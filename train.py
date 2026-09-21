# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import time
import uuid
import imageio
import datetime
import numpy as np
from tqdm import tqdm

import torch

from src.config import cfg, update_argparser, update_config, everytype2bool

from src.utils.system_utils import seed_everything
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.bounding_utils import decide_main_bounding
from src.utils import mono_utils
from src.utils import loss_utils

from src.dataloader.data_pack import DataPack, compute_iter_idx
from src.sparse_voxel_model import SparseVoxelModel
from src.cameras import CameraPoseOptimizer
from src.utils.camera_utils import compute_ate, compute_rpe

import svraster_cuda


def training(args):
    # Init and load data pack
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        mask_dir_name=cfg.data.mask_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
    )

    # Instantiate data loader
    tr_cams = data_pack.get_train_cameras()
    tr_cam_indices = compute_iter_idx(len(tr_cams), cfg.procedure.n_iter)

    if cfg.auto_exposure.enable:
        for cam in tr_cams:
            cam.auto_exposure_init()

    # Prepare monocular depth priors if instructed
    if cfg.regularizer.lambda_depthanythingv2 or cfg.regularizer.lambda_depth_ranking > 0:
    #if cfg.regularizer.lambda_depthanythingv2:
        mono_utils.prepare_depthanythingv2(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            force_rerun=False)

    if cfg.regularizer.lambda_mast3r_metric_depth:
        mono_utils.prepare_mast3r_metric_depth(
            cameras=tr_cams,
            source_path=cfg.data.source_path,
            mast3r_repo_path=cfg.regularizer.mast3r_repo_path)

    # Decide main (inside) region bounding box
    bounding = decide_main_bounding(
        bound_mode=cfg.bounding.bound_mode,
        forward_dist_scale=cfg.bounding.forward_dist_scale,
        pcd_density_rate=cfg.bounding.pcd_density_rate,
        bound_scale=cfg.bounding.bound_scale,
        tr_cams=tr_cams,
        pcd=data_pack.point_cloud,
        suggested_bounding=data_pack.suggested_bounding)

    # Init voxel model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    source_model_path = args.load_model_path if getattr(args, 'load_model_path', None) else args.model_path

    # Load checkpoint if specified, preview progress
    if args.load_iteration is not None:
        loaded_iter = voxel_model.load_iteration(
            source_model_path, args.load_iteration)
    else:
        loaded_iter = None
        voxel_model.model_init(
            bounding=bounding,
            outside_level=cfg.bounding.outside_level,
            init_n_level=cfg.init.init_n_level,
            init_out_ratio=cfg.init.init_out_ratio,
            sh_degree_init=cfg.init.sh_degree_init,
            geo_init=cfg.init.geo_init,
            sh0_init=cfg.init.sh0_init,
            shs_init=cfg.init.shs_init,
            cameras=tr_cams,
        )

    first_iter = loaded_iter if loaded_iter is not None else 1
    print(f"Start optmization from iters={first_iter}.")

    # Init optimizer
    def create_trainer():
        # The pytorch built-in `torch.optim.Adam` also works
        optimizer = svraster_cuda.sparse_adam.SparseAdam(
            [
                {'params': [voxel_model._geo_grid_pts], 'lr': cfg.optimizer.geo_lr},
                {'params': [voxel_model._sh0], 'lr': cfg.optimizer.sh0_lr},
                {'params': [voxel_model._shs], 'lr': cfg.optimizer.shs_lr},
            ],
            betas=(cfg.optimizer.optim_beta1, cfg.optimizer.optim_beta2),
            eps=cfg.optimizer.optim_eps)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optimizer.lr_decay_ckpt,
            gamma=cfg.optimizer.lr_decay_mult)
        return optimizer, scheduler

    optimizer, scheduler = create_trainer()
    if loaded_iter is not None and args.load_optimizer:
        optim_path = os.path.join(source_model_path, "optim.pt")
        if os.path.exists(optim_path):
            optim_ckpt = torch.load(optim_path)
            optimizer.load_state_dict(optim_ckpt['optim'])
            scheduler.load_state_dict(optim_ckpt['sched'])
            del optim_ckpt
            print(f"[OPTIM] Loaded optimizer and scheduler from {optim_path}")
        else:
            print(f"[OPTIM WARNING] Could not find {optim_path} to load optimizer state.")

    # Camera Pose Optimizer initialization
    pose_optimizer = None
    optim_pose = None
    sched_pose = None
    if cfg.pose_opt.pose_opt:
        print(f"[POSE OPT] Initializing CameraPoseOptimizer with {len(tr_cams)} cameras (mode='{cfg.pose_opt.pose_init_mode}', lr={cfg.pose_opt.lr_pose}, warmup={cfg.pose_opt.warmup_pose})")
        pose_optimizer = CameraPoseOptimizer(
            num_cams=len(tr_cams),
            init_c2w_list=tr_cams,
            mode=cfg.pose_opt.pose_init_mode
        ).cuda()
        optim_pose = torch.optim.Adam(
            pose_optimizer.parameters(),
            lr=cfg.pose_opt.lr_pose
        )
        total_decay_steps = max(1, cfg.procedure.n_iter - cfg.pose_opt.warmup_pose)
        gamma = (cfg.pose_opt.lr_pose_end / cfg.pose_opt.lr_pose) ** (1.0 / total_decay_steps)
        sched_pose = torch.optim.lr_scheduler.ExponentialLR(optim_pose, gamma=gamma)

        if loaded_iter is not None and args.load_optimizer:
            pose_ckpt_path = os.path.join(source_model_path, f"pose_opt_{loaded_iter:06d}.pt")
            if not os.path.exists(pose_ckpt_path):
                pose_ckpt_path = os.path.join(source_model_path, "pose_opt.pt")
            if os.path.exists(pose_ckpt_path):
                pose_ckpt = torch.load(pose_ckpt_path, map_location="cuda")
                pose_optimizer.load_state_dict_poses(pose_ckpt['poses'])
                if 'optim' in pose_ckpt and optim_pose is not None:
                    optim_pose.load_state_dict(pose_ckpt['optim'])
                if 'sched' in pose_ckpt and sched_pose is not None:
                    sched_pose.load_state_dict(pose_ckpt['sched'])
                print(f"[POSE OPT] Loaded pose optimizer state from {pose_ckpt_path}")

    # Some other initialization
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    elapsed = 0
    epoch_tracker = {
        'prev_c2w': pose_optimizer.get_all_c2w().detach().clone() if pose_optimizer is not None else None,
        'last_elapsed': 0.0
    }

    target_stat_iters = set()
    if not getattr(args, 'disable_voxel_stats', False):
        if hasattr(args, 'voxel_stats_iterations') and args.voxel_stats_iterations is not None:
            target_stat_iters.update(args.voxel_stats_iterations)
        else:
            target_stat_iters.update([5000, 10000, 15000, 20000])
        if hasattr(args, 'checkpoint_iterations') and args.checkpoint_iterations:
            target_stat_iters.update(args.checkpoint_iterations)
        target_stat_iters.add(cfg.procedure.n_iter)
    print(f"[VOXEL STATS] Configured iterations to save voxel statistics: {sorted(list(target_stat_iters))}")

    tr_render_opt = {
        'track_max_w': False,
        'lambda_R_concen': cfg.regularizer.lambda_R_concen,
        'output_T': False,
        'output_depth': False,
        'ss': 1.0,  # disable supersampling at first
        'rand_bg': cfg.regularizer.rand_bg,
        'use_auto_exposure': cfg.auto_exposure.enable,
    }

    sparse_depth_loss = loss_utils.SparseDepthLoss(
        iter_end=cfg.regularizer.sparse_depth_until)
    depthanythingv2_loss = loss_utils.DepthAnythingv2Loss(
        iter_from=cfg.regularizer.depthanythingv2_from,
        iter_end=cfg.regularizer.depthanythingv2_end,
        end_mult=cfg.regularizer.depthanythingv2_end_mult)
    mast3r_metric_depth_loss = loss_utils.Mast3rMetricDepthLoss(
        iter_from=cfg.regularizer.mast3r_metric_depth_from,
        iter_end=cfg.regularizer.mast3r_metric_depth_end,
        end_mult=cfg.regularizer.mast3r_metric_depth_end_mult)
    nd_loss = loss_utils.NormalDepthConsistencyLoss(
        iter_from=cfg.regularizer.n_dmean_from,
        iter_end=cfg.regularizer.n_dmean_end,
        ks=cfg.regularizer.n_dmean_ks,
        tol_deg=cfg.regularizer.n_dmean_tol_deg)
    nmed_loss = loss_utils.NormalMedianConsistencyLoss(
        iter_from=cfg.regularizer.n_dmed_from,
        iter_end=cfg.regularizer.n_dmed_end)
    
    # New
    depth_ranking_loss = loss_utils.DepthAnythingv2RankingLoss(
        iter_from=cfg.regularizer.depth_ranking_from,
        iter_end=cfg.regularizer.depth_ranking_end,
        end_mult=cfg.regularizer.depth_ranking_end_mult,
        patch_size=cfg.regularizer.depth_ranking_patch_size,
    )
    

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    iter_rng = range(first_iter, cfg.procedure.n_iter+1)
    progress_bar = tqdm(iter_rng, desc="Training")
    for iteration in iter_rng:

        # Start processing time tracking of this iteration
        iter_start.record()

        # Increase the degree of SH by one up to a maximum degree
        if iteration % 1000 == 0:
            voxel_model.sh_degree_add1()

        # Recompute sh from cameras
        if iteration in cfg.procedure.reset_sh_ckpt:
            print("Reset sh0 from cameras.")
            print("Reset shs to zero.")
            voxel_model.reset_sh_from_cameras(tr_cams)
            torch.cuda.empty_cache()

        # Use default super-sampling option ////////// RESOLUCIONNNNNNNNNNNNNNNNNN 
        if iteration > 1000:
            if cfg.regularizer.ss_aug_max > 1:
                tr_render_opt['ss'] = np.random.uniform(1, cfg.regularizer.ss_aug_max)
            elif 'ss' in tr_render_opt:
                tr_render_opt.pop('ss')  # Use default ss

        
        need_depth_ranking = (cfg.regularizer.lambda_depth_ranking > 0 and depth_ranking_loss.is_active(iteration))

        need_sparse_depth = cfg.regularizer.lambda_sparse_depth > 0 and sparse_depth_loss.is_active(iteration)
        need_depthanythingv2 = cfg.regularizer.lambda_depthanythingv2 > 0 and depthanythingv2_loss.is_active(iteration)
        need_mast3r_metric_depth = cfg.regularizer.lambda_mast3r_metric_depth > 0 and mast3r_metric_depth_loss.is_active(iteration)
        need_nd_loss = cfg.regularizer.lambda_normal_dmean > 0 and nd_loss.is_active(iteration)
        need_nmed_loss = cfg.regularizer.lambda_normal_dmed > 0 and nmed_loss.is_active(iteration)
        tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth or need_depth_ranking
        #tr_render_opt['output_T'] = cfg.regularizer.lambda_T_concen > 0 or cfg.regularizer.lambda_T_inside > 0 or cfg.regularizer.lambda_mask > 0 or need_sparse_depth or need_nd_loss or need_depthanythingv2 or need_mast3r_metric_depth
        tr_render_opt['output_normal'] = need_nd_loss or need_nmed_loss
        tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth or need_depth_ranking
        #tr_render_opt['output_depth'] = need_sparse_depth or need_nd_loss or need_nmed_loss or need_depthanythingv2 or need_mast3r_metric_depth
        if iteration >= cfg.regularizer.dist_from and cfg.regularizer.lambda_dist:
            tr_render_opt['lambda_dist'] = cfg.regularizer.lambda_dist

        if iteration >= cfg.regularizer.ascending_from and cfg.regularizer.lambda_ascending:
            tr_render_opt['lambda_ascending'] = cfg.regularizer.lambda_ascending

        # Update auto exposure
        if cfg.auto_exposure.enable and iteration in cfg.procedure.auto_exposure_upd_ckpt:
            for cam in tr_cams:
                with torch.no_grad():
                    ref = voxel_model.render(cam, ss=1.0)['color']
                cam.auto_exposure_update(ref, cam.image.cuda())

        # Pick a Camera
        cam_idx = tr_cam_indices[iteration-1]
        cam = tr_cams[cam_idx]
        if cfg.pose_opt.pose_opt and pose_optimizer is not None:
            cam_c2w = pose_optimizer.get_c2w(cam_idx)
            cam.c2w = cam_c2w
            cam.w2c = torch.inverse(cam_c2w).detach()

        # Get gt image
        gt_image = cam.image.cuda()
        if cfg.regularizer.lambda_R_concen > 0:
            tr_render_opt['gt_color'] = gt_image

        # Render
        render_pkg = voxel_model.render(cam, **tr_render_opt)
        render_image = render_pkg['color']

        # Loss
        mse = loss_utils.l2_loss(render_image, gt_image)

        if cfg.regularizer.use_l1:
            photo_loss = loss_utils.l1_loss(render_image, gt_image)
        elif cfg.regularizer.use_huber:
            photo_loss = loss_utils.huber_loss(render_image, gt_image, cfg.regularizer.huber_thres)
        else:
            photo_loss = mse
        loss = cfg.regularizer.lambda_photo * photo_loss

        if need_sparse_depth:
            loss += cfg.regularizer.lambda_sparse_depth * sparse_depth_loss(cam, render_pkg)

        if cfg.regularizer.lambda_mask:
            gt_T = 1 - cam.mask.cuda()
            loss += cfg.regularizer.lambda_mask * loss_utils.l2_loss(render_pkg['T'], gt_T)

        if need_depthanythingv2:
            loss += cfg.regularizer.lambda_depthanythingv2 * depthanythingv2_loss(cam, render_pkg, iteration)

        if need_mast3r_metric_depth:
            loss += cfg.regularizer.lambda_mast3r_metric_depth * mast3r_metric_depth_loss(cam, render_pkg, iteration)

        # Depth ranking loss .....
        if need_depth_ranking:
            loss += cfg.regularizer.lambda_depth_ranking * depth_ranking_loss(cam, render_pkg, iteration)

        if cfg.regularizer.lambda_ssim:
            loss += cfg.regularizer.lambda_ssim * loss_utils.fast_ssim_loss(render_image, gt_image)
        if cfg.regularizer.lambda_T_concen:
            loss += cfg.regularizer.lambda_T_concen * loss_utils.prob_concen_loss(render_pkg[f'raw_T'])
        if cfg.regularizer.lambda_T_inside:
            loss += cfg.regularizer.lambda_T_inside * render_pkg[f'raw_T'].square().mean()
        if need_nd_loss:
            loss += cfg.regularizer.lambda_normal_dmean * nd_loss(cam, render_pkg, iteration)
        if need_nmed_loss:
            loss += cfg.regularizer.lambda_normal_dmed * nmed_loss(cam, render_pkg, iteration)

        # Backward to get gradient of current iteration
        optimizer.zero_grad(set_to_none=True)
        if optim_pose is not None:
            optim_pose.zero_grad(set_to_none=True)
        loss.backward()

        # Total variation regularization
        if cfg.regularizer.lambda_tv_density and \
                iteration >= cfg.regularizer.tv_from and \
                iteration <= cfg.regularizer.tv_until:
            voxel_model.apply_tv_on_density_field(cfg.regularizer.lambda_tv_density)

        # Optimizer step
        optimizer.step()

        if optim_pose is not None and iteration > cfg.pose_opt.warmup_pose:
            if iteration % 1000  > 300:
                optim_pose.step()
        ######################################################
        # Start adaptive voxels pruning and subdividing
        ######################################################

        meet_adapt_period = (
            iteration % cfg.procedure.adapt_every == 0 and \
            iteration >= cfg.procedure.adapt_from and \
            iteration <= cfg.procedure.n_iter-500)
        need_pruning = (
            meet_adapt_period and \
            iteration <= cfg.procedure.prune_until)
        need_subdividing = (
            meet_adapt_period and \
            iteration <= cfg.procedure.subdivide_until and \
            voxel_model.num_voxels < cfg.procedure.subdivide_max_num)

        need_save_voxel_stats = (iteration in target_stat_iters)

        if need_pruning or need_subdividing or need_save_voxel_stats:
            # Track voxel statistic
            stat_pkg = voxel_model.compute_training_stat(camera_lst=tr_cams)
            if need_pruning or need_subdividing:
                # Cache scheduler state
                scheduler_state = scheduler.state_dict()

        if need_save_voxel_stats:
            save_voxel_training_stats(
                args=args,
                iteration=iteration,
                voxel_model=voxel_model,
                stat_pkg=stat_pkg
            )

        if need_pruning:
            ori_n = voxel_model.num_voxels

            # Compute pruning threshold
            prune_thres = np.interp(
                iteration,
                xp=[cfg.procedure.adapt_from, cfg.procedure.prune_until],
                fp=[cfg.procedure.prune_thres_init, cfg.procedure.prune_thres_final])

            # Prune voxels
            prune_mask = (stat_pkg['max_w'] < prune_thres).squeeze(1)

            # Pruning
            voxel_model.pruning(prune_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            print(f'[PRUNING]     {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f};  thres={prune_thres:.4f})')

        if need_subdividing:
            ori_n = voxel_model.num_voxels

            # Exclude some voxels
            min_samp_interval = stat_pkg['min_samp_interval']
            if need_pruning:
                min_samp_interval = min_samp_interval[~prune_mask]
            size_thres = min_samp_interval * cfg.procedure.subdivide_samp_thres
            large_enough = (voxel_model.vox_size * 0.5 > size_thres).squeeze(1)
            non_finest = voxel_model.octlevel.squeeze(1) < svraster_cuda.meta.MAX_NUM_LEVELS
            valid_mask = large_enough & non_finest

            # Compute subdivision threshold
            priority = voxel_model.subdivision_priority.squeeze(1) * valid_mask

            if iteration <= cfg.procedure.subdivide_all_until:
                thres = -1
            else:
                thres = priority.quantile(1 - cfg.procedure.subdivide_prop)

            subdivide_mask = (priority > thres) & valid_mask

            # In case the number of voxels over the threshold
            max_n_subdiv = round((cfg.procedure.subdivide_max_num - voxel_model.num_voxels) / 7)
            if subdivide_mask.sum() > max_n_subdiv:
                n_removed = subdivide_mask.sum() - max_n_subdiv
                subdivide_mask &= (priority > priority[subdivide_mask].sort().values[n_removed-1])

            # Subdivision
            voxel_model.subdividing(subdivide_mask)

            # Show statistic
            new_n = voxel_model.num_voxels
            in_p = voxel_model.inside_mask.float().mean().item()
            print(f'[SUBDIVIDING] {ori_n:7d} => {new_n:7d} (x{new_n/ori_n:.2f}; inside={in_p*100:.1f}%)')

            # Reset priority for the next round
            voxel_model.reset_subdivision_priority()

        if need_pruning or need_subdividing:
            # Re-create trainer for the updated parameters
            optimizer, scheduler = create_trainer()
            scheduler.load_state_dict(scheduler_state)
            del scheduler_state

            torch.cuda.empty_cache()

        ######################################################
        # End of adaptive voxels procedure
        ######################################################

        # Update learning rate
        scheduler.step()
        if sched_pose is not None and iteration > cfg.pose_opt.warmup_pose:
            sched_pose.step()

        # End processing time tracking of this iteration
        iter_end.record()
        torch.cuda.synchronize()
        elapsed += iter_start.elapsed_time(iter_end)

        # Logging
        with torch.no_grad():
            # Metric
            loss = loss.item()
            psnr = -10 * np.log10(mse.item())

            # Progress bar
            ema_p = max(0.01, 1 / (iteration - first_iter + 1))
            ema_loss_for_log += ema_p * (loss - ema_loss_for_log)
            ema_psnr_for_log += ema_p * (psnr - ema_psnr_for_log)
            if iteration % 10 == 0:
                pb_text = {
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "psnr": f"{ema_psnr_for_log:.2f}",
                }
                progress_bar.set_postfix(pb_text)
                progress_bar.update(10)
            if iteration == cfg.procedure.n_iter:
                progress_bar.close()

            # Log and save
            training_report(
                args=args,
                data_pack=data_pack,
                voxel_model=voxel_model,
                iteration=iteration,
                elapsed=elapsed,
                ema_psnr=ema_psnr_for_log,
                pose_optimizer=pose_optimizer)

            if iteration in args.checkpoint_iterations or iteration == cfg.procedure.n_iter:
                voxel_model.save_iteration(args.model_path, iteration, quantize=args.save_quantized)
                if args.save_optimizer:
                    torch.save(
                        {'optim': optimizer.state_dict(), 'sched': scheduler.state_dict()},
                        os.path.join(args.model_path, "optim.pt"))
                if cfg.pose_opt.pose_opt and pose_optimizer is not None:
                    pose_ckpt = {
                        'poses': pose_optimizer.state_dict_poses(),
                    }
                    if args.save_optimizer and optim_pose is not None:
                        pose_ckpt['optim'] = optim_pose.state_dict()
                        pose_ckpt['sched'] = sched_pose.state_dict()
                    torch.save(pose_ckpt, os.path.join(args.model_path, f"pose_opt_{iteration:06d}.pt"))
                    torch.save(pose_ckpt, os.path.join(args.model_path, "pose_opt.pt"))
                    print(f"[SAVE POSE] path={os.path.join(args.model_path, f'pose_opt_{iteration:06d}.pt')}")
                print(f"[SAVE] path={voxel_model.latest_save_path}")

            # Record per-epoch metrics (structure + pose variation)
            record_epoch_metrics(
                args=args,
                iteration=iteration,
                elapsed=elapsed,
                ema_loss=ema_loss_for_log,
                ema_psnr=ema_psnr_for_log,
                voxel_model=voxel_model,
                pose_optimizer=pose_optimizer,
                tr_cams=tr_cams,
                optimizer=optimizer,
                optim_pose=optim_pose,
                epoch_tracker=epoch_tracker,
                progress_bar=progress_bar
            )


def record_epoch_metrics(
    args, iteration, elapsed, ema_loss, ema_psnr,
    voxel_model, pose_optimizer, tr_cams,
    optimizer, optim_pose, epoch_tracker,
    progress_bar=None
):
    n_cams = max(1, len(tr_cams))
    is_epoch_end = (iteration % n_cams == 0) or (iteration == cfg.procedure.n_iter)
    if not is_epoch_end:
        return

    current_epoch = iteration // n_cams if (iteration % n_cams == 0) else (iteration // n_cams + 1)
    epoch_interval = getattr(cfg.pose_opt, 'pose_epoch_interval', 1)
    if (current_epoch % epoch_interval != 0) and (iteration != cfg.procedure.n_iter):
        return

    elapsed_sec = elapsed / 1000.0
    epoch_time_sec = elapsed_sec - epoch_tracker.get('last_elapsed', 0.0)
    epoch_tracker['last_elapsed'] = elapsed_sec

    # 1. Structure metrics
    num_voxels = voxel_model.num_voxels
    try:
        inside_mask = voxel_model.inside_mask
        inside_voxels = inside_mask.sum().item()
        inside_pct = (inside_voxels / max(1, num_voxels)) * 100.0
    except Exception:
        inside_voxels = num_voxels
        inside_pct = 100.0

    lr_geo = optimizer.param_groups[0]['lr'] if (optimizer is not None and len(optimizer.param_groups) > 0) else 0.0
    lr_pose = optim_pose.param_groups[0]['lr'] if (optim_pose is not None and iteration > cfg.pose_opt.warmup_pose) else 0.0

    # 2. Pose variation metrics
    epoch_drot_mean_deg = 0.0
    epoch_drot_max_deg = 0.0
    epoch_dtrans_mean_m = 0.0
    epoch_dtrans_max_m = 0.0

    tot_drot_mean_deg = 0.0
    tot_drot_max_deg = 0.0
    tot_dtrans_mean_m = 0.0
    tot_dtrans_max_m = 0.0

    ate_res = None
    rpe_res = None
    has_gt = False

    if pose_optimizer is not None:
        curr_c2w = pose_optimizer.get_all_c2w().detach()  # [N, 4, 4]

        # Pose variation in this epoch relative to previous epoch
        if epoch_tracker.get('prev_c2w') is not None:
            prev_c2w = epoch_tracker['prev_c2w']
            R_rel = torch.bmm(prev_c2w[:, :3, :3].transpose(-1, -2), curr_c2w[:, :3, :3])
            t_rel = curr_c2w[:, :3, 3] - prev_c2w[:, :3, 3]

            traces = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_ang = torch.clamp((traces - 1.0) / 2.0, -1.0, 1.0)
            epoch_rot_deg = torch.acos(cos_ang) * (180.0 / np.pi)
            epoch_trans_m = torch.norm(t_rel, dim=-1)

            epoch_drot_mean_deg = epoch_rot_deg.mean().item()
            epoch_drot_max_deg = epoch_rot_deg.max().item()
            epoch_dtrans_mean_m = epoch_trans_m.mean().item()
            epoch_dtrans_max_m = epoch_trans_m.max().item()

        epoch_tracker['prev_c2w'] = curr_c2w.clone()

        # Cumulative variation from initialization (COLMAP base poses)
        se3 = pose_optimizer.se3_refine.weight.detach()
        tot_rot_deg = se3[:, :3].norm(dim=1) * (180.0 / np.pi)
        tot_trans_m = se3[:, 3:].norm(dim=1)
        tot_drot_mean_deg = tot_rot_deg.mean().item()
        tot_drot_max_deg = tot_rot_deg.max().item()
        tot_dtrans_mean_m = tot_trans_m.mean().item()
        tot_dtrans_max_m = tot_trans_m.max().item()

        # ATE / RPE against Ground Truth if available
        has_gt = any(getattr(c, 'c2w_gt', None) is not None for c in tr_cams)
        if has_gt:
            gt_c2ws = torch.stack([
                c.c2w_gt if getattr(c, 'c2w_gt', None) is not None else c.c2w.detach()
                for c in tr_cams
            ]).to(curr_c2w.device)
            ate_res = compute_ate(curr_c2w, gt_c2ws)
            rpe_res = compute_rpe(curr_c2w, gt_c2ws)

    # 3. CSV Logging
    csv_path = os.path.join(args.model_path, "metrics_per_epoch.csv")
    csv_exists = os.path.exists(csv_path)

    header = (
        "epoch,iteration,elapsed_sec,epoch_time_sec,loss,psnr,num_voxels,inside_voxels,inside_pct,"
        "lr_geo,lr_pose,epoch_drot_mean_deg,epoch_drot_max_deg,epoch_dtrans_mean_m,epoch_dtrans_max_m,"
        "total_drot_mean_deg,total_drot_max_deg,total_dtrans_mean_m,total_dtrans_max_m,"
        "ate_trans_rmse_m,ate_rot_rmse_deg,rpe_trans_rmse_m,rpe_rot_rmse_deg\n"
    )

    ate_t_str = f"{ate_res['trans_rmse']:.6f}" if ate_res is not None else ""
    ate_r_str = f"{ate_res['rot_rmse_deg']:.4f}" if ate_res is not None else ""
    rpe_t_str = f"{rpe_res['trans_rmse']:.6f}" if rpe_res is not None else ""
    rpe_r_str = f"{rpe_res['rot_rmse_deg']:.4f}" if rpe_res is not None else ""

    row = (
        f"{current_epoch},{iteration},{elapsed_sec:.1f},{epoch_time_sec:.1f},"
        f"{ema_loss:.6f},{ema_psnr:.4f},{num_voxels},{inside_voxels},{inside_pct:.2f},"
        f"{lr_geo:.6e},{lr_pose:.6e},"
        f"{epoch_drot_mean_deg:.4f},{epoch_drot_max_deg:.4f},{epoch_dtrans_mean_m:.6f},{epoch_dtrans_max_m:.6f},"
        f"{tot_drot_mean_deg:.4f},{tot_drot_max_deg:.4f},{tot_dtrans_mean_m:.6f},{tot_dtrans_max_m:.6f},"
        f"{ate_t_str},{ate_r_str},{rpe_t_str},{rpe_r_str}\n"
    )

    with open(csv_path, 'a') as f:
        if not csv_exists:
            f.write(header)
        f.write(row)

    # 4. JSONL Logging
    jsonl_path = os.path.join(args.model_path, "metrics_per_epoch.jsonl")
    epoch_dict = {
        'epoch': current_epoch,
        'iteration': iteration,
        'elapsed_sec': round(elapsed_sec, 1),
        'epoch_time_sec': round(epoch_time_sec, 1),
        'loss': round(float(ema_loss), 6),
        'psnr': round(float(ema_psnr), 4),
        'num_voxels': int(num_voxels),
        'inside_voxels': int(inside_voxels),
        'inside_pct': round(float(inside_pct), 2),
        'lr_geo': float(lr_geo),
        'lr_pose': float(lr_pose),
        'epoch_drot_mean_deg': round(float(epoch_drot_mean_deg), 4),
        'epoch_drot_max_deg': round(float(epoch_drot_max_deg), 4),
        'epoch_dtrans_mean_m': round(float(epoch_dtrans_mean_m), 6),
        'epoch_dtrans_max_m': round(float(epoch_dtrans_max_m), 6),
        'total_drot_mean_deg': round(float(tot_drot_mean_deg), 4),
        'total_drot_max_deg': round(float(tot_drot_max_deg), 4),
        'total_dtrans_mean_m': round(float(tot_dtrans_mean_m), 6),
        'total_dtrans_max_m': round(float(tot_dtrans_max_m), 6),
    }
    if ate_res is not None:
        epoch_dict['ate_trans_rmse_m'] = round(float(ate_res['trans_rmse']), 6)
        epoch_dict['ate_rot_rmse_deg'] = round(float(ate_res['rot_rmse_deg']), 4)
    if rpe_res is not None:
        epoch_dict['rpe_trans_rmse_m'] = round(float(rpe_res['trans_rmse']), 6)
        epoch_dict['rpe_rot_rmse_deg'] = round(float(rpe_res['rot_rmse_deg']), 4)

    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(epoch_dict) + "\n")

    # 5. Clean terminal report
    gt_msg = f" | ATE: {ate_res['trans_rmse']:.4f}m, {ate_res['rot_rmse_deg']:.2f}deg" if (has_gt and ate_res is not None) else ""
    pose_msg = (
        f" | dPose/ep: {epoch_drot_mean_deg:.3f}deg, {epoch_dtrans_mean_m*100:.3f}cm"
        f" | Tot dPose: {tot_drot_mean_deg:.2f}deg, {tot_dtrans_mean_m*100:.2f}cm"
        if pose_optimizer is not None else ""
    )
    epoch_msg = (
        f"[EPOCH {current_epoch:03d} | iter {iteration:05d} | {epoch_time_sec:.1f}s] "
        f"Voxels: {num_voxels} ({inside_pct:.1f}% in) | Loss: {ema_loss:.5f} | PSNR: {ema_psnr:.2f}"
        f"{pose_msg}{gt_msg}"
    )
    if progress_bar is not None and hasattr(progress_bar, 'write'):
        progress_bar.write(epoch_msg)
    else:
        print(epoch_msg)


def save_voxel_training_stats(args, iteration, voxel_model, stat_pkg):
    """
    Save detailed per-voxel statistics and system VRAM before pruning, subdividing,
    and subdivision priority reset.

    Per-voxel fields:
    - voxel_model.vox_center: [N, 3] (float32)
    - voxel_model.vox_size: [N, 1] (float32)
    - voxel_model.octlevel: [N, 1] (int8)
    - voxel_model.subdivision_priority: [N, 1] (float32, or None)
    - stat_pkg['max_w']: [N, 1] (float32)
    - stat_pkg['view_cnt']: [N, 1] (float32)
    - stat_pkg['min_samp_interval']: [N, 1] (float32)

    Global metrics:
    - n_voxels: int
    - vram: dict with allocated, reserved, and peak memory in bytes, MB, GB.
    """
    stats_dir = os.path.join(args.model_path, "voxel_stats")
    os.makedirs(stats_dir, exist_ok=True)

    n_voxels = int(voxel_model.num_voxels)

    # Detach tensors and transfer to CPU
    vox_center = voxel_model.vox_center.detach().clone().cpu()
    vox_size = voxel_model.vox_size.detach().clone().cpu()
    octlevel = voxel_model.octlevel.detach().clone().cpu()

    subdiv_p = voxel_model.subdivision_priority
    if subdiv_p is not None:
        subdivision_priority = subdiv_p.detach().clone().cpu()
    else:
        subdivision_priority = None

    max_w = stat_pkg['max_w'].detach().clone().cpu()
    view_cnt = stat_pkg['view_cnt'].detach().clone().cpu()
    min_samp_interval = stat_pkg['min_samp_interval'].detach().clone().cpu()

    # VRAM statistics
    if torch.cuda.is_available():
        alloc_b = torch.cuda.memory_allocated()
        max_alloc_b = torch.cuda.max_memory_allocated()
        res_b = torch.cuda.memory_reserved()
        max_res_b = torch.cuda.max_memory_reserved()
    else:
        alloc_b = 0
        max_alloc_b = 0
        res_b = 0
        max_res_b = 0

    vram_dict = {
        'allocated_bytes': int(alloc_b),
        'allocated_mb': float(alloc_b / (1024 ** 2)),
        'allocated_gb': float(alloc_b / (1024 ** 3)),
        'max_allocated_bytes': int(max_alloc_b),
        'max_allocated_mb': float(max_alloc_b / (1024 ** 2)),
        'max_allocated_gb': float(max_alloc_b / (1024 ** 3)),
        'reserved_bytes': int(res_b),
        'reserved_mb': float(res_b / (1024 ** 2)),
        'reserved_gb': float(res_b / (1024 ** 3)),
        'max_reserved_bytes': int(max_res_b),
        'max_reserved_mb': float(max_res_b / (1024 ** 2)),
        'max_reserved_gb': float(max_res_b / (1024 ** 3)),
    }

    has_subdiv = subdivision_priority is not None and subdivision_priority.numel() > 0
    if has_subdiv:
        subdiv_flat = subdivision_priority.view(-1).float()
        sum_subdiv_p = float(subdiv_flat.sum().item())
        median_subdiv_p = float(subdiv_flat.median().item())
        quantiles = torch.quantile(subdiv_flat, torch.tensor([0.90, 0.95, 0.99]))
        p90_subdiv_p = float(quantiles[0].item())
        p95_subdiv_p = float(quantiles[1].item())
        p99_subdiv_p = float(quantiles[2].item())
    else:
        sum_subdiv_p = float('nan')
        median_subdiv_p = float('nan')
        p90_subdiv_p = float('nan')
        p95_subdiv_p = float('nan')
        p99_subdiv_p = float('nan')

    payload = {
        'iteration': int(iteration),
        'n_voxels': n_voxels,
        'vox_center': vox_center,
        'vox_size': vox_size,
        'octlevel': octlevel,
        'subdivision_priority': subdivision_priority,
        'subdivision_priority_stats': {
            'sum': sum_subdiv_p,
            'median': median_subdiv_p,
            'p90': p90_subdiv_p,
            'p95': p95_subdiv_p,
            'p99': p99_subdiv_p,
        },
        'max_w': max_w,
        'view_cnt': view_cnt,
        'min_samp_interval': min_samp_interval,
        'vram': vram_dict,
        'vram_allocated_mb': vram_dict['allocated_mb'],
        'vram_max_allocated_mb': vram_dict['max_allocated_mb'],
        'vram_reserved_mb': vram_dict['reserved_mb'],
        'vram_max_reserved_mb': vram_dict['max_reserved_mb'],
    }

    pt_filename = f"voxel_stats_iter{iteration:06d}.pt"
    pt_path = os.path.join(stats_dir, pt_filename)
    torch.save(payload, pt_path)

    # Append to summary CSV
    csv_path = os.path.join(stats_dir, "voxel_stats_summary.csv")
    csv_header = (
        "iteration,n_voxels,vram_allocated_mb,vram_max_allocated_mb,"
        "vram_reserved_mb,vram_max_reserved_mb,mean_max_w,mean_view_cnt,"
        "mean_min_samp_interval,has_subdiv_priority,sum_subdivision_priority,"
        "median_subdivision_priority,p90_subdivision_priority,p95_subdivision_priority,"
        "p99_subdivision_priority,file_path\n"
    )
    file_exists = os.path.isfile(csv_path)

    mean_max_w = float(max_w.mean().item()) if max_w.numel() > 0 else 0.0
    mean_view_cnt = float(view_cnt.mean().item()) if view_cnt.numel() > 0 else 0.0
    obs_mask = max_w > 0
    if obs_mask.any():
        mean_samp = float(min_samp_interval[obs_mask].mean().item())
    else:
        mean_samp = float(min_samp_interval.mean().item()) if min_samp_interval.numel() > 0 else 0.0

    sum_str = f"{sum_subdiv_p:.6e}" if not np.isnan(sum_subdiv_p) else "nan"
    median_str = f"{median_subdiv_p:.6e}" if not np.isnan(median_subdiv_p) else "nan"
    p90_str = f"{p90_subdiv_p:.6e}" if not np.isnan(p90_subdiv_p) else "nan"
    p95_str = f"{p95_subdiv_p:.6e}" if not np.isnan(p95_subdiv_p) else "nan"
    p99_str = f"{p99_subdiv_p:.6e}" if not np.isnan(p99_subdiv_p) else "nan"

    row_str = (
        f"{iteration},{n_voxels},{vram_dict['allocated_mb']:.2f},{vram_dict['max_allocated_mb']:.2f},"
        f"{vram_dict['reserved_mb']:.2f},{vram_dict['max_reserved_mb']:.2f},"
        f"{mean_max_w:.6f},{mean_view_cnt:.2f},{mean_samp:.6f},"
        f"{has_subdiv},{sum_str},{median_str},{p90_str},{p95_str},{p99_str},{pt_filename}\n"
    )

    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write(csv_header)
        f.write(row_str)

    subdiv_msg = f", SubdivP(med={median_str}, p90={p90_str}, p99={p99_str})" if has_subdiv else ""
    print(
        f"[VOXEL STATS] Iteration {iteration:06d}: N_voxels={n_voxels:,}, "
        f"VRAM_alloc={vram_dict['allocated_mb']:.1f}MB (Peak={vram_dict['max_allocated_mb']:.1f}MB)"
        f"{subdiv_msg} -> saved {pt_path}"
    )


def training_report(args, data_pack, voxel_model, iteration, elapsed, ema_psnr, pose_optimizer=None):

    voxel_model.freeze_vox_geo()

    # Progress view
    if args.pg_view_every > 0 and (iteration % args.pg_view_every == 0 or iteration == 1):
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        if len(test_cameras) == 0:
            test_cameras = data_pack.get_train_cameras()
        pg_idx = 0
        view = test_cameras[pg_idx]
        render_pkg = voxel_model.render(view, output_depth=True, output_normal=True, output_T=True)
        render_image = render_pkg['color']
        render_depth = render_pkg['depth'][0]
        render_depth_med = render_pkg['depth'][2]
        render_normal = render_pkg['normal']
        render_alpha = 1 - render_pkg['T'][0]

        im = np.concatenate([
            np.concatenate([
                im_tensor2np(render_image),
                im_tensor2np(render_alpha)[...,None].repeat(3, axis=-1),
            ], axis=1),
            np.concatenate([
                viz_tensordepth(render_depth, render_alpha),
                im_tensor2np(render_normal * 0.5 + 0.5),
            ], axis=1),
            np.concatenate([
                im_tensor2np(view.depth2normal(render_depth) * 0.5 + 0.5),
                im_tensor2np(view.depth2normal(render_depth_med) * 0.5 + 0.5),
            ], axis=1),
        ], axis=0)
        torch.cuda.empty_cache()

        outdir = os.path.join(args.model_path, "pg_view")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.jpg")
        os.makedirs(outdir, exist_ok=True)

        imageio.imwrite(outpath, im)

        eps_file = os.path.join(args.model_path, "pg_view", "eps.txt")
        with open(eps_file, 'a') as f:
            f.write(f"{iteration},{elapsed/1000:.1f}\n")

    # Report test and samples of training set
    if iteration in args.test_iterations:
        print(f"[EVAL] running...")
        torch.cuda.empty_cache()
        test_cameras = data_pack.get_test_cameras()
        save_every = max(1, len(test_cameras) // 8)
        outdir = os.path.join(args.model_path, "test_view")
        os.makedirs(outdir, exist_ok=True)
        psnr_lst = []
        video = []
        max_w = torch.zeros([voxel_model.num_voxels, 1], dtype=torch.float32, device="cuda")
        for idx, camera in enumerate(test_cameras):
            render_pkg = voxel_model.render(camera, output_normal=True, track_max_w=True)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            video.append(im)
            if idx % save_every == 0:
                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}.jpg")
                cat = np.concatenate([gt, im], axis=1)
                imageio.imwrite(outpath, cat)

                outpath = os.path.join(outdir, f"idx{idx:04d}_iter{iteration:06d}_normal.jpg")
                render_normal = render_pkg['normal']
                render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
                imageio.imwrite(outpath, render_normal)
            mse = np.square(im/255 - gt/255).mean()
            psnr_lst.append(-10 * np.log10(mse))
            max_w = torch.maximum(max_w, render_pkg['max_w'])
        avg_psnr = np.mean(psnr_lst)
        imageio.mimwrite(
            os.path.join(outdir, f"video_iter{iteration:06d}.mp4"),
            video, fps=30)
        torch.cuda.empty_cache()

        fps = time.time()
        for idx, camera in enumerate(test_cameras):
            voxel_model.render(camera, track_max_w=False)
        torch.cuda.synchronize()
        fps = len(test_cameras) / (time.time() - fps)
        torch.cuda.empty_cache()

        # Sample training views to render
        train_cameras = data_pack.get_train_cameras()
        for idx in range(0, len(train_cameras), max(1, len(train_cameras)//8)):
            camera = train_cameras[idx]
            render_pkg = voxel_model.render(
                camera, output_normal=True, track_max_w=True,
                use_auto_exposure=cfg.auto_exposure.enable)
            render_image = render_pkg['color']
            im = im_tensor2np(render_image)
            gt = im_tensor2np(camera.image)
            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}.jpg")
            cat = np.concatenate([gt, im], axis=1)
            imageio.imwrite(outpath, cat)

            outpath = os.path.join(outdir, f"train_idx{idx:04d}_iter{iteration:06d}_normal.jpg")
            render_normal = render_pkg['normal']
            render_normal = im_tensor2np(render_normal * 0.5 + 0.5)
            imageio.imwrite(outpath, render_normal)

        print(f"[EVAL] iter={iteration:6d}  psnr={avg_psnr:.2f}  fps={fps:.0f}")

        # Evaluate camera poses if GT is available
        pose_stat = None
        if pose_optimizer is not None:
            train_cameras = data_pack.get_train_cameras()
            has_gt = any(getattr(c, 'c2w_gt', None) is not None for c in train_cameras)
            if has_gt:
                pred_c2ws = pose_optimizer.get_all_c2w().detach()
                gt_c2ws = torch.stack([
                    c.c2w_gt if getattr(c, 'c2w_gt', None) is not None else c.c2w.detach()
                    for c in train_cameras
                ]).to(pred_c2ws.device)
                ate_res = compute_ate(pred_c2ws, gt_c2ws)
                rpe_res = compute_rpe(pred_c2ws, gt_c2ws)
                pose_stat = {
                    'ate': ate_res,
                    'rpe': rpe_res,
                }
                print(f"[POSE EVAL] iter={iteration:6d} | ATE trans (RMSE): {ate_res['trans_rmse']:.4f} m, rot: {ate_res['rot_rmse_deg']:.2f} deg | RPE trans: {rpe_res['trans_rmse']:.4f} m, rot: {rpe_res['rot_rmse_deg']:.2f} deg")

        outdir = os.path.join(args.model_path, "test_stat")
        outpath = os.path.join(outdir, f"iter{iteration:06d}.json")
        os.makedirs(outdir, exist_ok=True)
        with open(outpath, 'w') as f:
            q = torch.linspace(0,1,5, device="cuda")
            max_w_q = max_w.quantile(q).tolist()
            peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
            stat = {
                'psnr': avg_psnr,
                'ema_psnr': ema_psnr,
                'elapsed': elapsed,
                'fps': fps,
                'n_voxels': voxel_model.num_voxels,
                'max_w_q': max_w_q,
                'peak_mem': peak_mem,
            }
            if pose_stat is not None:
                stat['pose_metrics'] = pose_stat
            json.dump(stat, f, indent=4)

    voxel_model.unfreeze_vox_geo()



if __name__ == "__main__":

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster optimization."
        "You can specify a list of config files to overwrite the default setups."
        "All config fields can also be overwritten by command line.")
    parser.add_argument('--model_path')
    parser.add_argument('--cfg_files', default=[], nargs='*')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[-1])
    parser.add_argument("--pg_view_every", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--voxel_stats_iterations", nargs="*", type=int, default=[5000, 10000, 15000, 20000], help="Iterations to save detailed per-voxel statistics")
    parser.add_argument("--disable_voxel_stats", action="store_true", default=False, help="Disable saving per-voxel statistics")
    parser.add_argument("--load_iteration", type=int, default=None, help="Iteration of checkpoint to load (-1 for latest)")
    parser.add_argument("--load_model_path", type=str, default=None, help="Optional model directory path to load checkpoint from (defaults to --model_path)")
    parser.add_argument("--load_optimizer", action='store_true', help="Load optimizer and scheduler state")
    parser.add_argument("--save_optimizer", action='store_true', help="Save optimizer state at checkpoints")
    parser.add_argument("--save_quantized", action='store_true', help="Save quantized model checkpoint to reduce file size")
    parser.add_argument("--pose_opt", action='store_true', default=False, help="Enable camera pose optimization")
    parser.add_argument("--pose_init_mode", type=str, default=None, choices=['colmap', 'identity'], help="Pose initialization mode: 'colmap' or 'identity'")
    parser.add_argument("--lr_pose", type=float, default=None, help="Initial learning rate for pose refinement")
    parser.add_argument("--warmup_pose", type=int, default=None, help="Warmup iterations before pose optimization")
    parser.add_argument("--pose_epoch_interval", type=int, default=None, help="Epoch interval to record metrics")
    parser.add_argument("--fast_10k", "--10k", action="store_true", default=False, help="Shift and scale the entire 20000-step training logic to 10000 steps.")
    parser.add_argument("--n_iter", "--iter", type=int, default=None, help="Set total iterations (e.g. 10000) and scale schedules proportionally.")
    parser.add_argument("--sche_mult", type=float, default=None, help="Schedule multiplier factor (e.g. 0.5 for 10k)")
    parser.add_argument("--keep_lr", action="store_true", default=False, help="Do not scale learning rates when scaling iterations (keeps original LR)")
    args, cmd_lst = parser.parse_known_args()

    # Update config from files and command line
    update_config(args.cfg_files, cmd_lst)
    if args.fast_10k:
        cfg.procedure.sche_mult = 0.5
    elif args.sche_mult is not None:
        cfg.procedure.sche_mult = args.sche_mult
    elif args.n_iter is not None:
        cfg.procedure.sche_mult = float(args.n_iter) / 20000.0

    if args.pose_opt:
        cfg.pose_opt.pose_opt = True
    if args.pose_init_mode is not None:
        cfg.pose_opt.pose_init_mode = args.pose_init_mode
    if args.lr_pose is not None:
        cfg.pose_opt.lr_pose = args.lr_pose
    if args.warmup_pose is not None:
        cfg.pose_opt.warmup_pose = args.warmup_pose
    if args.pose_epoch_interval is not None:
        cfg.pose_opt.pose_epoch_interval = args.pose_epoch_interval

    # Global init
    seed_everything(cfg.procedure.seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Setup output folder
    if not args.model_path:
        datetime_str = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        unique_str = str(uuid.uuid4())[:6]
        folder_name = f"{datetime_str}-{unique_str}"
        args.model_path = os.path.join(f"./output", folder_name)

    os.makedirs(args.model_path, exist_ok=True)

    # Apply scheduler scaling
    if cfg.procedure.sche_mult != 1:
        sche_mult = cfg.procedure.sche_mult

        if not args.keep_lr:
            for key in ['geo_lr', 'sh0_lr', 'shs_lr']:
                cfg.optimizer[key] /= sche_mult

        cfg.optimizer.lr_decay_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.optimizer.lr_decay_ckpt]

        for key in [
                'dist_from', 'tv_from', 'tv_until',
                'n_dmean_from', 'n_dmean_end',
                'n_dmed_from', 'n_dmed_end',
                'depthanythingv2_from', 'depthanythingv2_end',
                'mast3r_metric_depth_from', 'mast3r_metric_depth_end',
                'depth_ranking_from', 'depth_ranking_end']:
            if key in cfg.regularizer:
                cfg.regularizer[key] = round(cfg.regularizer[key] * sche_mult)

        for key in [
                'n_iter',
                'adapt_from', 'adapt_every',
                'prune_until', 'subdivide_until', 'subdivide_all_until']:
            cfg.procedure[key] = round(cfg.procedure[key] * sche_mult)

        cfg.procedure.reset_sh_ckpt = [
            round(v * sche_mult) if v > 0 else v
            for v in cfg.procedure.reset_sh_ckpt]

        if hasattr(cfg, 'auto_exposure') and 'auto_exposure_upd_ckpt' in cfg.auto_exposure:
            cfg.auto_exposure.auto_exposure_upd_ckpt = [
                round(v * sche_mult) if v > 0 else v
                for v in cfg.auto_exposure.auto_exposure_upd_ckpt]

        if args.voxel_stats_iterations:
            args.voxel_stats_iterations = [
                round(v * sche_mult) if v > 0 else v
                for v in args.voxel_stats_iterations]

        if args.checkpoint_iterations:
            args.checkpoint_iterations = [
                round(v * sche_mult) if v > 0 else v
                for v in args.checkpoint_iterations]

        print("=" * 80)
        print(f"[SCHEDULE SCALING] Transferred 20,000 schedule to {cfg.procedure.n_iter:,} iterations (multiplier = {sche_mult}):")
        print(f"  - Total Iterations (n_iter)   : 20,000 -> {cfg.procedure.n_iter:,}")
        print(f"  - Adapt Every (adapt_every)   : 1,000  -> {cfg.procedure.adapt_every:,}")
        print(f"  - Adapt From (adapt_from)     : 1,000  -> {cfg.procedure.adapt_from:,}")
        print(f"  - Subdivide Until             : 15,000 -> {cfg.procedure.subdivide_until:,}")
        print(f"  - Prune Until                 : 18,000 -> {cfg.procedure.prune_until:,}")
        print(f"  - LR Decay Checkpoints        : {cfg.optimizer.lr_decay_ckpt}")
        print(f"  - Voxel Stats Checkpoints     : {args.voxel_stats_iterations}")
        print("=" * 80)

    with open(os.path.join(args.model_path, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    print(f"Output folder: {args.model_path}")

    # Update negative iterations
    for i in range(len(args.test_iterations)):
        if args.test_iterations[i] < 0:
            args.test_iterations[i] += cfg.procedure.n_iter + 1
    for i in range(len(args.checkpoint_iterations)):
        if args.checkpoint_iterations[i] < 0:
            args.checkpoint_iterations[i] += cfg.procedure.n_iter + 1
    if args.voxel_stats_iterations:
        for i in range(len(args.voxel_stats_iterations)):
            if args.voxel_stats_iterations[i] < 0:
                args.voxel_stats_iterations[i] += cfg.procedure.n_iter + 1

    # Launch training loop
    training(args)
    print("Everything done.")
