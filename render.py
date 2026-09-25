# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio

import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.utils.image_utils import im_tensor2np, viz_tensordepth


@torch.no_grad()
def render_set(name, iteration, suffix, args, views, voxel_model):

    render_path = os.path.join(args.model_path, name, f"ours_{iteration}{suffix}", "renders")
    gts_path = os.path.join(args.model_path, name, f"ours_{iteration}{suffix}", "gt")
    alpha_path = os.path.join(args.model_path, name, f"ours_{iteration}{suffix}", "alpha")
    viz_path = os.path.join(args.model_path, name, f"ours_{iteration}{suffix}", "viz")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    makedirs(viz_path, exist_ok=True)
    print(f'render_path: {render_path}')
    print(f'ss            =: {voxel_model.ss}')
    print(f'n_samp_per_vox=: {voxel_model.n_samp_per_vox}')

    if args.eval_fps:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tr_render_opt = {
        'track_max_w': False,
        'output_depth': not args.eval_fps,
        'output_normal': not args.eval_fps,
        'output_T': not args.eval_fps,
    }

    if args.eval_fps:
        # Warmup
        voxel_model.render(views[0], **tr_render_opt)

    eps_time = time.perf_counter()
    psnr_lst = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = voxel_model.render(view, **tr_render_opt)
        if not args.eval_fps:
            rendering = render_pkg['color']
            gt = view.image.cuda()
            mse = (rendering.clip(0,1) - gt.clip(0,1)).square().mean()
            psnr = -10 * torch.log10(mse)
            psnr_lst.append(psnr.item())
            fname = view.image_name

            # RGB
            imageio.imwrite(
                os.path.join(render_path, fname + (".jpg" if args.use_jpg else ".png")),
                im_tensor2np(rendering)
            )
            if args.rgb_only:
                continue
            imageio.imwrite(
                os.path.join(gts_path, fname + ".png"),
                im_tensor2np(gt)
            )
            # Alpha
            imageio.imwrite(
                os.path.join(alpha_path, fname + ".alpha.jpg"),
                im_tensor2np(1-render_pkg['T'])[...,None].repeat(3, axis=-1)
            )
            # Depth
            imageio.imwrite(
                os.path.join(viz_path, fname + ".depth_med_viz.jpg"),
                viz_tensordepth(render_pkg['depth'][2])
            )
            imageio.imwrite(
                os.path.join(viz_path, fname + ".depth_viz.jpg"),
                viz_tensordepth(render_pkg['depth'][0], 1-render_pkg['T'][0])
            )
            # Normal
            depth_med2normal = view.depth2normal(render_pkg['depth'][2])
            depth2normal = view.depth2normal(render_pkg['depth'][0])
            imageio.imwrite(
                os.path.join(viz_path, fname + ".depth_med2normal.jpg"),
                im_tensor2np(depth_med2normal * 0.5 + 0.5)
            )
            imageio.imwrite(
                os.path.join(viz_path, fname + ".depth2normal.jpg"),
                im_tensor2np(depth2normal * 0.5 + 0.5)
            )
            render_normal = render_pkg['normal']
            imageio.imwrite(
                os.path.join(viz_path, fname + ".normal.jpg"),
                im_tensor2np(render_normal * 0.5 + 0.5)
            )
    torch.cuda.synchronize()
    eps_time = time.perf_counter() - eps_time
    peak_mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3
    if args.eval_fps:
        print(f'Resolution:', tuple(render_pkg['color'].shape[-2:]))
        print(f'Eps time: {eps_time:.3f} sec')
        print(f"Peak mem: {peak_mem:.2f} GB")
        print(f'FPS     : {len(views)/eps_time:.0f}')
        outtxt = os.path.join(args.model_path, name, "ours_{}{}.txt".format(iteration, suffix))
        with open(outtxt, 'w') as f:
            f.write(f"n={len(views):.6f}\n")
            f.write(f"eps={eps_time:.6f}\n")
            f.write(f"peak_mem={peak_mem:.2f}\n")
            f.write(f"fps={len(views)/eps_time:.6f}\n")
    else:
        print('PSNR:', np.mean(psnr_lst))


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--eval_fps", action="store_true")
    parser.add_argument("--clear_res_down", action="store_true")
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--rgb_only", action="store_true")
    parser.add_argument("--use_jpg", action="store_true")
    parser.add_argument("--overwrite_ss", default=None, type=float)
    parser.add_argument("--overwrite_n_samp_per_vox", default=None)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    if args.clear_res_down:
        cfg.data.res_downscale = 0
        cfg.data.res_width = 0

    # Load data
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=args.eval_fps,
    )

    # Load model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
        deferred_appearance=cfg.model.deferred_appearance,
        appearance_feat_dim=cfg.model.appearance_feat_dim,
        appearance_hidden_dim=cfg.model.appearance_hidden_dim,
    )
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)

    # Output path suffix
    suffix = args.suffix
    if not args.suffix:
        if cfg.data.res_downscale > 0:
            suffix += f"_r{cfg.data.res_downscale}"
        if cfg.data.res_width > 0:
            suffix += f"_w{cfg.data.res_width}"

    if args.overwrite_ss:
        voxel_model.ss = args.overwrite_ss
        if not args.suffix:
            suffix += f"_ss{args.overwrite_ss:.2f}"
    
    if args.overwrite_n_samp_per_vox:
        voxel_model.n_samp_per_vox = args.overwrite_n_samp_per_vox
        if not args.suffix:
            suffix += f"_{args.overwrite_n_samp_per_vox}"

    voxel_model.freeze_vox_geo()

    if not args.skip_train:
        render_set(
            "train", loaded_iter, suffix, args,
            data_pack.get_train_cameras(), voxel_model)

    if not args.skip_test:
        render_set(
            "test", loaded_iter, suffix, args,
            data_pack.get_test_cameras(), voxel_model)
