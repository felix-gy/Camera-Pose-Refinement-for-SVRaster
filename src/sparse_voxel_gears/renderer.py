# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import svraster_cuda

from src.utils.image_utils import resize_rendering

class SVRenderer:

    def freeze_vox_geo(self):
        '''
        Freeze grid points parameter and pre-gather them to each voxel.
        '''
        with torch.no_grad():
            self.frozen_vox_geo = svraster_cuda.renderer.GatherGeoParams.apply(
                self.vox_key,
                torch.arange(self.num_voxels, device="cuda"),
                self._geo_grid_pts
            )
        self._geo_grid_pts.requires_grad = False

    def unfreeze_vox_geo(self):
        '''
        Unfreeze grid points parameter.
        '''
        del self.frozen_vox_geo
        self._geo_grid_pts.requires_grad = True

    def vox_fn(self, idx, cam_pos, color_mode=None, viewdir=None):
        '''
        Per-frame voxel property processing. Two important operations:
        1. Gather grid points parameter into each voxel.
        2. Compute view-dependent color of each voxel.

        Input:
            @idx        Indices for active voxel for current frame.
            @cam_pos    Camera position.
        Output:
            @vox_params A dictionary of the pre-process voxel properties.
        '''

        # Gather the density values at the eight corners of each voxel.
        # It defined a trilinear density field.
        # The final tensor are in shape [#vox, 8]
        if hasattr(self, 'frozen_vox_geo'):
            geos = self.frozen_vox_geo
        else:
            geos = svraster_cuda.renderer.GatherGeoParams.apply(
                self.vox_key,
                idx,
                self._geo_grid_pts
            )

        # Compute voxel colors
        if color_mode is None or color_mode == "sh":
            active_sh_degree = self.active_sh_degree
            color_mode = "sh"
        elif color_mode.startswith("sh"):
            active_sh_degree = int(color_mode[2])
            color_mode = "sh"

        if color_mode == "sh":
            rgbs = svraster_cuda.renderer.SH_eval.apply(
                active_sh_degree,
                idx,
                self.vox_center,
                cam_pos,
                viewdir, # Ignore above two when viewdir is not None
                self.sh0,
                self.shs,
            )
        elif color_mode == "rand":
            rgbs = torch.rand([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        elif color_mode == "dontcare":
            rgbs = torch.empty([self.num_voxels, 3], dtype=torch.float32, device="cuda")
        else:
            raise NotImplementedError

        # Gather appearance features if present
        if hasattr(self, '_feat_grid_pts') and self._feat_grid_pts is not None:
            feats = svraster_cuda.renderer.GatherFeatParams.apply(
                self.vox_key,
                idx,
                self._feat_grid_pts
            )
        else:
            feats = None

        # Pack everything
        vox_params = {
            'geos': geos,
            'rgbs': rgbs,
            'feats': feats,
            'subdiv_p': self._subdiv_p, # Dummy param to record subdivision priority
        }
        if vox_params['subdiv_p'] is None:
            vox_params['subdiv_p'] = torch.ones([self.num_voxels, 1], device="cuda")

        return vox_params

    def render(
            self,
            camera,
            color_mode=None,
            track_max_w=False,
            ss=None,
            output_depth=False,
            output_normal=False,
            output_T=False,
            rand_bg=False,
            use_auto_exposure=False,
            **other_opt):

        ###################################
        # Pre-processing
        ###################################
        if ss is None:
            ss = self.ss
        w_src, h_src = camera.image_width, camera.image_height
        w, h = round(w_src * ss), round(h_src * ss)
        w_ss, h_ss = w / w_src, h / h_src
        if ss != 1.0 and 'gt_color' in other_opt:
            other_opt['gt_color'] = resize_rendering(other_opt['gt_color'], size=(h, w))

        n_samp_per_vox = other_opt.pop('n_samp_per_vox', self.n_samp_per_vox)

        ###################################
        # Call low-level rasterization API
        ###################################
        raster_settings = svraster_cuda.renderer.RasterSettings(
            color_mode=color_mode,
            n_samp_per_vox=n_samp_per_vox,
            image_width=w,
            image_height=h,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            cx=camera.cx * w_ss,
            cy=camera.cy * h_ss,
            w2c_matrix=camera.w2c,
            c2w_matrix=camera.c2w,
            bg_color=float(self.white_background),
            near=camera.near,
            need_depth=output_depth,
            need_normal=output_normal,
            track_max_w=track_max_w,
            **other_opt)
        color, feat_img, depth, normal, T, max_w = svraster_cuda.renderer.rasterize_voxels(
            raster_settings,
            self.octpath,
            self.vox_center,
            self.vox_size,
            self.vox_fn)

        ###################################
        # Deferred Neural Appearance Evaluation
        ###################################
        base_color = color
        residual = None
        if getattr(self, 'deferred_appearance', False) and feat_img is not None and feat_img.numel() > 0 and getattr(self, 'appearance_mlp', None) is not None:
            from src.utils.ray_utils import compute_viewdirs
            viewdirs = compute_viewdirs(camera, h, w)  # [H, W, 3]
            feat_flat = feat_img.permute(1, 2, 0).reshape(-1, self.appearance_feat_dim)
            base_flat = base_color.permute(1, 2, 0).reshape(-1, 3)
            dir_flat = viewdirs.reshape(-1, 3)

            residual = self.appearance_mlp(feat_flat, base_flat, dir_flat)
            residual = residual.reshape(h, w, 3).permute(2, 0, 1)
            color = base_color + residual

        ###################################
        # Post-processing and pack output
        ###################################
        if rand_bg:
            bg_noise = torch.rand_like(color, requires_grad=False)
            color = color + T * bg_noise
            base_color = base_color + T * bg_noise
        elif not self.white_background and not self.black_background:
            color = color + T * color.mean((1,2), keepdim=True)
            base_color = base_color + T * base_color.mean((1,2), keepdim=True)

        if use_auto_exposure:
            color = camera.auto_exposure_apply(color)
            base_color = camera.auto_exposure_apply(base_color)

        render_pkg = {
            'color': color,
            'depth': depth if output_depth else None,
            'normal': normal if output_normal else None,
            'T': T if output_T else None,
            'max_w': max_w,
            'base_color': base_color,
            'residual': residual,
            'feature': feat_img,
            'view_residual': residual,
        }

        for k in ['color', 'depth', 'normal', 'T']:
            render_pkg[f'raw_{k}'] = render_pkg[k]

            # Post process super-sampling
            if render_pkg[k] is not None and render_pkg[k].shape[-2:] != (h_src, w_src):
                render_pkg[k] = resize_rendering(render_pkg[k], size=(h_src, w_src))

        # Clip intensity
        render_pkg['color'] = render_pkg['color'].clamp(0, 1)

        return render_pkg
