# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import torch
import svraster_cuda
from src.utils.camera_utils import se3_to_SE3


class CameraBase:

    '''
    Base class of perspective cameras.
    '''

    def __repr__(self):
        clsname = self.__class__.__name__
        fname = f"image_name='{self.image_name}'"
        res = f"HW=({self.image_height}x{self.image_width})"
        fov = f"fovx={np.rad2deg(self.fovx):.1f}deg"
        return f"{clsname}({fname}, {res}, {fov})"

    @property
    def lookat(self):
        return self.c2w[:3, 2]

    @property
    def position(self):
        return self.c2w[:3, 3]

    @property
    def down(self):
        return self.c2w[:3, 1]

    @property
    def right(self):
        return self.c2w[:3, 0]

    @property
    def cx(self):
        return self.image_width * self.cx_p

    @property
    def cy(self):
        return self.image_height * self.cy_p

    @property
    def pix_size(self):
        return 2 * self.tanfovx / self.image_width

    @property
    def tanfovx(self):
        return np.tan(self.fovx * 0.5)

    @property
    def tanfovy(self):
        return np.tan(self.fovy * 0.5)

    def compute_rd(self, wh=None, cxcy=None, device=None):
        '''Ray directions in world space.'''
        if wh is None:
            wh = (self.image_width, self.image_height)
        if cxcy is None:
            cxcy = (self.cx * wh[0] / self.image_width, self.cy * wh[1] / self.image_height)
        rd = svraster_cuda.utils.compute_rd(
            width=wh[0], height=wh[1],
            cx=cxcy[0], cy=cxcy[1],
            tanfovx=self.tanfovx, tanfovy=self.tanfovy,
            c2w_matrix=self.c2w.cuda())
        rd = rd.to(device if device is None else self.c2w.device)
        return rd

    def project(self, pts, return_depth=False):
        # Return normalized image coordinate in [-1, 1]
        cam_pts = pts @ self.w2c[:3, :3].T + self.w2c[:3, 3]
        depth = cam_pts[:, [2]]
        cam_uv = cam_pts[:, :2] / depth
        scale_x = 1 / self.tanfovx
        scale_y = 1 / self.tanfovy
        shift_x = 2 * self.cx_p - 1
        shift_y = 2 * self.cy_p - 1
        cam_uv[:, 0] = cam_uv[:, 0] * scale_x + shift_x
        cam_uv[:, 1] = cam_uv[:, 1] * scale_y + shift_y
        if return_depth:
            return cam_uv, depth
        return cam_uv

    def depth2pts(self, depth):
        device = depth.device
        h, w = depth.shape[-2:]
        rd = self.compute_rd(wh=(w, h), device=device)
        return self.position.view(3,1,1).to(device) + rd * depth

    def depth2normal(self, depth, ks=3, tol_cos=-1):
        assert ks % 2 == 1
        pad = ks // 2
        ks_1 = ks - 1
        pts = self.depth2pts(depth)
        normal_pseudo = torch.zeros_like(pts)
        dx = pts[:, pad:-pad, ks_1:] - pts[:, pad:-pad, :-ks_1]
        dy = pts[:, ks_1:, pad:-pad] - pts[:, :-ks_1, pad:-pad]
        normal_pseudo[:, pad:-pad, pad:-pad] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=0), dim=0)

        if tol_cos > 0:
            with torch.no_grad():
                pts_dir = torch.nn.functional.normalize(pts - self.position.view(3,1,1), dim=0)
                dot = (normal_pseudo * pts_dir).sum(0)
                mask = (dot > tol_cos)
            normal_pseudo = normal_pseudo * mask

        return normal_pseudo


class Camera(CameraBase):
    def __init__(
            self, image_name,
            w2c, fovx, fovy, cx_p, cy_p,
            near=0.02,
            image=None, mask=None, depth=None,
            sparse_pt=None,
            c2w_gt=None):

        self.image_name = image_name

        # Camera parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.w2c = torch.as_tensor(w2c, dtype=torch.float32, device=device)
        self.c2w = self.w2c.inverse().contiguous()
        self.c2w_gt = torch.as_tensor(c2w_gt, dtype=torch.float32, device=device) if c2w_gt is not None else None

        self.fovx = fovx
        self.fovy = fovy

        # Load frame
        self.image = image.cpu()

        # Other camera parameters
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        self.cx_p = (0.5 if cx_p is None else cx_p)
        self.cy_p = (0.5 if cy_p is None else cy_p)
        self.near = near

        # Load mask and depth if there are
        self.mask = mask.cpu() if mask is not None else None
        self.depth = depth.cpu() if depth is not None else None

        # Load sparse depth
        if sparse_pt is not None:
            self.sparse_pt = torch.tensor(sparse_pt, dtype=torch.float32, device="cpu")
        else:
            self.sparse_pt = None

    def to(self, device):
        self.image = self.image.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        return self

    def auto_exposure_init(self):
        self._exposure_A = torch.eye(3, dtype=torch.float32, device="cuda")
        self._exposure_t = torch.zeros([3,1,1], dtype=torch.float32, device="cuda")
        self.exposure_updated = False

    def auto_exposure_apply(self, image):
        if self.exposure_updated:
            image = torch.einsum('ij,jhw->ihw', self._exposure_A, image) + self._exposure_t
        return image

    def auto_exposure_update(self, ren, ref):
        self.exposure_updated = True
        self._exposure_A.requires_grad_()
        self._exposure_t.requires_grad_()
        optim = torch.optim.Adam([self._exposure_A, self._exposure_t], lr=1e-3)
        for _ in range(100):
            loss = (self.auto_exposure_apply(ren).clamp(0, 1) - ref).abs().mean()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        self._exposure_A.requires_grad_(False)
        self._exposure_t.requires_grad_(False)

    def clone_mini(self):
        return MiniCam(
            c2w=self.c2w.clone(),
            fovx=self.fovx, fovy=self.fovy,
            width=self.image_width, height=self.image_height,
            near=self.near,
            cx_p=self.cx_p, cy_p=self.cy_p)


class MiniCam(CameraBase):
    def __init__(self,
            c2w, fovx, fovy,
            width, height,
            near=0.02,
            cx_p=None, cy_p=None,
            image_name="minicam"):

        self.image_name = image_name
        self.c2w = torch.tensor(c2w).clone().cuda()
        self.w2c = self.c2w.inverse()

        self.fovx = fovx
        self.fovy = fovy
        self.image_width = width
        self.image_height = height
        self.cx_p = (0.5 if cx_p is None else cx_p)
        self.cy_p = (0.5 if cy_p is None else cy_p)
        self.near = near

        self.depth = None
        self.mask = None

    def clone_mini(self):
        return MiniCam(
            c2w=self.c2w.clone(),
            fovx=self.fovx, fovy=self.fovy,
            width=self.image_width, height=self.image_height,
            near=self.near,
            cx_p=self.cx_p, cy_p=self.cy_p)

    def move_forward(self, dist):
        new_position = self.position + dist * self.lookat
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_up(self, dist):
        return self.move_down(-dist)

    def move_down(self, dist):
        new_position = self.position + dist * self.down
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_right(self, dist):
        new_position = self.position + dist * self.right
        self.c2w[:3, 3] = new_position
        self.w2c = self.c2w.inverse()
        return self

    def move_left(self, dist):
        return self.move_right(-dist)

    def rotate(self, R):
        self.c2w[:3, :3] = (R @ self.w2c[:3, :3]).T
        self.w2c = self.c2w.inverse()
        return self

    def rotate_x(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)

    def rotate_y(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [np.cos(rad), 0, -np.sin(rad)],
            [0, 1, 0],
            [np.sin(rad), 0, np.cos(rad)],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)

    def rotate_z(self, rad=None, deg=None):
        assert rad is None or deg is None, "Can only specify rotation by either rad or deg."
        if rad is None:
            rad = np.deg2rad(deg)
        R = torch.tensor([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1],
        ], dtype=torch.float32, device="cuda")
        return self.rotate(R)


class CameraPoseOptimizer(torch.nn.Module):
    def __init__(self, num_cams: int, init_c2w_list=None, mode: str = "colmap", device=None):
        """
        Camera pose refinement module using se(3) Lie algebra parametrization.
        Optimizes a 6 DoF residual per camera: c2w_opt = c2w_base @ exp(se3_refine).

        Args:
            num_cams: Total number of training cameras.
            init_c2w_list: List of Camera objects or tensor of [N, 4, 4] initial c2w matrices.
            mode: 'colmap' (refine relative to provided initial poses)
                  'identity' (initialize all cameras at world origin I_4x4)
            device: Target device ('cuda', 'cpu', etc.). Defaults to cuda if available.
        """
        super().__init__()
        self.num_cams = num_cams
        self.mode = mode

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Learnable 6 DoF se(3) tangent vectors per camera, initialized to zeros (exp(0) = I)
        self.se3_refine = torch.nn.Embedding(num_cams, 6, device=device)
        torch.nn.init.zeros_(self.se3_refine.weight)

        # Store fixed base poses
        if mode == "colmap" and init_c2w_list is not None:
            if hasattr(init_c2w_list[0], 'c2w'):
                base_poses = torch.stack([cam.c2w.detach().clone().float().to(device) for cam in init_c2w_list])
            else:
                base_poses = torch.stack([torch.as_tensor(p, dtype=torch.float32, device=device) for p in init_c2w_list])
        else:
            # Identity initialization: all cameras start at world origin I_4x4
            base_poses = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_cams, 1, 1)

        self.register_buffer("base_c2w", base_poses)

    def get_c2w(self, cam_idx):
        """
        Computes the optimized camera-to-world (c2w) matrix for a given camera index:
        c2w_opt = c2w_base @ exp(se3_refine)
        """
        if not torch.is_tensor(cam_idx):
            cam_idx = torch.tensor(cam_idx, dtype=torch.long, device=self.base_c2w.device)
        elif cam_idx.device != self.base_c2w.device:
            cam_idx = cam_idx.to(self.base_c2w.device)

        delta_T = se3_to_SE3(self.se3_refine(cam_idx))
        c2w_opt = self.base_c2w[cam_idx] @ delta_T
        return c2w_opt

    def get_all_c2w(self):
        """Returns all optimized c2w matrices [N, 4, 4]."""
        all_indices = torch.arange(self.num_cams, device=self.base_c2w.device)
        delta_T = se3_to_SE3(self.se3_refine(all_indices))
        return self.base_c2w @ delta_T

    def state_dict_poses(self):
        return {
            'se3_refine': self.se3_refine.weight.detach().cpu(),
            'base_c2w': self.base_c2w.detach().cpu(),
            'mode': self.mode,
        }

    def load_state_dict_poses(self, state):
        self.se3_refine.weight.data.copy_(state['se3_refine'].to(self.se3_refine.weight.device))
        if 'base_c2w' in state:
            self.base_c2w.copy_(state['base_c2w'].to(self.base_c2w.device))
