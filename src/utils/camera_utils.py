# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from scipy.interpolate import make_interp_spline


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(0.5 * fov))

def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def interpolate_poses(poses, n_frame, periodic=True):

    assert len(poses) > 1

    poses = list(poses)
    bc_type = None

    if periodic:
        poses.append(poses[0])
        bc_type = "periodic"

    pos_lst = np.stack([pose[:3, 3] for pose in poses])
    lookat_lst = np.stack([pose[:3, 2] for pose in poses])
    right_lst = np.stack([pose[:3, 0] for pose in poses])

    ts = np.linspace(0, 1, len(poses))
    pos_interp_f = make_interp_spline(ts, pos_lst, bc_type=bc_type)
    lookat_interp_f = make_interp_spline(ts, lookat_lst, bc_type=bc_type)
    right_interp_f = make_interp_spline(ts, right_lst, bc_type=bc_type)

    samps = np.linspace(0, 1, n_frame+1)[:n_frame]
    pos_video = pos_interp_f(samps)
    lookat_video = lookat_interp_f(samps)
    right_video = right_interp_f(samps)
    interp_poses = []
    for i in range(n_frame):
        pos = pos_video[i]
        lookat = lookat_video[i] / np.linalg.norm(lookat_video[i])
        right_ = right_video[i] / np.linalg.norm(right_video[i])
        down = np.cross(lookat, right_)
        right = np.cross(down, lookat)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = lookat
        c2w[:3, 3] = pos
        interp_poses.append(c2w)

    return interp_poses


def gen_circular_poses(radius,
                       n_frame,
                       starting=1.5 * np.pi, # Starting from -z
                       ):
    poses = []
    for rad in np.linspace(starting, starting + 2 * np.pi, n_frame):
        pos = radius * np.array([np.cos(rad), 0, np.sin(rad)])
        lookat = -pos / np.linalg.norm(pos)
        down = np.array([0, 1, 0])
        right = np.cross(down, lookat)
        right = right / np.linalg.norm(right)
        down = np.cross(lookat, right)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = lookat
        c2w[:3, 3] = pos
        poses.append(c2w)
    return poses


# =========================================================================
# Lie Algebra SE(3) and Camera Pose Optimization Utilities
# =========================================================================

import torch


def skew_symmetric(w: torch.Tensor) -> torch.Tensor:
    """
    Computes the skew-symmetric matrix [w]_x from a 3D vector w.
    Shape: [..., 3] -> [..., 3, 3]
    """
    w0, w1, w2 = w.unbind(-1)
    zeros = torch.zeros_like(w0)
    wx = torch.stack([
        torch.stack([zeros, -w2, w1], dim=-1),
        torch.stack([w2, zeros, -w0], dim=-1),
        torch.stack([-w1, w0, zeros], dim=-1)
    ], dim=-2)
    return wx


def so3_to_SO3(w: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) vector to SO(3) 3x3 rotation matrix using Rodrigues formula.
    Shape: [..., 3] -> [..., 3, 3]
    """
    theta = w.norm(dim=-1, keepdim=True).unsqueeze(-1)
    theta_sq = theta ** 2

    # Taylor expansions to avoid division by zero
    small_angle = theta < 1e-5
    A = torch.where(small_angle, 1.0 - theta_sq / 6.0, torch.sin(theta) / (theta + 1e-8))
    B = torch.where(small_angle, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + 1e-8))

    wx = skew_symmetric(w)
    I = torch.eye(3, device=w.device, dtype=torch.float32).expand(*w.shape[:-1], 3, 3)
    R = I + A * wx + B * (wx @ wx)
    return R


def se3_to_SE3(wu: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) tangent vector (6 DoF) to SE(3) 4x4 transformation matrix.
    wu: [..., 6] where wu[..., :3] is rotation vector w, wu[..., 3:] is translation vector u.
    Returns: [..., 4, 4] homogeneous matrix.
    """
    w, u = wu.split([3, 3], dim=-1)
    theta = w.norm(dim=-1, keepdim=True).unsqueeze(-1)
    theta_sq = theta ** 2

    # Taylor expansions for stability
    small_angle = theta < 1e-5
    A = torch.where(small_angle, 1.0 - theta_sq / 6.0, torch.sin(theta) / (theta + 1e-8))
    B = torch.where(small_angle, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + 1e-8))
    C = torch.where(small_angle, 1.0 / 6.0 - theta_sq / 120.0, (theta - torch.sin(theta)) / (theta ** 3 + 1e-8))

    wx = skew_symmetric(w)
    I = torch.eye(3, device=wu.device, dtype=torch.float32).expand(*wu.shape[:-1], 3, 3)
    R = I + A * wx + B * (wx @ wx)
    V = I + B * wx + C * (wx @ wx)
    t = (V @ u.unsqueeze(-1)).squeeze(-1)

    T = torch.eye(4, device=wu.device, dtype=torch.float32).repeat(*wu.shape[:-1], 1, 1)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def umeyama_alignment(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """
    Computes optimal Sim(3) transformation (s, R, t) such that s*R*X + t ~= Y
    using the Umeyama algorithm. X, Y shape: [N, 3].
    """
    assert X.shape == Y.shape and len(X.shape) == 2 and X.shape[1] == 3
    n, m = X.shape

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    sigma_x = np.mean(np.sum((X - mu_x) ** 2, axis=1))

    # Cross-covariance matrix
    H = (Y - mu_y).T @ (X - mu_x) / n
    U, D, Vt = np.linalg.svd(H)

    d = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, np.sign(d)])

    R = U @ S @ Vt
    s = (1.0 / sigma_x) * np.trace(np.diag(D) @ S) if with_scale else 1.0
    t = mu_y - s * R @ mu_x

    return s, R, t


def compute_ate(pred_c2ws, gt_c2ws, align: bool = True):
    """
    Computes Absolute Trajectory Error (ATE) between estimated and GT camera poses.
    Returns:
        ate_trans: Translation RMSE
        ate_rot: Rotation RMSE in degrees
    """
    if torch.is_tensor(pred_c2ws):
        pred_c2ws = pred_c2ws.detach().cpu().numpy()
    if torch.is_tensor(gt_c2ws):
        gt_c2ws = gt_c2ws.detach().cpu().numpy()

    preds = np.asarray(pred_c2ws)
    gts = np.asarray(gt_c2ws)

    t_preds = preds[:, :3, 3]
    t_gts = gts[:, :3, 3]

    if align and len(t_preds) >= 3:
        s, R_align, t_align = umeyama_alignment(t_preds, t_gts, with_scale=True)
        t_aligned = s * (preds[:, :3, 3] @ R_align.T) + t_align
        R_aligned = R_align @ preds[:, :3, :3]
    else:
        t_aligned = t_preds
        R_aligned = preds[:, :3, :3]

    trans_errors = np.linalg.norm(t_aligned - t_gts, axis=1)
    ate_trans = np.sqrt(np.mean(trans_errors ** 2))

    # Rotation errors
    R_gts = gts[:, :3, :3]
    R_diff = np.einsum('nij,njk->nik', R_aligned, R_gts.transpose(0, 2, 1))
    traces = np.clip((np.trace(R_diff, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rot_errors_deg = np.rad2deg(np.arccos(traces))
    ate_rot = np.sqrt(np.mean(rot_errors_deg ** 2))

    return {
        'trans_rmse': float(ate_trans),
        'rot_rmse_deg': float(ate_rot)
    }


def compute_rpe(pred_c2ws, gt_c2ws):
    """
    Computes Relative Pose Error (RPE) between consecutive frames.
    """
    if torch.is_tensor(pred_c2ws):
        pred_c2ws = pred_c2ws.detach().cpu().numpy()
    if torch.is_tensor(gt_c2ws):
        gt_c2ws = gt_c2ws.detach().cpu().numpy()

    preds = np.asarray(pred_c2ws)
    gts = np.asarray(gt_c2ws)

    n = len(preds)
    if n < 2:
        return {'trans_rmse': 0.0, 'rot_rmse_deg': 0.0}

    rpe_trans_lst = []
    rpe_rot_lst = []

    for i in range(n - 1):
        # Relative transformation for pred
        rel_pred = np.linalg.inv(preds[i]) @ preds[i + 1]
        rel_gt = np.linalg.inv(gts[i]) @ gts[i + 1]

        rel_error = np.linalg.inv(rel_gt) @ rel_pred
        rpe_trans_lst.append(np.linalg.norm(rel_error[:3, 3]))

        R_err = rel_error[:3, :3]
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rpe_rot_lst.append(np.rad2deg(np.arccos(cos_angle)))

    rpe_trans = np.sqrt(np.mean(np.array(rpe_trans_lst) ** 2))
    rpe_rot = np.sqrt(np.mean(np.array(rpe_rot_lst) ** 2))

    return {
        'trans_rmse': float(rpe_trans),
        'rot_rmse_deg': float(rpe_rot)
    }
