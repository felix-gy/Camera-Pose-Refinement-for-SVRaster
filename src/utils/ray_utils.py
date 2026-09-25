import torch


def compute_viewdirs(camera, height, width):
    """
    Computes normalized camera ray view directions [height, width, 3] in world space,
    following the exact camera coordinate convention of SVRaster (matching compute_rd_kernel in CUDA).
    """
    device = camera.c2w.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    w_ss = width / camera.image_width
    h_ss = height / camera.image_height
    cx = camera.cx * w_ss
    cy = camera.cy * h_ss

    cam_x = (x + 0.5 - cx) * 2.0 * camera.tanfovx / float(width)
    cam_y = (y + 0.5 - cy) * 2.0 * camera.tanfovy / float(height)
    cam_z = torch.ones_like(cam_x)
    cam_dirs = torch.stack([cam_x, cam_y, cam_z], dim=-1)  # [H, W, 3]

    # Rotate by c2w 3x3 rotation: world_dir = cam_dirs @ R.T
    rot = camera.c2w[:3, :3]
    world_dirs = torch.matmul(cam_dirs, rot.T)  # [H, W, 3]
    norm = torch.linalg.norm(world_dirs, dim=-1, keepdim=True) + 1e-8
    return world_dirs / norm
