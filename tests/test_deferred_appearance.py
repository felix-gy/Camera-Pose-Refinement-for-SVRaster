# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Unit tests for Deferred Neural Appearance in SVRaster.

import os
import sys
import shutil
import tempfile
import unittest
import torch
import numpy as np

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.deferred_appearance import DeferredAppearanceMLP
from src.utils.ray_utils import compute_viewdirs
from src.sparse_voxel_model import SparseVoxelModel
from src.cameras import Camera


class TestDeferredAppearance(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_t1_mlp_architecture_and_residual_zero_init(self):
        """T1 & T2: Verify DeferredAppearanceMLP dimensions and initial near-zero residual."""
        feat_dim = 8
        hidden_dim = 32
        mlp = DeferredAppearanceMLP(feat_dim=feat_dim, hidden_dim=hidden_dim).to(self.device)

        batch_size = 1024
        feats = torch.randn(batch_size, feat_dim, device=self.device) * 0.01
        base_rgb = torch.rand(batch_size, 3, device=self.device)
        viewdirs = torch.randn(batch_size, 3, device=self.device)
        viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)

        residual = mlp(feats, base_rgb, viewdirs)

        # 1. Output shape check
        self.assertEqual(residual.shape, (batch_size, 3))

        # 2. Near-zero initial output check (std=1e-4 on last layer)
        max_abs_val = residual.abs().max().item()
        mean_abs_val = residual.abs().mean().item()
        print(f"[TEST T1/T2] Max initial residual: {max_abs_val:.6f}, Mean: {mean_abs_val:.6f}")
        self.assertLess(max_abs_val, 0.05, "Initial residual should be small so optimization starts at base RGB")
        self.assertLess(mean_abs_val, 0.01, "Mean initial residual should be near 0")

        # 3. Output range check (tanh ensures range in (-1, 1))
        self.assertTrue((residual >= -1.0).all().item() and (residual <= 1.0).all().item())

    def test_t2_ray_utils_compute_viewdirs(self):
        """Verify ray directions are normalized and follow camera orientation."""
        h, w = 64, 64
        w2c = torch.eye(4, device=self.device)
        w2c[2, 3] = 2.0  # Camera placed at (0, 0, -2) looking towards +z
        fovx = np.pi / 3
        fovy = np.pi / 3

        camera = Camera(
            image_name="test_cam",
            w2c=w2c,
            fovx=fovx,
            fovy=fovy,
            cx_p=0.5,
            cy_p=0.5,
            image=torch.zeros(3, h, w),
        )

        viewdirs = compute_viewdirs(camera, h, w)
        self.assertEqual(viewdirs.shape, (h, w, 3))

        norms = torch.linalg.norm(viewdirs, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                        "All computed view directions must be unit vectors")

        center_dir = viewdirs[h // 2, w // 2]
        self.assertGreater(center_dir[2].item(), 0.9, "Center ray should point forwards (+z)")

    def test_t3_model_initialization_with_deferred_appearance(self):
        """Verify SparseVoxelModel configures _feat_grid_pts and appearance_mlp properly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for SparseVoxelModel test")

        model = SparseVoxelModel(
            deferred_appearance=True,
            appearance_feat_dim=8,
            appearance_hidden_dim=32,
            sh_degree=1,  # Should be forced to 0
        )

        self.assertTrue(model.deferred_appearance)
        self.assertEqual(model.max_sh_degree, 0, "max_sh_degree must be forced to 0 when deferred_appearance=True")
        self.assertIn('_feat_grid_pts', model.grid_pts_param_lst)
        self.assertIsNotNone(model.appearance_mlp)

        bounding = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        model.model_init(
            bounding=bounding,
            outside_level=0,
            init_n_level=3,
            appearance_feat_dim=8,
            appearance_feat_std=0.01,
        )

        self.assertIsNotNone(model._feat_grid_pts)
        self.assertEqual(model._feat_grid_pts.shape, (model.num_grid_pts, 8))
        self.assertTrue(model._feat_grid_pts.requires_grad)
        self.assertTrue(model._geo_grid_pts.requires_grad)
        self.assertTrue(model._sh0.requires_grad)

    def test_t4_topological_adaptation_subdivision_and_pruning(self):
        """Verify _feat_grid_pts adapt correctly during voxel subdivision and pruning."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for topological adaptation test")

        model = SparseVoxelModel(
            deferred_appearance=True,
            appearance_feat_dim=8,
            appearance_hidden_dim=32,
        )
        bounding = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        model.model_init(bounding=bounding, outside_level=0, init_n_level=2)

        init_num_voxels = model.num_voxels
        init_num_pts = model.num_grid_pts

        # 1. Test Subdivision
        subdivide_mask = torch.zeros(init_num_voxels, dtype=torch.bool, device="cuda")
        subdivide_mask[:4] = True
        model.subdividing(subdivide_mask)

        new_num_voxels = model.num_voxels
        new_num_pts = model.num_grid_pts
        self.assertEqual(new_num_voxels, init_num_voxels + 4 * 7)
        self.assertEqual(model._feat_grid_pts.shape, (new_num_pts, 8))
        self.assertFalse(torch.isnan(model._feat_grid_pts).any(), "No NaNs after subdivision in _feat_grid_pts")
        self.assertTrue(model._feat_grid_pts.requires_grad)

        # 2. Test Pruning
        prune_mask = torch.zeros(new_num_voxels, dtype=torch.bool, device="cuda")
        prune_mask[:2] = True
        model.pruning(prune_mask)

        pruned_num_voxels = model.num_voxels
        self.assertEqual(pruned_num_voxels, new_num_voxels - 2)
        self.assertEqual(model._feat_grid_pts.shape, (model.num_grid_pts, 8))
        self.assertFalse(torch.isnan(model._feat_grid_pts).any(), "No NaNs after pruning in _feat_grid_pts")

    def test_t5_save_and_load_parity(self):
        """Verify that saving and loading checkpoints preserves _feat_grid_pts and appearance_mlp weights exactly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for save/load test")

        temp_dir = tempfile.mkdtemp()
        try:
            model = SparseVoxelModel(
                deferred_appearance=True,
                appearance_feat_dim=8,
                appearance_hidden_dim=32,
            )
            bounding = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
            model.model_init(bounding=bounding, outside_level=0, init_n_level=2)

            with torch.no_grad():
                model._feat_grid_pts.fill_(0.42)
                for p in model.appearance_mlp.parameters():
                    p.fill_(0.123)

            save_path = os.path.join(temp_dir, "test_model")
            model.save(save_path)

            loaded_model = SparseVoxelModel(
                deferred_appearance=True,
                appearance_feat_dim=8,
                appearance_hidden_dim=32,
            )
            loaded_model.load(save_path)

            self.assertEqual(loaded_model.num_grid_pts, model.num_grid_pts)
            self.assertEqual(loaded_model.num_voxels, model.num_voxels)
            self.assertIsNotNone(loaded_model._feat_grid_pts)
            self.assertTrue(torch.allclose(loaded_model._feat_grid_pts, model._feat_grid_pts))

            for p1, p2 in zip(model.appearance_mlp.parameters(), loaded_model.appearance_mlp.parameters()):
                self.assertTrue(torch.allclose(p1, p2), "MLP parameters should match exactly after loading")

            print("[TEST T5] Save and load parity verified successfully!")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
