import os
import unittest

import torch
import torch.nn.functional as F

from main import setup_cfg
from utils.phase5_space import (
    common_direction_removal,
    diagonal_whitening_apply,
    diagonal_whitening_fit,
    estimate_common_direction,
    lda_apply,
    lda_shrinkage_fit,
    normalize,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Phase5SpaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def _features(self, n=8, d=12, dtype=torch.float64):
        return F.normalize(torch.randn(n, d, dtype=dtype), dim=-1)

    def test_common_direction_removal_shape_normalization_and_stats(self):
        x = self._features()
        direction, stats = estimate_common_direction(prototypes=x, source="prototypes")

        out, apply_stats = common_direction_removal(x, direction, rho=0.25, return_stats=True)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(out.shape[0], dtype=out.dtype), atol=1e-6))
        self.assertAlmostEqual(stats["common_direction_norm"], 1.0, places=5)
        self.assertEqual(apply_stats["transform"], "common_direction_removal")

    def test_diagonal_whitening_shape_normalization_and_stats(self):
        features = self._features(n=10, d=7)
        transform = diagonal_whitening_fit(features, eps=1e-4)

        out = diagonal_whitening_apply(features, transform)

        self.assertEqual(out.shape, features.shape)
        self.assertEqual(out.dtype, features.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(out.shape[0], dtype=out.dtype), atol=1e-6))
        self.assertIn("whitening_var_mean", transform["stats"])

    def test_lda_shape_normalization_and_no_nan(self):
        base = {
            0: self._features(n=5, d=10),
            1: self._features(n=5, d=10),
            2: self._features(n=5, d=10),
        }
        support = {
            3: self._features(n=3, d=10),
        }

        transform = lda_shrinkage_fit(base, support, dim=3, gamma=1e-3, novel_weight=0.1)
        query = self._features(n=4, d=10)
        out = lda_apply(query, transform)

        self.assertEqual(out.shape[0], query.shape[0])
        self.assertLessEqual(out.shape[1], 3)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(out.shape[0], dtype=out.dtype), atol=1e-6))
        self.assertGreater(transform["stats"]["lda_dim_used"], 0)

    def test_normalize_avoids_nan_for_zero_vectors(self):
        x = torch.zeros(3, 5)
        out = normalize(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.all(out == 0))

    def test_default_phase567_config_disabled(self):
        cfg = setup_cfg(
            os.path.join(ROOT, "configs", "datasets", "cifar100.yaml"),
            os.path.join(ROOT, "configs", "trainers", "bimc.yaml"),
        )

        self.assertFalse(cfg.TRAINER.BiMC.PHASE5.ENABLED)
        self.assertEqual(cfg.TRAINER.BiMC.PHASE5.SPACE_TRANSFORM, "none")
        self.assertFalse(cfg.TRAINER.BiMC.PHASE6.ENABLED)
        self.assertEqual(cfg.TRAINER.BiMC.PHASE6.ALIGNMENT_MODE, "none")
        self.assertFalse(cfg.TRAINER.BiMC.PHASE7.ENABLED)
        self.assertEqual(cfg.TRAINER.BiMC.PHASE7.SEPARATION_MODE, "none")
        self.assertEqual(cfg.TRAINER.BiMC.FUSION_BETA_MODE, "fixed")
        self.assertEqual(cfg.TRAINER.BiMC.FUSION_GEOMETRY, "linear")


if __name__ == "__main__":
    unittest.main()
