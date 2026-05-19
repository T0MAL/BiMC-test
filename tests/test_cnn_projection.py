import unittest

import torch

from utils.phase1_fusion import project_cnn_features_to_clip


class CnnProjectionTests(unittest.TestCase):
    def test_projection_returns_clip_dimensional_normalized_features(self):
        torch.manual_seed(13)
        cnn_features = torch.randn(6, 11)

        projected = project_cnn_features_to_clip(
            cnn_features,
            clip_dim=7,
            projection="random_orthogonal",
            seed=0,
        )

        self.assertEqual(projected.shape, (6, 7))
        self.assertTrue(torch.allclose(projected.norm(dim=-1), torch.ones(6), atol=1e-5))

    def test_projection_is_deterministic(self):
        torch.manual_seed(17)
        cnn_features = torch.randn(4, 5)

        first = project_cnn_features_to_clip(cnn_features, clip_dim=9, seed=123)
        second = project_cnn_features_to_clip(cnn_features, clip_dim=9, seed=123)

        self.assertTrue(torch.allclose(first, second, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
