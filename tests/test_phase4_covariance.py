import unittest

import torch
import torch.nn.functional as F

from utils.phase4_covariance import (
    base_borrowed_diag_covariance,
    build_phase4_class_diag_vars,
    compute_base_class_diag_covariances,
    diag_shrinkage_covariance,
    mahalanobis_diag_score,
    safe_diag_variance,
)


class Phase4CovarianceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.dtype = torch.float64
        self.features = torch.randn(12, 6, dtype=self.dtype)
        self.labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.protos = F.normalize(torch.randn(4, 6, dtype=self.dtype), dim=-1)
        self.base_features = {
            0: self.features[self.labels == 0],
            1: self.features[self.labels == 1],
        }

    def test_safe_diag_variance_shape_and_positive(self):
        var = safe_diag_variance(self.features, eps=1e-4)
        self.assertEqual(var.shape, (6,))
        self.assertTrue((var > 0).all())

    def test_base_borrowed_covariance_shape_and_weights(self):
        base_vars = compute_base_class_diag_covariances(self.base_features, self.protos[:2])
        prior, weights, stats = base_borrowed_diag_covariance(
            self.protos[2],
            self.protos[:2],
            base_vars,
            tau=4.0,
        )
        self.assertEqual(prior.shape, (6,))
        self.assertTrue(torch.allclose(weights.sum(), torch.ones((), dtype=self.dtype), atol=1e-6))
        self.assertIn("borrowed_weight_entropy", stats)

    def test_diag_shrinkage_variance_positive(self):
        prior = torch.ones(6, dtype=self.dtype)
        var, stats = diag_shrinkage_covariance(
            self.features[self.labels == 2],
            self.protos[2],
            prior,
            shrinkage_lambda=0.5,
        )
        self.assertEqual(var.shape, (6,))
        self.assertTrue((var > 0).all())
        self.assertEqual(var.dtype, self.dtype)
        self.assertIn("shot_variance_mean", stats)

    def test_diag_mahalanobis_scores_shape(self):
        class_vars, _, _ = build_phase4_class_diag_vars(
            self.features,
            self.labels,
            self.protos,
            num_base_classes=2,
            cov_mode="hybrid_diag",
        )
        scores = mahalanobis_diag_score(self.features[:5], self.protos, class_vars, temp=1.0)
        self.assertEqual(scores.shape, (5, 4))

    def test_no_nan_or_inf_and_preserve_dtype_device(self):
        class_vars, _, _ = build_phase4_class_diag_vars(
            self.features,
            self.labels,
            self.protos,
            num_base_classes=2,
            cov_mode="base_borrowed_diag",
        )
        scores = mahalanobis_diag_score(self.features[:5], self.protos, class_vars)
        tensors = [
            safe_diag_variance(self.features),
            class_vars,
            scores,
        ]
        for tensor in tensors:
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(tensor.dtype, self.dtype)
            self.assertEqual(tensor.device, self.features.device)


if __name__ == "__main__":
    unittest.main()
