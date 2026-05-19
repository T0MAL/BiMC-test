import unittest

import torch
import torch.nn.functional as F

from utils.phase3_scores import (
    apply_hubness_correction,
    compute_prototype_hubness,
    dynamic_alpha_from_scores,
    energy_score,
    normalized_entropy,
    reliability_from_scores,
    top1_margin,
)


class Phase3ScoresTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.scores = torch.randn(5, 7, dtype=torch.float64)
        self.probs = F.softmax(self.scores, dim=-1)

    def test_entropy_returns_batch_shape(self):
        entropy = normalized_entropy(self.probs)
        self.assertEqual(entropy.shape, (5,))

    def test_margin_returns_batch_shape(self):
        margin = top1_margin(self.probs)
        self.assertEqual(margin.shape, (5,))

    def test_energy_returns_batch_shape(self):
        energy = energy_score(self.scores, temp=0.7)
        self.assertEqual(energy.shape, (5,))

    def test_dynamic_alpha_is_clipped(self):
        aux_scores = torch.randn_like(self.scores)
        alpha, stats = dynamic_alpha_from_scores(
            self.scores,
            aux_scores,
            mode="entropy_margin",
            min_alpha=0.2,
            max_alpha=0.8,
        )
        self.assertEqual(alpha.shape, (5,))
        self.assertGreaterEqual(alpha.min().item(), 0.2)
        self.assertLessEqual(alpha.max().item(), 0.8)
        self.assertIn("alpha_mean", stats)

    def test_hubness_returns_class_shape(self):
        protos = F.normalize(torch.randn(7, 11, dtype=torch.float64), dim=-1)
        hubness, stats = compute_prototype_hubness(protos, tau=0.05)
        self.assertEqual(hubness.shape, (7,))
        self.assertIn("hubness_mean", stats)

    def test_corrected_scores_keep_shape(self):
        protos = F.normalize(torch.randn(7, 11, dtype=torch.float64), dim=-1)
        hubness, _ = compute_prototype_hubness(protos)
        corrected = apply_hubness_correction(self.scores, hubness, lambda_h=0.05)
        self.assertEqual(corrected.shape, self.scores.shape)

    def test_no_nan_or_inf(self):
        protos = F.normalize(torch.randn(7, 11, dtype=torch.float64), dim=-1)
        hubness, _ = compute_prototype_hubness(protos)
        corrected = apply_hubness_correction(self.scores, hubness)
        reliability, _ = reliability_from_scores(self.scores, mode="entropy_margin_energy")
        tensors = [
            normalized_entropy(self.probs),
            top1_margin(self.probs),
            energy_score(self.scores),
            corrected,
            reliability,
            hubness,
        ]
        for tensor in tensors:
            self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
