import inspect
import unittest

import torch
import torch.nn.functional as F

from utils.phase1_fusion import compute_cnn_query_reliability_beta


class CnnBetaLogicTests(unittest.TestCase):
    def test_beta_decreases_when_cnn_visual_reliability_is_higher(self):
        query_clip = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        text_proto = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6]]), dim=-1)
        query_cnn = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
        high_reliability_cnn_proto = F.normalize(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), dim=-1)
        low_reliability_cnn_proto = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)

        beta_high_visual = compute_cnn_query_reliability_beta(
            query_clip,
            text_proto,
            query_cnn,
            high_reliability_cnn_proto,
            reliability_mode="margin",
            beta_clip_min=0.0,
            beta_clip_max=1.0,
            cnn_topk=2,
        )
        beta_low_visual = compute_cnn_query_reliability_beta(
            query_clip,
            text_proto,
            query_cnn,
            low_reliability_cnn_proto,
            reliability_mode="margin",
            beta_clip_min=0.0,
            beta_clip_max=1.0,
            cnn_topk=2,
        )

        self.assertLess(beta_high_visual.item(), beta_low_visual.item())

    def test_cnn_beta_does_not_require_query_labels(self):
        parameters = inspect.signature(compute_cnn_query_reliability_beta).parameters
        self.assertNotIn("query_labels", parameters)
        self.assertNotIn("test_labels", parameters)

        query_clip = F.normalize(torch.randn(3, 4), dim=-1)
        text_proto = F.normalize(torch.randn(5, 4), dim=-1)
        query_cnn = F.normalize(torch.randn(3, 6), dim=-1)
        cnn_proto = F.normalize(torch.randn(5, 6), dim=-1)

        beta = compute_cnn_query_reliability_beta(
            query_clip,
            text_proto,
            query_cnn,
            cnn_proto,
            reliability_mode="entropy_margin",
            beta_clip_min=0.05,
            beta_clip_max=0.95,
            cnn_topk=3,
        )

        self.assertEqual(beta.shape, (3,))
        self.assertGreaterEqual(beta.min().item(), 0.05)
        self.assertLessEqual(beta.max().item(), 0.95)

    def test_cnn_beta_accepts_mixed_clip_and_cnn_dtypes(self):
        query_clip = F.normalize(torch.randn(3, 4, dtype=torch.float32), dim=-1)
        text_proto = F.normalize(torch.randn(5, 4, dtype=torch.float64), dim=-1)
        query_cnn = F.normalize(torch.randn(3, 6, dtype=torch.float32), dim=-1)
        cnn_proto = F.normalize(torch.randn(5, 6, dtype=torch.float64), dim=-1)

        beta = compute_cnn_query_reliability_beta(
            query_clip,
            text_proto,
            query_cnn,
            cnn_proto,
            reliability_mode="entropy_margin",
            beta_clip_min=0.05,
            beta_clip_max=0.95,
            cnn_topk=3,
        )

        self.assertEqual(beta.dtype, query_clip.dtype)
        self.assertEqual(beta.shape, (3,))
        self.assertTrue(torch.isfinite(beta).all())


if __name__ == "__main__":
    unittest.main()
