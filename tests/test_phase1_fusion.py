import unittest

import torch
import torch.nn.functional as F

from utils.phase1_fusion import (
    compute_class_margin_beta,
    compute_query_reliability_beta,
    fuse_prototypes,
    slerp,
)


class Phase1FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_fixed_linear_matches_original_formula(self):
        text_proto = torch.randn(4, 8)
        visual_proto = torch.randn(4, 8)
        beta = 0.3

        fused = fuse_prototypes(visual_proto, text_proto, beta, geometry="linear")
        expected = F.normalize(beta * text_proto + (1 - beta) * visual_proto, dim=-1)

        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))

    def test_slerp_shape_and_normalization(self):
        visual_proto = F.normalize(torch.randn(4, 8), dim=-1)
        text_proto = F.normalize(torch.randn(4, 8), dim=-1)
        beta = torch.linspace(0.1, 0.9, steps=4)

        fused = slerp(visual_proto, text_proto, beta)

        self.assertEqual(fused.shape, visual_proto.shape)
        self.assertTrue(torch.allclose(fused.norm(dim=-1), torch.ones(4), atol=1e-5))

    def test_slerp_query_broadcast_shape(self):
        visual_proto = F.normalize(torch.randn(1, 4, 8), dim=-1)
        text_proto = F.normalize(torch.randn(1, 4, 8), dim=-1)
        beta = torch.linspace(0.1, 0.9, steps=5).view(5, 1, 1)

        fused = fuse_prototypes(visual_proto, text_proto, beta, geometry="slerp")

        self.assertEqual(fused.shape, (5, 4, 8))

    def test_class_margin_beta_is_clipped(self):
        support_features = F.normalize(torch.randn(12, 8), dim=-1)
        support_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        text_proto = F.normalize(torch.randn(3, 8), dim=-1)
        visual_proto = F.normalize(torch.randn(3, 8), dim=-1)

        beta = compute_class_margin_beta(
            support_features,
            support_labels,
            text_proto,
            visual_proto,
            beta_temperature=0.05,
            beta_clip_min=0.05,
            beta_clip_max=0.95,
        )

        self.assertEqual(beta.shape, (3,))
        self.assertGreaterEqual(beta.min().item(), 0.05)
        self.assertLessEqual(beta.max().item(), 0.95)

    def test_query_reliability_beta_per_query(self):
        query_features = F.normalize(torch.randn(5, 8), dim=-1)
        text_proto = F.normalize(torch.randn(4, 8), dim=-1)
        visual_proto = F.normalize(torch.randn(4, 8), dim=-1)

        beta = compute_query_reliability_beta(
            query_features,
            text_proto,
            visual_proto,
            reliability_mode="entropy_margin",
            beta_clip_min=0.05,
            beta_clip_max=0.95,
        )

        self.assertEqual(beta.shape, (5,))
        self.assertGreaterEqual(beta.min().item(), 0.05)
        self.assertLessEqual(beta.max().item(), 0.95)

    def test_reliability_helpers_accept_mixed_proto_dtypes(self):
        support_features = F.normalize(torch.randn(6, 8, dtype=torch.float32), dim=-1)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        text_proto = F.normalize(torch.randn(3, 8, dtype=torch.float32), dim=-1)
        visual_proto = F.normalize(torch.randn(3, 8, dtype=torch.float64), dim=-1)

        class_beta = compute_class_margin_beta(
            support_features,
            support_labels,
            text_proto,
            visual_proto,
            beta_temperature=0.05,
            beta_clip_min=0.05,
            beta_clip_max=0.95,
        )
        query_beta = compute_query_reliability_beta(
            support_features,
            text_proto,
            visual_proto,
            reliability_mode="entropy_margin",
            beta_clip_min=0.05,
            beta_clip_max=0.95,
        )

        self.assertEqual(class_beta.dtype, support_features.dtype)
        self.assertEqual(query_beta.dtype, support_features.dtype)
        self.assertTrue(torch.isfinite(class_beta).all())
        self.assertTrue(torch.isfinite(query_beta).all())


if __name__ == "__main__":
    unittest.main()
