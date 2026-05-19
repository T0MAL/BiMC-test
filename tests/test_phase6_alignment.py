import unittest

import torch
import torch.nn.functional as F

from utils.phase6_alignment import (
    apply_label_prior_correction,
    blackbox_shift_prior,
    ot_text_image_alignment,
    prediction_frequency_prior,
    sinkhorn_transport,
    support_aware_text_proto,
)


class Phase6AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def _features(self, n=6, d=8, dtype=torch.float64):
        return F.normalize(torch.randn(n, d, dtype=dtype), dim=-1)

    def test_sinkhorn_transport_uniform_marginals(self):
        cost = torch.rand(4, 5, dtype=torch.float64)

        transport = sinkhorn_transport(cost, eps=0.1, max_iter=100)

        self.assertEqual(transport.shape, cost.shape)
        self.assertTrue(torch.isfinite(transport).all())
        self.assertTrue(torch.allclose(transport.sum(dim=1), torch.full((4,), 1 / 4, dtype=transport.dtype), atol=1e-4))
        self.assertTrue(torch.allclose(transport.sum(dim=0), torch.full((5,), 1 / 5, dtype=transport.dtype), atol=1e-4))

    def test_ot_text_image_alignment_outputs_normalized_proto(self):
        desc = self._features(n=5, d=9)
        support = self._features(n=4, d=9)

        proto, weights, stats = ot_text_image_alignment(desc, support, eps=0.1, max_iter=80)

        self.assertEqual(proto.shape, (9,))
        self.assertEqual(weights.shape, (5,))
        self.assertTrue(torch.isfinite(proto).all())
        self.assertTrue(torch.allclose(weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(proto.norm(), torch.ones((), dtype=proto.dtype), atol=1e-6))
        self.assertIn("ot_cost_mean", stats)

    def test_support_aware_text_proto_topk(self):
        desc = self._features(n=6, d=7)
        support_proto = self._features(n=1, d=7).squeeze(0)

        proto, weights, stats = support_aware_text_proto(desc, support_proto, temp=0.07, topk=3)

        self.assertEqual(proto.shape, support_proto.shape)
        self.assertEqual(int((weights > 0).sum().item()), 3)
        self.assertTrue(torch.allclose(weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(proto.norm(), torch.ones((), dtype=proto.dtype), atol=1e-6))
        self.assertEqual(stats["alignment_mode"], "support_aware_text")

    def test_label_priors_sum_to_one_and_correction_keeps_shape(self):
        logits = torch.randn(11, 4, dtype=torch.float64)
        probs = torch.softmax(logits, dim=-1)

        freq = prediction_frequency_prior(probs)
        shift = blackbox_shift_prior(probs, max_iter=5)
        corrected = apply_label_prior_correction(torch.log(probs), shift, strength=0.5)

        self.assertTrue(torch.allclose(freq.sum(), torch.ones((), dtype=freq.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(shift.sum(), torch.ones((), dtype=shift.dtype), atol=1e-6))
        self.assertEqual(corrected.shape, logits.shape)
        self.assertTrue(torch.isfinite(corrected).all())


if __name__ == "__main__":
    unittest.main()
