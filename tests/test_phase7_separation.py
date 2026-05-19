import unittest

import torch
import torch.nn.functional as F

from utils.phase7_separation import (
    compute_pairwise_cosine,
    graph_highpass_correction,
    hubness_safe_repulsion,
    prototype_repulsion,
)


class Phase7SeparationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def _prototypes(self, n=6, d=8, dtype=torch.float64):
        return F.normalize(torch.randn(n, d, dtype=dtype), dim=-1)

    def test_pairwise_cosine_shape_and_finite(self):
        prototypes = self._prototypes()

        sim = compute_pairwise_cosine(prototypes)

        self.assertEqual(sim.shape, (prototypes.shape[0], prototypes.shape[0]))
        self.assertTrue(torch.isfinite(sim).all())
        self.assertTrue(torch.allclose(torch.diag(sim), torch.ones(prototypes.shape[0], dtype=sim.dtype), atol=1e-6))

    def test_prototype_repulsion_keeps_shape_and_normalization(self):
        prototypes = self._prototypes()

        updated, stats = prototype_repulsion(prototypes, delta=0.03, margin=-1.0, topk=3)

        self.assertEqual(updated.shape, prototypes.shape)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue(torch.allclose(updated.norm(dim=-1), torch.ones(updated.shape[0], dtype=updated.dtype), atol=1e-6))
        self.assertIn("affected_prototypes", stats)

    def test_graph_highpass_keeps_shape_and_normalization(self):
        prototypes = self._prototypes()

        updated, stats = graph_highpass_correction(prototypes, tau=0.1, gamma=0.05, topk=3)

        self.assertEqual(updated.shape, prototypes.shape)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue(torch.allclose(updated.norm(dim=-1), torch.ones(updated.shape[0], dtype=updated.dtype), atol=1e-6))
        self.assertEqual(stats["separation_mode"], "graph_highpass")

    def test_hubness_safe_repulsion_only_updates_high_hubness(self):
        prototypes = self._prototypes()
        hubness = torch.tensor([10, 1, 1, 7, 0, 0], dtype=prototypes.dtype)

        updated, stats = hubness_safe_repulsion(prototypes, hubness, delta=0.03, topk=2)

        self.assertEqual(updated.shape, prototypes.shape)
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue(torch.allclose(updated.norm(dim=-1), torch.ones(updated.shape[0], dtype=updated.dtype), atol=1e-6))
        self.assertGreaterEqual(stats["affected_prototypes"], 1)


if __name__ == "__main__":
    unittest.main()
