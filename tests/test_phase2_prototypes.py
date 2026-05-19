import os
import unittest

import torch
import torch.nn.functional as F

from main import setup_cfg
from utils.phase2_prototypes import (
    base_neighbor_prior,
    combined_description_reweight,
    discriminative_description_reweight,
    dynamic_lambda_i_from_quality,
    normalize,
    robust_weighted_visual_prototype,
    shrinkage_visual_prototype,
    visual_grounded_description_reweight,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Phase2PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def _features(self, n=5, d=8, dtype=torch.float64, device="cpu"):
        return F.normalize(torch.randn(n, d, dtype=dtype, device=device), dim=-1)

    def test_robust_weighted_visual_prototype_shape_normalization_and_weights(self):
        features = self._features(n=4, d=8)

        proto, weights, _ = robust_weighted_visual_prototype(features)

        self.assertEqual(proto.shape, (8,))
        self.assertTrue(torch.allclose(proto.norm(), torch.ones((), dtype=proto.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-6))

    def test_shrinkage_rho_range_and_normalization(self):
        features = self._features(n=5, d=8)
        shot_proto = normalize(features.mean(dim=0))
        prior_proto = normalize(torch.randn(8, dtype=features.dtype))

        proto, rho, _ = shrinkage_visual_prototype(features, shot_proto, prior_proto)

        self.assertGreaterEqual(rho.item(), 0.0)
        self.assertLessEqual(rho.item(), 1.0)
        self.assertTrue(torch.allclose(proto.norm(), torch.ones((), dtype=proto.dtype), atol=1e-6))

    def test_dynamic_lambda_inside_configured_range(self):
        q = torch.tensor(0.73, dtype=torch.float64)

        lambda_i = dynamic_lambda_i_from_quality(q, min_val=0.1, max_val=0.2)

        self.assertGreaterEqual(lambda_i.item(), 0.1)
        self.assertLessEqual(lambda_i.item(), 0.2)

    def test_description_reweighting_weights_and_normalization(self):
        desc = self._features(n=6, d=8)
        class_name = normalize(torch.randn(8, dtype=desc.dtype))
        all_names = F.normalize(torch.randn(4, 8, dtype=desc.dtype), dim=-1)
        all_names[2] = class_name

        proto, weights, _ = discriminative_description_reweight(desc, class_name, all_names)

        self.assertTrue(torch.allclose(weights.sum(), torch.ones((), dtype=weights.dtype), atol=1e-6))
        self.assertTrue(torch.allclose(proto.norm(), torch.ones((), dtype=proto.dtype), atol=1e-6))

    def test_all_functions_preserve_device_and_dtype(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            with self.subTest(device=device):
                dtype = torch.float64
                features = self._features(n=5, d=8, dtype=dtype, device=device)
                shot_proto = normalize(features.mean(dim=0))
                prior_proto = normalize(torch.randn(8, dtype=dtype, device=device))
                base_protos = self._features(n=3, d=8, dtype=dtype, device=device)
                desc = self._features(n=4, d=8, dtype=dtype, device=device)
                all_names = self._features(n=3, d=8, dtype=dtype, device=device)
                class_name = all_names[1]

                robust_proto, robust_weights, _ = robust_weighted_visual_prototype(features)
                prior = base_neighbor_prior(shot_proto, base_protos)
                shrink_proto, rho, _ = shrinkage_visual_prototype(features, shot_proto, prior_proto)
                lambda_i = dynamic_lambda_i_from_quality(torch.tensor(0.6, dtype=dtype, device=device))
                discrim_proto, discrim_weights, _ = discriminative_description_reweight(desc, class_name, all_names)
                visual_proto, visual_weights, _ = visual_grounded_description_reweight(desc, shot_proto)
                combined_proto, combined_weights, _ = combined_description_reweight(desc, shot_proto, class_name, all_names)

                tensors = [
                    robust_proto,
                    robust_weights,
                    prior,
                    shrink_proto,
                    rho,
                    lambda_i,
                    discrim_proto,
                    discrim_weights,
                    visual_proto,
                    visual_weights,
                    combined_proto,
                    combined_weights,
                ]
                for tensor in tensors:
                    self.assertEqual(tensor.device.type, torch.device(device).type)
                    self.assertEqual(tensor.dtype, dtype)

    def test_original_phase2_config_preserves_original_defaults(self):
        cfg = setup_cfg(
            os.path.join(ROOT, "configs", "datasets", "cifar100.yaml"),
            os.path.join(ROOT, "configs", "trainers", "bimc.yaml"),
        )

        self.assertFalse(cfg.TRAINER.BiMC.PHASE2.ENABLED)
        self.assertEqual(cfg.TRAINER.BiMC.PHASE2.VISUAL_PROTO_MODE, "mean")
        self.assertEqual(cfg.TRAINER.BiMC.PHASE2.TEXT_PROTO_MODE, "mean")
        self.assertFalse(cfg.TRAINER.BiMC.PHASE2.DYNAMIC_LAMBDA_I)
        self.assertEqual(cfg.TRAINER.BiMC.FUSION_BETA_MODE, "fixed")
        self.assertEqual(cfg.TRAINER.BiMC.FUSION_GEOMETRY, "linear")


if __name__ == "__main__":
    unittest.main()
