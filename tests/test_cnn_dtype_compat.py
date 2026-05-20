import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bimc import BiMC


def _cfg():
    bimc = SimpleNamespace(
        TEXT_CALIBRATION=False,
        LAMBDA_T=0.0,
        FUSION_BETA_MODE="fixed",
        FUSION_GEOMETRY="linear",
        RELIABILITY_MODE="entropy_margin",
        BETA_CLIP_MIN=0.05,
        BETA_CLIP_MAX=0.95,
        USE_CNN_BRANCH=True,
        CNN_EXPERIMENT_MODE="cnn_proto_adjust",
        CNN_TOPK=5,
        USING_ENSEMBLE=False,
    )
    return SimpleNamespace(
        TRAINER=SimpleNamespace(BiMC=bimc),
        DATASET=SimpleNamespace(ENSEMBLE_ALPHA=0.6),
    )


class DummyBiMC(BiMC):
    def __init__(self, image_features):
        nn.Module.__init__(self)
        self.cfg = _cfg()
        self.image_features = image_features

    def extract_img_feature(self, images):
        return self.image_features[: images.shape[0]]


class CnnDtypeCompatTests(unittest.TestCase):
    def test_proto_adjust_forward_accepts_mixed_feature_and_proto_dtypes(self):
        torch.manual_seed(19)
        model = DummyBiMC(F.normalize(torch.randn(2, 4, dtype=torch.float32), dim=-1))
        images = torch.zeros(2, 3, 4, 4)

        image_proto = F.normalize(torch.randn(3, 4, dtype=torch.float64), dim=-1)
        text_features = F.normalize(torch.randn(3, 4, dtype=torch.float32), dim=-1)
        description_proto = F.normalize(torch.randn(3, 4, dtype=torch.float32), dim=-1)
        description_features = F.normalize(torch.randn(6, 4, dtype=torch.float64), dim=-1)
        description_targets = torch.tensor([0, 0, 1, 1, 2, 2])
        cov_image = torch.eye(4, dtype=torch.float64)

        logits, beta_info = model.forward_ours(
            images,
            num_cls=3,
            num_base_cls=2,
            image_proto=image_proto,
            cov_image=cov_image,
            description_proto=description_proto,
            description_features=description_features,
            description_targets=description_targets,
            text_features=text_features,
            beta=0.65,
            return_beta_info=True,
        )

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(beta_info["mode"], "fixed")
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
