import os
import unittest

from main import setup_cfg
from utils.phase1_fusion import validate_phase1_options


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class CnnBiMCConfigTests(unittest.TestCase):
    def test_baseline_config_still_loads(self):
        cfg = setup_cfg(
            os.path.join(ROOT, "configs", "datasets", "cifar100.yaml"),
            os.path.join(ROOT, "configs", "trainers", "bimc.yaml"),
        )

        validate_phase1_options(cfg)

        self.assertEqual(cfg.TRAINER.BiMC.FUSION_BETA_MODE, "fixed")
        self.assertFalse(cfg.TRAINER.BiMC.USE_CNN_BRANCH)
        self.assertEqual(cfg.TRAINER.BiMC.CNN_EXPERIMENT_MODE, "none")
        self.assertEqual(cfg.TRAINER.BiMC.CNN_BACKBONE, "resnet50")


if __name__ == "__main__":
    unittest.main()
