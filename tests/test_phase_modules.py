import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F

    from models.bimc import BiMC
    from utils.phase2_prototypes import refine_description_prototypes, refine_visual_prototypes
    from utils.phase3_scores import compute_dynamic_alpha
    from utils.phase4_covariance import combine_novel_auxiliary, prepare_covariance
    from utils.phase5_space import common_direction, remove_common_direction
    from utils.phase6_alignment import align_text_prototypes, apply_label_prior
    from utils.phase7_separation import separate_prototypes

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this environment")
class PhaseModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_default_forward_matches_original_fixed_linear_fusion(self):
        cfg = ns(
            TRAINER=ns(
                BiMC=ns(
                    TEXT_CALIBRATION=True,
                    LAMBDA_T=0.5,
                    USING_ENSEMBLE=False,
                    FUSION_BETA_MODE="fixed",
                    FUSION_GEOMETRY="linear",
                    BETA_CLIP_MIN=0.05,
                    BETA_CLIP_MAX=0.95,
                    RELIABILITY_MODE="entropy_margin",
                    PHASE3=ns(
                        ENABLED=False,
                        HUBNESS_ENABLED=False,
                        TEMP_SCALING_ENABLED=False,
                        TEMP_VALUE=1.0,
                        DYNAMIC_ALPHA_ENABLED=False,
                    ),
                    PHASE4=ns(
                        ENABLED=False,
                        APPLY_TO_NOVEL=False,
                    ),
                    PHASE5=ns(
                        ENABLED=False,
                        SPACE_TRANSFORM="none",
                    ),
                    PHASE6=ns(
                        ENABLED=False,
                        LABEL_PRIOR_ENABLED=False,
                    ),
                    PHASE7=ns(
                        ENABLED=False,
                    ),
                )
            ),
            DATASET=ns(ENSEMBLE_ALPHA=0.6),
        )

        class Dummy:
            def __init__(self, cfg):
                self.cfg = cfg

            def extract_img_feature(self, images):
                return images

        images = F.normalize(torch.randn(2, 4), dim=-1)
        image_proto = F.normalize(torch.randn(3, 4), dim=-1)
        text_features = F.normalize(torch.randn(3, 4), dim=-1)
        description_proto = F.normalize(torch.randn(3, 4), dim=-1)
        description_features = F.normalize(torch.randn(6, 4), dim=-1)
        description_targets = torch.tensor([0, 0, 1, 1, 2, 2])
        cov = torch.eye(4)
        beta = 0.4

        out = BiMC.forward_ours(
            Dummy(cfg),
            images,
            num_cls=3,
            num_base_cls=3,
            image_proto=image_proto,
            cov_image=cov,
            description_proto=description_proto,
            description_features=description_features,
            description_targets=description_targets,
            text_features=text_features,
            beta=beta,
        )

        calibrated_text = (1 - cfg.TRAINER.BiMC.LAMBDA_T) * text_features + cfg.TRAINER.BiMC.LAMBDA_T * description_proto
        fused = F.normalize(beta * calibrated_text + (1 - beta) * image_proto, dim=-1)
        expected = F.softmax(images @ fused.t(), dim=-1)

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_phase2_refiners_preserve_shapes(self):
        visual = F.normalize(torch.randn(2, 4), dim=-1)
        base = F.normalize(torch.randn(3, 4), dim=-1)
        support = F.normalize(torch.randn(4, 4), dim=-1)
        labels = torch.tensor([5, 5, 6, 6])
        refined_visual = refine_visual_prototypes(
            visual,
            base_visual_proto=base,
            support_features=support,
            support_labels=labels,
            class_ids=[5, 6],
            mode="shrinkage_base_prior",
            dynamic_lambda=True,
        )
        self.assertEqual(refined_visual.shape, visual.shape)

        desc_features = F.normalize(torch.randn(6, 4), dim=-1)
        desc_targets = torch.tensor([5, 5, 5, 6, 6, 6])
        desc_proto = F.normalize(torch.randn(2, 4), dim=-1)
        text_proto = F.normalize(torch.randn(2, 4), dim=-1)
        refined_text = refine_description_prototypes(
            desc_features,
            desc_targets,
            desc_proto,
            text_proto,
            class_ids=[5, 6],
            mode="combined_reweight",
        )
        self.assertEqual(refined_text.shape, desc_proto.shape)

    def test_phase3_dynamic_alpha_range(self):
        proto_logits = torch.randn(5, 4)
        aux_logits = torch.randn(5, 4)
        alpha = compute_dynamic_alpha(proto_logits, aux_logits, mode="entropy_margin_energy")
        self.assertEqual(alpha.shape, (5,))
        self.assertGreaterEqual(alpha.min().item(), 0.05)
        self.assertLessEqual(alpha.max().item(), 0.95)

    def test_phase4_covariance_and_novel_aux(self):
        cov = torch.randn(4, 4)
        cov = cov @ cov.t()
        hybrid = prepare_covariance(cov, mode="hybrid_diag")
        self.assertEqual(hybrid.shape, cov.shape)
        knn = torch.tensor([[0.2, 0.8]])
        cov_prob = torch.tensor([[0.6, 0.4]])
        combined = combine_novel_auxiliary(knn, cov_prob, combine_mode="average")
        self.assertTrue(torch.allclose(combined, torch.tensor([[0.4, 0.6]])))

    def test_phase5_common_direction_removal_reduces_projection(self):
        values = F.normalize(torch.randn(6, 4), dim=-1)
        direction = common_direction([values])
        removed = remove_common_direction(values, direction, normalize=False)
        before = torch.abs(values @ direction).mean()
        after = torch.abs(removed @ direction).mean()
        self.assertLess(after.item(), before.item())

    def test_phase6_alignment_and_label_prior(self):
        text = F.normalize(torch.randn(3, 4), dim=-1)
        visual = F.normalize(torch.randn(3, 4), dim=-1)
        aligned = align_text_prototypes(text, visual, mode="support_aware_text")
        self.assertEqual(aligned.shape, text.shape)

        probs = torch.softmax(torch.randn(5, 3), dim=-1)
        adjusted = apply_label_prior(probs, enabled=True, transductive=True)
        self.assertTrue(torch.allclose(adjusted.sum(dim=-1), torch.ones(5), atol=1e-6))

    def test_phase7_separation_shape(self):
        prototypes = F.normalize(torch.randn(4, 8), dim=-1)
        separated = separate_prototypes(prototypes, mode="graph_highpass")
        self.assertEqual(separated.shape, prototypes.shape)
        self.assertTrue(torch.allclose(separated.norm(dim=-1), torch.ones(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
