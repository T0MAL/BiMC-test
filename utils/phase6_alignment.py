import torch
import torch.nn.functional as F


VALID_ALIGNMENT_MODES = ("none", "support_aware_text", "ot_text_image")


def align_text_prototypes(
    text_proto,
    visual_proto,
    mode="none",
    strength=0.25,
    ot_tau=10.0,
):
    if mode not in VALID_ALIGNMENT_MODES:
        raise ValueError(f"Unknown Phase 6 alignment mode: {mode}")
    if mode == "none":
        return F.normalize(text_proto, dim=-1)

    text_proto = F.normalize(text_proto, dim=-1)
    visual_proto = F.normalize(visual_proto.to(text_proto.device, text_proto.dtype), dim=-1)
    strength = float(strength)

    if mode == "support_aware_text":
        aligned = (1 - strength) * text_proto + strength * visual_proto
        return F.normalize(aligned, dim=-1)

    transport = torch.softmax((text_proto @ visual_proto.t()) * float(ot_tau), dim=-1)
    transported_visual = transport @ visual_proto
    aligned = (1 - strength) * text_proto + strength * transported_visual
    return F.normalize(aligned, dim=-1)


def apply_label_prior(probabilities, enabled=False, transductive=False, strength=0.25):
    if not enabled:
        return probabilities
    if not transductive:
        raise ValueError("LABEL_PRIOR_ENABLED requires LABEL_PRIOR_TRANSDUCTIVE=true.")

    num_classes = probabilities.shape[-1]
    if num_classes <= 1:
        return probabilities
    batch_prior = probabilities.mean(dim=0).clamp_min(1e-12)
    target_prior = torch.full_like(batch_prior, 1.0 / num_classes)
    correction = torch.pow(target_prior / batch_prior, float(strength))
    adjusted = probabilities * correction.unsqueeze(0)
    return adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
