import torch
import torch.nn.functional as F


VALID_VISUAL_PROTO_MODES = ("original", "shrinkage_base_prior")
VALID_TEXT_PROTO_MODES = ("original", "combined_reweight")


def refine_visual_prototypes(
    visual_proto,
    base_visual_proto=None,
    support_features=None,
    support_labels=None,
    class_ids=None,
    mode="original",
    shrinkage=0.1,
    dynamic_lambda=False,
    tau=16.0,
):
    if mode not in VALID_VISUAL_PROTO_MODES:
        raise ValueError(f"Unknown Phase 2 visual prototype mode: {mode}")
    if mode == "original" or base_visual_proto is None or base_visual_proto.numel() == 0:
        return F.normalize(visual_proto, dim=-1)

    visual_proto = F.normalize(visual_proto, dim=-1)
    base_visual_proto = F.normalize(base_visual_proto.to(visual_proto.device, visual_proto.dtype), dim=-1)
    weights = torch.softmax((visual_proto @ base_visual_proto.t()) * float(tau), dim=-1)
    base_prior = F.normalize(weights @ base_visual_proto, dim=-1)

    lambdas = _visual_shrinkage_lambdas(
        visual_proto=visual_proto,
        support_features=support_features,
        support_labels=support_labels,
        class_ids=class_ids,
        default_lambda=float(shrinkage),
        dynamic_lambda=dynamic_lambda,
    )
    refined = (1 - lambdas) * visual_proto + lambdas * base_prior
    return F.normalize(refined, dim=-1)


def refine_description_prototypes(
    description_features,
    description_targets,
    description_proto,
    text_proto,
    class_ids,
    mode="original",
    reweight_tau=10.0,
    combine_weight=0.5,
):
    if mode not in VALID_TEXT_PROTO_MODES:
        raise ValueError(f"Unknown Phase 2 text prototype mode: {mode}")
    if mode == "original":
        return F.normalize(description_proto, dim=-1)

    description_features = F.normalize(description_features, dim=-1)
    description_proto = F.normalize(description_proto, dim=-1)
    text_proto = F.normalize(text_proto, dim=-1)
    description_targets = description_targets.to(description_features.device).long()

    refined = []
    for local_idx, class_id in enumerate(class_ids):
        mask = description_targets == int(class_id)
        if not torch.any(mask):
            refined.append(description_proto[local_idx])
            continue
        class_descriptions = description_features[mask]
        query = text_proto[local_idx]
        weights = torch.softmax((class_descriptions @ query) * float(reweight_tau), dim=0)
        reweighted = F.normalize((weights.unsqueeze(-1) * class_descriptions).sum(dim=0), dim=-1)
        combined = (1 - float(combine_weight)) * description_proto[local_idx] + float(combine_weight) * reweighted
        refined.append(F.normalize(combined, dim=-1))
    return torch.stack(refined, dim=0)


def _visual_shrinkage_lambdas(
    visual_proto,
    support_features,
    support_labels,
    class_ids,
    default_lambda,
    dynamic_lambda,
):
    default = torch.full(
        (visual_proto.shape[0], 1),
        float(default_lambda),
        device=visual_proto.device,
        dtype=visual_proto.dtype,
    )
    if not dynamic_lambda or support_features is None or support_labels is None or class_ids is None:
        return default.clamp(0.0, 0.95)

    support_features = F.normalize(support_features.to(visual_proto.device, visual_proto.dtype), dim=-1)
    support_labels = support_labels.to(visual_proto.device).long()

    values = []
    for local_idx, class_id in enumerate(class_ids):
        mask = support_labels == int(class_id)
        count = int(mask.sum().item())
        if count == 0:
            values.append(min(0.95, max(default_lambda, 0.5)))
            continue
        class_features = support_features[mask]
        compactness = (class_features @ visual_proto[local_idx]).mean().clamp(min=0.0, max=1.0)
        low_shot_factor = 1.0 / (count ** 0.5)
        lambda_i = default_lambda + (1 - compactness.item()) * 0.25 + low_shot_factor * 0.10
        values.append(min(0.95, max(0.0, lambda_i)))

    return torch.tensor(values, device=visual_proto.device, dtype=visual_proto.dtype).view(-1, 1)
