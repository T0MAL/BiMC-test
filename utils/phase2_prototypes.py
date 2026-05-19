import math

import torch


VALID_VISUAL_PROTO_MODES = ("mean", "robust_weighted", "shrinkage_base_prior")
VALID_SHRINKAGE_PRIORS = ("base_neighbors", "text", "zero")
VALID_TEXT_PROTO_MODES = (
    "mean",
    "discriminative_reweight",
    "visual_grounded_reweight",
    "combined_reweight",
    "topk_discriminative",
)


def validate_phase2_options(cfg):
    opts = cfg.TRAINER.BiMC.PHASE2
    if opts.VISUAL_PROTO_MODE not in VALID_VISUAL_PROTO_MODES:
        raise ValueError(
            f"Invalid PHASE2.VISUAL_PROTO_MODE {opts.VISUAL_PROTO_MODE}. "
            f"Expected one of {VALID_VISUAL_PROTO_MODES}."
        )
    if opts.SHRINKAGE_PRIOR not in VALID_SHRINKAGE_PRIORS:
        raise ValueError(
            f"Invalid PHASE2.SHRINKAGE_PRIOR {opts.SHRINKAGE_PRIOR}. "
            f"Expected one of {VALID_SHRINKAGE_PRIORS}."
        )
    if opts.TEXT_PROTO_MODE not in VALID_TEXT_PROTO_MODES:
        raise ValueError(
            f"Invalid PHASE2.TEXT_PROTO_MODE {opts.TEXT_PROTO_MODE}. "
            f"Expected one of {VALID_TEXT_PROTO_MODES}."
        )
    if opts.ROBUST_KAPPA <= 0:
        raise ValueError("PHASE2.ROBUST_KAPPA must be positive.")
    if opts.SHRINKAGE_EPS <= 0:
        raise ValueError("PHASE2.SHRINKAGE_EPS must be positive.")
    if not 0 <= opts.SHRINKAGE_MIN <= opts.SHRINKAGE_MAX <= 1:
        raise ValueError("Expected 0 <= PHASE2.SHRINKAGE_MIN <= PHASE2.SHRINKAGE_MAX <= 1.")
    if not 0 <= opts.DYNAMIC_LAMBDA_MIN <= opts.DYNAMIC_LAMBDA_MAX <= 1:
        raise ValueError("Expected 0 <= PHASE2.DYNAMIC_LAMBDA_MIN <= PHASE2.DYNAMIC_LAMBDA_MAX <= 1.")
    if opts.DESC_TEMP <= 0:
        raise ValueError("PHASE2.DESC_TEMP must be positive.")
    if opts.DESC_TOPK <= 0:
        raise ValueError("PHASE2.DESC_TOPK must be positive.")


def normalize(x, dim=-1, eps=1e-8):
    safe_eps = _safe_eps(x, eps)
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    return x / norm.clamp_min(safe_eps)


def compute_visual_quality(features):
    if features.shape[0] == 0:
        return torch.zeros((), device=features.device, dtype=features.dtype)
    q = torch.linalg.vector_norm(features.mean(dim=0), ord=2)
    one = torch.ones((), device=features.device, dtype=features.dtype)
    zero = torch.zeros((), device=features.device, dtype=features.dtype)
    return torch.clamp(q, min=zero, max=one)


def robust_weighted_visual_prototype(features, kappa=16.0, drop_lowest=False):
    if features.shape[0] == 0:
        proto = torch.zeros(features.shape[-1], device=features.device, dtype=features.dtype)
        weights = torch.empty(0, device=features.device, dtype=features.dtype)
        return proto, weights, _weight_stats(weights, None, {"num_support": 0, "dropped_index": None})

    center = normalize(features.mean(dim=0), dim=-1)
    sim = features @ center
    kept_features = features
    kept_sim = sim
    dropped_index = None

    if drop_lowest and features.shape[0] > 2:
        dropped_index = int(torch.argmin(sim).detach().cpu().item())
        keep_mask = torch.ones(features.shape[0], device=features.device, dtype=torch.bool)
        keep_mask[dropped_index] = False
        kept_features = features[keep_mask]
        kept_sim = sim[keep_mask]

    weights = torch.softmax(kept_sim * float(kappa), dim=0)
    proto = normalize((weights.unsqueeze(-1) * kept_features).sum(dim=0), dim=-1)
    stats = _weight_stats(
        weights,
        kept_sim,
        {
            "num_support": int(features.shape[0]),
            "num_used": int(kept_features.shape[0]),
            "dropped_index": dropped_index,
            "visual_quality": _to_float(compute_visual_quality(features)),
        },
    )
    return proto, weights, stats


def base_neighbor_prior(novel_proto, base_protos, tau=16.0):
    if base_protos is None or base_protos.shape[0] == 0:
        return normalize(novel_proto, dim=-1)
    novel_proto = normalize(novel_proto, dim=-1)
    base_protos = normalize(base_protos, dim=-1)
    scores = base_protos @ novel_proto
    weights = torch.softmax(scores * float(tau), dim=0)
    return normalize((weights.unsqueeze(-1) * base_protos).sum(dim=0), dim=-1)


def shrinkage_visual_prototype(
    features,
    shot_proto,
    prior_proto,
    eps=1e-8,
    min_rho=0.0,
    max_rho=1.0,
):
    shot_proto = normalize(shot_proto, dim=-1)
    prior_proto = normalize(prior_proto, dim=-1)
    m = int(features.shape[0])
    safe_eps = _safe_eps(shot_proto, eps)

    if m == 0:
        rho = torch.as_tensor(max_rho, device=shot_proto.device, dtype=shot_proto.dtype)
        proto = normalize((1 - rho) * shot_proto + rho * prior_proto, dim=-1)
        return proto, rho, {
            "num_support": 0,
            "sigma2": None,
            "dist2": None,
            "rho": _to_float(rho),
        }

    diff = features - shot_proto.unsqueeze(0)
    sigma2 = (diff * diff).sum(dim=-1).mean()
    prior_diff = shot_proto - prior_proto
    dist2 = (prior_diff * prior_diff).sum()
    denom = sigma2 + (float(m) * dist2) + safe_eps
    rho = sigma2 / denom
    rho = torch.clamp(rho, min=float(min_rho), max=float(max_rho))
    proto = normalize((1 - rho) * shot_proto + rho * prior_proto, dim=-1)
    stats = {
        "num_support": m,
        "sigma2": _to_float(sigma2),
        "dist2": _to_float(dist2),
        "rho": _to_float(rho),
    }
    return proto, rho, stats


def dynamic_lambda_i_from_quality(q, min_val=0.0, max_val=0.5):
    if isinstance(q, torch.Tensor):
        return torch.clamp(1 - q, min=float(min_val), max=float(max_val))
    return max(float(min_val), min(float(max_val), 1 - float(q)))


def discriminative_description_reweight(
    desc_feats,
    class_name_proto,
    all_class_name_protos,
    temp=0.05,
    topk=None,
):
    score, own, other = _discriminative_scores(desc_feats, class_name_proto, all_class_name_protos)
    proto, weights = _weighted_description_proto(desc_feats, score, temp=temp, topk=topk)
    stats = _description_stats(weights, score, own=own, other=other, topk=topk)
    return proto, weights, stats


def visual_grounded_description_reweight(desc_feats, support_visual_proto, temp=0.05):
    if desc_feats.shape[0] == 0:
        score = torch.empty(0, device=desc_feats.device, dtype=desc_feats.dtype)
    else:
        score = desc_feats @ normalize(support_visual_proto, dim=-1)
    proto, weights = _weighted_description_proto(desc_feats, score, temp=temp)
    stats = _description_stats(weights, score)
    return proto, weights, stats


def combined_description_reweight(
    desc_feats,
    support_visual_proto,
    class_name_proto,
    all_class_name_protos,
    temp=0.05,
    lambda_d=0.5,
):
    discrim_score, own, other = _discriminative_scores(desc_feats, class_name_proto, all_class_name_protos)
    if desc_feats.shape[0] == 0:
        visual_score = discrim_score
    else:
        visual_score = desc_feats @ normalize(support_visual_proto, dim=-1)
    score = visual_score + float(lambda_d) * discrim_score
    proto, weights = _weighted_description_proto(desc_feats, score, temp=temp)
    stats = _description_stats(
        weights,
        score,
        own=own,
        other=other,
        visual_score=visual_score,
        discrim_score=discrim_score,
    )
    return proto, weights, stats


def _discriminative_scores(desc_feats, class_name_proto, all_class_name_protos):
    if desc_feats.shape[0] == 0:
        empty = torch.empty(0, device=desc_feats.device, dtype=desc_feats.dtype)
        return empty, empty, empty

    class_name_proto = normalize(class_name_proto, dim=-1)
    all_class_name_protos = normalize(all_class_name_protos, dim=-1)
    own = desc_feats @ class_name_proto

    if all_class_name_protos.shape[0] <= 1:
        other = torch.zeros_like(own)
        return own - other, own, other

    all_scores = desc_feats @ all_class_name_protos.t()
    own_index = torch.argmax(all_class_name_protos @ class_name_proto).detach()
    all_scores = all_scores.clone()
    all_scores[:, own_index] = -torch.inf
    other = all_scores.max(dim=1).values
    return own - other, own, other


def _weighted_description_proto(desc_feats, score, temp=0.05, topk=None):
    if desc_feats.shape[0] == 0:
        proto = torch.zeros(desc_feats.shape[-1], device=desc_feats.device, dtype=desc_feats.dtype)
        weights = torch.empty(0, device=desc_feats.device, dtype=desc_feats.dtype)
        return proto, weights

    temp = max(float(temp), _safe_eps(desc_feats, 1e-8))
    if topk is None:
        weights = torch.softmax(score / temp, dim=0)
    else:
        k = max(1, min(int(topk), score.shape[0]))
        keep = torch.topk(score, k=k, dim=0).indices
        weights = torch.zeros_like(score)
        weights[keep] = torch.softmax(score[keep] / temp, dim=0)

    proto = normalize((weights.unsqueeze(-1) * desc_feats).sum(dim=0), dim=-1)
    return proto, weights


def _description_stats(weights, score, **extra_scores):
    stats = {
        "num_descriptions": int(weights.shape[0]),
        "desc_weight_entropy": _entropy(weights),
        "desc_weight_min": _to_float(weights.min()) if weights.numel() else None,
        "desc_weight_max": _to_float(weights.max()) if weights.numel() else None,
        "desc_score_mean": _to_float(score.mean()) if score.numel() else None,
        "desc_score_std": _std(score),
        "desc_score_min": _to_float(score.min()) if score.numel() else None,
        "desc_score_max": _to_float(score.max()) if score.numel() else None,
    }
    for name, value in extra_scores.items():
        if isinstance(value, torch.Tensor):
            stats[f"{name}_mean"] = _to_float(value.mean()) if value.numel() else None
            stats[f"{name}_min"] = _to_float(value.min()) if value.numel() else None
            stats[f"{name}_max"] = _to_float(value.max()) if value.numel() else None
        else:
            stats[name] = value
    return stats


def _weight_stats(weights, score, extra):
    stats = dict(extra)
    stats.update(
        {
            "weight_entropy": _entropy(weights),
            "weight_min": _to_float(weights.min()) if weights.numel() else None,
            "weight_max": _to_float(weights.max()) if weights.numel() else None,
        }
    )
    if score is not None and score.numel():
        stats.update(
            {
                "sim_mean": _to_float(score.mean()),
                "sim_std": _std(score),
                "sim_min": _to_float(score.min()),
                "sim_max": _to_float(score.max()),
            }
        )
    return stats


def _entropy(weights, eps=1e-12):
    if weights.numel() == 0:
        return None
    weights_f = weights.float()
    entropy = -(weights_f * torch.log(weights_f.clamp_min(eps))).sum()
    return _to_float(entropy)


def _std(value):
    if value.numel() == 0:
        return None
    return _to_float(value.float().std(unbiased=False))


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().float().cpu().item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _safe_eps(ref, eps):
    if not torch.is_floating_point(ref):
        return float(eps)
    finfo = torch.finfo(ref.dtype)
    return max(float(eps), float(finfo.eps))
