import math

import torch
import torch.nn.functional as F


VALID_ALIGNMENT_MODES = ("none", "ot_text_image", "support_aware_text", "label_prior_correction")
VALID_LABEL_PRIOR_MODES = ("blackbox_shift", "prediction_frequency", "uniform_smoothing")


def validate_phase6_options(cfg):
    opts = cfg.TRAINER.BiMC.PHASE6
    if opts.ALIGNMENT_MODE not in VALID_ALIGNMENT_MODES:
        raise ValueError(
            f"Invalid PHASE6.ALIGNMENT_MODE {opts.ALIGNMENT_MODE}. "
            f"Expected one of {VALID_ALIGNMENT_MODES}."
        )
    if opts.LABEL_PRIOR_MODE not in VALID_LABEL_PRIOR_MODES:
        raise ValueError(
            f"Invalid PHASE6.LABEL_PRIOR_MODE {opts.LABEL_PRIOR_MODE}. "
            f"Expected one of {VALID_LABEL_PRIOR_MODES}."
        )
    if opts.OT_EPS <= 0:
        raise ValueError("PHASE6.OT_EPS must be positive.")
    if opts.OT_MAX_ITER <= 0:
        raise ValueError("PHASE6.OT_MAX_ITER must be positive.")
    if opts.SUPPORT_AWARE_TEXT_TEMP <= 0:
        raise ValueError("PHASE6.SUPPORT_AWARE_TEXT_TEMP must be positive.")
    if opts.SUPPORT_AWARE_TEXT_TOPK < 0:
        raise ValueError("PHASE6.SUPPORT_AWARE_TEXT_TOPK must be non-negative.")
    if opts.LABEL_PRIOR_STRENGTH < 0:
        raise ValueError("PHASE6.LABEL_PRIOR_STRENGTH must be non-negative.")
    if opts.LABEL_PRIOR_EPS <= 0:
        raise ValueError("PHASE6.LABEL_PRIOR_EPS must be positive.")
    if opts.LABEL_PRIOR_MAX_ITER <= 0:
        raise ValueError("PHASE6.LABEL_PRIOR_MAX_ITER must be positive.")


def normalize(x, dim=-1, eps=1e-8):
    safe_eps = _safe_eps(x, eps)
    x = torch.nan_to_num(x)
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    return torch.nan_to_num(x / norm.clamp_min(safe_eps))


def sinkhorn_transport(cost, eps=0.05, max_iter=50):
    if cost.ndim != 2:
        raise ValueError("Expected cost with shape [N, M].")
    if cost.numel() == 0:
        return torch.empty_like(cost)

    dtype = cost.dtype
    device = cost.device
    work = torch.nan_to_num(cost.float())
    n, m = work.shape
    eps = max(float(eps), 1e-8)
    row_marginal = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    col_marginal = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)

    work = work - work.min()
    kernel = torch.exp(-work / eps).clamp_min(1e-30)
    u = torch.ones_like(row_marginal)
    v = torch.ones_like(col_marginal)
    for _ in range(int(max_iter)):
        u = row_marginal / (kernel.matmul(v).clamp_min(1e-30))
        v = col_marginal / (kernel.t().matmul(u).clamp_min(1e-30))

    transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    transport = transport / transport.sum().clamp_min(1e-30)
    return torch.nan_to_num(transport).to(device=device, dtype=dtype)


def ot_text_image_alignment(desc_feats, support_feats, eps=0.05, max_iter=50, normalize_cost=False):
    desc_feats = normalize(desc_feats, dim=-1)
    support_feats = normalize(support_feats, dim=-1)
    if desc_feats.numel() == 0:
        proto = torch.zeros(desc_feats.shape[-1], device=desc_feats.device, dtype=desc_feats.dtype)
        weights = torch.empty(0, device=desc_feats.device, dtype=desc_feats.dtype)
        return proto, weights, _empty_alignment_stats("ot_text_image", "missing_descriptions")
    if support_feats.numel() == 0:
        weights = torch.full(
            (desc_feats.shape[0],),
            1.0 / desc_feats.shape[0],
            device=desc_feats.device,
            dtype=desc_feats.dtype,
        )
        proto = normalize((weights.unsqueeze(-1) * desc_feats).sum(dim=0), dim=-1)
        stats = _alignment_stats(weights, None, "ot_text_image")
        stats["fallback_reason"] = "missing_support_features"
        return proto, weights, stats

    cost = 1.0 - support_feats.matmul(desc_feats.t())
    if normalize_cost and cost.numel():
        cost_std = cost.float().std(unbiased=False).clamp_min(_safe_eps(cost, 1e-8))
        cost = (cost - cost.mean()) / cost_std
    transport = sinkhorn_transport(cost, eps=eps, max_iter=max_iter)
    weights = transport.sum(dim=0)
    weights = weights / weights.sum().clamp_min(_safe_eps(weights, 1e-8))
    proto = normalize((weights.unsqueeze(-1) * desc_feats).sum(dim=0), dim=-1)
    stats = _alignment_stats(weights, cost, "ot_text_image")
    stats["ot_eps"] = float(eps)
    stats["ot_max_iter"] = int(max_iter)
    return proto, weights, stats


def support_aware_text_proto(desc_feats, support_proto, temp=0.05, topk=0):
    desc_feats = normalize(desc_feats, dim=-1)
    support_proto = normalize(support_proto.to(device=desc_feats.device, dtype=desc_feats.dtype), dim=-1)
    if desc_feats.numel() == 0:
        proto = torch.zeros_like(support_proto)
        weights = torch.empty(0, device=support_proto.device, dtype=support_proto.dtype)
        return proto, weights, _empty_alignment_stats("support_aware_text", "missing_descriptions")

    score = desc_feats.matmul(support_proto)
    temp = max(float(temp), _safe_eps(desc_feats, 1e-8))
    if int(topk) > 0 and int(topk) < score.numel():
        keep = torch.topk(score, k=int(topk), dim=0).indices
        weights = torch.zeros_like(score)
        weights[keep] = torch.softmax(score[keep] / temp, dim=0)
    else:
        weights = torch.softmax(score / temp, dim=0)
    proto = normalize((weights.unsqueeze(-1) * desc_feats).sum(dim=0), dim=-1)
    stats = _alignment_stats(weights, score, "support_aware_text")
    stats["support_aware_text_temp"] = float(temp)
    stats["support_aware_text_topk"] = int(topk)
    return proto, weights, stats


def prediction_frequency_prior(probabilities, eps=1e-8):
    probabilities = _row_normalize_probabilities(probabilities, eps=eps)
    prior = probabilities.mean(dim=0)
    return prior / prior.sum().clamp_min(_safe_eps(prior, eps))


def blackbox_shift_prior(probabilities, max_iter=20, eps=1e-8):
    probabilities = _row_normalize_probabilities(probabilities, eps=eps)
    if probabilities.numel() == 0:
        return torch.empty(probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype)

    q = prediction_frequency_prior(probabilities, eps=eps)
    for _ in range(int(max_iter)):
        weighted = probabilities * q.unsqueeze(0)
        posterior = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(_safe_eps(weighted, eps))
        q = posterior.mean(dim=0)
        q = q / q.sum().clamp_min(_safe_eps(q, eps))
    return torch.nan_to_num(q)


def apply_label_prior_correction(scores, prior, strength=0.5, eps=1e-8):
    prior = prior.to(device=scores.device, dtype=scores.dtype)
    prior = prior / prior.sum().clamp_min(_safe_eps(prior, eps))
    corrected = scores + float(strength) * torch.log(prior.clamp_min(_safe_eps(prior, eps))).unsqueeze(0)
    return torch.nan_to_num(corrected)


def label_prior_stats(prior, mode, strength, transductive=True, level="session"):
    prior = prior.detach().float()
    return {
        "label_prior_mode": mode,
        "label_prior_strength": float(strength),
        "label_prior_entropy": _entropy(prior),
        "label_prior_min": _to_float(prior.min()) if prior.numel() else None,
        "label_prior_max": _to_float(prior.max()) if prior.numel() else None,
        "transductive": bool(transductive),
        "transductive_level": level,
    }


def _row_normalize_probabilities(probabilities, eps=1e-8):
    probs = torch.nan_to_num(probabilities)
    if torch.any(probs < 0) or torch.any(probs.sum(dim=1) <= 0):
        probs = F.softmax(probs, dim=-1)
    else:
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(_safe_eps(probs, eps))
    return torch.nan_to_num(probs)


def _alignment_stats(weights, score_or_cost, mode):
    stats = {
        "alignment_mode": mode,
        "weight_entropy": _entropy(weights),
        "weight_min": _to_float(weights.min()) if weights.numel() else None,
        "weight_max": _to_float(weights.max()) if weights.numel() else None,
        "fallback_reason": "",
    }
    if score_or_cost is not None and score_or_cost.numel():
        key = "ot_cost" if mode == "ot_text_image" and score_or_cost.ndim == 2 else "alignment_score"
        stats[f"{key}_mean"] = _to_float(score_or_cost.mean())
        stats[f"{key}_std"] = _to_float(score_or_cost.float().std(unbiased=False))
        stats[f"{key}_min"] = _to_float(score_or_cost.min())
        stats[f"{key}_max"] = _to_float(score_or_cost.max())
    return stats


def _empty_alignment_stats(mode, fallback_reason):
    return {
        "alignment_mode": mode,
        "weight_entropy": None,
        "weight_min": None,
        "weight_max": None,
        "fallback_reason": fallback_reason,
    }


def _entropy(weights, eps=1e-12):
    if weights.numel() == 0:
        return None
    weights_f = weights.float()
    entropy = -(weights_f * torch.log(weights_f.clamp_min(float(eps)))).sum()
    return _to_float(entropy)


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
