import math

import torch
import torch.nn.functional as F


VALID_HUBNESS_SOURCES = ("mixed", "text", "visual", "all")
VALID_DYNAMIC_ALPHA_MODES = (
    "margin",
    "entropy",
    "entropy_margin",
    "energy",
    "entropy_margin_energy",
)


def normalize_scores(scores, temp=1.0, eps=1e-8):
    """Apply scalar temperature normalization while preserving dtype/device."""
    safe_temp = _safe_scalar_tensor(temp, scores, eps)
    return torch.nan_to_num(scores / safe_temp)


def normalized_entropy(p, eps=1e-8):
    num_classes = p.shape[-1]
    if num_classes <= 1:
        return torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)

    safe_eps = _dtype_eps(p, eps)
    probs = torch.clamp(p, min=safe_eps)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    entropy = entropy / math.log(num_classes)
    return torch.nan_to_num(entropy)


def top1_margin(p):
    k = min(2, p.shape[-1])
    top_values = p.topk(k=k, dim=-1).values
    if k == 1:
        return torch.nan_to_num(top_values[..., 0])
    return torch.nan_to_num(top_values[..., 0] - top_values[..., 1])


def energy_score(scores, temp=1.0, eps=1e-8):
    safe_temp = _safe_scalar_tensor(temp, scores, eps)
    scores_t = scores / safe_temp
    energy = -safe_temp * torch.logsumexp(scores_t, dim=-1)
    return torch.nan_to_num(energy)


def confidence_from_energy(energy, normalize=True, eps=1e-8):
    rel = torch.nan_to_num(-energy)
    if not normalize:
        return rel

    safe_eps = _dtype_eps(rel, eps)
    rel_min = rel.min()
    rel_max = rel.max()
    rel = (rel - rel_min) / (rel_max - rel_min + safe_eps)
    return torch.nan_to_num(rel)


def reliability_from_scores(scores, mode="entropy_margin", temp=1.0, eps=1e-8):
    if mode not in VALID_DYNAMIC_ALPHA_MODES:
        raise ValueError(f"Invalid reliability mode {mode}. Expected one of {VALID_DYNAMIC_ALPHA_MODES}.")

    scores_t = normalize_scores(scores, temp=temp, eps=eps)
    p = F.softmax(scores_t, dim=-1)
    confidence = 1 - normalized_entropy(p, eps=eps)
    margin = top1_margin(p)
    energy = energy_score(scores, temp=temp, eps=eps)
    energy_rel = confidence_from_energy(energy, normalize=True, eps=eps)

    if mode == "margin":
        reliability = margin
    elif mode == "entropy":
        reliability = confidence
    elif mode == "entropy_margin":
        reliability = confidence * margin
    elif mode == "energy":
        reliability = energy_rel
    elif mode == "entropy_margin_energy":
        reliability = confidence * margin * energy_rel
    else:
        raise ValueError(f"Invalid reliability mode {mode}. Expected one of {VALID_DYNAMIC_ALPHA_MODES}.")

    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)
    stats = {
        "mode": mode,
        "temperature": _float_value(_safe_scalar_tensor(temp, scores, eps)),
        "confidence_mean": _mean_value(confidence),
        "confidence_std": _std_value(confidence),
        "margin_mean": _mean_value(margin),
        "margin_std": _std_value(margin),
        "energy_mean": _mean_value(energy),
        "energy_std": _std_value(energy),
        "energy_reliability_mean": _mean_value(energy_rel),
        "energy_reliability_std": _std_value(energy_rel),
        "reliability_mean": _mean_value(reliability),
        "reliability_std": _std_value(reliability),
        "reliability_min": _min_value(reliability),
        "reliability_max": _max_value(reliability),
    }
    return reliability, stats


def compute_prototype_hubness(protos, tau=0.05, topk=0, eps=1e-8):
    protos = F.normalize(protos, dim=-1)
    num_classes = protos.shape[0]
    if num_classes <= 1:
        hubness = torch.zeros(num_classes, device=protos.device, dtype=protos.dtype)
        return hubness, _hubness_stats(hubness, hubness, tau, topk)

    tau_t = _safe_scalar_tensor(tau, protos, eps)
    scores = protos @ protos.t()
    diagonal = torch.eye(num_classes, device=protos.device, dtype=torch.bool)
    scores = scores.masked_fill(diagonal, float("-inf"))

    if topk and topk > 0:
        k = min(int(topk), num_classes - 1)
        selected = scores.topk(k=k, dim=1).values
        raw_hubness = torch.logsumexp(selected / tau_t, dim=1)
    else:
        raw_hubness = torch.logsumexp(scores / tau_t, dim=1)

    raw_hubness = torch.nan_to_num(raw_hubness)
    mean = raw_hubness.mean()
    std = raw_hubness.std(unbiased=False)
    safe_eps = _dtype_eps(raw_hubness, eps)
    hubness = (raw_hubness - mean) / (std + safe_eps)
    hubness = torch.nan_to_num(hubness)
    return hubness, _hubness_stats(hubness, raw_hubness, tau, topk)


def apply_hubness_correction(scores, hubness, lambda_h=0.05):
    hubness = hubness.to(device=scores.device, dtype=scores.dtype)
    lambda_t = torch.as_tensor(lambda_h, device=scores.device, dtype=scores.dtype)
    corrected = scores - lambda_t * hubness.view(1, -1)
    return torch.nan_to_num(corrected)


def dynamic_alpha_from_scores(
    calib_scores,
    aux_scores,
    mode,
    temp_calib=1.0,
    temp_aux=1.0,
    min_alpha=0.05,
    max_alpha=0.95,
    eps=1e-8,
):
    r_calib, calib_stats = reliability_from_scores(calib_scores, mode, temp=temp_calib, eps=eps)
    r_aux, aux_stats = reliability_from_scores(aux_scores, mode, temp=temp_aux, eps=eps)
    safe_eps = _dtype_eps(calib_scores, eps)
    alpha_x = r_calib / (r_calib + r_aux + safe_eps)
    alpha_x = torch.nan_to_num(alpha_x, nan=0.5, posinf=max_alpha, neginf=min_alpha)
    alpha_x = torch.clamp(alpha_x, min=min_alpha, max=max_alpha)
    stats = {
        "mode": mode,
        "temp_calib": _float_value(_safe_scalar_tensor(temp_calib, calib_scores, eps)),
        "temp_aux": _float_value(_safe_scalar_tensor(temp_aux, aux_scores, eps)),
        "alpha_mean": _mean_value(alpha_x),
        "alpha_std": _std_value(alpha_x),
        "alpha_min": _min_value(alpha_x),
        "alpha_max": _max_value(alpha_x),
        "calib_reliability_mean": calib_stats["reliability_mean"],
        "calib_reliability_std": calib_stats["reliability_std"],
        "aux_reliability_mean": aux_stats["reliability_mean"],
        "aux_reliability_std": aux_stats["reliability_std"],
        "calib_energy_mean": calib_stats["energy_mean"],
        "calib_energy_std": calib_stats["energy_std"],
        "aux_energy_mean": aux_stats["energy_mean"],
        "aux_energy_std": aux_stats["energy_std"],
    }
    return alpha_x, stats


def combine_with_dynamic_alpha(p_calib, p_aux, alpha_x):
    alpha_x = alpha_x.to(device=p_calib.device, dtype=p_calib.dtype)
    combined = alpha_x.unsqueeze(-1) * p_calib + (1 - alpha_x.unsqueeze(-1)) * p_aux
    return torch.nan_to_num(combined)


def tensor_stats(values, prefix=None):
    values = _flatten_tensor(values)
    if values.numel() == 0:
        stats = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    else:
        stats = {
            "count": int(values.numel()),
            "mean": _mean_value(values),
            "std": _std_value(values),
            "min": _min_value(values),
            "max": _max_value(values),
        }
    if prefix:
        return {f"{prefix}_{key}": value for key, value in stats.items()}
    return stats


def _safe_scalar_tensor(value, ref, eps):
    eps_value = _dtype_eps(ref, eps)
    if isinstance(value, torch.Tensor):
        value_t = value.to(device=ref.device, dtype=ref.dtype)
    else:
        value_t = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
    return torch.clamp(value_t, min=eps_value)


def _dtype_eps(ref, eps):
    if torch.is_floating_point(ref):
        return max(float(eps), torch.finfo(ref.dtype).tiny)
    return float(eps)


def _hubness_stats(hubness, raw_hubness, tau, topk):
    return {
        "hubness_mean": _mean_value(hubness),
        "hubness_std": _std_value(hubness),
        "hubness_min": _min_value(hubness),
        "hubness_max": _max_value(hubness),
        "hubness_raw_mean": _mean_value(raw_hubness),
        "hubness_raw_std": _std_value(raw_hubness),
        "hubness_tau": float(tau),
        "hubness_topk": int(topk or 0),
    }


def _flatten_tensor(values):
    if values is None:
        return torch.empty(0)
    if isinstance(values, torch.Tensor):
        return values.detach().reshape(-1).float().cpu()
    return torch.as_tensor(values).detach().reshape(-1).float().cpu()


def _mean_value(values):
    values = _flatten_tensor(values)
    return None if values.numel() == 0 else float(values.mean().item())


def _std_value(values):
    values = _flatten_tensor(values)
    return None if values.numel() == 0 else float(values.std(unbiased=False).item())


def _min_value(values):
    values = _flatten_tensor(values)
    return None if values.numel() == 0 else float(values.min().item())


def _max_value(values):
    values = _flatten_tensor(values)
    return None if values.numel() == 0 else float(values.max().item())


def _float_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].float().cpu().item())
    return float(value)
