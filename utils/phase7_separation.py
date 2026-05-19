import math

import torch


VALID_SEPARATION_MODES = ("none", "hubness_safe_repulsion", "prototype_repulsion", "graph_highpass")
VALID_REPULSION_SOURCES = ("mixed", "text", "visual")


def validate_phase7_options(cfg):
    opts = cfg.TRAINER.BiMC.PHASE7
    if opts.SEPARATION_MODE not in VALID_SEPARATION_MODES:
        raise ValueError(
            f"Invalid PHASE7.SEPARATION_MODE {opts.SEPARATION_MODE}. "
            f"Expected one of {VALID_SEPARATION_MODES}."
        )
    if opts.REPULSION_SOURCE not in VALID_REPULSION_SOURCES:
        raise ValueError(
            f"Invalid PHASE7.REPULSION_SOURCE {opts.REPULSION_SOURCE}. "
            f"Expected one of {VALID_REPULSION_SOURCES}."
        )
    if opts.REPULSION_DELTA < 0:
        raise ValueError("PHASE7.REPULSION_DELTA must be non-negative.")
    if opts.REPULSION_TOPK < 0:
        raise ValueError("PHASE7.REPULSION_TOPK must be non-negative.")
    if opts.GRAPH_TAU <= 0:
        raise ValueError("PHASE7.GRAPH_TAU must be positive.")
    if opts.GRAPH_GAMMA < 0:
        raise ValueError("PHASE7.GRAPH_GAMMA must be non-negative.")
    if opts.GRAPH_TOPK < 0:
        raise ValueError("PHASE7.GRAPH_TOPK must be non-negative.")


def normalize(x, dim=-1, eps=1e-8):
    safe_eps = _safe_eps(x, eps)
    x = torch.nan_to_num(x)
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    return torch.nan_to_num(x / norm.clamp_min(safe_eps))


def compute_pairwise_cosine(prototypes):
    prototypes = normalize(prototypes, dim=-1)
    return torch.nan_to_num(prototypes.matmul(prototypes.t()))


def prototype_repulsion(prototypes, delta=0.03, margin=0.0, topk=5):
    prototypes = normalize(prototypes, dim=-1)
    if prototypes.shape[0] <= 1 or float(delta) == 0:
        return prototypes, _separation_stats(prototypes, prototypes, "prototype_repulsion", 0)

    sim = compute_pairwise_cosine(prototypes)
    sim = sim.clone()
    sim.fill_diagonal_(-torch.inf)
    k = _neighbor_count(sim.shape[0], topk)
    if k == 0:
        return prototypes, _separation_stats(prototypes, prototypes, "prototype_repulsion", 0)

    values, indices = torch.topk(sim, k=k, dim=1)
    neighbor_proto = prototypes[indices]
    weights = torch.relu(values - float(margin))
    repulsion = (weights.unsqueeze(-1) * neighbor_proto).sum(dim=1)
    affected = weights.sum(dim=1) > 0
    updated = prototypes.clone()
    updated[affected] = normalize(prototypes[affected] - float(delta) * repulsion[affected], dim=-1)
    return updated, _separation_stats(prototypes, updated, "prototype_repulsion", int(affected.sum().item()))


def graph_highpass_correction(prototypes, tau=0.05, gamma=0.03, topk=5, normalize_adj=True):
    prototypes = normalize(prototypes, dim=-1)
    c = prototypes.shape[0]
    if c <= 1 or float(gamma) == 0:
        return prototypes, _separation_stats(prototypes, prototypes, "graph_highpass", 0)

    sim = compute_pairwise_cosine(prototypes)
    sim = sim.clone()
    sim.fill_diagonal_(-torch.inf)
    k = _neighbor_count(c, topk)
    if k == 0:
        adj = torch.zeros_like(sim)
    else:
        values, indices = torch.topk(sim, k=k, dim=1)
        weights = torch.softmax(values / max(float(tau), _safe_eps(sim, 1e-8)), dim=1)
        adj = torch.zeros_like(sim)
        adj.scatter_(1, indices, weights)

    if normalize_adj:
        degree = adj.sum(dim=1).clamp_min(_safe_eps(adj, 1e-8))
        inv_sqrt = torch.rsqrt(degree)
        adj = inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
    else:
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(_safe_eps(adj, 1e-8))

    smoothed = adj.matmul(prototypes)
    highpass = prototypes + float(gamma) * (prototypes - smoothed)
    updated = normalize(highpass, dim=-1)
    affected = int((torch.linalg.vector_norm(updated - prototypes, ord=2, dim=-1) > 1e-12).sum().item())
    stats = _separation_stats(prototypes, updated, "graph_highpass", affected)
    stats["graph_tau"] = float(tau)
    stats["graph_gamma"] = float(gamma)
    stats["graph_topk"] = int(topk)
    stats["graph_normalize_adj"] = bool(normalize_adj)
    return updated, stats


def hubness_safe_repulsion(prototypes, hubness, delta=0.03, topk=5):
    prototypes = normalize(prototypes, dim=-1)
    hubness = hubness.to(device=prototypes.device, dtype=prototypes.dtype).reshape(-1)
    if hubness.numel() != prototypes.shape[0]:
        raise ValueError("Hubness vector must have one value per prototype.")
    if prototypes.shape[0] <= 1 or float(delta) == 0:
        return prototypes, _separation_stats(prototypes, prototypes, "hubness_safe_repulsion", 0, hubness, hubness)

    mean = hubness.mean()
    std = hubness.float().std(unbiased=False).to(dtype=hubness.dtype).clamp_min(_safe_eps(hubness, 1e-8))
    z = (hubness - mean) / std
    mask = z > 0
    repelled, _ = prototype_repulsion(prototypes, delta=delta, margin=0.0, topk=topk)
    updated = prototypes.clone()
    updated[mask] = repelled[mask]
    stats = _separation_stats(prototypes, updated, "hubness_safe_repulsion", int(mask.sum().item()), hubness, _estimate_hubness(updated, prototypes))
    stats["hubness_threshold"] = "z>0"
    return updated, stats


def _estimate_hubness(updated_prototypes, reference_features):
    scores = normalize(reference_features, dim=-1).matmul(normalize(updated_prototypes, dim=-1).t())
    winners = scores.argmax(dim=1)
    return torch.bincount(winners, minlength=updated_prototypes.shape[0]).to(device=updated_prototypes.device, dtype=updated_prototypes.dtype)


def _separation_stats(before, after, mode, affected_count, hubness_before=None, hubness_after=None):
    pairwise = compute_pairwise_cosine(before)
    off_diag = _off_diag(pairwise)
    displacement = torch.linalg.vector_norm((after - before).detach().float(), ord=2, dim=-1)
    stats = {
        "separation_mode": mode,
        "pairwise_similarity_mean": _to_float(off_diag.mean()) if off_diag.numel() else None,
        "pairwise_similarity_std": _to_float(off_diag.float().std(unbiased=False)) if off_diag.numel() else None,
        "pairwise_similarity_max": _to_float(off_diag.max()) if off_diag.numel() else None,
        "affected_prototypes": int(affected_count),
        "displacement_norm_mean": _to_float(displacement.mean()) if displacement.numel() else None,
        "displacement_norm_std": _to_float(displacement.std(unbiased=False)) if displacement.numel() else None,
        "fallback_reason": "",
    }
    if hubness_before is not None and hubness_before.numel():
        stats["hubness_before_mean"] = _to_float(hubness_before.float().mean())
        stats["hubness_before_std"] = _to_float(hubness_before.float().std(unbiased=False))
        stats["hubness_before_max"] = _to_float(hubness_before.float().max())
    if hubness_after is not None and hubness_after.numel():
        stats["hubness_after_mean"] = _to_float(hubness_after.float().mean())
        stats["hubness_after_std"] = _to_float(hubness_after.float().std(unbiased=False))
        stats["hubness_after_max"] = _to_float(hubness_after.float().max())
    return stats


def _off_diag(matrix):
    if matrix.shape[0] <= 1:
        return torch.empty(0, device=matrix.device, dtype=matrix.dtype)
    mask = ~torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.bool)
    return matrix[mask]


def _neighbor_count(num_prototypes, topk):
    if int(topk) <= 0:
        return max(0, num_prototypes - 1)
    return max(0, min(int(topk), num_prototypes - 1))


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
