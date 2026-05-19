import math

import torch


VALID_SPACE_TRANSFORMS = ("none", "common_direction_removal", "whitening", "lda_shrinkage")
VALID_PHASE5_APPLY_TO = ("query_only", "prototype_only", "all")
VALID_CDR_SOURCES = ("prototypes", "base_features", "all_seen_features")
VALID_WHITENING_SOURCES = ("base_features", "all_seen_features")
VALID_LDA_SOURCES = ("base_plus_support", "base_features")


def validate_phase5_options(cfg):
    opts = cfg.TRAINER.BiMC.PHASE5
    if opts.SPACE_TRANSFORM not in VALID_SPACE_TRANSFORMS:
        raise ValueError(
            f"Invalid PHASE5.SPACE_TRANSFORM {opts.SPACE_TRANSFORM}. "
            f"Expected one of {VALID_SPACE_TRANSFORMS}."
        )
    if opts.APPLY_TO not in VALID_PHASE5_APPLY_TO:
        raise ValueError(
            f"Invalid PHASE5.APPLY_TO {opts.APPLY_TO}. Expected one of {VALID_PHASE5_APPLY_TO}."
        )
    if opts.CDR_SOURCE not in VALID_CDR_SOURCES:
        raise ValueError(
            f"Invalid PHASE5.CDR_SOURCE {opts.CDR_SOURCE}. Expected one of {VALID_CDR_SOURCES}."
        )
    if opts.WHITENING_SOURCE not in VALID_WHITENING_SOURCES:
        raise ValueError(
            f"Invalid PHASE5.WHITENING_SOURCE {opts.WHITENING_SOURCE}. "
            f"Expected one of {VALID_WHITENING_SOURCES}."
        )
    if opts.LDA_SOURCE not in VALID_LDA_SOURCES:
        raise ValueError(
            f"Invalid PHASE5.LDA_SOURCE {opts.LDA_SOURCE}. Expected one of {VALID_LDA_SOURCES}."
        )
    if opts.WHITENING_EPS <= 0:
        raise ValueError("PHASE5.WHITENING_EPS must be positive.")
    if opts.LDA_DIM <= 0:
        raise ValueError("PHASE5.LDA_DIM must be positive.")
    if opts.LDA_GAMMA < 0:
        raise ValueError("PHASE5.LDA_GAMMA must be non-negative.")
    if opts.LDA_NOVEL_WEIGHT < 0:
        raise ValueError("PHASE5.LDA_NOVEL_WEIGHT must be non-negative.")


def normalize(x, dim=-1, eps=1e-8):
    safe_eps = _safe_eps(x, eps)
    x = torch.nan_to_num(x)
    norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
    out = x / norm.clamp_min(safe_eps)
    return torch.nan_to_num(out)


def common_direction_removal(x, direction, rho=0.5, return_stats=False):
    direction = normalize(direction.to(device=x.device, dtype=x.dtype), dim=-1)
    out = normalize(x - float(rho) * direction, dim=-1)
    stats = {
        "transform": "common_direction_removal",
        "rho": float(rho),
        "direction_norm": _to_float(torch.linalg.vector_norm(direction.float(), ord=2)),
        "input_norm_mean": _tensor_norm_mean(x),
        "output_norm_mean": _tensor_norm_mean(out),
        "fallback_reason": "",
    }
    if return_stats:
        return out, stats
    return out


def estimate_common_direction(prototypes=None, features=None, source="prototypes"):
    source_tensor = prototypes if source == "prototypes" else features
    if source_tensor is None or source_tensor.numel() == 0:
        ref = prototypes if prototypes is not None else features
        if ref is None:
            raise ValueError("No tensor is available to estimate common direction.")
        direction = torch.zeros(ref.shape[-1], device=ref.device, dtype=ref.dtype)
        stats = {
            "transform": "common_direction_removal",
            "source": source,
            "num_vectors": 0,
            "common_direction_norm": 0.0,
            "fallback_reason": f"missing_{source}",
        }
        return direction, stats

    direction = normalize(source_tensor.mean(dim=0), dim=-1)
    stats = {
        "transform": "common_direction_removal",
        "source": source,
        "num_vectors": int(source_tensor.shape[0]),
        "common_direction_norm": _to_float(torch.linalg.vector_norm(direction.float(), ord=2)),
        "fallback_reason": "",
    }
    return direction, stats


def diagonal_whitening_fit(features, eps=1e-4):
    if features is None or features.numel() == 0:
        raise ValueError("Cannot fit whitening without features.")

    dtype = features.dtype
    device = features.device
    work = features.float()
    mean = work.mean(dim=0)
    var = work.var(dim=0, unbiased=False) + float(eps)
    inv_std = torch.rsqrt(var.clamp_min(float(eps)))
    transform = {
        "type": "diagonal_whitening",
        "mean": mean.to(device=device, dtype=dtype),
        "inv_std": inv_std.to(device=device, dtype=dtype),
        "stats": _whitening_stats(var, "diagonal_whitening", fallback_reason=""),
    }
    return transform


def diagonal_whitening_apply(x, transform, return_stats=False):
    mean = transform["mean"].to(device=x.device, dtype=x.dtype)
    inv_std = transform["inv_std"].to(device=x.device, dtype=x.dtype)
    out = normalize((x - mean) * inv_std, dim=-1)
    stats = {
        "transform": "diagonal_whitening",
        "output_norm_mean": _tensor_norm_mean(out),
        "fallback_reason": transform.get("stats", {}).get("fallback_reason", ""),
    }
    if return_stats:
        return out, stats
    return out


def full_whitening_fit(features, eps=1e-4):
    if features is None or features.numel() == 0:
        raise ValueError("Cannot fit whitening without features.")
    if features.ndim != 2:
        raise ValueError("Expected features with shape [N, D].")
    if features.shape[0] < 2:
        transform = diagonal_whitening_fit(features, eps=eps)
        transform["type"] = "diagonal_whitening"
        transform["stats"]["fallback_reason"] = "full_whitening_needs_at_least_two_samples"
        return transform

    dtype = features.dtype
    device = features.device
    try:
        work = features.float()
        mean = work.mean(dim=0)
        centered = work - mean
        cov = centered.t().matmul(centered) / max(1, centered.shape[0] - 1)
        eye = torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
        cov = cov + float(eps) * eye
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(float(eps))
        inv_sqrt = eigvecs.matmul(torch.diag(torch.rsqrt(eigvals))).matmul(eigvecs.t())
        transform = {
            "type": "full_whitening",
            "mean": mean.to(device=device, dtype=dtype),
            "inv_sqrt": inv_sqrt.to(device=device, dtype=dtype),
            "stats": _whitening_stats(eigvals, "full_whitening", fallback_reason=""),
        }
        transform["stats"]["whitening_eig_min"] = _to_float(eigvals.min())
        transform["stats"]["whitening_eig_max"] = _to_float(eigvals.max())
        return transform
    except RuntimeError as exc:
        transform = diagonal_whitening_fit(features, eps=eps)
        transform["stats"]["fallback_reason"] = f"full_whitening_failed:{type(exc).__name__}"
        return transform


def full_whitening_apply(x, transform, return_stats=False):
    if transform.get("type") != "full_whitening":
        return diagonal_whitening_apply(x, transform, return_stats=return_stats)
    mean = transform["mean"].to(device=x.device, dtype=x.dtype)
    inv_sqrt = transform["inv_sqrt"].to(device=x.device, dtype=x.dtype)
    out = normalize((x - mean).matmul(inv_sqrt), dim=-1)
    stats = {
        "transform": "full_whitening",
        "output_norm_mean": _tensor_norm_mean(out),
        "fallback_reason": transform.get("stats", {}).get("fallback_reason", ""),
    }
    if return_stats:
        return out, stats
    return out


def lda_shrinkage_fit(
    base_features_by_class,
    support_features_by_class=None,
    dim=256,
    gamma=1e-3,
    novel_weight=0.1,
):
    class_blocks = _collect_lda_blocks(base_features_by_class, weight=1.0, prefix="base")
    class_blocks.extend(_collect_lda_blocks(support_features_by_class, weight=float(novel_weight), prefix="support"))
    class_blocks = [block for block in class_blocks if block["features"].numel() > 0]
    if len(class_blocks) < 2:
        raise ValueError("LDA needs features from at least two classes.")

    ref = class_blocks[0]["features"]
    device = ref.device
    dtype = ref.dtype
    work_blocks = []
    for block in class_blocks:
        work_blocks.append({
            "features": block["features"].to(device=device).float(),
            "weight": float(block["weight"]),
            "source": block["source"],
        })

    d = work_blocks[0]["features"].shape[-1]
    means = []
    class_weights = []
    sw = torch.zeros(d, d, device=device, dtype=torch.float32)
    for block in work_blocks:
        feats = block["features"]
        mean = feats.mean(dim=0)
        centered = feats - mean
        weight = block["weight"]
        means.append(mean)
        class_weights.append(weight * max(1, feats.shape[0]))
        sw = sw + weight * centered.t().matmul(centered) / max(1, feats.shape[0])

    means = torch.stack(means, dim=0)
    class_weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
    overall = (class_weights_tensor.unsqueeze(-1) * means).sum(dim=0) / class_weights_tensor.sum().clamp_min(1e-12)

    sb = torch.zeros_like(sw)
    for mean, weight in zip(means, class_weights):
        diff = (mean - overall).unsqueeze(1)
        sb = sb + float(weight) * diff.matmul(diff.t())

    trace = torch.trace(sw) / max(1, d)
    ridge = float(gamma) * (trace.item() if torch.isfinite(trace) else 1.0)
    if ridge <= 0:
        ridge = float(gamma) if gamma > 0 else 1e-6
    eye = torch.eye(d, device=device, dtype=torch.float32)
    sw = sw + ridge * eye

    fallback_reason = ""
    try:
        sw_eigvals, sw_eigvecs = torch.linalg.eigh(sw)
        sw_eigvals = sw_eigvals.clamp_min(max(ridge, 1e-6))
        sw_inv_sqrt = sw_eigvecs.matmul(torch.diag(torch.rsqrt(sw_eigvals))).matmul(sw_eigvecs.t())
        mat = sw_inv_sqrt.matmul(sb).matmul(sw_inv_sqrt)
        eigvals, eigvecs = torch.linalg.eigh(mat)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        max_rank = min(int(dim), d, max(1, len(work_blocks) - 1))
        projection = sw_inv_sqrt.matmul(eigvecs[:, :max_rank])
    except RuntimeError as exc:
        fallback_reason = f"lda_whitened_generalized_failed:{type(exc).__name__}"
        eigvals, eigvecs = torch.linalg.eigh(sb + ridge * eye)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        max_rank = min(int(dim), d, max(1, len(work_blocks) - 1))
        projection = eigvecs[:, :max_rank]

    projection = normalize(projection, dim=0).to(device=device, dtype=dtype)
    stats = {
        "transform": "lda_shrinkage",
        "lda_dim_requested": int(dim),
        "lda_dim_used": int(projection.shape[1]),
        "lda_num_classes": int(len(work_blocks)),
        "lda_num_base_classes": int(sum(1 for block in work_blocks if block["source"] == "base")),
        "lda_num_support_classes": int(sum(1 for block in work_blocks if block["source"] == "support")),
        "lda_gamma": float(gamma),
        "lda_novel_weight": float(novel_weight),
        "lda_eigen_min": _to_float(eigvals[:projection.shape[1]].min()) if projection.shape[1] else None,
        "lda_eigen_max": _to_float(eigvals[:projection.shape[1]].max()) if projection.shape[1] else None,
        "lda_eigen_mean": _to_float(eigvals[:projection.shape[1]].mean()) if projection.shape[1] else None,
        "fallback_reason": fallback_reason,
    }
    return {"type": "lda_shrinkage", "projection": projection, "stats": stats}


def lda_apply(x, projection, return_stats=False):
    if isinstance(projection, dict):
        projection_tensor = projection["projection"]
        fallback_reason = projection.get("stats", {}).get("fallback_reason", "")
    else:
        projection_tensor = projection
        fallback_reason = ""
    projection_tensor = projection_tensor.to(device=x.device, dtype=x.dtype)
    out = normalize(x.matmul(projection_tensor), dim=-1)
    stats = {
        "transform": "lda_shrinkage",
        "lda_dim_used": int(out.shape[-1]),
        "output_norm_mean": _tensor_norm_mean(out),
        "fallback_reason": fallback_reason,
    }
    if return_stats:
        return out, stats
    return out


def _collect_lda_blocks(features_by_class, weight, prefix):
    if not features_by_class:
        return []
    if isinstance(features_by_class, dict):
        values = features_by_class.values()
    else:
        values = features_by_class
    blocks = []
    for features in values:
        if features is None or features.numel() == 0:
            continue
        blocks.append({"features": features, "weight": weight, "source": prefix})
    return blocks


def _whitening_stats(values, transform, fallback_reason=""):
    values = values.detach().float()
    return {
        "transform": transform,
        "whitening_var_mean": _to_float(values.mean()),
        "whitening_var_std": _to_float(values.std(unbiased=False)),
        "whitening_var_min": _to_float(values.min()),
        "whitening_var_max": _to_float(values.max()),
        "fallback_reason": fallback_reason,
    }


def _tensor_norm_mean(x):
    if x.numel() == 0:
        return None
    return _to_float(torch.linalg.vector_norm(x.detach().float(), ord=2, dim=-1).mean())


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
