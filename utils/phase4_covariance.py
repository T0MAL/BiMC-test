import math

import torch
import torch.nn.functional as F


VALID_COV_MODES = ("original", "diag_shrinkage", "base_borrowed_diag", "hybrid_diag")
VALID_COV_PRIORS = ("base_neighbors", "global", "identity")
VALID_NOVEL_AUX_COMBINE_MODES = ("alpha", "average", "max", "replace")


def safe_diag_variance(features, center=None, eps=1e-4):
    if center is None:
        if features.shape[0] == 0:
            center = torch.zeros(features.shape[-1], device=features.device, dtype=features.dtype)
        else:
            center = features.mean(dim=0)
    else:
        center = center.to(device=features.device, dtype=features.dtype)

    if features.shape[0] == 0:
        var = torch.full_like(center, _dtype_eps(center, eps))
    else:
        var = ((features - center.view(1, -1)) ** 2).mean(dim=0)
        var = var + _dtype_eps(var, eps)
    return torch.nan_to_num(var, nan=_dtype_eps(var, eps), posinf=1.0, neginf=_dtype_eps(var, eps))


def compute_base_class_diag_covariances(base_features_by_class, base_protos, eps=1e-4, return_stats=False):
    vars_by_class = []
    class_records = []
    for class_id in range(base_protos.shape[0]):
        features = _features_for_class(base_features_by_class, class_id, base_protos)
        var = safe_diag_variance(features, center=base_protos[class_id], eps=eps)
        vars_by_class.append(var)
        class_records.append({
            "class_id": int(class_id),
            "var_mean": _mean_value(var),
            "var_std": _std_value(var),
            "var_min": _min_value(var),
            "var_max": _max_value(var),
        })
    base_diag_vars = torch.stack(vars_by_class, dim=0)
    stats = tensor_stats(base_diag_vars, prefix="base_diag_var")
    stats["base_class_count"] = int(base_protos.shape[0])
    if return_stats:
        return base_diag_vars, stats, class_records
    return base_diag_vars


def base_borrowed_diag_covariance(novel_proto, base_protos, base_diag_vars, tau=16.0, eps=1e-4):
    novel_proto = F.normalize(novel_proto, dim=-1)
    base_protos = F.normalize(base_protos.to(device=novel_proto.device, dtype=novel_proto.dtype), dim=-1)
    base_diag_vars = base_diag_vars.to(device=novel_proto.device, dtype=novel_proto.dtype)
    tau_t = _safe_scalar_tensor(tau, novel_proto, eps)

    scores = tau_t * (base_protos @ novel_proto)
    weights = F.softmax(scores, dim=0)
    weights = torch.nan_to_num(weights, nan=1.0 / max(1, weights.numel()))
    weights = weights / weights.sum().clamp_min(_dtype_eps(weights, eps))
    var_prior = weights @ base_diag_vars
    var_prior = torch.clamp(torch.nan_to_num(var_prior), min=_dtype_eps(var_prior, eps))

    entropy = -(weights * torch.log(weights.clamp_min(_dtype_eps(weights, eps)))).sum()
    entropy_norm = entropy / math.log(weights.numel()) if weights.numel() > 1 else torch.zeros_like(entropy)
    stats = {
        "borrowed_variance_mean": _mean_value(var_prior),
        "borrowed_variance_std": _std_value(var_prior),
        "borrowed_weight_entropy": _float_value(entropy),
        "borrowed_weight_entropy_norm": _float_value(entropy_norm),
        "borrowed_weight_max": _max_value(weights),
        "borrowed_tau": float(tau),
    }
    return var_prior, weights, stats


def diag_shrinkage_covariance(shot_features, shot_proto, prior_var, shrinkage_lambda=0.5, eps=1e-4):
    shot_features = shot_features.to(device=shot_proto.device, dtype=shot_proto.dtype)
    prior_var = prior_var.to(device=shot_proto.device, dtype=shot_proto.dtype)
    lam = min(1.0, max(0.0, float(shrinkage_lambda)))
    shot_var = safe_diag_variance(shot_features, center=shot_proto, eps=eps)
    var = (1 - lam) * shot_var + lam * prior_var + _dtype_eps(shot_var, eps)
    var = torch.clamp(torch.nan_to_num(var), min=_dtype_eps(var, eps))
    stats = {
        "shot_variance_mean": _mean_value(shot_var),
        "shot_variance_std": _std_value(shot_var),
        "prior_variance_mean": _mean_value(prior_var),
        "prior_variance_std": _std_value(prior_var),
        "shrinkage_lambda": lam,
        "diag_variance_mean": _mean_value(var),
        "diag_variance_std": _std_value(var),
    }
    return var, stats


def mahalanobis_diag_score(query_features, class_protos, class_diag_vars, temp=1.0, eps=1e-4):
    class_protos = class_protos.to(device=query_features.device, dtype=query_features.dtype)
    class_diag_vars = class_diag_vars.to(device=query_features.device, dtype=query_features.dtype)
    class_diag_vars = torch.clamp(class_diag_vars, min=_dtype_eps(class_diag_vars, eps))
    diff = query_features.unsqueeze(1) - class_protos.unsqueeze(0)
    scores = -((diff ** 2) / class_diag_vars.unsqueeze(0)).mean(dim=-1)
    scores = scores / _safe_scalar_tensor(temp, query_features, eps)
    return torch.nan_to_num(scores)


def build_phase4_class_diag_vars(
    support_features,
    support_labels,
    class_protos,
    num_base_classes,
    cov_mode="original",
    cov_prior="base_neighbors",
    shrinkage_lambda=0.5,
    eps=1e-4,
    tau=16.0,
    apply_to_novel=True,
    apply_to_base=False,
):
    if cov_mode not in VALID_COV_MODES:
        raise ValueError(f"Invalid COV_MODE {cov_mode}. Expected one of {VALID_COV_MODES}.")
    if cov_prior not in VALID_COV_PRIORS:
        raise ValueError(f"Invalid COV_PRIOR {cov_prior}. Expected one of {VALID_COV_PRIORS}.")
    if cov_mode == "original":
        return None, {"cov_mode": cov_mode}, []

    support_features = support_features.to(device=class_protos.device, dtype=class_protos.dtype)
    support_labels = support_labels.to(device=class_protos.device).long()
    class_protos = class_protos.to(device=support_features.device, dtype=support_features.dtype)

    num_classes, dim = class_protos.shape
    num_base = min(int(num_base_classes), num_classes)
    features_by_class = {
        class_id: support_features[support_labels == class_id]
        for class_id in range(num_classes)
    }
    base_features_by_class = {
        class_id: features_by_class[class_id]
        for class_id in range(num_base)
    }
    if num_base > 0:
        base_diag_vars, _, _ = compute_base_class_diag_covariances(
            base_features_by_class,
            class_protos[:num_base],
            eps=eps,
            return_stats=True,
        )
    else:
        base_diag_vars = torch.ones(0, dim, device=support_features.device, dtype=support_features.dtype)

    base_feature_chunks = [features_by_class[class_id] for class_id in range(num_base) if features_by_class[class_id].numel() > 0]
    if base_feature_chunks:
        global_features = torch.cat(base_feature_chunks, dim=0)
    elif support_features.numel() > 0:
        global_features = support_features
    else:
        global_features = torch.zeros(1, dim, device=support_features.device, dtype=support_features.dtype)
    global_var = safe_diag_variance(global_features, center=global_features.mean(dim=0), eps=eps)
    identity_var = torch.ones_like(global_var)

    class_vars = []
    class_records = []
    borrowed_entropies = []
    borrowed_means = []
    shot_means = []
    active_var_values = []

    for class_id in range(num_classes):
        is_base = class_id < num_base
        active = (is_base and apply_to_base) or ((not is_base) and apply_to_novel)
        shot_features = features_by_class[class_id]
        shot_proto = class_protos[class_id]
        shot_var = safe_diag_variance(shot_features, center=shot_proto, eps=eps)
        prior_var = _prior_for_mode(
            class_id=class_id,
            is_base=is_base,
            cov_mode=cov_mode,
            cov_prior=cov_prior,
            shot_proto=shot_proto,
            base_protos=class_protos[:num_base],
            base_diag_vars=base_diag_vars,
            global_var=global_var,
            identity_var=identity_var,
            tau=tau,
            eps=eps,
        )
        borrowed_stats = prior_var["stats"]
        prior_tensor = prior_var["var"]

        if not active:
            var = shot_var
            shrink_stats = {
                "shot_variance_mean": _mean_value(shot_var),
                "diag_variance_mean": _mean_value(var),
                "diag_variance_std": _std_value(var),
                "shrinkage_lambda": 0.0,
            }
        elif cov_mode == "base_borrowed_diag" and not is_base and num_base > 0:
            var = torch.clamp(prior_tensor + _dtype_eps(prior_tensor, eps), min=_dtype_eps(prior_tensor, eps))
            shrink_stats = {
                "shot_variance_mean": _mean_value(shot_var),
                "shot_variance_std": _std_value(shot_var),
                "prior_variance_mean": _mean_value(prior_tensor),
                "prior_variance_std": _std_value(prior_tensor),
                "shrinkage_lambda": 1.0,
                "diag_variance_mean": _mean_value(var),
                "diag_variance_std": _std_value(var),
            }
        else:
            var, shrink_stats = diag_shrinkage_covariance(
                shot_features,
                shot_proto,
                prior_tensor,
                shrinkage_lambda=shrinkage_lambda,
                eps=eps,
            )

        class_vars.append(var)
        if active:
            active_var_values.append(var)
        if shrink_stats.get("shot_variance_mean") is not None:
            shot_means.append(shrink_stats["shot_variance_mean"])
        if borrowed_stats.get("borrowed_variance_mean") is not None:
            borrowed_means.append(borrowed_stats["borrowed_variance_mean"])
        if borrowed_stats.get("borrowed_weight_entropy") is not None:
            borrowed_entropies.append(borrowed_stats["borrowed_weight_entropy"])

        class_records.append({
            "class_id": int(class_id),
            "is_base": bool(is_base),
            "active": bool(active),
            "cov_mode": cov_mode,
            "cov_prior": prior_var["name"],
            "var_mean": _mean_value(var),
            "var_std": _std_value(var),
            "var_min": _min_value(var),
            "var_max": _max_value(var),
            "shot_variance_mean": shrink_stats.get("shot_variance_mean"),
            "borrowed_variance_mean": borrowed_stats.get("borrowed_variance_mean"),
            "borrowed_weight_entropy": borrowed_stats.get("borrowed_weight_entropy"),
            "shrinkage_lambda": shrink_stats.get("shrinkage_lambda"),
        })

    class_diag_vars = torch.stack(class_vars, dim=0)
    active_tensor = torch.stack(active_var_values, dim=0) if active_var_values else class_diag_vars
    stats = tensor_stats(active_tensor, prefix="phase4_var")
    stats.update({
        "cov_mode": cov_mode,
        "cov_prior": cov_prior,
        "shrinkage_lambda": float(shrinkage_lambda),
        "shot_variance_mean": _average_or_none(shot_means),
        "borrowed_variance_mean": _average_or_none(borrowed_means),
        "borrowed_weight_entropy": _average_or_none(borrowed_entropies),
        "num_classes": int(num_classes),
        "num_base_classes": int(num_base),
        "apply_to_base": bool(apply_to_base),
        "apply_to_novel": bool(apply_to_novel),
    })
    return class_diag_vars, stats, class_records


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


def _prior_for_mode(
    class_id,
    is_base,
    cov_mode,
    cov_prior,
    shot_proto,
    base_protos,
    base_diag_vars,
    global_var,
    identity_var,
    tau,
    eps,
):
    if cov_mode in ("base_borrowed_diag", "hybrid_diag") and not is_base and base_protos.shape[0] > 0:
        prior, _, stats = base_borrowed_diag_covariance(
            shot_proto,
            base_protos,
            base_diag_vars,
            tau=tau,
            eps=eps,
        )
        return {"name": "base_neighbors", "var": prior, "stats": stats}

    if cov_prior == "identity":
        return {"name": "identity", "var": identity_var, "stats": {}}
    return {"name": "global", "var": global_var, "stats": {}}


def _features_for_class(features_by_class, class_id, base_protos):
    if isinstance(features_by_class, dict):
        features = features_by_class.get(class_id)
    elif isinstance(features_by_class, (list, tuple)):
        features = features_by_class[class_id] if class_id < len(features_by_class) else None
    else:
        features = None
    if features is None:
        return base_protos.new_empty((0, base_protos.shape[-1]))
    return features.to(device=base_protos.device, dtype=base_protos.dtype)


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


def _average_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))
