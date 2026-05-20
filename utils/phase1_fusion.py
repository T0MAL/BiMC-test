import math

import torch
import torch.nn.functional as F


VALID_FUSION_BETA_MODES = ("fixed", "class_margin", "query_reliability")
VALID_FUSION_GEOMETRIES = ("linear", "slerp")
VALID_RELIABILITY_MODES = ("margin", "entropy", "entropy_margin")
VALID_CNN_EXPERIMENT_MODES = (
    "none",
    "cnn_proto_adjust",
    "cnn_query_beta",
    "cnn_proto_adjust_plus_query_beta",
)
VALID_CNN_PROJECTIONS = ("random_orthogonal", "identity_pad")
CNN_PROTO_ADJUST_MODES = ("cnn_proto_adjust", "cnn_proto_adjust_plus_query_beta")
CNN_QUERY_BETA_MODES = ("cnn_query_beta", "cnn_proto_adjust_plus_query_beta")


def validate_phase1_options(cfg):
    opts = cfg.TRAINER.BiMC
    if opts.FUSION_BETA_MODE not in VALID_FUSION_BETA_MODES:
        raise ValueError(
            f"Invalid fusion beta mode {opts.FUSION_BETA_MODE}. "
            f"Expected one of {VALID_FUSION_BETA_MODES}."
        )
    if opts.FUSION_GEOMETRY not in VALID_FUSION_GEOMETRIES:
        raise ValueError(
            f"Invalid fusion geometry {opts.FUSION_GEOMETRY}. "
            f"Expected one of {VALID_FUSION_GEOMETRIES}."
        )
    if opts.RELIABILITY_MODE not in VALID_RELIABILITY_MODES:
        raise ValueError(
            f"Invalid reliability mode {opts.RELIABILITY_MODE}. "
            f"Expected one of {VALID_RELIABILITY_MODES}."
        )
    if opts.BETA_TEMPERATURE <= 0:
        raise ValueError("BETA_TEMPERATURE must be positive.")
    if not 0 <= opts.BETA_CLIP_MIN <= opts.BETA_CLIP_MAX <= 1:
        raise ValueError("Expected 0 <= BETA_CLIP_MIN <= BETA_CLIP_MAX <= 1.")
    if not hasattr(opts, "USE_CNN_BRANCH"):
        return
    if opts.CNN_EXPERIMENT_MODE not in VALID_CNN_EXPERIMENT_MODES:
        raise ValueError(
            f"Invalid CNN experiment mode {opts.CNN_EXPERIMENT_MODE}. "
            f"Expected one of {VALID_CNN_EXPERIMENT_MODES}."
        )
    if opts.CNN_PROJECTION not in VALID_CNN_PROJECTIONS:
        raise ValueError(
            f"Invalid CNN projection {opts.CNN_PROJECTION}. "
            f"Expected one of {VALID_CNN_PROJECTIONS}."
        )
    if not 0 <= opts.CNN_LAMBDA <= 1:
        raise ValueError("Expected 0 <= CNN_LAMBDA <= 1.")
    if opts.CNN_TOPK < 1:
        raise ValueError("CNN_TOPK must be at least 1.")


def get_active_cnn_mode(opts):
    if not getattr(opts, "USE_CNN_BRANCH", False):
        return "none"
    return getattr(opts, "CNN_EXPERIMENT_MODE", "none")


def cnn_uses_proto_adjust(mode):
    return mode in CNN_PROTO_ADJUST_MODES


def cnn_uses_query_beta(mode):
    return mode in CNN_QUERY_BETA_MODES


def clamp_beta(beta, beta_clip_min, beta_clip_max):
    return torch.clamp(beta, min=beta_clip_min, max=beta_clip_max)


def _to_ref_tensor(value, ref):
    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def _broadcast_beta(beta, target):
    beta = _to_ref_tensor(beta, target)

    if beta.ndim == 0:
        return beta

    if beta.shape == target.shape[:-1]:
        return beta.unsqueeze(-1)

    if beta.ndim == 1:
        if target.ndim == 2 and beta.numel() == target.shape[0]:
            return beta.view(-1, 1)
        if target.ndim == 3 and beta.numel() == target.shape[0]:
            return beta.view(-1, 1, 1)
        if target.ndim == 3 and beta.numel() == target.shape[1]:
            return beta.view(1, -1, 1)

    while beta.ndim < target.ndim:
        beta = beta.unsqueeze(-1)
    return beta


def slerp(a, b, t, eps=1e-6):
    """Spherical interpolation from normalized vector a to normalized vector b."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    a, b = torch.broadcast_tensors(a, b)
    t = _broadcast_beta(t, a)

    raw_dot = (a * b).sum(dim=-1, keepdim=True)
    dot = raw_dot.clamp(min=-1 + eps, max=1 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    weight_a = torch.sin((1 - t) * theta) / sin_theta
    weight_b = torch.sin(t * theta) / sin_theta
    spherical = weight_a * a + weight_b * b

    linear = (1 - t) * a + t * b
    small_angle = (1 - raw_dot.abs()) < eps
    out = torch.where(small_angle, linear, spherical)
    return F.normalize(out, dim=-1)


def fuse_prototypes(visual_proto, text_proto, beta, geometry="linear"):
    if geometry == "linear":
        beta = _broadcast_beta(beta, text_proto)
        out = beta * text_proto + (1 - beta) * visual_proto
        return F.normalize(out, dim=-1)
    if geometry == "slerp":
        return slerp(visual_proto, text_proto, beta)
    raise ValueError(f"Unknown fusion geometry: {geometry}")


def _top_other(scores, class_id):
    if scores.shape[1] == 1:
        return torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype)
    other_scores = scores.clone()
    other_scores[:, class_id] = float("-inf")
    return other_scores.max(dim=1).values


def compute_class_margin_beta(
    support_features,
    support_labels,
    text_proto,
    visual_proto,
    beta_temperature,
    beta_clip_min,
    beta_clip_max,
    default_beta=0.5,
    use_leave_one_out=True,
):
    """Compute one beta value per class from support-set text/visual margins."""
    support_features = F.normalize(support_features, dim=-1)
    text_proto = F.normalize(text_proto, dim=-1).to(
        device=support_features.device,
        dtype=support_features.dtype,
    )
    visual_proto = F.normalize(visual_proto, dim=-1).to(
        device=support_features.device,
        dtype=support_features.dtype,
    )
    support_labels = support_labels.to(device=support_features.device).long()

    num_classes = text_proto.shape[0]
    default = torch.full(
        (num_classes,),
        float(default_beta),
        device=support_features.device,
        dtype=support_features.dtype,
    )
    beta_c = clamp_beta(default, beta_clip_min, beta_clip_max)

    if support_features.numel() == 0:
        return beta_c

    scores_text = support_features @ text_proto.t()
    scores_visual = support_features @ visual_proto.t()

    for class_id in range(num_classes):
        class_mask = support_labels == class_id
        if not torch.any(class_mask):
            continue

        class_text_scores = scores_text[class_mask]
        text_correct = class_text_scores[:, class_id]
        text_other = _top_other(class_text_scores, class_id)
        margin_text = text_correct - text_other

        class_visual_scores = scores_visual[class_mask]
        if use_leave_one_out:
            class_features = support_features[class_mask]
            if class_features.shape[0] > 1:
                feature_sum = class_features.sum(dim=0, keepdim=True)
                loo_proto = F.normalize(
                    (feature_sum - class_features) / (class_features.shape[0] - 1),
                    dim=-1,
                )
                visual_correct = (class_features * loo_proto).sum(dim=-1)
            else:
                visual_correct = class_visual_scores[:, class_id]
        else:
            visual_correct = class_visual_scores[:, class_id]

        visual_other = _top_other(class_visual_scores, class_id)
        margin_visual = visual_correct - visual_other

        margin_delta = (margin_text.mean() - margin_visual.mean()) / beta_temperature
        beta_c[class_id] = torch.sigmoid(margin_delta)

    return clamp_beta(beta_c, beta_clip_min, beta_clip_max)


def _normalized_entropy(probabilities, eps=1e-12):
    num_classes = probabilities.shape[-1]
    if num_classes <= 1:
        return torch.zeros(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)

    entropy = -(probabilities * torch.log(probabilities + eps)).sum(dim=-1)
    return entropy / math.log(num_classes)


def _prob_margin(probabilities):
    k = min(2, probabilities.shape[-1])
    top_values = probabilities.topk(k=k, dim=-1).values
    if k == 1:
        return top_values[:, 0]
    return top_values[:, 0] - top_values[:, 1]


def _reliability(probabilities, mode):
    entropy_confidence = 1 - _normalized_entropy(probabilities)
    margin = _prob_margin(probabilities)

    if mode == "margin":
        return margin
    if mode == "entropy":
        return entropy_confidence
    if mode == "entropy_margin":
        return entropy_confidence * margin
    raise ValueError(f"Unknown reliability mode: {mode}")


def compute_reliability_from_scores(scores, reliability_mode, topk=None):
    if topk is not None and 0 < topk < scores.shape[-1]:
        scores = scores.topk(k=topk, dim=-1).values
    probabilities = F.softmax(scores, dim=-1)
    return _reliability(probabilities, reliability_mode)


def compute_query_reliability_beta(
    query_features,
    text_proto,
    visual_proto,
    reliability_mode,
    beta_clip_min,
    beta_clip_max,
    eps=1e-12,
):
    """Compute one beta value per query from unlabeled text/visual reliability."""
    query_features = F.normalize(query_features, dim=-1)
    text_proto = F.normalize(text_proto, dim=-1).to(
        device=query_features.device,
        dtype=query_features.dtype,
    )
    visual_proto = F.normalize(visual_proto, dim=-1).to(
        device=query_features.device,
        dtype=query_features.dtype,
    )

    scores_text = query_features @ text_proto.t()
    scores_visual = query_features @ visual_proto.t()

    reliability_text = compute_reliability_from_scores(scores_text, reliability_mode)
    reliability_visual = compute_reliability_from_scores(scores_visual, reliability_mode)

    beta_x = reliability_text / (reliability_text + reliability_visual + eps)
    return clamp_beta(beta_x, beta_clip_min, beta_clip_max)


def _orthogonal_projection_matrix(input_dim, output_dim, generator):
    raw = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    if input_dim >= output_dim:
        if hasattr(torch, "linalg") and hasattr(torch.linalg, "qr"):
            q, _ = torch.linalg.qr(raw, mode="reduced")
        else:
            q, _ = torch.qr(raw)
        return q[:, :output_dim]

    if hasattr(torch, "linalg") and hasattr(torch.linalg, "qr"):
        q, _ = torch.linalg.qr(raw.t(), mode="reduced")
    else:
        q, _ = torch.qr(raw.t())
    return q[:, :input_dim].t()


def deterministic_projection_matrix(
    input_dim,
    output_dim,
    projection="random_orthogonal",
    seed=0,
    device=None,
    dtype=None,
):
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Projection dimensions must be positive.")
    if projection not in VALID_CNN_PROJECTIONS:
        raise ValueError(
            f"Invalid CNN projection {projection}. Expected one of {VALID_CNN_PROJECTIONS}."
        )

    if projection == "identity_pad":
        matrix = torch.zeros(input_dim, output_dim, dtype=torch.float32)
        diagonal = min(input_dim, output_dim)
        matrix[:diagonal, :diagonal] = torch.eye(diagonal, dtype=torch.float32)
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + input_dim * 1_000_003 + output_dim * 9_176)
        matrix = _orthogonal_projection_matrix(input_dim, output_dim, generator)

    if device is not None or dtype is not None:
        matrix = matrix.to(device=device, dtype=dtype)
    return matrix


def project_cnn_features_to_clip(
    cnn_features,
    clip_dim,
    projection="random_orthogonal",
    seed=0,
    projection_matrix=None,
    return_matrix=False,
):
    cnn_features = F.normalize(cnn_features, dim=-1)
    input_dim = cnn_features.shape[-1]
    if projection_matrix is None:
        projection_matrix = deterministic_projection_matrix(
            input_dim=input_dim,
            output_dim=clip_dim,
            projection=projection,
            seed=seed,
            device=cnn_features.device,
            dtype=cnn_features.dtype,
        )
    else:
        projection_matrix = projection_matrix.to(
            device=cnn_features.device,
            dtype=cnn_features.dtype,
        )
    projected = cnn_features @ projection_matrix
    projected = F.normalize(projected, dim=-1)
    if return_matrix:
        return projected, projection_matrix
    return projected


def compute_cnn_query_reliability_beta(
    query_clip_features,
    text_proto,
    query_cnn_features,
    cnn_proto,
    reliability_mode,
    beta_clip_min,
    beta_clip_max,
    cnn_topk=5,
    eps=1e-12,
    return_info=False,
):
    """Compute query-wise beta from CLIP text reliability and CNN visual reliability."""
    query_clip_features = F.normalize(query_clip_features, dim=-1)
    query_cnn_features = F.normalize(query_cnn_features, dim=-1)
    text_proto = F.normalize(text_proto, dim=-1).to(
        device=query_clip_features.device,
        dtype=query_clip_features.dtype,
    )
    cnn_proto = F.normalize(cnn_proto, dim=-1).to(
        device=query_cnn_features.device,
        dtype=query_cnn_features.dtype,
    )

    scores_text = query_clip_features @ text_proto.t()
    scores_cnn = query_cnn_features @ cnn_proto.t()

    reliability_text = compute_reliability_from_scores(scores_text, reliability_mode)
    reliability_visual = compute_reliability_from_scores(
        scores_cnn,
        reliability_mode,
        topk=min(int(cnn_topk), scores_cnn.shape[-1]),
    )

    beta_x = reliability_text / (reliability_text + reliability_visual + eps)
    beta_x = clamp_beta(beta_x, beta_clip_min, beta_clip_max)

    if return_info:
        return beta_x, {
            "reliability_text": reliability_text.detach(),
            "reliability_visual": reliability_visual.detach(),
        }
    return beta_x


def beta_statistics(values):
    values = _flatten_beta_values(values)
    if values.numel() == 0:
        return {
            "beta_count": 0,
            "beta_mean": None,
            "beta_std": None,
            "beta_min": None,
            "beta_max": None,
            "beta_q05": None,
            "beta_q25": None,
            "beta_q50": None,
            "beta_q75": None,
            "beta_q95": None,
        }

    values = values.float().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]))
    return {
        "beta_count": int(values.numel()),
        "beta_mean": float(values.mean().item()),
        "beta_std": float(values.std(unbiased=False).item()),
        "beta_min": float(values.min().item()),
        "beta_max": float(values.max().item()),
        "beta_q05": float(quantiles[0].item()),
        "beta_q25": float(quantiles[1].item()),
        "beta_q50": float(quantiles[2].item()),
        "beta_q75": float(quantiles[3].item()),
        "beta_q95": float(quantiles[4].item()),
    }


def _flatten_beta_values(values):
    if values is None:
        return torch.empty(0)
    if isinstance(values, torch.Tensor):
        return values.detach().reshape(-1).cpu()
    tensors = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().reshape(-1).cpu())
        else:
            tensors.append(torch.as_tensor(value).reshape(-1).cpu())
    if not tensors:
        return torch.empty(0)
    return torch.cat(tensors, dim=0)
