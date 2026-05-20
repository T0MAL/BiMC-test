import math

import torch


VALID_COV_MODES = ("original", "hybrid_diag")
VALID_NOVEL_AUX_COMBINE_MODES = ("average", "cov", "knn", "weighted", "reliability_gated")


def prepare_covariance(cov, mode="original", hybrid_diag_weight=0.5, eps=1e-4):
    if mode not in VALID_COV_MODES:
        raise ValueError(f"Unknown Phase 4 covariance mode: {mode}")
    if mode == "original":
        return cov

    diag_cov = torch.diag_embed(torch.diagonal(cov))
    weight = float(hybrid_diag_weight)
    out = (1 - weight) * cov + weight * diag_cov
    eye = torch.eye(out.shape[0], device=out.device, dtype=out.dtype)
    return out + float(eps) * eye


def combine_novel_auxiliary(
    prob_knn,
    prob_cov,
    replace_novel_nn=False,
    combine_mode="average",
    cov_score_weight=0.5,
):
    if combine_mode not in VALID_NOVEL_AUX_COMBINE_MODES:
        raise ValueError(f"Unknown Phase 4 novel auxiliary combine mode: {combine_mode}")
    if replace_novel_nn or combine_mode == "cov":
        return prob_cov
    if combine_mode == "knn":
        return prob_knn
    if combine_mode == "weighted":
        weight = float(cov_score_weight)
        return (1 - weight) * prob_knn + weight * prob_cov
    if combine_mode == "reliability_gated":
        weight = compute_auxiliary_reliability_weights(prob_knn, prob_cov)
        return (1 - weight) * prob_knn + weight * prob_cov
    return 0.5 * (prob_knn + prob_cov)


def compute_auxiliary_reliability_weights(
    prob_knn,
    prob_cov,
    min_weight=0.05,
    max_weight=0.60,
    eps=1e-12,
):
    rel_knn = _auxiliary_reliability(prob_knn)
    rel_cov = _auxiliary_reliability(prob_cov)
    weight = rel_cov / (rel_cov + rel_knn + float(eps))
    return weight.view(-1, 1).clamp(float(min_weight), float(max_weight))


def _auxiliary_reliability(probabilities):
    probabilities = _renormalize_probabilities(probabilities)
    confidence = 1 - _normalized_entropy(probabilities)
    margin = _prob_margin(probabilities)
    return confidence * margin


def _renormalize_probabilities(probabilities):
    total = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probabilities / total


def _normalized_entropy(probabilities):
    num_classes = probabilities.shape[-1]
    if num_classes <= 1:
        return torch.zeros(probabilities.shape[0], device=probabilities.device, dtype=probabilities.dtype)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1)
    return entropy / math.log(num_classes)


def _prob_margin(probabilities):
    k = min(2, probabilities.shape[-1])
    top_values = probabilities.topk(k=k, dim=-1).values
    if k == 1:
        return top_values[:, 0]
    return top_values[:, 0] - top_values[:, 1]
