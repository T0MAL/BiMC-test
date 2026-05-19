import torch


VALID_COV_MODES = ("original", "hybrid_diag")
VALID_NOVEL_AUX_COMBINE_MODES = ("average", "cov", "knn")


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


def combine_novel_auxiliary(prob_knn, prob_cov, replace_novel_nn=False, combine_mode="average"):
    if combine_mode not in VALID_NOVEL_AUX_COMBINE_MODES:
        raise ValueError(f"Unknown Phase 4 novel auxiliary combine mode: {combine_mode}")
    if replace_novel_nn or combine_mode == "cov":
        return prob_cov
    if combine_mode == "knn":
        return prob_knn
    return 0.5 * (prob_knn + prob_cov)
