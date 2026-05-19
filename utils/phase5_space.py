import torch
import torch.nn.functional as F


VALID_SPACE_TRANSFORMS = ("none", "common_direction_removal")
VALID_APPLY_TO = ("all", "prototypes", "queries")


def common_direction(tensors, eps=1e-6):
    valid = [F.normalize(t.reshape(-1, t.shape[-1]), dim=-1) for t in tensors if t is not None and t.numel() > 0]
    if not valid:
        return None
    stacked = torch.cat(valid, dim=0)
    direction = stacked.mean(dim=0)
    norm = direction.norm()
    if norm <= eps:
        return None
    return direction / norm


def remove_common_direction(values, direction, strength=1.0, normalize=True):
    if direction is None:
        return values
    direction = direction.to(values.device, values.dtype)
    projected = values - float(strength) * ((values @ direction).unsqueeze(-1) * direction)
    if normalize:
        projected = F.normalize(projected, dim=-1)
    return projected


def transform_covariance(cov, direction, strength=1.0, eps=1e-6):
    if direction is None:
        return cov
    direction = direction.to(cov.device, cov.dtype)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    projector = eye - float(strength) * torch.outer(direction, direction)
    return projector @ cov @ projector.t() + float(eps) * eye
