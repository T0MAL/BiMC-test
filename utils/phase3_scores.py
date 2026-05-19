import math

import torch
import torch.nn.functional as F


VALID_DYNAMIC_ALPHA_MODES = ("entropy_margin", "entropy_margin_energy")
VALID_HUBNESS_SOURCES = ("support", "description", "mixed")


def apply_temperature(logits, enabled=False, temperature=1.0):
    if not enabled:
        return logits
    temperature = max(float(temperature), 1e-6)
    return logits / temperature


def compute_dynamic_alpha(
    proto_logits,
    aux_logits,
    mode="entropy_margin",
    alpha_clip_min=0.05,
    alpha_clip_max=0.95,
    energy_enabled=False,
):
    if mode not in VALID_DYNAMIC_ALPHA_MODES:
        raise ValueError(f"Unknown Phase 3 dynamic alpha mode: {mode}")

    proto_probs = F.softmax(proto_logits, dim=-1)
    aux_probs = F.softmax(aux_logits, dim=-1)
    proto_rel = _reliability(proto_probs)
    aux_rel = _reliability(aux_probs)

    if energy_enabled or mode.endswith("_energy"):
        proto_rel = proto_rel * _energy_confidence(proto_logits)
        aux_rel = aux_rel * _energy_confidence(aux_logits)

    alpha = proto_rel / (proto_rel + aux_rel + 1e-12)
    return alpha.clamp(float(alpha_clip_min), float(alpha_clip_max))


def compute_hubness_bias(reference_features, prototypes, strength=0.1):
    if reference_features is None or reference_features.numel() == 0 or prototypes.shape[0] <= 1:
        return torch.zeros(prototypes.shape[0], device=prototypes.device, dtype=prototypes.dtype)

    reference_features = F.normalize(reference_features.to(prototypes.device, prototypes.dtype), dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    nearest = torch.argmax(reference_features @ prototypes.t(), dim=-1)
    counts = torch.bincount(nearest, minlength=prototypes.shape[0]).to(prototypes.device, prototypes.dtype)
    bias = torch.log1p(counts)
    bias = bias - bias.mean()
    return float(strength) * bias


def select_hubness_reference(source, support_features=None, description_features=None):
    if source not in VALID_HUBNESS_SOURCES:
        raise ValueError(f"Unknown Phase 3 hubness source: {source}")
    tensors = []
    if source in ("support", "mixed") and support_features is not None:
        tensors.append(support_features)
    if source in ("description", "mixed") and description_features is not None:
        tensors.append(description_features)
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _reliability(probabilities):
    entropy_conf = 1 - _normalized_entropy(probabilities)
    margin = _prob_margin(probabilities)
    return entropy_conf * margin


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


def _energy_confidence(logits):
    energy = -torch.logsumexp(logits, dim=-1)
    centered = -(energy - energy.mean())
    return torch.sigmoid(centered)
