import torch
import torch.nn.functional as F


VALID_SEPARATION_MODES = ("none", "graph_highpass")


def separate_prototypes(prototypes, mode="none", strength=0.15, graph_k=5):
    if mode not in VALID_SEPARATION_MODES:
        raise ValueError(f"Unknown Phase 7 separation mode: {mode}")
    prototypes = F.normalize(prototypes, dim=-1)
    if mode == "none" or prototypes.shape[0] <= 1:
        return prototypes

    k = min(max(int(graph_k), 1), prototypes.shape[0] - 1)
    sim = prototypes @ prototypes.t()
    sim.fill_diagonal_(float("-inf"))
    nn_idx = sim.topk(k=k, dim=-1).indices
    neighbors = prototypes[nn_idx].mean(dim=1)
    highpass = prototypes + float(strength) * (prototypes - neighbors)
    return F.normalize(highpass, dim=-1)
