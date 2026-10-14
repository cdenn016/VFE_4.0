import torch


def coherent_partition(residual_norm: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor,
                       resolution: float, n_agents: int) -> torch.Tensor:
    r"""Components of Coh_a={e:||r_e||<=a}; c13 analogue when evaluated on recognition means."""
    active = residual_norm <= resolution
    src, dst = edge_src[active], edge_dst[active]
    labels = torch.arange(n_agents, device=edge_src.device)
    for _ in range(n_agents):
        proposed = labels.clone()
        proposed.scatter_reduce_(0, torch.cat([src, dst]), torch.cat([labels[dst], labels[src]]), reduce='amin')
        if torch.equal(proposed, labels):
            break
        labels = proposed
    return labels
