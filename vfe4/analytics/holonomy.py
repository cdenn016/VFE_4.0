from collections import deque

import torch

from ..state import LoopTable


def fixed_sector_rank(transports: torch.Tensor, edge_src: torch.Tensor, edge_dst: torch.Tensor,
                      n_agents: int, rtol: float = 1e-10) -> int:
    r"""dim ker L = sum dim Fix(Hol), c10 prop:cg-kernel-holonomy, with SPD weights.

    A spanning forest reduces constraints to K root coordinates per component,
    avoiding a dense (N*K)-dimensional Laplacian. Numerical rank uses declared rtol.
    """
    K = transports.shape[-1]
    adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(n_agents)]
    sources, targets = edge_src.cpu().tolist(), edge_dst.cpu().tolist()
    for e, (s, t) in enumerate(zip(sources, targets)):
        adjacency[s].append((t, e, False))
        adjacency[t].append((s, e, True))
    visited = [False] * n_agents
    total = 0
    eye = torch.eye(K, device=transports.device, dtype=transports.dtype)
    for root in range(n_agents):
        if visited[root]:
            continue
        visited[root] = True
        queue = deque([root])
        maps: dict[int, torch.Tensor] = {root: eye}
        while queue:
            parent = queue.popleft()
            for child, edge, inverse in adjacency[parent]:
                if visited[child]:
                    continue
                visited[child] = True
                matrix = torch.linalg.solve(transports[edge], eye) if inverse else transports[edge]
                maps[child] = matrix @ maps[parent]
                queue.append(child)
        indices = [e for e, s in enumerate(sources) if s in maps]
        if not indices:
            total += K
            continue
        rec_map = torch.stack([maps[targets[e]] for e in indices])
        src_map = torch.stack([maps[sources[e]] for e in indices])
        constraints = rec_map - transports[indices] @ src_map
        singular = torch.linalg.svdvals(constraints.reshape(-1, K))
        scale = max(float(torch.linalg.vector_norm(rec_map)), float(torch.linalg.vector_norm(transports[indices] @ src_map)), 1.0)
        total += K - int((singular > rtol * scale).sum())
    return total


def loop_holonomy_defect(omega: torch.Tensor, loops: LoopTable) -> torch.Tensor:
    r"""Q_p=1/2||I-product exp(omega_e)||_G^2, c08b, in the declared reference G=I."""
    lengths = loops.offset.diff()
    if not lengths.numel():
        return omega.new_empty(0)
    if not torch.all(lengths == lengths[0]):
        raise ValueError('Group loops by length before batched holonomy evaluation')
    width = int(lengths[0])
    generators = omega[loops.primitive_id].reshape(-1, width, omega.shape[-1], omega.shape[-1])
    signs = loops.orientation.reshape(-1, width, 1, 1)
    factors = torch.linalg.matrix_exp(generators * signs)
    holonomy = torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype).expand(factors.shape[0], -1, -1)
    for index in range(width):
        holonomy = holonomy @ factors[:, index]
    return 0.5 * (holonomy - torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype)).square().sum((-1, -2))
