from typing import Optional

import torch

from ..state import LoopTable


def fixed_sector_rank(
    transports: torch.Tensor,            # (E,K,K) receiver <- sender maps
    edge_src:   torch.Tensor,            # (E,) sender vertex
    edge_dst:   torch.Tensor,            # (E,) receiver vertex
    n_agents:   int,
    rtol:       Optional[float] = None,
) -> int:
    r"""dim ker(delta^T delta), the connection-Laplacian kernel of c10 prop:cg-kernel-holonomy.

    Edge e contributes the residual x_dst - T_e x_src. The (nK x nK) Gram matrix
    delta^T delta is assembled from O(1)-scaled blocks and diagonalized densely, so the
    caller must admit nK. Eigenvalues at most rtol * lambda_max are kernel, with default
    rtol = nK * eps. Rooted path products are avoided: on long spanning trees their
    norms grow exponentially and swamp any global threshold.
    """
    K = transports.shape[-1]
    D = n_agents * K
    gram = transports.new_zeros(D, D)
    eye = torch.eye(K, device=transports.device, dtype=transports.dtype).expand_as(transports)
    coordinate = torch.arange(K, device=transports.device)

    def add(rows: torch.Tensor, cols: torch.Tensor, blocks: torch.Tensor) -> None:
        index = ((rows[:, None, None] * K + coordinate[None, :, None]) * D
                 + cols[:, None, None] * K + coordinate[None, None, :])
        gram.view(-1).index_add_(0, index.flatten(), blocks.flatten())

    add(edge_dst, edge_dst, eye)
    add(edge_src, edge_src, transports.mT @ transports)
    add(edge_dst, edge_src, -transports)
    add(edge_src, edge_dst, -transports.mT)
    eigenvalues = torch.linalg.eigvalsh(gram)
    scale = float(eigenvalues[-1]) if D else 0.0
    threshold = (D * torch.finfo(transports.dtype).eps if rtol is None else rtol) * scale
    return int((eigenvalues <= threshold).sum())


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
