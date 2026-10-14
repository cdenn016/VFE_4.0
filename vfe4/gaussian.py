"""Normalized Gaussian algebra; constants are retained in every record cost."""

import math
from typing import Optional

import torch


def checked_cholesky(
    Sigma:  torch.Tensor,                    # (..., D, D)
    agents: Optional[torch.Tensor] = None,   # (B,) physical agent IDs of the leading batch axis
    where:  str                    = '',
) -> torch.Tensor:
    """Cholesky factor, or an error naming each failed block; nothing is repaired."""
    L, info = torch.linalg.cholesky_ex(Sigma, check_errors=False)
    bad = (info != 0) | ~torch.isfinite(L).flatten(-2).all(-1)
    if torch.any(bad):
        index = (bad.reshape(bad.shape[0], -1).any(-1) if bad.ndim else bad.reshape(1)).nonzero().flatten()
        named = agents[index].tolist() if agents is not None else index.tolist()
        label = 'agents' if agents is not None else 'batch indices'
        raise ValueError(f'Non-SPD or nonfinite Gaussian block{" in " + where if where else ""} at {label} {named}; '
                         'covariance is not repaired')
    return L


def linear_gaussian_moments(
    mu:    torch.Tensor,
    Sigma: torch.Tensor,
    A:     torch.Tensor,
    o:     torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""For a unique latent scope, E[r]=A mu-o, Cov(r)=A Sigma A^T."""
    return (A @ mu.unsqueeze(-1)).squeeze(-1) - o, A @ Sigma @ A.mT


def gaussian_kl(
    mu_q:    torch.Tensor,
    Sigma_q: torch.Tensor,
    mu_p:    torch.Tensor,
    Sigma_p: torch.Tensor,
) -> torch.Tensor:
    r"""D_KL(q||p)=1/2[tr(P_p Sigma_q)+delta^T P_p delta-K+log det(Sigma_p/Sigma_q)]."""
    L_p, L_q = checked_cholesky(Sigma_p), checked_cholesky(Sigma_q)
    delta = mu_q - mu_p
    trace = torch.cholesky_solve(Sigma_q, L_p).diagonal(dim1=-2, dim2=-1).sum(-1)
    quad = (delta * torch.cholesky_solve(delta.unsqueeze(-1), L_p).squeeze(-1)).sum(-1)
    log_ratio = 2 * (L_p.diagonal(dim1=-2, dim2=-1).log().sum(-1) - L_q.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    return 0.5 * (trace + quad - mu_q.shape[-1] + log_ratio)


def damp_moments(
    mu:         torch.Tensor,
    mu_star:    torch.Tensor,
    Sigma:      torch.Tensor,
    Sigma_star: torch.Tensor,
    w:          float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Damp sufficient statistics (mu,M2), retaining w(1-w) delta delta^T."""
    delta = mu - mu_star
    return ((1 - w) * mu + w * mu_star,
            (1 - w) * Sigma + w * Sigma_star + w * (1 - w) * delta.unsqueeze(-1) @ delta.unsqueeze(-2))


def gaussian_entropy(chol: torch.Tensor) -> torch.Tensor:
    r"""H(N)=D/2 log(2 pi e)+log det(L); node-block factors keep their own entropy."""
    if chol.ndim == 4:
        return (0.5 * chol.shape[-1] * (1 + math.log(2 * math.pi)) * chol.shape[-3]
                + chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=(-1, -2)))
    return 0.5 * chol.shape[-1] * (1 + math.log(2 * math.pi)) + chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def ragged_log_softmax(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets.diff()
    if values.numel() == 0:
        return values
    row = torch.repeat_interleave(torch.arange(lengths.numel(), device=values.device), lengths)
    maximum = torch.segment_reduce(values, 'max', lengths=lengths)
    shifted = values - maximum[row]
    total = torch.segment_reduce(shifted.exp(), 'sum', lengths=lengths)
    return shifted - total[row].log()
