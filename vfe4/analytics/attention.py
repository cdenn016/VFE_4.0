import torch

from ..state import CandidateRows


def row_statistics(beta: torch.Tensor, rows: CandidateRows) -> dict[str, torch.Tensor]:
    r"""Categorical entropy and effective degree exp(H), descriptive c06 row statistics."""
    lengths = rows.row_offset.diff()
    entropy = torch.segment_reduce(-torch.special.xlogy(beta, beta), 'sum', lengths=lengths)
    self_weight = torch.segment_reduce(beta * (rows.receiver_incidence == rows.source_incidence), 'sum', lengths=lengths)
    return {'row_entropy': entropy, 'effective_degree': entropy.exp(), 'self_weight': self_weight}
