"""Fail-closed categorical primitives for deterministic H1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vfe4.types._validation import (
    require_probability_vector as _require_probability_vector,
)


class AllInvalidSourceRowError(ValueError):
    """Raised when a source row has no declared causal support."""


@dataclass(frozen=True)
class MaskedLogProbabilities:
    """Live normalized log-probabilities and their exact Boolean support."""

    log_probs: Tensor
    support_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.log_probs, Tensor) or not isinstance(
            self.support_mask, Tensor
        ):
            raise ValueError("masked categorical values must be tensors")
        if self.log_probs.ndim != 1 or self.log_probs.numel() == 0:
            raise ValueError("masked log-probabilities must be a nonempty vector")
        if self.support_mask.dtype is not torch.bool:
            raise ValueError("support_mask must be boolean")
        if self.support_mask.shape != self.log_probs.shape:
            raise ValueError("support_mask must match the log-probability shape")


def require_probability_vector(value: Tensor, *, name: str) -> Tensor:
    """Validate and return an owned float64 probability vector."""
    return _require_probability_vector(value, name=name)


def categorical_kl(q: Tensor, p: Tensor, *, name: str) -> Tensor:
    """Compute KL(q || p), including the valid zero-mass convention for q."""
    checked_q = require_probability_vector(q, name=f"{name}.q")
    checked_p = require_probability_vector(p, name=f"{name}.p")
    if checked_q.shape != checked_p.shape or checked_q.device != checked_p.device:
        raise ValueError(f"{name} q and p must have the same shape and device")
    support = checked_q > 0
    if bool(torch.any(checked_p[support] <= 0)):
        raise ValueError(f"{name} has q mass outside p support")
    return torch.sum(
        checked_q[support]
        * (torch.log(checked_q[support]) - torch.log(checked_p[support]))
    )


def selected_log_softmax(logits: Tensor, index: int) -> Tensor:
    """Return the selected log-softmax entry after precision and shape checks."""
    _require_vector(logits, "logits")
    if type(index) is not int or index < 0 or index >= logits.numel():
        raise ValueError("index is out of range")
    return torch.log_softmax(logits, dim=0)[index]


def masked_log_softmax_from_parents(
    logits: Tensor,
    declared_parents: tuple[int, ...],
    receiver_t: int,
) -> MaskedLogProbabilities:
    """Normalize one causal source row after applying its declared support mask."""

    _require_vector(logits, "logits")
    if type(receiver_t) is not int or receiver_t <= 0:
        raise ValueError("receiver_t must be a positive integer")
    if logits.numel() != receiver_t:
        raise ValueError("logits must contain one entry for every earlier node")
    if type(declared_parents) is not tuple:
        raise ValueError("declared_parents must be a tuple")
    if not declared_parents:
        raise AllInvalidSourceRowError("source row has no valid declared parent")
    if (
        any(type(parent) is not int for parent in declared_parents)
        or tuple(sorted(set(declared_parents))) != declared_parents
        or any(parent < 0 or parent >= receiver_t for parent in declared_parents)
    ):
        raise ValueError(
            "declared parents must be unique increasing nodes strictly below receiver_t"
        )

    support_mask = torch.zeros_like(logits, dtype=torch.bool)
    support_mask[list(declared_parents)] = True
    if not bool(torch.any(support_mask)):
        raise AllInvalidSourceRowError("source row has no valid declared parent")
    masked_logits = logits.masked_fill(~support_mask, -torch.inf)
    return MaskedLogProbabilities(
        torch.log_softmax(masked_logits, dim=0), support_mask
    )


def _require_vector(value: object, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
