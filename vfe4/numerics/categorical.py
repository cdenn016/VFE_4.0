"""Fail-closed categorical primitives for deterministic H1 evaluation."""

from __future__ import annotations

import torch
from torch import Tensor

from vfe4.types._validation import (
    require_probability_vector as _require_probability_vector,
)


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


def _require_vector(value: object, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
