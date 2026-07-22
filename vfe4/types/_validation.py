"""Leaf tensor validation shared by foundational types and numerical APIs."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def require_probability_vector(value: Tensor, *, name: str) -> Tensor:
    """Validate and return an owned float64 probability vector."""
    _require_vector(value, name)
    if bool(torch.any(value < 0)):
        raise ValueError(f"{name} must be nonnegative")
    total = value.sum()
    rounding_allowance = 64.0 * math.ulp(1.0) * max(1, value.numel())
    if abs(float(total.item()) - 1.0) > rounding_allowance:
        raise ValueError(f"{name} must sum to one")
    return value.detach().clone()


def require_spd(matrix: Tensor, *, name: str) -> Tensor:
    """Validate a finite symmetric positive-definite float64 matrix."""
    if not isinstance(matrix, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if matrix.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must be finite")
    if not bool(torch.equal(matrix, matrix.transpose(-1, -2))):
        raise ValueError(f"{name} must be symmetric")
    _, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    if int(info.item()) != 0:
        raise ValueError(f"{name} must be symmetric positive definite")
    return matrix.detach().clone()


def _require_vector(value: object, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
