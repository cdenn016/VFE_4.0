"""Fail-closed Gaussian primitives without matrix inversion."""

from __future__ import annotations

import math

import torch
from torch import Tensor


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


def gaussian_log_prob(value: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
    """Evaluate a multivariate Gaussian log density via a Cholesky solve."""
    _require_finite_vector(value, "value")
    _require_finite_vector(mean, "mean")
    checked_covariance = require_spd(covariance, name="covariance")
    if value.shape != mean.shape:
        raise ValueError("value and mean must have the same shape")
    if covariance.shape != (value.numel(), value.numel()):
        raise ValueError("covariance shape must match value")
    if value.device != mean.device or value.device != covariance.device:
        raise ValueError("value, mean, and covariance must share a device")

    chol = torch.linalg.cholesky(checked_covariance)
    delta = (value - mean).unsqueeze(-1)
    solved = torch.cholesky_solve(delta, chol)
    quadratic = torch.sum(delta * solved)
    log_determinant = 2.0 * torch.log(torch.diagonal(chol)).sum()
    normalizer = value.numel() * math.log(2.0 * math.pi)
    return -0.5 * (quadratic + log_determinant + normalizer)


def _require_finite_vector(value: object, name: str) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
