"""Standard-normal Gauss-Hermite quadrature nodes and weights."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def probabilists_gauss_hermite(order: int, *, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return nodes and weights for expectations under a standard normal."""
    if type(order) is not int or order <= 0:
        raise ValueError("order must be a positive integer")
    if dtype is not torch.float64:
        raise ValueError("quadrature requires float64")
    hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(order)
    nodes = torch.tensor(math.sqrt(2.0) * hermite_nodes, dtype=dtype)
    weights = torch.tensor(hermite_weights / math.sqrt(math.pi), dtype=dtype)
    if not bool(torch.isfinite(nodes).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("quadrature nodes and weights must be finite")
    rounding_allowance = 64.0 * math.ulp(1.0)
    if abs(float(weights.sum().item()) - 1.0) > rounding_allowance:
        raise RuntimeError("quadrature weights failed normalization")
    return nodes, weights
