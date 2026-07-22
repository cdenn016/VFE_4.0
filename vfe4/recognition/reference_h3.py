"""Autograd-preserving Gaussian recognition families for H3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vfe4.types.h3 import H3InitializationConfig, H3RecognitionFamily


_DIMENSION = 4
_STRICT_LOWER_ROWS = (1, 2, 2, 3, 3, 3)
_STRICT_LOWER_COLUMNS = (0, 0, 1, 0, 1, 2)


def _live_tensor(
    value: object, shape: tuple[int, ...], name: str
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _derived_finite(value: Tensor, name: str) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, eq=False)
class H3VariationalGaussian:
    """A live Gaussian parameterized by a precision Cholesky factor."""

    family: H3RecognitionFamily
    mean: Tensor
    precision_cholesky: Tensor

    def __post_init__(self) -> None:
        if self.family not in (
            "structured_full_spd",
            "fine_factorized_diagonal",
        ):
            raise ValueError("family must be an H3 recognition family")
        mean = _live_tensor(self.mean, (_DIMENSION,), "mean")
        cholesky = _live_tensor(
            self.precision_cholesky,
            (_DIMENSION, _DIMENSION),
            "precision_cholesky",
        )
        if not torch.equal(cholesky, torch.tril(cholesky)):
            raise ValueError("precision_cholesky must be lower triangular")
        diagonal = torch.diagonal(cholesky)
        if not bool(torch.all(diagonal > 0.0)):
            raise ValueError("precision_cholesky must have positive diagonal")
        if self.family == "fine_factorized_diagonal" and not torch.equal(
            cholesky, torch.diag(diagonal)
        ):
            raise ValueError(
                "factorized precision_cholesky must be exactly diagonal"
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "precision_cholesky", cholesky)

    def precision(self) -> Tensor:
        result = self.precision_cholesky @ self.precision_cholesky.T
        return _derived_finite(result, "variational precision")

    def linear_variance(self, row: Tensor) -> Tensor:
        checked_row = _live_tensor(row, (_DIMENSION,), "row")
        solved = torch.linalg.solve_triangular(
            self.precision_cholesky,
            checked_row.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        result = torch.dot(solved, solved)
        return _derived_finite(result, "linear variance")

    def entropy(self) -> Tensor:
        result = (
            0.5 * _DIMENSION * (1.0 + math.log(2.0 * math.pi))
            - torch.log(torch.diagonal(self.precision_cholesky)).sum()
        )
        return _derived_finite(result, "variational entropy")


def _initial_parameter_values(
    initialization: H3InitializationConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(initialization, H3InitializationConfig):
        raise ValueError("initialization must be an H3InitializationConfig")
    mean = torch.tensor(
        initialization.mean, dtype=torch.float64, device="cpu"
    )
    precision = torch.tensor(
        initialization.precision, dtype=torch.float64, device="cpu"
    )
    cholesky = torch.linalg.cholesky(precision)
    raw_diagonal = torch.log(torch.diagonal(cholesky))
    raw_lower = cholesky[_STRICT_LOWER_ROWS, _STRICT_LOWER_COLUMNS]
    return mean, raw_diagonal, raw_lower


class StructuredH3Parameters(nn.Module):
    """Fresh full-SPD precision-Cholesky parameters for one H3 arm."""

    family: H3RecognitionFamily = "structured_full_spd"

    def __init__(self, initialization: H3InitializationConfig) -> None:
        super().__init__()
        mean, raw_diagonal, raw_lower = _initial_parameter_values(
            initialization
        )
        self.mean = nn.Parameter(mean)
        self.raw_diagonal = nn.Parameter(raw_diagonal)
        self.raw_lower = nn.Parameter(raw_lower)

    def forward(self) -> H3VariationalGaussian:
        cholesky = torch.zeros(
            (_DIMENSION, _DIMENSION),
            dtype=torch.float64,
            device="cpu",
        )
        cholesky[_STRICT_LOWER_ROWS, _STRICT_LOWER_COLUMNS] = self.raw_lower
        cholesky = cholesky + torch.diag(torch.exp(self.raw_diagonal))
        return H3VariationalGaussian(
            family=self.family,
            mean=self.mean,
            precision_cholesky=cholesky,
        )


class FactorizedH3Parameters(nn.Module):
    """Fresh diagonal precision-Cholesky parameters for one H3 arm."""

    family: H3RecognitionFamily = "fine_factorized_diagonal"

    def __init__(self, initialization: H3InitializationConfig) -> None:
        super().__init__()
        mean, raw_diagonal, _ = _initial_parameter_values(initialization)
        self.mean = nn.Parameter(mean)
        self.raw_diagonal = nn.Parameter(raw_diagonal)

    def forward(self) -> H3VariationalGaussian:
        return H3VariationalGaussian(
            family=self.family,
            mean=self.mean,
            precision_cholesky=torch.diag(torch.exp(self.raw_diagonal)),
        )


def make_h3_parameters(
    family: H3RecognitionFamily,
    initialization: H3InitializationConfig,
) -> StructuredH3Parameters | FactorizedH3Parameters:
    """Construct a fresh parameter module for one frozen H3 family."""

    if family == "structured_full_spd":
        return StructuredH3Parameters(initialization)
    if family == "fine_factorized_diagonal":
        return FactorizedH3Parameters(initialization)
    raise ValueError("family must be an H3 recognition family")


__all__ = [
    "FactorizedH3Parameters",
    "H3RecognitionFamily",
    "H3VariationalGaussian",
    "StructuredH3Parameters",
    "make_h3_parameters",
]
