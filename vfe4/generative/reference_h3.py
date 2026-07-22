"""Normalized source-free Gaussian factors for the frozen H3 fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from vfe4.types.h3 import H3Fixture


_DIMENSION = 4
_FACTOR_COUNT = 6


def _owned_tensor(value: object, shape: tuple[int, ...], name: str) -> Tensor:
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
    return value.detach().clone()


def _runtime_vector(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.shape != (_DIMENSION,):
        raise ValueError(f"{name} must have shape ({_DIMENSION},)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, init=False)
class H3ScalarGaussianFactor:
    """One normalized scalar Gaussian factor in four canonical coordinates."""

    _row: Tensor = field(repr=False, compare=False)
    _target: Tensor = field(repr=False, compare=False)
    _variance: Tensor = field(repr=False, compare=False)

    def __init__(self, row: Tensor, target: Tensor, variance: Tensor) -> None:
        checked_row = _owned_tensor(row, (_DIMENSION,), "row")
        checked_target = _owned_tensor(target, (), "target")
        checked_variance = _owned_tensor(variance, (), "variance")
        if not bool(checked_variance > 0.0):
            raise ValueError("variance must be positive")
        object.__setattr__(self, "_row", checked_row)
        object.__setattr__(self, "_target", checked_target)
        object.__setattr__(self, "_variance", checked_variance)

    @property
    def row(self) -> Tensor:
        return self._row.detach().clone()

    @property
    def target(self) -> Tensor:
        return self._target.detach().clone()

    @property
    def variance(self) -> Tensor:
        return self._variance.detach().clone()

    def log_prob(self, y: Tensor) -> Tensor:
        checked_y = _runtime_vector(y, "y")
        residual = self._row @ checked_y - self._target
        return -0.5 * (
            residual.square() / self._variance
            + torch.log(2.0 * torch.pi * self._variance)
        )


@dataclass(frozen=True, init=False)
class H3CanonicalJoint:
    """Owned diagnostic canonical coefficients for a normalized factor sum."""

    _precision: Tensor = field(repr=False, compare=False)
    _natural: Tensor = field(repr=False, compare=False)
    _log_constant: Tensor = field(repr=False, compare=False)

    def __init__(
        self, precision: Tensor, natural: Tensor, log_constant: Tensor
    ) -> None:
        checked_precision = _owned_tensor(
            precision, (_DIMENSION, _DIMENSION), "precision"
        )
        checked_natural = _owned_tensor(natural, (_DIMENSION,), "natural")
        checked_constant = _owned_tensor(log_constant, (), "log_constant")
        if not bool(torch.equal(checked_precision, checked_precision.T)):
            raise ValueError("precision must be exactly symmetric")
        _, info = torch.linalg.cholesky_ex(checked_precision, check_errors=False)
        if int(info.item()) != 0:
            raise ValueError("precision must be positive definite")
        object.__setattr__(self, "_precision", checked_precision)
        object.__setattr__(self, "_natural", checked_natural)
        object.__setattr__(self, "_log_constant", checked_constant)

    @property
    def precision(self) -> Tensor:
        return self._precision.detach().clone()

    @property
    def natural(self) -> Tensor:
        return self._natural.detach().clone()

    @property
    def log_constant(self) -> Tensor:
        return self._log_constant.detach().clone()


class H3GenerativeModel:
    """The six normalized factors from one frozen source-free H3 fixture."""

    def __init__(self, factors: tuple[H3ScalarGaussianFactor, ...]) -> None:
        if (
            type(factors) is not tuple
            or len(factors) != _FACTOR_COUNT
            or not all(isinstance(factor, H3ScalarGaussianFactor) for factor in factors)
        ):
            raise ValueError("factors must contain exactly six H3 scalar factors")
        self._factors = factors

    @classmethod
    def from_fixture(cls, fixture: H3Fixture) -> "H3GenerativeModel":
        if not isinstance(fixture, H3Fixture):
            raise ValueError("fixture must be an H3Fixture")
        if fixture.dimension != _DIMENSION or len(fixture.factors) != _FACTOR_COUNT:
            raise ValueError("fixture must contain six four-dimensional factors")
        factors = tuple(
            H3ScalarGaussianFactor(
                torch.tensor(record.row, dtype=torch.float64, device="cpu"),
                torch.tensor(record.target, dtype=torch.float64, device="cpu"),
                torch.tensor(record.variance, dtype=torch.float64, device="cpu"),
            )
            for record in fixture.factors
        )
        return cls(factors)

    @property
    def factors(self) -> tuple[H3ScalarGaussianFactor, ...]:
        return self._factors

    def log_joint(self, y: Tensor) -> Tensor:
        checked_y = _runtime_vector(y, "y")
        return torch.stack(
            tuple(factor.log_prob(checked_y) for factor in self._factors)
        ).sum()

    def canonical_joint(self) -> H3CanonicalJoint:
        precision_parts = tuple(
            torch.outer(factor.row, factor.row) / factor.variance
            for factor in self._factors
        )
        natural_parts = tuple(
            factor.target * factor.row / factor.variance
            for factor in self._factors
        )
        constant_parts = tuple(
            -0.5
            * (
                factor.target.square() / factor.variance
                + torch.log(2.0 * torch.pi * factor.variance)
            )
            for factor in self._factors
        )
        return H3CanonicalJoint(
            torch.stack(precision_parts).sum(dim=0),
            torch.stack(natural_parts).sum(dim=0),
            torch.stack(constant_parts).sum(),
        )


__all__ = [
    "H3CanonicalJoint",
    "H3GenerativeModel",
    "H3ScalarGaussianFactor",
]
