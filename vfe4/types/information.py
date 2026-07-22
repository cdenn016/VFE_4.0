"""Immutable information-form records and precision-factor interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from torch import Tensor


@dataclass(frozen=True)
class MatrixBlock:
    """A requested rectangular matrix block with ordered row and column indices."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_indices(self.rows, "rows")
        _require_indices(self.columns, "columns")


@dataclass(frozen=True)
class PrecisionDiagnostics:
    """Condition diagnostics recorded when a precision factor is built."""

    dimension: int
    min_cholesky_pivot: float
    lambda_min: float
    lambda_max: float
    kappa_2: float

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        for name in (
            "min_cholesky_pivot",
            "lambda_min",
            "lambda_max",
            "kappa_2",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite float")
        if self.lambda_max < self.lambda_min:
            raise ValueError("lambda_max must be at least lambda_min")


@runtime_checkable
class PrecisionFactor(Protocol):
    """Operations available from a factored positive-definite precision."""

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def diagnostics(self) -> PrecisionDiagnostics:
        raise NotImplementedError

    def solve(self, rhs: Tensor) -> Tensor:
        raise NotImplementedError

    def logdet(self) -> Tensor:
        raise NotImplementedError

    def selected_inverse(
        self, blocks: Sequence[MatrixBlock]
    ) -> Mapping[MatrixBlock, Tensor]:
        raise NotImplementedError

    def sample(self, noise: Tensor) -> Tensor:
        raise NotImplementedError

    def quadratic(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def trace_inverse_product(self, left: "PrecisionFactor") -> Tensor:
        raise NotImplementedError


def _require_indices(value: object, name: str) -> None:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    if any(type(index) is not int or index < 0 for index in value):
        raise ValueError(f"{name} must contain nonnegative integer indices")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate indices")
