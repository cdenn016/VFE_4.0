"""Information-form Gaussian operations backed by a precision factor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from vfe4.numerics.precision import DenseCholeskyPrecision
from vfe4.types.information import MatrixBlock, PrecisionFactor


@dataclass(frozen=True, init=False)
class InformationGaussian:
    """An immutable Gaussian in canonical information coordinates."""

    _h: Tensor = field(repr=False, compare=False)
    _J: Tensor = field(repr=False, compare=False)
    factor: PrecisionFactor = field(repr=False, compare=False)

    @classmethod
    def from_information(
        cls,
        h: Tensor,
        J: Tensor,
        factor_factory: Callable[[Tensor], PrecisionFactor] = DenseCholeskyPrecision,
    ) -> "InformationGaussian":
        checked_h = _require_vector(h, "h")
        checked_j = _require_information_matrix(J)
        if checked_j.shape != (checked_h.numel(), checked_h.numel()):
            raise ValueError("h and J must have the same dimension")
        if checked_h.device != checked_j.device:
            raise ValueError("h and J must share a device")
        if not callable(factor_factory):
            raise ValueError("factor_factory must be callable")
        factor = factor_factory(checked_j.detach().clone())
        if not isinstance(factor, PrecisionFactor):
            raise ValueError("factor_factory must produce a PrecisionFactor")
        if factor.dimension != checked_h.numel():
            raise ValueError("factor dimension must match h and J")
        result = cls.__new__(cls)
        object.__setattr__(result, "_h", checked_h)
        object.__setattr__(result, "_J", checked_j)
        object.__setattr__(result, "factor", factor)
        result.mean()
        result.log_normalizer()
        return result

    @property
    def dimension(self) -> int:
        return self._h.numel()

    @property
    def h(self) -> Tensor:
        return self._h.detach().clone()

    @property
    def J(self) -> Tensor:
        return self._J.detach().clone()

    def mean(self) -> Tensor:
        result = self.factor.solve(self._h)
        _require_derived_finite(result, "information Gaussian mean")
        return result.detach().clone()

    def log_normalizer(self) -> Tensor:
        mean = self.mean()
        result = 0.5 * (
            torch.dot(self._h, mean)
            - self.factor.logdet()
            + self.dimension * math.log(2.0 * math.pi)
        )
        _require_derived_finite(result, "information Gaussian log normalizer")
        return result.detach().clone()

    def entropy(self) -> Tensor:
        result = 0.5 * (
            self.dimension * (1.0 + math.log(2.0 * math.pi))
            - self.factor.logdet()
        )
        _require_derived_finite(result, "information Gaussian entropy")
        return result.detach().clone()

    def log_prob(self, value: Tensor) -> Tensor:
        checked = _require_vector(value, "value")
        if checked.shape != self._h.shape:
            raise ValueError("value dimension must match the information Gaussian")
        if checked.device != self._h.device:
            raise ValueError("value must share the information Gaussian device")
        try:
            quadratic = self.factor.quadratic(checked)
        except ValueError as error:
            raise ValueError("information Gaussian log_prob must be finite") from error
        result = torch.dot(self._h, checked) - 0.5 * quadratic - self.log_normalizer()
        _require_derived_finite(result, "information Gaussian log_prob")
        return result.detach().clone()

    def oriented_kl(self, other: "InformationGaussian") -> Tensor:
        if not isinstance(other, InformationGaussian):
            raise ValueError("other must be an InformationGaussian")
        if other.dimension != self.dimension:
            raise ValueError("information Gaussians must have the same dimension")
        if other._h.device != self._h.device:
            raise ValueError("information Gaussians must share a device")
        delta = other.mean() - self.mean()
        _require_derived_finite(delta, "information Gaussian KL displacement")
        result = 0.5 * (
            self.factor.trace_inverse_product(other.factor)
            + other.factor.quadratic(delta)
            - self.dimension
            + self.factor.logdet()
            - other.factor.logdet()
        )
        _require_derived_finite(result, "information Gaussian oriented KL")
        return result.detach().clone()

    def selected_moment_blocks(
        self, blocks: Sequence[MatrixBlock]
    ) -> Mapping[MatrixBlock, Tensor]:
        inverse_blocks = self.factor.selected_inverse(blocks)
        mean = self.mean()
        selected: dict[MatrixBlock, Tensor] = {}
        for block, inverse_block in inverse_blocks.items():
            row_indices = torch.tensor(block.rows, device=mean.device)
            column_indices = torch.tensor(block.columns, device=mean.device)
            outer = mean.index_select(0, row_indices).unsqueeze(1) * mean.index_select(
                0, column_indices
            ).unsqueeze(0)
            value = inverse_block + outer
            _require_derived_finite(value, "information Gaussian moment block")
            selected[block] = value.detach().clone()
        return MappingProxyType(selected)


def _require_vector(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value.detach().clone()


def _require_information_matrix(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError("J must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError("J must use float64")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ValueError("J must be a nonempty square matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("J must be finite")
    if not bool(torch.equal(value, value.transpose(-1, -2))):
        raise ValueError("J must be symmetric")
    checked = value.detach().clone()
    chol, info = torch.linalg.cholesky_ex(checked, check_errors=False)
    if int(info.item()) != 0:
        raise ValueError("J must be symmetric positive definite")
    _require_derived_finite(chol, "information Gaussian Cholesky factor")
    return checked


def _require_derived_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
