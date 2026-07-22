"""Dense Cholesky implementation of the bounded precision-factor seam."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor

from vfe4.types.information import MatrixBlock, PrecisionDiagnostics, PrecisionFactor


class DenseCholeskyPrecision:
    """A dense positive-definite precision represented only by its Cholesky factor."""

    def __init__(self, matrix: Tensor) -> None:
        checked = _require_precision_matrix(matrix)
        chol, info = torch.linalg.cholesky_ex(checked, check_errors=False)
        if int(info.item()) != 0:
            raise ValueError("matrix must be symmetric positive definite")
        _require_derived_finite(chol, "precision Cholesky factor")
        eigenvalues = torch.linalg.eigvalsh(checked)
        _require_derived_finite(eigenvalues, "precision eigenvalues")
        lambda_min = float(eigenvalues[0].item())
        lambda_max = float(eigenvalues[-1].item())
        diagnostics = PrecisionDiagnostics(
            dimension=checked.shape[0],
            min_cholesky_pivot=float(torch.diagonal(chol).min().item()),
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            kappa_2=lambda_max / lambda_min,
        )
        self._chol = chol.detach().clone()
        self._diagnostics = diagnostics

    @property
    def dimension(self) -> int:
        return self._diagnostics.dimension

    @property
    def diagnostics(self) -> PrecisionDiagnostics:
        return self._diagnostics

    def solve(self, rhs: Tensor) -> Tensor:
        _require_rhs(rhs, self.dimension, self._chol.device, "rhs", matrix_allowed=True)
        matrix_rhs = rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
        result = torch.cholesky_solve(matrix_rhs, self._chol)
        if rhs.ndim == 1:
            result = result.squeeze(-1)
        _require_derived_finite(result, "precision solve")
        return result.detach().clone()

    def logdet(self) -> Tensor:
        result = 2.0 * torch.log(torch.diagonal(self._chol)).sum()
        _require_derived_finite(result, "precision log determinant")
        return result.detach().clone()

    def selected_inverse(
        self, blocks: Sequence[MatrixBlock]
    ) -> Mapping[MatrixBlock, Tensor]:
        if not isinstance(blocks, Sequence):
            raise ValueError("blocks must be a sequence")
        ordered_columns: list[int] = []
        for block in blocks:
            if not isinstance(block, MatrixBlock):
                raise ValueError("blocks must contain MatrixBlock records")
            if any(index >= self.dimension for index in block.rows):
                raise ValueError("block row index is out of range")
            if any(index >= self.dimension for index in block.columns):
                raise ValueError("block column index is out of range")
            for column in block.columns:
                if column not in ordered_columns:
                    ordered_columns.append(column)
        if len(ordered_columns) == self.dimension:
            raise ValueError("selected inverse cannot request all dimension columns")
        if not blocks:
            return MappingProxyType({})

        selector = torch.zeros(
            (self.dimension, len(ordered_columns)),
            dtype=torch.float64,
            device=self._chol.device,
        )
        selector[
            torch.tensor(ordered_columns, device=self._chol.device),
            torch.arange(len(ordered_columns), device=self._chol.device),
        ] = 1.0
        solved = torch.cholesky_solve(selector, self._chol)
        _require_derived_finite(solved, "selected inverse solve")
        column_positions = {
            column: position for position, column in enumerate(ordered_columns)
        }
        selected: dict[MatrixBlock, Tensor] = {}
        for block in blocks:
            row_indices = torch.tensor(block.rows, device=self._chol.device)
            positions = torch.tensor(
                [column_positions[column] for column in block.columns],
                device=self._chol.device,
            )
            value = solved.index_select(0, row_indices).index_select(1, positions)
            _require_derived_finite(value, "selected inverse block")
            selected[block] = value.detach().clone()
        return MappingProxyType(selected)

    def sample(self, noise: Tensor) -> Tensor:
        _require_rhs(noise, self.dimension, self._chol.device, "noise", matrix_allowed=False)
        result = torch.linalg.solve_triangular(
            self._chol.transpose(-1, -2), noise.unsqueeze(-1), upper=True
        ).squeeze(-1)
        _require_derived_finite(result, "precision sample")
        return result.detach().clone()

    def quadratic(self, value: Tensor) -> Tensor:
        _require_rhs(value, self.dimension, self._chol.device, "value", matrix_allowed=False)
        transformed = self._chol.transpose(-1, -2) @ value
        _require_derived_finite(transformed, "precision quadratic transform")
        result = torch.sum(transformed * transformed)
        _require_derived_finite(result, "precision quadratic")
        return result.detach().clone()

    def trace_inverse_product(self, left: PrecisionFactor) -> Tensor:
        if not isinstance(left, DenseCholeskyPrecision):
            raise ValueError("left must use the same supported factor implementation")
        if left.dimension != self.dimension:
            raise ValueError("precision factors must have the same dimension")
        if left._chol.device != self._chol.device:
            raise ValueError("precision factors must share a device")
        whitened = torch.linalg.solve_triangular(
            self._chol, left._chol, upper=False
        )
        _require_derived_finite(whitened, "precision trace transform")
        result = torch.sum(whitened * whitened)
        _require_derived_finite(result, "precision trace inverse product")
        return result.detach().clone()


def _require_precision_matrix(matrix: object) -> Tensor:
    if not isinstance(matrix, Tensor):
        raise ValueError("matrix must be a torch.Tensor")
    if matrix.dtype is not torch.float64:
        raise ValueError("matrix must use float64")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a nonempty square matrix")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must be finite")
    if not bool(torch.equal(matrix, matrix.transpose(-1, -2))):
        raise ValueError("matrix must be symmetric")
    return matrix.detach().clone()


def _require_rhs(
    value: object,
    dimension: int,
    device: torch.device,
    name: str,
    *,
    matrix_allowed: bool,
) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64:
        raise ValueError(f"{name} must use float64")
    valid_rank = value.ndim in ((1, 2) if matrix_allowed else (1,))
    if not valid_rank or value.shape[0] != dimension:
        shape = f"({dimension},) or ({dimension}, K)" if matrix_allowed else f"({dimension},)"
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must share the factor device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _require_derived_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
