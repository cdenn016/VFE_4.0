"""Allocation-bounded block-tridiagonal factor operations for H8.

Only explicit ``[N,b,b]`` diagonal and ``[N-1,b,b]`` lower-adjacent
storage is admitted.  The implementation never flattens the population axis,
forms a global identity, or retains the input precision after factorization.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from torch import Tensor

from vfe4.numerics.block_layout import BlockChainLayout, BlockId
from vfe4.types.h8 import (
    H8_MIN_CHOLESKY_PIVOT,
    BackendCounterSnapshot,
    BlockFillRecord,
    BlockPatternRecord,
    BlockStorageRecord,
    BlockTridiagonalPrecision,
    BlockWorkspaceRecord,
    SelectedInverseBlocks,
    SparseConditionDiagnostics,
)


H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS = 8
H8_HAGER_HIGHAM_1NORM_POLICY = (
    ("initial_probe", "uniform_all_positive_1_over_dimension"),
    ("zero_sign", "positive_one"),
    ("argmax_tie_break", "first_population_major_index"),
    (
        "stop_order",
        ("repeated_index", "selected_less_than_or_equal_dot_product"),
    ),
    ("repeated_index_appended_before_stop", True),
    ("dot_product_stop_operator", "<="),
    ("next_probe", "one_hot_selected_index"),
    ("maximum_iterations", H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS),
    (
        "convergence_reasons",
        ("repeated_index", "dot_product_stop", "maximum_iterations"),
    ),
)


def _block_vector_raw_sha256(
    value: Tensor,
    layout: BlockChainLayout,
) -> str:
    """Stream one ``[N,b]`` diagnostic hash without a length-``D`` view."""

    checked = _require_finite_cpu_float64(value, name="diagnostic block vector")
    layout.require_block_vector_shape(
        tuple(int(size) for size in checked.shape),
        name="diagnostic block vector",
    )
    digest = hashlib.sha256()
    first = True
    for population in range(layout.population_size):
        for local in range(layout.block_size):
            if not first:
                digest.update(b"|")
            digest.update(
                format(float(checked[population, local].item()), ".17g").encode(
                    "ascii"
                )
            )
            first = False
    return digest.hexdigest()


def _require_finite_cpu_float64(value: object, *, name: str) -> Tensor:
    if type(value) is not Tensor:
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.float64 or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


class BlockTridiagonalCholesky:
    """Owned block-bidiagonal Cholesky factor of a symmetric precision.

    Mathematical symmetry is deliberately not repaired here.  The original
    block bytes are consumed as supplied and symmetry/reconstruction residuals
    remain obligations of the frozen operand-local H8 budget.
    """

    __slots__ = (
        "_layout",
        "_diag",
        "_lower",
        "_pivots",
        "_factorization_calls",
        "_forward_calls",
        "_backward_calls",
        "_solve_calls",
        "_logdet_calls",
        "_selected_inverse_calls",
        "_sample_calls",
        "_quadratic_calls",
        "_trace_calls",
        "_sparse_matvec_calls",
        "_maximum_rhs_width",
        "_maximum_sample_rhs_width",
        "_selected_block_ids",
        "_forbidden_selected_blocks",
        "_forbidden_rhs_widths",
        "_maximum_workspace_shape",
        "_input_precision_scalar_count",
        "_selected_inverse_scalar_count",
        "_information_scalar_count",
        "_stored_block_ids",
        "_observed_offband_blocks",
        "_duplicated_upper_blocks",
        "_upper_block_scalar_count",
        "_diagnostics",
    )

    def __init__(self) -> None:
        raise TypeError(
            "BlockTridiagonalCholesky is factory-only; use factorize()"
        )

    @classmethod
    def _from_validated_factors(
        cls,
        *,
        layout: BlockChainLayout,
        diagonal_factor: Tensor,
        lower_factor: Tensor,
        pivots: tuple[float, ...],
        input_precision_scalar_count: int,
    ) -> "BlockTridiagonalCholesky":
        if type(layout) is not BlockChainLayout:
            raise ValueError("layout must be an exact BlockChainLayout")
        diagonal = _require_finite_cpu_float64(
            diagonal_factor,
            name="diagonal factor",
        )
        lower = _require_finite_cpu_float64(
            lower_factor,
            name="lower factor",
        )
        expected_diagonal_shape = (
            layout.population_size,
            layout.block_size,
            layout.block_size,
        )
        expected_lower_shape = (
            layout.horizon,
            layout.block_size,
            layout.block_size,
        )
        if tuple(diagonal.shape) != expected_diagonal_shape:
            raise ValueError("diagonal factor has the wrong owned shape")
        if tuple(lower.shape) != expected_lower_shape:
            raise ValueError("lower factor has the wrong owned shape")
        if bool(torch.count_nonzero(torch.triu(diagonal, diagonal=1))):
            raise ValueError("diagonal factors must be lower triangular")
        observed_pivots = tuple(
            float(torch.diagonal(local).min().item()) for local in diagonal
        )
        if (
            type(pivots) is not tuple
            or len(pivots) != layout.population_size
            or any(
                type(pivot) is not float
                or not math.isfinite(pivot)
                or pivot <= 0.0
                for pivot in pivots
            )
        ):
            raise ValueError("factor pivots must be exact positive finite records")
        if pivots != observed_pivots:
            raise ValueError("factor pivots do not match the owned factors")
        expected_precision_count = layout.band_storage_scalar_count
        if (
            type(input_precision_scalar_count) is not int
            or input_precision_scalar_count != expected_precision_count
        ):
            raise ValueError("input precision count does not match the layout")
        instance = object.__new__(cls)
        instance._layout = layout
        instance._diag = diagonal.detach().clone().contiguous()
        instance._lower = lower.detach().clone().contiguous()
        instance._pivots = pivots
        instance._factorization_calls = 1
        instance._forward_calls = 0
        instance._backward_calls = 0
        instance._solve_calls = 0
        instance._logdet_calls = 0
        instance._selected_inverse_calls = 0
        instance._sample_calls = 0
        instance._quadratic_calls = 0
        instance._trace_calls = 0
        instance._sparse_matvec_calls = 0
        # Factorization's local E_t solve uses one [b,b] right-hand side.
        instance._maximum_rhs_width = layout.block_size
        instance._maximum_sample_rhs_width = 0
        instance._selected_block_ids = ()
        instance._forbidden_selected_blocks = 0
        instance._forbidden_rhs_widths = []
        instance._maximum_workspace_shape = (
            layout.block_size,
            layout.block_size,
        )
        instance._input_precision_scalar_count = input_precision_scalar_count
        instance._selected_inverse_scalar_count = 0
        instance._information_scalar_count = 0
        # These observations are derived from the only two validated owned
        # tensor inventories above; there is no upper/off-band storage field.
        instance._stored_block_ids = (
            tuple(
                BlockId.diagonal(index)
                for index in range(int(instance._diag.shape[0]))
            )
            + tuple(
                BlockId.lower(index + 1)
                for index in range(int(instance._lower.shape[0]))
            )
        )
        instance._observed_offband_blocks = sum(
            block not in layout.stored_block_ids
            for block in instance._stored_block_ids
        )
        instance._duplicated_upper_blocks = sum(
            block.kind != "diagonal" and block.row < block.column
            for block in instance._stored_block_ids
        )
        instance._upper_block_scalar_count = (
            instance._duplicated_upper_blocks
            * layout.block_size
            * layout.block_size
        )
        instance._diagnostics = None
        return instance

    @classmethod
    def factorize(
        cls,
        precision: BlockTridiagonalPrecision,
    ) -> "BlockTridiagonalCholesky":
        if type(precision) is not BlockTridiagonalPrecision:
            raise ValueError("precision must be an exact BlockTridiagonalPrecision")
        layout = precision.layout
        diagonal, lower = precision._block_refs()
        block = layout.block_size
        diagonal_factors: list[Tensor] = []
        lower_factors: list[Tensor] = []
        pivots: list[float] = []

        schur = diagonal[0]
        for population in range(layout.population_size):
            factor, info = torch.linalg.cholesky_ex(
                schur,
                upper=False,
                check_errors=False,
            )
            if int(info.item()) != 0 or not bool(torch.isfinite(factor).all()):
                raise ValueError(
                    f"block {population} is not positive definite"
                )
            pivot = float(torch.diagonal(factor).min().item())
            if not math.isfinite(pivot) or pivot <= 0.0:
                raise ValueError(f"block {population} has a nonpositive pivot")
            diagonal_factors.append(factor)
            pivots.append(pivot)
            if population == layout.horizon:
                continue
            # E_t L_t^T = J_{t+1,t}; solve locally without materializing
            # an inverse or any population-wide right-hand side.
            lower_factor = torch.linalg.solve_triangular(
                factor,
                lower[population].T,
                upper=False,
                left=True,
            ).T
            lower_factors.append(lower_factor)
            schur = (
                diagonal[population + 1]
                - lower_factor @ lower_factor.T
            )

        diagonal_owned = torch.stack(diagonal_factors, dim=0)
        lower_owned = torch.stack(lower_factors, dim=0)
        if tuple(diagonal_owned.shape) != (
            layout.population_size,
            block,
            block,
        ) or tuple(lower_owned.shape) != (layout.horizon, block, block):
            raise RuntimeError("block Cholesky construction changed storage shape")
        return cls._from_validated_factors(
            layout=layout,
            diagonal_factor=diagonal_owned,
            lower_factor=lower_owned,
            pivots=tuple(pivots),
            input_precision_scalar_count=(
                int(diagonal.numel()) + int(lower.numel())
            ),
        )

    @property
    def dimension(self) -> int:
        return self._layout.dimension

    @property
    def layout(self) -> BlockChainLayout:
        return self._layout

    @property
    def diagonal_factor(self) -> Tensor:
        return self._diag.clone()

    @property
    def lower_factor(self) -> Tensor:
        return self._lower.clone()

    @property
    def pattern(self) -> BlockPatternRecord:
        return BlockPatternRecord()

    @property
    def storage(self) -> BlockStorageRecord:
        return BlockStorageRecord(
            layout=self._layout,
            precision_scalar_count=self._input_precision_scalar_count,
            factor_scalar_count=int(self._diag.numel() + self._lower.numel()),
            selected_inverse_scalar_count=self._selected_inverse_scalar_count,
            information_scalar_count=self._information_scalar_count,
            upper_block_scalar_count=self._upper_block_scalar_count,
        )

    @property
    def fill(self) -> BlockFillRecord:
        return BlockFillRecord(
            layout=self._layout,
            stored_block_ids=self._stored_block_ids,
            observed_offband_blocks=self._observed_offband_blocks,
            duplicated_upper_blocks=self._duplicated_upper_blocks,
        )

    @property
    def workspace(self) -> BlockWorkspaceRecord:
        shape = self._maximum_workspace_shape
        return BlockWorkspaceRecord(
            maximum_shape=shape,
            maximum_scalar_count=math.prod(shape),
            maximum_rhs_width=self._maximum_rhs_width,
            attempted_forbidden_rhs_widths=tuple(
                self._forbidden_rhs_widths
            ),
        )

    @property
    def counters(self) -> BackendCounterSnapshot:
        return BackendCounterSnapshot(
            layout=self._layout,
            factorization_calls=self._factorization_calls,
            forward_substitution_calls=self._forward_calls,
            backward_substitution_calls=self._backward_calls,
            solve_calls=self._solve_calls,
            logdet_calls=self._logdet_calls,
            selected_inverse_calls=self._selected_inverse_calls,
            sample_calls=self._sample_calls,
            quadratic_calls=self._quadratic_calls,
            trace_calls=self._trace_calls,
            sparse_matvec_calls=self._sparse_matvec_calls,
            maximum_rhs_width=self._maximum_rhs_width,
            maximum_sample_rhs_width=self._maximum_sample_rhs_width,
            selected_block_ids=self._selected_block_ids,
            selected_block_count=len(self._selected_block_ids),
            attempted_forbidden_selected_blocks=(
                self._forbidden_selected_blocks
            ),
            attempted_forbidden_rhs_widths=tuple(
                self._forbidden_rhs_widths
            ),
        )

    @property
    def diagnostics(self) -> SparseConditionDiagnostics:
        if self._diagnostics is None:
            self._diagnostics = self._estimate_condition()
        return self._diagnostics

    def _checked_rhs(self, rhs: object, *, name: str) -> tuple[Tensor, bool, int]:
        value = _require_finite_cpu_float64(rhs, name=name)
        shape = tuple(int(size) for size in value.shape)
        try:
            self._layout.require_rhs_shape(shape, name=name)
        except ValueError:
            self._record_rejected_shape(shape)
            raise
        squeeze = value.ndim == 2
        width = 1 if squeeze else int(value.shape[2])
        self._maximum_rhs_width = max(self._maximum_rhs_width, width)
        self._maximum_workspace_shape = max(
            self._maximum_workspace_shape,
            (self._layout.block_size, width),
            key=lambda item: math.prod(item),
        )
        self._information_scalar_count = max(
            self._information_scalar_count,
            self._layout.information_scalar_count,
        )
        return (value.unsqueeze(-1) if squeeze else value), squeeze, width

    def _record_rejected_shape(self, shape: tuple[int, ...]) -> None:
        positive_dimensions = tuple(
            dimension
            for dimension in shape
            if type(dimension) is int and dimension > 0
        )
        # The backend schema currently records attempted widths.  Preserve a
        # witness for every rejected shape, including (D,D), by recording its
        # final positive axis; use D only for a scalar/empty malformed input.
        attempted_width = (
            positive_dimensions[-1]
            if positive_dimensions
            else self._layout.dimension
        )
        self._forbidden_rhs_widths.append(attempted_width)

    def solve_factor(self, rhs: Tensor, *, transpose: bool) -> Tensor:
        if type(transpose) is not bool:
            raise ValueError("transpose must be a bool")
        value, squeeze, _ = self._checked_rhs(rhs, name="factor rhs")
        outputs: list[Tensor] = [torch.empty(0)] * self._layout.population_size
        if not transpose:
            self._forward_calls += 1
            for population in range(self._layout.population_size):
                local = value[population]
                if population:
                    local = (
                        local
                        - self._lower[population - 1]
                        @ outputs[population - 1]
                    )
                outputs[population] = torch.linalg.solve_triangular(
                    self._diag[population],
                    local,
                    upper=False,
                    left=True,
                )
        else:
            self._backward_calls += 1
            for population in range(
                self._layout.population_size - 1,
                -1,
                -1,
            ):
                local = value[population]
                if population < self._layout.horizon:
                    local = (
                        local
                        - self._lower[population].T
                        @ outputs[population + 1]
                    )
                outputs[population] = torch.linalg.solve_triangular(
                    self._diag[population].T,
                    local,
                    upper=True,
                    left=True,
                )
        result = torch.stack(outputs, dim=0)
        return result.squeeze(-1) if squeeze else result

    def solve(self, rhs: Tensor) -> Tensor:
        self._solve_calls += 1
        forward = self.solve_factor(rhs, transpose=False)
        return self.solve_factor(forward, transpose=True)

    def logdet(self) -> Tensor:
        self._logdet_calls += 1
        return 2.0 * torch.log(torch.diagonal(self._diag, dim1=-2, dim2=-1)).sum()

    def selected_inverse(
        self,
        blocks: Sequence[BlockId],
    ) -> SelectedInverseBlocks:
        try:
            requested = tuple(blocks)
            self._layout.require_complete_stored_blocks(requested)
        except (TypeError, ValueError):
            self._forbidden_selected_blocks += 1
            raise ValueError(
                "selected inverse requires exactly the canonical diagonal and "
                "lower-adjacent blocks"
            ) from None
        self._selected_inverse_calls += 1
        self._selected_block_ids = requested
        block = self._layout.block_size
        identity = torch.eye(block, dtype=torch.float64, device="cpu")
        diagonal: list[Tensor] = [torch.empty(0)] * self._layout.population_size
        lower: list[Tensor] = [torch.empty(0)] * self._layout.horizon

        inverse_last = torch.linalg.solve_triangular(
            self._diag[-1], identity, upper=False, left=True
        )
        diagonal[-1] = inverse_last.T @ inverse_last
        for population in range(self._layout.horizon - 1, -1, -1):
            inverse_diagonal = torch.linalg.solve_triangular(
                self._diag[population],
                identity,
                upper=False,
                left=True,
            )
            next_covariance = diagonal[population + 1]
            coupling = self._lower[population]
            lower[population] = (
                -next_covariance @ coupling @ inverse_diagonal
            )
            middle = identity + coupling.T @ next_covariance @ coupling
            diagonal[population] = (
                inverse_diagonal.T @ middle @ inverse_diagonal
            )

        selected = SelectedInverseBlocks(
            self._layout,
            torch.stack(diagonal, dim=0),
            torch.stack(lower, dim=0),
        )
        diag_ref, lower_ref = selected._block_refs()
        self._selected_inverse_scalar_count = int(
            diag_ref.numel() + lower_ref.numel()
        )
        return selected

    def sample(self, noise: Tensor) -> Tensor:
        checked = _require_finite_cpu_float64(noise, name="sample noise")
        shape = tuple(int(size) for size in checked.shape)
        try:
            self._layout.require_sample_shape(shape, name="sample noise")
        except ValueError:
            self._record_rejected_shape(shape)
            if shape:
                self._maximum_sample_rhs_width = max(
                    self._maximum_sample_rhs_width,
                    shape[-1],
                )
            raise
        self._sample_calls += 1
        self._maximum_sample_rhs_width = max(
            self._maximum_sample_rhs_width,
            1,
        )
        return self.solve_factor(checked, transpose=True)

    def quadratic(self, value: Tensor) -> Tensor:
        checked = _require_finite_cpu_float64(value, name="quadratic value")
        self._layout.require_block_vector_shape(
            tuple(int(size) for size in checked.shape),
            name="quadratic value",
        )
        self._quadratic_calls += 1
        total = torch.zeros((), dtype=torch.float64, device="cpu")
        for population in range(self._layout.population_size):
            transformed = self._diag[population].T @ checked[population]
            if population < self._layout.horizon:
                transformed = (
                    transformed
                    + self._lower[population].T @ checked[population + 1]
                )
            total = total + torch.dot(transformed, transformed)
        return total

    def _precision_blocks(self) -> tuple[Tensor, Tensor]:
        diagonal: list[Tensor] = []
        lower: list[Tensor] = []
        for population in range(self._layout.population_size):
            local = self._diag[population] @ self._diag[population].T
            if population:
                local = (
                    local
                    + self._lower[population - 1]
                    @ self._lower[population - 1].T
                )
            diagonal.append(local)
            if population < self._layout.horizon:
                lower.append(
                    self._lower[population] @ self._diag[population].T
                )
        return torch.stack(diagonal, dim=0), torch.stack(lower, dim=0)

    def sparse_matvec(self, value: Tensor) -> Tensor:
        checked = _require_finite_cpu_float64(value, name="sparse matvec value")
        self._layout.require_block_vector_shape(
            tuple(int(size) for size in checked.shape),
            name="sparse matvec value",
        )
        self._sparse_matvec_calls += 1
        diagonal, lower = self._precision_blocks()
        outputs: list[Tensor] = []
        for population in range(self._layout.population_size):
            local = diagonal[population] @ checked[population]
            if population:
                local = local + lower[population - 1] @ checked[population - 1]
            if population < self._layout.horizon:
                local = local + lower[population].T @ checked[population + 1]
            outputs.append(local)
        return torch.stack(outputs, dim=0)

    def trace_inverse_product(
        self,
        left: BlockTridiagonalPrecision,
    ) -> Tensor:
        if type(left) is not BlockTridiagonalPrecision:
            raise ValueError("left must be an exact BlockTridiagonalPrecision")
        if left.layout != self._layout:
            raise ValueError("trace operands must share an exact layout")
        self._trace_calls += 1
        selected = self.selected_inverse(self._layout.stored_block_ids)
        covariance_diagonal, covariance_lower = selected._block_refs()
        left_diagonal, left_lower = left._block_refs()
        diagonal_trace = torch.einsum(
            "nij,nji->",
            left_diagonal,
            covariance_diagonal,
        )
        adjacent_trace = 2.0 * torch.sum(left_lower * covariance_lower)
        return diagonal_trace + adjacent_trace

    def _precision_one_norm(self) -> Tensor:
        diagonal, lower = self._precision_blocks()
        column_sums = diagonal.abs().sum(dim=1)
        column_sums[:-1] = column_sums[:-1] + lower.abs().sum(dim=1)
        column_sums[1:] = column_sums[1:] + lower.abs().sum(dim=2)
        return column_sums.max()

    def _estimate_condition(self) -> SparseConditionDiagnostics:
        layout = self._layout
        x = torch.full(
            (layout.population_size, layout.block_size),
            1.0 / layout.dimension,
            dtype=torch.float64,
            device="cpu",
        )
        estimate = 0.0
        indices: list[tuple[int, int]] = []
        sign_hashes: list[str] = []
        convergence_reason = "maximum_iterations"
        iterations = 0
        for iteration in range(1, H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS + 1):
            iterations = iteration
            y = self.solve(x)
            estimate = max(estimate, float(y.abs().sum().item()))
            sign = torch.where(y >= 0.0, 1.0, -1.0)
            sign_hashes.append(_block_vector_raw_sha256(sign, layout))
            z = self.solve(sign)
            selected_index = (0, 0)
            selected = -1.0
            for population in range(layout.population_size):
                local_values = z[population].abs()
                local = int(torch.argmax(local_values).item())
                candidate = float(local_values[local].item())
                if candidate > selected:
                    selected = candidate
                    selected_index = (population, local)
            dot = float(torch.sum(z * x).item())
            if selected_index in indices:
                convergence_reason = "repeated_index"
                indices.append(selected_index)
                break
            indices.append(selected_index)
            if selected <= dot:
                convergence_reason = "dot_product_stop"
                break
            x = torch.zeros_like(x)
            population, local = selected_index
            x[population, local] = 1.0

        index_sha256 = hashlib.sha256(
            ",".join(
                f"{population}:{local}"
                for population, local in indices
            ).encode("ascii")
        ).hexdigest()
        sign_sha256 = hashlib.sha256(
            "|".join(sign_hashes).encode("ascii")
        ).hexdigest()
        pivots = self._pivots
        margins = tuple(pivot - H8_MIN_CHOLESKY_PIVOT for pivot in pivots)
        return SparseConditionDiagnostics(
            estimator="HagerHigham1NormEstimate-v1",
            kappa_1_estimate=float(self._precision_one_norm().item()) * estimate,
            iterations=iterations,
            convergence_reason=convergence_reason,
            index_sha256=index_sha256,
            sign_sha256=sign_sha256,
            per_block_min_pivots=pivots,
            global_min_pivot=min(pivots),
            per_block_pivot_margins=margins,
            global_pivot_margin=min(pivots) - H8_MIN_CHOLESKY_PIVOT,
        )


__all__ = [
    "H8_HAGER_HIGHAM_1NORM_POLICY",
    "H8_HAGER_HIGHAM_MAXIMUM_ITERATIONS",
    "BlockTridiagonalCholesky",
]
