"""Identity-bound real-operation instrumentation for the H4 solver arms."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import replace
from typing import TypeAlias

import torch
from torch import Tensor

from vfe4.types.h4 import (
    H4MemoryRecord,
    H4OperationKind,
    H4OperationRecord,
    H4SolverArm,
)

_CHOLESKY = torch.linalg.cholesky
_SOLVE_TRIANGULAR = torch.linalg.solve_triangular
_MATRIX_MULTIPLY = torch.matmul

_OperationKey: TypeAlias = tuple[
    str,
    H4SolverArm,
    H4OperationKind,
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]


class _RecorderBase:
    __slots__ = ("_capability_id",)

    def __init__(self) -> None:
        self._capability_id: int | None = None

    def _bind(self, capability: _RecorderCapability) -> None:
        if self._capability_id is not None:
            raise ValueError("operation recorder is already bound to a facade")
        self._capability_id = id(capability)

    def _accept(
        self,
        capability: _RecorderCapability,
        record: H4OperationRecord,
    ) -> None:
        if id(capability) != self._capability_id:
            raise PermissionError("operation recorder mutation requires its facade capability")
        self._accept_authorized(record)

    def _accept_authorized(self, record: H4OperationRecord) -> None:
        raise NotImplementedError


class NullOperationRecorder(_RecorderBase):
    """Recorder that preserves numerics and intentionally retains no operations."""

    __slots__ = ()

    def snapshot(self) -> tuple[H4OperationRecord, ...]:
        return ()

    def _accept_authorized(self, record: H4OperationRecord) -> None:
        del record


class CountingOperationRecorder(_RecorderBase):
    """Aggregate successful real operations in first-successful-call order."""

    __slots__ = ("_records", "_positions")

    def __init__(self) -> None:
        super().__init__()
        self._records: list[H4OperationRecord] = []
        self._positions: dict[_OperationKey, int] = {}

    def snapshot(self) -> tuple[H4OperationRecord, ...]:
        return tuple(self._records)

    def _accept_authorized(self, record: H4OperationRecord) -> None:
        key: _OperationKey = (
            record.problem_id,
            record.arm,
            record.operation,
            record.operand_shapes,
            record.result_shape,
        )
        position = self._positions.get(key)
        if position is None:
            self._positions[key] = len(self._records)
            self._records.append(record)
            return
        self._records[position] = replace(
            self._records[position],
            count=self._records[position].count + 1,
        )


class _RecorderCapability:
    __slots__ = ("_recorder",)

    def __init__(self, recorder: _RecorderBase) -> None:
        self._recorder = recorder
        recorder._bind(self)

    def emit(self, record: H4OperationRecord) -> None:
        self._recorder._accept(self, record)


class InstrumentedLinearAlgebra:
    """Immutable identity-bound facade over the five counted matrix operations."""

    __slots__ = ("_problem_id", "_arm", "_recorder", "__capability", "_locked")

    def __init__(
        self,
        *,
        problem_id: str,
        arm: H4SolverArm,
        recorder: NullOperationRecorder | CountingOperationRecorder,
    ) -> None:
        if type(problem_id) is not str or not problem_id:
            raise ValueError("problem_id must be a nonempty string")
        if arm not in ("information", "moment"):
            raise ValueError("arm must be an H4 solver arm")
        if type(recorder) not in (NullOperationRecorder, CountingOperationRecorder):
            raise ValueError("recorder must be an exact H4 operation recorder")
        object.__setattr__(self, "_problem_id", problem_id)
        object.__setattr__(self, "_arm", arm)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_InstrumentedLinearAlgebra__capability", _RecorderCapability(recorder))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("InstrumentedLinearAlgebra identity is immutable")
        object.__setattr__(self, name, value)

    @property
    def problem_id(self) -> str:
        return self._problem_id

    @property
    def arm(self) -> H4SolverArm:
        return self._arm

    def cholesky(self, value: Tensor) -> Tensor:
        _tensor(value, "value")
        result = _CHOLESKY(value)
        self._emit("cholesky", (value,), result)
        return result

    def triangular_solve(
        self,
        triangular: Tensor,
        rhs: Tensor,
        *,
        upper: bool = False,
    ) -> Tensor:
        _tensor(triangular, "triangular")
        _tensor(rhs, "rhs")
        if type(upper) is not bool:
            raise ValueError("upper must be a bool")
        vector_rhs = rhs.ndim == 1
        prepared_rhs = rhs.unsqueeze(-1) if vector_rhs else rhs
        result = _SOLVE_TRIANGULAR(
            triangular,
            prepared_rhs,
            upper=upper,
        )
        if vector_rhs:
            result = result.squeeze(-1)
        self._emit("triangular_solve", (triangular, rhs), result)
        return result

    def matrix_multiply(self, left: Tensor, right: Tensor) -> Tensor:
        _tensor(left, "left")
        _tensor(right, "right")
        result = _MATRIX_MULTIPLY(left, right)
        self._emit("matrix_multiply", (left, right), result)
        return result

    def symmetric_rank_update(
        self,
        covariance: Tensor,
        gain: Tensor,
        innovation_covariance: Tensor,
    ) -> Tensor:
        _tensor(covariance, "covariance")
        _tensor(gain, "gain")
        _tensor(innovation_covariance, "innovation_covariance")
        result = covariance - _MATRIX_MULTIPLY(
            _MATRIX_MULTIPLY(gain, innovation_covariance),
            gain.T,
        )
        result = 0.5 * (result + result.T)
        self._emit(
            "symmetric_rank_update",
            (covariance, gain, innovation_covariance),
            result,
        )
        return result

    def selected_block_extract(
        self,
        value: Tensor,
        row_indices: tuple[int, ...],
        column_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        _tensor(value, "value")
        rows = _indices(row_indices, value.shape[0], "row_indices")
        row_tensor = torch.tensor(rows, dtype=torch.int64, device=value.device)
        if value.ndim == 1:
            if column_indices is not None:
                raise ValueError("column_indices must be None for a vector")
            result = torch.index_select(value, 0, row_tensor)
        elif value.ndim == 2:
            if column_indices is None:
                raise ValueError("column_indices are required for a matrix")
            columns = _indices(column_indices, value.shape[1], "column_indices")
            column_tensor = torch.tensor(columns, dtype=torch.int64, device=value.device)
            result = torch.index_select(
                torch.index_select(value, 0, row_tensor),
                1,
                column_tensor,
            )
        else:
            raise ValueError("selected_block_extract accepts only a vector or matrix")
        self._emit("selected_block_extract", (value,), result)
        return result

    def _emit(
        self,
        operation: H4OperationKind,
        operands: tuple[Tensor, ...],
        result: Tensor,
    ) -> None:
        record = H4OperationRecord(
            self._problem_id,
            self._arm,
            operation,
            tuple(_shape(operand) for operand in operands),
            _shape(result),
            1,
        )
        self.__capability.emit(record)


def measure_untimed_memory(
    problem_id: str,
    arm: H4SolverArm,
    callable: Callable[[], object],
) -> H4MemoryRecord:
    """Run one untimed call and return its separately measured Python peak."""

    if not isinstance(callable, Callable):
        raise ValueError("callable must be callable")
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
        baseline = 0
    try:
        callable()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()
    return H4MemoryRecord(
        problem_id,
        arm,
        max(0, peak - baseline),
        None,
        ("process_working_set_delta_bytes",),
    )


def _facade_binding(linalg: InstrumentedLinearAlgebra) -> tuple[str, H4SolverArm]:
    if not isinstance(linalg, InstrumentedLinearAlgebra):
        raise ValueError("linalg must be an InstrumentedLinearAlgebra facade")
    return linalg.problem_id, linalg.arm


def _facade_uses_null_recorder(linalg: InstrumentedLinearAlgebra) -> bool:
    if not isinstance(linalg, InstrumentedLinearAlgebra):
        return False
    return type(linalg._recorder) is NullOperationRecorder


def _tensor(value: object, name: str) -> None:
    if type(value) is not Tensor:
        raise ValueError(f"{name} must be an exact torch.Tensor")
    if value.ndim not in (1, 2) or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector or matrix")
    if value.dtype is not torch.float64 or value.device.type != "cpu":
        raise ValueError(f"{name} must be CPU float64")


def _indices(value: object, bound: int, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not int or item < 0 or item >= bound for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be nonempty unique indices in range")
    return value


def _shape(value: Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


__all__ = [
    "H4OperationKind",
    "NullOperationRecorder",
    "CountingOperationRecorder",
    "InstrumentedLinearAlgebra",
    "measure_untimed_memory",
]
