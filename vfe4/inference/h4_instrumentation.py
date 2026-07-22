"""Identity-bound real-operation instrumentation for the H4 solver arms."""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeAlias

import torch
from torch import Tensor

from vfe4.types.h4 import (
    H4MemoryRecord,
    H4OperationKind,
    H4OperationRecord,
    H4SolverArm,
    _H4_SPD_PROOF_ISSUER,
    _H4SpdProof,
    _issue_h4_spd_proof,
)

_CHOLESKY = torch.linalg.cholesky
_SOLVE_TRIANGULAR = torch.linalg.solve_triangular
_MATRIX_MULTIPLY = torch.matmul
_RECORDER_CAPABILITY_ISSUER = object()

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
        if type(capability) is not _RecorderCapability:
            raise PermissionError("operation recorder binding requires a private capability")
        if self._capability_id is not None:
            raise ValueError("operation recorder is already bound to a facade")
        self._capability_id = id(capability)

    def _accept(
        self,
        capability: _RecorderCapability,
        record: H4OperationRecord,
    ) -> None:
        self._require_capability(capability)
        self._accept_authorized(capability, record)

    def _require_capability(self, capability: object) -> None:
        if type(capability) is not _RecorderCapability or id(capability) != self._capability_id:
            raise PermissionError("operation recorder mutation requires its facade capability")

    def _accept_authorized(
        self,
        capability: _RecorderCapability,
        record: H4OperationRecord,
    ) -> None:
        self._require_capability(capability)
        raise NotImplementedError


class NullOperationRecorder(_RecorderBase):
    """Recorder that preserves numerics and intentionally retains no operations."""

    __slots__ = ()

    def snapshot(self) -> tuple[H4OperationRecord, ...]:
        return ()

    def _accept_authorized(
        self,
        capability: _RecorderCapability,
        record: H4OperationRecord,
    ) -> None:
        self._require_capability(capability)
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

    def _accept_authorized(
        self,
        capability: _RecorderCapability,
        record: H4OperationRecord,
    ) -> None:
        self._require_capability(capability)
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

    def __init__(self, recorder: _RecorderBase, issuer: object) -> None:
        if issuer is not _RECORDER_CAPABILITY_ISSUER:
            raise PermissionError("recorder capability requires its private issuer")
        self._recorder = recorder
        recorder._bind(self)

    def emit(self, record: H4OperationRecord) -> None:
        self._recorder._accept(self, record)


@dataclass(frozen=True, slots=True, eq=False)
class _SuccessfulCholesky:
    value: Tensor
    factor: Tensor
    value_shape: tuple[int, ...]
    factor_shape: tuple[int, ...]
    value_storage_pointer: int
    factor_storage_pointer: int
    value_version: int
    factor_version: int
    problem_id: str
    arm: H4SolverArm


class InstrumentedLinearAlgebra:
    """Immutable identity-bound facade over the five counted matrix operations."""

    __slots__ = (
        "_problem_id",
        "_arm",
        "_recorder",
        "_cholesky_receipt",
        "__capability",
        "_locked",
    )

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
        object.__setattr__(self, "_cholesky_receipt", None)
        object.__setattr__(
            self,
            "_InstrumentedLinearAlgebra__capability",
            _RecorderCapability(recorder, _RECORDER_CAPABILITY_ISSUER),
        )
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
        object.__setattr__(self, "_cholesky_receipt", None)
        _tensor(value, "value")
        result = _CHOLESKY(value)
        self._emit("cholesky", (value,), result)
        object.__setattr__(
            self,
            "_cholesky_receipt",
            _SuccessfulCholesky(
                value,
                result,
                _shape(value),
                _shape(result),
                int(value.untyped_storage().data_ptr()),
                int(result.untyped_storage().data_ptr()),
                int(value._version),
                int(result._version),
                self._problem_id,
                self._arm,
            ),
        )
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

    if type(problem_id) is not str or not problem_id:
        raise ValueError("problem_id must be a nonempty string")
    if arm not in ("information", "moment"):
        raise ValueError("arm must be an H4 solver arm")
    if not isinstance(callable, Callable):
        raise ValueError("callable must be callable")
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        callable()
        return H4MemoryRecord(
            problem_id,
            arm,
            None,
            None,
            ("python_peak_bytes", "process_working_set_delta_bytes"),
        )

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    try:
        callable()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    return H4MemoryRecord(
        problem_id,
        arm,
        max(0, peak - baseline),
        None,
        ("process_working_set_delta_bytes",),
    )


def _facade_binding(linalg: InstrumentedLinearAlgebra) -> tuple[str, H4SolverArm]:
    if type(linalg) is not InstrumentedLinearAlgebra:
        raise ValueError("linalg must be an exact InstrumentedLinearAlgebra facade")
    return linalg._problem_id, linalg._arm


def _facade_uses_null_recorder(linalg: InstrumentedLinearAlgebra) -> bool:
    if type(linalg) is not InstrumentedLinearAlgebra:
        return False
    return type(linalg._recorder) is NullOperationRecorder


def _facade_proven_spd_tuple(
    linalg: InstrumentedLinearAlgebra,
    value: Tensor,
    factor: Tensor,
) -> tuple[tuple[tuple[float, ...], ...], _H4SpdProof]:
    if type(linalg) is not InstrumentedLinearAlgebra:
        raise ValueError("SPD proof requires an exact InstrumentedLinearAlgebra facade")
    matching = linalg._cholesky_receipt
    object.__setattr__(linalg, "_cholesky_receipt", None)
    if (
        type(matching) is not _SuccessfulCholesky
        or matching.value is not value
        or matching.factor is not factor
    ):
        raise ValueError("SPD proof requires a successful facade Cholesky receipt")
    if (
        matching.problem_id != linalg._problem_id
        or matching.arm != linalg._arm
        or _shape(value) != matching.value_shape
        or _shape(factor) != matching.factor_shape
        or int(value.untyped_storage().data_ptr()) != matching.value_storage_pointer
        or int(factor.untyped_storage().data_ptr()) != matching.factor_storage_pointer
        or int(value._version) != matching.value_version
        or int(factor._version) != matching.factor_version
    ):
        raise ValueError("SPD proof tensors changed after facade Cholesky")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("SPD proof value must be a square matrix")
    raw = tuple(tuple(float(item) for item in row) for row in value.tolist())
    return _issue_h4_spd_proof(raw, _H4_SPD_PROOF_ISSUER)


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
