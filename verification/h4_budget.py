"""Operand-shaped, bounded-memory numerical budgets for H4."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from vfe4.config.schema import H4ConditionEnvelopeConfig
from vfe4.types.h4 import (
    H4_ALLOWANCE_ELEMENT_COUNTS,
    H4AllowanceElement,
    H4AllowanceInvariantName,
    H4AllowanceOperationCount,
    H4AllowanceOperand,
    H4ApplicableAllowance,
    H4NativeInformationState,
    H4NativeMomentState,
    H4_PROBLEM_SEEDS,
    H4SelectedMoment,
    H4SolverResult,
    H4SolverArm,
    H4TerminalLaw,
)
from vfe4.validation.h3_fixture import parse_h3_fixture_bytes

from .h4_records import H4InnovationConditionRecord, H4PosteriorConditionRecord
from .numpy_oracles import h4_gaussian
from .numpy_oracles.h4_gaussian import (
    H4OracleEvaluation,
    H4OracleKLEvaluation,
)

H4_EPSILON = 2.220446049250313e-16
H4_ROUNDING_CONSTANT = 4096.0
H4_SOLVER_RELATIVE_BUDGET = 1.0e-9
H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION = 1.0e-4
H4_MAXIMUM_CHUNK_ROWS = 4096
H4_ALLOWANCE_STREAM_DOMAIN = "vfe4.h4.allowance-element-stream.v1"
_H4_ALLOWANCE_GROUP_VECTOR_DOMAIN = b"vfe4.h4.allowance-group-vector.v1\x00"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PACKED_ROW_DTYPE = np.dtype(
    [
        ("left_value", "<f8"), ("right_value", "<f8"),
        ("left_value_norm", "<f8"), ("right_value_norm", "<f8"),
        ("left_absolute_sum", "<f8"), ("right_absolute_sum", "<f8"),
        ("left_rounding", "<f8"), ("left_solver", "<f8"),
        ("right_rounding", "<f8"), ("right_solver", "<f8"),
        ("comparison_allowance", "<f8"), ("residual", "<f8"),
        ("normalized_residual", "<f8"), ("final_allowance", "<f8"),
        ("allowance_scale_ratio", "<f8"), ("decisive", "u1"),
        ("passed", "u1"),
    ],
    align=False,
)
if _PACKED_ROW_DTYPE.itemsize != 122:  # pragma: no cover - import-time contract
    raise RuntimeError("H4 packed allowance row must be exactly 122 bytes")


def gamma_n(n: int) -> float:
    if type(n) is not int or n < 0 or n * H4_EPSILON >= 1.0:
        raise ValueError("n must be a nonnegative integer in the binary64 gamma domain")
    return (n * H4_EPSILON) / (1.0 - n * H4_EPSILON)


def dot_operation_count(k: int) -> int:
    if type(k) is not int or k < 0:
        raise ValueError("dot dimension must be nonnegative")
    return max(0, 2 * k - 1)


def matrix_multiply_operation_count(m: int, k: int, n: int) -> int:
    if any(type(value) is not int or value < 0 for value in (m, k, n)):
        raise ValueError("matrix dimensions must be nonnegative integers")
    return m * n * dot_operation_count(k)


def triangular_solve_operation_count(n: int, rhs_columns: int) -> int:
    if any(type(value) is not int or value < 0 for value in (n, rhs_columns)):
        raise ValueError("triangular solve dimensions must be nonnegative integers")
    return rhs_columns * n * n


def cholesky_operation_count(n: int) -> int:
    if type(n) is not int or n < 0:
        raise ValueError("Cholesky dimension must be nonnegative")
    return math.ceil(n * n * n / 3)


def operand_allowance(
    *,
    label: str,
    value: float,
    value_norm: float,
    absolute_summand_accumulation: float,
    condition_numbers: tuple[float, ...],
    operation_counts: tuple[H4AllowanceOperationCount, ...],
    solver_produced: bool,
    invariant_scale: float,
) -> H4AllowanceOperand:
    if type(invariant_scale) is not float or not math.isfinite(invariant_scale) or invariant_scale < 1.0:
        raise ValueError("invariant_scale must be a finite float at least one")
    if any(type(value_) is not float or not math.isfinite(value_) for value_ in (value, value_norm, absolute_summand_accumulation)):
        raise ValueError("operand scalar inputs must be exact finite floats")
    if type(condition_numbers) is not tuple or type(operation_counts) is not tuple:
        raise ValueError("operand evidence must be immutable tuples")
    operation_count = sum(item.count for item in operation_counts)
    kappa = max((1.0, *condition_numbers))
    rounding = (
        H4_ROUNDING_CONSTANT * gamma_n(operation_count) * kappa
        * max(1.0, value_norm, absolute_summand_accumulation)
    )
    solver = H4_SOLVER_RELATIVE_BUDGET * invariant_scale if solver_produced else 0.0
    return H4AllowanceOperand(
        label, value, value_norm, absolute_summand_accumulation,
        condition_numbers, operation_counts, solver_produced,
        rounding, solver, rounding + solver,
    )


def pair_element_allowance(
    *,
    stream_index: int,
    invariant: H4AllowanceInvariantName,
    problem_id: str,
    comparison_source: Literal[
        "solver_to_oracle", "adapter_to_h3_reference", "adapter_to_oracle"
    ],
    repetition_index: int | None,
    arm: H4SolverArm | None,
    path: str,
    shape: tuple[int, ...],
    flat_index: int,
    left: H4AllowanceOperand,
    right: H4AllowanceOperand,
) -> H4AllowanceElement:
    if type(left) is not H4AllowanceOperand or type(right) is not H4AllowanceOperand:
        raise ValueError("pair allowance requires exact operand records")
    scale = max(1.0, abs(left.value), left.value_norm, abs(right.value), right.value_norm)
    for operand in (left, right):
        expected_solver = H4_SOLVER_RELATIVE_BUDGET * scale if operand.solver_produced else 0.0
        if operand.solver_allowance != expected_solver:
            raise ValueError("operand solver allowance was not computed with this element scale")
    comparison = (
        H4_ROUNDING_CONSTANT * gamma_n(3)
        * max(1.0, abs(left.value), abs(right.value), abs(left.value) + abs(right.value))
    )
    residual = abs(left.value - right.value)
    final = left.total_allowance + right.total_allowance + comparison
    normalized = residual / final
    ratio = final / scale
    return H4AllowanceElement(
        stream_index, invariant, problem_id, comparison_source, repetition_index,
        arm, path, shape, flat_index, scale, left, right, comparison, residual,
        normalized, final, ratio,
        ratio < H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION, residual <= final,
    )


@dataclass(frozen=True, slots=True)
class _H4AllowanceOperandGroup:
    label: str
    values: NDArray[np.float64]
    value_norm: float
    absolute_summand_accumulations: NDArray[np.float64]
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[H4AllowanceOperationCount, ...]
    solver_produced: bool

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("allowance operand group label must be nonempty")
        for name in ("values", "absolute_summand_accumulations"):
            array = getattr(self, name)
            if (
                type(array) is not np.ndarray
                or array.dtype != np.dtype(np.float64)
                or array.ndim != 1
                or not array.flags.c_contiguous
                or array.flags.writeable
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{name} must be a finite read-only C-contiguous float64 vector")
        if self.values.size != self.absolute_summand_accumulations.size or self.values.size == 0:
            raise ValueError("allowance operand group arrays must have equal nonzero length")
        if np.any(self.absolute_summand_accumulations < 0.0):
            raise ValueError("absolute summand accumulations must be nonnegative")
        if type(self.value_norm) is not float or not math.isfinite(self.value_norm) or self.value_norm < 0.0 or self.value_norm < float(np.max(np.abs(self.values))):
            raise ValueError("group value_norm must dominate every scalar value")
        if type(self.condition_numbers) is not tuple or not self.condition_numbers or any(type(item) is not float or not math.isfinite(item) or item <= 0.0 for item in self.condition_numbers):
            raise ValueError("group condition numbers must be positive finite floats")
        if type(self.operation_counts) is not tuple or not all(type(item) is H4AllowanceOperationCount for item in self.operation_counts):
            raise ValueError("group operation counts must be exact immutable records")
        if len({item.label for item in self.operation_counts}) != len(self.operation_counts):
            raise ValueError("group operation labels must be unique")
        if type(self.solver_produced) is not bool:
            raise ValueError("group solver provenance must be bool")


@dataclass(frozen=True, slots=True)
class _H4AllowanceGroupInput:
    problem_id: str
    problem_sha256: str
    comparison_source: Literal[
        "solver_to_oracle", "adapter_to_h3_reference", "adapter_to_oracle"
    ]
    repetition_index: int | None
    arm: H4SolverArm | None
    path_prefix: str
    shape: tuple[int, ...]
    left: _H4AllowanceOperandGroup
    right: _H4AllowanceOperandGroup

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id or type(self.problem_sha256) is not str or _SHA256.fullmatch(self.problem_sha256) is None:
            raise ValueError("allowance group problem identity is invalid")
        if type(self.path_prefix) is not str or not self.path_prefix:
            raise ValueError("allowance group path prefix must be nonempty")
        if type(self.shape) is not tuple or not self.shape or any(type(item) is not int or item <= 0 for item in self.shape):
            raise ValueError("allowance group shape must be a positive tuple")
        if type(self.left) is not _H4AllowanceOperandGroup or type(self.right) is not _H4AllowanceOperandGroup:
            raise ValueError("allowance group must own exact operand groups")
        element_count = math.prod(self.shape)
        if self.left.values.size != element_count or self.right.values.size != element_count:
            raise ValueError("allowance group arrays must equal product(shape)")
        if self.comparison_source == "solver_to_oracle":
            if self.arm not in ("information", "moment"):
                raise ValueError("solver group requires a real arm")
            if self.problem_id.startswith("h4-anchor-"):
                if self.repetition_index is not None:
                    raise ValueError("anchor solver group has no repetition")
            elif type(self.repetition_index) is not int or self.repetition_index not in range(11):
                raise ValueError("scaled solver group requires repetition 0..10")
        elif self.comparison_source in ("adapter_to_h3_reference", "adapter_to_oracle"):
            if self.arm is not None or self.repetition_index is not None:
                raise ValueError("adapter group cannot carry arm/repetition")
        else:
            raise ValueError("invalid allowance comparison source")


@dataclass(frozen=True, slots=True)
class H4ResultAllowanceGroupBundle:
    kl_to_zero: _H4AllowanceGroupInput
    terminal_h: _H4AllowanceGroupInput
    terminal_J: _H4AllowanceGroupInput
    selected_mean_and_covariance: tuple[_H4AllowanceGroupInput, ...]
    complete_objective: _H4AllowanceGroupInput

    def __post_init__(self) -> None:
        if (
            type(self.kl_to_zero) is not _H4AllowanceGroupInput
            or type(self.terminal_h) is not _H4AllowanceGroupInput
            or type(self.terminal_J) is not _H4AllowanceGroupInput
            or type(self.selected_mean_and_covariance) is not tuple
            or not all(
                type(item) is _H4AllowanceGroupInput
                for item in self.selected_mean_and_covariance
            )
            or type(self.complete_objective) is not _H4AllowanceGroupInput
        ):
            raise ValueError("H4 result allowance bundle owns exact immutable groups")
        groups = (
            self.kl_to_zero, self.terminal_h, self.terminal_J,
            *self.selected_mean_and_covariance, self.complete_objective,
        )
        identity = (
            self.kl_to_zero.problem_id, self.kl_to_zero.problem_sha256,
            self.kl_to_zero.arm, self.kl_to_zero.repetition_index,
        )
        if any(
            (
                group.problem_id, group.problem_sha256, group.arm,
                group.repetition_index,
            ) != identity
            or group.comparison_source != "solver_to_oracle"
            for group in groups
        ):
            raise ValueError("allowance bundle group identity/arm/repetition is mixed")
        problem_id = self.kl_to_zero.problem_id
        if problem_id in (
            "h4-anchor-h3-coupled-v1", "h4-anchor-h3-zero-control-v1",
        ):
            horizon, expected_dimension, selected_dimension = 1, 4, 2
        else:
            match = re.fullmatch(
                r"h4-(?:coupled|zero_control)-T(7|15|31)-dz4-dm4-seed[1-9][0-9]*-v1",
                problem_id,
            )
            if match is None:
                raise ValueError("allowance bundle problem identity is not canonical")
            horizon = int(match.group(1))
            expected_dimension = 8 * (horizon + 1)
            selected_dimension = 8
        if (
            self.kl_to_zero.path_prefix != "kl_to_zero"
            or self.kl_to_zero.shape != (1,)
            or self.kl_to_zero.left.label != "kl_to_oracle"
            or self.kl_to_zero.right.label != "literal_zero"
            or self.terminal_h.path_prefix != "terminal_h"
            or self.terminal_h.shape != (expected_dimension,)
            or self.terminal_h.left.label != "terminal_h"
            or self.terminal_h.right.label != "oracle_natural"
            or self.terminal_J.path_prefix != "terminal_J"
            or self.terminal_J.shape != (expected_dimension, expected_dimension)
            or self.terminal_J.left.label != "terminal_J"
            or self.terminal_J.right.label != "oracle_precision"
            or self.complete_objective.path_prefix != "complete_objective"
            or self.complete_objective.shape != (1,)
            or self.complete_objective.left.label != "terminal_complete_objective"
            or self.complete_objective.right.label
            != "oracle_canonical_log_normalizer"
        ):
            raise ValueError("allowance bundle fixed group path/shape is incompatible")
        expected_labels = (
            "initial", "terminal",
            *(f"observation[{time_index}]" for time_index in range(1, horizon + 1)),
        )
        if len(self.selected_mean_and_covariance) != 2 * len(expected_labels):
            raise ValueError("allowance bundle selected group count is incomplete")
        for selected_index, label in enumerate(expected_labels):
            mean_group = self.selected_mean_and_covariance[2 * selected_index]
            covariance_group = self.selected_mean_and_covariance[2 * selected_index + 1]
            prefix = f"selected_moments.{label}"
            if (
                mean_group.path_prefix != f"{prefix}.mean"
                or mean_group.shape != (selected_dimension,)
                or mean_group.left.label != f"terminal_selected_mean:{label}"
                or mean_group.right.label != f"oracle_selected_mean:{label}"
                or covariance_group.path_prefix != f"{prefix}.covariance"
                or covariance_group.shape != (selected_dimension, selected_dimension)
                or covariance_group.left.label
                != f"terminal_selected_covariance:{label}"
                or covariance_group.right.label
                != f"oracle_selected_covariance:{label}"
            ):
                raise ValueError("allowance bundle selected mean/covariance sequence is invalid")


@dataclass(frozen=True, slots=True)
class H4AllowanceResultSource:
    problem_payload: bytes
    repetition_index: int | None
    oracle: H4OracleEvaluation
    result: H4SolverResult
    terminal: H4TerminalLaw
    kl_to_oracle: H4OracleKLEvaluation

    def __post_init__(self) -> None:
        if type(self.problem_payload) is not bytes or not self.problem_payload:
            raise ValueError("allowance result source requires immutable problem bytes")
        if self.repetition_index is not None and (
            type(self.repetition_index) is not int
            or self.repetition_index not in range(11)
        ):
            raise ValueError("allowance repetition must be None or 0..10")
        if (
            type(self.oracle) is not H4OracleEvaluation
            or type(self.result) is not H4SolverResult
            or type(self.terminal) is not H4TerminalLaw
            or type(self.kl_to_oracle) is not H4OracleKLEvaluation
        ):
            raise ValueError("allowance source numerical records must have exact types")
        if self.oracle.source_kind == "h3_anchor":
            if self.repetition_index is not None:
                raise ValueError("anchor allowance result source cannot carry a repetition")
        elif self.oracle.source_kind == "scaled_pcg64":
            if type(self.repetition_index) is not int or self.repetition_index not in range(11):
                raise ValueError("scaled allowance result source requires repetition 0..10")
        else:  # pragma: no cover - closed by H4OracleEvaluation
            raise ValueError("unknown allowance source kind")
        if (
            self.result.problem_id != self.oracle.problem_id
            or self.result.problem_sha256 != self.oracle.problem_sha256
            or self.terminal.arm != self.result.arm
            or self.result.factor_count != len(self.oracle.factor_ids)
        ):
            raise ValueError("allowance source problem/arm identity disagrees")
        if tuple(item.name for item in self.terminal.selected_moments) != tuple(
            item.name for item in self.oracle.selected_moments
        ):
            raise ValueError("allowance source selected labels disagree")


@dataclass(frozen=True, slots=True)
class H4AnchorAllowanceSource:
    h3_fixture_bytes: bytes
    information: H4AllowanceResultSource
    moment: H4AllowanceResultSource

    def __post_init__(self) -> None:
        if type(self.h3_fixture_bytes) is not bytes or not self.h3_fixture_bytes:
            raise ValueError("anchor source requires immutable H3 fixture bytes")
        if (
            type(self.information) is not H4AllowanceResultSource
            or type(self.moment) is not H4AllowanceResultSource
        ):
            raise ValueError("anchor source requires exact arm sources")
        if self.information.repetition_index is not None or self.moment.repetition_index is not None:
            raise ValueError("anchor allowance sources cannot carry a repetition")
        if self.information.result.arm != "information" or self.moment.result.arm != "moment":
            raise ValueError("anchor allowance arm order is frozen")
        if (
            self.information.problem_payload != self.moment.problem_payload
            or self.information.oracle.problem_id != self.moment.oracle.problem_id
            or self.information.oracle.problem_sha256 != self.moment.oracle.problem_sha256
            or self.information.oracle.source_kind != "h3_anchor"
            or self.moment.oracle.source_kind != "h3_anchor"
        ):
            raise ValueError("anchor allowance arms must bind one H3 problem")


def allowance_group_header(
    invariant: H4AllowanceInvariantName,
    group: _H4AllowanceGroupInput,
) -> bytes:
    if invariant not in dict(H4_ALLOWANCE_ELEMENT_COUNTS) or type(group) is not _H4AllowanceGroupInput:
        raise ValueError("invalid allowance group header inputs")
    def vector_digest(lane_name: str, array: NDArray[np.float64]) -> str:
        digest = hashlib.sha256(_H4_ALLOWANCE_GROUP_VECTOR_DOMAIN)
        digest.update(lane_name.encode("ascii"))
        digest.update(b"\x00")
        digest.update(array.size.to_bytes(8, "big"))
        digest.update(
            np.asarray(array, dtype=np.dtype("<f8"), order="C").tobytes(order="C")
        )
        return digest.hexdigest()

    payload = {
        "invariant": invariant,
        "problem_id": group.problem_id,
        "problem_sha256": group.problem_sha256,
        "comparison_source": group.comparison_source,
        "repetition_index": group.repetition_index,
        "arm": group.arm,
        "path_prefix": group.path_prefix,
        "shape": group.shape,
        "element_count": math.prod(group.shape),
        "left_label": group.left.label,
        "right_label": group.right.label,
        "left_operation_counts": tuple((item.label, item.count) for item in group.left.operation_counts),
        "right_operation_counts": tuple((item.label, item.count) for item in group.right.operation_counts),
        "left_condition_numbers": tuple(item.hex() for item in group.left.condition_numbers),
        "right_condition_numbers": tuple(item.hex() for item in group.right.condition_numbers),
        "left_value_norm": group.left.value_norm.hex(),
        "right_value_norm": group.right.value_norm.hex(),
        "left_values_sha256": vector_digest("left_value", group.left.values),
        "right_values_sha256": vector_digest("right_value", group.right.values),
        "left_absolute_summands_sha256": vector_digest(
            "left_absolute_summand", group.left.absolute_summand_accumulations,
        ),
        "right_absolute_summands_sha256": vector_digest(
            "right_absolute_summand", group.right.absolute_summand_accumulations,
        ),
        "left_solver_produced": group.left.solver_produced,
        "right_solver_produced": group.right.solver_produced,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _readonly_vector(values: object) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, order="C", copy=True).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("allowance operand vector must be finite")
    array.setflags(write=False)
    return array


def _operation_table(*items: tuple[str, int]) -> tuple[H4AllowanceOperationCount, ...]:
    return tuple(H4AllowanceOperationCount(label, count) for label, count in items)


def _vector_norm(values: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(values)))


def _matrix_norm(values: NDArray[np.float64]) -> float:
    return float(np.max(np.sum(np.abs(values), axis=1, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class _FactorRouteEvidence:
    factor_conditions: tuple[float, ...]
    posterior_condition: float
    J_absolute_summands: NDArray[np.float64]
    h_absolute_summands: NDArray[np.float64]
    c_absolute_summands: NDArray[np.float64]
    J_operations: tuple[H4AllowanceOperationCount, ...]
    h_operations: tuple[H4AllowanceOperationCount, ...]
    c_operations: tuple[H4AllowanceOperationCount, ...]
    selected_mean_operations: tuple[H4AllowanceOperationCount, ...]
    selected_covariance_base_operations: tuple[H4AllowanceOperationCount, ...]


def _factor_route_evidence(
    factors: tuple[object, ...], oracle: H4OracleEvaluation,
) -> _FactorRouteEvidence:
    dimension = oracle.dimension
    J_summands = np.zeros((dimension, dimension), dtype=np.float64)
    h_summands = np.zeros(dimension, dtype=np.float64)
    c_summand = 0.0
    conditions: list[float] = []
    cholesky_count = 0
    solve_A_count = 0
    solve_b_count = 0
    J_matmul_count = 0
    h_matmul_count = 0
    quadratic_count = 0
    logdet_count = 0
    for factor in factors:
        covariance = factor.covariance
        matrix = factor.matrix
        target = factor.target
        rows = int(target.size)
        lower = np.linalg.cholesky(covariance)
        solved_matrix = np.linalg.solve(
            lower.T, np.linalg.solve(lower, matrix),
        )
        solved_target = np.linalg.solve(
            lower.T, np.linalg.solve(lower, target),
        )
        J_contribution = matrix.T @ solved_matrix
        h_contribution = matrix.T @ solved_target
        quadratic = float(target @ solved_target)
        logdet = 2.0 * float(
            np.sum(np.log(np.diag(lower)), dtype=np.float64)
        )
        c_contribution = -0.5 * (
            quadratic + rows * math.log(2.0 * math.pi) + logdet
        )
        J_summands += np.abs(J_contribution)
        h_summands += np.abs(h_contribution)
        c_summand += abs(c_contribution)
        conditions.append(float(np.linalg.cond(covariance)))
        cholesky_count += cholesky_operation_count(rows)
        solve_A_count += 2 * triangular_solve_operation_count(rows, dimension)
        solve_b_count += 2 * triangular_solve_operation_count(rows, 1)
        J_matmul_count += matrix_multiply_operation_count(dimension, rows, dimension)
        h_matmul_count += matrix_multiply_operation_count(dimension, rows, 1)
        quadratic_count += dot_operation_count(rows)
        logdet_count += rows + max(0, rows - 1)
    factor_count = len(factors)
    J_sum = factor_count * dimension * dimension
    h_sum = factor_count * dimension
    precision_symmetrization = 2 * dimension * dimension
    posterior_cholesky = cholesky_operation_count(dimension)
    posterior_natural_solve = 2 * triangular_solve_operation_count(dimension, 1)
    posterior_condition = float(
        np.linalg.cond(np.asarray(oracle.precision, dtype=np.float64))
    )
    J_summands.setflags(write=False)
    h_summands.setflags(write=False)
    c_array = _readonly_vector((c_summand,))
    return _FactorRouteEvidence(
        tuple(conditions), posterior_condition,
        J_summands.reshape(-1), h_summands, c_array,
        _operation_table(
            ("factor_covariance_cholesky", cholesky_count),
            ("factor_precision_solves_A", solve_A_count),
            ("factor_J_assembly_matmuls", J_matmul_count),
            ("factor_J_sum_reduction", J_sum),
            ("posterior_precision_symmetrization", precision_symmetrization),
        ),
        _operation_table(
            ("factor_covariance_cholesky", cholesky_count),
            ("factor_precision_solves_b", solve_b_count),
            ("factor_h_assembly_matmuls", h_matmul_count),
            ("factor_h_sum_reduction", h_sum),
        ),
        _operation_table(
            ("factor_covariance_cholesky", cholesky_count),
            ("factor_precision_solves_b", solve_b_count),
            ("factor_c_quadratics", quadratic_count),
            ("factor_c_logdet_reductions", logdet_count),
            ("factor_c_scalar_combinations", 4 * factor_count),
            ("factor_c_sum_reduction", factor_count),
        ),
        _operation_table(
            ("factor_covariance_cholesky", cholesky_count),
            ("factor_precision_solves_A", solve_A_count),
            ("factor_precision_solves_b", solve_b_count),
            ("factor_J_assembly_matmuls", J_matmul_count),
            ("factor_h_assembly_matmuls", h_matmul_count),
            ("factor_J_sum_reduction", J_sum),
            ("factor_h_sum_reduction", h_sum),
            ("posterior_precision_symmetrization", precision_symmetrization),
            ("posterior_precision_cholesky", posterior_cholesky),
            ("posterior_natural_solve", posterior_natural_solve),
            ("selected_extract", 0),
        ),
        _operation_table(
            ("factor_covariance_cholesky", cholesky_count),
            ("factor_precision_solves_A", solve_A_count),
            ("factor_J_assembly_matmuls", J_matmul_count),
            ("factor_J_sum_reduction", J_sum),
            ("posterior_precision_symmetrization", precision_symmetrization),
            ("posterior_precision_cholesky", posterior_cholesky),
            ("posterior_covariance_solves", 2 * triangular_solve_operation_count(dimension, dimension)),
            ("selected_extract", 0),
        ),
    )


def _operand_group(
    label: str,
    values: object,
    *,
    value_norm: float,
    absolute_summands: object,
    conditions: tuple[float, ...],
    operations: tuple[H4AllowanceOperationCount, ...],
    solver_produced: bool,
) -> _H4AllowanceOperandGroup:
    return _H4AllowanceOperandGroup(
        label, _readonly_vector(values), float(value_norm),
        _readonly_vector(absolute_summands), conditions, operations,
        solver_produced,
    )


def _allowance_group(
    source: H4AllowanceResultSource,
    path: str,
    shape: tuple[int, ...],
    left: _H4AllowanceOperandGroup,
    right: _H4AllowanceOperandGroup,
) -> _H4AllowanceGroupInput:
    return _H4AllowanceGroupInput(
        source.oracle.problem_id, source.oracle.problem_sha256,
        "solver_to_oracle", source.repetition_index, source.result.arm,
        path, shape, left, right,
    )


def _validated_route(
    source: H4AllowanceResultSource,
) -> tuple[dict[str, object], tuple[object, ...]]:
    core, factors, digest = h4_gaussian._parse(source.problem_payload)
    if (
        digest != source.oracle.problem_sha256
        or core["problem_id"] != source.oracle.problem_id
        or core["source_kind"] != source.oracle.source_kind
        or core["seed"] != source.oracle.seed
        or core["kind"] != source.oracle.kind
        or core["horizon"] != source.oracle.horizon
        or core["dimension"] != source.oracle.dimension
        or tuple(item.factor_id for item in factors) != source.oracle.factor_ids
    ):
        raise ValueError("allowance source bytes have the wrong problem digest")
    dimension = source.oracle.dimension
    if (
        len(source.terminal.h) != dimension
        or len(source.terminal.J) != dimension
        or any(len(row) != dimension for row in source.terminal.J)
        or len(source.terminal.mean) != dimension
    ):
        raise ValueError("allowance terminal law has the wrong dimension")
    return core, factors


def _build_result_allowance_group_bundle(
    source: H4AllowanceResultSource,
    oracle: H4OracleEvaluation,
    *,
    observed: bool,
) -> H4ResultAllowanceGroupBundle:
    _, factors = _validated_route(source)
    if oracle.problem_id != source.oracle.problem_id or oracle.problem_sha256 != source.oracle.problem_sha256:
        raise ValueError("independent oracle identity disagrees")
    route = _factor_route_evidence(factors, oracle)
    dimension = oracle.dimension
    arm = source.result.arm
    terminal_h = np.asarray(source.terminal.h, dtype=np.float64)
    terminal_J = np.asarray(source.terminal.J, dtype=np.float64)
    terminal_mean = np.asarray(source.terminal.mean, dtype=np.float64)
    oracle_h = np.asarray(oracle.natural, dtype=np.float64)
    oracle_J = np.asarray(oracle.precision, dtype=np.float64)
    if arm == "information":
        native_information = source.result.native_information
        if (
            type(native_information) is not H4NativeInformationState
            or native_information.h != source.terminal.h
            or native_information.J != source.terminal.J
            or native_information.mean != source.terminal.mean
            or native_information.complete_objective
            != source.terminal.complete_objective
        ):
            raise ValueError("information terminal does not match its native result")
        native_precision_condition = float(
            np.linalg.cond(np.asarray(native_information.J, dtype=np.float64))
        )
        native_covariance_condition = None
    else:
        native_moment = source.result.native_moment
        if (
            type(native_moment) is not H4NativeMomentState
            or native_moment.mean != source.terminal.mean
            or native_moment.complete_objective != source.terminal.complete_objective
        ):
            raise ValueError("moment terminal does not match its native result")
        native_covariance = np.asarray(native_moment.covariance, dtype=np.float64)
        if native_covariance.shape != (dimension, dimension):
            raise ValueError("moment native covariance has the wrong dimension")
        native_covariance_condition = float(np.linalg.cond(native_covariance))
        native_precision_condition = float(np.linalg.cond(terminal_J))

    kl_record = (
        source.kl_to_oracle
        if observed
        else h4_gaussian.reverse_kl_to_h4_oracle(
            oracle, mean=source.terminal.mean, precision=source.terminal.J,
        )
    )
    zero = _operand_group(
        "literal_zero", (0.0,), value_norm=0.0, absolute_summands=(0.0,),
        conditions=(1.0,), operations=(), solver_produced=False,
    )
    kl_left = _operand_group(
        "kl_to_oracle", (kl_record.value,),
        value_norm=abs(kl_record.value),
        absolute_summands=(kl_record.absolute_summand_accumulation,),
        conditions=(
            kl_record.candidate_condition_number,
            kl_record.oracle_condition_number,
        ),
        operations=_operation_table(*kl_record.operation_counts),
        solver_produced=True,
    )
    kl_group = _allowance_group(source, "kl_to_zero", (1,), kl_left, zero)

    if arm == "information":
        h_operations: tuple[H4AllowanceOperationCount, ...] = ()
        h_conditions = (1.0,)
        h_summands = route.h_absolute_summands
        J_operations = _operation_table(
            ("terminal_information_precision_proof_cholesky", cholesky_operation_count(dimension)),
        )
        J_conditions = (native_precision_condition,)
        J_summands = route.J_absolute_summands
    else:
        moment_J_operations = _operation_table(
            ("terminal_moment_covariance_cholesky", cholesky_operation_count(dimension)),
            ("terminal_moment_precision_solves", 2 * triangular_solve_operation_count(dimension, dimension)),
            ("terminal_moment_precision_proof_cholesky", cholesky_operation_count(dimension)),
        )
        J_operations = moment_J_operations
        h_operations = moment_J_operations + _operation_table(
            ("terminal_moment_natural_matmul", matrix_multiply_operation_count(dimension, dimension, 1)),
        )
        assert native_covariance_condition is not None
        J_conditions = (native_covariance_condition, native_precision_condition)
        h_conditions = J_conditions
        J_summands = _readonly_vector(np.abs(terminal_J).reshape(-1))
        h_summands = _readonly_vector(
            np.sum(
                np.abs(terminal_J * terminal_mean[np.newaxis, :]),
                axis=1,
                dtype=np.float64,
            )
        )

    h_left = _operand_group(
        "terminal_h", terminal_h, value_norm=_vector_norm(terminal_h),
        absolute_summands=h_summands, conditions=h_conditions,
        operations=h_operations, solver_produced=True,
    )
    h_right = _operand_group(
        "oracle_natural", oracle_h, value_norm=_vector_norm(oracle_h),
        absolute_summands=route.h_absolute_summands,
        conditions=route.factor_conditions or (1.0,),
        operations=route.h_operations, solver_produced=False,
    )
    J_left = _operand_group(
        "terminal_J", terminal_J, value_norm=_matrix_norm(terminal_J),
        absolute_summands=J_summands, conditions=J_conditions,
        operations=J_operations, solver_produced=True,
    )
    J_right = _operand_group(
        "oracle_precision", oracle_J, value_norm=_matrix_norm(oracle_J),
        absolute_summands=route.J_absolute_summands,
        conditions=route.factor_conditions or (1.0,),
        operations=route.J_operations, solver_produced=False,
    )
    h_group = _allowance_group(source, "terminal_h", (dimension,), h_left, h_right)
    J_group = _allowance_group(source, "terminal_J", (dimension, dimension), J_left, J_right)

    if len(source.terminal.selected_moments) != len(oracle.selected_moments):
        raise ValueError("allowance selected-moment count disagrees")
    selected_groups: list[_H4AllowanceGroupInput] = []
    for terminal_selected, oracle_selected in zip(
        source.terminal.selected_moments, oracle.selected_moments, strict=True,
    ):
        if terminal_selected.name != oracle_selected.name:
            raise ValueError("allowance selected-moment order disagrees")
        size = len(oracle_selected.coordinate_indices)
        terminal_selected_mean = np.asarray(terminal_selected.mean, dtype=np.float64)
        terminal_selected_covariance = np.asarray(terminal_selected.covariance, dtype=np.float64)
        oracle_selected_mean = np.asarray(oracle_selected.mean, dtype=np.float64)
        oracle_selected_covariance = np.asarray(oracle_selected.covariance, dtype=np.float64)
        selected_mean_operations = _operation_table(("selected_extract", 0))
        if arm == "information":
            selected_covariance_operations = _operation_table(
                ("selected_information_precision_cholesky", cholesky_operation_count(dimension)),
                ("selected_information_covariance_solves", 2 * triangular_solve_operation_count(dimension, size)),
                ("selected_extract", 0),
                ("selected_covariance_proof_cholesky", cholesky_operation_count(size)),
            )
            selected_covariance_conditions = (native_precision_condition,)
        else:
            selected_covariance_operations = _operation_table(
                ("selected_extract", 0),
                ("selected_covariance_proof_cholesky", cholesky_operation_count(size)),
            )
            selected_covariance_conditions = (1.0,)
        mean_left = _operand_group(
            f"terminal_selected_mean:{terminal_selected.name}", terminal_selected_mean,
            value_norm=_vector_norm(terminal_selected_mean),
            absolute_summands=np.abs(terminal_selected_mean), conditions=(1.0,),
            operations=selected_mean_operations, solver_produced=True,
        )
        mean_right = _operand_group(
            f"oracle_selected_mean:{oracle_selected.name}", oracle_selected_mean,
            value_norm=_vector_norm(oracle_selected_mean),
            absolute_summands=np.abs(oracle_selected_mean),
            conditions=(*route.factor_conditions, route.posterior_condition),
            operations=route.selected_mean_operations, solver_produced=False,
        )
        covariance_left = _operand_group(
            f"terminal_selected_covariance:{terminal_selected.name}",
            terminal_selected_covariance,
            value_norm=_matrix_norm(terminal_selected_covariance),
            absolute_summands=np.abs(terminal_selected_covariance).reshape(-1),
            conditions=selected_covariance_conditions,
            operations=selected_covariance_operations, solver_produced=True,
        )
        covariance_right = _operand_group(
            f"oracle_selected_covariance:{oracle_selected.name}",
            oracle_selected_covariance,
            value_norm=_matrix_norm(oracle_selected_covariance),
            absolute_summands=np.abs(oracle_selected_covariance).reshape(-1),
            conditions=(*route.factor_conditions, route.posterior_condition),
            operations=route.selected_covariance_base_operations + _operation_table(
                ("selected_covariance_proof_cholesky", cholesky_operation_count(size)),
            ),
            solver_produced=False,
        )
        prefix = f"selected_moments.{terminal_selected.name}"
        selected_groups.extend((
            _allowance_group(source, f"{prefix}.mean", (size,), mean_left, mean_right),
            _allowance_group(
                source, f"{prefix}.covariance", (size, size),
                covariance_left, covariance_right,
            ),
        ))

    canonical_operand = oracle.route_agreement.canonical_operand
    objective_left = _operand_group(
        "terminal_complete_objective", (source.terminal.complete_objective,),
        value_norm=abs(source.terminal.complete_objective),
        absolute_summands=(abs(source.terminal.complete_objective),),
        conditions=(1.0,), operations=(), solver_produced=True,
    )
    objective_right = _operand_group(
        "oracle_canonical_log_normalizer", (oracle.canonical_log_normalizer,),
        value_norm=abs(oracle.canonical_log_normalizer),
        absolute_summands=(canonical_operand.absolute_summand_accumulation,),
        conditions=canonical_operand.condition_numbers,
        operations=_operation_table(*canonical_operand.operation_counts),
        solver_produced=False,
    )
    objective_group = _allowance_group(
        source, "complete_objective", (1,), objective_left, objective_right,
    )
    return H4ResultAllowanceGroupBundle(
        kl_group, h_group, J_group, tuple(selected_groups), objective_group,
    )


def h4_result_allowance_group_bundle(
    *, source: H4AllowanceResultSource,
) -> H4ResultAllowanceGroupBundle:
    if type(source) is not H4AllowanceResultSource:
        raise ValueError("result allowance producer requires an exact source")
    return _build_result_allowance_group_bundle(
        source, source.oracle, observed=True,
    )


def _expected_result_allowance_group_bundle(
    source: H4AllowanceResultSource,
) -> H4ResultAllowanceGroupBundle:
    recomputed_oracle = h4_gaussian.evaluate_h4_oracle(source.problem_payload)
    return _build_result_allowance_group_bundle(
        source, recomputed_oracle, observed=False,
    )


def _bundle_groups(
    bundle: H4ResultAllowanceGroupBundle,
) -> tuple[_H4AllowanceGroupInput, ...]:
    return (
        bundle.kl_to_zero,
        bundle.terminal_h,
        bundle.terminal_J,
        *bundle.selected_mean_and_covariance,
        bundle.complete_objective,
    )


def _adapter_group(
    *,
    oracle: H4OracleEvaluation,
    comparison_source: Literal["adapter_to_h3_reference", "adapter_to_oracle"],
    path: str,
    shape: tuple[int, ...],
    left: _H4AllowanceOperandGroup,
    right: _H4AllowanceOperandGroup,
) -> _H4AllowanceGroupInput:
    return _H4AllowanceGroupInput(
        oracle.problem_id, oracle.problem_sha256, comparison_source, None, None,
        path, shape, left, right,
    )


def _build_anchor_identity_groups(
    source: H4AnchorAllowanceSource,
    *,
    expected: bool,
) -> tuple[_H4AllowanceGroupInput, ...]:
    expected_fixture_id = (
        "h3-coupled-v1"
        if source.information.oracle.kind == "coupled"
        else "h3-zero-control-v1"
    )
    fixture = parse_h3_fixture_bytes(
        source.h3_fixture_bytes, expected_fixture_id=expected_fixture_id,
    )
    oracle = (
        h4_gaussian.evaluate_h4_oracle(source.information.problem_payload)
        if expected else source.information.oracle
    )
    if (
        oracle.problem_id != source.information.oracle.problem_id
        or oracle.problem_sha256 != source.information.oracle.problem_sha256
        or oracle.kind != fixture.kind
    ):
        raise ValueError("H3 fixture and anchor problem identity disagree")
    information_bundle = _build_result_allowance_group_bundle(
        source.information,
        oracle if expected else source.information.oracle,
        observed=not expected,
    )
    moment_oracle = (
        h4_gaussian.evaluate_h4_oracle(source.moment.problem_payload)
        if expected else source.moment.oracle
    )
    moment_bundle = _build_result_allowance_group_bundle(
        source.moment,
        moment_oracle if expected else source.moment.oracle,
        observed=not expected,
    )
    _, factors, _ = h4_gaussian._parse(source.information.problem_payload)
    route = _factor_route_evidence(factors, oracle)
    precision = np.asarray(oracle.precision, dtype=np.float64)
    natural = np.asarray(oracle.natural, dtype=np.float64)
    adapter_J = _operand_group(
        "adapter_precision", precision, value_norm=_matrix_norm(precision),
        absolute_summands=route.J_absolute_summands,
        conditions=route.factor_conditions or (1.0,), operations=route.J_operations,
        solver_produced=False,
    )
    adapter_h = _operand_group(
        "adapter_natural", natural, value_norm=_vector_norm(natural),
        absolute_summands=route.h_absolute_summands,
        conditions=route.factor_conditions or (1.0,), operations=route.h_operations,
        solver_produced=False,
    )
    adapter_c = _operand_group(
        "adapter_constant", (oracle.constant,), value_norm=abs(oracle.constant),
        absolute_summands=route.c_absolute_summands,
        conditions=route.factor_conditions or (1.0,), operations=route.c_operations,
        solver_produced=False,
    )
    canonical_operand = oracle.route_agreement.canonical_operand
    adapter_logZ = _operand_group(
        "adapter_logZ", (oracle.canonical_log_normalizer,),
        value_norm=abs(oracle.canonical_log_normalizer),
        absolute_summands=(canonical_operand.absolute_summand_accumulation,),
        conditions=canonical_operand.condition_numbers,
        operations=_operation_table(*canonical_operand.operation_counts),
        solver_produced=False,
    )
    if fixture.kind == "coupled":
        if fixture.reference_posterior_natural is None or fixture.reference_log_evidence is None:
            raise ValueError("coupled H3 fixture lacks frozen reference evidence")
        reference_J_values = np.asarray(
            fixture.reference_posterior_precision, dtype=np.float64,
        )
        reference_h_values = np.asarray(
            fixture.reference_posterior_natural, dtype=np.float64,
        )
        reference_J = _operand_group(
            "h3_reference_precision", reference_J_values,
            value_norm=_matrix_norm(reference_J_values),
            absolute_summands=np.abs(reference_J_values).reshape(-1),
            conditions=(1.0,), operations=(), solver_produced=False,
        )
        reference_h = _operand_group(
            "h3_reference_natural", reference_h_values,
            value_norm=_vector_norm(reference_h_values),
            absolute_summands=np.abs(reference_h_values), conditions=(1.0,),
            operations=(), solver_produced=False,
        )
        reference_c = _operand_group(
            "h3_reference_constant", (oracle.constant,),
            value_norm=abs(oracle.constant),
            absolute_summands=(abs(oracle.constant),), conditions=(1.0,),
            operations=(), solver_produced=False,
        )
        reference_logZ = _operand_group(
            "h3_reference_logZ", (fixture.reference_log_evidence,),
            value_norm=abs(fixture.reference_log_evidence),
            absolute_summands=(abs(fixture.reference_log_evidence),),
            conditions=(1.0,), operations=(), solver_produced=False,
        )
        adapter_groups = (
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_h3_reference",
                path="adapter.J", shape=(oracle.dimension, oracle.dimension),
                left=adapter_J, right=reference_J,
            ),
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_h3_reference",
                path="adapter.h", shape=(oracle.dimension,), left=adapter_h,
                right=reference_h,
            ),
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_h3_reference",
                path="adapter.c", shape=(1,), left=adapter_c, right=reference_c,
            ),
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_h3_reference",
                path="adapter.logZ", shape=(1,), left=adapter_logZ,
                right=reference_logZ,
            ),
        )
    else:
        independent = h4_gaussian.evaluate_h4_oracle(
            source.information.problem_payload,
        )
        independent_route = _factor_route_evidence(factors, independent)
        oracle_c = _operand_group(
            "oracle_constant", (independent.constant,),
            value_norm=abs(independent.constant),
            absolute_summands=independent_route.c_absolute_summands,
            conditions=independent_route.factor_conditions or (1.0,),
            operations=independent_route.c_operations, solver_produced=False,
        )
        independent_operand = independent.route_agreement.canonical_operand
        oracle_logZ = _operand_group(
            "oracle_logZ", (independent.canonical_log_normalizer,),
            value_norm=abs(independent.canonical_log_normalizer),
            absolute_summands=(independent_operand.absolute_summand_accumulation,),
            conditions=independent_operand.condition_numbers,
            operations=_operation_table(*independent_operand.operation_counts),
            solver_produced=False,
        )
        adapter_groups = (
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_oracle",
                path="adapter.c", shape=(1,), left=adapter_c, right=oracle_c,
            ),
            _adapter_group(
                oracle=oracle, comparison_source="adapter_to_oracle",
                path="adapter.logZ", shape=(1,), left=adapter_logZ,
                right=oracle_logZ,
            ),
        )
    return (
        *_bundle_groups(information_bundle),
        *_bundle_groups(moment_bundle),
        *adapter_groups,
    )


def h4_anchor_identity_groups(
    *, source: H4AnchorAllowanceSource,
) -> tuple[_H4AllowanceGroupInput, ...]:
    if type(source) is not H4AnchorAllowanceSource:
        raise ValueError("anchor allowance producer requires an exact source")
    return _build_anchor_identity_groups(source, expected=False)


def _expected_anchor_identity_groups(
    source: H4AnchorAllowanceSource,
) -> tuple[_H4AllowanceGroupInput, ...]:
    return _build_anchor_identity_groups(source, expected=True)


@dataclass(frozen=True, slots=True)
class _OperandWitness:
    label: str
    value: float
    value_norm: float
    absolute_summand_accumulation: float
    condition_numbers: tuple[float, ...]
    operation_counts: tuple[H4AllowanceOperationCount, ...]
    solver_produced: bool


@dataclass(frozen=True, slots=True)
class _WitnessCandidate:
    stream_index: int
    problem_id: str
    comparison_source: Literal[
        "solver_to_oracle", "adapter_to_h3_reference", "adapter_to_oracle"
    ]
    repetition_index: int | None
    arm: H4SolverArm | None
    path: str
    shape: tuple[int, ...]
    flat_index: int
    left: _OperandWitness
    right: _OperandWitness


def _freeze_witness_candidate(
    stream_index: int,
    group: _H4AllowanceGroupInput,
    flat_index: int,
) -> _WitnessCandidate:
    def operand(group_operand: _H4AllowanceOperandGroup) -> _OperandWitness:
        return _OperandWitness(
            group_operand.label,
            float(group_operand.values[flat_index]),
            group_operand.value_norm,
            float(group_operand.absolute_summand_accumulations[flat_index]),
            group_operand.condition_numbers,
            group_operand.operation_counts,
            group_operand.solver_produced,
        )

    coordinates = np.unravel_index(flat_index, group.shape, order="C")
    path = group.path_prefix + "[" + ",".join(
        str(item) for item in coordinates
    ) + "]"
    return _WitnessCandidate(
        stream_index, group.problem_id, group.comparison_source,
        group.repetition_index, group.arm, path, group.shape, flat_index,
        operand(group.left), operand(group.right),
    )


class _IncrementalAllowanceState:
    __slots__ = (
        "invariant", "expected_element_count", "digest", "observed",
        "maximum_normalized", "maximum_ratio",
        "maximum_normalized_candidate", "maximum_ratio_candidate",
        "first_failed_candidate", "first_indecisive_candidate", "closed",
    )

    def __init__(self, invariant: H4AllowanceInvariantName) -> None:
        self.invariant = invariant
        self.expected_element_count = dict(H4_ALLOWANCE_ELEMENT_COUNTS)[invariant]
        self.digest = hashlib.sha256(b"vfe4.h4.allowance-element-stream.v1\x00")
        self.observed = 0
        self.maximum_normalized = -1.0
        self.maximum_ratio = -1.0
        self.maximum_normalized_candidate: _WitnessCandidate | None = None
        self.maximum_ratio_candidate: _WitnessCandidate | None = None
        self.first_failed_candidate: _WitnessCandidate | None = None
        self.first_indecisive_candidate: _WitnessCandidate | None = None
        self.closed = False

    def consume(
        self,
        expected_group: _H4AllowanceGroupInput,
        observed_group: _H4AllowanceGroupInput,
    ) -> None:
        if self.closed:
            raise ValueError("allowance accumulator is already closed")
        if type(expected_group) is not _H4AllowanceGroupInput or type(observed_group) is not _H4AllowanceGroupInput:
            raise ValueError("allowance accumulator requires exact groups")
        expected_header = allowance_group_header(self.invariant, expected_group)
        actual_header = allowance_group_header(self.invariant, observed_group)
        if actual_header != expected_header:
            raise ValueError("independent numeric allowance group header mismatch")
        self.digest.update(len(actual_header).to_bytes(8, "big"))
        self.digest.update(actual_header)
        left_count = sum(item.count for item in observed_group.left.operation_counts)
        right_count = sum(item.count for item in observed_group.right.operation_counts)
        left_factor = H4_ROUNDING_CONSTANT * gamma_n(left_count) * max(
            (1.0, *observed_group.left.condition_numbers)
        )
        right_factor = H4_ROUNDING_CONSTANT * gamma_n(right_count) * max(
            (1.0, *observed_group.right.condition_numbers)
        )
        element_count = math.prod(observed_group.shape)
        for chunk_start in range(0, element_count, H4_MAXIMUM_CHUNK_ROWS):
            chunk_end = min(element_count, chunk_start + H4_MAXIMUM_CHUNK_ROWS)
            left_values = observed_group.left.values[chunk_start:chunk_end]
            right_values = observed_group.right.values[chunk_start:chunk_end]
            left_sums = observed_group.left.absolute_summand_accumulations[chunk_start:chunk_end]
            right_sums = observed_group.right.absolute_summand_accumulations[chunk_start:chunk_end]
            left_abs = np.abs(left_values)
            right_abs = np.abs(right_values)
            scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(scale, left_abs, out=scale)
            np.maximum(scale, observed_group.left.value_norm, out=scale)
            np.maximum(scale, right_abs, out=scale)
            np.maximum(scale, observed_group.right.value_norm, out=scale)
            left_round_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(left_round_scale, observed_group.left.value_norm, out=left_round_scale)
            np.maximum(left_round_scale, left_sums, out=left_round_scale)
            left_rounding = np.multiply(left_round_scale, left_factor)
            right_round_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(right_round_scale, observed_group.right.value_norm, out=right_round_scale)
            np.maximum(right_round_scale, right_sums, out=right_round_scale)
            right_rounding = np.multiply(right_round_scale, right_factor)
            left_solver = np.multiply(
                scale,
                H4_SOLVER_RELATIVE_BUDGET if observed_group.left.solver_produced else 0.0,
            )
            right_solver = np.multiply(
                scale,
                H4_SOLVER_RELATIVE_BUDGET if observed_group.right.solver_produced else 0.0,
            )
            comparison_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(comparison_scale, left_abs, out=comparison_scale)
            np.maximum(comparison_scale, right_abs, out=comparison_scale)
            absolute_pair_sum = np.add(left_abs, right_abs)
            np.maximum(comparison_scale, absolute_pair_sum, out=comparison_scale)
            comparison = np.multiply(
                comparison_scale, H4_ROUNDING_CONSTANT * gamma_n(3),
            )
            residual = np.abs(np.subtract(left_values, right_values))
            final = np.add(left_rounding, left_solver)
            np.add(final, right_rounding, out=final)
            np.add(final, right_solver, out=final)
            np.add(final, comparison, out=final)
            normalized = np.divide(residual, final)
            ratios = np.divide(final, scale)
            decisive = np.less(ratios, H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION)
            passed = np.less_equal(residual, final)
            if not all(
                np.isfinite(array).all()
                for array in (
                    scale, left_rounding, right_rounding, comparison, residual,
                    normalized, final, ratios,
                )
            ):
                raise ValueError("allowance arithmetic produced a nonfinite lane")
            scratch = np.empty(chunk_end - chunk_start, dtype=_PACKED_ROW_DTYPE)
            scratch["left_value"] = left_values
            scratch["right_value"] = right_values
            scratch["left_value_norm"] = observed_group.left.value_norm
            scratch["right_value_norm"] = observed_group.right.value_norm
            scratch["left_absolute_sum"] = left_sums
            scratch["right_absolute_sum"] = right_sums
            scratch["left_rounding"] = left_rounding
            scratch["left_solver"] = left_solver
            scratch["right_rounding"] = right_rounding
            scratch["right_solver"] = right_solver
            scratch["comparison_allowance"] = comparison
            scratch["residual"] = residual
            scratch["normalized_residual"] = normalized
            scratch["final_allowance"] = final
            scratch["allowance_scale_ratio"] = ratios
            scratch["decisive"] = decisive
            scratch["passed"] = passed
            self.digest.update(scratch.tobytes(order="C"))
            local_normalized = int(np.argmax(normalized))
            normalized_value = float(normalized[local_normalized])
            if normalized_value > self.maximum_normalized:
                self.maximum_normalized = normalized_value
                self.maximum_normalized_candidate = _freeze_witness_candidate(
                    self.observed + chunk_start + local_normalized,
                    observed_group, chunk_start + local_normalized,
                )
            local_ratio = int(np.argmax(ratios))
            ratio_value = float(ratios[local_ratio])
            if ratio_value > self.maximum_ratio:
                self.maximum_ratio = ratio_value
                self.maximum_ratio_candidate = _freeze_witness_candidate(
                    self.observed + chunk_start + local_ratio,
                    observed_group, chunk_start + local_ratio,
                )
            if self.first_failed_candidate is None:
                failed = np.flatnonzero(np.logical_not(passed))
                if failed.size:
                    local = int(failed[0])
                    self.first_failed_candidate = _freeze_witness_candidate(
                        self.observed + chunk_start + local,
                        observed_group, chunk_start + local,
                    )
            if self.first_indecisive_candidate is None:
                indecisive = np.flatnonzero(np.logical_not(decisive))
                if indecisive.size:
                    local = int(indecisive[0])
                    self.first_indecisive_candidate = _freeze_witness_candidate(
                        self.observed + chunk_start + local,
                        observed_group, chunk_start + local,
                    )
            del scratch
        self.observed += element_count
        if self.observed > self.expected_element_count:
            raise ValueError("allowance scalar stream has extra elements")

    def finalize(self) -> H4ApplicableAllowance:
        if self.closed:
            raise ValueError("allowance accumulator finalize is one-shot")
        self.closed = True
        if self.observed != self.expected_element_count:
            raise ValueError("allowance scalar stream count is incomplete")
        if self.maximum_normalized_candidate is None or self.maximum_ratio_candidate is None:
            raise ValueError("applicable allowance stream must be nonempty")
        candidates = (
            self.maximum_normalized_candidate, self.maximum_ratio_candidate,
            self.first_failed_candidate, self.first_indecisive_candidate,
        )
        materialized: dict[int, H4AllowanceElement] = {}
        for candidate in candidates:
            if candidate is not None and candidate.stream_index not in materialized:
                materialized[candidate.stream_index] = _materialize_candidate(
                    self.invariant, candidate,
                )
        maximum_normalized_element = materialized[
            self.maximum_normalized_candidate.stream_index
        ]
        maximum_ratio_element = materialized[self.maximum_ratio_candidate.stream_index]
        first_failed = (
            None if self.first_failed_candidate is None
            else materialized[self.first_failed_candidate.stream_index]
        )
        first_indecisive = (
            None if self.first_indecisive_candidate is None
            else materialized[self.first_indecisive_candidate.stream_index]
        )
        return H4ApplicableAllowance(
            True, self.invariant, H4_ALLOWANCE_STREAM_DOMAIN,
            self.expected_element_count, self.observed, self.digest.hexdigest(),
            self.maximum_normalized, maximum_normalized_element,
            self.maximum_ratio, maximum_ratio_element, first_failed,
            first_indecisive, first_indecisive is None, first_failed is None,
        )


def aggregate_allowance_groups(
    invariant: H4AllowanceInvariantName,
    *,
    expected_element_count: int,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance:
    frozen_count = dict(H4_ALLOWANCE_ELEMENT_COUNTS).get(invariant)
    if type(expected_element_count) is not int or expected_element_count != frozen_count:
        raise ValueError("allowance expected element count is frozen by invariant")
    if isinstance(expected_group_headers, (bytes, bytearray)) or isinstance(groups, (list, tuple)) and False:
        raise ValueError("allowance streams must be iterables of groups, not scalar bytes")
    expected_iterator = iter(expected_group_headers)
    digest = hashlib.sha256(b"vfe4.h4.allowance-element-stream.v1\x00")
    observed = 0
    seen_paths: set[tuple[str, int | None, str | None, str]] = set()
    maximum_normalized = -1.0
    maximum_ratio = -1.0
    maximum_normalized_candidate: _WitnessCandidate | None = None
    maximum_ratio_candidate: _WitnessCandidate | None = None
    first_failed_candidate: _WitnessCandidate | None = None
    first_indecisive_candidate: _WitnessCandidate | None = None

    for group in groups:
        if type(group) is not _H4AllowanceGroupInput:
            raise ValueError("allowance group stream requires exact group records")
        try:
            expected_header = next(expected_iterator)
        except StopIteration as error:
            raise ValueError("observed an extra allowance group") from error
        if type(expected_header) is not bytes:
            raise ValueError("expected allowance group headers must be bytes")
        actual_header = allowance_group_header(invariant, group)
        if actual_header != expected_header:
            raise ValueError("allowance group header/order mismatch")
        path_key = (group.problem_id, group.repetition_index, group.arm, group.path_prefix)
        if path_key in seen_paths:
            raise ValueError("duplicate allowance group path")
        seen_paths.add(path_key)
        digest.update(len(actual_header).to_bytes(8, "big"))
        digest.update(actual_header)

        left_count = sum(item.count for item in group.left.operation_counts)
        right_count = sum(item.count for item in group.right.operation_counts)
        left_factor = H4_ROUNDING_CONSTANT * gamma_n(left_count) * max((1.0, *group.left.condition_numbers))
        right_factor = H4_ROUNDING_CONSTANT * gamma_n(right_count) * max((1.0, *group.right.condition_numbers))
        element_count = math.prod(group.shape)
        for chunk_start in range(0, element_count, H4_MAXIMUM_CHUNK_ROWS):
            chunk_end = min(element_count, chunk_start + H4_MAXIMUM_CHUNK_ROWS)
            left_values = group.left.values[chunk_start:chunk_end]
            right_values = group.right.values[chunk_start:chunk_end]
            left_sums = group.left.absolute_summand_accumulations[chunk_start:chunk_end]
            right_sums = group.right.absolute_summand_accumulations[chunk_start:chunk_end]
            left_abs = np.abs(left_values)
            right_abs = np.abs(right_values)
            scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(scale, left_abs, out=scale)
            np.maximum(scale, group.left.value_norm, out=scale)
            np.maximum(scale, right_abs, out=scale)
            np.maximum(scale, group.right.value_norm, out=scale)

            left_round_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(left_round_scale, group.left.value_norm, out=left_round_scale)
            np.maximum(left_round_scale, left_sums, out=left_round_scale)
            left_rounding = np.multiply(left_round_scale, left_factor)
            right_round_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(right_round_scale, group.right.value_norm, out=right_round_scale)
            np.maximum(right_round_scale, right_sums, out=right_round_scale)
            right_rounding = np.multiply(right_round_scale, right_factor)
            left_solver = np.multiply(scale, H4_SOLVER_RELATIVE_BUDGET if group.left.solver_produced else 0.0)
            right_solver = np.multiply(scale, H4_SOLVER_RELATIVE_BUDGET if group.right.solver_produced else 0.0)

            comparison_scale = np.ones(left_values.shape, dtype=np.float64)
            np.maximum(comparison_scale, left_abs, out=comparison_scale)
            np.maximum(comparison_scale, right_abs, out=comparison_scale)
            absolute_pair_sum = np.add(left_abs, right_abs)
            np.maximum(comparison_scale, absolute_pair_sum, out=comparison_scale)
            comparison = np.multiply(comparison_scale, H4_ROUNDING_CONSTANT * gamma_n(3))
            residual = np.abs(np.subtract(left_values, right_values))
            final = np.add(left_rounding, left_solver)
            np.add(final, right_rounding, out=final)
            np.add(final, right_solver, out=final)
            np.add(final, comparison, out=final)
            normalized = np.divide(residual, final)
            ratios = np.divide(final, scale)
            decisive = np.less(ratios, H4_MAXIMUM_ALLOWANCE_SCALE_FRACTION)
            passed = np.less_equal(residual, final)
            if not all(np.isfinite(array).all() for array in (scale, left_rounding, right_rounding, comparison, residual, normalized, final, ratios)):
                raise ValueError("allowance arithmetic produced a nonfinite lane")

            scratch = np.empty(chunk_end - chunk_start, dtype=_PACKED_ROW_DTYPE)
            scratch["left_value"] = left_values
            scratch["right_value"] = right_values
            scratch["left_value_norm"] = group.left.value_norm
            scratch["right_value_norm"] = group.right.value_norm
            scratch["left_absolute_sum"] = left_sums
            scratch["right_absolute_sum"] = right_sums
            scratch["left_rounding"] = left_rounding
            scratch["left_solver"] = left_solver
            scratch["right_rounding"] = right_rounding
            scratch["right_solver"] = right_solver
            scratch["comparison_allowance"] = comparison
            scratch["residual"] = residual
            scratch["normalized_residual"] = normalized
            scratch["final_allowance"] = final
            scratch["allowance_scale_ratio"] = ratios
            scratch["decisive"] = decisive
            scratch["passed"] = passed
            digest.update(scratch.tobytes(order="C"))

            local_max_normalized = int(np.argmax(normalized))
            value_normalized = float(normalized[local_max_normalized])
            if value_normalized > maximum_normalized:
                maximum_normalized = value_normalized
                maximum_normalized_candidate = _freeze_witness_candidate(
                    observed + chunk_start + local_max_normalized, group,
                    chunk_start + local_max_normalized,
                )
            local_max_ratio = int(np.argmax(ratios))
            value_ratio = float(ratios[local_max_ratio])
            if value_ratio > maximum_ratio:
                maximum_ratio = value_ratio
                maximum_ratio_candidate = _freeze_witness_candidate(
                    observed + chunk_start + local_max_ratio, group,
                    chunk_start + local_max_ratio,
                )
            if first_failed_candidate is None:
                failed = np.flatnonzero(np.logical_not(passed))
                if failed.size:
                    local = int(failed[0])
                    first_failed_candidate = _freeze_witness_candidate(
                        observed + chunk_start + local, group, chunk_start + local,
                    )
            if first_indecisive_candidate is None:
                indecisive = np.flatnonzero(np.logical_not(decisive))
                if indecisive.size:
                    local = int(indecisive[0])
                    first_indecisive_candidate = _freeze_witness_candidate(
                        observed + chunk_start + local, group, chunk_start + local,
                    )
            del scratch
        observed += element_count

    try:
        next(expected_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("missing an expected allowance group")
    if observed != expected_element_count:
        raise ValueError("allowance scalar stream count is incomplete")
    if maximum_normalized_candidate is None or maximum_ratio_candidate is None:
        raise ValueError("applicable allowance stream must be nonempty")

    candidates = (
        maximum_normalized_candidate, maximum_ratio_candidate,
        first_failed_candidate, first_indecisive_candidate,
    )
    materialized: dict[int, H4AllowanceElement] = {}
    for candidate in candidates:
        if candidate is not None and candidate.stream_index not in materialized:
            materialized[candidate.stream_index] = _materialize_candidate(invariant, candidate)
    max_normalized_element = materialized[maximum_normalized_candidate.stream_index]
    max_ratio_element = materialized[maximum_ratio_candidate.stream_index]
    first_failed = None if first_failed_candidate is None else materialized[first_failed_candidate.stream_index]
    first_indecisive = None if first_indecisive_candidate is None else materialized[first_indecisive_candidate.stream_index]
    return H4ApplicableAllowance(
        True, invariant, H4_ALLOWANCE_STREAM_DOMAIN, expected_element_count, observed,
        digest.hexdigest(), maximum_normalized, max_normalized_element,
        maximum_ratio, max_ratio_element, first_failed, first_indecisive,
        first_indecisive is None, first_failed is None,
    )


def _materialize_candidate(
    invariant: H4AllowanceInvariantName,
    candidate: _WitnessCandidate,
) -> H4AllowanceElement:
    left_witness = candidate.left
    right_witness = candidate.right
    scale = max(
        1.0, abs(left_witness.value), left_witness.value_norm,
        abs(right_witness.value), right_witness.value_norm,
    )
    left = operand_allowance(
        label=left_witness.label, value=left_witness.value,
        value_norm=left_witness.value_norm,
        absolute_summand_accumulation=left_witness.absolute_summand_accumulation,
        condition_numbers=left_witness.condition_numbers,
        operation_counts=left_witness.operation_counts,
        solver_produced=left_witness.solver_produced, invariant_scale=scale,
    )
    right = operand_allowance(
        label=right_witness.label, value=right_witness.value,
        value_norm=right_witness.value_norm,
        absolute_summand_accumulation=right_witness.absolute_summand_accumulation,
        condition_numbers=right_witness.condition_numbers,
        operation_counts=right_witness.operation_counts,
        solver_produced=right_witness.solver_produced, invariant_scale=scale,
    )
    return pair_element_allowance(
        stream_index=candidate.stream_index, invariant=invariant,
        problem_id=candidate.problem_id,
        comparison_source=candidate.comparison_source,
        repetition_index=candidate.repetition_index, arm=candidate.arm,
        path=candidate.path, shape=candidate.shape,
        flat_index=candidate.flat_index, left=left, right=right,
    )


def allowance_is_decisive(record: H4ApplicableAllowance) -> bool:
    if type(record) is not H4ApplicableAllowance:
        raise ValueError("allowance_is_decisive requires an exact applicable record")
    return record.decisive


def _build(
    invariant: H4AllowanceInvariantName,
    *,
    expected_group_headers: Iterable[bytes],
    groups: Iterable[_H4AllowanceGroupInput],
) -> H4ApplicableAllowance:
    return aggregate_allowance_groups(
        invariant, expected_element_count=dict(H4_ALLOWANCE_ELEMENT_COUNTS)[invariant],
        expected_group_headers=expected_group_headers, groups=groups,
    )


def build_h4_anchor_identity_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("h3_anchor_identity", expected_group_headers=expected_group_headers, groups=groups)


def build_h4_exact_posterior_gap_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("exact_posterior_gap_equivalence", expected_group_headers=expected_group_headers, groups=groups)


def build_h4_terminal_h_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("terminal_h_equivalence", expected_group_headers=expected_group_headers, groups=groups)


def build_h4_terminal_j_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("terminal_J_equivalence", expected_group_headers=expected_group_headers, groups=groups)


def build_h4_selected_moment_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("selected_moment_equivalence", expected_group_headers=expected_group_headers, groups=groups)


def build_h4_complete_objective_allowance(*, expected_group_headers: Iterable[bytes], groups: Iterable[_H4AllowanceGroupInput]) -> H4ApplicableAllowance:
    return _build("complete_objective_equivalence", expected_group_headers=expected_group_headers, groups=groups)


_ALLOWANCE_INVARIANT_ORDER: tuple[H4AllowanceInvariantName, ...] = tuple(
    name for name, _ in H4_ALLOWANCE_ELEMENT_COUNTS
)


class H4SixInvariantAllowanceAccumulator:
    __slots__ = ("_states", "_position", "_failed", "_finalized")

    def __init__(self) -> None:
        self._states: dict[H4AllowanceInvariantName, _IncrementalAllowanceState] = {
            name: _IncrementalAllowanceState(name)
            for name in _ALLOWANCE_INVARIANT_ORDER
        }
        self._position = 0
        self._failed = False
        self._finalized = False

    def _consume_group_pairs(
        self,
        invariant: H4AllowanceInvariantName,
        expected_groups: tuple[_H4AllowanceGroupInput, ...],
        observed_groups: tuple[_H4AllowanceGroupInput, ...],
    ) -> None:
        if len(expected_groups) != len(observed_groups):
            raise ValueError("independent allowance producers disagree on group count")
        state = self._states[invariant]
        for expected_group, observed_group in zip(
            expected_groups, observed_groups, strict=True,
        ):
            state.consume(expected_group, observed_group)

    def consume(
        self,
        source: H4AnchorAllowanceSource | H4AllowanceResultSource,
    ) -> None:
        if self._finalized:
            raise ValueError("six-invariant allowance consumption is closed")
        if self._failed:
            raise ValueError("six-invariant allowance accumulator failed closed")
        try:
            if self._position < 2:
                if type(source) is not H4AnchorAllowanceSource:
                    raise ValueError("allowance source order requires both anchors first")
                expected_kind = ("coupled", "zero_control")[self._position]
                if source.information.oracle.kind != expected_kind:
                    raise ValueError("anchor allowance source order is frozen")
                expected_groups = _expected_anchor_identity_groups(source)
                observed_groups = h4_anchor_identity_groups(source=source)
                self._consume_group_pairs(
                    "h3_anchor_identity", expected_groups, observed_groups,
                )
            else:
                if type(source) is not H4AllowanceResultSource:
                    raise ValueError("scaled allowance source order cannot contain an anchor")
                scaled_index = self._position - 2
                if scaled_index >= 120 * 11 * 2:
                    raise ValueError("allowance source stream has extra scaled results")
                problem_index, within_problem = divmod(scaled_index, 22)
                repetition_index, arm_index = divmod(within_problem, 2)
                horizon_index, within_horizon = divmod(problem_index, 40)
                seed_index, kind_index = divmod(within_horizon, 2)
                expected_horizon = (7, 15, 31)[horizon_index]
                expected_seed = H4_PROBLEM_SEEDS[seed_index]
                expected_kind = ("coupled", "zero_control")[kind_index]
                expected_arm = ("information", "moment")[arm_index]
                if source.repetition_index is None:
                    raise ValueError("scaled allowance source requires repetition 0..10")
                if (
                    source.repetition_index != repetition_index
                    or source.result.arm != expected_arm
                    or source.oracle.source_kind != "scaled_pcg64"
                    or source.oracle.horizon != expected_horizon
                    or source.oracle.seed != expected_seed
                    or source.oracle.kind != expected_kind
                ):
                    raise ValueError("scaled allowance problem/repetition/arm order is frozen")
                expected_bundle = _expected_result_allowance_group_bundle(source)
                observed_bundle = h4_result_allowance_group_bundle(source=source)
                self._consume_group_pairs(
                    "exact_posterior_gap_equivalence",
                    (expected_bundle.kl_to_zero,),
                    (observed_bundle.kl_to_zero,),
                )
                self._consume_group_pairs(
                    "terminal_h_equivalence", (expected_bundle.terminal_h,),
                    (observed_bundle.terminal_h,),
                )
                self._consume_group_pairs(
                    "terminal_J_equivalence", (expected_bundle.terminal_J,),
                    (observed_bundle.terminal_J,),
                )
                self._consume_group_pairs(
                    "selected_moment_equivalence",
                    expected_bundle.selected_mean_and_covariance,
                    observed_bundle.selected_mean_and_covariance,
                )
                self._consume_group_pairs(
                    "complete_objective_equivalence",
                    (expected_bundle.complete_objective,),
                    (observed_bundle.complete_objective,),
                )
        except Exception:
            self._failed = True
            raise
        self._position += 1

    def finalize(self) -> tuple[
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
        H4ApplicableAllowance,
    ]:
        if self._finalized:
            raise ValueError("six-invariant allowance finalize is one-shot")
        if self._failed:
            raise ValueError("six-invariant allowance accumulator failed closed")
        if self._position != 2 + 120 * 11 * 2:
            raise ValueError("six-invariant allowance source stream is incomplete")
        self._finalized = True
        records = tuple(self._states[name].finalize() for name in _ALLOWANCE_INVARIANT_ORDER)
        self._states.clear()
        return records  # type: ignore[return-value]


def new_h4_six_invariant_allowance_accumulator(
) -> H4SixInvariantAllowanceAccumulator:
    return H4SixInvariantAllowanceAccumulator()


def posterior_condition_record(
    *,
    problem_id: str,
    problem_sha256: str,
    source: Literal["numpy_oracle", "information", "moment"],
    repetition_index: int | None,
    dimension: int,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    condition_number: float,
    minimum_cholesky_pivot: float,
    mean_infinity_norm: float,
    envelope: H4ConditionEnvelopeConfig,
) -> H4PosteriorConditionRecord:
    if type(envelope) is not H4ConditionEnvelopeConfig:
        raise ValueError("posterior classifier requires the exact resolved envelope")
    values = (
        minimum_eigenvalue, maximum_eigenvalue, condition_number,
        minimum_cholesky_pivot, mean_infinity_norm,
    )
    finite = all(type(value) is float and math.isfinite(value) for value in values)
    spd = finite and minimum_eigenvalue > 0.0 and minimum_cholesky_pivot > 0.0
    if not finite or not spd:
        raise ValueError("posterior raw diagnostic must be finite SPD evidence")
    eligible = (
        minimum_eigenvalue >= envelope.posterior_minimum_eigenvalue
        and maximum_eigenvalue <= envelope.posterior_maximum_eigenvalue
        and condition_number <= envelope.posterior_maximum_condition_number
        and minimum_cholesky_pivot >= envelope.posterior_minimum_cholesky_pivot
        and mean_infinity_norm <= envelope.posterior_maximum_mean_infinity_norm
    )
    return H4PosteriorConditionRecord(
        problem_id, problem_sha256, source, repetition_index, dimension,
        minimum_eigenvalue, maximum_eigenvalue, condition_number,
        minimum_cholesky_pivot, mean_infinity_norm, True, True, eligible,
    )


def innovation_condition_record(
    *,
    problem_id: str,
    problem_sha256: str,
    source: Literal["numpy_oracle", "moment"],
    repetition_index: int | None,
    factor_id: str,
    time_index: int,
    parent_coordinate_indices: tuple[int, ...],
    innovation_dimension: int,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    condition_number: float,
    envelope: H4ConditionEnvelopeConfig,
) -> H4InnovationConditionRecord:
    if type(envelope) is not H4ConditionEnvelopeConfig:
        raise ValueError("innovation classifier requires the exact resolved envelope")
    values = (minimum_eigenvalue, maximum_eigenvalue, condition_number)
    finite = all(type(value) is float and math.isfinite(value) for value in values)
    spd = finite and minimum_eigenvalue > 0.0
    if not finite or not spd:
        raise ValueError("innovation raw diagnostic must be finite SPD evidence")
    eligible = (
        minimum_eigenvalue >= envelope.innovation_minimum_eigenvalue
        and maximum_eigenvalue <= envelope.innovation_maximum_eigenvalue
        and condition_number <= envelope.innovation_maximum_condition_number
    )
    return H4InnovationConditionRecord(
        problem_id, problem_sha256, source, repetition_index, factor_id,
        time_index, parent_coordinate_indices, innovation_dimension,
        minimum_eigenvalue, maximum_eigenvalue, condition_number,
        True, True, eligible,
    )


__all__ = [
    "H4_ALLOWANCE_STREAM_DOMAIN", "H4_EPSILON", "H4_MAXIMUM_CHUNK_ROWS",
    "H4_ROUNDING_CONSTANT", "H4_SOLVER_RELATIVE_BUDGET",
    "H4AllowanceResultSource", "H4AnchorAllowanceSource",
    "H4ResultAllowanceGroupBundle", "H4SixInvariantAllowanceAccumulator",
    "aggregate_allowance_groups", "allowance_group_header", "allowance_is_decisive",
    "build_h4_anchor_identity_allowance", "build_h4_complete_objective_allowance",
    "build_h4_exact_posterior_gap_allowance", "build_h4_selected_moment_allowance",
    "build_h4_terminal_h_allowance", "build_h4_terminal_j_allowance",
    "cholesky_operation_count", "dot_operation_count", "gamma_n",
    "h4_anchor_identity_groups", "h4_result_allowance_group_bundle",
    "innovation_condition_record", "matrix_multiply_operation_count",
    "new_h4_six_invariant_allowance_accumulator",
    "operand_allowance", "pair_element_allowance", "posterior_condition_record",
    "triangular_solve_operation_count",
]
