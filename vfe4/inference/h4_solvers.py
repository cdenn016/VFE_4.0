"""Independent information- and moment-form Gaussian solvers for H4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

from vfe4.inference.h4_instrumentation import (
    InstrumentedLinearAlgebra,
    _facade_binding,
    _facade_uses_null_recorder,
)
from vfe4.types.h4 import (
    H4FactorRole,
    H4NativeInformationState,
    H4NativeMomentState,
    H4NeutralProblem,
    H4ProblemKind,
    H4ProblemSource,
    H4SelectedMoment,
    H4SolveProtocol,
    H4SolverArm,
    H4SolverResult,
    H4TerminalLaw,
    h4_problem_digest,
)

_MATERIALIZATION_VERSION = "h4-materialized-problem-v1"
_MATERIALIZATION_DOMAIN = b"vfe4.h4.materialized-problem.v1\x00"
_ACCESS_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class H4MaterializedProblem:
    materialization_version: Literal["h4-materialized-problem-v1"]
    problem_id: str
    problem_sha256: str
    protocol_id: Literal["h4-single-pass-v1"]
    dtype: Literal["float64"]
    device: Literal["cpu"]
    source_kind: H4ProblemSource
    seed: int
    kind: H4ProblemKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_ids: tuple[str, ...]
    factor_roles: tuple[H4FactorRole, ...]
    factor_time_indices: tuple[int, ...]
    factor_normalized_coordinate_indices: tuple[tuple[int, ...], ...]
    factor_parent_coordinate_indices: tuple[tuple[int, ...], ...]
    _factor_matrices: tuple[Tensor, ...]
    _factor_targets: tuple[Tensor, ...]
    _factor_covariances: tuple[Tensor, ...]
    tensor_sha256: str

    def __post_init__(self) -> None:
        _validate_materialized_structure(self)
        observed = _materialized_digest(self)
        if self.tensor_sha256 != observed:
            raise ValueError("tensor_sha256 does not match the owned raw tensors")


@dataclass(frozen=True, slots=True)
class H4InnovationDiagnostic:
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float

    def __post_init__(self) -> None:
        if type(self.factor_id) is not str or not self.factor_id:
            raise ValueError("factor_id must be a nonempty string")
        if type(self.time_index) is not int or self.time_index < 0:
            raise ValueError("time_index must be nonnegative")
        _strict_indices(
            self.parent_coordinate_indices,
            None,
            "parent_coordinate_indices",
        )
        if type(self.innovation_dimension) is not int or self.innovation_dimension <= 0:
            raise ValueError("innovation_dimension must be positive")
        values = (
            self.minimum_eigenvalue,
            self.maximum_eigenvalue,
            self.condition_number,
            self.minimum_cholesky_pivot,
        )
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
            raise ValueError("innovation diagnostics must be finite")
        if (
            self.minimum_eigenvalue <= 0.0
            or self.maximum_eigenvalue < self.minimum_eigenvalue
            or self.condition_number <= 0.0
            or self.minimum_cholesky_pivot <= 0.0
        ):
            raise ValueError("innovation covariance diagnostics must be positive")
        expected = self.maximum_eigenvalue / self.minimum_eigenvalue
        tolerance = 64.0 * math.ulp(1.0) * max(1.0, abs(expected), abs(self.condition_number))
        if abs(self.condition_number - expected) > tolerance:
            raise ValueError("condition_number is inconsistent with the eigenvalues")


@dataclass(frozen=True, slots=True)
class H4NativeDiagnostics:
    problem_id: str
    problem_sha256: str
    protocol_id: Literal["h4-single-pass-v1"]
    arm: H4SolverArm
    factor_count: int
    replayed_result: H4SolverResult
    innovation_diagnostics: tuple[H4InnovationDiagnostic, ...]
    finite: Literal[True]
    spd: Literal[True]
    replay_matches_result: Literal[True]

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id:
            raise ValueError("problem_id must be a nonempty string")
        _sha256(self.problem_sha256, "problem_sha256")
        if self.protocol_id != "h4-single-pass-v1" or self.arm not in ("information", "moment"):
            raise ValueError("native diagnostic identity is invalid")
        if type(self.factor_count) is not int or self.factor_count <= 0:
            raise ValueError("factor_count must be positive")
        if type(self.replayed_result) is not H4SolverResult:
            raise ValueError("replayed_result must be an exact H4SolverResult")
        if (
            self.replayed_result.problem_id != self.problem_id
            or self.replayed_result.problem_sha256 != self.problem_sha256
            or self.replayed_result.protocol_id != self.protocol_id
            or self.replayed_result.arm != self.arm
            or self.replayed_result.factor_count != self.factor_count
        ):
            raise ValueError("replayed_result identity does not match diagnostics")
        if type(self.innovation_diagnostics) is not tuple or not all(
            type(item) is H4InnovationDiagnostic for item in self.innovation_diagnostics
        ):
            raise ValueError("innovation_diagnostics must be exact immutable records")
        if self.arm == "information" and self.innovation_diagnostics:
            raise ValueError("information diagnostics must not contain innovations")
        if self.finite is not True or self.spd is not True or self.replay_matches_result is not True:
            raise ValueError("native diagnostics are emitted only after successful closure")


class H4GaussianSolver(Protocol):
    def solve(
        self,
        materialized: H4MaterializedProblem,
        protocol: H4SolveProtocol,
        linalg: InstrumentedLinearAlgebra,
    ) -> H4SolverResult: ...


class InformationFormH4Solver:
    def solve(
        self,
        materialized: H4MaterializedProblem,
        protocol: H4SolveProtocol,
        linalg: InstrumentedLinearAlgebra,
    ) -> H4SolverResult:
        return solve_information_form(materialized, protocol, linalg)


class MomentFormH4Solver:
    def solve(
        self,
        materialized: H4MaterializedProblem,
        protocol: H4SolveProtocol,
        linalg: InstrumentedLinearAlgebra,
    ) -> H4SolverResult:
        return solve_moment_form(materialized, protocol, linalg)


@dataclass(frozen=True, slots=True)
class _InnovationReplay:
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    covariance: Tensor
    cholesky: Tensor


def materialize_h4_problem(
    problem: H4NeutralProblem,
    protocol: H4SolveProtocol,
) -> H4MaterializedProblem:
    """Own one raw CPU-float64 tensor triple for every neutral factor."""

    if type(problem) is not H4NeutralProblem:
        raise ValueError("problem must be an exact H4NeutralProblem")
    _validate_protocol(protocol)
    if h4_problem_digest(problem) != problem.canonical_sha256:
        raise ValueError("problem canonical digest is invalid")
    matrices = tuple(_owned_tensor(factor.matrix) for factor in problem.factor_schedule)
    targets = tuple(_owned_tensor(factor.target) for factor in problem.factor_schedule)
    covariances = tuple(_owned_tensor(factor.covariance) for factor in problem.factor_schedule)
    values: dict[str, object] = {
        "materialization_version": _MATERIALIZATION_VERSION,
        "problem_id": problem.problem_id,
        "problem_sha256": problem.canonical_sha256,
        "protocol_id": protocol.protocol_id,
        "dtype": protocol.dtype,
        "device": protocol.device,
        "source_kind": problem.source_kind,
        "seed": problem.seed,
        "kind": problem.kind,
        "horizon": problem.horizon,
        "d_z": problem.d_z,
        "d_m": problem.d_m,
        "dimension": problem.dimension,
        "coordinate_order": problem.coordinate_order,
        "factor_ids": tuple(factor.factor_id for factor in problem.factor_schedule),
        "factor_roles": tuple(factor.role for factor in problem.factor_schedule),
        "factor_time_indices": tuple(factor.time_index for factor in problem.factor_schedule),
        "factor_normalized_coordinate_indices": tuple(
            factor.normalized_coordinate_indices for factor in problem.factor_schedule
        ),
        "factor_parent_coordinate_indices": tuple(
            factor.parent_coordinate_indices for factor in problem.factor_schedule
        ),
        "_factor_matrices": matrices,
        "_factor_targets": targets,
        "_factor_covariances": covariances,
    }
    digest = _materialized_digest_parts(values)
    return H4MaterializedProblem(**values, tensor_sha256=digest)  # type: ignore[arg-type]


def solve_information_form(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
) -> H4SolverResult:
    _validate_solver_context(materialized, protocol, linalg, "information")
    matrices, targets, covariances = _materialized_tensors(
        materialized,
        _ACCESS_CAPABILITY,
    )
    dimension = materialized.dimension
    precision = torch.zeros((dimension, dimension), dtype=torch.float64)
    natural = torch.zeros(dimension, dtype=torch.float64)
    constant = torch.zeros((), dtype=torch.float64)
    log_two_pi = math.log(2.0 * math.pi)
    for matrix, target, covariance in zip(matrices, targets, covariances, strict=True):
        cholesky = linalg.cholesky(covariance)
        whitened_matrix = linalg.triangular_solve(cholesky, matrix)
        whitened_target = linalg.triangular_solve(cholesky, target)
        precision = precision + linalg.matrix_multiply(
            whitened_matrix.T,
            whitened_matrix,
        )
        natural = natural + linalg.matrix_multiply(
            whitened_matrix.T,
            whitened_target,
        )
        constant = constant - 0.5 * (
            torch.sum(whitened_target * whitened_target)
            + target.numel() * log_two_pi
            + 2.0 * torch.sum(torch.log(torch.diagonal(cholesky)))
        )
    precision = _symmetrize(precision)
    precision_cholesky = linalg.cholesky(precision)
    intermediate = linalg.triangular_solve(precision_cholesky, natural)
    mean = linalg.triangular_solve(
        precision_cholesky.T,
        intermediate,
        upper=True,
    )
    complete_objective = (
        constant
        + 0.5 * torch.sum(natural * mean)
        - torch.sum(torch.log(torch.diagonal(precision_cholesky)))
        + dimension / 2.0 * log_two_pi
    )
    _require_finite(
        (precision, natural, mean, complete_objective),
        "information native state",
    )
    native = H4NativeInformationState(
        _vector_tuple(natural),
        _matrix_tuple(precision),
        _vector_tuple(mean),
        float(complete_objective.item()),
    )
    return H4SolverResult(
        materialized.problem_id,
        materialized.problem_sha256,
        "information",
        materialized.protocol_id,
        len(materialized.factor_ids),
        native,
        None,
    )


def solve_moment_form(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
) -> H4SolverResult:
    result, _ = _solve_moment_native(materialized, protocol, linalg, capture=False)
    return result


def to_common_terminal_law(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4TerminalLaw:
    _validate_external_context(materialized, result, linalg)
    blocks = _selected_blocks(materialized)
    dimension = materialized.dimension
    if result.arm == "information":
        assert result.native_information is not None
        native = result.native_information
        precision = _owned_from_tuple(native.J)
        natural = _owned_from_tuple(native.h)
        mean = _owned_from_tuple(native.mean)
        precision_cholesky = linalg.cholesky(precision)
        selected: list[H4SelectedMoment] = []
        for name, indices in blocks:
            columns = torch.zeros((dimension, len(indices)), dtype=torch.float64)
            column_positions = torch.tensor(indices, dtype=torch.int64)
            local_positions = torch.arange(len(indices), dtype=torch.int64)
            columns[column_positions, local_positions] = 1.0
            intermediate = linalg.triangular_solve(precision_cholesky, columns)
            inverse_columns = linalg.triangular_solve(
                precision_cholesky.T,
                intermediate,
                upper=True,
            )
            covariance_block = linalg.selected_block_extract(
                inverse_columns,
                indices,
                tuple(range(len(indices))),
            )
            mean_block = linalg.selected_block_extract(mean, indices)
            selected.append(
                H4SelectedMoment(
                    name,
                    _vector_tuple(mean_block),
                    _matrix_tuple(_symmetrize(covariance_block)),
                )
            )
        objective = native.complete_objective
    else:
        assert result.native_moment is not None
        native = result.native_moment
        mean = _owned_from_tuple(native.mean)
        covariance = _owned_from_tuple(native.covariance)
        covariance_cholesky = linalg.cholesky(covariance)
        identity = torch.eye(dimension, dtype=torch.float64)
        intermediate = linalg.triangular_solve(covariance_cholesky, identity)
        precision = linalg.triangular_solve(
            covariance_cholesky.T,
            intermediate,
            upper=True,
        )
        precision = _symmetrize(precision)
        natural = linalg.matrix_multiply(precision, mean)
        selected = []
        for name, indices in blocks:
            mean_block = linalg.selected_block_extract(mean, indices)
            covariance_block = linalg.selected_block_extract(
                covariance,
                indices,
                indices,
            )
            selected.append(
                H4SelectedMoment(
                    name,
                    _vector_tuple(mean_block),
                    _matrix_tuple(_symmetrize(covariance_block)),
                )
            )
        objective = native.complete_objective
    product = linalg.matrix_multiply(precision, mean)
    numerator = torch.max(torch.abs(product - natural))
    precision_norm = torch.max(torch.sum(torch.abs(precision), dim=1))
    mean_norm = torch.max(torch.abs(mean))
    natural_norm = torch.max(torch.abs(natural))
    scale = torch.maximum(
        torch.tensor(1.0, dtype=torch.float64),
        precision_norm * mean_norm + natural_norm,
    )
    residual = numerator / scale
    _require_finite((precision, natural, mean, residual), "common terminal law")
    return H4TerminalLaw(
        result.arm,
        _vector_tuple(natural),
        _matrix_tuple(_symmetrize(precision)),
        _vector_tuple(mean),
        tuple(selected),
        objective,
        float(residual.item()),
    )


def evaluate_h4_native_diagnostics(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> H4NativeDiagnostics:
    _validate_external_context(materialized, result, linalg)
    if not _facade_uses_null_recorder(linalg):
        raise ValueError("diagnostic replay requires a null-recorder facade")
    if result.arm == "information":
        replayed = solve_information_form(
            materialized,
            H4SolveProtocol(),
            linalg,
        )
        innovations: tuple[H4InnovationDiagnostic, ...] = ()
    else:
        replayed, captured = _solve_moment_native(
            materialized,
            H4SolveProtocol(),
            linalg,
            capture=True,
        )
        innovation_records: list[H4InnovationDiagnostic] = []
        for item in captured:
            eigenvalues = torch.linalg.eigvalsh(item.covariance)
            minimum = float(torch.min(eigenvalues).item())
            maximum = float(torch.max(eigenvalues).item())
            pivot = float(torch.min(torch.diagonal(item.cholesky)).item())
            innovation_records.append(
                H4InnovationDiagnostic(
                    item.factor_id,
                    item.time_index,
                    item.parent_coordinate_indices,
                    int(item.covariance.shape[0]),
                    minimum,
                    maximum,
                    maximum / minimum,
                    pivot,
                )
            )
        innovations = tuple(innovation_records)
    if replayed != result:
        raise ValueError("diagnostic replay does not exactly match the supplied result")
    return H4NativeDiagnostics(
        materialized.problem_id,
        materialized.problem_sha256,
        materialized.protocol_id,
        result.arm,
        len(materialized.factor_ids),
        replayed,
        innovations,
        True,
        True,
        True,
    )


def _solve_moment_native(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
    *,
    capture: bool,
) -> tuple[H4SolverResult, tuple[_InnovationReplay, ...]]:
    _validate_solver_context(materialized, protocol, linalg, "moment")
    matrices, targets, covariances = _materialized_tensors(
        materialized,
        _ACCESS_CAPABILITY,
    )
    dimension = materialized.dimension
    mean = torch.zeros(dimension, dtype=torch.float64)
    covariance = torch.zeros((dimension, dimension), dtype=torch.float64)
    active_set: set[int] = set()
    objective = torch.zeros((), dtype=torch.float64)
    innovations: list[_InnovationReplay] = []
    left_initial_region = False
    log_two_pi = math.log(2.0 * math.pi)
    for index, (matrix, target, noise) in enumerate(
        zip(matrices, targets, covariances, strict=True)
    ):
        role = materialized.factor_roles[index]
        normalized = materialized.factor_normalized_coordinate_indices[index]
        parents = materialized.factor_parent_coordinate_indices[index]
        if role == "initial":
            if left_initial_region:
                raise ValueError("initial factors must form one consecutive prefix")
            if any(coordinate in active_set for coordinate in normalized):
                raise ValueError("initial factor coordinates overlap")
            active = tuple(sorted(active_set))
            _scatter_vector(mean, normalized, target)
            _scatter_matrix(covariance, normalized, normalized, noise)
            if active:
                zeros = torch.zeros((len(normalized), len(active)), dtype=torch.float64)
                _scatter_matrix(covariance, normalized, active, zeros)
                _scatter_matrix(covariance, active, normalized, zeros.T)
            active_set.update(normalized)
            continue
        left_initial_region = True
        active = tuple(sorted(active_set))
        if role == "transition":
            if any(parent not in active_set for parent in parents):
                raise ValueError("transition parent is not active")
            if any(child in active_set for child in normalized):
                raise ValueError("transition child is already active")
            rows = tuple(range(int(matrix.shape[0])))
            dynamics = -linalg.selected_block_extract(matrix, rows, parents)
            parent_mean = linalg.selected_block_extract(mean, parents)
            parent_to_active = linalg.selected_block_extract(
                covariance,
                parents,
                active,
            )
            child_mean = linalg.matrix_multiply(dynamics, parent_mean) + target
            child_to_active = linalg.matrix_multiply(dynamics, parent_to_active)
            parent_positions = tuple(active.index(parent) for parent in parents)
            child_to_parents = linalg.selected_block_extract(
                child_to_active,
                rows,
                parent_positions,
            )
            child_covariance = _symmetrize(
                linalg.matrix_multiply(child_to_parents, dynamics.T) + noise
            )
            _scatter_vector(mean, normalized, child_mean)
            _scatter_matrix(covariance, normalized, active, child_to_active)
            _scatter_matrix(covariance, active, normalized, child_to_active.T)
            _scatter_matrix(covariance, normalized, normalized, child_covariance)
            active_set.update(normalized)
            continue
        if role != "observation":
            raise ValueError("unknown H4 factor role")
        if not active:
            raise ValueError("observation requires active coordinates")
        rows = tuple(range(int(matrix.shape[0])))
        active_matrix = linalg.selected_block_extract(matrix, rows, active)
        active_mean = linalg.selected_block_extract(mean, active)
        active_covariance = linalg.selected_block_extract(
            covariance,
            active,
            active,
        )
        residual = target - linalg.matrix_multiply(active_matrix, active_mean)
        covariance_times_map = linalg.matrix_multiply(
            active_covariance,
            active_matrix.T,
        )
        innovation_covariance = _symmetrize(
            noise + linalg.matrix_multiply(active_matrix, covariance_times_map)
        )
        innovation_cholesky = linalg.cholesky(innovation_covariance)
        whitened_residual = linalg.triangular_solve(
            innovation_cholesky,
            residual,
        )
        increment = -0.5 * (
            torch.sum(whitened_residual * whitened_residual)
            + target.numel() * log_two_pi
            + 2.0 * torch.sum(torch.log(torch.diagonal(innovation_cholesky)))
        )
        intermediate = linalg.triangular_solve(
            innovation_cholesky,
            covariance_times_map.T,
        )
        gain_transpose = linalg.triangular_solve(
            innovation_cholesky.T,
            intermediate,
            upper=True,
        )
        gain = gain_transpose.T
        active_mean = active_mean + linalg.matrix_multiply(gain, residual)
        active_covariance = linalg.symmetric_rank_update(
            active_covariance,
            gain,
            innovation_covariance,
        )
        _scatter_vector(mean, active, active_mean)
        _scatter_matrix(covariance, active, active, active_covariance)
        objective = objective + increment
        if capture:
            innovations.append(
                _InnovationReplay(
                    materialized.factor_ids[index],
                    materialized.factor_time_indices[index],
                    parents,
                    innovation_covariance,
                    innovation_cholesky,
                )
            )
    if active_set != set(range(dimension)):
        raise ValueError("moment schedule did not activate every global coordinate")
    covariance = _symmetrize(covariance)
    linalg.cholesky(covariance)
    _require_finite((mean, covariance, objective), "moment native state")
    native = H4NativeMomentState(
        _vector_tuple(mean),
        _matrix_tuple(covariance),
        float(objective.item()),
    )
    result = H4SolverResult(
        materialized.problem_id,
        materialized.problem_sha256,
        "moment",
        materialized.protocol_id,
        len(materialized.factor_ids),
        None,
        native,
    )
    return result, tuple(innovations)


def _validate_solver_context(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    linalg: InstrumentedLinearAlgebra,
    arm: H4SolverArm,
) -> None:
    if type(materialized) is not H4MaterializedProblem:
        raise ValueError("materialized must be an exact H4MaterializedProblem")
    _validate_protocol(protocol)
    if (
        materialized.materialization_version != _MATERIALIZATION_VERSION
        or materialized.protocol_id != protocol.protocol_id
        or materialized.dtype != protocol.dtype
        or materialized.device != protocol.device
    ):
        raise ValueError("materialization and solve protocol do not match")
    _validate_facade(materialized, linalg, arm)
    _validate_factor_metadata(materialized)


def _validate_external_context(
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    linalg: InstrumentedLinearAlgebra,
) -> None:
    if type(materialized) is not H4MaterializedProblem:
        raise ValueError("materialized must be an exact H4MaterializedProblem")
    _validate_materialized_structure(materialized)
    if _materialized_digest(materialized) != materialized.tensor_sha256:
        raise ValueError("materialized tensor digest is invalid")
    if type(result) is not H4SolverResult:
        raise ValueError("result must be an exact H4SolverResult")
    if (
        result.problem_id != materialized.problem_id
        or result.problem_sha256 != materialized.problem_sha256
        or result.protocol_id != materialized.protocol_id
        or result.factor_count != len(materialized.factor_ids)
    ):
        raise ValueError("result identity does not match materialized problem")
    if result.arm == "information":
        if type(result.native_information) is not H4NativeInformationState or result.native_moment is not None:
            raise ValueError("result native state does not match information arm")
    elif result.arm == "moment":
        if type(result.native_moment) is not H4NativeMomentState or result.native_information is not None:
            raise ValueError("result native state does not match moment arm")
    else:
        raise ValueError("result arm is invalid")
    _validate_facade(materialized, linalg, result.arm)
    _validate_factor_metadata(materialized)


def _validate_facade(
    materialized: H4MaterializedProblem,
    linalg: InstrumentedLinearAlgebra,
    arm: H4SolverArm,
) -> None:
    try:
        problem_id, observed_arm = _facade_binding(linalg)
    except ValueError as exc:
        raise ValueError("invalid H4 facade") from exc
    if problem_id != materialized.problem_id or observed_arm != arm:
        raise ValueError("facade binding does not match problem and arm")


def _validate_protocol(protocol: H4SolveProtocol) -> None:
    if type(protocol) is not H4SolveProtocol or (
        protocol.protocol_id,
        protocol.dtype,
        protocol.device,
        protocol.factor_passes,
        protocol.solver_relative_budget,
        protocol.stopping_rule,
    ) != (
        "h4-single-pass-v1",
        "float64",
        "cpu",
        1,
        1.0e-9,
        "complete_schedule_finite_spd",
    ):
        raise ValueError("H4 solve protocol is not the frozen protocol")


def _validate_materialized_structure(value: H4MaterializedProblem) -> None:
    if value.materialization_version != _MATERIALIZATION_VERSION:
        raise ValueError("invalid materialization_version")
    if type(value.problem_id) is not str or not value.problem_id:
        raise ValueError("problem_id must be a nonempty string")
    _sha256(value.problem_sha256, "problem_sha256")
    _sha256(value.tensor_sha256, "tensor_sha256")
    if (
        value.protocol_id != "h4-single-pass-v1"
        or value.dtype != "float64"
        or value.device != "cpu"
    ):
        raise ValueError("materialized protocol literals are invalid")
    if value.source_kind not in ("scaled_pcg64", "h3_anchor") or value.kind not in (
        "coupled",
        "zero_control",
    ):
        raise ValueError("materialized source/kind is invalid")
    if any(
        type(item) is not int
        for item in (value.seed, value.horizon, value.d_z, value.d_m, value.dimension)
    ) or value.dimension <= 0:
        raise ValueError("materialized dimensions must be exact integers")
    if (
        type(value.coordinate_order) is not tuple
        or len(value.coordinate_order) != value.dimension
        or not all(type(item) is str and item for item in value.coordinate_order)
    ):
        raise ValueError("coordinate_order is invalid")
    _validate_factor_metadata(value)
    tensors = (
        *value._factor_matrices,
        *value._factor_targets,
        *value._factor_covariances,
    )
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != len(tensors):
        raise ValueError("materialized tensors must own nonaliasing storage")
    for tensor in tensors:
        if (
            type(tensor) is not Tensor
            or tensor.dtype is not torch.float64
            or tensor.device.type != "cpu"
            or tensor.requires_grad
            or not tensor.is_leaf
            or not tensor.is_contiguous()
            or tensor._base is not None
        ):
            raise ValueError("materialized tensors must be owned contiguous CPU float64")
    for matrix, target, covariance in zip(
        value._factor_matrices,
        value._factor_targets,
        value._factor_covariances,
        strict=True,
    ):
        rows = int(target.shape[0]) if target.ndim == 1 else -1
        if (
            matrix.ndim != 2
            or matrix.shape != (rows, value.dimension)
            or covariance.ndim != 2
            or covariance.shape != (rows, rows)
            or rows <= 0
        ):
            raise ValueError("materialized tensor shapes do not match factor metadata")


def _validate_factor_metadata(value: H4MaterializedProblem) -> None:
    count = len(value.factor_ids) if type(value.factor_ids) is tuple else -1
    sequences = (
        value.factor_roles,
        value.factor_time_indices,
        value.factor_normalized_coordinate_indices,
        value.factor_parent_coordinate_indices,
        value._factor_matrices,
        value._factor_targets,
        value._factor_covariances,
    )
    if count <= 0 or any(type(sequence) is not tuple or len(sequence) != count for sequence in sequences):
        raise ValueError("materialized factor sequences must be parallel nonempty tuples")
    if len(set(value.factor_ids)) != count or any(type(item) is not str or not item for item in value.factor_ids):
        raise ValueError("materialized factor IDs must be unique nonempty strings")
    for index in range(count):
        role = value.factor_roles[index]
        time = value.factor_time_indices[index]
        normalized = _strict_indices(
            value.factor_normalized_coordinate_indices[index],
            value.dimension,
            "factor_normalized_coordinate_indices",
            allow_empty=True,
        )
        parents = _strict_indices(
            value.factor_parent_coordinate_indices[index],
            value.dimension,
            "factor_parent_coordinate_indices",
            allow_empty=True,
        )
        if role not in ("initial", "transition", "observation"):
            raise ValueError("materialized factor role is invalid")
        if type(time) is not int or time < 0:
            raise ValueError("materialized factor time is invalid")
        if set(normalized) & set(parents):
            raise ValueError("materialized normalized and parent coordinates overlap")
        if role == "observation" and normalized:
            raise ValueError("observation factor cannot normalize coordinates")
        if role == "initial" and parents:
            raise ValueError("initial factor cannot have parents")


def _materialized_tensors(
    materialized: H4MaterializedProblem,
    capability: object,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    if capability is not _ACCESS_CAPABILITY:
        raise PermissionError("materialized tensor access requires the solver capability")
    return (
        materialized._factor_matrices,
        materialized._factor_targets,
        materialized._factor_covariances,
    )


def _materialized_digest(value: H4MaterializedProblem) -> str:
    parts = {
        name: getattr(value, name)
        for name in H4MaterializedProblem.__slots__
        if name != "tensor_sha256"
    }
    return _materialized_digest_parts(parts)


def _materialized_digest_parts(parts: dict[str, object]) -> str:
    metadata = {
        name: parts[name]
        for name in (
            "materialization_version",
            "problem_id",
            "problem_sha256",
            "protocol_id",
            "dtype",
            "device",
            "source_kind",
            "seed",
            "kind",
            "horizon",
            "d_z",
            "d_m",
            "dimension",
            "coordinate_order",
            "factor_ids",
            "factor_roles",
            "factor_time_indices",
            "factor_normalized_coordinate_indices",
            "factor_parent_coordinate_indices",
        )
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(_MATERIALIZATION_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    for role, tensors in (
        ("matrix", parts["_factor_matrices"]),
        ("target", parts["_factor_targets"]),
        ("covariance", parts["_factor_covariances"]),
    ):
        assert isinstance(tensors, tuple)
        for index, tensor in enumerate(tensors):
            assert isinstance(tensor, Tensor)
            header = json.dumps(
                {
                    "role": role,
                    "index": index,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "device": str(tensor.device),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            raw = tensor.detach().numpy().tobytes(order="C")
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _selected_blocks(
    materialized: H4MaterializedProblem,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    initial = _coordinate_union(
        materialized,
        tuple(
            index
            for index, role in enumerate(materialized.factor_roles)
            if role == "initial"
        ),
        normalized=True,
    )
    terminal = _coordinate_union(
        materialized,
        tuple(
            index
            for index, (role, time) in enumerate(
                zip(materialized.factor_roles, materialized.factor_time_indices, strict=True)
            )
            if role == "transition" and time == materialized.horizon
        ),
        normalized=True,
    )
    blocks: list[tuple[str, tuple[int, ...]]] = [
        ("initial", initial),
        ("terminal", terminal),
    ]
    for time in range(1, materialized.horizon + 1):
        observation = _coordinate_union(
            materialized,
            tuple(
                index
                for index, (role, factor_time) in enumerate(
                    zip(materialized.factor_roles, materialized.factor_time_indices, strict=True)
                )
                if role == "observation" and factor_time == time
            ),
            normalized=False,
        )
        blocks.append((f"observation[{time}]", observation))
    expected_dimension = materialized.d_z + materialized.d_m
    if any(len(indices) != expected_dimension for _, indices in blocks):
        raise ValueError("selected coordinate block has the wrong dimension")
    return tuple(blocks)


def _coordinate_union(
    materialized: H4MaterializedProblem,
    factor_indices: tuple[int, ...],
    *,
    normalized: bool,
) -> tuple[int, ...]:
    if not factor_indices:
        raise ValueError("selected coordinate block is missing its factors")
    source = (
        materialized.factor_normalized_coordinate_indices
        if normalized
        else materialized.factor_parent_coordinate_indices
    )
    values = tuple(
        coordinate
        for factor_index in factor_indices
        for coordinate in source[factor_index]
    )
    if len(values) != len(set(values)):
        raise ValueError("selected coordinate block contains a duplicate")
    ordered = tuple(sorted(values))
    if any(coordinate < 0 or coordinate >= materialized.dimension for coordinate in ordered):
        raise ValueError("selected coordinate block contains an undeclared coordinate")
    return ordered


def _owned_tensor(value: object) -> Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu").detach().clone().contiguous()


def _owned_from_tuple(value: tuple) -> Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


def _scatter_vector(target: Tensor, indices: tuple[int, ...], value: Tensor) -> None:
    index = torch.tensor(indices, dtype=torch.int64)
    target.index_copy_(0, index, value)


def _scatter_matrix(
    target: Tensor,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    value: Tensor,
) -> None:
    if not rows or not columns:
        return
    row_index = torch.tensor(rows, dtype=torch.int64)[:, None]
    column_index = torch.tensor(columns, dtype=torch.int64)[None, :]
    target[row_index, column_index] = value


def _symmetrize(value: Tensor) -> Tensor:
    return 0.5 * (value + value.T)


def _require_finite(values: tuple[Tensor, ...], name: str) -> None:
    if any(not bool(torch.all(torch.isfinite(value)).item()) for value in values):
        raise ValueError(f"{name} must be finite")


def _strict_indices(
    value: object,
    bound: int | None,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if type(value) is not tuple or (not value and not allow_empty):
        raise ValueError(f"{name} must be an immutable index tuple")
    if any(
        type(item) is not int
        or item < 0
        or (bound is not None and item >= bound)
        for item in value
    ) or any(left >= right for left, right in zip(value, value[1:], strict=False)):
        raise ValueError(f"{name} must be strictly ascending unique indices")
    return value


def _vector_tuple(value: Tensor) -> tuple[float, ...]:
    if value.ndim != 1:
        raise ValueError("expected a vector")
    return tuple(float(item) for item in value.tolist())


def _matrix_tuple(value: Tensor) -> tuple[tuple[float, ...], ...]:
    if value.ndim != 2:
        raise ValueError("expected a matrix")
    return tuple(tuple(float(item) for item in row) for row in value.tolist())


def _sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "H4MaterializedProblem",
    "H4InnovationDiagnostic",
    "H4NativeDiagnostics",
    "H4GaussianSolver",
    "InformationFormH4Solver",
    "MomentFormH4Solver",
    "materialize_h4_problem",
    "solve_information_form",
    "solve_moment_form",
    "to_common_terminal_law",
    "evaluate_h4_native_diagnostics",
]
