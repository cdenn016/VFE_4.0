"""Immutable H4 protocol records and canonical neutral-problem encoding."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from .results import GateStatus, InvariantResult

H4SolverArm: TypeAlias = Literal["information", "moment"]
H4ProblemKind: TypeAlias = Literal["coupled", "zero_control"]
H4ProblemSource: TypeAlias = Literal["scaled_pcg64", "h3_anchor"]
H4PairOrder: TypeAlias = Literal["information_then_moment", "moment_then_information"]
H4FactorRole: TypeAlias = Literal["initial", "transition", "observation"]
H4OperationKind: TypeAlias = Literal["cholesky", "triangular_solve", "matrix_multiply", "symmetric_rank_update", "selected_block_extract"]
H4JsonScalar: TypeAlias = str | int | float | bool | None
H4JsonValue: TypeAlias = H4JsonScalar | tuple["H4JsonValue", ...] | Mapping[str, "H4JsonValue"]
H4JsonMapping: TypeAlias = Mapping[str, H4JsonValue]
H4MeasurementName: TypeAlias = Literal["primary_seed_ratio_geometric_mean", "primary_bootstrap_lower", "primary_bootstrap_upper", "primary_effect_threshold", "primary_timed_ab_total", "primary_timed_ba_total", "maximum_solver_stopping_residual", "maximum_allowance_scale_fraction"]
H4AllowanceInvariantName: TypeAlias = Literal["h3_anchor_identity", "exact_posterior_gap_equivalence", "terminal_h_equivalence", "terminal_J_equivalence", "selected_moment_equivalence", "complete_objective_equivalence"]

H4_INVARIANT_NAMES = (
    "h3_anchor_identity", "fixed_seed_problem_identity", "coupled_zero_control_contract",
    "cpu_float64_one_thread", "shared_protocol_identity", "scaled_condition_envelope",
    "complete_repetition_table", "primary_timed_order_balance", "exact_posterior_gap_equivalence",
    "terminal_h_equivalence", "terminal_J_equivalence", "selected_moment_equivalence",
    "complete_objective_equivalence", "all_equivalence_allowances_decisive",
    "real_operation_instrumentation", "primary_seed_level_inference", "primary_effect_threshold",
)
H4_MEASUREMENT_NAMES = (
    "primary_seed_ratio_geometric_mean", "primary_bootstrap_lower", "primary_bootstrap_upper",
    "primary_effect_threshold", "primary_timed_ab_total", "primary_timed_ba_total",
    "maximum_solver_stopping_residual", "maximum_allowance_scale_fraction",
)
H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL = (
    "primary_seed_ratio_geometric_mean", "primary_bootstrap_lower", "primary_bootstrap_upper",
    "primary_timed_ab_total", "primary_timed_ba_total",
)
H4_ALLOWANCE_INVARIANT_NAMES = (
    "h3_anchor_identity", "exact_posterior_gap_equivalence", "terminal_h_equivalence",
    "terminal_J_equivalence", "selected_moment_equivalence", "complete_objective_equivalence",
)

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_UNAVAILABLE_ANCHOR = "not_evaluated_after_decisive_h3_anchor_failure"
_UNAVAILABLE_ELIGIBILITY = "not_evaluated_after_inconclusive_eligibility"


@dataclass(frozen=True)
class H4RawDraw:
    draw_index: int
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.draw_index) is not int or self.draw_index < 0:
            raise ValueError("draw_index must be a nonnegative integer")
        _string(self.name, "name")
        if type(self.shape) is not tuple or any(type(v) is not int or v < 0 for v in self.shape):
            raise ValueError("shape must be a tuple of nonnegative integers")
        if type(self.values) is not tuple or _product(self.shape) != len(self.values):
            raise ValueError("values length must equal product(shape)")
        object.__setattr__(self, "values", _vector(self.values, len(self.values), "values"))


@dataclass(frozen=True)
class H4AffineGaussianFactor:
    factor_id: str
    role: H4FactorRole
    time_index: int
    normalized_coordinate_indices: tuple[int, ...]
    parent_coordinate_indices: tuple[int, ...]
    matrix: tuple[tuple[float, ...], ...]
    target: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    raw_draws: tuple[H4RawDraw, ...]

    def __post_init__(self) -> None:
        _string(self.factor_id, "factor_id")
        if self.role not in ("initial", "transition", "observation"):
            raise ValueError("role must be an H4 factor role")
        if type(self.time_index) is not int or self.time_index < 0:
            raise ValueError("time_index must be nonnegative")
        if type(self.matrix) is not tuple or not self.matrix or type(self.matrix[0]) is not tuple or not self.matrix[0]:
            raise ValueError("matrix must be nonempty")
        rows, dimension = len(self.matrix), len(self.matrix[0])
        matrix = _matrix(self.matrix, rows, dimension, "matrix")
        target = _vector(self.target, rows, "target")
        covariance = _matrix(self.covariance, rows, rows, "covariance")
        _spd(covariance, "covariance")
        normalized = _indices(self.normalized_coordinate_indices, dimension, "normalized_coordinate_indices")
        parents = _indices(self.parent_coordinate_indices, dimension, "parent_coordinate_indices")
        if set(normalized) & set(parents):
            raise ValueError("normalized and parent indices must be disjoint")
        if self.role in ("initial", "transition") and len(normalized) != rows:
            raise ValueError("initial and transition normalized indices must match residual dimension")
        if self.role == "observation" and normalized:
            raise ValueError("observation normalized indices must be empty")
        if self.role == "initial" and parents:
            raise ValueError("initial parent indices must be empty")
        if self.role in ("initial", "transition"):
            for row, column in enumerate(normalized):
                if any(matrix[row][j] != (1.0 if j == column else 0.0) for j in normalized):
                    raise ValueError("normalized columns must be identity")
        allowed = set(normalized) | set(parents)
        if any(matrix[i][j] != 0.0 for i in range(rows) for j in range(dimension) if j not in allowed):
            raise ValueError("matrix may support only normalized and parent columns")
        if type(self.raw_draws) is not tuple or not all(isinstance(draw, H4RawDraw) for draw in self.raw_draws):
            raise ValueError("raw_draws must be a tuple of H4RawDraw")
        if tuple(draw.draw_index for draw in self.raw_draws) != tuple(sorted(draw.draw_index for draw in self.raw_draws)):
            raise ValueError("raw_draws must be increasing by draw_index")
        if len({draw.name for draw in self.raw_draws}) != len(self.raw_draws):
            raise ValueError("raw draw names must be unique per factor")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "normalized_coordinate_indices", normalized)
        object.__setattr__(self, "parent_coordinate_indices", parents)


@dataclass(frozen=True)
class H4NeutralProblem:
    problem_id: str
    source_kind: H4ProblemSource
    seed: int
    kind: H4ProblemKind
    horizon: int
    d_z: int
    d_m: int
    dimension: int
    coordinate_order: tuple[str, ...]
    factor_schedule: tuple[H4AffineGaussianFactor, ...]
    canonical_sha256: str

    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if self.source_kind not in ("scaled_pcg64", "h3_anchor") or self.kind not in ("coupled", "zero_control"):
            raise ValueError("source_kind and kind must be valid")
        if type(self.seed) is not int or type(self.horizon) is not int or type(self.d_z) is not int or type(self.d_m) is not int or type(self.dimension) is not int:
            raise ValueError("problem dimensions and seed must be integers")
        if type(self.coordinate_order) is not tuple or len(self.coordinate_order) != self.dimension or not all(type(x) is str and x for x in self.coordinate_order):
            raise ValueError("coordinate_order must be a nonempty string tuple of dimension")
        if type(self.factor_schedule) is not tuple or not self.factor_schedule or not all(isinstance(f, H4AffineGaussianFactor) for f in self.factor_schedule):
            raise ValueError("factor_schedule must be nonempty H4 factors")
        if len({f.factor_id for f in self.factor_schedule}) != len(self.factor_schedule):
            raise ValueError("factor IDs must be unique")
        if any(len(f.matrix[0]) != self.dimension for f in self.factor_schedule):
            raise ValueError("factor matrix dimension mismatch")
        _sha(self.canonical_sha256, "canonical_sha256")
        if self.source_kind == "scaled_pcg64":
            if self.seed <= 0 or self.kind not in ("coupled", "zero_control") or self.horizon not in (7, 15, 31) or (self.d_z, self.d_m) != (4, 4) or self.dimension != (self.horizon + 1) * 8:
                raise ValueError("invalid scaled H4 problem")
            expected_coordinates = tuple(f"{prefix}[{t},{i}]" for t in range(self.horizon + 1) for prefix in ("z", "m") for i in range(4))
            if self.coordinate_order != expected_coordinates:
                raise ValueError("scaled coordinate order is frozen")
            expected_ids = ("initial_joint", *(item for t in range(1, self.horizon + 1) for item in (f"m_transition[{t}]", f"z_transition[{t}]", f"observation[{t}]")))
            if tuple(f.factor_id for f in self.factor_schedule) != expected_ids:
                raise ValueError("scaled factor schedule is frozen")
            all_draws = tuple(draw.draw_index for factor in self.factor_schedule for draw in factor.raw_draws)
            if len(set(all_draws)) != len(all_draws):
                raise ValueError("scaled draw indices must be globally unique")
        else:
            if self.seed != 0 or self.horizon != 1 or (self.d_z, self.d_m, self.dimension) != (1, 1, 4) or self.coordinate_order != ("z0", "m0", "z1", "m1"):
                raise ValueError("invalid H3 anchor problem")


@dataclass(frozen=True)
class H4SolveProtocol:
    protocol_id: Literal["h4-single-pass-v1"] = "h4-single-pass-v1"
    dtype: Literal["float64"] = "float64"
    device: Literal["cpu"] = "cpu"
    factor_passes: Literal[1] = 1
    solver_relative_budget: float = 1.0e-9
    stopping_rule: Literal["complete_schedule_finite_spd"] = "complete_schedule_finite_spd"
    def __post_init__(self) -> None:
        if (self.protocol_id, self.dtype, self.device, self.factor_passes, self.stopping_rule) != ("h4-single-pass-v1", "float64", "cpu", 1, "complete_schedule_finite_spd") or self.solver_relative_budget != 1.0e-9:
            raise ValueError("H4 solve protocol is frozen")


@dataclass(frozen=True)
class H4SelectedMoment:
    name: str
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    def __post_init__(self) -> None:
        _string(self.name, "name")
        mean = _vector(self.mean, len(self.mean) if type(self.mean) is tuple else -1, "mean")
        covariance = _matrix(self.covariance, len(mean), len(mean), "covariance")
        _spd(covariance, "covariance")
        object.__setattr__(self, "mean", mean); object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class H4TerminalLaw:
    arm: H4SolverArm; h: tuple[float, ...]; J: tuple[tuple[float, ...], ...]; mean: tuple[float, ...]; selected_moments: tuple[H4SelectedMoment, ...]; complete_objective: float; stopping_residual: float
    def __post_init__(self) -> None:
        _law(self.arm, self.h, self.J, self.mean, self.complete_objective, "terminal")
        if type(self.selected_moments) is not tuple or not self.selected_moments or not all(isinstance(value, H4SelectedMoment) for value in self.selected_moments) or len({value.name for value in self.selected_moments}) != len(self.selected_moments): raise ValueError("selected_moments must be unique immutable moments")
        _finite(self.stopping_residual, "stopping_residual")

@dataclass(frozen=True)
class H4NativeInformationState:
    h: tuple[float, ...]; J: tuple[tuple[float, ...], ...]; mean: tuple[float, ...]; complete_objective: float
    def __post_init__(self) -> None: _law("information", self.h, self.J, self.mean, self.complete_objective, "native_information")

@dataclass(frozen=True)
class H4NativeMomentState:
    mean: tuple[float, ...]; covariance: tuple[tuple[float, ...], ...]; complete_objective: float
    def __post_init__(self) -> None:
        mean = _vector(self.mean, len(self.mean) if type(self.mean) is tuple else -1, "mean"); covariance = _matrix(self.covariance, len(mean), len(mean), "covariance"); _spd(covariance, "covariance"); _finite(self.complete_objective, "complete_objective")
        object.__setattr__(self, "mean", mean); object.__setattr__(self, "covariance", covariance)

@dataclass(frozen=True)
class H4SolverResult:
    problem_id: str; problem_sha256: str; arm: H4SolverArm; protocol_id: Literal["h4-single-pass-v1"]; factor_count: int; native_information: H4NativeInformationState | None; native_moment: H4NativeMomentState | None
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id"); _sha(self.problem_sha256, "problem_sha256")
        if self.arm not in ("information", "moment") or self.protocol_id != "h4-single-pass-v1" or type(self.factor_count) is not int or self.factor_count <= 0: raise ValueError("invalid solver result identity")
        if (self.arm == "information") != (self.native_information is not None) or (self.arm == "moment") != (self.native_moment is not None): raise ValueError("solver result requires exactly its matching native state")

@dataclass(frozen=True)
class H4TimingRecord:
    problem_id: str; problem_index: int; horizon_index: int; seed_index: int; kind_index: int; seed: int; kind: H4ProblemKind; horizon: int; repetition_index: int; pair_index: int; order: H4PairOrder; information_nanoseconds: int; moment_nanoseconds: int
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if any(type(value) is not int or value < 0 for value in (self.problem_index, self.horizon_index, self.seed_index, self.kind_index, self.repetition_index, self.pair_index)) or type(self.seed) is not int or self.seed <= 0 or self.kind not in ("coupled", "zero_control") or self.horizon not in (7, 15, 31): raise ValueError("invalid timing identity")
        if self.pair_index != 3 + self.repetition_index or self.order not in ("information_then_moment", "moment_then_information") or self.order != ("information_then_moment" if (self.horizon_index + self.seed_index + self.kind_index + self.pair_index) % 2 == 0 else "moment_then_information") or type(self.information_nanoseconds) is not int or type(self.moment_nanoseconds) is not int or self.information_nanoseconds <= 0 or self.moment_nanoseconds <= 0: raise ValueError("invalid H4 timing record")

@dataclass(frozen=True)
class H4OperationRecord:
    problem_id: str; arm: H4SolverArm; operation: H4OperationKind; operand_shapes: tuple[tuple[int, ...], ...]; result_shape: tuple[int, ...]; count: int
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if self.arm not in ("information", "moment") or self.operation not in ("cholesky", "triangular_solve", "matrix_multiply", "symmetric_rank_update", "selected_block_extract") or type(self.operand_shapes) is not tuple or not self.operand_shapes or type(self.result_shape) is not tuple or not self.result_shape or type(self.count) is not int or self.count <= 0: raise ValueError("invalid operation record")
        if any(type(dimension) is not int or dimension <= 0 for shape in (*self.operand_shapes, self.result_shape) for dimension in shape): raise ValueError("operation shapes must be positive")

@dataclass(frozen=True)
class H4MemoryRecord:
    problem_id: str; arm: H4SolverArm; python_peak_bytes: int | None; process_working_set_delta_bytes: int | None; unavailable_fields: tuple[Literal["python_peak_bytes", "process_working_set_delta_bytes"], ...]
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if self.arm not in ("information", "moment") or type(self.unavailable_fields) is not tuple or tuple(sorted(set(self.unavailable_fields), key=("python_peak_bytes", "process_working_set_delta_bytes").index)) != self.unavailable_fields or any(value not in ("python_peak_bytes", "process_working_set_delta_bytes") for value in self.unavailable_fields): raise ValueError("invalid unavailable_fields")
        if (self.python_peak_bytes is None) != ("python_peak_bytes" in self.unavailable_fields) or (self.process_working_set_delta_bytes is None) != ("process_working_set_delta_bytes" in self.unavailable_fields) or (self.python_peak_bytes is not None and (type(self.python_peak_bytes) is not int or self.python_peak_bytes < 0)) or (self.process_working_set_delta_bytes is not None and type(self.process_working_set_delta_bytes) is not int): raise ValueError("invalid memory metrics")


@dataclass(frozen=True)
class H4GateResult:
    gate: Literal["H4"]
    status: GateStatus
    measurements: Mapping[H4MeasurementName, float | None]
    invariants: tuple[InvariantResult, ...]
    allowances_by_invariant: Mapping[H4AllowanceInvariantName, H4JsonMapping]
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate != "H4" or not isinstance(self.status, GateStatus):
            raise ValueError("invalid H4 gate result")
        measurements = dict(self.measurements)
        if tuple(measurements) != H4_MEASUREMENT_NAMES:
            raise ValueError("measurement names/order must equal H4_MEASUREMENT_NAMES")
        for name, value in measurements.items():
            _optional_finite(value, f"measurements[{name}]")
        if type(self.invariants) is not tuple or tuple(item.name for item in self.invariants) != H4_INVARIANT_NAMES:
            raise ValueError("invariants must equal H4_INVARIANT_NAMES in order")
        allowances = dict(self.allowances_by_invariant)
        if tuple(allowances) != H4_ALLOWANCE_INVARIANT_NAMES:
            raise ValueError("allowance names/order must equal H4_ALLOWANCE_INVARIANT_NAMES")
        if type(self.obligations) is not tuple or len(set(self.obligations)) != len(self.obligations) or not all(type(x) is str and x for x in self.obligations):
            raise ValueError("obligations must be unique nonempty strings")
        anchor = self.invariants[0]
        anchor_fail = self.status is GateStatus.FAIL and not anchor.passed and anchor.value is not None and anchor.limit is not None and all((not item.passed and item.value is None and item.limit is None and item.detail == _UNAVAILABLE_ANCHOR) for item in self.invariants[1:])
        if anchor_fail:
            if tuple(name for name, value in measurements.items() if value is None) != H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL or measurements["primary_effect_threshold"] != .80:
                raise ValueError("anchor failure measurements are frozen")
        elif self.status is not GateStatus.INCONCLUSIVE:
            if any(value is None for value in measurements.values()):
                raise ValueError("completed pass/fail measurements must be finite")
        elif any(value is None for value in measurements.values()) and not self.obligations:
            raise ValueError("inconclusive unavailable measurements require obligations")
        frozen: dict[str, H4JsonMapping] = {}
        for name, record in allowances.items():
            if not isinstance(record, Mapping) or not record:
                raise ValueError("allowance record must be nonempty mapping")
            copied = _freeze_json(record, f"allowances[{name}]")
            if not isinstance(copied, MappingProxyType):
                raise ValueError("allowance record must be mapping")
            applicable = copied.get("applicable")
            if applicable is False:
                if tuple(copied) != ("applicable", "reason") or copied["reason"] not in (_UNAVAILABLE_ANCHOR, _UNAVAILABLE_ELIGIBILITY):
                    raise ValueError("inapplicable allowance sentinel is malformed")
            elif applicable is True:
                if len(copied) < 2:
                    raise ValueError("applicable allowance requires numerical fields")
            else:
                raise ValueError("allowance applicable flag is required")
            frozen[name] = copied  # type: ignore[assignment]
        if anchor_fail:
            if frozen["h3_anchor_identity"].get("applicable") is not True or any(frozen[name] != MappingProxyType({"applicable": False, "reason": _UNAVAILABLE_ANCHOR}) for name in H4_ALLOWANCE_INVARIANT_NAMES[1:]):
                raise ValueError("anchor failure allowance applicability is frozen")
        object.__setattr__(self, "measurements", MappingProxyType(measurements))
        object.__setattr__(self, "allowances_by_invariant", MappingProxyType(frozen))


def h4_problem_core(problem: H4NeutralProblem) -> dict[str, object]:
    return {name: _json_value(getattr(problem, name)) for name in ("problem_id", "source_kind", "seed", "kind", "horizon", "d_z", "d_m", "dimension", "coordinate_order", "factor_schedule")}

def h4_problem_digest(problem: H4NeutralProblem) -> str:
    core = h4_problem_core(problem)
    return hashlib.sha256(b"vfe4.h4.neutral-problem.v1\0" + _compact_json(core)).hexdigest()

def canonical_h4_problem_bytes(problem: H4NeutralProblem) -> bytes:
    digest = h4_problem_digest(problem)
    if digest != problem.canonical_sha256:
        raise ValueError("canonical_sha256 does not match problem core")
    return _compact_json({"schema_version": "h4-neutral-problem-v1", "canonical_sha256": digest, "problem": h4_problem_core(problem)})


def _compact_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def _json_value(value: object) -> object:
    if isinstance(value, H4RawDraw): return {name: _json_value(getattr(value, name)) for name in ("draw_index", "name", "shape", "values")}
    if isinstance(value, H4AffineGaussianFactor): return {name: _json_value(getattr(value, name)) for name in ("factor_id", "role", "time_index", "normalized_coordinate_indices", "parent_coordinate_indices", "matrix", "target", "covariance", "raw_draws")}
    if type(value) is tuple: return [_json_value(item) for item in value]
    if type(value) in (str, int, float, bool) or value is None: return value
    raise ValueError("canonical JSON value is unsupported")

def _freeze_json(value: object, name: str) -> H4JsonValue:
    if isinstance(value, Mapping):
        copied: dict[str, H4JsonValue] = {}
        for key, item in value.items():
            _string(key, f"{name}.key")
            if key in copied: raise ValueError("duplicate JSON key")
            copied[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(copied)
    if type(value) is tuple: return tuple(_freeze_json(item, name) for item in value)
    if type(value) in (str, int, bool) or value is None: return value
    if type(value) is float:
        _finite(value, name); return value
    raise ValueError(f"{name} must be immutable JSON")

def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape: result *= value
    return result
def _string(value: object, name: str) -> None:
    if type(value) is not str or not value: raise ValueError(f"{name} must be a nonempty string")
def _finite(value: object, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)): raise ValueError(f"{name} must be finite")
def _optional_finite(value: object, name: str) -> None:
    if value is not None: _finite(value, name)
def _sha(value: object, name: str) -> None:
    if type(value) is not str or _HEX.fullmatch(value) is None: raise ValueError(f"{name} must be lowercase SHA-256")
def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != size: raise ValueError(f"{name} has invalid length")
    for item in value: _finite(item, name)
    return tuple(float(item) for item in value)
def _matrix(value: object, rows: int, columns: int, name: str) -> tuple[tuple[float, ...], ...]:
    if type(value) is not tuple or len(value) != rows: raise ValueError(f"{name} has invalid rows")
    return tuple(_vector(row, columns, f"{name}[{i}]") for i, row in enumerate(value))
def _indices(value: object, dimension: int, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int or item < 0 or item >= dimension for item in value) or len(set(value)) != len(value): raise ValueError(f"{name} must be unique indices in range")
    return value
def _spd(matrix: tuple[tuple[float, ...], ...], name: str) -> None:
    size = len(matrix)
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)): raise ValueError(f"{name} must be symmetric")
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0.0: raise ValueError(f"{name} must be positive definite")
                lower[i][j] = math.sqrt(value)
            else: lower[i][j] = value / lower[j][j]

def _law(arm: object, h: object, J: object, mean: object, objective: object, name: str) -> None:
    if arm not in ("information", "moment"): raise ValueError(f"{name} arm is invalid")
    vector = _vector(h, len(h) if type(h) is tuple else -1, f"{name}.h")
    matrix = _matrix(J, len(vector), len(vector), f"{name}.J")
    _spd(matrix, f"{name}.J")
    _vector(mean, len(vector), f"{name}.mean"); _finite(objective, f"{name}.complete_objective")

__all__ = [name for name in globals() if name.startswith("H4")] + ["canonical_h4_problem_bytes", "h4_problem_core", "h4_problem_digest"]
