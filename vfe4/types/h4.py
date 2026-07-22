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
H4IntervalClass: TypeAlias = Literal["support", "no_support", "crossing", "boundary"]

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
H4_APPLICABLE_ALLOWANCE_FIELDS = (
    "applicable", "dimension", "operands", "absolute_summands", "condition_numbers",
    "operation_counts", "solver_contribution", "invariant_scale", "final_allowance",
    "allowance_scale_ratio",
)
H4_PROBLEM_SEEDS = (
    104729, 130363, 155921, 181081, 206369, 231779, 257053, 282407,
    307831, 333271, 358747, 384253, 409891, 435437, 461009, 486587,
    512161, 537793, 563359, 588937,
)
H4_PRIMARY_EFFECT_THRESHOLD = 0.80

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
        indices = tuple(draw.draw_index for draw in self.raw_draws)
        if any(left >= right for left, right in zip(indices, indices[1:], strict=False)):
            raise ValueError("raw_draws must be strictly increasing by draw_index")
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
            if self.problem_id != f"h4-{self.kind}-T{self.horizon}-dz4-dm4-seed{self.seed}-v1":
                raise ValueError("scaled problem ID is frozen")
            expected_ids = ("initial_joint", *(item for t in range(1, self.horizon + 1) for item in (f"m_transition[{t}]", f"z_transition[{t}]", f"observation[{t}]")))
            if tuple(f.factor_id for f in self.factor_schedule) != expected_ids:
                raise ValueError("scaled factor schedule is frozen")
            all_draws = tuple(draw.draw_index for factor in self.factor_schedule for draw in factor.raw_draws)
            if all_draws and tuple(sorted(all_draws)) != tuple(range(11 * self.horizon)):
                raise ValueError("scaled draw indices must be globally unique")
            _validate_scaled_schedule(self.factor_schedule, self.kind, self.horizon, self.dimension)
        else:
            if self.seed != 0 or self.horizon != 1 or (self.d_z, self.d_m, self.dimension) != (1, 1, 4) or self.coordinate_order != ("z0", "m0", "z1", "m1"):
                raise ValueError("invalid H3 anchor problem")
            _validate_anchor_schedule(self)


@dataclass(frozen=True)
class H4SolveProtocol:
    protocol_id: Literal["h4-single-pass-v1"] = "h4-single-pass-v1"
    dtype: Literal["float64"] = "float64"
    device: Literal["cpu"] = "cpu"
    factor_passes: Literal[1] = 1
    solver_relative_budget: float = 1.0e-9
    stopping_rule: Literal["complete_schedule_finite_spd"] = "complete_schedule_finite_spd"
    def __post_init__(self) -> None:
        if type(self.factor_passes) is not int or (self.protocol_id, self.dtype, self.device, self.factor_passes, self.stopping_rule) != ("h4-single-pass-v1", "float64", "cpu", 1, "complete_schedule_finite_spd") or self.solver_relative_budget != 1.0e-9:
            raise ValueError("H4 solve protocol is frozen")


@dataclass(frozen=True)
class H4SelectedMoment:
    name: str
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    def __post_init__(self) -> None:
        _string(self.name, "name")
        mean = _vector(self.mean, len(self.mean) if type(self.mean) is tuple else -1, "mean")
        if not mean: raise ValueError("mean must be nonempty")
        covariance = _matrix(self.covariance, len(mean), len(mean), "covariance")
        _spd(covariance, "covariance")
        object.__setattr__(self, "mean", mean); object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class H4TerminalLaw:
    arm: H4SolverArm; h: tuple[float, ...]; J: tuple[tuple[float, ...], ...]; mean: tuple[float, ...]; selected_moments: tuple[H4SelectedMoment, ...]; complete_objective: float; stopping_residual: float
    def __post_init__(self) -> None:
        _law(self.arm, self.h, self.J, self.mean, self.complete_objective, "terminal")
        if type(self.selected_moments) is not tuple or len(self.selected_moments) < 3 or not all(isinstance(value, H4SelectedMoment) for value in self.selected_moments): raise ValueError("selected_moments must be immutable moments")
        expected = ("initial", "terminal", *(f"observation[{time}]" for time in range(1, len(self.selected_moments)-1)))
        if tuple(value.name for value in self.selected_moments) != expected or any(len(value.mean) != len(self.selected_moments[0].mean) for value in self.selected_moments): raise ValueError("selected moments must have exact names and equal blocks")
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
        if (self.arm == "information" and (type(self.native_information) is not H4NativeInformationState or self.native_moment is not None)) or (self.arm == "moment" and (type(self.native_moment) is not H4NativeMomentState or self.native_information is not None)): raise ValueError("solver result requires exactly its matching native state")

@dataclass(frozen=True)
class H4TimingRecord:
    problem_id: str; problem_index: int; horizon_index: int; seed_index: int; kind_index: int; seed: int; kind: H4ProblemKind; horizon: int; repetition_index: int; pair_index: int; order: H4PairOrder; information_nanoseconds: int; moment_nanoseconds: int
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if any(
            type(value) is not int
            for value in (
                self.problem_index, self.horizon_index, self.seed_index,
                self.kind_index, self.seed, self.horizon,
                self.repetition_index, self.pair_index,
                self.information_nanoseconds, self.moment_nanoseconds,
            )
        ):
            raise ValueError("H4 timing integer fields must have exact int type")
        horizons = (7,15,31)
        if self.horizon_index not in range(3) or self.seed_index not in range(20) or self.kind_index not in (0,1) or self.repetition_index not in range(11) or self.horizon != horizons[self.horizon_index] or self.seed != H4_PROBLEM_SEEDS[self.seed_index] or self.kind != ("coupled" if self.kind_index == 0 else "zero_control") or self.problem_index != ((self.horizon_index * 20 + self.seed_index) * 2 + self.kind_index): raise ValueError("invalid timing identity")
        if self.pair_index != 3 + self.repetition_index or self.order not in ("information_then_moment", "moment_then_information") or self.order != ("information_then_moment" if (self.horizon_index + self.seed_index + self.kind_index + self.pair_index) % 2 == 0 else "moment_then_information") or type(self.information_nanoseconds) is not int or type(self.moment_nanoseconds) is not int or self.information_nanoseconds <= 0 or self.moment_nanoseconds <= 0: raise ValueError("invalid H4 timing record")

@dataclass(frozen=True)
class H4OperationRecord:
    problem_id: str; arm: H4SolverArm; operation: H4OperationKind; operand_shapes: tuple[tuple[int, ...], ...]; result_shape: tuple[int, ...]; count: int
    def __post_init__(self) -> None:
        _string(self.problem_id, "problem_id")
        if self.arm not in ("information", "moment") or self.operation not in ("cholesky", "triangular_solve", "matrix_multiply", "symmetric_rank_update", "selected_block_extract") or type(self.operand_shapes) is not tuple or not self.operand_shapes or type(self.result_shape) is not tuple or not self.result_shape or type(self.count) is not int or self.count <= 0: raise ValueError("invalid operation record")
        if any(type(shape) is not tuple or not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape) for shape in (*self.operand_shapes, self.result_shape)): raise ValueError("operation shapes must be immutable positive tuples")

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
        if type(self.invariants) is not tuple or not all(
            type(item) is InvariantResult for item in self.invariants
        ):
            raise ValueError("invariants must be exact InvariantResult records")
        measurements = dict(self.measurements)
        if tuple(measurements) != H4_MEASUREMENT_NAMES:
            raise ValueError("measurement names/order must equal H4_MEASUREMENT_NAMES")
        for name, value in measurements.items():
            _optional_finite(value, f"measurements[{name}]")
        if measurements["primary_effect_threshold"] != H4_PRIMARY_EFFECT_THRESHOLD:
            raise ValueError("primary_effect_threshold is frozen at 0.80")
        if tuple(item.name for item in self.invariants) != H4_INVARIANT_NAMES:
            raise ValueError("invariants must equal H4_INVARIANT_NAMES in order")
        allowances = dict(self.allowances_by_invariant)
        if tuple(allowances) != H4_ALLOWANCE_INVARIANT_NAMES:
            raise ValueError("allowance names/order must equal H4_ALLOWANCE_INVARIANT_NAMES")
        if type(self.obligations) is not tuple or len(set(self.obligations)) != len(self.obligations) or not all(type(x) is str and x for x in self.obligations):
            raise ValueError("obligations must be unique nonempty strings")
        anchor = self.invariants[0]
        anchor_fail = (
            self.status is GateStatus.FAIL
            and not anchor.passed
            and anchor.value is not None
            and anchor.limit is not None
            and anchor.value > anchor.limit
            and not self.obligations
            and all(
                (not item.passed and item.value is None and item.limit is None
                 and item.detail == _UNAVAILABLE_ANCHOR)
                for item in self.invariants[1:]
            )
        )
        if anchor_fail:
            if tuple(name for name, value in measurements.items() if value is None) != H4_PRIMARY_MEASUREMENTS_UNAVAILABLE_AFTER_ANCHOR_FAIL:
                raise ValueError("anchor failure measurements are frozen")
        elif any(item.detail == _UNAVAILABLE_ANCHOR for item in self.invariants):
            raise ValueError("anchor-unavailable invariant sentinel is reserved for anchor FAIL")
        elif self.status in (GateStatus.PASS, GateStatus.FAIL) and any(
            value is None for value in measurements.values()
        ):
            raise ValueError("completed pass/fail measurements must be finite")

        interval_class: H4IntervalClass | None = None
        lower = measurements["primary_bootstrap_lower"]
        upper = measurements["primary_bootstrap_upper"]
        if (lower is None) != (upper is None):
            raise ValueError("bootstrap interval endpoints must be jointly available")
        if lower is not None and upper is not None:
            interval_class = _classify_h4_interval(lower, upper)
            _validate_interval_invariant(self.invariants[16], interval_class, lower, upper)
        elif not anchor_fail:
            _require_unavailable_invariant(
                self.invariants[15], "primary_seed_level_inference", self.obligations
            )
            _require_unavailable_invariant(
                self.invariants[16], "primary_effect_threshold", self.obligations
            )

        comparison_states = (
            ("unresolved",) * 5
            if anchor_fail
            else tuple(_comparison_state(item) for item in self.invariants[8:13])
        )
        preconditions_pass = all(item.passed for item in self.invariants[:8])
        eligibility_pass = all(item.passed for item in self.invariants[13:16])
        comparisons_complete = all(state in ("pass", "miss") for state in comparison_states)
        has_equivalence_miss = any(state == "miss" for state in comparison_states)
        upstream_ambiguity = (
            not preconditions_pass
            or not eligibility_pass
            or not comparisons_complete
            or interval_class is None
            or any(value is None for value in measurements.values())
        )

        if self.status is GateStatus.PASS:
            if (
                self.obligations
                or not preconditions_pass
                or not eligibility_pass
                or comparison_states != ("pass",) * 5
                or interval_class != "support"
            ):
                raise ValueError("PASS requires complete support evidence")
        elif self.status is GateStatus.FAIL and not anchor_fail:
            if self.obligations or upstream_ambiguity:
                raise ValueError("post-timing FAIL requires complete eligible evidence")
            if has_equivalence_miss:
                pass
            elif comparison_states == ("pass",) * 5 and interval_class == "no_support":
                pass
            else:
                raise ValueError("post-timing FAIL requires a decisive equivalence miss or bootstrap no-support interval")
        elif self.status is GateStatus.INCONCLUSIVE:
            if not self.obligations or not any(not item.passed for item in self.invariants):
                raise ValueError("INCONCLUSIVE requires failed evidence and obligation")
            if not upstream_ambiguity:
                if has_equivalence_miss or interval_class in ("support", "no_support"):
                    raise ValueError("complete decisive evidence cannot be INCONCLUSIVE")
                expected_obligation = (
                    "primary_effect_threshold: bootstrap_interval_crosses_threshold"
                    if interval_class == "crossing"
                    else "primary_effect_threshold: bootstrap_interval_equals_threshold"
                )
                if self.obligations != (expected_obligation,):
                    raise ValueError("interval precision obligation is frozen")
            producers = {
                "primary_seed_ratio_geometric_mean": "primary_seed_level_inference",
                "primary_bootstrap_lower": "primary_seed_level_inference",
                "primary_bootstrap_upper": "primary_seed_level_inference",
                "primary_timed_ab_total": "primary_timed_order_balance",
                "primary_timed_ba_total": "primary_timed_order_balance",
                "maximum_solver_stopping_residual": "shared_protocol_identity",
                "maximum_allowance_scale_fraction": "all_equivalence_allowances_decisive",
            }
            for measurement, producer in producers.items():
                if measurements[measurement] is None:
                    _require_unavailable_invariant(
                        self.invariants[H4_INVARIANT_NAMES.index(producer)],
                        producer,
                        self.obligations,
                    )
        elif not anchor_fail:
            raise ValueError("invalid H4 gate status")
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
                _validate_applicable_allowance(copied)
            else:
                raise ValueError("allowance applicable flag is required")
            frozen[name] = copied  # type: ignore[assignment]
        if anchor_fail:
            if frozen["h3_anchor_identity"].get("applicable") is not True or any(frozen[name] != MappingProxyType({"applicable": False, "reason": _UNAVAILABLE_ANCHOR}) for name in H4_ALLOWANCE_INVARIANT_NAMES[1:]):
                raise ValueError("anchor failure allowance applicability is frozen")
        elif self.status is GateStatus.PASS or self.status is GateStatus.FAIL:
            if any(record.get("applicable") is not True for record in frozen.values()): raise ValueError("conclusive post-timing allowances must be applicable")
        else:
            for name, record in frozen.items():
                if record.get("applicable") is False:
                    item = self.invariants[H4_INVARIANT_NAMES.index(name)]
                    if record.get("reason") != _UNAVAILABLE_ELIGIBILITY or (item.passed,item.value,item.limit,item.detail) != (False,None,None,_UNAVAILABLE_ELIGIBILITY): raise ValueError("inapplicable inconclusive allowance requires matching eligibility sentinel")
        object.__setattr__(self, "measurements", MappingProxyType(measurements))
        object.__setattr__(self, "allowances_by_invariant", MappingProxyType(frozen))


def _classify_h4_interval(lower: float, upper: float) -> H4IntervalClass:
    """Classify the frozen positive bootstrap interval without status side effects."""

    if not (
        math.isfinite(lower)
        and math.isfinite(upper)
        and lower > 0.0
        and upper > 0.0
        and lower <= upper
    ):
        raise ValueError("bootstrap interval must be finite, positive, and ordered")
    if lower == H4_PRIMARY_EFFECT_THRESHOLD and upper == H4_PRIMARY_EFFECT_THRESHOLD:
        return "boundary"
    if upper <= H4_PRIMARY_EFFECT_THRESHOLD:
        return "support"
    if lower >= H4_PRIMARY_EFFECT_THRESHOLD:
        return "no_support"
    return "crossing"


def _validate_interval_invariant(
    invariant: InvariantResult,
    interval_class: H4IntervalClass,
    lower: float,
    upper: float,
) -> None:
    expected = {
        "support": (True, upper, H4_PRIMARY_EFFECT_THRESHOLD, "bootstrap_interval_supports_effect"),
        "no_support": (False, lower, H4_PRIMARY_EFFECT_THRESHOLD, "bootstrap_interval_excludes_support"),
        "crossing": (False, lower, H4_PRIMARY_EFFECT_THRESHOLD, "bootstrap_interval_crosses_threshold"),
        "boundary": (False, H4_PRIMARY_EFFECT_THRESHOLD, H4_PRIMARY_EFFECT_THRESHOLD, "bootstrap_interval_equals_threshold"),
    }[interval_class]
    if (invariant.passed, invariant.value, invariant.limit, invariant.detail) != expected:
        raise ValueError("primary_effect_threshold invariant must match bootstrap classifier")


def _comparison_state(invariant: InvariantResult) -> Literal["pass", "miss", "unresolved"]:
    if invariant.value is None or invariant.limit is None:
        if (
            invariant.passed is False
            and invariant.value is None
            and invariant.limit is None
            and invariant.detail == _UNAVAILABLE_ELIGIBILITY
        ):
            return "unresolved"
        raise ValueError("equivalence comparison availability is malformed")
    if invariant.value < 0.0 or invariant.limit < 0.0:
        raise ValueError("equivalence comparison residual and limit must be nonnegative")
    if invariant.passed:
        if invariant.value > invariant.limit:
            raise ValueError("passing equivalence comparison exceeds its limit")
        return "pass"
    if invariant.value <= invariant.limit:
        raise ValueError("failed equivalence comparison is not a decisive miss")
    return "miss"


def _require_unavailable_invariant(
    invariant: InvariantResult,
    producer: str,
    obligations: tuple[str, ...],
) -> None:
    if (
        invariant.passed,
        invariant.value,
        invariant.limit,
        invariant.detail,
    ) != (False, None, None, _UNAVAILABLE_ELIGIBILITY) or (
        f"{producer}: {_UNAVAILABLE_ELIGIBILITY}" not in obligations
    ):
        raise ValueError("inconclusive unavailable producer evidence is malformed")


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
        if not value: raise ValueError(f"{name} mappings must be nonempty")
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
    if type(value) is not tuple or any(type(item) is not int or item < 0 or item >= dimension for item in value) or len(set(value)) != len(value) or any(left >= right for left, right in zip(value, value[1:], strict=False)): raise ValueError(f"{name} must be strictly ascending unique indices in range")
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

def _validate_scaled_schedule(schedule: tuple[H4AffineGaussianFactor, ...], kind: H4ProblemKind, horizon: int, dimension: int) -> None:
    initial = schedule[0]
    if initial.role != "initial" or initial.time_index != 0 or initial.normalized_coordinate_indices != tuple(range(8)) or initial.parent_coordinate_indices or initial.target != (0.0,) * 8 or initial.covariance != tuple(tuple(1.0 if i == j else 0.0 for j in range(8)) for i in range(8)) or initial.matrix != tuple(tuple(1.0 if i == j else 0.0 for j in range(dimension)) for i in range(8)) or initial.raw_draws:
        raise ValueError("scaled initial factor is frozen")
    names = ("A_m", "A_z", "B", "c_m", "c_z", "R_m", "R_z", "G", "observation_offset", "observation_noise", "observed_target")
    shapes = ((4,4),(4,4),(4,4),(4,),(4,),(4,),(4,),(8,8),(8,),(8,),(8,))
    for time in range(1, horizon + 1):
        m_factor, z_factor, observation = schedule[1 + 3 * (time - 1):1 + 3 * time]
        z_prev = tuple(range((time - 1) * 8, (time - 1) * 8 + 4)); m_prev = tuple(range((time - 1) * 8 + 4, time * 8)); z_now = tuple(range(time * 8, time * 8 + 4)); m_now = tuple(range(time * 8 + 4, (time + 1) * 8))
        expected = (("transition", time, m_now, m_prev, (0,3,5)), ("transition", time, z_now, (*z_prev,*m_now), (1,2,4,6)), ("observation", time, (), (*z_now,*m_now), (7,8,9,10)))
        for factor, (role, index, normalized, parents, local_indices) in zip((m_factor,z_factor,observation), expected, strict=True):
            if (factor.role, factor.time_index, factor.normalized_coordinate_indices, factor.parent_coordinate_indices) != (role,index,normalized,parents): raise ValueError("scaled factor metadata is frozen")
            if tuple(draw.draw_index for draw in factor.raw_draws) != tuple(11*(time-1)+item for item in local_indices): raise ValueError("scaled draw ownership is frozen")
            if tuple(draw.name for draw in factor.raw_draws) != tuple(f"{names[item]}[{time}]" for item in local_indices) or tuple(draw.shape for draw in factor.raw_draws) != tuple(shapes[item] for item in local_indices): raise ValueError("scaled draw names/shapes are frozen")
        if kind == "zero_control" and (any(value != 0.0 for row in m_factor.matrix for value in (row[index] for index in m_factor.parent_coordinate_indices)) or any(value != 0.0 for row in z_factor.matrix for value in (row[index] for index in z_factor.parent_coordinate_indices))):
            raise ValueError("zero-control transition parent blocks must be zero")

def _validate_anchor_schedule(problem: H4NeutralProblem) -> None:
    expected_ids = ("z0_prior", "m0_prior", "m1_transition", "z1_transition", "z1_observation", "m1_observation")
    metadata = (("initial",0,(0,),()), ("initial",0,(1,),()), ("transition",1,(3,),(1,)), ("transition",1,(2,),(0,3)), ("observation",1,(),(2,)), ("observation",1,(),(3,)))
    expected_problem_id = "h4-anchor-h3-coupled-v1" if problem.kind == "coupled" else "h4-anchor-h3-zero-control-v1"
    if problem.problem_id != expected_problem_id or tuple(f.factor_id for f in problem.factor_schedule) != expected_ids:
        raise ValueError("H3 anchor identity/schedule is frozen")
    for factor, item in zip(problem.factor_schedule, metadata, strict=True):
        if (factor.role, factor.time_index, factor.normalized_coordinate_indices, factor.parent_coordinate_indices) != item or factor.raw_draws:
            raise ValueError("H3 anchor metadata/provenance is frozen")

def _validate_applicable_allowance(record: Mapping[str, H4JsonValue]) -> None:
    if tuple(record) != H4_APPLICABLE_ALLOWANCE_FIELDS or record["applicable"] is not True: raise ValueError("applicable allowance fields/order are frozen")
    if type(record["dimension"]) is not int or record["dimension"] <= 0: raise ValueError("allowance dimension must be positive")
    for name in ("operands", "absolute_summands", "condition_numbers", "operation_counts"):
        value = record[name]
        if type(value) is not tuple or not value: raise ValueError("allowance operand fields must be nonempty tuples")
    if any(type(value) not in (int,float) or not math.isfinite(float(value)) for value in record["operands"]): raise ValueError("operands must be finite")
    if any(type(value) not in (int,float) or not math.isfinite(float(value)) or float(value) < 0 for value in record["absolute_summands"]): raise ValueError("absolute summands must be nonnegative")
    if any(type(value) not in (int,float) or not math.isfinite(float(value)) or float(value) <= 0 for value in record["condition_numbers"]): raise ValueError("condition numbers must be positive")
    if any(type(value) is not int or value <= 0 for value in record["operation_counts"]): raise ValueError("operation counts must be positive ints")
    for name in ("solver_contribution", "final_allowance", "allowance_scale_ratio"):
        value = record[name]
        if type(value) not in (int,float) or not math.isfinite(float(value)) or float(value) < 0: raise ValueError("allowance values must be nonnegative finite")
    scale = record["invariant_scale"]
    if type(scale) not in (int,float) or not math.isfinite(float(scale)) or float(scale) <= 0: raise ValueError("invariant scale must be positive finite")
    expected = float(record["final_allowance"]) / float(scale); actual = float(record["allowance_scale_ratio"])
    if abs(actual - expected) > 64.0 * math.ulp(1.0) * max(1.0, abs(actual), abs(expected)): raise ValueError("allowance ratio is inconsistent")

__all__ = [name for name in globals() if name.startswith("H4")] + ["canonical_h4_problem_bytes", "h4_problem_core", "h4_problem_digest"]
