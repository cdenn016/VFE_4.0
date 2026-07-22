"""Dependency-free immutable records shared by H4 verification phases."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCALED_PROBLEM_ID = re.compile(
    r"h4-(?:coupled|zero_control)-T(7|15|31)-dz4-dm4-seed[1-9][0-9]*-v1\Z"
)
_ERROR_LIMIT = 512


def _identity(problem_id: str, problem_sha256: str) -> None:
    if type(problem_id) is not str or not problem_id:
        raise ValueError("problem_id must be a nonempty string")
    if type(problem_sha256) is not str or _SHA256.fullmatch(problem_sha256) is None:
        raise ValueError("problem_sha256 must be lowercase SHA-256")


def _finite(value: object, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _error(value: str | None, name: str) -> None:
    if value is not None and (type(value) is not str or not value or len(value) > _ERROR_LIMIT):
        raise ValueError(f"{name} must be None or a nonempty error capped at 512 code points")


@dataclass(frozen=True, slots=True)
class H4PosteriorConditionRecord:
    problem_id: str
    problem_sha256: str
    source: Literal["numpy_oracle", "information", "moment"]
    repetition_index: int | None
    dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    minimum_cholesky_pivot: float
    mean_infinity_norm: float
    finite: Literal[True]
    spd: Literal[True]
    eligible: bool

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.source not in ("numpy_oracle", "information", "moment"):
            raise ValueError("invalid posterior condition source")
        if self.source == "numpy_oracle":
            if self.repetition_index is not None:
                raise ValueError("oracle posterior has no repetition")
        elif type(self.repetition_index) is not int or self.repetition_index not in range(11):
            raise ValueError("retained posterior requires repetition 0..10")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("posterior dimension must be positive")
        for name in (
            "minimum_eigenvalue", "maximum_eigenvalue", "condition_number",
            "minimum_cholesky_pivot", "mean_infinity_norm",
        ):
            _finite(getattr(self, name), name)
        if (
            self.minimum_eigenvalue <= 0.0
            or self.maximum_eigenvalue < self.minimum_eigenvalue
            or self.condition_number < 1.0
            or self.minimum_cholesky_pivot <= 0.0
            or self.mean_infinity_norm < 0.0
            or self.finite is not True
            or self.spd is not True
            or type(self.eligible) is not bool
        ):
            raise ValueError("posterior condition record is malformed")


@dataclass(frozen=True, slots=True)
class H4InnovationConditionRecord:
    problem_id: str
    problem_sha256: str
    source: Literal["numpy_oracle", "moment"]
    repetition_index: int | None
    factor_id: str
    time_index: int
    parent_coordinate_indices: tuple[int, ...]
    innovation_dimension: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    finite: Literal[True]
    spd: Literal[True]
    eligible: bool

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.source not in ("numpy_oracle", "moment"):
            raise ValueError("invalid innovation condition source")
        if self.source == "numpy_oracle":
            if self.repetition_index is not None:
                raise ValueError("oracle innovation has no repetition")
        elif type(self.repetition_index) is not int or self.repetition_index not in range(11):
            raise ValueError("moment innovation requires repetition 0..10")
        if type(self.factor_id) is not str or not self.factor_id:
            raise ValueError("factor_id must be nonempty")
        if type(self.time_index) is not int or self.time_index <= 0:
            raise ValueError("innovation time_index must be positive")
        if (
            type(self.parent_coordinate_indices) is not tuple
            or not self.parent_coordinate_indices
            or any(type(item) is not int or item < 0 for item in self.parent_coordinate_indices)
            or any(left >= right for left, right in zip(self.parent_coordinate_indices, self.parent_coordinate_indices[1:], strict=False))
        ):
            raise ValueError("innovation parent indices must be strictly ascending")
        if type(self.innovation_dimension) is not int or self.innovation_dimension <= 0:
            raise ValueError("innovation dimension must be positive")
        for name in ("minimum_eigenvalue", "maximum_eigenvalue", "condition_number"):
            _finite(getattr(self, name), name)
        if (
            self.minimum_eigenvalue <= 0.0
            or self.maximum_eigenvalue < self.minimum_eigenvalue
            or self.condition_number < 1.0
            or self.finite is not True
            or self.spd is not True
            or type(self.eligible) is not bool
        ):
            raise ValueError("innovation condition record is malformed")


@dataclass(frozen=True, slots=True)
class H4ConditionWitness:
    metric: Literal[
        "minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number",
        "minimum_cholesky_pivot", "maximum_mean_infinity_norm", "first_ineligible",
    ]
    stream_index: int
    record: H4PosteriorConditionRecord | H4InnovationConditionRecord

    def __post_init__(self) -> None:
        if self.metric not in (
            "minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number",
            "minimum_cholesky_pivot", "maximum_mean_infinity_norm", "first_ineligible",
        ):
            raise ValueError("invalid condition witness metric")
        _nonnegative_int(self.stream_index, "stream_index")
        if type(self.record) not in (H4PosteriorConditionRecord, H4InnovationConditionRecord):
            raise ValueError("condition witness must own an exact condition record")
        if type(self.record) is H4InnovationConditionRecord and self.metric in (
            "minimum_cholesky_pivot", "maximum_mean_infinity_norm",
        ):
            raise ValueError("innovation witnesses cannot use posterior-only metrics")
        if self.metric == "first_ineligible" and self.record.eligible:
            raise ValueError("first_ineligible witness must be ineligible")


@dataclass(frozen=True, slots=True)
class H4ConditionStreamSummary:
    name: Literal["oracle_posterior", "terminal_posterior", "oracle_innovation", "moment_innovation"]
    stream_domain: Literal["vfe4.h4.condition-record-stream.v1"]
    expected_record_count: int
    observed_record_count: int
    record_stream_sha256: str
    eligible_record_count: int
    ineligible_record_count: int
    witnesses: tuple[H4ConditionWitness, ...]
    all_eligible: bool

    def __post_init__(self) -> None:
        expected_counts = {
            "oracle_posterior": 120, "terminal_posterior": 2640,
            "oracle_innovation": 2120, "moment_innovation": 23320,
        }
        if self.name not in expected_counts or self.stream_domain != "vfe4.h4.condition-record-stream.v1":
            raise ValueError("condition stream identity is frozen")
        if (
            self.expected_record_count != expected_counts[self.name]
            or self.observed_record_count != self.expected_record_count
            or type(self.eligible_record_count) is not int
            or type(self.ineligible_record_count) is not int
            or self.eligible_record_count < 0
            or self.ineligible_record_count < 0
            or self.eligible_record_count + self.ineligible_record_count != self.observed_record_count
        ):
            raise ValueError("condition stream counts are incomplete")
        if _SHA256.fullmatch(self.record_stream_sha256) is None:
            raise ValueError("condition stream digest must be lowercase SHA-256")
        if type(self.witnesses) is not tuple or not all(type(item) is H4ConditionWitness for item in self.witnesses):
            raise ValueError("condition witnesses must be an exact tuple")
        base_metrics = (
            ("minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number")
            if "innovation" in self.name else
            ("minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number", "minimum_cholesky_pivot", "maximum_mean_infinity_norm")
        )
        expected_metrics = base_metrics + (() if self.ineligible_record_count == 0 else ("first_ineligible",))
        if tuple(item.metric for item in self.witnesses) != expected_metrics:
            raise ValueError("condition witness order is frozen")
        if any(item.stream_index >= self.observed_record_count for item in self.witnesses):
            raise ValueError("condition witness index is outside the stream")
        if type(self.all_eligible) is not bool or self.all_eligible != (self.ineligible_record_count == 0):
            raise ValueError("all_eligible must equal the full-stream conjunction")


@dataclass(frozen=True, slots=True)
class H4ProblemConditionSummary:
    problem_id: str
    problem_sha256: str
    name: Literal["oracle_posterior", "terminal_posterior", "oracle_innovation", "moment_innovation"]
    stream_domain: Literal["vfe4.h4.problem-condition-record-stream.v1"]
    expected_record_count: int
    observed_record_count: int
    record_stream_sha256: str
    eligible_record_count: int
    ineligible_record_count: int
    witnesses: tuple[H4ConditionWitness, ...]
    all_eligible: bool

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        match = _SCALED_PROBLEM_ID.fullmatch(self.problem_id)
        anchor = self.problem_id in (
            "h4-anchor-h3-coupled-v1", "h4-anchor-h3-zero-control-v1",
        )
        if anchor:
            if self.name != "oracle_innovation":
                raise ValueError(
                    "anchor problem condition summary accepts only oracle_innovation"
                )
            expected_counts = {"oracle_innovation": 2}
        elif match is not None:
            horizon = int(match.group(1))
            expected_counts = {
                "oracle_posterior": 1,
                "terminal_posterior": 22,
                "oracle_innovation": horizon,
                "moment_innovation": 11 * horizon,
            }
        else:
            raise ValueError(
                "problem condition summary requires a scaled or exact anchor identity"
            )
        if (
            self.name not in expected_counts
            or self.stream_domain != "vfe4.h4.problem-condition-record-stream.v1"
        ):
            raise ValueError("problem condition stream identity is frozen")
        if (
            self.expected_record_count != expected_counts[self.name]
            or self.observed_record_count != self.expected_record_count
            or type(self.eligible_record_count) is not int
            or type(self.ineligible_record_count) is not int
            or self.eligible_record_count < 0
            or self.ineligible_record_count < 0
            or self.eligible_record_count + self.ineligible_record_count
            != self.observed_record_count
        ):
            raise ValueError("problem condition stream count is incomplete")
        if type(self.record_stream_sha256) is not str or _SHA256.fullmatch(self.record_stream_sha256) is None:
            raise ValueError("problem condition stream digest must be lowercase SHA-256")
        if (
            type(self.witnesses) is not tuple
            or not all(type(item) is H4ConditionWitness for item in self.witnesses)
        ):
            raise ValueError("problem condition witnesses must be an exact tuple")
        posterior = "posterior" in self.name
        base_metrics = (
            (
                "minimum_eigenvalue", "maximum_eigenvalue",
                "maximum_condition_number", "minimum_cholesky_pivot",
                "maximum_mean_infinity_norm",
            )
            if posterior
            else (
                "minimum_eigenvalue", "maximum_eigenvalue",
                "maximum_condition_number",
            )
        )
        expected_metrics = base_metrics + (
            () if self.ineligible_record_count == 0 else ("first_ineligible",)
        )
        if tuple(item.metric for item in self.witnesses) != expected_metrics:
            raise ValueError("problem condition witness order is frozen")
        expected_record_type = (
            H4PosteriorConditionRecord if posterior else H4InnovationConditionRecord
        )
        expected_source = {
            "oracle_posterior": "numpy_oracle",
            "terminal_posterior": None,
            "oracle_innovation": "numpy_oracle",
            "moment_innovation": "moment",
        }[self.name]
        for witness in self.witnesses:
            record = witness.record
            if (
                type(record) is not expected_record_type
                or record.problem_id != self.problem_id
                or record.problem_sha256 != self.problem_sha256
                or witness.stream_index >= self.observed_record_count
                or (
                    expected_source is None
                    and getattr(record, "source") not in ("information", "moment")
                )
                or (
                    expected_source is not None
                    and getattr(record, "source") != expected_source
                )
            ):
                raise ValueError("problem condition witness identity does not match its stream")
        if type(self.all_eligible) is not bool or self.all_eligible != (
            self.ineligible_record_count == 0
        ):
            raise ValueError("problem all_eligible must equal the full-stream conjunction")


@dataclass(frozen=True, slots=True)
class H4CoverageRecord:
    name: Literal[
        "oracle_posterior", "terminal_posterior", "oracle_innovation", "moment_innovation",
        "native_replay", "operation_pass", "memory_pass", "execution_trace", "postflight_schedule",
    ]
    key_stream_domain: Literal["vfe4.h4.coverage-key-stream.v1"]
    expected_key_count: int
    observed_key_count: int
    expected_key_stream_sha256: str
    observed_key_stream_sha256: str
    missing_key_count: int
    extra_key_count: int
    duplicate_key_count: int
    first_missing_key: str | None
    first_extra_key: str | None
    first_duplicate_key: str | None
    complete: bool

    def __post_init__(self) -> None:
        counts = {
            "oracle_posterior": 120, "terminal_posterior": 2640,
            "oracle_innovation": 2120, "moment_innovation": 23320,
            "native_replay": 2640, "operation_pass": 240, "memory_pass": 240,
            "execution_trace": 120, "postflight_schedule": 146720,
        }
        if self.name not in counts or self.key_stream_domain != "vfe4.h4.coverage-key-stream.v1":
            raise ValueError("coverage identity is frozen")
        if type(self.expected_key_count) is not int or self.expected_key_count != counts[self.name]:
            raise ValueError("coverage expected count is frozen")
        _nonnegative_int(self.observed_key_count, "observed_key_count")
        if _SHA256.fullmatch(self.expected_key_stream_sha256) is None or _SHA256.fullmatch(self.observed_key_stream_sha256) is None:
            raise ValueError("coverage digests must be lowercase SHA-256")
        for name in ("missing_key_count", "extra_key_count", "duplicate_key_count"):
            _nonnegative_int(getattr(self, name), name)
        for count, witness, name in (
            (self.missing_key_count, self.first_missing_key, "first_missing_key"),
            (self.extra_key_count, self.first_extra_key, "first_extra_key"),
            (self.duplicate_key_count, self.first_duplicate_key, "first_duplicate_key"),
        ):
            if (count == 0) != (witness is None) or (witness is not None and (type(witness) is not str or not witness)):
                raise ValueError(f"{name} must agree with its discrepancy count")
        expected_complete = (
            self.observed_key_count == self.expected_key_count
            and self.expected_key_stream_sha256 == self.observed_key_stream_sha256
            and self.missing_key_count == self.extra_key_count == self.duplicate_key_count == 0
        )
        if type(self.complete) is not bool or self.complete != expected_complete:
            raise ValueError("coverage completeness must match counts, digests, and discrepancies")


@dataclass(frozen=True, slots=True)
class H4ArmCallSpan:
    problem_id: str
    phase: Literal["warmup", "timed"]
    pair_index: int
    repetition_index: int | None
    order: Literal["information_then_moment", "moment_then_information"]
    order_position: Literal[0, 1]
    arm: Literal["information", "moment"]
    start_nanoseconds: int
    end_nanoseconds: int
    duration_nanoseconds: int

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id:
            raise ValueError("span problem_id must be nonempty")
        if self.phase == "warmup":
            if type(self.pair_index) is not int or self.pair_index not in range(3) or self.repetition_index is not None:
                raise ValueError("warmup span identity is invalid")
        elif self.phase == "timed":
            if type(self.repetition_index) is not int or self.repetition_index not in range(11) or self.pair_index != 3 + self.repetition_index:
                raise ValueError("timed span identity is invalid")
        else:
            raise ValueError("span phase is invalid")
        if self.order not in ("information_then_moment", "moment_then_information") or self.order_position not in (0, 1) or self.arm not in ("information", "moment"):
            raise ValueError("span order/arm identity is invalid")
        expected_arm = (
            ("information", "moment") if self.order == "information_then_moment"
            else ("moment", "information")
        )[self.order_position]
        if self.arm != expected_arm:
            raise ValueError("span arm does not match order position")
        for name in ("start_nanoseconds", "end_nanoseconds", "duration_nanoseconds"):
            _nonnegative_int(getattr(self, name), name)
        if self.end_nanoseconds < self.start_nanoseconds or self.duration_nanoseconds != self.end_nanoseconds - self.start_nanoseconds or self.duration_nanoseconds <= 0:
            raise ValueError("span duration must equal its positive timer difference")


@dataclass(frozen=True, slots=True)
class H4PostflightEventKey:
    problem_id: str
    problem_sha256: str
    event_index: int
    phase: Literal[
        "materialized_integrity", "terminal_conversion", "native_diagnostic_replay",
        "terminal_posterior_condition", "moment_innovation_condition", "oracle_rehydration",
        "oracle_route_agreement", "equivalence_group", "operation_pass", "memory_pass",
        "stream_compaction",
    ]
    repetition_index: int | None
    arm: Literal["information", "moment"] | None
    factor_id: str | None
    selected_moment_name: str | None
    equivalence_component: Literal[
        "kl_to_zero", "h", "J", "selected_mean", "selected_covariance", "objective",
    ] | None
    integrity_phase: Literal["after_timed_batch", "after_postflight"] | None

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        _nonnegative_int(self.event_index, "event_index")
        phases = (
            "materialized_integrity", "terminal_conversion", "native_diagnostic_replay",
            "terminal_posterior_condition", "moment_innovation_condition", "oracle_rehydration",
            "oracle_route_agreement", "equivalence_group", "operation_pass", "memory_pass",
            "stream_compaction",
        )
        if self.phase not in phases:
            raise ValueError("unknown H4 postflight phase")
        optionals = (
            self.repetition_index, self.arm, self.factor_id, self.selected_moment_name,
            self.equivalence_component, self.integrity_phase,
        )
        if self.phase == "materialized_integrity":
            if self.integrity_phase not in ("after_timed_batch", "after_postflight") or any(item is not None for item in optionals[:-1]):
                raise ValueError("integrity postflight key is malformed")
        elif self.phase in ("terminal_conversion", "native_diagnostic_replay", "terminal_posterior_condition"):
            if type(self.repetition_index) is not int or self.repetition_index not in range(11) or self.arm not in ("information", "moment") or any(item is not None for item in optionals[2:]):
                raise ValueError("terminal postflight key is malformed")
        elif self.phase == "moment_innovation_condition":
            if type(self.repetition_index) is not int or self.repetition_index not in range(11) or self.arm != "moment" or type(self.factor_id) is not str or not self.factor_id or any(item is not None for item in optionals[3:]):
                raise ValueError("moment innovation postflight key is malformed")
        elif self.phase in ("oracle_rehydration", "oracle_route_agreement", "stream_compaction"):
            if any(item is not None for item in optionals):
                raise ValueError("oracle/compaction postflight key has fake discriminators")
        elif self.phase == "equivalence_group":
            if type(self.repetition_index) is not int or self.repetition_index not in range(11) or self.arm not in ("information", "moment") or self.equivalence_component not in ("kl_to_zero", "h", "J", "selected_mean", "selected_covariance", "objective") or self.factor_id is not None or self.integrity_phase is not None:
                raise ValueError("equivalence postflight key is malformed")
            selected = self.equivalence_component in ("selected_mean", "selected_covariance")
            if selected != (type(self.selected_moment_name) is str and bool(self.selected_moment_name)):
                raise ValueError("selected equivalence keys require exactly one selected label")
        else:
            if self.arm not in ("information", "moment") or any(item is not None for item in (self.repetition_index, self.factor_id, self.selected_moment_name, self.equivalence_component, self.integrity_phase)):
                raise ValueError("operation/memory postflight key is malformed")


@dataclass(frozen=True, slots=True)
class H4PostflightTimingWitness:
    event: H4PostflightEventKey
    timed_batch_end_nanoseconds: int
    start_nanoseconds: int
    end_nanoseconds: int

    def __post_init__(self) -> None:
        if type(self.event) is not H4PostflightEventKey:
            raise ValueError("timing witness requires an exact event")
        for name in ("timed_batch_end_nanoseconds", "start_nanoseconds", "end_nanoseconds"):
            _nonnegative_int(getattr(self, name), name)
        if self.end_nanoseconds < self.start_nanoseconds:
            raise ValueError("postflight witness timer interval is reversed")


@dataclass(frozen=True, slots=True)
class H4PostflightScheduleSummary:
    stream_domain: Literal["vfe4.h4.postflight-event-key-stream.v1"]
    expected_event_count: int
    observed_event_count: int
    expected_key_stream_sha256: str
    observed_key_stream_sha256: str
    first_mismatch_index: int | None
    first_expected_key: H4PostflightEventKey | None
    first_observed_key: H4PostflightEventKey | None
    timing_violation_count: int
    first_timing_violation: H4PostflightTimingWitness | None
    complete: bool

    def __post_init__(self) -> None:
        if self.stream_domain != "vfe4.h4.postflight-event-key-stream.v1":
            raise ValueError("postflight stream domain is frozen")
        _nonnegative_int(self.expected_event_count, "expected_event_count")
        _nonnegative_int(self.observed_event_count, "observed_event_count")
        if self.expected_event_count == 0:
            raise ValueError("postflight schedule must be nonempty")
        if _SHA256.fullmatch(self.expected_key_stream_sha256) is None or _SHA256.fullmatch(self.observed_key_stream_sha256) is None:
            raise ValueError("postflight stream digests must be lowercase SHA-256")
        mismatch = self.first_mismatch_index is not None
        if mismatch:
            _nonnegative_int(self.first_mismatch_index, "first_mismatch_index")
            if self.first_expected_key is None and self.first_observed_key is None:
                raise ValueError("postflight mismatch requires a key witness")
        elif self.first_expected_key is not None or self.first_observed_key is not None:
            raise ValueError("postflight key witness requires a mismatch index")
        for key in (self.first_expected_key, self.first_observed_key):
            if key is not None and type(key) is not H4PostflightEventKey:
                raise ValueError("postflight key witness must be exact")
        _nonnegative_int(self.timing_violation_count, "timing_violation_count")
        if (self.timing_violation_count == 0) != (self.first_timing_violation is None):
            raise ValueError("postflight timing witness must match violation count")
        if self.first_timing_violation is not None and type(self.first_timing_violation) is not H4PostflightTimingWitness:
            raise ValueError("postflight timing witness must be exact")
        expected_complete = (
            self.expected_event_count == self.observed_event_count
            and self.expected_key_stream_sha256 == self.observed_key_stream_sha256
            and not mismatch and self.timing_violation_count == 0
        )
        if type(self.complete) is not bool or self.complete != expected_complete:
            raise ValueError("postflight completeness is inconsistent")


@dataclass(frozen=True, slots=True)
class H4GarbageCollectorRecord:
    problem_id: str
    capture_attempted: Literal[True]
    capture_error: str | None
    prior_enabled: bool | None
    disable_required: bool | None
    disable_attempted: bool
    disable_error: str | None
    effective_state_capture_error: str | None
    disabled_during_batch: bool | None
    restore_attempted: bool
    restored_enabled: bool | None
    restoration_error: str | None
    restored_exact_prior_state: bool

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id or self.capture_attempted is not True:
            raise ValueError("GC capture identity is malformed")
        for name in ("capture_error", "disable_error", "effective_state_capture_error", "restoration_error"):
            _error(getattr(self, name), name)
        for name in ("disable_attempted", "restore_attempted", "restored_exact_prior_state"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("GC action fields must be booleans")
        if self.capture_error is not None:
            if (
                any(value is not None for value in (
                    self.prior_enabled, self.disable_required, self.disable_error,
                    self.effective_state_capture_error, self.disabled_during_batch,
                    self.restored_enabled, self.restoration_error,
                ))
                or self.disable_attempted
                or self.restore_attempted
                or self.restored_exact_prior_state
            ):
                raise ValueError("failed GC capture must suppress mutation and timing")
            return
        if type(self.prior_enabled) is not bool or self.disable_required != self.prior_enabled:
            raise ValueError("GC disable requirement must equal the prior enabled state")
        if self.disable_attempted != self.disable_required:
            raise ValueError("GC disable attempt must follow the captured prior state")
        if self.disable_error is not None:
            if (
                not self.disable_attempted
                or self.effective_state_capture_error is not None
                or self.disabled_during_batch is not None
            ):
                raise ValueError("GC disable failure cannot fabricate effective-state evidence")
        elif self.effective_state_capture_error is None:
            if type(self.disabled_during_batch) is not bool:
                raise ValueError("successful GC effective-state capture requires a boolean")
        elif self.disabled_during_batch is not None:
            raise ValueError("GC effective-state failure cannot retain a state value")
        if not self.restore_attempted:
            raise ValueError("known GC prior state must always be restored")
        if self.restoration_error is None:
            if type(self.restored_enabled) is not bool:
                raise ValueError("successful GC restoration requires an observed state")
        elif self.restored_enabled is not None:
            raise ValueError("GC restoration failure cannot retain a restored state")
        expected_restored = (
            self.restoration_error is None
            and self.restored_enabled is self.prior_enabled
        )
        if self.restored_exact_prior_state != expected_restored:
            raise ValueError("GC restoration result is inconsistent")


@dataclass(frozen=True, slots=True)
class H4ExecutionTrace:
    problem_id: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    warmup_spans: tuple[H4ArmCallSpan, ...]
    timed_batch_start_nanoseconds: int
    timed_spans: tuple[H4ArmCallSpan, ...]
    timed_batch_end_nanoseconds: int
    postflight_schedule: H4PostflightScheduleSummary
    garbage_collector: H4GarbageCollectorRecord
    warmups_count_toward_balance: Literal[False]
    timed_guard_violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id:
            raise ValueError("trace problem_id must be nonempty")
        for name, value, valid in (
            ("horizon_index", self.horizon_index, range(3)),
            ("seed_index", self.seed_index, range(20)),
            ("kind_index", self.kind_index, range(2)),
        ):
            if type(value) is not int or value not in valid:
                raise ValueError(f"invalid trace {name}")
        if type(self.problem_index) is not int or self.problem_index != ((self.horizon_index * 20 + self.seed_index) * 2 + self.kind_index):
            raise ValueError("trace problem_index must follow traversal")
        if type(self.warmup_spans) is not tuple or len(self.warmup_spans) != 6 or type(self.timed_spans) is not tuple or len(self.timed_spans) != 22:
            raise ValueError("complete trace requires six warmup and 22 timed spans")
        if not all(type(span) is H4ArmCallSpan and span.problem_id == self.problem_id for span in (*self.warmup_spans, *self.timed_spans)):
            raise ValueError("trace spans must be exact and belong to the problem")
        for spans, pairs, phase in (
            (self.warmup_spans, range(3), "warmup"),
            (self.timed_spans, range(3, 14), "timed"),
        ):
            for offset, pair_index in enumerate(pairs):
                pair = spans[2 * offset:2 * offset + 2]
                expected_order = (
                    "information_then_moment"
                    if (self.horizon_index + self.seed_index + self.kind_index + pair_index) % 2 == 0
                    else "moment_then_information"
                )
                if tuple((item.phase, item.pair_index, item.order, item.order_position) for item in pair) != (
                    (phase, pair_index, expected_order, 0),
                    (phase, pair_index, expected_order, 1),
                ):
                    raise ValueError("trace span pair/order identity is inconsistent")
        _nonnegative_int(self.timed_batch_start_nanoseconds, "timed_batch_start_nanoseconds")
        _nonnegative_int(self.timed_batch_end_nanoseconds, "timed_batch_end_nanoseconds")
        if self.timed_batch_end_nanoseconds < self.timed_batch_start_nanoseconds or any(
            span.start_nanoseconds < self.timed_batch_start_nanoseconds
            or span.end_nanoseconds > self.timed_batch_end_nanoseconds
            for span in self.timed_spans
        ):
            raise ValueError("timed spans must lie inside the timed batch")
        if type(self.postflight_schedule) is not H4PostflightScheduleSummary or not self.postflight_schedule.complete:
            raise ValueError("complete trace requires complete postflight schedule")
        if self.postflight_schedule.first_timing_violation is not None:
            raise ValueError("complete trace cannot have a postflight timing violation")
        if (
            type(self.garbage_collector) is not H4GarbageCollectorRecord
            or self.garbage_collector.problem_id != self.problem_id
            or self.garbage_collector.capture_error is not None
            or self.garbage_collector.disable_error is not None
            or self.garbage_collector.effective_state_capture_error is not None
            or self.garbage_collector.disabled_during_batch is not True
            or self.garbage_collector.restoration_error is not None
            or not self.garbage_collector.restored_exact_prior_state
        ):
            raise ValueError("complete trace requires successful GC timing and restoration")
        if self.warmups_count_toward_balance is not False:
            raise ValueError("warmups never count toward balance")
        if type(self.timed_guard_violations) is not tuple or any(type(item) is not str or not item for item in self.timed_guard_violations):
            raise ValueError("timed guard violations must be an immutable string tuple")


@dataclass(frozen=True, slots=True)
class H4ThreadStateRecord:
    capture_error: str | None
    prior_intra_op_threads: int | None
    set_attempted: bool
    set_error: str | None
    effective_intra_op_threads: int | None
    verified_one: bool
    prior_inter_op_threads: int | None
    final_inter_op_threads: int | None
    inter_op_unchanged: bool
    restore_attempted: bool
    restored_intra_op_threads: int | None
    restoration_error: str | None
    restored_exact_prior_state: bool

    def __post_init__(self) -> None:
        for name in ("capture_error", "set_error", "restoration_error"):
            _error(getattr(self, name), name)
        for name in ("set_attempted", "verified_one", "inter_op_unchanged", "restore_attempted", "restored_exact_prior_state"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("thread action fields must be booleans")
        for name in (
            "prior_intra_op_threads", "effective_intra_op_threads", "prior_inter_op_threads",
            "final_inter_op_threads", "restored_intra_op_threads",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when available")
        if self.capture_error is not None:
            if (
                any(value is not None for value in (
                    self.set_error, self.effective_intra_op_threads,
                    self.final_inter_op_threads, self.restored_intra_op_threads,
                    self.restoration_error,
                ))
                or self.prior_inter_op_threads is not None
                or self.set_attempted
                or self.restore_attempted
                or self.verified_one
                or self.inter_op_unchanged
                or self.restored_exact_prior_state
            ):
                raise ValueError(
                    "thread partial capture may retain only phase-valid priors and must suppress mutation"
                )
            return
        if self.prior_intra_op_threads is None or self.prior_inter_op_threads is None:
            raise ValueError("successful thread capture requires both prior counts")
        if not self.set_attempted or not self.restore_attempted:
            raise ValueError("thread set and restoration must be attempted after capture")
        if self.verified_one != (self.set_error is None and self.effective_intra_op_threads == 1):
            raise ValueError("verified_one is inconsistent")
        if self.set_error is None and self.effective_intra_op_threads != 1:
            raise ValueError("successful thread set requires an observed single thread")
        if self.set_error is not None and self.effective_intra_op_threads == 1:
            raise ValueError("thread set failure cannot coexist with successful evidence")
        if self.inter_op_unchanged != (
            self.final_inter_op_threads is not None
            and self.final_inter_op_threads == self.prior_inter_op_threads
        ):
            raise ValueError("inter-op threads must remain unchanged")
        if self.restoration_error is None:
            if self.restored_intra_op_threads is None:
                raise ValueError("successful thread restoration requires an observed count")
        elif self.restored_intra_op_threads is not None:
            raise ValueError("thread restoration failure cannot retain a restored count")
        expected_restored = (
            self.restoration_error is None
            and self.restored_intra_op_threads == self.prior_intra_op_threads
        )
        if self.restored_exact_prior_state != expected_restored:
            raise ValueError("thread restoration result is inconsistent")


__all__ = [name for name in globals() if name.startswith("H4")]
