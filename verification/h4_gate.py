"""H4 information-form cost gate, compact evidence records, and payload authority.

The timed region in this module contains only fresh native solver calls, timer
reads, and assignments into preallocated slots.  Every digest, conversion,
diagnostic, comparison, record construction, and serialization step is
deliberately outside that region.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import time
from contextlib import redirect_stdout
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Literal, Protocol, TypeAlias

import numpy as np
import torch

from vfe4.artifacts.provenance import process_cpu_affinity
from verification.h4_budget import (
    H4AllowanceResultSource,
    H4AnchorAllowanceSource,
    H4SixInvariantAllowanceAccumulator,
    innovation_condition_record,
    posterior_condition_record,
)
from verification.h4_records import (
    H4ArmCallSpan,
    H4ConditionStreamSummary,
    H4ConditionWitness,
    H4CoverageRecord,
    H4ExecutionTrace,
    H4GarbageCollectorRecord,
    H4InnovationConditionRecord,
    H4PostflightEventKey,
    H4PostflightScheduleSummary,
    H4PostflightTimingWitness,
    H4PosteriorConditionRecord,
    H4ProblemConditionSummary,
    H4ThreadStateRecord,
)
from verification.h4_statistics import (
    H4BootstrapInterval,
    H4PrimaryTimedOrderBalance,
    H4TimingSummary,
    decide_h4_interval,
    paired_log_bootstrap_interval,
    summarize_primary_timed_order,
    summarize_seed_ratios,
)
from verification.numpy_oracles.h4_gaussian import (
    H4OracleEvaluation,
    H4OracleKLEvaluation,
    H4OracleRouteAgreement,
    evaluate_h4_oracle,
    reverse_kl_to_h4_oracle,
)
from vfe4.config.schema import H4ValidationConfig
from vfe4.generative.reference_h4 import h4_anchor_from_h3, make_h4_problem
from vfe4.inference.h4_instrumentation import (
    CountingOperationRecorder,
    InstrumentedLinearAlgebra,
    NullOperationRecorder,
    measure_untimed_memory,
)
from vfe4.inference.h4_solvers import (
    H4InnovationDiagnostic,
    H4MaterializedProblem,
    H4NativeDiagnostics,
    _assert_h4_materialized_integrity,
    evaluate_h4_native_diagnostics,
    materialize_h4_problem,
    solve_information_form,
    solve_moment_form,
    to_common_terminal_law,
)
from vfe4.validation.h3_fixture import parse_h3_fixture_bytes
from vfe4.types.h4 import (
    H4_ALLOWANCE_INVARIANT_NAMES,
    H4_INVARIANT_NAMES,
    H4_MEASUREMENT_NAMES,
    H4ApplicableAllowance,
    H4AllowanceRecord,
    H4GateResult,
    H4InapplicableAllowance,
    H4IntervalDecision,
    H4MemoryRecord,
    H4NeutralProblem,
    H4OperationRecord,
    H4SelectedMoment,
    H4SolveProtocol,
    H4SolverArm,
    H4SolverResult,
    H4TerminalLaw,
    H4TimingRecord,
    canonical_h4_problem_bytes,
)
from vfe4.types.results import GateStatus, InvariantResult


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCALED_ID = re.compile(
    r"h4-(coupled|zero_control)-T(7|15|31)-dz4-dm4-seed([1-9][0-9]*)-v1\Z"
)
_ERROR_LIMIT = 512
_POWER_VALUE_LIMIT = 4096
_MAX_PAYLOAD_BYTES = 67_108_864
_POSTFLIGHT_DOMAIN = b"vfe4.h4.postflight-event-key-stream.v1\x00"
_STREAM_DOMAINS = (
    "vfe4.h4.oracle-evaluation-stream.v1",
    "vfe4.h4.native-result-stream.v1",
    "vfe4.h4.terminal-law-stream.v1",
    "vfe4.h4.native-diagnostic-stream.v1",
)
_CONDITION_NAMES = (
    "oracle_posterior", "terminal_posterior",
    "oracle_innovation", "moment_innovation",
)
_COVERAGE_NAMES = (
    "oracle_posterior", "terminal_posterior", "oracle_innovation",
    "moment_innovation", "native_replay", "operation_pass", "memory_pass",
    "execution_trace", "postflight_schedule",
)
_POWER_NAMES = (
    "active_power_scheme", "cpu_frequency_governor",
    "energy_performance_preference", "low_power_mode",
)
_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
_INCOMPLETE_OBLIGATIONS = {
    "warmup": "complete all six H4 warmup arm calls without exception",
    "gc_capture": "capture cyclic GC state before H4 timing",
    "gc_disable": "disable and verify cyclic GC before H4 timing",
    "timed_batch": "complete all 22 H4 timed arm calls and restore process-global state",
    "gc_restore": "restore exact prior cyclic GC state after H4 timing",
    "postflight": "complete exact H4 postflight schedule and release full problem objects",
}

H4MaterializedIntegrityPhase: TypeAlias = Literal[
    "after_materialization", "after_anchor_information", "after_anchor_moment",
    "before_timed_batch", "after_timed_batch", "after_postflight",
]
H4ScaledMaterializedIntegrityCheckpoint: TypeAlias = Literal[
    "after_materialization", "before_timed_batch", "after_timed_batch", "after_postflight",
]


def _sha(value: object, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _identity(problem_id: object, problem_sha256: object) -> None:
    if type(problem_id) is not str or not problem_id:
        raise ValueError("problem_id must be a nonempty string")
    _sha(problem_sha256, "problem_sha256")


def _finite(value: object, name: str, *, nonnegative: bool = False) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _strict_tuple(value: object, expected_type: type, name: str) -> tuple[object, ...]:
    if type(value) is not tuple or not all(type(item) is expected_type for item in value):
        raise ValueError(f"{name} must be an exact immutable tuple")
    return value


def _stable_error(error: BaseException) -> str:
    text = (
        f"{type(error).__module__}.{type(error).__qualname__}: {error}"
        .replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "\ufffd")
    )
    return text[:_ERROR_LIMIT]


def _bounded_text(value: object, name: str, *, maximum: int = _ERROR_LIMIT) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{name} must be nonempty and capped at {maximum} code points")
    return value


@dataclass(frozen=True, slots=True)
class H4MaterializedIntegrityCheck:
    phase: H4MaterializedIntegrityPhase
    expected_tensor_sha256: str
    observed_tensor_sha256: str
    exact_match: Literal[True]

    def __post_init__(self) -> None:
        if self.phase not in (
            "after_materialization", "after_anchor_information", "after_anchor_moment",
            "before_timed_batch", "after_timed_batch", "after_postflight",
        ):
            raise ValueError("invalid H4 materialized integrity phase")
        _sha(self.expected_tensor_sha256, "expected_tensor_sha256")
        _sha(self.observed_tensor_sha256, "observed_tensor_sha256")
        if self.exact_match is not True or self.observed_tensor_sha256 != self.expected_tensor_sha256:
            raise ValueError("completed integrity check requires exact digest equality")


@dataclass(frozen=True, slots=True)
class H4MaterializationIdentity:
    problem_id: str
    problem_sha256: str
    materialization_version: Literal["h4-materialized-problem-v1"]
    protocol_id: Literal["h4-single-pass-v1"]
    tensor_sha256: str
    materialization_count: Literal[1]
    shared_by_identity: Literal[True]
    integrity_checks: tuple[H4MaterializedIntegrityCheck, ...]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        _sha(self.tensor_sha256, "tensor_sha256")
        if (
            self.materialization_version != "h4-materialized-problem-v1"
            or self.protocol_id != "h4-single-pass-v1"
            or self.materialization_count != 1
            or self.shared_by_identity is not True
        ):
            raise ValueError("materialization identity is frozen")
        checks = _strict_tuple(self.integrity_checks, H4MaterializedIntegrityCheck, "integrity_checks")
        phases = tuple(item.phase for item in checks)
        valid = (
            (),
            ("after_materialization",),
            ("after_materialization", "after_anchor_information"),
            ("after_materialization", "after_anchor_information", "after_anchor_moment"),
            ("after_materialization", "before_timed_batch"),
            ("after_materialization", "before_timed_batch", "after_timed_batch"),
            ("after_materialization", "before_timed_batch", "after_timed_batch", "after_postflight"),
        )
        if phases not in valid:
            raise ValueError("integrity checks are not a canonical phase prefix")
        if any(
            item.expected_tensor_sha256 != self.tensor_sha256
            or item.observed_tensor_sha256 != self.tensor_sha256
            for item in checks
        ):
            raise ValueError("integrity check digest must match enclosing materialization")


@dataclass(frozen=True, slots=True)
class H4CanonicalStreamDigest:
    domain: Literal[
        "vfe4.h4.oracle-evaluation-stream.v1",
        "vfe4.h4.native-result-stream.v1",
        "vfe4.h4.terminal-law-stream.v1",
        "vfe4.h4.native-diagnostic-stream.v1",
    ]
    record_count: int
    scalar_count: int
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.domain not in _STREAM_DOMAINS:
            raise ValueError("unknown H4 canonical stream domain")
        for name in ("record_count", "scalar_count", "byte_count"):
            _nonnegative_int(getattr(self, name), name)
        if self.byte_count <= len(self.domain) + 1:
            raise ValueError("canonical stream must contain encoded evidence")
        _sha(self.sha256, "sha256")


@dataclass(frozen=True, slots=True)
class H4SelectedMomentSummary:
    name: str
    coordinate_indices: tuple[int, ...]
    dimension: int
    mean_scalar_count: int
    mean_sha256: str
    mean_infinity_norm: float
    covariance_scalar_count: int
    covariance_sha256: str
    covariance_trace: float
    covariance_maximum_absolute_value: float

    def __post_init__(self) -> None:
        _bounded_text(self.name, "selected moment name")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("selected dimension must be positive")
        if (
            type(self.coordinate_indices) is not tuple
            or len(self.coordinate_indices) != self.dimension
            or any(type(item) is not int or item < 0 for item in self.coordinate_indices)
            or any(left >= right for left, right in zip(self.coordinate_indices, self.coordinate_indices[1:]))
        ):
            raise ValueError("selected coordinate indices must be strictly ascending")
        if self.mean_scalar_count != self.dimension or self.covariance_scalar_count != self.dimension ** 2:
            raise ValueError("selected scalar counts do not match dimension")
        _sha(self.mean_sha256, "mean_sha256")
        _sha(self.covariance_sha256, "covariance_sha256")
        for name in (
            "mean_infinity_norm", "covariance_trace", "covariance_maximum_absolute_value",
        ):
            _finite(getattr(self, name), name, nonnegative=True)


@dataclass(frozen=True, slots=True)
class H4CompactKLSummary:
    value: float
    trace_term: float
    quadratic_mean_term: float
    minus_dimension_term: float
    candidate_logdet_precision_term: float
    minus_oracle_logdet_precision_term: float
    absolute_summand_accumulation: float
    candidate_condition_number: float
    oracle_condition_number: float
    operation_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in tuple(self.__dataclass_fields__)[:-1]:
            _finite(getattr(self, name), name)
        if (
            self.absolute_summand_accumulation < 0.0
            or self.candidate_condition_number < 1.0
            or self.oracle_condition_number < 1.0
            or type(self.operation_counts) is not tuple
            or any(
                type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
                or not item[0] or type(item[1]) is not int or item[1] < 0
                for item in self.operation_counts
            )
            or len({item[0] for item in self.operation_counts}) != len(self.operation_counts)
        ):
            raise ValueError("compact KL evidence is malformed")


@dataclass(frozen=True, slots=True)
class H4CompactResultRecord:
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    repetition_index: int | None
    arm: H4SolverArm
    native_stream: H4CanonicalStreamDigest
    terminal_stream: H4CanonicalStreamDigest
    oracle_kl_q_to_p: H4CompactKLSummary
    native_complete_objective: float
    terminal_complete_objective: float
    stopping_residual: float
    selected_moments: tuple[H4SelectedMomentSummary, ...]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.source_kind == "scaled_pcg64":
            if type(self.repetition_index) is not int or self.repetition_index not in range(11):
                raise ValueError("scaled compact result requires repetition 0..10")
        elif self.source_kind == "h3_anchor":
            if self.repetition_index is not None:
                raise ValueError("anchor compact result has no timed repetition")
        else:
            raise ValueError("invalid compact result source")
        if self.arm not in ("information", "moment"):
            raise ValueError("invalid compact result arm")
        if (
            type(self.native_stream) is not H4CanonicalStreamDigest
            or self.native_stream.domain != "vfe4.h4.native-result-stream.v1"
            or type(self.terminal_stream) is not H4CanonicalStreamDigest
            or self.terminal_stream.domain != "vfe4.h4.terminal-law-stream.v1"
            or type(self.oracle_kl_q_to_p) is not H4CompactKLSummary
        ):
            raise ValueError("compact result requires exact stream and KL records")
        for name in (
            "native_complete_objective", "terminal_complete_objective", "stopping_residual",
        ):
            _finite(getattr(self, name), name, nonnegative=name == "stopping_residual")
        selected = _strict_tuple(self.selected_moments, H4SelectedMomentSummary, "selected_moments")
        expected_count = 3 if self.source_kind == "h3_anchor" else int(_SCALED_ID.fullmatch(self.problem_id).group(2)) + 2 if _SCALED_ID.fullmatch(self.problem_id) else -1
        if len(selected) != expected_count or len({item.name for item in selected}) != len(selected):
            raise ValueError("compact result selected moment schedule is incomplete")


@dataclass(frozen=True, slots=True)
class H4CompactOracleRecord:
    problem_id: str
    problem_sha256: str
    source_kind: Literal["scaled_pcg64", "h3_anchor"]
    dimension: int
    oracle_stream: H4CanonicalStreamDigest
    canonical_log_normalizer: float
    predictive_log_normalizer: float
    route_agreement: H4OracleRouteAgreement
    selected_moments: tuple[H4SelectedMomentSummary, ...]
    posterior_condition: H4PosteriorConditionRecord
    innovation_conditions: H4ProblemConditionSummary

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.source_kind not in ("scaled_pcg64", "h3_anchor"):
            raise ValueError("invalid compact oracle source")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("compact oracle dimension must be positive")
        if (
            type(self.oracle_stream) is not H4CanonicalStreamDigest
            or self.oracle_stream.domain != "vfe4.h4.oracle-evaluation-stream.v1"
        ):
            raise ValueError("compact oracle requires its canonical stream")
        _finite(self.canonical_log_normalizer, "canonical_log_normalizer")
        _finite(self.predictive_log_normalizer, "predictive_log_normalizer")
        if (
            type(self.route_agreement) is not H4OracleRouteAgreement
            or self.route_agreement.problem_id != self.problem_id
            or self.route_agreement.problem_sha256 != self.problem_sha256
            or self.route_agreement.canonical_operand.value != self.canonical_log_normalizer
            or self.route_agreement.predictive_operand.value != self.predictive_log_normalizer
        ):
            raise ValueError("compact oracle route agreement does not bind its retained values")
        selected = _strict_tuple(self.selected_moments, H4SelectedMomentSummary, "selected_moments")
        if len({item.name for item in selected}) != len(selected):
            raise ValueError("compact oracle selected names must be unique")
        if (
            type(self.posterior_condition) is not H4PosteriorConditionRecord
            or self.posterior_condition.problem_id != self.problem_id
            or self.posterior_condition.problem_sha256 != self.problem_sha256
            or self.posterior_condition.source != "numpy_oracle"
            or self.posterior_condition.dimension != self.dimension
            or type(self.innovation_conditions) is not H4ProblemConditionSummary
            or self.innovation_conditions.problem_id != self.problem_id
            or self.innovation_conditions.problem_sha256 != self.problem_sha256
            or self.innovation_conditions.name != "oracle_innovation"
        ):
            raise ValueError("compact oracle condition evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class H4NativeReplayRecord:
    problem_id: str
    problem_sha256: str
    repetition_index: int
    arm: H4SolverArm
    reference_native_sha256: str
    replayed_native_sha256: str
    diagnostic_stream: H4CanonicalStreamDigest
    innovation_record_count: int
    exact_result_match: Literal[True]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if type(self.repetition_index) is not int or self.repetition_index not in range(11):
            raise ValueError("native replay requires repetition 0..10")
        if self.arm not in ("information", "moment"):
            raise ValueError("invalid native replay arm")
        _sha(self.reference_native_sha256, "reference_native_sha256")
        _sha(self.replayed_native_sha256, "replayed_native_sha256")
        if self.exact_result_match is not True or self.reference_native_sha256 != self.replayed_native_sha256:
            raise ValueError("native replay requires exact result equality")
        if (
            type(self.diagnostic_stream) is not H4CanonicalStreamDigest
            or self.diagnostic_stream.domain != "vfe4.h4.native-diagnostic-stream.v1"
            or self.innovation_record_count != self.diagnostic_stream.record_count
            or self.diagnostic_stream.scalar_count != 4 * self.innovation_record_count
            or (self.arm == "information" and self.innovation_record_count != 0)
            or (self.arm == "moment" and self.innovation_record_count <= 0)
        ):
            raise ValueError("native diagnostic stream counts are inconsistent")


@dataclass(frozen=True, slots=True)
class H4CountingPassRecord:
    problem_id: str
    problem_sha256: str
    arm: H4SolverArm
    reference_repetition_index: Literal[0]
    reference_native_sha256: str
    replayed_native_sha256: str
    reference_terminal_sha256: str
    replayed_terminal_sha256: str
    exact_result_match: Literal[True]
    solver_operations: tuple[H4OperationRecord, ...]
    terminal_conversion_operations: tuple[H4OperationRecord, ...]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.arm not in ("information", "moment") or self.reference_repetition_index != 0:
            raise ValueError("counting pass identity is frozen")
        for name in (
            "reference_native_sha256", "replayed_native_sha256",
            "reference_terminal_sha256", "replayed_terminal_sha256",
        ):
            _sha(getattr(self, name), name)
        if (
            self.exact_result_match is not True
            or self.reference_native_sha256 != self.replayed_native_sha256
            or self.reference_terminal_sha256 != self.replayed_terminal_sha256
        ):
            raise ValueError("counting pass must exactly replay native and terminal results")
        for name in ("solver_operations", "terminal_conversion_operations"):
            records = _strict_tuple(getattr(self, name), H4OperationRecord, name)
            if not records or any(item.problem_id != self.problem_id or item.arm != self.arm for item in records):
                raise ValueError("counting operations must be nonempty and identity-bound")
        selected_count = _selected_label_count_from_problem_id(self.problem_id)
        expected_cholesky = selected_count + (1 if self.arm == "information" else 2)
        actual_cholesky = sum(
            item.count for item in self.terminal_conversion_operations
            if item.operation == "cholesky"
        )
        if actual_cholesky != expected_cholesky:
            raise ValueError("terminal conversion Cholesky count is not explicit and complete")


@dataclass(frozen=True, slots=True)
class H4MemoryPassRecord:
    problem_id: str
    problem_sha256: str
    arm: H4SolverArm
    reference_repetition_index: Literal[0]
    reference_native_sha256: str
    replayed_native_sha256: str
    exact_result_match: Literal[True]
    memory: H4MemoryRecord

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.arm not in ("information", "moment") or self.reference_repetition_index != 0:
            raise ValueError("memory pass identity is frozen")
        _sha(self.reference_native_sha256, "reference_native_sha256")
        _sha(self.replayed_native_sha256, "replayed_native_sha256")
        if self.exact_result_match is not True or self.reference_native_sha256 != self.replayed_native_sha256:
            raise ValueError("memory pass must exactly replay the native result")
        if type(self.memory) is not H4MemoryRecord or self.memory.problem_id != self.problem_id or self.memory.arm != self.arm:
            raise ValueError("memory pass record is not identity-bound")


def _selected_label_count_from_problem_id(problem_id: str) -> int:
    match = _SCALED_ID.fullmatch(problem_id)
    if match is not None:
        return int(match.group(2)) + 2
    if problem_id in ("h4-anchor-h3-coupled-v1", "h4-anchor-h3-zero-control-v1"):
        return 3
    raise ValueError("unknown H4 problem identity")


def _scaled_indices(
    problem_id: str, problem_index: int, horizon_index: int, seed_index: int, kind_index: int,
) -> int:
    match = _SCALED_ID.fullmatch(problem_id)
    if (
        match is None
        or type(horizon_index) is not int or horizon_index not in range(3)
        or type(seed_index) is not int or seed_index not in range(20)
        or type(kind_index) is not int or kind_index not in range(2)
        or type(problem_index) is not int
        or problem_index != ((horizon_index * 20 + seed_index) * 2 + kind_index)
        or int(match.group(2)) != (7, 15, 31)[horizon_index]
        or match.group(1) != ("coupled", "zero_control")[kind_index]
    ):
        raise ValueError("scaled problem indices do not match canonical traversal")
    return int(match.group(2))


@dataclass(frozen=True, slots=True)
class H4ProblemEvaluation:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    oracle: H4CompactOracleRecord
    materialization: H4MaterializationIdentity
    execution_trace: H4ExecutionTrace
    retained_results: tuple[H4CompactResultRecord, ...]
    native_replays: tuple[H4NativeReplayRecord, ...]
    condition_summaries: tuple[
        H4ProblemConditionSummary, H4ProblemConditionSummary,
        H4ProblemConditionSummary, H4ProblemConditionSummary,
    ]
    counting_passes: tuple[H4CountingPassRecord, H4CountingPassRecord]
    memory_passes: tuple[H4MemoryPassRecord, H4MemoryPassRecord]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        horizon = _scaled_indices(
            self.problem_id, self.problem_index, self.horizon_index,
            self.seed_index, self.kind_index,
        )
        if (
            type(self.oracle) is not H4CompactOracleRecord
            or self.oracle.source_kind != "scaled_pcg64"
            or (self.oracle.problem_id, self.oracle.problem_sha256)
            != (self.problem_id, self.problem_sha256)
            or type(self.materialization) is not H4MaterializationIdentity
            or (self.materialization.problem_id, self.materialization.problem_sha256)
            != (self.problem_id, self.problem_sha256)
            or tuple(item.phase for item in self.materialization.integrity_checks)
            != ("after_materialization", "before_timed_batch", "after_timed_batch", "after_postflight")
            or type(self.execution_trace) is not H4ExecutionTrace
            or self.execution_trace.problem_id != self.problem_id
            or (
                self.execution_trace.problem_index, self.execution_trace.horizon_index,
                self.execution_trace.seed_index, self.execution_trace.kind_index,
            ) != (self.problem_index, self.horizon_index, self.seed_index, self.kind_index)
        ):
            raise ValueError("complete H4 problem identity is inconsistent")
        results = _strict_tuple(self.retained_results, H4CompactResultRecord, "retained_results")
        replays = _strict_tuple(self.native_replays, H4NativeReplayRecord, "native_replays")
        if len(results) != 22 or len(replays) != 22:
            raise ValueError("complete problem requires 22 compact results and replays")
        expected = tuple((repetition, arm) for repetition in range(11) for arm in ("information", "moment"))
        if (
            tuple((item.repetition_index, item.arm) for item in results) != expected
            or tuple((item.repetition_index, item.arm) for item in replays) != expected
        ):
            raise ValueError("compact result/replay order is frozen")
        for result, replay in zip(results, replays, strict=True):
            if (
                (result.problem_id, result.problem_sha256) != (self.problem_id, self.problem_sha256)
                or (replay.problem_id, replay.problem_sha256) != (self.problem_id, self.problem_sha256)
                or replay.reference_native_sha256 != result.native_stream.sha256
                or tuple((item.name, item.coordinate_indices) for item in result.selected_moments)
                != tuple((item.name, item.coordinate_indices) for item in self.oracle.selected_moments)
            ):
                raise ValueError("compact result/replay/oracle identities disagree")
        summaries = _strict_tuple(self.condition_summaries, H4ProblemConditionSummary, "condition_summaries")
        expected_counts = (1, 22, horizon, 11 * horizon)
        if (
            len(summaries) != 4
            or tuple(item.name for item in summaries) != _CONDITION_NAMES
            or tuple(item.observed_record_count for item in summaries) != expected_counts
            or any((item.problem_id, item.problem_sha256) != (self.problem_id, self.problem_sha256) for item in summaries)
        ):
            raise ValueError("per-problem condition summaries are incomplete")
        for name, records, expected_arms in (
            ("counting_passes", self.counting_passes, ("information", "moment")),
            ("memory_passes", self.memory_passes, ("information", "moment")),
        ):
            exact = H4CountingPassRecord if name == "counting_passes" else H4MemoryPassRecord
            values = _strict_tuple(records, exact, name)
            if len(values) != 2 or tuple(item.arm for item in values) != expected_arms or any(
                (item.problem_id, item.problem_sha256) != (self.problem_id, self.problem_sha256)
                for item in values
            ):
                raise ValueError(f"{name} must contain one identity-bound pass per arm")


@dataclass(frozen=True, slots=True)
class H4ScaledIncompletePhaseRecord:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    phase: Literal["warmup", "gc_capture", "gc_disable", "timed_batch", "gc_restore", "postflight"]
    materialization: H4MaterializationIdentity
    warmup_spans: tuple[H4ArmCallSpan, ...]
    partial_timed_spans: tuple[H4ArmCallSpan, ...]
    garbage_collector: H4GarbageCollectorRecord | None
    postflight_schedule: H4PostflightScheduleSummary | None
    stable_error: str
    obligation: str

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        _scaled_indices(
            self.problem_id, self.problem_index, self.horizon_index,
            self.seed_index, self.kind_index,
        )
        if self.phase not in _INCOMPLETE_OBLIGATIONS or self.obligation != _INCOMPLETE_OBLIGATIONS[self.phase]:
            raise ValueError("scaled incomplete phase obligation is frozen")
        _bounded_text(self.stable_error, "stable_error")
        if (
            type(self.materialization) is not H4MaterializationIdentity
            or (self.materialization.problem_id, self.materialization.problem_sha256)
            != (self.problem_id, self.problem_sha256)
        ):
            raise ValueError("incomplete phase materialization identity disagrees")
        warmups = _strict_tuple(self.warmup_spans, H4ArmCallSpan, "warmup_spans")
        timed = _strict_tuple(self.partial_timed_spans, H4ArmCallSpan, "partial_timed_spans")
        if len(warmups) > 6 or len(timed) > 22:
            raise ValueError("incomplete spans must be strict prefixes")
        expected_warmups = _expected_span_identities(
            self.problem_id, self.horizon_index, self.seed_index, self.kind_index,
            "warmup", length=len(warmups),
        )
        expected_timed = _expected_span_identities(
            self.problem_id, self.horizon_index, self.seed_index, self.kind_index,
            "timed", length=len(timed),
        )
        if _span_identities(warmups) != expected_warmups or _span_identities(timed) != expected_timed:
            raise ValueError("incomplete arm spans are not the canonical prefix")
        if self.phase == "warmup":
            if len(warmups) >= 6 or timed or self.garbage_collector is not None or self.postflight_schedule is not None:
                raise ValueError("warmup failure cannot fabricate GC, timing, or postflight evidence")
            return
        if len(warmups) != 6 or type(self.garbage_collector) is not H4GarbageCollectorRecord:
            raise ValueError("post-warmup failure requires all warmups and a typed GC record")
        if self.garbage_collector.problem_id != self.problem_id:
            raise ValueError("incomplete GC record belongs to another problem")
        gc_record = self.garbage_collector
        if self.phase == "gc_capture":
            valid = gc_record.capture_error is not None and not timed
        elif self.phase == "gc_disable":
            valid = (
                gc_record.capture_error is None
                and (
                    gc_record.disable_error is not None
                    or gc_record.effective_state_capture_error is not None
                    or gc_record.disabled_during_batch is not True
                )
                and not timed
            )
        elif self.phase == "timed_batch":
            valid = len(timed) < 22 and gc_record.capture_error is None
        elif self.phase == "gc_restore":
            valid = not gc_record.restored_exact_prior_state
        else:
            valid = len(timed) == 22 and gc_record.restored_exact_prior_state
        if not valid:
            raise ValueError("incomplete phase evidence does not match the phase boundary")
        if (self.phase == "postflight") != (self.postflight_schedule is not None):
            raise ValueError("postflight summary is present exactly when postflight began")


@dataclass(frozen=True, slots=True)
class H4ScaledMaterializedIntegrityFailureRecord:
    problem_id: str
    problem_sha256: str
    problem_index: int
    horizon_index: int
    seed_index: int
    kind_index: int
    materialization_version: Literal["h4-materialized-problem-v1"]
    protocol_id: Literal["h4-single-pass-v1"]
    materialization_count: Literal[1]
    shared_by_identity: Literal[True]
    checkpoint: H4ScaledMaterializedIntegrityCheckpoint
    expected_tensor_sha256: str
    completed_integrity_checks: tuple[H4MaterializedIntegrityCheck, ...]
    failure_kind: Literal["seam_exception", "digest_mismatch"]
    observed_tensor_sha256: str | None
    seam_error: str | None
    warmup_spans: tuple[H4ArmCallSpan, ...]
    timed_spans: tuple[H4ArmCallSpan, ...]
    garbage_collector: H4GarbageCollectorRecord | None
    postflight_schedule: H4PostflightScheduleSummary | None
    obligation: Literal["materialized_integrity"]

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        horizon = _scaled_indices(
            self.problem_id, self.problem_index, self.horizon_index,
            self.seed_index, self.kind_index,
        )
        if (
            self.materialization_version != "h4-materialized-problem-v1"
            or self.protocol_id != "h4-single-pass-v1"
            or self.materialization_count != 1
            or self.shared_by_identity is not True
            or self.obligation != "materialized_integrity"
        ):
            raise ValueError("scaled materialized-integrity carrier identity is frozen")
        _sha(self.expected_tensor_sha256, "expected_tensor_sha256")
        expected_prefix = {
            "after_materialization": (),
            "before_timed_batch": ("after_materialization",),
            "after_timed_batch": ("after_materialization", "before_timed_batch"),
            "after_postflight": ("after_materialization", "before_timed_batch", "after_timed_batch"),
        }
        if self.checkpoint not in expected_prefix:
            raise ValueError("invalid scaled integrity checkpoint")
        checks = _strict_tuple(
            self.completed_integrity_checks, H4MaterializedIntegrityCheck,
            "completed_integrity_checks",
        )
        if tuple(item.phase for item in checks) != expected_prefix[self.checkpoint] or any(
            item.expected_tensor_sha256 != self.expected_tensor_sha256
            or item.observed_tensor_sha256 != self.expected_tensor_sha256
            for item in checks
        ):
            raise ValueError("completed integrity-check prefix is inconsistent")
        if self.failure_kind == "seam_exception":
            if self.observed_tensor_sha256 is not None or self.seam_error is None:
                raise ValueError("seam exception requires only a bounded seam error")
            _bounded_text(self.seam_error, "seam_error")
        elif self.failure_kind == "digest_mismatch":
            if self.observed_tensor_sha256 is None or self.seam_error is not None:
                raise ValueError("digest mismatch requires only an observed digest")
            _sha(self.observed_tensor_sha256, "observed_tensor_sha256")
            if self.observed_tensor_sha256 == self.expected_tensor_sha256:
                raise ValueError("digest mismatch cannot carry an equal digest")
        else:
            raise ValueError("invalid integrity failure kind")
        warmups = _strict_tuple(self.warmup_spans, H4ArmCallSpan, "warmup_spans")
        timed = _strict_tuple(self.timed_spans, H4ArmCallSpan, "timed_spans")
        expected_counts = {
            "after_materialization": (0, 0, False),
            "before_timed_batch": (6, 0, False),
            "after_timed_batch": (6, 22, True),
            "after_postflight": (6, 22, True),
        }[self.checkpoint]
        if (len(warmups), len(timed), self.garbage_collector is not None) != expected_counts:
            raise ValueError("integrity checkpoint execution prefix is inconsistent")
        if _span_identities(warmups) != _expected_span_identities(
            self.problem_id, self.horizon_index, self.seed_index, self.kind_index,
            "warmup", length=len(warmups),
        ) or _span_identities(timed) != _expected_span_identities(
            self.problem_id, self.horizon_index, self.seed_index, self.kind_index,
            "timed", length=len(timed),
        ):
            raise ValueError("integrity checkpoint spans are not canonical")
        if self.garbage_collector is not None and (
            type(self.garbage_collector) is not H4GarbageCollectorRecord
            or self.garbage_collector.problem_id != self.problem_id
            or not self.garbage_collector.restored_exact_prior_state
        ):
            raise ValueError("post-timed integrity failure requires restored GC evidence")
        if self.checkpoint in ("after_materialization", "before_timed_batch"):
            if self.postflight_schedule is not None:
                raise ValueError("pre-timing integrity failure has no postflight schedule")
        else:
            if type(self.postflight_schedule) is not H4PostflightScheduleSummary:
                raise ValueError("post-timing integrity failure requires schedule evidence")
            if self.postflight_schedule.expected_event_count != 251 + 55 * horizon:
                raise ValueError("integrity failure schedule count does not match horizon")


@dataclass(frozen=True, slots=True)
class H4AnchorEvaluation:
    problem_id: str
    problem_sha256: str
    oracle: H4CompactOracleRecord
    materialization: H4MaterializationIdentity
    information_result: H4CompactResultRecord
    information_diagnostic_stream: H4CanonicalStreamDigest
    moment_result: H4CompactResultRecord
    moment_diagnostic_stream: H4CanonicalStreamDigest

    def __post_init__(self) -> None:
        _identity(self.problem_id, self.problem_sha256)
        if self.problem_id not in (
            "h4-anchor-h3-coupled-v1", "h4-anchor-h3-zero-control-v1",
        ):
            raise ValueError("anchor evaluation requires an exact H3 anchor ID")
        if (
            type(self.oracle) is not H4CompactOracleRecord
            or self.oracle.source_kind != "h3_anchor"
            or (self.oracle.problem_id, self.oracle.problem_sha256) != (self.problem_id, self.problem_sha256)
            or type(self.materialization) is not H4MaterializationIdentity
            or (self.materialization.problem_id, self.materialization.problem_sha256) != (self.problem_id, self.problem_sha256)
            or tuple(item.phase for item in self.materialization.integrity_checks)
            != ("after_materialization", "after_anchor_information", "after_anchor_moment")
        ):
            raise ValueError("anchor oracle/materialization identity disagrees")
        for arm, result, diagnostic in (
            ("information", self.information_result, self.information_diagnostic_stream),
            ("moment", self.moment_result, self.moment_diagnostic_stream),
        ):
            if (
                type(result) is not H4CompactResultRecord
                or result.source_kind != "h3_anchor" or result.repetition_index is not None
                or result.arm != arm
                or (result.problem_id, result.problem_sha256) != (self.problem_id, self.problem_sha256)
                or type(diagnostic) is not H4CanonicalStreamDigest
                or diagnostic.domain != "vfe4.h4.native-diagnostic-stream.v1"
            ):
                raise ValueError("anchor arm compact evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class H4UnavailablePhaseRecord:
    phase: Literal["anchor_coupled", "anchor_zero_control", "scaled_preflight", "statistics"]
    reason: str
    obligation: str

    def __post_init__(self) -> None:
        if self.phase not in ("anchor_coupled", "anchor_zero_control", "scaled_preflight", "statistics"):
            raise ValueError("unknown H4 unavailable phase")
        _bounded_text(self.reason, "unavailable reason")
        _bounded_text(self.obligation, "unavailable obligation")


@dataclass(frozen=True, slots=True)
class H4PowerPolicyField:
    name: Literal[
        "active_power_scheme", "cpu_frequency_governor",
        "energy_performance_preference", "low_power_mode",
    ]
    availability: Literal["available", "not_applicable", "unavailable"]
    source: Literal["powercfg", "linux_sysfs", "pmset", "none"]
    value: str | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if self.name not in _POWER_NAMES or self.availability not in (
            "available", "not_applicable", "unavailable",
        ) or self.source not in ("powercfg", "linux_sysfs", "pmset", "none"):
            raise ValueError("invalid H4 power-policy field")
        if self.availability == "available":
            if self.source == "none" or self.value is None or self.unavailable_reason is not None:
                raise ValueError("available power-policy field requires source and value only")
            _bounded_text(self.value, "power policy value", maximum=_POWER_VALUE_LIMIT)
        elif self.availability == "not_applicable":
            if self.source != "none" or self.value is not None or self.unavailable_reason is not None:
                raise ValueError("not-applicable power field must use the none source")
        else:
            if self.source == "none" or self.value is not None or self.unavailable_reason is None:
                raise ValueError("unavailable applicable field requires source and reason")
            _bounded_text(self.unavailable_reason, "power unavailable reason")


@dataclass(frozen=True, slots=True)
class H4EnvironmentRecord:
    clock_implementation: str
    clock_resolution_seconds: float
    clock_monotonic: bool
    processor: str
    platform: str
    platform_system: Literal["Windows", "Linux", "Darwin", "Other"]
    affinity_cpu_ids: tuple[int, ...] | None
    logical_cpu_count: int
    physical_cpu_count: int | None
    torch_version: str
    numpy_version: str
    torch_config_text: str
    torch_config_sha256: str
    numpy_blas_text: str
    numpy_blas_sha256: str
    cuda_available: Literal[False]
    environment_variables: tuple[tuple[str, bool, str | None], ...]
    power_policy_fields: tuple[H4PowerPolicyField, H4PowerPolicyField, H4PowerPolicyField, H4PowerPolicyField]
    power_policy_category_complete: Literal[True]
    unavailable_fields: tuple[str, ...]
    mandatory_facts_complete: bool

    def __post_init__(self) -> None:
        for name in ("clock_implementation", "processor", "platform", "torch_version", "numpy_version", "torch_config_text", "numpy_blas_text"):
            _bounded_text(getattr(self, name), name, maximum=1_000_000)
        _finite(self.clock_resolution_seconds, "clock_resolution_seconds")
        if self.clock_resolution_seconds <= 0.0 or type(self.clock_monotonic) is not bool:
            raise ValueError("clock facts are malformed")
        if self.platform_system not in ("Windows", "Linux", "Darwin", "Other"):
            raise ValueError("platform system category is closed")
        if type(self.unavailable_fields) is not tuple or len(set(self.unavailable_fields)) != len(self.unavailable_fields) or any(type(item) is not str or not item for item in self.unavailable_fields):
            raise ValueError("environment unavailable fields must be unique strings")
        affinity_unavailable = "affinity_cpu_ids" in self.unavailable_fields
        if (self.affinity_cpu_ids is None) != affinity_unavailable:
            raise ValueError("affinity_cpu_ids availability is inconsistent")
        if (
            self.affinity_cpu_ids is not None
            and (
                type(self.affinity_cpu_ids) is not tuple or not self.affinity_cpu_ids
                or any(type(item) is not int or item < 0 for item in self.affinity_cpu_ids)
                or tuple(sorted(set(self.affinity_cpu_ids))) != self.affinity_cpu_ids
            )
        ):
            raise ValueError("affinity_cpu_ids must be a sorted unique nonempty tuple")
        if (
            type(self.logical_cpu_count) is not int or self.logical_cpu_count <= 0
            or (self.physical_cpu_count is not None and (type(self.physical_cpu_count) is not int or self.physical_cpu_count <= 0))
        ):
            raise ValueError("CPU facts are malformed")
        _sha(self.torch_config_sha256, "torch_config_sha256")
        _sha(self.numpy_blas_sha256, "numpy_blas_sha256")
        if (
            hashlib.sha256(self.torch_config_text.encode()).hexdigest() != self.torch_config_sha256
            or hashlib.sha256(self.numpy_blas_text.encode()).hexdigest() != self.numpy_blas_sha256
        ):
            raise ValueError("library configuration digests do not match retained text")
        if self.cuda_available is not False:
            raise ValueError("H4 environment is CPU-only")
        variables = self.environment_variables
        if type(variables) is not tuple or tuple(item[0] for item in variables) != _ENVIRONMENT_NAMES:
            raise ValueError("environment-variable order is frozen")
        for name, present, value in variables:
            if type(present) is not bool or present != (value is not None) or (value is not None and type(value) is not str):
                raise ValueError(f"environment variable {name} presence/value is inconsistent")
        policy = _strict_tuple(self.power_policy_fields, H4PowerPolicyField, "power_policy_fields")
        if tuple(item.name for item in policy) != _POWER_NAMES or self.power_policy_category_complete is not True:
            raise ValueError("power-policy category/order must be complete")
        expected_complete = not any(
            name in self.unavailable_fields for name in (
                "clock", "processor", "platform", "affinity_cpu_ids", "logical_cpu_count",
                "torch_version", "numpy_version", "torch_config", "numpy_blas", "cuda_available",
                "environment_variables",
            )
        )
        if type(self.mandatory_facts_complete) is not bool or self.mandatory_facts_complete != expected_complete:
            raise ValueError("mandatory_facts_complete is inconsistent")


@dataclass(frozen=True, slots=True)
class H4PayloadSizeRecord:
    encoding: Literal["utf8-compact-sorted-key-json-v1"]
    observed_bytes: int
    maximum_bytes: Literal[67108864]
    fixed_point_iterations: int
    within_limit: bool

    def __post_init__(self) -> None:
        if self.encoding != "utf8-compact-sorted-key-json-v1" or self.maximum_bytes != _MAX_PAYLOAD_BYTES:
            raise ValueError("H4 payload encoding/ceiling is frozen")
        if type(self.observed_bytes) is not int or self.observed_bytes < 0:
            raise ValueError("payload observed bytes must be nonnegative")
        if type(self.fixed_point_iterations) is not int or self.fixed_point_iterations not in range(1, 5):
            raise ValueError("payload fixed point must converge in one through four iterations")
        if type(self.within_limit) is not bool or self.within_limit != (self.observed_bytes <= self.maximum_bytes):
            raise ValueError("payload size decision is inconsistent")


@dataclass(frozen=True, slots=True)
class H4GateEvaluation:
    schema_version: Literal["h4-gate-evaluation-v1"]
    payload_representation: Literal["bounded-stream-summaries-v1"]
    maximum_payload_bytes: Literal[67108864]
    result: H4GateResult
    h4_config_sha256: str
    anchors: tuple[
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
    ]
    unavailable_phases: tuple[H4UnavailablePhaseRecord, ...]
    problems: tuple[
        H4ProblemEvaluation | H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        ...,
    ]
    allowances: tuple[H4AllowanceRecord, ...]
    coverage: tuple[H4CoverageRecord, ...]
    condition_summaries: tuple[
        H4ConditionStreamSummary, H4ConditionStreamSummary,
        H4ConditionStreamSummary, H4ConditionStreamSummary,
    ]
    raw_timings: tuple[H4TimingRecord, ...]
    primary_timed_order_balance: H4PrimaryTimedOrderBalance | None
    timing_summary: H4TimingSummary | None
    bootstrap_interval: H4BootstrapInterval | None
    interval_decision: H4IntervalDecision | None
    thread_state: H4ThreadStateRecord
    environment: H4EnvironmentRecord
    payload_size: H4PayloadSizeRecord
    bounded_claim: str
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "h4-gate-evaluation-v1":
            raise ValueError("H4 gate evaluation schema is frozen")
        _validate_gate_container(self)


@dataclass(frozen=True, slots=True)
class H4ValidationArtifact:
    schema_version: Literal["vfe4-validation-h4-v1"]
    payload_representation: Literal["bounded-stream-summaries-v1"]
    maximum_payload_bytes: Literal[67108864]
    gate: Literal["H4"]
    status: GateStatus
    h4_config_sha256: str
    result: H4GateResult
    anchors: tuple[
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
    ]
    unavailable_phases: tuple[H4UnavailablePhaseRecord, ...]
    problems: tuple[
        H4ProblemEvaluation | H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        ...,
    ]
    allowances: tuple[H4AllowanceRecord, ...]
    coverage: tuple[H4CoverageRecord, ...]
    condition_summaries: tuple[
        H4ConditionStreamSummary, H4ConditionStreamSummary,
        H4ConditionStreamSummary, H4ConditionStreamSummary,
    ]
    raw_timings: tuple[H4TimingRecord, ...]
    primary_timed_order_balance: H4PrimaryTimedOrderBalance | None
    timing_summary: H4TimingSummary | None
    bootstrap_interval: H4BootstrapInterval | None
    interval_decision: H4IntervalDecision | None
    thread_state: H4ThreadStateRecord
    environment: H4EnvironmentRecord
    payload_size: H4PayloadSizeRecord
    bounded_claim: str
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "vfe4-validation-h4-v1" or self.gate != "H4":
            raise ValueError("H4 validation artifact identity is frozen")
        if not isinstance(self.status, GateStatus) or type(self.result) is not H4GateResult or self.status is not self.result.status:
            raise ValueError("artifact status must equal its gate result")
        _validate_gate_container(self)


def _validate_gate_container(value: H4GateEvaluation | H4ValidationArtifact) -> None:
    if (
        value.payload_representation != "bounded-stream-summaries-v1"
        or value.maximum_payload_bytes != _MAX_PAYLOAD_BYTES
        or type(value.result) is not H4GateResult
    ):
        raise ValueError("H4 compact container identity is frozen")
    _sha(value.h4_config_sha256, "h4_config_sha256")
    if type(value.anchors) is not tuple or len(value.anchors) != 2:
        raise ValueError("H4 container requires exactly two ordered anchor slots")
    for index, anchor in enumerate(value.anchors):
        expected_phase = ("anchor_coupled", "anchor_zero_control")[index]
        expected_id = (
            "h4-anchor-h3-coupled-v1", "h4-anchor-h3-zero-control-v1",
        )[index]
        if type(anchor) is H4UnavailablePhaseRecord:
            if anchor.phase != expected_phase:
                raise ValueError("anchor unavailable record is in the wrong slot")
        elif type(anchor) is H4AnchorEvaluation:
            if anchor.problem_id != expected_id:
                raise ValueError("completed anchor is in the wrong slot")
        else:
            raise ValueError("anchor slot must contain a typed evaluation or unavailable record")
    unavailable = _strict_tuple(
        value.unavailable_phases, H4UnavailablePhaseRecord, "unavailable_phases",
    )
    phases = tuple(item.phase for item in unavailable)
    if phases not in ((), ("scaled_preflight",), ("statistics",), ("scaled_preflight", "statistics")):
        raise ValueError("top-level unavailable phases are duplicate-free and canonically ordered")
    problems = value.problems
    if type(problems) is not tuple or len(problems) > 120:
        raise ValueError("H4 problem sequence must be a bounded tuple")
    failure_seen = False
    completed_count = 0
    for index, item in enumerate(problems):
        if type(item) is H4ProblemEvaluation:
            if failure_seen or item.problem_index != index:
                raise ValueError("complete problems must be the canonical traversal prefix")
            completed_count += 1
        elif type(item) is H4ScaledIncompletePhaseRecord:
            if failure_seen or item.problem_index != index or index != len(problems) - 1:
                raise ValueError("scaled incomplete record must terminate the canonical prefix")
            failure_seen = True
        elif type(item) is H4ScaledMaterializedIntegrityFailureRecord:
            if failure_seen or index != len(problems) - 1:
                raise ValueError("scaled integrity failure must terminate the problem sequence")
            if item.checkpoint == "after_materialization":
                if len(problems) != 1:
                    raise ValueError("preflight integrity failure is the sole problem record")
            elif item.problem_index != index:
                raise ValueError("later integrity failure must follow the completed prefix")
            failure_seen = True
        else:
            raise ValueError("H4 problem tuple contains a broad or unknown record")
    complete = len(problems) == 120 and completed_count == 120 and not failure_seen
    allowances = value.allowances
    if type(allowances) is not tuple or len(allowances) != 6 or any(
        type(item) not in (H4ApplicableAllowance, H4InapplicableAllowance)
        for item in allowances
    ):
        raise ValueError("container must own exactly six typed allowance records")
    expected_allowances = tuple(
        value.result.allowances_by_invariant[name] for name in H4_ALLOWANCE_INVARIANT_NAMES
    )
    if allowances != expected_allowances:
        raise ValueError("container allowances must equal its gate-result mapping")
    coverage = _strict_tuple(value.coverage, H4CoverageRecord, "coverage")
    conditions = _strict_tuple(
        value.condition_summaries, H4ConditionStreamSummary, "condition_summaries",
    )
    timings = _strict_tuple(value.raw_timings, H4TimingRecord, "raw_timings")
    if complete:
        if (
            tuple(item.name for item in coverage) != _COVERAGE_NAMES
            or not all(item.complete for item in coverage)
            or tuple(item.name for item in conditions) != _CONDITION_NAMES
            or tuple(item.observed_record_count for item in conditions) != (120, 2640, 2120, 23320)
            or len(timings) != 1320
        ):
            raise ValueError("complete H4 container requires closed coverage, conditions, and timings")
        _validate_complete_condition_evidence(value.result, conditions)
        if tuple(
            (item.problem_index, item.repetition_index) for item in timings
        ) != tuple((problem_index, repetition) for problem_index in range(120) for repetition in range(11)):
            raise ValueError("raw timing traversal/repetition order is frozen")
        for problem in problems:
            assert type(problem) is H4ProblemEvaluation
            rows = timings[11 * problem.problem_index:11 * (problem.problem_index + 1)]
            spans = {
                (item.repetition_index, item.arm): item
                for item in problem.execution_trace.timed_spans
            }
            if any(
                row.problem_id != problem.problem_id
                or row.information_nanoseconds != spans[(row.repetition_index, "information")].duration_nanoseconds
                or row.moment_nanoseconds != spans[(row.repetition_index, "moment")].duration_nanoseconds
                for row in rows
            ):
                raise ValueError("timing rows do not match retained trace spans")
        if (
            type(value.primary_timed_order_balance) is not H4PrimaryTimedOrderBalance
            or not value.primary_timed_order_balance.matches
            or type(value.timing_summary) is not H4TimingSummary
            or type(value.bootstrap_interval) is not H4BootstrapInterval
            or value.interval_decision is None
        ):
            raise ValueError("complete H4 container requires exact paired statistics")
        if decide_h4_interval(value.bootstrap_interval) != value.interval_decision:
            raise ValueError("public interval decision identity is broken")
    else:
        if any(item.complete for item in coverage):
            raise ValueError("incomplete H4 branch cannot claim complete global coverage")
        if value.primary_timed_order_balance is not None or value.timing_summary is not None or value.bootstrap_interval is not None or value.interval_decision is not None:
            raise ValueError("incomplete H4 branch cannot fabricate statistics")
        if "statistics" not in phases:
            raise ValueError("incomplete H4 branch must type the unavailable statistics phase")
    if type(value.thread_state) is not H4ThreadStateRecord or type(value.environment) is not H4EnvironmentRecord or type(value.payload_size) is not H4PayloadSizeRecord:
        raise ValueError("thread, environment, and payload-size records must be exact")
    _bounded_text(value.bounded_claim, "bounded_claim", maximum=4096)
    if type(value.nonclaims) is not tuple or not value.nonclaims or len(set(value.nonclaims)) != len(value.nonclaims) or any(type(item) is not str or not item for item in value.nonclaims):
        raise ValueError("H4 nonclaims must be a unique nonempty tuple")


def _validate_complete_condition_evidence(
    result: H4GateResult,
    conditions: tuple[H4ConditionStreamSummary, ...],
) -> None:
    if type(result) is not H4GateResult:
        raise ValueError("complete condition-envelope evidence requires an exact result")
    records = _strict_tuple(
        conditions, H4ConditionStreamSummary, "condition_summaries",
    )
    if (
        tuple(item.name for item in records) != _CONDITION_NAMES
        or tuple(item.observed_record_count for item in records)
        != (120, 2640, 2120, 23320)
    ):
        raise ValueError("complete condition-envelope evidence has wrong streams")
    eligible = sum(item.eligible_record_count for item in records)
    observed = sum(item.observed_record_count for item in records)
    all_eligible = all(item.all_eligible for item in records)
    invariant = result.invariants[
        H4_INVARIANT_NAMES.index("scaled_condition_envelope")
    ]
    expected = (
        all_eligible, float(eligible), float(observed), "full_condition_stream",
    )
    actual = (
        invariant.passed, invariant.value, invariant.limit, invariant.detail,
    )
    if actual != expected:
        raise ValueError("condition-envelope evidence disagrees with its invariant")
    obligation = (
        "scaled_condition_envelope: resolve incomplete H4 eligibility evidence"
    )
    if all_eligible:
        if obligation in result.obligations:
            raise ValueError("condition-envelope evidence has a spurious obligation")
    elif result.status is not GateStatus.INCONCLUSIVE or obligation not in result.obligations:
        raise ValueError("condition-envelope evidence lacks its exact obligation")


def _order_for_pair(
    horizon_index: int, seed_index: int, kind_index: int, pair_index: int,
) -> Literal["information_then_moment", "moment_then_information"]:
    for name, value in (
        ("horizon_index", horizon_index), ("seed_index", seed_index),
        ("kind_index", kind_index), ("pair_index", pair_index),
    ):
        _nonnegative_int(value, name)
    return (
        "information_then_moment"
        if (horizon_index + seed_index + kind_index + pair_index) % 2 == 0
        else "moment_then_information"
    )


def _span_identities(spans: tuple[H4ArmCallSpan, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.problem_id, item.phase, item.pair_index, item.repetition_index,
         item.order, item.order_position, item.arm)
        for item in spans
    )


def _expected_span_identities(
    problem_id: str,
    horizon_index: int,
    seed_index: int,
    kind_index: int,
    phase: Literal["warmup", "timed"],
    *,
    length: int,
) -> tuple[tuple[object, ...], ...]:
    maximum = 6 if phase == "warmup" else 22
    if type(length) is not int or length < 0 or length > maximum:
        raise ValueError("span-prefix length is outside the phase")
    rows: list[tuple[object, ...]] = []
    pair_indices = range(3) if phase == "warmup" else range(3, 14)
    for pair_index in pair_indices:
        order = _order_for_pair(horizon_index, seed_index, kind_index, pair_index)
        arms = (
            ("information", "moment")
            if order == "information_then_moment"
            else ("moment", "information")
        )
        repetition = None if phase == "warmup" else pair_index - 3
        for position, arm in enumerate(arms):
            rows.append((problem_id, phase, pair_index, repetition, order, position, arm))
    return tuple(rows[:length])


class _TorchThreadAPI(Protocol):
    def get_num_threads(self) -> int: ...
    def get_num_interop_threads(self) -> int: ...
    def set_num_threads(self, value: int) -> None: ...


@dataclass(frozen=True, slots=True)
class _H4ThreadGuardOutcome:
    value: object | None
    work_error: str | None
    state: H4ThreadStateRecord


def _run_thread_guard(
    work: Callable[[], object], *, torch_api: _TorchThreadAPI = torch,
) -> _H4ThreadGuardOutcome:
    if not callable(work):
        raise ValueError("thread-guard work must be callable")
    prior_intra: int | None = None
    prior_inter: int | None = None
    try:
        prior_intra = torch_api.get_num_threads()
        if type(prior_intra) is not int or prior_intra <= 0:
            raise RuntimeError("intra-op thread capture was not a positive integer")
        prior_inter = torch_api.get_num_interop_threads()
        if type(prior_inter) is not int or prior_inter <= 0:
            raise RuntimeError("inter-op thread capture was not a positive integer")
    except Exception as error:
        state = H4ThreadStateRecord(
            _stable_error(error), prior_intra, False, None, None, False,
            None, None, False, False, None, None, False,
        )
        return _H4ThreadGuardOutcome(None, None, state)

    set_error: str | None = None
    effective: int | None = None
    work_value: object | None = None
    work_error: str | None = None
    restored: int | None = None
    final_inter: int | None = None
    restoration_error: str | None = None
    try:
        try:
            torch_api.set_num_threads(1)
        except Exception as error:
            set_error = _stable_error(error)
        if set_error is None:
            try:
                effective = torch_api.get_num_threads()
                if effective != 1:
                    raise RuntimeError("intra-op thread verification failed")
            except Exception as error:
                set_error = _stable_error(error)
        if set_error is None and effective == 1:
            try:
                work_value = work()
            except Exception as error:
                work_error = _stable_error(error)
    finally:
        try:
            torch_api.set_num_threads(prior_intra)
            restored = torch_api.get_num_threads()
            final_inter = torch_api.get_num_interop_threads()
        except Exception as error:
            restoration_error = _stable_error(error)
            restored = None
            final_inter = None
    state = H4ThreadStateRecord(
        None, prior_intra, True, set_error, effective, set_error is None and effective == 1,
        prior_inter, final_inter, final_inter == prior_inter, True, restored,
        restoration_error, restoration_error is None and restored == prior_intra,
    )
    return _H4ThreadGuardOutcome(work_value, work_error, state)


@dataclass(frozen=True, slots=True)
class _H4TimedBatchOutcome:
    results: tuple[H4SolverResult, ...]
    warmup_spans: tuple[H4ArmCallSpan, ...]
    timed_spans: tuple[H4ArmCallSpan, ...]
    timings: tuple[H4TimingRecord, ...]
    timed_batch_start_nanoseconds: int
    timed_batch_end_nanoseconds: int
    garbage_collector: H4GarbageCollectorRecord


class _H4BatchFailure(RuntimeError):
    def __init__(
        self,
        phase: str,
        stable_error: str,
        warmup_spans: tuple[H4ArmCallSpan, ...],
        timed_spans: tuple[H4ArmCallSpan, ...],
        garbage_collector: H4GarbageCollectorRecord | None,
    ) -> None:
        super().__init__(stable_error)
        self.phase = phase
        self.stable_error = stable_error
        self.warmup_spans = warmup_spans
        self.timed_spans = timed_spans
        self.garbage_collector = garbage_collector


class _H4IntegrityCheckFailure(RuntimeError):
    def __init__(
        self,
        phase: H4MaterializedIntegrityPhase,
        expected_tensor_sha256: str,
        failure_kind: Literal["seam_exception", "digest_mismatch"],
        observed_tensor_sha256: str | None,
        seam_error: str | None,
    ) -> None:
        detail = seam_error or (
            f"materialized digest mismatch: expected {expected_tensor_sha256}, "
            f"observed {observed_tensor_sha256}"
        )
        super().__init__(detail)
        self.phase = phase
        self.expected_tensor_sha256 = expected_tensor_sha256
        self.failure_kind = failure_kind
        self.observed_tensor_sha256 = observed_tensor_sha256
        self.seam_error = seam_error


class _H4WarmupBoundaryIntegrityFailure(RuntimeError):
    def __init__(
        self,
        failure: _H4IntegrityCheckFailure,
        warmup_spans: tuple[H4ArmCallSpan, ...],
    ) -> None:
        super().__init__(str(failure))
        self.failure = failure
        self.warmup_spans = warmup_spans


class _H4ScaledCarrierFailure(RuntimeError):
    def __init__(
        self,
        record: H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        stable_error: str,
    ) -> None:
        super().__init__(stable_error)
        self.record = record
        self.stable_error = stable_error


_TIMED_BATCH_ACTIVE = False


def _assert_outside_timed_batch(operation: str) -> None:
    if _TIMED_BATCH_ACTIVE:
        raise RuntimeError(f"{operation} is excluded from the H4 timed batch")


def _fresh_null_linalg(problem_id: str, arm: H4SolverArm) -> InstrumentedLinearAlgebra:
    _assert_outside_timed_batch("facade construction")
    return InstrumentedLinearAlgebra(
        problem_id=problem_id, arm=arm, recorder=NullOperationRecorder(),
    )


def _solver_for_arm(arm: H4SolverArm) -> Callable[
    [H4MaterializedProblem, H4SolveProtocol, InstrumentedLinearAlgebra], H4SolverResult
]:
    return solve_information_form if arm == "information" else solve_moment_form


def _span_from_slots(
    *,
    problem_id: str,
    phase: Literal["warmup", "timed"],
    pair_index: int,
    repetition_index: int | None,
    order: Literal["information_then_moment", "moment_then_information"],
    position: Literal[0, 1],
    arm: H4SolverArm,
    start: int,
    end: int,
) -> H4ArmCallSpan:
    _assert_outside_timed_batch("span construction")
    return H4ArmCallSpan(
        problem_id, phase, pair_index, repetition_index, order, position, arm,
        start, end, end - start,
    )


def _run_warmup_and_timed_batch(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    *,
    horizon_index: int,
    seed_index: int,
    kind_index: int,
    perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
    before_timed_hook: Callable[[], None] | None = None,
) -> _H4TimedBatchOutcome:
    global _TIMED_BATCH_ACTIVE
    if type(materialized) is not H4MaterializedProblem or type(protocol) is not H4SolveProtocol:
        raise ValueError("timed H4 batch requires exact materialized problem/protocol")
    if materialized.protocol_id != protocol.protocol_id:
        raise ValueError("materialized problem and solve protocol disagree")
    if not callable(perf_counter_ns) or (before_timed_hook is not None and not callable(before_timed_hook)):
        raise ValueError("timed H4 hooks must be callable")

    warmup_spans: list[H4ArmCallSpan] = []
    for pair_index in range(3):
        order = _order_for_pair(horizon_index, seed_index, kind_index, pair_index)
        arms: tuple[H4SolverArm, H4SolverArm] = (
            ("information", "moment") if order == "information_then_moment"
            else ("moment", "information")
        )
        for position, arm in enumerate(arms):
            try:
                start = perf_counter_ns()
                _solver_for_arm(arm)(
                    materialized, protocol, _fresh_null_linalg(materialized.problem_id, arm),
                )
                end = perf_counter_ns()
                warmup_spans.append(_span_from_slots(
                    problem_id=materialized.problem_id, phase="warmup",
                    pair_index=pair_index, repetition_index=None, order=order,
                    position=position, arm=arm, start=start, end=end,  # type: ignore[arg-type]
                ))
            except Exception as error:
                raise _H4BatchFailure(
                    "warmup", _stable_error(error), tuple(warmup_spans), (), None,
                ) from error
    if before_timed_hook is not None:
        try:
            before_timed_hook()
        except _H4IntegrityCheckFailure as failure:
            raise _H4WarmupBoundaryIntegrityFailure(
                failure, tuple(warmup_spans),
            ) from failure

    result_slots: list[H4SolverResult | None] = [None] * 22
    start_slots: list[int] = [0] * 22
    end_slots: list[int] = [0] * 22
    facades: list[InstrumentedLinearAlgebra] = []
    arm_slots: list[H4SolverArm] = []
    order_slots: list[Literal["information_then_moment", "moment_then_information"]] = []
    for repetition_index, pair_index in enumerate(range(3, 14)):
        order = _order_for_pair(horizon_index, seed_index, kind_index, pair_index)
        arms = (
            ("information", "moment") if order == "information_then_moment"
            else ("moment", "information")
        )
        for arm in arms:
            arm_slots.append(arm)
            order_slots.append(order)
            facades.append(_fresh_null_linalg(materialized.problem_id, arm))

    capture_error: str | None = None
    prior_enabled: bool | None = None
    disable_required: bool | None = None
    disable_attempted = False
    disable_error: str | None = None
    effective_error: str | None = None
    disabled_during: bool | None = None
    restore_attempted = False
    restored_enabled: bool | None = None
    restoration_error: str | None = None
    timed_error: str | None = None
    completed_calls = 0

    try:
        try:
            prior_enabled = gc.isenabled()
        except Exception as error:
            capture_error = _stable_error(error)
        if capture_error is None:
            disable_required = prior_enabled
            if prior_enabled:
                disable_attempted = True
                try:
                    gc.disable()
                except Exception as error:
                    disable_error = _stable_error(error)
            if disable_error is None:
                try:
                    disabled_during = not gc.isenabled()
                except Exception as error:
                    effective_error = _stable_error(error)
            if disable_error is None and effective_error is None and disabled_during is True:
                _TIMED_BATCH_ACTIVE = True
                try:
                    for slot in range(22):
                        arm = arm_slots[slot]
                        start_slots[slot] = perf_counter_ns()
                        result_slots[slot] = _solver_for_arm(arm)(
                            materialized, protocol, facades[slot],
                        )
                        end_slots[slot] = perf_counter_ns()
                        completed_calls = slot + 1
                except Exception as error:
                    timed_error = _stable_error(error)
                finally:
                    _TIMED_BATCH_ACTIVE = False
    finally:
        if prior_enabled is not None:
            restore_attempted = True
            try:
                if prior_enabled:
                    gc.enable()
                else:
                    gc.disable()
                restored_enabled = gc.isenabled()
            except Exception as error:
                restoration_error = _stable_error(error)

    gc_record = H4GarbageCollectorRecord(
        materialized.problem_id, True, capture_error, prior_enabled,
        disable_required, disable_attempted, disable_error, effective_error,
        disabled_during, restore_attempted, restored_enabled, restoration_error,
        restoration_error is None and prior_enabled is not None and restored_enabled is prior_enabled,
    )

    timed_spans = tuple(
        _span_from_slots(
            problem_id=materialized.problem_id, phase="timed",
            pair_index=3 + slot // 2, repetition_index=slot // 2,
            order=order_slots[slot], position=slot % 2, arm=arm_slots[slot],
            start=start_slots[slot], end=end_slots[slot],  # type: ignore[arg-type]
        )
        for slot in range(completed_calls)
    )
    if capture_error is not None:
        raise _H4BatchFailure("gc_capture", capture_error, tuple(warmup_spans), (), gc_record)
    if (
        disable_error is not None
        or effective_error is not None
        or disabled_during is not True
    ):
        raise _H4BatchFailure(
            "gc_disable",
            disable_error or effective_error or "cyclic GC remained enabled",
            tuple(warmup_spans), (), gc_record,
        )
    if timed_error is not None:
        raise _H4BatchFailure("timed_batch", timed_error, tuple(warmup_spans), timed_spans, gc_record)
    if not gc_record.restored_exact_prior_state:
        raise _H4BatchFailure(
            "gc_restore", restoration_error or "cyclic GC exact restoration failed",
            tuple(warmup_spans), timed_spans, gc_record,
        )
    if any(item is None for item in result_slots) or completed_calls != 22:
        raise RuntimeError("complete timed batch did not fill all preallocated result slots")
    results = tuple(result_slots)  # type: ignore[arg-type]
    timings: list[H4TimingRecord] = []
    for repetition_index in range(11):
        pair = timed_spans[2 * repetition_index:2 * repetition_index + 2]
        durations = {item.arm: item.duration_nanoseconds for item in pair}
        timings.append(H4TimingRecord(
            materialized.problem_id,
            ((horizon_index * 20 + seed_index) * 2 + kind_index),
            horizon_index, seed_index, kind_index, materialized.seed,
            materialized.kind, materialized.horizon, repetition_index,
            3 + repetition_index, order_slots[2 * repetition_index],
            durations["information"], durations["moment"],
        ))
    return _H4TimedBatchOutcome(
        results, tuple(warmup_spans), timed_spans, tuple(timings),
        timed_spans[0].start_nanoseconds, timed_spans[-1].end_nanoseconds,
        gc_record,
    )


def _iter_expected_postflight_event_keys(
    problem: H4NeutralProblem,
) -> Iterator[H4PostflightEventKey]:
    _assert_outside_timed_batch("postflight key generation")
    if type(problem) is not H4NeutralProblem or problem.source_kind != "scaled_pcg64":
        raise ValueError("postflight schedule requires an exact scaled neutral problem")
    event_index = 0

    def event(
        phase: str,
        *,
        repetition_index: int | None = None,
        arm: H4SolverArm | None = None,
        factor_id: str | None = None,
        selected_moment_name: str | None = None,
        equivalence_component: str | None = None,
        integrity_phase: str | None = None,
    ) -> H4PostflightEventKey:
        nonlocal event_index
        value = H4PostflightEventKey(
            problem.problem_id, problem.canonical_sha256, event_index, phase,
            repetition_index, arm, factor_id, selected_moment_name,
            equivalence_component, integrity_phase,
        )  # type: ignore[arg-type]
        event_index += 1
        return value

    yield event("materialized_integrity", integrity_phase="after_timed_batch")
    for phase in (
        "terminal_conversion", "native_diagnostic_replay",
        "terminal_posterior_condition",
    ):
        for repetition in range(11):
            for arm in ("information", "moment"):
                yield event(phase, repetition_index=repetition, arm=arm)
    observation_ids = tuple(
        factor.factor_id for factor in problem.factor_schedule
        if factor.role == "observation"
    )
    if observation_ids != tuple(f"observation[{time_index}]" for time_index in range(1, problem.horizon + 1)):
        raise ValueError("postflight factor schedule is not the frozen observation order")
    for repetition in range(11):
        for factor_id in observation_ids:
            yield event(
                "moment_innovation_condition", repetition_index=repetition,
                arm="moment", factor_id=factor_id,
            )
    yield event("oracle_rehydration")
    yield event("oracle_route_agreement")
    selected_names = (
        "initial", "terminal",
        *(f"observation[{time_index}]" for time_index in range(1, problem.horizon + 1)),
    )
    for repetition in range(11):
        for arm in ("information", "moment"):
            for component in ("kl_to_zero", "h", "J"):
                yield event(
                    "equivalence_group", repetition_index=repetition, arm=arm,
                    equivalence_component=component,
                )
            for selected_name in selected_names:
                for component in ("selected_mean", "selected_covariance"):
                    yield event(
                        "equivalence_group", repetition_index=repetition, arm=arm,
                        selected_moment_name=selected_name,
                        equivalence_component=component,
                    )
            yield event(
                "equivalence_group", repetition_index=repetition, arm=arm,
                equivalence_component="objective",
            )
    for arm in ("information", "moment"):
        yield event("operation_pass", arm=arm)
    for arm in ("information", "moment"):
        yield event("memory_pass", arm=arm)
    yield event("materialized_integrity", integrity_phase="after_postflight")
    yield event("stream_compaction")
    expected = 251 + 55 * problem.horizon
    if event_index != expected:
        raise RuntimeError("independent H4 postflight schedule count drifted")


def _event_bytes(event: H4PostflightEventKey) -> bytes:
    if type(event) is not H4PostflightEventKey:
        raise ValueError("postflight stream requires exact event keys")
    return json.dumps(
        asdict(event), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _postflight_key_stream_digest(
    problem_sha256: str, events: Iterable[H4PostflightEventKey],
) -> tuple[int, str]:
    _assert_outside_timed_batch("postflight key hashing")
    _sha(problem_sha256, "problem_sha256")
    digest = hashlib.sha256(_POSTFLIGHT_DOMAIN + problem_sha256.encode("ascii") + b"\x00")
    count = 0
    for event in events:
        payload = _event_bytes(event)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    return count, digest.hexdigest()


def _complete_postflight_summary(
    problem: H4NeutralProblem,
    observed: Iterable[H4PostflightEventKey],
    *,
    timed_batch_end_nanoseconds: int,
    event_spans: Iterable[tuple[H4PostflightEventKey, int, int]],
) -> H4PostflightScheduleSummary:
    _assert_outside_timed_batch("postflight schedule finalization")
    expected_count, expected_digest = _postflight_key_stream_digest(
        problem.canonical_sha256, _iter_expected_postflight_event_keys(problem),
    )
    observed_events = iter(observed)
    observed_digest = hashlib.sha256(
        _POSTFLIGHT_DOMAIN + problem.canonical_sha256.encode("ascii") + b"\x00"
    )
    expected_events = _iter_expected_postflight_event_keys(problem)
    observed_count = 0
    mismatch_index: int | None = None
    first_expected: H4PostflightEventKey | None = None
    first_observed: H4PostflightEventKey | None = None
    sentinel = object()
    while True:
        expected_event = next(expected_events, sentinel)
        observed_event = next(observed_events, sentinel)
        if expected_event is sentinel and observed_event is sentinel:
            break
        if observed_event is not sentinel:
            payload = _event_bytes(observed_event)  # type: ignore[arg-type]
            observed_digest.update(len(payload).to_bytes(8, "big"))
            observed_digest.update(payload)
            observed_count += 1
        if mismatch_index is None and expected_event != observed_event:
            mismatch_index = max(0, observed_count - (0 if observed_event is sentinel else 1))
            first_expected = None if expected_event is sentinel else expected_event  # type: ignore[assignment]
            first_observed = None if observed_event is sentinel else observed_event  # type: ignore[assignment]
    timing_count = 0
    first_timing: H4PostflightTimingWitness | None = None
    prior_end = timed_batch_end_nanoseconds
    for event, start, end in event_spans:
        violation = (
            type(start) is not int or type(end) is not int
            or start < timed_batch_end_nanoseconds or start < prior_end or end < start
        )
        if violation:
            timing_count += 1
            if first_timing is None:
                first_timing = H4PostflightTimingWitness(
                    event, timed_batch_end_nanoseconds, max(0, start), max(0, end),
                )
        prior_end = max(prior_end, end)
    observed_sha = observed_digest.hexdigest()
    complete = (
        expected_count == observed_count and expected_digest == observed_sha
        and mismatch_index is None and timing_count == 0
    )
    return H4PostflightScheduleSummary(
        "vfe4.h4.postflight-event-key-stream.v1", expected_count, observed_count,
        expected_digest, observed_sha, mismatch_index, first_expected, first_observed,
        timing_count, first_timing, complete,
    )


def _incomplete_postflight_summary_for_test(horizon: int) -> H4PostflightScheduleSummary:
    """Return a bounded typed prefix summary used by failure-carrier tests.

    This is deliberately private and never appears on a completed execution path.
    """
    if horizon not in (7, 15, 31):
        raise ValueError("H4 horizon must be 7, 15, or 31")
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=horizon)  # type: ignore[arg-type]
    first = next(_iter_expected_postflight_event_keys(problem))
    count, digest = _postflight_key_stream_digest(
        problem.canonical_sha256, _iter_expected_postflight_event_keys(problem),
    )
    empty = hashlib.sha256(
        _POSTFLIGHT_DOMAIN + problem.canonical_sha256.encode("ascii") + b"\x00"
    ).hexdigest()
    return H4PostflightScheduleSummary(
        "vfe4.h4.postflight-event-key-stream.v1", count, 0, digest, empty,
        0, first, None, 0, None, False,
    )


def _probe_windows_power_scheme() -> str:
    completed = subprocess.run(
        ["powercfg", "/getactivescheme"], check=True, capture_output=True,
        text=True, timeout=10,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("powercfg returned no active power scheme")
    return value


def _probe_linux_policy(filename: str) -> str:
    observations: list[str] = []
    for directory in sorted(
        Path("/sys/devices/system/cpu/cpufreq").glob("policy*"),
        key=lambda item: item.name,
    ):
        path = directory / filename
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                observations.append(f"{directory.name}={value}")
    if not observations:
        raise RuntimeError(f"no Linux {filename} observations were available")
    return ",".join(observations)


def _probe_macos_low_power_mode() -> str:
    completed = subprocess.run(
        ["pmset", "-g", "custom"], check=True, capture_output=True,
        text=True, timeout=10,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if "lowpowermode" in line.lower()]
    if not lines:
        raise RuntimeError("pmset returned no low-power-mode observation")
    return ",".join(lines)


def _available_power_field(
    name: str, source: str, probe: Callable[[], str],
) -> H4PowerPolicyField:
    try:
        value = probe()
        if type(value) is not str or not value:
            raise RuntimeError("power-policy probe returned an empty value")
        if len(value) > _POWER_VALUE_LIMIT:
            raise RuntimeError("power-policy observation exceeded 4096 code points")
        return H4PowerPolicyField(name, "available", source, value, None)  # type: ignore[arg-type]
    except Exception as error:
        return H4PowerPolicyField(
            name, "unavailable", source, None, _stable_error(error),
        )  # type: ignore[arg-type]


def _capture_power_policy_fields() -> tuple[
    H4PowerPolicyField, H4PowerPolicyField, H4PowerPolicyField, H4PowerPolicyField,
]:
    _assert_outside_timed_batch("power-policy capture")
    system = platform.system()
    if system == "Windows":
        records = (
            _available_power_field("active_power_scheme", "powercfg", _probe_windows_power_scheme),
            H4PowerPolicyField("cpu_frequency_governor", "not_applicable", "none", None, None),
            H4PowerPolicyField("energy_performance_preference", "not_applicable", "none", None, None),
            H4PowerPolicyField("low_power_mode", "not_applicable", "none", None, None),
        )
    elif system == "Linux":
        records = (
            H4PowerPolicyField("active_power_scheme", "not_applicable", "none", None, None),
            _available_power_field(
                "cpu_frequency_governor", "linux_sysfs",
                lambda: _probe_linux_policy("scaling_governor"),
            ),
            _available_power_field(
                "energy_performance_preference", "linux_sysfs",
                lambda: _probe_linux_policy("energy_performance_preference"),
            ),
            H4PowerPolicyField("low_power_mode", "not_applicable", "none", None, None),
        )
    elif system == "Darwin":
        records = (
            H4PowerPolicyField("active_power_scheme", "not_applicable", "none", None, None),
            H4PowerPolicyField("cpu_frequency_governor", "not_applicable", "none", None, None),
            H4PowerPolicyField("energy_performance_preference", "not_applicable", "none", None, None),
            _available_power_field("low_power_mode", "pmset", _probe_macos_low_power_mode),
        )
    else:
        records = tuple(
            H4PowerPolicyField(name, "not_applicable", "none", None, None)
            for name in _POWER_NAMES
        )
    return records  # type: ignore[return-value]


def _physical_cpu_count() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        value = psutil.cpu_count(logical=False)
        return value if type(value) is int and value > 0 else None
    except Exception:
        return None


def _numpy_config_text() -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    value = stream.getvalue().strip()
    return value or "numpy configuration unavailable"


def _windows_registry_processor_identity() -> str | None:
    """Read a stable host CPU identity when Windows omits process variables."""

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            values: list[str] = []
            for name in ("ProcessorNameString", "Identifier", "VendorIdentifier"):
                try:
                    value = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    continue
                if type(value) is str and value.strip() and value.strip() not in values:
                    values.append(value.strip())
        return " | ".join(values) or None
    except (ImportError, OSError):
        return None


def _processor_identity(system_raw: str) -> str | None:
    if system_raw == "Windows":
        registry_identity = _windows_registry_processor_identity()
        if registry_identity is not None:
            return registry_identity
    for probe in (platform.processor, platform.machine):
        value = probe()
        if type(value) is str and value.strip():
            return value.strip()
    return None


def _capture_environment() -> H4EnvironmentRecord:
    _assert_outside_timed_batch("environment capture")
    unavailable: list[str] = []
    clock = time.get_clock_info("perf_counter")
    system_raw = platform.system()
    system: Literal["Windows", "Linux", "Darwin", "Other"] = (
        system_raw if system_raw in ("Windows", "Linux", "Darwin") else "Other"
    )  # type: ignore[assignment]
    processor = _processor_identity(system_raw)
    if processor is None:
        processor = "unavailable"
        unavailable.append("processor")
    platform_text = platform.platform()
    if not platform_text:
        platform_text = "unavailable"
        unavailable.append("platform")
    try:
        affinity: tuple[int, ...] | None = process_cpu_affinity()
    except RuntimeError:
        affinity = None
        unavailable.append("affinity_cpu_ids")
    logical = os.cpu_count()
    if type(logical) is not int or logical <= 0:
        logical = len(affinity) if affinity is not None else 1
        unavailable.append("logical_cpu_count")
    physical = _physical_cpu_count()
    if physical is None:
        unavailable.append("physical_cpu_count")
    torch_text = str(torch.__config__.show()).strip() or "torch configuration unavailable"
    numpy_text = _numpy_config_text()
    cuda = bool(torch.cuda.is_available())
    if cuda:
        unavailable.append("cuda_available")
    environment_variables = tuple(
        (name, name in os.environ, os.environ.get(name)) for name in _ENVIRONMENT_NAMES
    )
    policy = _capture_power_policy_fields()
    return H4EnvironmentRecord(
        clock.implementation, float(clock.resolution), bool(clock.monotonic),
        processor, platform_text, system, affinity, logical, physical,
        str(torch.__version__), str(np.__version__), torch_text,
        hashlib.sha256(torch_text.encode()).hexdigest(), numpy_text,
        hashlib.sha256(numpy_text.encode()).hexdigest(), False,
        environment_variables, policy, True, tuple(unavailable),
        not any(
            name in unavailable for name in (
                "clock", "processor", "platform", "affinity_cpu_ids", "logical_cpu_count",
                "torch_version", "numpy_version", "torch_config", "numpy_blas",
                "cuda_available", "environment_variables",
            )
        ),
    )


def _recursive_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError("payload mapping keys must be nonempty strings")
            keys.add(key)
            keys.update(_recursive_mapping_keys(item))
    elif type(value) in (tuple, list):
        for item in value:
            keys.update(_recursive_mapping_keys(item))
    return keys


def _assert_finite_json(value: object, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} mapping key must be nonempty string")
            _assert_finite_json(item, f"{path}.{key}")
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite float")
    elif type(value) not in (str, int, bool) and value is not None:
        raise ValueError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _compact_json_bytes(value: object) -> bytes:
    _assert_outside_timed_batch("JSON serialization")
    _assert_finite_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _thaw_typed(value: object) -> object:
    _assert_outside_timed_batch("typed artifact thaw")
    if isinstance(value, GateStatus):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        module = type(value).__module__
        if not (
            module.startswith("verification.h4")
            or module.startswith("vfe4.types.h4")
            or module == "vfe4.types.results"
        ):
            raise ValueError(f"artifact rejects unexpected dataclass {type(value).__name__}")
        return {field.name: _thaw_typed(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _thaw_typed(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_typed(item) for item in value]
    if type(value) in (str, int, float, bool) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("artifact rejects nonfinite floats")
        return value
    if isinstance(value, (torch.Tensor, np.ndarray)):
        raise ValueError("artifact rejects full numerical arrays")
    raise ValueError(f"artifact rejects unsupported value {type(value).__name__}")


def _solve_payload_size_fixed_point(
    builder: Callable[[H4PayloadSizeRecord], dict[str, object]],
    *,
    maximum_bytes: int,
) -> tuple[H4PayloadSizeRecord, dict[str, object]]:
    _assert_outside_timed_batch("payload-size fixed point")
    if not callable(builder) or maximum_bytes != _MAX_PAYLOAD_BYTES:
        raise ValueError("H4 payload fixed-point inputs are frozen")
    observed = 0
    for iteration in range(1, 5):
        size = H4PayloadSizeRecord(
            "utf8-compact-sorted-key-json-v1", observed, _MAX_PAYLOAD_BYTES,
            iteration, observed <= _MAX_PAYLOAD_BYTES,
        )
        payload = builder(size)
        if type(payload) is not dict:
            raise ValueError("payload builder must return a fresh exact dictionary")
        encoded = _compact_json_bytes(payload)
        if len(encoded) == observed:
            return size, payload
        observed = len(encoded)
    raise RuntimeError("H4 payload-size fixed point did not converge within four iterations")


def h4_validation_artifact(evaluation: H4GateEvaluation) -> H4ValidationArtifact:
    _assert_outside_timed_batch("H4 artifact construction")
    if type(evaluation) is not H4GateEvaluation:
        raise ValueError("H4 artifact construction requires an exact gate evaluation")
    return H4ValidationArtifact(
        "vfe4-validation-h4-v1", evaluation.payload_representation,
        evaluation.maximum_payload_bytes, "H4", evaluation.result.status,
        evaluation.h4_config_sha256, evaluation.result, evaluation.anchors,
        evaluation.unavailable_phases, evaluation.problems, evaluation.allowances,
        evaluation.coverage, evaluation.condition_summaries, evaluation.raw_timings,
        evaluation.primary_timed_order_balance, evaluation.timing_summary,
        evaluation.bootstrap_interval, evaluation.interval_decision,
        evaluation.thread_state, evaluation.environment, evaluation.payload_size,
        evaluation.bounded_claim, evaluation.nonclaims,
    )


def _artifact_payload(artifact: H4ValidationArtifact) -> dict[str, object]:
    if type(artifact) is not H4ValidationArtifact:
        raise ValueError("artifact payload requires the exact H4 artifact")
    thawed = _thaw_typed(artifact)
    if type(thawed) is not dict:
        raise RuntimeError("typed H4 artifact did not thaw to a dictionary")
    expected = tuple(field.name for field in fields(H4ValidationArtifact))
    if tuple(thawed) != expected:
        raise RuntimeError("H4 artifact top-level field order drifted")
    forbidden = {"native_result", "terminal_law", "replayed_result", "precision", "covariance"}
    if forbidden.intersection(_recursive_mapping_keys(thawed)):
        raise RuntimeError("H4 compact artifact contains a forbidden full-array field")
    return thawed


def h4_validation_payload(artifact: H4ValidationArtifact) -> dict[str, object]:
    _assert_outside_timed_batch("H4 validation serialization")
    if type(artifact) is not H4ValidationArtifact:
        raise ValueError("H4 validation payload requires the exact H4 artifact")
    if not artifact.payload_size.within_limit:
        raise RuntimeError("H4 validation payload exceeds the 67108864-byte ceiling")
    payload = _artifact_payload(artifact)
    observed = len(_compact_json_bytes(payload))
    if observed != artifact.payload_size.observed_bytes:
        raise RuntimeError("H4 payload byte length does not equal its fixed-point record")
    return payload


class _CanonicalStreamBuilder:
    __slots__ = ("domain", "digest", "record_count", "scalar_count", "byte_count")

    def __init__(self, domain: str) -> None:
        _assert_outside_timed_batch("canonical stream creation")
        if domain not in _STREAM_DOMAINS:
            raise ValueError("unknown canonical H4 stream domain")
        initial = domain.encode("ascii") + b"\x00"
        self.domain = domain
        self.digest = hashlib.sha256(initial)
        self.record_count = 0
        self.scalar_count = 0
        self.byte_count = len(initial)

    def _write(self, value: bytes) -> None:
        self.digest.update(value)
        self.byte_count += len(value)

    def header(self, value: Mapping[str, object]) -> None:
        payload = _compact_json_bytes(value)
        prefix = len(payload).to_bytes(8, "big")
        self._write(prefix)
        self._write(payload)

    def floats(self, values: object) -> None:
        array = np.asarray(values, dtype=np.dtype("<f8"), order="C").reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("canonical stream float lane is nonfinite")
        payload = array.tobytes(order="C")
        self._write(payload)
        self.scalar_count += int(array.size)

    def finish(self, *, record_count: int | None = None) -> H4CanonicalStreamDigest:
        if record_count is not None:
            self.record_count = record_count
        return H4CanonicalStreamDigest(
            self.domain, self.record_count, self.scalar_count, self.byte_count,
            self.digest.hexdigest(),
        )


def _selected_hash(
    *,
    domain: Literal["vfe4.h4.selected-mean.v1", "vfe4.h4.selected-covariance.v1"],
    problem_sha256: str,
    repetition_identity: str,
    arm: str,
    name: str,
    coordinate_indices: tuple[int, ...],
    parent_dimension: int,
    selected_dimension: int,
    values: object,
) -> str:
    _assert_outside_timed_batch("selected-moment hashing")
    _sha(problem_sha256, "problem_sha256")
    array = np.asarray(values, dtype=np.dtype("<f8"), order="C").reshape(-1)
    expected = selected_dimension if domain.endswith("mean.v1") else selected_dimension ** 2
    if array.size != expected or not np.all(np.isfinite(array)):
        raise ValueError("selected-moment hash lane has the wrong finite shape")
    header = {
        "problem_sha256": problem_sha256,
        "repetition_identity": repetition_identity,
        "arm": arm,
        "name": name,
        "coordinate_indices": coordinate_indices,
        "parent_dimension": parent_dimension,
        "selected_dimension": selected_dimension,
        "scalar_count": expected,
    }
    payload = _compact_json_bytes(header)
    digest = hashlib.sha256(domain.encode("ascii") + b"\x00")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _selected_summary(
    *,
    problem_sha256: str,
    repetition_identity: str,
    arm: str,
    coordinate_indices: tuple[int, ...],
    parent_dimension: int,
    selected: H4SelectedMoment | object,
) -> H4SelectedMomentSummary:
    _assert_outside_timed_batch("selected-moment compaction")
    name = getattr(selected, "name")
    mean = np.asarray(getattr(selected, "mean"), dtype=np.float64)
    covariance = np.asarray(getattr(selected, "covariance"), dtype=np.float64)
    dimension = len(coordinate_indices)
    if mean.shape != (dimension,) or covariance.shape != (dimension, dimension):
        raise ValueError("selected-moment arrays do not match their bound indices")
    return H4SelectedMomentSummary(
        name, coordinate_indices, dimension, dimension,
        _selected_hash(
            domain="vfe4.h4.selected-mean.v1", problem_sha256=problem_sha256,
            repetition_identity=repetition_identity, arm=arm, name=name,
            coordinate_indices=coordinate_indices, parent_dimension=parent_dimension,
            selected_dimension=dimension, values=mean,
        ),
        float(np.max(np.abs(mean), initial=0.0)), dimension ** 2,
        _selected_hash(
            domain="vfe4.h4.selected-covariance.v1", problem_sha256=problem_sha256,
            repetition_identity=repetition_identity, arm=arm, name=name,
            coordinate_indices=coordinate_indices, parent_dimension=parent_dimension,
            selected_dimension=dimension, values=covariance,
        ),
        float(np.trace(covariance)), float(np.max(np.abs(covariance), initial=0.0)),
    )


def _oracle_stream(oracle: H4OracleEvaluation) -> H4CanonicalStreamDigest:
    _assert_outside_timed_batch("oracle stream hashing")
    if type(oracle) is not H4OracleEvaluation:
        raise ValueError("oracle stream requires the exact NumPy oracle record")
    builder = _CanonicalStreamBuilder("vfe4.h4.oracle-evaluation-stream.v1")
    builder.header({
        "schema_version": oracle.schema_version,
        "problem_id": oracle.problem_id,
        "problem_sha256": oracle.problem_sha256,
        "source_kind": oracle.source_kind,
        "seed": oracle.seed,
        "kind": oracle.kind,
        "horizon": oracle.horizon,
        "d_z": oracle.d_z,
        "d_m": oracle.d_m,
        "dimension": oracle.dimension,
        "coordinate_order": oracle.coordinate_order,
        "factor_ids": oracle.factor_ids,
        "selected_blocks": tuple({
            "name": item.name,
            "coordinate_indices": item.coordinate_indices,
            "dimension": len(item.coordinate_indices),
        } for item in oracle.selected_moments),
    })
    for lane in (
        oracle.precision, oracle.natural, (oracle.constant,), oracle.mean,
        oracle.covariance, (oracle.canonical_log_normalizer,),
        (oracle.predictive_log_normalizer,),
    ):
        builder.floats(lane)
    for selected in oracle.selected_moments:
        builder.floats(selected.mean)
        builder.floats(selected.covariance)
    stream = builder.finish(record_count=1)
    selected_dimension = oracle.d_z + oracle.d_m
    expected = (
        2 * oracle.dimension ** 2 + 2 * oracle.dimension + 3
        + (oracle.horizon + 2) * (selected_dimension + selected_dimension ** 2)
    )
    if stream.scalar_count != expected:
        raise RuntimeError("oracle canonical stream scalar count drifted")
    return stream


def _native_stream(result: H4SolverResult) -> H4CanonicalStreamDigest:
    _assert_outside_timed_batch("native result hashing")
    if type(result) is not H4SolverResult:
        raise ValueError("native stream requires the exact solver result")
    builder = _CanonicalStreamBuilder("vfe4.h4.native-result-stream.v1")
    builder.header({
        "problem_id": result.problem_id, "problem_sha256": result.problem_sha256,
        "arm": result.arm, "protocol_id": result.protocol_id,
        "factor_count": result.factor_count,
    })
    if result.arm == "information":
        assert result.native_information is not None
        native = result.native_information
        for lane in (native.h, native.J, native.mean, (native.complete_objective,)):
            builder.floats(lane)
    else:
        assert result.native_moment is not None
        native = result.native_moment
        for lane in (native.mean, native.covariance, (native.complete_objective,)):
            builder.floats(lane)
    dimension = len(native.mean)
    expected = dimension ** 2 + (2 * dimension if result.arm == "information" else dimension) + 1
    stream = builder.finish(record_count=1)
    if stream.scalar_count != expected:
        raise RuntimeError("native canonical stream scalar count drifted")
    return stream


def _terminal_stream(
    terminal: H4TerminalLaw,
    *,
    problem_id: str,
    problem_sha256: str,
    coordinate_indices: tuple[tuple[str, tuple[int, ...]], ...],
) -> H4CanonicalStreamDigest:
    _assert_outside_timed_batch("terminal result hashing")
    if type(terminal) is not H4TerminalLaw:
        raise ValueError("terminal stream requires the exact common terminal law")
    if tuple(item.name for item in terminal.selected_moments) != tuple(item[0] for item in coordinate_indices):
        raise ValueError("terminal selected labels do not match bound coordinate indices")
    builder = _CanonicalStreamBuilder("vfe4.h4.terminal-law-stream.v1")
    builder.header({
        "problem_id": problem_id, "problem_sha256": problem_sha256,
        "arm": terminal.arm,
        "selected_blocks": tuple({"name": name, "coordinate_indices": indices} for name, indices in coordinate_indices),
    })
    for lane in (terminal.h, terminal.J, terminal.mean):
        builder.floats(lane)
    for selected in terminal.selected_moments:
        builder.floats(selected.mean)
        builder.floats(selected.covariance)
    builder.floats((terminal.complete_objective, terminal.stopping_residual))
    dimension = len(terminal.mean)
    selected_scalars = sum(len(indices) + len(indices) ** 2 for _, indices in coordinate_indices)
    expected = dimension ** 2 + 2 * dimension + selected_scalars + 2
    stream = builder.finish(record_count=1)
    if stream.scalar_count != expected:
        raise RuntimeError("terminal canonical stream scalar count drifted")
    return stream


def _diagnostic_stream(diagnostics: H4NativeDiagnostics) -> H4CanonicalStreamDigest:
    _assert_outside_timed_batch("native diagnostic hashing")
    if type(diagnostics) is not H4NativeDiagnostics:
        raise ValueError("diagnostic stream requires an exact native diagnostic record")
    replayed = _native_stream(diagnostics.replayed_result)
    builder = _CanonicalStreamBuilder("vfe4.h4.native-diagnostic-stream.v1")
    builder.header({
        "problem_id": diagnostics.problem_id,
        "problem_sha256": diagnostics.problem_sha256,
        "protocol_id": diagnostics.protocol_id,
        "arm": diagnostics.arm,
        "factor_count": diagnostics.factor_count,
        "replayed_native_sha256": replayed.sha256,
        "innovation_records": tuple({
            "factor_id": item.factor_id,
            "time_index": item.time_index,
            "parent_coordinate_indices": item.parent_coordinate_indices,
            "innovation_dimension": item.innovation_dimension,
        } for item in diagnostics.innovation_diagnostics),
    })
    for item in diagnostics.innovation_diagnostics:
        builder.floats((
            item.minimum_eigenvalue, item.maximum_eigenvalue,
            item.condition_number, item.minimum_cholesky_pivot,
        ))
    return builder.finish(record_count=len(diagnostics.innovation_diagnostics))


def _kl_summary(value: H4OracleKLEvaluation) -> H4CompactKLSummary:
    if type(value) is not H4OracleKLEvaluation:
        raise ValueError("compact KL requires the exact NumPy KL evaluation")
    return H4CompactKLSummary(**asdict(value))


def _terminal_precision(terminal: H4TerminalLaw) -> tuple[tuple[float, ...], ...]:
    return terminal.J


def _native_objective(result: H4SolverResult) -> float:
    native = result.native_information if result.arm == "information" else result.native_moment
    assert native is not None
    return native.complete_objective


def _compact_result_from_kl(
    *,
    result: H4SolverResult,
    terminal: H4TerminalLaw,
    oracle: H4OracleEvaluation,
    repetition_index: int | None,
    kl: H4OracleKLEvaluation,
) -> H4CompactResultRecord:
    _assert_outside_timed_batch("result compaction")
    if type(kl) is not H4OracleKLEvaluation:
        raise ValueError("result compaction requires an exact KL evaluation")
    native_stream = _native_stream(result)
    coordinate_indices = tuple(
        (item.name, item.coordinate_indices) for item in oracle.selected_moments
    )
    terminal_stream = _terminal_stream(
        terminal, problem_id=result.problem_id, problem_sha256=result.problem_sha256,
        coordinate_indices=coordinate_indices,
    )
    repetition_identity = (
        "anchor" if repetition_index is None else f"repetition[{repetition_index}]"
    )
    summaries = tuple(
        _selected_summary(
            problem_sha256=result.problem_sha256,
            repetition_identity=repetition_identity,
            arm=result.arm,
            coordinate_indices=indices,
            parent_dimension=oracle.dimension,
            selected=selected,
        )
        for selected, (_, indices) in zip(
            terminal.selected_moments, coordinate_indices, strict=True,
        )
    )
    return H4CompactResultRecord(
        result.problem_id, result.problem_sha256, oracle.source_kind,
        repetition_index, result.arm, native_stream, terminal_stream,
        _kl_summary(kl), _native_objective(result), terminal.complete_objective,
        terminal.stopping_residual, summaries,
    )


def _compact_result(
    *,
    result: H4SolverResult,
    terminal: H4TerminalLaw,
    oracle: H4OracleEvaluation,
    repetition_index: int | None,
) -> tuple[H4CompactResultRecord, H4OracleKLEvaluation]:
    kl = reverse_kl_to_h4_oracle(
        oracle, mean=terminal.mean, precision=_terminal_precision(terminal),
    )
    return (
        _compact_result_from_kl(
            result=result, terminal=terminal, oracle=oracle,
            repetition_index=repetition_index, kl=kl,
        ),
        kl,
    )


def _float_hex_json(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _float_hex_json(getattr(value, field.name)) for field in fields(value)}
    if type(value) is tuple:
        return [_float_hex_json(item) for item in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("condition stream rejects nonfinite floats")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError("condition stream encountered unsupported record value")


class _ConditionAccumulator:
    __slots__ = (
        "name", "expected", "problem_id", "problem_sha256", "digest", "count",
        "eligible", "ineligible", "extrema", "first_ineligible",
    )

    def __init__(
        self,
        name: str,
        expected: int,
        *,
        problem_id: str | None = None,
        problem_sha256: str | None = None,
    ) -> None:
        if name not in _CONDITION_NAMES or type(expected) is not int or expected <= 0:
            raise ValueError("condition accumulator identity is invalid")
        if (problem_id is None) != (problem_sha256 is None):
            raise ValueError("per-problem condition identity must be jointly present")
        self.name = name
        self.expected = expected
        self.problem_id = problem_id
        self.problem_sha256 = problem_sha256
        if problem_id is None:
            initial = b"vfe4.h4.condition-record-stream.v1\x00" + name.encode("ascii") + b"\x00"
        else:
            _identity(problem_id, problem_sha256)
            initial = (
                b"vfe4.h4.problem-condition-record-stream.v1\x00"
                + problem_sha256.encode("ascii") + b"\x00" + name.encode("ascii") + b"\x00"
            )
        self.digest = hashlib.sha256(initial)
        self.count = 0
        self.eligible = 0
        self.ineligible = 0
        self.extrema: dict[str, tuple[float, int, H4PosteriorConditionRecord | H4InnovationConditionRecord]] = {}
        self.first_ineligible: tuple[int, H4PosteriorConditionRecord | H4InnovationConditionRecord] | None = None

    def consume(self, record: H4PosteriorConditionRecord | H4InnovationConditionRecord) -> None:
        _assert_outside_timed_batch("condition accumulation")
        posterior = "posterior" in self.name
        expected_type = H4PosteriorConditionRecord if posterior else H4InnovationConditionRecord
        if type(record) is not expected_type:
            raise ValueError("condition record type does not match its stream")
        if self.problem_id is not None and (
            record.problem_id != self.problem_id or record.problem_sha256 != self.problem_sha256
        ):
            raise ValueError("per-problem condition stream received another identity")
        payload = _compact_json_bytes(_float_hex_json(record))
        self.digest.update(len(payload).to_bytes(8, "big"))
        self.digest.update(payload)
        index = self.count
        self.count += 1
        if record.eligible:
            self.eligible += 1
        else:
            self.ineligible += 1
            if self.first_ineligible is None:
                self.first_ineligible = (index, record)
        metrics: tuple[tuple[str, float, bool], ...] = (
            ("minimum_eigenvalue", record.minimum_eigenvalue, True),
            ("maximum_eigenvalue", record.maximum_eigenvalue, False),
            ("maximum_condition_number", record.condition_number, False),
            *((
                ("minimum_cholesky_pivot", record.minimum_cholesky_pivot, True),
                ("maximum_mean_infinity_norm", record.mean_infinity_norm, False),
            ) if posterior else ()),
        )
        for metric, scalar, minimum in metrics:
            current = self.extrema.get(metric)
            if current is None or (scalar < current[0] if minimum else scalar > current[0]):
                self.extrema[metric] = (scalar, index, record)

    def finish(self) -> H4ConditionStreamSummary | H4ProblemConditionSummary:
        if self.count != self.expected or self.eligible + self.ineligible != self.count:
            raise ValueError("condition stream is incomplete")
        metric_order = (
            ("minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number")
            if "innovation" in self.name else
            (
                "minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number",
                "minimum_cholesky_pivot", "maximum_mean_infinity_norm",
            )
        )
        witnesses = [
            H4ConditionWitness(metric, self.extrema[metric][1], self.extrema[metric][2])
            for metric in metric_order
        ]
        if self.first_ineligible is not None:
            witnesses.append(H4ConditionWitness(
                "first_ineligible", self.first_ineligible[0], self.first_ineligible[1],
            ))
        common = (
            self.name,
            self.expected,
            self.count,
            self.digest.hexdigest(),
            self.eligible,
            self.ineligible,
            tuple(witnesses),
            self.ineligible == 0,
        )
        if self.problem_id is None:
            return H4ConditionStreamSummary(
                common[0], "vfe4.h4.condition-record-stream.v1", *common[1:],
            )  # type: ignore[arg-type]
        return H4ProblemConditionSummary(
            self.problem_id, self.problem_sha256, common[0],
            "vfe4.h4.problem-condition-record-stream.v1", *common[1:],
        )  # type: ignore[arg-type]


def _oracle_posterior_condition(
    oracle: H4OracleEvaluation, config: H4ValidationConfig,
) -> H4PosteriorConditionRecord:
    diagnostic = oracle.posterior_diagnostic
    return posterior_condition_record(
        problem_id=oracle.problem_id, problem_sha256=oracle.problem_sha256,
        source="numpy_oracle", repetition_index=None, dimension=diagnostic.dimension,
        minimum_eigenvalue=diagnostic.minimum_eigenvalue,
        maximum_eigenvalue=diagnostic.maximum_eigenvalue,
        condition_number=diagnostic.condition_number,
        minimum_cholesky_pivot=diagnostic.minimum_cholesky_pivot,
        mean_infinity_norm=diagnostic.mean_infinity_norm,
        envelope=config.condition_envelope,
    )


def _oracle_innovation_conditions(
    oracle: H4OracleEvaluation, config: H4ValidationConfig,
) -> tuple[H4InnovationConditionRecord, ...]:
    return tuple(
        innovation_condition_record(
            problem_id=oracle.problem_id, problem_sha256=oracle.problem_sha256,
            source="numpy_oracle", repetition_index=None,
            factor_id=item.factor_id, time_index=item.time_index,
            parent_coordinate_indices=item.parent_coordinate_indices,
            innovation_dimension=item.innovation_dimension,
            minimum_eigenvalue=item.minimum_eigenvalue,
            maximum_eigenvalue=item.maximum_eigenvalue,
            condition_number=item.condition_number,
            envelope=config.condition_envelope,
        )
        for item in oracle.innovation_diagnostics
    )


def _terminal_posterior_condition(
    terminal: H4TerminalLaw,
    *,
    problem_id: str,
    problem_sha256: str,
    repetition_index: int,
    config: H4ValidationConfig,
) -> H4PosteriorConditionRecord:
    precision = np.asarray(terminal.J, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(precision)
    cholesky = np.linalg.cholesky(precision)
    source: Literal["information", "moment"] = terminal.arm
    return posterior_condition_record(
        problem_id=problem_id, problem_sha256=problem_sha256, source=source,
        repetition_index=repetition_index, dimension=precision.shape[0],
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        minimum_cholesky_pivot=float(np.min(np.diag(cholesky))),
        mean_infinity_norm=float(np.max(np.abs(np.asarray(terminal.mean)))),
        envelope=config.condition_envelope,
    )


def _moment_innovation_condition(
    diagnostic: H4InnovationDiagnostic,
    *,
    problem_id: str,
    problem_sha256: str,
    repetition_index: int,
    config: H4ValidationConfig,
) -> H4InnovationConditionRecord:
    if type(diagnostic) is not H4InnovationDiagnostic:
        raise ValueError("moment innovation condition requires an exact diagnostic")
    return innovation_condition_record(
        problem_id=problem_id, problem_sha256=problem_sha256,
        source="moment", repetition_index=repetition_index,
        factor_id=diagnostic.factor_id, time_index=diagnostic.time_index,
        parent_coordinate_indices=diagnostic.parent_coordinate_indices,
        innovation_dimension=diagnostic.innovation_dimension,
        minimum_eigenvalue=diagnostic.minimum_eigenvalue,
        maximum_eigenvalue=diagnostic.maximum_eigenvalue,
        condition_number=diagnostic.condition_number,
        envelope=config.condition_envelope,
    )


def _moment_innovation_conditions(
    diagnostics: H4NativeDiagnostics,
    *,
    repetition_index: int,
    config: H4ValidationConfig,
) -> tuple[H4InnovationConditionRecord, ...]:
    if diagnostics.arm != "moment":
        raise ValueError("moment innovation conditions require the moment diagnostic replay")
    return tuple(
        _moment_innovation_condition(
            item, problem_id=diagnostics.problem_id,
            problem_sha256=diagnostics.problem_sha256,
            repetition_index=repetition_index, config=config,
        )
        for item in diagnostics.innovation_diagnostics
    )


def _coverage_key(**values: object) -> str:
    return _compact_json_bytes(values).decode("utf-8")


def _expected_coverage_keys(
    name: str, problems: tuple[H4NeutralProblem, ...],
) -> Iterator[str]:
    if name not in _COVERAGE_NAMES or len(problems) != 120:
        raise ValueError("coverage-key generation requires the full frozen traversal")
    for problem in problems:
        if name == "oracle_posterior":
            yield _coverage_key(problem_id=problem.problem_id, problem_sha256=problem.canonical_sha256)
        elif name in ("terminal_posterior", "native_replay"):
            for repetition in range(11):
                for arm in ("information", "moment"):
                    yield _coverage_key(
                        problem_id=problem.problem_id,
                        problem_sha256=problem.canonical_sha256,
                        repetition_index=repetition, arm=arm,
                    )
        elif name == "oracle_innovation":
            for factor in problem.factor_schedule:
                if factor.role == "observation":
                    yield _coverage_key(
                        problem_id=problem.problem_id,
                        problem_sha256=problem.canonical_sha256,
                        factor_id=factor.factor_id,
                    )
        elif name == "moment_innovation":
            for repetition in range(11):
                for factor in problem.factor_schedule:
                    if factor.role == "observation":
                        yield _coverage_key(
                            problem_id=problem.problem_id,
                            problem_sha256=problem.canonical_sha256,
                            repetition_index=repetition, factor_id=factor.factor_id,
                        )
        elif name in ("operation_pass", "memory_pass"):
            for arm in ("information", "moment"):
                yield _coverage_key(
                    problem_id=problem.problem_id,
                    problem_sha256=problem.canonical_sha256, arm=arm,
                )
        elif name == "execution_trace":
            yield _coverage_key(problem_id=problem.problem_id, problem_sha256=problem.canonical_sha256)
        else:
            for event in _iter_expected_postflight_event_keys(problem):
                yield _event_bytes(event).decode("utf-8")


class _CoverageAccumulator:
    __slots__ = (
        "name", "expected_count", "expected_digest", "expected_iter",
        "observed_digest", "observed_count", "missing", "extra", "duplicates",
        "first_missing", "first_extra", "first_duplicate", "seen",
    )

    def __init__(self, name: str, problems: tuple[H4NeutralProblem, ...]) -> None:
        expected_counts = {
            "oracle_posterior": 120, "terminal_posterior": 2640,
            "oracle_innovation": 2120, "moment_innovation": 23320,
            "native_replay": 2640, "operation_pass": 240, "memory_pass": 240,
            "execution_trace": 120, "postflight_schedule": 146720,
        }
        if name not in expected_counts:
            raise ValueError("unknown H4 coverage stream")
        self.name = name
        self.expected_count = expected_counts[name]
        initial = b"vfe4.h4.coverage-key-stream.v1\x00" + name.encode("ascii") + b"\x00"
        expected_digest = hashlib.sha256(initial)
        count = 0
        for key in _expected_coverage_keys(name, problems):
            payload = key.encode("utf-8")
            expected_digest.update(len(payload).to_bytes(8, "big"))
            expected_digest.update(payload)
            count += 1
        if count != self.expected_count:
            raise RuntimeError("independent H4 coverage count drifted")
        self.expected_digest = expected_digest.hexdigest()
        self.expected_iter = iter(_expected_coverage_keys(name, problems))
        self.observed_digest = hashlib.sha256(initial)
        self.observed_count = 0
        self.missing = 0
        self.extra = 0
        self.duplicates = 0
        self.first_missing: str | None = None
        self.first_extra: str | None = None
        self.first_duplicate: str | None = None
        self.seen: set[str] = set()

    def consume(self, key: str) -> None:
        if type(key) is not str or not key:
            raise ValueError("coverage key must be a nonempty string")
        expected = next(self.expected_iter, None)
        if expected is None:
            self.extra += 1
            if self.first_extra is None:
                self.first_extra = key
        elif key != expected:
            self.missing += 1
            self.extra += 1
            if self.first_missing is None:
                self.first_missing = expected
            if self.first_extra is None:
                self.first_extra = key
        if key in self.seen:
            self.duplicates += 1
            if self.first_duplicate is None:
                self.first_duplicate = key
        else:
            self.seen.add(key)
        payload = key.encode("utf-8")
        self.observed_digest.update(len(payload).to_bytes(8, "big"))
        self.observed_digest.update(payload)
        self.observed_count += 1

    def finish(self) -> H4CoverageRecord:
        remaining_count = 0
        first_remaining: str | None = None
        for key in self.expected_iter:
            if first_remaining is None:
                first_remaining = key
            remaining_count += 1
        if remaining_count:
            self.missing += remaining_count
            if self.first_missing is None:
                self.first_missing = first_remaining
        observed = self.observed_digest.hexdigest()
        complete = (
            self.observed_count == self.expected_count
            and observed == self.expected_digest
            and self.missing == self.extra == self.duplicates == 0
        )
        self.seen.clear()
        return H4CoverageRecord(
            self.name, "vfe4.h4.coverage-key-stream.v1", self.expected_count,
            self.observed_count, self.expected_digest, observed, self.missing,
            self.extra, self.duplicates, self.first_missing, self.first_extra,
            self.first_duplicate, complete,
        )  # type: ignore[arg-type]


class _PostflightTracker:
    __slots__ = (
        "problem", "expected_iter", "expected_count", "expected_digest",
        "observed_digest", "observed_count", "first_mismatch_index",
        "first_expected", "first_observed", "timing_violations", "first_timing",
        "timed_batch_end", "prior_end", "coverage",
    )

    def __init__(
        self,
        problem: H4NeutralProblem,
        *,
        timed_batch_end_nanoseconds: int,
        coverage: _CoverageAccumulator,
    ) -> None:
        self.problem = problem
        self.expected_count, self.expected_digest = _postflight_key_stream_digest(
            problem.canonical_sha256, _iter_expected_postflight_event_keys(problem),
        )
        self.expected_iter = iter(_iter_expected_postflight_event_keys(problem))
        self.observed_digest = hashlib.sha256(
            _POSTFLIGHT_DOMAIN + problem.canonical_sha256.encode("ascii") + b"\x00"
        )
        self.observed_count = 0
        self.first_mismatch_index: int | None = None
        self.first_expected: H4PostflightEventKey | None = None
        self.first_observed: H4PostflightEventKey | None = None
        self.timing_violations = 0
        self.first_timing: H4PostflightTimingWitness | None = None
        self.timed_batch_end = timed_batch_end_nanoseconds
        self.prior_end = timed_batch_end_nanoseconds
        self.coverage = coverage

    def record(self, event: H4PostflightEventKey, start: int, end: int) -> None:
        expected = next(self.expected_iter, None)
        if self.first_mismatch_index is None and expected != event:
            self.first_mismatch_index = self.observed_count
            self.first_expected = expected
            self.first_observed = event
        payload = _event_bytes(event)
        self.observed_digest.update(len(payload).to_bytes(8, "big"))
        self.observed_digest.update(payload)
        self.coverage.consume(payload.decode("utf-8"))
        violation = (
            type(start) is not int or type(end) is not int
            or start < self.prior_end or end < start
        )
        if violation:
            self.timing_violations += 1
            if self.first_timing is None:
                self.first_timing = H4PostflightTimingWitness(
                    event, self.timed_batch_end, max(0, start), max(0, end),
                )
        self.prior_end = max(self.prior_end, end)
        self.observed_count += 1

    def finish(self) -> H4PostflightScheduleSummary:
        remaining = next(self.expected_iter, None)
        if remaining is not None and self.first_mismatch_index is None:
            self.first_mismatch_index = self.observed_count
            self.first_expected = remaining
        observed = self.observed_digest.hexdigest()
        complete = (
            self.observed_count == self.expected_count
            and observed == self.expected_digest
            and self.first_mismatch_index is None
            and self.timing_violations == 0
        )
        return H4PostflightScheduleSummary(
            "vfe4.h4.postflight-event-key-stream.v1", self.expected_count,
            self.observed_count, self.expected_digest, observed,
            self.first_mismatch_index, self.first_expected, self.first_observed,
            self.timing_violations, self.first_timing, complete,
        )


def _compact_oracle(
    oracle: H4OracleEvaluation, config: H4ValidationConfig,
) -> H4CompactOracleRecord:
    _assert_outside_timed_batch("oracle compaction")
    if type(oracle) is not H4OracleEvaluation or type(config) is not H4ValidationConfig:
        raise ValueError("oracle compaction requires exact oracle/config records")
    agreement = oracle.route_agreement
    if not agreement.eligible or agreement.eligible != (
        agreement.passed and agreement.decisive
    ):
        raise ValueError(
            "oracle route agreement is not eligible: "
            f"problem_id={agreement.problem_id!r}; "
            f"canonical_value={agreement.canonical_operand.value!r}; "
            f"canonical_rounding_depth="
            f"{agreement.canonical_operand.rounding_depth!r}; "
            f"predictive_value={agreement.predictive_operand.value!r}; "
            f"predictive_rounding_depth="
            f"{agreement.predictive_operand.rounding_depth!r}; "
            f"residual={agreement.residual!r}; "
            f"final_allowance={agreement.final_allowance!r}; "
            f"allowance_scale_ratio={agreement.allowance_scale_ratio!r}; "
            f"decisive={agreement.decisive!r}; passed={agreement.passed!r}; "
            f"eligible={agreement.eligible!r}"
        )
    posterior = _oracle_posterior_condition(oracle, config)
    innovations = _oracle_innovation_conditions(oracle, config)
    condition_accumulator = _ConditionAccumulator(
        "oracle_innovation", len(innovations), problem_id=oracle.problem_id,
        problem_sha256=oracle.problem_sha256,
    )
    for item in innovations:
        condition_accumulator.consume(item)
    innovation_summary = condition_accumulator.finish()
    assert type(innovation_summary) is H4ProblemConditionSummary
    selected = tuple(
        _selected_summary(
            problem_sha256=oracle.problem_sha256,
            repetition_identity="oracle", arm="oracle",
            coordinate_indices=item.coordinate_indices,
            parent_dimension=oracle.dimension, selected=item,
        )
        for item in oracle.selected_moments
    )
    return H4CompactOracleRecord(
        oracle.problem_id, oracle.problem_sha256, oracle.source_kind,
        oracle.dimension, _oracle_stream(oracle), oracle.canonical_log_normalizer,
        oracle.predictive_log_normalizer, oracle.route_agreement, selected,
        posterior, innovation_summary,
    )


def _integrity_check(
    materialized: H4MaterializedProblem, phase: H4MaterializedIntegrityPhase,
) -> H4MaterializedIntegrityCheck:
    _assert_outside_timed_batch("materialized-integrity hashing")
    try:
        observed = _assert_h4_materialized_integrity(materialized)
    except Exception as error:
        raise _H4IntegrityCheckFailure(
            phase, materialized.tensor_sha256, "seam_exception", None,
            _stable_error(error),
        ) from error
    if type(observed) is not str or _SHA256.fullmatch(observed) is None:
        error = RuntimeError("materialized-integrity seam returned a malformed digest")
        raise _H4IntegrityCheckFailure(
            phase, materialized.tensor_sha256, "seam_exception", None,
            _stable_error(error),
        ) from error
    if observed != materialized.tensor_sha256:
        raise _H4IntegrityCheckFailure(
            phase, materialized.tensor_sha256, "digest_mismatch", observed, None,
        )
    return H4MaterializedIntegrityCheck(
        phase, materialized.tensor_sha256, observed, True,
    )


def _materialization_identity(
    materialized: H4MaterializedProblem,
    checks: tuple[H4MaterializedIntegrityCheck, ...],
) -> H4MaterializationIdentity:
    return H4MaterializationIdentity(
        materialized.problem_id, materialized.problem_sha256,
        materialized.materialization_version, materialized.protocol_id,
        materialized.tensor_sha256, 1, True, checks,
    )


def _scaled_integrity_failure_record(
    materialized: H4MaterializedProblem,
    *,
    problem_index: int,
    horizon_index: int,
    seed_index: int,
    kind_index: int,
    checkpoint: H4ScaledMaterializedIntegrityCheckpoint,
    completed_integrity_checks: tuple[H4MaterializedIntegrityCheck, ...],
    failure: _H4IntegrityCheckFailure,
    warmup_spans: tuple[H4ArmCallSpan, ...],
    timed_spans: tuple[H4ArmCallSpan, ...],
    garbage_collector: H4GarbageCollectorRecord | None,
    postflight_schedule: H4PostflightScheduleSummary | None,
) -> H4ScaledMaterializedIntegrityFailureRecord:
    if failure.phase != checkpoint:
        raise ValueError("integrity failure phase and scaled checkpoint disagree")
    return H4ScaledMaterializedIntegrityFailureRecord(
        materialized.problem_id, materialized.problem_sha256, problem_index,
        horizon_index, seed_index, kind_index,
        materialized.materialization_version, materialized.protocol_id, 1, True,
        checkpoint, failure.expected_tensor_sha256, completed_integrity_checks,
        failure.failure_kind, failure.observed_tensor_sha256, failure.seam_error,
        warmup_spans, timed_spans, garbage_collector, postflight_schedule,
        "materialized_integrity",
    )


def _facade(
    problem_id: str, arm: H4SolverArm,
    recorder: NullOperationRecorder | CountingOperationRecorder,
) -> InstrumentedLinearAlgebra:
    _assert_outside_timed_batch("postflight facade construction")
    return InstrumentedLinearAlgebra(
        problem_id=problem_id, arm=arm, recorder=recorder,
    )


def _operation_delta(
    before: tuple[H4OperationRecord, ...], after: tuple[H4OperationRecord, ...],
) -> tuple[H4OperationRecord, ...]:
    keys = lambda item: (
        item.problem_id, item.arm, item.operation, item.operand_shapes, item.result_shape,
    )
    before_by_key = {keys(item): item.count for item in before}
    delta: list[H4OperationRecord] = []
    for item in after:
        difference = item.count - before_by_key.get(keys(item), 0)
        if difference > 0:
            delta.append(replace(item, count=difference))
        elif difference < 0:
            raise RuntimeError("operation recorder count decreased")
    if not delta:
        raise RuntimeError("terminal conversion emitted no real operations")
    return tuple(delta)


def _counting_pass(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    arm: H4SolverArm,
    reference: H4CompactResultRecord,
) -> H4CountingPassRecord:
    _assert_outside_timed_batch("operation counting pass")
    recorder = CountingOperationRecorder()
    linalg = _facade(materialized.problem_id, arm, recorder)
    replayed = _solver_for_arm(arm)(materialized, protocol, linalg)
    solver_operations = recorder.snapshot()
    native = _native_stream(replayed)
    terminal = to_common_terminal_law(materialized, replayed, linalg)
    after = recorder.snapshot()
    conversion_operations = _operation_delta(solver_operations, after)
    terminal_stream = _terminal_stream(
        terminal, problem_id=materialized.problem_id,
        problem_sha256=materialized.problem_sha256,
        coordinate_indices=tuple(
            (item.name, indices)
            for item, indices in zip(
                terminal.selected_moments,
                _selected_coordinate_indices(materialized), strict=True,
            )
        ),
    )
    return H4CountingPassRecord(
        materialized.problem_id, materialized.problem_sha256, arm, 0,
        reference.native_stream.sha256, native.sha256,
        reference.terminal_stream.sha256, terminal_stream.sha256, True,
        solver_operations, conversion_operations,
    )


def _memory_pass(
    materialized: H4MaterializedProblem,
    protocol: H4SolveProtocol,
    arm: H4SolverArm,
    reference: H4CompactResultRecord,
) -> H4MemoryPassRecord:
    _assert_outside_timed_batch("memory measurement pass")
    holder: list[H4SolverResult] = []

    def measured() -> None:
        if holder:
            raise RuntimeError("H4 memory holder received multiple solver results")
        holder.append(_solver_for_arm(arm)(
            materialized, protocol,
            _facade(materialized.problem_id, arm, NullOperationRecorder()),
        ))

    memory = measure_untimed_memory(materialized.problem_id, arm, measured)
    if len(holder) != 1:
        raise RuntimeError("H4 memory holder did not receive exactly one solver result")
    replayed = _native_stream(holder.pop())
    return H4MemoryPassRecord(
        materialized.problem_id, materialized.problem_sha256, arm, 0,
        reference.native_stream.sha256, replayed.sha256, True, memory,
    )


def _selected_coordinate_indices(
    materialized: H4MaterializedProblem,
) -> tuple[tuple[int, ...], ...]:
    initial = tuple(materialized.factor_normalized_coordinate_indices[0])
    terminal_time = materialized.horizon
    terminal = tuple(
        index for index, name in enumerate(materialized.coordinate_order)
        if name.startswith(f"z[{terminal_time},") or name.startswith(f"m[{terminal_time},")
    )
    observations = tuple(
        tuple(parents)
        for role, parents in zip(
            materialized.factor_roles,
            materialized.factor_parent_coordinate_indices,
            strict=True,
        ) if role == "observation"
    )
    result = (initial, terminal, *observations)
    if len(result) != materialized.horizon + 2 or any(len(item) != materialized.d_z + materialized.d_m for item in result):
        raise ValueError("materialized selected-coordinate schedule is incomplete")
    return result


@dataclass(slots=True)
class _PreflightProblem:
    problem: H4NeutralProblem
    payload: bytes
    materialized: H4MaterializedProblem
    oracle: H4CompactOracleRecord
    integrity_checks: list[H4MaterializedIntegrityCheck]


@dataclass(frozen=True, slots=True)
class _AnchorWork:
    evaluation: H4AnchorEvaluation
    allowance_source: H4AnchorAllowanceSource


def _evaluate_anchor(
    fixture_bytes: bytes,
    *,
    expected_fixture_id: Literal["h3-coupled-v1", "h3-zero-control-v1"],
    config: H4ValidationConfig,
) -> _AnchorWork:
    _assert_outside_timed_batch("H4 anchor evaluation")
    if type(fixture_bytes) is not bytes or not fixture_bytes:
        raise ValueError("H3 anchor bytes must be nonempty immutable bytes")
    fixture = parse_h3_fixture_bytes(
        fixture_bytes, expected_fixture_id=expected_fixture_id,
    )
    problem = h4_anchor_from_h3(fixture)
    payload = canonical_h4_problem_bytes(problem)
    oracle = evaluate_h4_oracle(payload)
    compact_oracle = _compact_oracle(oracle, config)
    if not compact_oracle.posterior_condition.eligible or not compact_oracle.innovation_conditions.all_eligible:
        raise ValueError("H3 anchor oracle condition evidence is outside the envelope")
    materialized = materialize_h4_problem(problem, config.solve_protocol)
    checks = [_integrity_check(materialized, "after_materialization")]

    results: dict[H4SolverArm, H4SolverResult] = {}
    terminals: dict[H4SolverArm, H4TerminalLaw] = {}
    compacts: dict[H4SolverArm, H4CompactResultRecord] = {}
    kls: dict[H4SolverArm, H4OracleKLEvaluation] = {}
    diagnostic_streams: dict[H4SolverArm, H4CanonicalStreamDigest] = {}
    for arm, phase in (
        ("information", "after_anchor_information"),
        ("moment", "after_anchor_moment"),
    ):
        linalg = _facade(problem.problem_id, arm, NullOperationRecorder())
        result = _solver_for_arm(arm)(materialized, config.solve_protocol, linalg)
        terminal = to_common_terminal_law(
            materialized, result,
            _facade(problem.problem_id, arm, NullOperationRecorder()),
        )
        compact, kl = _compact_result(
            result=result, terminal=terminal, oracle=oracle,
            repetition_index=None,
        )
        diagnostics = evaluate_h4_native_diagnostics(
            materialized, result,
            _facade(problem.problem_id, arm, NullOperationRecorder()),
        )
        diagnostic = _diagnostic_stream(diagnostics)
        if _native_stream(diagnostics.replayed_result).sha256 != compact.native_stream.sha256:
            raise ValueError("anchor diagnostic replay changed the native result")
        results[arm] = result
        terminals[arm] = terminal
        compacts[arm] = compact
        kls[arm] = kl
        diagnostic_streams[arm] = diagnostic
        checks.append(_integrity_check(materialized, phase))  # type: ignore[arg-type]

    identity = _materialization_identity(materialized, tuple(checks))
    evaluation = H4AnchorEvaluation(
        problem.problem_id, problem.canonical_sha256, compact_oracle, identity,
        compacts["information"], diagnostic_streams["information"],
        compacts["moment"], diagnostic_streams["moment"],
    )
    information_source = H4AllowanceResultSource(
        payload, None, oracle, results["information"], terminals["information"],
        kls["information"],
    )
    moment_source = H4AllowanceResultSource(
        payload, None, oracle, results["moment"], terminals["moment"],
        kls["moment"],
    )
    return _AnchorWork(
        evaluation,
        H4AnchorAllowanceSource(fixture_bytes, information_source, moment_source),
    )


def _preflight_scaled(
    config: H4ValidationConfig,
    *,
    problems: tuple[H4NeutralProblem, ...],
    global_conditions: dict[str, _ConditionAccumulator],
    coverage: dict[str, _CoverageAccumulator],
) -> tuple[_PreflightProblem, ...]:
    _assert_outside_timed_batch("scaled H4 preflight")
    values: list[_PreflightProblem] = []
    if len(problems) != 120:
        raise ValueError("scaled preflight requires the generated 120-problem tuple")
    problem_iterator = iter(problems)
    for horizon_index, horizon in enumerate(config.traversal.horizons):
        for seed_index, seed in enumerate(config.traversal.seeds):
            for kind_index, kind in enumerate(config.traversal.kinds):
                problem = next(problem_iterator)
                expected_index = ((horizon_index * 20 + seed_index) * 2 + kind_index)
                _scaled_indices(
                    problem.problem_id, expected_index, horizon_index, seed_index,
                    kind_index,
                )
                payload = canonical_h4_problem_bytes(problem)
                oracle = evaluate_h4_oracle(payload)
                compact = _compact_oracle(oracle, config)
                if not compact.posterior_condition.eligible or not compact.innovation_conditions.all_eligible:
                    raise ValueError(f"scaled condition envelope failed for {problem.problem_id}")
                global_conditions["oracle_posterior"].consume(compact.posterior_condition)
                coverage["oracle_posterior"].consume(_coverage_key(
                    problem_id=problem.problem_id,
                    problem_sha256=problem.canonical_sha256,
                ))
                for condition in _oracle_innovation_conditions(oracle, config):
                    global_conditions["oracle_innovation"].consume(condition)
                    coverage["oracle_innovation"].consume(_coverage_key(
                        problem_id=problem.problem_id,
                        problem_sha256=problem.canonical_sha256,
                        factor_id=condition.factor_id,
                    ))
                materialized = materialize_h4_problem(problem, config.solve_protocol)
                try:
                    check = _integrity_check(materialized, "after_materialization")
                except _H4IntegrityCheckFailure as failure:
                    record = _scaled_integrity_failure_record(
                        materialized, problem_index=expected_index,
                        horizon_index=horizon_index, seed_index=seed_index,
                        kind_index=kind_index, checkpoint="after_materialization",
                        completed_integrity_checks=(), failure=failure,
                        warmup_spans=(), timed_spans=(), garbage_collector=None,
                        postflight_schedule=None,
                    )
                    raise _H4ScaledCarrierFailure(record, str(failure)) from failure
                values.append(_PreflightProblem(
                    problem, payload, materialized, compact, [check],
                ))
                del oracle
    if len(values) != 120:
        raise RuntimeError("H4 scaled preflight did not close all 120 problems")
    return tuple(values)


def _generate_scaled_problems(config: H4ValidationConfig) -> tuple[H4NeutralProblem, ...]:
    _assert_outside_timed_batch("scaled H4 problem generation")
    values = tuple(
        make_h4_problem(seed=seed, kind=kind, horizon=horizon, d_z=4, d_m=4)
        for horizon in config.traversal.horizons
        for seed in config.traversal.seeds
        for kind in config.traversal.kinds
    )
    if len(values) != 120 or len({item.problem_id for item in values}) != 120:
        raise RuntimeError("scaled H4 generator did not produce the frozen traversal")
    return values


class _ObservedPostflight:
    __slots__ = (
        "tracker", "problem", "event_index", "materialized",
        "integrity_checks", "problem_index", "horizon_index", "seed_index",
        "kind_index", "warmup_spans", "timed_spans", "garbage_collector",
    )

    def __init__(
        self,
        tracker: _PostflightTracker,
        problem: H4NeutralProblem,
        *,
        materialized: H4MaterializedProblem,
        integrity_checks: list[H4MaterializedIntegrityCheck],
        problem_index: int,
        horizon_index: int,
        seed_index: int,
        kind_index: int,
        warmup_spans: tuple[H4ArmCallSpan, ...],
        timed_spans: tuple[H4ArmCallSpan, ...],
        garbage_collector: H4GarbageCollectorRecord,
    ) -> None:
        self.tracker = tracker
        self.problem = problem
        self.event_index = 0
        self.materialized = materialized
        self.integrity_checks = integrity_checks
        self.problem_index = problem_index
        self.horizon_index = horizon_index
        self.seed_index = seed_index
        self.kind_index = kind_index
        self.warmup_spans = warmup_spans
        self.timed_spans = timed_spans
        self.garbage_collector = garbage_collector

    def call(
        self,
        phase: str,
        action: Callable[[], object],
        *,
        repetition_index: int | None = None,
        arm: H4SolverArm | None = None,
        factor_id: str | None = None,
        selected_moment_name: str | None = None,
        equivalence_component: str | None = None,
        integrity_phase: str | None = None,
    ) -> object:
        event = H4PostflightEventKey(
            self.problem.problem_id, self.problem.canonical_sha256,
            self.event_index, phase, repetition_index, arm, factor_id,
            selected_moment_name, equivalence_component, integrity_phase,
        )  # type: ignore[arg-type]
        start = time.perf_counter_ns()
        try:
            value = action()
        except _H4IntegrityCheckFailure as failure:
            schedule = self.tracker.finish()
            if integrity_phase not in ("after_timed_batch", "after_postflight"):
                raise RuntimeError(
                    "postflight integrity failure lacked an exact checkpoint"
                ) from failure
            record = _scaled_integrity_failure_record(
                self.materialized, problem_index=self.problem_index,
                horizon_index=self.horizon_index, seed_index=self.seed_index,
                kind_index=self.kind_index, checkpoint=integrity_phase,
                completed_integrity_checks=tuple(self.integrity_checks),
                failure=failure, warmup_spans=self.warmup_spans,
                timed_spans=self.timed_spans,
                garbage_collector=self.garbage_collector,
                postflight_schedule=schedule,
            )
            raise _H4ScaledCarrierFailure(record, str(failure)) from failure
        except Exception as error:
            schedule = self.tracker.finish()
            stable_error = _stable_error(error)
            identity = _materialization_identity(
                self.materialized, tuple(self.integrity_checks),
            )
            record = H4ScaledIncompletePhaseRecord(
                self.materialized.problem_id, self.materialized.problem_sha256,
                self.problem_index, self.horizon_index, self.seed_index,
                self.kind_index, "postflight", identity, self.warmup_spans,
                self.timed_spans, self.garbage_collector, schedule, stable_error,
                _INCOMPLETE_OBLIGATIONS["postflight"],
            )
            raise _H4ScaledCarrierFailure(record, stable_error) from error
        end = time.perf_counter_ns()
        self.tracker.record(event, start, end)
        self.event_index += 1
        return value


def _postflight_schedule_snapshot(
    tracker: _PostflightTracker,
) -> H4PostflightScheduleSummary:
    mismatch_index = tracker.first_mismatch_index
    first_expected = tracker.first_expected
    first_observed = tracker.first_observed
    if mismatch_index is None and tracker.observed_count < tracker.expected_count:
        mismatch_index = tracker.observed_count
        first_expected = next(
            (
                event for index, event in enumerate(
                    _iter_expected_postflight_event_keys(tracker.problem)
                )
                if index == tracker.observed_count
            ),
            None,
        )
    observed_digest = tracker.observed_digest.copy().hexdigest()
    complete = (
        tracker.observed_count == tracker.expected_count
        and observed_digest == tracker.expected_digest
        and mismatch_index is None
        and tracker.timing_violations == 0
    )
    return H4PostflightScheduleSummary(
        "vfe4.h4.postflight-event-key-stream.v1", tracker.expected_count,
        tracker.observed_count, tracker.expected_digest, observed_digest,
        mismatch_index, first_expected, first_observed,
        tracker.timing_violations, tracker.first_timing, complete,
    )


def _materialization_identity_snapshot(
    observed: _ObservedPostflight,
) -> H4MaterializationIdentity:
    materialized = observed.materialized
    return H4MaterializationIdentity(
        materialized.problem_id, materialized.problem_sha256,
        materialized.materialization_version, materialized.protocol_id,
        materialized.tensor_sha256, 1, True,
        tuple(observed.integrity_checks),
    )


def _finalize_postflight_boundary(
    observed: _ObservedPostflight,
    finalize: Callable[[H4PostflightScheduleSummary], object],
) -> object:
    if not callable(finalize):
        raise ValueError("postflight finalizer must be callable")
    schedule = _postflight_schedule_snapshot(observed.tracker)
    identity = _materialization_identity_snapshot(observed)
    try:
        schedule = observed.tracker.finish()
        if not schedule.complete:
            raise RuntimeError("H4 postflight event schedule did not close exactly")
        return finalize(schedule)
    except _H4ScaledCarrierFailure:
        raise
    except Exception as error:
        stable_error = _stable_error(error)
        record = H4ScaledIncompletePhaseRecord(
            observed.materialized.problem_id,
            observed.materialized.problem_sha256,
            observed.problem_index, observed.horizon_index,
            observed.seed_index, observed.kind_index, "postflight", identity,
            observed.warmup_spans, observed.timed_spans,
            observed.garbage_collector, schedule, stable_error,
            _INCOMPLETE_OBLIGATIONS["postflight"],
        )
        raise _H4ScaledCarrierFailure(record, stable_error) from error


def _observe_native_diagnostic_replay(
    observed: _ObservedPostflight,
    *,
    materialized: H4MaterializedProblem,
    result: H4SolverResult,
    repetition_index: int,
    arm: H4SolverArm,
    replay_records: list[H4NativeReplayRecord],
    moment_diagnostics: dict[int, tuple[H4InnovationDiagnostic, ...]],
    coverage: object,
) -> H4NativeReplayRecord:
    consume_coverage = getattr(coverage, "consume", None)
    if not callable(consume_coverage):
        raise ValueError("native replay coverage accumulator must be callable")

    def replay_action() -> H4NativeReplayRecord:
        diagnostics = evaluate_h4_native_diagnostics(
            materialized, result,
            _facade(materialized.problem_id, arm, NullOperationRecorder()),
        )
        if type(diagnostics) is not H4NativeDiagnostics:
            raise RuntimeError("native diagnostic replay returned a wrong record type")
        reference_native = _native_stream(result)
        replayed_native = _native_stream(diagnostics.replayed_result)
        diagnostic_stream = _diagnostic_stream(diagnostics)
        record = H4NativeReplayRecord(
            materialized.problem_id, materialized.problem_sha256,
            repetition_index, arm, reference_native.sha256,
            replayed_native.sha256, diagnostic_stream,
            len(diagnostics.innovation_diagnostics), True,
        )
        consume_coverage(_coverage_key(
            problem_id=materialized.problem_id,
            problem_sha256=materialized.problem_sha256,
            repetition_index=repetition_index, arm=arm,
        ))
        replay_records.append(record)
        if arm == "moment":
            moment_diagnostics[repetition_index] = (
                diagnostics.innovation_diagnostics
            )
        return record

    record = observed.call(
        "native_diagnostic_replay", replay_action,
        repetition_index=repetition_index, arm=arm,
    )
    if type(record) is not H4NativeReplayRecord:
        raise RuntimeError("native diagnostic event returned a wrong record type")
    return record


def _observe_moment_innovation_conditions(
    observed: _ObservedPostflight,
    diagnostics: tuple[H4InnovationDiagnostic, ...],
    *,
    problem_id: str,
    problem_sha256: str,
    repetition_index: int,
    config: H4ValidationConfig,
    condition_accumulator: object | None = None,
    coverage: object | None = None,
) -> tuple[H4InnovationConditionRecord, ...]:
    if type(diagnostics) is not tuple or not all(
        type(item) is H4InnovationDiagnostic for item in diagnostics
    ):
        raise ValueError("moment innovation schedule requires exact diagnostics")
    consume_condition = (
        None if condition_accumulator is None
        else getattr(condition_accumulator, "consume", None)
    )
    consume_coverage = None if coverage is None else getattr(coverage, "consume", None)
    if condition_accumulator is not None and not callable(consume_condition):
        raise ValueError("moment condition accumulator must be callable")
    if coverage is not None and not callable(consume_coverage):
        raise ValueError("moment condition coverage accumulator must be callable")
    values: list[H4InnovationConditionRecord] = []
    for diagnostic in diagnostics:
        def condition_action(
            diagnostic=diagnostic,
        ) -> H4InnovationConditionRecord:
            condition = _moment_innovation_condition(
                diagnostic, problem_id=problem_id,
                problem_sha256=problem_sha256,
                repetition_index=repetition_index, config=config,
            )
            if consume_condition is not None:
                consume_condition(condition)
            if consume_coverage is not None:
                consume_coverage(_coverage_key(
                    problem_id=problem_id, problem_sha256=problem_sha256,
                    repetition_index=repetition_index,
                    factor_id=condition.factor_id,
                ))
            values.append(condition)
            return condition

        condition = observed.call(
            "moment_innovation_condition",
            condition_action,
            repetition_index=repetition_index, arm="moment",
            factor_id=diagnostic.factor_id,
        )
        if type(condition) is not H4InnovationConditionRecord:
            raise RuntimeError("moment innovation event returned a wrong record type")
    return tuple(values)


_AllowanceGroupAction = Callable[[], None]
_AllowanceGroupObserver = Callable[
    [str, str | None, _AllowanceGroupAction], None
]


@dataclass(frozen=True, slots=True)
class _DeferredAllowanceGroup:
    invariant: object
    expected_group: object
    observed_group: object
    component: str
    selected_name: str | None


def _allowance_event_metadata(
    invariant: str, expected_group: object,
) -> tuple[str, str | None]:
    path = getattr(expected_group, "path_prefix", None)
    if invariant == "exact_posterior_gap_equivalence" and path == "kl_to_zero":
        return "kl_to_zero", None
    if invariant == "terminal_h_equivalence" and path == "terminal_h":
        return "h", None
    if invariant == "terminal_J_equivalence" and path == "terminal_J":
        return "J", None
    if invariant == "complete_objective_equivalence" and path == "complete_objective":
        return "objective", None
    if invariant == "selected_moment_equivalence" and type(path) is str:
        match = re.fullmatch(
            r"selected_moments\.(.+)\.(mean|covariance)", path,
        )
        if match is not None:
            component = (
                "selected_mean" if match.group(2) == "mean"
                else "selected_covariance"
            )
            return component, match.group(1)
    raise ValueError("allowance group cannot be assigned to a declared H4 event")


class _H4ObservedAllowanceAccumulator(H4SixInvariantAllowanceAccumulator):
    __slots__ = ("_group_observer", "_deferred_groups")

    def __init__(self) -> None:
        super().__init__()
        self._group_observer: _AllowanceGroupObserver | None = None
        self._deferred_groups: list[_DeferredAllowanceGroup] | None = None

    def _consume_group_pairs(
        self,
        invariant,
        expected_groups,
        observed_groups,
    ) -> None:
        observer = self._group_observer
        deferred_groups = self._deferred_groups
        if deferred_groups is not None:
            if len(expected_groups) != len(observed_groups):
                raise ValueError(
                    "independent allowance producers disagree on group count"
                )
            for expected_group, observed_group in zip(
                expected_groups, observed_groups, strict=True,
            ):
                component, selected_name = _allowance_event_metadata(
                    invariant, expected_group,
                )
                deferred_groups.append(_DeferredAllowanceGroup(
                    invariant, expected_group, observed_group,
                    component, selected_name,
                ))
            return
        if observer is None:
            H4SixInvariantAllowanceAccumulator._consume_group_pairs(
                self, invariant, expected_groups, observed_groups,
            )
            return
        if len(expected_groups) != len(observed_groups):
            raise ValueError("independent allowance producers disagree on group count")
        for expected_group, observed_group in zip(
            expected_groups, observed_groups, strict=True,
        ):
            component, selected_name = _allowance_event_metadata(
                invariant, expected_group,
            )
            executed = False

            def consume_group(
                expected_group=expected_group,
                observed_group=observed_group,
            ) -> None:
                nonlocal executed
                if executed:
                    raise RuntimeError("allowance group event executed more than once")
                H4SixInvariantAllowanceAccumulator._consume_group_pairs(
                    self, invariant, (expected_group,), (observed_group,),
                )
                executed = True

            observer(component, selected_name, consume_group)
            if not executed:
                raise RuntimeError("allowance group event did not execute its consumption")

    def consume_observed(
        self,
        source: H4AllowanceResultSource,
        observer: _AllowanceGroupObserver,
    ) -> None:
        if self._group_observer is not None:
            raise RuntimeError("nested allowance observation is forbidden")
        if not callable(observer):
            raise ValueError("allowance group observer must be callable")
        self._group_observer = observer
        try:
            self.consume(source)
        finally:
            self._group_observer = None

    def defer_source(
        self,
        source_factory: Callable[[], object],
    ) -> tuple[_DeferredAllowanceGroup, ...]:
        if self._group_observer is not None or self._deferred_groups is not None:
            raise RuntimeError("nested allowance observation is forbidden")
        if not callable(source_factory):
            raise ValueError("allowance source factory must be callable")
        deferred_groups: list[_DeferredAllowanceGroup] = []
        self._deferred_groups = deferred_groups
        try:
            source = source_factory()
            self.consume(source)  # type: ignore[arg-type]
        except Exception:
            self._failed = True
            raise
        finally:
            self._deferred_groups = None
        if not deferred_groups:
            self._failed = True
            raise RuntimeError("allowance source produced no equivalence groups")
        return tuple(deferred_groups)

    def consume_deferred_group(self, group: _DeferredAllowanceGroup) -> None:
        if type(group) is not _DeferredAllowanceGroup:
            self._failed = True
            raise ValueError("deferred allowance group has a wrong record type")
        if self._failed:
            raise ValueError("six-invariant allowance accumulator failed closed")
        try:
            H4SixInvariantAllowanceAccumulator._consume_group_pairs(
                self, group.invariant, (group.expected_group,),
                (group.observed_group,),
            )
        except Exception:
            self._failed = True
            raise


def new_h4_six_invariant_allowance_accumulator(
) -> H4SixInvariantAllowanceAccumulator:
    return _H4ObservedAllowanceAccumulator()


def _consume_allowance_source_at_events(
    observed: _ObservedPostflight,
    allowance_accumulator: object,
    source: object,
    *,
    repetition_index: int,
    arm: H4SolverArm,
    before_first_group: Callable[[], None] | None = None,
    source_factory: Callable[[], object] | None = None,
) -> None:
    if before_first_group is not None and not callable(before_first_group):
        raise ValueError("first allowance-group preparation must be callable")
    if source_factory is not None:
        if source is not None:
            raise ValueError("deferred allowance source cannot also be preconstructed")
        if not callable(source_factory):
            raise ValueError("allowance source factory must be callable")
        defer_source = getattr(allowance_accumulator, "defer_source", None)
        if not callable(defer_source):
            raise ValueError("scaled allowance accumulator lacks deferred consumption")
        remaining_groups: list[_DeferredAllowanceGroup] = []

        def first_event_action() -> None:
            groups = defer_source(source_factory)
            if not groups:
                raise RuntimeError("allowance source produced no equivalence groups")
            first = groups[0]
            if (
                getattr(first, "component", None) != "kl_to_zero"
                or getattr(first, "selected_name", None) is not None
            ):
                raise RuntimeError("first allowance group is not KL equivalence")
            if before_first_group is not None:
                before_first_group()
            consume_deferred = getattr(
                allowance_accumulator, "consume_deferred_group", None,
            )
            if not callable(consume_deferred):
                raise ValueError(
                    "scaled allowance accumulator lacks deferred group consumption"
                )
            consume_deferred(first)
            remaining_groups.extend(groups[1:])

        observed.call(
            "equivalence_group", first_event_action,
            repetition_index=repetition_index, arm=arm,
            equivalence_component="kl_to_zero",
        )
        consume_deferred = getattr(
            allowance_accumulator, "consume_deferred_group", None,
        )
        if not callable(consume_deferred):
            raise ValueError(
                "scaled allowance accumulator lacks deferred group consumption"
            )
        for group in remaining_groups:
            observed.call(
                "equivalence_group",
                lambda group=group: consume_deferred(group),
                repetition_index=repetition_index, arm=arm,
                selected_moment_name=group.selected_name,
                equivalence_component=group.component,
            )
        return

    consume_observed = getattr(allowance_accumulator, "consume_observed", None)
    if not callable(consume_observed):
        raise ValueError("scaled allowance accumulator lacks observed consumption")
    first_group = True

    def observe_group(
        component: str,
        selected_name: str | None,
        action: _AllowanceGroupAction,
    ) -> None:
        nonlocal first_group
        prefix = before_first_group if first_group else None

        def event_action() -> None:
            if prefix is not None:
                prefix()
            action()

        observed.call(
            "equivalence_group", event_action,
            repetition_index=repetition_index,
            arm=arm, selected_moment_name=selected_name,
            equivalence_component=component,
        )
        first_group = False

    consume_observed(source, observe_group)


def _observe_stream_compaction(
    observed: _ObservedPostflight,
    release_action: Callable[[], int],
) -> int:
    if not callable(release_action):
        raise ValueError("stream release action must be callable")
    released = observed.call("stream_compaction", release_action)
    if type(released) is not int or released <= 0:
        raise RuntimeError("H4 stream compaction released no full numerical objects")
    return released


def _evaluate_scaled_problem(
    preflight: _PreflightProblem,
    *,
    config: H4ValidationConfig,
    horizon_index: int,
    seed_index: int,
    kind_index: int,
    allowance_accumulator: H4SixInvariantAllowanceAccumulator,
    global_conditions: dict[str, _ConditionAccumulator],
    coverage: dict[str, _CoverageAccumulator],
) -> tuple[H4ProblemEvaluation, tuple[H4TimingRecord, ...]]:
    materialized = preflight.materialized

    def before_timed() -> None:
        preflight.integrity_checks.append(
            _integrity_check(materialized, "before_timed_batch")
        )

    problem_index = ((horizon_index * 20 + seed_index) * 2 + kind_index)
    try:
        batch = _run_warmup_and_timed_batch(
            materialized, config.solve_protocol, horizon_index=horizon_index,
            seed_index=seed_index, kind_index=kind_index,
            before_timed_hook=before_timed,
        )
    except _H4WarmupBoundaryIntegrityFailure as boundary:
        record = _scaled_integrity_failure_record(
            materialized, problem_index=problem_index,
            horizon_index=horizon_index, seed_index=seed_index,
            kind_index=kind_index, checkpoint="before_timed_batch",
            completed_integrity_checks=tuple(preflight.integrity_checks),
            failure=boundary.failure, warmup_spans=boundary.warmup_spans,
            timed_spans=(), garbage_collector=None, postflight_schedule=None,
        )
        raise _H4ScaledCarrierFailure(record, str(boundary.failure)) from boundary
    warmup_spans = batch.warmup_spans
    timed_spans = batch.timed_spans
    timings = batch.timings
    timed_start = batch.timed_batch_start_nanoseconds
    timed_end = batch.timed_batch_end_nanoseconds
    garbage_collector = batch.garbage_collector
    results_by_identity: dict[tuple[int, H4SolverArm], H4SolverResult] = {}
    for span, result in zip(timed_spans, batch.results, strict=True):
        assert span.repetition_index is not None
        results_by_identity[(span.repetition_index, span.arm)] = result
    del batch

    tracker = _PostflightTracker(
        preflight.problem, timed_batch_end_nanoseconds=timed_end,
        coverage=coverage["postflight_schedule"],
    )
    observed = _ObservedPostflight(
        tracker, preflight.problem, materialized=materialized,
        integrity_checks=preflight.integrity_checks,
        problem_index=problem_index, horizon_index=horizon_index,
        seed_index=seed_index, kind_index=kind_index,
        warmup_spans=warmup_spans, timed_spans=timed_spans,
        garbage_collector=garbage_collector,
    )
    check_after_timed = observed.call(
        "materialized_integrity",
        lambda: _integrity_check(materialized, "after_timed_batch"),
        integrity_phase="after_timed_batch",
    )
    assert type(check_after_timed) is H4MaterializedIntegrityCheck
    preflight.integrity_checks.append(check_after_timed)

    terminals: dict[tuple[int, H4SolverArm], H4TerminalLaw] = {}
    for repetition in range(11):
        for arm in ("information", "moment"):
            result = results_by_identity[(repetition, arm)]
            terminal = observed.call(
                "terminal_conversion",
                lambda result=result, arm=arm: to_common_terminal_law(
                    materialized, result,
                    _facade(materialized.problem_id, arm, NullOperationRecorder()),
                ),
                repetition_index=repetition, arm=arm,
            )
            assert type(terminal) is H4TerminalLaw
            terminals[(repetition, arm)] = terminal

    replay_records: list[H4NativeReplayRecord] = []
    moment_innovation_diagnostics: dict[
        int, tuple[H4InnovationDiagnostic, ...]
    ] = {}
    for repetition in range(11):
        for arm in ("information", "moment"):
            result = results_by_identity[(repetition, arm)]
            _observe_native_diagnostic_replay(
                observed, materialized=materialized, result=result,
                repetition_index=repetition, arm=arm,
                replay_records=replay_records,
                moment_diagnostics=moment_innovation_diagnostics,
                coverage=coverage["native_replay"],
            )

    terminal_condition_records: list[H4PosteriorConditionRecord] = []
    for repetition in range(11):
        for arm in ("information", "moment"):
            def terminal_condition_action(
                repetition=repetition, arm=arm,
            ) -> H4PosteriorConditionRecord:
                condition = _terminal_posterior_condition(
                    terminals[(repetition, arm)],
                    problem_id=materialized.problem_id,
                    problem_sha256=materialized.problem_sha256,
                    repetition_index=repetition, config=config,
                )
                global_conditions["terminal_posterior"].consume(condition)
                coverage["terminal_posterior"].consume(_coverage_key(
                    problem_id=materialized.problem_id,
                    problem_sha256=materialized.problem_sha256,
                    repetition_index=repetition, arm=arm,
                ))
                terminal_condition_records.append(condition)
                return condition

            condition = observed.call(
                "terminal_posterior_condition",
                terminal_condition_action,
                repetition_index=repetition, arm=arm,
            )
            assert type(condition) is H4PosteriorConditionRecord
    flat_moment_conditions: list[H4InnovationConditionRecord] = []
    for repetition in range(11):
        conditions = _observe_moment_innovation_conditions(
            observed, moment_innovation_diagnostics[repetition],
            problem_id=materialized.problem_id,
            problem_sha256=materialized.problem_sha256,
            repetition_index=repetition, config=config,
            condition_accumulator=global_conditions["moment_innovation"],
            coverage=coverage["moment_innovation"],
        )
        for condition in conditions:
            flat_moment_conditions.append(condition)

    rehydrated = observed.call(
        "oracle_rehydration", lambda: evaluate_h4_oracle(preflight.payload),
    )
    assert type(rehydrated) is H4OracleEvaluation
    observed.call(
        "oracle_route_agreement",
        lambda: _assert_rehydrated_oracle(preflight.oracle, rehydrated),
    )

    compact_results: list[H4CompactResultRecord] = []
    for repetition in range(11):
        for arm in ("information", "moment"):
            result = results_by_identity[(repetition, arm)]
            terminal = terminals[(repetition, arm)]
            kl_holder: list[H4OracleKLEvaluation] = []

            def make_source(
                result=result, terminal=terminal, repetition=repetition,
            ) -> H4AllowanceResultSource:
                kl = reverse_kl_to_h4_oracle(
                    rehydrated, mean=terminal.mean,
                    precision=_terminal_precision(terminal),
                )
                source = H4AllowanceResultSource(
                    preflight.payload, repetition, rehydrated,
                    result, terminal, kl,
                )
                kl_holder.append(kl)
                return source

            def prepare_compact_result(
                result=result, terminal=terminal, repetition=repetition,
            ) -> None:
                if len(kl_holder) != 1:
                    raise RuntimeError(
                        "compact result requires exactly one observed KL evaluation"
                    )
                compact_results.append(_compact_result_from_kl(
                    result=result, terminal=terminal, oracle=rehydrated,
                    repetition_index=repetition, kl=kl_holder[0],
                ))

            _consume_allowance_source_at_events(
                observed, allowance_accumulator, None,
                repetition_index=repetition, arm=arm,
                before_first_group=prepare_compact_result,
                source_factory=make_source,
            )
            kl_holder.clear()

    counting: list[H4CountingPassRecord] = []
    for arm_index, arm in enumerate(("information", "moment")):
        def counting_action(
            arm=arm, arm_index=arm_index,
        ) -> H4CountingPassRecord:
            record = _counting_pass(
                materialized, config.solve_protocol, arm,
                compact_results[arm_index],
            )
            coverage["operation_pass"].consume(_coverage_key(
                problem_id=materialized.problem_id,
                problem_sha256=materialized.problem_sha256, arm=arm,
            ))
            counting.append(record)
            return record

        record = observed.call(
            "operation_pass", counting_action, arm=arm,
        )
        assert type(record) is H4CountingPassRecord
    memory: list[H4MemoryPassRecord] = []
    for arm_index, arm in enumerate(("information", "moment")):
        def memory_action(
            arm=arm, arm_index=arm_index,
        ) -> H4MemoryPassRecord:
            record = _memory_pass(
                materialized, config.solve_protocol, arm,
                compact_results[arm_index],
            )
            coverage["memory_pass"].consume(_coverage_key(
                problem_id=materialized.problem_id,
                problem_sha256=materialized.problem_sha256, arm=arm,
            ))
            memory.append(record)
            return record

        record = observed.call(
            "memory_pass", memory_action, arm=arm,
        )
        assert type(record) is H4MemoryPassRecord

    check_after_postflight = observed.call(
        "materialized_integrity",
        lambda: _integrity_check(materialized, "after_postflight"),
        integrity_phase="after_postflight",
    )
    assert type(check_after_postflight) is H4MaterializedIntegrityCheck
    preflight.integrity_checks.append(check_after_postflight)

    problem_condition_summaries: list[H4ProblemConditionSummary] = []
    condition_groups: tuple[
        tuple[str, Iterable[H4PosteriorConditionRecord | H4InnovationConditionRecord]], ...
    ] = (
        ("oracle_posterior", (preflight.oracle.posterior_condition,)),
        ("terminal_posterior", terminal_condition_records),
        ("oracle_innovation", _oracle_innovation_conditions(rehydrated, config)),
        ("moment_innovation", flat_moment_conditions),
    )
    for name, condition_records in condition_groups:
        expected = {
            "oracle_posterior": 1, "terminal_posterior": 22,
            "oracle_innovation": preflight.problem.horizon,
            "moment_innovation": 11 * preflight.problem.horizon,
        }[name]
        accumulator = _ConditionAccumulator(
            name, expected, problem_id=materialized.problem_id,
            problem_sha256=materialized.problem_sha256,
        )
        for condition in condition_records:
            accumulator.consume(condition)
        summary = accumulator.finish()
        assert type(summary) is H4ProblemConditionSummary
        problem_condition_summaries.append(summary)

    # The final scheduled event performs the actual release of retained full
    # numerical objects; it is not a synthetic schedule marker.
    retained_full_objects = [rehydrated, result, terminal]
    del rehydrated, result, terminal, make_source, prepare_compact_result
    del kl_holder

    def release_full_objects() -> int:
        released = (
            len(terminals) + len(results_by_identity)
            + len(moment_innovation_diagnostics)
            + len(retained_full_objects)
        )
        terminals.clear()
        results_by_identity.clear()
        moment_innovation_diagnostics.clear()
        retained_full_objects.clear()
        return released

    _observe_stream_compaction(observed, release_full_objects)

    def finalize_problem(
        schedule: H4PostflightScheduleSummary,
    ) -> H4ProblemEvaluation:
        trace = H4ExecutionTrace(
            materialized.problem_id,
            problem_index,
            horizon_index, seed_index, kind_index, warmup_spans, timed_start,
            timed_spans, timed_end, schedule, garbage_collector, False, (),
        )
        coverage["execution_trace"].consume(_coverage_key(
            problem_id=materialized.problem_id,
            problem_sha256=materialized.problem_sha256,
        ))
        identity = _materialization_identity(
            materialized, tuple(preflight.integrity_checks),
        )
        return H4ProblemEvaluation(
            materialized.problem_id, materialized.problem_sha256,
            problem_index,
            horizon_index, seed_index, kind_index, preflight.oracle,
            identity, trace, tuple(compact_results), tuple(replay_records),
            tuple(problem_condition_summaries), tuple(counting), tuple(memory),
        )

    evaluation = _finalize_postflight_boundary(observed, finalize_problem)
    assert type(evaluation) is H4ProblemEvaluation
    return evaluation, timings


def _assert_rehydrated_oracle(
    compact: H4CompactOracleRecord, oracle: H4OracleEvaluation,
) -> Literal[True]:
    stream = _oracle_stream(oracle)
    if (
        (oracle.problem_id, oracle.problem_sha256) != (compact.problem_id, compact.problem_sha256)
        or stream != compact.oracle_stream
        or oracle.route_agreement != compact.route_agreement
        or oracle.canonical_log_normalizer != compact.canonical_log_normalizer
        or oracle.predictive_log_normalizer != compact.predictive_log_normalizer
    ):
        raise ValueError("rehydrated H4 oracle differs from preflight")
    return True


@dataclass(frozen=True, slots=True)
class _H4CoreData:
    result: H4GateResult
    anchors: tuple[H4AnchorEvaluation | H4UnavailablePhaseRecord, H4AnchorEvaluation | H4UnavailablePhaseRecord]
    unavailable_phases: tuple[H4UnavailablePhaseRecord, ...]
    problems: tuple[H4ProblemEvaluation | H4ScaledIncompletePhaseRecord | H4ScaledMaterializedIntegrityFailureRecord, ...]
    allowances: tuple[H4AllowanceRecord, ...]
    coverage: tuple[H4CoverageRecord, ...]
    condition_summaries: tuple[H4ConditionStreamSummary, ...]
    raw_timings: tuple[H4TimingRecord, ...]
    primary_timed_order_balance: H4PrimaryTimedOrderBalance | None
    timing_summary: H4TimingSummary | None
    bootstrap_interval: H4BootstrapInterval | None
    interval_decision: H4IntervalDecision | None


def _all_unavailable_result(obligation: str) -> H4GateResult:
    sentinel = "not_evaluated_after_inconclusive_eligibility"
    measurements = {
        name: (0.80 if name == "primary_effect_threshold" else None)
        for name in H4_MEASUREMENT_NAMES
    }
    invariants = tuple(
        InvariantResult(name, False, None, None, sentinel)
        for name in H4_INVARIANT_NAMES
    )
    allowances = {
        name: H4InapplicableAllowance(False, sentinel)
        for name in H4_ALLOWANCE_INVARIANT_NAMES
    }
    obligations = tuple(
        dict.fromkeys((
            *(f"{name}: {sentinel}" for name in H4_INVARIANT_NAMES),
            obligation,
        ))
    )
    return H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, measurements, invariants, allowances,
        obligations,
    )


def _empty_core(
    obligation: str,
    *,
    anchors: tuple[
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
        H4AnchorEvaluation | H4UnavailablePhaseRecord,
    ] | None = None,
    problems: tuple[
        H4ProblemEvaluation | H4ScaledIncompletePhaseRecord
        | H4ScaledMaterializedIntegrityFailureRecord,
        ...,
    ] = (),
    raw_timings: tuple[H4TimingRecord, ...] = (),
) -> _H4CoreData:
    result = _all_unavailable_result(obligation)
    if anchors is None:
        anchors = (
            H4UnavailablePhaseRecord("anchor_coupled", obligation, obligation),
            H4UnavailablePhaseRecord("anchor_zero_control", obligation, obligation),
        )
    scaled_preflight_completed = bool(problems) and not (
        len(problems) == 1
        and type(problems[0]) is H4ScaledMaterializedIntegrityFailureRecord
        and problems[0].checkpoint == "after_materialization"
    )
    unavailable = (
        *(() if scaled_preflight_completed else (
            H4UnavailablePhaseRecord("scaled_preflight", obligation, obligation),
        )),
        H4UnavailablePhaseRecord("statistics", obligation, obligation),
    )
    return _H4CoreData(
        result, anchors, unavailable, problems,
        tuple(result.allowances_by_invariant[name] for name in H4_ALLOWANCE_INVARIANT_NAMES),
        (), (), raw_timings, None, None, None, None,
    )


def _early_anchor_failure_result(
    anchor: H4ApplicableAllowance,
    *,
    maximum_stopping_residual: float,
) -> H4GateResult:
    if anchor.invariant != "h3_anchor_identity" or anchor.passed:
        raise ValueError("early anchor failure requires a failed anchor allowance")
    measurements = {
        "primary_seed_ratio_geometric_mean": None,
        "primary_bootstrap_lower": None,
        "primary_bootstrap_upper": None,
        "primary_effect_threshold": 0.80,
        "primary_timed_ab_total": None,
        "primary_timed_ba_total": None,
        "maximum_solver_stopping_residual": maximum_stopping_residual,
        "maximum_allowance_scale_fraction": anchor.maximum_allowance_scale_ratio,
    }
    invariants = (
        InvariantResult(
            "h3_anchor_identity", False, anchor.maximum_normalized_residual,
            1.0, "anchor_comparison_miss",
        ),
        *(InvariantResult(
            name, False, None, None,
            "not_evaluated_after_decisive_h3_anchor_failure",
        ) for name in H4_INVARIANT_NAMES[1:]),
    )
    allowances = {
        name: (
            anchor if name == "h3_anchor_identity" else
            H4InapplicableAllowance(
                False, "not_evaluated_after_decisive_h3_anchor_failure",
            )
        )
        for name in H4_ALLOWANCE_INVARIANT_NAMES
    }
    return H4GateResult(
        "H4", GateStatus.FAIL, measurements, invariants, allowances, (),
    )


def _anchor_restoration_inconclusive_result(
    result: H4GateResult,
) -> H4GateResult:
    unavailable = "not_evaluated_after_inconclusive_eligibility"
    if (
        type(result) is not H4GateResult
        or result.status is not GateStatus.FAIL
        or result.invariants[0].passed
        or result.invariants[0].value is None
        or any(
            item.detail != "not_evaluated_after_decisive_h3_anchor_failure"
            for item in result.invariants[1:]
        )
        or type(result.allowances_by_invariant["h3_anchor_identity"]) is not H4ApplicableAllowance
    ):
        raise ValueError("anchor restoration rebuild requires the exact early FAIL")
    invariants = (
        result.invariants[0],
        *(InvariantResult(name, False, None, None, unavailable)
          for name in H4_INVARIANT_NAMES[1:]),
    )
    allowances = {
        name: (
            result.allowances_by_invariant[name]
            if name == "h3_anchor_identity"
            else H4InapplicableAllowance(False, unavailable)
        )
        for name in H4_ALLOWANCE_INVARIANT_NAMES
    }
    return H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, result.measurements, invariants,
        allowances,
        ("restore H4 process-global state before closing anchor result",),
    )


_THREAD_STATE_OBLIGATION = (
    "restore H4 process-global thread state before closing the gate"
)


def _thread_state_inconclusive_result(result: H4GateResult) -> H4GateResult:
    if type(result) is not H4GateResult:
        raise ValueError("thread-state rebuild requires an exact H4 result")
    invariants = list(result.invariants)
    index = H4_INVARIANT_NAMES.index("cpu_float64_one_thread")
    invariants[index] = InvariantResult(
        "cpu_float64_one_thread", False, 0.0, 1.0,
        "process_global_thread_state_not_restored",
    )
    obligations = tuple(dict.fromkeys((*result.obligations, _THREAD_STATE_OBLIGATION)))
    return H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, result.measurements, tuple(invariants),
        result.allowances_by_invariant, obligations,
    )


def _complete_gate_result(
    *,
    allowances: tuple[H4ApplicableAllowance, ...],
    conditions: tuple[H4ConditionStreamSummary, ...],
    coverage: tuple[H4CoverageRecord, ...],
    timing_summary: H4TimingSummary,
    balance: H4PrimaryTimedOrderBalance,
    bootstrap: H4BootstrapInterval,
    interval_decision: H4IntervalDecision,
    maximum_stopping_residual: float,
    environment_complete: bool,
) -> H4GateResult:
    if len(allowances) != 6:
        raise ValueError("complete gate requires all six allowances")
    maximum_ratio = max(item.maximum_allowance_scale_ratio for item in allowances)
    measurements = {
        "primary_seed_ratio_geometric_mean": timing_summary.geometric_mean_ratio,
        "primary_bootstrap_lower": bootstrap.lower,
        "primary_bootstrap_upper": bootstrap.upper,
        "primary_effect_threshold": 0.80,
        "primary_timed_ab_total": float(balance.observed_ab_total),
        "primary_timed_ba_total": float(balance.observed_ba_total),
        "maximum_solver_stopping_residual": maximum_stopping_residual,
        "maximum_allowance_scale_fraction": maximum_ratio,
    }
    invariants: list[InvariantResult] = [
        InvariantResult(
            "h3_anchor_identity", allowances[0].passed,
            allowances[0].maximum_normalized_residual, 1.0,
            "complete_element_local_anchor_comparison",
        ),
        InvariantResult("fixed_seed_problem_identity", True, 120.0, 120.0, "complete_canonical_traversal"),
        InvariantResult("coupled_zero_control_contract", True, 60.0, 60.0, "matched_problem_pairs"),
        InvariantResult("cpu_float64_one_thread", environment_complete, 1.0 if environment_complete else 0.0, 1.0, "captured_environment_and_thread_state"),
        InvariantResult("shared_protocol_identity", maximum_stopping_residual <= 1.0e-9, maximum_stopping_residual, 1.0e-9, "single_pass_solver_residual"),
        InvariantResult("scaled_condition_envelope", all(item.all_eligible for item in conditions), float(sum(item.eligible_record_count for item in conditions)), float(sum(item.observed_record_count for item in conditions)), "full_condition_stream"),
        InvariantResult("complete_repetition_table", True, 1320.0, 1320.0, "all_timed_rows_present"),
        InvariantResult("primary_timed_order_balance", balance.matches, float(balance.observed_ab_total + balance.observed_ba_total), 220.0, "preregistered_primary_balance"),
    ]
    for invariant, allowance in zip(H4_ALLOWANCE_INVARIANT_NAMES[1:], allowances[1:], strict=True):
        invariants.append(InvariantResult(
            invariant, allowance.passed, allowance.maximum_normalized_residual,
            1.0, "complete_element_local_comparison",
        ))
    invariants.extend((
        InvariantResult(
            "all_equivalence_allowances_decisive",
            all(item.decisive for item in allowances), maximum_ratio, 1.0e-4,
            "strict_element_local_allowance_scale_cap",
        ),
        InvariantResult(
            "real_operation_instrumentation",
            all(item.complete for item in coverage),
            float(sum(item.observed_key_count for item in coverage)),
            float(sum(item.expected_key_count for item in coverage)),
            "complete_operation_memory_replay_and_schedule_coverage",
        ),
        InvariantResult(
            "primary_seed_level_inference", True, 20.0, 20.0,
            "paired_seed_level_bootstrap",
        ),
        InvariantResult(
            "primary_effect_threshold", interval_decision.invariant_passed,
            interval_decision.invariant_value, interval_decision.invariant_limit,
            interval_decision.invariant_detail,
        ),
    ))
    obligations: tuple[str, ...] = ()
    precondition_eligible = all(item.passed for item in invariants[:8]) and all(
        item.passed for item in invariants[13:16]
    )
    has_comparison_miss = any(not item.passed for item in invariants[8:13])
    if not precondition_eligible:
        status = GateStatus.INCONCLUSIVE
        obligations = tuple(
            f"{item.name}: resolve incomplete H4 eligibility evidence"
            for item in invariants[:16] if not item.passed
        )
    elif has_comparison_miss:
        status = GateStatus.FAIL
    else:
        status = interval_decision.status_if_other_invariants_eligible
        if interval_decision.obligation is not None:
            obligations = (f"primary_effect_threshold: {interval_decision.invariant_detail}",)
    allowance_map = {
        name: item for name, item in zip(H4_ALLOWANCE_INVARIANT_NAMES, allowances, strict=True)
    }
    return H4GateResult(
        "H4", status, measurements, tuple(invariants), allowance_map, obligations,
    )


def _unavailable_anchor(
    phase: Literal["anchor_coupled", "anchor_zero_control"],
    error: Exception,
) -> H4UnavailablePhaseRecord:
    label = "coupled" if phase == "anchor_coupled" else "zero-control"
    return H4UnavailablePhaseRecord(
        phase, _stable_error(error),
        f"complete {label} H3 anchor evaluation before scaled H4 work",
    )


def _evaluate_complete_core(
    config: H4ValidationConfig,
    *,
    h3_coupled_bytes: bytes,
    h3_zero_bytes: bytes,
    environment_complete: bool,
) -> _H4CoreData:
    _assert_outside_timed_batch("complete H4 gate evaluation")
    coupled: _AnchorWork | None = None
    zero: _AnchorWork | None = None
    coupled_slot: H4AnchorEvaluation | H4UnavailablePhaseRecord
    zero_slot: H4AnchorEvaluation | H4UnavailablePhaseRecord
    try:
        coupled = _evaluate_anchor(
            h3_coupled_bytes, expected_fixture_id="h3-coupled-v1", config=config,
        )
        coupled_slot = coupled.evaluation
    except Exception as error:
        coupled_slot = _unavailable_anchor("anchor_coupled", error)
    try:
        zero = _evaluate_anchor(
            h3_zero_bytes, expected_fixture_id="h3-zero-control-v1", config=config,
        )
        zero_slot = zero.evaluation
    except Exception as error:
        zero_slot = _unavailable_anchor("anchor_zero_control", error)
    anchors = (coupled_slot, zero_slot)
    if coupled is None or zero is None:
        return _empty_core(
            "complete both H3 anchor slots before scaled H4 work",
            anchors=anchors,
        )

    allowance_accumulator = new_h4_six_invariant_allowance_accumulator()
    try:
        allowance_accumulator.consume(coupled.allowance_source)
    except Exception as error:
        anchors = (_unavailable_anchor("anchor_coupled", error), zero.evaluation)
        return _empty_core(
            "complete both H3 anchor slots before scaled H4 work",
            anchors=anchors,
        )
    try:
        allowance_accumulator.consume(zero.allowance_source)
    except Exception as error:
        anchors = (coupled.evaluation, _unavailable_anchor("anchor_zero_control", error))
        return _empty_core(
            "complete both H3 anchor slots before scaled H4 work",
            anchors=anchors,
        )
    anchors: tuple[H4AnchorEvaluation, H4AnchorEvaluation] = (
        coupled.evaluation, zero.evaluation,
    )
    anchor_allowance = allowance_accumulator.anchor_identity_record()
    if type(anchor_allowance) is not H4ApplicableAllowance:
        raise RuntimeError("H4 anchor snapshot did not return an applicable allowance")
    maximum_anchor_residual = max(
        coupled.evaluation.information_result.stopping_residual,
        coupled.evaluation.moment_result.stopping_residual,
        zero.evaluation.information_result.stopping_residual,
        zero.evaluation.moment_result.stopping_residual,
    )
    if not anchor_allowance.passed:
        if not anchor_allowance.decisive:
            return _empty_core(
                "h3_anchor_identity: resolve indecisive anchor allowance",
                anchors=anchors,
            )
        result = _early_anchor_failure_result(
            anchor_allowance,
            maximum_stopping_residual=maximum_anchor_residual,
        )
        unavailable = (
            H4UnavailablePhaseRecord(
                "scaled_preflight", "decisive H3 anchor identity miss",
                "repair H3 anchor identity before scaled H4 timing",
            ),
            H4UnavailablePhaseRecord(
                "statistics", "decisive H3 anchor identity miss",
                "repair H3 anchor identity before H4 statistics",
            ),
        )
        allowances = tuple(
            result.allowances_by_invariant[name] for name in H4_ALLOWANCE_INVARIANT_NAMES
        )
        return _H4CoreData(
            result, anchors, unavailable, (), allowances, (), (), (),
            None, None, None, None,
        )

    neutral_problems = _generate_scaled_problems(config)
    global_conditions = {
        name: _ConditionAccumulator(name, expected)
        for name, expected in zip(
            _CONDITION_NAMES, (120, 2640, 2120, 23320), strict=True,
        )
    }
    coverage = {
        name: _CoverageAccumulator(name, neutral_problems)
        for name in _COVERAGE_NAMES
    }
    try:
        preflight = _preflight_scaled(
            config, problems=neutral_problems,
            global_conditions=global_conditions, coverage=coverage,
        )
    except _H4ScaledCarrierFailure as failure:
        return _empty_core(
            f"materialized_integrity: {failure.stable_error}", anchors=anchors,
            problems=(failure.record,),
        )
    except Exception as error:
        return _empty_core(
            f"scaled_preflight: {_stable_error(error)}", anchors=anchors,
        )
    completed: list[H4ProblemEvaluation] = []
    timings: list[H4TimingRecord] = []
    for problem_index, item in enumerate(preflight):
        horizon_index, within_horizon = divmod(problem_index, 40)
        seed_index, kind_index = divmod(within_horizon, 2)
        try:
            evaluation, problem_timings = _evaluate_scaled_problem(
                item, config=config, horizon_index=horizon_index,
                seed_index=seed_index, kind_index=kind_index,
                allowance_accumulator=allowance_accumulator,
                global_conditions=global_conditions, coverage=coverage,
            )
        except _H4BatchFailure as failure:
            identity = _materialization_identity(
                item.materialized, tuple(item.integrity_checks),
            )
            incomplete = H4ScaledIncompletePhaseRecord(
                item.problem.problem_id, item.problem.canonical_sha256,
                problem_index, horizon_index, seed_index, kind_index,
                failure.phase, identity, failure.warmup_spans,
                failure.timed_spans, failure.garbage_collector, None,
                failure.stable_error, _INCOMPLETE_OBLIGATIONS[failure.phase],
            )  # type: ignore[arg-type]
            return _empty_core(
                f"{failure.phase}: {failure.stable_error}", anchors=anchors,
                problems=(*completed, incomplete),
                raw_timings=tuple(timings),
            )
        except _H4ScaledCarrierFailure as failure:
            phase = (
                "materialized_integrity"
                if type(failure.record) is H4ScaledMaterializedIntegrityFailureRecord
                else failure.record.phase
            )
            return _empty_core(
                f"{phase}: {failure.stable_error}", anchors=anchors,
                problems=(*completed, failure.record),
                raw_timings=tuple(timings),
            )
        completed.append(evaluation)
        timings.extend(problem_timings)
    if len(completed) != 120 or len(timings) != 1320:
        raise RuntimeError("H4 complete traversal did not close all problems/timings")
    allowance_records = allowance_accumulator.finalize()
    if allowance_records[0] is not anchor_allowance:
        raise RuntimeError("H4 final allowance changed the cached anchor record identity")
    conditions = tuple(
        global_conditions[name].finish() for name in _CONDITION_NAMES
    )
    if not all(type(item) is H4ConditionStreamSummary for item in conditions):
        raise RuntimeError("global condition accumulators returned wrong record types")
    coverage_records = tuple(coverage[name].finish() for name in _COVERAGE_NAMES)
    if not all(item.complete for item in coverage_records):
        raise RuntimeError("H4 global coverage did not close")
    raw_timings = tuple(timings)
    primary = tuple(
        item for item in raw_timings
        if item.horizon_index == 2 and item.kind_index == 0
    )
    timing_summary = summarize_seed_ratios(primary)
    traces = tuple(item.execution_trace for item in completed)
    balance = summarize_primary_timed_order(primary, traces)
    bootstrap = paired_log_bootstrap_interval(timing_summary)
    decision = decide_h4_interval(bootstrap)
    maximum_stopping_residual = max(
        item.stopping_residual
        for problem in completed for item in problem.retained_results
    )
    result = _complete_gate_result(
        allowances=allowance_records,
        conditions=conditions,  # type: ignore[arg-type]
        coverage=coverage_records,
        timing_summary=timing_summary, balance=balance, bootstrap=bootstrap,
        interval_decision=decision,
        maximum_stopping_residual=maximum_stopping_residual,
        environment_complete=environment_complete,
    )
    return _H4CoreData(
        result, anchors, (), tuple(completed), allowance_records,
        coverage_records, conditions, raw_timings, balance, timing_summary,
        bootstrap, decision,
    )  # type: ignore[arg-type]


_BOUNDED_CLAIM = (
    "On the frozen CPU-float64 protocol, the H4 artifact compares complete "
    "information-form and moment-form Gaussian inference using element-local "
    "allowances and paired seed-level timing evidence."
)
_NONCLAIMS = (
    "H4 does not establish H5 generalized-EM update coherence.",
    "H4 does not establish H6 prefix prediction.",
    "H4 does not establish H7 frame covariance.",
    "H4 does not establish H8 sparse scaling.",
    "H4 does not establish language-model training quality or WikiText-103 performance.",
)

_PAYLOAD_SIZE_OBLIGATION = (
    "reduce H4 validation payload below 67108864 bytes without dropping scalar coverage"
)


def _payload_size_inconclusive_result(
    result: H4GateResult, *, observed_bytes: int,
) -> H4GateResult:
    if type(result) is not H4GateResult or type(observed_bytes) is not int:
        raise ValueError("payload-size status rebuild requires exact evidence")
    invariants = list(result.invariants)
    invariants[H4_INVARIANT_NAMES.index("real_operation_instrumentation")] = InvariantResult(
        "real_operation_instrumentation", False, float(observed_bytes),
        float(_MAX_PAYLOAD_BYTES), "validation_payload_exceeds_limit",
    )
    obligations = tuple(dict.fromkeys((*result.obligations, _PAYLOAD_SIZE_OBLIGATION)))
    return H4GateResult(
        "H4", GateStatus.INCONCLUSIVE, result.measurements, tuple(invariants),
        result.allowances_by_invariant, obligations,
    )


def _assemble_evaluation(
    core: _H4CoreData,
    *,
    config: H4ValidationConfig,
    thread_state: H4ThreadStateRecord,
    environment: H4EnvironmentRecord,
) -> H4GateEvaluation:
    active_core = core

    def make(size: H4PayloadSizeRecord) -> H4GateEvaluation:
        return H4GateEvaluation(
            "h4-gate-evaluation-v1", "bounded-stream-summaries-v1",
            _MAX_PAYLOAD_BYTES, active_core.result, config.config_sha256, active_core.anchors,
            active_core.unavailable_phases, active_core.problems, active_core.allowances,
            active_core.coverage, active_core.condition_summaries, active_core.raw_timings,
            active_core.primary_timed_order_balance, active_core.timing_summary,
            active_core.bootstrap_interval, active_core.interval_decision, thread_state,
            environment, size, _BOUNDED_CLAIM, _NONCLAIMS,
        )  # type: ignore[arg-type]

    def builder(size: H4PayloadSizeRecord) -> dict[str, object]:
        return _artifact_payload(h4_validation_artifact(make(size)))

    size, _ = _solve_payload_size_fixed_point(
        builder, maximum_bytes=config.maximum_validation_payload_bytes,
    )
    if not size.within_limit:
        active_core = replace(
            active_core,
            result=_payload_size_inconclusive_result(
                active_core.result, observed_bytes=size.observed_bytes,
            ),
        )
        size, _ = _solve_payload_size_fixed_point(
            builder, maximum_bytes=config.maximum_validation_payload_bytes,
        )
    return make(size)


def evaluate_h4(
    config: H4ValidationConfig,
    *,
    h3_coupled_bytes: bytes,
    h3_zero_bytes: bytes,
) -> H4GateEvaluation:
    """Run the frozen H4 protocol and return its compact typed evidence."""
    _assert_outside_timed_batch("H4 gate entry")
    if type(config) is not H4ValidationConfig:
        raise ValueError("evaluate_h4 requires the exact standalone H4 config")
    # Re-run the exact public constructor so mutated/forged nested records cannot
    # authorize any thread mutation or timing work.
    H4ValidationConfig(**{
        field.name: getattr(config, field.name) for field in fields(H4ValidationConfig)
    })
    if type(h3_coupled_bytes) is not bytes or type(h3_zero_bytes) is not bytes:
        raise ValueError("H4 anchor inputs must be immutable bytes")
    environment = _capture_environment()

    def work() -> _H4CoreData:
        if not environment.mandatory_facts_complete:
            return _empty_core(
                "capture complete mandatory H4 environment facts before timing",
            )
        return _evaluate_complete_core(
            config, h3_coupled_bytes=h3_coupled_bytes,
            h3_zero_bytes=h3_zero_bytes,
            environment_complete=environment.mandatory_facts_complete,
        )

    guarded = _run_thread_guard(work)
    core = guarded.value if type(guarded.value) is _H4CoreData else None
    if core is None:
        reason = guarded.work_error or guarded.state.capture_error or guarded.state.set_error or "H4 thread guard did not execute"
        core = _empty_core(reason)
    if (
        not guarded.state.verified_one
        or not guarded.state.inter_op_unchanged
        or not guarded.state.restored_exact_prior_state
    ):
        early_anchor_fail = (
            core.result.status is GateStatus.FAIL
            and all(
                item.detail == "not_evaluated_after_decisive_h3_anchor_failure"
                for item in core.result.invariants[1:]
            )
        )
        if early_anchor_fail:
            restored_result = _anchor_restoration_inconclusive_result(core.result)
            core = replace(
                core, result=restored_result,
                allowances=tuple(
                    restored_result.allowances_by_invariant[name]
                    for name in H4_ALLOWANCE_INVARIANT_NAMES
                ),
            )
        else:
            core = replace(
                core, result=_thread_state_inconclusive_result(core.result),
            )
    return _assemble_evaluation(
        core, config=config, thread_state=guarded.state, environment=environment,
    )


__all__ = [
    name for name in globals()
    if name.startswith("H4") or name in (
        "evaluate_h4", "h4_validation_artifact", "h4_validation_payload",
    )
]
