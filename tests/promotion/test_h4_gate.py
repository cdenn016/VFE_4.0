from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import pytest
import torch

import vfe4.artifacts as artifacts
import verification.h4_gate as gate
import verification.numpy_oracles.h4_gaussian as h4_gaussian
from verification.h4_gate import (
    H4AnchorEvaluation,
    H4CanonicalStreamDigest,
    H4CompactKLSummary,
    H4CompactOracleRecord,
    H4CompactResultRecord,
    H4CountingPassRecord,
    H4EnvironmentRecord,
    H4GateEvaluation,
    H4MaterializationIdentity,
    H4MaterializedIntegrityCheck,
    H4MemoryPassRecord,
    H4NativeReplayRecord,
    H4PayloadSizeRecord,
    H4PowerPolicyField,
    H4ProblemEvaluation,
    H4ScaledIncompletePhaseRecord,
    H4ScaledMaterializedIntegrityFailureRecord,
    H4SelectedMomentSummary,
    H4UnavailablePhaseRecord,
    H4ValidationArtifact,
    h4_validation_artifact,
    h4_validation_payload,
)
from verification.h4_records import H4GarbageCollectorRecord
from verification.h4_records import H4ThreadStateRecord
from vfe4.config import resolve_h4_validation_config
from vfe4.generative.reference_h4 import make_h4_problem
from vfe4.inference.h4_solvers import H4InnovationDiagnostic, materialize_h4_problem
from vfe4.types.h4 import (
    H4_ALLOWANCE_ELEMENT_COUNTS,
    H4_PRIMARY_TIMED_BALANCE,
    H4_PROBLEM_SEEDS,
    H4AllowanceElement,
    H4AllowanceOperand,
    H4SolveProtocol,
    H4TimingRecord,
    canonical_h4_problem_bytes,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


RECORD_FIELDS = {
    H4MaterializedIntegrityCheck: (
        "phase", "expected_tensor_sha256", "observed_tensor_sha256", "exact_match",
    ),
    H4MaterializationIdentity: (
        "problem_id", "problem_sha256", "materialization_version", "protocol_id",
        "tensor_sha256", "materialization_count", "shared_by_identity", "integrity_checks",
    ),
    H4CanonicalStreamDigest: ("domain", "record_count", "scalar_count", "byte_count", "sha256"),
    H4SelectedMomentSummary: (
        "name", "coordinate_indices", "dimension", "mean_scalar_count", "mean_sha256",
        "mean_infinity_norm", "covariance_scalar_count", "covariance_sha256",
        "covariance_trace", "covariance_maximum_absolute_value",
    ),
    H4CompactKLSummary: (
        "value", "trace_term", "quadratic_mean_term", "minus_dimension_term",
        "candidate_logdet_precision_term", "minus_oracle_logdet_precision_term",
        "absolute_summand_accumulation", "candidate_condition_number",
        "oracle_condition_number", "operation_counts",
    ),
    H4CompactResultRecord: (
        "problem_id", "problem_sha256", "source_kind", "repetition_index", "arm",
        "native_stream", "terminal_stream", "oracle_kl_q_to_p",
        "native_complete_objective", "terminal_complete_objective", "stopping_residual",
        "selected_moments",
    ),
    H4CompactOracleRecord: (
        "problem_id", "problem_sha256", "source_kind", "dimension", "oracle_stream",
        "canonical_log_normalizer", "predictive_log_normalizer", "route_agreement",
        "selected_moments", "posterior_condition", "innovation_conditions",
    ),
    H4NativeReplayRecord: (
        "problem_id", "problem_sha256", "repetition_index", "arm",
        "reference_native_sha256", "replayed_native_sha256", "diagnostic_stream",
        "innovation_record_count", "exact_result_match",
    ),
    H4CountingPassRecord: (
        "problem_id", "problem_sha256", "arm", "reference_repetition_index",
        "reference_native_sha256", "replayed_native_sha256", "reference_terminal_sha256",
        "replayed_terminal_sha256", "exact_result_match", "solver_operations",
        "terminal_conversion_operations",
    ),
    H4MemoryPassRecord: (
        "problem_id", "problem_sha256", "arm", "reference_repetition_index",
        "reference_native_sha256", "replayed_native_sha256", "exact_result_match", "memory",
    ),
    H4ProblemEvaluation: (
        "problem_id", "problem_sha256", "problem_index", "horizon_index", "seed_index",
        "kind_index", "oracle", "materialization", "execution_trace", "retained_results",
        "native_replays", "condition_summaries", "counting_passes", "memory_passes",
    ),
    H4ScaledIncompletePhaseRecord: (
        "problem_id", "problem_sha256", "problem_index", "horizon_index", "seed_index",
        "kind_index", "phase", "materialization", "warmup_spans", "partial_timed_spans",
        "garbage_collector", "postflight_schedule", "stable_error", "obligation",
    ),
    H4ScaledMaterializedIntegrityFailureRecord: (
        "problem_id", "problem_sha256", "problem_index", "horizon_index", "seed_index",
        "kind_index", "materialization_version", "protocol_id", "materialization_count",
        "shared_by_identity", "checkpoint", "expected_tensor_sha256",
        "completed_integrity_checks", "failure_kind", "observed_tensor_sha256", "seam_error",
        "warmup_spans", "timed_spans", "garbage_collector", "postflight_schedule", "obligation",
    ),
    H4AnchorEvaluation: (
        "problem_id", "problem_sha256", "oracle", "materialization", "information_result",
        "information_diagnostic_stream", "moment_result", "moment_diagnostic_stream",
    ),
    H4UnavailablePhaseRecord: ("phase", "reason", "obligation"),
    H4PowerPolicyField: ("name", "availability", "source", "value", "unavailable_reason"),
    H4EnvironmentRecord: (
        "clock_implementation", "clock_resolution_seconds", "clock_monotonic", "processor",
        "platform", "platform_system", "affinity_cpu_ids", "logical_cpu_count",
        "physical_cpu_count", "torch_version", "numpy_version", "torch_config_text",
        "torch_config_sha256", "numpy_blas_text", "numpy_blas_sha256", "cuda_available",
        "environment_variables", "power_policy_fields", "power_policy_category_complete",
        "unavailable_fields", "mandatory_facts_complete",
    ),
    H4PayloadSizeRecord: (
        "encoding", "observed_bytes", "maximum_bytes", "fixed_point_iterations", "within_limit",
    ),
    H4GateEvaluation: (
        "schema_version", "payload_representation", "maximum_payload_bytes", "result",
        "h4_config_sha256", "anchors", "unavailable_phases", "problems", "allowances",
        "coverage", "condition_summaries", "raw_timings", "primary_timed_order_balance",
        "timing_summary", "bootstrap_interval", "interval_decision", "thread_state",
        "environment", "payload_size", "bounded_claim", "nonclaims",
    ),
    H4ValidationArtifact: (
        "schema_version", "payload_representation", "maximum_payload_bytes", "gate", "status",
        "h4_config_sha256", "result", "anchors", "unavailable_phases", "problems", "allowances",
        "coverage", "condition_summaries", "raw_timings", "primary_timed_order_balance",
        "timing_summary", "bootstrap_interval", "interval_decision", "thread_state",
        "environment", "payload_size", "bounded_claim", "nonclaims",
    ),
}


def _protocol() -> H4SolveProtocol:
    return H4SolveProtocol()


@pytest.fixture(scope="module")
def h4_config():
    return resolve_h4_validation_config({
        "schema_version": "h4-validation-config-v1",
        "solve_protocol": {
            "protocol_id": "h4-single-pass-v1", "dtype": "float64",
            "device": "cpu", "factor_passes": 1,
            "solver_relative_budget": 1.0e-9,
            "stopping_rule": "complete_schedule_finite_spd",
        },
        "traversal": {
            "horizons": [7, 15, 31], "seeds": list(H4_PROBLEM_SEEDS),
            "kinds": ["coupled", "zero_control"], "d_z": 4, "d_m": 4,
            "dimensions": [64, 128, 256], "primary_horizon": 31,
            "primary_kind": "coupled", "primary_dimension": 256,
        },
        "timing": {
            "parity_expression": "(horizon_index + seed_index + kind_index + pair_index) % 2 == 0",
            "warmup_pair_indices": [0, 1, 2],
            "timed_pair_indices": list(range(3, 14)),
            "timed_repetitions_per_problem": 11,
            "warmups_count_toward_balance": False,
            "primary_timed_balance": [list(row) for row in H4_PRIMARY_TIMED_BALANCE],
            "primary_5_ab_6_ba_rows": 10, "primary_6_ab_5_ba_rows": 10,
            "primary_timed_ab_total": 110, "primary_timed_ba_total": 110,
            "clock": "time.perf_counter_ns",
            "timer_boundary": "fresh_native_solver_call_v1",
            "between_repetitions": "timer_reads_and_preallocated_assignments_only",
        },
        "bootstrap": {
            "seed": 20260721, "replicates": 100000, "inferential_units": 20,
            "index_low": 0, "index_high": 20, "endpoint": False,
            "index_dtype": "<i8", "index_shape": [100000, 20],
            "statistic": "mean_log_seed_ratio", "percentiles": [2.5, 97.5],
            "percentile_method": "linear", "percentile_space": "log_then_exp",
            "digest_domain": "vfe4.h4.bootstrap-indices.v1",
            "expected_index_sha256": "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14",
        },
        "condition_envelope": {
            "posterior_minimum_eigenvalue": 1.0e-6,
            "posterior_maximum_eigenvalue": 1.0e6,
            "posterior_maximum_condition_number": 1.0e8,
            "posterior_minimum_cholesky_pivot": 1.0e-3,
            "posterior_maximum_mean_infinity_norm": 16.0,
            "innovation_minimum_eigenvalue": 1.0e-6,
            "innovation_maximum_eigenvalue": 1.0e6,
            "innovation_maximum_condition_number": 1.0e8,
            "inclusive": True,
        },
        "allowance": {
            "float64_epsilon": 2.220446049250313e-16,
            "rounding_constant": 4096, "solver_relative_budget": 1.0e-9,
            "maximum_allowance_scale_fraction": 1.0e-4,
            "decisiveness_comparison": "strict_less_than",
            "element_stream_domain": "vfe4.h4.allowance-element-stream.v1",
            "maximum_chunk_rows": 4096,
        },
        "environment": {
            "device": "cpu", "dtype": "float64", "intra_op_threads": 1,
            "alter_inter_op_threads": False, "cuda_expected": False,
            "gc_policy": "restore_exact_prior_enabled_state",
            "power_policy_field_order": [
                "active_power_scheme", "cpu_frequency_governor",
                "energy_performance_preference", "low_power_mode",
            ],
            "power_policy_capture": "typed_best_effort_outside_timing",
        },
        "primary_effect_threshold": 0.80,
        "maximum_validation_payload_bytes": 67_108_864,
    })


@pytest.fixture(scope="module")
def scaled_materialized():
    problem = make_h4_problem(seed=104729, kind="coupled", horizon=7)
    return problem, materialize_h4_problem(problem, _protocol())


@pytest.fixture(scope="module")
def scaled_information_result(scaled_materialized):
    _, materialized = scaled_materialized
    return gate.solve_information_form(
        materialized, _protocol(),
        gate._fresh_null_linalg(materialized.problem_id, "information"),
    )


def _check(phase: str) -> H4MaterializedIntegrityCheck:
    return H4MaterializedIntegrityCheck(phase, SHA_A, SHA_A, True)  # type: ignore[arg-type]


def _materialization(phases: tuple[str, ...]) -> H4MaterializationIdentity:
    return H4MaterializationIdentity(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", SHA_B,
        "h4-materialized-problem-v1", "h4-single-pass-v1", SHA_A, 1, True,
        tuple(_check(phase) for phase in phases),
    )


def _spans(phase: str, count: int):
    result = []
    for offset in range(count // 2):
        pair = offset if phase == "warmup" else offset + 3
        repetition = None if phase == "warmup" else offset
        order = "information_then_moment" if pair % 2 == 0 else "moment_then_information"
        arms = ("information", "moment") if order == "information_then_moment" else ("moment", "information")
        for position, arm in enumerate(arms):
            start = 10 * (2 * offset + position + 1)
            result.append(gate.H4ArmCallSpan(
                "h4-coupled-T7-dz4-dm4-seed104729-v1", phase, pair, repetition,
                order, position, arm, start, start + 1, 1,
            ))
    return tuple(result)


def _complete_gc() -> H4GarbageCollectorRecord:
    return H4GarbageCollectorRecord(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", True, None, True, True, True,
        None, None, True, True, True, None, True,
    )


def _complete_environment() -> H4EnvironmentRecord:
    torch_config = "torch-config"
    numpy_blas = "numpy-blas"
    return H4EnvironmentRecord(
        "clock", 1.0e-9, True, "processor", "platform", "Other", (0,),
        1, 1, "torch", "numpy", torch_config,
        hashlib.sha256(torch_config.encode()).hexdigest(), numpy_blas,
        hashlib.sha256(numpy_blas.encode()).hexdigest(), False,
        tuple((name, False, None) for name in gate._ENVIRONMENT_NAMES),
        tuple(
            H4PowerPolicyField(name, "not_applicable", "none", None, None)
            for name in gate._POWER_NAMES
        ),
        True, (), True,
    )  # type: ignore[arg-type]


def _not_applicable_power_policy_fields() -> tuple[H4PowerPolicyField, ...]:
    return tuple(
        H4PowerPolicyField(name, "not_applicable", "none", None, None)
        for name in gate._POWER_NAMES
    )


def test_h4_environment_uses_the_shared_live_process_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = getattr(artifacts, "process_cpu_affinity", None)
    assert callable(provider)
    monkeypatch.setattr(
        gate, "_capture_power_policy_fields", _not_applicable_power_policy_fields,
    )

    affinity = provider()
    h4_environment = gate._capture_environment()
    artifact_environment = artifacts.build_environment(SimpleNamespace(run=SimpleNamespace(
        device="cpu", dtype="float64", seed=20260721, deterministic=True,
    )))

    assert h4_environment.affinity_cpu_ids == affinity
    assert artifact_environment["process_cpu_affinity"] == affinity
    assert "affinity_cpu_ids" not in h4_environment.unavailable_fields


def test_h4_windows_processor_identity_uses_registry_when_platform_probes_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.platform, "processor", lambda: "")
    monkeypatch.setattr(gate.platform, "machine", lambda: "")
    monkeypatch.setattr(
        gate, "_windows_registry_processor_identity",
        lambda: "AMD Ryzen 9 9900X 12-Core Processor",
    )

    assert gate._processor_identity("Windows") == "AMD Ryzen 9 9900X 12-Core Processor"


def test_h4_processor_identity_remains_fail_closed_when_every_probe_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.platform, "processor", lambda: "")
    monkeypatch.setattr(gate.platform, "machine", lambda: "")
    monkeypatch.setattr(gate, "_windows_registry_processor_identity", lambda: None)

    assert gate._processor_identity("Windows") is None


def test_h4_ineligible_oracle_route_error_retains_decision_evidence(h4_config) -> None:
    oracle = h4_gaussian.evaluate_h4_oracle(canonical_h4_problem_bytes(
        make_h4_problem(seed=104729, kind="coupled", horizon=7),
    ))
    predictive_value = oracle.route_agreement.predictive_operand.value + 1.0
    predictive = replace(
        oracle.route_agreement.predictive_operand,
        value=predictive_value, value_norm=abs(predictive_value),
    )
    agreement = h4_gaussian._route_agreement(
        oracle.problem_id, oracle.problem_sha256,
        oracle.route_agreement.canonical_operand, predictive,
    )
    assert not agreement.eligible
    forged = replace(
        oracle,
        predictive_log_normalizer=predictive.value,
        route_agreement=agreement,
        operand_evidence=(agreement.canonical_operand, agreement.predictive_operand),
    )

    with pytest.raises(ValueError) as captured:
        gate._compact_oracle(forged, h4_config)

    message = str(captured.value)
    for field in (
        "problem_id=", "canonical_value=", "canonical_rounding_depth=",
        "predictive_value=", "predictive_rounding_depth=", "residual=",
        "final_allowance=", "allowance_scale_ratio=", "decisive=",
        "passed=", "eligible=",
    ):
        assert field in message


def test_h4_environment_affinity_availability_is_an_exact_iff() -> None:
    unavailable = replace(
        _complete_environment(), affinity_cpu_ids=None,
        unavailable_fields=("affinity_cpu_ids",), mandatory_facts_complete=False,
    )
    assert unavailable.affinity_cpu_ids is None

    with pytest.raises(ValueError, match="affinity_cpu_ids"):
        replace(
            _complete_environment(), unavailable_fields=("affinity_cpu_ids",),
            mandatory_facts_complete=False,
        )
    with pytest.raises(ValueError, match="affinity_cpu_ids"):
        replace(_complete_environment(), affinity_cpu_ids=None)


def test_affinity_unavailability_stops_h4_before_problems_or_timings(
    monkeypatch: pytest.MonkeyPatch, h4_config,
) -> None:
    def unavailable() -> tuple[int, ...]:
        raise RuntimeError("injected process-affinity unavailability")

    monkeypatch.setattr(gate, "process_cpu_affinity", unavailable, raising=False)
    monkeypatch.setattr(
        gate, "_capture_power_policy_fields", _not_applicable_power_policy_fields,
    )
    environment = gate._capture_environment()
    assert environment.affinity_cpu_ids is None
    assert "affinity_cpu_ids" in environment.unavailable_fields
    assert environment.mandatory_facts_complete is False

    captured: dict[str, object] = {}
    state = _valid_thread_state()

    def guard(work):
        return gate._H4ThreadGuardOutcome(work(), None, state)

    def assemble(core, **kwargs):
        captured["core"] = core
        captured["environment"] = kwargs["environment"]
        return core

    monkeypatch.setattr(gate, "_capture_environment", lambda: environment)
    monkeypatch.setattr(gate, "_run_thread_guard", guard)
    monkeypatch.setattr(gate, "_assemble_evaluation", assemble)

    returned = gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    core = captured["core"]
    assert returned is core
    assert captured["environment"] is environment
    assert core.result.status is gate.GateStatus.INCONCLUSIVE
    assert core.problems == ()
    assert core.raw_timings == ()
    assert (
        "capture complete mandatory H4 environment facts before timing"
        in core.result.obligations
    )


def _valid_thread_state() -> H4ThreadStateRecord:
    return H4ThreadStateRecord(
        None, 8, True, None, 1, True, 4, 4, True, True, 8, None, True,
    )


def _restoration_failure_state() -> H4ThreadStateRecord:
    return H4ThreadStateRecord(
        None, 8, True, None, 1, True, 4, 4, True, True, None,
        "builtins.RuntimeError: restore", False,
    )


def _inter_op_change_state() -> H4ThreadStateRecord:
    return H4ThreadStateRecord(
        None, 8, True, None, 1, True, 4, 5, False, True, 8, None, True,
    )


def _incomplete_phase_record(phase: str) -> H4ScaledIncompletePhaseRecord:
    obligations = {
        "warmup": "complete all six H4 warmup arm calls without exception",
        "gc_capture": "capture cyclic GC state before H4 timing",
        "gc_disable": "disable and verify cyclic GC before H4 timing",
        "timed_batch": "complete all 22 H4 timed arm calls and restore process-global state",
        "gc_restore": "restore exact prior cyclic GC state after H4 timing",
        "postflight": "complete exact H4 postflight schedule and release full problem objects",
    }
    warmups = () if phase == "warmup" else _spans("warmup", 6)
    timed = (
        _spans("timed", 2) if phase == "timed_batch"
        else (_spans("timed", 22) if phase in ("gc_restore", "postflight") else ())
    )
    gc_record = None if phase == "warmup" else _complete_gc()
    if phase == "gc_capture":
        gc_record = H4GarbageCollectorRecord(
            "h4-coupled-T7-dz4-dm4-seed104729-v1", True,
            "builtins.RuntimeError: capture", None, None, False, None, None,
            None, False, None, None, False,
        )
    elif phase == "gc_disable":
        gc_record = _gc_disable_failure()
    elif phase == "gc_restore":
        gc_record = _gc_restore_failure()
    schedule = (
        gate._incomplete_postflight_summary_for_test(7)
        if phase == "postflight" else None
    )
    return H4ScaledIncompletePhaseRecord(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", SHA_B, 0, 0, 0, 0,
        phase, _materialization(("after_materialization",)), warmups, timed,
        gc_record, schedule, "builtins.RuntimeError: bounded",
        obligations[phase],
    )  # type: ignore[arg-type]


def _install_top_level_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    thread_state: H4ThreadStateRecord | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    state = _valid_thread_state() if thread_state is None else thread_state
    monkeypatch.setattr(gate, "_capture_environment", _complete_environment)

    def guard(work):
        try:
            value = work()
            error = None
        except Exception as exception:
            value = None
            error = gate._stable_error(exception)
        return gate._H4ThreadGuardOutcome(value, error, state)

    def assemble(core, **_kwargs):
        captured["core"] = core
        return core

    monkeypatch.setattr(gate, "_run_thread_guard", guard)
    monkeypatch.setattr(gate, "_assemble_evaluation", assemble)
    return captured


def _observed_postflight(problem, materialized):
    class _ScheduleCoverage:
        def consume(self, _key: str) -> None:
            return None

    tracker = gate._PostflightTracker(
        problem, timed_batch_end_nanoseconds=1,
        coverage=_ScheduleCoverage(),  # type: ignore[arg-type]
    )
    return gate._ObservedPostflight(
        tracker, problem, materialized=materialized,
        integrity_checks=[
            gate._integrity_check(materialized, "after_materialization"),
            gate._integrity_check(materialized, "before_timed_batch"),
        ],
        problem_index=0, horizon_index=0, seed_index=0, kind_index=0,
        warmup_spans=_spans("warmup", 6), timed_spans=_spans("timed", 22),
        garbage_collector=_complete_gc(),
    )


def _gc_disable_failure() -> H4GarbageCollectorRecord:
    return H4GarbageCollectorRecord(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", True, None, True, True, True,
        "builtins.RuntimeError: disable", None, None, True, True, None, True,
    )


def _gc_restore_failure() -> H4GarbageCollectorRecord:
    return H4GarbageCollectorRecord(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", True, None, True, True, True,
        None, None, True, True, None, "builtins.RuntimeError: restore", False,
    )


def _typed_allowance(invariant: str, *, failed: bool = False):
    left_value = 1.0e-8 if failed else 0.0
    left = H4AllowanceOperand(
        "left", left_value, abs(left_value), 0.0, (1.0,), (), False,
        0.0, 0.0, 0.0,
    )
    right = H4AllowanceOperand(
        "right", 0.0, 0.0, 0.0, (1.0,), (), False,
        0.0, 0.0, 0.0,
    )
    comparison = 4096.0 * (
        3.0 * 2.220446049250313e-16
        / (1.0 - 3.0 * 2.220446049250313e-16)
    )
    residual = abs(left_value)
    anchor = invariant == "h3_anchor_identity"
    element = H4AllowanceElement(
        0, invariant,
        "h4-anchor-h3-zero-control-v1" if anchor else "h4-coupled-T7-dz4-dm4-seed104729-v1",
        "adapter_to_oracle" if anchor else "solver_to_oracle",
        None if anchor else 0, None if anchor else "information", "value",
        (1,), 0, 1.0, left, right, comparison, residual,
        residual / comparison, comparison, comparison, True, not failed,
    )
    count = dict(H4_ALLOWANCE_ELEMENT_COUNTS)[invariant]
    return gate.H4ApplicableAllowance(
        True, invariant, "vfe4.h4.allowance-element-stream.v1", count, count,
        SHA_A, element.normalized_residual, element,
        element.allowance_scale_ratio, element,
        element if failed else None, None, True, not failed,
    )


def _complete_pass_result():
    measurements = {
        "primary_seed_ratio_geometric_mean": 0.75,
        "primary_bootstrap_lower": 0.70,
        "primary_bootstrap_upper": 0.75,
        "primary_effect_threshold": 0.80,
        "primary_timed_ab_total": 110.0,
        "primary_timed_ba_total": 110.0,
        "maximum_solver_stopping_residual": 0.0,
        "maximum_allowance_scale_fraction": 0.0,
    }
    invariants = [
        gate.InvariantResult(name, True, 0.0, 1.0, "closed")
        for name in gate.H4_INVARIANT_NAMES
    ]
    invariants[-1] = gate.InvariantResult(
        "primary_effect_threshold", True, 0.75, 0.80,
        "bootstrap_interval_supports_effect",
    )
    allowances = {
        name: _typed_allowance(name) for name in gate.H4_ALLOWANCE_INVARIANT_NAMES
    }
    return gate.H4GateResult(
        "H4", gate.GateStatus.PASS, measurements, tuple(invariants), allowances, (),
    )


def _condition_summary(
    name: str,
    record,
) -> gate.H4ConditionStreamSummary:
    expected = {
        "oracle_posterior": 120,
        "terminal_posterior": 2640,
        "oracle_innovation": 2120,
        "moment_innovation": 23320,
    }[name]
    metrics = (
        (
            "minimum_eigenvalue", "maximum_eigenvalue",
            "maximum_condition_number", "minimum_cholesky_pivot",
            "maximum_mean_infinity_norm",
        )
        if "posterior" in name
        else (
            "minimum_eigenvalue", "maximum_eigenvalue",
            "maximum_condition_number",
        )
    )
    if not record.eligible:
        metrics = (*metrics, "first_ineligible")
    witnesses = tuple(
        gate.H4ConditionWitness(metric, 0, record) for metric in metrics
    )
    return gate.H4ConditionStreamSummary(
        name, "vfe4.h4.condition-record-stream.v1", expected, expected,
        SHA_A, expected if record.eligible else expected - 1,
        0 if record.eligible else 1, witnesses, record.eligible,
    )


def _condition_streams(h4_config, *, outside: bool):
    envelope = h4_config.condition_envelope
    posterior = gate.posterior_condition_record(
        problem_id="condition-problem", problem_sha256=SHA_A,
        source="information", repetition_index=0, dimension=1,
        minimum_eigenvalue=envelope.posterior_minimum_eigenvalue,
        maximum_eigenvalue=envelope.posterior_maximum_eigenvalue,
        condition_number=envelope.posterior_maximum_condition_number,
        minimum_cholesky_pivot=envelope.posterior_minimum_cholesky_pivot,
        mean_infinity_norm=envelope.posterior_maximum_mean_infinity_norm,
        envelope=envelope,
    )
    minimum = envelope.innovation_minimum_eigenvalue
    if outside:
        minimum = math.nextafter(minimum, 0.0)
    innovation = gate.innovation_condition_record(
        problem_id="condition-problem", problem_sha256=SHA_A,
        source="moment", repetition_index=0, factor_id="observation[1]",
        time_index=1, parent_coordinate_indices=(0,), innovation_dimension=1,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=envelope.innovation_maximum_eigenvalue,
        condition_number=envelope.innovation_maximum_condition_number,
        envelope=envelope,
    )
    return (
        _condition_summary("oracle_posterior", posterior),
        _condition_summary("terminal_posterior", posterior),
        _condition_summary("oracle_innovation", innovation if not outside else replace(innovation, eligible=True, minimum_eigenvalue=envelope.innovation_minimum_eigenvalue)),
        _condition_summary("moment_innovation", innovation),
    )


def _complete_result_for_conditions(conditions):
    interval = SimpleNamespace(
        invariant_passed=True, invariant_value=0.75, invariant_limit=0.80,
        invariant_detail="bootstrap_interval_supports_effect",
        status_if_other_invariants_eligible=gate.GateStatus.PASS,
        obligation=None,
    )
    return gate._complete_gate_result(
        allowances=tuple(
            _typed_allowance(name) for name in gate.H4_ALLOWANCE_INVARIANT_NAMES
        ),
        conditions=conditions,
        coverage=(SimpleNamespace(
            complete=True, observed_key_count=1, expected_key_count=1,
        ),),
        timing_summary=SimpleNamespace(geometric_mean_ratio=0.75),
        balance=SimpleNamespace(
            matches=True, observed_ab_total=110, observed_ba_total=110,
        ),
        bootstrap=SimpleNamespace(lower=0.70, upper=0.75),
        interval_decision=interval,
        maximum_stopping_residual=0.0, environment_complete=True,
    )


def test_task4_records_have_exact_frozen_slotted_field_order() -> None:
    for record, expected in RECORD_FIELDS.items():
        assert tuple(field.name for field in fields(record)) == expected
        assert record.__dataclass_params__.frozen is True
        assert "__dict__" not in record.__slots__
    sample = _check("after_materialization")
    with pytest.raises(FrozenInstanceError):
        sample.phase = "before_timed_batch"  # type: ignore[misc]


def test_integrity_records_reject_wrong_digest_phase_prefix_and_failure_carrier() -> None:
    with pytest.raises(ValueError):
        H4MaterializedIntegrityCheck("after_materialization", SHA_A, SHA_B, True)
    with pytest.raises(ValueError):
        H4MaterializedIntegrityCheck("unknown", SHA_A, SHA_A, True)  # type: ignore[arg-type]
    complete = _materialization((
        "after_materialization", "before_timed_batch", "after_timed_batch", "after_postflight",
    ))
    assert complete.materialization_count == 1
    with pytest.raises(ValueError):
        replace(complete, integrity_checks=(complete.integrity_checks[1], complete.integrity_checks[0]))


@pytest.mark.parametrize("phase", ("warmup", "gc_capture", "gc_disable", "timed_batch", "gc_restore", "postflight"))
def test_scaled_incomplete_phase_has_exact_prefix_and_obligation(phase: str) -> None:
    record = _incomplete_phase_record(phase)
    assert record.obligation == gate._INCOMPLETE_OBLIGATIONS[phase]
    with pytest.raises(ValueError):
        replace(record, obligation="retry")


@pytest.mark.parametrize("failure_kind", ("seam_exception", "digest_mismatch"))
@pytest.mark.parametrize("checkpoint", ("after_materialization", "before_timed_batch", "after_timed_batch", "after_postflight"))
def test_scaled_integrity_failure_matrix_is_typed_and_prefix_checked(checkpoint: str, failure_kind: str) -> None:
    prefix = {
        "after_materialization": (),
        "before_timed_batch": ("after_materialization",),
        "after_timed_batch": ("after_materialization", "before_timed_batch"),
        "after_postflight": ("after_materialization", "before_timed_batch", "after_timed_batch"),
    }[checkpoint]
    warmups = () if checkpoint == "after_materialization" else _spans("warmup", 6)
    timed = _spans("timed", 22) if checkpoint in ("after_timed_batch", "after_postflight") else ()
    gc_record = _complete_gc() if timed else None
    schedule = gate._incomplete_postflight_summary_for_test(7) if timed else None
    record = H4ScaledMaterializedIntegrityFailureRecord(
        "h4-coupled-T7-dz4-dm4-seed104729-v1", SHA_B, 0, 0, 0, 0,
        "h4-materialized-problem-v1", "h4-single-pass-v1", 1, True, checkpoint,
        SHA_A, tuple(_check(item) for item in prefix), failure_kind,
        None if failure_kind == "seam_exception" else SHA_B,
        "builtins.RuntimeError: seam" if failure_kind == "seam_exception" else None,
        warmups, timed, gc_record, schedule, "materialized_integrity",
    )
    assert record.checkpoint == checkpoint
    with pytest.raises(ValueError):
        replace(record, completed_integrity_checks=()) if prefix else replace(record, warmup_spans=_spans("warmup", 6))


def test_expected_postflight_schedule_is_exact_and_streamed() -> None:
    aggregate = 0
    for horizon, expected_count in ((7, 636), (15, 1076), (31, 1956)):
        problem = make_h4_problem(seed=104729, kind="coupled", horizon=horizon)  # type: ignore[arg-type]
        keys = tuple(gate._iter_expected_postflight_event_keys(problem))
        assert len(keys) == expected_count == 251 + 55 * horizon
        assert tuple(key.event_index for key in keys) == tuple(range(expected_count))
        assert (keys[0].phase, keys[0].integrity_phase) == ("materialized_integrity", "after_timed_batch")
        assert keys[-1].phase == "stream_compaction"
        count, digest = gate._postflight_key_stream_digest(problem.canonical_sha256, iter(keys))
        local = hashlib.sha256(b"vfe4.h4.postflight-event-key-stream.v1\x00" + problem.canonical_sha256.encode("ascii") + b"\x00")
        for key in keys:
            payload = json.dumps(dataclasses.asdict(key), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            local.update(len(payload).to_bytes(8, "big")); local.update(payload)
        assert count == expected_count
        assert digest == local.hexdigest()
        aggregate += 40 * expected_count
    assert aggregate == 146720


@pytest.mark.parametrize(
    "outside, expected_status, expected_obligations",
    (
        (False, gate.GateStatus.PASS, ()),
        (
            True, gate.GateStatus.INCONCLUSIVE,
            ("scaled_condition_envelope: resolve incomplete H4 eligibility evidence",),
        ),
    ),
)
def test_complete_condition_stream_accepts_inclusive_boundary_and_types_just_outside(
    h4_config, outside: bool, expected_status, expected_obligations,
) -> None:
    conditions = _condition_streams(h4_config, outside=outside)
    assert conditions[-1].all_eligible is (not outside)
    assert conditions[-1].ineligible_record_count == (1 if outside else 0)
    result = _complete_result_for_conditions(conditions)
    assert result.status is expected_status
    assert result.obligations == expected_obligations
    condition_invariant = result.invariants[
        gate.H4_INVARIANT_NAMES.index("scaled_condition_envelope")
    ]
    assert condition_invariant.passed is (not outside)
    assert condition_invariant.value == float(
        sum(item.eligible_record_count for item in conditions)
    )
    assert condition_invariant.limit == float(
        sum(item.observed_record_count for item in conditions)
    )
    gate._validate_complete_condition_evidence(result, conditions)

    opposite = _complete_result_for_conditions(
        _condition_streams(h4_config, outside=not outside)
    )
    with pytest.raises(ValueError, match="condition-envelope evidence"):
        gate._validate_complete_condition_evidence(opposite, conditions)


def test_real_scaled_problem_keeps_all_three_warmups_and_eleven_timed_pairs(
    scaled_materialized,
) -> None:
    protocol = _protocol()
    _, materialized = scaled_materialized
    ticks = iter(range(1, 1000))
    prior_gc = gc.isenabled()
    outcome = gate._run_warmup_and_timed_batch(
        materialized, protocol, horizon_index=0, seed_index=0, kind_index=0,
        perf_counter_ns=lambda: next(ticks),
    )
    assert len(outcome.warmup_spans) == 6
    assert len(outcome.timed_spans) == 22
    assert len(outcome.results) == 22
    assert len(outcome.timings) == 11
    assert tuple(item.repetition_index for item in outcome.timings) == tuple(range(11))
    assert all(type(item) is H4TimingRecord for item in outcome.timings)
    assert outcome.garbage_collector.restored_exact_prior_state is True
    assert gc.isenabled() is prior_gc


@pytest.mark.parametrize("earliest_phase", ("gc_disable", "timed_batch"))
def test_batch_reports_earliest_failure_while_retaining_later_gc_restore_evidence(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized, earliest_phase: str,
) -> None:
    _, materialized = scaled_materialized
    enabled = True

    def isenabled() -> bool:
        return enabled

    def disable() -> None:
        nonlocal enabled
        if earliest_phase == "gc_disable":
            raise RuntimeError("disable first")
        enabled = False

    def enable() -> None:
        raise RuntimeError("restore later")

    calls = 0

    def solve(*_args):
        nonlocal calls
        calls += 1
        if earliest_phase == "timed_batch" and calls == 7:
            raise RuntimeError("timed first")
        return object()

    monkeypatch.setattr(gate.gc, "isenabled", isenabled)
    monkeypatch.setattr(gate.gc, "disable", disable)
    monkeypatch.setattr(gate.gc, "enable", enable)
    monkeypatch.setattr(gate, "solve_information_form", solve)
    monkeypatch.setattr(gate, "solve_moment_form", solve)

    ticks = iter(range(1, 1000))
    with pytest.raises(gate._H4BatchFailure) as failure:
        gate._run_warmup_and_timed_batch(
            materialized, _protocol(), horizon_index=0, seed_index=0,
            kind_index=0, perf_counter_ns=lambda: next(ticks),
        )
    assert failure.value.phase == earliest_phase
    assert failure.value.garbage_collector is not None
    assert failure.value.garbage_collector.restoration_error == (
        "builtins.RuntimeError: restore later"
    )
    assert failure.value.garbage_collector.restored_exact_prior_state is False


def test_integrity_seam_classifies_exception_and_returned_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized,
) -> None:
    _, materialized = scaled_materialized

    def fail_seam(_materialized):
        raise RuntimeError("seam broke")

    monkeypatch.setattr(gate, "_assert_h4_materialized_integrity", fail_seam)
    with pytest.raises(gate._H4IntegrityCheckFailure) as seam:
        gate._integrity_check(materialized, "after_materialization")
    assert seam.value.failure_kind == "seam_exception"
    assert seam.value.observed_tensor_sha256 is None
    assert seam.value.seam_error == "builtins.RuntimeError: seam broke"

    monkeypatch.setattr(
        gate, "_assert_h4_materialized_integrity", lambda _materialized: SHA_B,
    )
    with pytest.raises(gate._H4IntegrityCheckFailure) as mismatch:
        gate._integrity_check(materialized, "before_timed_batch")
    assert mismatch.value.failure_kind == "digest_mismatch"
    assert mismatch.value.observed_tensor_sha256 == SHA_B
    assert mismatch.value.seam_error is None


def test_postflight_action_and_integrity_failures_emit_typed_terminal_carriers(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized,
) -> None:
    problem, materialized = scaled_materialized
    checks = [
        gate._integrity_check(materialized, "after_materialization"),
        gate._integrity_check(materialized, "before_timed_batch"),
    ]
    warmups = _spans("warmup", 6)
    timed = _spans("timed", 22)

    def observed_path():
        class _CoverageSink:
            def consume(self, _key: str) -> None:
                return None

        tracker = gate._PostflightTracker(
            problem, timed_batch_end_nanoseconds=timed[-1].end_nanoseconds,
            coverage=_CoverageSink(),  # type: ignore[arg-type]
        )
        return gate._ObservedPostflight(
            tracker, problem, materialized=materialized,
            integrity_checks=list(checks), problem_index=0, horizon_index=0,
            seed_index=0, kind_index=0, warmup_spans=warmups,
            timed_spans=timed, garbage_collector=_complete_gc(),
        )

    observed = observed_path()
    with pytest.raises(gate._H4ScaledCarrierFailure) as postflight:
        observed.call(
            "terminal_conversion",
            lambda: (_ for _ in ()).throw(RuntimeError("convert broke")),
            repetition_index=0, arm="information",
        )
    assert type(postflight.value.record) is H4ScaledIncompletePhaseRecord
    assert postflight.value.record.phase == "postflight"
    assert postflight.value.record.postflight_schedule is not None
    assert postflight.value.record.postflight_schedule.complete is False

    monkeypatch.setattr(
        gate, "_assert_h4_materialized_integrity", lambda _materialized: SHA_B,
    )
    observed = observed_path()
    with pytest.raises(gate._H4ScaledCarrierFailure) as integrity:
        observed.call(
            "materialized_integrity",
            lambda: gate._integrity_check(materialized, "after_timed_batch"),
            integrity_phase="after_timed_batch",
        )
    assert type(integrity.value.record) is H4ScaledMaterializedIntegrityFailureRecord
    assert integrity.value.record.checkpoint == "after_timed_batch"
    assert integrity.value.record.failure_kind == "digest_mismatch"


@pytest.mark.parametrize(
    "category",
    ("native_hash", "diagnostic_hash", "replay_record", "coverage_consume"),
)
def test_native_diagnostic_event_owns_hash_replay_and_coverage_failures(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized,
    scaled_information_result, category: str,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)

    class _Coverage:
        def consume(self, _key: str) -> None:
            if category == "coverage_consume":
                raise RuntimeError("coverage consume injection")

    def injected(*_args, **_kwargs):
        raise RuntimeError(f"{category} injection")

    if category == "native_hash":
        monkeypatch.setattr(gate, "_native_stream", injected)
    elif category == "diagnostic_hash":
        monkeypatch.setattr(gate, "_diagnostic_stream", injected)
    elif category == "replay_record":
        monkeypatch.setattr(gate, "H4NativeReplayRecord", injected)

    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._observe_native_diagnostic_replay(
            observed, materialized=materialized,
            result=scaled_information_result, repetition_index=0,
            arm="information", replay_records=[], moment_diagnostics={},
            coverage=_Coverage(),
        )
    record = failure.value.record
    assert type(record) is H4ScaledIncompletePhaseRecord
    assert record.phase == "postflight"
    assert record.postflight_schedule is not None
    assert record.postflight_schedule.complete is False
    assert record.postflight_schedule.observed_event_count == 0
    assert tuple(
        item.phase for item in record.materialization.integrity_checks
    ) == ("after_materialization", "before_timed_batch")
    assert "injection" in record.stable_error


def test_compact_result_preparation_is_owned_by_first_equivalence_event(
    scaled_materialized,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)

    class _OneGroupAccumulator:
        def consume_observed(self, _source, observer) -> None:
            observer("kl_to_zero", None, lambda: None)

    def fail_compaction() -> None:
        raise RuntimeError("compact result injection")

    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._consume_allowance_source_at_events(
            observed, _OneGroupAccumulator(), object(), repetition_index=0,
            arm="information", before_first_group=fail_compaction,
        )
    assert type(failure.value.record) is H4ScaledIncompletePhaseRecord
    assert failure.value.record.phase == "postflight"
    assert failure.value.record.postflight_schedule is not None
    assert failure.value.record.postflight_schedule.observed_event_count == 0
    assert failure.value.record.stable_error == (
        "builtins.RuntimeError: compact result injection"
    )


@pytest.mark.parametrize("category", ("reverse_kl", "allowance_source"))
def test_first_equivalence_event_owns_kl_and_allowance_source_construction(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized, category: str,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)

    class _DeferredAccumulator:
        def defer_source(self, source_factory):
            source_factory()
            return ()

    def injected(*_args, **_kwargs):
        raise RuntimeError(f"{category} injection")

    if category == "reverse_kl":
        monkeypatch.setattr(gate, "reverse_kl_to_h4_oracle", injected)
    else:
        monkeypatch.setattr(gate, "H4AllowanceResultSource", injected)

    def source_factory():
        if category == "reverse_kl":
            return gate.reverse_kl_to_h4_oracle(object(), mean=(), precision=())
        return gate.H4AllowanceResultSource(
            b"payload", 0, object(), object(), object(), object(),
        )

    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._consume_allowance_source_at_events(
            observed, _DeferredAccumulator(), None, repetition_index=0,
            arm="information", source_factory=source_factory,
        )
    record = failure.value.record
    assert type(record) is H4ScaledIncompletePhaseRecord
    assert record.phase == "postflight"
    assert record.postflight_schedule is not None
    assert record.postflight_schedule.observed_event_count == 0
    assert record.postflight_schedule.complete is False
    assert tuple(
        item.phase for item in record.materialization.integrity_checks
    ) == ("after_materialization", "before_timed_batch")
    assert record.stable_error == f"builtins.RuntimeError: {category} injection"


def test_stream_release_operation_is_owned_by_compaction_event(
    scaled_materialized,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)

    def fail_release() -> int:
        raise RuntimeError("release injection")

    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._observe_stream_compaction(observed, fail_release)
    assert type(failure.value.record) is H4ScaledIncompletePhaseRecord
    assert failure.value.record.phase == "postflight"
    assert failure.value.record.postflight_schedule is not None
    assert failure.value.record.postflight_schedule.observed_event_count == 0
    assert failure.value.record.stable_error == (
        "builtins.RuntimeError: release injection"
    )


@pytest.mark.parametrize(
    "category",
    (
        "trace_construction", "execution_trace_coverage",
        "materialization_identity", "problem_evaluation",
    ),
)
def test_postflight_finalization_converts_failures_after_last_event_to_typed_carrier(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized, category: str,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)
    observed.integrity_checks.extend((
        gate._integrity_check(materialized, "after_timed_batch"),
        gate._integrity_check(materialized, "after_postflight"),
    ))
    expected = tuple(gate._iter_expected_postflight_event_keys(problem))
    for offset, event in enumerate(expected):
        instant = 1_000 + offset
        observed.tracker.record(event, instant, instant)
    observed.event_index = len(expected)
    injection_calls: list[str] = []

    def injected(*_args, **_kwargs):
        injection_calls.append(category)
        raise RuntimeError(f"{category} injection")

    if category == "trace_construction":
        monkeypatch.setattr(gate, "H4ExecutionTrace", injected)
    elif category == "materialization_identity":
        monkeypatch.setattr(gate, "_materialization_identity", injected)
    elif category == "problem_evaluation":
        monkeypatch.setattr(gate, "H4ProblemEvaluation", injected)

    class _Coverage:
        def consume(self, _key: str) -> None:
            injected()

    def finalize(_schedule):
        if category == "trace_construction":
            return gate.H4ExecutionTrace()
        if category == "execution_trace_coverage":
            _Coverage().consume("execution-trace")
        elif category == "materialization_identity":
            gate._materialization_identity(
                materialized, tuple(observed.integrity_checks),
            )
        else:
            return gate.H4ProblemEvaluation()
        return object()

    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._finalize_postflight_boundary(observed, finalize)
    record = failure.value.record
    assert type(record) is H4ScaledIncompletePhaseRecord
    assert record.phase == "postflight"
    assert record.postflight_schedule is not None
    assert record.postflight_schedule.complete is True
    assert record.postflight_schedule.observed_event_count == len(expected)
    assert tuple(
        item.phase for item in record.materialization.integrity_checks
    ) == (
        "after_materialization", "before_timed_batch",
        "after_timed_batch", "after_postflight",
    )
    assert record.stable_error == f"builtins.RuntimeError: {category} injection"
    assert injection_calls == [category]


def test_postflight_finalization_converts_schedule_finish_failure_to_typed_carrier(
    monkeypatch: pytest.MonkeyPatch, scaled_materialized,
) -> None:
    problem, materialized = scaled_materialized
    observed = _observed_postflight(problem, materialized)
    observed.integrity_checks.extend((
        gate._integrity_check(materialized, "after_timed_batch"),
        gate._integrity_check(materialized, "after_postflight"),
    ))
    expected = tuple(gate._iter_expected_postflight_event_keys(problem))
    for offset, event in enumerate(expected):
        instant = 1_000 + offset
        observed.tracker.record(event, instant, instant)
    observed.event_index = len(expected)

    finish_calls: list[str] = []

    def injected(_tracker):
        finish_calls.append("finish")
        raise RuntimeError("schedule_finish injection")

    monkeypatch.setattr(gate._PostflightTracker, "finish", injected)
    with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
        gate._finalize_postflight_boundary(observed, lambda _schedule: object())
    record = failure.value.record
    assert type(record) is H4ScaledIncompletePhaseRecord
    assert record.phase == "postflight"
    assert record.postflight_schedule is not None
    assert record.postflight_schedule.complete is True
    assert record.postflight_schedule.observed_event_count == len(expected)
    assert tuple(
        item.phase for item in record.materialization.integrity_checks
    ) == (
        "after_materialization", "before_timed_batch",
        "after_timed_batch", "after_postflight",
    )
    assert record.stable_error == (
        "builtins.RuntimeError: schedule_finish injection"
    )
    assert finish_calls == ["finish"]


def test_moment_condition_computation_occurs_inside_its_declared_postflight_event(
    monkeypatch: pytest.MonkeyPatch, h4_config,
) -> None:
    diagnostics = (
        H4InnovationDiagnostic(
            "observation[1]", 1, (0,), 1, 1.0, 2.0, 2.0, 1.0,
        ),
        H4InnovationDiagnostic(
            "observation[2]", 2, (1,), 1, 2.0, 4.0, 2.0, 1.0,
        ),
    )
    timeline: list[str] = []
    original = gate.innovation_condition_record

    def compute(**kwargs):
        timeline.append(f"compute:{kwargs['factor_id']}")
        return original(**kwargs)

    class _Observed:
        def call(self, phase, action, **kwargs):
            factor_id = kwargs["factor_id"]
            timeline.append(f"enter:{phase}:{factor_id}")
            value = action()
            timeline.append(f"exit:{phase}:{factor_id}")
            return value

    monkeypatch.setattr(gate, "innovation_condition_record", compute)
    records = gate._observe_moment_innovation_conditions(
        _Observed(), diagnostics, problem_id="problem", problem_sha256=SHA_A,
        repetition_index=3, config=h4_config,
    )
    assert tuple(item.factor_id for item in records) == (
        "observation[1]", "observation[2]",
    )
    assert timeline == [
        "enter:moment_innovation_condition:observation[1]",
        "compute:observation[1]",
        "exit:moment_innovation_condition:observation[1]",
        "enter:moment_innovation_condition:observation[2]",
        "compute:observation[2]",
        "exit:moment_innovation_condition:observation[2]",
    ]


@pytest.mark.parametrize("mutate_last_group", (False, True))
def test_each_allowance_group_is_consumed_inside_its_event_and_last_mutation_fails_closed(
    scaled_materialized, mutate_last_group: bool,
) -> None:
    problem, materialized = scaled_materialized
    warmups = _spans("warmup", 6)
    timed = _spans("timed", 22)
    timeline: list[str] = []

    class _CoverageSink:
        def consume(self, key: str) -> None:
            event = json.loads(key)
            timeline.append(f"record:{event['equivalence_component']}")

    tracker = gate._PostflightTracker(
        problem, timed_batch_end_nanoseconds=timed[-1].end_nanoseconds,
        coverage=_CoverageSink(),  # type: ignore[arg-type]
    )
    expected = tuple(gate._iter_expected_postflight_event_keys(problem))
    allowance_start = next(
        index for index, event in enumerate(expected)
        if event.phase == "equivalence_group"
    )
    for offset, event in enumerate(expected[:allowance_start]):
        instant = 1_000 + offset
        tracker.record(event, instant, instant)
    timeline.clear()
    observed = gate._ObservedPostflight(
        tracker, problem, materialized=materialized,
        integrity_checks=[
            gate._integrity_check(materialized, "after_materialization"),
            gate._integrity_check(materialized, "before_timed_batch"),
        ],
        problem_index=0, horizon_index=0, seed_index=0, kind_index=0,
        warmup_spans=warmups, timed_spans=timed,
        garbage_collector=_complete_gc(),
    )
    observed.event_index = allowance_start
    selected_names = (
        "initial", "terminal",
        *(f"observation[{index}]" for index in range(1, 8)),
    )
    descriptors = (
        ("kl_to_zero", None), ("h", None), ("J", None),
        *((component, name) for name in selected_names
          for component in ("selected_mean", "selected_covariance")),
        ("objective", None),
    )

    class _FakeAccumulator:
        def consume_observed(self, _source, observer) -> None:
            for index, (component, selected_name) in enumerate(descriptors):
                def consume(index=index, component=component):
                    timeline.append(f"consume:{component}")
                    if mutate_last_group and index == len(descriptors) - 1:
                        raise RuntimeError("last allowance chunk mutated")
                observer(component, selected_name, consume)

    if mutate_last_group:
        with pytest.raises(gate._H4ScaledCarrierFailure) as failure:
            gate._consume_allowance_source_at_events(
                observed, _FakeAccumulator(), object(), repetition_index=0,
                arm="information",
            )
        assert type(failure.value.record) is H4ScaledIncompletePhaseRecord
        assert failure.value.record.phase == "postflight"
        assert failure.value.record.stable_error == (
            "builtins.RuntimeError: last allowance chunk mutated"
        )
        assert failure.value.record.postflight_schedule is not None
        assert failure.value.record.postflight_schedule.observed_event_count == (
            allowance_start + len(descriptors) - 1
        )
        assert timeline[-1] == "consume:objective"
    else:
        gate._consume_allowance_source_at_events(
            observed, _FakeAccumulator(), object(), repetition_index=0,
            arm="information",
        )
        assert len(timeline) == 2 * len(descriptors)
        assert all(
            timeline[2 * index:2 * index + 2]
            == [f"consume:{component}", f"record:{component}"]
            for index, (component, _) in enumerate(descriptors)
        )


class _FakeTorch:
    def __init__(self, *, intra: int = 8, inter: int = 4, fail_get_inter: bool = False, fail_set: bool = False):
        self.intra = intra; self.inter = inter; self.fail_get_inter = fail_get_inter; self.fail_set = fail_set
        self.set_calls: list[int] = []

    def get_num_threads(self) -> int:
        return self.intra

    def get_num_interop_threads(self) -> int:
        if self.fail_get_inter:
            raise RuntimeError("inter capture")
        return self.inter

    def set_num_threads(self, value: int) -> None:
        self.set_calls.append(value)
        self.intra = value
        if self.fail_set and value == 1:
            raise RuntimeError("set failed after mutation")


def test_thread_guard_suppresses_mutation_on_partial_capture_and_restores_after_set_attempt() -> None:
    partial = _FakeTorch(fail_get_inter=True)
    skipped = gate._run_thread_guard(lambda: "unreachable", torch_api=partial)
    assert skipped.value is None and skipped.work_error is None
    assert skipped.state.prior_intra_op_threads == 8
    assert skipped.state.capture_error == "builtins.RuntimeError: inter capture"
    assert partial.set_calls == []
    assert skipped.state.set_attempted is False and skipped.state.restore_attempted is False

    failed_set = _FakeTorch(fail_set=True)
    restored = gate._run_thread_guard(lambda: "unreachable", torch_api=failed_set)
    assert failed_set.set_calls == [1, 8]
    assert restored.state.set_attempted is True and restored.state.restore_attempted is True
    assert restored.state.restored_exact_prior_state is True
    assert restored.state.set_error == "builtins.RuntimeError: set failed after mutation"

    success = _FakeTorch()
    completed = gate._run_thread_guard(lambda: "done", torch_api=success)
    assert completed.value == "done" and completed.work_error is None
    assert completed.state.verified_one is True
    assert completed.state.inter_op_unchanged is True
    assert completed.state.restored_exact_prior_state is True


@pytest.mark.parametrize(
    "failing_fixture, failure_detail",
    (
        ("h3-coupled-v1", "oracle integrity failed"),
        ("h3-zero-control-v1", "solver failed"),
    ),
)
def test_evaluate_h4_maps_each_anchor_exception_to_its_slot_and_preserves_the_other(
    monkeypatch: pytest.MonkeyPatch, h4_config, failing_fixture: str,
    failure_detail: str,
) -> None:
    captured = _install_top_level_harness(monkeypatch)
    coupled = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
        marker="coupled-complete",
    )
    zero = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
        marker="zero-complete",
    )
    calls: list[str] = []

    def evaluate_anchor(_fixture, *, expected_fixture_id, config):
        assert config is h4_config
        calls.append(expected_fixture_id)
        if expected_fixture_id == failing_fixture:
            raise RuntimeError(failure_detail)
        evaluation = coupled if expected_fixture_id == "h3-coupled-v1" else zero
        return gate._AnchorWork(evaluation, object())

    monkeypatch.setattr(gate, "_evaluate_anchor", evaluate_anchor)
    result = gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    assert result is captured["core"]
    assert calls == ["h3-coupled-v1", "h3-zero-control-v1"]
    core = captured["core"]
    failed_index = 0 if failing_fixture == "h3-coupled-v1" else 1
    completed_index = 1 - failed_index
    failed = core.anchors[failed_index]
    assert type(failed) is H4UnavailablePhaseRecord
    assert failed.phase == (
        "anchor_coupled" if failed_index == 0 else "anchor_zero_control"
    )
    assert failed.reason == f"builtins.RuntimeError: {failure_detail}"
    assert core.anchors[completed_index] is (zero if completed_index else coupled)
    assert core.result.status is gate.GateStatus.INCONCLUSIVE


def test_evaluate_h4_preserves_anchor_slots_when_scaled_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, h4_config, scaled_materialized,
) -> None:
    captured = _install_top_level_harness(monkeypatch)
    problem, materialized = scaled_materialized
    coupled = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
    )
    zero = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
    )

    def evaluate_anchor(_fixture, *, expected_fixture_id, config):
        assert config is h4_config
        evaluation = coupled if expected_fixture_id == "h3-coupled-v1" else zero
        return gate._AnchorWork(evaluation, object())

    class _AnchorAccumulator:
        def consume(self, _source) -> None:
            return None

        def anchor_identity_record(self):
            return _typed_allowance("h3_anchor_identity")

    failure_record = H4ScaledMaterializedIntegrityFailureRecord(
        problem.problem_id, problem.canonical_sha256, 0, 0, 0, 0,
        materialized.materialization_version, materialized.protocol_id, 1, True,
        "after_materialization", materialized.tensor_sha256, (),
        "seam_exception", None, "builtins.RuntimeError: preflight seam",
        (), (), None, None, "materialized_integrity",
    )

    def fail_preflight(*_args, **_kwargs):
        raise gate._H4ScaledCarrierFailure(failure_record, "preflight seam")

    monkeypatch.setattr(gate, "_evaluate_anchor", evaluate_anchor)
    monkeypatch.setattr(
        gate, "new_h4_six_invariant_allowance_accumulator",
        lambda: _AnchorAccumulator(),
    )
    monkeypatch.setattr(gate, "_generate_scaled_problems", lambda _config: (object(),) * 120)
    monkeypatch.setattr(gate, "_ConditionAccumulator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gate, "_CoverageAccumulator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gate, "_preflight_scaled", fail_preflight)
    gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    core = captured["core"]
    assert core.anchors == (coupled, zero)
    assert core.problems == (failure_record,)
    assert tuple(item.phase for item in core.unavailable_phases) == (
        "scaled_preflight", "statistics",
    )
    assert core.result.status is gate.GateStatus.INCONCLUSIVE


def test_evaluate_h4_preserves_anchors_on_plain_scaled_preflight_error(
    monkeypatch: pytest.MonkeyPatch, h4_config,
) -> None:
    captured = _install_top_level_harness(monkeypatch)
    coupled = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
    )
    zero = SimpleNamespace(
        information_result=SimpleNamespace(stopping_residual=0.0),
        moment_result=SimpleNamespace(stopping_residual=0.0),
    )

    def evaluate_anchor(_fixture, *, expected_fixture_id, config):
        assert config is h4_config
        evaluation = coupled if expected_fixture_id == "h3-coupled-v1" else zero
        return gate._AnchorWork(evaluation, object())

    class _AnchorAccumulator:
        def consume(self, _source) -> None:
            return None

        def anchor_identity_record(self):
            return _typed_allowance("h3_anchor_identity")

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("scaled problem h4-coupled-T15 route is ineligible")

    monkeypatch.setattr(gate, "_evaluate_anchor", evaluate_anchor)
    monkeypatch.setattr(
        gate, "new_h4_six_invariant_allowance_accumulator",
        lambda: _AnchorAccumulator(),
    )
    monkeypatch.setattr(gate, "_generate_scaled_problems", lambda _config: (object(),) * 120)
    monkeypatch.setattr(gate, "_ConditionAccumulator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gate, "_CoverageAccumulator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gate, "_preflight_scaled", fail_preflight)

    gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    core = captured["core"]
    assert core.anchors == (coupled, zero)
    assert core.problems == ()
    assert tuple(item.phase for item in core.unavailable_phases) == (
        "scaled_preflight", "statistics",
    )
    assert any(
        "scaled problem h4-coupled-T15 route is ineligible" in item
        for item in core.result.obligations
    )


@pytest.mark.parametrize(
    "phase", ("warmup", "gc_capture", "gc_disable", "timed_batch", "gc_restore", "postflight"),
)
def test_top_level_evaluate_h4_preserves_all_six_typed_incomplete_injections(
    monkeypatch: pytest.MonkeyPatch, h4_config, phase: str,
) -> None:
    captured = _install_top_level_harness(monkeypatch)
    record = _incomplete_phase_record(phase)
    expected_core = gate._empty_core(
        f"{phase}: bounded", problems=(record,),
        raw_timings=() if phase not in ("timed_batch", "gc_restore", "postflight") else (),
    )
    monkeypatch.setattr(gate, "_evaluate_complete_core", lambda *_args, **_kwargs: expected_core)
    returned = gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    assert returned is captured["core"] is expected_core
    assert type(returned.problems[-1]) is H4ScaledIncompletePhaseRecord
    assert returned.problems[-1].phase == phase
    assert returned.result.status is gate.GateStatus.INCONCLUSIVE


@pytest.mark.parametrize("thread_state", (_restoration_failure_state(), _inter_op_change_state()))
def test_complete_evidence_survives_thread_or_inter_op_restoration_failure(
    monkeypatch: pytest.MonkeyPatch, h4_config, thread_state: H4ThreadStateRecord,
) -> None:
    captured = _install_top_level_harness(monkeypatch, thread_state=thread_state)
    result = _complete_pass_result()
    anchors = (object(), object())
    problems = tuple(object() for _ in range(120))
    allowances = tuple(
        result.allowances_by_invariant[name]
        for name in gate.H4_ALLOWANCE_INVARIANT_NAMES
    )
    coverage = tuple(object() for _ in range(9))
    conditions = tuple(object() for _ in range(4))
    timings = tuple(object() for _ in range(1320))
    balance, timing_summary, bootstrap, interval = object(), object(), object(), object()
    completed = gate._H4CoreData(
        result, anchors, (), problems, allowances, coverage, conditions, timings,
        balance, timing_summary, bootstrap, interval,
    )
    monkeypatch.setattr(gate, "_evaluate_complete_core", lambda *_args, **_kwargs: completed)
    returned = gate.evaluate_h4(
        h4_config, h3_coupled_bytes=b"coupled", h3_zero_bytes=b"zero",
    )
    assert returned is captured["core"]
    assert returned.result.status is gate.GateStatus.INCONCLUSIVE
    thread_invariant = returned.result.invariants[
        gate.H4_INVARIANT_NAMES.index("cpu_float64_one_thread")
    ]
    assert thread_invariant.passed is False
    assert thread_invariant.detail == "process_global_thread_state_not_restored"
    assert returned.anchors is anchors
    assert returned.problems is problems
    assert returned.allowances is allowances
    assert returned.coverage is coverage
    assert returned.condition_summaries is conditions
    assert returned.raw_timings is timings
    assert returned.primary_timed_order_balance is balance
    assert returned.timing_summary is timing_summary
    assert returned.bootstrap_interval is bootstrap
    assert returned.interval_decision is interval


def test_stable_errors_are_bounded_and_power_policy_is_exactly_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gate._stable_error(RuntimeError("a\r\nb\x00" + "x" * 600)) == (
        "builtins.RuntimeError: a\nb\ufffd" + "x" * (512 - len("builtins.RuntimeError: a\nb\ufffd"))
    )
    monkeypatch.setattr(gate.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gate, "_probe_windows_power_scheme", lambda: "balanced")
    fields_value = gate._capture_power_policy_fields()
    assert tuple(item.name for item in fields_value) == (
        "active_power_scheme", "cpu_frequency_governor", "energy_performance_preference", "low_power_mode",
    )
    assert fields_value[0] == H4PowerPolicyField("active_power_scheme", "available", "powercfg", "balanced", None)
    assert all(item.availability == "not_applicable" for item in fields_value[1:])


def test_payload_fixed_point_and_exact_top_level_artifact_authority() -> None:
    calls = []
    def builder(size: H4PayloadSizeRecord) -> dict[str, object]:
        calls.append(size.observed_bytes)
        return {"schema_version": "probe-v1", "payload_size": dataclasses.asdict(size)}
    size, payload = gate._solve_payload_size_fixed_point(builder, maximum_bytes=67108864)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert size.observed_bytes == len(encoded)
    assert 1 <= size.fixed_point_iterations <= 4
    assert calls[0] == 0
    assert size.within_limit is True


def test_anchor_restoration_and_payload_oversize_rebuild_inconclusive_status() -> None:
    anchor = _typed_allowance("h3_anchor_identity", failed=True)
    early = gate._early_anchor_failure_result(
        anchor, maximum_stopping_residual=0.0,
    )
    restored = gate._anchor_restoration_inconclusive_result(early)
    assert restored.status is gate.GateStatus.INCONCLUSIVE
    assert restored.allowances_by_invariant["h3_anchor_identity"] is anchor
    assert restored.obligations == (
        "restore H4 process-global state before closing anchor result",
    )
    assert all(
        item.detail == "not_evaluated_after_inconclusive_eligibility"
        for item in restored.invariants[1:]
    )

    oversized = gate._payload_size_inconclusive_result(
        _complete_pass_result(), observed_bytes=67_108_865,
    )
    assert oversized.status is gate.GateStatus.INCONCLUSIVE
    instrumentation = oversized.invariants[
        gate.H4_INVARIANT_NAMES.index("real_operation_instrumentation")
    ]
    assert instrumentation.passed is False
    assert instrumentation.detail == "validation_payload_exceeds_limit"
    assert oversized.obligations == (
        "reduce H4 validation payload below 67108864 bytes without dropping scalar coverage",
    )


def test_synthetic_120_problem_compact_payload_stays_below_64_mib_without_full_arrays() -> None:
    problems = []
    for index in range(120):
        horizon = (7, 15, 31)[index // 40]
        problems.append({
            "problem_index": index,
            "horizon": horizon,
            "retained_results": [{
                "native_stream": {"scalar_count": (8 * (horizon + 1)) ** 2 + 2 * 8 * (horizon + 1) + 1, "sha256": SHA_A},
                "terminal_stream": {"sha256": SHA_B},
                "selected_moments": [{"name": "initial", "mean_scalar_count": 8, "mean_sha256": SHA_A, "covariance_scalar_count": 64, "covariance_sha256": SHA_B}],
            } for _ in range(22)],
        })
    payload = {"schema_version": "vfe4-validation-h4-v1", "problems": problems}
    encoded = gate._compact_json_bytes(payload)
    assert len(encoded) < 67108864
    forbidden = {"native_result", "terminal_law", "replayed_result", "precision", "covariance"}
    assert forbidden.isdisjoint(gate._recursive_mapping_keys(payload))


def test_payload_records_reject_nonfinite_and_inconsistent_ceiling() -> None:
    with pytest.raises(ValueError):
        H4PayloadSizeRecord("utf8-compact-sorted-key-json-v1", 10, 67108864, 0, True)
    with pytest.raises(ValueError):
        H4PayloadSizeRecord("utf8-compact-sorted-key-json-v1", 67108865, 67108864, 2, True)
    with pytest.raises(ValueError):
        H4SelectedMomentSummary("initial", (0,), 1, 1, SHA_A, math.inf, 1, SHA_B, 1.0, 1.0)


def test_public_artifact_functions_are_present_and_do_not_accept_broad_mappings() -> None:
    assert callable(h4_validation_artifact)
    assert callable(h4_validation_payload)
    with pytest.raises((TypeError, ValueError)):
        h4_validation_artifact({})  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        h4_validation_payload({})  # type: ignore[arg-type]


def test_validation_payload_accepts_only_the_built_artifact_and_thaws_it_directly(
    monkeypatch: pytest.MonkeyPatch, h4_config,
) -> None:
    evaluation = gate._assemble_evaluation(
        gate._empty_core("bounded serialization fixture"), config=h4_config,
        thread_state=_valid_thread_state(), environment=_complete_environment(),
    )
    artifact = h4_validation_artifact(evaluation)

    def forbid_reconstruction(_evaluation):
        raise AssertionError("payload serialization reconstructed the artifact")

    monkeypatch.setattr(gate, "h4_validation_artifact", forbid_reconstruction)
    payload = h4_validation_payload(artifact)
    assert tuple(payload) == tuple(field.name for field in fields(H4ValidationArtifact))
    assert payload["status"] == artifact.status.value
    assert payload["coverage"] == []
    assert payload["payload_size"] == dataclasses.asdict(artifact.payload_size)
    assert len(gate._compact_json_bytes(payload)) == artifact.payload_size.observed_bytes
    with pytest.raises(ValueError, match="exact H4 artifact"):
        h4_validation_payload(evaluation)  # type: ignore[arg-type]
