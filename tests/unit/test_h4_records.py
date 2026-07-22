from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from verification import h4_records as h4_records_module
from verification.h4_records import (
    H4ArmCallSpan,
    H4ConditionStreamSummary,
    H4ConditionWitness,
    H4CoverageRecord,
    H4ExecutionTrace,
    H4GarbageCollectorRecord,
    H4InnovationConditionRecord,
    H4PosteriorConditionRecord,
    H4PostflightEventKey,
    H4PostflightScheduleSummary,
    H4PostflightTimingWitness,
    H4ProblemConditionSummary,
    H4ThreadStateRecord,
)


def test_h4_shared_record_fields_are_exact_frozen_and_slotted() -> None:
    assert "H4ProblemConditionSummary" in h4_records_module.__all__
    assert tuple(field.name for field in fields(H4PosteriorConditionRecord)) == (
        "problem_id", "problem_sha256", "source", "repetition_index", "dimension",
        "minimum_eigenvalue", "maximum_eigenvalue", "condition_number",
        "minimum_cholesky_pivot", "mean_infinity_norm", "finite", "spd", "eligible",
    )
    assert tuple(field.name for field in fields(H4InnovationConditionRecord)) == (
        "problem_id", "problem_sha256", "source", "repetition_index", "factor_id",
        "time_index", "parent_coordinate_indices", "innovation_dimension",
        "minimum_eigenvalue", "maximum_eigenvalue", "condition_number", "finite",
        "spd", "eligible",
    )
    assert tuple(field.name for field in fields(H4ConditionWitness)) == (
        "metric", "stream_index", "record",
    )
    assert tuple(field.name for field in fields(H4ConditionStreamSummary)) == (
        "name", "stream_domain", "expected_record_count", "observed_record_count",
        "record_stream_sha256", "eligible_record_count", "ineligible_record_count",
        "witnesses", "all_eligible",
    )
    assert tuple(field.name for field in fields(H4ProblemConditionSummary)) == (
        "problem_id", "problem_sha256", "name", "stream_domain",
        "expected_record_count", "observed_record_count", "record_stream_sha256",
        "eligible_record_count", "ineligible_record_count", "witnesses",
        "all_eligible",
    )
    assert tuple(field.name for field in fields(H4CoverageRecord))[-4:] == (
        "first_missing_key", "first_extra_key", "first_duplicate_key", "complete",
    )
    assert tuple(field.name for field in fields(H4ExecutionTrace))[-4:] == (
        "postflight_schedule", "garbage_collector", "warmups_count_toward_balance",
        "timed_guard_violations",
    )
    for record_type in (
        H4PosteriorConditionRecord, H4InnovationConditionRecord, H4ConditionWitness,
        H4ConditionStreamSummary, H4ProblemConditionSummary, H4CoverageRecord, H4ArmCallSpan,
        H4PostflightEventKey, H4PostflightTimingWitness,
        H4PostflightScheduleSummary, H4GarbageCollectorRecord, H4ExecutionTrace,
        H4ThreadStateRecord,
    ):
        assert "__dict__" not in record_type.__slots__


def test_h4_condition_and_coverage_records_fail_closed() -> None:
    posterior = H4PosteriorConditionRecord(
        "p", "a" * 64, "numpy_oracle", None, 4, 1.0, 2.0, 2.0, 1.0,
        0.5, True, True, True,
    )
    with pytest.raises(FrozenInstanceError):
        posterior.dimension = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        H4PosteriorConditionRecord(
            "p", "a" * 64, "numpy_oracle", None, 4, float("nan"), 2.0, 2.0,
            1.0, 0.5, True, True, True,
        )


def test_h4_problem_condition_summary_is_distinct_and_problem_bound() -> None:
    problem_id = "h4-coupled-T7-dz4-dm4-seed104729-v1"
    posterior = H4PosteriorConditionRecord(
        problem_id, "a" * 64, "numpy_oracle", None, 64, 1.0, 2.0, 2.0,
        1.0, 0.5, True, True, True,
    )
    witnesses = tuple(
        H4ConditionWitness(metric, 0, posterior)
        for metric in (
            "minimum_eigenvalue", "maximum_eigenvalue",
            "maximum_condition_number", "minimum_cholesky_pivot",
            "maximum_mean_infinity_norm",
        )
    )
    summary = H4ProblemConditionSummary(
        problem_id, "a" * 64, "oracle_posterior",
        "vfe4.h4.problem-condition-record-stream.v1", 1, 1, "b" * 64,
        1, 0, witnesses, True,
    )
    assert type(summary) is not H4ConditionStreamSummary
    with pytest.raises(ValueError, match="problem|identity"):
        replace(summary, problem_id="h4-zero_control-T7-dz4-dm4-seed104729-v1")
    with pytest.raises(ValueError, match="count"):
        replace(summary, name="terminal_posterior", expected_record_count=1)


def test_h4_problem_condition_summary_uses_horizon_local_innovation_counts() -> None:
    problem_id = "h4-zero_control-T15-dz4-dm4-seed130363-v1"
    record = H4InnovationConditionRecord(
        problem_id, "c" * 64, "numpy_oracle", None, "observation[1]", 1,
        (0, 1), 2, 1.0, 2.0, 2.0, True, True, True,
    )
    witnesses = tuple(
        H4ConditionWitness(metric, 0, record)
        for metric in (
            "minimum_eigenvalue", "maximum_eigenvalue", "maximum_condition_number",
        )
    )
    summary = H4ProblemConditionSummary(
        problem_id, "c" * 64, "oracle_innovation",
        "vfe4.h4.problem-condition-record-stream.v1", 15, 15, "d" * 64,
        15, 0, witnesses, True,
    )
    assert summary.expected_record_count == 15
    with pytest.raises(ValueError, match="identity"):
        replace(summary, problem_sha256="e" * 64)
    with pytest.raises(ValueError):
        H4CoverageRecord(
            "execution_trace", "vfe4.h4.coverage-key-stream.v1", 120, 119,
            "a" * 64, "a" * 64, 0, 0, 0, None, None, None, True,
        )


def _complete_gc_record(*, prior_enabled: bool = True) -> H4GarbageCollectorRecord:
    return H4GarbageCollectorRecord(
        "p", True, None, prior_enabled, prior_enabled, prior_enabled, None,
        None, True, True, prior_enabled, None, True,
    )


def _gc_capture_failure() -> H4GarbageCollectorRecord:
    return H4GarbageCollectorRecord(
        "p", True, "capture failed", None, None, False, None, None, None,
        False, None, None, False,
    )


def test_h4_gc_record_rejects_contradictory_phase_evidence() -> None:
    complete = _complete_gc_record()
    capture_failure = _gc_capture_failure()
    invalid = (
        (complete, {"disable_error": "disable failed"}),
        (complete, {
            "effective_state_capture_error": "inspection failed",
            "disabled_during_batch": True,
        }),
        (complete, {"disabled_during_batch": None}),
        (_complete_gc_record(prior_enabled=False), {
            "disable_error": "disable was never attempted",
        }),
        (complete, {
            "restoration_error": "restore failed",
            "restored_enabled": True,
            "restored_exact_prior_state": False,
        }),
        (capture_failure, {"disable_error": "fabricated downstream error"}),
        (capture_failure, {
            "effective_state_capture_error": "fabricated downstream error",
        }),
        (capture_failure, {"restoration_error": "fabricated downstream error"}),
    )
    for record, changes in invalid:
        with pytest.raises(ValueError, match="GC|capture|disable|effective|restor"):
            replace(record, **changes)


def _complete_thread_record() -> H4ThreadStateRecord:
    return H4ThreadStateRecord(
        None, 4, True, None, 1, True, 8, 8, True, True, 4, None, True,
    )


def _thread_capture_failure() -> H4ThreadStateRecord:
    return H4ThreadStateRecord(
        "capture failed", None, False, None, None, False, None, None, False,
        False, None, None, False,
    )


def test_h4_thread_record_rejects_contradictory_phase_evidence() -> None:
    complete = _complete_thread_record()
    capture_failure = _thread_capture_failure()
    invalid = (
        (capture_failure, {"final_inter_op_threads": 8}),
        (capture_failure, {"inter_op_unchanged": True}),
        (capture_failure, {"set_error": "fabricated downstream error"}),
        (capture_failure, {"restoration_error": "fabricated downstream error"}),
        (complete, {"set_error": "set failed", "verified_one": False}),
        (complete, {"effective_intra_op_threads": None, "verified_one": False}),
        (complete, {
            "restoration_error": "restore failed",
            "restored_intra_op_threads": 4,
            "restored_exact_prior_state": False,
        }),
        (complete, {
            "restored_intra_op_threads": None,
            "restored_exact_prior_state": False,
        }),
    )
    for record, changes in invalid:
        with pytest.raises(ValueError, match="thread|capture|set|restor|inter-op"):
            replace(record, **changes)


def test_h4_thread_partial_capture_retains_only_phase_valid_prior() -> None:
    partial = H4ThreadStateRecord(
        "inter-op getter failed", 4, False, None, None, False, None, None,
        False, False, None, None, False,
    )
    assert partial.prior_intra_op_threads == 4
    with pytest.raises(ValueError, match="capture|partial"):
        replace(partial, prior_inter_op_threads=8)
    with pytest.raises(ValueError, match="capture|mutation"):
        replace(partial, set_attempted=True)
    with pytest.raises(ValueError, match="capture|partial|order"):
        H4ThreadStateRecord(
            "intra-op getter failed", None, False, None, None, False, 8, None,
            False, False, None, None, False,
        )


def _complete_trace(garbage_collector: H4GarbageCollectorRecord) -> H4ExecutionTrace:
    def spans(
        phase: str, pair_indices: range, start_nanoseconds: int,
    ) -> tuple[tuple[H4ArmCallSpan, ...], int]:
        result = []
        cursor = start_nanoseconds
        for pair_index in pair_indices:
            order = (
                "information_then_moment"
                if pair_index % 2 == 0
                else "moment_then_information"
            )
            arms = (
                ("information", "moment")
                if order == "information_then_moment"
                else ("moment", "information")
            )
            repetition_index = None if phase == "warmup" else pair_index - 3
            for order_position, arm in enumerate(arms):
                result.append(H4ArmCallSpan(
                    "p", phase, pair_index, repetition_index, order,
                    order_position, arm, cursor, cursor + 1, 1,
                ))
                cursor += 1
        return tuple(result), cursor

    warmups, _ = spans("warmup", range(3), 1)
    timed, timed_end = spans("timed", range(3, 14), 100)
    postflight = H4PostflightScheduleSummary(
        "vfe4.h4.postflight-event-key-stream.v1", 1, 1, "a" * 64, "a" * 64,
        None, None, None, 0, None, True,
    )
    return H4ExecutionTrace(
        "p", 0, 0, 0, 0, warmups, 100, timed, timed_end, postflight,
        garbage_collector, False, (),
    )


def test_complete_trace_rejects_gc_effective_state_capture_error() -> None:
    garbage_collector = replace(
        _complete_gc_record(),
        effective_state_capture_error="inspection failed",
        disabled_during_batch=None,
    )
    with pytest.raises(ValueError, match="complete trace requires successful GC"):
        _complete_trace(garbage_collector)


def test_complete_trace_rejects_gc_observed_enabled_during_batch() -> None:
    garbage_collector = replace(
        _complete_gc_record(), disabled_during_batch=False,
    )
    with pytest.raises(ValueError, match="complete trace requires successful GC"):
        _complete_trace(garbage_collector)
