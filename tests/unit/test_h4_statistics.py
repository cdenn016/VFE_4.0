from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from vfe4.types.h4 import H4_PROBLEM_SEEDS, H4TimingRecord
from vfe4.types.results import GateStatus
from verification.h4_records import (
    H4ArmCallSpan,
    H4ExecutionTrace,
    H4GarbageCollectorRecord,
    H4PostflightScheduleSummary,
)
from verification.h4_statistics import (
    H4BootstrapInterval,
    decide_h4_interval,
    paired_log_bootstrap_interval,
    summarize_primary_timed_order,
    summarize_seed_ratios,
)


def _primary_records() -> tuple[H4TimingRecord, ...]:
    rows = []
    for seed_index, seed in enumerate(H4_PROBLEM_SEEDS):
        for repetition in range(11):
            pair_index = 3 + repetition
            order = (
                "information_then_moment"
                if (2 + seed_index + pair_index) % 2 == 0
                else "moment_then_information"
            )
            rows.append(H4TimingRecord(
                f"h4-coupled-T31-dz4-dm4-seed{seed}-v1",
                (2 * 20 + seed_index) * 2, 2, seed_index, 0, seed, "coupled", 31,
                repetition, pair_index, order, 80 + repetition, 100 + repetition,
            ))
    return tuple(rows)


def _pair_order(seed_index: int, pair_index: int) -> str:
    return (
        "information_then_moment"
        if (2 + seed_index + pair_index) % 2 == 0
        else "moment_then_information"
    )


def _span_pair(
    problem_id: str,
    seed_index: int,
    pair_index: int,
    *,
    phase: str,
    repetition_index: int | None,
    start_nanoseconds: int,
) -> tuple[tuple[H4ArmCallSpan, H4ArmCallSpan], int]:
    order = _pair_order(seed_index, pair_index)
    arms = (
        ("information", "moment")
        if order == "information_then_moment"
        else ("moment", "information")
    )
    cursor = start_nanoseconds
    spans = []
    for position, arm in enumerate(arms):
        duration = (
            1 if phase == "warmup"
            else (80 + repetition_index if arm == "information" else 100 + repetition_index)
        )
        assert repetition_index is None or type(duration) is int
        spans.append(H4ArmCallSpan(
            problem_id, phase, pair_index, repetition_index, order, position, arm,
            cursor, cursor + duration, duration,
        ))
        cursor += duration
    return (spans[0], spans[1]), cursor


def _primary_trace(seed_index: int, *, problem_id: str | None = None) -> H4ExecutionTrace:
    seed = H4_PROBLEM_SEEDS[seed_index]
    identity = problem_id or f"h4-coupled-T31-dz4-dm4-seed{seed}-v1"
    warmups = []
    cursor = 1
    for pair_index in range(3):
        pair, cursor = _span_pair(
            identity, seed_index, pair_index, phase="warmup",
            repetition_index=None, start_nanoseconds=cursor,
        )
        warmups.extend(pair)
    timed = []
    timed_start = 1_000
    cursor = timed_start
    for repetition_index in range(11):
        pair, cursor = _span_pair(
            identity, seed_index, 3 + repetition_index, phase="timed",
            repetition_index=repetition_index, start_nanoseconds=cursor,
        )
        timed.extend(pair)
    postflight = H4PostflightScheduleSummary(
        "vfe4.h4.postflight-event-key-stream.v1", 1, 1, "a" * 64, "a" * 64,
        None, None, None, 0, None, True,
    )
    garbage_collector = H4GarbageCollectorRecord(
        identity, True, None, True, True, True, None, None, True, True, True,
        None, True,
    )
    return H4ExecutionTrace(
        identity, (2 * 20 + seed_index) * 2, 2, seed_index, 0,
        tuple(warmups), timed_start, tuple(timed), cursor, postflight,
        garbage_collector, False, (),
    )


def _bootstrap_interval(
    lower: float, upper: float, estimate: float,
) -> H4BootstrapInterval:
    return H4BootstrapInterval(
        20260721, 100000, tuple(range(20)), (100000, 20), "<i8",
        "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14",
        "mean_log_seed_ratio", "linear", "log_then_exp", estimate, lower, upper,
    )


def test_h4_seed_summary_and_bootstrap_are_exact_and_paired() -> None:
    summary = summarize_seed_ratios(_primary_records())
    assert len(summary.seed_summaries) == 20
    assert summary.seed_summaries[0].information_median_nanoseconds == 85
    assert summary.seed_summaries[0].moment_median_nanoseconds == 105
    assert math.isclose(summary.geometric_mean_ratio, 85 / 105)
    interval = paired_log_bootstrap_interval(summary)
    assert interval.resample_index_shape == (100000, 20)
    assert interval.resample_index_dtype == "<i8"
    assert interval.resample_index_sha256 == "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"
    assert interval.lower == interval.upper == interval.estimate


def test_h4_primary_timed_order_requires_matching_problem_identity() -> None:
    records = _primary_records()
    traces = tuple(_primary_trace(seed_index) for seed_index in range(20))
    balance = summarize_primary_timed_order(records, traces)
    assert balance.matches
    mismatched = (
        _primary_trace(0, problem_id="h4-coupled-T31-dz4-dm4-seed104729-alias"),
        *traces[1:],
    )
    with pytest.raises(ValueError, match="problem ID"):
        summarize_primary_timed_order(records, mismatched)


@pytest.mark.parametrize(
    ("interval", "classification", "status"),
    (
        (_bootstrap_interval(0.70, 0.80, 0.75), "support", GateStatus.PASS),
        (_bootstrap_interval(0.80, 0.90, 0.85), "no_support", GateStatus.FAIL),
        (_bootstrap_interval(0.70, 0.90, 0.80), "crossing", GateStatus.INCONCLUSIVE),
        (_bootstrap_interval(0.80, 0.80, 0.80), "boundary", GateStatus.INCONCLUSIVE),
    ),
)
def test_h4_interval_decision_delegates_all_boundaries(
    interval: H4BootstrapInterval,
    classification: str,
    status: GateStatus,
) -> None:
    decision = decide_h4_interval(interval)
    assert decision.classification == classification
    assert decision.status_if_other_invariants_eligible is status
