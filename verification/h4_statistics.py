"""Paired seed-level timing summaries and preregistered H4 bootstrap."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vfe4.types.h4 import (
    H4_PRIMARY_TIMED_AB_TOTAL,
    H4_PRIMARY_TIMED_BALANCE,
    H4_PRIMARY_TIMED_BA_TOTAL,
    H4_PROBLEM_SEEDS,
    H4IntervalDecision,
    H4TimingRecord,
    classify_h4_interval,
)

from .h4_records import H4ExecutionTrace

_BOOTSTRAP_HEADER = b'{"dtype":"<i8","endpoint":false,"high":20,"low":0,"seed":20260721,"shape":[100000,20]}'
_BOOTSTRAP_DIGEST = "a254e18bccc519a719e9f4b409f45cc9ae4a2a321903531cd8fd73433687cd14"


def _finite_positive(value: object, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")


@dataclass(frozen=True, slots=True)
class H4SeedTimingSummary:
    seed_index: int
    seed: int
    information_median_nanoseconds: int
    moment_median_nanoseconds: int
    ratio: float
    log_ratio: float

    def __post_init__(self) -> None:
        if type(self.seed_index) is not int or self.seed_index not in range(20) or type(self.seed) is not int or self.seed != H4_PROBLEM_SEEDS[self.seed_index]:
            raise ValueError("seed summary identity is frozen")
        if type(self.information_median_nanoseconds) is not int or self.information_median_nanoseconds <= 0 or type(self.moment_median_nanoseconds) is not int or self.moment_median_nanoseconds <= 0:
            raise ValueError("seed medians must be positive integers")
        expected_ratio = self.information_median_nanoseconds / self.moment_median_nanoseconds
        if type(self.ratio) is not float or self.ratio != expected_ratio or type(self.log_ratio) is not float or self.log_ratio != math.log(expected_ratio):
            raise ValueError("seed ratio fields are inconsistent")


@dataclass(frozen=True, slots=True)
class H4TimingSummary:
    primary_problem_ids: tuple[str, ...]
    seed_summaries: tuple[H4SeedTimingSummary, ...]
    geometric_mean_ratio: float

    def __post_init__(self) -> None:
        if type(self.primary_problem_ids) is not tuple or len(self.primary_problem_ids) != 20 or len(set(self.primary_problem_ids)) != 20 or not all(type(item) is str and item for item in self.primary_problem_ids):
            raise ValueError("timing summary requires 20 unique primary problem IDs")
        if type(self.seed_summaries) is not tuple or tuple(item.seed_index for item in self.seed_summaries) != tuple(range(20)) or not all(type(item) is H4SeedTimingSummary for item in self.seed_summaries):
            raise ValueError("timing summary requires all seed summaries in order")
        expected = math.exp(math.fsum(item.log_ratio for item in self.seed_summaries) / 20)
        if type(self.geometric_mean_ratio) is not float or self.geometric_mean_ratio != expected:
            raise ValueError("geometric mean ratio is inconsistent")


@dataclass(frozen=True, slots=True)
class H4PrimaryTimedOrderBalance:
    expected_rows: tuple[tuple[int, int, int], ...]
    observed_rows: tuple[tuple[int, int, int], ...]
    expected_pattern_counts: tuple[tuple[int, int, int], ...]
    observed_pattern_counts: tuple[tuple[int, int, int], ...]
    expected_ab_total: int
    expected_ba_total: int
    observed_ab_total: int
    observed_ba_total: int
    warmup_contribution: Literal[0]
    matches: bool

    def __post_init__(self) -> None:
        if self.expected_rows != H4_PRIMARY_TIMED_BALANCE or self.expected_pattern_counts != ((5, 6, 10), (6, 5, 10)) or (self.expected_ab_total, self.expected_ba_total) != (110, 110) or self.warmup_contribution != 0:
            raise ValueError("primary timed-order expectation is frozen")
        for rows, name in ((self.observed_rows, "observed_rows"), (self.observed_pattern_counts, "observed_pattern_counts")):
            if type(rows) is not tuple or not all(type(row) is tuple and len(row) == 3 and all(type(item) is int for item in row) for row in rows):
                raise ValueError(f"{name} must be an immutable integer-row tuple")
        if any(type(value) is not int or value < 0 for value in (self.observed_ab_total, self.observed_ba_total)):
            raise ValueError("observed timed totals must be nonnegative integers")
        expected_match = (
            self.observed_rows == self.expected_rows
            and self.observed_pattern_counts == self.expected_pattern_counts
            and self.observed_ab_total == self.expected_ab_total
            and self.observed_ba_total == self.expected_ba_total
        )
        if type(self.matches) is not bool or self.matches != expected_match:
            raise ValueError("timed-order match flag is inconsistent")


@dataclass(frozen=True, slots=True)
class H4BootstrapInterval:
    bootstrap_seed: int
    replicate_count: int
    inferential_seed_indices: tuple[int, ...]
    resample_index_shape: tuple[int, int]
    resample_index_dtype: Literal["<i8"]
    resample_index_sha256: str
    statistic: Literal["mean_log_seed_ratio"]
    percentile_method: Literal["linear"]
    percentile_space: Literal["log_then_exp"]
    estimate: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            type(self.bootstrap_seed) is not int or self.bootstrap_seed != 20260721
            or type(self.replicate_count) is not int or self.replicate_count != 100000
            or self.inferential_seed_indices != tuple(range(20))
            or self.resample_index_shape != (100000, 20)
            or self.resample_index_dtype != "<i8"
            or self.resample_index_sha256 != _BOOTSTRAP_DIGEST
            or self.statistic != "mean_log_seed_ratio"
            or self.percentile_method != "linear"
            or self.percentile_space != "log_then_exp"
        ):
            raise ValueError("H4 bootstrap identity is frozen")
        for name in ("estimate", "lower", "upper"):
            _finite_positive(getattr(self, name), name)
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("bootstrap estimate must lie inside its ordered interval")


def summarize_seed_ratios(records: tuple[H4TimingRecord, ...]) -> H4TimingSummary:
    if type(records) is not tuple or len(records) != 220 or not all(type(item) is H4TimingRecord for item in records):
        raise ValueError("H4 primary summary requires exactly 220 exact timing records")
    by_seed: list[list[H4TimingRecord]] = [[] for _ in range(20)]
    for record in records:
        if record.horizon_index != 2 or record.horizon != 31 or record.kind_index != 0 or record.kind != "coupled":
            raise ValueError("H4 primary summary rejects nonprimary timing rows")
        by_seed[record.seed_index].append(record)
    summaries: list[H4SeedTimingSummary] = []
    problem_ids: list[str] = []
    for seed_index, rows in enumerate(by_seed):
        if len(rows) != 11 or tuple(sorted(item.repetition_index for item in rows)) != tuple(range(11)) or len({(item.repetition_index, item.pair_index) for item in rows}) != 11:
            raise ValueError("each primary seed requires exactly 11 distinct repetitions")
        rows.sort(key=lambda item: item.repetition_index)
        if len({item.problem_id for item in rows}) != 1:
            raise ValueError("one primary seed cannot alias multiple problem IDs")
        information_values = tuple(item.information_nanoseconds for item in rows)
        moment_values = tuple(item.moment_nanoseconds for item in rows)
        information_median = statistics.median(information_values)
        moment_median = statistics.median(moment_values)
        if type(information_median) is not int or type(moment_median) is not int:
            raise ValueError("eleven-value timing medians must remain exact integers")
        ratio = information_median / moment_median
        summaries.append(H4SeedTimingSummary(
            seed_index, H4_PROBLEM_SEEDS[seed_index], information_median,
            moment_median, ratio, math.log(ratio),
        ))
        problem_ids.append(rows[0].problem_id)
    summary_tuple = tuple(summaries)
    geometric = math.exp(math.fsum(item.log_ratio for item in summary_tuple) / 20)
    return H4TimingSummary(tuple(problem_ids), summary_tuple, geometric)


def summarize_primary_timed_order(
    records: tuple[H4TimingRecord, ...],
    traces: tuple[H4ExecutionTrace, ...],
) -> H4PrimaryTimedOrderBalance:
    summary = summarize_seed_ratios(records)
    if type(traces) is not tuple or not all(type(item) is H4ExecutionTrace for item in traces):
        raise ValueError("timed-order balance requires exact execution traces")
    primary_traces = [item for item in traces if item.horizon_index == 2 and item.kind_index == 0]
    if len(primary_traces) != 20 or tuple(sorted(item.seed_index for item in primary_traces)) != tuple(range(20)):
        raise ValueError("timed-order balance requires all 20 primary traces")
    traces_by_seed = {item.seed_index: item for item in primary_traces}
    observed_rows: list[tuple[int, int, int]] = []
    for seed_summary in summary.seed_summaries:
        trace = traces_by_seed[seed_summary.seed_index]
        rows = [item for item in records if item.seed_index == seed_summary.seed_index]
        expected_problem_id = summary.primary_problem_ids[seed_summary.seed_index]
        if (
            trace.problem_id != expected_problem_id
            or any(row.problem_id != expected_problem_id for row in rows)
            or any(span.problem_id != expected_problem_id for span in trace.timed_spans)
        ):
            raise ValueError("timing rows, trace, and spans must share the primary problem ID")
        spans = {(item.repetition_index, item.arm): item for item in trace.timed_spans}
        ab = 0
        for row in rows:
            if spans[(row.repetition_index, "information")].duration_nanoseconds != row.information_nanoseconds or spans[(row.repetition_index, "moment")].duration_nanoseconds != row.moment_nanoseconds:
                raise ValueError("timing record does not equal its matching arm span")
            ab += row.order == "information_then_moment"
        observed_rows.append((seed_summary.seed, ab, 11 - ab))
    patterns = tuple(
        (ab, ba, sum((row[1], row[2]) == (ab, ba) for row in observed_rows))
        for ab, ba in ((5, 6), (6, 5))
    )
    ab_total = sum(row[1] for row in observed_rows)
    ba_total = sum(row[2] for row in observed_rows)
    observed_tuple = tuple(observed_rows)
    return H4PrimaryTimedOrderBalance(
        H4_PRIMARY_TIMED_BALANCE, observed_tuple, ((5, 6, 10), (6, 5, 10)),
        patterns, H4_PRIMARY_TIMED_AB_TOTAL, H4_PRIMARY_TIMED_BA_TOTAL,
        ab_total, ba_total, 0,
        observed_tuple == H4_PRIMARY_TIMED_BALANCE
        and patterns == ((5, 6, 10), (6, 5, 10))
        and (ab_total, ba_total) == (110, 110),
    )


def paired_log_bootstrap_interval(summary: H4TimingSummary) -> H4BootstrapInterval:
    if type(summary) is not H4TimingSummary:
        raise ValueError("paired bootstrap requires the exact timing summary")
    rng = np.random.Generator(np.random.PCG64(20260721))
    indices = rng.integers(0, 20, size=(100000, 20), endpoint=False, dtype=np.int64)
    bytes_indices = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256(
        b"vfe4.h4.bootstrap-indices.v1\x00" + _BOOTSTRAP_HEADER + b"\x00"
        + bytes_indices.tobytes(order="C")
    ).hexdigest()
    if digest != _BOOTSTRAP_DIGEST:
        raise RuntimeError("H4 bootstrap resample bytes do not match preregistration")
    seed_log_ratios = np.asarray(
        tuple(item.log_ratio for item in summary.seed_summaries), dtype=np.float64,
    )
    replicate_mean_log_ratios = np.mean(
        seed_log_ratios[indices], axis=1, dtype=np.float64,
    )
    log_lower, log_upper = np.percentile(
        replicate_mean_log_ratios, (2.5, 97.5), method="linear",
    )
    lower, upper = math.exp(float(log_lower)), math.exp(float(log_upper))
    return H4BootstrapInterval(
        20260721, 100000, tuple(range(20)), (100000, 20), "<i8", digest,
        "mean_log_seed_ratio", "linear", "log_then_exp",
        summary.geometric_mean_ratio, lower, upper,
    )


def decide_h4_interval(interval: H4BootstrapInterval) -> H4IntervalDecision:
    if type(interval) is not H4BootstrapInterval:
        raise ValueError("decide_h4_interval requires an exact bootstrap interval")
    return classify_h4_interval(interval.lower, interval.upper)


__all__ = [
    "H4BootstrapInterval", "H4PrimaryTimedOrderBalance", "H4SeedTimingSummary",
    "H4TimingSummary", "decide_h4_interval", "paired_log_bootstrap_interval",
    "summarize_primary_timed_order", "summarize_seed_ratios",
]
