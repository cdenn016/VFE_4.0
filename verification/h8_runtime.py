"""Lossless, immutable runtime-evidence views for the H8 parent."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

from vfe4.types.h8 import H8ChildAttemptRecord, H8ChildResult
from vfe4.types.results import GateStatus
from verification.h8_budget import _require_h8_budget_issued_attempt
from verification.h8_wire import (
    H8_COLD_REPETITIONS,
    H8_PRODUCTION_SAMPLE_SEEDS,
    H8_PRODUCTION_SEEDS,
    canonical_json_bytes,
)


_PROBLEM_KEYS = (
    "problem_seed",
    "sample_noise_seed",
    "input_sha256",
    "sample_noise_sha256",
    "generative_sha256",
    "recognition_sha256",
    "local_spd_diagnostics",
    "transition_norms",
    "observation_sha256",
)
_FACTOR_RUN_KEYS = (
    "mode",
    "seed",
    "repetition",
    "input_sha256",
    "fill",
    "workspace",
    "condition_diagnostics",
    "counters",
    "reconstruction_invariants",
)
_ALLOCATION_RUN_KEYS = (
    "mode",
    "seed",
    "repetition",
    "input_sha256",
    "allocation",
    "resources",
)
_EXPECTED_RUN_ORDER = (
    *(
        ("production", seed, repetition)
        for seed in H8_PRODUCTION_SEEDS
        for repetition in range(H8_COLD_REPETITIONS)
    ),
    *(("profiler", seed, None) for seed in H8_PRODUCTION_SEEDS),
)
_PRODUCTION_RUN_COUNT = len(H8_PRODUCTION_SEEDS) * H8_COLD_REPETITIONS


def _plain_json_value(value: object, name: str) -> object:
    """Copy supported typed evidence into an ordinary JSON value."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain_json_value(
                getattr(value, field.name),
                f"{name}.{field.name}",
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} contains a non-string JSON key")
            copied[key] = _plain_json_value(item, f"{name}.{key}")
        return copied
    if type(value) in (tuple, list):
        return [
            _plain_json_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Enum):
        return _plain_json_value(value.value, name)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{name} is not a finite JSON-ready evidence value")


def _freeze_plain_json(value: object) -> object:
    """Recursively freeze a value already copied by ``_plain_json_value``."""

    if type(value) is dict:
        plain = cast(dict[str, object], value)
        return MappingProxyType(
            {key: _freeze_plain_json(item) for key, item in plain.items()}
        )
    if type(value) is list:
        return tuple(_freeze_plain_json(item) for item in cast(list[object], value))
    return value


def _immutable_mapping(
    items: tuple[tuple[str, object], ...],
    *,
    name: str,
) -> Mapping[str, object]:
    plain = {key: _plain_json_value(value, f"{name}.{key}") for key, value in items}
    frozen = _freeze_plain_json(plain)
    if not isinstance(frozen, Mapping):  # pragma: no cover - construction guard
        raise AssertionError("immutable JSON mapping construction failed")
    return cast(Mapping[str, object], frozen)


@dataclass(frozen=True, slots=True, init=False)
class H8LosslessRuntimeEvidenceViews:
    """Factory-built runtime sections backed only by exact issued attempts."""

    problems: tuple[Mapping[str, object], ...]
    factor_runs: tuple[Mapping[str, object], ...]
    allocation_runs: tuple[Mapping[str, object], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "H8LosslessRuntimeEvidenceViews is factory-only; "
            "use build_h8_lossless_runtime_evidence_views"
        )

    def __post_init__(self) -> None:
        inventories = (
            ("problems", self.problems, 3, _PROBLEM_KEYS),
            ("factor_runs", self.factor_runs, 18, _FACTOR_RUN_KEYS),
            (
                "allocation_runs",
                self.allocation_runs,
                18,
                _ALLOCATION_RUN_KEYS,
            ),
        )
        for name, records, expected_length, expected_keys in inventories:
            if (
                type(records) is not tuple
                or len(records) != expected_length
                or any(
                    type(record) is not MappingProxyType
                    or tuple(record) != expected_keys
                    for record in records
                )
            ):
                raise ValueError(f"{name} is not the exact immutable H8 inventory")


def _make_views(
    *,
    problems: tuple[Mapping[str, object], ...],
    factor_runs: tuple[Mapping[str, object], ...],
    allocation_runs: tuple[Mapping[str, object], ...],
) -> H8LosslessRuntimeEvidenceViews:
    views = object.__new__(H8LosslessRuntimeEvidenceViews)
    object.__setattr__(views, "problems", problems)
    object.__setattr__(views, "factor_runs", factor_runs)
    object.__setattr__(views, "allocation_runs", allocation_runs)
    views.__post_init__()
    return views


def _issued_attempt_inventory(
    child_attempts: object,
) -> tuple[H8ChildAttemptRecord, ...]:
    if type(child_attempts) is not tuple or not child_attempts:
        raise ValueError("child_attempts must be a nonempty exact tuple")
    attempts = tuple(
        _require_h8_budget_issued_attempt(attempt) for attempt in child_attempts
    )
    if len({id(attempt) for attempt in attempts}) != len(attempts):
        raise ValueError("child_attempts contains a duplicate issued attempt")
    identities = tuple(
        (
            attempt.request.mode,
            attempt.request.seed,
            attempt.request.repetition,
            attempt.request.control_id,
        )
        for attempt in attempts
    )
    if len(set(identities)) != len(identities):
        raise ValueError("child_attempts contains a duplicate request identity")
    if any(attempt.status is not GateStatus.PASS for attempt in attempts):
        raise ValueError("every supplied H8 attempt must have exact PASS status")
    return attempts


def _run_attempt_inventory(
    attempts: tuple[H8ChildAttemptRecord, ...],
) -> tuple[H8ChildAttemptRecord, ...]:
    run_attempts = tuple(
        attempt
        for attempt in attempts
        if attempt.request.mode in ("production", "profiler")
    )
    observed_order = tuple(
        (
            attempt.request.mode,
            attempt.request.seed,
            attempt.request.repetition,
        )
        for attempt in run_attempts
    )
    if observed_order != _EXPECTED_RUN_ORDER:
        raise ValueError(
            "run-bearing attempts must match the authoritative "
            "15-production-then-3-profiler order"
        )
    for attempt in run_attempts:
        if type(attempt.result) is not H8ChildResult or attempt.pass_evidence is None:
            raise ValueError(
                "every PASS production/profiler attempt must retain its "
                "exact result and private evidence"
            )
    return run_attempts


def _cross_bind_runs(
    *,
    run_attempts: tuple[H8ChildAttemptRecord, ...],
    production_runs: object,
    profiler_runs: object,
) -> None:
    supplied = (
        ("production_runs", production_runs, _PRODUCTION_RUN_COUNT),
        ("profiler_runs", profiler_runs, len(H8_PRODUCTION_SEEDS)),
    )
    for name, runs, expected_length in supplied:
        if (
            type(runs) is not tuple
            or len(runs) != expected_length
            or any(type(run) is not H8ChildResult for run in runs)
        ):
            raise ValueError(f"{name} must retain the exact typed run inventory")
    all_runs = cast(tuple[H8ChildResult, ...], production_runs) + cast(
        tuple[H8ChildResult, ...],
        profiler_runs,
    )
    if any(
        attempt.result is not run
        for attempt, run in zip(run_attempts, all_runs, strict=True)
    ):
        raise ValueError(
            "supplied production/profiler runs must be the exact result "
            "objects retained by their issued attempts"
        )


def _problem_views(
    run_attempts: tuple[H8ChildAttemptRecord, ...],
) -> tuple[Mapping[str, object], ...]:
    problems: list[Mapping[str, object]] = []
    for seed in H8_PRODUCTION_SEEDS:
        seed_attempts = tuple(
            attempt for attempt in run_attempts if attempt.request.seed == seed
        )
        if len(seed_attempts) != H8_COLD_REPETITIONS + 1:
            raise ValueError(f"problem consensus for seed {seed} requires six attempts")
        canonical_candidates: list[bytes] = []
        for attempt in seed_attempts:
            result = cast(H8ChildResult, attempt.result)
            evidence = attempt.pass_evidence
            if evidence is None:  # pragma: no cover - checked above
                raise AssertionError("run attempt lost private PASS evidence")
            problem = evidence.problem_evidence
            candidate = cast(
                dict[str, object],
                _plain_json_value(
                    {
                        "problem_seed": seed,
                        "sample_noise_seed": H8_PRODUCTION_SAMPLE_SEEDS[seed],
                        "input_sha256": result.input_sha256,
                        "sample_noise_sha256": evidence.sample_noise_sha256,
                        "generative_sha256": problem.generative_sha256,
                        "recognition_sha256": problem.recognition_sha256,
                        "local_spd_diagnostics": (problem.local_spd_diagnostics),
                        "transition_norms": problem.transition_norms,
                        "observation_sha256": problem.observation_sha256,
                    },
                    f"problem_consensus[{seed}]",
                ),
            )
            canonical_candidates.append(canonical_json_bytes(candidate))
        consensus_bytes = set(canonical_candidates)
        if len(consensus_bytes) != 1:
            raise ValueError(f"problem consensus disagreement for seed {seed}")
        consensus = json.loads(consensus_bytes.pop())
        if type(consensus) is not dict:  # pragma: no cover - construction guard
            raise AssertionError("problem consensus is not a JSON object")
        ordered_consensus = {key: consensus[key] for key in _PROBLEM_KEYS}
        frozen = _freeze_plain_json(ordered_consensus)
        problems.append(cast(Mapping[str, object], frozen))
    return tuple(problems)


def _factor_run_view(
    attempt: H8ChildAttemptRecord,
) -> Mapping[str, object]:
    result = cast(H8ChildResult, attempt.result)
    evidence = attempt.pass_evidence
    decisions = attempt.resource_decisions
    if (
        evidence is None
        or result.fill is None
        or result.workspace is None
        or result.counters is None
        or attempt.residuals is None
        or decisions is None
        or "residual_allowances" not in decisions
    ):
        raise ValueError("PASS run is missing lossless factor evidence")
    reconstruction = {
        "residuals": attempt.residuals,
        "residual_allowances": decisions["residual_allowances"],
    }
    return _immutable_mapping(
        (
            ("mode", result.mode),
            ("seed", result.seed),
            ("repetition", result.repetition),
            ("input_sha256", result.input_sha256),
            ("fill", result.fill),
            ("workspace", result.workspace),
            ("condition_diagnostics", evidence.condition_diagnostics),
            ("counters", result.counters),
            ("reconstruction_invariants", reconstruction),
        ),
        name=f"factor_run[{result.mode},{result.seed},{result.repetition}]",
    )


def _allocation_run_view(
    attempt: H8ChildAttemptRecord,
) -> Mapping[str, object]:
    result = cast(H8ChildResult, attempt.result)
    evidence = attempt.pass_evidence
    if evidence is None:  # pragma: no cover - checked above
        raise AssertionError("run attempt lost private PASS evidence")
    return _immutable_mapping(
        (
            ("mode", result.mode),
            ("seed", result.seed),
            ("repetition", result.repetition),
            ("input_sha256", result.input_sha256),
            ("allocation", evidence.allocation),
            ("resources", result.resources),
        ),
        name=f"allocation_run[{result.mode},{result.seed},{result.repetition}]",
    )


def build_h8_lossless_runtime_evidence_views(
    *,
    child_attempts: object,
    production_runs: object,
    profiler_runs: object,
) -> H8LosslessRuntimeEvidenceViews:
    """Build exact H8 runtime views from issued, cross-bound PASS attempts."""

    attempts = _issued_attempt_inventory(child_attempts)
    run_attempts = _run_attempt_inventory(attempts)
    _cross_bind_runs(
        run_attempts=run_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
    )
    return _make_views(
        problems=_problem_views(run_attempts),
        factor_runs=tuple(_factor_run_view(attempt) for attempt in run_attempts),
        allocation_runs=tuple(
            _allocation_run_view(attempt) for attempt in run_attempts
        ),
    )


__all__ = [
    "H8LosslessRuntimeEvidenceViews",
    "build_h8_lossless_runtime_evidence_views",
]
