"""Lossless, immutable runtime-evidence views for the H8 parent."""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

from vfe4.config.schema import H8ValidationConfig
from vfe4.inference.h8_allocation import H8AllocationPolicy
from vfe4.numerics.block_layout import (
    H8_MAX_STORAGE_SCALARS,
    BlockChainLayout,
)
from vfe4.types.h5_schema import H5_OBJECTIVE_SCHEMA_SHA256
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    H8_NEGATIVE_CONTROL_IDS,
    H8_PROFILER_API_CONTRACT_SHA256,
    H8_PROFILER_MEMORY_SOURCE_SHA256,
    H8_PROFILER_SOURCE_SHA256,
    H8_REQUIRED_OPERATIONS,
    H8_VERIFIER_PREFIX,
    BlockStorageRecord,
    H8ChildAttemptRecord,
    H8ChildResult,
    H8ControlResult,
    H8CorrectnessCell,
)
from vfe4.types.results import GateStatus
from verification.h8_budget import (
    EPS,
    MAX_ALLOWANCE_FRACTION,
    ROUNDING_MULTIPLIER,
    SOLVER_RELATIVE_BUDGET,
    _require_h8_budget_issued_attempt,
)
from verification.h8_protocol import build_h8_protocol_sha256
from verification.h8_wire import (
    H8_COLD_REPETITIONS,
    H8_MAX_PROCESS_INCREMENTAL_BYTES,
    H8_MAX_SECONDS,
    H8_MAX_TORCH_POPULATION_BYTES,
    H8_MIN_CHOLESKY_PIVOT,
    H8_PRODUCTION_SAMPLE_SEED_PAIRS,
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
_MANUSCRIPT_SHA256 = (
    "d733880d3613d32a97b7a12c93ff6c037d0abdfd9ce4810e411769997dbad03c"
)
_SECTION_KEYS = (
    "revision",
    "config",
    "interpretation",
    "protocol",
    "environment",
    "problems",
    "storage",
    "factor",
    "allocation",
    "budgets",
    "invariants",
    "artifacts",
)
_EXPECTED_ATTEMPT_ORDER = (
    *(
        ("production", seed, repetition, None)
        for seed in H8_PRODUCTION_SEEDS
        for repetition in range(H8_COLD_REPETITIONS)
    ),
    *(("profiler", seed, None, None) for seed in H8_PRODUCTION_SEEDS),
    *(
        ("negative_control", H8_PRODUCTION_SEEDS[0], None, control_id)
        for control_id in H8_NEGATIVE_CONTROL_IDS
    ),
)
_PROFILER_ACTIONS = frozenset(
    ("PREEXISTING", "CREATE", "INCREMENT_VERSION", "DESTROY")
)
_PRODUCTION_CHANNELS = frozenset(
    ("dispatch", "backend", "numpy_guard", "os_hwm")
)
_PROFILER_CHANNELS = _PRODUCTION_CHANNELS | {"profiler"}
_HEX = frozenset("0123456789abcdef")


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


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _git_object_id(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git object ID")
    return value


def _attempt_identity(
    attempt: H8ChildAttemptRecord,
) -> tuple[object, object, object, object]:
    return (
        attempt.request.mode,
        attempt.request.seed,
        attempt.request.repetition,
        attempt.request.control_id,
    )


def _exact_v4_inventories(
    *,
    child_attempts: object,
    production_runs: object,
    profiler_runs: object,
    controls: object,
    config: H8ValidationConfig,
    protocol_sha256: str,
) -> tuple[
    tuple[H8ChildAttemptRecord, ...],
    tuple[H8ChildAttemptRecord, ...],
    tuple[H8ChildResult, ...],
    tuple[H8ChildResult, ...],
    tuple[H8ControlResult, ...],
]:
    attempts = _issued_attempt_inventory(child_attempts)
    if tuple(_attempt_identity(attempt) for attempt in attempts) != (
        _EXPECTED_ATTEMPT_ORDER
    ):
        raise ValueError("child attempts must retain the exact 30-run order")
    if any(
        attempt.request.config_sha256 != config.config_sha256
        or attempt.request.protocol_sha256 != protocol_sha256
        for attempt in attempts
    ):
        raise ValueError("child attempts do not bind the exact H8 config/protocol")
    run_attempts = _run_attempt_inventory(attempts)
    _cross_bind_runs(
        run_attempts=run_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
    )
    if (
        type(production_runs) is not tuple
        or type(profiler_runs) is not tuple
        or type(controls) is not tuple
        or len(controls) != len(H8_NEGATIVE_CONTROL_IDS)
        or any(type(control) is not H8ControlResult for control in controls)
    ):
        raise ValueError("runtime inventories must retain exact typed tuples")
    typed_production = cast(tuple[H8ChildResult, ...], production_runs)
    typed_profiler = cast(tuple[H8ChildResult, ...], profiler_runs)
    typed_controls = cast(tuple[H8ControlResult, ...], controls)
    control_attempts = attempts[len(run_attempts) :]
    if any(
        attempt.result is not control
        for attempt, control in zip(control_attempts, typed_controls, strict=True)
    ):
        raise ValueError(
            "supplied controls must be the exact result objects retained by "
            "their issued attempts"
        )
    if tuple(control.control_id for control in typed_controls) != (
        H8_NEGATIVE_CONTROL_IDS
    ):
        raise ValueError("controls must retain the exact frozen order")
    return (
        attempts,
        run_attempts,
        typed_production,
        typed_profiler,
        typed_controls,
    )


def _correctness_inventory(
    value: object,
) -> tuple[H8CorrectnessCell, ...]:
    if (
        type(value) is not tuple
        or len(value) != len(H8_CORRECTNESS_CASES)
        or any(type(cell) is not H8CorrectnessCell for cell in value)
    ):
        raise ValueError("correctness must retain the exact typed H8 grid")
    cells = cast(tuple[H8CorrectnessCell, ...], value)
    if tuple(cell.cell_id for cell in cells) != tuple(
        range(1, len(H8_CORRECTNESS_CASES) + 1)
    ):
        raise ValueError("correctness cells are incomplete or reordered")
    for cell in cells:
        cell.__post_init__()
    return cells


def _identity_consensus(
    run_attempts: tuple[H8ChildAttemptRecord, ...],
) -> dict[str, dict[str, object]]:
    candidates: list[bytes] = []
    for attempt in run_attempts:
        evidence = attempt.pass_evidence
        if evidence is None:
            raise ValueError("run attempt is missing private identity evidence")
        candidates.append(
            canonical_json_bytes(
                _plain_json_value(
                    evidence.child_identities,
                    "child_identities",
                )
            )
        )
    if len(set(candidates)) != 1:
        raise ValueError(
            "hardware, affinity, thread, and BLAS identities must agree "
            "across all 18 runs"
        )
    decoded = json.loads(candidates[0])
    if type(decoded) is not dict or tuple(decoded) != (
        "affinity",
        "blas",
        "hardware",
        "thread",
    ):
        raise ValueError("identity consensus has the wrong exact inventory")
    identities: dict[str, dict[str, object]] = {}
    for kind in ("hardware", "affinity", "thread", "blas"):
        raw = decoded.get(kind)
        if type(raw) is not dict:
            raise ValueError(f"{kind} identity is not an exact mapping")
        record = cast(dict[str, object], raw)
        digest = record.get("sha256")
        body = {key: item for key, item in record.items() if key != "sha256"}
        if (
            record.get("kind") != kind
            or _sha256(digest, f"{kind}.sha256")
            != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        ):
            raise ValueError(f"{kind} identity hash does not bind its payload")
        identities[kind] = record
    return identities


def _environment_section(
    run_attempts: tuple[H8ChildAttemptRecord, ...],
    *,
    config: H8ValidationConfig,
) -> dict[str, object]:
    identities = _identity_consensus(run_attempts)
    hardware = identities["hardware"]
    affinity = identities["affinity"]
    thread = identities["thread"]
    blas = identities["blas"]
    if (
        set(hardware)
        != {
            "kind",
            "platform",
            "release",
            "system",
            "machine",
            "processor",
            "cpu_count",
            "python",
            "implementation",
            "sha256",
        }
        or set(thread)
        != {
            "kind",
            "environment",
            "torch_num_threads",
            "torch_num_interop_threads",
            "sha256",
        }
        or set(blas)
        != {
            "kind",
            "torch_version",
            "numpy_version",
            "torch_config",
            "numpy_config",
            "sha256",
        }
        or blas["torch_version"] != config.torch_version
    ):
        raise ValueError("runtime identity payloads differ from the v4 schema")
    affinity_payload = {
        key: affinity[key]
        for key in affinity
        if key not in ("kind", "sha256")
    }
    if tuple(affinity_payload) not in (
        ("adapter", "cpus"),
        ("adapter", "process_mask", "system_mask"),
    ):
        raise ValueError("affinity identity has the wrong exact schema")
    thread_environment = thread["environment"]
    if not isinstance(thread_environment, Mapping):
        raise ValueError("thread environment identity is unavailable")
    return {
        "platform": hardware["platform"],
        "platform_release": hardware["release"],
        "processor": hardware["processor"],
        "cpu_count": hardware["cpu_count"],
        "affinity": affinity_payload,
        "python_version": hardware["python"],
        "pytorch_version": blas["torch_version"],
        "numpy_version": blas["numpy_version"],
        "device": "cpu",
        "dtype": "float64",
        "grad_enabled": False,
        "intraop_threads": thread["torch_num_threads"],
        "interop_threads": thread["torch_num_interop_threads"],
        "thread_environment": dict(thread_environment),
        "blas_identity": {
            "torch_config": blas["torch_config"],
            "numpy_config": blas["numpy_config"],
        },
        "hardware_identity_sha256": hardware["sha256"],
        "affinity_sha256": affinity["sha256"],
        "thread_identity_sha256": thread["sha256"],
        "blas_identity_sha256": blas["sha256"],
    }


def _storage_section(
    runs: tuple[H8ChildResult, ...],
    *,
    config: H8ValidationConfig,
) -> tuple[dict[str, object], tuple[BlockStorageRecord, ...]]:
    storages = tuple(run.storage for run in runs)
    if any(type(storage) is not BlockStorageRecord for storage in storages):
        raise ValueError("all H8 runs must retain exact storage records")
    typed = cast(tuple[BlockStorageRecord, ...], storages)
    expected_layout = BlockChainLayout(
        horizon=config.T,
        d_z=config.d_z,
        d_m=config.d_m,
    )
    if any(
        storage.layout != expected_layout or not storage.matches_expectation
        for storage in typed
    ):
        raise ValueError("H8 storage records differ from the frozen layout")
    count_rows = {
        (
            storage.information_scalar_count,
            storage.precision_scalar_count,
            storage.factor_scalar_count,
            storage.selected_inverse_scalar_count,
        )
        for storage in typed
    }
    if len(count_rows) != 1:
        raise ValueError("H8 storage records disagree across runs")
    h_scalars, precision, factor, selected = count_rows.pop()
    return (
        {
            "h_scalars": h_scalars,
            "input_precision_scalars": precision,
            "factor_scalars": factor,
            "selected_inverse_scalars": selected,
            "category_cap_scalars": H8_MAX_STORAGE_SCALARS,
            "dense_forbidden_scalars": config.D * config.D,
            "input_within_cap": precision <= H8_MAX_STORAGE_SCALARS,
            "factor_within_cap": factor <= H8_MAX_STORAGE_SCALARS,
            "selected_within_cap": selected <= H8_MAX_STORAGE_SCALARS,
        },
        typed,
    )


def _decision_all(
    run_attempts: tuple[H8ChildAttemptRecord, ...],
    key: str,
) -> bool:
    return all(
        attempt.resource_decisions is not None
        and attempt.resource_decisions.get(key) is True
        for attempt in run_attempts
    )


def _derived_invariants(
    *,
    correctness: tuple[H8CorrectnessCell, ...],
    attempts: tuple[H8ChildAttemptRecord, ...],
    run_attempts: tuple[H8ChildAttemptRecord, ...],
    production_runs: tuple[H8ChildResult, ...],
    profiler_runs: tuple[H8ChildResult, ...],
    controls: tuple[H8ControlResult, ...],
    storages: tuple[BlockStorageRecord, ...],
    prerequisites_current_and_pass: bool,
    result_status: GateStatus,
    result_obligations: tuple[str, ...],
    config: H8ValidationConfig,
) -> dict[str, bool]:
    production_complete = tuple(
        (run.seed, run.repetition) for run in production_runs
    ) == tuple(
        (seed, repetition)
        for seed in H8_PRODUCTION_SEEDS
        for repetition in range(H8_COLD_REPETITIONS)
    )
    profiler_complete = tuple(run.seed for run in profiler_runs) == (
        H8_PRODUCTION_SEEDS
    )
    child_attempts_complete = len(attempts) == len(_EXPECTED_ATTEMPT_ORDER)
    child_attempts_exact_order = tuple(
        _attempt_identity(attempt) for attempt in attempts
    ) == _EXPECTED_ATTEMPT_ORDER
    child_attempts_cross_bound = (
        all(
            attempt.result is run
            for attempt, run in zip(
                run_attempts,
                (*production_runs, *profiler_runs),
                strict=True,
            )
        )
        and all(
            attempt.result is control
            for attempt, control in zip(
                attempts[len(run_attempts) :],
                controls,
                strict=True,
            )
        )
    )
    observability_complete = all(
        set(cast(H8ChildResult, attempt.result).allocation.observed_channels)
        == (
            _PROFILER_CHANNELS
            if attempt.request.mode == "profiler"
            else _PRODUCTION_CHANNELS
        )
        for attempt in run_attempts
    )
    every_profiler_action_joined = all(
        run.allocation.profiler_all_joined_and_liveness_reconciled is True
        and {event.action for event in run.allocation.profiler_events}
        == _PROFILER_ACTIONS
        for run in profiler_runs
    )
    required_operations_reached = all(
        attempt.operation_reachability is not None
        and tuple(attempt.operation_reachability) == H8_REQUIRED_OPERATIONS
        and all(attempt.operation_reachability.values())
        for attempt in run_attempts
    )
    forbidden_attempts_zero = (
        _decision_all(run_attempts, "forbidden_attempts_zero")
        and all(
            run.allocation.dispatch_forbidden_attempt_count == 0
            and run.allocation.backend_forbidden_attempt_count == 0
            for run in (*production_runs, *profiler_runs)
        )
    )
    offband_fill_zero = (
        _decision_all(run_attempts, "offband_fill_pass")
        and all(
            run.fill is not None
            and run.fill.observed_offband_blocks == 0
            and run.fill.duplicated_upper_blocks == 0
            for run in (*production_runs, *profiler_runs)
        )
    )
    pivot_margin_pass = (
        _decision_all(run_attempts, "pivot_margin_pass")
        and all(
            attempt.pass_evidence is not None
            and attempt.pass_evidence.condition_diagnostics.global_min_pivot
            >= H8_MIN_CHOLESKY_PIVOT
            for attempt in run_attempts
        )
    )
    rhs_width_pass = (
        _decision_all(run_attempts, "rhs_width_pass")
        and all(
            run.workspace is not None
            and run.workspace.maximum_rhs_width <= config.max_rhs_width
            and run.counters is not None
            and run.counters.maximum_rhs_width <= config.max_rhs_width
            for run in (*production_runs, *profiler_runs)
        )
    )
    sample_width_pass = (
        _decision_all(run_attempts, "sample_width_pass")
        and all(
            run.counters is not None
            and run.counters.maximum_sample_rhs_width <= config.sample_width
            for run in (*production_runs, *profiler_runs)
        )
    )
    time_pass = (
        _decision_all(run_attempts, "time_pass")
        and all(
            attempt.parent_elapsed_ns <= int(H8_MAX_SECONDS * 1e9)
            and cast(H8ChildResult, attempt.result).resources.child_elapsed_ns
            <= int(H8_MAX_SECONDS * 1e9)
            for attempt in run_attempts
        )
    )
    process_memory_pass = (
        _decision_all(run_attempts, "process_memory_pass")
        and all(
            run.resources.conservative_incremental_hwm_bytes
            <= H8_MAX_PROCESS_INCREMENTAL_BYTES
            for run in (*production_runs, *profiler_runs)
        )
    )
    torch_memory_pass = (
        _decision_all(run_attempts, "torch_memory_pass")
        and all(
            run.allocation.torch_population_peak_bytes
            <= H8_MAX_TORCH_POPULATION_BYTES
            for run in (*production_runs, *profiler_runs)
        )
    )
    residuals_pass = (
        _decision_all(run_attempts, "residual_allowances_pass")
        and all(
            attempt.resource_decisions is not None
            and isinstance(
                attempt.resource_decisions.get("residual_allowances"),
                Mapping,
            )
            and all(
                isinstance(group, Mapping) and group.get("passed") is True
                for group in cast(
                    Mapping[str, object],
                    attempt.resource_decisions["residual_allowances"],
                ).values()
            )
            for attempt in run_attempts
        )
    )
    correctness_complete = tuple(cell.cell_id for cell in correctness) == (
        tuple(range(1, len(H8_CORRECTNESS_CASES) + 1))
    )
    controls_complete = (
        tuple(control.control_id for control in controls)
        == H8_NEGATIVE_CONTROL_IDS
        and all(control.status is GateStatus.PASS for control in controls)
    )
    exact_inventory_complete = (
        correctness_complete
        and child_attempts_complete
        and child_attempts_exact_order
        and child_attempts_cross_bound
        and production_complete
        and profiler_complete
        and controls_complete
    )
    retained_statuses = (
        *(cell.status for cell in correctness),
        *(attempt.status for attempt in attempts),
        *(control.status for control in controls),
        *(
            invariant.status
            for run in (*production_runs, *profiler_runs)
            for invariant in run.invariants
        ),
    )
    expected_status = (
        GateStatus.FAIL
        if GateStatus.FAIL in retained_statuses
        else GateStatus.INCONCLUSIVE
        if (
            result_obligations
            or not exact_inventory_complete
            or not retained_statuses
            or GateStatus.INCONCLUSIVE in retained_statuses
        )
        else GateStatus.PASS
    )
    invariants = {
        "prerequisites_current_and_pass": prerequisites_current_and_pass,
        "interpretation_hash_current": (
            config.interpretation_sha256 == H8_INTERPRETATION_SHA256
        ),
        "correctness_cells_complete": correctness_complete,
        "correctness_pass": correctness_complete
        and all(cell.status is GateStatus.PASS for cell in correctness),
        "controls_complete": controls_complete,
        "child_attempts_complete": child_attempts_complete,
        "child_attempts_exact_order": child_attempts_exact_order,
        "child_attempts_cross_bound": child_attempts_cross_bound,
        "observability_complete": observability_complete,
        "every_profiler_action_joined_and_liveness_reconciled": (
            every_profiler_action_joined
        ),
        "production_runs_complete": production_complete,
        "profiler_runs_complete": profiler_complete,
        "required_operations_reached": required_operations_reached,
        "storage_pass": _decision_all(run_attempts, "storage_pass")
        and all(storage.matches_expectation for storage in storages),
        "forbidden_attempts_zero": forbidden_attempts_zero,
        "offband_fill_zero": offband_fill_zero,
        "pivot_margin_pass": pivot_margin_pass,
        "rhs_width_pass": rhs_width_pass,
        "sample_width_pass": sample_width_pass,
        "time_pass": time_pass,
        "process_memory_pass": process_memory_pass,
        "torch_memory_pass": torch_memory_pass,
        "finite_pass": _decision_all(run_attempts, "finite_pass"),
        "residuals_pass": residuals_pass,
        "witnessed_failure_dominance_applied": result_status is expected_status,
    }
    return {**invariants, "all_pass": all(invariants.values())}


def build_h8_v4_runtime_sections(
    *,
    config: H8ValidationConfig,
    candidate_head: str,
    candidate_dirty_digest: str,
    candidate_junit_sha256: str,
    current_refs_registry_sha256: str,
    dependency_closure_sha256: str,
    preregistration_sha256: str,
    prerequisites_current_and_pass: bool,
    correctness: object,
    child_attempts: object,
    production_runs: object,
    profiler_runs: object,
    controls: object,
    result_status: GateStatus,
    result_obligations: tuple[str, ...],
) -> Mapping[str, object]:
    """Build all v4 runtime sections solely from exact typed H8 evidence."""

    if type(config) is not H8ValidationConfig:
        raise ValueError("config must be an exact H8ValidationConfig")
    config.__post_init__()
    if config != H8ValidationConfig.create():
        raise ValueError("H8 validation configuration is stale")
    _git_object_id(candidate_head, "candidate_head")
    for name, value in (
        ("candidate_dirty_digest", candidate_dirty_digest),
        ("candidate_junit_sha256", candidate_junit_sha256),
        ("current_refs_registry_sha256", current_refs_registry_sha256),
        ("dependency_closure_sha256", dependency_closure_sha256),
        ("preregistration_sha256", preregistration_sha256),
    ):
        _sha256(value, name)
    if type(prerequisites_current_and_pass) is not bool:
        raise ValueError("prerequisites_current_and_pass must be a bool")
    if type(result_status) is not GateStatus:
        raise ValueError("result_status must be an exact GateStatus")
    if (
        type(result_obligations) is not tuple
        or any(
            type(obligation) is not str or not obligation
            for obligation in result_obligations
        )
        or len(set(result_obligations)) != len(result_obligations)
    ):
        raise ValueError("result_obligations must be unique nonempty strings")

    protocol_sha256 = build_h8_protocol_sha256(config)
    (
        attempts,
        run_attempts,
        typed_production,
        typed_profiler,
        typed_controls,
    ) = _exact_v4_inventories(
        child_attempts=child_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
        controls=controls,
        config=config,
        protocol_sha256=protocol_sha256,
    )
    typed_correctness = _correctness_inventory(correctness)
    views = build_h8_lossless_runtime_evidence_views(
        child_attempts=attempts,
        production_runs=typed_production,
        profiler_runs=typed_profiler,
    )
    all_runs = (*typed_production, *typed_profiler)
    storage, storages = _storage_section(all_runs, config=config)
    layout = BlockChainLayout(
        horizon=config.T,
        d_z=config.d_z,
        d_m=config.d_m,
    )
    allocation_policy = H8AllocationPolicy(layout)
    no_forbidden_attempts = (
        _decision_all(run_attempts, "forbidden_attempts_zero")
        and all(
            run.allocation.dispatch_forbidden_attempt_count == 0
            and run.allocation.backend_forbidden_attempt_count == 0
            for run in all_runs
        )
    )
    all_observable = all(
        set(run.allocation.observed_channels)
        == (
            _PROFILER_CHANNELS
            if run.mode == "profiler"
            else _PRODUCTION_CHANNELS
        )
        for run in all_runs
    )
    invariants = _derived_invariants(
        correctness=typed_correctness,
        attempts=attempts,
        run_attempts=run_attempts,
        production_runs=typed_production,
        profiler_runs=typed_profiler,
        controls=typed_controls,
        storages=storages,
        prerequisites_current_and_pass=prerequisites_current_and_pass,
        result_status=result_status,
        result_obligations=result_obligations,
        config=config,
    )
    sections = {
        "revision": {
            "git_head": candidate_head,
            "dirty_digest": candidate_dirty_digest,
            "dependency_closure_sha256": dependency_closure_sha256,
            "manuscript_sha256": _MANUSCRIPT_SHA256,
            "preregistration_sha256": preregistration_sha256,
            "h7_plan_sha256": H8_H7_PLAN_SHA256,
        },
        "config": {
            "config_sha256": config.config_sha256,
            "objective_schema_sha256": H5_OBJECTIVE_SCHEMA_SHA256,
            "protocol_sha256": protocol_sha256,
            "canonical_json_sha256": hashlib.sha256(
                config.canonical_json.encode("utf-8")
            ).hexdigest(),
            "selected_operation": config.operation,
            "ordered_gates": H8_VERIFIER_PREFIX,
            "current_refs_registry_sha256": current_refs_registry_sha256,
            "candidate_junit_sha256": candidate_junit_sha256,
        },
        "interpretation": {
            "interpretation_sha256": config.interpretation_sha256,
            "choice_kind": config.choice_kind,
            "K_semantics": config.k_semantics,
            "T": config.T,
            "N": config.N,
            "K": config.K,
            "d_z": config.d_z,
            "d_m": config.d_m,
            "b": config.b,
            "D": config.D,
            "V": config.V,
            "coordinate_order": config.coordinate_order,
            "state_parent_sets": "t0:none;t>=1:{t-1}",
            "model_parent_sets": "t0:none;t>=1:{t-1}",
            "state_source_support": "singleton_previous_slice",
            "model_source_support": "singleton_previous_slice",
            "ambiguity_policy": (
                "changed_or_clarified_K_invalidates_and_yields_INCONCLUSIVE"
            ),
        },
        "protocol": {
            "generator_schema": config.generator_schema,
            "generator_draw_schema_sha256": (
                config.problem_draw_schema_sha256
            ),
            "sample_schema": config.sample_schema,
            "factor_schema": config.factor_schema,
            "selected_inverse_schema": config.selected_inverse_schema,
            "condition_estimator_schema": config.condition_estimator_schema,
            "allocation_schema": config.allocation_schema,
            "torch_version": config.torch_version,
            "profiler_source_hashes": {
                "memory_profile": H8_PROFILER_MEMORY_SOURCE_SHA256,
                "profiler": H8_PROFILER_SOURCE_SHA256,
            },
            "profiler_api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
            "profiler_raw_event_schema": config.profiler_raw_event_schema,
            "child_schema": config.child_schema,
            "production_seed_order": config.seeds,
            "production_sample_seed_map": H8_PRODUCTION_SAMPLE_SEED_PAIRS,
            "repetition_order": tuple(range(config.cold_repetitions)),
            "correctness_seed_table": config.correctness_seed_table,
            "required_operations": H8_REQUIRED_OPERATIONS,
            "negative_control_order": H8_NEGATIVE_CONTROL_IDS,
        },
        "environment": _environment_section(
            run_attempts,
            config=config,
        ),
        "problems": views.problems,
        "storage": storage,
        "factor": {
            "schema_version": "h8-factor-evidence-v1",
            "algorithm": "block_tridiagonal_cholesky_local_recursion",
            "pattern": "symmetric_block_tridiagonal_diag_lower_only",
            "runs": views.factor_runs,
        },
        "allocation": {
            "schema_version": "h8-allocation-evidence-v1",
            "whitelist": allocation_policy.descriptor(),
            "runs": views.allocation_runs,
            "tracemalloc_supplementary": None,
            "all_observable": all_observable,
            "no_forbidden_attempts": no_forbidden_attempts,
        },
        "budgets": {
            "eps": EPS,
            "rounding_multiplier": ROUNDING_MULTIPLIER,
            "solver_relative_budget": SOLVER_RELATIVE_BUDGET,
            "maximum_allowance_scale_fraction": MAX_ALLOWANCE_FRACTION,
            "min_cholesky_pivot": H8_MIN_CHOLESKY_PIVOT,
            "max_seconds": H8_MAX_SECONDS,
            "max_process_incremental_bytes": (
                H8_MAX_PROCESS_INCREMENTAL_BYTES
            ),
            "max_torch_population_bytes": H8_MAX_TORCH_POPULATION_BYTES,
            "max_storage_scalars": H8_MAX_STORAGE_SCALARS,
            "boundary_policy": (
                "residual_le_allowance_and_fraction_lt_limit"
            ),
        },
        "invariants": invariants,
        "artifacts": {
            "config_path": "config.json",
            "provenance_path": "provenance.json",
            "environment_path": "environment.json",
            "h7_reference_path": "references/h7.json",
            "h6_prediction_reference_path": (
                "references/h6_prediction.json"
            ),
            "validation_path": "validation/h8.json",
            "manifest_path": "manifest.sha256",
        },
    }
    if tuple(sections) != _SECTION_KEYS:
        raise RuntimeError("internal H8 v4 section inventory drifted")
    frozen = _freeze_plain_json(
        _plain_json_value(sections, "h8_v4_runtime_sections")
    )
    if not isinstance(frozen, Mapping):  # pragma: no cover - construction guard
        raise AssertionError("H8 v4 runtime sections did not freeze as a mapping")
    return cast(Mapping[str, object], frozen)


__all__ = [
    "H8LosslessRuntimeEvidenceViews",
    "build_h8_lossless_runtime_evidence_views",
    "build_h8_v4_runtime_sections",
]
