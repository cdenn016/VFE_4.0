"""Fail-closed assembly for the bounded H8 sparse-scale gate.

This module owns evidence assembly only.  It deliberately does not import or
launch the H8 child process, profiler, correctness grid, or predecessor gates.
Runtime orchestration supplies already-decoded typed records in a later layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import cast

from vfe4.artifacts.atomic import canonical_json_bytes
from vfe4.artifacts.h6 import (
    CandidateArtifactReference,
    reopen_bounded_prefix_certificate_set,
)
from vfe4.artifacts.provenance import (
    H8_PROVENANCE_KEYS,
    source_candidate_sha256,
)
from vfe4.types.h8 import (
    H8_CORRECTNESS_CASES,
    H8_H7_PLAN_SHA256,
    H8_INTERPRETATION_SHA256,
    H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION,
    H8_MAX_PROCESS_INCREMENTAL_BYTES,
    H8_MAX_SECONDS,
    H8_MAX_TORCH_POPULATION_BYTES,
    H8_MIN_CHOLESKY_PIVOT,
    H8_NEGATIVE_CONTROL_IDS,
    H8_NONCLAIMS,
    H8_PROBLEM_DRAW_SCHEMA_SHA256,
    H8_PRODUCTION_SAMPLE_SEED_PAIRS,
    H8_PRODUCTION_SEEDS,
    H8_PROFILER_API_CONTRACT_SHA256,
    H8_PROFILER_MEMORY_SOURCE_SHA256,
    H8_PROFILER_SOURCE_SHA256,
    H8_REQUIRED_OPERATIONS,
    H8_VERIFIER_PREFIX,
    CurrentH8PrerequisiteRefs,
    H8ChildAttemptRecord,
    H8ChildResult,
    H8ControlResult,
    H8CorrectnessCell,
    H8GateEvaluation,
    H8H6PredictionReference,
    H8H6PrefixReference,
    H8H6PrefixSemanticFamilyReference,
    H8H7Reference,
    H8LegacyH6PrefixReference,
    H8LegacyH6PredictionReference,
)
from vfe4.types.h6 import BoundedPrefixCertificateSet
from vfe4.types.results import GateStatus, H8GateResult


H8_VALIDATION_SCHEMA = "h8-sparse-scale-v3"
H8_CURRENT_CANDIDATE_RESULT_SCHEMA = "h8-current-candidate-result-v2"
H8_BOUNDED_CLAIM = (
    "The frozen T=128, K=d_z=d_m=20 synthetic chain completed within the "
    "preregistered sparse storage, allocation, numerical, time, and memory "
    "contract."
)

H8_VALIDATION_TOP_LEVEL_KEYS = (
    "schema_version",
    "gate",
    "status",
    "obligations",
    "bounded_claim",
    "nonclaims",
    "revision",
    "config",
    "prerequisites",
    "interpretation",
    "protocol",
    "environment",
    "problems",
    "storage",
    "factor",
    "correctness",
    "allocation",
    "controls",
    "child_attempts",
    "production_runs",
    "profiler_runs",
    "budgets",
    "invariants",
    "artifacts",
)

H8_PUBLICATION_PAYLOAD_KEYS = (
    "config.json",
    "environment.json",
    "provenance.json",
    "references/h6_prediction.json",
    "references/h7.json",
    "validation/h8.json",
)

H8_POINTER_PREDECESSOR_KEYS = (
    "h1_h5",
    "h1_prefix_prior",
    "h6_prefix",
    "h7",
    "h6_prediction",
)

H8_H6_PREFIX_PAYLOAD_KEYS = (
    "certificates/prefix_set.json",
    "config.json",
    "environment.json",
    "provenance.json",
    "validation/h6_prefix.json",
)

H8_SOURCE_ONLY_OBLIGATIONS = (
    "h8_correctness_grid_not_executed",
    "h8_production_runs_not_executed",
    "h8_profiler_runs_not_executed",
    "h8_negative_controls_not_executed",
    "h8_runtime_sections_not_bound",
    "h8_parent_orchestrator_not_implemented",
)
H8_EXPECTED_CHILD_ATTEMPT_IDS = (
    *(
        ("production", seed, repetition, None)
        for seed in H8_PRODUCTION_SEEDS
        for repetition in range(5)
    ),
    *(("profiler", seed, None, None) for seed in H8_PRODUCTION_SEEDS),
    *(
        (
            "negative_control",
            H8_PRODUCTION_SEEDS[0],
            None,
            control_id,
        )
        for control_id in H8_NEGATIVE_CONTROL_IDS
    ),
)

@dataclass(frozen=True, slots=True)
class H8PrerequisiteArtifactValidation:
    registry_sha256: str
    revalidated_reference_names: tuple[str, ...]
    obligations: tuple[str, ...]
    validation_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.registry_sha256, "registry_sha256")
        allowed_names = (
            "h1_h5",
            "h1_prefix_prior",
            "h6_prefix",
            "h7",
            "h6_prediction",
        )
        if (
            type(self.revalidated_reference_names) is not tuple
            or any(
                type(item) is not str or item not in allowed_names
                for item in self.revalidated_reference_names
            )
            or tuple(
                name
                for name in allowed_names
                if name in self.revalidated_reference_names
            )
            != self.revalidated_reference_names
        ):
            raise ValueError("revalidated prerequisite names are not exact and ordered")
        if (
            type(self.obligations) is not tuple
            or any(type(item) is not str or not item for item in self.obligations)
            or len(set(self.obligations)) != len(self.obligations)
        ):
            raise ValueError("prerequisite obligations must be unique strings")
        expected = hashlib.sha256(
            canonical_h8_json_bytes(
                {
                    "domain": "vfe4.h8.prerequisite-artifact-validation.v2",
                    "registry_sha256": self.registry_sha256,
                    "revalidated_reference_names": (
                        self.revalidated_reference_names
                    ),
                    "obligations": self.obligations,
                }
            )
        ).hexdigest()
        if self.validation_sha256 != expected:
            raise ValueError("prerequisite artifact validation hash is stale")

    @classmethod
    def create(
        cls,
        *,
        registry_sha256: str,
        revalidated_reference_names: tuple[str, ...],
        obligations: tuple[str, ...],
    ) -> "H8PrerequisiteArtifactValidation":
        payload = {
            "domain": "vfe4.h8.prerequisite-artifact-validation.v2",
            "registry_sha256": registry_sha256,
            "revalidated_reference_names": revalidated_reference_names,
            "obligations": obligations,
        }
        return cls(
            registry_sha256=registry_sha256,
            revalidated_reference_names=revalidated_reference_names,
            obligations=obligations,
            validation_sha256=hashlib.sha256(
                canonical_h8_json_bytes(payload)
            ).hexdigest(),
        )


_HEX = frozenset("0123456789abcdef")
_SUPPLIED_RUNTIME_SECTION_KEYS = (
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

_NESTED_KEYS: Mapping[str, tuple[str, ...]] = {
    "revision": (
        "git_head",
        "dirty_digest",
        "dependency_closure_sha256",
        "manuscript_sha256",
        "preregistration_sha256",
        "h7_plan_sha256",
    ),
    "config": (
        "config_sha256",
        "objective_schema_sha256",
        "protocol_sha256",
        "canonical_json_sha256",
        "selected_operation",
        "ordered_gates",
        "current_refs_registry_sha256",
        "candidate_junit_sha256",
    ),
    "interpretation": (
        "interpretation_sha256",
        "choice_kind",
        "K_semantics",
        "T",
        "N",
        "K",
        "d_z",
        "d_m",
        "b",
        "D",
        "V",
        "coordinate_order",
        "state_parent_sets",
        "model_parent_sets",
        "state_source_support",
        "model_source_support",
        "ambiguity_policy",
    ),
    "protocol": (
        "generator_schema",
        "generator_draw_schema_sha256",
        "sample_schema",
        "factor_schema",
        "selected_inverse_schema",
        "condition_estimator_schema",
        "allocation_schema",
        "torch_version",
        "profiler_source_hashes",
        "profiler_api_contract_sha256",
        "profiler_raw_event_schema",
        "child_schema",
        "production_seed_order",
        "production_sample_seed_map",
        "repetition_order",
        "correctness_seed_table",
        "required_operations",
        "negative_control_order",
    ),
    "environment": (
        "platform",
        "platform_release",
        "processor",
        "cpu_count",
        "affinity",
        "python_version",
        "pytorch_version",
        "numpy_version",
        "device",
        "dtype",
        "grad_enabled",
        "intraop_threads",
        "interop_threads",
        "thread_environment",
        "blas_identity",
        "hardware_identity_sha256",
        "affinity_sha256",
        "thread_identity_sha256",
        "blas_identity_sha256",
    ),
    "storage": (
        "h_scalars",
        "input_precision_scalars",
        "factor_scalars",
        "selected_inverse_scalars",
        "category_cap_scalars",
        "dense_forbidden_scalars",
        "input_within_cap",
        "factor_within_cap",
        "selected_within_cap",
    ),
    "factor": (
        "algorithm",
        "pattern",
        "fill",
        "workspace",
        "condition_estimate",
        "per_block_min_pivots",
        "per_block_pivot_margins",
        "global_min_pivot",
        "global_pivot_margin",
        "counters",
        "reconstruction_invariants",
    ),
    "allocation": (
        "whitelist",
        "dispatch",
        "live_storage",
        "profiler_api",
        "profiler_raw_events",
        "preexisting_storage_count",
        "preexisting_bytes",
        "baseline_live_bytes",
        "profiler_reconstructed_live_peak",
        "profiler_net_deltas_supplementary",
        "backend",
        "os_hwm",
        "tracemalloc_supplementary",
        "cross_checks",
        "all_observable",
        "no_forbidden_attempts",
    ),
    "budgets": (
        "eps",
        "rounding_multiplier",
        "solver_relative_budget",
        "maximum_allowance_scale_fraction",
        "min_cholesky_pivot",
        "max_seconds",
        "max_process_incremental_bytes",
        "max_torch_population_bytes",
        "max_storage_scalars",
        "boundary_policy",
    ),
    "invariants": (
        "prerequisites_current_and_pass",
        "interpretation_hash_current",
        "correctness_cells_complete",
        "correctness_pass",
        "controls_complete",
        "child_attempts_complete",
        "child_attempts_exact_order",
        "child_attempts_cross_bound",
        "observability_complete",
        "every_profiler_action_joined_and_liveness_reconciled",
        "production_runs_complete",
        "profiler_runs_complete",
        "required_operations_reached",
        "storage_pass",
        "forbidden_attempts_zero",
        "offband_fill_zero",
        "pivot_margin_pass",
        "rhs_width_pass",
        "sample_width_pass",
        "time_pass",
        "process_memory_pass",
        "torch_memory_pass",
        "finite_pass",
        "residuals_pass",
        "witnessed_failure_dominance_applied",
        "all_pass",
    ),
    "artifacts": (
        "config_path",
        "provenance_path",
        "environment_path",
        "h7_reference_path",
        "h6_prediction_reference_path",
        "validation_path",
        "manifest_path",
    ),
}


def _sha256(value: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _json_value(value: object) -> object:
    """Return the exact JSON value accepted by the artifact serializer."""

    return json.loads(canonical_h8_json_bytes(value))


def canonical_h8_json_bytes(value: object) -> bytes:
    """Serialize one H8 record with the generic artifact canonicalizer."""

    return canonical_json_bytes(value)


def classify_h8_status(
    *,
    retained_statuses: tuple[GateStatus, ...],
    exact_inventory_complete: bool,
    open_obligations: tuple[str, ...],
) -> GateStatus:
    """Apply witnessed-failure dominance to already-typed evidence."""

    if type(retained_statuses) is not tuple or any(
        type(status) is not GateStatus for status in retained_statuses
    ):
        raise ValueError("retained_statuses must contain exact GateStatus values")
    if type(exact_inventory_complete) is not bool:
        raise ValueError("exact_inventory_complete must be a bool")
    if (
        type(open_obligations) is not tuple
        or any(type(item) is not str or not item for item in open_obligations)
        or len(set(open_obligations)) != len(open_obligations)
    ):
        raise ValueError("open_obligations must be unique nonempty strings")
    if GateStatus.FAIL in retained_statuses:
        return GateStatus.FAIL
    if (
        open_obligations
        or not exact_inventory_complete
        or not retained_statuses
        or GateStatus.INCONCLUSIVE in retained_statuses
    ):
        return GateStatus.INCONCLUSIVE
    return GateStatus.PASS


def make_h8_preflight_inconclusive(
    *,
    config_sha256: str,
    obligations: tuple[str, ...],
    candidate_junit_sha256: str | None = None,
    current_refs_registry_sha256: str | None = None,
    h7_manifest_sha256: str | None = None,
    h6_prediction_manifest_sha256: str | None = None,
) -> H8GateResult:
    """Retain an honest preflight result when current prerequisites are absent."""

    return H8GateResult(
        gate="H8",
        status=GateStatus.INCONCLUSIVE,
        config_sha256=_sha256(config_sha256, "config_sha256"),
        candidate_junit_sha256=candidate_junit_sha256,
        current_refs_registry_sha256=current_refs_registry_sha256,
        h7_manifest_sha256=h7_manifest_sha256,
        h6_prediction_manifest_sha256=h6_prediction_manifest_sha256,
        correctness=(),
        child_attempts=(),
        production_runs=(),
        profiler_runs=(),
        controls=(),
        obligations=obligations,
    )


def _retained_statuses(
    correctness: tuple[H8CorrectnessCell, ...],
    child_attempts: tuple[H8ChildAttemptRecord, ...],
    production_runs: tuple[H8ChildResult, ...],
    profiler_runs: tuple[H8ChildResult, ...],
    controls: tuple[H8ControlResult, ...],
) -> tuple[GateStatus, ...]:
    return (
        *(item.status for item in correctness),
        *(item.status for item in child_attempts),
        *(item.status for item in controls),
        *(
            invariant.status
            for child in (*production_runs, *profiler_runs)
            for invariant in child.invariants
        ),
    )


def _inventory_complete(
    correctness: tuple[H8CorrectnessCell, ...],
    child_attempts: tuple[H8ChildAttemptRecord, ...],
    production_runs: tuple[H8ChildResult, ...],
    profiler_runs: tuple[H8ChildResult, ...],
    controls: tuple[H8ControlResult, ...],
) -> bool:
    return (
        tuple(item.cell_id for item in correctness)
        == tuple(range(1, len(H8_CORRECTNESS_CASES) + 1))
        and _attempt_ids(child_attempts) == H8_EXPECTED_CHILD_ATTEMPT_IDS
        and all(item.status is GateStatus.PASS for item in child_attempts)
        and _attempts_cross_bound(
            child_attempts,
            production_runs,
            profiler_runs,
            controls,
        )
        and tuple((item.seed, item.repetition) for item in production_runs)
        == tuple(
            (seed, repetition)
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        )
        and tuple(item.seed for item in profiler_runs) == H8_PRODUCTION_SEEDS
        and tuple(item.control_id for item in controls) == H8_NEGATIVE_CONTROL_IDS
    )


def _attempt_ids(
    child_attempts: tuple[H8ChildAttemptRecord, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.request.mode,
            item.request.seed,
            item.request.repetition,
            item.request.control_id,
        )
        for item in child_attempts
    )


def _attempts_cross_bound(
    child_attempts: tuple[H8ChildAttemptRecord, ...],
    production_runs: tuple[H8ChildResult, ...],
    profiler_runs: tuple[H8ChildResult, ...],
    controls: tuple[H8ControlResult, ...],
) -> bool:
    return (
        production_runs
        == tuple(
            item.result
            for item in child_attempts
            if item.request.mode == "production"
            and type(item.result) is H8ChildResult
        )
        and profiler_runs
        == tuple(
            item.result
            for item in child_attempts
            if item.request.mode == "profiler"
            and type(item.result) is H8ChildResult
        )
        and controls
        == tuple(
            item.result
            for item in child_attempts
            if item.request.mode == "negative_control"
            and type(item.result) is H8ControlResult
        )
    )


def _typed_records(
    values: object,
    expected: type[object],
    name: str,
) -> tuple[object, ...]:
    if type(values) is not tuple or any(type(item) is not expected for item in values):
        raise ValueError(f"{name} must retain exact typed records")
    for item in values:
        cast(object, item).__post_init__()  # type: ignore[attr-defined]
    return values


def _source_only_sections(
    *,
    result: H8GateResult,
    refs: CurrentH8PrerequisiteRefs,
    dependency_closure_sha256: str,
    preregistration_sha256: str,
    prerequisites_current_and_pass: bool,
) -> dict[str, object]:
    """Build an exact-key, explicitly unavailable source-only section set."""

    return {
        "revision": {
            "git_head": refs.candidate_head,
            "dirty_digest": refs.candidate_dirty_digest,
            "dependency_closure_sha256": dependency_closure_sha256,
            "manuscript_sha256": None,
            "preregistration_sha256": preregistration_sha256,
            "h7_plan_sha256": H8_H7_PLAN_SHA256,
        },
        "config": {
            "config_sha256": result.config_sha256,
            "objective_schema_sha256": None,
            "protocol_sha256": None,
            "canonical_json_sha256": result.config_sha256,
            "selected_operation": "H8",
            "ordered_gates": H8_VERIFIER_PREFIX,
            "current_refs_registry_sha256": refs.registry_sha256,
            "candidate_junit_sha256": refs.candidate_junit_sha256,
        },
        "interpretation": {
            "interpretation_sha256": H8_INTERPRETATION_SHA256,
            "choice_kind": "operational_preregistration_not_manuscript_theorem",
            "K_semantics": "each_channel_dimension",
            "T": 128,
            "N": 129,
            "K": 20,
            "d_z": 20,
            "d_m": 20,
            "b": 40,
            "D": 5160,
            "V": 3,
            "coordinate_order": "[z_0,m_0,...,z_T,m_T]",
            "state_parent_sets": None,
            "model_parent_sets": None,
            "state_source_support": None,
            "model_source_support": None,
            "ambiguity_policy": "reject_alternative_K_semantics",
        },
        "protocol": {
            "generator_schema": "h8-synthetic-chain-v1",
            "generator_draw_schema_sha256": H8_PROBLEM_DRAW_SCHEMA_SHA256,
            "sample_schema": "h8-pcg64-sample-v1",
            "factor_schema": None,
            "selected_inverse_schema": None,
            "condition_estimator_schema": None,
            "allocation_schema": None,
            "torch_version": "2.9.1",
            "profiler_source_hashes": {
                "memory_profile": H8_PROFILER_MEMORY_SOURCE_SHA256,
                "profiler": H8_PROFILER_SOURCE_SHA256,
            },
            "profiler_api_contract_sha256": H8_PROFILER_API_CONTRACT_SHA256,
            "profiler_raw_event_schema": None,
            "child_schema": "h8-child-v1",
            "production_seed_order": H8_PRODUCTION_SEEDS,
            "production_sample_seed_map": H8_PRODUCTION_SAMPLE_SEED_PAIRS,
            "repetition_order": tuple(range(5)),
            "correctness_seed_table": H8_CORRECTNESS_CASES,
            "required_operations": H8_REQUIRED_OPERATIONS,
            "negative_control_order": H8_NEGATIVE_CONTROL_IDS,
        },
        "environment": {
            "platform": None,
            "platform_release": None,
            "processor": None,
            "cpu_count": None,
            "affinity": None,
            "python_version": None,
            "pytorch_version": "2.9.1",
            "numpy_version": None,
            "device": "cpu",
            "dtype": "float64",
            "grad_enabled": False,
            "intraop_threads": None,
            "interop_threads": None,
            "thread_environment": None,
            "blas_identity": None,
            "hardware_identity_sha256": None,
            "affinity_sha256": None,
            "thread_identity_sha256": None,
            "blas_identity_sha256": None,
        },
        "problems": (),
        "storage": {key: None for key in _NESTED_KEYS["storage"]},
        "factor": {key: None for key in _NESTED_KEYS["factor"]},
        "allocation": {
            **{key: None for key in _NESTED_KEYS["allocation"]},
            "profiler_raw_events": (),
            "all_observable": False,
            "no_forbidden_attempts": False,
        },
        "budgets": {
            "eps": 2.220446049250313e-16,
            "rounding_multiplier": 4096,
            "solver_relative_budget": 1e-9,
            "maximum_allowance_scale_fraction": (
                H8_MAXIMUM_ALLOWANCE_SCALE_FRACTION
            ),
            "min_cholesky_pivot": H8_MIN_CHOLESKY_PIVOT,
            "max_seconds": H8_MAX_SECONDS,
            "max_process_incremental_bytes": H8_MAX_PROCESS_INCREMENTAL_BYTES,
            "max_torch_population_bytes": H8_MAX_TORCH_POPULATION_BYTES,
            "max_storage_scalars": None,
            "boundary_policy": "residual_le_allowance_and_fraction_lt_limit",
        },
        "invariants": {
            **{key: False for key in _NESTED_KEYS["invariants"]},
            "prerequisites_current_and_pass": prerequisites_current_and_pass,
            "interpretation_hash_current": True,
            "child_attempts_complete": (
                len(result.child_attempts)
                == len(H8_EXPECTED_CHILD_ATTEMPT_IDS)
            ),
            "child_attempts_exact_order": (
                _attempt_ids(result.child_attempts)
                == H8_EXPECTED_CHILD_ATTEMPT_IDS
            ),
            "child_attempts_cross_bound": (
                bool(result.child_attempts)
                and _attempts_cross_bound(
                    result.child_attempts,
                    result.production_runs,
                    result.profiler_runs,
                    result.controls,
                )
            ),
            "witnessed_failure_dominance_applied": True,
            "all_pass": result.status is GateStatus.PASS,
        },
        "artifacts": {
            "config_path": "config.json",
            "provenance_path": "provenance.json",
            "environment_path": "environment.json",
            "h7_reference_path": "references/h7.json",
            "h6_prediction_reference_path": "references/h6_prediction.json",
            "validation_path": "validation/h8.json",
            "manifest_path": "manifest.sha256",
        },
    }


def _validate_runtime_sections(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping) or tuple(value) != _SUPPLIED_RUNTIME_SECTION_KEYS:
        raise ValueError("runtime sections must retain the exact H8 section order")
    copied: dict[str, object] = {}
    for name in _SUPPLIED_RUNTIME_SECTION_KEYS:
        section = value[name]
        if name == "problems":
            if type(section) not in (tuple, list):
                raise ValueError("problems must be an ordered array")
            ordered_problems: list[dict[str, object]] = []
            for problem in section:
                if not isinstance(problem, Mapping) or tuple(problem) != _PROBLEM_KEYS:
                    raise ValueError(
                        "every problem must retain the exact H8 problem schema"
                    )
                ordered_problems.append(
                    {
                        key: _json_value(problem[key])
                        for key in _PROBLEM_KEYS
                    }
                )
            copied[name] = tuple(ordered_problems)
        else:
            if not isinstance(section, Mapping) or tuple(section) != _NESTED_KEYS[name]:
                raise ValueError(f"{name} must retain its exact nested key inventory")
            copied[name] = {
                key: _json_value(section[key])
                for key in _NESTED_KEYS[name]
            }
    return copied


def _validate_section_bindings(
    sections: Mapping[str, object],
    *,
    result: H8GateResult,
    refs: CurrentH8PrerequisiteRefs,
    dependency_closure_sha256: str,
    preregistration_sha256: str,
    prerequisites_current_and_pass: bool,
) -> None:
    """Reject section identities or decisions that contradict typed evidence."""

    revision = cast(Mapping[str, object], sections["revision"])
    config = cast(Mapping[str, object], sections["config"])
    interpretation = cast(Mapping[str, object], sections["interpretation"])
    invariants = cast(Mapping[str, object], sections["invariants"])
    artifacts = cast(Mapping[str, object], sections["artifacts"])
    if (
        revision["git_head"] != refs.candidate_head
        or revision["dirty_digest"] != refs.candidate_dirty_digest
        or revision["dependency_closure_sha256"] != dependency_closure_sha256
        or revision["preregistration_sha256"] != preregistration_sha256
        or revision["h7_plan_sha256"] != H8_H7_PLAN_SHA256
    ):
        raise ValueError("H8 revision section is not bound to current inputs")
    if (
        config["config_sha256"] != result.config_sha256
        or config["canonical_json_sha256"] != result.config_sha256
        or config["selected_operation"] != "H8"
        or tuple(cast(object, config["ordered_gates"])) != H8_VERIFIER_PREFIX
        or config["current_refs_registry_sha256"] != refs.registry_sha256
        or config["candidate_junit_sha256"] != refs.candidate_junit_sha256
    ):
        raise ValueError("H8 config section is not bound to the resolved candidate")
    if (
        interpretation["interpretation_sha256"] != H8_INTERPRETATION_SHA256
        or interpretation["choice_kind"]
        != "operational_preregistration_not_manuscript_theorem"
        or interpretation["K_semantics"] != "each_channel_dimension"
        or tuple(
            interpretation[name]
            for name in ("T", "N", "K", "d_z", "d_m", "b", "D", "V")
        )
        != (128, 129, 20, 20, 20, 40, 5160, 3)
    ):
        raise ValueError("H8 interpretation section is not the frozen interpretation")
    if (
        invariants["prerequisites_current_and_pass"]
        is not prerequisites_current_and_pass
        or invariants["interpretation_hash_current"] is not True
        or invariants["child_attempts_complete"]
        is not (
            len(result.child_attempts)
            == len(H8_EXPECTED_CHILD_ATTEMPT_IDS)
        )
        or invariants["child_attempts_exact_order"]
        is not (
            _attempt_ids(result.child_attempts)
            == H8_EXPECTED_CHILD_ATTEMPT_IDS
        )
        or invariants["child_attempts_cross_bound"]
        is not (
            bool(result.child_attempts)
            and _attempts_cross_bound(
                result.child_attempts,
                result.production_runs,
                result.profiler_runs,
                result.controls,
            )
        )
        or invariants["witnessed_failure_dominance_applied"] is not True
        or invariants["all_pass"] is not (result.status is GateStatus.PASS)
    ):
        raise ValueError("H8 invariant decisions contradict the gate result")
    if tuple(artifacts.values()) != (
        "config.json",
        "provenance.json",
        "environment.json",
        "references/h7.json",
        "references/h6_prediction.json",
        "validation/h8.json",
        "manifest.sha256",
    ):
        raise ValueError("H8 artifact paths differ from the exact publication inventory")


def _read_immutable_file(
    raw_path: str,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[Path, bytes]:
    """Reopen one exact immutable file without projecting or copying its payload."""

    _sha256(expected_sha256, f"{name}_sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    if path.is_symlink():
        raise ValueError(f"{name} path cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} path is unavailable") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} path must be a regular non-symlink file")
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{name} bytes differ from the immutable reference")
    return resolved, payload


def _safe_artifact_payload_path(root: Path, raw_name: str) -> Path:
    relative = PurePosixPath(raw_name)
    if (
        not raw_name
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError("artifact payload name is not a safe relative POSIX path")
    path = root.joinpath(*relative.parts)
    if path.is_symlink():
        raise ValueError("artifact payload cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("artifact payload is unavailable") from exc
    if root not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise ValueError("artifact payload escapes its immutable artifact root")
    return resolved


def _manifest_entries_exact(manifest_bytes: bytes) -> tuple[tuple[str, str], ...]:
    try:
        text = manifest_bytes.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ValueError("prerequisite manifest is not strict ASCII") from exc
    if not text.endswith("\n"):
        raise ValueError("prerequisite manifest must end in one newline")
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or not name
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise ValueError("prerequisite manifest entry is malformed")
        entries.append((name, digest))
    if not entries or len({name for name, _ in entries}) != len(entries):
        raise ValueError("prerequisite manifest inventory is empty or duplicated")
    return tuple(entries)


def _reopen_manifested_artifact(
    reference: object,
    *,
    require_content_hashes: bool = True,
) -> tuple[Path, Mapping[str, bytes]]:
    raw_root = getattr(reference, "artifact_path", None)
    if type(raw_root) is not str:
        raise ValueError("prerequisite artifact path is missing")
    root, entries, payloads = _reopen_manifested_root(
        raw_root,
        expected_manifest_sha256=getattr(reference, "manifest_sha256", None),
    )
    payload_hashes = getattr(reference, "payload_hashes", None)
    if (
        not isinstance(payload_hashes, Mapping)
        or tuple(entries) != tuple(payload_hashes.items())
    ):
        raise ValueError("prerequisite manifest differs from keyed payload hashes")
    if require_content_hashes:
        _revalidate_content_hashes(reference, payloads)
    return root, payloads


def _resolve_immutable_artifact_root(raw_root: str, *, name: str) -> Path:
    if type(raw_root) is not str:
        raise ValueError(f"{name} path is missing")
    root_path = Path(raw_root)
    if not root_path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    if root_path.is_symlink():
        raise ValueError(f"{name} root cannot be a symlink")
    try:
        root = root_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} root is unavailable") from exc
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} root must be a regular directory")
    return root


def _reopen_manifested_root(
    raw_root: str,
    *,
    expected_manifest_sha256: object,
) -> tuple[Path, tuple[tuple[str, str], ...], Mapping[str, bytes]]:
    root = _resolve_immutable_artifact_root(
        raw_root,
        name="prerequisite artifact",
    )
    if type(expected_manifest_sha256) is not str:
        raise ValueError("prerequisite manifest SHA-256 is missing")
    _sha256(expected_manifest_sha256, "prerequisite_manifest_sha256")
    manifest_path = root / "manifest.sha256"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("prerequisite artifact manifest is unavailable")
    manifest_bytes = manifest_path.read_bytes()
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != expected_manifest_sha256
    ):
        raise ValueError("prerequisite artifact manifest hash is stale")
    entries = _manifest_entries_exact(manifest_bytes)
    payloads: dict[str, bytes] = {}
    for name, expected_sha256 in entries:
        path = _safe_artifact_payload_path(root, name)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("prerequisite artifact payload hash is stale")
        payloads[name] = payload
    observed_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed_files != {*payloads, "manifest.sha256"}:
        raise ValueError("prerequisite artifact file inventory is not exact")
    return root, entries, payloads


def _revalidate_content_hashes(
    reference: object,
    payloads: Mapping[str, bytes],
) -> None:
    """Rehash every named content preimage rather than trusting registry digests."""

    content_hashes = getattr(reference, "content_hashes", None)
    if not isinstance(content_hashes, Mapping) or not content_hashes:
        raise ValueError("prerequisite content hashes are missing")
    for name, expected_sha256 in content_hashes.items():
        if type(name) is not str or name not in payloads:
            raise ValueError("content hash does not name a manifested artifact payload")
        _sha256(expected_sha256, f"content_hashes[{name!r}]")
        if hashlib.sha256(payloads[name]).hexdigest() != expected_sha256:
            raise ValueError("manifested content bytes differ from their registry hash")


def _canonical_mapping(payload: bytes, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_h8_json_bytes(value) != payload:
        raise ValueError(f"{name} is not one canonical JSON object")
    return value


def _reopen_reference_common(reference: object) -> tuple[Path, Mapping[str, bytes]]:
    root, payloads = _reopen_manifested_artifact(reference)
    _read_immutable_file(
        getattr(reference, "result_path"),
        expected_sha256=getattr(reference, "result_sha256"),
        name=f"{getattr(reference, 'kind')}_result",
    )
    _read_immutable_file(
        getattr(reference, "ledger_path"),
        expected_sha256=getattr(reference, "ledger_sha256"),
        name=f"{getattr(reference, 'kind')}_ledger",
    )
    return root, payloads


def _reopen_h7_compatibility_references(
    refs: CurrentH8PrerequisiteRefs,
) -> None:
    candidate_junit_path: Path | None = None
    for key, reference in refs.h7_compatibility_refs.items():
        _reopen_manifested_artifact(
            reference,
            require_content_hashes=False,
        )
        _read_immutable_file(
            reference.ledger_path,
            expected_sha256=reference.ledger_sha256,
            name=f"h7_transitive_{key}_ledger",
        )
        junit_path, _ = _read_immutable_file(
            reference.junit_path,
            expected_sha256=reference.junit_sha256,
            name=f"h7_transitive_{key}_junit",
        )
        if candidate_junit_path is None:
            candidate_junit_path = junit_path
        elif junit_path != candidate_junit_path:
            raise ValueError("H7 transitive references do not share one JUnit preimage")


def _revalidate_h6_prefix_certificates(
    reference: object,
    payloads: Mapping[str, bytes],
) -> None:
    if type(reference) is not H8H6PrefixReference:
        raise ValueError(
            "H8 requires an exact bounded H6-Prefix reference"
        )
    reference.__post_init__()
    if tuple(payloads) != H8_H6_PREFIX_PAYLOAD_KEYS:
        raise ValueError(
            "bounded H6-Prefix artifact must retain its exact five payloads"
        )
    certificate_set = reopen_bounded_prefix_certificate_set(
        root=Path(getattr(reference, "artifact_path")),
        expected_manifest_sha256=getattr(reference, "manifest_sha256"),
        expected_git_head=getattr(reference, "producer_head"),
        expected_dirty_digest=getattr(reference, "producer_dirty_digest"),
        expected_junit_sha256=getattr(
            reference,
            "candidate_junit_sha256",
        ),
    )
    if type(certificate_set) is not BoundedPrefixCertificateSet:
        raise ValueError(
            "bounded H6-Prefix reopen must return its exact certificate set"
        )
    observed_families = tuple(
        H8H6PrefixSemanticFamilyReference(
            semantic_family_index=index,
            semantic_family_sha256=certificate.semantic_family_sha256,
            validation_payload_sha256=(
                certificate.validation_payload_sha256
            ),
            certificate_sha256=certificate.certificate_sha256,
        )
        for index, certificate in enumerate(certificate_set.certificates)
    )
    if (
        certificate_set.schema_version
        != reference.certificate_set_schema
        or certificate_set.config_sha256 != reference.config_sha256
        or certificate_set.workload_plan_sha256
        != reference.workload_plan_sha256
        or certificate_set.validation_payload_sha256
        != reference.validation_payload_sha256
        or certificate_set.prefix_certificate_set_sha256
        != reference.prefix_certificate_set_sha256
        or observed_families != reference.semantic_families
    ):
        raise ValueError(
            "bounded H6-Prefix schemas, aggregate hashes, or ordered "
            "semantic families differ from the registry reference"
        )


def _revalidate_h7_fixture_set(
    reference: H8H7Reference,
    payloads: Mapping[str, bytes],
) -> None:
    from vfe4.types.h7 import h7_owned_sha256
    from vfe4.types.results import H7GateResult
    from vfe4.validation import parse_h7_fixture_bytes

    raw_validation = payloads.get("validation/h7.json")
    if raw_validation is None:
        raise ValueError("H7 artifact lacks validation/h7.json")
    validation = _canonical_mapping(raw_validation, "H7 validation")
    result = validation.get("result")
    if (
        validation.get("schema") != "h7-frame-covariance-validation-v1"
        or type(result) is not dict
        or result.get("status") != "pass"
        or result.get("obligations") != []
        or not isinstance(result.get("fixture_hashes"), Mapping)
    ):
        raise ValueError("H7 validation is not one exact PASS fixture record")

    repository_root = Path(__file__).resolve().parents[1]
    fixture_root = repository_root / "vfe4" / "validation" / "fixtures"
    fixture_paths = {
        "h1_fixture_raw_sha256": fixture_root / "h1_v1.json",
        "h7_fixture_raw_sha256": fixture_root / "h7_v1.json",
        "density_probe_table_raw_sha256": (
            fixture_root / "h7_density_probes_v1.json"
        ),
    }
    fixture_bytes: dict[str, bytes] = {}
    for key, path in fixture_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"H7 current-candidate fixture is unavailable: {key}")
        resolved = path.resolve(strict=True)
        if (
            repository_root not in resolved.parents
            or resolved.is_symlink()
            or not resolved.is_file()
        ):
            raise ValueError(f"H7 current-candidate fixture escapes the repository: {key}")
        fixture_bytes[key] = resolved.read_bytes()
    parsed_fixture = parse_h7_fixture_bytes(
        fixture_bytes["h7_fixture_raw_sha256"],
    )
    if (
        fixture_paths["density_probe_table_raw_sha256"].read_bytes()
        != fixture_bytes["density_probe_table_raw_sha256"]
    ):
        raise ValueError("H7 density-probe bytes changed during fixture parsing")
    derived_fixture_hashes = {
        key: hashlib.sha256(payload).hexdigest()
        for key, payload in fixture_bytes.items()
    }
    derived_fixture_hashes["density_probe_set_sha256"] = (
        parsed_fixture.density_probe_set_sha256
    )
    if (
        tuple(derived_fixture_hashes) != H7GateResult.fixture_hash_keys
        or result["fixture_hashes"] != derived_fixture_hashes
    ):
        raise ValueError(
            "H7 validation fixture hashes differ from current-candidate fixture bytes"
        )
    derived_fixture_set_sha256 = h7_owned_sha256(
        "vfe4.h7.fixture-set.v1",
        derived_fixture_hashes,
    )
    if (
        validation.get("fixture_set_sha256") != derived_fixture_set_sha256
        or reference.fixture_set_sha256 != derived_fixture_set_sha256
    ):
        raise ValueError("H7 fixture-set hash is not derivable from reopened fixture bytes")


def _validate_blinded_data_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
) -> None:
    _sha256(expected_manifest_sha256, "blinded_data_manifest_sha256")
    manifest_path = root / "manifest.sha256"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("blinded-data manifest is unavailable")
    if manifest_path.read_bytes() != (expected_manifest_sha256 + "\n").encode("ascii"):
        raise ValueError("blinded-data directory manifest identity is stale")


def _reconstruct_h6_prediction_readiness(
    *,
    reference: H8H6PredictionReference,
    config: Mapping[str, object],
) -> object:
    from vfe4.config import resolve_h6_prediction_v2_config
    from vfe4.training import h6_readiness
    from vfe4.types.h6 import issue_prediction_readiness_v2

    repository_root = Path(__file__).resolve().parents[1]
    resolved = resolve_h6_prediction_v2_config(
        config,
        repo_root=repository_root,
    )
    source = resolved.source
    if (
        resolved.config_sha256 != reference.config_sha256
        or resolved.schema_version != reference.config_schema
        or source.git_head != reference.producer_head
        or source.dirty_digest != reference.producer_dirty_digest
        or source.source_sha256
        != source_candidate_sha256(
            git_head_value=reference.producer_head,
            dirty_digest_value=reference.producer_dirty_digest,
        )
        or resolved.matching_set_sha256 != reference.matching_set_sha256
        or resolved.h1_prefix_prior_generative_factor_schema_sha256
        != reference.h1_prefix_prior_generative_factor_schema_sha256
        or resolved.smc_bias_semantics_sha256
        != reference.smc_bias_semantics_sha256
        or resolved.objective_gate.spec_sha256
        != reference.objective_gate_spec_sha256
    ):
        raise ValueError("H6-Prediction resolved config/source bindings are stale")

    correctness_manifests = dict(resolved.correctness_manifests)
    if tuple(correctness_manifests) != ("H1", "H2", "H3", "H5"):
        raise ValueError("H6-Prediction correctness manifest inventory is not exact")
    correctness = []
    for gate in ("H1", "H2", "H3", "H5"):
        raw_path = reference.correctness_artifact_paths[gate]
        _reopen_manifested_root(
            raw_path,
            expected_manifest_sha256=correctness_manifests[gate],
        )
        correctness.append(
            h6_readiness._load_prediction_correctness_artifact(
                gate=gate,
                root=Path(raw_path),
                expected_manifest_sha256=correctness_manifests[gate],
                expected_git_head=source.git_head,
                expected_dirty_digest=source.dirty_digest,
            )
        )

    _reopen_manifested_root(
        reference.h1_prefix_prior_artifact_path,
        expected_manifest_sha256=resolved.h1_prefix_prior_manifest_sha256,
    )
    h1_prefix = h6_readiness._load_h1_prefix_prior_artifact(
        root=Path(reference.h1_prefix_prior_artifact_path),
        expected_manifest_sha256=resolved.h1_prefix_prior_manifest_sha256,
        expected_generative_factor_schema_sha256=(
            resolved.h1_prefix_prior_generative_factor_schema_sha256
        ),
        expected_git_head=source.git_head,
        expected_dirty_digest=source.dirty_digest,
        expected_source_sha256=source.source_sha256,
    )

    if (
        resolved.smc_validation_manifest_sha256
        != reference.smc_accuracy_manifest_sha256
    ):
        raise ValueError("finite-SMC manifest differs from the H6-Prediction config")
    _reopen_manifested_root(
        reference.smc_accuracy_artifact_path,
        expected_manifest_sha256=reference.smc_accuracy_manifest_sha256,
    )
    smc = h6_readiness._load_smc_accuracy_artifact(
        root=Path(reference.smc_accuracy_artifact_path),
        expected_manifest_sha256=reference.smc_accuracy_manifest_sha256,
        expected_git_head=source.git_head,
        expected_dirty_digest=source.dirty_digest,
    )

    _reopen_manifested_root(
        reference.h6_prefix_artifact_path,
        expected_manifest_sha256=reference.h6_prefix_manifest_sha256,
    )
    certificates = h6_readiness._load_prefix_certificates(
        root=Path(reference.h6_prefix_artifact_path),
        expected_set_sha256=resolved.prefix_certificate_set_sha256,
        expected_git_head=source.git_head,
        expected_dirty_digest=source.dirty_digest,
    )
    h5_binding = h6_readiness._load_h5_update_binding(
        Path(reference.correctness_artifact_paths["H5"]),
        expected_binding_sha256=resolved.h5_update_binding_sha256,
    )

    blinded_root = _resolve_immutable_artifact_root(
        reference.blinded_data_artifact_path,
        name="blinded-data artifact",
    )
    _validate_blinded_data_manifest(
        blinded_root,
        expected_manifest_sha256=reference.blinded_data_manifest_sha256,
    )
    observed_archive = resolved.data.observed_archive
    if observed_archive is None:
        raise ValueError("H6-Prediction config lacks its observed archive identity")
    data_identity = h6_readiness._load_blinded_data_identity(
        blinded_root,
        expected_archive_sha256=observed_archive.archive_sha256,
        expected_data_identity_sha256=resolved.data_identity_sha256,
        expected_access_policy_sha256=resolved.access_policy_sha256,
    )

    _reopen_manifested_root(
        reference.matching_artifact_path,
        expected_manifest_sha256=reference.matching_manifest_sha256,
    )
    matching_set = h6_readiness._validate_matching_artifact(
        Path(reference.matching_artifact_path),
        expected_set_sha256=resolved.matching_set_sha256,
        expected_git_head=source.git_head,
        expected_dirty_digest=source.dirty_digest,
    )

    return issue_prediction_readiness_v2(
        git_head=source.git_head,
        dirty_digest=source.dirty_digest,
        experiment_config_sha256=resolved.config_sha256,
        correctness_artifacts=tuple(correctness),
        h1_prefix_prior_artifact=h1_prefix,
        h1_prefix_prior_generative_factor_schema_sha256=(
            resolved.h1_prefix_prior_generative_factor_schema_sha256
        ),
        smc_bias_semantics_sha256=resolved.smc_bias_semantics_sha256,
        objective_gate_spec=resolved.objective_gate,
        h5_update_binding=h5_binding,
        h6_training_schedule=resolved.training_schedule,
        smc_accuracy_artifact=smc,
        critical_values_sha256=resolved.critical_values_sha256,
        endpoint_smc_protocol=resolved.endpoint_smc_protocol,
        attribution_matrix_sha256=resolved.attribution_matrix_sha256,
        matching_set_sha256=matching_set.matching_set_sha256,
        prefix_certificates=certificates,
        data_identity=data_identity,
        matching_set=matching_set,
    )


def _reopen_h6_prediction_v2(
    reference: H8H6PredictionReference,
) -> None:
    from vfe4.artifacts.h6 import read_h6_prediction_result
    from vfe4.training.h6_readiness import _readiness_payload

    artifact_root, payloads = _reopen_reference_common(reference)
    expected_result_path = (
        artifact_root / "validation" / "h6_prediction_result.json"
    ).resolve(strict=False)
    if Path(reference.result_path).resolve(strict=True) != expected_result_path:
        raise ValueError("H6-Prediction result path is not its manifested result")
    expected_payload_names = (
        "raw/h6_endpoint_records.json",
        "validation/h6_prediction_metrics.json",
        "validation/h6_prediction_result.json",
    )
    if tuple(payloads) != expected_payload_names:
        raise ValueError("H6-Prediction artifact payload inventory is not amended v2")
    readiness_root, _entries, readiness_payloads = _reopen_manifested_root(
        reference.readiness_artifact_path,
        expected_manifest_sha256=reference.readiness_manifest_sha256,
    )
    if (
        tuple(readiness_payloads)
        != (
            "config.json",
            "validation/h6_prediction_readiness.json",
        )
        or readiness_root.name
        != f"h6-prediction-readiness-{reference.readiness_sha256[:16]}"
    ):
        raise ValueError("H6-Prediction readiness artifact inventory is not exact")
    config_bytes = readiness_payloads["config.json"]
    config = _canonical_mapping(config_bytes, "H6-Prediction config")
    readiness = _canonical_mapping(
        readiness_payloads["validation/h6_prediction_readiness.json"],
        "H6-Prediction readiness",
    )
    typed_readiness = _reconstruct_h6_prediction_readiness(
        reference=reference,
        config=config,
    )
    expected_readiness_bytes = canonical_h8_json_bytes(
        _readiness_payload(typed_readiness)
    )
    if (
        hashlib.sha256(config_bytes).hexdigest() != reference.config_sha256
        or config.get("schema_version") != reference.config_schema
        or canonical_h8_json_bytes(config) != config_bytes
        or readiness_payloads["validation/h6_prediction_readiness.json"]
        != expected_readiness_bytes
        or readiness.get("readiness_schema") != reference.readiness_schema
        or readiness.get("status") != "PASS"
        or readiness.get("readiness_sha256") != reference.readiness_sha256
        or readiness.get("experiment_config_sha256") != reference.config_sha256
        or readiness.get("matching_set_sha256") != reference.matching_set_sha256
        or readiness.get("h1_prefix_prior_generative_factor_schema_sha256")
        != reference.h1_prefix_prior_generative_factor_schema_sha256
        or readiness.get("smc_bias_semantics_sha256")
        != reference.smc_bias_semantics_sha256
        or readiness.get("objective_gate_spec_sha256")
        != reference.objective_gate_spec_sha256
    ):
        raise ValueError("H6-Prediction readiness/config bindings are stale")

    result = read_h6_prediction_result(
        artifact_root=artifact_root,
        readiness=typed_readiness,
    )
    metrics_bytes = payloads["validation/h6_prediction_metrics.json"]
    metrics = _canonical_mapping(metrics_bytes, "H6-Prediction metrics")
    result_payload = _canonical_mapping(
        payloads["validation/h6_prediction_result.json"],
        "H6-Prediction result",
    )
    if (
        result.status is not GateStatus.PASS
        or result.obligations
        or result.readiness_sha256 != reference.readiness_sha256
        or result.smc_bias_semantics_sha256
        != reference.smc_bias_semantics_sha256
        or result.metrics_sha256 != reference.metrics_sha256
        or hashlib.sha256(metrics_bytes).hexdigest() != reference.metrics_sha256
        or metrics.get("schema") != reference.metrics_schema
        or metrics.get("objective_gate_spec_sha256")
        != reference.objective_gate_spec_sha256
        or result_payload.get("schema_version") != reference.result_schema
        or result_payload.get("status") != "pass"
    ):
        raise ValueError("H6-Prediction raw-derived metrics/result bindings are stale")


def validate_h8_prerequisite_artifacts(
    refs: CurrentH8PrerequisiteRefs,
) -> H8PrerequisiteArtifactValidation:
    """Reopen exact referenced bytes and return one hash-bound validation.

    The registry remains the sole source of reference records. This function
    validates the immutable paths and hashes but never reconstructs a reference
    from the reopened payload and never copies predecessor bytes into H8.
    """

    if type(refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("refs must be exact CurrentH8PrerequisiteRefs")
    refs.__post_init__()
    obligations = list(refs.prerequisite_obligations)
    revalidated: list[str] = []
    try:
        _reopen_h7_compatibility_references(refs)
    except (OSError, RuntimeError, TypeError, ValueError):
        obligations.append(
            "h8_prerequisite_h7_transitive_junit_artifact_revalidation_required"
        )
    for name in ("h1_h5", "h1_prefix_prior", "h6_prefix", "h7"):
        reference = getattr(refs, name)
        try:
            _root, payloads = _reopen_reference_common(reference)
            if name == "h6_prefix":
                if type(reference) is H8H6PrefixReference:
                    _revalidate_h6_prefix_certificates(reference, payloads)
                elif type(reference) is not H8LegacyH6PrefixReference:
                    raise ValueError("H6-Prefix reference variant is not exact")
            if name == "h7":
                _read_immutable_file(
                    reference.result_pointer_path,
                    expected_sha256=reference.result_pointer_sha256,
                    name="h7_result_pointer",
                )
                _revalidate_h7_fixture_set(reference, payloads)
            revalidated.append(name)
        except (OSError, RuntimeError, TypeError, ValueError):
            obligations.append(
                f"h8_prerequisite_{name}_immutable_artifact_revalidation_required"
            )
    if type(refs.h6_prediction) is H8H6PredictionReference:
        try:
            _reopen_h6_prediction_v2(
                refs.h6_prediction,
            )
            revalidated.append("h6_prediction")
        except (OSError, RuntimeError, TypeError, ValueError):
            obligations.append(
                "h8_prerequisite_h6_prediction_v2_artifact_revalidation_required"
            )
    elif type(refs.h6_prediction) is not H8LegacyH6PredictionReference:
        raise ValueError("H6-Prediction reference variant is not exact")
    return H8PrerequisiteArtifactValidation.create(
        registry_sha256=refs.registry_sha256,
        revalidated_reference_names=tuple(revalidated),
        obligations=tuple(dict.fromkeys(obligations)),
    )


def _prerequisite_payload(
    refs: CurrentH8PrerequisiteRefs,
    *,
    prerequisite_obligations: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "h7_compatibility_refs": {
            key: _json_value(reference)
            for key, reference in refs.h7_compatibility_refs.items()
        },
        "h1_h5": _json_value(refs.h1_h5),
        "h1_prefix_prior": _json_value(refs.h1_prefix_prior),
        "h6_prefix": _json_value(refs.h6_prefix),
        "h7": _json_value(refs.h7),
        "h6_prediction": _json_value(refs.h6_prediction),
        "compatibility_checks": (
            "h7_chain_same_candidate_head",
            "h7_chain_same_candidate_dirty_digest",
            "h7_transitive_reference_bytes_equal",
            "immutable_artifacts_reopened_and_revalidated",
            "h6_prediction_frozen_scientific_dependency_closure",
            "lossless_reference_round_trip",
        ),
        "obligations": prerequisite_obligations,
        "all_current_and_pass": not prerequisite_obligations,
    }


def h8_current_refs_registry_payload(
    refs: CurrentH8PrerequisiteRefs,
) -> dict[str, object]:
    """Serialize the deterministic registry preimage without a self-hash."""

    if type(refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("refs must be an exact CurrentH8PrerequisiteRefs")
    refs.__post_init__()
    return {
        "schema_version": refs.registry_schema_version,
        "candidate": {
            "git_head": refs.candidate_head,
            "dirty_digest": refs.candidate_dirty_digest,
            "junit_sha256": refs.candidate_junit_sha256,
        },
        "h7_compatibility_refs": {
            key: _json_value(reference)
            for key, reference in refs.h7_compatibility_refs.items()
        },
        "references": {
            name: _json_value(getattr(refs, name))
            for name in H8_POINTER_PREDECESSOR_KEYS
        },
    }


def _correctness_cell_payload(item: H8CorrectnessCell) -> dict[str, object]:
    """Project one typed cell to the exact flattened Task-7 wire schema."""

    layout = item.layout
    return {
        "T": layout.horizon,
        "d_z": layout.d_z,
        "d_m": layout.d_m,
        "N": layout.population_size,
        "b": layout.block_size,
        "D": layout.total_dimension,
        "problem_seed": item.problem_seed,
        "sample_noise_seed": item.sample_noise_seed,
        "problem_sha256": item.problem_sha256,
        "sample_noise_sha256": item.sample_noise_sha256,
        "source_results": tuple(_json_value(value) for value in item.source_results),
        "pair_comparisons": tuple(
            _json_value(value) for value in item.pair_comparisons
        ),
        "wrong_path_controls": tuple(
            _json_value(value) for value in item.wrong_path_controls
        ),
        "status": item.status.value,
        "obligations": item.obligations,
    }


def _correctness_payload(result: H8GateResult) -> dict[str, object]:
    cells = tuple(_correctness_cell_payload(item) for item in result.correctness)
    return {
        "grid_order": tuple(item.cell_id for item in result.correctness),
        "cells": cells,
        "cell_count": len(cells),
        "all_complete": len(cells) == len(H8_CORRECTNESS_CASES),
        "all_decisive": bool(cells)
        and all(
            control.decisive
            for cell in result.correctness
            for control in cell.wrong_path_controls
        ),
        "all_pass": bool(cells)
        and all(item.status is GateStatus.PASS for item in result.correctness),
    }


def _attempt_payload(item: H8ChildAttemptRecord) -> dict[str, object]:
    """Serialize one parent-owned launch without claiming a missing child result."""

    if type(item.result) is H8ChildResult:
        result_kind: str | None = "child"
        result_identity: dict[str, object] | None = {
            "mode": item.result.mode,
            "seed": item.result.seed,
            "repetition": item.result.repetition,
            "input_sha256": item.result.input_sha256,
        }
    elif type(item.result) is H8ControlResult:
        result_kind = "control"
        result_identity = {
            "control_id": item.result.control_id,
            "event_sha256": item.result.event_sha256,
        }
    else:
        result_kind = None
        result_identity = None
    return {
        "request": _json_value(item.request),
        "status": item.status.value,
        "reasons": item.reasons,
        "result_kind": result_kind,
        "result_identity": result_identity,
        "nonpass_envelope": (
            None
            if item.nonpass_envelope is None
            else _json_value(item.nonpass_envelope)
        ),
        "timed_out": item.timed_out,
        "exit_code": item.exit_code,
        "parent_elapsed_ns": item.parent_elapsed_ns,
        "request_sha256": item.request_sha256,
        "identities_sha256": item.identities_sha256,
        "stdout_sha256": item.stdout_sha256,
        "stderr_sha256": item.stderr_sha256,
        "operation_reachability": (
            None
            if item.operation_reachability is None
            else _json_value(item.operation_reachability)
        ),
        "residuals": (
            None if item.residuals is None else _json_value(item.residuals)
        ),
        "resource_decisions": (
            None
            if item.resource_decisions is None
            else _json_value(item.resource_decisions)
        ),
    }


def _child_payload(
    item: H8ChildResult,
    attempt: H8ChildAttemptRecord,
) -> dict[str, object]:
    """Merge child-authored endpoints with separately witnessed parent facts."""

    if attempt.result != item:
        raise ValueError("H8 child result is not bound to its parent attempt")

    return {
        "mode": item.mode,
        "seed": item.seed,
        "repetition": item.repetition,
        "input_sha256": item.input_sha256,
        "objective": _json_value(item.objective),
        "storage": _json_value(item.storage),
        "fill": _json_value(item.fill),
        "workspace": _json_value(item.workspace),
        "counters": _json_value(item.counters),
        "allocation": _json_value(item.allocation),
        "resources": _json_value(item.resources),
        "invariants": tuple(_json_value(value) for value in item.invariants),
        "parent_elapsed_ns": attempt.parent_elapsed_ns,
        "child_elapsed_ns": item.resources.child_elapsed_ns,
        "exit_code": attempt.exit_code,
        "stdout_sha256": attempt.stdout_sha256,
        "stderr_sha256": attempt.stderr_sha256,
        "operation_reachability": _json_value(
            attempt.operation_reachability
        ),
        "residuals": _json_value(attempt.residuals),
        "resource_decisions": _json_value(attempt.resource_decisions),
    }


def _evaluation_sha256(
    *,
    result: H8GateResult,
    current_refs: CurrentH8PrerequisiteRefs,
    prerequisite_obligations: tuple[str, ...],
    runtime_sections: Mapping[str, object] | None,
    validation_payload_sha256: str,
    dependency_closure_sha256: str,
    preregistration_sha256: str,
) -> str:
    preimage = canonical_h8_json_bytes(
        {
            "domain": "vfe4.h8.gate-evaluation.v1",
            "result": result,
            "current_refs": current_refs,
            "prerequisite_obligations": prerequisite_obligations,
            "runtime_sections": runtime_sections,
            "validation_payload_sha256": validation_payload_sha256,
            "dependency_closure_sha256": dependency_closure_sha256,
            "preregistration_sha256": preregistration_sha256,
            "interpretation_sha256": H8_INTERPRETATION_SHA256,
        }
    )
    return hashlib.sha256(preimage).hexdigest()


def _validation_payload_from_context(
    *,
    result: H8GateResult,
    current_refs: CurrentH8PrerequisiteRefs,
    prerequisite_obligations: tuple[str, ...],
    runtime_sections: Mapping[str, object] | None,
    dependency_closure_sha256: str,
    preregistration_sha256: str,
) -> dict[str, object]:
    """Reconstruct every validation field from independently retained context."""

    if type(result) is not H8GateResult:
        raise ValueError("H8 validation context requires an exact typed result")
    if type(current_refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("H8 validation context requires exact current references")
    result.__post_init__()
    current_refs.__post_init__()
    _sha256(dependency_closure_sha256, "dependency_closure_sha256")
    _sha256(preregistration_sha256, "preregistration_sha256")
    if (
        type(prerequisite_obligations) is not tuple
        or any(
            type(item) is not str or not item
            for item in prerequisite_obligations
        )
        or len(set(prerequisite_obligations)) != len(prerequisite_obligations)
        or any(
            item not in prerequisite_obligations
            for item in current_refs.prerequisite_obligations
        )
    ):
        raise ValueError("H8 prerequisite obligations are not exact")
    if (
        result.candidate_junit_sha256
        != current_refs.candidate_junit_sha256
        or result.current_refs_registry_sha256
        != current_refs.registry_sha256
        or result.h7_manifest_sha256
        != current_refs.h7.manifest_sha256
        or result.h6_prediction_manifest_sha256
        != current_refs.h6_prediction.manifest_sha256
    ):
        raise ValueError("H8 result is not bound to its retained current references")
    if (
        result.status is not GateStatus.FAIL
        and (
            any(
                item not in result.obligations
                for item in prerequisite_obligations
            )
            or H8_SOURCE_ONLY_OBLIGATIONS[5] not in result.obligations
        )
    ):
        raise ValueError(
            "H8 result does not retain its prerequisite/orchestrator obligations"
        )

    prerequisites_current_and_pass = not prerequisite_obligations
    if runtime_sections is None:
        if (
            result.status is not GateStatus.FAIL
            and H8_SOURCE_ONLY_OBLIGATIONS[4] not in result.obligations
        ):
            raise ValueError(
                "source-only H8 context must retain its runtime-section obligation"
            )
        sections = _source_only_sections(
            result=result,
            refs=current_refs,
            dependency_closure_sha256=dependency_closure_sha256,
            preregistration_sha256=preregistration_sha256,
            prerequisites_current_and_pass=prerequisites_current_and_pass,
        )
    else:
        sections = _validate_runtime_sections(runtime_sections)
    _validate_section_bindings(
        sections,
        result=result,
        refs=current_refs,
        dependency_closure_sha256=dependency_closure_sha256,
        preregistration_sha256=preregistration_sha256,
        prerequisites_current_and_pass=prerequisites_current_and_pass,
    )

    production_attempts = tuple(
        attempt
        for attempt in result.child_attempts
        if attempt.request.mode == "production"
        and type(attempt.result) is H8ChildResult
    )
    profiler_attempts = tuple(
        attempt
        for attempt in result.child_attempts
        if attempt.request.mode == "profiler"
        and type(attempt.result) is H8ChildResult
    )
    payload = {
        "schema_version": H8_VALIDATION_SCHEMA,
        "gate": "H8",
        "status": result.status.value,
        "obligations": result.obligations,
        "bounded_claim": (
            H8_BOUNDED_CLAIM
            if result.status is GateStatus.PASS
            else f"NOT ESTABLISHED: {H8_BOUNDED_CLAIM}"
        ),
        "nonclaims": H8_NONCLAIMS,
        "revision": sections["revision"],
        "config": sections["config"],
        "prerequisites": _prerequisite_payload(
            current_refs,
            prerequisite_obligations=prerequisite_obligations,
        ),
        "interpretation": sections["interpretation"],
        "protocol": sections["protocol"],
        "environment": sections["environment"],
        "problems": sections["problems"],
        "storage": sections["storage"],
        "factor": sections["factor"],
        "correctness": _correctness_payload(result),
        "allocation": sections["allocation"],
        "controls": tuple(_json_value(item) for item in result.controls),
        "child_attempts": tuple(
            _attempt_payload(item) for item in result.child_attempts
        ),
        "production_runs": tuple(
            _child_payload(item, attempt)
            for item, attempt in zip(
                result.production_runs,
                production_attempts,
                strict=True,
            )
        ),
        "profiler_runs": tuple(
            _child_payload(item, attempt)
            for item, attempt in zip(
                result.profiler_runs,
                profiler_attempts,
                strict=True,
            )
        ),
        "budgets": sections["budgets"],
        "invariants": sections["invariants"],
        "artifacts": sections["artifacts"],
    }
    if tuple(payload) != H8_VALIDATION_TOP_LEVEL_KEYS:
        raise RuntimeError("internal H8 payload inventory drifted")
    return payload


def assemble_h8_gate_evaluation(
    *,
    config_sha256: str,
    current_refs: CurrentH8PrerequisiteRefs,
    correctness: tuple[H8CorrectnessCell, ...],
    production_runs: tuple[H8ChildResult, ...],
    profiler_runs: tuple[H8ChildResult, ...],
    controls: tuple[H8ControlResult, ...],
    child_attempts: tuple[H8ChildAttemptRecord, ...] = (),
    dependency_closure_sha256: str,
    preregistration_sha256: str,
    additional_obligations: tuple[str, ...] = (),
    prerequisite_validation: H8PrerequisiteArtifactValidation | None = None,
    runtime_sections: Mapping[str, object] | None = None,
) -> H8GateEvaluation:
    """Freeze typed evidence; never launch, retry, or synthesize a child run."""

    if type(current_refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("current_refs must be an exact bound H8 registry")
    current_refs.__post_init__()
    _sha256(config_sha256, "config_sha256")
    _sha256(dependency_closure_sha256, "dependency_closure_sha256")
    _sha256(preregistration_sha256, "preregistration_sha256")
    _typed_records(correctness, H8CorrectnessCell, "correctness")
    _typed_records(child_attempts, H8ChildAttemptRecord, "child_attempts")
    _typed_records(production_runs, H8ChildResult, "production_runs")
    _typed_records(profiler_runs, H8ChildResult, "profiler_runs")
    _typed_records(controls, H8ControlResult, "controls")
    if (
        type(additional_obligations) is not tuple
        or any(type(item) is not str or not item for item in additional_obligations)
    ):
        raise ValueError("additional_obligations must be nonempty strings")
    if prerequisite_validation is None:
        validated_obligations = (
            "h8_prerequisite_artifact_revalidation_not_supplied",
        )
    else:
        if type(prerequisite_validation) is not H8PrerequisiteArtifactValidation:
            raise ValueError("prerequisite validation must retain its exact type")
        prerequisite_validation.__post_init__()
        if prerequisite_validation.registry_sha256 != current_refs.registry_sha256:
            raise ValueError("prerequisite validation belongs to another registry")
        validated_obligations = prerequisite_validation.obligations
        if (
            not validated_obligations
            and prerequisite_validation.revalidated_reference_names
            != H8_POINTER_PREDECESSOR_KEYS
        ):
            raise ValueError(
                "successful prerequisite validation must reopen every reference"
            )
    bound_prerequisite_obligations = tuple(
        dict.fromkeys(
            (
                *current_refs.prerequisite_obligations,
                *validated_obligations,
            )
        )
    )
    complete = _inventory_complete(
        correctness,
        child_attempts,
        production_runs,
        profiler_runs,
        controls,
    )
    obligations = [
        *bound_prerequisite_obligations,
        *additional_obligations,
    ]
    if not complete:
        if tuple(item.cell_id for item in correctness) != tuple(
            range(1, len(H8_CORRECTNESS_CASES) + 1)
        ):
            obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[0])
        if tuple((item.seed, item.repetition) for item in production_runs) != tuple(
            (seed, repetition)
            for seed in H8_PRODUCTION_SEEDS
            for repetition in range(5)
        ):
            obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[1])
        if tuple(item.seed for item in profiler_runs) != H8_PRODUCTION_SEEDS:
            obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[2])
        if tuple(item.control_id for item in controls) != H8_NEGATIVE_CONTROL_IDS:
            obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[3])
    if runtime_sections is None:
        obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[4])
    # The v3 attempt schema retains parent evidence, but this source slice does
    # not yet own the runtime orchestrator/protocol/environment revalidation
    # required to authorize PASS. Witnessed FAIL still dominates this blocker.
    obligations.append(H8_SOURCE_ONLY_OBLIGATIONS[5])
    obligations = list(dict.fromkeys(obligations))
    statuses = _retained_statuses(
        correctness,
        child_attempts,
        production_runs,
        profiler_runs,
        controls,
    )
    status = classify_h8_status(
        retained_statuses=statuses,
        exact_inventory_complete=complete,
        open_obligations=tuple(obligations),
    )
    final_obligations = () if status is GateStatus.FAIL else tuple(obligations)
    result = H8GateResult(
        gate="H8",
        status=status,
        config_sha256=config_sha256,
        candidate_junit_sha256=current_refs.candidate_junit_sha256,
        current_refs_registry_sha256=current_refs.registry_sha256,
        h7_manifest_sha256=current_refs.h7.manifest_sha256,
        h6_prediction_manifest_sha256=(
            current_refs.h6_prediction.manifest_sha256
        ),
        correctness=correctness,
        child_attempts=child_attempts,
        production_runs=production_runs,
        profiler_runs=profiler_runs,
        controls=controls,
        obligations=final_obligations,
    )
    retained_runtime_sections = (
        None
        if runtime_sections is None
        else _validate_runtime_sections(runtime_sections)
    )
    payload = _validation_payload_from_context(
        result=result,
        current_refs=current_refs,
        prerequisite_obligations=bound_prerequisite_obligations,
        runtime_sections=retained_runtime_sections,
        dependency_closure_sha256=dependency_closure_sha256,
        preregistration_sha256=preregistration_sha256,
    )
    payload_bytes = canonical_h8_json_bytes(payload)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    return H8GateEvaluation(
        result=result,
        current_refs=current_refs,
        prerequisite_obligations=bound_prerequisite_obligations,
        runtime_sections=retained_runtime_sections,
        validation_payload_canonical_json=payload_bytes,
        validation_payload_sha256=payload_sha256,
        dependency_closure_sha256=dependency_closure_sha256,
        preregistration_sha256=preregistration_sha256,
        interpretation_sha256=H8_INTERPRETATION_SHA256,
        evaluation_sha256=_evaluation_sha256(
            result=result,
            current_refs=current_refs,
            prerequisite_obligations=bound_prerequisite_obligations,
            runtime_sections=retained_runtime_sections,
            validation_payload_sha256=payload_sha256,
            dependency_closure_sha256=dependency_closure_sha256,
            preregistration_sha256=preregistration_sha256,
        ),
    )


def h8_validation_payload(evaluation: H8GateEvaluation) -> dict[str, object]:
    """Revalidate evaluation identity and return the plan-ordered payload."""

    if type(evaluation) is not H8GateEvaluation:
        raise ValueError("evaluation must be an exact H8GateEvaluation")
    evaluation.__post_init__()
    payload_bytes = evaluation.validation_payload_canonical_json
    if hashlib.sha256(payload_bytes).hexdigest() != evaluation.validation_payload_sha256:
        raise ValueError("H8 validation payload hash differs from its bytes")
    parsed = json.loads(payload_bytes)
    if not isinstance(parsed, dict) or set(parsed) != set(H8_VALIDATION_TOP_LEVEL_KEYS):
        raise ValueError("H8 validation payload has the wrong top-level schema")
    result = cast(H8GateResult, evaluation.result)
    if type(result) is not H8GateResult:
        raise ValueError("H8 validation payload requires an exact typed result")
    result.__post_init__()
    if (
        parsed.get("schema_version") != H8_VALIDATION_SCHEMA
        or parsed.get("gate") != "H8"
        or parsed.get("status") != result.status.value
        or tuple(parsed.get("obligations", ())) != result.obligations
        or parsed.get("nonclaims") != list(H8_NONCLAIMS)
    ):
        raise ValueError("H8 validation payload is not bound to its result")
    expected_payload = _json_value(
        _validation_payload_from_context(
            result=result,
            current_refs=evaluation.current_refs,
            prerequisite_obligations=evaluation.prerequisite_obligations,
            runtime_sections=evaluation.runtime_sections,
            dependency_closure_sha256=evaluation.dependency_closure_sha256,
            preregistration_sha256=evaluation.preregistration_sha256,
        )
    )
    if not isinstance(expected_payload, dict):
        raise RuntimeError("internal H8 context reconstruction is not a mapping")
    expected_inventories = {
        name: expected_payload[name]
        for name in (
            "correctness",
            "controls",
            "child_attempts",
            "production_runs",
            "profiler_runs",
        )
    }
    if any(
        parsed.get(name) != expected
        for name, expected in expected_inventories.items()
    ):
        raise ValueError(
            "H8 validation inventories are not bound to the typed result"
        )
    if parsed != expected_payload:
        raise ValueError(
            "H8 validation payload differs from its exact context reconstruction"
        )
    expected_evaluation_sha256 = _evaluation_sha256(
        result=result,
        current_refs=evaluation.current_refs,
        prerequisite_obligations=evaluation.prerequisite_obligations,
        runtime_sections=evaluation.runtime_sections,
        validation_payload_sha256=evaluation.validation_payload_sha256,
        dependency_closure_sha256=evaluation.dependency_closure_sha256,
        preregistration_sha256=evaluation.preregistration_sha256,
    )
    if evaluation.evaluation_sha256 != expected_evaluation_sha256:
        raise ValueError("H8 evaluation hash differs from its complete identity")
    return {key: parsed[key] for key in H8_VALIDATION_TOP_LEVEL_KEYS}


def build_h8_publication_payloads(
    config: object,
    evaluation: H8GateEvaluation,
    *,
    h7_reference: H8H7Reference,
    h6_prediction_reference: (
        H8H6PredictionReference | H8LegacyH6PredictionReference
    ),
    provenance: Mapping[str, object] | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build exactly the six JSON payloads; the generic publisher adds manifest."""

    canonical_json = getattr(config, "canonical_json", None)
    config_sha256 = getattr(config, "config_sha256", None)
    refs = getattr(config, "h8_current_refs", None)
    if (
        type(canonical_json) is not str
        or type(config_sha256) is not str
        or hashlib.sha256(canonical_json.encode("utf-8")).hexdigest() != config_sha256
        or type(refs) is not CurrentH8PrerequisiteRefs
        or type(evaluation) is not H8GateEvaluation
        or evaluation.current_refs != refs
        or evaluation.result.config_sha256 != config_sha256
        or h7_reference != refs.h7
        or h6_prediction_reference != refs.h6_prediction
    ):
        raise ValueError("H8 publication inputs are not bound to one exact config")
    try:
        config_payload = json.loads(canonical_json)
    except json.JSONDecodeError as exc:
        raise ValueError("H8 canonical config is not JSON") from exc
    if canonical_h8_json_bytes(config_payload) != canonical_json.encode("utf-8"):
        raise ValueError("H8 config bytes are not canonical")
    validation_payload = h8_validation_payload(evaluation)
    payloads = {
        "config.json": config_payload,
        "environment.json": dict(environment)
        if environment is not None
        else cast(Mapping[str, object], validation_payload["environment"]),
        "provenance.json": dict(provenance)
        if provenance is not None
        else {
            "schema_version": "vfe4-h8-provenance-v1",
            "git_head": refs.candidate_head,
            "dirty_digest": refs.candidate_dirty_digest,
            "config_sha256": config_sha256,
            "junit_sha256": refs.candidate_junit_sha256,
            "current_refs_registry_sha256": refs.registry_sha256,
            "validation_sha256": evaluation.validation_payload_sha256,
            "status": evaluation.result.status.value,
            "obligations": evaluation.result.obligations,
            "evaluation_sha256": evaluation.evaluation_sha256,
        },
        "references/h6_prediction.json": _json_value(h6_prediction_reference),
        "references/h7.json": _json_value(h7_reference),
        "validation/h8.json": validation_payload,
    }
    if tuple(payloads) != H8_PUBLICATION_PAYLOAD_KEYS:
        raise RuntimeError("internal H8 publication inventory drifted")
    return payloads


def _manifest_entries(manifest_bytes: bytes) -> tuple[tuple[str, str], ...]:
    try:
        text = manifest_bytes.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ValueError("H8 artifact manifest is not strict ASCII") from exc
    if not text.endswith("\n"):
        raise ValueError("H8 artifact manifest must end with one newline")
    entries: list[tuple[str, str]] = []
    for expected_name, line in zip(
        H8_PUBLICATION_PAYLOAD_KEYS,
        text.splitlines(),
        strict=True,
    ):
        digest, separator, name = line.partition("  ")
        if separator != "  " or name != expected_name:
            raise ValueError("H8 artifact manifest inventory is not exact")
        entries.append((name, _sha256(digest, f"manifest[{name!r}]")))
    return tuple(entries)


def _revalidate_h8_published_artifact(
    artifact: CandidateArtifactReference,
) -> Path:
    """Re-read every published byte before an external pointer can be built."""

    if type(artifact) is not CandidateArtifactReference:
        raise ValueError("artifact must be an exact CandidateArtifactReference")
    artifact.__post_init__()
    if artifact.artifact_path.is_symlink():
        raise ValueError("H8 artifact directory cannot be a symlink")
    try:
        root = artifact.artifact_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("H8 artifact directory is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise ValueError("H8 artifact path must be a real directory")
    manifest_path = root / "manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("H8 artifact lacks a regular manifest")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact.manifest_sha256:
        raise ValueError("H8 artifact manifest hash differs from its reference")
    entries = _manifest_entries(manifest_bytes)
    if dict(entries) != dict(artifact.payload_hashes):
        raise ValueError("H8 artifact manifest differs from its payload reference")

    descendants = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("H8 artifact cannot contain symlinks")
    observed_files = {
        path.relative_to(root).as_posix()
        for path in descendants
        if path.is_file()
    }
    if observed_files != {*H8_PUBLICATION_PAYLOAD_KEYS, "manifest.sha256"}:
        raise ValueError("H8 artifact contains an unlisted or missing payload")
    for name, expected_digest in entries:
        relative = PurePosixPath(name)
        path = root / Path(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("H8 manifest names an invalid payload")
        payload_bytes = path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != expected_digest:
            raise ValueError(f"H8 payload hash mismatch: {name}")
        try:
            parsed = json.loads(payload_bytes.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"H8 payload is not strict JSON: {name}") from exc
        if canonical_h8_json_bytes(parsed) != payload_bytes:
            raise ValueError(f"H8 payload is not canonical JSON: {name}")
    return root


def _read_canonical_payload(root: Path, name: str) -> object:
    path = root / Path(*PurePosixPath(name).parts)
    payload_bytes = path.read_bytes()
    parsed = json.loads(payload_bytes.decode("utf-8", errors="strict"))
    if canonical_h8_json_bytes(parsed) != payload_bytes:
        raise ValueError(f"H8 payload is not canonical JSON: {name}")
    return parsed


def _revalidate_h8_artifact_semantics(
    root: Path,
    *,
    config_sha256: str,
    validation_sha256: str,
    junit_sha256: str,
    current_refs: CurrentH8PrerequisiteRefs,
    evaluation: H8GateEvaluation,
    source_sha256: str,
    registry_path: Path,
) -> None:
    config = _read_canonical_payload(root, "config.json")
    provenance = _read_canonical_payload(root, "provenance.json")
    environment = _read_canonical_payload(root, "environment.json")
    h7_reference = _read_canonical_payload(root, "references/h7.json")
    h6_prediction_reference = _read_canonical_payload(
        root,
        "references/h6_prediction.json",
    )
    validation = _read_canonical_payload(root, "validation/h8.json")
    if not isinstance(config, Mapping):
        raise ValueError("H8 config payload must be a mapping")
    validation_config = config.get("validation")
    h8_config = config.get("h8")
    if (
        hashlib.sha256(canonical_h8_json_bytes(config)).hexdigest()
        != config_sha256
        or not isinstance(validation_config, Mapping)
        or tuple(validation_config.get("gates", ())) != H8_VERIFIER_PREFIX
        or not isinstance(h8_config, Mapping)
        or h8_config.get("h7_plan_sha256") != H8_H7_PLAN_SHA256
        or h8_config.get("interpretation_sha256") != H8_INTERPRETATION_SHA256
        or config.get("h8_current_refs") != _json_value(current_refs)
    ):
        raise ValueError("H8 config payload is not the bound selected operation")
    if (
        h7_reference != _json_value(current_refs.h7)
        or h6_prediction_reference != _json_value(current_refs.h6_prediction)
        or evaluation.current_refs != current_refs
    ):
        raise ValueError("H8 artifact reference payloads are not lossless")
    if (
        not isinstance(validation, Mapping)
        or set(validation) != set(H8_VALIDATION_TOP_LEVEL_KEYS)
        or hashlib.sha256(canonical_h8_json_bytes(validation)).hexdigest()
        != validation_sha256
        or validation.get("schema_version") != H8_VALIDATION_SCHEMA
        or validation.get("gate") != "H8"
        or validation.get("nonclaims") != list(H8_NONCLAIMS)
    ):
        raise ValueError("validation/h8.json is not the exact H8 schema")
    expected_validation = h8_validation_payload(evaluation)
    validation_obligations = validation.get("obligations")
    if (
        validation != expected_validation
        or not isinstance(validation_obligations, list)
        or validation.get("environment") != environment
    ):
        raise ValueError("H8 validation identities do not match artifact inputs")
    reference_registry = (
        provenance.get("reference_registry")
        if isinstance(provenance, Mapping)
        else None
    )
    started_utc = provenance.get("started_utc") if isinstance(provenance, Mapping) else None
    ended_utc = provenance.get("ended_utc") if isinstance(provenance, Mapping) else None
    try:
        started = _strict_utc_timestamp(started_utc, "started_utc")
        ended = _strict_utc_timestamp(ended_utc, "ended_utc")
    except ValueError as exc:
        raise ValueError("H8 provenance timestamps are invalid") from exc
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != set(H8_PROVENANCE_KEYS)
        or provenance.get("schema_version") != "vfe4-h8-provenance-v1"
        or provenance.get("git_head") != current_refs.candidate_head
        or provenance.get("dirty_digest") != current_refs.candidate_dirty_digest
        or provenance.get("dirty_content_digest")
        != current_refs.candidate_dirty_digest
        or provenance.get("source_sha256") != source_sha256
        or source_sha256
        != source_candidate_sha256(
            git_head_value=current_refs.candidate_head,
            dirty_digest_value=current_refs.candidate_dirty_digest,
        )
        or provenance.get("config_sha256") != config_sha256
        or provenance.get("junit_sha256") != junit_sha256
        or provenance.get("current_refs_registry_sha256")
        != current_refs.registry_sha256
        or reference_registry
        != {
            "path": registry_path.resolve(strict=False).as_posix(),
            "sha256": current_refs.registry_sha256,
        }
        or provenance.get("dependency_closure_sha256")
        != evaluation.dependency_closure_sha256
        or provenance.get("preregistration_sha256")
        != evaluation.preregistration_sha256
        or provenance.get("interpretation_sha256")
        != evaluation.interpretation_sha256
        or provenance.get("validation_sha256") != validation_sha256
        or provenance.get("evaluation_sha256") != evaluation.evaluation_sha256
        or provenance.get("status") != validation.get("status")
        or provenance.get("obligations") != validation_obligations
        or provenance.get("selected_operation") != "H8"
        or tuple(cast(object, provenance.get("ordered_gates", ())))
        != H8_VERIFIER_PREFIX
        or provenance.get("execution_scope")
        != (
            "h8-parent-orchestrated-runtime-v1"
            if evaluation.result.child_attempts
            else "source-only-empty-runtime-records"
        )
        or provenance.get("external_pointer_in_artifact") is not False
        or started > ended
    ):
        raise ValueError("H8 provenance is not bound to validation and candidate")


def _strict_utc_timestamp(value: object, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UTC timestamp") from exc
    canonical = parsed.astimezone(timezone.utc).isoformat(
        timespec="microseconds",
    ).replace("+00:00", "Z")
    if parsed.tzinfo is None or canonical != value:
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    return parsed


def _validate_captured_h8_registry(
    *,
    repo_root: Path,
    artifact_head: str,
    current_refs: CurrentH8PrerequisiteRefs,
    registry_path: Path,
    registry_bytes: bytes,
) -> Path:
    if not isinstance(registry_path, Path):
        raise ValueError("registry_path must be a Path")
    if type(registry_bytes) is not bytes or not registry_bytes:
        raise ValueError("registry_bytes must be the nonempty captured bytes")
    expected_refs_root = (repo_root.resolve(strict=False) / ".verification").resolve(
        strict=False
    )
    expected_path = (
        expected_refs_root
        / f"h8-current-candidate-{artifact_head}-refs.json"
    ).resolve(strict=False)
    observed_path = registry_path.resolve(strict=False)
    if (
        observed_path != expected_path
        or observed_path.parent != expected_refs_root
        or hashlib.sha256(registry_bytes).hexdigest()
        != current_refs.registry_sha256
        or registry_bytes
        != canonical_h8_json_bytes(h8_current_refs_registry_payload(current_refs))
    ):
        raise ValueError("captured current-candidate H8 registry is stale or lossy")
    return observed_path


def h8_current_candidate_result_payload(
    artifact: CandidateArtifactReference,
    *,
    repo_root: Path,
    config_sha256: str,
    validation_sha256: str,
    junit_sha256: str,
    current_refs: CurrentH8PrerequisiteRefs,
    evaluation: H8GateEvaluation,
    source_sha256: str,
    registry_path: Path,
    registry_bytes: bytes,
) -> dict[str, object]:
    """Build the external pointer from a hash-complete published reference."""

    if not isinstance(repo_root, Path):
        raise ValueError("repo_root must be a Path")
    if type(current_refs) is not CurrentH8PrerequisiteRefs:
        raise ValueError("current_refs must be an exact bound H8 registry")
    if type(evaluation) is not H8GateEvaluation:
        raise ValueError("evaluation must be an exact H8GateEvaluation")
    artifact_root = _revalidate_h8_published_artifact(artifact)
    current_refs.__post_init__()
    evaluation.__post_init__()
    _sha256(config_sha256, "config_sha256")
    _sha256(validation_sha256, "validation_sha256")
    _sha256(junit_sha256, "junit_sha256")
    _sha256(source_sha256, "source_sha256")
    refs_path = _validate_captured_h8_registry(
        repo_root=repo_root,
        artifact_head=artifact.git_head,
        current_refs=current_refs,
        registry_path=registry_path,
        registry_bytes=registry_bytes,
    )
    if (
        artifact.git_head != current_refs.candidate_head
        or artifact.dirty_digest != current_refs.candidate_dirty_digest
        or junit_sha256 != current_refs.candidate_junit_sha256
        or config_sha256 != evaluation.result.config_sha256
        or validation_sha256 != evaluation.validation_payload_sha256
        or junit_sha256 != evaluation.result.candidate_junit_sha256
        or current_refs.registry_sha256
        != evaluation.result.current_refs_registry_sha256
        or evaluation.current_refs != current_refs
        or set(artifact.payload_hashes) != set(H8_PUBLICATION_PAYLOAD_KEYS)
        or artifact.payload_hashes["config.json"] != config_sha256
        or artifact.payload_hashes["validation/h8.json"] != validation_sha256
        or artifact.payload_hashes["references/h7.json"]
        != hashlib.sha256(canonical_h8_json_bytes(current_refs.h7)).hexdigest()
        or artifact.payload_hashes["references/h6_prediction.json"]
        != hashlib.sha256(
            canonical_h8_json_bytes(current_refs.h6_prediction)
        ).hexdigest()
    ):
        raise ValueError("published H8 reference is not bound to current inputs")
    _revalidate_h8_artifact_semantics(
        artifact_root,
        config_sha256=config_sha256,
        validation_sha256=validation_sha256,
        junit_sha256=junit_sha256,
        current_refs=current_refs,
        evaluation=evaluation,
        source_sha256=source_sha256,
        registry_path=refs_path,
    )
    predecessors = {
        name: _json_value(getattr(current_refs, name))
        for name in H8_POINTER_PREDECESSOR_KEYS
    }
    return {
        "schema_version": H8_CURRENT_CANDIDATE_RESULT_SCHEMA,
        "candidate": {
            "git_head": artifact.git_head,
            "dirty_digest": artifact.dirty_digest,
            "junit_sha256": junit_sha256,
        },
        "artifact": {
            "path": artifact_root.as_posix(),
            "manifest_sha256": artifact.manifest_sha256,
            "config_sha256": config_sha256,
            "validation_sha256": validation_sha256,
        },
        "current_refs": {
            "path": refs_path.as_posix(),
            "sha256": current_refs.registry_sha256,
        },
        "predecessors": predecessors,
    }


__all__ = [
    "H8_BOUNDED_CLAIM",
    "H8_CURRENT_CANDIDATE_RESULT_SCHEMA",
    "H8_POINTER_PREDECESSOR_KEYS",
    "H8PrerequisiteArtifactValidation",
    "H8_PUBLICATION_PAYLOAD_KEYS",
    "H8_SOURCE_ONLY_OBLIGATIONS",
    "H8_VALIDATION_SCHEMA",
    "H8_VALIDATION_TOP_LEVEL_KEYS",
    "assemble_h8_gate_evaluation",
    "build_h8_publication_payloads",
    "canonical_h8_json_bytes",
    "classify_h8_status",
    "h8_current_candidate_result_payload",
    "h8_current_refs_registry_payload",
    "h8_validation_payload",
    "make_h8_preflight_inconclusive",
    "validate_h8_prerequisite_artifacts",
]
