"""Report-bound composition and atomic publication for the independent H6-Prefix gate.

The public runner performs a four-case focused check by default. Full Prefix
enumeration remains behind exact authorization and a complete frozen
validation fixture. Report composition and publication revalidate their owned
identities and emit only the independent six-file H6-Prefix artifact.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import torch

from verification.numpy_oracles.h6_prefix import enumerate_ordered_tail_pairs
from verification.h6_validation_candidate import (
    H6ValidationPerturbationArtifactPayload,
    load_h6_validation_perturbation_artifact_payload,
)
from vfe4.artifacts.atomic import publish_run_directory
from vfe4.artifacts.provenance import (
    current_source_identity,
    source_candidate_sha256,
)
from vfe4.config import resolve_h6_prefix_config
from vfe4.config.schema import (
    H6_PREFIX_V2_AUTHORIZATION_SHA256,
    H6PrefixResolvedConfig,
    H6PrefixV3ResolvedConfig,
)
from vfe4.h6_validation_fixture import (
    ValidationSafetyFixturePayload,
    read_validation_safety_fixture_payload,
)
from vfe4.data.windows import CausalPrefix
from vfe4.numerics.categorical import masked_log_softmax_from_parents
from vfe4.predictive import (
    BootstrapSmcPredictor,
    EstimatorIdentity,
    PriorPrediction,
    vocabulary_identity_sha256,
)
from vfe4.training.arms import BuiltArm, LatentLanguageArmModel, build_arm
from vfe4.types.h6 import (
    ArmConfig,
    BoundedPrefixCertificate,
    BoundedPrefixCertificateSet,
    BoundedPrefixReportBinding,
    BoundedPrefixReportReference,
    EstimatorSpec,
    H6_PREFIX_REQUIRED_CHECKS,
    EvidenceStatus,
    H6LanguageStructure,
    H6PrefixProfilePair,
    H6PrefixWorkloadPlan,
    PrefixCaseKey,
    PrefixCertificate,
    PrefixReportBinding,
    ValidationSafetyFixture,
    VocabularyIdentity,
    canonical_json_bytes,
)
from vfe4.types.results import (
    H6BoundedPrefixGateResult,
    H6PrefixGateResult,
)
from vfe4.validation.h6_prefix import (
    SMALL_EXPECTED_BY_POSITION,
    SMALL_EXPECTED_TOTAL,
    VALIDATION_EXPECTED_TOTAL,
    DynamicCheckResult,
    DynamicExecutionPlan,
    DynamicPrefixCase,
    DynamicPrefixReport,
    FrozenValidationPerturbations,
    PairSideHarness,
    AllInvalidSourceObservation,
    SourceMaskObservation,
    SourceMaskObserver,
    load_frozen_validation_perturbations,
    observe_all_invalid_source_rejection,
    run_dynamic_prefix_checks,
)
from vfe4.validation.h6_static_audit import (
    StaticAuditCheck,
    StaticAuditFinding,
    StaticAuditReport,
    audit_h6_static_source,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FOCUSED_SMALL_CASE_INDICES = (0, 6561, 8748, 9477)
_FULL_PREFIX_AUTHORIZATION_SHA256 = hashlib.sha256(
    b"AUTHORIZE_VFE4_H6_PREFIX_FULL_INVENTORIES_V1"
).hexdigest()
_RESOLVED_PREFIX_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "source",
        "execution_mode",
        "profiles",
        "authorization_sha256",
        "artifact_root",
    }
)
_RESOLVED_PREFIX_CONFIG_V2_FIELDS = frozenset(
    {
        *_RESOLVED_PREFIX_CONFIG_FIELDS,
        "workload_plan",
        "workload_plan_sha256",
    }
)
_RESOLVED_PREFIX_CONFIG_V3_FIELDS = frozenset(
    {
        *_RESOLVED_PREFIX_CONFIG_V2_FIELDS,
        "workload_authorization_sha256",
        "validation_fixture_reference",
        "validation_perturbation_reference",
    }
)
_DYNAMIC_CHECK_NAMES = (
    "signature_and_identity",
    "dynamic_target_suffix_leakage",
    "cache_identity",
    "source_mask",
    "case_inventory",
    "validation_data_safety",
)
_STATIC_CHECK_NAMES = (
    "import_signature_access",
    "taint_cache_capability",
    "mask_normalization_support",
    "inventory_identity",
)
_DISCHARGED_COMPANION_OBLIGATIONS = frozenset(
    {
        "companion 4,096 validation-family report is required",
        "companion 9,720 small-family report is required",
    }
)
_FORBIDDEN_PREFIX_DEPENDENCY_FIELDS = (
    "predecessor",
    "correctness_artifact",
    "smc_accuracy",
    "training_schedule",
    "h6_schedule",
    "matching",
    "tuning",
    "checkpoint",
    "opening",
    "prediction",
)
_OWNED_PREFIX_WORKLOAD_DEPENDENCY_LIKE_FIELDS = frozenset(
    {
        "prediction_calls_per_case",
        "representative_prediction_calls",
        "ladder_subset_prediction_calls",
        "amended_total_prediction_calls",
        "finite_smc_accuracy_particle_count",
    }
)


def _bounded_indices(
    case_family: Literal["small", "validation"],
    scope: Literal["representative_exhaustive", "estimator_stratified"],
) -> tuple[int, ...]:
    workload = H6PrefixWorkloadPlan()
    if scope == "representative_exhaustive":
        return tuple(
            range(
                SMALL_EXPECTED_TOTAL
                if case_family == "small"
                else VALIDATION_EXPECTED_TOTAL
            )
        )
    return (
        workload.small_global_case_indices
        if case_family == "small"
        else workload.validation_global_case_indices
    )


@dataclass(frozen=True, slots=True)
class _H6PrefixDynamicJob:
    semantic_family_index: int
    profile: H6PrefixProfilePair
    case_family: Literal["small", "validation"]
    scope: Literal["representative_exhaustive", "estimator_stratified"]
    particle_count: int
    selected_global_indices: tuple[int, ...]
    report_key: PrefixCaseKey
    expected_case_count: int
    expected_predictor_call_count: int
    expected_particle_call_units: int
    collect_source_masks: bool
    collect_validation_safety: bool


@dataclass(frozen=True, slots=True)
class _H6PrefixSemanticFamilyPlan:
    semantic_family_index: int
    semantic_key: bytes
    profiles: tuple[H6PrefixProfilePair, ...]
    dynamic_jobs: tuple[_H6PrefixDynamicJob, ...]
    arm_build_count: int
    predictor_boundary_count: int
    dynamic_report_count: int
    representative_mask_collector_count: int
    expected_case_count: int
    expected_predictor_call_count: int
    expected_particle_call_units: int


@dataclass(frozen=True, slots=True)
class _H6PrefixStaticAuditJob:
    report_keys: tuple[PrefixCaseKey, ...]
    case_key_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _H6PrefixRunnerPlan:
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig
    semantic_families: tuple[_H6PrefixSemanticFamilyPlan, ...]
    expected_validation_vocabulary: VocabularyIdentity
    static_audit_job: _H6PrefixStaticAuditJob
    fixture_load_count: int
    static_audit_count: int
    expected_case_count: int
    expected_predictor_call_count: int
    expected_particle_call_units: int

@runtime_checkable
class _H6PrefixDynamicJobResult(Protocol):
    execution_plan: DynamicExecutionPlan
    report: DynamicPrefixReport
    observed_predictor_call_count: int


@runtime_checkable
class _H6PrefixRunnerExecutor(Protocol):
    def load_validation_perturbations(
        self, *, expected_vocabulary: VocabularyIdentity
    ) -> object: ...

    def build_arm(
        self,
        *,
        arm_config: ArmConfig,
        structure: H6LanguageStructure,
    ) -> object: ...

    def build_predictor_boundary(
        self,
        *,
        built_arm: object,
        estimator: EstimatorSpec,
    ) -> object: ...

    def execute_dynamic_job(
        self,
        *,
        job: _H6PrefixDynamicJob,
        built_arm: object,
        predictor: object,
        validation_perturbations: object,
    ) -> _H6PrefixDynamicJobResult: ...

    def execute_static_audit(
        self, *, job: _H6PrefixStaticAuditJob
    ) -> StaticAuditReport: ...


@dataclass(frozen=True, slots=True)
class _H6PrefixDynamicJobEvidence:
    job: _H6PrefixDynamicJob
    execution_plan: DynamicExecutionPlan
    report: DynamicPrefixReport
    observed_predictor_call_count: int


@dataclass(frozen=True, slots=True)
class _H6PrefixRunnerEvidence:
    plan: _H6PrefixRunnerPlan
    dynamic_results: tuple[_H6PrefixDynamicJobEvidence, ...]
    static_report: StaticAuditReport
    observed_predictor_call_count: int


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _canonical_object(
    value: Mapping[str, object], name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    decoded = json.loads(canonical_json_bytes(value))
    if type(decoded) is not dict:
        raise ValueError(f"{name} must canonicalize to one object")
    return decoded


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _vocabulary_sha256(profile: H6PrefixProfilePair, *, small: bool) -> str:
    vocabulary = (
        profile.small_arm_config.vocabulary
        if small
        else profile.production_arm_config.vocabulary
    )
    return _owned_hash(
        "vfe4.h6.vocabulary-identity.v1",
        {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
    )


def _resolved_structure_payload(
    structure: object,
) -> dict[str, object]:
    return {
        "base_sha256": structure.base.canonical_sha256,
        "dag_sha256": structure.dag.canonical_sha256,
        "receiver_labels": structure.receiver_labels,
        "structure_sha256": structure.structure_sha256,
    }


def _resolved_estimator_payload(profile: H6PrefixProfilePair) -> dict[str, object]:
    estimator = profile.estimator
    return {
        "schema_version": estimator.schema_version,
        "kind": estimator.kind,
        "particle_count": estimator.particle_count,
        "resampling": estimator.resampling,
        "dtype": estimator.dtype,
        "device": estimator.device,
        "estimator_sha256": estimator.estimator_sha256,
    }


def _resolved_profile_payload(
    profile: H6PrefixProfilePair,
) -> dict[str, object]:
    profile.__post_init__()
    return {
        "profile_id": profile.profile_id,
        "small_arm_config": {
            **profile.small_arm_config.canonical_payload(),
            "config_sha256": profile.small_arm_config.config_sha256,
        },
        "production_arm_config": {
            **profile.production_arm_config.canonical_payload(),
            "config_sha256": profile.production_arm_config.config_sha256,
        },
        "estimator": _resolved_estimator_payload(profile),
        "small_structure": _resolved_structure_payload(
            profile.small_structure
        ),
        "production_structure": _resolved_structure_payload(
            profile.production_structure
        ),
        "data_safety_sha256": profile.data_safety_sha256,
        "small_model_family_sha256": profile.small_model_family_sha256,
        "production_model_family_sha256": (
            profile.production_model_family_sha256
        ),
        "profile_pair_sha256": profile.profile_pair_sha256,
    }


def _resolver_arm_payload(config: ArmConfig) -> dict[str, object]:
    payload = config.canonical_payload()
    payload.pop("capacity_allocation_sha256")
    return payload


def _resolver_structure_payload(
    structure: H6LanguageStructure,
) -> dict[str, object]:
    structure.__post_init__()
    return {
        "base": {
            "base_id": structure.base.base_id,
            "points": list(structure.base.points),
            "dimension": structure.base.dimension,
        },
        "dag": {
            "labeling": structure.dag.labeling,
            "node_labels": list(structure.dag.node_labels),
            "rows": [
                {
                    "receiver_t": row.receiver_t,
                    "parents": list(row.parents),
                }
                for row in structure.dag.rows
            ],
        },
        "receiver_labels": list(structure.receiver_labels),
    }


def _resolver_profile_payload(
    profile: H6PrefixProfilePair,
) -> dict[str, object]:
    profile.__post_init__()
    estimator = profile.estimator
    return {
        "profile_id": profile.profile_id,
        "small_arm_config": _resolver_arm_payload(
            profile.small_arm_config
        ),
        "production_arm_config": _resolver_arm_payload(
            profile.production_arm_config
        ),
        "estimator": {
            "schema_version": estimator.schema_version,
            "kind": estimator.kind,
            "particle_count": estimator.particle_count,
            "resampling": estimator.resampling,
            "dtype": estimator.dtype,
            "device": estimator.device,
        },
        "small_structure": _resolver_structure_payload(
            profile.small_structure
        ),
        "production_structure": _resolver_structure_payload(
            profile.production_structure
        ),
        "data_safety_sha256": profile.data_safety_sha256,
        "small_model_family_sha256": profile.small_model_family_sha256,
        "production_model_family_sha256": (
            profile.production_model_family_sha256
        ),
        "profile_pair_sha256": profile.profile_pair_sha256,
    }


def _profile_semantic_key(profile: H6PrefixProfilePair) -> bytes:
    return canonical_json_bytes(
        {
            "small": {
                "arm": profile.small_arm_config.arm.value,
                "config_id": profile.small_arm_config.config_id,
                **profile.small_arm_config.semantic_payload(),
            },
            "production": {
                "arm": profile.production_arm_config.arm.value,
                "config_id": profile.production_arm_config.config_id,
                **profile.production_arm_config.semantic_payload(),
            },
            "small_structure_sha256": (
                profile.small_structure.structure_sha256
            ),
            "production_structure_sha256": (
                profile.production_structure.structure_sha256
            ),
            "data_safety_sha256": profile.data_safety_sha256,
        }
    )


def _validate_execution_profile_inventory(
    *,
    execution_mode: object,
    authorization_sha256: object,
    profiles: tuple[H6PrefixProfilePair, ...],
    workload_plan: H6PrefixWorkloadPlan | None = None,
) -> None:
    observed = tuple(
        (_profile_semantic_key(profile), profile.estimator.particle_count)
        for profile in profiles
    )
    ordered_semantics = tuple(dict.fromkeys(key for key, _ in observed))
    if execution_mode == "focused_subset":
        if authorization_sha256 is not None:
            raise ValueError(
                "focused H6-Prefix cannot carry full-inventory authorization"
            )
        expected = tuple((key, 4) for key in ordered_semantics)
        if observed != expected:
            raise ValueError(
                "focused H6-Prefix requires exactly one four-particle profile "
                "per semantic profile"
            )
        return
    if execution_mode != "authorized_full":
        raise ValueError("unsupported H6-Prefix execution mode")
    expected_authorization = (
        _FULL_PREFIX_AUTHORIZATION_SHA256
        if workload_plan is None
        else H6_PREFIX_V2_AUTHORIZATION_SHA256
    )
    if authorization_sha256 != expected_authorization:
        raise ValueError(
            "authorized-full H6-Prefix requires the exact operation authorization"
        )
    particle_counts = (
        (128, 256, 512, 1024)
        if workload_plan is None
        else workload_plan.production_particle_counts
    )
    expected = tuple(
        (key, particle_count)
        for key in ordered_semantics
        for particle_count in particle_counts
    )
    if observed != expected:
        raise ValueError(
            "authorized-full H6-Prefix requires the complete ordered "
            "128, 256, 512, 1024 ladder per semantic profile"
        )


def _verify_profile_key(
    *,
    profile: H6PrefixProfilePair,
    key: PrefixCaseKey,
    small: bool,
) -> None:
    if type(key) is not PrefixCaseKey:
        raise ValueError("Prefix reports require exact PrefixCaseKey records")
    key.__post_init__()
    config = (
        profile.small_arm_config
        if small
        else profile.production_arm_config
    )
    model_family_sha256 = (
        profile.small_model_family_sha256
        if small
        else profile.production_model_family_sha256
    )
    expected = {
        "arm": config.arm,
        "predictor_config_sha256": config.config_sha256,
        "estimator_sha256": profile.estimator.estimator_sha256,
        "model_family_sha256": model_family_sha256,
        "vocabulary_sha256": _vocabulary_sha256(profile, small=small),
        "data_safety_sha256": profile.data_safety_sha256,
    }
    for name, value in expected.items():
        if getattr(key, name) != value:
            label = "small" if small else "production"
            raise ValueError(
                f"{label} report key {name} does not match its Prefix profile"
            )


def _case_key_manifest(keys: tuple[PrefixCaseKey, ...]) -> str:
    encoded_payloads = tuple(
        (canonical_json_bytes(key.canonical_payload()), key.canonical_payload())
        for key in keys
    )
    if len({encoded for encoded, _ in encoded_payloads}) != len(keys):
        raise ValueError("static Prefix key inventory contains duplicates")
    ordered = tuple(
        payload
        for _, payload in sorted(encoded_payloads, key=lambda item: item[0])
    )
    return _owned_hash("vfe4.h6.static-audit-case-keys.v1", ordered)


def _dynamic_checks(
    report: DynamicPrefixReport,
) -> dict[str, DynamicCheckResult]:
    checks = {check.name: check for check in report.checks}
    if tuple(checks) != _DYNAMIC_CHECK_NAMES:
        raise ValueError("dynamic Prefix report check inventory is incomplete")
    return checks


def _static_checks(
    report: StaticAuditReport,
) -> dict[str, StaticAuditCheck]:
    checks = {check.name: check for check in report.checks}
    if tuple(checks) != _STATIC_CHECK_NAMES:
        raise ValueError("static Prefix report check inventory is incomplete")
    return checks


def _no_witnessed_failure(*checks: DynamicCheckResult | StaticAuditCheck) -> bool:
    return all(check.status is not EvidenceStatus.FAIL for check in checks)


def _report_obligations(
    *,
    small_report: DynamicPrefixReport,
    validation_report: DynamicPrefixReport,
    static_report: StaticAuditReport,
) -> tuple[str, ...]:
    obligations: list[str] = []
    for label, report in (
        ("small", small_report),
        ("validation", validation_report),
    ):
        obligations.extend(
            f"{label}: {value}"
            for value in report.obligations
            if value not in _DISCHARGED_COMPANION_OBLIGATIONS
        )
        obligations.extend(
            f"{label}: {value}" for value in report.unresolved_diagnostics
        )
        obligations.extend(
            f"{label}/{check.name}: {value}"
            for check in report.checks
            for value in check.obligations
        )
    obligations.extend(f"static: {value}" for value in static_report.obligations)
    obligations.extend(
        f"static/{check.name}: {value}"
        for check in static_report.checks
        for value in check.obligations
    )
    return tuple(sorted(set(obligations)))


def _complete_case_inventory(
    *,
    report: DynamicPrefixReport,
    small: bool,
) -> bool:
    checks = _dynamic_checks(report)
    inventory = checks["case_inventory"]
    if small:
        exact_positions = report.completed_by_position == (
            SMALL_EXPECTED_BY_POSITION
        )
        expected_total = SMALL_EXPECTED_TOTAL
    else:
        exact_positions = report.completed_by_position == (
            VALIDATION_EXPECTED_TOTAL,
        )
        expected_total = VALIDATION_EXPECTED_TOTAL
    return (
        inventory.status is EvidenceStatus.PASS
        and inventory.expected_count == expected_total
        and inventory.completed_count == expected_total
        and exact_positions
        and report.complete_case_manifest_sha256 is not None
    )


def compose_prefix_certificate(
    *,
    profile: H6PrefixProfilePair,
    small_report: DynamicPrefixReport,
    validation_report: DynamicPrefixReport,
    static_report: StaticAuditReport,
) -> PrefixCertificate:
    """Derive the sole report-bound certificate for one exact profile pair."""

    if type(profile) is not H6PrefixProfilePair:
        raise ValueError("profile must be an exact H6PrefixProfilePair")
    if type(small_report) is not DynamicPrefixReport:
        raise ValueError("small_report must be an exact DynamicPrefixReport")
    if type(validation_report) is not DynamicPrefixReport:
        raise ValueError(
            "validation_report must be an exact DynamicPrefixReport"
        )
    if type(static_report) is not StaticAuditReport:
        raise ValueError("static_report must be an exact StaticAuditReport")
    if (
        small_report.schema_version != "h6-dynamic-prefix-report-v1"
        or validation_report.schema_version != "h6-dynamic-prefix-report-v1"
    ):
        raise ValueError(
            "legacy Prefix composition explicitly rejects v2 dynamic reports"
        )
    profile.__post_init__()
    small_report.__post_init__()
    validation_report.__post_init__()
    static_report.__post_init__()

    _verify_profile_key(profile=profile, key=small_report.key, small=True)
    _verify_profile_key(
        profile=profile, key=validation_report.key, small=False
    )
    if (
        small_report.key.git_head != validation_report.key.git_head
        or small_report.key.dirty_digest != validation_report.key.dirty_digest
    ):
        raise ValueError(
            "small and validation report keys must bind one source candidate"
        )
    expected_static_manifest = _case_key_manifest(
        (small_report.key, validation_report.key)
    )
    if static_report.case_key_manifest_sha256 != expected_static_manifest:
        raise ValueError(
            "static case-key manifest does not bind exactly both Prefix keys"
        )

    small_checks = _dynamic_checks(small_report)
    validation_checks = _dynamic_checks(validation_report)
    static_checks = _static_checks(static_report)
    complete_small = _complete_case_inventory(
        report=small_report, small=True
    )
    complete_validation = _complete_case_inventory(
        report=validation_report, small=False
    )
    checks: dict[str, bool] = {
        "signature_import": _no_witnessed_failure(
            small_checks["signature_and_identity"],
            validation_checks["signature_and_identity"],
            static_checks["import_signature_access"],
        ),
        "taint_dataflow": _no_witnessed_failure(
            static_checks["taint_cache_capability"]
        ),
        "dynamic_target_suffix_leakage": _no_witnessed_failure(
            small_checks["dynamic_target_suffix_leakage"],
            validation_checks["dynamic_target_suffix_leakage"],
        ),
        "source_mask": _no_witnessed_failure(
            small_checks["source_mask"],
            validation_checks["source_mask"],
            static_checks["mask_normalization_support"],
        ),
        "cache_identity": _no_witnessed_failure(
            small_checks["cache_identity"],
            validation_checks["cache_identity"],
        ),
        "case_inventory": (
            _no_witnessed_failure(
                small_checks["case_inventory"],
                validation_checks["case_inventory"],
                static_checks["inventory_identity"],
            )
            and (
                complete_small
                or small_checks["case_inventory"].status
                is EvidenceStatus.INCONCLUSIVE
            )
            and (
                complete_validation
                or validation_checks["case_inventory"].status
                is EvidenceStatus.INCONCLUSIVE
            )
        ),
        "artifact_identity": True,
        "data_safety": _no_witnessed_failure(
            small_checks["validation_data_safety"],
            validation_checks["validation_data_safety"],
        ),
    }
    if tuple(sorted(checks)) != tuple(sorted(H6_PREFIX_REQUIRED_CHECKS)):
        raise RuntimeError("internal H6-Prefix check mapping is incomplete")

    obligations = list(
        _report_obligations(
            small_report=small_report,
            validation_report=validation_report,
            static_report=static_report,
        )
    )
    if not complete_small and small_checks["case_inventory"].status is not (
        EvidenceStatus.FAIL
    ):
        obligations.append("small: complete 9,720-case manifest is required")
    if not complete_validation and validation_checks[
        "case_inventory"
    ].status is not EvidenceStatus.FAIL:
        obligations.append(
            "validation: complete 4,096-case manifest is required"
        )
    if profile.estimator.particle_count == 4:
        obligations.append(
            "four-particle focused profile is development-only Prefix evidence"
        )

    witnessed_failure = (
        any(not value for value in checks.values())
        or small_report.status is EvidenceStatus.FAIL
        or validation_report.status is EvidenceStatus.FAIL
        or static_report.status is EvidenceStatus.FAIL
    )
    unresolved = (
        bool(obligations)
        or any(
            check.status is EvidenceStatus.INCONCLUSIVE
            for check in (
                *small_report.checks,
                *validation_report.checks,
                *static_report.checks,
            )
        )
    )
    status = (
        EvidenceStatus.FAIL
        if witnessed_failure
        else EvidenceStatus.INCONCLUSIVE
        if unresolved
        else EvidenceStatus.PASS
    )
    final_obligations = (
        ()
        if status is EvidenceStatus.FAIL
        else tuple(sorted(set(obligations)))
    )
    binding = PrefixReportBinding.create(
        small_report_sha256=small_report.report_sha256,
        small_case_manifest_sha256=(
            small_report.complete_case_manifest_sha256
            or small_report.case_result_manifest_sha256
        ),
        validation_report_sha256=validation_report.report_sha256,
        validation_case_manifest_sha256=(
            validation_report.complete_case_manifest_sha256
            or validation_report.case_result_manifest_sha256
        ),
        static_report_sha256=static_report.report_sha256,
        static_source_manifest_sha256=(
            static_report.source_manifest_sha256
        ),
        static_rules_sha256=static_report.rules_sha256,
        static_case_key_manifest_sha256=(
            static_report.case_key_manifest_sha256
        ),
    )
    validation_payload = canonical_json_bytes(
        {
            "schema_version": "h6-prefix-certificate-validation-v2",
            "profile": {
                **profile.canonical_payload(),
                "profile_pair_sha256": profile.profile_pair_sha256,
            },
            "small_key": small_report.key.canonical_payload(),
            "key": validation_report.key.canonical_payload(),
            "report_binding": binding.canonical_payload(),
            "checks": checks,
            "status": status.value,
            "obligations": final_obligations,
        }
    )
    validation_payload_sha256 = hashlib.sha256(validation_payload).hexdigest()
    certificate_sha256 = _owned_hash(
        "vfe4.h6.prefix-certificate.v1",
        {
            "key": validation_report.key.canonical_payload(),
            "validation_payload_sha256": validation_payload_sha256,
            "status": status.value,
            "obligations": final_obligations,
        },
    )
    return PrefixCertificate(
        key=validation_report.key,
        validation_payload_canonical_json=validation_payload,
        validation_payload_sha256=validation_payload_sha256,
        status=status,
        obligations=final_obligations,
        certificate_sha256=certificate_sha256,
    )


@dataclass(frozen=True, slots=True)
class H6BoundedPrefixFamilyBundle:
    """Pure input bundle for one bounded four-profile semantic family."""

    profiles: tuple[H6PrefixProfilePair, ...]
    reports: tuple[DynamicPrefixReport, ...]

    def __post_init__(self) -> None:
        if type(self) is not H6BoundedPrefixFamilyBundle:
            raise TypeError(
                "bundle requires the exact H6BoundedPrefixFamilyBundle type"
            )
        if (
            type(self.profiles) is not tuple
            or any(
                type(profile) is not H6PrefixProfilePair
                for profile in self.profiles
            )
        ):
            raise ValueError("bounded family profiles must be exact records")
        if (
            type(self.reports) is not tuple
            or any(
                type(report) is not DynamicPrefixReport
                for report in self.reports
            )
        ):
            raise ValueError("bounded family reports must be exact records")


def _bounded_report_obligations(
    reports: tuple[DynamicPrefixReport, ...],
    static_report: StaticAuditReport,
) -> tuple[str, ...]:
    obligations: list[str] = []
    for report in reports:
        label = f"N{report.particle_count}/{report.case_family}"
        obligations.extend(
            f"{label}: {value}" for value in report.obligations
        )
        obligations.extend(
            f"{label}: {value}" for value in report.unresolved_diagnostics
        )
        obligations.extend(
            f"{label}/{check.name}: {value}"
            for check in report.checks
            for value in check.obligations
        )
    obligations.extend(
        f"static: {value}" for value in static_report.obligations
    )
    obligations.extend(
        f"static/{check.name}: {value}"
        for check in static_report.checks
        for value in check.obligations
    )
    return tuple(sorted(set(obligations)))


def compose_bounded_prefix_certificate(
    *,
    family_bundle: H6BoundedPrefixFamilyBundle,
    static_report: StaticAuditReport,
    global_case_keys: tuple[PrefixCaseKey, ...],
    source_sha256: str,
) -> BoundedPrefixCertificate:
    """Compose one certificate from a complete, already-produced v2 matrix."""

    if type(family_bundle) is not H6BoundedPrefixFamilyBundle:
        raise ValueError(
            "family_bundle must be an exact H6BoundedPrefixFamilyBundle"
        )
    family_bundle.__post_init__()
    if type(static_report) is not StaticAuditReport:
        raise ValueError("static_report must be an exact StaticAuditReport")
    static_report.__post_init__()
    _require_sha256(source_sha256, "source_sha256")
    if (
        type(global_case_keys) is not tuple
        or not global_case_keys
        or any(type(key) is not PrefixCaseKey for key in global_case_keys)
    ):
        raise ValueError(
            "global_case_keys must be a nonempty exact ordered key inventory"
        )
    for key in global_case_keys:
        key.__post_init__()
    encoded_global_keys = tuple(
        canonical_json_bytes(key.canonical_payload())
        for key in global_case_keys
    )
    if len(set(encoded_global_keys)) != len(encoded_global_keys):
        raise ValueError("global Prefix key inventory contains duplicates")

    profiles = family_bundle.profiles
    reports = family_bundle.reports
    workload = H6PrefixWorkloadPlan()
    if (
        len(profiles) != 4
        or tuple(
            profile.estimator.particle_count for profile in profiles
        )
        != workload.production_particle_counts
    ):
        raise ValueError(
            "bounded family requires the exact ordered four-profile N ladder"
        )
    if len(reports) != 8:
        raise ValueError(
            "bounded family requires exactly eight ordered dynamic reports"
        )
    for profile in profiles:
        profile.__post_init__()
    if len({profile.profile_pair_sha256 for profile in profiles}) != 4:
        raise ValueError("bounded family profile identities must be unique")
    semantic_payload = _non_particle_profile_payload(profiles[0])
    semantic_bytes = canonical_json_bytes(semantic_payload)
    if any(
        canonical_json_bytes(_non_particle_profile_payload(profile))
        != semantic_bytes
        for profile in profiles[1:]
    ):
        raise ValueError(
            "only particle count may vary within a bounded semantic family"
        )
    semantic_family_sha256 = _owned_hash(
        "vfe4.h6.bounded-prefix-semantic-family.v2",
        semantic_payload,
    )

    expected_matrix = tuple(
        (
            profile,
            case_family,
            (
                "representative_exhaustive"
                if profile.estimator.particle_count == 128
                else "estimator_stratified"
            ),
        )
        for profile in profiles
        for case_family in ("small", "validation")
    )
    common_git_head = reports[0].key.git_head
    common_dirty_digest = reports[0].key.dirty_digest
    expected_source_sha256 = source_candidate_sha256(
        git_head_value=common_git_head,
        dirty_digest_value=common_dirty_digest,
    )
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "source_sha256 does not match the bounded source candidate"
        )
    references: list[BoundedPrefixReportReference] = []
    for report, (profile, case_family, scope) in zip(
        reports,
        expected_matrix,
        strict=True,
    ):
        report.__post_init__()
        if report.schema_version != "h6-dynamic-prefix-report-v2":
            raise ValueError(
                "bounded composition rejects mixed v1/v2 dynamic evidence"
            )
        small = case_family == "small"
        _verify_profile_key(
            profile=profile,
            key=report.key,
            small=small,
        )
        expected_indices = _bounded_indices(case_family, scope)
        expected_completed = (
            SMALL_EXPECTED_BY_POSITION
            if scope == "representative_exhaustive"
            and case_family == "small"
            else (VALIDATION_EXPECTED_TOTAL,)
            if scope == "representative_exhaustive"
            else (4, 4, 4, 4)
            if case_family == "small"
            else (16,)
        )
        if (
            report.particle_count != profile.estimator.particle_count
            or report.case_family != case_family
            or report.scope != scope
            or report.workload_plan_sha256
            != workload.workload_plan_sha256
            or report.selected_global_indices != expected_indices
            or report.completed_by_position != expected_completed
            or (
                scope == "representative_exhaustive"
                and report.complete_case_manifest_sha256 is None
            )
            or (
                scope == "estimator_stratified"
                and report.complete_case_manifest_sha256 is not None
            )
        ):
            raise ValueError(
                "bounded dynamic report does not match its exact matrix slot"
            )
        if (
            report.key.git_head != common_git_head
            or report.key.dirty_digest != common_dirty_digest
        ):
            raise ValueError(
                "bounded dynamic reports do not bind one source candidate"
            )
        estimator_identity = EstimatorIdentity.from_spec(profile.estimator)
        if (
            report.estimator_semantic_sha256
            != estimator_identity.semantic_sha256
            or report.estimator_artifact_bytes_sha256
            != estimator_identity.artifact_bytes_sha256
        ):
            raise ValueError(
                "bounded report estimator identity does not match its profile"
            )
        references.append(
            BoundedPrefixReportReference.create(
                profile_pair_sha256=profile.profile_pair_sha256,
                particle_count=profile.estimator.particle_count,
                case_family=case_family,
                scope=scope,
                report_key=report.key,
                report_sha256=report.report_sha256,
                execution_plan_sha256=report.execution_plan_sha256,
                workload_plan_sha256=report.workload_plan_sha256,
                selected_global_indices=report.selected_global_indices,
                selection_manifest_sha256=report.selection_manifest_sha256,
                completed_by_position=report.completed_by_position,
                complete_case_manifest_sha256=(
                    report.complete_case_manifest_sha256
                ),
                model_state_sha256=report.model_state_sha256,
                proposal_identity_sha256=(
                    report.proposal_identity_sha256
                ),
                estimator_semantic_sha256=(
                    report.estimator_semantic_sha256
                ),
                estimator_artifact_bytes_sha256=(
                    report.estimator_artifact_bytes_sha256
                ),
            )
        )

    if any(
        key.git_head != common_git_head
        or key.dirty_digest != common_dirty_digest
        for key in global_case_keys
    ):
        raise ValueError(
            "global Prefix key inventory crosses source candidates"
        )
    if static_report.case_key_manifest_sha256 != _case_key_manifest(
        global_case_keys
    ):
        raise ValueError(
            "global static report does not bind the supplied key inventory"
        )
    global_key_payloads = set(encoded_global_keys)
    if any(
        canonical_json_bytes(report.key.canonical_payload())
        not in global_key_payloads
        for report in reports
    ):
        raise ValueError(
            "global static key inventory omits a bounded family report"
        )
    family_key_order = tuple(
        canonical_json_bytes(report.key.canonical_payload())
        for report in reports
    )
    first_family_key_index = encoded_global_keys.index(family_key_order[0])
    if (
        encoded_global_keys[
            first_family_key_index : first_family_key_index + len(family_key_order)
        ]
        != family_key_order
    ):
        raise ValueError(
            "global Prefix keys do not preserve the exact family plan order"
        )
    global_case_key_order_sha256 = _owned_hash(
        "vfe4.h6.bounded-prefix-global-case-key-order.v2",
        tuple(key.canonical_payload() for key in global_case_keys),
    )

    report_references = tuple(references)
    higher_n_small_selection = report_references[
        2
    ].selection_manifest_sha256
    higher_n_validation_selection = report_references[
        3
    ].selection_manifest_sha256
    binding = BoundedPrefixReportBinding.create(
        workload_plan_sha256=workload.workload_plan_sha256,
        semantic_family_sha256=semantic_family_sha256,
        git_head=common_git_head,
        dirty_digest=common_dirty_digest,
        source_sha256=source_sha256,
        global_case_key_order_sha256=global_case_key_order_sha256,
        profile_pair_sha256s=tuple(
            profile.profile_pair_sha256 for profile in profiles
        ),
        report_references=report_references,
        higher_n_small_selection_manifest_sha256=(
            higher_n_small_selection
        ),
        higher_n_validation_selection_manifest_sha256=(
            higher_n_validation_selection
        ),
        static_report_sha256=static_report.report_sha256,
        static_source_manifest_sha256=(
            static_report.source_manifest_sha256
        ),
        static_rules_sha256=static_report.rules_sha256,
        static_case_key_manifest_sha256=(
            static_report.case_key_manifest_sha256
        ),
    )

    dynamic = tuple(_dynamic_checks(report) for report in reports)
    static = _static_checks(static_report)
    representative = dynamic[:2]
    artifact_identity = (
        binding.git_head == common_git_head
        and binding.dirty_digest == common_dirty_digest
        and binding.source_sha256 == expected_source_sha256
        and binding.global_case_key_order_sha256
        == global_case_key_order_sha256
        and binding.profile_pair_sha256s
        == tuple(profile.profile_pair_sha256 for profile in profiles)
        and tuple(
            reference.report_sha256
            for reference in binding.report_references
        )
        == tuple(report.report_sha256 for report in reports)
        and binding.static_report_sha256 == static_report.report_sha256
        and binding.static_source_manifest_sha256
        == static_report.source_manifest_sha256
        and binding.static_rules_sha256 == static_report.rules_sha256
        and binding.static_case_key_manifest_sha256
        == static_report.case_key_manifest_sha256
    )
    checks: dict[str, bool] = {
        "signature_import": _no_witnessed_failure(
            *(item["signature_and_identity"] for item in dynamic),
            static["import_signature_access"],
        ),
        "taint_dataflow": _no_witnessed_failure(
            static["taint_cache_capability"]
        ),
        "dynamic_target_suffix_leakage": _no_witnessed_failure(
            *(item["dynamic_target_suffix_leakage"] for item in dynamic)
        ),
        "source_mask": _no_witnessed_failure(
            *(item["source_mask"] for item in representative),
            static["mask_normalization_support"],
        ),
        "cache_identity": _no_witnessed_failure(
            *(item["cache_identity"] for item in dynamic)
        ),
        "case_inventory": _no_witnessed_failure(
            *(item["case_inventory"] for item in representative),
            static["inventory_identity"],
        ),
        "artifact_identity": artifact_identity,
        "data_safety": _no_witnessed_failure(
            *(item["validation_data_safety"] for item in representative)
        ),
    }
    if tuple(checks) != H6_PREFIX_REQUIRED_CHECKS:
        raise RuntimeError("internal bounded Prefix check mapping is incomplete")
    obligations = _bounded_report_obligations(reports, static_report)
    witnessed_failure = (
        any(not value for value in checks.values())
        or any(report.status is EvidenceStatus.FAIL for report in reports)
        or static_report.status is EvidenceStatus.FAIL
    )
    unresolved = (
        bool(obligations)
        or any(
            report.status is EvidenceStatus.INCONCLUSIVE
            for report in reports
        )
        or static_report.status is EvidenceStatus.INCONCLUSIVE
    )
    status = (
        EvidenceStatus.FAIL
        if witnessed_failure
        else EvidenceStatus.INCONCLUSIVE
        if unresolved
        else EvidenceStatus.PASS
    )
    return BoundedPrefixCertificate.create(
        semantic_family_sha256=semantic_family_sha256,
        report_binding=binding,
        status=status,
        obligations=() if status is EvidenceStatus.FAIL else obligations,
        checks=checks,
    )


@dataclass(frozen=True)
class H6PrefixReportBundle:
    """Exact profile/reports/certificate unit consumed by pure publication."""

    profile: H6PrefixProfilePair
    small_report: DynamicPrefixReport
    validation_report: DynamicPrefixReport
    static_report: StaticAuditReport
    certificate: PrefixCertificate

    def __post_init__(self) -> None:
        expected = compose_prefix_certificate(
            profile=self.profile,
            small_report=self.small_report,
            validation_report=self.validation_report,
            static_report=self.static_report,
        )
        if self.certificate != expected:
            raise ValueError(
                "certificate does not equal the report-derived Prefix certificate"
            )


def _dynamic_report_payload(report: DynamicPrefixReport) -> dict[str, object]:
    return {**report.canonical_payload(), "report_sha256": report.report_sha256}


def _static_finding_payload(
    finding: StaticAuditFinding,
) -> dict[str, object]:
    return {
        "rule_id": finding.rule_id,
        "status": finding.status.value,
        "path": finding.path,
        "line": finding.line,
        "message": finding.message,
        "witness_sha256": finding.witness_sha256,
        "finding_sha256": finding.finding_sha256,
    }


def _static_check_payload(check: StaticAuditCheck) -> dict[str, object]:
    return {
        "name": check.name,
        "status": check.status.value,
        "finding_sha256s": check.finding_sha256s,
        "obligations": check.obligations,
        "check_sha256": check.check_sha256,
    }


def _static_report_payload(report: StaticAuditReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "source_manifest_sha256": report.source_manifest_sha256,
        "rules_sha256": report.rules_sha256,
        "case_key_manifest_sha256": report.case_key_manifest_sha256,
        "checks": tuple(_static_check_payload(check) for check in report.checks),
        "findings": tuple(
            _static_finding_payload(finding) for finding in report.findings
        ),
        "status": report.status.value,
        "obligations": report.obligations,
        "report_sha256": report.report_sha256,
    }


def _compose_bounded_prefix_certificate_set(
    *,
    config: H6PrefixV3ResolvedConfig,
    runner_evidence: _H6PrefixRunnerEvidence,
) -> BoundedPrefixCertificateSet:
    """Compose the ordered bounded family set from retained runner evidence."""

    if type(config) is not H6PrefixV3ResolvedConfig:
        raise ValueError("bounded publication requires an exact v3 config")
    _validate_v3_resolved_runner_config(config)
    if (
        type(runner_evidence) is not _H6PrefixRunnerEvidence
        or type(runner_evidence.plan) is not _H6PrefixRunnerPlan
        or runner_evidence.plan.config != config
        or runner_evidence.plan
        != _build_h6_prefix_runner_plan(config)
    ):
        raise ValueError(
            "bounded publication evidence differs from its exact v3 plan"
        )
    plan = runner_evidence.plan
    if type(runner_evidence.static_report) is not StaticAuditReport:
        raise ValueError(
            "bounded publication requires one exact static report"
        )
    runner_evidence.static_report.__post_init__()
    if (
        runner_evidence.static_report.case_key_manifest_sha256
        != plan.static_audit_job.case_key_manifest_sha256
    ):
        raise ValueError(
            "bounded publication static report differs from the plan"
        )
    expected_jobs = tuple(
        job
        for family in plan.semantic_families
        for job in family.dynamic_jobs
    )
    if (
        type(runner_evidence.dynamic_results) is not tuple
        or len(runner_evidence.dynamic_results) != len(expected_jobs)
        or tuple(
            result.job for result in runner_evidence.dynamic_results
        )
        != expected_jobs
    ):
        raise ValueError(
            "bounded publication dynamic evidence is missing or reordered"
        )
    observed_calls = 0
    for job, evidence in zip(
        expected_jobs,
        runner_evidence.dynamic_results,
        strict=True,
    ):
        if type(evidence) is not _H6PrefixDynamicJobEvidence:
            raise ValueError(
                "bounded publication requires exact dynamic evidence"
            )
        _, _, calls = _validated_dynamic_job_result(
            plan=plan,
            job=job,
            result=evidence,
        )
        observed_calls += calls
    if (
        observed_calls != runner_evidence.observed_predictor_call_count
        or observed_calls != plan.expected_predictor_call_count
    ):
        raise ValueError(
            "bounded publication predictor-call total differs from the plan"
        )

    certificates: list[BoundedPrefixCertificate] = []
    offset = 0
    for family in plan.semantic_families:
        family_results = runner_evidence.dynamic_results[
            offset : offset + len(family.dynamic_jobs)
        ]
        offset += len(family_results)
        certificate = compose_bounded_prefix_certificate(
            family_bundle=H6BoundedPrefixFamilyBundle(
                profiles=family.profiles,
                reports=tuple(result.report for result in family_results),
            ),
            static_report=runner_evidence.static_report,
            global_case_keys=plan.static_audit_job.report_keys,
            source_sha256=config.source.source_sha256,
        )
        certificates.append(certificate)
    if offset != len(runner_evidence.dynamic_results):
        raise ValueError("bounded publication retained extra dynamic evidence")
    ordered_certificates = tuple(certificates)
    return BoundedPrefixCertificateSet.create(
        config_sha256=config.config_sha256,
        semantic_family_sha256s=tuple(
            certificate.semantic_family_sha256
            for certificate in ordered_certificates
        ),
        certificates=ordered_certificates,
    )


def _bounded_execution_plan_payload(
    plan: DynamicExecutionPlan,
) -> dict[str, object]:
    plan.__post_init__()
    return {**plan.canonical_payload(), "plan_sha256": plan.plan_sha256}


def _bounded_certificate_payload(
    certificate: BoundedPrefixCertificate,
) -> dict[str, object]:
    certificate.__post_init__()
    return {
        "schema_version": certificate.schema_version,
        "semantic_family_sha256": certificate.semantic_family_sha256,
        "report_binding": certificate.report_binding.canonical_payload(),
        "validation_payload": json.loads(
            certificate.validation_payload_canonical_json
        ),
        "validation_payload_sha256": (
            certificate.validation_payload_sha256
        ),
        "status": certificate.status.value,
        "obligations": certificate.obligations,
        "checks": dict(certificate.checks),
        "certificate_sha256": certificate.certificate_sha256,
    }


def h6_bounded_prefix_artifact_payloads(
    *,
    config_payload: Mapping[str, object],
    provenance_payload: Mapping[str, object],
    environment_payload: Mapping[str, object],
    runner_evidence: _H6PrefixRunnerEvidence,
    certificate_set: BoundedPrefixCertificateSet,
) -> dict[str, object]:
    """Construct the exact bounded-v3 five-payload artifact."""

    if type(certificate_set) is not BoundedPrefixCertificateSet:
        raise ValueError(
            "bounded payloads require an exact certificate set"
        )
    certificate_set.__post_init__()
    if (
        type(runner_evidence) is not _H6PrefixRunnerEvidence
        or type(runner_evidence.plan) is not _H6PrefixRunnerPlan
        or type(runner_evidence.plan.config) is not H6PrefixV3ResolvedConfig
    ):
        raise ValueError("bounded payloads require exact v3 runner evidence")
    config = runner_evidence.plan.config
    expected_set = _compose_bounded_prefix_certificate_set(
        config=config,
        runner_evidence=runner_evidence,
    )
    if certificate_set != expected_set:
        raise ValueError(
            "bounded certificate set differs from runner evidence"
        )
    result = H6BoundedPrefixGateResult.from_certificate_set(
        certificate_set
    )
    resolved_config = _canonical_object(config_payload, "config_payload")
    provenance = _canonical_object(
        provenance_payload,
        "provenance_payload",
    )
    environment = _canonical_object(
        environment_payload,
        "environment_payload",
    )
    if (
        frozenset(resolved_config) != _RESOLVED_PREFIX_CONFIG_V3_FIELDS
        or resolved_config.get("schema_version") != "h6-prefix-config-v3"
        or resolved_config.get("operation") != "H6-Prefix"
        or canonical_json_bytes(resolved_config)
        != config.canonical_json.encode("ascii")
        or hashlib.sha256(canonical_json_bytes(resolved_config)).hexdigest()
        != config.config_sha256
    ):
        raise ValueError("bounded payload config differs from exact v3 config")
    if (
        frozenset(provenance)
        != frozenset(
            {
                "schema_version",
                "git_head",
                "dirty_digest",
                "source_sha256",
                "junit_sha256",
            }
        )
        or provenance.get("schema_version")
        != "h6-prefix-provenance-v1"
        or provenance.get("git_head") != certificate_set.git_head
        or provenance.get("dirty_digest") != certificate_set.dirty_digest
        or provenance.get("source_sha256")
        != certificate_set.source_sha256
    ):
        raise ValueError(
            "bounded provenance differs from certificate-set source"
        )
    junit_sha256 = provenance.get("junit_sha256")
    if junit_sha256 is not None:
        _require_sha256(junit_sha256, "provenance.junit_sha256")
    if (
        frozenset(environment)
        != frozenset(
            {
                "schema_version",
                "device",
                "dtype",
                "python_implementation",
                "python_version",
            }
        )
        or environment.get("schema_version")
        != "h6-prefix-environment-v1"
    ):
        raise ValueError("bounded environment payload is not exact")
    for name, payload in (
        ("config_payload", resolved_config),
        ("provenance_payload", provenance),
        ("environment_payload", environment),
    ):
        _reject_prefix_dependencies(payload, name)

    plan = runner_evidence.plan
    family_entries: list[dict[str, object]] = []
    offset = 0
    for family, certificate in zip(
        plan.semantic_families,
        certificate_set.certificates,
        strict=True,
    ):
        family_results = runner_evidence.dynamic_results[
            offset : offset + len(family.dynamic_jobs)
        ]
        offset += len(family_results)
        jobs = tuple(
            {
                "job_index": job_index,
                "particle_count": evidence.job.particle_count,
                "case_family": evidence.job.case_family,
                "scope": evidence.job.scope,
                "profile_pair_sha256": (
                    evidence.job.profile.profile_pair_sha256
                ),
                "execution_plan": _bounded_execution_plan_payload(
                    evidence.execution_plan
                ),
                "dynamic_report": _dynamic_report_payload(
                    evidence.report
                ),
                "observed_predictor_call_count": (
                    evidence.observed_predictor_call_count
                ),
            }
            for job_index, evidence in enumerate(family_results)
        )
        family_entries.append(
            {
                "semantic_family_index": family.semantic_family_index,
                "semantic_family_sha256": (
                    certificate.semantic_family_sha256
                ),
                "jobs": jobs,
                "validation_payload_sha256": (
                    certificate.validation_payload_sha256
                ),
                "certificate_sha256": certificate.certificate_sha256,
            }
        )
    if offset != len(runner_evidence.dynamic_results):
        raise ValueError("bounded payloads retained extra dynamic evidence")
    runner_totals = {
        "semantic_family_count": len(plan.semantic_families),
        "planned_fixture_load_count": plan.fixture_load_count,
        "planned_static_audit_count": plan.static_audit_count,
        "planned_arm_build_count": sum(
            family.arm_build_count for family in plan.semantic_families
        ),
        "planned_predictor_boundary_count": sum(
            family.predictor_boundary_count
            for family in plan.semantic_families
        ),
        "planned_dynamic_report_count": sum(
            family.dynamic_report_count
            for family in plan.semantic_families
        ),
        "observed_dynamic_report_count": len(
            runner_evidence.dynamic_results
        ),
        "planned_case_count": plan.expected_case_count,
        "planned_predictor_call_count": (
            plan.expected_predictor_call_count
        ),
        "observed_predictor_call_count": (
            runner_evidence.observed_predictor_call_count
        ),
        "planned_particle_call_units": (
            plan.expected_particle_call_units
        ),
    }
    validation = {
        "schema_version": "h6-prefix-validation-set-v2",
        "gate": "H6-Prefix",
        "status": result.status.value,
        "obligations": result.obligations,
        "config_sha256": result.config_sha256,
        "workload_plan_sha256": result.workload_plan_sha256,
        "validation_payload_sha256": result.validation_payload_sha256,
        "prefix_certificate_set_sha256": (
            result.prefix_certificate_set_sha256
        ),
        "runner_totals": runner_totals,
        "semantic_families": tuple(family_entries),
        "static_report": _static_report_payload(
            runner_evidence.static_report
        ),
    }
    certificates_payload = {
        "schema_version": certificate_set.schema_version,
        "config_sha256": certificate_set.config_sha256,
        "workload_plan_sha256": certificate_set.workload_plan_sha256,
        "git_head": certificate_set.git_head,
        "dirty_digest": certificate_set.dirty_digest,
        "source_sha256": certificate_set.source_sha256,
        "semantic_family_sha256s": (
            certificate_set.semantic_family_sha256s
        ),
        "validation_payload_sha256": (
            certificate_set.validation_payload_sha256
        ),
        "prefix_certificate_set_sha256": (
            certificate_set.prefix_certificate_set_sha256
        ),
        "certificates": tuple(
            _bounded_certificate_payload(certificate)
            for certificate in certificate_set.certificates
        ),
    }
    return {
        "certificates/prefix_set.json": certificates_payload,
        "config.json": resolved_config,
        "environment.json": environment,
        "provenance.json": provenance,
        "validation/h6_prefix.json": validation,
    }


def publish_h6_bounded_prefix_artifact(
    *,
    artifact_root: Path,
    run_name: str,
    config_payload: Mapping[str, object],
    provenance_payload: Mapping[str, object],
    environment_payload: Mapping[str, object],
    runner_evidence: _H6PrefixRunnerEvidence,
    certificate_set: BoundedPrefixCertificateSet,
) -> tuple[H6BoundedPrefixGateResult, Path]:
    """Atomically publish one complete bounded Prefix v3 artifact."""

    payloads = h6_bounded_prefix_artifact_payloads(
        config_payload=config_payload,
        provenance_payload=provenance_payload,
        environment_payload=environment_payload,
        runner_evidence=runner_evidence,
        certificate_set=certificate_set,
    )
    result = H6BoundedPrefixGateResult.from_certificate_set(
        certificate_set
    )
    run_dir = publish_run_directory(artifact_root, run_name, payloads)
    return result, run_dir


def _reject_prefix_dependencies(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} contains an invalid field name")
            folded = key.casefold()
            owned_workload_metric = (
                path == "config_payload.workload_plan"
                and key
                in _OWNED_PREFIX_WORKLOAD_DEPENDENCY_LIKE_FIELDS
            )
            if (
                not owned_workload_metric
                and (
                    folded in {"h1", "h2", "h3", "h4", "h5"}
                    or any(
                        fragment in folded
                        for fragment in _FORBIDDEN_PREFIX_DEPENDENCY_FIELDS
                    )
                )
            ):
                raise ValueError(
                    f"{path}.{key} is not allowed in independent H6-Prefix"
                )
            _reject_prefix_dependencies(item, f"{path}.{key}")
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            _reject_prefix_dependencies(item, f"{path}[{index}]")
    elif type(value) is str and (
        value in {"H1", "H2", "H3", "H4", "H5", "H6-Prediction"}
        or value.startswith(("H1-", "H2-", "H3-", "H4-", "H5-"))
    ):
        raise ValueError(
            f"{path} contains a forbidden predecessor or Prediction identity"
        )


def h6_prefix_artifact_payloads(
    *,
    config_payload: Mapping[str, object],
    provenance_payload: Mapping[str, object],
    environment_payload: Mapping[str, object],
    report_bundles: tuple[H6PrefixReportBundle, ...],
) -> dict[str, object]:
    """Construct the exact five JSON payloads preceding the atomic manifest."""

    if (
        type(report_bundles) is not tuple
        or not report_bundles
        or any(type(bundle) is not H6PrefixReportBundle for bundle in report_bundles)
    ):
        raise ValueError(
            "report_bundles must be a nonempty exact H6PrefixReportBundle tuple"
        )
    for bundle in report_bundles:
        bundle.__post_init__()
    config = _canonical_object(config_payload, "config_payload")
    provenance = _canonical_object(provenance_payload, "provenance_payload")
    environment = _canonical_object(environment_payload, "environment_payload")
    if (
        frozenset(config) != _RESOLVED_PREFIX_CONFIG_FIELDS
        or config.get("schema_version") != "h6-prefix-config-v1"
        or config.get("operation") != "H6-Prefix"
    ):
        raise ValueError(
            "config_payload must be the exact resolved H6-Prefix schema"
        )
    for name, payload in (
        ("config_payload", config),
        ("provenance_payload", provenance),
        ("environment_payload", environment),
    ):
        _reject_prefix_dependencies(payload, name)

    ordered = report_bundles
    if len({bundle.profile.profile_id for bundle in ordered}) != len(ordered):
        raise ValueError("Prefix profile IDs must be unique")
    if len(
        {
            canonical_json_bytes(bundle.certificate.key.canonical_payload())
            for bundle in ordered
        }
    ) != len(ordered):
        raise ValueError("Prefix production certificate keys must be unique")
    profile_hashes = tuple(
        bundle.profile.profile_pair_sha256 for bundle in ordered
    )
    configured_profiles = config["profiles"]
    if (
        type(configured_profiles) is not list
        or not configured_profiles
        or any(type(profile) is not dict for profile in configured_profiles)
    ):
        raise ValueError(
            "resolved config profiles must be a nonempty ordered object list"
        )
    configured_profile_hashes = tuple(
        _require_sha256(
            profile.get("profile_pair_sha256"),
            f"config.profiles[{index}].profile_pair_sha256",
        )
        for index, profile in enumerate(configured_profiles)
    )
    if configured_profile_hashes != profile_hashes:
        raise ValueError(
            "ordered embedded profile_pair_sha256 values do not match "
            "the report bundles"
        )
    expected_profiles = tuple(
        _resolved_profile_payload(bundle.profile) for bundle in ordered
    )
    if canonical_json_bytes(configured_profiles) != canonical_json_bytes(
        expected_profiles
    ):
        raise ValueError(
            "resolved config profiles do not equal the exact typed report profiles"
        )
    _validate_execution_profile_inventory(
        execution_mode=config["execution_mode"],
        authorization_sha256=config["authorization_sha256"],
        profiles=tuple(bundle.profile for bundle in ordered),
    )
    source = config["source"]
    if (
        type(source) is not dict
        or frozenset(source)
        != frozenset({"git_head", "dirty_digest", "source_sha256"})
    ):
        raise ValueError("resolved config source identity is incomplete")
    source_git_head = source["git_head"]
    if (
        type(source_git_head) is not str
        or len(source_git_head) != 40
        or any(character not in "0123456789abcdef" for character in source_git_head)
    ):
        raise ValueError("config.source.git_head must be a lowercase Git SHA")
    source_dirty_digest = _require_sha256(
        source["dirty_digest"], "config.source.dirty_digest"
    )
    source_sha256 = _require_sha256(
        source["source_sha256"], "config.source.source_sha256"
    )
    candidate_heads = {
        bundle.certificate.key.git_head for bundle in ordered
    }
    candidate_dirty = {
        bundle.certificate.key.dirty_digest for bundle in ordered
    }
    if (
        len(candidate_heads) != 1
        or len(candidate_dirty) != 1
        or source_git_head != next(iter(candidate_heads))
        or source_dirty_digest != next(iter(candidate_dirty))
        or provenance.get("git_head") != source_git_head
        or provenance.get("dirty_digest") != source_dirty_digest
        or provenance.get("source_sha256") != source_sha256
    ):
        raise ValueError(
            "config/provenance do not match the exact Prefix report candidate"
        )
    junit_sha256 = provenance.get("junit_sha256")
    if junit_sha256 is not None:
        _require_sha256(junit_sha256, "provenance.junit_sha256")

    certificates = {
        bundle.certificate.key: bundle.certificate for bundle in ordered
    }
    result = H6PrefixGateResult.from_certificates(certificates)
    validation_entries = tuple(
        {
            "profile": {
                **bundle.profile.canonical_payload(),
                "profile_pair_sha256": bundle.profile.profile_pair_sha256,
            },
            "small_report": _dynamic_report_payload(bundle.small_report),
            "validation_report": _dynamic_report_payload(
                bundle.validation_report
            ),
            "static_report": _static_report_payload(bundle.static_report),
            "certificate_sha256": bundle.certificate.certificate_sha256,
            "validation_payload_sha256": (
                bundle.certificate.validation_payload_sha256
            ),
        }
        for bundle in ordered
    )
    validation = {
        "schema_version": "h6-prefix-validation-set-v1",
        "gate": "H6-Prefix",
        "status": result.status.value,
        "validation_payload_sha256": result.validation_payload_sha256,
        "prefix_certificate_set_sha256": (
            result.prefix_certificate_set_sha256
        ),
        "obligations": result.obligations,
        "profiles": validation_entries,
    }
    certificate_entries = tuple(
        {
            "key": bundle.certificate.key.canonical_payload(),
            "status": bundle.certificate.status.value,
            "obligations": bundle.certificate.obligations,
            "validation_payload_sha256": (
                bundle.certificate.validation_payload_sha256
            ),
            "validation_payload": json.loads(
                bundle.certificate.validation_payload_canonical_json
            ),
            "certificate_sha256": bundle.certificate.certificate_sha256,
        }
        for bundle in ordered
    )
    certificate_set = {
        "schema_version": "h6-prefix-certificate-set-v1",
        "prefix_certificate_set_sha256": (
            result.prefix_certificate_set_sha256
        ),
        "certificates": certificate_entries,
    }
    return {
        "config.json": config,
        "provenance.json": provenance,
        "environment.json": environment,
        "validation/h6_prefix.json": validation,
        "certificates/prefix_set.json": certificate_set,
    }


def publish_h6_prefix_artifact(
    *,
    artifact_root: Path,
    run_name: str,
    config_payload: Mapping[str, object],
    provenance_payload: Mapping[str, object],
    environment_payload: Mapping[str, object],
    report_bundles: tuple[H6PrefixReportBundle, ...],
) -> tuple[H6PrefixGateResult, Path]:
    """Atomically publish one already-composed independent H6-Prefix artifact."""

    payloads = h6_prefix_artifact_payloads(
        config_payload=config_payload,
        provenance_payload=provenance_payload,
        environment_payload=environment_payload,
        report_bundles=report_bundles,
    )
    certificates = {
        bundle.certificate.key: bundle.certificate
        for bundle in report_bundles
    }
    result = H6PrefixGateResult.from_certificates(certificates)
    run_dir = publish_run_directory(
        artifact_root,
        run_name,
        payloads,
    )
    return result, run_dir


def _prefix_key(
    *,
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig,
    profile: H6PrefixProfilePair,
    small: bool,
) -> PrefixCaseKey:
    arm_config = (
        profile.small_arm_config
        if small
        else profile.production_arm_config
    )
    return PrefixCaseKey(
        arm=arm_config.arm,
        predictor_config_sha256=arm_config.config_sha256,
        estimator_sha256=profile.estimator.estimator_sha256,
        model_family_sha256=(
            profile.small_model_family_sha256
            if small
            else profile.production_model_family_sha256
        ),
        vocabulary_sha256=vocabulary_identity_sha256(
            arm_config.vocabulary
        ),
        data_safety_sha256=profile.data_safety_sha256,
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
    )


def _non_particle_profile_payload(
    profile: H6PrefixProfilePair,
) -> dict[str, object]:
    payload = _resolved_profile_payload(profile)
    payload.pop("profile_id")
    payload.pop("profile_pair_sha256")
    estimator = dict(payload["estimator"])
    estimator.pop("particle_count")
    estimator.pop("estimator_sha256")
    payload["estimator"] = estimator
    return payload


def _validate_v2_resolved_runner_config(
    config: H6PrefixResolvedConfig,
) -> None:
    if type(config) is not H6PrefixResolvedConfig:
        raise ValueError("config must be an exact H6PrefixResolvedConfig")
    config.__post_init__()
    if config.schema_version != "h6-prefix-config-v2":
        raise ValueError("bounded runner plans require h6-prefix-config-v2")
    workload = config.workload_plan
    if type(workload) is not H6PrefixWorkloadPlan:
        raise ValueError("bounded runner config requires the exact workload plan")
    workload.__post_init__()
    try:
        payload = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("resolved v2 config canonical_json is invalid") from exc
    if (
        type(payload) is not dict
        or frozenset(payload) != _RESOLVED_PREFIX_CONFIG_V2_FIELDS
        or payload.get("schema_version") != "h6-prefix-config-v2"
        or payload.get("operation") != "H6-Prefix"
    ):
        raise ValueError("resolved v2 config field inventory is not exact")
    supplied_workload = payload.get("workload_plan")
    if canonical_json_bytes(supplied_workload) != canonical_json_bytes(
        workload.canonical_payload()
    ):
        raise ValueError("resolved v2 workload subtree is not canonical")
    if (
        payload.get("workload_plan_sha256")
        != workload.workload_plan_sha256
        or config.workload_plan_sha256
        != workload.workload_plan_sha256
        or config.authorization_sha256
        != H6_PREFIX_V2_AUTHORIZATION_SHA256
        or payload.get("authorization_sha256")
        != H6_PREFIX_V2_AUTHORIZATION_SHA256
    ):
        raise ValueError("resolved v2 workload authorization identity is stale")
    reresolved = resolve_h6_prefix_config(
        {
            "schema_version": config.schema_version,
            "operation": config.operation,
            "source": {
                "git_head": config.source.git_head,
                "dirty_digest": config.source.dirty_digest,
                "source_sha256": config.source.source_sha256,
            },
            "execution_mode": config.execution_mode,
            "profiles": [
                _resolver_profile_payload(profile)
                for profile in config.profiles
            ],
            "workload_plan_sha256": workload.workload_plan_sha256,
            "authorization_sha256": config.authorization_sha256,
            "artifact_root": str(config.artifact_root),
        },
        repo_root=_REPO_ROOT,
    )
    if reresolved != config:
        raise ValueError(
            "resolved v2 config does not equal the public resolver result"
        )
    expected = {
        "schema_version": config.schema_version,
        "operation": config.operation,
        "source": {
            "git_head": config.source.git_head,
            "dirty_digest": config.source.dirty_digest,
            "source_sha256": config.source.source_sha256,
        },
        "execution_mode": config.execution_mode,
        "profiles": tuple(
            _resolved_profile_payload(profile) for profile in config.profiles
        ),
        "artifact_root": config.artifact_root.as_posix(),
        "workload_plan": workload.canonical_payload(),
        "workload_plan_sha256": workload.workload_plan_sha256,
        "authorization_sha256": config.authorization_sha256,
    }
    canonical = canonical_json_bytes(expected)
    if (
        config.canonical_json.encode("utf-8") != canonical
        or config.config_sha256 != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError("resolved v2 config does not equal its typed reconstruction")
    if (
        len({profile.profile_id for profile in config.profiles})
        != len(config.profiles)
        or len(
            {profile.profile_pair_sha256 for profile in config.profiles}
        )
        != len(config.profiles)
    ):
        raise ValueError("resolved v2 profile identities must be unique")
    _validate_execution_profile_inventory(
        execution_mode=config.execution_mode,
        authorization_sha256=config.authorization_sha256,
        profiles=config.profiles,
        workload_plan=workload,
    )
    dependency_payload = dict(payload)
    dependency_payload.pop("workload_plan")
    _reject_prefix_dependencies(
        dependency_payload,
        "resolved_h6_prefix_v2",
    )


def _validate_v3_resolved_runner_config(
    config: H6PrefixV3ResolvedConfig,
) -> None:
    if type(config) is not H6PrefixV3ResolvedConfig:
        raise ValueError("config must be an exact H6PrefixV3ResolvedConfig")
    config.__post_init__()
    workload = config.workload_plan
    if type(workload) is not H6PrefixWorkloadPlan:
        raise ValueError("v3 runner config requires the exact workload plan")
    workload.__post_init__()
    try:
        payload = json.loads(config.canonical_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("resolved v3 config canonical_json is invalid") from exc
    if (
        type(payload) is not dict
        or frozenset(payload) != _RESOLVED_PREFIX_CONFIG_V3_FIELDS
        or payload.get("schema_version") != "h6-prefix-config-v3"
        or payload.get("operation") != "H6-Prefix"
        or payload.get("workload_plan_sha256")
        != workload.workload_plan_sha256
        or payload.get("workload_authorization_sha256")
        != H6_PREFIX_V2_AUTHORIZATION_SHA256
        or config.workload_authorization_sha256
        != H6_PREFIX_V2_AUTHORIZATION_SHA256
        or payload.get("authorization_sha256")
        != config.authorization_sha256
        or payload.get("validation_fixture_reference")
        != config.validation_fixture_reference.to_payload()
        or payload.get("validation_perturbation_reference")
        != config.validation_perturbation_reference.to_payload()
    ):
        raise ValueError(
            "resolved v3 field, workload, authorization, or reference "
            "binding is stale"
        )
    if canonical_json_bytes(payload.get("workload_plan")) != (
        canonical_json_bytes(workload.canonical_payload())
    ):
        raise ValueError("resolved v3 workload subtree is not canonical")
    reresolved = resolve_h6_prefix_config(
        {
            "schema_version": config.schema_version,
            "operation": config.operation,
            "source": {
                "git_head": config.source.git_head,
                "dirty_digest": config.source.dirty_digest,
                "source_sha256": config.source.source_sha256,
            },
            "execution_mode": config.execution_mode,
            "profiles": [
                _resolver_profile_payload(profile)
                for profile in config.profiles
            ],
            "workload_plan_sha256": workload.workload_plan_sha256,
            "workload_authorization_sha256": (
                config.workload_authorization_sha256
            ),
            "validation_fixture_reference": (
                config.validation_fixture_reference.to_payload()
            ),
            "validation_perturbation_reference": (
                config.validation_perturbation_reference.to_payload()
            ),
            "authorization_sha256": config.authorization_sha256,
            "artifact_root": str(config.artifact_root),
        },
        repo_root=_REPO_ROOT,
    )
    if type(reresolved) is not H6PrefixV3ResolvedConfig or reresolved != config:
        raise ValueError(
            "resolved v3 config does not equal the public resolver result"
        )
    if (
        len({profile.profile_id for profile in config.profiles})
        != len(config.profiles)
        or len(
            {profile.profile_pair_sha256 for profile in config.profiles}
        )
        != len(config.profiles)
    ):
        raise ValueError("resolved v3 profile identities must be unique")
    _validate_execution_profile_inventory(
        execution_mode=config.execution_mode,
        authorization_sha256=config.workload_authorization_sha256,
        profiles=config.profiles,
        workload_plan=workload,
    )
    dependency_payload = dict(payload)
    dependency_payload.pop("workload_plan")
    _reject_prefix_dependencies(
        dependency_payload,
        "resolved_h6_prefix_v3",
    )


def _build_h6_prefix_runner_plan(
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig,
) -> _H6PrefixRunnerPlan:
    """Purely validate and freeze an exact bounded v2/v3 execution graph."""

    if type(config) is H6PrefixResolvedConfig:
        _validate_v2_resolved_runner_config(config)
    elif type(config) is H6PrefixV3ResolvedConfig:
        _validate_v3_resolved_runner_config(config)
    else:
        raise ValueError("bounded runner requires an exact v2 or v3 config")
    workload = config.workload_plan
    assert type(workload) is H6PrefixWorkloadPlan
    observed = tuple(
        (_profile_semantic_key(profile), profile.estimator.particle_count)
        for profile in config.profiles
    )
    ordered_semantics = tuple(dict.fromkeys(key for key, _ in observed))
    expected_inventory = tuple(
        (semantic_key, particle_count)
        for semantic_key in ordered_semantics
        for particle_count in workload.production_particle_counts
    )
    if observed != expected_inventory:
        raise ValueError(
            "bounded runner requires complete ordered, unmixed semantic families"
        )
    families: list[_H6PrefixSemanticFamilyPlan] = []
    profile_offset = 0
    for family_index, semantic_key in enumerate(ordered_semantics):
        profiles = config.profiles[
            profile_offset : profile_offset
            + len(workload.production_particle_counts)
        ]
        profile_offset += len(profiles)
        if (
            len(profiles) != len(workload.production_particle_counts)
            or any(
                _profile_semantic_key(profile) != semantic_key
                for profile in profiles
            )
        ):
            raise ValueError("bounded semantic family is missing or mixed")
        baseline = canonical_json_bytes(
            _non_particle_profile_payload(profiles[0])
        )
        if any(
            canonical_json_bytes(_non_particle_profile_payload(profile))
            != baseline
            for profile in profiles[1:]
        ):
            raise ValueError(
                "only particle count may vary within a bounded semantic family"
            )
        jobs: list[_H6PrefixDynamicJob] = []
        for profile in profiles:
            particle_count = profile.estimator.particle_count
            representative = (
                particle_count == workload.representative_particle_count
            )
            scope: Literal[
                "representative_exhaustive", "estimator_stratified"
            ] = (
                "representative_exhaustive"
                if representative
                else "estimator_stratified"
            )
            for case_family, small in (
                ("small", True),
                ("validation", False),
            ):
                selected = _bounded_indices(case_family, scope)
                expected_calls = (
                    len(selected) * workload.prediction_calls_per_case
                )
                jobs.append(
                    _H6PrefixDynamicJob(
                        semantic_family_index=family_index,
                        profile=profile,
                        case_family=case_family,
                        scope=scope,
                        particle_count=particle_count,
                        selected_global_indices=selected,
                        report_key=_prefix_key(
                            config=config,
                            profile=profile,
                            small=small,
                        ),
                        expected_case_count=len(selected),
                        expected_predictor_call_count=expected_calls,
                        expected_particle_call_units=(
                            expected_calls * particle_count
                        ),
                        collect_source_masks=representative,
                        collect_validation_safety=(
                            representative and not small
                        ),
                    )
                )
        family = _H6PrefixSemanticFamilyPlan(
            semantic_family_index=family_index,
            semantic_key=semantic_key,
            profiles=profiles,
            dynamic_jobs=tuple(jobs),
            arm_build_count=2,
            predictor_boundary_count=8,
            dynamic_report_count=8,
            representative_mask_collector_count=2,
            expected_case_count=sum(job.expected_case_count for job in jobs),
            expected_predictor_call_count=sum(
                job.expected_predictor_call_count for job in jobs
            ),
            expected_particle_call_units=sum(
                job.expected_particle_call_units for job in jobs
            ),
        )
        families.append(family)
    if profile_offset != len(config.profiles):
        raise ValueError("bounded profile inventory contains an extra profile")
    report_keys = tuple(
        job.report_key for family in families for job in family.dynamic_jobs
    )
    static_job = _H6PrefixStaticAuditJob(
        report_keys=report_keys,
        case_key_manifest_sha256=_case_key_manifest(report_keys),
    )
    runner_plan = _H6PrefixRunnerPlan(
        config=config,
        semantic_families=tuple(families),
        expected_validation_vocabulary=(
            families[0].profiles[0].production_arm_config.vocabulary
        ),
        static_audit_job=static_job,
        fixture_load_count=1,
        static_audit_count=1,
        expected_case_count=sum(
            family.expected_case_count for family in families
        ),
        expected_predictor_call_count=sum(
            family.expected_predictor_call_count for family in families
        ),
        expected_particle_call_units=sum(
            family.expected_particle_call_units for family in families
        ),
    )
    return runner_plan


def _bounded_workload_authorization_sha256(
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig,
) -> str:
    if (
        type(config) is H6PrefixResolvedConfig
        and config.schema_version == "h6-prefix-config-v2"
        and config.authorization_sha256
        == H6_PREFIX_V2_AUTHORIZATION_SHA256
    ):
        return config.authorization_sha256
    if (
        type(config) is H6PrefixV3ResolvedConfig
        and config.workload_authorization_sha256
        == H6_PREFIX_V2_AUTHORIZATION_SHA256
    ):
        return config.workload_authorization_sha256
    raise ValueError(
        "bounded runner config does not bind the v2 workload authorization"
    )


def _validated_dynamic_job_result(
    *,
    plan: _H6PrefixRunnerPlan,
    job: _H6PrefixDynamicJob,
    result: object,
) -> tuple[DynamicExecutionPlan, DynamicPrefixReport, int]:
    execution_plan = getattr(result, "execution_plan", None)
    report = getattr(result, "report", None)
    observed_calls = getattr(
        result,
        "observed_predictor_call_count",
        None,
    )
    if type(execution_plan) is not DynamicExecutionPlan:
        raise RuntimeError(
            "bounded dynamic job result requires an exact execution plan"
        )
    if type(report) is not DynamicPrefixReport:
        raise RuntimeError(
            "bounded dynamic job result requires an exact dynamic report"
        )
    execution_plan.__post_init__()
    report.__post_init__()
    if (
        execution_plan.schema_version
        != "h6-dynamic-execution-plan-v2"
        or execution_plan.scope != job.scope
        or execution_plan.case_family != job.case_family
        or execution_plan.particle_count != job.particle_count
        or execution_plan.workload_plan_sha256
        != plan.config.workload_plan_sha256
        or execution_plan.authorization_sha256
        != _bounded_workload_authorization_sha256(plan.config)
        or execution_plan.selected_global_indices
        != job.selected_global_indices
        or report.schema_version != "h6-dynamic-prefix-report-v2"
        or report.key != job.report_key
        or report.scope != job.scope
        or report.case_family != job.case_family
        or report.particle_count != job.particle_count
        or report.workload_plan_sha256
        != plan.config.workload_plan_sha256
        or report.selected_global_indices != job.selected_global_indices
        or report.execution_plan_sha256 != execution_plan.plan_sha256
        or report.selection_manifest_sha256
        != execution_plan.selection_manifest_sha256
    ):
        raise RuntimeError(
            "bounded dynamic job result does not bind its frozen job"
        )
    if (
        type(observed_calls) is not int
        or observed_calls != job.expected_predictor_call_count
    ):
        raise RuntimeError(
            "bounded dynamic job violated the exact five-call budget"
        )
    return execution_plan, report, observed_calls


def _load_h6_prefix_v3_validation_inputs(
    config: H6PrefixV3ResolvedConfig,
    *,
    expected_vocabulary: VocabularyIdentity,
) -> object:
    """Load, reconstruct, and fully cross-check v3 inputs before any arm."""

    _validate_v3_resolved_runner_config(config)
    if type(expected_vocabulary) is not VocabularyIdentity:
        raise ValueError("expected validation vocabulary must be exact")
    expected_vocabulary.__post_init__()
    candidate_reference = config.validation_perturbation_reference
    if (
        expected_vocabulary.vocabulary_id
        != candidate_reference.vocabulary_id
        or expected_vocabulary.size
        != candidate_reference.vocabulary_size
        or expected_vocabulary.tokenizer_spec_sha256
        != candidate_reference.tokenizer_spec_sha256
        or vocabulary_identity_sha256(expected_vocabulary)
        != candidate_reference.vocabulary_sha256
    ):
        raise ValueError(
            "expected validation vocabulary differs from the v3 reference"
        )

    fixture_payload = read_validation_safety_fixture_payload(
        config.validation_fixture_reference
    )
    if type(fixture_payload) is not ValidationSafetyFixturePayload:
        raise RuntimeError(
            "fixture loader did not return an exact validation fixture payload"
        )
    fixture_payload.__post_init__()
    if fixture_payload.reference != config.validation_fixture_reference:
        raise RuntimeError(
            "loaded validation fixture reference differs from v3 config"
        )
    fixture_bytes = fixture_payload.fixture_bytes
    starts = fixture_payload.starts
    real_target_counts = fixture_payload.real_target_counts
    if (
        type(fixture_bytes) is not bytes
        or len(fixture_bytes)
        != config.validation_fixture_reference.fixture_raw_length
        or hashlib.sha256(fixture_bytes).hexdigest()
        != config.validation_fixture_reference.fixture_raw_sha256
        or fixture_payload.validation_token_sha256
        != config.validation_fixture_reference.validation_token_sha256
        or type(starts) is not tuple
        or len(starts) != 4096
        or any(type(value) is not int for value in starts)
        or type(real_target_counts) is not tuple
        or len(real_target_counts) != 4096
        or any(
            type(value) is not int or not 1 <= value <= 32
            for value in real_target_counts
        )
    ):
        raise RuntimeError(
            "loaded validation fixture bytes or scalars differ from v3 config"
        )

    candidate_payload = load_h6_validation_perturbation_artifact_payload(
        candidate_reference.local_artifact_path
    )
    if type(candidate_payload) is not H6ValidationPerturbationArtifactPayload:
        raise RuntimeError(
            "candidate loader did not return an exact same-pass payload"
        )
    candidate_payload.__post_init__()
    if candidate_payload.reference != candidate_reference:
        raise RuntimeError(
            "loaded perturbation reference differs from v3 config"
        )

    validation_fixture = ValidationSafetyFixture.create(
        validation_token_sha256=(
            config.validation_fixture_reference.validation_token_sha256
        ),
        starts=starts,
        real_target_counts=real_target_counts,
        fixture_bytes=fixture_bytes,
    )
    if (
        validation_fixture.fixture_sha256
        != config.validation_fixture_reference.fixture_raw_sha256
        or validation_fixture.validation_token_sha256
        != config.validation_fixture_reference.validation_token_sha256
    ):
        raise RuntimeError(
            "reconstructed validation fixture differs from v3 references"
        )
    perturbations = load_frozen_validation_perturbations(
        candidate_payload.candidate_bytes,
        expected_vocabulary=expected_vocabulary,
        validation_fixture=validation_fixture,
        validation_fixture_bytes=fixture_bytes,
    )
    records = getattr(perturbations, "records", None)
    if (
        getattr(perturbations, "source_fixture_verified", None) is not True
        or getattr(perturbations, "materialization", None)
        != "authorized_full"
        or getattr(perturbations, "schema_version", None)
        != "h6-validation-perturbations-v1"
        or getattr(perturbations, "generator_version", None)
        != "h6-validation-perturbations-v1"
        or getattr(perturbations, "seed", None) != 2026072197
        or type(getattr(perturbations, "seed", None)) is not int
        or getattr(perturbations, "full_count", None) != 4096
        or type(getattr(perturbations, "full_count", None)) is not int
        or getattr(perturbations, "materialized_count", None) != 4096
        or type(getattr(perturbations, "materialized_count", None)) is not int
        or type(records) is not tuple
        or len(records) != 4096
        or tuple(getattr(record, "case_index", None) for record in records)
        != tuple(range(4096))
        or getattr(perturbations, "vocabulary", None)
        != expected_vocabulary
        or getattr(perturbations, "vocabulary_sha256", None)
        != candidate_reference.vocabulary_sha256
        or getattr(perturbations, "validation_token_sha256", None)
        != candidate_reference.validation_token_sha256
        or getattr(
            perturbations,
            "validation_safety_fixture_sha256",
            None,
        )
        != candidate_reference.fixture_raw_sha256
        or getattr(perturbations, "manifest_sha256", None)
        != candidate_reference.perturbation_inner_manifest_sha256
        or getattr(perturbations, "raw_sha256", None)
        != candidate_reference.perturbation_raw_sha256
        or getattr(perturbations, "canonical_bytes", None)
        != candidate_payload.candidate_bytes
    ):
        raise RuntimeError(
            "loaded perturbations are incomplete or differ from v3 references"
        )
    return perturbations


def _execute_h6_prefix_plan(
    plan: _H6PrefixRunnerPlan,
    executor: _H6PrefixRunnerExecutor,
) -> _H6PrefixRunnerEvidence:
    """Execute a frozen plan while retaining orchestration and budgets here."""

    if type(plan) is not _H6PrefixRunnerPlan:
        raise ValueError("plan must be an exact bounded runner plan")
    if plan != _build_h6_prefix_runner_plan(plan.config):
        raise ValueError("bounded runner plan differs from its pure reconstruction")
    if not isinstance(executor, _H6PrefixRunnerExecutor):
        raise ValueError("executor does not implement the bounded runner protocol")
    production_vocabularies = tuple(
        family.profiles[0].production_arm_config.vocabulary
        for family in plan.semantic_families
    )
    if any(
        vocabulary != plan.expected_validation_vocabulary
        for vocabulary in production_vocabularies
    ):
        raise RuntimeError(
            "bounded semantic families disagree on production vocabulary"
        )
    perturbations = executor.load_validation_perturbations(
        expected_vocabulary=plan.expected_validation_vocabulary
    )
    dynamic_results: list[_H6PrefixDynamicJobEvidence] = []
    for family in plan.semantic_families:
        representative = family.profiles[0]
        small_arm = executor.build_arm(
            arm_config=representative.small_arm_config,
            structure=representative.small_structure,
        )
        production_arm = executor.build_arm(
            arm_config=representative.production_arm_config,
            structure=representative.production_structure,
        )
        for profile_index, profile in enumerate(family.profiles):
            small_job = family.dynamic_jobs[2 * profile_index]
            validation_job = family.dynamic_jobs[2 * profile_index + 1]
            small_predictor = executor.build_predictor_boundary(
                built_arm=small_arm,
                estimator=profile.estimator,
            )
            production_predictor = executor.build_predictor_boundary(
                built_arm=production_arm,
                estimator=profile.estimator,
            )
            for job, built_arm_value, predictor in (
                (small_job, small_arm, small_predictor),
                (
                    validation_job,
                    production_arm,
                    production_predictor,
                ),
            ):
                result = executor.execute_dynamic_job(
                    job=job,
                    built_arm=built_arm_value,
                    predictor=predictor,
                    validation_perturbations=perturbations,
                )
                (
                    execution_plan,
                    report,
                    observed_calls,
                ) = _validated_dynamic_job_result(
                    plan=plan,
                    job=job,
                    result=result,
                )
                dynamic_results.append(
                    _H6PrefixDynamicJobEvidence(
                        job=job,
                        execution_plan=execution_plan,
                        report=report,
                        observed_predictor_call_count=observed_calls,
                    )
                )
    static_report = executor.execute_static_audit(
        job=plan.static_audit_job
    )
    if type(static_report) is not StaticAuditReport:
        raise RuntimeError(
            "bounded static audit report must be an exact StaticAuditReport"
        )
    static_report.__post_init__()
    if (
        static_report.case_key_manifest_sha256
        != plan.static_audit_job.case_key_manifest_sha256
    ):
        raise RuntimeError(
            "bounded static audit report does not bind the frozen report keys"
        )
    observed_total = sum(
        result.observed_predictor_call_count
        for result in dynamic_results
    )
    if observed_total != plan.expected_predictor_call_count:
        raise RuntimeError("bounded runner predictor-call total is stale")
    return _H6PrefixRunnerEvidence(
        plan=plan,
        dynamic_results=tuple(dynamic_results),
        static_report=static_report,
        observed_predictor_call_count=observed_total,
    )


def _execute_h6_prefix_v3_plan(
    plan: _H6PrefixRunnerPlan,
    executor: _H6PrefixRunnerExecutor,
) -> _H6PrefixRunnerEvidence:
    """Private pre-publication entry point for focused v3 orchestration."""

    if (
        type(plan) is not _H6PrefixRunnerPlan
        or type(plan.config) is not H6PrefixV3ResolvedConfig
    ):
        raise ValueError("private v3 execution requires an exact v3 plan")
    return _execute_h6_prefix_plan(plan, executor)


def _small_cases(
    execution_mode: str,
) -> tuple[DynamicPrefixCase, ...]:
    if execution_mode == "focused_subset":
        oracle_cases = enumerate_ordered_tail_pairs(
            case_indices=_FOCUSED_SMALL_CASE_INDICES,
            max_cases=len(_FOCUSED_SMALL_CASE_INDICES),
        )
    elif execution_mode == "authorized_full":
        oracle_cases = enumerate_ordered_tail_pairs()
    else:
        raise ValueError("unsupported H6-Prefix execution mode")
    return tuple(
        DynamicPrefixCase.create(
            ordinal=case.case_index,
            receiver_t=case.receiver_t,
            shared_prefix=case.prefix,
            left_tail=case.left_tail,
            right_tail=case.right_tail,
        )
        for case in oracle_cases
    )


def _source_mask_evidence(
    *,
    built_arm: BuiltArm,
    predictor: BootstrapSmcPredictor,
    arm_config: ArmConfig,
    probe_receiver_t: int,
) -> tuple[
    SourceMaskObserver | None,
    AllInvalidSourceObservation | None,
]:
    """Bind a collector to the already-required first validated prediction."""

    if arm_config.source_mode != "categorical":
        return None, None
    if (
        built_arm.config is not arm_config
        or type(built_arm.model) is not LatentLanguageArmModel
        or predictor.proposal.model is not built_arm.model
        or predictor.predictor_config_sha256 != arm_config.config_sha256
    ):
        raise RuntimeError(
            "source-mask evidence must use the exact built arm and predictor"
        )
    source_prior = built_arm.model.source_prior
    if source_prior is None:
        raise RuntimeError("categorical arm has no live source prior")
    if type(probe_receiver_t) is not int or probe_receiver_t <= 0:
        raise ValueError("source-mask probe receiver must be positive")

    def observe(
        case: DynamicPrefixCase,
        prediction: PriorPrediction,
    ) -> tuple[SourceMaskObservation, ...]:
        if type(case) is not DynamicPrefixCase:
            raise ValueError("source-mask observer requires an exact case")
        case.__post_init__()
        if type(prediction) is not PriorPrediction:
            raise ValueError(
                "source-mask observer requires the validated PriorPrediction"
            )
        prediction.__post_init__()
        prefix = CausalPrefix.create(
            receiver_t=case.receiver_t,
            vocabulary=arm_config.vocabulary,
            token_ids=torch.tensor(case.shared_prefix, dtype=torch.int64),
        )
        population = prediction.cache.filtered_population
        row = source_prior.structure.dag.rows[case.receiver_t - 1]
        if row.receiver_t != case.receiver_t:
            raise RuntimeError("source-prior receiver is not the Prefix receiver")
        histories = {
            "state": population.component("state_history")[0],
        }
        if arm_config.model_channel_enabled:
            histories["model"] = population.component("model_history")[0]
        observations: list[SourceMaskObservation] = []
        for bank, history in histories.items():
            if arm_config.prior_variant == "fixed":
                log_probabilities = (
                    built_arm.model.state_source_log_probs(case.receiver_t)
                    if bank == "state"
                    else built_arm.model.model_source_log_probs(case.receiver_t)
                )
            elif arm_config.prior_variant == "parent_specific_pooled_prefix":
                log_probabilities = (
                    built_arm.model.state_source_log_probs(
                        case.receiver_t,
                        prefix=prefix,
                        earlier_latents=history,
                    )
                    if bank == "state"
                    else built_arm.model.model_source_log_probs(
                        case.receiver_t,
                        prefix=prefix,
                        earlier_latents=history,
                    )
                )
            else:
                raise RuntimeError("categorical Prefix arm has an unsupported prior")
            observations.append(
                SourceMaskObservation.capture(
                    case_sha256=case.case_sha256,
                    config_sha256=arm_config.config_sha256,
                    bank=bank,
                    receiver_t=case.receiver_t,
                    declared_parents=row.parents,
                    log_probabilities=log_probabilities,
                )
            )
        return tuple(observations)

    def all_invalid_probe() -> object:
        return masked_log_softmax_from_parents(
            torch.zeros(probe_receiver_t, dtype=torch.float64),
            (),
            probe_receiver_t,
        )

    return (
        observe,
        observe_all_invalid_source_rejection(
            config_sha256=arm_config.config_sha256,
            receiver_t=probe_receiver_t,
            probe=all_invalid_probe,
        ),
    )


def _validate_resolved_runner_config(
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig,
) -> None:
    if type(config) is H6PrefixV3ResolvedConfig:
        _validate_v3_resolved_runner_config(config)
        observed_source = current_source_identity(
            _REPO_ROOT,
            config.artifact_root,
        )
        configured_source = (
            config.source.git_head,
            config.source.dirty_digest,
            config.source.source_sha256,
        )
        if observed_source != configured_source:
            raise ValueError(
                "H6-Prefix source identity is stale for the live candidate"
            )
        return
    if type(config) is not H6PrefixResolvedConfig:
        raise ValueError(
            "config must be an exact H6PrefixResolvedConfig or "
            "H6PrefixV3ResolvedConfig"
        )
    config.__post_init__()
    if config.schema_version == "h6-prefix-config-v2":
        _validate_v2_resolved_runner_config(config)
        observed_source = current_source_identity(
            _REPO_ROOT,
            config.artifact_root,
        )
        configured_source = (
            config.source.git_head,
            config.source.dirty_digest,
            config.source.source_sha256,
        )
        if observed_source != configured_source:
            raise ValueError(
                "H6-Prefix source identity is stale for the live candidate"
            )
        return
    if config.schema_version != "h6-prefix-config-v1":
        raise ValueError("unsupported resolved H6-Prefix schema version")
    if type(config.canonical_json) is not str or not isinstance(
        config.artifact_root, Path
    ):
        raise ValueError("resolved H6-Prefix config identity is incomplete")
    if (
        type(config.profiles) is not tuple
        or not config.profiles
        or any(
            type(profile) is not H6PrefixProfilePair
            for profile in config.profiles
        )
    ):
        raise ValueError("resolved H6-Prefix profiles are incomplete")
    for profile in config.profiles:
        profile.__post_init__()
    _validate_execution_profile_inventory(
        execution_mode=config.execution_mode,
        authorization_sha256=config.authorization_sha256,
        profiles=config.profiles,
        workload_plan=None,
    )
    expected_config = {
        "schema_version": config.schema_version,
        "operation": config.operation,
        "source": {
            "git_head": config.source.git_head,
            "dirty_digest": config.source.dirty_digest,
            "source_sha256": config.source.source_sha256,
        },
        "execution_mode": config.execution_mode,
        "profiles": tuple(
            _resolved_profile_payload(profile) for profile in config.profiles
        ),
        "authorization_sha256": config.authorization_sha256,
        "artifact_root": config.artifact_root.as_posix(),
    }
    canonical = canonical_json_bytes(expected_config)
    if (
        config.canonical_json.encode("utf-8") != canonical
        or config.config_sha256 != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError(
            "resolved H6-Prefix canonical config/source identity is stale"
        )
    observed_source = current_source_identity(
        _REPO_ROOT,
        config.artifact_root,
    )
    configured_source = (
        config.source.git_head,
        config.source.dirty_digest,
        config.source.source_sha256,
    )
    if observed_source != configured_source:
        raise ValueError(
            "H6-Prefix source identity is stale for the live candidate"
        )


class _CountingPriorPredictor:
    def __init__(self, delegate: BootstrapSmcPredictor) -> None:
        self.delegate = delegate
        self.call_count = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: object,
        cache: object = None,
    ) -> PriorPrediction:
        self.call_count += 1
        return self.delegate.next_token_log_probs(
            prefix_tokens,
            estimator_rng,
            cache,
        )


@dataclass(frozen=True, slots=True)
class _LiveH6PrefixDynamicJobResult:
    execution_plan: DynamicExecutionPlan
    report: DynamicPrefixReport
    observed_predictor_call_count: int


def _require_live_structure_binding(
    expected_structure: H6LanguageStructure,
    live_structure_sha256: str,
    *,
    live_fixture_sha256: str | None = None,
) -> None:
    if type(expected_structure) is not H6LanguageStructure:
        raise ValueError(
            "expected language structure must be an exact record"
        )
    expected_structure.__post_init__()
    observed = _require_sha256(
        live_structure_sha256,
        "live language structure SHA-256",
    )
    fixture = (
        observed
        if live_fixture_sha256 is None
        else _require_sha256(
            live_fixture_sha256,
            "live source-prior fixture SHA-256",
        )
    )
    if (
        observed != expected_structure.structure_sha256
        or fixture != expected_structure.structure_sha256
    ):
        raise RuntimeError(
            "live arm language structure does not bind the planned structure"
        )


class _LiveH6PrefixRunnerExecutor:
    """Real adapter kept private until Task 3E can compose bounded artifacts."""

    def load_validation_perturbations(
        self, *, expected_vocabulary: VocabularyIdentity
    ) -> FrozenValidationPerturbations:
        return load_frozen_validation_perturbations(
            expected_vocabulary=expected_vocabulary
        )

    def build_arm(
        self,
        *,
        arm_config: ArmConfig,
        structure: H6LanguageStructure,
    ) -> BuiltArm:
        arm_config.__post_init__()
        structure.__post_init__()
        if (
            structure.receiver_labels
            != tuple(range(1, arm_config.horizon + 1))
        ):
            raise RuntimeError("live arm structure does not match its horizon")
        built_arm = build_arm(arm_config.arm, arm_config)
        if arm_config.source_mode == "categorical":
            if (
                type(built_arm.model) is not LatentLanguageArmModel
                or built_arm.model.source_prior is None
                or type(built_arm.model.source_prior.structure)
                is not H6LanguageStructure
            ):
                raise RuntimeError(
                    "categorical live arm does not expose its language structure"
                )
            live_structure = built_arm.model.source_prior.structure
            live_structure.__post_init__()
            _require_live_structure_binding(
                structure,
                live_structure.structure_sha256,
                live_fixture_sha256=(
                    built_arm.model.source_prior.fixture_sha256
                ),
            )
        return built_arm

    def build_predictor_boundary(
        self,
        *,
        built_arm: object,
        estimator: EstimatorSpec,
    ) -> _CountingPriorPredictor:
        if type(built_arm) is not BuiltArm:
            raise ValueError("predictor boundary requires an exact BuiltArm")
        estimator.__post_init__()
        _, predictor = built_arm.rebuild_predictive_boundary(estimator)
        return _CountingPriorPredictor(predictor)

    @staticmethod
    def _small_job_cases(
        job: _H6PrefixDynamicJob,
    ) -> tuple[DynamicPrefixCase, ...]:
        if job.scope == "representative_exhaustive":
            cases = _small_cases("authorized_full")
        else:
            oracle_cases = enumerate_ordered_tail_pairs(
                case_indices=job.selected_global_indices,
                max_cases=len(job.selected_global_indices),
            )
            cases = tuple(
                DynamicPrefixCase.create(
                    ordinal=case.case_index,
                    receiver_t=case.receiver_t,
                    shared_prefix=case.prefix,
                    left_tail=case.left_tail,
                    right_tail=case.right_tail,
                )
                for case in oracle_cases
            )
        if tuple(case.ordinal for case in cases) != job.selected_global_indices:
            raise RuntimeError("small Prefix selection is stale")
        return cases

    @staticmethod
    def _validation_job_cases(
        job: _H6PrefixDynamicJob,
        perturbations: FrozenValidationPerturbations,
    ) -> tuple[DynamicPrefixCase, ...]:
        perturbations.__post_init__()
        full_cases = perturbations.dynamic_cases
        if (
            perturbations.materialization,
            perturbations.materialized_count,
            len(full_cases),
        ) != ("authorized_full", VALIDATION_EXPECTED_TOTAL, VALIDATION_EXPECTED_TOTAL):
            raise RuntimeError(
                "bounded Prefix execution requires the verified full validation fixture"
            )
        if job.scope == "representative_exhaustive":
            selected = full_cases
        else:
            selected = tuple(
                full_cases[index] for index in job.selected_global_indices
            )
        if tuple(case.ordinal for case in selected) != job.selected_global_indices:
            raise RuntimeError("validation Prefix selection is stale")
        return selected

    def execute_dynamic_job(
        self,
        *,
        job: _H6PrefixDynamicJob,
        built_arm: object,
        predictor: object,
        validation_perturbations: object,
    ) -> _LiveH6PrefixDynamicJobResult:
        if (
            type(job) is not _H6PrefixDynamicJob
            or type(built_arm) is not BuiltArm
            or type(predictor) is not _CountingPriorPredictor
            or predictor.call_count != 0
        ):
            raise ValueError("live dynamic boundary is not fresh and exact")
        expected_config = (
            job.profile.small_arm_config
            if job.case_family == "small"
            else job.profile.production_arm_config
        )
        expected_model_family = (
            job.profile.small_model_family_sha256
            if job.case_family == "small"
            else job.profile.production_model_family_sha256
        )
        if (
            built_arm.config is not expected_config
            or built_arm.model_family_sha256 != expected_model_family
            or predictor.predictor_config_sha256
            != expected_config.config_sha256
            or predictor.data_safety_sha256
            != job.profile.data_safety_sha256
        ):
            raise RuntimeError("live dynamic boundary differs from its frozen job")
        if job.case_family == "small":
            cases = self._small_job_cases(job)
            bound_perturbations = None
        else:
            if type(validation_perturbations) is not FrozenValidationPerturbations:
                raise ValueError("live validation fixture must be exact")
            cases = self._validation_job_cases(
                job,
                validation_perturbations,
            )
            bound_perturbations = (
                validation_perturbations
                if job.collect_validation_safety
                else None
            )
        selection_rows = tuple(
            (case.ordinal, case.case_sha256) for case in cases
        )
        workload = H6PrefixWorkloadPlan()
        execution_plan = DynamicExecutionPlan.create_scoped(
            scope=job.scope,
            case_family=job.case_family,
            particle_count=job.particle_count,
            workload_plan=workload,
            authorization_sha256=H6_PREFIX_V2_AUTHORIZATION_SHA256,
            selection_rows=selection_rows,
        )
        source_mask_observer: SourceMaskObserver | None = None
        all_invalid: AllInvalidSourceObservation | None = None
        if job.collect_source_masks:
            source_mask_observer, all_invalid = _source_mask_evidence(
                built_arm=built_arm,
                predictor=predictor.delegate,
                arm_config=expected_config,
                probe_receiver_t=cases[0].receiver_t,
            )
        report = run_dynamic_prefix_checks(
            key=job.report_key,
            predictor=predictor,
            arm_config=expected_config,
            cases=cases,
            plan=execution_plan,
            stream_seed=2026072197,
            perturbations=bound_perturbations,
            source_mask_observations=None,
            all_invalid_observation=all_invalid,
            pair_side_harness=PairSideHarness(),
            source_mask_observer=source_mask_observer,
        )
        if predictor.call_count != job.expected_predictor_call_count:
            raise RuntimeError(
                "live bounded job made other than five predictor calls per case"
            )
        return _LiveH6PrefixDynamicJobResult(
            execution_plan=execution_plan,
            report=report,
            observed_predictor_call_count=predictor.call_count,
        )

    def execute_static_audit(
        self,
        *,
        job: _H6PrefixStaticAuditJob,
    ) -> StaticAuditReport:
        return audit_h6_static_source(_REPO_ROOT, job.report_keys)


class _LiveH6PrefixV3RunnerExecutor(_LiveH6PrefixRunnerExecutor):
    """Private live adapter that binds v3 inputs before inherited arm work."""

    def __init__(self, config: H6PrefixV3ResolvedConfig) -> None:
        if type(config) is not H6PrefixV3ResolvedConfig:
            raise ValueError("live v3 executor requires an exact v3 config")
        _validate_v3_resolved_runner_config(config)
        self._config = config

    def load_validation_perturbations(
        self,
        *,
        expected_vocabulary: VocabularyIdentity,
    ) -> object:
        return _load_h6_prefix_v3_validation_inputs(
            self._config,
            expected_vocabulary=expected_vocabulary,
        )


def run_h6_prefix(
    *,
    config: H6PrefixResolvedConfig | H6PrefixV3ResolvedConfig,
    junit_sha256: str | None,
) -> tuple[H6PrefixGateResult | H6BoundedPrefixGateResult, Path]:
    """Run bounded typed Prefix checks and publish only their derived status."""

    _validate_resolved_runner_config(config)
    if junit_sha256 is not None:
        _require_sha256(junit_sha256, "junit_sha256")
    if type(config) is H6PrefixV3ResolvedConfig:
        plan = _build_h6_prefix_runner_plan(config)
        runner_evidence = _execute_h6_prefix_v3_plan(
            plan,
            _LiveH6PrefixV3RunnerExecutor(config),
        )
        certificate_set = _compose_bounded_prefix_certificate_set(
            config=config,
            runner_evidence=runner_evidence,
        )
        _validate_resolved_runner_config(config)
        return publish_h6_bounded_prefix_artifact(
            artifact_root=config.artifact_root,
            run_name=(
                f"h6-prefix-{config.source.git_head}-"
                f"{config.config_sha256[:16]}"
            ),
            config_payload=json.loads(config.canonical_json),
            provenance_payload={
                "schema_version": "h6-prefix-provenance-v1",
                "git_head": config.source.git_head,
                "dirty_digest": config.source.dirty_digest,
                "source_sha256": config.source.source_sha256,
                "junit_sha256": junit_sha256,
            },
            environment_payload={
                "schema_version": "h6-prefix-environment-v1",
                "device": "cpu",
                "dtype": "float64",
                "python_implementation": platform.python_implementation(),
                "python_version": sys.version.split()[0],
            },
            runner_evidence=runner_evidence,
            certificate_set=certificate_set,
        )
    if getattr(config, "schema_version", "h6-prefix-config-v1") == (
        "h6-prefix-config-v2"
    ):
        _build_h6_prefix_runner_plan(config)
        raise RuntimeError(
            "bounded h6-prefix-config-v2 execution is blocked until Task 3E "
            "supplies the bounded certificate and artifact composer"
        )
    perturbations = load_frozen_validation_perturbations()
    if (
        config.execution_mode == "authorized_full"
        and perturbations.materialization != "authorized_full"
    ):
        raise RuntimeError(
            "authorized-full H6-Prefix requires the complete frozen "
            "4,096-record validation perturbation fixture"
        )
    small_cases = _small_cases(config.execution_mode)
    bundles: list[H6PrefixReportBundle] = []
    for profile in config.profiles:
        built_small = build_arm(
            profile.small_arm_config.arm,
            profile.small_arm_config,
        )
        built_production = build_arm(
            profile.production_arm_config.arm,
            profile.production_arm_config,
        )
        if (
            built_small.model_family_sha256
            != profile.small_model_family_sha256
            or built_production.model_family_sha256
            != profile.production_model_family_sha256
            or built_small.predictor.data_safety_sha256
            != profile.data_safety_sha256
            or built_production.predictor.data_safety_sha256
            != profile.data_safety_sha256
        ):
            raise RuntimeError(
                "resolved Prefix profile differs from the implemented arm "
                "factory/predictive safety boundary"
            )
        _, small_predictor = built_small.rebuild_predictive_boundary(
            profile.estimator
        )
        _, production_predictor = (
            built_production.rebuild_predictive_boundary(profile.estimator)
        )
        small_key = _prefix_key(
            config=config,
            profile=profile,
            small=True,
        )
        production_key = _prefix_key(
            config=config,
            profile=profile,
            small=False,
        )
        small_source_observer, small_all_invalid = _source_mask_evidence(
            built_arm=built_small,
            predictor=small_predictor,
            arm_config=profile.small_arm_config,
            probe_receiver_t=small_cases[0].receiver_t,
        )
        validation_source_observer, validation_all_invalid = (
            _source_mask_evidence(
                built_arm=built_production,
                predictor=production_predictor,
                arm_config=profile.production_arm_config,
                probe_receiver_t=(
                    perturbations.dynamic_cases[0].receiver_t
                ),
            )
        )
        small_report = run_dynamic_prefix_checks(
            key=small_key,
            predictor=small_predictor,
            arm_config=profile.small_arm_config,
            cases=small_cases,
            plan=DynamicExecutionPlan.create(
                mode=config.execution_mode,
                case_family="small",
                authorization_sha256=config.authorization_sha256,
            ),
            stream_seed=2026072197,
            source_mask_observations=None,
            all_invalid_observation=small_all_invalid,
            pair_side_harness=PairSideHarness(),
            source_mask_observer=small_source_observer,
        )
        validation_report = run_dynamic_prefix_checks(
            key=production_key,
            predictor=production_predictor,
            arm_config=profile.production_arm_config,
            cases=perturbations.dynamic_cases,
            plan=DynamicExecutionPlan.create(
                mode=config.execution_mode,
                case_family="validation",
                authorization_sha256=config.authorization_sha256,
            ),
            stream_seed=2026072197,
            perturbations=perturbations,
            source_mask_observations=None,
            all_invalid_observation=validation_all_invalid,
            pair_side_harness=PairSideHarness(),
            source_mask_observer=validation_source_observer,
        )
        static_report = audit_h6_static_source(
            _REPO_ROOT,
            (small_key, production_key),
        )
        certificate = compose_prefix_certificate(
            profile=profile,
            small_report=small_report,
            validation_report=validation_report,
            static_report=static_report,
        )
        bundles.append(
            H6PrefixReportBundle(
                profile=profile,
                small_report=small_report,
                validation_report=validation_report,
                static_report=static_report,
                certificate=certificate,
            )
        )
    config_payload = json.loads(config.canonical_json)
    provenance_payload = {
        "schema_version": "h6-prefix-provenance-v1",
        "git_head": config.source.git_head,
        "dirty_digest": config.source.dirty_digest,
        "source_sha256": config.source.source_sha256,
        "junit_sha256": junit_sha256,
    }
    environment_payload = {
        "schema_version": "h6-prefix-environment-v1",
        "device": "cpu",
        "dtype": "float64",
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version.split()[0],
    }
    return publish_h6_prefix_artifact(
        artifact_root=config.artifact_root,
        run_name=(
            f"h6-prefix-{config.source.git_head}-"
            f"{config.config_sha256[:16]}"
        ),
        config_payload=config_payload,
        provenance_payload=provenance_payload,
        environment_payload=environment_payload,
        report_bundles=tuple(bundles),
    )


__all__ = [
    "H6BoundedPrefixFamilyBundle",
    "H6PrefixReportBundle",
    "compose_bounded_prefix_certificate",
    "compose_prefix_certificate",
    "h6_bounded_prefix_artifact_payloads",
    "h6_prefix_artifact_payloads",
    "publish_h6_bounded_prefix_artifact",
    "publish_h6_prefix_artifact",
    "run_h6_prefix",
]
