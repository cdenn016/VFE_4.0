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

from verification.numpy_oracles.h6_prefix import enumerate_ordered_tail_pairs
from vfe4.artifacts.atomic import publish_run_directory
from vfe4.artifacts.provenance import current_source_identity
from vfe4.config.schema import H6PrefixResolvedConfig
from vfe4.predictive import vocabulary_identity_sha256
from vfe4.training.arms import build_arm
from vfe4.types.h6 import (
    H6_PREFIX_REQUIRED_CHECKS,
    EvidenceStatus,
    H6PrefixProfilePair,
    PrefixCaseKey,
    PrefixCertificate,
    PrefixReportBinding,
    canonical_json_bytes,
)
from vfe4.types.results import H6PrefixGateResult
from vfe4.validation.h6_prefix import (
    SMALL_EXPECTED_BY_POSITION,
    SMALL_EXPECTED_TOTAL,
    VALIDATION_EXPECTED_TOTAL,
    DynamicCheckResult,
    DynamicExecutionPlan,
    DynamicPrefixCase,
    DynamicPrefixReport,
    PairSideHarness,
    load_frozen_validation_perturbations,
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
            "data_safety_sha256": profile.data_safety_sha256,
        }
    )


def _validate_execution_profile_inventory(
    *,
    execution_mode: object,
    authorization_sha256: object,
    profiles: tuple[H6PrefixProfilePair, ...],
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
    if authorization_sha256 != _FULL_PREFIX_AUTHORIZATION_SHA256:
        raise ValueError(
            "authorized-full H6-Prefix requires the exact operation authorization"
        )
    expected = tuple(
        (key, particle_count)
        for key in ordered_semantics
        for particle_count in (128, 256, 512, 1024)
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
        exact_positions = (
            len(report.completed_by_position) == 32
            and sum(report.completed_by_position) == VALIDATION_EXPECTED_TOTAL
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


def _reject_prefix_dependencies(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} contains an invalid field name")
            folded = key.casefold()
            if (
                folded in {"h1", "h2", "h3", "h4", "h5"}
                or any(
                    fragment in folded
                    for fragment in _FORBIDDEN_PREFIX_DEPENDENCY_FIELDS
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
    config: H6PrefixResolvedConfig,
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


def _validate_resolved_runner_config(config: H6PrefixResolvedConfig) -> None:
    if type(config) is not H6PrefixResolvedConfig:
        raise ValueError("config must be an exact H6PrefixResolvedConfig")
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


def run_h6_prefix(
    *,
    config: H6PrefixResolvedConfig,
    junit_sha256: str | None,
) -> tuple[H6PrefixGateResult, Path]:
    """Run bounded typed Prefix checks and publish only their derived status."""

    _validate_resolved_runner_config(config)
    if junit_sha256 is not None:
        _require_sha256(junit_sha256, "junit_sha256")
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
            pair_side_harness=PairSideHarness(),
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
            pair_side_harness=PairSideHarness(),
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
    "H6PrefixReportBundle",
    "compose_prefix_certificate",
    "h6_prefix_artifact_payloads",
    "publish_h6_prefix_artifact",
    "run_h6_prefix",
]
