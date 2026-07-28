"""Frozen H6 current-candidate projections and verified artifact references."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Literal

from vfe4.config import validate_h6_prefix_v3_resolved_payload
from vfe4.types.h6 import (
    H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS,
    H6_A0_DIRECT_EXACT_PREFIX_WITNESS_CHECKS,
    H6_PREFIX_REQUIRED_CHECKS,
    A0DirectExactPrefixCertificateV1,
    A0DirectExactPrefixWitnessV1,
    ArmConfig,
    ArmId,
    BoundedPrefixCertificate,
    BoundedPrefixCertificateSet,
    BoundedPrefixReportBinding,
    BoundedPrefixReportReference,
    CapacityAllocation,
    EstimatorSpec,
    EvidenceStatus,
    H6PrefixWorkloadPlan,
    PrefixCaseKey,
    VocabularyIdentity,
)

from .atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOWER_HEX = frozenset("0123456789abcdef")
_MANIFEST_LINE = re.compile(
    r"(?P<sha256>[0-9a-f]{64})  (?P<path>[^\r\n]+)\n"
)
_EXPECTED_PAYLOADS = {
    "H1-Prefix-Prior": (
        "config.json",
        "schemas/generative_factor.json",
        "validation/h1_prefix_prior.json",
    ),
    "H6-Prefix": (
        "certificates/prefix_set.json",
        "config.json",
        "environment.json",
        "provenance.json",
        "validation/h6_prefix.json",
    ),
}
_EXPECTED_H6_PREFIX_V3_PAYLOADS = (
    "certificates/a0_direct_exact.json",
    "certificates/prefix_set.json",
    "config.json",
    "environment.json",
    "provenance.json",
    "validation/h6_prefix.json",
)
_ProjectedGateRunner = Callable[
    [Literal["H1-Prefix-Prior", "H6-Prefix"], object, str | None],
    tuple[object, Path],
]
_PROJECTED_CURRENT_CANDIDATE_RUNNER: _ProjectedGateRunner | None = None


def _install_projected_current_candidate_runner(
    runner: _ProjectedGateRunner,
) -> None:
    """Install one external execution callback exactly once."""

    if not callable(runner):
        raise ValueError("projected current-candidate runner must be callable")
    global _PROJECTED_CURRENT_CANDIDATE_RUNNER
    current = _PROJECTED_CURRENT_CANDIDATE_RUNNER
    if current is not None and current is not runner:
        raise RuntimeError("another projected current-candidate runner is installed")
    _PROJECTED_CURRENT_CANDIDATE_RUNNER = runner


def _require_lower_hex(value: object, length: int, location: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(
            f"{location} must be exactly {length} lowercase hexadecimal characters"
        )
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_json(item)
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _owned_config(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError("projected scientific config must be a string-keyed mapping")
    try:
        copied = json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (ArtifactPublicationError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"projected scientific config is not canonical JSON: {exc}") from exc
    frozen = _freeze_json(copied)
    if not isinstance(frozen, Mapping):
        raise ValueError("projected scientific config must remain a mapping")
    return frozen


def _resolve_projected_config(
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
    raw_config: Mapping[str, object],
) -> object:
    import vfe4.config as config_module

    thawed = _thaw_json(raw_config)
    if not isinstance(thawed, Mapping):
        raise ValueError("projected scientific config is not a mapping")
    if operation == "H1-Prefix-Prior":
        if thawed.get("schema_version") == "h1-prefix-prior-config-v2":
            return config_module.resolve_h1_prefix_prior_v2_config(
                thawed,
                repo_root=_REPO_ROOT,
            )
        return config_module.resolve_h1_prefix_prior_config(
            thawed,
            repo_root=_REPO_ROOT,
        )
    return config_module.resolve_h6_prefix_config(
        thawed,
        repo_root=_REPO_ROOT,
    )


@dataclass(frozen=True)
class ProjectedCurrentCandidateConfig:
    """Owned, validated projection for one independent current-candidate gate."""

    operation: Literal["H1-Prefix-Prior", "H6-Prefix"]
    raw_config: Mapping[str, object]
    canonical_sha256: str

    def __post_init__(self) -> None:
        if self.operation not in ("H1-Prefix-Prior", "H6-Prefix"):
            raise ValueError("operation must be H1-Prefix-Prior or H6-Prefix")
        owned = _owned_config(self.raw_config)
        if owned.get("operation") != self.operation:
            raise ValueError("projected raw config operation differs from its record")
        expected_sha256 = _require_lower_hex(
            self.canonical_sha256,
            64,
            "canonical_sha256",
        )
        resolved = _resolve_projected_config(self.operation, owned)
        if (
            getattr(resolved, "operation", None) != self.operation
            or getattr(resolved, "config_sha256", None) != expected_sha256
        ):
            raise ValueError(
                "projected raw config does not reproduce its operation and canonical SHA-256"
            )
        object.__setattr__(self, "raw_config", owned)


def _canonical_payload_name(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("payload hash keys must be canonical POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.suffix != ".json"
    ):
        raise ValueError("payload hash keys must name canonical relative JSON files")
    return value


@dataclass(frozen=True)
class CandidateArtifactReference:
    """Hash-complete reference reconstructed from a published run directory."""

    artifact_path: Path
    git_head: str
    dirty_digest: str
    manifest_sha256: str
    payload_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_path, Path):
            raise ValueError("artifact_path must be a Path")
        git_head = _require_lower_hex(self.git_head, 40, "git_head")
        dirty_digest = _require_lower_hex(
            self.dirty_digest,
            64,
            "dirty_digest",
        )
        manifest_sha256 = _require_lower_hex(
            self.manifest_sha256,
            64,
            "manifest_sha256",
        )
        if (
            not isinstance(self.payload_hashes, Mapping)
            or not self.payload_hashes
            or any(type(key) is not str for key in self.payload_hashes)
        ):
            raise ValueError("payload_hashes must be a nonempty string-keyed mapping")
        payload_hashes: dict[str, str] = {}
        aliases: set[str] = set()
        for name, digest in sorted(self.payload_hashes.items()):
            canonical_name = _canonical_payload_name(name)
            alias = canonical_name.casefold()
            if alias in aliases:
                raise ValueError("payload_hashes contains a portable path collision")
            aliases.add(alias)
            payload_hashes[canonical_name] = _require_lower_hex(
                digest,
                64,
                f"payload_hashes[{canonical_name!r}]",
            )
        object.__setattr__(
            self,
            "artifact_path",
            self.artifact_path.resolve(strict=False),
        )
        object.__setattr__(self, "git_head", git_head)
        object.__setattr__(self, "dirty_digest", dirty_digest)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(
            self,
            "payload_hashes",
            MappingProxyType(payload_hashes),
        )


def _extract_operation_config(
    raw_config: Mapping[str, object],
    *,
    operation_key: Literal[
        "h1_prefix_prior",
        "h1_prefix_prior_v2",
        "h6_prefix",
    ],
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
) -> Mapping[str, object]:
    if (
        not isinstance(raw_config, Mapping)
        or any(type(key) is not str for key in raw_config)
    ):
        raise ValueError("raw_config must be a string-keyed mapping")
    direct = "operation" in raw_config
    nested = "operations" in raw_config
    if direct and nested:
        raise ValueError(
            "raw_config cannot be both a direct operation and an operations root"
        )
    if direct:
        selected: object = raw_config
    elif nested:
        operations = raw_config["operations"]
        if (
            not isinstance(operations, Mapping)
            or operation_key not in operations
            or any(type(key) is not str for key in operations)
        ):
            raise ValueError(
                f"raw_config operations must contain {operation_key!r}"
            )
        entry = operations[operation_key]
        if not isinstance(entry, Mapping) or "config" not in entry:
            raise ValueError(
                f"raw_config operations[{operation_key!r}] must contain config"
            )
        selected = entry["config"]
    else:
        raise ValueError(
            "raw_config must be a direct operation mapping or an operations root"
        )
    if not isinstance(selected, Mapping) or selected.get("operation") != operation:
        raise ValueError(f"selected config must have operation {operation!r}")
    return selected


def _project(
    raw_config: Mapping[str, object],
    *,
    operation_key: Literal[
        "h1_prefix_prior",
        "h1_prefix_prior_v2",
        "h6_prefix",
    ],
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
    schema_versions: tuple[
        Literal[
            "h1-prefix-prior-config-v1",
            "h1-prefix-prior-config-v2",
            "h6-prefix-config-v1",
            "h6-prefix-config-v2",
            "h6-prefix-config-v3",
        ],
        ...,
    ],
) -> ProjectedCurrentCandidateConfig:
    selected = _extract_operation_config(
        raw_config,
        operation_key=operation_key,
        operation=operation,
    )
    owned = _owned_config(selected)
    if owned.get("schema_version") not in schema_versions:
        raise ValueError(
            f"{operation_key} requires schema_version in {schema_versions!r}"
        )
    resolved = _resolve_projected_config(operation, owned)
    if (
        getattr(resolved, "operation", None) != operation
        or getattr(resolved, "schema_version", None) not in schema_versions
    ):
        raise ValueError(
            "resolved projection returned another operation or schema version"
        )
    canonical_sha256 = _require_lower_hex(
        getattr(resolved, "config_sha256", None),
        64,
        "resolved config_sha256",
    )
    return ProjectedCurrentCandidateConfig(
        operation,
        owned,
        canonical_sha256,
    )


def project_h1_prefix_prior_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig:
    """Purely select and validate the H1 prefix-prior scientific config."""

    return _project(
        raw_config,
        operation_key="h1_prefix_prior",
        operation="H1-Prefix-Prior",
        schema_versions=("h1-prefix-prior-config-v1",),
    )


def project_h1_prefix_prior_v2_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig:
    """Purely select and validate the parent-specific scorer-v2 config."""

    return _project(
        raw_config,
        operation_key="h1_prefix_prior_v2",
        operation="H1-Prefix-Prior",
        schema_versions=("h1-prefix-prior-config-v2",),
    )


def project_h6_prefix_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig:
    """Purely select and validate the predecessor-free H6 Prefix config."""

    return _project(
        raw_config,
        operation_key="h6_prefix",
        operation="H6-Prefix",
        schema_versions=(
            "h6-prefix-config-v1",
            "h6-prefix-config-v2",
            "h6-prefix-config-v3",
        ),
    )


def _validated_junit_sha256(value: object) -> str | None:
    if value is None:
        return None
    try:
        return _require_lower_hex(value, 64, "JUnit SHA-256")
    except ValueError as exc:
        raise ValueError("JUnit SHA-256 must be None or exact lowercase SHA-256") from exc


def _validate_predecessor_mapping(
    predecessor_refs: Mapping[str, CandidateArtifactReference],
    operation: str,
) -> None:
    if (
        not isinstance(predecessor_refs, Mapping)
        or any(
            type(key) is not str or type(value) is not CandidateArtifactReference
            for key, value in predecessor_refs.items()
        )
    ):
        raise ValueError(
            "predecessor_refs must map strings to CandidateArtifactReference records"
        )
    if predecessor_refs:
        raise ValueError(f"{operation} does not consume predecessor references")


def _manifest_entries(manifest_bytes: bytes) -> tuple[tuple[str, str], ...]:
    if not manifest_bytes or len(manifest_bytes) > 65_536:
        raise ArtifactPublicationError("artifact manifest is empty or exceeds its bound")
    try:
        text = manifest_bytes.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ArtifactPublicationError("artifact manifest is not strict ASCII") from exc
    if "\r" in text or not text.endswith("\n"):
        raise ArtifactPublicationError("artifact manifest must use final LF-only records")
    entries: list[tuple[str, str]] = []
    offset = 0
    aliases: set[str] = set()
    while offset < len(text):
        match = _MANIFEST_LINE.match(text, offset)
        if match is None:
            raise ArtifactPublicationError("artifact manifest has a malformed record")
        digest = match.group("sha256")
        try:
            name = _canonical_payload_name(match.group("path"))
        except ValueError as exc:
            raise ArtifactPublicationError(str(exc)) from exc
        alias = name.casefold()
        if alias in aliases:
            raise ArtifactPublicationError(
                "artifact manifest contains duplicate or colliding paths"
            )
        aliases.add(alias)
        entries.append((name, digest))
        offset = match.end()
    if tuple(name for name, _ in entries) != tuple(
        sorted(name for name, _ in entries)
    ):
        raise ArtifactPublicationError("artifact manifest paths are not sorted")
    return tuple(entries)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactPublicationError(f"artifact payload is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_json_payload(path: Path, *, maximum_bytes: int = 16_777_216) -> object:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactPublicationError(f"artifact JSON is unreadable: {path}") from exc
    if not payload or len(payload) > maximum_bytes:
        raise ArtifactPublicationError("artifact JSON is empty or exceeds its bound")
    try:
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactPublicationError("artifact JSON is not canonical UTF-8 JSON") from exc
    if canonical_json_bytes(parsed) != payload:
        raise ArtifactPublicationError("artifact JSON bytes are not canonical")
    return parsed


def _contains_bounded_external_reference(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key
            in {
                "validation_fixture_reference",
                "validation_perturbation_reference",
            }
            or _contains_bounded_external_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(
            _contains_bounded_external_reference(item) for item in value
        )
    return False


_BOUNDED_EXECUTION_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "case_family",
        "particle_count",
        "expected_by_position",
        "full_expected_count",
        "selected_global_indices",
        "workload_plan_sha256",
        "authorization_sha256",
        "selection_rows",
        "selection_manifest_sha256",
        "plan_sha256",
    }
)
_BOUNDED_DYNAMIC_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "key",
        "execution_plan_sha256",
        "model_state_sha256",
        "proposal_identity_sha256",
        "estimator_semantic_sha256",
        "estimator_artifact_bytes_sha256",
        "stream_seed",
        "completed_by_position",
        "checks",
        "status",
        "obligations",
        "unresolved_diagnostics",
        "first_counterexample",
        "case_result_manifest_sha256",
        "cache_manifest_sha256",
        "pair_harness_manifest_sha256",
        "mask_manifest_sha256",
        "complete_case_manifest_sha256",
        "scope",
        "case_family",
        "particle_count",
        "workload_plan_sha256",
        "selected_global_indices",
        "selection_manifest_sha256",
        "applicable_check_names",
        "report_sha256",
    }
)
_BOUNDED_DYNAMIC_CHECK_FIELDS = frozenset(
    {
        "name",
        "status",
        "expected_count",
        "completed_count",
        "violation_count",
        "first_counterexample",
        "obligations",
    }
)
_BOUNDED_STATIC_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "source_manifest_sha256",
        "rules_sha256",
        "case_key_manifest_sha256",
        "checks",
        "findings",
        "status",
        "obligations",
        "report_sha256",
    }
)
_BOUNDED_STATIC_CHECK_FIELDS = frozenset(
    {
        "name",
        "status",
        "finding_sha256s",
        "obligations",
        "check_sha256",
    }
)
_BOUNDED_STATIC_FINDING_FIELDS = frozenset(
    {
        "rule_id",
        "status",
        "path",
        "line",
        "message",
        "witness_sha256",
        "finding_sha256",
    }
)
_BOUNDED_FAMILY_FIELDS = frozenset(
    {
        "semantic_family_index",
        "semantic_family_sha256",
        "jobs",
        "validation_payload_sha256",
        "certificate_sha256",
    }
)
_BOUNDED_JOB_FIELDS = frozenset(
    {
        "job_index",
        "particle_count",
        "case_family",
        "scope",
        "profile_pair_sha256",
        "execution_plan",
        "dynamic_report",
        "observed_predictor_call_count",
    }
)
_BOUNDED_RUNNER_TOTAL_FIELDS = frozenset(
    {
        "semantic_family_count",
        "planned_fixture_load_count",
        "planned_static_audit_count",
        "planned_arm_build_count",
        "planned_predictor_boundary_count",
        "planned_dynamic_report_count",
        "observed_dynamic_report_count",
        "planned_case_count",
        "planned_predictor_call_count",
        "observed_predictor_call_count",
        "planned_particle_call_units",
    }
)
_BOUNDED_CASE_KEY_FIELDS = frozenset(
    {
        "arm",
        "predictor_config_sha256",
        "estimator_sha256",
        "model_family_sha256",
        "vocabulary_sha256",
        "data_safety_sha256",
        "git_head",
        "dirty_digest",
    }
)
_BOUNDED_REPORT_REFERENCE_FIELDS = frozenset(
    {
        "schema_version",
        "profile_pair_sha256",
        "particle_count",
        "case_family",
        "scope",
        "report_key",
        "report_key_sha256",
        "report_sha256",
        "execution_plan_sha256",
        "workload_plan_sha256",
        "selected_global_indices",
        "selection_manifest_sha256",
        "completed_by_position",
        "complete_case_manifest_sha256",
        "model_state_sha256",
        "proposal_identity_sha256",
        "estimator_semantic_sha256",
        "estimator_artifact_bytes_sha256",
        "reference_sha256",
    }
)
_BOUNDED_REPORT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "workload_plan_sha256",
        "semantic_family_sha256",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "global_case_key_order_sha256",
        "profile_pair_sha256s",
        "report_references",
        "higher_n_small_selection_manifest_sha256",
        "higher_n_validation_selection_manifest_sha256",
        "static_report_sha256",
        "static_source_manifest_sha256",
        "static_rules_sha256",
        "static_case_key_manifest_sha256",
        "binding_sha256",
    }
)
_BOUNDED_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "semantic_family_sha256",
        "report_binding",
        "validation_payload",
        "validation_payload_sha256",
        "status",
        "obligations",
        "checks",
        "certificate_sha256",
    }
)
_BOUNDED_CERTIFICATE_SET_FIELDS = frozenset(
    {
        "schema_version",
        "config_sha256",
        "workload_plan_sha256",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "semantic_family_sha256s",
        "validation_payload_sha256",
        "prefix_certificate_set_sha256",
        "certificates",
    }
)
_A0_DIRECT_ARM_CONFIG_FIELDS = frozenset(
    {
        "arm",
        "config_id",
        "vocabulary",
        "horizon",
        "latent_enabled",
        "state_channel_enabled",
        "model_channel_enabled",
        "source_mode",
        "map_mode",
        "recognition_family",
        "recognition_conditioning",
        "prior_variant",
        "mixture_mode",
        "objective_kind",
        "capacity_allocation",
        "capacity_allocation_sha256",
        "config_sha256",
    }
)
_A0_DIRECT_WITNESS_FIELDS = frozenset(
    {
        "schema_version",
        "small_case_count",
        "validation_case_count",
        "small_complete_case_manifest_sha256",
        "validation_complete_case_manifest_sha256",
        "direct_case_manifest_sha256",
        "small_model_state_sha256",
        "production_model_state_sha256",
        "small_proposal_identity_sha256",
        "production_proposal_identity_sha256",
        "direct_predictor_path_sha256",
        "case_witness_manifest_sha256",
        "checks",
        "first_counterexample_sha256",
        "witness_sha256",
    }
)
_A0_DIRECT_VALIDATION_FIELDS = frozenset(
    {
        "schema_version",
        "arm",
        "endpoint_config",
        "estimator",
        "predictor_config_sha256",
        "model_family_sha256",
        "vocabulary_sha256",
        "data_safety_sha256",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "direct_predictor_path_sha256",
        "heldout_scorer_path_sha256",
        "bounded_a0_certificate_sha256",
        "bounded_a0_report_binding_sha256",
        "direct_case_manifest_sha256",
        "static_report_sha256",
        "static_report_status",
        "direct_path_witness_sha256",
        "checks",
        "status",
        "obligations",
    }
)
_A0_DIRECT_CERTIFICATE_FIELDS = frozenset(
    {
        *_A0_DIRECT_VALIDATION_FIELDS,
        "direct_witness",
        "validation_payload",
        "validation_payload_sha256",
        "certificate_sha256",
    }
)
_BOUNDED_V3_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "source",
        "execution_mode",
        "profiles",
        "authorization_sha256",
        "artifact_root",
        "workload_plan",
        "workload_plan_sha256",
        "workload_authorization_sha256",
        "validation_fixture_reference",
        "validation_perturbation_reference",
    }
)
_BOUNDED_SOURCE_FIELDS = frozenset(
    {"git_head", "dirty_digest", "source_sha256"}
)
_BOUNDED_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "small_arm_config",
        "production_arm_config",
        "estimator",
        "small_structure",
        "production_structure",
        "data_safety_sha256",
        "small_model_family_sha256",
        "production_model_family_sha256",
        "profile_pair_sha256",
    }
)
_BOUNDED_ESTIMATOR_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "particle_count",
        "resampling",
        "dtype",
        "device",
        "estimator_sha256",
    }
)
_BOUNDED_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "device",
        "dtype",
        "python_implementation",
        "python_version",
    }
)
_BOUNDED_DYNAMIC_CHECK_NAMES = (
    "signature_and_identity",
    "dynamic_target_suffix_leakage",
    "cache_identity",
    "source_mask",
    "case_inventory",
    "validation_data_safety",
)
_BOUNDED_STATIC_CHECK_NAMES = (
    "import_signature_access",
    "taint_cache_capability",
    "mask_normalization_support",
    "inventory_identity",
)


def _bounded_owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _bounded_exact_object(
    value: object,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != expected_fields:
        raise ArtifactPublicationError(f"{label} fields are not exact")
    return value


def _bounded_config_family_bindings(
    config_payload: Mapping[str, object],
    *,
    workload: H6PrefixWorkloadPlan,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    config = _bounded_exact_object(
        config_payload,
        _BOUNDED_V3_CONFIG_FIELDS,
        "bounded H6 Prefix config",
    )
    profiles = config.get("profiles")
    if type(profiles) is not list or not profiles:
        raise ArtifactPublicationError(
            "bounded H6 Prefix config profiles are not a nonempty list"
        )
    particle_counts = workload.production_particle_counts
    if len(profiles) % len(particle_counts) != 0:
        raise ArtifactPublicationError(
            "bounded H6 Prefix config profiles do not form complete families"
        )
    families: list[tuple[str, tuple[str, ...]]] = []
    for offset in range(0, len(profiles), len(particle_counts)):
        family_profiles = profiles[offset : offset + len(particle_counts)]
        semantic_payload: dict[str, object] | None = None
        profile_pair_sha256s: list[str] = []
        for position, raw_profile in enumerate(family_profiles):
            profile = _bounded_exact_object(
                raw_profile,
                _BOUNDED_PROFILE_FIELDS,
                "bounded H6 Prefix resolved profile",
            )
            estimator = _bounded_exact_object(
                profile.get("estimator"),
                _BOUNDED_ESTIMATOR_FIELDS,
                "bounded H6 Prefix resolved estimator",
            )
            if estimator.get("particle_count") != particle_counts[position]:
                raise ArtifactPublicationError(
                    "bounded H6 Prefix config particle ladder changed"
                )
            profile_pair_sha256 = _require_lower_hex(
                profile.get("profile_pair_sha256"),
                64,
                "bounded H6 Prefix profile-pair SHA-256",
            )
            profile_pair_sha256s.append(profile_pair_sha256)
            current_semantic = dict(profile)
            current_semantic.pop("profile_id")
            current_semantic.pop("profile_pair_sha256")
            semantic_estimator = dict(estimator)
            semantic_estimator.pop("particle_count")
            semantic_estimator.pop("estimator_sha256")
            current_semantic["estimator"] = semantic_estimator
            if semantic_payload is None:
                semantic_payload = current_semantic
            elif canonical_json_bytes(current_semantic) != canonical_json_bytes(
                semantic_payload
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix family changes beyond particle count"
                )
        assert semantic_payload is not None
        families.append(
            (
                _bounded_owned_hash(
                    "vfe4.h6.bounded-prefix-semantic-family.v2",
                    semantic_payload,
                ),
                tuple(profile_pair_sha256s),
            )
        )
    return tuple(families)


def _bounded_require_canonical_match(
    observed: object,
    expected: object,
    label: str,
) -> None:
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ArtifactPublicationError(
            f"bounded H6 Prefix {label} differs from its recomputed identity"
        )


def _bounded_prefix_case_key_from_payload(value: object) -> PrefixCaseKey:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_CASE_KEY_FIELDS,
        "bounded H6 Prefix case key",
    )
    try:
        key = PrefixCaseKey(
            arm=ArmId(payload["arm"]),
            predictor_config_sha256=payload["predictor_config_sha256"],
            estimator_sha256=payload["estimator_sha256"],
            model_family_sha256=payload["model_family_sha256"],
            vocabulary_sha256=payload["vocabulary_sha256"],
            data_safety_sha256=payload["data_safety_sha256"],
            git_head=payload["git_head"],
            dirty_digest=payload["dirty_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix case key is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        key.canonical_payload(),
        "case key",
    )
    return key


def _bounded_report_reference_from_payload(
    value: object,
) -> BoundedPrefixReportReference:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_REPORT_REFERENCE_FIELDS,
        "bounded H6 Prefix report reference",
    )
    selected_global_indices = payload["selected_global_indices"]
    completed_by_position = payload["completed_by_position"]
    if (
        type(selected_global_indices) is not list
        or type(completed_by_position) is not list
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix report-reference sequences are not exact"
        )
    report_key = _bounded_prefix_case_key_from_payload(payload["report_key"])
    try:
        reference = BoundedPrefixReportReference.create(
            profile_pair_sha256=payload["profile_pair_sha256"],
            particle_count=payload["particle_count"],
            case_family=payload["case_family"],
            scope=payload["scope"],
            report_key=report_key,
            report_sha256=payload["report_sha256"],
            execution_plan_sha256=payload["execution_plan_sha256"],
            workload_plan_sha256=payload["workload_plan_sha256"],
            selected_global_indices=tuple(selected_global_indices),
            selection_manifest_sha256=payload[
                "selection_manifest_sha256"
            ],
            completed_by_position=tuple(completed_by_position),
            complete_case_manifest_sha256=payload[
                "complete_case_manifest_sha256"
            ],
            model_state_sha256=payload["model_state_sha256"],
            proposal_identity_sha256=payload[
                "proposal_identity_sha256"
            ],
            estimator_semantic_sha256=payload[
                "estimator_semantic_sha256"
            ],
            estimator_artifact_bytes_sha256=payload[
                "estimator_artifact_bytes_sha256"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix report reference is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        reference.canonical_payload(),
        "report reference",
    )
    return reference


def _bounded_report_binding_from_payload(
    value: object,
) -> BoundedPrefixReportBinding:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_REPORT_BINDING_FIELDS,
        "bounded H6 Prefix report binding",
    )
    profile_pair_sha256s = payload["profile_pair_sha256s"]
    report_reference_payloads = payload["report_references"]
    if (
        type(profile_pair_sha256s) is not list
        or type(report_reference_payloads) is not list
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix report-binding sequences are not exact"
        )
    report_references = tuple(
        _bounded_report_reference_from_payload(reference)
        for reference in report_reference_payloads
    )
    try:
        binding = BoundedPrefixReportBinding.create(
            workload_plan_sha256=payload["workload_plan_sha256"],
            semantic_family_sha256=payload["semantic_family_sha256"],
            git_head=payload["git_head"],
            dirty_digest=payload["dirty_digest"],
            source_sha256=payload["source_sha256"],
            global_case_key_order_sha256=payload[
                "global_case_key_order_sha256"
            ],
            profile_pair_sha256s=tuple(profile_pair_sha256s),
            report_references=report_references,
            higher_n_small_selection_manifest_sha256=payload[
                "higher_n_small_selection_manifest_sha256"
            ],
            higher_n_validation_selection_manifest_sha256=payload[
                "higher_n_validation_selection_manifest_sha256"
            ],
            static_report_sha256=payload["static_report_sha256"],
            static_source_manifest_sha256=payload[
                "static_source_manifest_sha256"
            ],
            static_rules_sha256=payload["static_rules_sha256"],
            static_case_key_manifest_sha256=payload[
                "static_case_key_manifest_sha256"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix report binding is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        binding.canonical_payload(),
        "report binding",
    )
    return binding


def _bounded_certificate_payload(
    certificate: BoundedPrefixCertificate,
) -> dict[str, object]:
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


def _bounded_certificate_from_payload(
    value: object,
) -> BoundedPrefixCertificate:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_CERTIFICATE_FIELDS,
        "bounded H6 Prefix certificate",
    )
    obligations = payload["obligations"]
    checks = payload["checks"]
    if (
        type(obligations) is not list
        or type(checks) is not dict
        or frozenset(checks) != frozenset(H6_PREFIX_REQUIRED_CHECKS)
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate checks or obligations are not exact"
        )
    report_binding = _bounded_report_binding_from_payload(
        payload["report_binding"]
    )
    ordered_checks = {
        name: checks[name] for name in H6_PREFIX_REQUIRED_CHECKS
    }
    try:
        certificate = BoundedPrefixCertificate.create(
            semantic_family_sha256=payload["semantic_family_sha256"],
            report_binding=report_binding,
            status=EvidenceStatus(payload["status"]),
            obligations=tuple(obligations),
            checks=ordered_checks,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        _bounded_certificate_payload(certificate),
        "certificate",
    )
    return certificate


def _bounded_certificate_set_payload(
    certificate_set: BoundedPrefixCertificateSet,
) -> dict[str, object]:
    return {
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


def _bounded_certificate_set_from_payload(
    value: object,
) -> BoundedPrefixCertificateSet:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_CERTIFICATE_SET_FIELDS,
        "bounded H6 Prefix certificate set",
    )
    semantic_family_sha256s = payload["semantic_family_sha256s"]
    certificate_payloads = payload["certificates"]
    if (
        type(semantic_family_sha256s) is not list
        or type(certificate_payloads) is not list
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate-set sequences are not exact"
        )
    certificates = tuple(
        _bounded_certificate_from_payload(certificate)
        for certificate in certificate_payloads
    )
    try:
        certificate_set = BoundedPrefixCertificateSet.create(
            config_sha256=payload["config_sha256"],
            semantic_family_sha256s=tuple(semantic_family_sha256s),
            certificates=certificates,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate set is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        _bounded_certificate_set_payload(certificate_set),
        "certificate set",
    )
    return certificate_set


def _a0_direct_arm_config_from_payload(value: object) -> ArmConfig:
    payload = _bounded_exact_object(
        value,
        _A0_DIRECT_ARM_CONFIG_FIELDS,
        "direct-A0 endpoint config",
    )
    vocabulary_payload = _bounded_exact_object(
        payload["vocabulary"],
        frozenset(
            {"vocabulary_id", "size", "tokenizer_spec_sha256"}
        ),
        "direct-A0 endpoint vocabulary",
    )
    allocation_payload = _bounded_exact_object(
        payload["capacity_allocation"],
        frozenset(
            {
                "emission_width",
                "latent_width",
                "recognition_width",
                "prior_context_width",
            }
        ),
        "direct-A0 endpoint allocation",
    )
    try:
        vocabulary = VocabularyIdentity(
            vocabulary_id=vocabulary_payload["vocabulary_id"],
            size=vocabulary_payload["size"],
            tokenizer_spec_sha256=vocabulary_payload[
                "tokenizer_spec_sha256"
            ],
        )
        allocation = CapacityAllocation.create(
            emission_width=allocation_payload["emission_width"],
            latent_width=allocation_payload["latent_width"],
            recognition_width=allocation_payload["recognition_width"],
            prior_context_width=allocation_payload[
                "prior_context_width"
            ],
        )
        config = ArmConfig.create(
            arm=ArmId(payload["arm"]),
            config_id=payload["config_id"],
            vocabulary=vocabulary,
            horizon=payload["horizon"],
            latent_enabled=payload["latent_enabled"],
            state_channel_enabled=payload["state_channel_enabled"],
            model_channel_enabled=payload["model_channel_enabled"],
            source_mode=payload["source_mode"],
            map_mode=payload["map_mode"],
            recognition_family=payload["recognition_family"],
            recognition_conditioning=payload[
                "recognition_conditioning"
            ],
            prior_variant=payload["prior_variant"],
            mixture_mode=payload["mixture_mode"],
            objective_kind=payload["objective_kind"],
            capacity_allocation=allocation,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "direct-A0 endpoint config is invalid"
        ) from exc
    expected = {
        **config.canonical_payload(),
        "config_sha256": config.config_sha256,
    }
    _bounded_require_canonical_match(
        payload,
        expected,
        "direct-A0 endpoint config",
    )
    return config


def _a0_direct_estimator_from_payload(value: object) -> EstimatorSpec:
    payload = _bounded_exact_object(
        value,
        _BOUNDED_ESTIMATOR_FIELDS,
        "direct-A0 estimator",
    )
    try:
        estimator = EstimatorSpec.create(
            kind=payload["kind"],
            particle_count=payload["particle_count"],
            resampling=payload["resampling"],
            dtype=payload["dtype"],
            device=payload["device"],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "direct-A0 estimator is invalid"
        ) from exc
    expected = {
        "schema_version": estimator.schema_version,
        "kind": estimator.kind,
        "particle_count": estimator.particle_count,
        "resampling": estimator.resampling,
        "dtype": estimator.dtype,
        "device": estimator.device,
        "estimator_sha256": estimator.estimator_sha256,
    }
    _bounded_require_canonical_match(
        payload,
        expected,
        "direct-A0 estimator",
    )
    return estimator


def _a0_direct_witness_from_payload(
    value: object,
) -> A0DirectExactPrefixWitnessV1:
    payload = _bounded_exact_object(
        value,
        _A0_DIRECT_WITNESS_FIELDS,
        "direct-A0 path witness",
    )
    checks = payload["checks"]
    if (
        type(checks) is not dict
        or frozenset(checks)
        != frozenset(H6_A0_DIRECT_EXACT_PREFIX_WITNESS_CHECKS)
    ):
        raise ArtifactPublicationError(
            "direct-A0 witness checks are incomplete"
        )
    try:
        witness = A0DirectExactPrefixWitnessV1.create(
            small_complete_case_manifest_sha256=payload[
                "small_complete_case_manifest_sha256"
            ],
            validation_complete_case_manifest_sha256=payload[
                "validation_complete_case_manifest_sha256"
            ],
            small_model_state_sha256=payload[
                "small_model_state_sha256"
            ],
            production_model_state_sha256=payload[
                "production_model_state_sha256"
            ],
            small_proposal_identity_sha256=payload[
                "small_proposal_identity_sha256"
            ],
            production_proposal_identity_sha256=payload[
                "production_proposal_identity_sha256"
            ],
            direct_predictor_path_sha256=payload[
                "direct_predictor_path_sha256"
            ],
            case_witness_manifest_sha256=payload[
                "case_witness_manifest_sha256"
            ],
            checks={
                name: checks[name]
                for name in H6_A0_DIRECT_EXACT_PREFIX_WITNESS_CHECKS
            },
            first_counterexample_sha256=payload[
                "first_counterexample_sha256"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "direct-A0 path witness is invalid"
        ) from exc
    _bounded_require_canonical_match(
        payload,
        witness.canonical_payload(),
        "direct-A0 path witness",
    )
    return witness


def _a0_direct_certificate_from_payload(
    value: object,
    *,
    certificate_set: BoundedPrefixCertificateSet,
) -> A0DirectExactPrefixCertificateV1:
    payload = _bounded_exact_object(
        value,
        _A0_DIRECT_CERTIFICATE_FIELDS,
        "direct-A0 Prefix certificate",
    )
    validation_payload = _bounded_exact_object(
        payload["validation_payload"],
        _A0_DIRECT_VALIDATION_FIELDS,
        "direct-A0 validation payload",
    )
    checks = payload["checks"]
    obligations = payload["obligations"]
    if (
        type(checks) is not dict
        or frozenset(checks)
        != frozenset(H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS)
        or type(obligations) is not list
    ):
        raise ArtifactPublicationError(
            "direct-A0 certificate checks or obligations are incomplete"
        )
    bounded_matches = tuple(
        certificate
        for certificate in certificate_set.certificates
        if certificate.certificate_sha256
        == payload["bounded_a0_certificate_sha256"]
    )
    if len(bounded_matches) != 1:
        raise ArtifactPublicationError(
            "direct-A0 certificate lacks one typed bounded premise"
        )
    endpoint_config = _a0_direct_arm_config_from_payload(
        payload["endpoint_config"]
    )
    estimator = _a0_direct_estimator_from_payload(payload["estimator"])
    witness = _a0_direct_witness_from_payload(payload["direct_witness"])
    try:
        certificate = A0DirectExactPrefixCertificateV1.create(
            endpoint_config=endpoint_config,
            estimator=estimator,
            model_family_sha256=payload["model_family_sha256"],
            data_safety_sha256=payload["data_safety_sha256"],
            git_head=payload["git_head"],
            dirty_digest=payload["dirty_digest"],
            source_sha256=payload["source_sha256"],
            direct_predictor_path_sha256=payload[
                "direct_predictor_path_sha256"
            ],
            heldout_scorer_path_sha256=payload[
                "heldout_scorer_path_sha256"
            ],
            bounded_a0_certificate=bounded_matches[0],
            direct_witness=witness,
            static_report_sha256=payload["static_report_sha256"],
            static_report_status=EvidenceStatus(
                payload["static_report_status"]
            ),
            checks={
                name: checks[name]
                for name in H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
            },
            status=EvidenceStatus(payload["status"]),
            obligations=tuple(obligations),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "direct-A0 Prefix certificate is invalid"
        ) from exc
    _bounded_require_canonical_match(
        validation_payload,
        json.loads(certificate.validation_payload_canonical_json),
        "direct-A0 validation payload",
    )
    _bounded_require_canonical_match(
        payload,
        certificate.artifact_payload(),
        "direct-A0 Prefix certificate",
    )
    return certificate


def _a0_direct_source_path_identity(
    *,
    domain: str,
    relative_paths: tuple[str, ...],
) -> str:
    rows = []
    for relative in relative_paths:
        raw = _REPO_ROOT.joinpath(*relative.split("/")).read_bytes()
        rows.append(
            {
                "path": relative,
                "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return _bounded_owned_hash(domain, tuple(rows))


def _current_a0_direct_path_identities() -> tuple[str, str]:
    return (
        _a0_direct_source_path_identity(
            domain="vfe4.h6.a0-direct-prefix-predictor-path.v1",
            relative_paths=(
                "vfe4/training/h6_transformer.py",
                "vfe4/training/arms.py",
            ),
        ),
        _a0_direct_source_path_identity(
            domain="vfe4.h6.a0-heldout-scorer-path.v1",
            relative_paths=(
                "vfe4/training/h6_heldout_scoring_v3.py",
            ),
        ),
    )


def _current_a0_target_read_follows_prediction() -> bool:
    path = _REPO_ROOT / "vfe4" / "training" / "h6_heldout_scoring_v3.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return False
    functions = tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "score_h6_exact_a0_total_v3"
    )
    if len(functions) != 1:
        return False
    prediction_lines = tuple(
        node.lineno
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prefix_log_probs"
    )
    target_lines = tuple(
        node.lineno
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Attribute)
        and node.attr == "targets"
        and isinstance(node.value, ast.Name)
        and node.value.id == "windows"
    )
    return (
        len(prediction_lines) == 1
        and len(target_lines) == 1
        and prediction_lines[0] < target_lines[0]
    )


def _bounded_expected_job_contract(
    *,
    scope: object,
    case_family: object,
    particle_count: object,
    workload: object,
) -> tuple[list[int], int, list[int], list[str]]:
    representative = scope == "representative_exhaustive"
    if case_family == "small":
        expected_by_position = (
            [6561, 2187, 729, 243] if representative else [4, 4, 4, 4]
        )
        full_expected_count = 9720
        selected = (
            list(range(full_expected_count))
            if representative
            else list(getattr(workload, "small_global_case_indices"))
        )
    elif case_family == "validation":
        expected_by_position = [4096] if representative else [16]
        full_expected_count = 4096
        selected = (
            list(range(full_expected_count))
            if representative
            else list(getattr(workload, "validation_global_case_indices"))
        )
    else:
        raise ArtifactPublicationError(
            "bounded H6 Prefix job case family is unsupported"
        )
    if (
        (representative and particle_count != 128)
        or (
            not representative
            and (
                scope != "estimator_stratified"
                or particle_count not in (256, 512, 1024)
            )
        )
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix job scope/particle count is not frozen"
        )
    applicable = (
        list(_BOUNDED_DYNAMIC_CHECK_NAMES)
        if representative
        else list(_BOUNDED_DYNAMIC_CHECK_NAMES[:3])
    )
    return expected_by_position, full_expected_count, selected, applicable


def _validate_bounded_static_report_payload(
    value: object,
) -> dict[str, object]:
    report = _bounded_exact_object(
        value,
        _BOUNDED_STATIC_REPORT_FIELDS,
        "bounded H6 Prefix static report",
    )
    checks = report["checks"]
    findings = report["findings"]
    if (
        type(checks) is not list
        or len(checks) != len(_BOUNDED_STATIC_CHECK_NAMES)
        or type(findings) is not list
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix static report inventory is incomplete"
        )
    check_hashes: list[str] = []
    for index, value_check in enumerate(checks):
        check = _bounded_exact_object(
            value_check,
            _BOUNDED_STATIC_CHECK_FIELDS,
            "bounded H6 Prefix static check",
        )
        expected_check = {
            "name": check["name"],
            "status": check["status"],
            "finding_sha256s": check["finding_sha256s"],
            "obligations": check["obligations"],
        }
        expected_check_sha256 = _bounded_owned_hash(
            "vfe4.h6.static-audit-check.v1",
            expected_check,
        )
        if (
            check["name"] != _BOUNDED_STATIC_CHECK_NAMES[index]
            or check["check_sha256"] != expected_check_sha256
        ):
            raise ArtifactPublicationError(
                "bounded H6 Prefix static check identity is stale"
            )
        check_hashes.append(expected_check_sha256)
    finding_hashes: list[str] = []
    for value_finding in findings:
        finding = _bounded_exact_object(
            value_finding,
            _BOUNDED_STATIC_FINDING_FIELDS,
            "bounded H6 Prefix static finding",
        )
        expected_finding_sha256 = _bounded_owned_hash(
            "vfe4.h6.static-audit-finding.v1",
            {
                key: finding[key]
                for key in _BOUNDED_STATIC_FINDING_FIELDS
                if key != "finding_sha256"
            },
        )
        if finding["finding_sha256"] != expected_finding_sha256:
            raise ArtifactPublicationError(
                "bounded H6 Prefix static finding identity is stale"
            )
        finding_hashes.append(expected_finding_sha256)
    expected_report_sha256 = _bounded_owned_hash(
        "vfe4.h6.static-audit-report.v1",
        {
            "schema_version": report["schema_version"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "rules_sha256": report["rules_sha256"],
            "case_key_manifest_sha256": report[
                "case_key_manifest_sha256"
            ],
            "checks": check_hashes,
            "findings": finding_hashes,
            "status": report["status"],
            "obligations": report["obligations"],
        },
    )
    if (
        report["schema_version"] != "h6-static-audit-v1"
        or report["report_sha256"] != expected_report_sha256
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix static report identity is stale"
        )
    return report


def _validate_bounded_prefix_reference_payloads(
    *,
    resolved_config: object,
    result: object,
    config_payload: Mapping[str, object],
    config_sha256: str,
    source: object,
    validation: Mapping[str, object],
    direct_certificate_payload: Mapping[str, object],
    certificate_set_payload: Mapping[str, object],
    provenance: Mapping[str, object],
    environment: Mapping[str, object],
    junit_sha256: str | None,
) -> A0DirectExactPrefixCertificateV1:
    from vfe4.config.schema import (
        H6_PREFIX_V2_AUTHORIZATION_SHA256,
        H6_PREFIX_V3_AUTHORIZATION_SHA256,
        H6PrefixV3ResolvedConfig,
    )
    from vfe4.types.h6 import (
        BoundedPrefixCertificateSet,
        H6PrefixWorkloadPlan,
    )
    from vfe4.types.results import H6BoundedPrefixGateResult

    if (
        (
            resolved_config is not None
            and type(resolved_config) is not H6PrefixV3ResolvedConfig
        )
        or type(result) is not H6BoundedPrefixGateResult
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix reference requires exact v3 config/result types"
        )
    result.__post_init__()
    certificate_set = result._certificate_set
    if type(certificate_set) is not BoundedPrefixCertificateSet:
        raise ArtifactPublicationError(
            "bounded H6 Prefix result lacks its exact certificate set"
        )
    certificate_set.__post_init__()
    workload = H6PrefixWorkloadPlan()
    expected_validation_fields = {
        "a0_direct_exact_certificate_sha256",
        "schema_version",
        "gate",
        "status",
        "obligations",
        "config_sha256",
        "workload_plan_sha256",
        "validation_payload_sha256",
        "prefix_certificate_set_sha256",
        "runner_totals",
        "semantic_families",
        "static_report",
    }
    expected_certificate_set_fields = {
        "schema_version",
        "config_sha256",
        "workload_plan_sha256",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "semantic_family_sha256s",
        "validation_payload_sha256",
        "prefix_certificate_set_sha256",
        "certificates",
    }
    if (
        set(validation) != expected_validation_fields
        or validation.get("schema_version")
        != "h6-prefix-validation-set-v2"
        or set(certificate_set_payload)
        != expected_certificate_set_fields
        or certificate_set_payload.get("schema_version")
        != "h6-prefix-certificate-set-v2"
        or config_payload.get("schema_version") != "h6-prefix-config-v3"
        or config_payload.get("authorization_sha256")
        != H6_PREFIX_V3_AUTHORIZATION_SHA256
        or config_payload.get("workload_authorization_sha256")
        != H6_PREFIX_V2_AUTHORIZATION_SHA256
        or "validation_fixture_reference" not in config_payload
        or "validation_perturbation_reference" not in config_payload
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix schemas or config references are incomplete"
        )
    try:
        config_bindings = validate_h6_prefix_v3_resolved_payload(
            config_payload
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix resolved-v3 config is invalid"
        ) from exc
    if (
        config_bindings.config_sha256 != config_sha256
        or config_bindings.source.git_head != certificate_set.git_head
        or config_bindings.source.dirty_digest
        != certificate_set.dirty_digest
        or config_bindings.source.source_sha256
        != certificate_set.source_sha256
        or config_bindings.workload_plan_sha256
        != certificate_set.workload_plan_sha256
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix config source or workload is stale"
        )
    if resolved_config is not None and (
        config_bindings.config_sha256 != resolved_config.config_sha256
        or config_bindings.source != resolved_config.source
        or config_bindings.workload_plan_sha256
        != resolved_config.workload_plan_sha256
        or config_bindings.artifact_root != resolved_config.artifact_root
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix config differs from its typed projection"
        )
    certificate_families = tuple(
        (
            certificate.semantic_family_sha256,
            certificate.report_binding.profile_pair_sha256s,
        )
        for certificate in certificate_set.certificates
    )
    if config_bindings.semantic_family_bindings != certificate_families:
        raise ArtifactPublicationError(
            "bounded H6 Prefix config families differ from certificates"
        )
    if any(
        _contains_bounded_external_reference(payload)
        for payload in (
            validation,
            direct_certificate_payload,
            certificate_set_payload,
            provenance,
            environment,
        )
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix external references escaped config.json"
        )
    if (
        result.config_sha256 != config_sha256
        or result.config_sha256 != certificate_set.config_sha256
        or result.workload_plan_sha256
        != certificate_set.workload_plan_sha256
        or validation.get("config_sha256") != result.config_sha256
        or validation.get("workload_plan_sha256")
        != result.workload_plan_sha256
        or validation.get("validation_payload_sha256")
        != result.validation_payload_sha256
        or validation.get("prefix_certificate_set_sha256")
        != result.prefix_certificate_set_sha256
        or certificate_set_payload.get("config_sha256")
        != certificate_set.config_sha256
        or certificate_set_payload.get("workload_plan_sha256")
        != certificate_set.workload_plan_sha256
        or certificate_set_payload.get("validation_payload_sha256")
        != certificate_set.validation_payload_sha256
        or certificate_set_payload.get("prefix_certificate_set_sha256")
        != certificate_set.prefix_certificate_set_sha256
        or certificate_set_payload.get("git_head")
        != certificate_set.git_head
        or certificate_set_payload.get("dirty_digest")
        != certificate_set.dirty_digest
        or certificate_set_payload.get("source_sha256")
        != certificate_set.source_sha256
        or certificate_set.git_head != getattr(source, "git_head", None)
        or certificate_set.dirty_digest
        != getattr(source, "dirty_digest", None)
        or certificate_set.source_sha256
        != getattr(source, "source_sha256", None)
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix config/workload/source/result hashes differ"
        )
    if (
        validation.get("gate") != "H6-Prefix"
        or validation.get("status") != result.status.value
        or validation.get("obligations") != list(result.obligations)
        or certificate_set_payload.get("semantic_family_sha256s")
        != list(certificate_set.semantic_family_sha256s)
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix result or semantic-family order differs"
        )
    serialized_certificates = certificate_set_payload.get("certificates")
    if (
        type(serialized_certificates) is not list
        or len(serialized_certificates)
        != len(certificate_set.certificates)
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate inventory is incomplete"
        )
    for payload, certificate in zip(
        serialized_certificates,
        certificate_set.certificates,
        strict=True,
    ):
        expected = {
            "schema_version": certificate.schema_version,
            "semantic_family_sha256": (
                certificate.semantic_family_sha256
            ),
            "report_binding": (
                certificate.report_binding.canonical_payload()
            ),
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
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            raise ArtifactPublicationError(
                "bounded H6 Prefix certificate payload differs from result"
            )
    direct_certificate = _a0_direct_certificate_from_payload(
        dict(direct_certificate_payload),
        certificate_set=certificate_set,
    )
    current_direct_path, current_scorer_path = (
        _current_a0_direct_path_identities()
    )
    if (
        direct_certificate.status is not EvidenceStatus.PASS
        or direct_certificate.obligations != ()
        or tuple(direct_certificate.checks)
        != H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
        or not all(direct_certificate.checks.values())
        or direct_certificate.git_head != certificate_set.git_head
        or direct_certificate.dirty_digest != certificate_set.dirty_digest
        or direct_certificate.source_sha256 != certificate_set.source_sha256
        or direct_certificate.direct_predictor_path_sha256
        != current_direct_path
        or direct_certificate.heldout_scorer_path_sha256
        != current_scorer_path
        or direct_certificate.checks["target_read_after_prediction"]
        is not True
        or not _current_a0_target_read_follows_prediction()
        or validation.get("a0_direct_exact_certificate_sha256")
        != direct_certificate.certificate_sha256
    ):
        raise ArtifactPublicationError(
            "direct-A0 authority must PASS every required check without "
            "obligations and match current source and validation"
        )
    families = validation.get("semantic_families")
    if (
        type(families) is not list
        or len(families) != len(certificate_set.certificates)
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix validation families are incomplete"
        )
    observed_predictor_call_count = 0
    for family_index, (family_value, certificate) in enumerate(
        zip(families, certificate_set.certificates, strict=True)
    ):
        family = _bounded_exact_object(
            family_value,
            _BOUNDED_FAMILY_FIELDS,
            "bounded H6 Prefix validation family",
        )
        if (
            family.get("semantic_family_index") != family_index
            or family.get("semantic_family_sha256")
            != certificate.semantic_family_sha256
            or family.get("validation_payload_sha256")
            != certificate.validation_payload_sha256
            or family.get("certificate_sha256")
            != certificate.certificate_sha256
        ):
            raise ArtifactPublicationError(
                "bounded H6 Prefix validation family order differs"
            )
        jobs = family.get("jobs")
        if type(jobs) is not list or len(jobs) != 8:
            raise ArtifactPublicationError(
                "bounded H6 Prefix family does not contain eight jobs"
            )
        references = certificate.report_binding.report_references
        for job_index, (job_value, reference) in enumerate(
            zip(jobs, references, strict=True)
        ):
            job = _bounded_exact_object(
                job_value,
                _BOUNDED_JOB_FIELDS,
                "bounded H6 Prefix validation job",
            )
            execution_plan = _bounded_exact_object(
                job["execution_plan"],
                _BOUNDED_EXECUTION_PLAN_FIELDS,
                "bounded H6 Prefix execution plan",
            )
            dynamic_report = _bounded_exact_object(
                job["dynamic_report"],
                _BOUNDED_DYNAMIC_REPORT_FIELDS,
                "bounded H6 Prefix dynamic report",
            )
            (
                expected_by_position,
                full_expected_count,
                selected_global_indices,
                applicable_check_names,
            ) = _bounded_expected_job_contract(
                scope=reference.scope,
                case_family=reference.case_family,
                particle_count=reference.particle_count,
                workload=workload,
            )
            selection_rows = execution_plan["selection_rows"]
            if (
                type(selection_rows) is not list
                or len(selection_rows) != len(selected_global_indices)
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix selection rows are incomplete"
                )
            selection_case_hashes: list[str] = []
            for row_index, row_value in enumerate(selection_rows):
                row = _bounded_exact_object(
                    row_value,
                    frozenset({"global_index", "case_sha256"}),
                    "bounded H6 Prefix selection row",
                )
                case_sha256 = row["case_sha256"]
                if (
                    row["global_index"] != selected_global_indices[row_index]
                    or type(case_sha256) is not str
                    or len(case_sha256) != 64
                    or any(character not in _LOWER_HEX for character in case_sha256)
                ):
                    raise ArtifactPublicationError(
                        "bounded H6 Prefix selection row is stale"
                    )
                selection_case_hashes.append(case_sha256)
            expected_selection_sha256 = _bounded_owned_hash(
                "vfe4.h6.dynamic-selection-manifest.v2",
                selection_rows,
            )
            execution_plan_preimage = {
                key: execution_plan[key]
                for key in _BOUNDED_EXECUTION_PLAN_FIELDS
                if key != "plan_sha256"
            }
            expected_plan_sha256 = _bounded_owned_hash(
                "vfe4.h6.dynamic-execution-plan.v2",
                execution_plan_preimage,
            )
            if (
                len(set(selection_case_hashes))
                != len(selection_case_hashes)
                or execution_plan.get("schema_version")
                != "h6-dynamic-execution-plan-v2"
                or execution_plan.get("scope") != reference.scope
                or execution_plan.get("case_family")
                != reference.case_family
                or execution_plan.get("particle_count")
                != reference.particle_count
                or execution_plan.get("expected_by_position")
                != expected_by_position
                or execution_plan.get("full_expected_count")
                != full_expected_count
                or execution_plan.get("selected_global_indices")
                != selected_global_indices
                or execution_plan.get("authorization_sha256")
                != H6_PREFIX_V2_AUTHORIZATION_SHA256
                or execution_plan.get("workload_plan_sha256")
                != result.workload_plan_sha256
                or execution_plan.get("selection_manifest_sha256")
                != expected_selection_sha256
                or execution_plan.get("selection_manifest_sha256")
                != reference.selection_manifest_sha256
                or execution_plan.get("plan_sha256")
                != expected_plan_sha256
                or execution_plan.get("plan_sha256")
                != reference.execution_plan_sha256
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix execution plan differs from its reference"
                )

            dynamic_checks = dynamic_report["checks"]
            if (
                type(dynamic_checks) is not list
                or len(dynamic_checks) != len(_BOUNDED_DYNAMIC_CHECK_NAMES)
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix dynamic checks are incomplete"
                )
            for check_index, check_value in enumerate(dynamic_checks):
                check = _bounded_exact_object(
                    check_value,
                    _BOUNDED_DYNAMIC_CHECK_FIELDS,
                    "bounded H6 Prefix dynamic check",
                )
                if check.get("name") != _BOUNDED_DYNAMIC_CHECK_NAMES[
                    check_index
                ]:
                    raise ArtifactPublicationError(
                        "bounded H6 Prefix dynamic checks are reordered"
                    )
            dynamic_report_preimage = {
                key: dynamic_report[key]
                for key in _BOUNDED_DYNAMIC_REPORT_FIELDS
                if key != "report_sha256"
            }
            expected_report_sha256 = _bounded_owned_hash(
                "vfe4.h6.dynamic-prefix-report.v2",
                dynamic_report_preimage,
            )
            report_key = dynamic_report.get("key")
            expected_report_key = reference.report_key.canonical_payload()
            expected_calls = sum(expected_by_position) * getattr(
                workload,
                "prediction_calls_per_case",
            )
            if (
                dynamic_report.get("schema_version")
                != "h6-dynamic-prefix-report-v2"
                or canonical_json_bytes(report_key)
                != canonical_json_bytes(expected_report_key)
                or dynamic_report.get("scope") != reference.scope
                or dynamic_report.get("case_family")
                != reference.case_family
                or dynamic_report.get("particle_count")
                != reference.particle_count
                or dynamic_report.get("workload_plan_sha256")
                != result.workload_plan_sha256
                or dynamic_report.get("selected_global_indices")
                != selected_global_indices
                or dynamic_report.get("execution_plan_sha256")
                != execution_plan.get("plan_sha256")
                or dynamic_report.get("execution_plan_sha256")
                != reference.execution_plan_sha256
                or dynamic_report.get("selection_manifest_sha256")
                != expected_selection_sha256
                or dynamic_report.get("selection_manifest_sha256")
                != reference.selection_manifest_sha256
                or dynamic_report.get("completed_by_position")
                != list(reference.completed_by_position)
                or dynamic_report.get("completed_by_position")
                != expected_by_position
                or dynamic_report.get("complete_case_manifest_sha256")
                != reference.complete_case_manifest_sha256
                or dynamic_report.get("model_state_sha256")
                != reference.model_state_sha256
                or dynamic_report.get("proposal_identity_sha256")
                != reference.proposal_identity_sha256
                or dynamic_report.get("estimator_semantic_sha256")
                != reference.estimator_semantic_sha256
                or dynamic_report.get("estimator_artifact_bytes_sha256")
                != reference.estimator_artifact_bytes_sha256
                or dynamic_report.get("applicable_check_names")
                != applicable_check_names
                or dynamic_report.get("report_sha256")
                != expected_report_sha256
                or dynamic_report.get("report_sha256")
                != reference.report_sha256
                or job.get("job_index") != job_index
                or job.get("particle_count") != reference.particle_count
                or job.get("case_family") != reference.case_family
                or job.get("scope") != reference.scope
                or job.get("profile_pair_sha256")
                != reference.profile_pair_sha256
                or type(job.get("observed_predictor_call_count")) is not int
                or job.get("observed_predictor_call_count") != expected_calls
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix dynamic job differs from its report reference"
                )
            observed_predictor_call_count += expected_calls

    static_report = _validate_bounded_static_report_payload(
        validation.get("static_report")
    )
    for certificate in certificate_set.certificates:
        binding = certificate.report_binding
        if (
            static_report["report_sha256"] != binding.static_report_sha256
            or static_report["source_manifest_sha256"]
            != binding.static_source_manifest_sha256
            or static_report["rules_sha256"]
            != binding.static_rules_sha256
            or static_report["case_key_manifest_sha256"]
            != binding.static_case_key_manifest_sha256
        ):
            raise ArtifactPublicationError(
                "bounded H6 Prefix static report differs from a family binding"
            )
    if (
        direct_certificate.static_report_sha256
        != static_report["report_sha256"]
        or direct_certificate.static_report_status
        is not EvidenceStatus(static_report["status"])
    ):
        raise ArtifactPublicationError(
            "direct-A0 authority differs from the global static report"
        )
    expected_direct_obligations = tuple(
        sorted(
            {
                *(
                    f"bounded A0: {value}"
                    for value in (
                        direct_certificate.bounded_a0_certificate.obligations
                    )
                ),
                *(
                    f"static: {value}"
                    for value in static_report["obligations"]
                ),
                *(
                    f"static/{check['name']}: {value}"
                    for check in static_report["checks"]
                    for value in check["obligations"]
                ),
            }
        )
    )
    if (
        direct_certificate.status is EvidenceStatus.INCONCLUSIVE
        and direct_certificate.obligations != expected_direct_obligations
    ):
        raise ArtifactPublicationError(
            "direct-A0 obligations differ from the unresolved premises"
        )

    family_count = len(certificate_set.certificates)
    runner_totals = _bounded_exact_object(
        validation.get("runner_totals"),
        _BOUNDED_RUNNER_TOTAL_FIELDS,
        "bounded H6 Prefix runner totals",
    )
    expected_runner_totals = {
        "semantic_family_count": family_count,
        "planned_fixture_load_count": 1,
        "planned_static_audit_count": 1,
        "planned_arm_build_count": 2 * family_count,
        "planned_predictor_boundary_count": 8 * family_count,
        "planned_dynamic_report_count": 8 * family_count,
        "observed_dynamic_report_count": 8 * family_count,
        "planned_case_count": workload.amended_total_cases * family_count,
        "planned_predictor_call_count": (
            workload.amended_total_prediction_calls * family_count
        ),
        "observed_predictor_call_count": observed_predictor_call_count,
        "planned_particle_call_units": (
            workload.amended_total_particle_call_units * family_count
        ),
    }
    if (
        runner_totals != expected_runner_totals
        or observed_predictor_call_count
        != expected_runner_totals["planned_predictor_call_count"]
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix runner totals differ from the exact workload"
        )
    if (
        set(provenance)
        != {
            "schema_version",
            "git_head",
            "dirty_digest",
            "source_sha256",
            "junit_sha256",
        }
        or provenance.get("schema_version") != "h6-prefix-provenance-v1"
        or provenance.get("git_head") != certificate_set.git_head
        or provenance.get("dirty_digest") != certificate_set.dirty_digest
        or provenance.get("source_sha256")
        != certificate_set.source_sha256
        or provenance.get("junit_sha256") != junit_sha256
        or frozenset(environment) != _BOUNDED_ENVIRONMENT_FIELDS
        or environment.get("schema_version")
        != "h6-prefix-environment-v1"
        or environment.get("device") != "cpu"
        or environment.get("dtype") != "float64"
        or type(environment.get("python_implementation")) is not str
        or not environment.get("python_implementation")
        or type(environment.get("python_version")) is not str
        or not environment.get("python_version")
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix provenance/environment differ"
        )
    return direct_certificate


def reopen_h6_prefix_authorities(
    root: Path,
    expected_manifest_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
    expected_junit_sha256: str,
) -> tuple[
    BoundedPrefixCertificateSet,
    A0DirectExactPrefixCertificateV1,
]:
    """Reopen both exact typed authorities from one current H6 Prefix artifact."""

    from vfe4.types.results import H6BoundedPrefixGateResult

    if not isinstance(root, Path):
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact root must be a Path"
        )
    try:
        manifest_sha256 = _require_lower_hex(
            expected_manifest_sha256,
            64,
            "bounded H6 Prefix manifest SHA-256",
        )
        git_head = _require_lower_hex(
            expected_git_head,
            40,
            "bounded H6 Prefix Git HEAD",
        )
        dirty_digest = _require_lower_hex(
            expected_dirty_digest,
            64,
            "bounded H6 Prefix dirty digest",
        )
        junit_sha256 = _require_lower_hex(
            expected_junit_sha256,
            64,
            "bounded H6 Prefix JUnit SHA-256",
        )
    except ValueError as exc:
        raise ArtifactPublicationError(str(exc)) from exc
    if root.is_symlink():
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact root cannot be a symlink"
        )
    try:
        artifact_root = root.resolve(strict=True)
    except OSError as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact root is unavailable"
        ) from exc
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact root must be a real directory"
        )

    manifest_path = artifact_root / "manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact lacks a regular manifest"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix manifest is unreadable"
        ) from exc
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ArtifactPublicationError(
            "bounded H6 Prefix manifest differs from its expected digest"
        )
    entries = _manifest_entries(manifest_bytes)
    expected_names = _EXPECTED_H6_PREFIX_V3_PAYLOADS
    if tuple(name for name, _digest in entries) != expected_names:
        raise ArtifactPublicationError(
            "bounded H6 Prefix payload inventory differs from its exact six files"
        )

    observed_files: set[str] = set()
    try:
        descendants = tuple(artifact_root.rglob("*"))
    except OSError as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact cannot be enumerated"
        ) from exc
    for path in descendants:
        if path.is_symlink():
            raise ArtifactPublicationError(
                "bounded H6 Prefix artifact contains a symlink"
            )
        if path.is_file():
            observed_files.add(path.relative_to(artifact_root).as_posix())
        elif not path.is_dir():
            raise ArtifactPublicationError(
                "bounded H6 Prefix artifact contains a non-file entry"
            )
    if observed_files != {*expected_names, "manifest.sha256"}:
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact has unlisted or missing files"
        )

    payloads: dict[str, object] = {}
    manifest_hashes = dict(entries)
    for name in expected_names:
        path = artifact_root.joinpath(*PurePosixPath(name).parts)
        if not path.is_file() or path.is_symlink():
            raise ArtifactPublicationError(
                f"bounded H6 Prefix payload is not a regular file: {name}"
            )
        if _file_sha256(path) != manifest_hashes[name]:
            raise ArtifactPublicationError(
                f"bounded H6 Prefix payload hash differs: {name}"
            )
        payloads[name] = _read_json_payload(path)

    config_payload = _bounded_exact_object(
        payloads["config.json"],
        _BOUNDED_V3_CONFIG_FIELDS,
        "bounded H6 Prefix config",
    )
    validation = payloads["validation/h6_prefix.json"]
    direct_certificate_payload = payloads[
        "certificates/a0_direct_exact.json"
    ]
    certificate_set_payload = payloads["certificates/prefix_set.json"]
    provenance = payloads["provenance.json"]
    environment = payloads["environment.json"]
    if not all(
        type(payload) is dict
        for payload in (
            validation,
            direct_certificate_payload,
            certificate_set_payload,
            provenance,
            environment,
        )
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix payload roots must be exact JSON objects"
        )

    config_bytes = canonical_json_bytes(config_payload)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    certificate_set = _bounded_certificate_set_from_payload(
        certificate_set_payload
    )
    if (
        certificate_set.config_sha256 != config_sha256
        or certificate_set.git_head != git_head
        or certificate_set.dirty_digest != dirty_digest
    ):
        raise ArtifactPublicationError(
            "bounded H6 Prefix certificate set differs from config/source"
        )
    try:
        result = H6BoundedPrefixGateResult.from_certificate_set(
            certificate_set
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix result cannot be reconstructed"
        ) from exc
    try:
        direct_certificate = _validate_bounded_prefix_reference_payloads(
            resolved_config=None,
            result=result,
            config_payload=config_payload,
            config_sha256=config_sha256,
            source=certificate_set,
            validation=validation,
            direct_certificate_payload=direct_certificate_payload,
            certificate_set_payload=certificate_set_payload,
            provenance=provenance,
            environment=environment,
            junit_sha256=junit_sha256,
        )
    except ArtifactPublicationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "bounded H6 Prefix artifact payload is invalid"
        ) from exc
    return certificate_set, direct_certificate


def reopen_bounded_prefix_certificate_set(
    root: Path,
    expected_manifest_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
    expected_junit_sha256: str,
) -> BoundedPrefixCertificateSet:
    """Compatibility projection after validating both current authorities."""

    certificate_set, _direct = reopen_h6_prefix_authorities(
        root,
        expected_manifest_sha256,
        expected_git_head,
        expected_dirty_digest,
        expected_junit_sha256,
    )
    return certificate_set


def _reference_from_published_directory(
    *,
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
    resolved_config: object,
    result: object,
    run_directory: Path,
    junit_sha256: str | None,
) -> CandidateArtifactReference:
    if not isinstance(run_directory, Path):
        raise ArtifactPublicationError("gate runner did not return a Path")
    if run_directory.is_symlink():
        raise ArtifactPublicationError("published run directory cannot be a symlink")
    try:
        root = run_directory.resolve(strict=True)
    except OSError as exc:
        raise ArtifactPublicationError("published run directory is unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise ArtifactPublicationError(
            "published run path must be a real, non-symlink directory"
        )
    manifest_path = root / "manifest.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArtifactPublicationError("published artifact lacks a regular manifest")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ArtifactPublicationError("published artifact manifest is unreadable") from exc
    entries = _manifest_entries(manifest_bytes)
    names = tuple(name for name, _ in entries)
    expected_names = (
        _EXPECTED_H6_PREFIX_V3_PAYLOADS
        if (
            operation == "H6-Prefix"
            and getattr(resolved_config, "schema_version", None)
            == "h6-prefix-config-v3"
        )
        else _EXPECTED_PAYLOADS[operation]
    )
    if names != expected_names:
        raise ArtifactPublicationError(
            f"{operation} artifact payload inventory differs from the frozen contract"
        )

    observed_files: set[str] = set()
    try:
        descendants = tuple(root.rglob("*"))
    except OSError as exc:
        raise ArtifactPublicationError("published artifact cannot be enumerated") from exc
    for path in descendants:
        if path.is_symlink():
            raise ArtifactPublicationError("published artifact contains a symlink")
        if path.is_file():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ArtifactPublicationError(
                    "published artifact payload escapes its directory"
                ) from exc
            observed_files.add(relative)
        elif not path.is_dir():
            raise ArtifactPublicationError(
                "published artifact contains a non-file, non-directory entry"
            )
    if observed_files != {*expected_names, "manifest.sha256"}:
        raise ArtifactPublicationError(
            "published artifact contains an unlisted or missing payload"
        )

    payload_hashes: dict[str, str] = {}
    for name, expected_digest in entries:
        path = root / Path(*PurePosixPath(name).parts)
        if not path.is_file() or path.is_symlink():
            raise ArtifactPublicationError("artifact manifest names an invalid payload")
        observed_digest = _file_sha256(path)
        if observed_digest != expected_digest:
            raise ArtifactPublicationError(
                f"artifact payload hash differs from manifest: {name}"
            )
        payload_hashes[name] = observed_digest

    canonical_json = getattr(resolved_config, "canonical_json", None)
    if type(canonical_json) is not str:
        raise ArtifactPublicationError("resolved config lacks canonical JSON")
    try:
        config_bytes = (root / "config.json").read_bytes()
    except OSError as exc:
        raise ArtifactPublicationError("artifact config is unreadable") from exc
    if config_bytes != canonical_json.encode("utf-8"):
        raise ArtifactPublicationError(
            "artifact config bytes differ from the independently resolved projection"
        )
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != getattr(resolved_config, "config_sha256", None):
        raise ArtifactPublicationError(
            "artifact config SHA-256 differs from the resolved projection"
        )
    config_payload = _read_json_payload(root / "config.json")
    if not isinstance(config_payload, Mapping):
        raise ArtifactPublicationError("artifact config must be a JSON object")
    source_payload = config_payload.get("source")
    source = getattr(resolved_config, "source", None)
    if (
        not isinstance(source_payload, Mapping)
        or source_payload.get("git_head") != getattr(source, "git_head", None)
        or source_payload.get("dirty_digest") != getattr(source, "dirty_digest", None)
        or source_payload.get("source_sha256")
        != getattr(source, "source_sha256", None)
    ):
        raise ArtifactPublicationError(
            "artifact source identity differs from the resolved projection"
        )

    validation_name = (
        "validation/h1_prefix_prior.json"
        if operation == "H1-Prefix-Prior"
        else "validation/h6_prefix.json"
    )
    validation = _read_json_payload(root / Path(*validation_name.split("/")))
    expected_status = getattr(getattr(result, "status", None), "value", None)
    result_obligations = getattr(result, "obligations", None)
    validation_obligations = (
        validation.get("obligations") if isinstance(validation, Mapping) else None
    )
    if (
        getattr(result, "gate", None) != operation
        or not isinstance(validation, Mapping)
        or validation.get("gate") != operation
        or validation.get("status") != expected_status
        or type(result_obligations) is not tuple
        or type(validation_obligations) is not list
        or tuple(validation_obligations) != result_obligations
    ):
        raise ArtifactPublicationError(
            "gate result and published validation payload do not agree"
        )
    if operation == "H1-Prefix-Prior":
        if (
            validation.get("git_head") != getattr(source, "git_head", None)
            or validation.get("dirty_digest")
            != getattr(source, "dirty_digest", None)
            or validation.get("config_sha256") != config_sha256
            or (
                validation.get("schema_version")
                == "h1-prefix-prior-validation-v3"
                and validation.get("junit_sha256") != junit_sha256
            )
        ):
            raise ArtifactPublicationError(
                "H1 prefix-prior validation identity differs from its config"
            )
    else:
        provenance = _read_json_payload(root / "provenance.json")
        certificate_set = _read_json_payload(
            root / "certificates" / "prefix_set.json"
        )
        from vfe4.types.results import H6BoundedPrefixGateResult

        bounded = (
            config_payload.get("schema_version") == "h6-prefix-config-v3"
            or type(result) is H6BoundedPrefixGateResult
        )
        if bounded:
            direct_certificate = _read_json_payload(
                root / "certificates" / "a0_direct_exact.json"
            )
            environment = _read_json_payload(root / "environment.json")
            if (
                not isinstance(provenance, Mapping)
                or not isinstance(certificate_set, Mapping)
                or not isinstance(direct_certificate, Mapping)
                or not isinstance(environment, Mapping)
            ):
                raise ArtifactPublicationError(
                    "bounded H6 Prefix payloads must be JSON objects"
                )
            _validate_bounded_prefix_reference_payloads(
                resolved_config=resolved_config,
                result=result,
                config_payload=config_payload,
                config_sha256=config_sha256,
                source=source,
                validation=validation,
                direct_certificate_payload=direct_certificate,
                certificate_set_payload=certificate_set,
                provenance=provenance,
                environment=environment,
                junit_sha256=junit_sha256,
            )
        elif (
            not isinstance(provenance, Mapping)
            or provenance.get("git_head") != getattr(source, "git_head", None)
            or provenance.get("dirty_digest")
            != getattr(source, "dirty_digest", None)
            or provenance.get("source_sha256")
            != getattr(source, "source_sha256", None)
            or provenance.get("junit_sha256") != junit_sha256
            or not isinstance(certificate_set, Mapping)
            or validation.get("validation_payload_sha256")
            != getattr(result, "validation_payload_sha256", None)
            or validation.get("prefix_certificate_set_sha256")
            != getattr(result, "prefix_certificate_set_sha256", None)
            or certificate_set.get("prefix_certificate_set_sha256")
            != getattr(result, "prefix_certificate_set_sha256", None)
        ):
            raise ArtifactPublicationError(
                "H6 Prefix provenance and certificate identities do not match "
                "the requested candidate/result"
            )

    return CandidateArtifactReference(
        root,
        getattr(source, "git_head", None),
        getattr(source, "dirty_digest", None),
        hashlib.sha256(manifest_bytes).hexdigest(),
        payload_hashes,
    )


def run_projected_current_candidate(
    *,
    config: ProjectedCurrentCandidateConfig,
    junit_sha256: str | None,
    predecessor_refs: Mapping[str, CandidateArtifactReference],
) -> CandidateArtifactReference:
    """Run one projected gate and reconstruct its reference from published bytes."""

    if type(config) is not ProjectedCurrentCandidateConfig:
        raise ValueError("config must be an exact ProjectedCurrentCandidateConfig")
    validated_junit = _validated_junit_sha256(junit_sha256)
    _validate_predecessor_mapping(predecessor_refs, config.operation)
    resolved = _resolve_projected_config(config.operation, config.raw_config)
    if (
        getattr(resolved, "operation", None) != config.operation
        or getattr(resolved, "config_sha256", None) != config.canonical_sha256
    ):
        raise ValueError("projected config changed after projection")

    runner = _PROJECTED_CURRENT_CANDIDATE_RUNNER
    if runner is None:
        raise ArtifactPublicationError(
            "no eligible projected current-candidate runner is installed"
        )
    output = runner(config.operation, resolved, validated_junit)
    if type(output) is not tuple or len(output) != 2:
        raise ArtifactPublicationError(
            "projected gate runner did not return its exact result/path pair"
        )
    result, run_directory = output
    return _reference_from_published_directory(
        operation=config.operation,
        resolved_config=resolved,
        result=result,
        run_directory=run_directory,
        junit_sha256=validated_junit,
    )


def _canonical_json_object_from_bytes(
    payload: bytes,
    *,
    name: str,
) -> dict[str, object]:
    if type(payload) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical UTF-8 JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


_A0_ENDPOINT_ID = "h6-a0-transformer-v2"
_A5_COMPLETE_ENDPOINT_ID = (
    "h6-a5-structured-parent-specific-prefix-exact-complete-"
    "latent-smoothing-v2"
)
_A5_EMISSION_ENDPOINT_ID = (
    "h6-a5-structured-parent-specific-prefix-exact-emission-"
    "latent-smoothing-v2"
)
_PREDICTION_ENDPOINT_IDS = (
    _A0_ENDPOINT_ID,
    "h6-a1-ordinary-latent-v1",
    "h6-a2-generic-map-v1",
    "h6-a3-immediate-predecessor-v1",
    "h6-a4-state-only-v1",
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
    _A5_COMPLETE_ENDPOINT_ID,
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1",
    _A5_EMISSION_ENDPOINT_ID,
    (
        "h6-a5-structured-fixed-exact-complete-"
        "nolatent-norecognition-v1"
    ),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1",
)
_PRIMARY_OBJECTIVE_ENDPOINT_IDS = (
    _A0_ENDPOINT_ID,
    _A5_COMPLETE_ENDPOINT_ID,
    _A5_EMISSION_ENDPOINT_ID,
)
_PREDICTION_COMPARISONS = {
    "OBJECTIVE": [
        _A5_COMPLETE_ENDPOINT_ID,
        _A5_EMISSION_ENDPOINT_ID,
    ],
    "PRIMARY": [
        _A0_ENDPOINT_ID,
        _A5_COMPLETE_ENDPOINT_ID,
    ],
}
_H6_CONFIRMATORY_SEEDS = tuple(range(2026072101, 2026072109))
_ENDPOINT_CONFIG_FIELDS = {
    "endpoint_id",
    "endpoint_config_sha256",
    "config",
}
_ARM_CONFIG_FIELDS = {
    "arm",
    "config_id",
    "vocabulary",
    "horizon",
    "latent_enabled",
    "state_channel_enabled",
    "model_channel_enabled",
    "source_mode",
    "map_mode",
    "recognition_family",
    "recognition_conditioning",
    "prior_variant",
    "mixture_mode",
    "objective_kind",
    "capacity_allocation",
    "capacity_allocation_sha256",
}
_ENDPOINT_OBSERVATION_FIELDS = {
    "endpoint_id",
    "endpoint_config_sha256",
    "checkpoint_sha256",
    "checkpoint_git_head",
    "checkpoint_dirty_digest",
    "confirmatory_seed",
    "replicate_id",
    "particle_count",
    "common_stream_sha256",
    "log_normalizer_sum",
    "negative_log_likelihood_sum",
    "counted_targets",
    "counter_consumption",
    "cache_audit",
    "failure",
    "test_opening_sha256",
}
_COUNTER_CONSUMPTION_FIELDS = {
    "schema",
    "consumption_count",
    "counter_trace_sha256",
    "complete",
    "consumption_sha256",
}
_CACHE_AUDIT_FIELDS = {
    "schema",
    "mode_order",
    "cold_result_sha256",
    "warm_result_sha256",
    "cold_negative_log_likelihood_sum",
    "warm_negative_log_likelihood_sum",
    "exact_match",
    "cache_audit_sha256",
}
_FAILURE_FIELDS = {
    "schema",
    "status",
    "failure_kind",
    "details_sha256",
}


@dataclass(frozen=True)
class _ValidatedPredictionRawReceipt:
    """Private receipt derived only from the immutable raw inventory."""

    payload: dict[str, object]
    raw_sha256: str
    objective_interval: object
    primary_interval: object
    objective_estimator_complete: bool
    primary_estimator_complete: bool


def _raw_record_identity_sha256(domain: bytes, payload: object) -> str:
    return hashlib.sha256(
        domain + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _validate_counter_consumption(
    value: object,
    *,
    protocol_sha256: str,
    endpoint_id: str,
    endpoint_config_sha256: str,
    checkpoint_sha256: str,
    confirmatory_seed: int,
    replicate_id: int,
    particle_count: int,
    common_stream_sha256: str,
) -> None:
    if type(value) is not dict or set(value) != _COUNTER_CONSUMPTION_FIELDS:
        raise ValueError("counter-consumption record fields are not exact")
    consumption_count = value["consumption_count"]
    if type(consumption_count) is not int or consumption_count < 0:
        raise ValueError("counter-consumption count must be nonnegative")
    counter_trace_sha256 = _require_lower_hex(
        value["counter_trace_sha256"],
        64,
        "counter_trace_sha256",
    )
    if (
        value["schema"] != "h6-endpoint-counter-consumption-v1"
        or value["complete"] is not True
    ):
        raise ValueError("counter-consumption record is not complete")
    expected_sha256 = _raw_record_identity_sha256(
        b"vfe4.h6.endpoint-counter-consumption.v1",
        {
            "schema": value["schema"],
            "protocol_sha256": protocol_sha256,
            "endpoint_id": endpoint_id,
            "endpoint_config_sha256": endpoint_config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "confirmatory_seed": confirmatory_seed,
            "replicate_id": replicate_id,
            "particle_count": particle_count,
            "common_stream_sha256": common_stream_sha256,
            "consumption_count": consumption_count,
            "counter_trace_sha256": counter_trace_sha256,
            "complete": True,
        },
    )
    if value["consumption_sha256"] != expected_sha256:
        raise ValueError(
            "counter-consumption digest does not bind its endpoint/stream context"
        )


def _validate_cache_audit(
    value: object,
    *,
    endpoint_id: str,
    endpoint_config_sha256: str,
    checkpoint_sha256: str,
    confirmatory_seed: int,
    replicate_id: int,
    particle_count: int,
    negative_log_likelihood_sum: float,
) -> None:
    if type(value) is not dict or set(value) != _CACHE_AUDIT_FIELDS:
        raise ValueError("cache-audit record fields are not exact")
    cold_sum = value["cold_negative_log_likelihood_sum"]
    warm_sum = value["warm_negative_log_likelihood_sum"]
    if type(cold_sum) is not float or type(warm_sum) is not float:
        raise ValueError("cache-audit totals must be binary64 floats")
    cold_sha256 = _require_lower_hex(
        value["cold_result_sha256"],
        64,
        "cold_result_sha256",
    )
    warm_sha256 = _require_lower_hex(
        value["warm_result_sha256"],
        64,
        "warm_result_sha256",
    )
    if (
        value["schema"] != "h6-endpoint-cache-audit-v1"
        or value["mode_order"] != ["cold", "warm"]
        or value["exact_match"] is not True
        or not math.isfinite(cold_sum)
        or not math.isfinite(warm_sum)
        or cold_sum != negative_log_likelihood_sum
        or warm_sum != negative_log_likelihood_sum
        or cold_sha256 != warm_sha256
    ):
        raise ValueError(
            "cache audit must prove exact cold-then-warm scoring identity"
        )
    expected_sha256 = _raw_record_identity_sha256(
        b"vfe4.h6.endpoint-cache-audit.v1",
        {
            "schema": value["schema"],
            "endpoint_id": endpoint_id,
            "endpoint_config_sha256": endpoint_config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "confirmatory_seed": confirmatory_seed,
            "replicate_id": replicate_id,
            "particle_count": particle_count,
            "mode_order": value["mode_order"],
            "cold_result_sha256": cold_sha256,
            "warm_result_sha256": warm_sha256,
            "cold_negative_log_likelihood_sum": cold_sum,
            "warm_negative_log_likelihood_sum": warm_sum,
            "exact_match": True,
        },
    )
    if value["cache_audit_sha256"] != expected_sha256:
        raise ValueError(
            "cache-audit digest does not bind its endpoint/particle context"
        )


def _validate_no_endpoint_failure(value: object) -> None:
    if (
        type(value) is not dict
        or set(value) != _FAILURE_FIELDS
        or value["schema"] != "h6-endpoint-failure-v1"
    ):
        raise ValueError("endpoint failure record fields are not exact")
    if (
        value["status"] != "NONE"
        or value["failure_kind"] is not None
        or value["details_sha256"] is not None
    ):
        raise ValueError(
            "a retained endpoint failure prevents H6 Prediction closure"
        )


def _read_prediction_endpoint_config(
    value: object,
    *,
    expected_endpoint_id: str,
) -> object:
    from vfe4.types.h6 import (
        ArmConfig,
        ArmId,
        CapacityAllocation,
        VocabularyIdentity,
    )

    if type(value) is not dict or set(value) != _ENDPOINT_CONFIG_FIELDS:
        raise ValueError("raw endpoint config record fields are not exact")
    if value["endpoint_id"] != expected_endpoint_id:
        raise ValueError("raw endpoint configs are missing or reordered")
    declared_sha256 = _require_lower_hex(
        value["endpoint_config_sha256"],
        64,
        "endpoint_config_sha256",
    )
    raw_config = value["config"]
    if type(raw_config) is not dict or set(raw_config) != _ARM_CONFIG_FIELDS:
        raise ValueError("raw endpoint ArmConfig fields are not exact")
    raw_vocabulary = raw_config["vocabulary"]
    raw_allocation = raw_config["capacity_allocation"]
    if (
        type(raw_vocabulary) is not dict
        or set(raw_vocabulary)
        != {"vocabulary_id", "size", "tokenizer_spec_sha256"}
        or type(raw_allocation) is not dict
        or set(raw_allocation)
        != {
            "emission_width",
            "latent_width",
            "recognition_width",
            "prior_context_width",
        }
    ):
        raise ValueError("raw endpoint vocabulary/allocation fields are not exact")
    try:
        vocabulary = VocabularyIdentity(
            raw_vocabulary["vocabulary_id"],
            raw_vocabulary["size"],
            raw_vocabulary["tokenizer_spec_sha256"],
        )
        allocation = CapacityAllocation.create(
            emission_width=raw_allocation["emission_width"],
            latent_width=raw_allocation["latent_width"],
            recognition_width=raw_allocation["recognition_width"],
            prior_context_width=raw_allocation["prior_context_width"],
        )
        arm = ArmConfig.create(
            arm=ArmId(raw_config["arm"]),
            config_id=raw_config["config_id"],
            vocabulary=vocabulary,
            horizon=raw_config["horizon"],
            latent_enabled=raw_config["latent_enabled"],
            state_channel_enabled=raw_config["state_channel_enabled"],
            model_channel_enabled=raw_config["model_channel_enabled"],
            source_mode=raw_config["source_mode"],
            map_mode=raw_config["map_mode"],
            recognition_family=raw_config["recognition_family"],
            recognition_conditioning=raw_config["recognition_conditioning"],
            prior_variant=raw_config["prior_variant"],
            mixture_mode=raw_config["mixture_mode"],
            objective_kind=raw_config["objective_kind"],
            capacity_allocation=allocation,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("raw endpoint config is not a valid typed ArmConfig") from exc
    if (
        raw_config["capacity_allocation_sha256"]
        != allocation.allocation_sha256
        or raw_config != arm.canonical_payload()
        or arm.config_id != expected_endpoint_id
        or arm.config_sha256 != declared_sha256
    ):
        raise ValueError(
            "raw endpoint config does not reproduce its exact semantic/allocation "
            "identity"
        )
    return arm


def _validate_prediction_endpoint_configs(
    value: object,
    *,
    readiness: object,
) -> dict[str, object]:
    from vfe4.artifacts.h6_matching import H6MatchingSetRecord

    if type(value) is not list or len(value) != len(_PREDICTION_ENDPOINT_IDS):
        raise ValueError("raw endpoint config inventory is not exact")
    configs = {
        endpoint_id: _read_prediction_endpoint_config(
            raw_config,
            expected_endpoint_id=endpoint_id,
        )
        for endpoint_id, raw_config in zip(
            _PREDICTION_ENDPOINT_IDS,
            value,
            strict=True,
        )
    }
    schedule = getattr(readiness, "_training_schedule", None)
    endpoint_phases = getattr(schedule, "endpoint_phases", None)
    if type(endpoint_phases) is not tuple:
        raise ValueError("readiness lacks its private validated training schedule")
    phase_by_config_sha256 = {
        item.endpoint_config_sha256: item for item in endpoint_phases
    }
    if any(
        config.config_sha256 not in phase_by_config_sha256
        for config in configs.values()
    ):
        raise ValueError(
            "raw endpoint config was not authorized by the readiness schedule"
        )
    if (
        phase_by_config_sha256[
            configs[_A0_ENDPOINT_ID].config_sha256
        ].latent_enabled
        or not phase_by_config_sha256[
            configs[_A5_COMPLETE_ENDPOINT_ID].config_sha256
        ].latent_enabled
        or not phase_by_config_sha256[
            configs[_A5_EMISSION_ENDPOINT_ID].config_sha256
        ].latent_enabled
    ):
        raise ValueError("raw endpoint phase identities contradict their arm configs")

    matching_set = getattr(readiness, "_matching_set", None)
    if type(matching_set) is not H6MatchingSetRecord:
        raise ValueError(
            "readiness lacks its private validated matching-set receipt"
        )
    matching_set.__post_init__()
    matching_configs = {
        item.config.config_id: item.config
        for item in matching_set.ownership_inventories
    }
    matching_prediction_subset = {
        endpoint_id: matching_configs.get(endpoint_id)
        for endpoint_id in _PRIMARY_OBJECTIVE_ENDPOINT_IDS
    }
    if (
        matching_set.status != "ELIGIBLE"
        or matching_set.obligations
        or matching_set.matching_set_sha256
        != getattr(readiness, "matching_set_sha256", None)
        or matching_set.git_head != getattr(readiness, "git_head", None)
        or matching_set.dirty_digest
        != getattr(readiness, "dirty_digest", None)
        or tuple(matching_prediction_subset)
        != _PRIMARY_OBJECTIVE_ENDPOINT_IDS
        or any(
            configs[endpoint_id] != matching_prediction_subset[endpoint_id]
            for endpoint_id in _PRIMARY_OBJECTIVE_ENDPOINT_IDS
        )
    ):
        raise ValueError(
            "raw endpoint configs do not equal the exact eligible matching-set "
            "PRIMARY/OBJECTIVE ownership subset"
        )

    complete = configs[_A5_COMPLETE_ENDPOINT_ID]
    emission = configs[_A5_EMISSION_ENDPOINT_ID]
    semantic_differences = tuple(
        name
        for name, item in complete.semantic_payload().items()
        if emission.semantic_payload()[name] != item
    )
    if (
        complete.vocabulary != emission.vocabulary
        or complete.horizon != emission.horizon
        or complete.capacity_allocation != emission.capacity_allocation
        or complete.prior_variant != "parent_specific_pooled_prefix"
        or emission.prior_variant != "parent_specific_pooled_prefix"
        or complete.objective_kind != "complete_elbo"
        or emission.objective_kind != "emission_only_ablation_non_elbo"
        or semantic_differences != ("objective_kind",)
    ):
        raise ValueError(
            "OBJECTIVE raw endpoints must share the readiness-selected nuisance "
            "allocation and differ only by objective_kind"
        )
    return configs


def _paired_q2_interval(
    *,
    aggregates: Mapping[tuple[str, int], object],
    left_endpoint_id: str,
    right_endpoint_id: str,
) -> tuple[object, bool]:
    from vfe4.evaluation.smc_uncertainty import (
        ENDPOINT_DEGREES_OF_FREEDOM,
        ENDPOINT_REPLICATE_COUNT,
        EndpointSmcAggregate,
        inflate_paired_interval,
    )
    from vfe4.numerics.critical_values import ENDPOINT_T_DF63

    values: list[float] = []
    paired_half_widths: list[float] = []
    left_bias_bounds: list[float] = []
    right_bias_bounds: list[float] = []
    estimator_complete = True
    for seed in _H6_CONFIRMATORY_SEEDS:
        left = aggregates[(left_endpoint_id, seed)]
        right = aggregates[(right_endpoint_id, seed)]
        if (
            type(left) is not EndpointSmcAggregate
            or type(right) is not EndpointSmcAggregate
        ):
            raise ValueError("paired Q2 arithmetic requires exact endpoint aggregates")
        differences = tuple(
            left_value - right_value
            for left_value, right_value in zip(
                left.q2,
                right.q2,
                strict=True,
            )
        )
        mean = math.fsum(differences) / ENDPOINT_REPLICATE_COUNT
        variance = math.fsum(
            (value - mean) ** 2 for value in differences
        ) / ENDPOINT_DEGREES_OF_FREEDOM
        paired_half_width = ENDPOINT_T_DF63 * math.sqrt(
            variance / ENDPOINT_REPLICATE_COUNT
        )
        if not all(
            math.isfinite(value)
            for value in (mean, variance, paired_half_width)
        ):
            raise ValueError("paired Q2 arithmetic produced a nonfinite value")
        values.append(mean)
        paired_half_widths.append(paired_half_width)
        left_bias_bounds.append(left.bias_bound)
        right_bias_bounds.append(right.bias_bound)
        estimator_complete = (
            estimator_complete and left.eligible and right.eligible
        )
    return (
        inflate_paired_interval(
            tuple(values),
            tuple(paired_half_widths),
            tuple(left_bias_bounds),
            tuple(right_bias_bounds),
        ),
        estimator_complete,
    )


def _validate_prediction_raw_inventory(
    raw_endpoint_records_bytes: bytes,
    *,
    readiness: object,
    test_opening_sha256: str,
) -> _ValidatedPredictionRawReceipt:
    from vfe4.evaluation.smc_uncertainty import (
        ENDPOINT_PARTICLE_COUNTS,
        ENDPOINT_REPLICATE_COUNT,
        EndpointSmcAggregate,
        EndpointSmcObservation,
        aggregate_endpoint_smc,
    )

    raw = _canonical_json_object_from_bytes(
        raw_endpoint_records_bytes,
        name="raw endpoint inventory",
    )
    if set(raw) != {
        "schema",
        "readiness_sha256",
        "matching_set_sha256",
        "git_head",
        "dirty_digest",
        "endpoint_smc_protocol_sha256",
        "smc_bias_semantics_sha256",
        "data_identity_sha256",
        "opening_count",
        "test_opening_sha256",
        "comparison_endpoint_ids",
        "endpoint_configs",
        "records",
    }:
        raise ValueError("raw endpoint inventory fields are not exact")
    readiness_sha256 = _require_lower_hex(
        getattr(readiness, "readiness_sha256", None),
        64,
        "readiness_sha256",
    )
    matching_set_sha256 = _require_lower_hex(
        getattr(readiness, "matching_set_sha256", None),
        64,
        "matching_set_sha256",
    )
    git_head = _require_lower_hex(
        getattr(readiness, "git_head", None),
        40,
        "git_head",
    )
    dirty_digest = _require_lower_hex(
        getattr(readiness, "dirty_digest", None),
        64,
        "dirty_digest",
    )
    endpoint_smc_protocol_sha256 = _require_lower_hex(
        getattr(readiness, "endpoint_smc_protocol_sha256", None),
        64,
        "endpoint_smc_protocol_sha256",
    )
    smc_bias_semantics_sha256 = _require_lower_hex(
        getattr(readiness, "smc_bias_semantics_sha256", None),
        64,
        "smc_bias_semantics_sha256",
    )
    data_identity_sha256 = _require_lower_hex(
        getattr(readiness, "data_identity_sha256", None),
        64,
        "data_identity_sha256",
    )
    test_opening_sha256 = _require_lower_hex(
        test_opening_sha256,
        64,
        "test_opening_sha256",
    )
    if (
        raw["schema"] != "h6-raw-endpoint-inventory-v3"
        or raw["readiness_sha256"] != readiness_sha256
        or raw["matching_set_sha256"] != matching_set_sha256
        or raw["git_head"] != git_head
        or raw["dirty_digest"] != dirty_digest
        or raw["endpoint_smc_protocol_sha256"]
        != endpoint_smc_protocol_sha256
        or raw["smc_bias_semantics_sha256"]
        != smc_bias_semantics_sha256
        or raw["data_identity_sha256"] != data_identity_sha256
        or raw["opening_count"] != 1
        or raw["test_opening_sha256"] != test_opening_sha256
        or raw["comparison_endpoint_ids"] != _PREDICTION_COMPARISONS
    ):
        raise ValueError(
            "raw endpoint inventory does not bind the amended readiness, "
            "matching set, source, comparisons, and single opening"
        )
    configs = _validate_prediction_endpoint_configs(
        raw["endpoint_configs"],
        readiness=readiness,
    )
    data_identity = getattr(readiness, "_data_identity", None)
    test_tokens = getattr(data_identity, "test_tokens", None)
    token_count = getattr(test_tokens, "token_count", None)
    if type(token_count) is not int or token_count <= 1:
        raise ValueError(
            "readiness lacks its private validated test-token count"
        )
    expected_counted_targets = token_count - 1
    records = raw["records"]
    expected_keys = tuple(
        (endpoint_id, seed, replicate_id, particle_count)
        for endpoint_id in _PREDICTION_ENDPOINT_IDS
        for seed in _H6_CONFIRMATORY_SEEDS
        for replicate_id in range(ENDPOINT_REPLICATE_COUNT)
        for particle_count in ENDPOINT_PARTICLE_COUNTS
    )
    if type(records) is not list or len(records) != len(expected_keys):
        raise ValueError(
            "raw endpoint inventory must contain the exact complete "
            "endpoint/seed/replicate/particle table"
        )
    observations: dict[tuple[str, int], list[EndpointSmcObservation]] = {
        (endpoint_id, seed): []
        for endpoint_id in _PREDICTION_ENDPOINT_IDS
        for seed in _H6_CONFIRMATORY_SEEDS
    }
    checkpoint_by_group: dict[tuple[str, int], str] = {}
    group_by_checkpoint: dict[str, tuple[str, int]] = {}
    common_stream_by_replicate: dict[int, str] = {}
    counted_targets: set[int] = set()
    for record, expected_key in zip(records, expected_keys, strict=True):
        if (
            type(record) is not dict
            or set(record) != _ENDPOINT_OBSERVATION_FIELDS
        ):
            raise ValueError("raw endpoint observation fields are not exact")
        endpoint_id, seed, replicate_id, particle_count = expected_key
        if (
            (
                record["endpoint_id"],
                record["confirmatory_seed"],
                record["replicate_id"],
                record["particle_count"],
            )
            != expected_key
            or record["endpoint_config_sha256"]
            != configs[endpoint_id].config_sha256
            or record["checkpoint_git_head"] != git_head
            or record["checkpoint_dirty_digest"] != dirty_digest
            or record["test_opening_sha256"] != test_opening_sha256
        ):
            raise ValueError(
                "raw endpoint observations are reordered or contradict their "
                "config/source/seed/opening identities"
            )
        checkpoint_sha256 = _require_lower_hex(
            record["checkpoint_sha256"],
            64,
            "checkpoint_sha256",
        )
        common_stream_sha256 = _require_lower_hex(
            record["common_stream_sha256"],
            64,
            "common_stream_sha256",
        )
        group = (endpoint_id, seed)
        prior_checkpoint = checkpoint_by_group.setdefault(
            group,
            checkpoint_sha256,
        )
        prior_group = group_by_checkpoint.setdefault(
            checkpoint_sha256,
            group,
        )
        if prior_checkpoint != checkpoint_sha256 or prior_group != group:
            raise ValueError(
                "each endpoint/seed must bind one unique terminal checkpoint"
            )
        prior_stream = common_stream_by_replicate.setdefault(
            replicate_id,
            common_stream_sha256,
        )
        if prior_stream != common_stream_sha256:
            raise ValueError(
                "compared endpoints do not share the same replicate stream"
            )
        log_normalizer_sum = record["log_normalizer_sum"]
        negative_log_likelihood_sum = record[
            "negative_log_likelihood_sum"
        ]
        if (
            type(log_normalizer_sum) is not float
            or type(negative_log_likelihood_sum) is not float
            or not math.isfinite(log_normalizer_sum)
            or not math.isfinite(negative_log_likelihood_sum)
            or negative_log_likelihood_sum != -log_normalizer_sum
        ):
            raise ValueError(
                "raw endpoint log-normalizer and NLL totals must be exact "
                "finite opposites"
            )
        _validate_counter_consumption(
            record["counter_consumption"],
            protocol_sha256=endpoint_smc_protocol_sha256,
            endpoint_id=endpoint_id,
            endpoint_config_sha256=configs[endpoint_id].config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            confirmatory_seed=seed,
            replicate_id=replicate_id,
            particle_count=particle_count,
            common_stream_sha256=common_stream_sha256,
        )
        _validate_cache_audit(
            record["cache_audit"],
            endpoint_id=endpoint_id,
            endpoint_config_sha256=configs[endpoint_id].config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            confirmatory_seed=seed,
            replicate_id=replicate_id,
            particle_count=particle_count,
            negative_log_likelihood_sum=negative_log_likelihood_sum,
        )
        _validate_no_endpoint_failure(record["failure"])
        try:
            observation = EndpointSmcObservation(
                checkpoint_sha256=checkpoint_sha256,
                replicate_id=replicate_id,
                particle_count=particle_count,
                common_stream_sha256=common_stream_sha256,
                negative_log_likelihood_sum=negative_log_likelihood_sum,
                counted_targets=record["counted_targets"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("raw endpoint observation is not typed SMC evidence") from exc
        observations[group].append(observation)
        counted_targets.add(observation.counted_targets)
    if (
        len(checkpoint_by_group) != len(observations)
        or len(group_by_checkpoint) != len(observations)
        or len(common_stream_by_replicate) != ENDPOINT_REPLICATE_COUNT
        or len(set(common_stream_by_replicate.values()))
        != ENDPOINT_REPLICATE_COUNT
        or counted_targets != {expected_counted_targets}
    ):
        raise ValueError(
            "raw endpoint checkpoint, stream, or counted-target inventory "
            "is contradictory"
        )

    aggregates: dict[tuple[str, int], EndpointSmcAggregate] = {}
    for group, group_observations in observations.items():
        aggregate = aggregate_endpoint_smc(group_observations)
        if type(aggregate) is not EndpointSmcAggregate:
            raise ValueError(
                "raw endpoint inventory cannot produce a complete finite "
                f"SMC aggregate for {group!r}: "
                f"{getattr(aggregate, 'failure_kinds', ())!r}"
            )
        if aggregate.checkpoint_sha256 != checkpoint_by_group[group]:
            raise ValueError("SMC aggregate checkpoint identity changed")
        aggregates[group] = aggregate

    objective_interval, objective_complete = _paired_q2_interval(
        aggregates=aggregates,
        left_endpoint_id=_A5_COMPLETE_ENDPOINT_ID,
        right_endpoint_id=_A5_EMISSION_ENDPOINT_ID,
    )
    primary_interval, primary_complete = _paired_q2_interval(
        aggregates=aggregates,
        left_endpoint_id=_A0_ENDPOINT_ID,
        right_endpoint_id=_A5_COMPLETE_ENDPOINT_ID,
    )
    return _ValidatedPredictionRawReceipt(
        payload=raw,
        raw_sha256=hashlib.sha256(raw_endpoint_records_bytes).hexdigest(),
        objective_interval=objective_interval,
        primary_interval=primary_interval,
        objective_estimator_complete=objective_complete,
        primary_estimator_complete=primary_complete,
    )


def _prediction_result_payload(
    *,
    result: object,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    status = getattr(getattr(result, "status", None), "value", None)
    obligations = getattr(result, "obligations", None)
    if (
        type(status) is not str
        or type(obligations) is not tuple
        or metrics.get("schema") != "h6-prediction-metrics-v2"
    ):
        raise ValueError("prediction result/metrics are not amended typed records")
    readiness_sha256 = _require_lower_hex(
        getattr(result, "readiness_sha256", None),
        64,
        "prediction result readiness_sha256",
    )
    metrics_sha256 = _require_lower_hex(
        getattr(result, "metrics_sha256", None),
        64,
        "prediction result metrics_sha256",
    )
    smc_bias_semantics_sha256 = _require_lower_hex(
        getattr(result, "smc_bias_semantics_sha256", None),
        64,
        "prediction result smc_bias_semantics_sha256",
    )
    if (
        metrics.get("smc_bias_semantics_sha256")
        != smc_bias_semantics_sha256
    ):
        raise ValueError(
            "prediction result and metrics bind different SMC bias semantics"
        )
    result_identity_sha256 = hashlib.sha256(
        b"vfe4.h6.prediction-result.v2\x00"
        + canonical_json_bytes(
            {
                "readiness_sha256": readiness_sha256,
                "smc_bias_semantics_sha256": (
                    smc_bias_semantics_sha256
                ),
                "metrics_sha256": metrics_sha256,
            }
        )
    ).hexdigest()
    return {
        "schema_version": "h6-prediction-result-v2",
        "gate": "H6-Prediction",
        "status": status,
        "obligations": obligations,
        "readiness_sha256": readiness_sha256,
        "smc_bias_semantics_sha256": smc_bias_semantics_sha256,
        "metrics_sha256": metrics_sha256,
        "result_identity_sha256": result_identity_sha256,
        "objective_gate_spec_sha256": metrics[
            "objective_gate_spec_sha256"
        ],
        "test_opening_sha256": metrics["test_opening_sha256"],
        "raw_endpoint_inventory_sha256": metrics[
            "raw_endpoint_inventory_sha256"
        ],
        "objective_status": metrics["objective_status"],
        "primary_disposition": metrics["primary_disposition"],
        "opening_count": metrics["opening_count"],
        "compute_savings_claim": False,
    }


def publish_h6_prediction_result(
    *,
    artifact_root: Path,
    readiness: object,
    validated_opening: object,
    raw_endpoint_records_bytes: bytes,
) -> tuple[object, Path]:
    """Publish one result whose metrics are derived only from raw SMC bytes."""

    from vfe4.training.h6_readiness import adjudicate_h6_prediction_opening
    from vfe4.data.access import validated_test_opening_identity
    from vfe4.types.h6 import (
        H6PredictionReadinessToken,
    )
    from vfe4.types.results import H6PredictionResult

    if type(readiness) is not H6PredictionReadinessToken:
        raise ValueError("typed amended H6 readiness is required")
    readiness.__post_init__()
    if readiness.readiness_schema != "h6-prediction-readiness-v2":
        raise ValueError("legacy readiness cannot publish amended metrics")
    test_opening_sha256 = _require_lower_hex(
        validated_test_opening_identity(validated_opening),
        64,
        "test_opening_sha256",
    )
    receipt = _validate_prediction_raw_inventory(
        raw_endpoint_records_bytes,
        readiness=readiness,
        test_opening_sha256=test_opening_sha256,
    )
    decision, metrics_bytes = adjudicate_h6_prediction_opening(
        objective_interval=receipt.objective_interval,
        primary_interval=receipt.primary_interval,
        objective_estimator_complete=(
            receipt.objective_estimator_complete
        ),
        primary_estimator_complete=receipt.primary_estimator_complete,
        test_opening_sha256=test_opening_sha256,
        raw_endpoint_inventory_sha256=receipt.raw_sha256,
        opening_count=1,
    )
    result = H6PredictionResult.from_metrics(
        readiness=readiness,
        metrics_bytes=metrics_bytes,
    )
    if getattr(result, "_decision", None) != decision:
        raise ValueError("published metrics do not reproduce the ordered decision")
    metrics = _canonical_json_object_from_bytes(
        metrics_bytes,
        name="H6 Prediction metrics",
    )
    result_payload = _prediction_result_payload(
        result=result,
        metrics=metrics,
    )
    run_directory = publish_run_directory(
        artifact_root,
        (
            "h6-prediction-result-"
            f"{result_payload['result_identity_sha256']}"
        ),
        {
            "raw/h6_endpoint_records.json": receipt.payload,
            "validation/h6_prediction_metrics.json": metrics,
            "validation/h6_prediction_result.json": result_payload,
        },
    )
    return result, run_directory


def read_h6_prediction_result(
    *,
    artifact_root: Path,
    readiness: object,
) -> object:
    """Reconstruct one amended Prediction result from immutable bytes."""

    from vfe4.types.h6 import H6PredictionReadinessToken
    from vfe4.types.results import H6PredictionResult

    if type(readiness) is not H6PredictionReadinessToken:
        raise ValueError("typed amended H6 readiness is required")
    readiness.__post_init__()
    if readiness.readiness_schema != "h6-prediction-readiness-v2":
        raise ValueError("legacy readiness cannot read amended metrics")
    if not isinstance(artifact_root, Path):
        raise ValueError("artifact_root must be pathlib.Path")
    try:
        root = artifact_root.resolve(strict=True)
    except OSError as exc:
        raise ArtifactPublicationError(
            "Prediction artifact root is unavailable"
        ) from exc
    expected_names = (
        "raw/h6_endpoint_records.json",
        "validation/h6_prediction_metrics.json",
        "validation/h6_prediction_result.json",
    )
    manifest_bytes = (root / "manifest.sha256").read_bytes()
    entries = _manifest_entries(manifest_bytes)
    if tuple(name for name, _ in entries) != expected_names:
        raise ArtifactPublicationError(
            "Prediction artifact payload inventory is not exact"
        )
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactPublicationError(
                "Prediction artifact cannot contain a symlink"
            )
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ArtifactPublicationError(
                "Prediction artifact contains a non-file entry"
            )
    if observed_files != {*expected_names, "manifest.sha256"}:
        raise ArtifactPublicationError(
            "Prediction artifact has a missing or unlisted file"
        )
    payloads: dict[str, bytes] = {}
    for name, expected_sha256 in entries:
        path = root / Path(*PurePosixPath(name).parts)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ArtifactPublicationError(
                f"Prediction artifact payload hash differs: {name}"
            )
        _canonical_json_object_from_bytes(payload, name=name)
        payloads[name] = payload
    metrics_bytes = payloads["validation/h6_prediction_metrics.json"]
    metrics = _canonical_json_object_from_bytes(
        metrics_bytes,
        name="H6 Prediction metrics",
    )
    raw_bytes = payloads["raw/h6_endpoint_records.json"]
    receipt = _validate_prediction_raw_inventory(
        raw_bytes,
        readiness=readiness,
        test_opening_sha256=metrics["test_opening_sha256"],
    )
    if receipt.raw_sha256 != metrics["raw_endpoint_inventory_sha256"]:
        raise ArtifactPublicationError(
            "Prediction metrics do not bind the raw endpoint inventory"
        )
    from vfe4.training.h6_readiness import adjudicate_h6_prediction_opening

    derived_decision, derived_metrics_bytes = adjudicate_h6_prediction_opening(
        objective_interval=receipt.objective_interval,
        primary_interval=receipt.primary_interval,
        objective_estimator_complete=(
            receipt.objective_estimator_complete
        ),
        primary_estimator_complete=receipt.primary_estimator_complete,
        test_opening_sha256=metrics["test_opening_sha256"],
        raw_endpoint_inventory_sha256=receipt.raw_sha256,
        opening_count=1,
    )
    if metrics_bytes != derived_metrics_bytes:
        raise ArtifactPublicationError(
            "Prediction metrics are not exactly derivable from raw SMC records"
        )
    result = H6PredictionResult.from_metrics(
        readiness=readiness,
        metrics_bytes=metrics_bytes,
    )
    if getattr(result, "_decision", None) != derived_decision:
        raise ArtifactPublicationError(
            "Prediction result does not retain the raw-derived decision"
        )
    expected_result_bytes = canonical_json_bytes(
        _prediction_result_payload(result=result, metrics=metrics)
    )
    expected_result_payload = _canonical_json_object_from_bytes(
        expected_result_bytes,
        name="derived H6 Prediction result",
    )
    if (
        payloads["validation/h6_prediction_result.json"]
        != expected_result_bytes
    ):
        raise ArtifactPublicationError(
            "Prediction result payload does not reproduce typed metrics"
        )
    if root.name != (
        "h6-prediction-result-"
        f"{expected_result_payload['result_identity_sha256']}"
    ):
        raise ArtifactPublicationError(
            "Prediction artifact directory does not match its full result identity"
        )
    return result


__all__ = [
    "CandidateArtifactReference",
    "ProjectedCurrentCandidateConfig",
    "project_h1_prefix_prior_config",
    "project_h1_prefix_prior_v2_config",
    "project_h6_prefix_config",
    "publish_h6_prediction_result",
    "read_h6_prediction_result",
    "reopen_bounded_prefix_certificate_set",
    "reopen_h6_prefix_authorities",
    "run_projected_current_candidate",
]
