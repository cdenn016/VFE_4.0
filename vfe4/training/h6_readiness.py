"""Fail-closed H6 Prediction prerequisite loading and readiness publication."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from vfe4.artifacts.atomic import canonical_json_bytes, publish_run_directory
from vfe4.artifacts.h6 import reopen_h6_prefix_authorities
from vfe4.config.schema import (
    H6PredictionV2ResolvedConfig,
    H6PredictionV3ResolvedConfig,
)
from vfe4.numerics.critical_values import CRITICAL_VALUES_PROTOCOL_SHA256
from vfe4.training.matching import ARM_MATRIX_ROWS, arm_matrix_sha256
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6MatchingSetV3,
    H6_MATCHING_POLICY_V3,
)
from vfe4.types.h6 import (
    H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS,
    A0DirectExactPrefixCertificateV1,
    ArmId,
    BoundedPrefixCertificateSet,
    DataIdentity,
    EvidenceStatus,
    H1PrefixPriorArtifactRef,
    H5UpdateBinding,
    H6PredictionReadinessToken,
    OrderedPredictionDecision,
    PrefixCaseKey,
    PrefixCertificate,
    PredictionCorrectnessArtifactRef,
    SmcAccuracyArtifactRef,
    canonical_json_bytes as h6_canonical_json_bytes,
    issue_prediction_readiness_v2,
)
from vfe4.types.h6_prediction_v3 import (
    H6_CHECKPOINT_CODEC_SHA256,
    H6_COUNTER_MAPPING_SHA256,
    H6_DETERMINISTIC_POLICY_SHA256,
    H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6_SCORING_INVENTORY_SHA256,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
)
from vfe4.types.results import (
    GateStatus,
    H1PrefixPriorV2GateResult,
    H6BoundedPrefixGateResult,
    H6PrefixGateResult,
    InvariantResult,
)

if TYPE_CHECKING:
    from vfe4.artifacts.h6_matching import H6MatchingSetRecord
    from vfe4.evaluation.smc_uncertainty import InflatedPairedInterval


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORRECTNESS_GATES = ("H1", "H2", "H3", "H5")
_REFERENCE_FIELDS = frozenset(
    {
        "correctness_artifact_roots",
        "h1_prefix_prior_artifact_root",
        "smc_accuracy_artifact_root",
        "h6_prefix_artifact_root",
        "blinded_data_artifact_root",
        "matching_artifact_root",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")
_H5_LABELS = (
    "exact_coordinate",
    "generalized_em",
    "natural_gradient_proposal",
)
_H5_INTRINSIC_PREIMAGE_FIELDS = (
    "update_spec_raw_sha256",
    "update_spec_canonical_sha256",
    "objective_schema_sha256",
    "factor_input_schema_sha256",
    "reference_sha256",
    "recognition_state_sha256",
    "model_state_sha256",
    "validation_payload_sha256",
)
PREDICTION_READINESS_SOURCE_BLOCKERS: tuple[()] = ()


class ProducerCompatibilityError(RuntimeError):
    """A current producer cannot supply the bytes required by readiness."""


def _raise_source_blockers() -> None:
    if PREDICTION_READINESS_SOURCE_BLOCKERS:
        raise ProducerCompatibilityError(
            "H6 Prediction readiness is unavailable at this source revision: "
            + "; ".join(PREDICTION_READINESS_SOURCE_BLOCKERS)
        )


def _path_from_value(value: object, *, repo_root: Path, name: str) -> Path:
    if type(value) is str:
        candidate = Path(value)
    elif isinstance(value, Path):
        candidate = value
    else:
        raise ValueError(f"{name} must be a string or exact pathlib.Path")
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


@dataclass(frozen=True)
class CurrentPredictionPrerequisiteRefs:
    """Exact artifact roots consumed by H6 Prediction readiness; H4 is absent."""

    correctness_artifact_roots: tuple[tuple[Literal["H1", "H2", "H3", "H5"], Path], ...]
    h1_prefix_prior_artifact_root: Path
    smc_accuracy_artifact_root: Path
    h6_prefix_artifact_root: Path
    blinded_data_artifact_root: Path
    matching_artifact_root: Path

    def __post_init__(self) -> None:
        if (
            type(self.correctness_artifact_roots) is not tuple
            or tuple(gate for gate, _ in self.correctness_artifact_roots)
            != _CORRECTNESS_GATES
            or any(
                not isinstance(root, Path)
                for _, root in self.correctness_artifact_roots
            )
        ):
            raise ValueError(
                "correctness_artifact_roots must contain exactly H1, H2, H3, H5 "
                "in frozen order; H4 is forbidden"
            )
        for name in (
            "h1_prefix_prior_artifact_root",
            "smc_accuracy_artifact_root",
            "h6_prefix_artifact_root",
            "blinded_data_artifact_root",
            "matching_artifact_root",
        ):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be an exact pathlib.Path")

    @classmethod
    def from_mapping(
        cls,
        references: Mapping[str, object],
        *,
        repo_root: Path | None = None,
        h1_prefix_prior_artifact_root: Path | None = None,
        smc_accuracy_artifact_root: Path | None = None,
        h6_prefix_artifact_root: Path | None = None,
        blinded_data_artifact_root: Path | None = None,
        matching_artifact_root: Path | None = None,
    ) -> "CurrentPredictionPrerequisiteRefs":
        """Own launcher mappings and freeze the four-gate order without H4."""

        if not isinstance(references, Mapping) or any(
            type(key) is not str for key in references
        ):
            raise ValueError(
                "Prediction prerequisite references must be a string mapping"
            )
        base = Path.cwd() if repo_root is None else repo_root
        if not isinstance(base, Path):
            raise ValueError("repo_root must be an exact pathlib.Path")
        base = base.resolve(strict=False)
        explicit = (
            h1_prefix_prior_artifact_root,
            smc_accuracy_artifact_root,
            h6_prefix_artifact_root,
            blinded_data_artifact_root,
            matching_artifact_root,
        )
        if any(value is not None for value in explicit):
            if not all(value is not None for value in explicit):
                raise ValueError(
                    "all five named prerequisite roots must be supplied together"
                )
            correctness: object = references
            named_values = {
                "h1_prefix_prior_artifact_root": explicit[0],
                "smc_accuracy_artifact_root": explicit[1],
                "h6_prefix_artifact_root": explicit[2],
                "blinded_data_artifact_root": explicit[3],
                "matching_artifact_root": explicit[4],
            }
        else:
            if set(references) != _REFERENCE_FIELDS:
                raise ValueError(
                    "prerequisite_refs must contain correctness_artifact_roots and "
                    "the five exact named artifact roots"
                )
            correctness = references["correctness_artifact_roots"]
            named_values = {
                name: references[name]
                for name in _REFERENCE_FIELDS
                if name != "correctness_artifact_roots"
            }
        if not isinstance(correctness, Mapping) or set(correctness) != set(
            _CORRECTNESS_GATES
        ):
            raise ValueError(
                "correctness_artifact_roots must contain exactly H1, H2, H3, H5; "
                "H4 is forbidden"
            )
        ordered = tuple(
            (
                gate,
                _path_from_value(
                    correctness[gate],
                    repo_root=base,
                    name=f"correctness_artifact_roots[{gate}]",
                ),
            )
            for gate in _CORRECTNESS_GATES
        )
        return cls(
            correctness_artifact_roots=ordered,  # type: ignore[arg-type]
            **{
                name: _path_from_value(value, repo_root=base, name=name)
                for name, value in named_values.items()
            },
        )


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _require_git_head(value: object, name: str = "git_head") -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 40-hex")
    return value


def _read_regular_file(path: Path, *, maximum_bytes: int = 33_554_432) -> bytes:
    try:
        stat_result = os.lstat(path)
    except OSError as exc:
        raise ProducerCompatibilityError(
            f"required producer file is unavailable: {path}"
        ) from exc
    if not path.is_file() or path.is_symlink() or stat_result.st_size > maximum_bytes:
        raise ProducerCompatibilityError(
            f"required producer path is not a bounded regular file: {path}"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProducerCompatibilityError(
            f"required producer file is unreadable: {path}"
        ) from exc


def _manifest_inventory(manifest_bytes: bytes) -> dict[str, str]:
    try:
        text = manifest_bytes.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ProducerCompatibilityError(
            "producer manifest is not strict ASCII"
        ) from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise ProducerCompatibilityError("producer manifest is not canonical LF text")
    inventory: dict[str, str] = {}
    names: list[str] = []
    for line in text.splitlines():
        if line.count("  ") != 1:
            raise ProducerCompatibilityError("producer manifest record is malformed")
        digest, name = line.split("  ", 1)
        _require_sha256(digest, "producer manifest digest")
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or path.as_posix() != name
            or any(part in ("", ".", "..") for part in path.parts)
            or name in inventory
        ):
            raise ProducerCompatibilityError(
                "producer manifest contains a noncanonical or duplicate path"
            )
        inventory[name] = digest
        names.append(name)
    if names != sorted(names):
        raise ProducerCompatibilityError("producer manifest paths are not sorted")
    return inventory


def _load_manifested_files(
    root: Path,
    *,
    required_paths: tuple[str, ...],
    expected_manifest_sha256: str | None,
) -> tuple[bytes, dict[str, str], dict[str, bytes]]:
    if not isinstance(root, Path):
        raise ValueError("artifact root must be an exact pathlib.Path")
    if root.is_symlink():
        raise ProducerCompatibilityError(f"producer root cannot be a symlink: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProducerCompatibilityError(
            f"producer root is unavailable: {root}"
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ProducerCompatibilityError(
            f"producer root is not a real directory: {root}"
        )
    manifest = _read_regular_file(resolved / "manifest.sha256", maximum_bytes=65_536)
    observed_manifest = hashlib.sha256(manifest).hexdigest()
    if (
        expected_manifest_sha256 is not None
        and observed_manifest != expected_manifest_sha256
    ):
        raise ValueError("producer manifest SHA-256 differs from the frozen config")
    inventory = _manifest_inventory(manifest)
    payloads: dict[str, bytes] = {}
    for name in required_paths:
        expected = inventory.get(name)
        if expected is None:
            raise ProducerCompatibilityError(
                f"producer manifest does not publish required payload {name}"
            )
        payload = _read_regular_file(resolved / Path(*PurePosixPath(name).parts))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"producer payload hash differs from manifest: {name}")
        payloads[name] = payload
    return manifest, inventory, payloads


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProducerCompatibilityError(f"{name} is not canonical JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ProducerCompatibilityError(f"{name} is not one canonical JSON object")
    return value


def _load_prediction_correctness_artifact(
    *,
    gate: Literal["H1", "H2", "H3", "H5"],
    root: Path,
    expected_manifest_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
) -> PredictionCorrectnessArtifactRef:
    validation_name = f"validation/{gate.lower()}.json"
    manifest, inventory, payloads = _load_manifested_files(
        root,
        required_paths=("config.json", "provenance.json", validation_name),
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if "validation/h4.json" in inventory:
        raise ProducerCompatibilityError(
            f"{gate} correctness artifact includes validation/h4.json; H4 cannot "
            "become a Prediction prerequisite through a shared manifest"
        )
    provenance = _json_object(payloads["provenance.json"], name="provenance.json")
    git_head = _require_git_head(provenance.get("git_head"), "producer git_head")
    dirty_digest = _require_sha256(
        provenance.get("dirty_digest"), "producer dirty_digest"
    )
    if git_head != expected_git_head or dirty_digest != expected_dirty_digest:
        raise ValueError(
            f"{gate} correctness artifact is stale for the current candidate"
        )
    validation = _json_object(payloads[validation_name], name=validation_name)
    direct_fields = {
        "gate",
        "git_head",
        "dirty_digest",
        "config_sha256",
        "status",
        "obligations",
    }
    if not direct_fields.issubset(validation):
        raise ProducerCompatibilityError(
            f"{gate} correctness producer schema cannot satisfy "
            "PredictionCorrectnessArtifactRef: its validation payload lacks direct "
            "gate/git_head/dirty_digest/config_sha256/status/obligations fields"
        )
    try:
        artifact = PredictionCorrectnessArtifactRef.from_bytes(
            gate=gate,
            artifact_path=Path(validation_name),
            manifest_bytes=manifest,
            git_head=git_head,
            dirty_digest=dirty_digest,
            config_bytes=payloads["config.json"],
            validation_payload_bytes=payloads[validation_name],
        )
    except ValueError as exc:
        raise ProducerCompatibilityError(
            f"{gate} correctness producer bytes do not satisfy the typed reference: {exc}"
        ) from exc
    if artifact.status is not GateStatus.PASS:
        raise ValueError(f"{gate} correctness artifact is not PASS")
    return artifact


def _load_h1_prefix_prior_artifact(
    *,
    root: Path,
    expected_manifest_sha256: str,
    expected_generative_factor_schema_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
    expected_source_sha256: str,
) -> H1PrefixPriorArtifactRef:
    manifest, _, payloads = _load_manifested_files(
        root,
        required_paths=(
            "config.json",
            "schemas/generative_factor.json",
            "validation/h1_prefix_prior.json",
        ),
        expected_manifest_sha256=expected_manifest_sha256,
    )
    try:
        artifact = H1PrefixPriorArtifactRef.from_bytes(
            artifact_path=Path("validation/h1_prefix_prior.json"),
            manifest_bytes=manifest,
            git_head=expected_git_head,
            dirty_digest=expected_dirty_digest,
            generative_factor_schema_bytes=payloads["schemas/generative_factor.json"],
            config_bytes=payloads["config.json"],
            validation_payload_bytes=payloads["validation/h1_prefix_prior.json"],
        )
    except ValueError as exc:
        raise ProducerCompatibilityError(
            f"H1-prefix-prior producer bytes do not satisfy the typed reference: {exc}"
        ) from exc
    from vfe4.config import resolve_h1_prefix_prior_v2_config

    try:
        raw_config = _json_object(
            payloads["config.json"],
            name="H1-prefix-prior config.json",
        )
        scorer_config = resolve_h1_prefix_prior_v2_config(
            raw_config,
            repo_root=_REPO_ROOT,
        )
        validation = _json_object(
            payloads["validation/h1_prefix_prior.json"],
            name="H1-prefix-prior validation",
        )
    except (TypeError, ValueError) as exc:
        raise ProducerCompatibilityError(
            "H1-prefix-prior artifact is not an exact scorer-v2 producer"
        ) from exc
    expected_validation_fields = {
        "schema_version",
        "gate",
        "status",
        "obligations",
        "git_head",
        "dirty_digest",
        "source_sha256",
        "config_sha256",
        "junit_sha256",
        "fixture_id",
        "fixture_sha256",
        "base_fixture_sha256",
        "generative_factor_schema_sha256",
        "scorer_schema",
        "latent_projection_policy",
        "parent_history_policy",
        "invariants",
        "computation",
    }
    if set(validation) != expected_validation_fields:
        raise ProducerCompatibilityError(
            "H1-prefix-prior validation inventory is not exact scorer-v2"
        )
    raw_obligations = validation.get("obligations")
    raw_invariants = validation.get("invariants")
    raw_junit_sha256 = validation.get("junit_sha256")
    if (
        type(raw_obligations) is not list
        or type(raw_invariants) is not list
        or (
            raw_junit_sha256 is not None
            and (
                type(raw_junit_sha256) is not str
                or len(raw_junit_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in raw_junit_sha256
                )
            )
        )
    ):
        raise ProducerCompatibilityError(
            "H1-prefix-prior obligations, invariants, or JUnit identity are malformed"
        )
    try:
        typed_invariants = tuple(
            InvariantResult(
                name=item["name"],
                passed=item["passed"],
                value=item["value"],
                limit=item["limit"],
                detail=item["detail"],
            )
            for item in raw_invariants
            if type(item) is dict
            and set(item) == {"name", "passed", "value", "limit", "detail"}
        )
        if len(typed_invariants) != len(raw_invariants):
            raise ValueError("invariant record inventory is malformed")
        typed_result = H1PrefixPriorV2GateResult(
            gate=validation.get("gate"),
            status=GateStatus(validation.get("status")),
            fixture_id=validation.get("fixture_id"),
            scorer_schema=validation.get("scorer_schema"),
            fixture_sha256=validation.get("fixture_sha256"),
            generative_factor_schema_sha256=validation.get(
                "generative_factor_schema_sha256"
            ),
            invariants=typed_invariants,
            obligations=tuple(raw_obligations),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProducerCompatibilityError(
            "H1-prefix-prior validation does not reconstruct as scorer-v2"
        ) from exc
    computation = validation.get("computation")
    if (
        type(computation) is not dict
        or set(computation)
        != {
            "production_priors",
            "independent_priors",
            "production_complete_objectives",
            "independent_complete_objectives",
        }
        or any(type(computation[name]) is not dict for name in computation)
        or set(computation["production_priors"])
        != {"active", "swapped", "target_suffix_a", "target_suffix_b"}
        or set(computation["independent_priors"]) != {"active", "swapped"}
        or set(computation["production_complete_objectives"]) != {"active", "swapped"}
        or set(computation["independent_complete_objectives"]) != {"active", "swapped"}
    ):
        raise ProducerCompatibilityError(
            "H1-prefix-prior scorer-v2 computation inventory is incomplete"
        )
    if (
        payloads["config.json"] != scorer_config.canonical_json.encode("utf-8")
        or validation.get("schema_version") != "h1-prefix-prior-validation-v3"
        or validation.get("fixture_id") != "h1-prefix-prior-scorer-v2"
        or validation.get("scorer_schema")
        != "parent-specific-pooled-prefix-bilinear-v1"
        or validation.get("latent_projection_policy") != "nonzero_bank_projections"
        or validation.get("parent_history_policy") != "active_swapped_distinct_nonzero"
        or scorer_config.source.git_head != expected_git_head
        or scorer_config.source.dirty_digest != expected_dirty_digest
        or scorer_config.source.source_sha256 != expected_source_sha256
        or validation.get("source_sha256") != expected_source_sha256
        or validation.get("source_sha256") != scorer_config.source.source_sha256
        or validation.get("config_sha256") != scorer_config.config_sha256
        or validation.get("fixture_sha256") != scorer_config.fixture_sha256
        or validation.get("base_fixture_sha256") != scorer_config.base_fixture_sha256
        or typed_result.status is not GateStatus.PASS
        or typed_result.fixture_sha256 != scorer_config.fixture_sha256
        or scorer_config.generative_factor_schema_sha256
        != expected_generative_factor_schema_sha256
    ):
        raise ProducerCompatibilityError(
            "H1-prefix-prior artifact does not carry the exact scorer-v2 "
            "fixture, policies, schema, and source identity"
        )
    if artifact.status is not GateStatus.PASS:
        raise ValueError("H1-prefix-prior artifact is not PASS")
    if (
        artifact.generative_factor_schema_sha256
        != expected_generative_factor_schema_sha256
    ):
        raise ValueError(
            "H1-prefix-prior artifact does not bind the configured scorer-v2 schema"
        )
    return artifact


def _load_smc_accuracy_artifact(
    *,
    root: Path,
    expected_manifest_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
) -> SmcAccuracyArtifactRef:
    required = (
        "config.json",
        "protocol/estimator.json",
        "fixtures/finite_smc.json",
        "validation/h6_smc_accuracy.json",
    )
    manifest, _, payloads = _load_manifested_files(
        root,
        required_paths=required,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    try:
        artifact = SmcAccuracyArtifactRef.from_bytes(
            artifact_path=Path("validation/h6_smc_accuracy.json"),
            manifest_bytes=manifest,
            git_head=expected_git_head,
            dirty_digest=expected_dirty_digest,
            estimator_preimage_bytes=payloads["protocol/estimator.json"],
            fixture_set_bytes=payloads["fixtures/finite_smc.json"],
            config_bytes=payloads["config.json"],
            validation_payload_bytes=payloads["validation/h6_smc_accuracy.json"],
        )
    except ValueError as exc:
        raise ProducerCompatibilityError(
            f"finite-SMC producer bytes do not satisfy the typed reference: {exc}"
        ) from exc
    if artifact.status is not GateStatus.PASS:
        raise ValueError("finite-SMC accuracy artifact is not PASS")
    return artifact


def _prefix_key(raw: object) -> PrefixCaseKey:
    if type(raw) is not dict or set(raw) != {
        "arm",
        "predictor_config_sha256",
        "estimator_sha256",
        "model_family_sha256",
        "vocabulary_sha256",
        "data_safety_sha256",
        "git_head",
        "dirty_digest",
    }:
        raise ProducerCompatibilityError("Prefix certificate key schema is not exact")
    try:
        return PrefixCaseKey(
            ArmId(raw["arm"]),
            raw["predictor_config_sha256"],
            raw["estimator_sha256"],
            raw["model_family_sha256"],
            raw["vocabulary_sha256"],
            raw["data_safety_sha256"],
            raw["git_head"],
            raw["dirty_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise ProducerCompatibilityError(
            f"Prefix certificate key is invalid: {exc}"
        ) from exc


def _load_prefix_certificates(
    *,
    root: Path,
    expected_set_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
) -> dict[PrefixCaseKey, PrefixCertificate]:
    _, _, payloads = _load_manifested_files(
        root,
        required_paths=(
            "certificates/prefix_set.json",
            "provenance.json",
            "validation/h6_prefix.json",
        ),
        expected_manifest_sha256=None,
    )
    provenance = _json_object(payloads["provenance.json"], name="Prefix provenance")
    if (
        provenance.get("git_head") != expected_git_head
        or provenance.get("dirty_digest") != expected_dirty_digest
    ):
        raise ValueError("H6-Prefix artifact is stale for the current candidate")
    certificate_set = _json_object(
        payloads["certificates/prefix_set.json"], name="Prefix certificate set"
    )
    if (
        certificate_set.get("schema_version") != "h6-prefix-certificate-set-v1"
        or certificate_set.get("prefix_certificate_set_sha256") != expected_set_sha256
        or type(certificate_set.get("certificates")) is not list
        or not certificate_set["certificates"]
    ):
        raise ProducerCompatibilityError(
            "H6-Prefix producer did not publish the exact nonempty certificate set"
        )
    result: dict[PrefixCaseKey, PrefixCertificate] = {}
    for raw_entry in certificate_set["certificates"]:
        if type(raw_entry) is not dict:
            raise ProducerCompatibilityError(
                "Prefix certificate entry is not an object"
            )
        key = _prefix_key(raw_entry.get("key"))
        try:
            status = EvidenceStatus(raw_entry.get("status"))
            obligations_raw = raw_entry.get("obligations")
            validation_payload = raw_entry.get("validation_payload")
            if (
                type(obligations_raw) is not list
                or type(validation_payload) is not dict
            ):
                raise ValueError("obligations/payload types are not canonical")
            certificate = PrefixCertificate(
                key=key,
                validation_payload_canonical_json=h6_canonical_json_bytes(
                    validation_payload
                ),
                validation_payload_sha256=raw_entry.get("validation_payload_sha256"),
                status=status,
                obligations=tuple(obligations_raw),
                certificate_sha256=raw_entry.get("certificate_sha256"),
            )
        except (TypeError, ValueError) as exc:
            raise ProducerCompatibilityError(
                f"Prefix certificate bytes are invalid: {exc}"
            ) from exc
        if (
            certificate.status is not EvidenceStatus.PASS
            or key.git_head != expected_git_head
            or key.dirty_digest != expected_dirty_digest
            or key in result
        ):
            raise ValueError(
                "every H6-Prefix certificate must be unique, current, and PASS"
            )
        result[key] = certificate
    if (
        H6PrefixGateResult.from_certificates(result).prefix_certificate_set_sha256
        != expected_set_sha256
    ):
        raise ValueError("H6-Prefix certificate-set digest is stale")
    return result


def _reopen_h6_prefix_authorities_v3(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_junit_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
    expected_source_sha256: str,
    expected_set_sha256: str,
    expected_direct_certificate_sha256: str,
) -> tuple[
    BoundedPrefixCertificateSet,
    A0DirectExactPrefixCertificateV1,
]:
    """Reopen both exact current PASS Prefix authorities."""

    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("H6-Prefix artifact root must be an absolute Path")
    canonical_root = root.resolve(strict=False)
    if canonical_root.as_posix() != root.as_posix():
        raise ValueError("H6-Prefix artifact root must be canonical")
    certificate_set, direct_certificate = reopen_h6_prefix_authorities(
        canonical_root,
        expected_manifest_sha256=_require_sha256(
            expected_manifest_sha256,
            "h6_prefix_manifest_sha256",
        ),
        expected_git_head=_require_git_head(expected_git_head),
        expected_dirty_digest=_require_sha256(
            expected_dirty_digest,
            "dirty_digest",
        ),
        expected_junit_sha256=_require_sha256(
            expected_junit_sha256,
            "h6_prefix_junit_sha256",
        ),
    )
    if type(certificate_set) is not BoundedPrefixCertificateSet:
        raise ValueError(
            "H6-Prefix reopener did not return an exact bounded certificate set"
        )
    certificate_set.__post_init__()
    if (
        type(direct_certificate)
        is not A0DirectExactPrefixCertificateV1
    ):
        raise ValueError(
            "H6-Prefix reopener did not return an exact direct-A0 certificate"
        )
    direct_certificate.__post_init__()
    gate = H6BoundedPrefixGateResult.from_certificate_set(certificate_set)
    if (
        gate.status is not GateStatus.PASS
        or gate.obligations != ()
        or certificate_set.git_head != expected_git_head
        or certificate_set.dirty_digest != expected_dirty_digest
        or certificate_set.source_sha256 != expected_source_sha256
        or certificate_set.prefix_certificate_set_sha256
        != expected_set_sha256
    ):
        raise ValueError(
            "bounded H6-Prefix certificate set is not exact current PASS"
        )
    if (
        direct_certificate.status is not EvidenceStatus.PASS
        or direct_certificate.obligations != ()
        or tuple(direct_certificate.checks)
        != H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
        or not all(direct_certificate.checks.values())
        or direct_certificate.git_head != expected_git_head
        or direct_certificate.dirty_digest != expected_dirty_digest
        or direct_certificate.source_sha256 != expected_source_sha256
        or direct_certificate.certificate_sha256
        != expected_direct_certificate_sha256
        or direct_certificate.bounded_a0_certificate_sha256
        not in tuple(
            certificate.certificate_sha256
            for certificate in certificate_set.certificates
        )
    ):
        raise ValueError(
            "direct-A0 Prefix certificate is not exact current PASS"
        )
    return certificate_set, direct_certificate


def read_h6_prefix_authorities_for_scoring_v3(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_junit_sha256: str,
    readiness: H6PredictionV3ReadinessToken,
) -> tuple[
    BoundedPrefixCertificateSet,
    A0DirectExactPrefixCertificateV1,
]:
    """Reopen both exact Prefix authorities authorized by v3 readiness."""

    if type(readiness) is not H6PredictionV3ReadinessToken:
        raise ValueError("H6-Prefix scoring requires exact v3 readiness")
    readiness.__post_init__()
    if readiness.status != "PASS":
        raise ValueError("H6-Prefix scoring requires PASS v3 readiness")
    expected_source_sha256 = hashlib.sha256(
        b"VFE4-H6-SOURCE-CANDIDATE-V1\x00"
        + bytes.fromhex(readiness.git_head)
        + bytes.fromhex(readiness.dirty_digest)
    ).hexdigest()
    return _reopen_h6_prefix_authorities_v3(
        root,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_junit_sha256=expected_junit_sha256,
        expected_git_head=readiness.git_head,
        expected_dirty_digest=readiness.dirty_digest,
        expected_source_sha256=expected_source_sha256,
        expected_set_sha256=readiness.prefix_certificate_set_sha256,
        expected_direct_certificate_sha256=(
            readiness.a0_direct_exact_prefix_certificate_sha256
        ),
    )


def read_h6_bounded_prefix_certificate_set_for_scoring_v3(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_junit_sha256: str,
    readiness: H6PredictionV3ReadinessToken,
) -> BoundedPrefixCertificateSet:
    """Compatibility projection after reopening both v3 Prefix authorities."""

    certificate_set, _direct_certificate = (
        read_h6_prefix_authorities_for_scoring_v3(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_junit_sha256=expected_junit_sha256,
            readiness=readiness,
        )
    )
    return certificate_set


def _load_h5_update_binding(
    root: Path, *, expected_binding_sha256: str
) -> H5UpdateBinding:
    manifest, _, payloads = _load_manifested_files(
        root,
        required_paths=("provenance.json", "validation/h5.json"),
        expected_manifest_sha256=None,
    )
    provenance = _json_object(payloads["provenance.json"], name="H5 provenance")
    h5_config = provenance.get("h5_config")
    h5_state = provenance.get("h5_state_hashes")
    if type(h5_config) is not dict or type(h5_state) is not dict:
        raise ProducerCompatibilityError(
            "H5 producer does not publish its field/state digest inventories"
        )
    labels = h5_config.get("enabled_update_labels")
    if type(labels) is not list or tuple(labels) != _H5_LABELS:
        raise ValueError("H5 enabled_update_labels differ from the three actual labels")
    config_digest_names = (
        "update_spec_raw_sha256",
        "update_spec_canonical_sha256",
        "objective_schema_sha256",
        "factor_input_schema_sha256",
    )
    state_digest_names = (
        "reference_sha256",
        "recognition_sha256",
        "model_sha256",
        "validation_payload_sha256",
    )
    config_digests = {
        name: _require_sha256(h5_config.get(name), f"H5 {name}")
        for name in config_digest_names
    }
    state_digests = {
        name: _require_sha256(h5_state.get(name), f"H5 {name}")
        for name in state_digest_names
    }
    encoded_preimages = provenance.get("h5_update_binding_preimages")
    if (
        type(encoded_preimages) is not dict
        or set(encoded_preimages) != {"schema_version", "encoding", "preimages"}
        or encoded_preimages.get("schema_version") != "h5-update-binding-preimages-v1"
        or encoded_preimages.get("encoding") != "hex"
    ):
        raise ProducerCompatibilityError(
            "H5 update-binding preimage envelope is not the exact v1 hex schema"
        )
    encoded_values = encoded_preimages.get("preimages")
    if type(encoded_values) is not dict or set(encoded_values) != set(
        _H5_INTRINSIC_PREIMAGE_FIELDS
    ):
        raise ProducerCompatibilityError(
            "H5 update-binding preimages do not equal the eight intrinsic fields"
        )
    intrinsic_preimages: dict[str, bytes] = {}
    for name in _H5_INTRINSIC_PREIMAGE_FIELDS:
        encoded = encoded_values[name]
        if (
            type(encoded) is not str
            or not encoded
            or len(encoded) % 2 != 0
            or any(character not in _LOWER_HEX for character in encoded)
        ):
            raise ProducerCompatibilityError(
                f"H5 update-binding preimage {name} is not canonical lowercase hex"
            )
        intrinsic_preimages[name] = bytes.fromhex(encoded)
    summary_digests = {
        **config_digests,
        "reference_sha256": state_digests["reference_sha256"],
        "recognition_state_sha256": state_digests["recognition_sha256"],
        "model_state_sha256": state_digests["model_sha256"],
        "validation_payload_sha256": state_digests["validation_payload_sha256"],
    }
    for name in _H5_INTRINSIC_PREIMAGE_FIELDS:
        if (
            hashlib.sha256(intrinsic_preimages[name]).hexdigest()
            != summary_digests[name]
        ):
            raise ValueError(
                f"H5 update-binding preimage does not match digest summary: {name}"
            )
    validation = _json_object(payloads["validation/h5.json"], name="validation/h5.json")
    producer_validation = validation.get("producer_validation")
    if type(producer_validation) is not dict:
        raise ProducerCompatibilityError(
            "H5 correctness artifact does not contain its producer validation"
        )
    validation_payload_sha256 = _require_sha256(
        producer_validation.get("payload_sha256"),
        "H5 producer validation payload_sha256",
    )
    if validation_payload_sha256 != state_digests["validation_payload_sha256"]:
        raise ValueError(
            "H5 validation payload digest differs from provenance state hashes"
        )
    _require_sha256(expected_binding_sha256, "h5_update_binding_sha256")
    producer_preimages = {
        "h5_manifest_sha256": manifest,
        "h5_payload_sha256": payloads["validation/h5.json"],
        **intrinsic_preimages,
    }
    binding = H5UpdateBinding.from_producer_preimages(
        producer_preimages=producer_preimages,
        enabled_update_labels=tuple(labels),
    )
    binding.verify_producer_preimages(producer_preimages)
    for name, expected_digest in summary_digests.items():
        if getattr(binding, name) != expected_digest:
            raise ValueError(
                f"H5 update-binding digest/name mapping is inconsistent: {name}"
            )
    if binding.binding_sha256 != expected_binding_sha256:
        raise ValueError(
            "H5 update-binding SHA-256 differs from the frozen Prediction config"
        )
    return binding


@dataclass(frozen=True, slots=True)
class H6PredictionV3PrerequisiteEvidence:
    """Mechanically reopened producer bytes required before v3 token issuance."""

    correctness_artifacts: tuple[PredictionCorrectnessArtifactRef, ...]
    h1_prefix_prior_artifact: H1PrefixPriorArtifactRef
    smc_accuracy_artifact: SmcAccuracyArtifactRef
    h5_update_binding: H5UpdateBinding
    bounded_prefix_certificate_set: BoundedPrefixCertificateSet
    a0_direct_exact_prefix_certificate: A0DirectExactPrefixCertificateV1

    def __post_init__(self) -> None:
        if (
            type(self.correctness_artifacts) is not tuple
            or tuple(item.gate for item in self.correctness_artifacts)
            != _CORRECTNESS_GATES
            or any(
                type(item) is not PredictionCorrectnessArtifactRef
                for item in self.correctness_artifacts
            )
        ):
            raise ValueError(
                "v3 prerequisite evidence requires exact H1, H2, H3, H5 "
                "correctness artifacts in frozen order"
            )
        for artifact in self.correctness_artifacts:
            artifact.__post_init__()
            if artifact.status is not GateStatus.PASS:
                raise ValueError(
                    "v3 prerequisite correctness artifacts must all be PASS"
                )
        if type(self.h1_prefix_prior_artifact) is not H1PrefixPriorArtifactRef:
            raise ValueError(
                "v3 prerequisite evidence requires exact H1-Prefix-Prior bytes"
            )
        self.h1_prefix_prior_artifact.__post_init__()
        if self.h1_prefix_prior_artifact.status is not GateStatus.PASS:
            raise ValueError("H1-Prefix-Prior prerequisite must be PASS")
        if type(self.smc_accuracy_artifact) is not SmcAccuracyArtifactRef:
            raise ValueError(
                "v3 prerequisite evidence requires exact finite-SMC bytes"
            )
        self.smc_accuracy_artifact.__post_init__()
        if self.smc_accuracy_artifact.status is not GateStatus.PASS:
            raise ValueError("finite-SMC prerequisite must be PASS")
        if type(self.h5_update_binding) is not H5UpdateBinding:
            raise ValueError(
                "v3 prerequisite evidence requires an exact H5 update binding"
            )
        self.h5_update_binding.__post_init__()
        if (
            type(self.bounded_prefix_certificate_set)
            is not BoundedPrefixCertificateSet
        ):
            raise ValueError(
                "v3 prerequisite evidence requires an exact bounded Prefix set"
            )
        self.bounded_prefix_certificate_set.__post_init__()
        prefix_gate = H6BoundedPrefixGateResult.from_certificate_set(
            self.bounded_prefix_certificate_set
        )
        if (
            prefix_gate.status is not GateStatus.PASS
            or prefix_gate.obligations != ()
        ):
            raise ValueError("bounded Prefix prerequisite must be exact PASS")
        if (
            type(self.a0_direct_exact_prefix_certificate)
            is not A0DirectExactPrefixCertificateV1
        ):
            raise ValueError(
                "v3 prerequisite evidence requires an exact direct-A0 "
                "Prefix certificate"
            )
        direct = self.a0_direct_exact_prefix_certificate
        direct.__post_init__()
        if (
            direct.status is not EvidenceStatus.PASS
            or direct.obligations != ()
            or tuple(direct.checks)
            != H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
            or not all(direct.checks.values())
            or direct.bounded_a0_certificate_sha256
            not in tuple(
                certificate.certificate_sha256
                for certificate in self.bounded_prefix_certificate_set.certificates
            )
        ):
            raise ValueError(
                "direct-A0 Prefix prerequisite must be exact PASS"
            )


def _validate_h6_prediction_v3_prerequisite_evidence(
    *,
    config: H6PredictionV3ResolvedConfig,
    evidence: H6PredictionV3PrerequisiteEvidence,
) -> None:
    if type(evidence) is not H6PredictionV3PrerequisiteEvidence:
        raise ValueError(
            "readiness v3 requires exact mechanically reopened prerequisite evidence"
        )
    evidence.__post_init__()
    expected_correctness = dict(config.correctness_manifests)
    if (
        tuple(expected_correctness) != _CORRECTNESS_GATES
        or any(
            artifact.manifest_sha256
            != expected_correctness[artifact.gate]
            or artifact.git_head != config.source.git_head
            or artifact.dirty_digest != config.source.dirty_digest
            for artifact in evidence.correctness_artifacts
        )
    ):
        raise ValueError(
            "reopened correctness artifacts differ from v3 config/source"
        )
    h1_prefix = evidence.h1_prefix_prior_artifact
    if (
        h1_prefix.manifest_sha256
        != config.h1_prefix_prior_manifest_sha256
        or h1_prefix.generative_factor_schema_sha256
        != config.h1_prefix_prior_generative_factor_schema_sha256
        or h1_prefix.git_head != config.source.git_head
        or h1_prefix.dirty_digest != config.source.dirty_digest
    ):
        raise ValueError(
            "reopened H1-Prefix-Prior artifact differs from v3 config/source"
        )
    smc = evidence.smc_accuracy_artifact
    if (
        smc.manifest_sha256 != config.smc_validation_manifest_sha256
        or smc.git_head != config.source.git_head
        or smc.dirty_digest != config.source.dirty_digest
    ):
        raise ValueError(
            "reopened finite-SMC artifact differs from v3 config/source"
        )
    if (
        evidence.h5_update_binding.binding_sha256
        != config.h5_update_binding_sha256
    ):
        raise ValueError("reopened H5 update binding differs from v3 config")
    prefix_set = evidence.bounded_prefix_certificate_set
    if (
        prefix_set.prefix_certificate_set_sha256
        != config.prefix_certificate_set_sha256
        or prefix_set.git_head != config.source.git_head
        or prefix_set.dirty_digest != config.source.dirty_digest
        or prefix_set.source_sha256 != config.source.source_sha256
    ):
        raise ValueError(
            "reopened bounded Prefix set differs from v3 config/source"
        )
    direct = evidence.a0_direct_exact_prefix_certificate
    if (
        direct.certificate_sha256
        != config.a0_direct_exact_prefix_certificate_sha256
        or direct.git_head != config.source.git_head
        or direct.dirty_digest != config.source.dirty_digest
        or direct.source_sha256 != config.source.source_sha256
    ):
        raise ValueError(
            "reopened direct-A0 Prefix certificate differs from v3 "
            "config/source"
        )


def reopen_h6_prediction_v3_prerequisite_evidence(
    *,
    config: H6PredictionV3ResolvedConfig,
    correctness_artifact_roots: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path], ...
    ],
    h1_prefix_prior_artifact_root: Path,
    smc_accuracy_artifact_root: Path,
    h6_prefix_artifact_root: Path,
    h6_prefix_manifest_sha256: str,
    h6_prefix_junit_sha256: str,
) -> H6PredictionV3PrerequisiteEvidence:
    """Reopen all producer artifacts required before a v3 readiness mint."""

    if type(config) is not H6PredictionV3ResolvedConfig:
        raise ValueError("v3 prerequisite reopening requires exact config")
    if (
        type(correctness_artifact_roots) is not tuple
        or tuple(gate for gate, _ in correctness_artifact_roots)
        != _CORRECTNESS_GATES
        or any(not isinstance(root, Path) for _, root in correctness_artifact_roots)
    ):
        raise ValueError(
            "correctness roots must contain exact H1, H2, H3, H5 paths"
        )
    correctness_roots = dict(correctness_artifact_roots)
    configured_manifests = dict(config.correctness_manifests)
    correctness = tuple(
        _load_prediction_correctness_artifact(
            gate=gate,  # type: ignore[arg-type]
            root=correctness_roots[gate],
            expected_manifest_sha256=configured_manifests[gate],
            expected_git_head=config.source.git_head,
            expected_dirty_digest=config.source.dirty_digest,
        )
        for gate in _CORRECTNESS_GATES
    )
    h1_prefix = _load_h1_prefix_prior_artifact(
        root=h1_prefix_prior_artifact_root,
        expected_manifest_sha256=config.h1_prefix_prior_manifest_sha256,
        expected_generative_factor_schema_sha256=(
            config.h1_prefix_prior_generative_factor_schema_sha256
        ),
        expected_git_head=config.source.git_head,
        expected_dirty_digest=config.source.dirty_digest,
        expected_source_sha256=config.source.source_sha256,
    )
    smc = _load_smc_accuracy_artifact(
        root=smc_accuracy_artifact_root,
        expected_manifest_sha256=config.smc_validation_manifest_sha256,
        expected_git_head=config.source.git_head,
        expected_dirty_digest=config.source.dirty_digest,
    )
    h5_binding = _load_h5_update_binding(
        correctness_roots["H5"],
        expected_binding_sha256=config.h5_update_binding_sha256,
    )
    prefix_set, direct_certificate = _reopen_h6_prefix_authorities_v3(
        h6_prefix_artifact_root,
        expected_manifest_sha256=h6_prefix_manifest_sha256,
        expected_junit_sha256=h6_prefix_junit_sha256,
        expected_git_head=config.source.git_head,
        expected_dirty_digest=config.source.dirty_digest,
        expected_source_sha256=config.source.source_sha256,
        expected_set_sha256=config.prefix_certificate_set_sha256,
        expected_direct_certificate_sha256=(
            config.a0_direct_exact_prefix_certificate_sha256
        ),
    )
    evidence = H6PredictionV3PrerequisiteEvidence(
        correctness_artifacts=correctness,
        h1_prefix_prior_artifact=h1_prefix,
        smc_accuracy_artifact=smc,
        h5_update_binding=h5_binding,
        bounded_prefix_certificate_set=prefix_set,
        a0_direct_exact_prefix_certificate=direct_certificate,
    )
    _validate_h6_prediction_v3_prerequisite_evidence(
        config=config,
        evidence=evidence,
    )
    return evidence


def _load_blinded_data_identity(
    root: Path,
    *,
    expected_archive_sha256: str,
    expected_data_identity_sha256: str,
    expected_access_policy_sha256: str,
) -> DataIdentity:
    from vfe4.data.access import (
        _revalidate_blinded_data_identity_for_readiness,
    )

    return _revalidate_blinded_data_identity_for_readiness(
        root,
        expected_archive_sha256=expected_archive_sha256,
        expected_data_identity_sha256=expected_data_identity_sha256,
        expected_access_policy_sha256=expected_access_policy_sha256,
    )


def _validate_matching_artifact(
    root: Path,
    *,
    expected_set_sha256: str,
    expected_git_head: str,
    expected_dirty_digest: str,
) -> "H6MatchingSetRecord":
    _require_sha256(expected_set_sha256, "matching_set_sha256")
    _require_git_head(expected_git_head, "matching artifact git_head")
    _require_sha256(expected_dirty_digest, "matching artifact dirty_digest")
    from vfe4.artifacts.h6_matching import read_h6_matching_set

    try:
        record = read_h6_matching_set(
            root,
            expected_set_sha256=expected_set_sha256,
            expected_git_head=expected_git_head,
            expected_dirty_digest=expected_dirty_digest,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ProducerCompatibilityError(
            "H6 matching artifact is not an eligible exact v2 evidence set"
        ) from exc
    if record.matching_set_sha256 != expected_set_sha256:
        raise ValueError("matching artifact differs from Prediction config")
    if (
        record.git_head != expected_git_head
        or record.dirty_digest != expected_dirty_digest
    ):
        raise ValueError("matching artifact source differs from the current candidate")
    if record.status != "ELIGIBLE" or record.obligations:
        raise ProducerCompatibilityError(
            "H6 matching artifact remains INCONCLUSIVE and cannot authorize "
            "Prediction readiness"
        )
    if record.authorizing_matching_report_ids != ("PRIMARY",):
        raise ProducerCompatibilityError(
            "H6 matching artifact lacks the sole exact eligible PRIMARY "
            "authorization report"
        )
    selected_primary = record.primary_selection.selected_candidate
    primary_records = tuple(
        item for item in record.matrix_reports if item.row.row_id == "PRIMARY"
    )
    if selected_primary is None or len(primary_records) != 1:
        raise ProducerCompatibilityError(
            "eligible matching artifact must bind one selected PRIMARY "
            "candidate and report"
        )
    primary_record = primary_records[0]
    objective_records = tuple(
        item for item in record.matrix_reports if item.row.row_id == "OBJECTIVE"
    )
    if len(objective_records) != 1:
        raise ProducerCompatibilityError(
            "eligible matching artifact must contain exactly one OBJECTIVE row"
        )
    objective_record = objective_records[0]
    inventories = {
        item.config.config_id: item.config for item in record.ownership_inventories
    }
    primary_left = inventories.get(primary_record.row.left_config_id)
    primary_right = inventories.get(primary_record.row.right_config_id)
    objective_left = inventories.get(objective_record.row.left_config_id)
    objective_right = inventories.get(objective_record.row.right_config_id)
    if (
        primary_left is None
        or primary_right is None
        or objective_left is None
        or objective_right is None
    ):
        raise ProducerCompatibilityError(
            "PRIMARY/OBJECTIVE rows do not bind their literal endpoint configs"
        )
    expected_primary_allocation = (
        selected_primary.emission_width,
        selected_primary.latent_width,
        selected_primary.recognition_width,
        selected_primary.prior_context_width,
    )
    observed_primary_allocation = (
        primary_right.capacity_allocation.emission_width,
        primary_right.capacity_allocation.latent_width,
        primary_right.capacity_allocation.recognition_width,
        primary_right.capacity_allocation.prior_context_width,
    )
    if (
        not primary_record.matched_claim_authorized
        or not primary_record.report.eligible
        or primary_record.report.obligations
        or primary_left != record.primary_matching_config.a0_config
        or primary_record.report.reference_config_sha256 != primary_left.config_sha256
        or primary_record.report.endpoint_config_sha256 != primary_right.config_sha256
        or primary_record.report.reference_parameter_count
        != selected_primary.a0_parameter_count
        or primary_record.report.endpoint_parameter_count
        != selected_primary.a5_parameter_count
        or primary_record.report.reference_training_flops
        != selected_primary.a0_training_flops
        or primary_record.report.endpoint_training_flops
        != selected_primary.a5_training_flops
        or observed_primary_allocation != expected_primary_allocation
        or primary_right.prior_variant != "parent_specific_pooled_prefix"
        or primary_right.objective_kind != "complete_elbo"
    ):
        raise ProducerCompatibilityError(
            "PRIMARY report and endpoint ownership do not equal the exact "
            "eligible joint selection"
        )
    semantic_differences = tuple(
        name
        for name, value in objective_left.semantic_payload().items()
        if objective_right.semantic_payload()[name] != value
    )
    if (
        objective_record.matched_claim_authorized
        or objective_left != primary_right
        or objective_left.prior_variant != objective_right.prior_variant
        or objective_left.prior_variant != "parent_specific_pooled_prefix"
        or objective_left.capacity_allocation != objective_right.capacity_allocation
        or objective_left.capacity_allocation != primary_right.capacity_allocation
        or objective_left.objective_kind != "complete_elbo"
        or objective_right.objective_kind != "emission_only_ablation_non_elbo"
        or semantic_differences != ("objective_kind",)
        or objective_record.report.reference_config_sha256
        != objective_left.config_sha256
        or objective_record.report.endpoint_config_sha256
        != objective_right.config_sha256
    ):
        raise ProducerCompatibilityError(
            "OBJECTIVE endpoints must share parent-specific prior and capacity "
            "from the selected A5 and differ only by complete versus "
            "emission-only objective_kind"
        )
    return record


def _current_source_identity(
    config: H6PredictionV2ResolvedConfig,
) -> tuple[str, str, str]:
    from vfe4.artifacts.provenance import current_source_identity

    return current_source_identity(_REPO_ROOT, config.artifact_root)


def _revalidate_h6_prediction_readiness_inputs(
    *,
    config: H6PredictionV2ResolvedConfig,
    prerequisite_refs: CurrentPredictionPrerequisiteRefs,
) -> H6PredictionReadinessToken:
    """Revalidate inputs without publishing; experiment dispatch uses this seam."""

    _raise_source_blockers()
    if type(config) is not H6PredictionV2ResolvedConfig:
        raise ValueError(
            "amended readiness requires an exact H6PredictionV2ResolvedConfig; "
            "legacy v1 cannot authorize H6"
        )
    from vfe4.config import resolve_h6_prediction_v2_config

    try:
        raw_config = json.loads(config.canonical_json)
        canonical_config = resolve_h6_prediction_v2_config(
            raw_config,
            repo_root=_REPO_ROOT,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("amended H6 Prediction config is not canonical v2") from exc
    if (
        canonical_config != config
        or hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
        != config.config_sha256
    ):
        raise ValueError("amended H6 Prediction typed fields differ from canonical v2")
    if type(prerequisite_refs) is not CurrentPredictionPrerequisiteRefs:
        raise ValueError(
            "prerequisite_refs must be exact CurrentPredictionPrerequisiteRefs"
        )
    prerequisite_refs.__post_init__()
    if tuple(gate for gate, _ in config.correctness_manifests) != _CORRECTNESS_GATES:
        raise ValueError("Prediction config must require exactly H1, H2, H3, H5")
    git_head_value = _require_git_head(config.source.git_head)
    dirty_digest = _require_sha256(config.source.dirty_digest, "dirty_digest")
    observed_source = _current_source_identity(config)
    configured_source = (
        git_head_value,
        dirty_digest,
        _require_sha256(config.source.source_sha256, "source_sha256"),
    )
    if observed_source != configured_source:
        raise ValueError("Prediction config is stale for the live source candidate")
    config.training_schedule.__post_init__()
    config.endpoint_smc_protocol.__post_init__()
    from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS

    if config.smc_bias_semantics_sha256 != SMC_BIAS_SEMANTICS.semantics_sha256:
        raise ValueError(
            "SMC bias semantics hash differs from the frozen production semantics"
        )
    from vfe4.data.wikitext2 import ACCESS_POLICY_SHA256

    if config.access_policy_sha256 != ACCESS_POLICY_SHA256:
        raise ValueError("blinded-data access policy differs from the frozen policy")
    if config.critical_values_sha256 != CRITICAL_VALUES_PROTOCOL_SHA256:
        raise ValueError(
            "critical-values protocol hash differs from the frozen literals"
        )
    if config.attribution_matrix_sha256 != arm_matrix_sha256(ARM_MATRIX_ROWS):
        raise ValueError("attribution-matrix hash differs from the literal eight rows")

    roots = dict(prerequisite_refs.correctness_artifact_roots)
    correctness = tuple(
        _load_prediction_correctness_artifact(
            gate=gate,  # type: ignore[arg-type]
            root=roots[gate],
            expected_manifest_sha256=dict(config.correctness_manifests)[gate],
            expected_git_head=git_head_value,
            expected_dirty_digest=dirty_digest,
        )
        for gate in _CORRECTNESS_GATES
    )
    h1_prefix = _load_h1_prefix_prior_artifact(
        root=prerequisite_refs.h1_prefix_prior_artifact_root,
        expected_manifest_sha256=config.h1_prefix_prior_manifest_sha256,
        expected_generative_factor_schema_sha256=(
            config.h1_prefix_prior_generative_factor_schema_sha256
        ),
        expected_git_head=git_head_value,
        expected_dirty_digest=dirty_digest,
        expected_source_sha256=config.source.source_sha256,
    )
    smc = _load_smc_accuracy_artifact(
        root=prerequisite_refs.smc_accuracy_artifact_root,
        expected_manifest_sha256=config.smc_validation_manifest_sha256,
        expected_git_head=git_head_value,
        expected_dirty_digest=dirty_digest,
    )
    h5_binding = _load_h5_update_binding(
        roots["H5"], expected_binding_sha256=config.h5_update_binding_sha256
    )
    certificates = _load_prefix_certificates(
        root=prerequisite_refs.h6_prefix_artifact_root,
        expected_set_sha256=config.prefix_certificate_set_sha256,
        expected_git_head=git_head_value,
        expected_dirty_digest=dirty_digest,
    )
    observed_archive = config.data.observed_archive
    if observed_archive is None:
        raise ProducerCompatibilityError(
            "Prediction readiness requires the frozen observed WikiText-2 archive identity"
        )
    data_identity = _load_blinded_data_identity(
        prerequisite_refs.blinded_data_artifact_root,
        expected_archive_sha256=observed_archive.archive_sha256,
        expected_data_identity_sha256=config.data_identity_sha256,
        expected_access_policy_sha256=config.access_policy_sha256,
    )
    if (
        data_identity.data_identity_sha256 != config.data_identity_sha256
        or data_identity.access_policy_sha256 != config.access_policy_sha256
    ):
        raise ValueError("blinded data/access identities differ from Prediction config")
    matching_set = _validate_matching_artifact(
        prerequisite_refs.matching_artifact_root,
        expected_set_sha256=config.matching_set_sha256,
        expected_git_head=git_head_value,
        expected_dirty_digest=dirty_digest,
    )
    return issue_prediction_readiness_v2(
        git_head=git_head_value,
        dirty_digest=dirty_digest,
        experiment_config_sha256=config.config_sha256,
        correctness_artifacts=correctness,
        h1_prefix_prior_artifact=h1_prefix,
        h1_prefix_prior_generative_factor_schema_sha256=(
            config.h1_prefix_prior_generative_factor_schema_sha256
        ),
        smc_bias_semantics_sha256=config.smc_bias_semantics_sha256,
        objective_gate_spec=config.objective_gate,
        h5_update_binding=h5_binding,
        h6_training_schedule=config.training_schedule,
        smc_accuracy_artifact=smc,
        critical_values_sha256=config.critical_values_sha256,
        endpoint_smc_protocol=config.endpoint_smc_protocol,
        attribution_matrix_sha256=config.attribution_matrix_sha256,
        matching_set_sha256=matching_set.matching_set_sha256,
        prefix_certificates=certificates,
        data_identity=data_identity,
        matching_set=matching_set,
    )


def _readiness_payload(token: H6PredictionReadinessToken) -> dict[str, object]:
    payload: dict[str, object] = {
        "readiness_schema": token.readiness_schema,
        "git_head": token.git_head,
        "dirty_digest": token.dirty_digest,
        "experiment_config_sha256": token.experiment_config_sha256,
        "correctness_manifests": dict(token.correctness_manifests),
        "h1_prefix_prior_manifest_sha256": (token.h1_prefix_prior_manifest_sha256),
        "h5_update_binding_sha256": token.h5_update_binding_sha256,
        "h6_training_schedule_sha256": token.h6_training_schedule_sha256,
        "smc_validation_manifest_sha256": token.smc_validation_manifest_sha256,
        "critical_values_sha256": token.critical_values_sha256,
        "endpoint_smc_protocol_sha256": token.endpoint_smc_protocol_sha256,
        "attribution_matrix_sha256": token.attribution_matrix_sha256,
        "matching_set_sha256": token.matching_set_sha256,
        "prefix_certificate_set_sha256": token.prefix_certificate_set_sha256,
        "data_identity_sha256": token.data_identity_sha256,
        "access_policy_sha256": token.access_policy_sha256,
        "readiness_sha256": token.readiness_sha256,
        "status": token.status,
    }
    if token.readiness_schema == "h6-prediction-readiness-v2":
        payload.update(
            {
                "h1_prefix_prior_generative_factor_schema_sha256": (
                    token.h1_prefix_prior_generative_factor_schema_sha256
                ),
                "smc_bias_semantics_sha256": (token.smc_bias_semantics_sha256),
                "objective_gate_spec_sha256": (token.objective_gate_spec_sha256),
            }
        )
    return payload


def _load_published_h6_prediction_readiness(
    *,
    config: H6PredictionV2ResolvedConfig,
    prerequisite_refs: CurrentPredictionPrerequisiteRefs,
    artifact_root: Path,
) -> H6PredictionReadinessToken:
    """Revalidate and bind an already-published, separately authorized token."""

    if not isinstance(artifact_root, Path):
        raise ValueError("artifact_root must be a pathlib.Path")
    fresh = _revalidate_h6_prediction_readiness_inputs(
        config=config,
        prerequisite_refs=prerequisite_refs,
    )
    expected_name = f"h6-prediction-readiness-{fresh.readiness_sha256[:16]}"
    if artifact_root.name != expected_name:
        raise ValueError(
            "readiness artifact directory name does not match the fresh token"
        )
    _, inventory, payloads = _load_manifested_files(
        artifact_root,
        required_paths=(
            "config.json",
            "validation/h6_prediction_readiness.json",
        ),
        expected_manifest_sha256=None,
    )
    if set(inventory) != {
        "config.json",
        "validation/h6_prediction_readiness.json",
    }:
        raise ValueError("readiness manifest must bind exactly its two frozen payloads")
    resolved_root = artifact_root.resolve(strict=True)
    observed_files: set[str] = set()
    for path in resolved_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("readiness artifact cannot contain a symlink")
        if path.is_file():
            observed_files.add(path.relative_to(resolved_root).as_posix())
        elif not path.is_dir():
            raise ValueError("readiness artifact contains a non-file entry")
    if observed_files != {
        "config.json",
        "manifest.sha256",
        "validation/h6_prediction_readiness.json",
    }:
        raise ValueError("readiness artifact has a missing or unlisted file")
    expected_config = canonical_json_bytes(json.loads(config.canonical_json))
    if payloads["config.json"] != expected_config:
        raise ValueError("published readiness config differs from the current config")
    expected_readiness = canonical_json_bytes(_readiness_payload(fresh))
    if payloads["validation/h6_prediction_readiness.json"] != expected_readiness:
        raise ValueError(
            "published readiness fields or hashes differ from fresh revalidation"
        )
    fresh.__post_init__()
    return fresh


def validate_h6_prediction_readiness(
    *,
    config: H6PredictionV2ResolvedConfig,
    prerequisite_refs: CurrentPredictionPrerequisiteRefs,
) -> tuple[H6PredictionReadinessToken, Path]:
    """Issue and atomically publish readiness only after exact typed revalidation."""

    token = _revalidate_h6_prediction_readiness_inputs(
        config=config,
        prerequisite_refs=prerequisite_refs,
    )
    token.__post_init__()
    run_directory = publish_run_directory(
        config.artifact_root,
        f"h6-prediction-readiness-{token.readiness_sha256[:16]}",
        {
            "config.json": json.loads(config.canonical_json),
            "validation/h6_prediction_readiness.json": _readiness_payload(token),
        },
    )
    return token, run_directory


def adjudicate_h6_prediction_opening(
    *,
    objective_interval: "InflatedPairedInterval",
    primary_interval: "InflatedPairedInterval",
    objective_estimator_complete: bool,
    primary_estimator_complete: bool,
    test_opening_sha256: str,
    raw_endpoint_inventory_sha256: str,
    opening_count: int,
) -> tuple[OrderedPredictionDecision, bytes]:
    """Derive canonical v2 metrics from one already-scored test opening."""

    from vfe4.evaluation.smc_uncertainty import (
        H6_OBJECTIVE_GATE_SPEC,
        SMC_BIAS_SEMANTICS,
        decide_ordered_prediction,
    )

    decision = decide_ordered_prediction(
        objective_interval=objective_interval,
        primary_interval=primary_interval,
        objective_estimator_complete=objective_estimator_complete,
        primary_estimator_complete=primary_estimator_complete,
        test_opening_sha256=test_opening_sha256,
        raw_endpoint_inventory_sha256=raw_endpoint_inventory_sha256,
        opening_count=opening_count,
        objective_gate_spec=H6_OBJECTIVE_GATE_SPEC,
    )
    objective_lower, objective_upper = decision.objective.objective_interval
    primary_payload = (
        None
        if decision.primary_interval is None
        else {
            "lower": decision.primary_interval[0],
            "upper": decision.primary_interval[1],
        }
    )
    metrics_bytes = canonical_json_bytes(
        {
            "schema": "h6-prediction-metrics-v2",
            "objective_gate_spec_sha256": (decision.objective_gate_spec_sha256),
            "smc_bias_semantics_sha256": (SMC_BIAS_SEMANTICS.semantics_sha256),
            "opening_policy": H6_OBJECTIVE_GATE_SPEC.opening_policy,
            "evaluation_order": H6_OBJECTIVE_GATE_SPEC.evaluation_order,
            "opening_count": decision.opening_count,
            "test_opening_sha256": decision.test_opening_sha256,
            "raw_endpoint_inventory_sha256": (decision.raw_endpoint_inventory_sha256),
            "objective_estimator_complete": (decision.objective.estimator_complete),
            "objective_interval_eligible": (decision.objective.interval_eligible),
            "objective_interval": {
                "lower": objective_lower,
                "upper": objective_upper,
            },
            "objective_status": decision.objective.status.value,
            "primary_estimator_complete": (decision.primary_estimator_complete),
            "primary_interval_eligible": (decision.primary_interval_eligible),
            "primary_interval": primary_payload,
            "primary_disposition": decision.primary_disposition,
        }
    )
    return decision, metrics_bytes


def _derive_h6_prediction_readiness_v3(
    *,
    config: H6PredictionV3ResolvedConfig,
    matching_set: H6MatchingSetV3,
    git_head: str,
    dirty_digest: str,
) -> H6PredictionV3ReadinessToken:
    """Issue the executable token only from exact already-validated v3 inputs.

    Artifact loading remains owned by the historical prerequisite loaders.
    This final typed seam refuses legacy configs/matching sets and binds every
    prerequisite identity carried by the resolved v3 configuration.
    """

    if type(config) is not H6PredictionV3ResolvedConfig:
        raise ValueError("readiness v3 requires an exact H6PredictionV3ResolvedConfig")
    if type(matching_set) is not H6MatchingSetV3:
        raise ValueError(
            "readiness v3 requires an exact H6MatchingSetV3; v2 is forbidden"
        )
    from vfe4.config import resolve_h6_prediction_v3_config

    try:
        raw_config = json.loads(config.canonical_json)
        canonical_config = resolve_h6_prediction_v3_config(
            raw_config,
            repo_root=_REPO_ROOT,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("readiness v3 config is not exact canonical v3") from exc
    if (
        canonical_config != config
        or hashlib.sha256(config.canonical_json.encode("utf-8")).hexdigest()
        != config.config_sha256
    ):
        raise ValueError("readiness v3 config is not exact canonical v3")
    if type(config.recognition_estimator) is not H6RecognitionEstimatorSpec:
        raise ValueError("readiness v3 requires the exact recognition estimator")
    if type(config.runtime) is not H6PredictionRuntimeIdentity:
        raise ValueError("readiness v3 requires the exact runtime identity")
    if type(config.training_schedule) is not H6TrainingScheduleV3:
        raise ValueError("readiness v3 requires the exact training schedule")
    config.recognition_estimator.__post_init__()
    config.runtime.__post_init__()
    config.training_schedule.__post_init__()
    config.endpoint_smc_protocol.__post_init__()
    config.objective_gate.__post_init__()
    matching_set.__post_init__()
    # Component selectors are fully validated and digest-bound by the matching
    # set above, but only PRIMARY is an authorization gate for readiness.
    if (
        matching_set.status != "ELIGIBLE"
        or matching_set.primary_selection.status != "ELIGIBLE"
        or matching_set.primary_selection.selected_candidate is None
        or not matching_set.matrix_reports[0].report.eligible
    ):
        raise ValueError(
            "readiness v3 requires an exact ELIGIBLE first-lexicographic "
            "PRIMARY selection; INCONCLUSIVE hard gates fail closed"
        )
    if (
        config.schema_version != "h6-prediction-config-v3"
        or config.operation != "H6-Prediction"
        or config.source.git_head != git_head
        or config.source.dirty_digest != dirty_digest
    ):
        raise ValueError("readiness source does not match the resolved config")
    if (
        config.training_schedule.recognition_estimator_sha256
        != config.recognition_estimator.estimator_sha256
    ):
        raise ValueError(
            "readiness recognition estimator identity differs from schedule"
        )
    if (
        config.training_schedule.runtime_identity_sha256
        != config.runtime.runtime_identity_sha256
    ):
        raise ValueError("readiness runtime identity differs from schedule")
    if (
        config.counter_mapping_sha256 != H6_COUNTER_MAPPING_SHA256
        or config.training_schedule.counter_mapping_sha256
        != config.counter_mapping_sha256
    ):
        raise ValueError("readiness counter mapping identity is stale")
    if (
        config.phase_ownership_sha256 != H6_PHASE_OWNERSHIP_SHA256
        or config.training_schedule.phase_ownership_sha256
        != config.phase_ownership_sha256
    ):
        raise ValueError("readiness phase ownership identity is stale")
    if (
        config.checkpoint_codec_sha256 != H6_CHECKPOINT_CODEC_SHA256
        or config.training_schedule.checkpoint_codec_sha256
        != config.checkpoint_codec_sha256
    ):
        raise ValueError("readiness checkpoint codec identity is stale")
    if config.runtime.deterministic_policy_sha256 != H6_DETERMINISTIC_POLICY_SHA256:
        raise ValueError("readiness deterministic policy identity is stale")
    if (
        config.scoring_inventory_sha256 != H6_SCORING_INVENTORY_SHA256
        or config.expected_test_row_count != 4104
    ):
        raise ValueError("readiness scoring inventory identity is stale")
    if (
        matching_set.git_head != git_head
        or matching_set.dirty_digest != dirty_digest
        or matching_set.matching_policy_sha256 != H6_MATCHING_POLICY_V3.policy_sha256
        or config.matching_policy_sha256 != matching_set.matching_policy_sha256
        or config.matching_set_schema != "h6-amended-matching-set-v3"
        or config.matching_set_sha256 != matching_set.matching_set_sha256
    ):
        raise ValueError("v3 matching identities do not authorize readiness")
    return H6PredictionV3ReadinessToken.create(
        git_head=git_head,
        dirty_digest=dirty_digest,
        experiment_config_sha256=config.config_sha256,
        correctness_manifests=config.correctness_manifests,
        h1_prefix_prior_manifest_sha256=(config.h1_prefix_prior_manifest_sha256),
        h1_prefix_prior_generative_factor_schema_sha256=(
            config.h1_prefix_prior_generative_factor_schema_sha256
        ),
        smc_bias_semantics_sha256=config.smc_bias_semantics_sha256,
        smc_validation_manifest_sha256=(config.smc_validation_manifest_sha256),
        prefix_certificate_set_sha256=(config.prefix_certificate_set_sha256),
        a0_direct_exact_prefix_certificate_sha256=(
            config.a0_direct_exact_prefix_certificate_sha256
        ),
        h5_update_binding_sha256=config.h5_update_binding_sha256,
        critical_values_sha256=config.critical_values_sha256,
        endpoint_smc_protocol_sha256=(config.endpoint_smc_protocol.protocol_sha256),
        attribution_matrix_sha256=config.attribution_matrix_sha256,
        objective_gate_spec_sha256=config.objective_gate.spec_sha256,
        matching_policy_sha256=config.matching_policy_sha256,
        matching_set_sha256=config.matching_set_sha256,
        training_schedule_sha256=config.training_schedule.schedule_sha256,
        recognition_estimator_sha256=(config.recognition_estimator.estimator_sha256),
        runtime_identity_sha256=config.runtime.runtime_identity_sha256,
        counter_mapping_sha256=config.counter_mapping_sha256,
        phase_ownership_sha256=config.phase_ownership_sha256,
        objective_manifest_schema_sha256=(H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256),
        data_identity_sha256=config.data_identity_sha256,
        access_policy_sha256=config.access_policy_sha256,
    )


def validate_h6_prediction_readiness_v3(
    *,
    config: H6PredictionV3ResolvedConfig,
    matching_set: H6MatchingSetV3,
    git_head: str,
    dirty_digest: str,
    prerequisite_evidence: H6PredictionV3PrerequisiteEvidence,
) -> H6PredictionV3ReadinessToken:
    """Issue v3 readiness only after every producer artifact was reopened."""

    if type(config) is not H6PredictionV3ResolvedConfig:
        raise ValueError("readiness v3 requires an exact H6PredictionV3ResolvedConfig")
    _validate_h6_prediction_v3_prerequisite_evidence(
        config=config,
        evidence=prerequisite_evidence,
    )
    direct_certificate = (
        prerequisite_evidence.a0_direct_exact_prefix_certificate
    )
    if (
        direct_certificate.endpoint_config
        != matching_set.endpoint_configs[0]
        or direct_certificate.endpoint_config.config_id
        != H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
    ):
        raise ValueError(
            "direct-A0 Prefix certificate differs from the exact matched "
            "A0 endpoint"
        )
    return _derive_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=git_head,
        dirty_digest=dirty_digest,
    )


def validate_existing_h6_prediction_readiness_v3(
    *,
    config: H6PredictionV3ResolvedConfig,
    matching_set: H6MatchingSetV3,
    readiness: H6PredictionV3ReadinessToken,
) -> H6PredictionV3ReadinessToken:
    """Validate a retained token intrinsically without claiming producer reopen."""

    if type(readiness) is not H6PredictionV3ReadinessToken:
        raise ValueError("existing readiness must be an exact v3 token")
    readiness.__post_init__()
    expected = _derive_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=config.source.git_head,
        dirty_digest=config.source.dirty_digest,
    )
    if readiness != expected:
        raise ValueError("existing v3 readiness differs from intrinsic authority")
    return readiness


__all__ = [
    "adjudicate_h6_prediction_opening",
    "CurrentPredictionPrerequisiteRefs",
    "H6PredictionV3PrerequisiteEvidence",
    "PREDICTION_READINESS_SOURCE_BLOCKERS",
    "ProducerCompatibilityError",
    "read_h6_bounded_prefix_certificate_set_for_scoring_v3",
    "read_h6_prefix_authorities_for_scoring_v3",
    "reopen_h6_prediction_v3_prerequisite_evidence",
    "validate_existing_h6_prediction_readiness_v3",
    "validate_h6_prediction_readiness",
    "validate_h6_prediction_readiness_v3",
]
