"""Fail-closed H6 Prediction prerequisite loading and readiness publication."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from vfe4.artifacts.atomic import canonical_json_bytes, publish_run_directory
from vfe4.config.schema import H6PredictionResolvedConfig
from vfe4.numerics.critical_values import CRITICAL_VALUES_PROTOCOL_SHA256
from vfe4.training.matching import ARM_MATRIX_ROWS, arm_matrix_sha256
from vfe4.types.h6 import (
    ArmId,
    DataIdentity,
    EvidenceStatus,
    H1PrefixPriorArtifactRef,
    H5UpdateBinding,
    H6PredictionReadinessToken,
    PrefixCaseKey,
    PrefixCertificate,
    PredictionCorrectnessArtifactRef,
    SmcAccuracyArtifactRef,
    canonical_json_bytes as h6_canonical_json_bytes,
    issue_prediction_readiness,
)
from vfe4.types.results import GateStatus, H6PrefixGateResult


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
PREDICTION_READINESS_SOURCE_BLOCKERS = (
    "separate manifest-linked H1/H2/H3/H5 correctness producers are absent",
    "finite-SMC lacks a manifest-linked config/estimator/fixture publisher",
    "H5 does not publish the ten exact update-binding preimages",
    "blinded data does not publish retained typed DataIdentity preimages",
    "arm matching lacks an immutable manifest-linked matching-set publisher",
)


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

    correctness_artifact_roots: tuple[
        tuple[Literal["H1", "H2", "H3", "H5"], Path], ...
    ]
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
            or any(not isinstance(root, Path) for _, root in self.correctness_artifact_roots)
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
            raise ValueError("Prediction prerequisite references must be a string mapping")
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
                raise ValueError("all five named prerequisite roots must be supplied together")
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
                name: _path_from_value(
                    value, repo_root=base, name=name
                )
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
        raise ProducerCompatibilityError("producer manifest is not strict ASCII") from exc
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
        raise ProducerCompatibilityError(f"producer root is unavailable: {root}") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise ProducerCompatibilityError(f"producer root is not a real directory: {root}")
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
        raise ValueError(f"{gate} correctness artifact is stale for the current candidate")
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
    expected_git_head: str,
    expected_dirty_digest: str,
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
            generative_factor_schema_bytes=payloads[
                "schemas/generative_factor.json"
            ],
            config_bytes=payloads["config.json"],
            validation_payload_bytes=payloads["validation/h1_prefix_prior.json"],
        )
    except ValueError as exc:
        raise ProducerCompatibilityError(
            f"H1-prefix-prior producer bytes do not satisfy the typed reference: {exc}"
        ) from exc
    if artifact.status is not GateStatus.PASS:
        raise ValueError("H1-prefix-prior artifact is not PASS")
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
        raise ProducerCompatibilityError(f"Prefix certificate key is invalid: {exc}") from exc


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
            raise ProducerCompatibilityError("Prefix certificate entry is not an object")
        key = _prefix_key(raw_entry.get("key"))
        try:
            status = EvidenceStatus(raw_entry.get("status"))
            obligations_raw = raw_entry.get("obligations")
            validation_payload = raw_entry.get("validation_payload")
            if type(obligations_raw) is not list or type(validation_payload) is not dict:
                raise ValueError("obligations/payload types are not canonical")
            certificate = PrefixCertificate(
                key=key,
                validation_payload_canonical_json=h6_canonical_json_bytes(
                    validation_payload
                ),
                validation_payload_sha256=raw_entry.get(
                    "validation_payload_sha256"
                ),
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
            raise ValueError("every H6-Prefix certificate must be unique, current, and PASS")
        result[key] = certificate
    if (
        H6PrefixGateResult.from_certificates(result).prefix_certificate_set_sha256
        != expected_set_sha256
    ):
        raise ValueError("H6-Prefix certificate-set digest is stale")
    return result


def _load_h5_update_binding(
    root: Path, *, expected_binding_sha256: str
) -> H5UpdateBinding:
    _, _, payloads = _load_manifested_files(
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
    for name in (
        "update_spec_raw_sha256",
        "update_spec_canonical_sha256",
        "objective_schema_sha256",
        "factor_input_schema_sha256",
    ):
        _require_sha256(h5_config.get(name), f"H5 {name}")
    for name in (
        "reference_sha256",
        "recognition_sha256",
        "model_sha256",
        "validation_payload_sha256",
    ):
        _require_sha256(h5_state.get(name), f"H5 {name}")
    _require_sha256(expected_binding_sha256, "h5_update_binding_sha256")
    raise ProducerCompatibilityError(
        "H5 producer schema records digest summaries but does not publish the exact "
        "ten named producer preimages required by "
        "H5UpdateBinding.from_producer_preimages; readiness will not fabricate them"
    )


def _load_blinded_data_identity(root: Path) -> DataIdentity:
    _load_manifested_files(
        root,
        required_paths=("data_identity.json",),
        expected_manifest_sha256=None,
    )
    raise ProducerCompatibilityError(
        "blinded-data producer schema does not serialize the retained typed "
        "EncodedTokenStorageIdentity/ValidationSafetyFixture bytes needed to "
        "reconstruct DataIdentity without decoding sealed corpus members"
    )


def _validate_matching_artifact(root: Path, *, expected_set_sha256: str) -> str:
    _require_sha256(expected_set_sha256, "matching_set_sha256")
    if not root.exists():
        raise ProducerCompatibilityError(
            "matching artifact root is absent; Task 7 has no immutable matching-set "
            "publisher for Prediction readiness"
        )
    raise ProducerCompatibilityError(
        "matching producer schema is undefined: Task 7 exposes typed in-memory "
        "MatchingReport records but no manifest-linked matching-set artifact that "
        "readiness can reproduce"
    )


def _current_source_identity(
    config: H6PredictionResolvedConfig,
) -> tuple[str, str, str]:
    from vfe4.artifacts.provenance import current_source_identity

    return current_source_identity(_REPO_ROOT, config.artifact_root)


def _revalidate_h6_prediction_readiness_inputs(
    *,
    config: H6PredictionResolvedConfig,
    prerequisite_refs: CurrentPredictionPrerequisiteRefs,
) -> H6PredictionReadinessToken:
    """Revalidate inputs without publishing; experiment dispatch uses this seam."""

    _raise_source_blockers()
    if type(config) is not H6PredictionResolvedConfig:
        raise ValueError("config must be an exact H6PredictionResolvedConfig")
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
    from vfe4.data.wikitext2 import ACCESS_POLICY_SHA256

    if config.access_policy_sha256 != ACCESS_POLICY_SHA256:
        raise ValueError("blinded-data access policy differs from the frozen policy")
    if config.critical_values_sha256 != CRITICAL_VALUES_PROTOCOL_SHA256:
        raise ValueError("critical-values protocol hash differs from the frozen literals")
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
        expected_git_head=git_head_value,
        expected_dirty_digest=dirty_digest,
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
    data_identity = _load_blinded_data_identity(
        prerequisite_refs.blinded_data_artifact_root
    )
    if (
        data_identity.data_identity_sha256 != config.data_identity_sha256
        or data_identity.access_policy_sha256 != config.access_policy_sha256
    ):
        raise ValueError("blinded data/access identities differ from Prediction config")
    matching_set_sha256 = _validate_matching_artifact(
        prerequisite_refs.matching_artifact_root,
        expected_set_sha256=config.matching_set_sha256,
    )
    return issue_prediction_readiness(
        git_head=git_head_value,
        dirty_digest=dirty_digest,
        experiment_config_sha256=config.config_sha256,
        correctness_artifacts=correctness,
        h1_prefix_prior_artifact=h1_prefix,
        h5_update_binding=h5_binding,
        h6_training_schedule=config.training_schedule,
        smc_accuracy_artifact=smc,
        critical_values_sha256=config.critical_values_sha256,
        endpoint_smc_protocol=config.endpoint_smc_protocol,
        attribution_matrix_sha256=config.attribution_matrix_sha256,
        matching_set_sha256=matching_set_sha256,
        prefix_certificates=certificates,
        data_identity=data_identity,
    )


def _readiness_payload(token: H6PredictionReadinessToken) -> dict[str, object]:
    return {
        "readiness_schema": token.readiness_schema,
        "git_head": token.git_head,
        "dirty_digest": token.dirty_digest,
        "experiment_config_sha256": token.experiment_config_sha256,
        "correctness_manifests": dict(token.correctness_manifests),
        "h1_prefix_prior_manifest_sha256": (
            token.h1_prefix_prior_manifest_sha256
        ),
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


def _load_published_h6_prediction_readiness(
    *,
    config: H6PredictionResolvedConfig,
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
        raise ValueError("readiness artifact directory name does not match the fresh token")
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
    if (
        payloads["validation/h6_prediction_readiness.json"]
        != expected_readiness
    ):
        raise ValueError(
            "published readiness fields or hashes differ from fresh revalidation"
        )
    fresh.__post_init__()
    return fresh


def validate_h6_prediction_readiness(
    *,
    config: H6PredictionResolvedConfig,
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


__all__ = [
    "CurrentPredictionPrerequisiteRefs",
    "PREDICTION_READINESS_SOURCE_BLOCKERS",
    "ProducerCompatibilityError",
    "validate_h6_prediction_readiness",
]
