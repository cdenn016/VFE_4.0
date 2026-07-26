"""Frozen H6 current-candidate projections and verified artifact references."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Literal

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
        schema_versions=("h6-prefix-config-v1", "h6-prefix-config-v2"),
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
    expected_names = _EXPECTED_PAYLOADS[operation]
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
        if (
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
    "run_projected_current_candidate",
]
