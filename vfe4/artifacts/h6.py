"""Frozen H6 current-candidate projections and verified artifact references."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Literal

from .atomic import ArtifactPublicationError, canonical_json_bytes


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
    operation_key: Literal["h1_prefix_prior", "h6_prefix"],
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
    operation_key: Literal["h1_prefix_prior", "h6_prefix"],
    operation: Literal["H1-Prefix-Prior", "H6-Prefix"],
) -> ProjectedCurrentCandidateConfig:
    selected = _extract_operation_config(
        raw_config,
        operation_key=operation_key,
        operation=operation,
    )
    owned = _owned_config(selected)
    resolved = _resolve_projected_config(operation, owned)
    if getattr(resolved, "operation", None) != operation:
        raise ValueError("resolved projection returned another operation")
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
    )


def project_h6_prefix_config(
    raw_config: Mapping[str, object],
) -> ProjectedCurrentCandidateConfig:
    """Purely select and validate the predecessor-free H6 Prefix config."""

    return _project(
        raw_config,
        operation_key="h6_prefix",
        operation="H6-Prefix",
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


__all__ = [
    "CandidateArtifactReference",
    "ProjectedCurrentCandidateConfig",
    "project_h1_prefix_prior_config",
    "project_h6_prefix_config",
    "run_projected_current_candidate",
]
