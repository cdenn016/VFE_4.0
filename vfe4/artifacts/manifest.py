"""Closed, canonical, recursive artifact-manifest validation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from .durability import canonical_json_bytes_generic


_ARTIFACT_INTEGRITY_DOMAIN = b"vfe4.artifact-integrity-record.v1\0"
_CLOSED_MANIFEST_IDENTITY_DOMAIN = b"vfe4.closed-manifest-identity.v1\0"
_MANIFEST_KEYS = frozenset({"entries", "schema_version"})
_ENTRY_KEYS = frozenset({"kind", "relative_path", "sha256", "size_bytes"})
_DEFAULT_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_INVALID_WINDOWS_CHARACTERS = frozenset('<>:"|?*')
_RESERVED_WINDOWS_STEMS = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
_RESERVED_WINDOWS_STEMS.update(f"COM{index}" for index in range(1, 10))
_RESERVED_WINDOWS_STEMS.update(f"LPT{index}" for index in range(1, 10))


class IntegrityValidationError(RuntimeError):
    """An artifact or its closed manifest failed fail-closed validation."""


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes_generic(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactIntegrityRecord:
    """Content identity for one validated regular nonlink artifact."""

    schema_version: Literal["vfe4-artifact-integrity-v1"]
    kind: Literal["file", "manifest"]
    relative_path: str
    size_bytes: int
    sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "vfe4-artifact-integrity-v1":
            raise IntegrityValidationError("artifact integrity schema is unsupported")
        if self.kind not in ("file", "manifest"):
            raise IntegrityValidationError("artifact integrity kind is unsupported")
        _canonical_relative_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise IntegrityValidationError(
                "artifact integrity size_bytes must be a nonnegative exact int"
            )
        if not _valid_sha256(self.sha256):
            raise IntegrityValidationError(
                "artifact integrity sha256 must be lowercase hexadecimal"
            )
        body = {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        expected = _domain_hash(_ARTIFACT_INTEGRITY_DOMAIN, body)
        if self.record_sha256 != expected:
            raise IntegrityValidationError(
                "artifact integrity record_sha256 does not match its body"
            )

    @classmethod
    def create(
        cls,
        *,
        kind: Literal["file", "manifest"],
        relative_path: str,
        size_bytes: int,
        sha256: str,
    ) -> ArtifactIntegrityRecord:
        body = {
            "kind": kind,
            "relative_path": relative_path,
            "schema_version": "vfe4-artifact-integrity-v1",
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
        return cls(
            schema_version="vfe4-artifact-integrity-v1",
            kind=kind,
            relative_path=relative_path,
            size_bytes=size_bytes,
            sha256=sha256,
            record_sha256=_domain_hash(_ARTIFACT_INTEGRITY_DOMAIN, body),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ClosedManifestIdentity:
    """Typed validation result for one complete recursive manifest."""

    schema_version: Literal["vfe4-closed-manifest-identity-v1"]
    manifest: ArtifactIntegrityRecord
    entries: tuple[ArtifactIntegrityRecord, ...]
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest: ArtifactIntegrityRecord,
        entries: tuple[ArtifactIntegrityRecord, ...],
    ) -> ClosedManifestIdentity:
        body = {
            "entries": entries,
            "manifest": manifest,
            "schema_version": "vfe4-closed-manifest-identity-v1",
        }
        return cls(
            schema_version="vfe4-closed-manifest-identity-v1",
            manifest=manifest,
            entries=entries,
            identity_sha256=_domain_hash(_CLOSED_MANIFEST_IDENTITY_DOMAIN, body),
        )


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _canonical_component(component: str) -> None:
    if (
        type(component) is not str
        or not component
        or unicodedata.normalize("NFC", component) != component
        or component in (".", "..")
        or component.endswith((".", " "))
        or any(
            ord(character) < 32 or character in _INVALID_WINDOWS_CHARACTERS
            for character in component
        )
        or component.split(".", 1)[0].upper() in _RESERVED_WINDOWS_STEMS
    ):
        raise IntegrityValidationError(
            "manifest relative_path has a nonportable component"
        )


def _canonical_relative_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise IntegrityValidationError(
            "manifest relative_path must be a nonempty canonical POSIX path"
        )
    if unicodedata.normalize("NFC", value) != value:
        raise IntegrityValidationError(
            "manifest relative_path must use canonical Unicode spelling"
        )
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or path.as_posix() != value
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise IntegrityValidationError(
            "manifest relative_path is noncanonical or escapes its manifest"
        )
    for component in path.parts:
        _canonical_component(component)
    return path


def _regular_status(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise IntegrityValidationError(
            f"artifact metadata could not be read: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise IntegrityValidationError(
            f"artifact must be a regular nonlink file: {path}"
        )
    return status


def _regular_directory(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise IntegrityValidationError(
            f"manifest root metadata could not be read: {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise IntegrityValidationError(
            f"manifest root must be a regular nonlink directory: {path}"
        )
    return status


def _sha256_regular_file(path: Path, size_bytes: int) -> tuple[str, bytes]:
    """Hash one already-size-checked regular file through an exact descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IntegrityValidationError(
            f"artifact could not be opened for hashing: {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != size_bytes:
            raise IntegrityValidationError(
                f"artifact changed before hashing: {path}"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
            observed_size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = _regular_status(path)
    if (
        observed_size != size_bytes
        or opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
        or opened.st_size != current.st_size
        or opened.st_mtime_ns != current.st_mtime_ns
    ):
        raise IntegrityValidationError(
            f"artifact changed while hashing: {path}"
        )
    return digest.hexdigest(), b"".join(chunks)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise IntegrityValidationError(
            f"artifact escapes the closed manifest root: {path}"
        ) from exc


def _validated_record(
    path: Path,
    *,
    root: Path,
    kind: Literal["file", "manifest"],
    expected_size: int | None,
    expected_sha256: str | None,
    max_size: int | None = None,
) -> tuple[ArtifactIntegrityRecord, bytes]:
    status = _regular_status(path)
    if max_size is not None and status.st_size > max_size:
        raise IntegrityValidationError(
            f"manifest exceeds its maximum byte size: {path}"
        )
    if expected_size is not None and status.st_size != expected_size:
        raise IntegrityValidationError(
            f"artifact size mismatch before hashing: {path}"
        )
    digest, payload = _sha256_regular_file(path, status.st_size)
    if expected_sha256 is not None and digest != expected_sha256:
        raise IntegrityValidationError(f"artifact SHA-256 mismatch: {path}")
    relative_path = _relative_to_root(path, root)
    return (
        ArtifactIntegrityRecord.create(
            kind=kind,
            relative_path=relative_path,
            size_bytes=status.st_size,
            sha256=digest,
        ),
        payload,
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityValidationError(
                f"manifest contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _parse_manifest(payload: bytes) -> tuple[dict[str, object], ...]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        document = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except IntegrityValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityValidationError(f"manifest JSON is invalid: {exc}") from exc
    if type(document) is not dict or frozenset(document) != _MANIFEST_KEYS:
        raise IntegrityValidationError("manifest has an open or invalid top-level key set")
    if canonical_json_bytes_generic(document) != payload:
        raise IntegrityValidationError("manifest bytes are not canonical JSON")
    if document["schema_version"] != "vfe4-closed-manifest-v1":
        raise IntegrityValidationError("closed manifest schema is unsupported")
    entries = document["entries"]
    if type(entries) is not list:
        raise IntegrityValidationError("manifest entries must be an exact JSON list")
    parsed: list[dict[str, object]] = []
    aliases: set[str] = set()
    previous_path: str | None = None
    for raw_entry in entries:
        if type(raw_entry) is not dict or frozenset(raw_entry) != _ENTRY_KEYS:
            raise IntegrityValidationError(
                "manifest entry has an open or invalid key set"
            )
        kind = raw_entry["kind"]
        if kind not in ("file", "manifest"):
            raise IntegrityValidationError("manifest entry kind is unsupported")
        relative_path = raw_entry["relative_path"]
        _canonical_relative_path(relative_path)
        if type(raw_entry["size_bytes"]) is not int or raw_entry["size_bytes"] < 0:
            raise IntegrityValidationError(
                "manifest entry size_bytes must be a nonnegative exact int"
            )
        if not _valid_sha256(raw_entry["sha256"]):
            raise IntegrityValidationError(
                "manifest entry sha256 must be lowercase hexadecimal"
            )
        assert isinstance(relative_path, str)
        alias = relative_path.casefold()
        if alias in aliases:
            raise IntegrityValidationError(
                "manifest contains duplicate or portable-alias paths"
            )
        aliases.add(alias)
        if previous_path is not None and relative_path <= previous_path:
            raise IntegrityValidationError(
                "manifest entries must be strictly sorted by relative_path"
            )
        previous_path = relative_path
        parsed.append(raw_entry)
    return tuple(parsed)


def _validate_manifest_recursive(
    manifest_path: Path,
    *,
    root: Path,
    active_manifests: set[Path],
    max_manifest_bytes: int,
) -> tuple[ArtifactIntegrityRecord, tuple[ArtifactIntegrityRecord, ...]]:
    resolved_manifest = manifest_path.resolve(strict=True)
    if resolved_manifest in active_manifests:
        raise IntegrityValidationError("recursive manifest cycle detected")
    active_manifests.add(resolved_manifest)
    try:
        manifest_record, payload = _validated_record(
            manifest_path,
            root=root,
            kind="manifest",
            expected_size=None,
            expected_sha256=None,
            max_size=max_manifest_bytes,
        )
        parsed_entries = _parse_manifest(payload)
        records: list[ArtifactIntegrityRecord] = []
        for entry in parsed_entries:
            relative = _canonical_relative_path(entry["relative_path"])
            candidate = manifest_path.parent.joinpath(*relative.parts)
            candidate_relative = _relative_to_root(candidate, root)
            if candidate_relative == manifest_record.relative_path:
                raise IntegrityValidationError(
                    "manifest cannot declare itself as an entry"
                )
            kind = entry["kind"]
            assert kind in ("file", "manifest")
            record, _ = _validated_record(
                candidate,
                root=root,
                kind=kind,
                expected_size=entry["size_bytes"],
                expected_sha256=entry["sha256"],
                max_size=max_manifest_bytes if kind == "manifest" else None,
            )
            records.append(record)
            if kind == "manifest":
                nested_manifest, nested_records = _validate_manifest_recursive(
                    candidate,
                    root=root,
                    active_manifests=active_manifests,
                    max_manifest_bytes=max_manifest_bytes,
                )
                if nested_manifest != record:
                    raise IntegrityValidationError(
                        "recursive manifest identity changed during validation"
                    )
                records.extend(nested_records)
        aliases: set[str] = set()
        for record in records:
            alias = record.relative_path.casefold()
            if alias in aliases:
                raise IntegrityValidationError(
                    "recursive manifest expands to duplicate artifact paths"
                )
            aliases.add(alias)
        return manifest_record, tuple(records)
    finally:
        active_manifests.remove(resolved_manifest)


def validate_closed_manifest(
    manifest_path: Path,
    *,
    root: Path | None = None,
    max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
) -> ClosedManifestIdentity:
    """Validate canonical bytes and every recursively declared artifact.

    Entry sizes are checked from metadata before any entry is opened or
    hashed.  Every returned record is bound to a regular nonlink file under
    the single resolved root.
    """

    if not isinstance(manifest_path, Path):
        raise IntegrityValidationError("manifest_path must be pathlib.Path")
    if type(max_manifest_bytes) is not int or max_manifest_bytes <= 0:
        raise IntegrityValidationError(
            "max_manifest_bytes must be a positive exact int"
        )
    root_path = manifest_path.parent if root is None else root
    if not isinstance(root_path, Path):
        raise IntegrityValidationError("root must be pathlib.Path")
    _regular_directory(root_path)
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError as exc:
        raise IntegrityValidationError(
            f"manifest root could not be resolved: {exc}"
        ) from exc
    manifest_record, entries = _validate_manifest_recursive(
        manifest_path,
        root=resolved_root,
        active_manifests=set(),
        max_manifest_bytes=max_manifest_bytes,
    )
    return ClosedManifestIdentity.create(
        manifest=manifest_record,
        entries=entries,
    )


__all__ = [
    "ArtifactIntegrityRecord",
    "ClosedManifestIdentity",
    "IntegrityValidationError",
    "validate_closed_manifest",
]
