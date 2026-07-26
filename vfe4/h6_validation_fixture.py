"""Narrow, stdlib-only access to the sealed H6 validation safety fixture."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal


_REFERENCE_SCHEMA: Final = "vfe4-h6-validation-safety-fixture-reference-v1"
_REFERENCE_HASH_DOMAIN: Final = (
    "vfe4.h6.validation-safety-fixture-reference.v1"
)
_LOGICAL_PAYLOAD_NAME: Final = "validation_safety_fixture.bin"
_MANIFEST_NAME: Final = "manifest.sha256"
_FIXTURE_DOMAIN: Final = b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
_ROW_COUNT: Final = 4096
_ROW_STRUCT: Final = struct.Struct("<QH33H")
_HEADER_LENGTH: Final = len(_FIXTURE_DOMAIN) + 32 + struct.calcsize("<I")
_FIXTURE_RAW_LENGTH: Final = _HEADER_LENGTH + _ROW_COUNT * _ROW_STRUCT.size
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_WINDOWS_REPARSE_POINT: Final = 0x400

if _FIXTURE_RAW_LENGTH != 311_369:  # pragma: no cover - import-time invariant
    raise RuntimeError("H6 validation fixture format length is inconsistent")


def _canonical_json_bytes(value: object) -> bytes:
    """Return the program's stdlib-only canonical JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _owned_hash(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_json_bytes(value)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _is_link_like(path: Path, metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
        or bool(file_attributes & _WINDOWS_REPARSE_POINT)
        or path.is_symlink()
        or (hasattr(path, "is_junction") and path.is_junction())
    )


@dataclass(frozen=True, slots=True)
class ValidationSafetyFixtureReference:
    """Identity-only authority for the one permitted H6 validation payload."""

    schema_version: Literal[
        "vfe4-h6-validation-safety-fixture-reference-v1"
    ]
    logical_payload_name: Literal["validation_safety_fixture.bin"]
    local_payload_path: Path
    binary_directory_manifest_sha256: str
    data_identity_sha256: str
    access_policy_sha256: str
    validation_token_sha256: str
    fixture_raw_sha256: str
    fixture_raw_length: Literal[311369]
    row_count: Literal[4096]
    reference_sha256: str

    def _identity_payload(self) -> dict[str, object]:
        return {
            "access_policy_sha256": self.access_policy_sha256,
            "binary_directory_manifest_sha256": (
                self.binary_directory_manifest_sha256
            ),
            "data_identity_sha256": self.data_identity_sha256,
            "fixture_raw_length": self.fixture_raw_length,
            "fixture_raw_sha256": self.fixture_raw_sha256,
            "logical_payload_name": self.logical_payload_name,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "validation_token_sha256": self.validation_token_sha256,
        }

    def __post_init__(self) -> None:
        if type(self) is not ValidationSafetyFixtureReference:
            raise TypeError(
                "reference requires the exact ValidationSafetyFixtureReference type"
            )
        if (
            type(self.schema_version) is not str
            or self.schema_version != _REFERENCE_SCHEMA
        ):
            raise ValueError("validation fixture reference schema is closed")
        if (
            type(self.logical_payload_name) is not str
            or self.logical_payload_name != _LOGICAL_PAYLOAD_NAME
        ):
            raise ValueError("validation fixture logical payload name is closed")
        if (
            not isinstance(self.local_payload_path, Path)
            or not self.local_payload_path.is_absolute()
            or self.local_payload_path.name != _LOGICAL_PAYLOAD_NAME
            or self.local_payload_path.resolve(strict=False)
            != self.local_payload_path
        ):
            raise ValueError(
                "local validation fixture path must be normalized and exact"
            )
        for name in (
            "binary_directory_manifest_sha256",
            "data_identity_sha256",
            "access_policy_sha256",
            "validation_token_sha256",
            "fixture_raw_sha256",
            "reference_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.fixture_raw_length) is not int
            or self.fixture_raw_length != _FIXTURE_RAW_LENGTH
        ):
            raise ValueError("fixture_raw_length must be exactly 311369")
        if type(self.row_count) is not int or self.row_count != _ROW_COUNT:
            raise ValueError("row_count must be exactly 4096")
        if (
            _canonical_json_bytes(self._identity_payload())
            != _canonical_json_bytes(
                json.loads(
                    _canonical_json_bytes(self._identity_payload()).decode(
                        "utf-8"
                    )
                )
            )
        ):
            raise ValueError("reference identity payload is not canonical JSON")
        expected = _owned_hash(
            _REFERENCE_HASH_DOMAIN, self._identity_payload()
        )
        if self.reference_sha256 != expected:
            raise ValueError("validation fixture reference identity is stale")

    @classmethod
    def create(
        cls,
        *,
        local_payload_path: Path,
        binary_directory_manifest_sha256: str,
        data_identity_sha256: str,
        access_policy_sha256: str,
        validation_token_sha256: str,
        fixture_raw_sha256: str,
        fixture_raw_length: int,
        row_count: int,
    ) -> "ValidationSafetyFixtureReference":
        if not isinstance(local_payload_path, Path):
            raise ValueError("local_payload_path must be a pathlib.Path")
        normalized_path = local_payload_path.resolve(strict=False)
        values = {
            "schema_version": _REFERENCE_SCHEMA,
            "logical_payload_name": _LOGICAL_PAYLOAD_NAME,
            "local_payload_path": normalized_path,
            "binary_directory_manifest_sha256": (
                binary_directory_manifest_sha256
            ),
            "data_identity_sha256": data_identity_sha256,
            "access_policy_sha256": access_policy_sha256,
            "validation_token_sha256": validation_token_sha256,
            "fixture_raw_sha256": fixture_raw_sha256,
            "fixture_raw_length": fixture_raw_length,
            "row_count": row_count,
        }
        identity_payload = {
            key: value
            for key, value in values.items()
            if key != "local_payload_path"
        }
        return cls(
            **values,
            reference_sha256=_owned_hash(
                _REFERENCE_HASH_DOMAIN, identity_payload
            ),
        )


def _parse_fixture_bytes(
    reference: ValidationSafetyFixtureReference,
    fixture_bytes: bytes,
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    if type(fixture_bytes) is not bytes:
        raise ValueError("fixture bytes must be exact immutable bytes")
    if len(fixture_bytes) != reference.fixture_raw_length:
        raise ValueError("fixture byte length does not match its reference")
    if hashlib.sha256(fixture_bytes).hexdigest() != reference.fixture_raw_sha256:
        raise ValueError("fixture raw SHA-256 does not match its reference")
    if fixture_bytes[: len(_FIXTURE_DOMAIN)] != _FIXTURE_DOMAIN:
        raise ValueError("fixture header domain is invalid")
    token_start = len(_FIXTURE_DOMAIN)
    token_end = token_start + 32
    validation_token_sha256 = fixture_bytes[token_start:token_end].hex()
    if validation_token_sha256 != reference.validation_token_sha256:
        raise ValueError("fixture validation-token SHA-256 is stale")
    (row_count,) = struct.unpack_from("<I", fixture_bytes, token_end)
    if row_count != _ROW_COUNT or row_count != reference.row_count:
        raise ValueError("fixture row count is not exactly 4096")

    starts: list[int] = []
    real_target_counts: list[int] = []
    seen_starts: set[int] = set()
    offset = _HEADER_LENGTH
    for _ in range(_ROW_COUNT):
        row = _ROW_STRUCT.unpack_from(fixture_bytes, offset)
        offset += _ROW_STRUCT.size
        start, real_target_count, *token_ids = row
        if start in seen_starts:
            raise ValueError("fixture starts must be ordered unique values")
        seen_starts.add(start)
        if not 1 <= real_target_count <= 32:
            raise ValueError("fixture real-target count must be in 1..32")
        if any(token_id > 257 for token_id in token_ids):
            raise ValueError("fixture token ID must be in 0..257")
        starts.append(start)
        real_target_counts.append(real_target_count)
    if offset != len(fixture_bytes):
        raise ValueError("fixture row payload length is invalid")
    return (
        validation_token_sha256,
        tuple(starts),
        tuple(real_target_counts),
    )


@dataclass(frozen=True, slots=True)
class ValidationSafetyFixturePayload:
    """Exact immutable bytes and parsed fields from the narrow fixture."""

    reference: ValidationSafetyFixtureReference
    fixture_bytes: bytes
    validation_token_sha256: str
    starts: tuple[int, ...]
    real_target_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self) is not ValidationSafetyFixturePayload:
            raise TypeError(
                "payload requires the exact ValidationSafetyFixturePayload type"
            )
        if type(self.reference) is not ValidationSafetyFixtureReference:
            raise ValueError("payload requires an exact validation fixture reference")
        self.reference.__post_init__()
        if type(self.fixture_bytes) is not bytes:
            raise ValueError("fixture bytes must be exact immutable bytes")
        if type(self.validation_token_sha256) is not str:
            raise ValueError("parsed validation-token SHA-256 must be exact text")
        if type(self.starts) is not tuple:
            raise ValueError("parsed starts must be an immutable tuple")
        if type(self.real_target_counts) is not tuple:
            raise ValueError(
                "parsed real-target counts must be an immutable tuple"
            )
        parsed = _parse_fixture_bytes(self.reference, self.fixture_bytes)
        if parsed != (
            self.validation_token_sha256,
            self.starts,
            self.real_target_counts,
        ):
            raise ValueError("retained fixture bytes differ from parsed fields")


def _normalized_root(reference: ValidationSafetyFixtureReference) -> Path:
    payload_path = reference.local_payload_path
    root = payload_path.parent
    try:
        root_metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("validation fixture directory is unavailable") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or _is_link_like(root, root_metadata)
        or resolved_root != root
    ):
        raise ValueError(
            "validation fixture directory must be non-redirected"
        )
    try:
        common = Path(
            os.path.commonpath(
                (os.fspath(resolved_root), os.fspath(payload_path))
            )
        )
    except ValueError as exc:
        raise ValueError("validation fixture path escapes its directory") from exc
    if common != resolved_root:
        raise ValueError("validation fixture path escapes its directory")
    return resolved_root


def _read_bounded_regular_file(
    root: Path,
    path: Path,
    *,
    maximum_length: int,
    exact_length: int | None,
    label: str,
) -> bytes:
    if path.parent != root:
        raise ValueError(f"{label} path escapes the validation fixture directory")
    try:
        root_before = root.stat()
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if (
        resolved != path
        or resolved.parent != root
        or not stat.S_ISREG(metadata.st_mode)
        or _is_link_like(path, metadata)
    ):
        raise ValueError(f"{label} must be a regular nonlink file")
    if metadata.st_size > maximum_length or (
        exact_length is not None and metadata.st_size != exact_length
    ):
        raise ValueError(f"{label} length violates its frozen bound")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_link_like(path, opened)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
        ):
            raise ValueError(f"{label} changed before opening")
        while True:
            chunk = os.read(
                descriptor, min(65_536, maximum_length + 1 - total)
            )
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_length:
                raise ValueError(f"{label} exceeds its frozen bound")
            chunks.append(chunk)
        closed_snapshot = os.fstat(descriptor)
        current = path.lstat()
        root_after = root.stat()
        if (
            _is_link_like(path, current)
            or (closed_snapshot.st_dev, closed_snapshot.st_ino, closed_snapshot.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (current.st_dev, current.st_ino, current.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or (root_after.st_dev, root_after.st_ino)
            != (root_before.st_dev, root_before.st_ino)
        ):
            raise ValueError(f"{label} changed while reading")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{label} cannot be read safely") from exc
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if len(content) != metadata.st_size or (
        exact_length is not None and len(content) != exact_length
    ):
        raise ValueError(f"{label} length changed while reading")
    return content


def read_validation_safety_fixture_payload(
    reference: ValidationSafetyFixtureReference,
) -> ValidationSafetyFixturePayload:
    """Read only the frozen manifest sibling and validation fixture payload."""

    if type(reference) is not ValidationSafetyFixtureReference:
        raise ValueError("reader requires an exact validation fixture reference")
    reference.__post_init__()
    root = _normalized_root(reference)
    manifest_path = root / _MANIFEST_NAME
    payload_path = reference.local_payload_path
    if (
        manifest_path.name != _MANIFEST_NAME
        or payload_path.name != _LOGICAL_PAYLOAD_NAME
    ):
        raise ValueError("validation fixture logical paths are not exact")

    manifest_bytes = _read_bounded_regular_file(
        root,
        manifest_path,
        maximum_length=65,
        exact_length=None,
        label="manifest",
    )
    expected_manifest = (
        reference.binary_directory_manifest_sha256 + "\n"
    ).encode("ascii")
    if manifest_bytes != expected_manifest:
        raise ValueError("manifest SHA-256 does not match the reference")
    fixture_bytes = _read_bounded_regular_file(
        root,
        payload_path,
        maximum_length=reference.fixture_raw_length,
        exact_length=reference.fixture_raw_length,
        label="fixture",
    )
    parsed = _parse_fixture_bytes(reference, fixture_bytes)
    return ValidationSafetyFixturePayload(
        reference=reference,
        fixture_bytes=fixture_bytes,
        validation_token_sha256=parsed[0],
        starts=parsed[1],
        real_target_counts=parsed[2],
    )


__all__ = [
    "ValidationSafetyFixturePayload",
    "ValidationSafetyFixtureReference",
    "read_validation_safety_fixture_payload",
]
