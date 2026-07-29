"""Generic, platform-probed durable file publication primitives.

The guarantees in this module are deliberately narrow: a successful operation
proves that the configured create/write/flush/reopen or
create/write/flush/reopen/replace/reopen sequence completed on the recorded
volume.  It does not make a broader claim about an unprobed filesystem.
"""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import hashlib
import inspect
import json
import math
import os
import platform
import stat
import uuid
from collections.abc import Callable, Iterable
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


POSIX_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0x10000)

WINDOWS_GENERIC_WRITE = 0x40000000
WINDOWS_CREATE_NEW = 1
WINDOWS_FILE_ATTRIBUTE_NORMAL = 0x00000080
WINDOWS_FILE_FLAG_WRITE_THROUGH = 0x80000000
WINDOWS_MOVEFILE_REPLACE_EXISTING = 0x00000001
WINDOWS_MOVEFILE_WRITE_THROUGH = 0x00000008

_DURABLE_FILE_DOMAIN = b"vfe4.durable-file-identity.v1\0"
_DURABILITY_IDENTITY_DOMAIN = b"vfe4.durability-identity.v1\0"
_VOLUME_IDENTITY_DOMAIN = b"vfe4.volume-identity.v1\0"
_PROBE_CREATE_PAYLOAD = b"vfe4-durability-probe-create-v1\n"
_PROBE_OLD_PAYLOAD = b"vfe4-durability-probe-old-v1\n"
_PROBE_REPLACE_PAYLOAD = b"vfe4-durability-probe-replace-v1\n"
DEFAULT_STREAM_CHUNK_SIZE_LIMIT = 1024 * 1024
DEFAULT_STREAM_REOPEN_BLOCK_SIZE = 1024 * 1024

_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "cifs",
        "davfs",
        "gcsfuse",
        "nfs",
        "nfs4",
        "s3fs",
        "smb",
        "smb2",
        "smb3",
        "sshfs",
    }
)


class DurabilityError(RuntimeError):
    """Base class for a rejected or unverified durability operation."""


class DurabilityCollisionError(DurabilityError):
    """An exclusive-create destination already exists."""


class DurabilityOperationError(DurabilityError):
    """A durability operation failed with an explicit uncertainty boundary."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        indeterminate: bool,
        error_code: int | None = None,
        obligations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.indeterminate = indeterminate
        self.error_code = error_code
        self.obligations = obligations


def _canonical_value(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise DurabilityError("canonical JSON mapping keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DurabilityError("canonical JSON rejects nonfinite floats")
        return value
    raise DurabilityError(
        f"unsupported canonical JSON value type: {type(value).__name__}"
    )


def canonical_json_bytes_generic(value: object) -> bytes:
    """Return canonical UTF-8 JSON for generic records without domain imports."""

    try:
        return json.dumps(
            _canonical_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except DurabilityError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise DurabilityError(f"canonical JSON serialization failed: {exc}") from exc


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes_generic(value)).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclasses.dataclass(frozen=True, slots=True)
class VolumeFacts:
    """Filesystem facts used to bind a durability result to one volume."""

    volume_path: str
    volume_serial: str
    filesystem_type: str
    is_remote: bool
    is_cloud_sync: bool = False

    def __post_init__(self) -> None:
        for name in ("volume_path", "volume_serial", "filesystem_type"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise DurabilityError(f"{name} must be a nonempty string")
        if type(self.is_remote) is not bool or type(self.is_cloud_sync) is not bool:
            raise DurabilityError("volume flags must be exact bool values")

    @property
    def identity(self) -> str:
        return _domain_hash(
            _VOLUME_IDENTITY_DOMAIN,
            {
                "filesystem_type": self.filesystem_type,
                "is_cloud_sync": self.is_cloud_sync,
                "is_remote": self.is_remote,
                "volume_path": self.volume_path,
                "volume_serial": self.volume_serial,
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DurableFileIdentity:
    """Typed identity returned only after exact reopen validation."""

    schema_version: Literal["vfe4-durable-file-v1"]
    operation: Literal["exclusive_create", "replace", "content_addressed"]
    size_bytes: int
    sha256: str
    volume_identity: str
    reopen_verified: Literal[True]
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        operation: Literal["exclusive_create", "replace"],
        payload: bytes,
        volume_identity: str,
    ) -> DurableFileIdentity:
        return cls.create_verified(
            operation=operation,
            size_bytes=len(payload),
            sha256=_sha256(payload),
            volume_identity=volume_identity,
        )

    @classmethod
    def create_verified(
        cls,
        *,
        operation: Literal[
            "exclusive_create",
            "replace",
            "content_addressed",
        ],
        size_bytes: int,
        sha256: str,
        volume_identity: str,
    ) -> DurableFileIdentity:
        """Construct an identity from already reopen-verified size/SHA facts."""

        if operation not in (
            "exclusive_create",
            "replace",
            "content_addressed",
        ):
            raise DurabilityError("durable file operation is invalid")
        if type(size_bytes) is not int or size_bytes < 0:
            raise DurabilityError("verified size_bytes must be a nonnegative integer")
        if not _valid_sha256(sha256):
            raise DurabilityError("verified sha256 must be lowercase hexadecimal")
        if type(volume_identity) is not str or not volume_identity:
            raise DurabilityError("volume_identity must be a nonempty string")
        body = {
            "operation": operation,
            "reopen_verified": True,
            "schema_version": "vfe4-durable-file-v1",
            "sha256": sha256,
            "size_bytes": size_bytes,
            "volume_identity": volume_identity,
        }
        return cls(
            schema_version="vfe4-durable-file-v1",
            operation=operation,
            size_bytes=size_bytes,
            sha256=sha256,
            volume_identity=volume_identity,
            reopen_verified=True,
            identity_sha256=_domain_hash(_DURABLE_FILE_DOMAIN, body),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedFileFacts:
    """Bounded-read facts for one stable regular nonlink file."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise DurabilityError("verified size_bytes must be a nonnegative integer")
        if not _valid_sha256(self.sha256):
            raise DurabilityError("verified sha256 must be lowercase hexadecimal")


@dataclasses.dataclass(frozen=True, slots=True)
class DurabilityErrorRecord:
    phase: str
    exception_type: str
    error_code: int | None
    message: str
    message_sha256: str

    @classmethod
    def capture(cls, phase: str, error: BaseException) -> DurabilityErrorRecord:
        message = str(error)
        return cls(
            phase=phase,
            exception_type=type(error).__name__,
            error_code=_error_code(error),
            message=message,
            message_sha256=_sha256(message.encode("utf-8", errors="strict")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DurabilityIdentity:
    """Result of probing one root on one recorded filesystem volume."""

    schema_version: Literal["vfe4-durability-backend-v1"]
    backend_kind: Literal["posix", "windows"]
    implementation_sha256: str
    os_name: str
    platform_system: str
    platform_release: str
    platform_version: str
    filesystem_type: str
    volume_identity: str
    supported_operations: tuple[str, ...]
    create_sha256: str | None
    replace_sha256: str | None
    status: Literal["pass", "inconclusive"]
    errors: tuple[DurabilityErrorRecord, ...]
    obligations: tuple[str, ...]
    identity_sha256: str

    @classmethod
    def create(
        cls,
        *,
        backend_kind: Literal["posix", "windows"],
        implementation_sha256: str,
        volume: VolumeFacts | None,
        create_sha256: str | None,
        replace_sha256: str | None,
        errors: tuple[DurabilityErrorRecord, ...],
        obligations: tuple[str, ...],
    ) -> DurabilityIdentity:
        status: Literal["pass", "inconclusive"] = (
            "pass" if not errors and not obligations else "inconclusive"
        )
        body = {
            "backend_kind": backend_kind,
            "create_sha256": create_sha256,
            "errors": errors,
            "filesystem_type": volume.filesystem_type if volume else "unknown",
            "implementation_sha256": implementation_sha256,
            "obligations": obligations,
            "os_name": os.name,
            "platform_release": platform.release(),
            "platform_system": platform.system(),
            "platform_version": platform.version(),
            "replace_sha256": replace_sha256,
            "schema_version": "vfe4-durability-backend-v1",
            "status": status,
            "supported_operations": (
                "exclusive_create_flush_reopen",
                "same_volume_replace_flush_reopen",
            ),
            "volume_identity": volume.identity if volume else "unknown",
        }
        return cls(
            schema_version="vfe4-durability-backend-v1",
            backend_kind=backend_kind,
            implementation_sha256=implementation_sha256,
            os_name=os.name,
            platform_system=body["platform_system"],
            platform_release=body["platform_release"],
            platform_version=body["platform_version"],
            filesystem_type=body["filesystem_type"],
            volume_identity=body["volume_identity"],
            supported_operations=body["supported_operations"],
            create_sha256=create_sha256,
            replace_sha256=replace_sha256,
            status=status,
            errors=errors,
            obligations=obligations,
            identity_sha256=_domain_hash(_DURABILITY_IDENTITY_DOMAIN, body),
        )


@runtime_checkable
class DurabilityBackend(Protocol):
    """Minimal durability surface consumed by scientific artifact modules."""

    def probe(self, root: Path) -> DurabilityIdentity: ...

    def create_exclusive(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity: ...

    def replace_durable(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity: ...

    def publish_bytes(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity: ...


@runtime_checkable
class ContentAddressedDurabilityBackend(DurabilityBackend, Protocol):
    """Additive bounded-stream surface; byte-only backend checks stay stable."""

    def publish_content_addressed_stream(
        self,
        directory: Path,
        chunks: Iterable[bytes],
        *,
        suffix: str,
        chunk_size_limit: int = DEFAULT_STREAM_CHUNK_SIZE_LIMIT,
        reopen_block_size: int = DEFAULT_STREAM_REOPEN_BLOCK_SIZE,
    ) -> DurableFileIdentity: ...


def _error_code(error: BaseException) -> int | None:
    for name in ("winerror", "errno"):
        value = getattr(error, name, None)
        if type(value) is int:
            return value
    return None


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_regular_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise DurabilityOperationError(
            f"durability parent is unavailable: {path}: {exc}",
            phase="validate_parent",
            indeterminate=False,
            error_code=_error_code(exc),
        ) from exc
    if not stat.S_ISDIR(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise DurabilityOperationError(
            f"durability parent must be a regular nonlink directory: {path}",
            phase="validate_parent",
            indeterminate=False,
        )


def _require_replaceable_target(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DurabilityOperationError(
            f"target metadata could not be read: {path}: {exc}",
            phase="validate_target",
            indeterminate=False,
            error_code=_error_code(exc),
        ) from exc
    if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(path, status):
        raise DurabilityOperationError(
            f"target must be absent or a regular nonlink file: {path}",
            phase="validate_target",
            indeterminate=False,
        )


def _require_path(path: Path) -> Path:
    if not isinstance(path, Path) or path.name in ("", ".", ".."):
        raise DurabilityOperationError(
            "durability path must be a concrete pathlib.Path file",
            phase="validate_path",
            indeterminate=False,
        )
    _require_regular_directory(path.parent)
    return path


def _require_payload(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise DurabilityOperationError(
            "durability payload must be exact bytes",
            phase="validate_payload",
            indeterminate=False,
        )
    return payload


def _open_regular_read(path: Path) -> Any:
    return path.open("rb", buffering=0)


def validate_regular_nonlink_sha256(
    path: Path,
    *,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
    block_size: int = DEFAULT_STREAM_REOPEN_BLOCK_SIZE,
    opener: Callable[[Path], Any] | None = None,
) -> VerifiedFileFacts:
    """Validate one stable regular nonlink using bounded sequential reads."""

    if not isinstance(path, Path) or path.name in ("", ".", ".."):
        raise DurabilityOperationError(
            "validation path must be a concrete pathlib.Path file",
            phase="validate_regular_file",
            indeterminate=False,
        )
    if type(block_size) is not int or block_size <= 0:
        raise DurabilityOperationError(
            "validation block_size must be a positive integer",
            phase="validate_regular_file",
            indeterminate=False,
        )
    if (
        expected_size_bytes is not None
        and (
            type(expected_size_bytes) is not int
            or expected_size_bytes < 0
        )
    ):
        raise DurabilityOperationError(
            "expected size must be a nonnegative integer or None",
            phase="validate_regular_file",
            indeterminate=False,
        )
    if expected_sha256 is not None and not _valid_sha256(expected_sha256):
        raise DurabilityOperationError(
            "expected SHA-256 must be lowercase hexadecimal or None",
            phase="validate_regular_file",
            indeterminate=False,
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise DurabilityOperationError(
            f"regular-file metadata failed for {path}: {exc}",
            phase="validate_regular_file",
            indeterminate=False,
            error_code=_error_code(exc),
        ) from exc
    if not stat.S_ISREG(before.st_mode) or _is_redirect_or_reparse(path, before):
        raise DurabilityOperationError(
            f"validation target must be a regular nonlink file: {path}",
            phase="validate_regular_file",
            indeterminate=False,
        )
    if (
        expected_size_bytes is not None
        and before.st_size != expected_size_bytes
    ):
        raise DurabilityOperationError(
            f"regular-file size did not match for {path}",
            phase="validate_regular_file",
            indeterminate=False,
        )

    digest = hashlib.sha256()
    size_bytes = 0
    open_regular = _open_regular_read if opener is None else opener
    try:
        with open_regular(path) as handle:
            while True:
                block = handle.read(block_size)
                if type(block) is not bytes or len(block) > block_size:
                    raise DurabilityOperationError(
                        "regular-file reader violated the bounded bytes contract",
                        phase="validate_regular_file",
                        indeterminate=False,
                    )
                if not block:
                    break
                digest.update(block)
                size_bytes += len(block)
        after = path.lstat()
    except DurabilityOperationError:
        raise
    except OSError as exc:
        raise DurabilityOperationError(
            f"regular-file bounded read failed for {path}: {exc}",
            phase="validate_regular_file",
            indeterminate=False,
            error_code=_error_code(exc),
        ) from exc

    observed_sha256 = digest.hexdigest()
    if (
        not stat.S_ISREG(after.st_mode)
        or _is_redirect_or_reparse(path, after)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size_bytes != before.st_size
    ):
        raise DurabilityOperationError(
            f"regular-file identity changed during validation: {path}",
            phase="validate_regular_file",
            indeterminate=False,
        )
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise DurabilityOperationError(
            f"regular-file SHA-256 did not match for {path}",
            phase="validate_regular_file",
            indeterminate=False,
        )
    return VerifiedFileFacts(
        size_bytes=size_bytes,
        sha256=observed_sha256,
    )


def _implementation_sha256() -> str:
    try:
        return _sha256(Path(__file__).read_bytes())
    except OSError:
        source = inspect.getsource(DurabilityBackend).encode("utf-8")
        return _sha256(source)


def _supported_volume(volume: VolumeFacts) -> tuple[bool, str | None]:
    filesystem = volume.filesystem_type.strip().lower()
    if volume.is_remote:
        return False, "remote filesystem semantics are unsupported"
    if volume.is_cloud_sync:
        return False, "cloud-synchronized filesystem semantics are unsupported"
    if not filesystem or filesystem == "unknown":
        return False, "filesystem type is unknown"
    if filesystem.startswith("fuse.") or filesystem in _NETWORK_FILESYSTEMS:
        return False, f"filesystem type {volume.filesystem_type!r} is unsupported"
    return True, None


class _BackendBase:
    backend_kind: Literal["posix", "windows"]

    def __init__(self, *, syscalls: object) -> None:
        self._syscalls = syscalls

    def _volume(self, path: Path) -> VolumeFacts:
        try:
            volume = self._syscalls.volume_facts(path)
        except OSError as exc:
            raise DurabilityOperationError(
                f"volume identity failed for {path}: {exc}",
                phase="volume_identity",
                indeterminate=False,
                error_code=_error_code(exc),
            ) from exc
        if not isinstance(volume, VolumeFacts):
            raise DurabilityOperationError(
                "syscall adapter returned an untyped volume identity",
                phase="volume_identity",
                indeterminate=False,
            )
        return volume

    def _same_volume(
        self, destination: Path, staging: Path
    ) -> VolumeFacts:
        destination_volume = self._volume(destination)
        staging_volume = self._volume(staging)
        if destination_volume.identity != staging_volume.identity:
            raise DurabilityOperationError(
                "staging and destination are on different volumes",
                phase="same_volume",
                indeterminate=False,
            )
        supported, reason = _supported_volume(destination_volume)
        if not supported:
            raise DurabilityOperationError(
                reason or "volume semantics are unsupported",
                phase="volume_support",
                indeterminate=False,
            )
        return destination_volume

    def _read_back(self, path: Path, payload: bytes, *, phase: str) -> None:
        try:
            before = path.lstat()
        except OSError as exc:
            raise DurabilityOperationError(
                f"{phase} metadata failed: {exc}",
                phase=phase,
                indeterminate=True,
                error_code=_error_code(exc),
                obligations=("the durable state of the target must be investigated",),
            ) from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or _is_redirect_or_reparse(path, before)
            or before.st_size != len(payload)
        ):
            raise DurabilityOperationError(
                f"{phase} did not reopen the expected regular file",
                phase=phase,
                indeterminate=True,
                obligations=("the durable state of the target must be investigated",),
            )
        try:
            observed = self._syscalls.read_regular_bytes(path)
            after = path.lstat()
        except OSError as exc:
            raise DurabilityOperationError(
                f"{phase} read-back failed: {exc}",
                phase=phase,
                indeterminate=True,
                error_code=_error_code(exc),
                obligations=("the durable state of the target must be investigated",),
            ) from exc
        if (
            observed != payload
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise DurabilityOperationError(
                f"{phase} read-back identity or SHA-256 did not match",
                phase=phase,
                indeterminate=True,
                obligations=("the durable state of the target must be investigated",),
            )

    def _stage_path(self, path: Path) -> Path:
        return path.with_name(f".{path.name}.vfe4-stage-{uuid.uuid4().hex}")

    def _create_file(
        self,
        path: Path,
        payload: bytes,
        *,
        operation: Literal["exclusive_create", "replace"],
        target_at_risk: bool,
    ) -> None:
        raise NotImplementedError

    def _sync_parent(self, parent: Path, *, target_at_risk: bool) -> None:
        raise NotImplementedError

    def _replace(self, staging: Path, destination: Path) -> None:
        raise NotImplementedError

    def _create_stream_file(
        self,
        path: Path,
        chunks: Iterable[bytes],
    ) -> None:
        raise NotImplementedError

    def _promote_no_replace(
        self,
        staging: Path,
        destination: Path,
    ) -> None:
        raise NotImplementedError

    def _validated_stream_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        chunk_size_limit: int,
    ) -> Iterable[bytes]:
        try:
            iterator = iter(chunks)
        except TypeError as exc:
            raise DurabilityOperationError(
                "stream chunks must be an iterable of exact bytes",
                phase="validate_stream",
                indeterminate=False,
            ) from exc
        for chunk in iterator:
            if type(chunk) is not bytes:
                raise DurabilityOperationError(
                    "stream chunks must contain exact bytes",
                    phase="validate_stream_chunk",
                    indeterminate=False,
                )
            if len(chunk) > chunk_size_limit:
                raise DurabilityOperationError(
                    "stream chunk exceeds chunk_size_limit",
                    phase="validate_stream_chunk",
                    indeterminate=False,
                )
            if chunk:
                yield chunk

    def _validate_stream_file(
        self,
        path: Path,
        *,
        expected_size_bytes: int | None,
        expected_sha256: str | None,
        block_size: int,
        phase: str,
        indeterminate: bool,
    ) -> VerifiedFileFacts:
        try:
            return validate_regular_nonlink_sha256(
                path,
                expected_size_bytes=expected_size_bytes,
                expected_sha256=expected_sha256,
                block_size=block_size,
                opener=self._syscalls.open_regular_read,
            )
        except DurabilityOperationError as exc:
            raise DurabilityOperationError(
                str(exc),
                phase=phase,
                indeterminate=indeterminate,
                error_code=exc.error_code,
                obligations=(
                    ("the durable state of the target must be investigated",)
                    if indeterminate
                    else (f"the retained staging file must be inspected: {path.name}",)
                ),
            ) from exc

    def _remove_owned_stage(
        self,
        staging: Path,
        *,
        target_at_risk: bool,
    ) -> None:
        try:
            self._syscalls.unlink(staging)
        except OSError as exc:
            raise DurabilityOperationError(
                f"owned stream staging cleanup failed for {staging}: {exc}",
                phase="cleanup_stream_staging",
                indeterminate=target_at_risk,
                error_code=_error_code(exc),
                obligations=(
                    f"the retained staging file must be inspected: {staging.name}",
                ),
            ) from exc

    def create_exclusive(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity:
        path = _require_path(path)
        payload = _require_payload(payload)
        volume = self._volume(path)
        supported, reason = _supported_volume(volume)
        if not supported:
            raise DurabilityOperationError(
                reason or "volume semantics are unsupported",
                phase="volume_support",
                indeterminate=False,
            )
        self._create_file(
            path,
            payload,
            operation="exclusive_create",
            target_at_risk=True,
        )
        self._read_back(path, payload, phase="reopen_exclusive")
        self._sync_parent(path.parent, target_at_risk=True)
        return DurableFileIdentity.create(
            operation="exclusive_create",
            payload=payload,
            volume_identity=volume.identity,
        )

    def replace_durable(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity:
        path = _require_path(path)
        payload = _require_payload(payload)
        _require_replaceable_target(path)
        staging = self._stage_path(path)
        volume = self._same_volume(path, staging)
        self._create_file(
            staging,
            payload,
            operation="replace",
            target_at_risk=False,
        )
        try:
            self._read_back(staging, payload, phase="reopen_staging")
        except DurabilityOperationError as exc:
            raise DurabilityOperationError(
                str(exc),
                phase=exc.phase,
                indeterminate=False,
                error_code=exc.error_code,
                obligations=(
                    f"the retained staging file must be inspected: {staging.name}",
                ),
            ) from exc
        try:
            self._replace(staging, path)
        except OSError as exc:
            raise DurabilityOperationError(
                f"durable replacement returned an error: {exc}",
                phase="replace",
                indeterminate=True,
                error_code=_error_code(exc),
                obligations=(
                    "the destination may contain either the predecessor or replacement",
                    "the destination must be reopened and investigated before retry",
                ),
            ) from exc
        self._sync_parent(path.parent, target_at_risk=True)
        self._read_back(path, payload, phase="reopen_destination")
        return DurableFileIdentity.create(
            operation="replace",
            payload=payload,
            volume_identity=volume.identity,
        )

    def publish_bytes(
        self, path: Path, payload: bytes
    ) -> DurableFileIdentity:
        """Create an absent target or replace a present regular target.

        A competing create after the absence check is reported as an
        exclusive-create collision; it is never converted into permission to
        overwrite the competing publisher.
        """

        path = _require_path(path)
        payload = _require_payload(payload)
        try:
            status = path.lstat()
        except FileNotFoundError:
            return self.create_exclusive(path, payload)
        except OSError as exc:
            raise DurabilityOperationError(
                f"publication target metadata failed: {exc}",
                phase="validate_target",
                indeterminate=False,
                error_code=_error_code(exc),
            ) from exc
        if not stat.S_ISREG(status.st_mode) or _is_redirect_or_reparse(path, status):
            raise DurabilityOperationError(
                f"publication target must be absent or a regular nonlink file: {path}",
                phase="validate_target",
                indeterminate=False,
            )
        return self.replace_durable(path, payload)

    def publish_content_addressed_stream(
        self,
        directory: Path,
        chunks: Iterable[bytes],
        *,
        suffix: str,
        chunk_size_limit: int = DEFAULT_STREAM_CHUNK_SIZE_LIMIT,
        reopen_block_size: int = DEFAULT_STREAM_REOPEN_BLOCK_SIZE,
    ) -> DurableFileIdentity:
        """Publish bounded chunks to ``<sha256><suffix>`` without overwrite."""

        if not isinstance(directory, Path):
            raise DurabilityOperationError(
                "stream publication directory must be a pathlib.Path",
                phase="validate_stream",
                indeterminate=False,
            )
        _require_regular_directory(directory)
        if (
            type(suffix) is not str
            or not suffix.startswith(".")
            or suffix in (".", "..")
            or "/" in suffix
            or "\\" in suffix
            or Path(suffix).name != suffix
        ):
            raise DurabilityOperationError(
                "stream suffix must be one safe filename suffix",
                phase="validate_stream",
                indeterminate=False,
            )
        for value, field in (
            (chunk_size_limit, "chunk_size_limit"),
            (reopen_block_size, "reopen_block_size"),
        ):
            if type(value) is not int or value <= 0:
                raise DurabilityOperationError(
                    f"{field} must be a positive integer",
                    phase="validate_stream",
                    indeterminate=False,
                )

        staging = directory / f".vfe4-stream-stage-{uuid.uuid4().hex}"
        volume = self._same_volume(directory, staging)
        try:
            self._create_stream_file(
                staging,
                self._validated_stream_chunks(
                    chunks,
                    chunk_size_limit=chunk_size_limit,
                ),
            )
        except DurabilityOperationError as exc:
            if exc.phase in ("validate_stream", "validate_stream_chunk"):
                self._remove_owned_stage(staging, target_at_risk=False)
            raise
        except DurabilityError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            self._remove_owned_stage(staging, target_at_risk=False)
            raise DurabilityOperationError(
                f"stream producer failed while writing {staging}: {exc}",
                phase="write_stream",
                indeterminate=False,
            ) from exc

        staged = self._validate_stream_file(
            staging,
            expected_size_bytes=None,
            expected_sha256=None,
            block_size=reopen_block_size,
            phase="reopen_staging_stream",
            indeterminate=False,
        )
        destination = directory / f"{staged.sha256}{suffix}"

        def recover_existing() -> DurableFileIdentity:
            try:
                self._validate_stream_file(
                    destination,
                    expected_size_bytes=staged.size_bytes,
                    expected_sha256=staged.sha256,
                    block_size=reopen_block_size,
                    phase="reopen_existing_stream",
                    indeterminate=False,
                )
            except DurabilityOperationError as exc:
                self._remove_owned_stage(staging, target_at_risk=False)
                raise DurabilityCollisionError(
                    "content-addressed destination exists with conflicting content: "
                    f"{destination}"
                ) from exc
            self._remove_owned_stage(staging, target_at_risk=False)
            self._sync_parent(directory, target_at_risk=False)
            return DurableFileIdentity.create_verified(
                operation="content_addressed",
                size_bytes=staged.size_bytes,
                sha256=staged.sha256,
                volume_identity=volume.identity,
            )

        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise DurabilityOperationError(
                f"stream destination metadata failed: {exc}",
                phase="validate_stream_destination",
                indeterminate=False,
                error_code=_error_code(exc),
            ) from exc
        else:
            return recover_existing()

        try:
            self._promote_no_replace(staging, destination)
        except OSError as exc:
            if _error_code(exc) in (errno.EEXIST, 80, 183):
                return recover_existing()
            raise DurabilityOperationError(
                f"content-addressed promotion returned an error: {exc}",
                phase="promote_stream",
                indeterminate=True,
                error_code=_error_code(exc),
                obligations=(
                    "the destination and retained staging file must be investigated",
                ),
            ) from exc
        self._sync_parent(directory, target_at_risk=True)
        self._validate_stream_file(
            destination,
            expected_size_bytes=staged.size_bytes,
            expected_sha256=staged.sha256,
            block_size=reopen_block_size,
            phase="reopen_destination_stream",
            indeterminate=True,
        )
        return DurableFileIdentity.create_verified(
            operation="content_addressed",
            size_bytes=staged.size_bytes,
            sha256=staged.sha256,
            volume_identity=volume.identity,
        )

    def probe(self, root: Path) -> DurabilityIdentity:
        errors: list[DurabilityErrorRecord] = []
        obligations: list[str] = []
        create_digest: str | None = None
        replace_digest: str | None = None
        volume: VolumeFacts | None = None
        token = uuid.uuid4().hex
        create_path = root / f".vfe4-durability-probe-{token}-create"
        replace_path = root / f".vfe4-durability-probe-{token}-replace"
        owned_payloads: dict[Path, bytes] = {}
        phase = "validate_root"
        try:
            _require_regular_directory(root)
            phase = "volume_identity"
            volume = self._volume(root)
            supported, reason = _supported_volume(volume)
            if not supported:
                obligations.append(reason or "volume semantics are unsupported")
            else:
                phase = "exclusive_create"
                created = self.create_exclusive(create_path, _PROBE_CREATE_PAYLOAD)
                owned_payloads[create_path] = _PROBE_CREATE_PAYLOAD
                create_digest = created.sha256
                phase = "exclusive_collision"
                try:
                    self.create_exclusive(create_path, b"collision")
                except DurabilityCollisionError:
                    pass
                else:
                    raise DurabilityOperationError(
                        "exclusive-create collision was not rejected",
                        phase=phase,
                        indeterminate=True,
                        obligations=(
                            "the backend exclusive-create contract is unverified",
                        ),
                    )
                phase = "replace_seed"
                self.create_exclusive(replace_path, _PROBE_OLD_PAYLOAD)
                owned_payloads[replace_path] = _PROBE_OLD_PAYLOAD
                phase = "replace"
                replaced = self.replace_durable(
                    replace_path, _PROBE_REPLACE_PAYLOAD
                )
                owned_payloads[replace_path] = _PROBE_REPLACE_PAYLOAD
                replace_digest = replaced.sha256
        except (DurabilityError, OSError, RuntimeError) as exc:
            errors.append(DurabilityErrorRecord.capture(phase, exc))
            if isinstance(exc, DurabilityOperationError):
                obligations.extend(exc.obligations)
            obligations.append(
                "scientific state must not begin until durability probes pass"
            )
        finally:
            cleanup_errors: list[DurabilityErrorRecord] = []
            for candidate, expected_payload in owned_payloads.items():
                try:
                    status = candidate.lstat()
                    if (
                        not stat.S_ISREG(status.st_mode)
                        or _is_redirect_or_reparse(candidate, status)
                        or status.st_size != len(expected_payload)
                        or self._syscalls.read_regular_bytes(candidate)
                        != expected_payload
                    ):
                        raise DurabilityError(
                            "owned probe path changed before cleanup"
                        )
                    self._syscalls.unlink(candidate)
                except (DurabilityError, OSError) as exc:
                    cleanup_errors.append(
                        DurabilityErrorRecord.capture("probe_cleanup", exc)
                    )
            if cleanup_errors:
                errors.extend(cleanup_errors)
                obligations.append(
                    "owned durability probe remnants require manual investigation"
                )
        return DurabilityIdentity.create(
            backend_kind=self.backend_kind,
            implementation_sha256=_implementation_sha256(),
            volume=volume,
            create_sha256=create_digest,
            replace_sha256=replace_digest,
            errors=tuple(errors),
            obligations=tuple(dict.fromkeys(obligations)),
        )


class _RealPosixSyscalls:
    def volume_facts(self, path: Path) -> VolumeFacts:
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        resolved = existing.resolve(strict=True)
        status = resolved.stat()
        mount_path, filesystem_type = _posix_mount_facts(resolved)
        return VolumeFacts(
            volume_path=mount_path,
            volume_serial=str(status.st_dev),
            filesystem_type=filesystem_type,
            is_remote=(
                filesystem_type.lower() in _NETWORK_FILESYSTEMS
                or filesystem_type.lower().startswith("fuse.")
            ),
            is_cloud_sync=False,
        )

    def open_exclusive(self, path: Path, flags: int, mode: int) -> int:
        return os.open(path, flags, mode)

    def write_all(self, handle: int, payload: bytes) -> None:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(handle, view[offset:])
            if written <= 0:
                raise OSError("write returned no progress")
            offset += written

    def flush_file(self, handle: int) -> None:
        os.fsync(handle)

    def close(self, handle: int) -> None:
        os.close(handle)

    def read_regular_bytes(self, path: Path) -> bytes:
        with path.open("rb") as handle:
            return handle.read()

    def open_regular_read(self, path: Path) -> Any:
        return path.open("rb", buffering=0)

    def replace(self, source: Path, destination: Path) -> None:
        os.replace(source, destination)

    def link(self, source: Path, destination: Path) -> None:
        os.link(source, destination)

    def open_directory(self, path: Path, flags: int) -> int:
        return os.open(path, flags)

    def flush_directory(self, handle: int) -> None:
        os.fsync(handle)

    def unlink(self, path: Path) -> None:
        path.unlink()


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _posix_mount_facts(path: Path) -> tuple[str, str]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return (path.anchor or "/", "unknown")
    best_mount = ""
    best_filesystem = "unknown"
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return (path.anchor or "/", "unknown")
    target = path.as_posix()
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount = _decode_mount_path(left_fields[4])
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            if len(mount) >= len(best_mount):
                best_mount = mount
                best_filesystem = right_fields[0]
    return (best_mount or (path.anchor or "/"), best_filesystem)


class PosixDurabilityBackend(_BackendBase):
    """POSIX O_EXCL/fsync/replace/directory-fsync implementation."""

    backend_kind: Literal["posix"] = "posix"

    def __init__(self, *, syscalls: object | None = None) -> None:
        super().__init__(
            syscalls=_RealPosixSyscalls() if syscalls is None else syscalls
        )

    def _create_file(
        self,
        path: Path,
        payload: bytes,
        *,
        operation: Literal["exclusive_create", "replace"],
        target_at_risk: bool,
    ) -> None:
        handle: object | None = None
        phase = "open_exclusive"
        try:
            handle = self._syscalls.open_exclusive(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            phase = "write"
            self._syscalls.write_all(handle, payload)
            phase = "flush_staging" if operation == "replace" else "flush_exclusive"
            self._syscalls.flush_file(handle)
        except FileExistsError as exc:
            raise DurabilityCollisionError(
                f"exclusive-create target already exists: {path}"
            ) from exc
        except OSError as exc:
            indeterminate = handle is not None and target_at_risk
            raise DurabilityOperationError(
                f"{phase} failed for {path}: {exc}",
                phase=phase,
                indeterminate=indeterminate,
                error_code=_error_code(exc),
                obligations=(
                    ("the created file requires investigation",)
                    if indeterminate
                    else ()
                ),
            ) from exc
        finally:
            if handle is not None:
                try:
                    self._syscalls.close(handle)
                except OSError as exc:
                    raise DurabilityOperationError(
                        f"close failed for {path}: {exc}",
                        phase="close",
                        indeterminate=target_at_risk,
                        error_code=_error_code(exc),
                        obligations=(
                            ("the created file requires investigation",)
                            if target_at_risk
                            else ()
                        ),
                    ) from exc

    def _sync_parent(self, parent: Path, *, target_at_risk: bool) -> None:
        handle: object | None = None
        try:
            handle = self._syscalls.open_directory(
                parent, POSIX_DIRECTORY_OPEN_FLAGS
            )
            self._syscalls.flush_directory(handle)
        except OSError as exc:
            raise DurabilityOperationError(
                f"directory fsync failed for {parent}: {exc}",
                phase="directory_fsync",
                indeterminate=target_at_risk,
                error_code=_error_code(exc),
                obligations=(
                    "the published directory entry has unverified crash durability",
                ),
            ) from exc
        finally:
            if handle is not None:
                try:
                    self._syscalls.close(handle)
                except OSError as exc:
                    raise DurabilityOperationError(
                        f"directory close failed for {parent}: {exc}",
                        phase="directory_close",
                        indeterminate=target_at_risk,
                        error_code=_error_code(exc),
                        obligations=(
                            "the published directory entry requires investigation",
                        ),
                    ) from exc

    def _replace(self, staging: Path, destination: Path) -> None:
        self._syscalls.replace(staging, destination)

    def _create_stream_file(
        self,
        path: Path,
        chunks: Iterable[bytes],
    ) -> None:
        handle: object | None = None
        phase = "create_stream_staging"
        try:
            handle = self._syscalls.open_exclusive(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            phase = "write_stream"
            for chunk in chunks:
                self._syscalls.write_all(handle, chunk)
            phase = "flush_stream_staging"
            self._syscalls.flush_file(handle)
        except DurabilityOperationError:
            raise
        except FileExistsError as exc:
            raise DurabilityCollisionError(
                f"stream staging target already exists: {path}"
            ) from exc
        except OSError as exc:
            raise DurabilityOperationError(
                f"{phase} failed for {path}: {exc}",
                phase=phase,
                indeterminate=False,
                error_code=_error_code(exc),
                obligations=(
                    f"the retained staging file must be inspected: {path.name}",
                ),
            ) from exc
        finally:
            if handle is not None:
                try:
                    self._syscalls.close(handle)
                except OSError as exc:
                    raise DurabilityOperationError(
                        f"close failed for {path}: {exc}",
                        phase="close_stream_staging",
                        indeterminate=False,
                        error_code=_error_code(exc),
                        obligations=(
                            f"the retained staging file must be inspected: {path.name}",
                        ),
                    ) from exc

    def _promote_no_replace(
        self,
        staging: Path,
        destination: Path,
    ) -> None:
        self._syscalls.link(staging, destination)
        self._syscalls.unlink(staging)


class _RealWindowsSyscalls:
    def __init__(self) -> None:
        if os.name != "nt":
            raise DurabilityError("real Windows syscalls require os.name == 'nt'")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self._create_file.restype = ctypes.c_void_p
        self._write_file = self._kernel32.WriteFile
        self._write_file.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        )
        self._write_file.restype = ctypes.c_int
        self._flush_file = self._kernel32.FlushFileBuffers
        self._flush_file.argtypes = (ctypes.c_void_p,)
        self._flush_file.restype = ctypes.c_int
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (ctypes.c_void_p,)
        self._close_handle.restype = ctypes.c_int
        self._move_file_ex = self._kernel32.MoveFileExW
        self._move_file_ex.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        self._move_file_ex.restype = ctypes.c_int
        self._delete_file = self._kernel32.DeleteFileW
        self._delete_file.argtypes = (ctypes.c_wchar_p,)
        self._delete_file.restype = ctypes.c_int

    def volume_facts(self, path: Path) -> VolumeFacts:
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        resolved = existing.resolve(strict=True)
        volume_path = ctypes.create_unicode_buffer(32768)
        get_volume_path = self._kernel32.GetVolumePathNameW
        if not get_volume_path(os.fspath(resolved), volume_path, len(volume_path)):
            raise ctypes.WinError(ctypes.get_last_error())
        serial = ctypes.c_uint32()
        maximum_component = ctypes.c_uint32()
        filesystem_flags = ctypes.c_uint32()
        filesystem_name = ctypes.create_unicode_buffer(256)
        get_volume_information = self._kernel32.GetVolumeInformationW
        if not get_volume_information(
            volume_path.value,
            None,
            0,
            ctypes.byref(serial),
            ctypes.byref(maximum_component),
            ctypes.byref(filesystem_flags),
            filesystem_name,
            len(filesystem_name),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        get_drive_type = self._kernel32.GetDriveTypeW
        get_drive_type.argtypes = (ctypes.c_wchar_p,)
        get_drive_type.restype = ctypes.c_uint32
        drive_type = int(get_drive_type(volume_path.value))
        lowered_parts = {part.casefold() for part in resolved.parts}
        cloud_markers = {
            "dropbox",
            "google drive",
            "icloud drive",
            "onedrive",
        }
        return VolumeFacts(
            volume_path=volume_path.value,
            volume_serial=f"{serial.value:08x}",
            filesystem_type=filesystem_name.value or "unknown",
            is_remote=drive_type == 4,
            is_cloud_sync=bool(lowered_parts & cloud_markers),
        )

    def create_file(
        self,
        path: Path,
        *,
        desired_access: int,
        share_mode: int,
        creation_disposition: int,
        flags_and_attributes: int,
    ) -> int:
        handle = self._create_file(
            os.fspath(path),
            desired_access,
            share_mode,
            None,
            creation_disposition,
            flags_and_attributes,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(handle)

    def write_all(self, handle: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 0x7FFFFFFF]
            buffer = ctypes.create_string_buffer(chunk)
            written = ctypes.c_uint32()
            if not self._write_file(
                handle,
                buffer,
                len(chunk),
                ctypes.byref(written),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if written.value == 0:
                raise OSError("WriteFile returned no progress")
            offset += written.value

    def flush_file(self, handle: int) -> None:
        if not self._flush_file(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self, handle: int) -> None:
        if not self._close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def read_regular_bytes(self, path: Path) -> bytes:
        with path.open("rb") as handle:
            return handle.read()

    def open_regular_read(self, path: Path) -> Any:
        return path.open("rb", buffering=0)

    def move_file_ex(
        self, source: Path, destination: Path, flags: int
    ) -> None:
        if not self._move_file_ex(
            os.fspath(source), os.fspath(destination), flags
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def unlink(self, path: Path) -> None:
        if not self._delete_file(os.fspath(path)):
            raise ctypes.WinError(ctypes.get_last_error())


class WindowsDurabilityBackend(_BackendBase):
    """Win32 write-through create/flush/move implementation."""

    backend_kind: Literal["windows"] = "windows"

    def __init__(self, *, syscalls: object | None = None) -> None:
        super().__init__(
            syscalls=_RealWindowsSyscalls() if syscalls is None else syscalls
        )

    def _create_file(
        self,
        path: Path,
        payload: bytes,
        *,
        operation: Literal["exclusive_create", "replace"],
        target_at_risk: bool,
    ) -> None:
        handle: object | None = None
        phase = "create_new"
        try:
            handle = self._syscalls.create_file(
                path,
                desired_access=WINDOWS_GENERIC_WRITE,
                share_mode=0,
                creation_disposition=WINDOWS_CREATE_NEW,
                flags_and_attributes=(
                    WINDOWS_FILE_ATTRIBUTE_NORMAL
                    | WINDOWS_FILE_FLAG_WRITE_THROUGH
                ),
            )
            phase = "write"
            self._syscalls.write_all(handle, payload)
            phase = "flush_staging" if operation == "replace" else "flush_exclusive"
            self._syscalls.flush_file(handle)
        except OSError as exc:
            if _error_code(exc) in (errno.EEXIST, 80, 183):
                raise DurabilityCollisionError(
                    f"exclusive-create target already exists: {path}"
                ) from exc
            indeterminate = handle is not None and target_at_risk
            raise DurabilityOperationError(
                f"{phase} failed for {path}: {exc}",
                phase=phase,
                indeterminate=indeterminate,
                error_code=_error_code(exc),
                obligations=(
                    ("the created file requires investigation",)
                    if indeterminate
                    else ()
                ),
            ) from exc
        finally:
            if handle is not None:
                try:
                    self._syscalls.close(handle)
                except OSError as exc:
                    raise DurabilityOperationError(
                        f"CloseHandle failed for {path}: {exc}",
                        phase="close",
                        indeterminate=target_at_risk,
                        error_code=_error_code(exc),
                        obligations=(
                            ("the created file requires investigation",)
                            if target_at_risk
                            else ()
                        ),
                    ) from exc

    def _sync_parent(self, parent: Path, *, target_at_risk: bool) -> None:
        del parent, target_at_risk

    def _replace(self, staging: Path, destination: Path) -> None:
        self._syscalls.move_file_ex(
            staging,
            destination,
            WINDOWS_MOVEFILE_REPLACE_EXISTING
            | WINDOWS_MOVEFILE_WRITE_THROUGH,
        )

    def _create_stream_file(
        self,
        path: Path,
        chunks: Iterable[bytes],
    ) -> None:
        handle: object | None = None
        phase = "create_stream_staging"
        try:
            handle = self._syscalls.create_file(
                path,
                desired_access=WINDOWS_GENERIC_WRITE,
                share_mode=0,
                creation_disposition=WINDOWS_CREATE_NEW,
                flags_and_attributes=(
                    WINDOWS_FILE_ATTRIBUTE_NORMAL
                    | WINDOWS_FILE_FLAG_WRITE_THROUGH
                ),
            )
            phase = "write_stream"
            for chunk in chunks:
                self._syscalls.write_all(handle, chunk)
            phase = "flush_stream_staging"
            self._syscalls.flush_file(handle)
        except DurabilityOperationError:
            raise
        except OSError as exc:
            if _error_code(exc) in (errno.EEXIST, 80, 183):
                raise DurabilityCollisionError(
                    f"stream staging target already exists: {path}"
                ) from exc
            raise DurabilityOperationError(
                f"{phase} failed for {path}: {exc}",
                phase=phase,
                indeterminate=False,
                error_code=_error_code(exc),
                obligations=(
                    f"the retained staging file must be inspected: {path.name}",
                ),
            ) from exc
        finally:
            if handle is not None:
                try:
                    self._syscalls.close(handle)
                except OSError as exc:
                    raise DurabilityOperationError(
                        f"CloseHandle failed for {path}: {exc}",
                        phase="close_stream_staging",
                        indeterminate=False,
                        error_code=_error_code(exc),
                        obligations=(
                            f"the retained staging file must be inspected: {path.name}",
                        ),
                    ) from exc

    def _promote_no_replace(
        self,
        staging: Path,
        destination: Path,
    ) -> None:
        self._syscalls.move_file_ex(
            staging,
            destination,
            WINDOWS_MOVEFILE_WRITE_THROUGH,
        )


def probe_durability(root: Path) -> DurabilityIdentity:
    """Probe the current platform explicitly; unknown platforms fail closed."""

    if os.name == "nt":
        return WindowsDurabilityBackend().probe(root)
    if os.name == "posix":
        return PosixDurabilityBackend().probe(root)
    raise DurabilityError(
        f"no durability backend is configured for platform {os.name!r}"
    )


def create_canonical_json(
    path: Path,
    value: object,
    *,
    backend: DurabilityBackend,
) -> DurableFileIdentity:
    """Canonicalize one generic JSON value and exclusive-create it durably."""

    if not isinstance(backend, DurabilityBackend):
        raise DurabilityError("backend does not implement DurabilityBackend")
    return backend.create_exclusive(path, canonical_json_bytes_generic(value))


def replace_canonical_json(
    path: Path,
    value: object,
    *,
    backend: DurabilityBackend,
) -> DurableFileIdentity:
    """Canonicalize one generic JSON value and durably replace its target."""

    if not isinstance(backend, DurabilityBackend):
        raise DurabilityError("backend does not implement DurabilityBackend")
    return backend.replace_durable(path, canonical_json_bytes_generic(value))


__all__ = [
    "ContentAddressedDurabilityBackend",
    "DEFAULT_STREAM_CHUNK_SIZE_LIMIT",
    "DEFAULT_STREAM_REOPEN_BLOCK_SIZE",
    "DurabilityBackend",
    "DurabilityCollisionError",
    "DurabilityError",
    "DurabilityErrorRecord",
    "DurabilityIdentity",
    "DurabilityOperationError",
    "DurableFileIdentity",
    "POSIX_DIRECTORY_OPEN_FLAGS",
    "PosixDurabilityBackend",
    "VolumeFacts",
    "WINDOWS_CREATE_NEW",
    "WINDOWS_FILE_ATTRIBUTE_NORMAL",
    "WINDOWS_FILE_FLAG_WRITE_THROUGH",
    "WINDOWS_GENERIC_WRITE",
    "WINDOWS_MOVEFILE_REPLACE_EXISTING",
    "WINDOWS_MOVEFILE_WRITE_THROUGH",
    "WindowsDurabilityBackend",
    "VerifiedFileFacts",
    "canonical_json_bytes_generic",
    "create_canonical_json",
    "probe_durability",
    "replace_canonical_json",
    "validate_regular_nonlink_sha256",
]
