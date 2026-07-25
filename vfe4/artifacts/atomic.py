"""Fail-closed, whole-directory atomic JSON artifact publication."""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import hashlib
import json
import math
import os
import stat
import sys
import unicodedata
import uuid
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch


class ArtifactPublicationError(RuntimeError):
    """A run could not be durably published as one complete directory."""


ATOMIC_FAILURE_CLEANUP_OBLIGATIONS = (
    (
        "failed staging directories are retained because portable "
        "handle-bound recursive deletion of the captured top directory "
        "cannot be proven race-free"
    ),
)


def _json_value(value: object) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactPublicationError("JSON mapping keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactPublicationError("nonfinite floats cannot be serialized")
        return value
    raise ArtifactPublicationError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize supported records recursively with exact JSON settings."""
    try:
        return json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ArtifactPublicationError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactPublicationError(f"JSON serialization failed: {exc}") from exc


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


_INVALID_WINDOWS_CHARACTERS = frozenset('<>:"|?*')
_RESERVED_WINDOWS_STEMS = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
_RESERVED_WINDOWS_STEMS.update(f"COM{index}" for index in range(1, 10))
_RESERVED_WINDOWS_STEMS.update(f"LPT{index}" for index in range(1, 10))
_RESERVED_WINDOWS_STEMS.update(f"COM{digit}" for digit in ("¹", "²", "³"))
_RESERVED_WINDOWS_STEMS.update(f"LPT{digit}" for digit in ("¹", "²", "³"))


def _portable_component(name: str, *, context: str) -> str:
    if type(name) is not str or not name:
        raise ArtifactPublicationError(f"{context} must be a nonempty component")
    if unicodedata.normalize("NFC", name) != name:
        raise ArtifactPublicationError(f"{context} must use canonical Unicode spelling")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        name in (".", "..")
        or "/" in name
        or "\\" in name
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or len(posix.parts) != 1
        or len(windows.parts) != 1
        or name.endswith((".", " "))
        or any(
            ord(character) < 32 or character in _INVALID_WINDOWS_CHARACTERS
            for character in name
        )
    ):
        raise ArtifactPublicationError(f"{context} is not a portable canonical component")
    if name.split(".", 1)[0].upper() in _RESERVED_WINDOWS_STEMS:
        raise ArtifactPublicationError(f"{context} is reserved on Windows")
    return name


def _payload_path(name: str) -> PurePosixPath:
    if type(name) is not str or not name or "\\" in name:
        raise ArtifactPublicationError("artifact names must be nonempty canonical POSIX paths")
    if unicodedata.normalize("NFC", name) != name:
        raise ArtifactPublicationError("artifact paths must use canonical Unicode spelling")
    path = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or path.as_posix() != name
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ArtifactPublicationError("artifact path is noncanonical or escapes the run")
    for component in path.parts:
        _portable_component(component, context="artifact path component")
    if path.name.casefold() == "manifest.sha256":
        raise ArtifactPublicationError("artifact manifest path is reserved")
    if path.suffix != ".json":
        raise ArtifactPublicationError("artifact payloads must be JSON files")
    return path


def _validated_payloads(
    payloads: Mapping[str, object],
) -> list[tuple[PurePosixPath, object]]:
    paths: list[tuple[PurePosixPath, object]] = []
    aliases: dict[tuple[str, ...], str] = {}
    for name, value in payloads.items():
        relative = _payload_path(name)
        alias = tuple(component.casefold() for component in relative.parts)
        if alias in aliases:
            raise ArtifactPublicationError(
                f"artifact payload paths collide portably: {aliases[alias]!r} and {name!r}"
            )
        aliases[alias] = name
        paths.append((relative, value))
    return sorted(paths, key=lambda item: item[0].as_posix())


def _run_component(name: str) -> str:
    """Return one portable, canonical directory component for a run."""
    try:
        return _portable_component(name, context="run_name")
    except ArtifactPublicationError as exc:
        if "run_name" in str(exc):
            raise
        raise ArtifactPublicationError(f"run_name is invalid: {exc}") from exc


def _destination_entry_exists(path: Path) -> bool:
    """Return whether the directory entry exists, including dangling links."""

    return os.path.lexists(os.fspath(path))


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


@dataclasses.dataclass(frozen=True, slots=True)
class _OwnedDirectoryIdentity:
    device: int
    inode: int
    marker_name: str | None = None
    marker_token: bytes | None = None
    marker_device: int | None = None
    marker_inode: int | None = None

    def matches(self, path: Path, *, require_marker: bool) -> bool:
        try:
            directory_status = path.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(directory_status.st_mode)
            or _is_redirect_or_reparse(path, directory_status)
            or directory_status.st_dev != self.device
            or directory_status.st_ino != self.inode
        ):
            return False
        if (
            self.marker_name is None
            or self.marker_token is None
            or self.marker_device is None
            or self.marker_inode is None
        ):
            return not require_marker
        marker = path / self.marker_name
        try:
            marker_status = marker.lstat()
        except FileNotFoundError:
            return not require_marker
        except OSError:
            return False
        if not require_marker:
            return False
        if (
            not stat.S_ISREG(marker_status.st_mode)
            or _is_redirect_or_reparse(marker, marker_status)
            or marker_status.st_dev != self.marker_device
            or marker_status.st_ino != self.marker_inode
        ):
            return False
        try:
            return marker.read_bytes() == self.marker_token
        except OSError:
            return False


def _capture_owned_directory(path: Path) -> _OwnedDirectoryIdentity:
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or _is_redirect_or_reparse(path, status)
        or status.st_ino == 0
    ):
        raise ArtifactPublicationError(
            "exclusive staging directory lacks a stable safe identity"
        )
    return _OwnedDirectoryIdentity(
        device=status.st_dev,
        inode=status.st_ino,
    )


def _install_ownership_marker(
    path: Path,
    identity: _OwnedDirectoryIdentity,
) -> _OwnedDirectoryIdentity:
    if not identity.matches(path, require_marker=False):
        raise ArtifactPublicationError(
            "exclusive staging ownership changed before marker creation"
        )
    marker_name = f".owner-{uuid.uuid4().hex}"
    marker_token = uuid.uuid4().hex.encode("ascii")
    marker = path / marker_name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(marker, flags, 0o600)
    try:
        if os.write(descriptor, marker_token) != len(marker_token):
            raise ArtifactPublicationError(
                "exclusive staging ownership marker write was incomplete"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    marker_status = marker.lstat()
    if (
        not stat.S_ISREG(marker_status.st_mode)
        or _is_redirect_or_reparse(marker, marker_status)
        or marker_status.st_ino == 0
    ):
        raise ArtifactPublicationError(
            "exclusive staging ownership marker is not a safe regular file"
        )
    marked_identity = _OwnedDirectoryIdentity(
        device=identity.device,
        inode=identity.inode,
        marker_name=marker_name,
        marker_token=marker_token,
        marker_device=marker_status.st_dev,
        marker_inode=marker_status.st_ino,
    )
    if not marked_identity.matches(path, require_marker=True):
        raise ArtifactPublicationError(
            "exclusive staging ownership identity changed during capture"
        )
    return marked_identity


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically move one directory while refusing any existing destination.

    Python's portable ``os.rename`` permits replacement of an empty directory
    on some POSIX systems.  Publication needs the stronger no-replace
    primitive supplied by Windows, Linux, and Darwin.  Unsupported POSIX
    kernels fail closed instead of falling back to a racy exists/rename pair.
    """

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
        move_file_ex.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        move_file_ex.restype = ctypes.c_int
        if move_file_ex(os.fspath(source), os.fspath(destination), 0):
            return
        error_code = ctypes.get_last_error()
        if _destination_entry_exists(destination):
            raise ArtifactPublicationError(
                f"run directory already exists: {destination}"
            )
        raise OSError(
            error_code,
            f"atomic no-replace directory move failed: {destination}",
        )

    if os.name != "posix":
        raise ArtifactPublicationError(
            "atomic no-replace directory publication is unsupported "
            f"on platform {os.name!r}"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result: int
    if sys.platform.startswith("linux"):
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise ArtifactPublicationError(
                "Linux libc lacks atomic renameat2(RENAME_NOREPLACE)"
            ) from exc
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    elif sys.platform == "darwin":
        try:
            renamex_np = libc.renamex_np
        except AttributeError as exc:
            raise ArtifactPublicationError(
                "Darwin libc lacks atomic renamex_np(RENAME_EXCL)"
            ) from exc
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise ArtifactPublicationError(
            "this POSIX platform lacks a configured atomic no-replace "
            "directory primitive"
        )
    if result == 0:
        return
    error_code = ctypes.get_errno()
    if error_code in (errno.EEXIST, errno.ENOTEMPTY) or (
        _destination_entry_exists(destination)
    ):
        raise ArtifactPublicationError(
            f"run directory already exists: {destination}"
        )
    raise OSError(
        error_code,
        f"atomic no-replace directory move failed: {destination}",
    )


def publish_run_directory(
    run_root: Path, run_name: str, payloads: Mapping[str, object]
) -> Path:
    """Publish every JSON plus a manifest by one absent-directory rename.

    Failed stages are retained under their private names; see
    ``ATOMIC_FAILURE_CLEANUP_OBLIGATIONS``.
    """
    if not isinstance(run_root, Path):
        raise ArtifactPublicationError("run_root and run_name are required")
    if not isinstance(payloads, Mapping) or not payloads:
        raise ArtifactPublicationError("payloads must be a nonempty mapping")
    safe_run_name = _run_component(run_name)
    paths = _validated_payloads(payloads)
    staging: Path | None = None
    staging_identity: _OwnedDirectoryIdentity | None = None
    try:
        root = run_root.resolve()
        final = (root / safe_run_name).resolve()
        if final.parent != root:
            raise ArtifactPublicationError("run_name escapes its resolved run_root")
        staging = root / f".staging-{uuid.uuid4().hex}"
        resolved_staging = staging.resolve()
        targets: list[tuple[PurePosixPath, Path, object]] = []
        target_aliases: set[tuple[str, ...]] = set()
        for relative, value in paths:
            target = (staging / Path(*relative.parts)).resolve()
            try:
                target_relative = target.relative_to(resolved_staging)
            except ValueError as exc:
                raise ArtifactPublicationError(
                    "artifact payload path escapes staging"
                ) from exc
            if not target_relative.parts:
                raise ArtifactPublicationError("artifact payload path must be under staging")
            target_alias = tuple(part.casefold() for part in target.parts)
            if target_alias in target_aliases:
                raise ArtifactPublicationError("resolved artifact payload paths collide")
            target_aliases.add(target_alias)
            targets.append((relative, target, value))
        root.mkdir(parents=True, exist_ok=True)
        if _destination_entry_exists(final):
            raise ArtifactPublicationError(f"run directory already exists: {final}")
        staging.mkdir()
        staging_identity = _capture_owned_directory(staging)
        staging_identity = _install_ownership_marker(
            staging, staging_identity
        )
        for _, target, value in targets:
            _atomic_write_bytes(target, canonical_json_bytes(value))
        manifest_lines = []
        for relative, target, _ in targets:
            data = target.read_bytes()
            manifest_lines.append(f"{hashlib.sha256(data).hexdigest()}  {relative.as_posix()}\n")
        _atomic_write_bytes(
            staging / "manifest.sha256", "".join(manifest_lines).encode("utf-8")
        )
        if not staging_identity.matches(staging, require_marker=True):
            raise ArtifactPublicationError(
                "exclusive staging ownership changed before commit"
            )
        marker_name = staging_identity.marker_name
        if marker_name is None:
            raise ArtifactPublicationError(
                "exclusive staging ownership marker is unavailable"
            )
        (staging / marker_name).unlink()
        if not staging_identity.matches(staging, require_marker=False):
            raise ArtifactPublicationError(
                "exclusive staging ownership changed before final commit"
            )
        _rename_directory_no_replace(staging, final)
        if not staging_identity.matches(final, require_marker=False):
            raise ArtifactPublicationError(
                "installed final directory identity differs from the "
                "captured staging directory"
            )
        return final
    except ArtifactPublicationError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError, TypeError) as exc:
        raise ArtifactPublicationError(f"artifact publication failed: {exc}") from exc
