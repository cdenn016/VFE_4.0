"""Fail-closed, whole-directory atomic JSON artifact publication."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import unicodedata
import uuid
from collections.abc import Mapping
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class ArtifactPublicationError(RuntimeError):
    """A run could not be durably published as one complete directory."""


def _json_value(value: object) -> Any:
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


def publish_run_directory(
    run_root: Path, run_name: str, payloads: Mapping[str, object]
) -> Path:
    """Publish every JSON plus a final manifest by one absent-directory rename."""
    if not isinstance(run_root, Path):
        raise ArtifactPublicationError("run_root and run_name are required")
    if not isinstance(payloads, Mapping) or not payloads:
        raise ArtifactPublicationError("payloads must be a nonempty mapping")
    safe_run_name = _run_component(run_name)
    paths = _validated_payloads(payloads)
    staging: Path | None = None
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
        if final.exists():
            raise ArtifactPublicationError(f"run directory already exists: {final}")
        staging.mkdir()
        for _, target, value in targets:
            _atomic_write_bytes(target, canonical_json_bytes(value))
        manifest_lines = []
        for relative, target, _ in targets:
            data = target.read_bytes()
            manifest_lines.append(f"{hashlib.sha256(data).hexdigest()}  {relative.as_posix()}\n")
        _atomic_write_bytes(
            staging / "manifest.sha256", "".join(manifest_lines).encode("utf-8")
        )
        if final.exists():
            raise ArtifactPublicationError(f"run directory already exists: {final}")
        os.rename(staging, final)
        return final
    except ArtifactPublicationError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError, TypeError) as exc:
        raise ArtifactPublicationError(f"artifact publication failed: {exc}") from exc
    finally:
        if staging is not None and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
