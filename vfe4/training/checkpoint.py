"""Deterministic, no-replace H6 checkpoint envelopes.

The envelope retains canonical bytes for the exact attempt, cursor, and
objective records.  Loading is intentionally not a discovery operation: the
caller must supply the three expected typed records, and their canonical bytes
must match before any active state is returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vfe4.types.h6 import canonical_json_bytes

from .language import H6AttemptCursor, H6AttemptSpec, H6ObjectiveManifest


_MAGIC = b"VFE4-H6-CHECKPOINT-V2\x00"
_TRAILER_BYTES = hashlib.sha256().digest_size
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_STATE_COUNT = 100_000
_LOWER_HEX = frozenset("0123456789abcdef")
_RECORD_NAMES = ("attempt_spec", "cursor", "objective_manifest")
_HEADER_KEYS = frozenset(
    {
        "checkpoint_schema",
        "records",
        "record_manifest",
        "state_manifest",
        "checkpoint_sha256",
    }
)
_ROLE_PREFIXES = {
    "model": "model.",
    "recognition_store": "recognition_store.",
    "optimizer_model_ce": "optimizer.model_ce_adamw.",
    "optimizer_model": "optimizer.model_adamw.",
    "optimizer_recognition": "optimizer.recognition_adamw.",
}
_SUFFIX = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*\Z")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _canonical_object(raw: bytes, name: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"{name} must be one canonical JSON object")
    return value


def _record_bytes(
    record: H6AttemptSpec | H6AttemptCursor | H6ObjectiveManifest,
) -> bytes:
    """Return one record's explicit canonical payload bytes."""

    record.__post_init__()
    canonical_bytes = getattr(record, "canonical_bytes", None)
    if callable(canonical_bytes):
        raw = canonical_bytes()
    else:
        canonical_payload = getattr(record, "canonical_payload", None)
        if not callable(canonical_payload):
            raise ValueError(
                f"{type(record).__name__} must expose canonical_payload()"
            )
        raw = canonical_json_bytes(canonical_payload())
    if type(raw) is not bytes:
        raise ValueError("canonical record serialization must return bytes")
    _canonical_object(raw, type(record).__name__)
    return raw


def _record_identity(
    record: H6AttemptSpec | H6AttemptCursor | H6ObjectiveManifest,
) -> str:
    if type(record) is H6AttemptSpec:
        return _require_sha256(
            record.attempt_spec_sha256, "attempt_spec_sha256"
        )
    if type(record) is H6AttemptCursor:
        return _require_sha256(record.cursor_sha256, "cursor_sha256")
    if type(record) is H6ObjectiveManifest:
        return _require_sha256(
            record.objective_manifest_sha256,
            "objective_manifest_sha256",
        )
    raise ValueError("unsupported H6 checkpoint record")


def _state_role(name: str, *, latent_enabled: bool) -> str:
    if type(name) is not str or not name or not name.isascii():
        raise ValueError("checkpoint state name must be nonempty ASCII")
    matches = tuple(
        (role, prefix)
        for role, prefix in _ROLE_PREFIXES.items()
        if name.startswith(prefix)
    )
    if len(matches) != 1:
        raise ValueError("checkpoint state has an unknown active-role prefix")
    role, prefix = matches[0]
    suffix = name[len(prefix) :]
    if not suffix or _SUFFIX.fullmatch(suffix) is None:
        raise ValueError("checkpoint state suffix is not canonical")
    allowed = (
        {
            "model",
            "recognition_store",
            "optimizer_model",
            "optimizer_recognition",
        }
        if latent_enabled
        else {"model", "optimizer_model_ce"}
    )
    if role not in allowed:
        raise ValueError(
            "checkpoint state role is inactive for the endpoint phase schedule"
        )
    return role


def _required_roles(*, latent_enabled: bool) -> frozenset[str]:
    return (
        frozenset(
            {
                "model",
                "recognition_store",
                "optimizer_model",
                "optimizer_recognition",
            }
        )
        if latent_enabled
        else frozenset({"model", "optimizer_model_ce"})
    )


@dataclass(frozen=True, slots=True)
class _CheckpointState:
    role: str
    name: str
    byte_length: int
    raw_bytes_sha256: str
    _raw_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role not in _ROLE_PREFIXES:
            raise ValueError("unsupported checkpoint state role")
        if type(self._raw_bytes) is not bytes:
            raise ValueError("checkpoint state must own immutable bytes")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ValueError("checkpoint state byte_length must be nonnegative")
        if self.byte_length != len(self._raw_bytes):
            raise ValueError("checkpoint state byte length mismatch")
        _require_sha256(self.raw_bytes_sha256, "raw_bytes_sha256")
        if hashlib.sha256(self._raw_bytes).hexdigest() != self.raw_bytes_sha256:
            raise ValueError("checkpoint state raw-byte digest mismatch")

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        raw_bytes: bytes,
        latent_enabled: bool,
    ) -> "_CheckpointState":
        if type(raw_bytes) is not bytes:
            raise ValueError("active checkpoint states must be immutable bytes")
        role = _state_role(name, latent_enabled=latent_enabled)
        owned = bytes(raw_bytes)
        return cls(
            role,
            name,
            len(owned),
            hashlib.sha256(owned).hexdigest(),
            owned,
        )

    def bytes(self) -> bytes:
        self.__post_init__()
        return bytes(self._raw_bytes)

    def manifest_item(self) -> tuple[str, str, int, str]:
        self.__post_init__()
        return (
            self.role,
            self.name,
            self.byte_length,
            self.raw_bytes_sha256,
        )


@dataclass(frozen=True, slots=True)
class H6CheckpointManifest:
    """Typed exact-resume checkpoint with retained canonical record bytes."""

    checkpoint_schema: Literal["h6-checkpoint-v2"]
    attempt_spec: H6AttemptSpec
    cursor: H6AttemptCursor
    objective_manifest: H6ObjectiveManifest
    attempt_spec_sha256: str
    cursor_sha256: str
    objective_manifest_sha256: str
    record_manifest: tuple[tuple[str, int, str, str], ...]
    state_manifest: tuple[tuple[str, str, int, str], ...]
    checkpoint_sha256: str
    _record_bytes: tuple[tuple[str, bytes], ...] = field(
        repr=False, compare=False
    )
    _states: tuple[_CheckpointState, ...] = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.checkpoint_schema != "h6-checkpoint-v2":
            raise ValueError("unsupported H6 checkpoint schema")
        if type(self.attempt_spec) is not H6AttemptSpec:
            raise ValueError("attempt_spec must be an exact H6AttemptSpec")
        if type(self.cursor) is not H6AttemptCursor:
            raise ValueError("cursor must be an exact H6AttemptCursor")
        if type(self.objective_manifest) is not H6ObjectiveManifest:
            raise ValueError(
                "objective_manifest must be an exact H6ObjectiveManifest"
            )
        for record in (
            self.attempt_spec,
            self.cursor,
            self.objective_manifest,
        ):
            record.__post_init__()
        if (
            self.cursor.attempt_spec_sha256
            != self.attempt_spec.attempt_spec_sha256
            or self.objective_manifest.attempt_spec_sha256
            != self.attempt_spec.attempt_spec_sha256
        ):
            raise ValueError("checkpoint records do not share one attempt")
        if (
            self.cursor.phase_schedule_sha256
            != self.attempt_spec.phase_schedule_sha256
            or self.cursor.latent_enabled
            is not self.attempt_spec.latent_enabled
            or self.objective_manifest.endpoint_config_sha256
            != self.attempt_spec.endpoint_config_sha256
            or self.objective_manifest.inventory_sha256
            != self.attempt_spec.objective_inventory_sha256
            or self.objective_manifest.adapter_sha256
            != self.attempt_spec.objective_adapter_sha256
            or self.objective_manifest.objective_kind
            != self.attempt_spec.objective_kind
        ):
            raise ValueError(
                "checkpoint cursor/objective records do not match the attempt"
            )
        detached_sha = (
            self.objective_manifest.detached_recognition_snapshot_sha256
        )
        if self.attempt_spec.latent_enabled != (detached_sha is not None):
            raise ValueError(
                "checkpoint recognition snapshot applicability is inconsistent"
            )
        if (
            self.attempt_spec_sha256
            != self.attempt_spec.attempt_spec_sha256
            or self.cursor_sha256 != self.cursor.cursor_sha256
            or self.objective_manifest_sha256
            != self.objective_manifest.objective_manifest_sha256
        ):
            raise ValueError("checkpoint record identities are inconsistent")

        records = (
            ("attempt_spec", self.attempt_spec),
            ("cursor", self.cursor),
            ("objective_manifest", self.objective_manifest),
        )
        expected_record_bytes = tuple(
            (name, _record_bytes(record)) for name, record in records
        )
        if self._record_bytes != expected_record_bytes:
            raise ValueError("retained canonical checkpoint records changed")
        expected_record_manifest = tuple(
            (
                name,
                len(raw),
                hashlib.sha256(raw).hexdigest(),
                _record_identity(record),
            )
            for (name, record), (_, raw) in zip(
                records, expected_record_bytes, strict=True
            )
        )
        if self.record_manifest != expected_record_manifest:
            raise ValueError("checkpoint record manifest is inconsistent")

        if (
            type(self._states) is not tuple
            or not self._states
            or any(type(state) is not _CheckpointState for state in self._states)
            or len(self._states) > _MAX_STATE_COUNT
        ):
            raise ValueError("checkpoint requires exact active state records")
        names = tuple(state.name for state in self._states)
        if (
            len(set(names)) != len(names)
            or len({name.casefold() for name in names}) != len(names)
        ):
            raise ValueError("checkpoint state names must be unique")
        latent_enabled = self.attempt_spec.latent_enabled
        if type(latent_enabled) is not bool:
            raise ValueError("attempt latent_enabled must be boolean")
        observed_roles: set[str] = set()
        for state in self._states:
            state.__post_init__()
            expected_role = _state_role(
                state.name, latent_enabled=latent_enabled
            )
            if state.role != expected_role:
                raise ValueError("checkpoint state role does not match its name")
            observed_roles.add(state.role)
        if observed_roles != _required_roles(
            latent_enabled=latent_enabled
        ):
            raise ValueError(
                "checkpoint must contain every and only active state role"
            )
        expected_order = tuple(
            sorted(self._states, key=lambda item: (item.role, item.name))
        )
        if self._states != expected_order:
            raise ValueError("checkpoint state records must be canonical-sorted")
        expected_state_manifest = tuple(
            state.manifest_item() for state in self._states
        )
        if self.state_manifest != expected_state_manifest:
            raise ValueError("checkpoint state manifest is inconsistent")

        if self.checkpoint_sha256 != _owned_hash(
            "vfe4.h6.checkpoint-manifest.v2",
            self.canonical_payload(include_checkpoint_sha256=False),
        ):
            raise ValueError("checkpoint manifest digest mismatch")

    @classmethod
    def capture(
        cls,
        *,
        attempt_spec: H6AttemptSpec,
        cursor: H6AttemptCursor,
        objective_manifest: H6ObjectiveManifest,
        active_state_bytes: tuple[tuple[str, bytes], ...],
    ) -> "H6CheckpointManifest":
        if type(attempt_spec) is not H6AttemptSpec:
            raise ValueError("attempt_spec must be an exact H6AttemptSpec")
        if type(cursor) is not H6AttemptCursor:
            raise ValueError("cursor must be an exact H6AttemptCursor")
        if type(objective_manifest) is not H6ObjectiveManifest:
            raise ValueError(
                "objective_manifest must be an exact H6ObjectiveManifest"
            )
        attempt_spec.__post_init__()
        cursor.__post_init__()
        objective_manifest.__post_init__()
        if (
            type(active_state_bytes) is not tuple
            or not active_state_bytes
            or len(active_state_bytes) > _MAX_STATE_COUNT
        ):
            raise ValueError("active_state_bytes must be a bounded nonempty tuple")
        states = tuple(
            sorted(
                (
                    _CheckpointState.capture(
                        name=name,
                        raw_bytes=raw_bytes,
                        latent_enabled=attempt_spec.latent_enabled,
                    )
                    for name, raw_bytes in active_state_bytes
                ),
                key=lambda item: (item.role, item.name),
            )
        )
        records = (
            ("attempt_spec", attempt_spec),
            ("cursor", cursor),
            ("objective_manifest", objective_manifest),
        )
        retained = tuple(
            (name, _record_bytes(record)) for name, record in records
        )
        record_manifest = tuple(
            (
                name,
                len(raw),
                hashlib.sha256(raw).hexdigest(),
                _record_identity(record),
            )
            for (name, record), (_, raw) in zip(
                records, retained, strict=True
            )
        )
        state_manifest = tuple(
            state.manifest_item() for state in states
        )
        values: dict[str, object] = {
            "checkpoint_schema": "h6-checkpoint-v2",
            "attempt_spec": attempt_spec,
            "cursor": cursor,
            "objective_manifest": objective_manifest,
            "attempt_spec_sha256": attempt_spec.attempt_spec_sha256,
            "cursor_sha256": cursor.cursor_sha256,
            "objective_manifest_sha256": (
                objective_manifest.objective_manifest_sha256
            ),
            "record_manifest": record_manifest,
            "state_manifest": state_manifest,
            "_record_bytes": retained,
            "_states": states,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        digest = _owned_hash(
            "vfe4.h6.checkpoint-manifest.v2",
            provisional.canonical_payload(
                include_checkpoint_sha256=False
            ),
        )
        return cls(**values, checkpoint_sha256=digest)

    def canonical_payload(
        self, *, include_checkpoint_sha256: bool = True
    ) -> dict[str, object]:
        records = {
            name: _canonical_object(raw, name)
            for name, raw in self._record_bytes
        }
        payload: dict[str, object] = {
            "checkpoint_schema": self.checkpoint_schema,
            "records": records,
            "record_manifest": self.record_manifest,
            "state_manifest": self.state_manifest,
        }
        if include_checkpoint_sha256:
            payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload

    def state_bytes(self, name: str) -> bytes:
        self.__post_init__()
        for state in self._states:
            if state.name == name:
                return state.bytes()
        raise KeyError(name)

    def to_bytes(self) -> bytes:
        self.__post_init__()
        header = canonical_json_bytes(self.canonical_payload())
        payload = b"".join(state.bytes() for state in self._states)
        body = _MAGIC + len(header).to_bytes(8, "big") + header + payload
        return body + hashlib.sha256(body).digest()

    def assert_resume_identity(
        self,
        *,
        attempt_spec: H6AttemptSpec,
        cursor: H6AttemptCursor,
        objective_manifest: H6ObjectiveManifest,
    ) -> None:
        expected = (
            _record_bytes(attempt_spec),
            _record_bytes(cursor),
            _record_bytes(objective_manifest),
        )
        observed = tuple(raw for _, raw in self._record_bytes)
        if observed != expected:
            raise ValueError("checkpoint exact resume identity mismatch")


def _manifest_from_bytes(
    raw: bytes,
    *,
    expected_attempt_spec: H6AttemptSpec,
    expected_cursor: H6AttemptCursor,
    expected_objective_manifest: H6ObjectiveManifest,
) -> H6CheckpointManifest:
    if type(raw) is not bytes:
        raise ValueError("checkpoint input must be immutable bytes")
    minimum = len(_MAGIC) + 8 + _TRAILER_BYTES
    if len(raw) < minimum or not raw.startswith(_MAGIC):
        raise ValueError("checkpoint schema or integrity marker is invalid")
    body = raw[:-_TRAILER_BYTES]
    trailer = raw[-_TRAILER_BYTES:]
    if hashlib.sha256(body).digest() != trailer:
        raise ValueError("checkpoint integrity digest mismatch; possible tamper")
    offset = len(_MAGIC)
    header_length = int.from_bytes(raw[offset : offset + 8], "big")
    header_start = offset + 8
    header_end = header_start + header_length
    if (
        header_length <= 0
        or header_length > _MAX_HEADER_BYTES
        or header_end > len(body)
    ):
        raise ValueError("checkpoint integrity header length is invalid")
    header_bytes = raw[header_start:header_end]
    header = _canonical_object(header_bytes, "checkpoint header")
    if set(header) != _HEADER_KEYS:
        raise ValueError("checkpoint header contains missing or unknown fields")
    if header["checkpoint_schema"] != "h6-checkpoint-v2":
        raise ValueError("unsupported H6 checkpoint schema")

    records = header["records"]
    if type(records) is not dict or set(records) != set(_RECORD_NAMES):
        raise ValueError("checkpoint record inventory is not exact")
    try:
        loaded_attempt_spec = H6AttemptSpec.from_payload(
            records["attempt_spec"]
        )
        loaded_cursor = H6AttemptCursor.from_payload(records["cursor"])
        loaded_objective_manifest = H6ObjectiveManifest.from_payload(
            records["objective_manifest"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint typed record reconstruction failed"
        ) from exc
    loaded_records = (
        ("attempt_spec", loaded_attempt_spec, expected_attempt_spec),
        ("cursor", loaded_cursor, expected_cursor),
        (
            "objective_manifest",
            loaded_objective_manifest,
            expected_objective_manifest,
        ),
    )
    for name, loaded, expected in loaded_records:
        if (
            canonical_json_bytes(records[name]) != _record_bytes(loaded)
            or _record_bytes(loaded) != _record_bytes(expected)
            or _record_identity(loaded) != _record_identity(expected)
        ):
            raise ValueError(f"checkpoint {name} identity mismatch")

    record_manifest = header["record_manifest"]
    state_manifest = header["state_manifest"]
    if (
        type(record_manifest) is not list
        or len(record_manifest) != len(_RECORD_NAMES)
        or any(type(item) is not list or len(item) != 4 for item in record_manifest)
    ):
        raise ValueError("checkpoint record manifest is malformed")
    if (
        type(state_manifest) is not list
        or not state_manifest
        or len(state_manifest) > _MAX_STATE_COUNT
        or any(type(item) is not list or len(item) != 4 for item in state_manifest)
    ):
        raise ValueError("checkpoint state manifest is malformed")

    payload = body[header_end:]
    cursor_offset = 0
    active: list[tuple[str, bytes]] = []
    for role, name, byte_length, digest in state_manifest:
        if (
            type(role) is not str
            or type(name) is not str
            or type(byte_length) is not int
            or byte_length < 0
            or type(digest) is not str
        ):
            raise ValueError("checkpoint state manifest fields are invalid")
        _require_sha256(digest, "checkpoint state digest")
        end = cursor_offset + byte_length
        state_bytes = payload[cursor_offset:end]
        if (
            end > len(payload)
            or hashlib.sha256(state_bytes).hexdigest() != digest
        ):
            raise ValueError("checkpoint state digest mismatch; possible tamper")
        expected_role = _state_role(
            name,
            latent_enabled=loaded_attempt_spec.latent_enabled,
        )
        if role != expected_role:
            raise ValueError("checkpoint state role/name mismatch")
        active.append((name, state_bytes))
        cursor_offset = end
    if cursor_offset != len(payload):
        raise ValueError("checkpoint contains unbound trailing state bytes")

    manifest = H6CheckpointManifest.capture(
        attempt_spec=loaded_attempt_spec,
        cursor=loaded_cursor,
        objective_manifest=loaded_objective_manifest,
        active_state_bytes=tuple(active),
    )
    expected_record_manifest = [
        list(item) for item in manifest.record_manifest
    ]
    expected_state_manifest = [
        list(item) for item in manifest.state_manifest
    ]
    if (
        record_manifest != expected_record_manifest
        or state_manifest != expected_state_manifest
        or header["checkpoint_sha256"] != manifest.checkpoint_sha256
        or manifest.to_bytes() != raw
    ):
        raise ValueError("checkpoint manifest digest mismatch; possible tamper")
    return manifest


def _exclusive_write(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _install_file_no_replace(source: Path, destination: Path) -> None:
    """Publish a complete sibling file without replacing an existing target."""

    if os.name == "nt":
        # Windows os.rename fails if destination already exists.
        os.rename(source, destination)
        return
    # A same-directory hard link is atomic and fails with EEXIST.
    os.link(source, destination)
    source.unlink()


def save_h6_checkpoint(
    path: str | os.PathLike[str],
    manifest: H6CheckpointManifest,
) -> None:
    """Durably publish one complete checkpoint without replacement."""

    if type(manifest) is not H6CheckpointManifest:
        raise ValueError("manifest must be an exact H6CheckpointManifest")
    content = manifest.to_bytes()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{uuid.uuid4().hex}.complete"
    )
    installed = False
    try:
        _exclusive_write(temporary, content)
        _install_file_no_replace(temporary, target)
        installed = True
        _fsync_directory(target.parent)
    finally:
        if not installed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def load_h6_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_attempt_spec: H6AttemptSpec,
    expected_cursor: H6AttemptCursor,
    expected_objective_manifest: H6ObjectiveManifest,
) -> H6CheckpointManifest:
    """Load only when every retained scientific identity matches exactly."""

    if type(expected_attempt_spec) is not H6AttemptSpec:
        raise ValueError("expected_attempt_spec must be exact and typed")
    if type(expected_cursor) is not H6AttemptCursor:
        raise ValueError("expected_cursor must be exact and typed")
    if type(expected_objective_manifest) is not H6ObjectiveManifest:
        raise ValueError(
            "expected_objective_manifest must be exact and typed"
        )
    return _manifest_from_bytes(
        Path(path).read_bytes(),
        expected_attempt_spec=expected_attempt_spec,
        expected_cursor=expected_cursor,
        expected_objective_manifest=expected_objective_manifest,
    )


__all__ = [
    "H6CheckpointManifest",
    "load_h6_checkpoint",
    "save_h6_checkpoint",
]
