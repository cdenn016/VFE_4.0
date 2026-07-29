"""Read-only validation of finalized WT103 figure experiment indexes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from vfe4.types.figures import FigureExperimentIndexIdentity
from vfe4.types.training import (
    EndpointInventory,
    WT103CheckpointIdentity,
    canonical_json_bytes,
    owned_sha256,
)


_ZERO_SHA256 = "0" * 64
_MAX_JSON_BYTES = 16 * 1024 * 1024
_INVALID_WINDOWS_CHARACTERS = frozenset('<>:"|?*')
_RESERVED_WINDOWS_STEMS = {
    "AUX",
    "CON",
    "CONIN$",
    "CONOUT$",
    "NUL",
    "PRN",
}
_RESERVED_WINDOWS_STEMS.update(f"COM{index}" for index in range(1, 10))
_RESERVED_WINDOWS_STEMS.update(f"LPT{index}" for index in range(1, 10))

_PLAN_KEYS = frozenset(
    {
        "checkpoint_schema_sha256",
        "config_sha256",
        "dirty_digest",
        "endpoint_inventory_sha256",
        "experiment_id",
        "experiment_plan_sha256",
        "expected_group_artifact_paths",
        "expected_run_artifact_paths",
        "factory_set_sha256",
        "figure_panel_count",
        "figure_series_count",
        "git_head",
        "objective_sha256",
        "raw_score_record_count",
        "resource_forecast_sha256",
        "result_row_count",
        "schedule_set_sha256",
        "schema_version",
        "source_record_sha256",
        "terminal_checkpoint_count",
        "terminal_checkpoint_keys",
        "test_endpoint_count",
        "token_cache_set_sha256",
        "tokenizer_spec_sha256",
        "tuning_attempt_count",
        "tuning_attempt_keys",
        "validation_endpoint_count",
        "window_manifest_sha256s",
    }
)
_INDEX_KEYS = frozenset(
    {
        "artifact_records",
        "experiment_plan_sha256",
        "index_sha256",
        "runs",
        "schema_version",
        "stage",
    }
)
_INDEX_RUN_KEYS = frozenset(
    {
        "disposition",
        "manifest_identity_sha256",
        "manifest_sha256",
        "relative_manifest_path",
        "run_id",
        "run_role",
    }
)
_MANIFEST_BASE_KEYS = frozenset(
    {
        "artifact_records",
        "checkpoint_artifact_records",
        "checkpoints",
        "disposition",
        "ended_utc",
        "environment_sha256",
        "experiment_plan_sha256",
        "failure_record_sha256",
        "manifest_sha256",
        "monotonic_duration_seconds",
        "provenance_sha256",
        "reservation_sha256",
        "resume_count",
        "resume_lineage_sha256",
        "run_id",
        "run_role",
        "schema_version",
        "started_utc",
    }
)
_ARTIFACT_RECORD_KEYS = frozenset(
    {
        "kind",
        "record_sha256",
        "relative_path",
        "schema_version",
        "sha256",
        "size_bytes",
    }
)
_CHECKPOINT_KEYS = frozenset(
    {
        "artifact_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_manifest_body_sha256",
        "checkpoint_payload_sha256",
        "checkpoint_role",
        "logical_key",
        "schema_version",
        "scientific_state_sha256",
        "size_bytes",
    }
)


class ReadOnlyFigureIndexError(RuntimeError):
    """A finalized figure input failed immutable, read-only validation."""


@dataclass(frozen=True, slots=True)
class ValidatedExperimentPlan:
    """The immutable subset of a canonical experiment plan used by figures."""

    plan_path: Path
    document: dict[str, object]
    payload: bytes
    experiment_plan_sha256: str
    endpoint_inventory_sha256: str
    tuning_attempt_keys: tuple[str, ...]
    terminal_checkpoint_keys: tuple[str, ...]
    expected_run_artifact_paths: tuple[str, ...]
    expected_group_artifact_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidatedRunManifest:
    """Canonical manifest bytes plus the lifecycle-compatible read identity."""

    manifest_path: Path
    document: dict[str, object]
    payload: bytes
    run_id: str
    run_role: Literal["tuning", "confirmation"]
    tuning_attempt_key: str | None
    disposition: Literal["success", "failure"]
    experiment_plan_sha256: str
    manifest_sha256: str
    manifest_payload_sha256: str
    manifest_size_bytes: int
    checkpoint_identity_sha256s: tuple[str, ...]
    checkpoint_artifact_record_sha256s: tuple[str, ...]
    artifact_record_sha256s: tuple[str, ...]
    resume_lineage_sha256: str | None
    terminal_checkpoint_key: str | None
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedFinalExperimentIndex:
    """A complete finalized index, its plan, and all referenced manifests."""

    identity: FigureExperimentIndexIdentity
    document: dict[str, object]
    payload: bytes
    plan: ValidatedExperimentPlan
    manifests: tuple[ValidatedRunManifest, ...]

    def manifest_by_relative_path(self) -> dict[str, ValidatedRunManifest]:
        """Return the already-validated explicit manifest lookup."""

        return {
            manifest.manifest_path.relative_to(
                self.identity.index_path.parent
            ).as_posix(): manifest
            for manifest in self.manifests
        }


@dataclass(frozen=True, slots=True)
class _ArtifactRecord:
    schema_version: Literal["vfe4-artifact-integrity-v1"]
    kind: Literal["file", "manifest"]
    relative_path: str
    size_bytes: int
    sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != "vfe4-artifact-integrity-v1"
            or self.kind not in ("file", "manifest")
            or type(self.size_bytes) is not int
            or self.size_bytes < 0
        ):
            raise ReadOnlyFigureIndexError(
                "artifact integrity record fields are invalid"
            )
        _relative_path(self.relative_path)
        _sha256(self.sha256, "artifact sha256")
        body = {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.record_sha256 != owned_sha256(
            "vfe4.artifact-integrity-record.v1",
            body,
        ):
            raise ReadOnlyFigureIndexError(
                "artifact record hash does not match its body"
            )


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReadOnlyFigureIndexError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ReadOnlyFigureIndexError(f"{name} must be nonempty text")
    return value


def _portable_component(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ReadOnlyFigureIndexError(f"{name} must be a nonempty component")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        unicodedata.normalize("NFC", value) != value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or len(posix.parts) != 1
        or len(windows.parts) != 1
        or value.endswith((".", " "))
        or any(
            ord(character) < 32
            or character in _INVALID_WINDOWS_CHARACTERS
            for character in value
        )
        or value.split(".", 1)[0].upper() in _RESERVED_WINDOWS_STEMS
    ):
        raise ReadOnlyFigureIndexError(
            f"{name} is not a portable canonical component"
        )
    return value


def _relative_path(value: object) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise ReadOnlyFigureIndexError(
            "artifact path must be canonical POSIX text"
        )
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        unicodedata.normalize("NFC", value) != value
        or path.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or path.as_posix() != value
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ReadOnlyFigureIndexError(
            "artifact path is noncanonical or escapes its root"
        )
    for part in path.parts:
        _portable_component(part, "artifact path component")
    return path


def _is_redirect_or_reparse(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if bool(getattr(status, "st_file_attributes", 0) & reparse):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _regular_directory(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ReadOnlyFigureIndexError(
            f"directory metadata is unavailable: {path}"
        ) from exc
    if (
        not stat.S_ISDIR(status.st_mode)
        or _is_redirect_or_reparse(path, status)
    ):
        raise ReadOnlyFigureIndexError(
            f"path must be a regular nonlink directory: {path}"
        )
    return status


def _regular_file_status(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise ReadOnlyFigureIndexError(
            f"file metadata is unavailable: {path}"
        ) from exc
    if (
        not stat.S_ISREG(status.st_mode)
        or _is_redirect_or_reparse(path, status)
    ):
        raise ReadOnlyFigureIndexError(
            f"path must be a regular nonlink file: {path}"
        )
    return status


def _entry_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _contained_path(root: Path, relative_path: str) -> Path:
    relative = _relative_path(relative_path)
    current = root
    _regular_directory(current)
    for component in relative.parts[:-1]:
        current = current / component
        _regular_directory(current)
    return root.joinpath(*relative.parts)


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    status = _regular_file_status(path)
    if status.st_size > maximum_bytes:
        raise ReadOnlyFigureIndexError(
            f"file exceeds its bounded read size: {path}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReadOnlyFigureIndexError(f"file read failed: {path}") from exc
    reopened = _regular_file_status(path)
    if (
        len(payload) != status.st_size
        or reopened.st_size != status.st_size
        or reopened.st_dev != status.st_dev
        or reopened.st_ino != status.st_ino
    ):
        raise ReadOnlyFigureIndexError(
            f"file identity changed while reading: {path}"
        )
    return payload


def _hash_regular_file(path: Path, expected_size: int) -> str:
    status = _regular_file_status(path)
    if status.st_size != expected_size:
        raise ReadOnlyFigureIndexError(
            f"artifact size differs from its record: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReadOnlyFigureIndexError(
            f"artifact open failed: {path}"
        ) from exc
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size:
            raise ReadOnlyFigureIndexError(
                f"artifact changed before hashing: {path}"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        closed_view = os.fstat(descriptor)
    except OSError as exc:
        raise ReadOnlyFigureIndexError(
            f"artifact hash read failed: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    reopened = _regular_file_status(path)
    if (
        total != expected_size
        or closed_view.st_size != expected_size
        or reopened.st_size != expected_size
        or (opened.st_dev, opened.st_ino)
        != (status.st_dev, status.st_ino)
        or (closed_view.st_dev, closed_view.st_ino)
        != (status.st_dev, status.st_ino)
        or (reopened.st_dev, reopened.st_ino)
        != (status.st_dev, status.st_ino)
    ):
        raise ReadOnlyFigureIndexError(
            f"artifact identity changed while hashing: {path}"
        )
    return digest.hexdigest()


def _json_object(payload: bytes, *, context: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReadOnlyFigureIndexError(
            f"{context} is not strict JSON"
        ) from exc
    if type(document) is not dict:
        raise ReadOnlyFigureIndexError(
            f"{context} must be a JSON object"
        )
    try:
        canonical = canonical_json_bytes(document)
    except (TypeError, ValueError) as exc:
        raise ReadOnlyFigureIndexError(
            f"{context} contains a noncanonical value"
        ) from exc
    if canonical != payload:
        raise ReadOnlyFigureIndexError(
            f"{context} is not canonical JSON"
        )
    return document


def _read_canonical_object(
    path: Path,
    *,
    context: str,
) -> tuple[dict[str, object], bytes]:
    payload = _read_regular_bytes(path)
    return _json_object(payload, context=context), payload


def _artifact_record(value: object) -> _ArtifactRecord:
    if type(value) is not dict or set(value) != _ARTIFACT_RECORD_KEYS:
        raise ReadOnlyFigureIndexError(
            "artifact integrity record is not closed"
        )
    try:
        return _ArtifactRecord(**value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ReadOnlyFigureIndexError(
            "artifact integrity record is invalid"
        ) from exc


def _artifact_records(
    value: object,
    *,
    name: str,
) -> tuple[_ArtifactRecord, ...]:
    if type(value) is not list:
        raise ReadOnlyFigureIndexError(f"{name} must be a JSON array")
    records = tuple(_artifact_record(item) for item in value)
    record_hashes = tuple(item.record_sha256 for item in records)
    if len(set(record_hashes)) != len(record_hashes):
        raise ReadOnlyFigureIndexError(
            f"{name} contains duplicate record identities"
        )
    return records


def _verify_artifact_record(root: Path, record: _ArtifactRecord) -> None:
    target = _contained_path(root, record.relative_path)
    observed = _hash_regular_file(target, record.size_bytes)
    if observed != record.sha256:
        raise ReadOnlyFigureIndexError(
            f"artifact integrity mismatch: {record.relative_path}"
        )


def _path_inventory(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ReadOnlyFigureIndexError(f"{name} must be a nonempty JSON array")
    paths = tuple(str(item) for item in value)
    if any(type(item) is not str for item in value):
        raise ReadOnlyFigureIndexError(f"{name} entries must be exact strings")
    for path in paths:
        _relative_path(path)
    if (
        paths != tuple(sorted(paths))
        or len({path.casefold() for path in paths}) != len(paths)
    ):
        raise ReadOnlyFigureIndexError(f"{name} must be sorted and unique")
    return paths


def _load_experiment_plan(
    experiment_root: Path,
    *,
    endpoint_inventory: EndpointInventory,
) -> ValidatedExperimentPlan:
    if type(endpoint_inventory) is not EndpointInventory:
        raise ReadOnlyFigureIndexError(
            "endpoint_inventory must be an exact EndpointInventory"
        )
    try:
        endpoint_inventory.__post_init__()
    except ValueError as exc:
        raise ReadOnlyFigureIndexError(
            "endpoint inventory is invalid"
        ) from exc
    plan_path = experiment_root / "experiment-plan.json"
    document, payload = _read_canonical_object(
        plan_path,
        context="experiment plan",
    )
    if set(document) != _PLAN_KEYS:
        raise ReadOnlyFigureIndexError(
            "experiment plan key set is open"
        )
    if document["schema_version"] != "wt103-experiment-plan-v1":
        raise ReadOnlyFigureIndexError(
            "experiment plan schema is invalid"
        )
    _portable_component(document["experiment_id"], "experiment_id")
    git_head = document["git_head"]
    if (
        type(git_head) is not str
        or len(git_head) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise ReadOnlyFigureIndexError(
            "experiment plan git_head is invalid"
        )
    for name in (
        "endpoint_inventory_sha256",
        "dirty_digest",
        "config_sha256",
        "source_record_sha256",
        "tokenizer_spec_sha256",
        "token_cache_set_sha256",
        "schedule_set_sha256",
        "factory_set_sha256",
        "objective_sha256",
        "checkpoint_schema_sha256",
        "resource_forecast_sha256",
    ):
        _sha256(document[name], name)
    windows = document["window_manifest_sha256s"]
    if type(windows) is not list or len(windows) != 3:
        raise ReadOnlyFigureIndexError(
            "experiment plan window inventory is invalid"
        )
    for digest in windows:
        _sha256(digest, "window manifest sha256")
    tuning_attempt_keys = tuple(endpoint_inventory.tuning_attempt_keys)
    terminal_checkpoint_keys = tuple(
        endpoint_inventory.terminal_checkpoint_keys
    )
    if (
        document["endpoint_inventory_sha256"]
        != endpoint_inventory.endpoint_inventory_sha256
        or document["tuning_attempt_keys"] != list(tuning_attempt_keys)
        or document["terminal_checkpoint_keys"]
        != list(terminal_checkpoint_keys)
    ):
        raise ReadOnlyFigureIndexError(
            "experiment plan differs from the frozen endpoint inventory"
        )
    count_bindings = {
        "tuning_attempt_count": endpoint_inventory.tuning_attempt_count,
        "terminal_checkpoint_count": (
            endpoint_inventory.terminal_checkpoint_count
        ),
        "validation_endpoint_count": (
            endpoint_inventory.validation_endpoint_count
        ),
        "test_endpoint_count": endpoint_inventory.test_endpoint_count,
        "raw_score_record_count": (
            endpoint_inventory.raw_score_record_count
        ),
        "result_row_count": endpoint_inventory.result_row_count,
        "figure_panel_count": endpoint_inventory.figure_panel_count,
        "figure_series_count": endpoint_inventory.figure_series_count,
    }
    for name, expected in count_bindings.items():
        if type(document[name]) is not int or document[name] != expected:
            raise ReadOnlyFigureIndexError(
                f"experiment plan {name} differs from its inventory"
            )
    expected_run_paths = _path_inventory(
        document["expected_run_artifact_paths"],
        name="expected run artifact paths",
    )
    expected_group_paths = _path_inventory(
        document["expected_group_artifact_paths"],
        name="expected group artifact paths",
    )
    body = dict(document)
    observed_hash = _sha256(
        body.pop("experiment_plan_sha256", None),
        "experiment_plan_sha256",
    )
    if observed_hash != owned_sha256(
        "vfe4.wt103.experiment-plan.v1",
        body,
    ):
        raise ReadOnlyFigureIndexError(
            "experiment plan hash does not match"
        )
    return ValidatedExperimentPlan(
        plan_path=plan_path,
        document=document,
        payload=payload,
        experiment_plan_sha256=observed_hash,
        endpoint_inventory_sha256=(
            endpoint_inventory.endpoint_inventory_sha256
        ),
        tuning_attempt_keys=tuning_attempt_keys,
        terminal_checkpoint_keys=terminal_checkpoint_keys,
        expected_run_artifact_paths=expected_run_paths,
        expected_group_artifact_paths=expected_group_paths,
    )


def _checkpoint(value: object) -> WT103CheckpointIdentity:
    if type(value) is not dict or set(value) != _CHECKPOINT_KEYS:
        raise ReadOnlyFigureIndexError(
            "checkpoint identity is not closed"
        )
    try:
        checkpoint = WT103CheckpointIdentity(
            **value  # type: ignore[arg-type]
        )
        checkpoint.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ReadOnlyFigureIndexError(
            "checkpoint identity is invalid"
        ) from exc
    return checkpoint


def _validate_reservation(
    run_path: Path,
    *,
    run_id: str,
    run_role: str,
    tuning_attempt_key: str | None,
    experiment_plan_sha256: str,
    started_utc: str,
) -> str:
    document, _payload = _read_canonical_object(
        run_path / "reservation.json",
        context="run reservation",
    )
    if set(document) != {
        "experiment_plan_sha256",
        "reservation_sha256",
        "run_id",
        "run_role",
        "schema_version",
        "started_utc",
        "tuning_attempt_key",
    }:
        raise ReadOnlyFigureIndexError(
            "run reservation key set is open"
        )
    body = dict(document)
    observed = _sha256(
        body.pop("reservation_sha256", None),
        "reservation_sha256",
    )
    if (
        document["schema_version"] != "wt103-run-reservation-v1"
        or document["run_id"] != run_id
        or document["run_role"] != run_role
        or document["tuning_attempt_key"] != tuning_attempt_key
        or document["experiment_plan_sha256"]
        != experiment_plan_sha256
        or document["started_utc"] != started_utc
        or observed
        != owned_sha256(
            "vfe4.wt103.run-reservation.v1",
            body,
        )
    ):
        raise ReadOnlyFigureIndexError(
            "run reservation differs from its manifest"
        )
    return observed


def _lineage_records(payload: bytes) -> tuple[dict[str, object], ...]:
    if not payload:
        raise ReadOnlyFigureIndexError("resume lineage ledger is empty")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ReadOnlyFigureIndexError(
                "resume lineage has an incomplete final line"
            )
        record = _json_object(
            line[:-1],
            context="resume lineage event",
        )
        if set(record) != {
            "cursor_sha256",
            "environment_sha256",
            "lineage_sha256",
            "parent_artifact_sha256",
            "parent_checkpoint_identity_sha256",
            "parent_scientific_state_sha256",
            "reason",
            "resumed_utc",
            "schema_version",
        }:
            raise ReadOnlyFigureIndexError(
                "resume lineage event key set is open"
            )
        body = dict(record)
        observed = _sha256(
            body.pop("lineage_sha256", None),
            "lineage_sha256",
        )
        if (
            record["schema_version"]
            != "wt103-resume-lineage-event-v1"
            or observed
            != owned_sha256(
                "vfe4.wt103.resume-lineage-event.v1",
                body,
            )
        ):
            raise ReadOnlyFigureIndexError(
                "resume lineage event hash does not match"
            )
        for name in (
            "cursor_sha256",
            "environment_sha256",
            "parent_artifact_sha256",
            "parent_checkpoint_identity_sha256",
            "parent_scientific_state_sha256",
        ):
            _sha256(record[name], name)
        _text(record["reason"], "resume reason")
        _text(record["resumed_utc"], "resumed_utc")
        records.append(record)
    identities = tuple(str(item["lineage_sha256"]) for item in records)
    if len(set(identities)) != len(identities):
        raise ReadOnlyFigureIndexError(
            "resume lineage contains duplicate events"
        )
    return tuple(records)


def _resume_owner_body(
    *,
    run_id: str,
    plan_sha256: str,
    reservation_sha256: str,
    ordinal: int,
    previous_lineage_sha256: str,
    lineage_sha256: str,
    state: str,
    terminal_manifest_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": "wt103-resume-owner-v1",
        "run_id": run_id,
        "experiment_plan_sha256": plan_sha256,
        "reservation_sha256": reservation_sha256,
        "resume_ordinal": ordinal,
        "previous_owner_lineage_sha256": previous_lineage_sha256,
        "lineage_sha256": lineage_sha256,
        "state": state,
        "terminal_manifest_sha256": terminal_manifest_sha256,
    }


def _resume_owner_document(**kwargs: object) -> dict[str, object]:
    body = _resume_owner_body(**kwargs)  # type: ignore[arg-type]
    return {
        **body,
        "owner_sha256": owned_sha256(
            "vfe4.wt103.resume-owner.v1",
            body,
        ),
    }


def _validate_resume_state(
    run_path: Path,
    *,
    run_id: str,
    plan_sha256: str,
    reservation_sha256: str,
    resume_count: int,
    resume_lineage_sha256: str | None,
    manifest_sha256: str,
) -> None:
    lineage_path = run_path / "resume-lineage.jsonl"
    owner_path = run_path / "resume-owner.json"
    leases_path = run_path / "resume-leases"
    takeovers_path = run_path / "resume-owner-takeovers"
    if resume_lineage_sha256 is None:
        if resume_count != 0 or any(
            _entry_exists(path)
            for path in (
                lineage_path,
                owner_path,
                leases_path,
                takeovers_path,
            )
        ):
            raise ReadOnlyFigureIndexError(
                "unresumed run contains resume state"
            )
        return
    _sha256(resume_lineage_sha256, "resume_lineage_sha256")
    if resume_count <= 0:
        raise ReadOnlyFigureIndexError(
            "resume lineage requires a positive resume count"
        )
    lineage_payload = _read_regular_bytes(lineage_path)
    records = _lineage_records(lineage_payload)
    if (
        hashlib.sha256(lineage_payload).hexdigest()
        != resume_lineage_sha256
        or len(records) != resume_count
    ):
        raise ReadOnlyFigureIndexError(
            "resume lineage bytes differ from the manifest"
        )
    _regular_directory(leases_path)
    observed_lease_names = tuple(
        sorted(item.name for item in leases_path.iterdir())
    )
    expected_lease_names = tuple(
        f"{ordinal:08d}.json"
        for ordinal in range(1, resume_count + 1)
    )
    if observed_lease_names != expected_lease_names:
        raise ReadOnlyFigureIndexError(
            "resume lease ordinal inventory is not exact"
        )
    prefix = b""
    for ordinal, record in enumerate(records, start=1):
        lease_path = leases_path / f"{ordinal:08d}.json"
        lease, _payload = _read_canonical_object(
            lease_path,
            context="resume lease",
        )
        body = {
            "schema_version": "wt103-resume-lease-v1",
            "run_id": run_id,
            "experiment_plan_sha256": plan_sha256,
            "reservation_sha256": reservation_sha256,
            "resume_ordinal": ordinal,
            "previous_lineage_sha256": (
                hashlib.sha256(prefix).hexdigest()
                if prefix
                else _ZERO_SHA256
            ),
            "lineage_sha256": record["lineage_sha256"],
        }
        expected = {
            **body,
            "lease_sha256": owned_sha256(
                "vfe4.wt103.resume-lease.v1",
                body,
            ),
        }
        if lease != expected:
            raise ReadOnlyFigureIndexError(
                "resume lease differs from its lineage event"
            )
        prefix += canonical_json_bytes(record) + b"\n"
    expected_owner = _resume_owner_document(
        run_id=run_id,
        plan_sha256=plan_sha256,
        reservation_sha256=reservation_sha256,
        ordinal=resume_count,
        previous_lineage_sha256=(
            _ZERO_SHA256
            if resume_count == 1
            else str(records[-2]["lineage_sha256"])
        ),
        lineage_sha256=str(records[-1]["lineage_sha256"]),
        state="terminal_closed",
        terminal_manifest_sha256=manifest_sha256,
    )
    owner, _payload = _read_canonical_object(
        owner_path,
        context="resume owner",
    )
    if owner != expected_owner:
        raise ReadOnlyFigureIndexError(
            "terminal resume owner does not close the manifest"
        )
    if resume_count == 1:
        if _entry_exists(takeovers_path):
            raise ReadOnlyFigureIndexError(
                "single-owner lineage contains takeover records"
            )
        return
    _regular_directory(takeovers_path)
    expected_takeovers: dict[str, bytes] = {}
    for ordinal in range(2, resume_count + 1):
        previous = records[ordinal - 2]
        replacement = records[ordinal - 1]
        previous_owner = _resume_owner_document(
            run_id=run_id,
            plan_sha256=plan_sha256,
            reservation_sha256=reservation_sha256,
            ordinal=ordinal - 1,
            previous_lineage_sha256=(
                _ZERO_SHA256
                if ordinal == 2
                else str(records[ordinal - 3]["lineage_sha256"])
            ),
            lineage_sha256=str(previous["lineage_sha256"]),
            state="active",
            terminal_manifest_sha256=None,
        )
        replacement_owner = _resume_owner_document(
            run_id=run_id,
            plan_sha256=plan_sha256,
            reservation_sha256=reservation_sha256,
            ordinal=ordinal,
            previous_lineage_sha256=str(previous["lineage_sha256"]),
            lineage_sha256=str(replacement["lineage_sha256"]),
            state="active",
            terminal_manifest_sha256=None,
        )
        takeover_body = {
            "schema_version": "wt103-resume-owner-takeover-v1",
            "run_id": run_id,
            "previous_owner_sha256": previous_owner["owner_sha256"],
            "previous_owner_lineage_sha256": previous["lineage_sha256"],
            "replacement_owner_sha256": replacement_owner["owner_sha256"],
            "replacement_owner_lineage_sha256": replacement[
                "lineage_sha256"
            ],
        }
        takeover = {
            **takeover_body,
            "takeover_sha256": owned_sha256(
                "vfe4.wt103.resume-owner-takeover.v1",
                takeover_body,
            ),
        }
        expected_takeovers[
            f"{previous_owner['owner_sha256']}.json"
        ] = canonical_json_bytes(takeover)
    observed_takeovers: dict[str, bytes] = {}
    for item in takeovers_path.iterdir():
        _portable_component(item.name, "resume takeover filename")
        observed_takeovers[item.name] = _read_regular_bytes(item)
    if observed_takeovers != expected_takeovers:
        raise ReadOnlyFigureIndexError(
            "resume takeover inventory is not exact"
        )


def validate_finalized_run_manifest(
    manifest_path: Path,
    *,
    plan: ValidatedExperimentPlan,
) -> ValidatedRunManifest:
    """Validate one exact terminal run manifest without lifecycle mutation."""

    if not isinstance(manifest_path, Path):
        raise ReadOnlyFigureIndexError(
            "manifest_path must be pathlib.Path"
        )
    if manifest_path.name != "run-manifest.json":
        raise ReadOnlyFigureIndexError(
            "manifest path must name run-manifest.json"
        )
    if type(plan) is not ValidatedExperimentPlan:
        raise ReadOnlyFigureIndexError(
            "plan must be an exact ValidatedExperimentPlan"
        )
    run_path = manifest_path.parent
    _regular_directory(run_path)
    document, payload = _read_canonical_object(
        manifest_path,
        context="run manifest",
    )
    role = document.get("run_role")
    expected_keys = set(_MANIFEST_BASE_KEYS)
    if role == "tuning":
        expected_keys.add("tuning_attempt_key")
    if set(document) != expected_keys:
        raise ReadOnlyFigureIndexError(
            "run manifest key set is open"
        )
    if document["schema_version"] != "wt103-run-manifest-v1":
        raise ReadOnlyFigureIndexError(
            "run manifest schema is invalid"
        )
    run_id = _portable_component(document["run_id"], "run_id")
    if (
        run_path.name != run_id
        or run_path.parent.name != "runs"
        or run_path.parent.parent != plan.plan_path.parent
    ):
        raise ReadOnlyFigureIndexError(
            "run manifest path and run ID are not exact"
        )
    if role not in ("tuning", "confirmation"):
        raise ReadOnlyFigureIndexError(
            "run manifest role is invalid"
        )
    tuning_attempt_key = document.get("tuning_attempt_key")
    if (
        role == "tuning"
        and (
            type(tuning_attempt_key) is not str
            or tuning_attempt_key not in plan.tuning_attempt_keys
        )
    ) or (
        role == "confirmation"
        and tuning_attempt_key is not None
    ):
        raise ReadOnlyFigureIndexError(
            "run manifest tuning key differs from its role/plan"
        )
    disposition = document["disposition"]
    if disposition not in ("success", "failure"):
        raise ReadOnlyFigureIndexError(
            "run manifest disposition is invalid"
        )
    plan_sha = _sha256(
        document["experiment_plan_sha256"],
        "experiment_plan_sha256",
    )
    if plan_sha != plan.experiment_plan_sha256:
        raise ReadOnlyFigureIndexError(
            "run manifest does not bind its published plan"
        )
    for name in (
        "environment_sha256",
        "provenance_sha256",
        "reservation_sha256",
    ):
        _sha256(document[name], name)
    started_utc = _text(document["started_utc"], "started_utc")
    _text(document["ended_utc"], "ended_utc")
    duration = document["monotonic_duration_seconds"]
    resume_count = document["resume_count"]
    if (
        type(duration) is not float
        or not math.isfinite(duration)
        or duration < 0.0
        or type(resume_count) is not int
        or resume_count < 0
    ):
        raise ReadOnlyFigureIndexError(
            "run duration/resume count is invalid"
        )
    body = dict(document)
    manifest_sha = _sha256(
        body.pop("manifest_sha256", None),
        "manifest_sha256",
    )
    if manifest_sha != owned_sha256(
        "vfe4.wt103.run-manifest.v1",
        body,
    ):
        raise ReadOnlyFigureIndexError(
            "run manifest hash does not match"
        )
    reservation_sha = _validate_reservation(
        run_path,
        run_id=run_id,
        run_role=role,
        tuning_attempt_key=tuning_attempt_key,
        experiment_plan_sha256=plan_sha,
        started_utc=started_utc,
    )
    if reservation_sha != document["reservation_sha256"]:
        raise ReadOnlyFigureIndexError(
            "run manifest reservation digest does not match"
        )
    checkpoint_values = document["checkpoints"]
    if type(checkpoint_values) is not list:
        raise ReadOnlyFigureIndexError(
            "checkpoint inventory must be a JSON array"
        )
    checkpoints = tuple(_checkpoint(item) for item in checkpoint_values)
    checkpoint_keys = tuple(item.logical_key for item in checkpoints)
    checkpoint_hashes = tuple(
        item.checkpoint_identity_sha256 for item in checkpoints
    )
    if (
        len(set(checkpoint_keys)) != len(checkpoint_keys)
        or len(set(checkpoint_hashes)) != len(checkpoint_hashes)
    ):
        raise ReadOnlyFigureIndexError(
            "checkpoint keys and identities must be unique"
        )
    terminal = tuple(
        item
        for item in checkpoints
        if item.checkpoint_role == "terminal_scoring"
    )
    if role == "tuning" and terminal:
        raise ReadOnlyFigureIndexError(
            "tuning run contains a terminal-scoring checkpoint"
        )
    if disposition == "failure" and terminal:
        raise ReadOnlyFigureIndexError(
            "failed run contains a terminal-scoring checkpoint"
        )
    terminal_key: str | None = None
    if role == "confirmation" and disposition == "success":
        if (
            len(terminal) != 1
            or terminal[0].logical_key
            not in plan.terminal_checkpoint_keys
        ):
            raise ReadOnlyFigureIndexError(
                "successful confirmation terminal checkpoint is invalid"
            )
        terminal_key = terminal[0].logical_key
    checkpoint_records = _artifact_records(
        document["checkpoint_artifact_records"],
        name="checkpoint artifact records",
    )
    artifact_records = _artifact_records(
        document["artifact_records"],
        name="run artifact records",
    )
    if len(checkpoint_records) != len(checkpoints):
        raise ReadOnlyFigureIndexError(
            "checkpoint artifact inventory is incomplete"
        )
    planned_paths = {
        item.relative_path.casefold() for item in artifact_records
    }
    checkpoint_paths: set[str] = set()
    reserved_names = {
        "experiment-plan.json",
        "failures.jsonl",
        "reservation.json",
        "resume-lineage.jsonl",
        "resume-owner.json",
        "run-manifest.json",
    }
    for checkpoint, record in zip(
        checkpoints,
        checkpoint_records,
        strict=True,
    ):
        alias = record.relative_path.casefold()
        parts = PurePosixPath(record.relative_path).parts
        if (
            alias in checkpoint_paths
            or alias in planned_paths
            or record.kind != "file"
            or record.sha256 != checkpoint.checkpoint_payload_sha256
            or record.size_bytes != checkpoint.size_bytes
            or PurePosixPath(record.relative_path).name.casefold()
            in reserved_names
            or any(
                part.casefold()
                in {"resume-leases", "resume-owner-takeovers"}
                for part in parts
            )
        ):
            raise ReadOnlyFigureIndexError(
                "checkpoint artifact differs from its checkpoint identity"
            )
        checkpoint_paths.add(alias)
    artifact_paths = tuple(
        item.relative_path for item in artifact_records
    )
    if (
        artifact_paths != tuple(sorted(artifact_paths))
        or len({path.casefold() for path in artifact_paths})
        != len(artifact_paths)
    ):
        raise ReadOnlyFigureIndexError(
            "run artifact paths must be sorted and unique"
        )
    for record in artifact_records:
        _verify_artifact_record(run_path, record)
    failure_head = document["failure_record_sha256"]
    if disposition == "success":
        if (
            artifact_paths != plan.expected_run_artifact_paths
            or failure_head is not None
        ):
            raise ReadOnlyFigureIndexError(
                "successful run artifacts differ from the plan"
            )
    else:
        _sha256(failure_head, "failure_record_sha256")
        allowed = set(plan.expected_run_artifact_paths) | {
            "failures.jsonl"
        }
        failure_records = tuple(
            item
            for item in artifact_records
            if item.relative_path == "failures.jsonl"
        )
        if (
            not set(artifact_paths).issubset(allowed)
            or len(failure_records) != 1
        ):
            raise ReadOnlyFigureIndexError(
                "failed run artifact inventory is invalid"
            )
    lineage_sha = document["resume_lineage_sha256"]
    if lineage_sha is not None:
        _sha256(lineage_sha, "resume_lineage_sha256")
    _validate_resume_state(
        run_path,
        run_id=run_id,
        plan_sha256=plan_sha,
        reservation_sha256=reservation_sha,
        resume_count=resume_count,
        resume_lineage_sha256=lineage_sha,
        manifest_sha256=manifest_sha,
    )
    semantic = {
        "run_id": run_id,
        "run_role": role,
        "tuning_attempt_key": tuning_attempt_key,
        "disposition": disposition,
        "experiment_plan_sha256": plan_sha,
        "manifest_sha256": manifest_sha,
        "manifest_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_size_bytes": len(payload),
        "checkpoint_identity_sha256s": checkpoint_hashes,
        "checkpoint_artifact_record_sha256s": tuple(
            item.record_sha256 for item in checkpoint_records
        ),
        "artifact_record_sha256s": tuple(
            item.record_sha256 for item in artifact_records
        ),
        "resume_lineage_sha256": lineage_sha,
    }
    identity_sha = owned_sha256(
        "vfe4.wt103.run-manifest-identity.v1",
        semantic,
    )
    return ValidatedRunManifest(
        manifest_path=manifest_path,
        document=document,
        payload=payload,
        terminal_checkpoint_key=terminal_key,
        identity_sha256=identity_sha,
        **semantic,
    )  # type: ignore[arg-type]


def validate_finalized_experiment_index(
    index_path: Path,
    *,
    endpoint_inventory: EndpointInventory,
) -> ValidatedFinalExperimentIndex:
    """Validate index identities and figure-authoritative artifact bytes."""

    if not isinstance(index_path, Path):
        raise ReadOnlyFigureIndexError(
            "index_path must be pathlib.Path"
        )
    if index_path.name != "experiment-index.json":
        raise ReadOnlyFigureIndexError(
            "index path must name experiment-index.json"
        )
    experiment_root = index_path.parent
    _regular_directory(experiment_root)
    document, payload = _read_canonical_object(
        index_path,
        context="experiment index",
    )
    if set(document) != _INDEX_KEYS:
        raise ReadOnlyFigureIndexError(
            "experiment index key set is open"
        )
    if (
        document["schema_version"] != "wt103-experiment-index-v1"
        or document["stage"] != "final"
    ):
        raise ReadOnlyFigureIndexError(
            "figure input requires a finalized experiment index"
        )
    body = dict(document)
    observed_index_sha = _sha256(
        body.pop("index_sha256", None),
        "index_sha256",
    )
    if observed_index_sha != owned_sha256(
        "vfe4.wt103.experiment-index.v1",
        body,
    ):
        raise ReadOnlyFigureIndexError(
            "experiment index hash does not match"
        )
    plan = _load_experiment_plan(
        experiment_root,
        endpoint_inventory=endpoint_inventory,
    )
    plan_sha = _sha256(
        document["experiment_plan_sha256"],
        "experiment_plan_sha256",
    )
    if plan_sha != plan.experiment_plan_sha256:
        raise ReadOnlyFigureIndexError(
            "experiment index does not bind its published plan"
        )
    group_records = _artifact_records(
        document["artifact_records"],
        name="group artifact records",
    )
    group_paths = tuple(item.relative_path for item in group_records)
    if (
        group_paths != tuple(sorted(group_paths))
        or len({path.casefold() for path in group_paths})
        != len(group_paths)
        or group_paths != plan.expected_group_artifact_paths
    ):
        raise ReadOnlyFigureIndexError(
            "final group artifacts differ from the experiment plan"
        )
    result_table_records = tuple(
        item
        for item in group_records
        if item.relative_path == "result-table.json"
    )
    if (
        len(result_table_records) != 1
        or result_table_records[0].kind != "file"
    ):
        raise ReadOnlyFigureIndexError(
            "final index lacks one exact result-table record"
        )
    for record in group_records:
        _verify_artifact_record(experiment_root, record)
    entries = document["runs"]
    if type(entries) is not list or not entries:
        raise ReadOnlyFigureIndexError(
            "experiment index runs must be a nonempty JSON array"
        )
    expected_count = (
        len(plan.tuning_attempt_keys)
        + len(plan.terminal_checkpoint_keys)
    )
    if len(entries) != expected_count:
        raise ReadOnlyFigureIndexError(
            "final experiment index run count differs from its plan"
        )
    manifests: list[ValidatedRunManifest] = []
    run_ids: list[str] = []
    manifest_shas: list[str] = []
    manifest_identity_shas: list[str] = []
    relative_paths: list[str] = []
    run_roles: list[str] = []
    tuning_keys: list[str] = []
    terminal_keys: list[str] = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != _INDEX_RUN_KEYS:
            raise ReadOnlyFigureIndexError(
                "experiment index run entry is not closed"
            )
        run_id = _portable_component(entry["run_id"], "run_id")
        role = entry["run_role"]
        disposition = entry["disposition"]
        if role not in ("tuning", "confirmation"):
            raise ReadOnlyFigureIndexError(
                "experiment index run role is invalid"
            )
        if disposition not in ("success", "failure"):
            raise ReadOnlyFigureIndexError(
                "experiment index disposition is invalid"
            )
        manifest_sha = _sha256(
            entry["manifest_sha256"],
            "manifest_sha256",
        )
        manifest_identity_sha = _sha256(
            entry["manifest_identity_sha256"],
            "manifest_identity_sha256",
        )
        expected_relative = f"runs/{run_id}/run-manifest.json"
        if entry["relative_manifest_path"] != expected_relative:
            raise ReadOnlyFigureIndexError(
                "experiment index manifest path is not exact"
            )
        manifest_path = _contained_path(
            experiment_root,
            expected_relative,
        )
        manifest = validate_finalized_run_manifest(
            manifest_path,
            plan=plan,
        )
        if (
            manifest.run_id != run_id
            or manifest.run_role != role
            or manifest.disposition != disposition
            or manifest.manifest_sha256 != manifest_sha
            or manifest.identity_sha256 != manifest_identity_sha
            or manifest.experiment_plan_sha256 != plan_sha
        ):
            raise ReadOnlyFigureIndexError(
                "experiment index entry differs from its terminal manifest"
            )
        manifests.append(manifest)
        run_ids.append(run_id)
        manifest_shas.append(manifest_sha)
        manifest_identity_shas.append(manifest_identity_sha)
        relative_paths.append(expected_relative)
        run_roles.append(role)
        if role == "tuning":
            if type(manifest.tuning_attempt_key) is not str:
                raise ReadOnlyFigureIndexError(
                    "tuning manifest lacks its exact attempt key"
                )
            tuning_keys.append(manifest.tuning_attempt_key)
        if manifest.terminal_checkpoint_key is not None:
            terminal_keys.append(manifest.terminal_checkpoint_key)
    for values, name in (
        (run_ids, "run IDs"),
        (manifest_shas, "manifest hashes"),
        (manifest_identity_shas, "manifest identity hashes"),
        (relative_paths, "manifest paths"),
    ):
        if len(set(values)) != len(values):
            raise ReadOnlyFigureIndexError(
                f"experiment index {name} must be unique"
            )
    expected_roles = (
        ("tuning",) * len(plan.tuning_attempt_keys)
        + ("confirmation",) * len(plan.terminal_checkpoint_keys)
    )
    if (
        tuple(run_roles) != expected_roles
        or tuple(tuning_keys) != plan.tuning_attempt_keys
        or tuple(terminal_keys) != plan.terminal_checkpoint_keys
    ):
        raise ReadOnlyFigureIndexError(
            "final ordered run inventory differs from the experiment plan"
        )
    semantic = {
        "stage": "final",
        "experiment_plan_sha256": plan_sha,
        "run_manifest_sha256s": tuple(manifest_shas),
        "artifact_record_sha256s": tuple(
            item.record_sha256 for item in group_records
        ),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    try:
        identity = FigureExperimentIndexIdentity(
            index_path=index_path,
            **semantic,
            identity_sha256=owned_sha256(
                "vfe4.wt103.experiment-index-identity.v1",
                semantic,
            ),
        )
        identity.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ReadOnlyFigureIndexError(
            "figure experiment index identity is invalid"
        ) from exc
    return ValidatedFinalExperimentIndex(
        identity=identity,
        document=document,
        payload=payload,
        plan=plan,
        manifests=tuple(manifests),
    )


__all__ = [
    "ReadOnlyFigureIndexError",
    "ValidatedExperimentPlan",
    "ValidatedFinalExperimentIndex",
    "ValidatedRunManifest",
    "validate_finalized_experiment_index",
    "validate_finalized_run_manifest",
]
