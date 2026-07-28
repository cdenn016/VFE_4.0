"""Authenticated, no-replace terminal-checkpoint catalog for H6 v3."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vfe4.artifacts.atomic import (
    ArtifactPublicationError,
    canonical_json_bytes,
    publish_run_directory,
)
from vfe4.artifacts.h6_prediction_v3 import (
    H6PredictionV3Authorities,
    H6TuningSelectionV3,
    _optimizer_cell,
    _validate_planned_checkpoint_v3,
)
from vfe4.training.checkpoint_v3 import (
    H6CheckpointV3,
    read_h6_checkpoint_file_v3,
)
from vfe4.training.h6_execution_v3 import (
    H6ExecutableAttemptV3,
    bind_h6_executable_attempt_v3,
)
_ENTRY_FILENAME = "checkpoint_catalog_entry.json"
_MAXIMUM_ENTRY_BYTES = 16 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")


def _hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib Path")
    resolved = value.resolve(strict=False)
    if resolved.as_posix() != value.as_posix():
        raise ValueError(f"{name} must be canonical")
    return resolved


def _is_redirect(path: Path, status: os.stat_result) -> bool:
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(status, "st_file_attributes", 0) & reparse_flag:
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _read_bounded_regular_file_once(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    try:
        parent_before = path.parent.lstat()
        path_before = path.lstat()
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or _is_redirect(path.parent, parent_before)
        or not stat.S_ISREG(path_before.st_mode)
        or _is_redirect(path, path_before)
        or path_before.st_size > maximum_bytes
    ):
        raise ArtifactPublicationError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (path_before.st_dev, path_before.st_ino)
            or opened.st_size > maximum_bytes
        ):
            raise ArtifactPublicationError(f"{label} identity or bound changed")
        raw = os.read(descriptor, maximum_bytes + 1)
        if len(raw) > maximum_bytes or os.read(descriptor, 1):
            raise ArtifactPublicationError(f"{label} exceeds its byte bound")
        opened_after = os.fstat(descriptor)
        if (
            (
                opened_after.st_dev,
                opened_after.st_ino,
                opened_after.st_size,
                opened_after.st_mtime_ns,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
        ):
            raise ArtifactPublicationError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    try:
        parent_after = path.parent.lstat()
        path_after = path.lstat()
    except OSError as exc:
        raise ArtifactPublicationError(f"{label} changed after reading") from exc
    if (
        (parent_after.st_dev, parent_after.st_ino)
        != (parent_before.st_dev, parent_before.st_ino)
        or (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
        != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        or _is_redirect(path.parent, parent_after)
        or _is_redirect(path, path_after)
    ):
        raise ArtifactPublicationError(f"{label} path identity changed")
    return raw


@dataclass(frozen=True, slots=True)
class H6CheckpointCatalogEntryV3:
    entry_schema: Literal["h6-checkpoint-catalog-entry-v3"]
    authority_sha256: str
    experiment_config_sha256: str
    readiness_sha256: str
    plan_sha256: str
    matching_set_sha256: str
    data_identity_sha256: str
    runtime_identity_sha256: str
    stage: Literal["tuning", "confirmatory"]
    planned_attempt_sha256: str
    executable_attempt_sha256: str
    endpoint_config_id: str
    endpoint_config_sha256: str
    tuning_cell_sha256: str
    tuning_cell_source: str
    training_seed: int
    attempt_spec_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_bytes_sha256: str
    checkpoint_byte_count: int
    entry_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def artifact_payload(self) -> dict[str, object]:
        return self.canonical_payload() | {"entry_sha256": self.entry_sha256}

    def __post_init__(self) -> None:
        if (
            self.entry_schema != "h6-checkpoint-catalog-entry-v3"
            or self.stage not in ("tuning", "confirmatory")
        ):
            raise ValueError("checkpoint catalog entry schema/stage is stale")
        for name in (
            "authority_sha256",
            "experiment_config_sha256",
            "readiness_sha256",
            "plan_sha256",
            "matching_set_sha256",
            "data_identity_sha256",
            "runtime_identity_sha256",
            "planned_attempt_sha256",
            "executable_attempt_sha256",
            "endpoint_config_sha256",
            "tuning_cell_sha256",
            "attempt_spec_sha256",
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "entry_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.endpoint_config_id) is not str
            or not self.endpoint_config_id
            or type(self.tuning_cell_source) is not str
            or not self.tuning_cell_source
            or type(self.training_seed) is not int
            or type(self.checkpoint_byte_count) is not int
            or self.checkpoint_byte_count <= 0
        ):
            raise ValueError("checkpoint catalog entry fields are malformed")
        checkpoint_path = Path(self.checkpoint_path)
        _canonical_absolute_path(checkpoint_path, "checkpoint_path")
        if checkpoint_path.as_posix() != self.checkpoint_path:
            raise ValueError("checkpoint_path must use canonical POSIX spelling")
        if self.entry_sha256 != _hash(
            "vfe4.h6.checkpoint-catalog-entry.v3",
            self.canonical_payload(),
        ):
            raise ValueError("checkpoint catalog entry identity is stale")

    @classmethod
    def create(
        cls,
        *,
        executable_attempt: H6ExecutableAttemptV3,
        checkpoint: H6CheckpointV3,
        checkpoint_path: Path,
    ) -> "H6CheckpointCatalogEntryV3":
        if type(executable_attempt) is not H6ExecutableAttemptV3:
            raise ValueError("catalog publication requires exact executable attempt")
        executable_attempt.__post_init__()
        if type(checkpoint) is not H6CheckpointV3:
            raise ValueError("catalog publication requires exact checkpoint v3")
        checkpoint.__post_init__()
        canonical_path = _canonical_absolute_path(
            checkpoint_path,
            "checkpoint_path",
        )
        authorities = executable_attempt.authorities
        attempt = executable_attempt.planned_attempt
        raw = _validate_planned_checkpoint_v3(
            checkpoint=checkpoint,
            planned_attempt=attempt,
            plan=authorities.plan,
            stage=attempt.stage,
        )
        if _optimizer_cell(checkpoint) != (
            executable_attempt.tuning_cell.learning_rate,
            executable_attempt.tuning_cell.weight_decay,
        ):
            raise ValueError("catalog checkpoint optimizer cell is not executable")
        values = {
            "entry_schema": "h6-checkpoint-catalog-entry-v3",
            "authority_sha256": authorities.authority_sha256,
            "experiment_config_sha256": authorities.config.config_sha256,
            "readiness_sha256": authorities.readiness.readiness_sha256,
            "plan_sha256": authorities.plan.plan_sha256,
            "matching_set_sha256": authorities.matching_set.matching_set_sha256,
            "data_identity_sha256": authorities.config.data_identity_sha256,
            "runtime_identity_sha256": (
                authorities.config.runtime.runtime_identity_sha256
            ),
            "stage": attempt.stage,
            "planned_attempt_sha256": attempt.planned_attempt_sha256,
            "executable_attempt_sha256": (
                executable_attempt.executable_attempt_sha256
            ),
            "endpoint_config_id": attempt.endpoint_config_id,
            "endpoint_config_sha256": attempt.endpoint_config_sha256,
            "tuning_cell_sha256": executable_attempt.tuning_cell.cell_sha256,
            "tuning_cell_source": executable_attempt.tuning_cell_source,
            "training_seed": attempt.training_seed,
            "attempt_spec_sha256": attempt.attempt_spec.attempt_spec_sha256,
            "checkpoint_path": canonical_path.as_posix(),
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "checkpoint_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint_byte_count": len(raw),
        }
        return cls(
            **values,  # type: ignore[arg-type]
            entry_sha256=_hash(
                "vfe4.h6.checkpoint-catalog-entry.v3",
                values,
            ),
        )


@dataclass(frozen=True, slots=True)
class H6CheckpointCatalogItemV3:
    entry: H6CheckpointCatalogEntryV3
    executable_attempt: H6ExecutableAttemptV3
    checkpoint: H6CheckpointV3

    def __post_init__(self) -> None:
        if (
            type(self.entry) is not H6CheckpointCatalogEntryV3
            or type(self.executable_attempt) is not H6ExecutableAttemptV3
            or type(self.checkpoint) is not H6CheckpointV3
        ):
            raise ValueError("checkpoint catalog item contains a non-v3 record")
        self.entry.__post_init__()
        self.executable_attempt.__post_init__()
        self.checkpoint.__post_init__()
        attempt = self.executable_attempt.planned_attempt
        if (
            self.entry.planned_attempt_sha256
            != attempt.planned_attempt_sha256
            or self.entry.executable_attempt_sha256
            != self.executable_attempt.executable_attempt_sha256
            or self.entry.checkpoint_sha256
            != self.checkpoint.checkpoint_sha256
            or self.entry.checkpoint_bytes_sha256
            != hashlib.sha256(self.checkpoint.to_bytes()).hexdigest()
        ):
            raise ValueError("checkpoint catalog item identity drift")


@dataclass(frozen=True, slots=True)
class H6CheckpointCatalogV3:
    catalog_schema: Literal["h6-checkpoint-catalog-v3"]
    authority_sha256: str
    items: tuple[H6CheckpointCatalogItemV3, ...]
    catalog_snapshot_sha256: str
    _plan_order: tuple[str, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.catalog_schema != "h6-checkpoint-catalog-v3":
            raise ValueError("checkpoint catalog schema is stale")
        _require_sha256(self.authority_sha256, "authority_sha256")
        if (
            type(self.items) is not tuple
            or any(type(item) is not H6CheckpointCatalogItemV3 for item in self.items)
            or type(self._plan_order) is not tuple
        ):
            raise ValueError("checkpoint catalog item inventory is malformed")
        for item in self.items:
            item.__post_init__()
        planned = tuple(
            item.entry.planned_attempt_sha256 for item in self.items
        )
        if (
            len(set(planned)) != len(planned)
            or tuple(value for value in self._plan_order if value in set(planned))
            != planned
            or {
                item.entry.authority_sha256 for item in self.items
            }
            - {self.authority_sha256}
        ):
            raise ValueError("checkpoint catalog is duplicated, foreign, or unordered")
        checkpoint_ids = tuple(item.entry.checkpoint_sha256 for item in self.items)
        checkpoint_paths = tuple(item.entry.checkpoint_path for item in self.items)
        if (
            len(set(checkpoint_ids)) != len(checkpoint_ids)
            or len(set(checkpoint_paths)) != len(checkpoint_paths)
        ):
            raise ValueError("checkpoint catalog checkpoint inventory is duplicated")
        payload = {
            "catalog_schema": self.catalog_schema,
            "authority_sha256": self.authority_sha256,
            "entries": tuple(item.entry.artifact_payload() for item in self.items),
        }
        if self.catalog_snapshot_sha256 != _hash(
            "vfe4.h6.checkpoint-catalog-snapshot.v3",
            payload,
        ):
            raise ValueError("checkpoint catalog snapshot identity is stale")

    @property
    def tuning_items(self) -> tuple[H6CheckpointCatalogItemV3, ...]:
        return tuple(item for item in self.items if item.entry.stage == "tuning")

    @property
    def confirmatory_items(self) -> tuple[H6CheckpointCatalogItemV3, ...]:
        return tuple(
            item for item in self.items if item.entry.stage == "confirmatory"
        )


def publish_h6_checkpoint_catalog_entry_v3(
    *,
    catalog_root: Path,
    checkpoint_path: Path,
    maximum_checkpoint_bytes: int,
    executable_attempt: H6ExecutableAttemptV3,
    checkpoint: H6CheckpointV3,
) -> Path:
    """Publish one terminal checkpoint authority under its plan digest."""

    root = _canonical_absolute_path(catalog_root, "catalog_root")
    reopened = read_h6_checkpoint_file_v3(
        _canonical_absolute_path(checkpoint_path, "checkpoint_path"),
        maximum_bytes=maximum_checkpoint_bytes,
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )
    if reopened.to_bytes() != checkpoint.to_bytes():
        raise ValueError("checkpoint file differs from the terminal checkpoint")
    entry = H6CheckpointCatalogEntryV3.create(
        executable_attempt=executable_attempt,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    return publish_run_directory(
        root,
        entry.planned_attempt_sha256,
        {_ENTRY_FILENAME: entry.artifact_payload()},
    )


def _decode_entry(raw: bytes) -> H6CheckpointCatalogEntryV3:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("checkpoint catalog entry has duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "checkpoint catalog entry is not canonical JSON"
        ) from exc
    expected_fields = frozenset(H6CheckpointCatalogEntryV3.__dataclass_fields__)
    if (
        type(payload) is not dict
        or frozenset(payload) != expected_fields
        or canonical_json_bytes(payload) != raw
    ):
        raise ArtifactPublicationError(
            "checkpoint catalog entry field inventory is not exact"
        )
    try:
        return H6CheckpointCatalogEntryV3(**payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ArtifactPublicationError(
            "checkpoint catalog entry cannot be authenticated"
        ) from exc


def _read_entry_directory(directory: Path) -> H6CheckpointCatalogEntryV3:
    try:
        root_status = directory.lstat()
        children = tuple(directory.iterdir())
    except OSError as exc:
        raise ArtifactPublicationError(
            "checkpoint catalog entry directory is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_status.st_mode)
        or _is_redirect(directory, root_status)
        or {path.name for path in children}
        != {_ENTRY_FILENAME, "manifest.sha256"}
    ):
        raise ArtifactPublicationError(
            "checkpoint catalog entry inventory is not exact"
        )
    payload_raw = _read_bounded_regular_file_once(
        directory / _ENTRY_FILENAME,
        maximum_bytes=_MAXIMUM_ENTRY_BYTES,
        label="checkpoint catalog entry",
    )
    manifest_raw = _read_bounded_regular_file_once(
        directory / "manifest.sha256",
        maximum_bytes=256,
        label="checkpoint catalog manifest",
    )
    expected_manifest = (
        f"{hashlib.sha256(payload_raw).hexdigest()}  {_ENTRY_FILENAME}\n"
    ).encode("ascii")
    if manifest_raw != expected_manifest:
        raise ArtifactPublicationError("checkpoint catalog manifest changed")
    return _decode_entry(payload_raw)


def read_h6_checkpoint_catalog_v3(
    catalog_root: Path,
    *,
    authorities: H6PredictionV3Authorities,
    maximum_checkpoint_bytes: int,
    tuning_selection: H6TuningSelectionV3 | None = None,
    required_inventory: Literal["partial", "tuning", "complete"] = "partial",
) -> H6CheckpointCatalogV3:
    """Reopen every present entry and optionally require a complete stage set."""

    root = _canonical_absolute_path(catalog_root, "catalog_root")
    if type(authorities) is not H6PredictionV3Authorities:
        raise ValueError("catalog reopen requires exact v3 authorities")
    authorities.__post_init__()
    if type(maximum_checkpoint_bytes) is not int or maximum_checkpoint_bytes <= 0:
        raise ValueError("maximum_checkpoint_bytes must be positive")
    if required_inventory not in ("partial", "tuning", "complete"):
        raise ValueError("required_inventory is invalid")
    if tuning_selection is not None:
        if type(tuning_selection) is not H6TuningSelectionV3:
            raise ValueError("catalog tuning selection must be exact v3")
        tuning_selection.__post_init__()
    try:
        root_status = root.lstat()
        children = tuple(root.iterdir())
    except OSError as exc:
        raise ArtifactPublicationError("checkpoint catalog is unavailable") from exc
    if not stat.S_ISDIR(root_status.st_mode) or _is_redirect(root, root_status):
        raise ArtifactPublicationError("checkpoint catalog is not a safe directory")
    attempts = {
        attempt.planned_attempt_sha256: attempt
        for attempt in authorities.plan.attempts
    }
    names = {path.name for path in children}
    if len(names) != len(children) or not names <= set(attempts):
        raise ArtifactPublicationError(
            "checkpoint catalog inventory contains a foreign plan entry"
        )
    expected_tuning = {
        attempt.planned_attempt_sha256
        for attempt in authorities.plan.tuning_attempts
    }
    if required_inventory == "tuning" and names != expected_tuning:
        raise ArtifactPublicationError(
            "checkpoint catalog tuning inventory is incomplete"
        )
    if required_inventory == "complete" and names != set(attempts):
        raise ArtifactPublicationError(
            "checkpoint catalog complete inventory is incomplete"
        )

    items_by_attempt: dict[str, H6CheckpointCatalogItemV3] = {}
    for child in children:
        entry = _read_entry_directory(child)
        attempt = attempts.get(child.name)
        if attempt is None or entry.planned_attempt_sha256 != child.name:
            raise ArtifactPublicationError(
                "checkpoint catalog directory is foreign to the plan"
            )
        selection = (
            tuning_selection if attempt.stage == "confirmatory" else None
        )
        try:
            executable = bind_h6_executable_attempt_v3(
                authorities=authorities,
                planned_attempt=attempt,
                tuning_selection=selection,
            )
            checkpoint = read_h6_checkpoint_file_v3(
                Path(entry.checkpoint_path),
                maximum_bytes=maximum_checkpoint_bytes,
                expected_checkpoint_sha256=entry.checkpoint_sha256,
            )
            expected_entry = H6CheckpointCatalogEntryV3.create(
                executable_attempt=executable,
                checkpoint=checkpoint,
                checkpoint_path=Path(entry.checkpoint_path),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ArtifactPublicationError(
                "checkpoint catalog entry does not reopen its exact authority"
            ) from exc
        if expected_entry != entry:
            raise ArtifactPublicationError(
                "checkpoint catalog entry authority changed"
            )
        item = H6CheckpointCatalogItemV3(
            entry=entry,
            executable_attempt=executable,
            checkpoint=checkpoint,
        )
        if entry.planned_attempt_sha256 in items_by_attempt:
            raise ArtifactPublicationError(
                "checkpoint catalog contains a duplicate planned attempt"
            )
        items_by_attempt[entry.planned_attempt_sha256] = item

    plan_order = tuple(
        attempt.planned_attempt_sha256 for attempt in authorities.plan.attempts
    )
    ordered = tuple(
        items_by_attempt[digest]
        for digest in plan_order
        if digest in items_by_attempt
    )
    values = {
        "catalog_schema": "h6-checkpoint-catalog-v3",
        "authority_sha256": authorities.authority_sha256,
        "items": ordered,
    }
    payload = {
        "catalog_schema": values["catalog_schema"],
        "authority_sha256": values["authority_sha256"],
        "entries": tuple(item.entry.artifact_payload() for item in ordered),
    }
    catalog = H6CheckpointCatalogV3(
        **values,  # type: ignore[arg-type]
        catalog_snapshot_sha256=_hash(
            "vfe4.h6.checkpoint-catalog-snapshot.v3",
            payload,
        ),
        _plan_order=plan_order,
    )
    catalog.__post_init__()
    return catalog


__all__ = [
    "H6CheckpointCatalogEntryV3",
    "H6CheckpointCatalogItemV3",
    "H6CheckpointCatalogV3",
    "publish_h6_checkpoint_catalog_entry_v3",
    "read_h6_checkpoint_catalog_v3",
]
