"""Atomic WikiText-103 experiment/run lifecycle above generic durability."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from vfe4.types.training import (
    EndpointInventory,
    WT103CheckpointIdentity,
    owned_sha256,
)
from vfe4.recording.failures import FailureLogError, validate_failure_log

from .atomic import _rename_directory_no_replace, _run_component
from .durability import (
    DurableFileIdentity,
    DurabilityBackend,
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
    canonical_json_bytes_generic,
)
from .manifest import ArtifactIntegrityRecord


class RunLifecycleError(RuntimeError):
    """An experiment lifecycle transition failed closed."""


class AttemptExecutionLease:
    """Process-lifetime, nonblocking OS lease for one attempt."""

    __slots__ = ("path", "run_id", "_file_descriptor")

    def __init__(
        self,
        *,
        path: Path,
        run_id: str,
        file_descriptor: int,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._file_descriptor: int | None = file_descriptor

    @property
    def active(self) -> bool:
        return self._file_descriptor is not None

    def release(self) -> None:
        """Release explicitly; process exit/crash also releases the OS lock."""

        file_descriptor = self._file_descriptor
        if file_descriptor is None:
            return
        self._file_descriptor = None
        try:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.lockf(file_descriptor, fcntl.LOCK_UN, 1)
        finally:
            os.close(file_descriptor)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def _acquire_attempt_execution_lease(
    experiment_root: Path,
    run_id: str,
) -> AttemptExecutionLease:
    lease_parent = experiment_root / "runs" / ".execution-leases"
    try:
        lease_parent.mkdir(parents=True, exist_ok=True)
        _regular_directory(lease_parent)
    except RunLifecycleError:
        raise
    except OSError as exc:
        raise RunLifecycleError(
            "attempt execution lease directory is unavailable"
        ) from exc
    path = lease_parent / f"{run_id}.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(path, flags, 0o600)
        os.set_inheritable(file_descriptor, False)
        opened = os.fstat(file_descriptor)
        observed = path.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or bool(getattr(observed, "st_file_attributes", 0) & reparse)
            or (
                opened.st_ino
                and observed.st_ino
                and (
                    opened.st_dev != observed.st_dev
                    or opened.st_ino != observed.st_ino
                )
            )
        ):
            raise RunLifecycleError(
                "attempt execution lease must be a regular nonlink file"
            )
        if opened.st_size == 0:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            if os.write(file_descriptor, b"\0") != 1:
                raise RunLifecycleError(
                    "attempt execution lease initialization was incomplete"
                )
            os.fsync(file_descriptor)
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file_descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.lockf(
                    file_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                    1,
                )
        except OSError as exc:
            raise RunLifecycleError(
                f"attempt execution lease is already held: {run_id}"
            ) from exc
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        if os.fstat(file_descriptor).st_size != 1:
            raise RunLifecycleError(
                "attempt execution lease file is malformed"
            )
        if os.read(file_descriptor, 1) != b"\0":
            raise RunLifecycleError(
                "attempt execution lease file is malformed"
            )
        return AttemptExecutionLease(
            path=path,
            run_id=run_id,
            file_descriptor=file_descriptor,
        )
    except RunLifecycleError:
        if file_descriptor is not None:
            os.close(file_descriptor)
        raise
    except OSError as exc:
        if file_descriptor is not None:
            os.close(file_descriptor)
        raise RunLifecycleError(
            f"attempt execution lease acquisition failed: {run_id}"
        ) from exc


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _valid_sha256(value: object) -> bool:
    try:
        _sha256(value, "digest")
    except ValueError:
        return False
    return True


def _git_head(value: object) -> str:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("git_head must be a concrete lowercase object ID")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_artifact_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("expected artifact path must be canonical POSIX text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
        or path.name in {
            "experiment-plan.json",
            "failures.jsonl",
            "run-manifest.json",
            "reservation.json",
        }
    ):
        raise ValueError("expected artifact path is unsafe or reserved")
    return value


def _regular_nonlink(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise RunLifecycleError(f"artifact metadata unavailable: {path}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(path, "is_junction", None)
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
        or bool(is_junction is not None and is_junction())
    ):
        raise RunLifecycleError(f"artifact must be a regular nonlink file: {path}")
    return status


def _regular_directory(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as exc:
        raise RunLifecycleError(f"run directory metadata unavailable: {path}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(path, "is_junction", None)
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or bool(getattr(status, "st_file_attributes", 0) & reparse)
        or bool(is_junction is not None and is_junction())
    ):
        raise RunLifecycleError(f"run path must be a regular directory: {path}")
    return status


def _read_regular_bytes(path: Path, *, maximum_bytes: int = 16 * 1024 * 1024) -> bytes:
    status = _regular_nonlink(path)
    if status.st_size > maximum_bytes:
        raise RunLifecycleError(f"artifact exceeds bounded read size: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RunLifecycleError(f"artifact read failed: {path}") from exc
    if len(payload) != status.st_size:
        raise RunLifecycleError(f"artifact size changed while reading: {path}")
    return payload


def _json_object(payload: bytes, *, context: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunLifecycleError(f"{context} is not strict JSON") from exc
    if type(value) is not dict:
        raise RunLifecycleError(f"{context} must be a JSON object")
    if canonical_json_bytes_generic(value) != payload:
        raise RunLifecycleError(f"{context} is not canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Immutable pre-attempt plan with inventory-derived counts only."""

    schema_version: Literal["wt103-experiment-plan-v1"]
    experiment_id: str
    endpoint_inventory_sha256: str
    git_head: str
    dirty_digest: str
    config_sha256: str
    source_record_sha256: str
    tokenizer_spec_sha256: str
    token_cache_set_sha256: str
    window_manifest_sha256s: tuple[str, ...]
    schedule_set_sha256: str
    factory_set_sha256: str
    objective_sha256: str
    checkpoint_schema_sha256: str
    resource_forecast_sha256: str
    tuning_attempt_keys: tuple[str, ...]
    terminal_checkpoint_keys: tuple[str, ...]
    expected_run_artifact_paths: tuple[str, ...]
    expected_group_artifact_paths: tuple[str, ...]
    tuning_attempt_count: int
    terminal_checkpoint_count: int
    validation_endpoint_count: int
    test_endpoint_count: int
    raw_score_record_count: int
    result_row_count: int
    figure_panel_count: int
    figure_series_count: int
    experiment_plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-experiment-plan-v1":
            raise ValueError("experiment plan schema is invalid")
        _run_component(self.experiment_id)
        _git_head(self.git_head)
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
            _sha256(getattr(self, name), name)
        if (
            type(self.window_manifest_sha256s) is not tuple
            or len(self.window_manifest_sha256s) != 3
        ):
            raise ValueError("window manifest inventory must contain three splits")
        for value in self.window_manifest_sha256s:
            _sha256(value, "window_manifest_sha256")
        for name in (
            "expected_run_artifact_paths",
            "expected_group_artifact_paths",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} inventory is empty")
            normalized = tuple(
                _canonical_artifact_path(item) for item in values
            )
            if (
                normalized != tuple(sorted(normalized))
                or len(set(normalized)) != len(normalized)
            ):
                raise ValueError(
                    f"{name} must be sorted and unique"
                )
        if (
            type(self.tuning_attempt_keys) is not tuple
            or len(self.tuning_attempt_keys) != self.tuning_attempt_count
            or len(set(self.tuning_attempt_keys))
            != len(self.tuning_attempt_keys)
            or any(
                type(key) is not str
                or not key.startswith("tuning/")
                or "/cell=" not in key
                or "/seed=" not in key
                for key in self.tuning_attempt_keys
            )
        ):
            raise ValueError(
                "tuning attempt inventory must contain the exact distinct keys"
            )
        if (
            type(self.terminal_checkpoint_keys) is not tuple
            or len(self.terminal_checkpoint_keys) != 40
            or len(set(self.terminal_checkpoint_keys)) != 40
            or self.terminal_checkpoint_count != 40
            or any(
                type(key) is not str
                or not key.startswith("terminal/")
                or "/seed=" not in key
                for key in self.terminal_checkpoint_keys
            )
        ):
            raise ValueError(
                "terminal checkpoint inventory must contain 40 distinct keys"
            )
        for name in (
            "tuning_attempt_count",
            "terminal_checkpoint_count",
            "validation_endpoint_count",
            "test_endpoint_count",
            "raw_score_record_count",
            "result_row_count",
            "figure_panel_count",
            "figure_series_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a derived positive exact int")
        expected = owned_sha256(
            "vfe4.wt103.experiment-plan.v1",
            self.semantic_payload(),
        )
        _sha256(self.experiment_plan_sha256, "experiment_plan_sha256")
        if self.experiment_plan_sha256 != expected:
            raise ValueError("experiment plan hash does not match")

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        endpoint_inventory: EndpointInventory,
        git_head: str,
        dirty_digest: str,
        config_sha256: str,
        source_record_sha256: str,
        tokenizer_spec_sha256: str,
        token_cache_set_sha256: str,
        window_manifest_sha256s: tuple[str, ...],
        schedule_set_sha256: str,
        factory_set_sha256: str,
        objective_sha256: str,
        checkpoint_schema_sha256: str,
        resource_forecast_sha256: str,
        expected_run_artifact_paths: tuple[str, ...],
        expected_group_artifact_paths: tuple[str, ...],
    ) -> "ExperimentPlan":
        if type(endpoint_inventory) is not EndpointInventory:
            raise ValueError("endpoint_inventory must be exact")
        endpoint_inventory.__post_init__()
        payload = {
            "schema_version": "wt103-experiment-plan-v1",
            "experiment_id": experiment_id,
            "endpoint_inventory_sha256": (
                endpoint_inventory.endpoint_inventory_sha256
            ),
            "git_head": git_head,
            "dirty_digest": dirty_digest,
            "config_sha256": config_sha256,
            "source_record_sha256": source_record_sha256,
            "tokenizer_spec_sha256": tokenizer_spec_sha256,
            "token_cache_set_sha256": token_cache_set_sha256,
            "window_manifest_sha256s": window_manifest_sha256s,
            "schedule_set_sha256": schedule_set_sha256,
            "factory_set_sha256": factory_set_sha256,
            "objective_sha256": objective_sha256,
            "checkpoint_schema_sha256": checkpoint_schema_sha256,
            "resource_forecast_sha256": resource_forecast_sha256,
            "tuning_attempt_keys": endpoint_inventory.tuning_attempt_keys,
            "terminal_checkpoint_keys": (
                endpoint_inventory.terminal_checkpoint_keys
            ),
            "expected_run_artifact_paths": expected_run_artifact_paths,
            "expected_group_artifact_paths": (
                expected_group_artifact_paths
            ),
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
        return cls(
            **payload,
            experiment_plan_sha256=owned_sha256(
                "vfe4.wt103.experiment-plan.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentPlanIdentity:
    plan_path: Path
    plan: ExperimentPlan
    durable_file: DurableFileIdentity
    identity_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.plan_path, Path)
            or type(self.plan) is not ExperimentPlan
            or type(self.durable_file) is not DurableFileIdentity
            or self.plan_path.name != "experiment-plan.json"
        ):
            raise ValueError("experiment plan identity fields are invalid")
        self.plan.__post_init__()
        if (
            self.durable_file.sha256
            != hashlib.sha256(
                canonical_json_bytes_generic(self.plan)
            ).hexdigest()
            or self.durable_file.size_bytes
            != len(canonical_json_bytes_generic(self.plan))
        ):
            raise ValueError("durable plan identity does not bind plan bytes")
        expected = owned_sha256(
            "vfe4.wt103.experiment-plan-identity.v1",
            {
                "experiment_plan_sha256": (
                    self.plan.experiment_plan_sha256
                ),
                "durable_file_identity_sha256": (
                    self.durable_file.identity_sha256
                ),
            },
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("experiment plan identity hash does not match")


def publish_experiment_plan(
    experiment_root: Path,
    plan: ExperimentPlan,
    *,
    backend: DurabilityBackend,
) -> ExperimentPlanIdentity:
    """Durably create the immutable experiment plan before any run attempt."""

    if not isinstance(experiment_root, Path):
        raise RunLifecycleError("experiment_root must be pathlib.Path")
    if type(plan) is not ExperimentPlan:
        raise RunLifecycleError("plan must be an exact ExperimentPlan")
    plan.__post_init__()
    try:
        experiment_root.mkdir(parents=True, exist_ok=True)
        _regular_directory(experiment_root)
        path = experiment_root / "experiment-plan.json"
        payload = canonical_json_bytes_generic(plan)
        durable = backend.create_exclusive(path, payload)
        identity = ExperimentPlanIdentity(
            plan_path=path,
            plan=plan,
            durable_file=durable,
            identity_sha256=owned_sha256(
                "vfe4.wt103.experiment-plan-identity.v1",
                {
                    "experiment_plan_sha256": plan.experiment_plan_sha256,
                    "durable_file_identity_sha256": durable.identity_sha256,
                },
            ),
        )
        _validate_plan_identity(experiment_root, identity)
        return identity
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(
            f"experiment plan publication failed: {exc}"
        ) from exc


def _validate_plan_identity(
    experiment_root: Path,
    identity: ExperimentPlanIdentity,
) -> None:
    if type(identity) is not ExperimentPlanIdentity:
        raise RunLifecycleError("plan identity must be exact")
    identity.__post_init__()
    expected_path = experiment_root / "experiment-plan.json"
    if identity.plan_path != expected_path:
        raise RunLifecycleError("experiment plan path is not explicit/exact")
    payload = _read_regular_bytes(identity.plan_path)
    expected = canonical_json_bytes_generic(identity.plan)
    if (
        payload != expected
        or hashlib.sha256(payload).hexdigest() != identity.durable_file.sha256
    ):
        raise RunLifecycleError("experiment plan bytes do not match identity")


def _load_experiment_plan(experiment_root: Path) -> ExperimentPlan:
    payload = _json_object(
        _read_regular_bytes(experiment_root / "experiment-plan.json"),
        context="experiment plan",
    )
    if set(payload) != set(ExperimentPlan.__dataclass_fields__):
        raise RunLifecycleError("experiment plan key set is open")
    normalized = dict(payload)
    for name in (
        "window_manifest_sha256s",
        "tuning_attempt_keys",
        "terminal_checkpoint_keys",
        "expected_run_artifact_paths",
        "expected_group_artifact_paths",
    ):
        value = normalized[name]
        if type(value) is not list:
            raise RunLifecycleError(
                f"experiment plan {name} must be a JSON array"
            )
        normalized[name] = tuple(value)
    try:
        plan = ExperimentPlan(**normalized)  # type: ignore[arg-type]
        plan.__post_init__()
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError("experiment plan is invalid") from exc
    return plan


@dataclass(frozen=True, slots=True)
class ResumeLineageEvent:
    schema_version: Literal["wt103-resume-lineage-event-v1"]
    parent_checkpoint_identity_sha256: str
    parent_scientific_state_sha256: str
    parent_artifact_sha256: str
    environment_sha256: str
    cursor_sha256: str
    reason: str
    resumed_utc: str
    lineage_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-resume-lineage-event-v1":
            raise ValueError("resume lineage schema is invalid")
        for name in (
            "parent_checkpoint_identity_sha256",
            "parent_scientific_state_sha256",
            "parent_artifact_sha256",
            "environment_sha256",
            "cursor_sha256",
        ):
            _sha256(getattr(self, name), name)
        _text(self.reason, "reason")
        _text(self.resumed_utc, "resumed_utc")
        expected = owned_sha256(
            "vfe4.wt103.resume-lineage-event.v1",
            self.semantic_payload(),
        )
        _sha256(self.lineage_sha256, "lineage_sha256")
        if self.lineage_sha256 != expected:
            raise ValueError("resume lineage hash does not match")

    @classmethod
    def create(
        cls,
        *,
        parent_checkpoint: WT103CheckpointIdentity,
        environment_sha256: str,
        cursor_sha256: str,
        reason: str,
        resumed_utc: str,
    ) -> "ResumeLineageEvent":
        if type(parent_checkpoint) is not WT103CheckpointIdentity:
            raise ValueError("parent_checkpoint must be exact")
        parent_checkpoint.__post_init__()
        payload = {
            "schema_version": "wt103-resume-lineage-event-v1",
            "parent_checkpoint_identity_sha256": (
                parent_checkpoint.checkpoint_identity_sha256
            ),
            "parent_scientific_state_sha256": (
                parent_checkpoint.scientific_state_sha256
            ),
            "parent_artifact_sha256": parent_checkpoint.artifact_sha256,
            "environment_sha256": environment_sha256,
            "cursor_sha256": cursor_sha256,
            "reason": reason,
            "resumed_utc": resumed_utc,
        }
        return cls(
            **payload,
            lineage_sha256=owned_sha256(
                "vfe4.wt103.resume-lineage-event.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class ReservedRun:
    experiment_root: Path
    run_id: str
    run_role: Literal["tuning", "confirmation"]
    tuning_attempt_key: str | None
    started_utc: str
    inprogress_path: Path
    final_path: Path
    experiment_plan_sha256: str
    reservation_sha256: str
    resume_count: int
    resume_owner_lineage_sha256: str | None
    execution_lease: AttemptExecutionLease

    def __post_init__(self) -> None:
        if (
            not isinstance(self.experiment_root, Path)
            or not isinstance(self.inprogress_path, Path)
            or not isinstance(self.final_path, Path)
        ):
            raise ValueError("reserved run paths must be pathlib.Path")
        _run_component(self.run_id)
        if self.run_role not in ("tuning", "confirmation"):
            raise ValueError("reserved run role is invalid")
        if (
            (self.run_role == "tuning" and type(self.tuning_attempt_key) is not str)
            or (
                self.run_role == "confirmation"
                and self.tuning_attempt_key is not None
            )
        ):
            raise ValueError(
                "tuning attempt key presence must match the run role"
            )
        if self.tuning_attempt_key is not None:
            _text(self.tuning_attempt_key, "tuning_attempt_key")
        _text(self.started_utc, "started_utc")
        _sha256(
            self.experiment_plan_sha256,
            "experiment_plan_sha256",
        )
        _sha256(self.reservation_sha256, "reservation_sha256")
        if type(self.resume_count) is not int or self.resume_count < 0:
            raise ValueError("resume_count must be a nonnegative exact int")
        if self.resume_owner_lineage_sha256 is not None:
            _sha256(
                self.resume_owner_lineage_sha256,
                "resume_owner_lineage_sha256",
            )
        if (self.resume_count == 0) != (
            self.resume_owner_lineage_sha256 is None
        ):
            raise ValueError(
                "resume owner presence must match the resume count"
            )
        expected_lease_path = (
            self.experiment_root
            / "runs"
            / ".execution-leases"
            / f"{self.run_id}.lock"
        )
        if (
            type(self.execution_lease) is not AttemptExecutionLease
            or self.execution_lease.path != expected_lease_path
            or self.execution_lease.run_id != self.run_id
        ):
            raise ValueError(
                "reserved run does not bind its exact execution lease"
            )


def release_run_execution_lease(reserved: ReservedRun) -> None:
    """Release a reservation's process lease after execution terminates."""

    if type(reserved) is not ReservedRun:
        raise RunLifecycleError("reserved must be exact ReservedRun")
    reserved.execution_lease.release()


def _reservation_payload(
    run_id: str,
    run_role: Literal["tuning", "confirmation"],
    tuning_attempt_key: str | None,
    started_utc: str,
    experiment_plan_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-run-reservation-v1",
        "run_id": run_id,
        "run_role": run_role,
        "tuning_attempt_key": tuning_attempt_key,
        "started_utc": started_utc,
        "experiment_plan_sha256": experiment_plan_sha256,
    }
    return {
        **body,
        "reservation_sha256": owned_sha256(
            "vfe4.wt103.run-reservation.v1",
            body,
        ),
    }


def _validate_reservation(
    path: Path,
    *,
    run_id: str,
    run_role: Literal["tuning", "confirmation"],
    tuning_attempt_key: str | None,
    experiment_plan_sha256: str,
) -> tuple[str, str, str | None]:
    payload = _read_regular_bytes(path)
    decoded = _json_object(payload, context="run reservation")
    if set(decoded) != {
        "experiment_plan_sha256",
        "reservation_sha256",
        "run_id",
        "run_role",
        "schema_version",
        "started_utc",
        "tuning_attempt_key",
    }:
        raise RunLifecycleError("run reservation key set is open")
    body = dict(decoded)
    reservation_sha = body.pop("reservation_sha256", None)
    if (
        decoded["schema_version"] != "wt103-run-reservation-v1"
        or decoded["run_id"] != run_id
        or decoded["run_role"] != run_role
        or decoded["tuning_attempt_key"] != tuning_attempt_key
        or decoded["experiment_plan_sha256"]
        != experiment_plan_sha256
        or reservation_sha
        != owned_sha256(
            "vfe4.wt103.run-reservation.v1",
            body,
        )
    ):
        raise RunLifecycleError("run reservation bytes do not match")
    started_utc = decoded["started_utc"]
    _text(started_utc, "started_utc")
    return str(reservation_sha), started_utc, tuning_attempt_key


def _validate_lineage_ledger(payload: bytes) -> tuple[dict[str, object], ...]:
    if not payload:
        raise RunLifecycleError("resume lineage ledger is empty")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise RunLifecycleError("resume lineage has an incomplete final line")
        record = _json_object(line[:-1], context="resume lineage event")
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
            raise RunLifecycleError("resume lineage event key set is open")
        body = dict(record)
        observed = body.pop("lineage_sha256", None)
        if observed != owned_sha256(
            "vfe4.wt103.resume-lineage-event.v1",
            body,
        ):
            raise RunLifecycleError("resume lineage event hash does not match")
        records.append(record)
        if len(records) > 1:
            raise RunLifecycleError(
                "frozen infrastructure retry budget permits one lineage"
            )
    hashes = tuple(str(item["lineage_sha256"]) for item in records)
    if len(set(hashes)) != len(hashes):
        raise RunLifecycleError("resume lineage contains duplicate events")
    return tuple(records)


def _lineage_event_from_record(
    record: dict[str, object],
) -> ResumeLineageEvent:
    try:
        event = ResumeLineageEvent(**record)  # type: ignore[arg-type]
        event.__post_init__()
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError(
            "resume lineage event cannot be reopened"
        ) from exc
    return event


def _validate_resume_lineage_intent(path: Path) -> ResumeLineageEvent:
    payload = _read_regular_bytes(path)
    record = _json_object(payload, context="resume lineage intent")
    if canonical_json_bytes_generic(record) != payload:
        raise RunLifecycleError("resume lineage intent is not canonical")
    records = _validate_lineage_ledger(payload + b"\n")
    if len(records) != 1:
        raise RunLifecycleError("resume lineage intent is not exact")
    return _lineage_event_from_record(records[0])


def _publish_resume_lineage_intent(
    inprogress: Path,
    *,
    resume_lineage: ResumeLineageEvent,
    backend: DurabilityBackend,
) -> ResumeLineageEvent:
    intent_path = inprogress / "resume-lineage-intent.json"
    payload = canonical_json_bytes_generic(resume_lineage)
    if intent_path.exists():
        reopened = _validate_resume_lineage_intent(intent_path)
        if reopened != resume_lineage:
            raise RunLifecycleError(
                "frozen infrastructure retry budget is exhausted by "
                "another lineage intent"
            )
        return reopened
    try:
        backend.create_exclusive(intent_path, payload)
    except Exception as exc:
        if (
            not intent_path.exists()
            or _read_regular_bytes(intent_path) != payload
        ):
            raise RunLifecycleError(
                f"resume lineage intent publication failed: {exc}"
            ) from exc
    if _read_regular_bytes(intent_path) != payload:
        raise RunLifecycleError(
            "resume lineage intent did not commit exactly"
        )
    reopened = _validate_resume_lineage_intent(intent_path)
    if reopened != resume_lineage:
        raise RunLifecycleError(
            "resume lineage intent changed on durable reopen"
        )
    return reopened


def _require_resume_lineage_intent(
    run_path: Path,
    *,
    lineage_records: tuple[dict[str, object], ...],
) -> ResumeLineageEvent:
    intent = _validate_resume_lineage_intent(
        run_path / "resume-lineage-intent.json"
    )
    if (
        len(lineage_records) != 1
        or intent != _lineage_event_from_record(lineage_records[0])
    ):
        raise RunLifecycleError(
            "resume lineage ledger differs from its immutable intent"
        )
    return intent


def reopen_resume_lineage_event(
    inprogress_path: Path,
) -> ResumeLineageEvent | None:
    """Reopen immutable intent, including a pre-ledger crash recovery."""

    if not isinstance(inprogress_path, Path):
        raise RunLifecycleError("in-progress path must be pathlib.Path")
    _regular_directory(inprogress_path)
    intent_path = inprogress_path / "resume-lineage-intent.json"
    lineage_path = inprogress_path / "resume-lineage.jsonl"
    if not intent_path.exists() and not lineage_path.exists():
        return None
    if not intent_path.exists():
        raise RunLifecycleError(
            "resume lineage ledger lacks its immutable intent"
        )
    intent = _validate_resume_lineage_intent(intent_path)
    if not lineage_path.exists():
        return intent
    records = _validate_lineage_ledger(
        _read_regular_bytes(lineage_path)
    )
    if len(records) != 1:
        raise RunLifecycleError(
            "frozen infrastructure retry lineage is not exact"
        )
    event = _lineage_event_from_record(records[0])
    if event != intent:
        raise RunLifecycleError(
            "resume lineage ledger differs from its immutable intent"
        )
    return event


_ZERO_SHA256 = "0" * 64


def _resume_lease_payload(
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_ordinal: int,
    previous_lineage_sha256: str,
    lineage_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-resume-lease-v1",
        "run_id": run_id,
        "experiment_plan_sha256": experiment_plan_sha256,
        "reservation_sha256": reservation_sha256,
        "resume_ordinal": resume_ordinal,
        "previous_lineage_sha256": previous_lineage_sha256,
        "lineage_sha256": lineage_sha256,
    }
    return {
        **body,
        "lease_sha256": owned_sha256(
            "vfe4.wt103.resume-lease.v1",
            body,
        ),
    }


def _validate_resume_lease(
    path: Path,
    *,
    expected: dict[str, object],
) -> None:
    decoded = _json_object(
        _read_regular_bytes(path),
        context="resume lease",
    )
    if set(decoded) != {
        "experiment_plan_sha256",
        "lease_sha256",
        "lineage_sha256",
        "previous_lineage_sha256",
        "reservation_sha256",
        "resume_ordinal",
        "run_id",
        "schema_version",
    }:
        raise RunLifecycleError("resume lease key set is open")
    body = dict(decoded)
    observed = body.pop("lease_sha256", None)
    if (
        decoded != expected
        or observed
        != owned_sha256(
            "vfe4.wt103.resume-lease.v1",
            body,
        )
    ):
        raise RunLifecycleError("resume lease does not match its CAS event")


def _resume_owner_payload(
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_ordinal: int,
    previous_owner_lineage_sha256: str,
    lineage_sha256: str,
    state: Literal["active", "terminal_closed"],
    terminal_manifest_sha256: str | None,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-resume-owner-v1",
        "run_id": run_id,
        "experiment_plan_sha256": experiment_plan_sha256,
        "reservation_sha256": reservation_sha256,
        "resume_ordinal": resume_ordinal,
        "previous_owner_lineage_sha256": previous_owner_lineage_sha256,
        "lineage_sha256": lineage_sha256,
        "state": state,
        "terminal_manifest_sha256": terminal_manifest_sha256,
    }
    return {
        **body,
        "owner_sha256": owned_sha256(
            "vfe4.wt103.resume-owner.v1",
            body,
        ),
    }


def _validate_resume_owner(
    path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
) -> dict[str, object]:
    decoded = _json_object(
        _read_regular_bytes(path),
        context="resume owner",
    )
    if set(decoded) != {
        "experiment_plan_sha256",
        "lineage_sha256",
        "owner_sha256",
        "previous_owner_lineage_sha256",
        "reservation_sha256",
        "resume_ordinal",
        "run_id",
        "schema_version",
        "state",
        "terminal_manifest_sha256",
    }:
        raise RunLifecycleError("resume owner key set is open")
    body = dict(decoded)
    observed = body.pop("owner_sha256", None)
    state = decoded["state"]
    terminal_manifest_sha256 = decoded["terminal_manifest_sha256"]
    if (
        decoded["schema_version"] != "wt103-resume-owner-v1"
        or decoded["run_id"] != run_id
        or decoded["experiment_plan_sha256"] != experiment_plan_sha256
        or decoded["reservation_sha256"] != reservation_sha256
        or type(decoded["resume_ordinal"]) is not int
        or int(decoded["resume_ordinal"]) < 1
        or state not in ("active", "terminal_closed")
        or (
            state == "active"
            and terminal_manifest_sha256 is not None
        )
        or (
            state == "terminal_closed"
            and not _valid_sha256(terminal_manifest_sha256)
        )
        or not _valid_sha256(decoded["lineage_sha256"])
        or not _valid_sha256(decoded["previous_owner_lineage_sha256"])
        or observed
        != owned_sha256(
            "vfe4.wt103.resume-owner.v1",
            body,
        )
    ):
        raise RunLifecycleError("resume owner is invalid")
    return decoded


def _resume_execution_started_payload(
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_ordinal: int,
    lineage_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-resume-execution-started-v1",
        "run_id": run_id,
        "experiment_plan_sha256": experiment_plan_sha256,
        "reservation_sha256": reservation_sha256,
        "resume_ordinal": resume_ordinal,
        "lineage_sha256": lineage_sha256,
    }
    return {
        **body,
        "execution_started_sha256": owned_sha256(
            "vfe4.wt103.resume-execution-started.v1",
            body,
        ),
    }


def _validate_resume_execution_started(
    path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
) -> dict[str, object]:
    decoded = _json_object(
        _read_regular_bytes(path),
        context="resume execution-started transition",
    )
    if set(decoded) != {
        "execution_started_sha256",
        "experiment_plan_sha256",
        "lineage_sha256",
        "reservation_sha256",
        "resume_ordinal",
        "run_id",
        "schema_version",
    }:
        raise RunLifecycleError(
            "resume execution-started transition key set is open"
        )
    body = dict(decoded)
    observed = body.pop("execution_started_sha256", None)
    if (
        decoded["schema_version"]
        != "wt103-resume-execution-started-v1"
        or decoded["run_id"] != run_id
        or decoded["experiment_plan_sha256"] != experiment_plan_sha256
        or decoded["reservation_sha256"] != reservation_sha256
        or decoded["resume_ordinal"] != 1
        or not _valid_sha256(decoded["lineage_sha256"])
        or observed
        != owned_sha256(
            "vfe4.wt103.resume-execution-started.v1",
            body,
        )
    ):
        raise RunLifecycleError(
            "resume execution-started transition is invalid"
        )
    return decoded


def _require_resume_execution_started(
    run_path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    lineage_records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    marker = _validate_resume_execution_started(
        run_path / "resume-execution-started.json",
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    if (
        len(lineage_records) != 1
        or marker["resume_ordinal"] != 1
        or marker["lineage_sha256"]
        != lineage_records[0]["lineage_sha256"]
    ):
        raise RunLifecycleError(
            "resume execution-started transition differs from its lineage"
        )
    return marker


def _reject_consumed_resume_retry(
    inprogress: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
) -> None:
    marker_path = inprogress / "resume-execution-started.json"
    if not marker_path.exists():
        return
    lineage_path = inprogress / "resume-lineage.jsonl"
    records = _validate_lineage_ledger(_read_regular_bytes(lineage_path))
    _require_resume_lineage_intent(
        inprogress,
        lineage_records=records,
    )
    marker = _require_resume_execution_started(
        inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        lineage_records=records,
    )
    _validate_resume_lease_inventory(
        inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        records=records,
    )
    owner = _validate_resume_owner(
        inprogress / "resume-owner.json",
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    if (
        owner["resume_ordinal"] != 1
        or owner["lineage_sha256"] != marker["lineage_sha256"]
    ):
        raise RunLifecycleError(
            "resume execution-started transition differs from its owner"
        )
    raise RunLifecycleError(
        "frozen infrastructure retry is already consumed"
    )


def _resume_takeover_payload(
    *,
    run_id: str,
    previous_owner_sha256: str,
    previous_owner_lineage_sha256: str,
    replacement_owner_sha256: str,
    replacement_owner_lineage_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-resume-owner-takeover-v1",
        "run_id": run_id,
        "previous_owner_sha256": previous_owner_sha256,
        "previous_owner_lineage_sha256": previous_owner_lineage_sha256,
        "replacement_owner_sha256": replacement_owner_sha256,
        "replacement_owner_lineage_sha256": (
            replacement_owner_lineage_sha256
        ),
    }
    return {
        **body,
        "takeover_sha256": owned_sha256(
            "vfe4.wt103.resume-owner-takeover.v1",
            body,
        ),
    }


def _acquire_resume_owner(
    *,
    inprogress: Path,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_ordinal: int,
    lineage_sha256: str,
    expected_owner_lineage_sha256: str | None,
    backend: DurabilityBackend,
) -> dict[str, object]:
    owner_path = inprogress / "resume-owner.json"
    if not owner_path.exists():
        if expected_owner_lineage_sha256 is not None:
            raise RunLifecycleError(
                "expected owner was supplied but no active resume owner exists"
            )
        desired = _resume_owner_payload(
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            resume_ordinal=resume_ordinal,
            previous_owner_lineage_sha256=_ZERO_SHA256,
            lineage_sha256=lineage_sha256,
            state="active",
            terminal_manifest_sha256=None,
        )
        desired_payload = canonical_json_bytes_generic(desired)
        try:
            backend.create_exclusive(owner_path, desired_payload)
        except Exception as exc:
            if (
                not owner_path.exists()
                or _read_regular_bytes(owner_path) != desired_payload
            ):
                raise RunLifecycleError(
                    f"active resume owner acquisition failed: {exc}"
                ) from exc
        if _read_regular_bytes(owner_path) != desired_payload:
            raise RunLifecycleError(
                "active resume owner reopen validation failed"
            )
        return desired

    current_payload = _read_regular_bytes(owner_path)
    current = _validate_resume_owner(
        owner_path,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    if current["state"] != "active":
        raise RunLifecycleError("resume owner is already terminally closed")
    if current["lineage_sha256"] == lineage_sha256:
        if (
            current["resume_ordinal"] != resume_ordinal
            or expected_owner_lineage_sha256 not in (None, lineage_sha256)
        ):
            raise RunLifecycleError(
                "active resume owner retry does not match its ordinal"
            )
        return current
    if expected_owner_lineage_sha256 is None:
        raise RunLifecycleError(
            "active resume owner blocks another resume mutation"
        )
    if current["lineage_sha256"] != expected_owner_lineage_sha256:
        raise RunLifecycleError(
            "expected owner does not match the active resume owner"
        )
    desired = _resume_owner_payload(
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=resume_ordinal,
        previous_owner_lineage_sha256=expected_owner_lineage_sha256,
        lineage_sha256=lineage_sha256,
        state="active",
        terminal_manifest_sha256=None,
    )
    takeover = _resume_takeover_payload(
        run_id=run_id,
        previous_owner_sha256=str(current["owner_sha256"]),
        previous_owner_lineage_sha256=expected_owner_lineage_sha256,
        replacement_owner_sha256=str(desired["owner_sha256"]),
        replacement_owner_lineage_sha256=lineage_sha256,
    )
    transitions = inprogress / "resume-owner-takeovers"
    transitions.mkdir(exist_ok=True)
    _regular_directory(transitions)
    transition_path = transitions / (
        f"{current['owner_sha256']}.json"
    )
    transition_payload = canonical_json_bytes_generic(takeover)
    preexisting = transition_path.exists()
    try:
        backend.create_exclusive(transition_path, transition_payload)
    except Exception as exc:
        if (
            not preexisting
            or not transition_path.exists()
            or _read_regular_bytes(transition_path) != transition_payload
        ):
            raise RunLifecycleError(
                f"resume owner takeover acquisition failed: {exc}"
            ) from exc
    if _read_regular_bytes(transition_path) != transition_payload:
        raise RunLifecycleError("resume owner takeover reopen failed")
    if _read_regular_bytes(owner_path) != current_payload:
        raise RunLifecycleError("resume owner takeover CAS changed")
    desired_payload = canonical_json_bytes_generic(desired)
    backend.replace_durable(owner_path, desired_payload)
    if _read_regular_bytes(owner_path) != desired_payload:
        raise RunLifecycleError("resume owner takeover did not commit")
    return desired


def _close_resume_owner(
    run_path: Path,
    *,
    reserved: ReservedRun,
    manifest_sha256: str,
    backend: DurabilityBackend,
) -> None:
    owner_path = run_path / "resume-owner.json"
    if reserved.resume_count == 0:
        if owner_path.exists():
            raise RunLifecycleError(
                "unresumed run cannot contain a resume owner"
            )
        return
    if reserved.resume_owner_lineage_sha256 is None:
        raise RunLifecycleError("resumed run lacks its owner identity")
    current_payload = _read_regular_bytes(owner_path)
    current = _validate_resume_owner(
        owner_path,
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
    )
    if (
        current["resume_ordinal"] != reserved.resume_count
        or current["lineage_sha256"]
        != reserved.resume_owner_lineage_sha256
    ):
        raise RunLifecycleError(
            "reserved run is no longer the active resume owner"
        )
    terminal = _resume_owner_payload(
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
        resume_ordinal=reserved.resume_count,
        previous_owner_lineage_sha256=str(
            current["previous_owner_lineage_sha256"]
        ),
        lineage_sha256=reserved.resume_owner_lineage_sha256,
        state="terminal_closed",
        terminal_manifest_sha256=manifest_sha256,
    )
    terminal_payload = canonical_json_bytes_generic(terminal)
    if current["state"] == "terminal_closed":
        if current_payload != terminal_payload:
            raise RunLifecycleError(
                "terminal resume owner differs from retry transition"
            )
        return
    backend.replace_durable(owner_path, terminal_payload)
    if _read_regular_bytes(owner_path) != terminal_payload:
        raise RunLifecycleError(
            "terminal resume owner reopen validation failed"
        )


def _require_active_resume_owner(
    run_path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_ordinal: int,
    lineage_sha256: str,
) -> None:
    owner = _validate_resume_owner(
        run_path / "resume-owner.json",
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    if (
        owner["state"] != "active"
        or owner["resume_ordinal"] != resume_ordinal
        or owner["lineage_sha256"] != lineage_sha256
    ):
        raise RunLifecycleError(
            "resume mutation no longer owns the active run lease"
        )


def _validate_terminal_resume_owner(
    run_path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_count: int,
    lineage_records: tuple[dict[str, object], ...],
    manifest_sha256: str,
    allow_active_recovery: bool = False,
) -> Literal["absent", "active", "terminal_closed"]:
    if type(allow_active_recovery) is not bool:
        raise RunLifecycleError(
            "active-owner recovery authority must be exact bool"
        )
    owner_path = run_path / "resume-owner.json"
    if resume_count == 0:
        if owner_path.exists():
            raise RunLifecycleError(
                "unresumed terminal run contains a resume owner"
            )
        return "absent"
    owner = _validate_resume_owner(
        owner_path,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    if (
        len(lineage_records) != resume_count
        or owner["resume_ordinal"] != resume_count
        or owner["lineage_sha256"]
        != lineage_records[-1]["lineage_sha256"]
        or owner["previous_owner_lineage_sha256"]
        != (
            _ZERO_SHA256
            if resume_count == 1
            else lineage_records[-2]["lineage_sha256"]
        )
    ):
        raise RunLifecycleError(
            "terminal resume owner does not close the manifest lineage"
        )
    expected = _resume_owner_payload(
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=resume_count,
        previous_owner_lineage_sha256=str(
            owner["previous_owner_lineage_sha256"]
        ),
        lineage_sha256=str(lineage_records[-1]["lineage_sha256"]),
        state=owner["state"],  # type: ignore[arg-type]
        terminal_manifest_sha256=(
            manifest_sha256
            if owner["state"] == "terminal_closed"
            else None
        ),
    )
    if owner != expected:
        raise RunLifecycleError(
            "terminal resume owner does not close the manifest lineage"
        )
    if owner["state"] == "terminal_closed":
        return "terminal_closed"
    if allow_active_recovery and owner["state"] == "active":
        return "active"
    raise RunLifecycleError(
        "terminal resume owner does not close the manifest lineage"
    )


def _lineage_payload(
    records: tuple[dict[str, object], ...],
) -> bytes:
    return b"".join(
        canonical_json_bytes_generic(record) + b"\n"
        for record in records
    )


def _validate_resume_lease_inventory(
    run_path: Path,
    *,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    records: tuple[dict[str, object], ...],
) -> None:
    leases_path = run_path / "resume-leases"
    takeovers_path = run_path / "resume-owner-takeovers"
    if not records:
        if leases_path.exists():
            raise RunLifecycleError(
                "resume lease inventory exists without lineage"
            )
        if takeovers_path.exists():
            raise RunLifecycleError(
                "resume takeover inventory exists without lineage"
            )
        return
    _regular_directory(leases_path)
    expected_names: list[str] = []
    prefix: list[dict[str, object]] = []
    for ordinal, record in enumerate(records, start=1):
        name = f"{ordinal:08d}.json"
        expected_names.append(name)
        previous_payload = _lineage_payload(tuple(prefix))
        expected = _resume_lease_payload(
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            resume_ordinal=ordinal,
            previous_lineage_sha256=(
                hashlib.sha256(previous_payload).hexdigest()
                if previous_payload
                else _ZERO_SHA256
            ),
            lineage_sha256=str(record["lineage_sha256"]),
        )
        _validate_resume_lease(leases_path / name, expected=expected)
        prefix.append(record)
    try:
        observed_names = tuple(
            sorted(item.name for item in leases_path.iterdir())
        )
    except OSError as exc:
        raise RunLifecycleError("resume lease inventory is unreadable") from exc
    if observed_names != tuple(expected_names):
        raise RunLifecycleError("resume lease ordinal inventory is not exact")
    if len(records) == 1:
        if takeovers_path.exists():
            raise RunLifecycleError(
                "single-owner lineage cannot contain takeover records"
            )
        return
    _regular_directory(takeovers_path)
    expected_takeovers: dict[str, bytes] = {}
    for ordinal in range(2, len(records) + 1):
        previous_record = records[ordinal - 2]
        replacement_record = records[ordinal - 1]
        previous_owner = _resume_owner_payload(
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            resume_ordinal=ordinal - 1,
            previous_owner_lineage_sha256=(
                _ZERO_SHA256
                if ordinal == 2
                else str(records[ordinal - 3]["lineage_sha256"])
            ),
            lineage_sha256=str(previous_record["lineage_sha256"]),
            state="active",
            terminal_manifest_sha256=None,
        )
        replacement_owner = _resume_owner_payload(
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            resume_ordinal=ordinal,
            previous_owner_lineage_sha256=str(
                previous_record["lineage_sha256"]
            ),
            lineage_sha256=str(replacement_record["lineage_sha256"]),
            state="active",
            terminal_manifest_sha256=None,
        )
        takeover = _resume_takeover_payload(
            run_id=run_id,
            previous_owner_sha256=str(previous_owner["owner_sha256"]),
            previous_owner_lineage_sha256=str(
                previous_record["lineage_sha256"]
            ),
            replacement_owner_sha256=str(
                replacement_owner["owner_sha256"]
            ),
            replacement_owner_lineage_sha256=str(
                replacement_record["lineage_sha256"]
            ),
        )
        expected_takeovers[
            f"{previous_owner['owner_sha256']}.json"
        ] = canonical_json_bytes_generic(takeover)
    try:
        observed_takeovers = {
            item.name: _read_regular_bytes(item)
            for item in takeovers_path.iterdir()
        }
    except OSError as exc:
        raise RunLifecycleError(
            "resume takeover inventory is unreadable"
        ) from exc
    if observed_takeovers != expected_takeovers:
        raise RunLifecycleError(
            "resume takeover inventory is not exact"
        )


def _append_resume_lineage_with_lease(
    *,
    inprogress: Path,
    run_id: str,
    experiment_plan_sha256: str,
    reservation_sha256: str,
    resume_lineage: ResumeLineageEvent,
    expected_owner_lineage_sha256: str | None,
    backend: DurabilityBackend,
) -> int:
    intent = _publish_resume_lineage_intent(
        inprogress,
        resume_lineage=resume_lineage,
        backend=backend,
    )
    lineage_path = inprogress / "resume-lineage.jsonl"
    if lineage_path.exists():
        previous = _read_regular_bytes(lineage_path)
        records = _validate_lineage_ledger(previous)
        _require_resume_lineage_intent(
            inprogress,
            lineage_records=records,
        )
    else:
        previous = b""
        records = ()
    if intent != resume_lineage:
        raise RunLifecycleError(
            "resume lineage mutation differs from its immutable intent"
        )
    matching = tuple(
        index
        for index, record in enumerate(records)
        if record["lineage_sha256"] == resume_lineage.lineage_sha256
    )
    if matching and matching != (len(records) - 1,):
        raise RunLifecycleError(
            "resume event ownership is already claimed"
        )
    claimed_lineages = {
        str(record["lineage_sha256"]) for record in records
    }
    owner_path = inprogress / "resume-owner.json"
    if owner_path.exists():
        owner = _validate_resume_owner(
            owner_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
        )
        claimed_lineages.add(str(owner["lineage_sha256"]))
    first_lease_path = inprogress / "resume-leases" / "00000001.json"
    if first_lease_path.exists():
        first_lease = _json_object(
            _read_regular_bytes(first_lease_path),
            context="resume lease",
        )
        claimed = first_lease.get("lineage_sha256")
        if not _valid_sha256(claimed):
            raise RunLifecycleError(
                "resume lease claim has an invalid lineage"
            )
        expected_first_lease = _resume_lease_payload(
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            resume_ordinal=1,
            previous_lineage_sha256=_ZERO_SHA256,
            lineage_sha256=str(claimed),
        )
        _validate_resume_lease(
            first_lease_path,
            expected=expected_first_lease,
        )
        claimed_lineages.add(str(claimed))
    if claimed_lineages - {resume_lineage.lineage_sha256}:
        raise RunLifecycleError(
            "frozen infrastructure retry budget is exhausted"
        )
    if records and not matching:
        raise RunLifecycleError(
            "frozen infrastructure retry budget is exhausted"
        )

    ordinal = matching[0] + 1 if matching else len(records) + 1
    _acquire_resume_owner(
        inprogress=inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=ordinal,
        lineage_sha256=resume_lineage.lineage_sha256,
        expected_owner_lineage_sha256=expected_owner_lineage_sha256,
        backend=backend,
    )
    if matching:
        _validate_resume_lease_inventory(
            inprogress,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha256,
            reservation_sha256=reservation_sha256,
            records=records,
        )
        return len(records)

    leases_path = inprogress / "resume-leases"
    leases_path.mkdir(exist_ok=True)
    _regular_directory(leases_path)
    expected = _resume_lease_payload(
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=ordinal,
        previous_lineage_sha256=(
            hashlib.sha256(previous).hexdigest()
            if previous
            else _ZERO_SHA256
        ),
        lineage_sha256=resume_lineage.lineage_sha256,
    )
    lease_path = leases_path / f"{ordinal:08d}.json"
    lease_payload = canonical_json_bytes_generic(expected)
    preexisting = lease_path.exists()
    try:
        backend.create_exclusive(lease_path, lease_payload)
    except Exception as exc:
        if not preexisting:
            raise RunLifecycleError(
                f"resume append-CAS acquisition failed at ordinal "
                f"{ordinal}: {exc}"
            ) from exc
        if _read_regular_bytes(lease_path) != lease_payload:
            raise RunLifecycleError(
                f"resume ordinal ownership is already claimed:{ordinal}"
            ) from exc
    if _read_regular_bytes(lease_path) != lease_payload:
        raise RunLifecycleError("resume lease reopen validation failed")
    _require_active_resume_owner(
        inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=ordinal,
        lineage_sha256=resume_lineage.lineage_sha256,
    )
    current = (
        _read_regular_bytes(lineage_path)
        if lineage_path.exists()
        else b""
    )
    if current != previous:
        raise RunLifecycleError("resume lineage CAS ordinal changed")
    line = canonical_json_bytes_generic(resume_lineage) + b"\n"
    if previous:
        backend.replace_durable(lineage_path, previous + line)
    else:
        backend.create_exclusive(lineage_path, line)
    appended = _read_regular_bytes(lineage_path)
    _require_active_resume_owner(
        inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=ordinal,
        lineage_sha256=resume_lineage.lineage_sha256,
    )
    appended_records = _validate_lineage_ledger(appended)
    if (
        len(appended_records) != ordinal
        or appended_records[-1]["lineage_sha256"]
        != resume_lineage.lineage_sha256
    ):
        raise RunLifecycleError("resume lineage append did not commit CAS event")
    _require_resume_lineage_intent(
        inprogress,
        lineage_records=appended_records,
    )
    _validate_resume_lease_inventory(
        inprogress,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        records=appended_records,
    )
    return ordinal


def consume_resume_execution_retry(
    reserved: ReservedRun,
    *,
    backend: DurabilityBackend,
) -> str:
    """Durably consume the one retry immediately before resumed execution."""

    if type(reserved) is not ReservedRun:
        raise RunLifecycleError("reserved must be exact ReservedRun")
    reserved.__post_init__()
    if not reserved.execution_lease.active:
        raise RunLifecycleError(
            "retry execution start requires the active attempt execution lease"
        )
    if (
        reserved.resume_count != 1
        or reserved.resume_owner_lineage_sha256 is None
    ):
        raise RunLifecycleError(
            "retry execution start requires the one exact resumed run"
        )
    if reserved.final_path.exists():
        raise RunLifecycleError(
            "terminal run cannot start resumed execution"
        )
    _regular_directory(reserved.inprogress_path)
    plan = _load_experiment_plan(reserved.experiment_root)
    if plan.experiment_plan_sha256 != reserved.experiment_plan_sha256:
        raise RunLifecycleError(
            "retry execution start differs from its experiment plan"
        )
    reservation_sha, started_utc, tuning_attempt_key = _validate_reservation(
        reserved.inprogress_path / "reservation.json",
        run_id=reserved.run_id,
        run_role=reserved.run_role,
        tuning_attempt_key=reserved.tuning_attempt_key,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
    )
    if (
        reservation_sha != reserved.reservation_sha256
        or started_utc != reserved.started_utc
        or tuning_attempt_key != reserved.tuning_attempt_key
    ):
        raise RunLifecycleError(
            "retry execution start differs from its durable reservation"
        )
    lineage_path = reserved.inprogress_path / "resume-lineage.jsonl"
    records = _validate_lineage_ledger(_read_regular_bytes(lineage_path))
    if (
        len(records) != 1
        or records[0]["lineage_sha256"]
        != reserved.resume_owner_lineage_sha256
    ):
        raise RunLifecycleError(
            "retry execution start differs from its one resume lineage"
        )
    _require_resume_lineage_intent(
        reserved.inprogress_path,
        lineage_records=records,
    )
    _validate_resume_lease_inventory(
        reserved.inprogress_path,
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
        records=records,
    )
    _require_active_resume_owner(
        reserved.inprogress_path,
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
        resume_ordinal=1,
        lineage_sha256=reserved.resume_owner_lineage_sha256,
    )
    marker_path = (
        reserved.inprogress_path / "resume-execution-started.json"
    )
    if marker_path.exists():
        _validate_resume_execution_started(
            marker_path,
            run_id=reserved.run_id,
            experiment_plan_sha256=reserved.experiment_plan_sha256,
            reservation_sha256=reserved.reservation_sha256,
        )
        raise RunLifecycleError(
            "frozen infrastructure retry is already consumed"
        )
    expected = _resume_execution_started_payload(
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
        resume_ordinal=1,
        lineage_sha256=reserved.resume_owner_lineage_sha256,
    )
    payload = canonical_json_bytes_generic(expected)
    try:
        backend.create_exclusive(marker_path, payload)
    except Exception as exc:
        if (
            not marker_path.exists()
            or _read_regular_bytes(marker_path) != payload
        ):
            raise RunLifecycleError(
                f"retry execution-start transition failed: {exc}"
            ) from exc
    if _read_regular_bytes(marker_path) != payload:
        raise RunLifecycleError(
            "retry execution-start transition did not commit"
        )
    reopened = _validate_resume_execution_started(
        marker_path,
        run_id=reserved.run_id,
        experiment_plan_sha256=reserved.experiment_plan_sha256,
        reservation_sha256=reserved.reservation_sha256,
    )
    if reopened != expected:
        raise RunLifecycleError(
            "retry execution-start transition changed on reopen"
        )
    return str(reopened["execution_started_sha256"])


def reserve_run(
    experiment_root: Path,
    run_id: str,
    *,
    run_role: Literal["tuning", "confirmation"],
    started_utc: str | None,
    plan: ExperimentPlanIdentity,
    backend: DurabilityBackend,
    mode: Literal["new", "resume"] = "new",
    resume_lineage: ResumeLineageEvent | None = None,
    expected_resume_owner_lineage_sha256: str | None = None,
    tuning_attempt_key: str | None = None,
) -> ReservedRun:
    """Reserve one explicit run or explicitly resume one retained crash."""

    execution_lease: AttemptExecutionLease | None = None
    try:
        _validate_plan_identity(experiment_root, plan)
        safe_id = _run_component(run_id)
        if run_role not in ("tuning", "confirmation"):
            raise RunLifecycleError("run role is invalid")
        if run_role == "tuning":
            if tuning_attempt_key not in plan.plan.tuning_attempt_keys:
                raise RunLifecycleError(
                    "tuning run requires one exact planned attempt key"
                )
        elif tuning_attempt_key is not None:
            raise RunLifecycleError(
                "confirmation run cannot claim a tuning attempt key"
            )
        if mode not in ("new", "resume"):
            raise RunLifecycleError("run reservation mode is invalid")
        inprogress_parent = experiment_root / "runs" / ".inprogress"
        final_parent = experiment_root / "runs"
        inprogress = inprogress_parent / safe_id
        final = final_parent / safe_id
        if final.exists():
            raise RunLifecycleError(f"terminal run already exists: {safe_id}")
        execution_lease = _acquire_attempt_execution_lease(
            experiment_root,
            safe_id,
        )
        reservation_path = inprogress / "reservation.json"
        if mode == "new":
            if resume_lineage is not None:
                raise RunLifecycleError("new run cannot carry resume lineage")
            if expected_resume_owner_lineage_sha256 is not None:
                raise RunLifecycleError(
                    "new run cannot carry an expected resume owner"
                )
            if started_utc is None:
                raise RunLifecycleError(
                    "new run requires an explicit start UTC"
                )
            _text(started_utc, "started_utc")
            expected = _reservation_payload(
                safe_id,
                run_role,
                tuning_attempt_key,
                started_utc,
                plan.plan.experiment_plan_sha256,
            )
            inprogress_parent.mkdir(parents=True, exist_ok=True)
            _regular_directory(inprogress_parent)
            try:
                inprogress.mkdir()
            except FileExistsError:
                _regular_directory(inprogress)
                try:
                    entries = tuple(
                        sorted(item.name for item in inprogress.iterdir())
                    )
                except OSError as exc:
                    raise RunLifecycleError(
                        "run reservation recovery directory is unreadable"
                    ) from exc
                if entries:
                    raise RunLifecycleError(
                        f"run reservation already exists: {safe_id}"
                    )
            reservation_payload = canonical_json_bytes_generic(expected)
            try:
                backend.create_exclusive(
                    reservation_path,
                    reservation_payload,
                )
            except Exception as exc:
                raise RunLifecycleError(
                    f"reservation publication failed: {exc}"
                ) from exc
            if _read_regular_bytes(
                reservation_path
            ) != reservation_payload:
                raise RunLifecycleError(
                    "reservation reopen validation failed"
                )
            (
                reservation_sha,
                _reservation_started,
                _reservation_tuning_attempt_key,
            ) = _validate_reservation(
                reservation_path,
                run_id=safe_id,
                run_role=run_role,
                tuning_attempt_key=tuning_attempt_key,
                experiment_plan_sha256=(
                    plan.plan.experiment_plan_sha256
                )
            )
            resume_count = 0
            resume_owner_lineage_sha256 = None
        else:
            if started_utc is not None:
                raise RunLifecycleError(
                    "resume reads the immutable reservation start UTC"
                )
            if type(resume_lineage) is not ResumeLineageEvent:
                raise RunLifecycleError(
                    "explicit resume requires exact resume lineage"
                )
            resume_lineage.__post_init__()
            _regular_directory(inprogress)
            (
                reservation_sha,
                _reservation_started,
                _reservation_tuning_attempt_key,
            ) = _validate_reservation(
                reservation_path,
                run_id=safe_id,
                run_role=run_role,
                tuning_attempt_key=tuning_attempt_key,
                experiment_plan_sha256=(
                    plan.plan.experiment_plan_sha256
                ),
            )
            _reject_consumed_resume_retry(
                inprogress,
                run_id=safe_id,
                experiment_plan_sha256=(
                    plan.plan.experiment_plan_sha256
                ),
                reservation_sha256=reservation_sha,
            )
            resume_count = _append_resume_lineage_with_lease(
                inprogress=inprogress,
                run_id=safe_id,
                experiment_plan_sha256=(
                    plan.plan.experiment_plan_sha256
                ),
                reservation_sha256=reservation_sha,
                resume_lineage=resume_lineage,
                expected_owner_lineage_sha256=(
                    expected_resume_owner_lineage_sha256
                ),
                backend=backend,
            )
            resume_owner_lineage_sha256 = resume_lineage.lineage_sha256
        (
            reservation_sha,
            reservation_started_utc,
            reservation_tuning_attempt_key,
        ) = _validate_reservation(
            reservation_path,
            run_id=safe_id,
            run_role=run_role,
            tuning_attempt_key=tuning_attempt_key,
            experiment_plan_sha256=plan.plan.experiment_plan_sha256,
        )
        return ReservedRun(
            experiment_root=experiment_root,
            run_id=safe_id,
            run_role=run_role,
            tuning_attempt_key=reservation_tuning_attempt_key,
            started_utc=reservation_started_utc,
            inprogress_path=inprogress,
            final_path=final,
            experiment_plan_sha256=plan.plan.experiment_plan_sha256,
            reservation_sha256=reservation_sha,
            resume_count=resume_count,
            resume_owner_lineage_sha256=resume_owner_lineage_sha256,
            execution_lease=execution_lease,
        )
    except RunLifecycleError:
        if execution_lease is not None:
            execution_lease.release()
        raise
    except Exception as exc:
        if execution_lease is not None:
            execution_lease.release()
        raise RunLifecycleError(f"run reservation failed: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RunManifestIdentity:
    run_path: Path
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
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
            if name != "run_path"
        }

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_path, Path)
            or self.run_role not in ("tuning", "confirmation")
            or self.disposition not in ("success", "failure")
        ):
            raise ValueError("run manifest identity fields are invalid")
        _run_component(self.run_id)
        if (
            (self.run_role == "tuning" and type(self.tuning_attempt_key) is not str)
            or (
                self.run_role == "confirmation"
                and self.tuning_attempt_key is not None
            )
        ):
            raise ValueError(
                "manifest tuning attempt key presence differs from its role"
            )
        for name in (
            "experiment_plan_sha256",
            "manifest_sha256",
            "manifest_payload_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.manifest_size_bytes) is not int or self.manifest_size_bytes <= 0:
            raise ValueError("manifest_size_bytes must be positive")
        for name in (
            "checkpoint_identity_sha256s",
            "checkpoint_artifact_record_sha256s",
            "artifact_record_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise ValueError(f"{name} must be an immutable tuple")
            for value in values:
                _sha256(value, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        if self.resume_lineage_sha256 is not None:
            _sha256(
                self.resume_lineage_sha256,
                "resume_lineage_sha256",
            )
        expected = owned_sha256(
            "vfe4.wt103.run-manifest-identity.v1",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("run manifest identity hash does not match")


def _verify_artifact_record(
    run_path: Path,
    record: ArtifactIntegrityRecord,
) -> None:
    record.__post_init__()
    relative = Path(*PurePosixPath(record.relative_path).parts)
    target = run_path / relative
    payload = _read_regular_bytes(target, maximum_bytes=max(record.size_bytes, 1))
    if (
        len(payload) != record.size_bytes
        or hashlib.sha256(payload).hexdigest() != record.sha256
    ):
        raise RunLifecycleError(
            f"artifact integrity mismatch: {record.relative_path}"
        )


def _validate_checkpoint_inventory(
    checkpoints: tuple[WT103CheckpointIdentity, ...],
    *,
    run_role: Literal["tuning", "confirmation"],
    disposition: Literal["success", "failure"],
    plan: ExperimentPlan,
) -> None:
    if (
        type(checkpoints) is not tuple
        or any(
            type(item) is not WT103CheckpointIdentity
            for item in checkpoints
        )
    ):
        raise RunLifecycleError("checkpoint inventory is not exact")
    for checkpoint in checkpoints:
        checkpoint.__post_init__()
    logical_keys = tuple(item.logical_key for item in checkpoints)
    identities = tuple(
        item.checkpoint_identity_sha256 for item in checkpoints
    )
    if (
        len(set(logical_keys)) != len(logical_keys)
        or len(set(identities)) != len(identities)
    ):
        raise RunLifecycleError(
            "checkpoint logical/identity keys must be unique"
        )
    terminal = tuple(
        item
        for item in checkpoints
        if item.checkpoint_role == "terminal_scoring"
    )
    if run_role == "tuning" and terminal:
        raise RunLifecycleError(
            "tuning run cannot publish a terminal_scoring checkpoint"
        )
    if disposition == "failure" and terminal:
        raise RunLifecycleError(
            "failed run cannot publish a terminal_scoring checkpoint"
        )
    if disposition == "success" and run_role == "confirmation":
        if len(terminal) != 1:
            raise RunLifecycleError(
                "successful confirmation requires exactly one "
                "terminal_scoring checkpoint"
            )
        if terminal[0].logical_key not in plan.terminal_checkpoint_keys:
            raise RunLifecycleError(
                "confirmation terminal checkpoint is outside the plan"
            )


def _validate_checkpoint_artifact_inventory(
    run_path: Path,
    *,
    checkpoints: tuple[WT103CheckpointIdentity, ...],
    records: tuple[ArtifactIntegrityRecord, ...],
    planned_artifact_records: tuple[ArtifactIntegrityRecord, ...],
) -> None:
    if (
        type(records) is not tuple
        or any(type(item) is not ArtifactIntegrityRecord for item in records)
        or len(records) != len(checkpoints)
    ):
        raise RunLifecycleError(
            "checkpoint artifact inventory must bind every checkpoint exactly"
        )
    paths = tuple(item.relative_path for item in records)
    planned_paths = tuple(
        item.relative_path for item in planned_artifact_records
    )
    aliases = tuple(path.casefold() for path in paths)
    if (
        len(set(aliases)) != len(aliases)
        or set(aliases).intersection(
            path.casefold() for path in planned_paths
        )
    ):
        raise RunLifecycleError(
            "checkpoint artifact inventory paths must be distinct"
        )
    reserved_names = {
        "experiment-plan.json",
        "failures.jsonl",
        "reservation.json",
        "resume-execution-started.json",
        "resume-lineage-intent.json",
        "resume-lineage.jsonl",
        "resume-owner.json",
        "run-manifest.json",
    }
    for checkpoint, record in zip(checkpoints, records, strict=True):
        parts = PurePosixPath(record.relative_path).parts
        if (
            record.kind != "file"
            or record.sha256 != checkpoint.checkpoint_payload_sha256
            or record.size_bytes != checkpoint.size_bytes
            or any(
                part.casefold()
                in {"resume-leases", "resume-owner-takeovers"}
                for part in parts
            )
            or PurePosixPath(record.relative_path).name.casefold()
            in reserved_names
        ):
            raise RunLifecycleError(
                "checkpoint artifact inventory differs from checkpoint bytes"
            )
        _verify_artifact_record(run_path, record)


def _validate_failure_artifact(
    run_path: Path,
    *,
    run_id: str,
    records: tuple[ArtifactIntegrityRecord, ...],
    failure_record_sha256: str,
) -> None:
    failure_records = tuple(
        item for item in records if item.relative_path == "failures.jsonl"
    )
    if len(failure_records) != 1:
        raise RunLifecycleError(
            "failed run requires exactly one verified failures.jsonl artifact"
        )
    try:
        ledger = validate_failure_log(run_path / "failures.jsonl")
    except (FailureLogError, OSError) as exc:
        raise RunLifecycleError("failure ledger is invalid") from exc
    if (
        not ledger
        or ledger[-1].record_sha256 != failure_record_sha256
        or ledger[-1].run_id != run_id
        or ledger[-1].terminal is not True
    ):
        raise RunLifecycleError(
            "failure ledger head does not match the terminal manifest"
        )


def _validate_artifact_inventory(
    run_path: Path,
    *,
    run_id: str,
    disposition: Literal["success", "failure"],
    records: tuple[ArtifactIntegrityRecord, ...],
    plan: ExperimentPlan,
    failure_record_sha256: str | None,
) -> None:
    if (
        type(records) is not tuple
        or any(
            type(item) is not ArtifactIntegrityRecord
            for item in records
        )
    ):
        raise RunLifecycleError("artifact record inventory is not exact")
    paths = tuple(item.relative_path for item in records)
    if (
        paths != tuple(sorted(paths))
        or len(set(path.casefold() for path in paths)) != len(paths)
    ):
        raise RunLifecycleError(
            "artifact record paths must be sorted and unique"
        )
    for record in records:
        _verify_artifact_record(run_path, record)
    expected = plan.expected_run_artifact_paths
    if disposition == "success":
        if paths != expected:
            raise RunLifecycleError(
                "successful run artifact inventory differs from plan"
            )
        if failure_record_sha256 is not None:
            raise RunLifecycleError(
                "successful run cannot claim a failure record"
            )
        return
    _sha256(failure_record_sha256, "failure_record_sha256")
    allowed = set(expected) | {"failures.jsonl"}
    if not set(paths).issubset(allowed):
        raise RunLifecycleError(
            "failed run contains an unplanned artifact"
        )
    _validate_failure_artifact(
        run_path,
        run_id=run_id,
        records=records,
        failure_record_sha256=failure_record_sha256,
    )


def _manifest_body(
    reserved: ReservedRun,
    *,
    disposition: Literal["success", "failure"],
    checkpoints: tuple[WT103CheckpointIdentity, ...],
    checkpoint_artifact_records: tuple[ArtifactIntegrityRecord, ...],
    artifact_records: tuple[ArtifactIntegrityRecord, ...],
    environment_sha256: str,
    provenance_sha256: str,
    ended_utc: str,
    monotonic_duration_seconds: float,
    failure_record_sha256: str | None,
    resume_lineage_sha256: str | None,
) -> dict[str, object]:
    body = {
        "schema_version": "wt103-run-manifest-v1",
        "run_id": reserved.run_id,
        "run_role": reserved.run_role,
        "started_utc": reserved.started_utc,
        "disposition": disposition,
        "experiment_plan_sha256": reserved.experiment_plan_sha256,
        "reservation_sha256": reserved.reservation_sha256,
        "environment_sha256": environment_sha256,
        "provenance_sha256": provenance_sha256,
        "ended_utc": ended_utc,
        "monotonic_duration_seconds": monotonic_duration_seconds,
        "failure_record_sha256": failure_record_sha256,
        "resume_count": reserved.resume_count,
        "resume_lineage_sha256": resume_lineage_sha256,
        "checkpoints": checkpoints,
        "checkpoint_artifact_records": checkpoint_artifact_records,
        "artifact_records": artifact_records,
    }
    if reserved.tuning_attempt_key is not None:
        body["tuning_attempt_key"] = reserved.tuning_attempt_key
    return body


def finalize_run(
    reserved: ReservedRun,
    *,
    disposition: Literal["success", "failure"],
    checkpoints: tuple[WT103CheckpointIdentity, ...],
    checkpoint_artifact_records: tuple[ArtifactIntegrityRecord, ...] = (),
    artifact_records: tuple[ArtifactIntegrityRecord, ...],
    environment_sha256: str,
    provenance_sha256: str,
    ended_utc: str,
    monotonic_duration_seconds: float,
    failure_record_sha256: str | None,
    backend: DurabilityBackend,
) -> RunManifestIdentity:
    """Close one attempt by manifest publication then atomic no-replace move."""

    if type(reserved) is not ReservedRun:
        raise RunLifecycleError("reserved must be exact ReservedRun")
    reserved.__post_init__()
    if not reserved.execution_lease.active:
        raise RunLifecycleError(
            "run finalization requires the active attempt execution lease"
        )
    if disposition not in ("success", "failure"):
        raise RunLifecycleError("terminal disposition is invalid")
    try:
        plan = _load_experiment_plan(reserved.experiment_root)
        if plan.experiment_plan_sha256 != reserved.experiment_plan_sha256:
            raise RunLifecycleError(
                "reserved run no longer binds the experiment plan"
            )
        final_exists = reserved.final_path.exists()
        inprogress_exists = reserved.inprogress_path.exists()
        if final_exists and inprogress_exists:
            raise RunLifecycleError(
                "terminal and in-progress run paths both exist"
            )
        if final_exists:
            run_path = reserved.final_path
            already_terminal = True
        elif inprogress_exists:
            run_path = reserved.inprogress_path
            already_terminal = False
        else:
            raise RunLifecycleError("reserved run path is missing")
        _regular_directory(run_path)
        (
            reservation_sha,
            reservation_started,
            reservation_tuning_attempt_key,
        ) = _validate_reservation(
            run_path / "reservation.json",
            run_id=reserved.run_id,
            run_role=reserved.run_role,
            tuning_attempt_key=reserved.tuning_attempt_key,
            experiment_plan_sha256=reserved.experiment_plan_sha256,
        )
        if (
            reservation_sha != reserved.reservation_sha256
            or reservation_started != reserved.started_utc
            or reservation_tuning_attempt_key != reserved.tuning_attempt_key
        ):
            raise RunLifecycleError(
                "reserved run differs from its durable reservation"
            )
        _validate_checkpoint_inventory(
            checkpoints,
            run_role=reserved.run_role,
            disposition=disposition,
            plan=plan,
        )
        _validate_checkpoint_artifact_inventory(
            run_path,
            checkpoints=checkpoints,
            records=checkpoint_artifact_records,
            planned_artifact_records=artifact_records,
        )
        _validate_artifact_inventory(
            run_path,
            run_id=reserved.run_id,
            disposition=disposition,
            records=artifact_records,
            plan=plan,
            failure_record_sha256=failure_record_sha256,
        )
        _sha256(environment_sha256, "environment_sha256")
        _sha256(provenance_sha256, "provenance_sha256")
        _text(ended_utc, "ended_utc")
        if (
            type(monotonic_duration_seconds) is not float
            or not math.isfinite(monotonic_duration_seconds)
            or monotonic_duration_seconds < 0.0
        ):
            raise RunLifecycleError(
                "monotonic_duration_seconds must be finite and nonnegative"
            )
        lineage_path = run_path / "resume-lineage.jsonl"
        if lineage_path.exists():
            lineage_payload = _read_regular_bytes(lineage_path)
            records = _validate_lineage_ledger(lineage_payload)
            if len(records) != reserved.resume_count:
                raise RunLifecycleError("resume count differs from lineage ledger")
            _require_resume_lineage_intent(
                run_path,
                lineage_records=records,
            )
            _validate_resume_lease_inventory(
                run_path,
                run_id=reserved.run_id,
                experiment_plan_sha256=reserved.experiment_plan_sha256,
                reservation_sha256=reserved.reservation_sha256,
                records=records,
            )
            _require_resume_execution_started(
                run_path,
                run_id=reserved.run_id,
                experiment_plan_sha256=reserved.experiment_plan_sha256,
                reservation_sha256=reserved.reservation_sha256,
                lineage_records=records,
            )
            lineage_sha: str | None = hashlib.sha256(
                lineage_payload
            ).hexdigest()
            owner = _validate_resume_owner(
                run_path / "resume-owner.json",
                run_id=reserved.run_id,
                experiment_plan_sha256=reserved.experiment_plan_sha256,
                reservation_sha256=reserved.reservation_sha256,
            )
            if (
                reserved.resume_owner_lineage_sha256 is None
                or owner["resume_ordinal"] != reserved.resume_count
                or owner["lineage_sha256"]
                != reserved.resume_owner_lineage_sha256
                or owner["lineage_sha256"]
                != records[-1]["lineage_sha256"]
                or (
                    owner["state"] == "terminal_closed"
                    and not already_terminal
                    and not (run_path / "run-manifest.json").exists()
                )
            ):
                raise RunLifecycleError(
                    "reserved run is no longer the active resume owner"
                )
        else:
            if reserved.resume_count != 0:
                raise RunLifecycleError("resume lineage ledger is missing")
            if (run_path / "resume-lineage-intent.json").exists():
                raise RunLifecycleError(
                    "unresumed run contains a resume lineage intent"
                )
            if (run_path / "resume-execution-started.json").exists():
                raise RunLifecycleError(
                    "unresumed run contains a retry execution-started transition"
                )
            _validate_resume_lease_inventory(
                run_path,
                run_id=reserved.run_id,
                experiment_plan_sha256=reserved.experiment_plan_sha256,
                reservation_sha256=reserved.reservation_sha256,
                records=(),
            )
            lineage_sha = None
        body = _manifest_body(
            reserved,
            disposition=disposition,
            checkpoints=checkpoints,
            checkpoint_artifact_records=checkpoint_artifact_records,
            artifact_records=artifact_records,
            environment_sha256=environment_sha256,
            provenance_sha256=provenance_sha256,
            ended_utc=ended_utc,
            monotonic_duration_seconds=monotonic_duration_seconds,
            failure_record_sha256=failure_record_sha256,
            resume_lineage_sha256=lineage_sha,
        )
        manifest_sha = owned_sha256(
            "vfe4.wt103.run-manifest.v1",
            body,
        )
        payload = canonical_json_bytes_generic(
            {**body, "manifest_sha256": manifest_sha}
        )
        manifest_path = run_path / "run-manifest.json"
        if manifest_path.exists():
            if _read_regular_bytes(manifest_path) != payload:
                raise RunLifecycleError(
                    "existing run manifest differs from retry transition"
                )
        else:
            if already_terminal:
                raise RunLifecycleError(
                    "terminal run is missing its manifest"
                )
            try:
                backend.create_exclusive(manifest_path, payload)
            except Exception as exc:
                if (
                    not manifest_path.exists()
                    or _read_regular_bytes(manifest_path) != payload
                ):
                    raise RunLifecycleError(
                        f"run manifest publication failed: {exc}"
                    ) from exc
        if _read_regular_bytes(manifest_path) != payload:
            raise RunLifecycleError("run manifest reopen validation failed")
        _close_resume_owner(
            run_path,
            reserved=reserved,
            manifest_sha256=manifest_sha,
            backend=backend,
        )
        if not already_terminal:
            try:
                _rename_directory_no_replace(
                    reserved.inprogress_path,
                    reserved.final_path,
                )
            except Exception as exc:
                raise RunLifecycleError(
                    f"terminal run rename failed: {exc}"
                ) from exc
        identity = _identity_from_manifest_payload(
            reserved.final_path,
            payload,
        )
        validated = validate_run_manifest(
            reserved.final_path / "run-manifest.json",
            expected=identity,
        )
        release_run_execution_lease(reserved)
        return validated
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(f"run finalization failed: {exc}") from exc


def _checkpoint_from_json(value: object) -> WT103CheckpointIdentity:
    if type(value) is not dict or set(value) != {
        "artifact_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_manifest_body_sha256",
        "checkpoint_payload_sha256",
        "checkpoint_role",
        "logical_key",
        "schema_version",
        "scientific_state_sha256",
        "size_bytes",
    }:
        raise RunLifecycleError("checkpoint identity JSON is not closed")
    try:
        identity = WT103CheckpointIdentity(**value)  # type: ignore[arg-type]
        identity.__post_init__()
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError("checkpoint identity JSON is invalid") from exc
    return identity


def _artifact_record_from_json(value: object) -> ArtifactIntegrityRecord:
    if type(value) is not dict or set(value) != {
        "kind",
        "record_sha256",
        "relative_path",
        "schema_version",
        "sha256",
        "size_bytes",
    }:
        raise RunLifecycleError("artifact integrity JSON is not closed")
    try:
        record = ArtifactIntegrityRecord(**value)  # type: ignore[arg-type]
        record.__post_init__()
    except (TypeError, ValueError) as exc:
        raise RunLifecycleError("artifact integrity JSON is invalid") from exc
    return record


def _identity_from_manifest_payload(
    run_path: Path,
    payload: bytes,
    *,
    allow_inprogress: bool = False,
    allow_active_resume_owner_recovery: bool = False,
) -> RunManifestIdentity:
    if (
        type(allow_inprogress) is not bool
        or type(allow_active_resume_owner_recovery) is not bool
        or (
            allow_active_resume_owner_recovery
            and not allow_inprogress
        )
    ):
        raise RunLifecycleError(
            "run manifest recovery authority is invalid"
        )
    decoded = _json_object(payload, context="run manifest")
    expected_keys = {
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
    if decoded.get("run_role") == "tuning":
        expected_keys.add("tuning_attempt_key")
    if set(decoded) != expected_keys:
        raise RunLifecycleError("run manifest key set is open")
    if decoded["schema_version"] != "wt103-run-manifest-v1":
        raise RunLifecycleError("run manifest schema is invalid")
    try:
        run_id = _run_component(decoded["run_id"])
        run_role = decoded["run_role"]
        tuning_attempt_key = decoded.get("tuning_attempt_key")
        disposition = decoded["disposition"]
        if run_role not in ("tuning", "confirmation"):
            raise RunLifecycleError("run manifest role is invalid")
        if disposition not in ("success", "failure"):
            raise RunLifecycleError("run manifest disposition is invalid")
        if (
            run_role == "tuning"
            and type(tuning_attempt_key) is not str
        ) or (
            run_role == "confirmation"
            and tuning_attempt_key is not None
        ):
            raise RunLifecycleError(
                "run manifest tuning attempt key differs from its role"
            )
        experiment_plan_sha = _sha256(
            decoded["experiment_plan_sha256"],
            "experiment_plan_sha256",
        )
        reservation_sha = _sha256(
            decoded["reservation_sha256"],
            "reservation_sha256",
        )
        _sha256(decoded["environment_sha256"], "environment_sha256")
        _sha256(decoded["provenance_sha256"], "provenance_sha256")
        started_utc = _text(decoded["started_utc"], "started_utc")
        _text(decoded["ended_utc"], "ended_utc")
    except ValueError as exc:
        raise RunLifecycleError("run manifest scalar identity is invalid") from exc
    if allow_inprogress:
        path_is_exact = (
            run_path.name == run_id
            and run_path.parent.name == ".inprogress"
            and run_path.parent.parent.name == "runs"
        )
        experiment_root = run_path.parent.parent.parent
    else:
        path_is_exact = (
            run_path.name == run_id
            and run_path.parent.name == "runs"
        )
        experiment_root = run_path.parent.parent
    if not path_is_exact:
        raise RunLifecycleError("run manifest path/run ID is not exact")
    duration = decoded["monotonic_duration_seconds"]
    resume_count = decoded["resume_count"]
    if (
        type(duration) is not float
        or not math.isfinite(duration)
        or duration < 0.0
        or type(resume_count) is not int
        or resume_count < 0
    ):
        raise RunLifecycleError(
            "run manifest duration/resume count is invalid"
        )
    body = dict(decoded)
    manifest_sha = body.pop("manifest_sha256", None)
    try:
        _sha256(manifest_sha, "manifest_sha256")
    except ValueError as exc:
        raise RunLifecycleError("run manifest hash is invalid") from exc
    if (
        manifest_sha
        != owned_sha256(
            "vfe4.wt103.run-manifest.v1",
            body,
        )
    ):
        raise RunLifecycleError("run manifest hash does not match")
    plan = _load_experiment_plan(experiment_root)
    if plan.experiment_plan_sha256 != experiment_plan_sha:
        raise RunLifecycleError("run manifest does not bind its plan")
    if (
        tuning_attempt_key is not None
        and tuning_attempt_key not in plan.tuning_attempt_keys
    ):
        raise RunLifecycleError(
            "run manifest tuning attempt key is outside its plan"
        )
    (
        observed_reservation_sha,
        reservation_started,
        reservation_tuning_attempt_key,
    ) = _validate_reservation(
        run_path / "reservation.json",
        run_id=run_id,
        run_role=run_role,
        tuning_attempt_key=tuning_attempt_key,
        experiment_plan_sha256=experiment_plan_sha,
    )
    if (
        observed_reservation_sha != reservation_sha
        or reservation_started != started_utc
        or reservation_tuning_attempt_key != tuning_attempt_key
    ):
        raise RunLifecycleError(
            "run manifest does not bind its durable reservation"
        )
    checkpoints_value = decoded["checkpoints"]
    checkpoint_artifacts_value = decoded["checkpoint_artifact_records"]
    artifacts_value = decoded["artifact_records"]
    if (
        type(checkpoints_value) is not list
        or type(checkpoint_artifacts_value) is not list
        or type(artifacts_value) is not list
    ):
        raise RunLifecycleError("run manifest inventories must be JSON arrays")
    checkpoints = tuple(
        _checkpoint_from_json(item) for item in checkpoints_value
    )
    checkpoint_artifacts = tuple(
        _artifact_record_from_json(item)
        for item in checkpoint_artifacts_value
    )
    artifacts = tuple(
        _artifact_record_from_json(item) for item in artifacts_value
    )
    _validate_checkpoint_inventory(
        checkpoints,
        run_role=run_role,
        disposition=disposition,
        plan=plan,
    )
    _validate_checkpoint_artifact_inventory(
        run_path,
        checkpoints=checkpoints,
        records=checkpoint_artifacts,
        planned_artifact_records=artifacts,
    )
    failure_head = decoded["failure_record_sha256"]
    _validate_artifact_inventory(
        run_path,
        run_id=run_id,
        disposition=disposition,
        records=artifacts,
        plan=plan,
        failure_record_sha256=failure_head,
    )
    lineage_sha = decoded["resume_lineage_sha256"]
    if lineage_sha is not None:
        try:
            _sha256(lineage_sha, "resume_lineage_sha256")
        except ValueError as exc:
            raise RunLifecycleError(
                "resume lineage digest is invalid"
            ) from exc
        if resume_count == 0:
            raise RunLifecycleError(
                "zero resume count cannot claim a lineage digest"
            )
        lineage_path = run_path / "resume-lineage.jsonl"
        lineage_payload = _read_regular_bytes(lineage_path)
        lineage_records = _validate_lineage_ledger(lineage_payload)
        if hashlib.sha256(lineage_payload).hexdigest() != lineage_sha:
            raise RunLifecycleError("resume lineage digest does not match")
        if len(lineage_records) != resume_count:
            raise RunLifecycleError(
                "resume count differs from lineage ledger"
            )
        _require_resume_lineage_intent(
            run_path,
            lineage_records=lineage_records,
        )
        _validate_resume_lease_inventory(
            run_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha,
            reservation_sha256=reservation_sha,
            records=lineage_records,
        )
        _require_resume_execution_started(
            run_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha,
            reservation_sha256=reservation_sha,
            lineage_records=lineage_records,
        )
        _validate_terminal_resume_owner(
            run_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha,
            reservation_sha256=reservation_sha,
            resume_count=resume_count,
            lineage_records=lineage_records,
            manifest_sha256=str(manifest_sha),
            allow_active_recovery=(
                allow_active_resume_owner_recovery
            ),
        )
    else:
        if resume_count != 0 or (run_path / "resume-lineage.jsonl").exists():
            raise RunLifecycleError(
                "resume count/lineage presence does not match"
            )
        if (run_path / "resume-lineage-intent.json").exists():
            raise RunLifecycleError(
                "unresumed run contains a resume lineage intent"
            )
        if (run_path / "resume-execution-started.json").exists():
            raise RunLifecycleError(
                "unresumed run contains a retry execution-started transition"
            )
        _validate_resume_lease_inventory(
            run_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha,
            reservation_sha256=reservation_sha,
            records=(),
        )
        _validate_terminal_resume_owner(
            run_path,
            run_id=run_id,
            experiment_plan_sha256=experiment_plan_sha,
            reservation_sha256=reservation_sha,
            resume_count=resume_count,
            lineage_records=(),
            manifest_sha256=str(manifest_sha),
        )
    semantic = {
        "run_id": run_id,
        "run_role": run_role,
        "tuning_attempt_key": tuning_attempt_key,
        "disposition": disposition,
        "experiment_plan_sha256": experiment_plan_sha,
        "manifest_sha256": manifest_sha,
        "manifest_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_size_bytes": len(payload),
        "checkpoint_identity_sha256s": tuple(
            item.checkpoint_identity_sha256 for item in checkpoints
        ),
        "checkpoint_artifact_record_sha256s": tuple(
            item.record_sha256 for item in checkpoint_artifacts
        ),
        "artifact_record_sha256s": tuple(
            item.record_sha256 for item in artifacts
        ),
        "resume_lineage_sha256": lineage_sha,
    }
    return RunManifestIdentity(
        run_path=run_path,
        **semantic,
        identity_sha256=owned_sha256(
            "vfe4.wt103.run-manifest-identity.v1",
            semantic,
        ),
    )  # type: ignore[arg-type]


def validate_run_manifest(
    manifest_path: Path,
    *,
    expected: RunManifestIdentity | None = None,
) -> RunManifestIdentity:
    """Validate one explicit terminal run path; never select newest/glob."""

    if not isinstance(manifest_path, Path):
        raise RunLifecycleError("manifest_path must be pathlib.Path")
    if manifest_path.name != "run-manifest.json":
        raise RunLifecycleError("manifest path must name run-manifest.json")
    run_path = manifest_path.parent
    _regular_directory(run_path)
    payload = _read_regular_bytes(manifest_path)
    try:
        identity = _identity_from_manifest_payload(run_path, payload)
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(
            f"run manifest validation failed: {exc}"
        ) from exc
    if expected is not None:
        if type(expected) is not RunManifestIdentity:
            raise RunLifecycleError("expected identity must be exact")
        expected.__post_init__()
        if identity != expected:
            raise RunLifecycleError("run manifest differs from expected identity")
    return identity


def _repair_manifested_active_resume_owner(
    run_path: Path,
    manifest_payload: bytes,
    *,
    backend: DurabilityBackend,
) -> None:
    """Close only the active owner exactly bound to a validated manifest."""

    decoded = _json_object(
        manifest_payload,
        context="run manifest recovery transition",
    )
    resume_count = decoded.get("resume_count")
    if resume_count == 0:
        return
    if type(resume_count) is not int or resume_count < 1:
        raise RunLifecycleError(
            "run manifest recovery resume count is invalid"
        )
    try:
        run_id = _run_component(decoded["run_id"])
        experiment_plan_sha256 = _sha256(
            decoded["experiment_plan_sha256"],
            "experiment_plan_sha256",
        )
        reservation_sha256 = _sha256(
            decoded["reservation_sha256"],
            "reservation_sha256",
        )
        manifest_sha256 = _sha256(
            decoded["manifest_sha256"],
            "manifest_sha256",
        )
    except (KeyError, ValueError) as exc:
        raise RunLifecycleError(
            "run manifest recovery identity is invalid"
        ) from exc
    lineage_payload = _read_regular_bytes(
        run_path / "resume-lineage.jsonl"
    )
    lineage_records = _validate_lineage_ledger(lineage_payload)
    owner_state = _validate_terminal_resume_owner(
        run_path,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_count=resume_count,
        lineage_records=lineage_records,
        manifest_sha256=manifest_sha256,
        allow_active_recovery=True,
    )
    if owner_state == "terminal_closed":
        return
    if owner_state != "active":
        raise RunLifecycleError(
            "run manifest recovery owner state is invalid"
        )
    owner_path = run_path / "resume-owner.json"
    current_payload = _read_regular_bytes(owner_path)
    current = _validate_resume_owner(
        owner_path,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
    )
    expected_active = _resume_owner_payload(
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=resume_count,
        previous_owner_lineage_sha256=(
            _ZERO_SHA256
            if resume_count == 1
            else str(lineage_records[-2]["lineage_sha256"])
        ),
        lineage_sha256=str(lineage_records[-1]["lineage_sha256"]),
        state="active",
        terminal_manifest_sha256=None,
    )
    if current != expected_active:
        raise RunLifecycleError(
            "run manifest recovery owner is not the exact active lineage"
        )
    terminal = _resume_owner_payload(
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_ordinal=resume_count,
        previous_owner_lineage_sha256=str(
            current["previous_owner_lineage_sha256"]
        ),
        lineage_sha256=str(current["lineage_sha256"]),
        state="terminal_closed",
        terminal_manifest_sha256=manifest_sha256,
    )
    terminal_payload = canonical_json_bytes_generic(terminal)
    if (
        _read_regular_bytes(run_path / "run-manifest.json")
        != manifest_payload
        or _read_regular_bytes(owner_path) != current_payload
    ):
        raise RunLifecycleError(
            "run manifest recovery transition changed before commit"
        )
    try:
        backend.replace_durable(owner_path, terminal_payload)
    except Exception as exc:
        if (
            not owner_path.exists()
            or _read_regular_bytes(owner_path) != terminal_payload
        ):
            raise RunLifecycleError(
                f"run manifest recovery owner closure failed: {exc}"
            ) from exc
    if _read_regular_bytes(owner_path) != terminal_payload:
        raise RunLifecycleError(
            "run manifest recovery owner closure did not commit"
        )
    _validate_terminal_resume_owner(
        run_path,
        run_id=run_id,
        experiment_plan_sha256=experiment_plan_sha256,
        reservation_sha256=reservation_sha256,
        resume_count=resume_count,
        lineage_records=lineage_records,
        manifest_sha256=manifest_sha256,
    )


def recover_terminal_run(
    experiment_root: Path,
    run_id: str,
    *,
    plan: ExperimentPlanIdentity,
    backend: DurabilityBackend | None = None,
) -> RunManifestIdentity:
    """Complete only a durably manifested terminal rename after process loss."""

    execution_lease: AttemptExecutionLease | None = None
    try:
        _validate_plan_identity(experiment_root, plan)
        safe_id = _run_component(run_id)
        inprogress = experiment_root / "runs" / ".inprogress" / safe_id
        final = experiment_root / "runs" / safe_id
        inprogress_exists = inprogress.exists()
        final_exists = final.exists()
        if inprogress_exists and final_exists:
            raise RunLifecycleError(
                "terminal and in-progress run paths both exist"
            )
        if final_exists:
            identity = validate_run_manifest(final / "run-manifest.json")
            if (
                identity.experiment_plan_sha256
                != plan.plan.experiment_plan_sha256
            ):
                raise RunLifecycleError(
                    "terminal run differs from the recovery plan"
                )
            return identity
        if not inprogress_exists:
            raise RunLifecycleError(
                "terminal recovery has no retained run directory"
            )
        execution_lease = _acquire_attempt_execution_lease(
            experiment_root,
            safe_id,
        )
        _regular_directory(inprogress)
        manifest_path = inprogress / "run-manifest.json"
        payload = _read_regular_bytes(manifest_path)
        candidate = _identity_from_manifest_payload(
            inprogress,
            payload,
            allow_inprogress=True,
            allow_active_resume_owner_recovery=True,
        )
        if (
            candidate.experiment_plan_sha256
            != plan.plan.experiment_plan_sha256
        ):
            raise RunLifecycleError(
                "terminal recovery candidate differs from its plan"
            )
        recovery_backend = backend
        if recovery_backend is None:
            recovery_backend = (
                WindowsDurabilityBackend()
                if os.name == "nt"
                else PosixDurabilityBackend()
            )
        _repair_manifested_active_resume_owner(
            inprogress,
            payload,
            backend=recovery_backend,
        )
        strict_candidate = _identity_from_manifest_payload(
            inprogress,
            payload,
            allow_inprogress=True,
        )
        if (
            strict_candidate.manifest_payload_sha256
            != candidate.manifest_payload_sha256
            or strict_candidate.identity_sha256
            != candidate.identity_sha256
        ):
            raise RunLifecycleError(
                "terminal owner repair changed the manifest identity"
            )
        if _read_regular_bytes(manifest_path) != payload:
            raise RunLifecycleError(
                "terminal recovery manifest changed before rename"
            )
        try:
            _rename_directory_no_replace(inprogress, final)
        except Exception as exc:
            raise RunLifecycleError(
                f"terminal recovery rename failed: {exc}"
            ) from exc
        recovered = validate_run_manifest(final / "run-manifest.json")
        if (
            recovered.manifest_payload_sha256
            != candidate.manifest_payload_sha256
            or recovered.identity_sha256 != candidate.identity_sha256
        ):
            raise RunLifecycleError(
                "terminal recovery changed the validated manifest identity"
            )
        return recovered
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(
            f"terminal run recovery failed: {exc}"
        ) from exc
    finally:
        if execution_lease is not None:
            execution_lease.release()


@dataclass(frozen=True, slots=True)
class ExperimentIndexIdentity:
    index_path: Path
    stage: Literal["pretest", "final"]
    experiment_plan_sha256: str
    run_manifest_sha256s: tuple[str, ...]
    artifact_record_sha256s: tuple[str, ...]
    payload_sha256: str
    size_bytes: int
    identity_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
            if name != "index_path"
        }

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index_path, Path)
            or self.index_path.name != "experiment-index.json"
            or self.stage not in ("pretest", "final")
            or type(self.run_manifest_sha256s) is not tuple
            or type(self.artifact_record_sha256s) is not tuple
        ):
            raise ValueError("experiment index identity is invalid")
        _sha256(
            self.experiment_plan_sha256,
            "experiment_plan_sha256",
        )
        for value in self.run_manifest_sha256s:
            _sha256(value, "run_manifest_sha256")
        if len(set(self.run_manifest_sha256s)) != len(
            self.run_manifest_sha256s
        ):
            raise ValueError("run manifest identities must be unique")
        for value in self.artifact_record_sha256s:
            _sha256(value, "artifact_record_sha256")
        if len(set(self.artifact_record_sha256s)) != len(
            self.artifact_record_sha256s
        ):
            raise ValueError("group artifact identities must be unique")
        _sha256(self.payload_sha256, "payload_sha256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("index size_bytes must be positive")
        expected = owned_sha256(
            "vfe4.wt103.experiment-index-identity.v1",
            self.semantic_payload(),
        )
        _sha256(self.identity_sha256, "identity_sha256")
        if self.identity_sha256 != expected:
            raise ValueError("experiment index identity hash does not match")


def _confirmation_terminal_key(
    manifest_path: Path,
) -> str | None:
    document = _json_object(
        _read_regular_bytes(manifest_path),
        context="run manifest",
    )
    if (
        document.get("run_role") != "confirmation"
        or document.get("disposition") != "success"
    ):
        return None
    values = document.get("checkpoints")
    if type(values) is not list:
        raise RunLifecycleError(
            "confirmation checkpoint inventory is not a JSON array"
        )
    terminal = tuple(
        _checkpoint_from_json(value)
        for value in values
        if type(value) is dict
        and value.get("checkpoint_role") == "terminal_scoring"
    )
    if len(terminal) != 1:
        raise RunLifecycleError(
            "successful confirmation lacks one terminal checkpoint"
        )
    return terminal[0].logical_key


def _validate_final_confirmation_keys(
    observed: tuple[str, ...],
    *,
    plan: ExperimentPlan,
) -> None:
    if (
        len(observed) != 40
        or len(set(observed)) != 40
        or observed != plan.terminal_checkpoint_keys
    ):
        raise RunLifecycleError(
            "final confirmation terminal checkpoint inventory must be "
            "the exact ordered 40 plan keys"
        )


def _validate_final_run_inventory(
    *,
    run_roles: tuple[str, ...],
    tuning_attempt_keys: tuple[str, ...],
    confirmation_terminal_keys: tuple[str, ...],
    plan: ExperimentPlan,
) -> None:
    tuning_count = len(plan.tuning_attempt_keys)
    if (
        run_roles[:tuning_count] != ("tuning",) * tuning_count
        or tuning_attempt_keys != plan.tuning_attempt_keys
        or len(set(tuning_attempt_keys)) != len(tuning_attempt_keys)
    ):
        raise RunLifecycleError(
            "final tuning attempt inventory must be the exact ordered plan keys"
        )
    if run_roles[tuning_count:] != (
        ("confirmation",) * len(plan.terminal_checkpoint_keys)
    ):
        raise RunLifecycleError(
            "final confirmation terminal checkpoint inventory must be "
            "the exact ordered 40 plan keys"
        )
    _validate_final_confirmation_keys(
        confirmation_terminal_keys,
        plan=plan,
    )


def validate_experiment_index(
    index_path: Path,
    *,
    expected: ExperimentIndexIdentity | None = None,
) -> ExperimentIndexIdentity:
    """Validate an exact index and every explicitly referenced terminal run."""

    if not isinstance(index_path, Path):
        raise RunLifecycleError("index_path must be pathlib.Path")
    if index_path.name != "experiment-index.json":
        raise RunLifecycleError(
            "index path must name experiment-index.json"
        )
    try:
        experiment_root = index_path.parent
        _regular_directory(experiment_root)
        payload = _read_regular_bytes(index_path)
        decoded = _json_object(payload, context="experiment index")
        if set(decoded) != {
            "artifact_records",
            "experiment_plan_sha256",
            "index_sha256",
            "runs",
            "schema_version",
            "stage",
        }:
            raise RunLifecycleError("experiment index key set is open")
        if decoded["schema_version"] != "wt103-experiment-index-v1":
            raise RunLifecycleError("experiment index schema is invalid")
        stage = decoded["stage"]
        if stage not in ("pretest", "final"):
            raise RunLifecycleError("experiment index stage is invalid")
        body = dict(decoded)
        index_sha = body.pop("index_sha256", None)
        _sha256(index_sha, "index_sha256")
        if index_sha != owned_sha256(
            "vfe4.wt103.experiment-index.v1",
            body,
        ):
            raise RunLifecycleError("experiment index hash does not match")
        experiment_plan_sha = decoded["experiment_plan_sha256"]
        _sha256(
            experiment_plan_sha,
            "experiment_plan_sha256",
        )

        plan = _load_experiment_plan(experiment_root)
        if plan.experiment_plan_sha256 != experiment_plan_sha:
            raise RunLifecycleError(
                "experiment index does not bind the published plan"
            )
        artifact_values = decoded["artifact_records"]
        if type(artifact_values) is not list:
            raise RunLifecycleError(
                "experiment index artifact records must be a JSON array"
            )
        artifacts = tuple(
            _artifact_record_from_json(item)
            for item in artifact_values
        )
        artifact_paths = tuple(item.relative_path for item in artifacts)
        if (
            artifact_paths != tuple(sorted(artifact_paths))
            or len(set(path.casefold() for path in artifact_paths))
            != len(artifact_paths)
        ):
            raise RunLifecycleError(
                "experiment index artifact paths must be sorted and unique"
            )
        expected_group_paths = plan.expected_group_artifact_paths
        if stage == "pretest":
            if artifacts:
                raise RunLifecycleError(
                    "pretest index cannot contain result artifacts"
                )
        elif artifact_paths != expected_group_paths:
            raise RunLifecycleError(
                "final index group artifacts differ from plan"
            )
        for record in artifacts:
            _verify_artifact_record(experiment_root, record)

        entries = decoded["runs"]
        if type(entries) is not list or not entries:
            raise RunLifecycleError(
                "experiment index runs must be a nonempty JSON array"
            )
        run_ids: list[str] = []
        manifest_shas: list[str] = []
        manifest_identity_shas: list[str] = []
        relative_paths: list[str] = []
        run_roles: list[str] = []
        tuning_attempt_keys: list[str] = []
        confirmation_terminal_keys: list[str] = []
        for entry in entries:
            if type(entry) is not dict or set(entry) != {
                "disposition",
                "manifest_identity_sha256",
                "manifest_sha256",
                "relative_manifest_path",
                "run_id",
                "run_role",
            }:
                raise RunLifecycleError(
                    "experiment index run entry is not closed"
                )
            run_id = _run_component(entry["run_id"])
            disposition = entry["disposition"]
            run_role = entry["run_role"]
            if run_role not in ("tuning", "confirmation"):
                raise RunLifecycleError(
                    "experiment index run role is invalid"
                )
            if disposition not in ("success", "failure"):
                raise RunLifecycleError(
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
            relative_path = entry["relative_manifest_path"]
            expected_relative = f"runs/{run_id}/run-manifest.json"
            if relative_path != expected_relative:
                raise RunLifecycleError(
                    "experiment index manifest path is not exact"
                )
            manifest = validate_run_manifest(
                experiment_root
                / Path(*PurePosixPath(expected_relative).parts)
            )
            if (
                manifest.run_id != run_id
                or manifest.run_role != run_role
                or manifest.disposition != disposition
                or manifest.manifest_sha256 != manifest_sha
                or manifest.identity_sha256 != manifest_identity_sha
                or manifest.experiment_plan_sha256
                != experiment_plan_sha
            ):
                raise RunLifecycleError(
                    "experiment index entry differs from terminal manifest"
                )
            run_ids.append(run_id)
            manifest_shas.append(manifest_sha)
            manifest_identity_shas.append(manifest_identity_sha)
            relative_paths.append(relative_path)
            run_roles.append(run_role)
            if run_role == "tuning":
                if type(manifest.tuning_attempt_key) is not str:
                    raise RunLifecycleError(
                        "tuning index entry lacks its exact attempt key"
                    )
                tuning_attempt_keys.append(manifest.tuning_attempt_key)
            terminal_key = _confirmation_terminal_key(
                experiment_root
                / Path(*PurePosixPath(expected_relative).parts)
            )
            if terminal_key is not None:
                confirmation_terminal_keys.append(terminal_key)
        for values, name in (
            (run_ids, "run IDs"),
            (manifest_shas, "manifest hashes"),
            (manifest_identity_shas, "manifest identity hashes"),
            (relative_paths, "manifest paths"),
        ):
            if len(set(values)) != len(values):
                raise RunLifecycleError(
                    f"experiment index {name} must be unique"
                )
        if stage == "final":
            _validate_final_run_inventory(
                run_roles=tuple(run_roles),
                tuning_attempt_keys=tuple(tuning_attempt_keys),
                confirmation_terminal_keys=tuple(
                    confirmation_terminal_keys
                ),
                plan=plan,
            )

        semantic = {
            "stage": stage,
            "experiment_plan_sha256": experiment_plan_sha,
            "run_manifest_sha256s": tuple(manifest_shas),
            "artifact_record_sha256s": tuple(
                item.record_sha256 for item in artifacts
            ),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        identity = ExperimentIndexIdentity(
            index_path=index_path,
            **semantic,
            identity_sha256=owned_sha256(
                "vfe4.wt103.experiment-index-identity.v1",
                semantic,
            ),
        )
        if expected is not None:
            if type(expected) is not ExperimentIndexIdentity:
                raise RunLifecycleError(
                    "expected index identity must be exact"
                )
            expected.__post_init__()
            if identity != expected:
                raise RunLifecycleError(
                    "experiment index differs from expected identity"
                )
        return identity
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(
            f"experiment index validation failed: {exc}"
        ) from exc


def publish_experiment_index(
    experiment_root: Path,
    *,
    plan: ExperimentPlanIdentity,
    run_manifests: tuple[RunManifestIdentity, ...],
    stage: Literal["pretest", "final"],
    artifact_records: tuple[ArtifactIntegrityRecord, ...],
    backend: DurabilityBackend,
) -> ExperimentIndexIdentity:
    """Publish the exact terminal manifest index after run closure."""

    try:
        _validate_plan_identity(experiment_root, plan)
        if (
            type(run_manifests) is not tuple
            or not run_manifests
            or any(
                type(item) is not RunManifestIdentity
                for item in run_manifests
            )
        ):
            raise RunLifecycleError("run_manifests must be a nonempty exact tuple")
        run_ids = tuple(item.run_id for item in run_manifests)
        if len(set(run_ids)) != len(run_ids):
            raise RunLifecycleError("experiment index run IDs must be unique")
        if stage not in ("pretest", "final"):
            raise RunLifecycleError("experiment index stage is invalid")
        if (
            type(artifact_records) is not tuple
            or any(
                type(item) is not ArtifactIntegrityRecord
                for item in artifact_records
            )
        ):
            raise RunLifecycleError(
                "group artifact record inventory is not exact"
            )
        artifact_paths = tuple(
            item.relative_path for item in artifact_records
        )
        if (
            artifact_paths != tuple(sorted(artifact_paths))
            or len(set(path.casefold() for path in artifact_paths))
            != len(artifact_paths)
        ):
            raise RunLifecycleError(
                "group artifact paths must be sorted and unique"
            )
        expected_group_paths = plan.plan.expected_group_artifact_paths
        if stage == "pretest":
            if artifact_records:
                raise RunLifecycleError(
                    "pretest index cannot publish result artifacts"
                )
        elif artifact_paths != expected_group_paths:
            raise RunLifecycleError(
                "final group artifacts differ from experiment plan"
            )
        for record in artifact_records:
            _verify_artifact_record(experiment_root, record)
        entries = []
        run_roles: list[str] = []
        tuning_attempt_keys: list[str] = []
        confirmation_terminal_keys: list[str] = []
        for manifest in run_manifests:
            manifest.__post_init__()
            if (
                manifest.experiment_plan_sha256
                != plan.plan.experiment_plan_sha256
                or manifest.run_path
                != experiment_root / "runs" / manifest.run_id
            ):
                raise RunLifecycleError(
                    "run manifest is outside the exact experiment plan/root"
                )
            manifest_path = manifest.run_path / "run-manifest.json"
            validate_run_manifest(
                manifest_path,
                expected=manifest,
            )
            terminal_key = _confirmation_terminal_key(manifest_path)
            if terminal_key is not None:
                confirmation_terminal_keys.append(terminal_key)
            run_roles.append(manifest.run_role)
            if manifest.run_role == "tuning":
                if manifest.tuning_attempt_key is None:
                    raise RunLifecycleError(
                        "tuning manifest lacks its exact attempt key"
                    )
                tuning_attempt_keys.append(manifest.tuning_attempt_key)
            entries.append(
                {
                    "run_id": manifest.run_id,
                    "run_role": manifest.run_role,
                    "disposition": manifest.disposition,
                    "manifest_sha256": manifest.manifest_sha256,
                    "manifest_identity_sha256": manifest.identity_sha256,
                    "relative_manifest_path": (
                        f"runs/{manifest.run_id}/run-manifest.json"
                    ),
                }
            )
        if stage == "final":
            _validate_final_run_inventory(
                run_roles=tuple(run_roles),
                tuning_attempt_keys=tuple(tuning_attempt_keys),
                confirmation_terminal_keys=tuple(
                    confirmation_terminal_keys
                ),
                plan=plan.plan,
            )
        body = {
            "schema_version": "wt103-experiment-index-v1",
            "stage": stage,
            "experiment_plan_sha256": plan.plan.experiment_plan_sha256,
            "runs": entries,
            "artifact_records": artifact_records,
        }
        index_sha = owned_sha256(
            "vfe4.wt103.experiment-index.v1",
            body,
        )
        payload = canonical_json_bytes_generic(
            {**body, "index_sha256": index_sha}
        )
        path = experiment_root / "experiment-index.json"
        previous_payload: bytes | None = None
        if path.exists():
            existing_payload = _read_regular_bytes(path)
            previous_payload = existing_payload
            previous = _json_object(
                existing_payload,
                context="experiment index",
            )
            old_runs = previous.get("runs")
            old_artifacts = previous.get("artifact_records")
            old_stage = previous.get("stage")
            if (
                type(old_runs) is not list
                or entries[: len(old_runs)] != old_runs
                or type(old_artifacts) is not list
                or (old_stage == "pretest" and old_artifacts)
                or old_stage not in ("pretest", "final")
                or (old_stage == "final" and existing_payload != payload)
                or (old_stage == "final" and stage != "final")
            ):
                raise RunLifecycleError(
                    "experiment index update would rewrite prior entries"
                )
        if previous_payload != payload:
            transition_previous_sha256 = (
                _ZERO_SHA256
                if previous_payload is None
                else hashlib.sha256(previous_payload).hexdigest()
            )
            transition_body = {
                "schema_version": "wt103-index-transition-v1",
                "experiment_plan_sha256": (
                    plan.plan.experiment_plan_sha256
                ),
                "previous_index_payload_sha256": (
                    transition_previous_sha256
                ),
                "next_index_payload_sha256": (
                    hashlib.sha256(payload).hexdigest()
                ),
                "stage": stage,
            }
            transition = canonical_json_bytes_generic(
                {
                    **transition_body,
                    "transition_sha256": owned_sha256(
                        "vfe4.wt103.index-transition.v1",
                        transition_body,
                    ),
                }
            )
            transition_root = (
                experiment_root / ".index-transitions"
            )
            transition_root.mkdir(exist_ok=True)
            _regular_directory(transition_root)
            transition_path = (
                transition_root
                / f"{transition_previous_sha256}.json"
            )
            try:
                backend.create_exclusive(
                    transition_path,
                    transition,
                )
            except Exception as exc:
                if (
                    not transition_path.exists()
                    or _read_regular_bytes(transition_path)
                    != transition
                ):
                    raise RunLifecycleError(
                        "experiment index transition ownership "
                        "is already claimed"
                    ) from exc
            if _read_regular_bytes(transition_path) != transition:
                raise RunLifecycleError(
                    "experiment index transition reopen failed"
                )
            current_payload = (
                _read_regular_bytes(path)
                if path.exists()
                else None
            )
            if current_payload != previous_payload:
                raise RunLifecycleError(
                    "experiment index transition lost its compare-and-swap "
                    "precondition"
                )
            if previous_payload is None:
                backend.create_exclusive(path, payload)
            else:
                backend.replace_durable(path, payload)
        reopened = _read_regular_bytes(path)
        if reopened != payload:
            raise RunLifecycleError("experiment index reopen validation failed")
        semantic = {
            "stage": stage,
            "experiment_plan_sha256": plan.plan.experiment_plan_sha256,
            "run_manifest_sha256s": tuple(
                item.manifest_sha256 for item in run_manifests
            ),
            "artifact_record_sha256s": tuple(
                item.record_sha256 for item in artifact_records
            ),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        identity = ExperimentIndexIdentity(
            index_path=path,
            **semantic,
            identity_sha256=owned_sha256(
                "vfe4.wt103.experiment-index-identity.v1",
                semantic,
            ),
        )
        return validate_experiment_index(path, expected=identity)
    except RunLifecycleError:
        raise
    except Exception as exc:
        raise RunLifecycleError(
            f"experiment index publication failed: {exc}"
        ) from exc


__all__ = [
    "AttemptExecutionLease",
    "ExperimentIndexIdentity",
    "ExperimentPlan",
    "ExperimentPlanIdentity",
    "ReservedRun",
    "ResumeLineageEvent",
    "RunLifecycleError",
    "RunManifestIdentity",
    "finalize_run",
    "publish_experiment_index",
    "publish_experiment_plan",
    "recover_terminal_run",
    "reopen_resume_lineage_event",
    "release_run_execution_lease",
    "reserve_run",
    "validate_experiment_index",
    "validate_run_manifest",
]
