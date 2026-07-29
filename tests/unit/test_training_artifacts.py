from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import vfe4.artifacts.run_directory as run_directory_module
from vfe4.artifacts.durability import (
    canonical_json_bytes_generic,
    DurableFileIdentity,
    DurabilityCollisionError,
    DurabilityIdentity,
    VolumeFacts,
)
from vfe4.artifacts.environment import (
    AllocationObservation,
    ComponentBenchmark,
    ComponentForecast,
    DependencyLockIdentity,
    DiskByteForecast,
    DistributionIdentity,
    EnvironmentObservation,
    LockInputManifest,
    LockRequirement,
    PowerProviderIdentity,
    ResourceForecast,
    ResourceUsageEvent,
    ResourceUsageLedger,
    ResourceWorkload,
    TrainingExecutionIdentity,
    authorize_test_reservation,
    capture_environment,
    forecast_resources,
    required_resource_components,
    render_dependency_lock,
    run_allocation_preflight,
)
from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.artifacts.provenance import (
    build_training_provenance,
    capture_git_identity,
)
from vfe4.artifacts.run_directory import (
    ExperimentPlan,
    ResumeLineageEvent,
    RunLifecycleError,
    finalize_run,
    publish_experiment_index,
    publish_experiment_plan,
    reserve_run,
    validate_experiment_index,
    validate_run_manifest,
)
from vfe4.checkpoint.schema import make_checkpoint_identity
from vfe4.recording.failures import create_failure_record
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    EndpointInventory,
    EstimatorProtocol,
    ResourceProfile,
    WT103_CONFIRMATORY_SEED_IDS,
    WT103_TUNING_CELLS,
    WT103_TUNING_SEED_IDS,
    default_wt103_arm_specs,
    default_wt103_gate_specs,
    owned_sha256,
)


def _sha(character: str) -> str:
    return character * 64


def _inventory() -> EndpointInventory:
    return EndpointInventory.create(
        default_wt103_arm_specs(),
        default_wt103_gate_specs(),
        WT103_TUNING_CELLS,
        WT103_TUNING_SEED_IDS,
        WT103_CONFIRMATORY_SEED_IDS,
        EstimatorProtocol.create(),
    )


class _FilesystemBackend:
    """Small injected durability boundary; lifecycle behavior remains real."""

    def __init__(self) -> None:
        self.fail_name: str | None = None
        self.volume = VolumeFacts(
            volume_path="/fixture",
            volume_serial="fixture-volume",
            filesystem_type="ext4",
            is_remote=False,
        )

    def _write(
        self,
        path: Path,
        payload: bytes,
        *,
        operation: str,
    ) -> DurableFileIdentity:
        if path.name == self.fail_name:
            raise OSError(f"injected write failure for {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if operation == "exclusive_create":
            try:
                with path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise DurabilityCollisionError(str(exc)) from exc
        else:
            temporary = path.with_name(f".{path.name}.test-stage")
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        assert path.read_bytes() == payload
        return DurableFileIdentity.create(
            operation=operation,  # type: ignore[arg-type]
            payload=payload,
            volume_identity=self.volume.identity,
        )

    def create_exclusive(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        return self._write(path, payload, operation="exclusive_create")

    def replace_durable(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if not path.is_file():
            raise OSError("replacement target is absent")
        return self._write(path, payload, operation="replace")

    def publish_bytes(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path.exists():
            return self.replace_durable(path, payload)
        return self.create_exclusive(path, payload)

    def probe(self, root: Path) -> DurabilityIdentity:
        return DurabilityIdentity.create(
            backend_kind="posix",
            implementation_sha256=_sha("a"),
            volume=self.volume,
            create_sha256=_sha("b"),
            replace_sha256=_sha("c"),
            errors=(),
            obligations=(),
        )


def _attempt_resume_in_spawned_process(
    root: Path,
    plan_identity: object,
    event: ResumeLineageEvent,
    result_queue: object,
) -> None:
    """Exercise the real cross-process reservation boundary."""

    reserved = None
    try:
        reserved = reserve_run(
            root,
            "attempt-process-lease",
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,  # type: ignore[arg-type]
            mode="resume",
            resume_lineage=event,
            backend=_FilesystemBackend(),
        )
    except BaseException as exc:
        result_queue.put(  # type: ignore[attr-defined]
            ("error", type(exc).__name__, str(exc))
        )
    else:
        result_queue.put(  # type: ignore[attr-defined]
            ("ok", reserved.resume_count, "")
        )
    finally:
        release = getattr(
            run_directory_module,
            "release_run_execution_lease",
            None,
        )
        if reserved is not None and release is not None:
            release(reserved)


def _resume_then_exit_without_releasing(
    root: Path,
    plan_identity: object,
    event: ResumeLineageEvent,
    acquired_event: object,
) -> None:
    reserve_run(
        root,
        "attempt-process-lease",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,  # type: ignore[arg-type]
        mode="resume",
        resume_lineage=event,
        backend=_FilesystemBackend(),
    )
    acquired_event.set()  # type: ignore[attr-defined]
    os._exit(23)


def _plan() -> ExperimentPlan:
    return ExperimentPlan.create(
        experiment_id="wt103-post-h8-fixture",
        endpoint_inventory=_inventory(),
        git_head="1" * 40,
        dirty_digest=_sha("2"),
        config_sha256=_sha("3"),
        source_record_sha256=_sha("4"),
        tokenizer_spec_sha256=_sha("5"),
        token_cache_set_sha256=_sha("6"),
        window_manifest_sha256s=(_sha("7"), _sha("8"), _sha("9")),
        schedule_set_sha256=_sha("a"),
        factory_set_sha256=_sha("b"),
        objective_sha256=_sha("c"),
        checkpoint_schema_sha256=_sha("d"),
        resource_forecast_sha256=_sha("e"),
        expected_run_artifact_paths=(
            "metrics.csv",
            "metrics.jsonl",
        ),
        expected_group_artifact_paths=(
            "result-table.json",
        ),
    )


def _terminal_checkpoint(
    plan: ExperimentPlan,
    logical_key: str | None = None,
    *,
    checkpoint_payload_sha256: str | None = None,
    size_bytes: int = 128,
):
    return make_checkpoint_identity(
        logical_key=(
            plan.terminal_checkpoint_keys[0]
            if logical_key is None
            else logical_key
        ),
        checkpoint_role="terminal_scoring",
        scientific_state_sha256=_sha("1"),
        checkpoint_payload_sha256=(
            _sha("2")
            if checkpoint_payload_sha256 is None
            else checkpoint_payload_sha256
        ),
        checkpoint_manifest_body_sha256=_sha("3"),
        size_bytes=size_bytes,
    )


def _integrity(relative_path: str, payload: bytes) -> ArtifactIntegrityRecord:
    return ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _write_run_artifacts(
    backend: _FilesystemBackend,
    run_path: Path,
) -> tuple[ArtifactIntegrityRecord, ...]:
    payloads = {
        "metrics.csv": b"ordinal,loss\n0,1.0\n",
        "metrics.jsonl": b'{"loss":1.0,"ordinal":0}\n',
    }
    for relative_path, payload in payloads.items():
        backend.create_exclusive(run_path / relative_path, payload)
    return tuple(
        _integrity(relative_path, payload)
        for relative_path, payload in payloads.items()
    )


def _write_terminal_checkpoint(
    backend: _FilesystemBackend,
    run_path: Path,
    plan: ExperimentPlan,
    *,
    logical_key: str | None = None,
    relative_path: str = "checkpoints/terminal-scoring.pt",
):
    key = (
        plan.terminal_checkpoint_keys[0]
        if logical_key is None
        else logical_key
    )
    payload = f"durable checkpoint bytes for {key}".encode()
    backend.create_exclusive(run_path / relative_path, payload)
    identity = _terminal_checkpoint(
        plan,
        key,
        checkpoint_payload_sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    return identity, _integrity(relative_path, payload)


def _failure_artifact(
    *,
    run_id: str,
) -> tuple[bytes, str]:
    record = create_failure_record(
        ordinal=0,
        utc_timestamp="2026-07-28T03:01:00.000000Z",
        monotonic_ns=1,
        run_id=run_id,
        arm_id="WT103-A0-AR-v1",
        seed_id=2026072101,
        phase="training",
        step=1,
        pass_index=0,
        cursor_sha256=_sha("7"),
        checkpoint_identity_sha256=None,
        retry_classification="not_retryable",
        scientific_state_advanced=False,
        terminal_disposition="failed",
        exception=RuntimeError("fixture terminal failure"),
        previous_record_sha256="0" * 64,
    )
    return canonical_json_bytes_generic(record) + b"\n", record.record_sha256


def _finalize_confirmation(
    *,
    root: Path,
    plan_identity,
    backend: _FilesystemBackend,
    ordinal: int,
    logical_key: str,
):
    reserved = reserve_run(
        root,
        f"confirmation-{ordinal:04d}",
        run_role="confirmation",
        started_utc=f"2026-07-28T01:{ordinal:02d}:00Z",
        plan=plan_identity,
        backend=backend,
    )
    records = _write_run_artifacts(backend, reserved.inprogress_path)
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan_identity.plan,
        logical_key=logical_key,
    )
    return finalize_run(
        reserved,
        disposition="success",
        checkpoints=(checkpoint,),
        checkpoint_artifact_records=(checkpoint_record,),
        artifact_records=records,
        environment_sha256=_sha("4"),
        provenance_sha256=_sha("5"),
        ended_utc=f"2026-07-28T02:{ordinal:02d}:00Z",
        monotonic_duration_seconds=12.5,
        failure_record_sha256=None,
        backend=backend,
    )


def _finalize_tuning(
    *,
    root: Path,
    plan_identity,
    backend: _FilesystemBackend,
    ordinal: int,
    tuning_attempt_key: str,
    disposition: str = "success",
):
    reserved = reserve_run(
        root,
        f"tuning-{ordinal:04d}",
        run_role="tuning",
        tuning_attempt_key=tuning_attempt_key,
        started_utc=f"2026-07-28T00:{ordinal:02d}:00Z",
        plan=plan_identity,
        backend=backend,
    )
    if disposition == "success":
        records = _write_run_artifacts(backend, reserved.inprogress_path)
        failure_record_sha256 = None
    else:
        failure_payload, failure_record_sha256 = _failure_artifact(
            run_id=reserved.run_id,
        )
        backend.create_exclusive(
            reserved.inprogress_path / "failures.jsonl",
            failure_payload,
        )
        records = (_integrity("failures.jsonl", failure_payload),)
    return finalize_run(
        reserved,
        disposition=disposition,  # type: ignore[arg-type]
        checkpoints=(),
        checkpoint_artifact_records=(),
        artifact_records=records,
        environment_sha256=_sha("4"),
        provenance_sha256=_sha("5"),
        ended_utc=f"2026-07-28T01:{ordinal:02d}:00Z",
        monotonic_duration_seconds=12.5,
        failure_record_sha256=failure_record_sha256,
        backend=backend,
    )


def test_run_lifecycle_publishes_plan_reserves_finalizes_validates_then_indexes(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    plan = _plan()
    root = tmp_path / "experiment"

    plan_identity = publish_experiment_plan(root, plan, backend=backend)
    reserved = reserve_run(
        root,
        "attempt-0001",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    with pytest.raises(RunLifecycleError, match="execution lease"):
        reserve_run(
            root,
            "attempt-0001",
            run_role="confirmation",
            started_utc="2026-07-28T01:00:00Z",
            plan=plan_identity,
            backend=backend,
        )
    artifact_records = _write_run_artifacts(
        backend,
        reserved.inprogress_path,
    )
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan,
    )
    with pytest.raises(
        RunLifecycleError,
        match="checkpoint artifact inventory",
    ):
        finalize_run(
            reserved,
            disposition="success",
            checkpoints=(checkpoint,),
            checkpoint_artifact_records=(),
            artifact_records=artifact_records,
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=12.5,
            failure_record_sha256=None,
            backend=backend,
        )
    manifest = finalize_run(
        reserved,
        disposition="success",
        checkpoints=(checkpoint,),
        checkpoint_artifact_records=(checkpoint_record,),
        artifact_records=artifact_records,
        environment_sha256=_sha("4"),
        provenance_sha256=_sha("5"),
        ended_utc="2026-07-28T02:00:00Z",
        monotonic_duration_seconds=12.5,
        failure_record_sha256=None,
        backend=backend,
    )

    assert not reserved.inprogress_path.exists()
    assert manifest.run_path == root / "runs" / "attempt-0001"
    assert manifest.checkpoint_artifact_record_sha256s == (
        checkpoint_record.record_sha256,
    )
    assert validate_run_manifest(
        manifest.run_path / "run-manifest.json",
        expected=manifest,
    ) == manifest
    tuning_manifests = tuple(
        _finalize_tuning(
            root=root,
            plan_identity=plan_identity,
            backend=backend,
            ordinal=ordinal,
            tuning_attempt_key=key,
        )
        for ordinal, key in enumerate(plan.tuning_attempt_keys)
    )
    pretest_index = publish_experiment_index(
        root,
        plan=plan_identity,
        run_manifests=(*tuning_manifests, manifest),
        stage="pretest",
        artifact_records=(),
        backend=backend,
    )
    assert pretest_index.stage == "pretest"
    assert pretest_index.artifact_record_sha256s == ()
    remaining = tuple(
        _finalize_confirmation(
            root=root,
            plan_identity=plan_identity,
            backend=backend,
            ordinal=ordinal,
            logical_key=logical_key,
        )
        for ordinal, logical_key in enumerate(
            plan.terminal_checkpoint_keys[1:],
            start=1,
        )
    )
    complete_manifests = (*tuning_manifests, manifest, *remaining)
    result_table_payload = b'{"rows":[]}'
    backend.create_exclusive(
        root / "result-table.json",
        result_table_payload,
    )
    with pytest.raises(
        RunLifecycleError,
        match="confirmation terminal checkpoint inventory",
    ):
        publish_experiment_index(
            root,
            plan=plan_identity,
            run_manifests=(*tuning_manifests, manifest),
            stage="final",
            artifact_records=(
                _integrity("result-table.json", result_table_payload),
            ),
            backend=backend,
        )
    with pytest.raises(
        RunLifecycleError,
        match="confirmation terminal checkpoint inventory",
    ):
        publish_experiment_index(
            root,
            plan=plan_identity,
            run_manifests=(
                *tuning_manifests,
                manifest,
                *reversed(remaining),
            ),
            stage="final",
            artifact_records=(
                _integrity("result-table.json", result_table_payload),
            ),
            backend=backend,
        )
    index = publish_experiment_index(
        root,
        plan=plan_identity,
        run_manifests=complete_manifests,
        stage="final",
        artifact_records=(
            _integrity("result-table.json", result_table_payload),
        ),
        backend=backend,
    )
    assert index.run_manifest_sha256s == tuple(
        item.manifest_sha256 for item in complete_manifests
    )
    assert index.stage == "final"
    assert index.artifact_record_sha256s == (
        _integrity(
            "result-table.json",
            result_table_payload,
        ).record_sha256,
    )
    assert (root / "experiment-index.json").is_file()
    assert validate_experiment_index(
        root / "experiment-index.json",
        expected=index,
    ) == index

    index_path = root / "experiment-index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["runs"] = list(reversed(document["runs"]))
    body = dict(document)
    body.pop("index_sha256")
    document["index_sha256"] = owned_sha256(
        "vfe4.wt103.experiment-index.v1",
        body,
    )
    index_path.write_bytes(canonical_json_bytes_generic(document))
    with pytest.raises(
        RunLifecycleError,
        match="tuning attempt inventory",
    ):
        validate_experiment_index(index_path)

    with pytest.raises(RunLifecycleError, match="already"):
        reserve_run(
            root,
            "attempt-0001",
            run_role="confirmation",
            started_utc="2026-07-28T01:00:00Z",
            plan=plan_identity,
            backend=backend,
        )
    checkpoint_path = manifest.run_path / checkpoint_record.relative_path
    checkpoint_path.write_bytes(b"x" * checkpoint_record.size_bytes)
    with pytest.raises(
        RunLifecycleError,
        match="artifact integrity mismatch",
    ):
        validate_run_manifest(manifest.run_path / "run-manifest.json")


def test_run_role_separates_tuning_from_terminal_confirmation(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    tuning = reserve_run(
        root,
        "tuning-0001",
        run_role="tuning",
        tuning_attempt_key=plan_identity.plan.tuning_attempt_keys[0],
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    payloads = {
        "metrics.csv": b"ordinal,loss\n0,1.0\n",
        "metrics.jsonl": b'{"loss":1.0,"ordinal":0}\n',
    }
    for relative_path, payload in payloads.items():
        backend.create_exclusive(
            tuning.inprogress_path / relative_path,
            payload,
        )
    tuned = finalize_run(
        tuning,
        disposition="success",
        checkpoints=(),
        checkpoint_artifact_records=(),
        artifact_records=tuple(
            _integrity(relative_path, payload)
            for relative_path, payload in payloads.items()
        ),
        environment_sha256=_sha("4"),
        provenance_sha256=_sha("5"),
        ended_utc="2026-07-28T02:00:00Z",
        monotonic_duration_seconds=12.5,
        failure_record_sha256=None,
        backend=backend,
    )
    assert tuned.run_role == "tuning"
    assert tuned.checkpoint_identity_sha256s == ()

    confirmation = reserve_run(
        root,
        "confirmation-0001",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    for relative_path, payload in payloads.items():
        backend.create_exclusive(
            confirmation.inprogress_path / relative_path,
            payload,
        )
    with pytest.raises(RunLifecycleError, match="exactly one"):
        finalize_run(
            confirmation,
            disposition="success",
            checkpoints=(),
            checkpoint_artifact_records=(),
            artifact_records=tuple(
                _integrity(relative_path, payload)
                for relative_path, payload in payloads.items()
            ),
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=12.5,
            failure_record_sha256=None,
            backend=backend,
        )
    assert confirmation.inprogress_path.is_dir()


def test_experiment_index_transition_is_compare_and_swap_exclusive(
    tmp_path: Path,
) -> None:
    class _IndexBarrierBackend(_FilesystemBackend):
        def __init__(self) -> None:
            super().__init__()
            self.race_enabled = False
            self.barrier = threading.Barrier(2)

        def create_exclusive(
            self,
            path: Path,
            payload: bytes,
        ) -> DurableFileIdentity:
            if (
                self.race_enabled
                and path.parent.name == ".index-transitions"
            ):
                self.barrier.wait(timeout=5.0)
            return super().create_exclusive(path, payload)

    backend = _IndexBarrierBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    manifests = tuple(
        _finalize_confirmation(
            root=root,
            plan_identity=plan_identity,
            backend=backend,
            ordinal=index,
            logical_key=logical_key,
        )
        for index, logical_key in enumerate(
            plan_identity.plan.terminal_checkpoint_keys[:3],
            start=1,
        )
    )
    publish_experiment_index(
        root,
        plan=plan_identity,
        run_manifests=(manifests[0],),
        stage="pretest",
        artifact_records=(),
        backend=backend,
    )
    backend.race_enabled = True

    def publish(candidate):
        return publish_experiment_index(
            root,
            plan=plan_identity,
            run_manifests=(manifests[0], candidate),
            stage="pretest",
            artifact_records=(),
            backend=backend,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(publish, manifests[1]),
            executor.submit(publish, manifests[2]),
        )
        outcomes = tuple(
            future.exception() or future.result() for future in futures
        )
    successes = tuple(
        item for item in outcomes if not isinstance(item, BaseException)
    )
    failures = tuple(
        item for item in outcomes if isinstance(item, BaseException)
    )
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RunLifecycleError)
    assert "transition" in str(failures[0])
    reopened = validate_experiment_index(root / "experiment-index.json")
    assert reopened == successes[0]


def test_explicit_resume_appends_lineage_and_failure_is_terminally_retained(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    first = reserve_run(
        root,
        "attempt-crash",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(first)
    parent = _terminal_checkpoint(_plan())
    lineage = ResumeLineageEvent.create(
        parent_checkpoint=parent,
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="explicit operator resume after process loss",
        resumed_utc="2026-07-28T03:00:00Z",
    )

    resumed = reserve_run(
        root,
        "attempt-crash",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=lineage,
        backend=backend,
    )
    assert resumed.inprogress_path == first.inprogress_path
    assert resumed.resume_count == 1
    lineage_payload = (
        resumed.inprogress_path / "resume-lineage.jsonl"
    ).read_bytes()
    assert lineage.lineage_sha256.encode("ascii") in lineage_payload
    run_directory_module.consume_resume_execution_retry(
        resumed,
        backend=backend,
    )

    failure_payload, failure_head = _failure_artifact(
        run_id=resumed.run_id,
    )
    backend.create_exclusive(
        resumed.inprogress_path / "failures.jsonl",
        failure_payload,
    )
    failed = finalize_run(
        resumed,
        disposition="failure",
        checkpoints=(),
        checkpoint_artifact_records=(),
        artifact_records=(
            _integrity("failures.jsonl", failure_payload),
        ),
        environment_sha256=_sha("8"),
        provenance_sha256=_sha("9"),
        ended_utc="2026-07-28T03:01:00Z",
        monotonic_duration_seconds=60.0,
        failure_record_sha256=failure_head,
        backend=backend,
    )
    assert failed.disposition == "failure"
    assert failed.run_path.is_dir()
    assert not (root / "runs" / ".inprogress" / "attempt-crash").exists()
    assert validate_run_manifest(
        failed.run_path / "run-manifest.json",
        expected=failed,
    ) == failed


def test_attempt_execution_lease_blocks_cross_process_resume_before_mutation(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-process-lease",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    event = ResumeLineageEvent.create(
        parent_checkpoint=_terminal_checkpoint(plan_identity.plan),
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="resume after authenticated process loss",
        resumed_utc="2026-07-28T03:00:00Z",
    )
    context = multiprocessing.get_context("spawn")

    def attempt() -> tuple[object, ...]:
        queue = context.Queue()
        process = context.Process(
            target=_attempt_resume_in_spawned_process,
            args=(root, plan_identity, event, queue),
        )
        process.start()
        process.join(timeout=15.0)
        assert process.exitcode == 0
        result = queue.get(timeout=2.0)
        queue.close()
        queue.join_thread()
        return result

    blocked = attempt()
    assert blocked[0] == "error"
    assert blocked[1] == "RunLifecycleError"
    assert "execution lease" in str(blocked[2])
    assert not (original.inprogress_path / "resume-owner.json").exists()
    assert not (original.inprogress_path / "resume-lineage.jsonl").exists()
    assert not (original.inprogress_path / "resume-leases").exists()

    release = getattr(
        run_directory_module,
        "release_run_execution_lease",
        None,
    )
    assert release is not None
    release(original)
    acquired = context.Event()
    crashed = context.Process(
        target=_resume_then_exit_without_releasing,
        args=(root, plan_identity, event, acquired),
    )
    crashed.start()
    assert acquired.wait(timeout=10.0)
    crashed.join(timeout=10.0)
    assert crashed.exitcode == 23

    reopened = reserve_run(
        root,
        "attempt-process-lease",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=event,
        backend=backend,
    )
    assert reopened.resume_count == 1
    release(reopened)


def test_run_finalization_requires_the_live_execution_lease(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    reserved = reserve_run(
        root,
        "attempt-released-before-finalize",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    records = _write_run_artifacts(backend, reserved.inprogress_path)
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan_identity.plan,
        logical_key=plan_identity.plan.terminal_checkpoint_keys[0],
    )
    run_directory_module.release_run_execution_lease(reserved)

    with pytest.raises(RunLifecycleError, match="active attempt execution lease"):
        finalize_run(
            reserved,
            disposition="success",
            checkpoints=(checkpoint,),
            checkpoint_artifact_records=(checkpoint_record,),
            artifact_records=records,
            environment_sha256=_sha("8"),
            provenance_sha256=_sha("9"),
            ended_utc="2026-07-28T03:01:00Z",
            monotonic_duration_seconds=1.0,
            failure_record_sha256=None,
            backend=backend,
        )
    assert not (root / "runs" / reserved.run_id).exists()
    assert reserved.inprogress_path.is_dir()


def test_second_new_infrastructure_retry_fails_before_resume_mutation(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-retry-budget",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    release = getattr(
        run_directory_module,
        "release_run_execution_lease",
        None,
    )
    if release is not None:
        release(original)
    parent = _terminal_checkpoint(plan_identity.plan)
    first_event = ResumeLineageEvent.create(
        parent_checkpoint=parent,
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="the one preregistered infrastructure retry",
        resumed_utc="2026-07-28T03:00:00Z",
    )
    second_event = ResumeLineageEvent.create(
        parent_checkpoint=parent,
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("8"),
        reason="a forbidden second infrastructure retry",
        resumed_utc="2026-07-28T03:01:00Z",
    )
    first = reserve_run(
        root,
        "attempt-retry-budget",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=first_event,
        backend=backend,
    )
    owner_path = first.inprogress_path / "resume-owner.json"
    lineage_path = first.inprogress_path / "resume-lineage.jsonl"
    lease_path = first.inprogress_path / "resume-leases" / "00000001.json"
    before = (
        owner_path.read_bytes(),
        lineage_path.read_bytes(),
        lease_path.read_bytes(),
    )
    assert (
        run_directory_module.reopen_resume_lineage_event(
            first.inprogress_path
        )
        == first_event
    )
    if release is not None:
        release(first)

    with pytest.raises(
        RunLifecycleError,
        match="infrastructure retry budget",
    ):
        reserve_run(
            root,
            "attempt-retry-budget",
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=second_event,
            expected_resume_owner_lineage_sha256=(
                first_event.lineage_sha256
            ),
            backend=backend,
        )
    assert (
        owner_path.read_bytes(),
        lineage_path.read_bytes(),
        lease_path.read_bytes(),
    ) == before
    assert not (
        first.inprogress_path / "resume-leases" / "00000002.json"
    ).exists()
    assert not (
        first.inprogress_path / "resume-owner-takeovers"
    ).exists()

    idempotent = reserve_run(
        root,
        "attempt-retry-budget",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=first_event,
        expected_resume_owner_lineage_sha256=(
            first_event.lineage_sha256
        ),
        backend=backend,
    )
    assert idempotent.resume_count == 1
    assert (
        owner_path.read_bytes(),
        lineage_path.read_bytes(),
        lease_path.read_bytes(),
    ) == before
    if release is not None:
        release(idempotent)


def test_resume_execution_start_consumes_same_lineage_before_retry_mutation(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-consumed-retry",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(original)
    event = ResumeLineageEvent.create(
        parent_checkpoint=_terminal_checkpoint(plan_identity.plan),
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="the one preregistered infrastructure retry",
        resumed_utc="2026-07-28T03:00:00Z",
    )
    resumed = reserve_run(
        root,
        original.run_id,
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=event,
        backend=backend,
    )
    marker_path = resumed.inprogress_path / "resume-execution-started.json"
    assert not marker_path.exists()
    run_directory_module.release_run_execution_lease(resumed)
    with pytest.raises(
        RunLifecycleError,
        match="active attempt execution lease",
    ):
        run_directory_module.consume_resume_execution_retry(
            resumed,
            backend=backend,
        )
    assert not marker_path.exists()
    resumed = reserve_run(
        root,
        original.run_id,
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=event,
        expected_resume_owner_lineage_sha256=event.lineage_sha256,
        backend=backend,
    )

    run_directory_module.consume_resume_execution_retry(
        resumed,
        backend=backend,
    )

    assert marker_path.is_file()
    before = {
        path.name: path.read_bytes()
        for path in (
            resumed.inprogress_path / "resume-owner.json",
            resumed.inprogress_path / "resume-lineage.jsonl",
            resumed.inprogress_path / "resume-leases" / "00000001.json",
            marker_path,
        )
    }
    run_directory_module.release_run_execution_lease(resumed)

    with pytest.raises(
        RunLifecycleError,
        match="infrastructure retry.*consumed",
    ):
        reserve_run(
            root,
            original.run_id,
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=event,
            expected_resume_owner_lineage_sha256=event.lineage_sha256,
            backend=backend,
        )
    assert {
        path.name: path.read_bytes()
        for path in (
            resumed.inprogress_path / "resume-owner.json",
            resumed.inprogress_path / "resume-lineage.jsonl",
            resumed.inprogress_path / "resume-leases" / "00000001.json",
            marker_path,
        )
    } == before


def test_production_resume_consumes_retry_before_scientific_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt.ProductionAttemptSpec.create(
        ordinal=0,
        role="confirmation",
        arm_id=training.endpoint_inventory.arms[0].arm_id,
        seed_id=training.profile.statistics.confirmatory_seed_ids[0],
        learning_rate=1.0e-4,
        weight_decay=0.0,
        pass_count=training.profile.cadence.confirmatory_passes,
        quarter_pass=False,
    )
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        attempt.attempt_id,
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(original)
    event = ResumeLineageEvent.create(
        parent_checkpoint=_terminal_checkpoint(plan_identity.plan),
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="resume after original process loss",
        resumed_utc="2026-07-28T03:00:00Z",
    )
    resumed = reserve_run(
        root,
        attempt.attempt_id,
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=event,
        backend=backend,
    )
    marker_path = resumed.inprogress_path / "resume-execution-started.json"
    ledger_path = tmp_path / "resource-usage-ledger.json"
    real_debit = production_attempt._debit_resource_usage
    fail_predebit = True

    def debit_with_one_prepublication_failure(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal fail_predebit
        if fail_predebit:
            fail_predebit = False
            raise RuntimeError("injected pre-debit publication failure")
        return real_debit(**kwargs)

    monkeypatch.setattr(
        production_attempt,
        "_debit_resource_usage",
        debit_with_one_prepublication_failure,
    )

    def stop_at_scientific_boundary(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        assert marker_path.is_file()
        reopened = production_attempt._reopen_resource_usage_ledger(
            path=ledger_path,
            experiment_plan_sha256=(
                plan_identity.plan.experiment_plan_sha256
            ),
        )
        assert len(reopened.events) == 1
        raise RuntimeError("stopped after retry execution start")

    monkeypatch.setattr(
        production_attempt,
        "_measure_attempt_resource_usage",
        stop_at_scientific_boundary,
    )
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=plan_identity.plan.experiment_plan_sha256,
        resource_profile=training.profile.resources,
    )

    with pytest.raises(
        RuntimeError,
        match="injected pre-debit publication failure",
    ):
        production_attempt._execute_reserved_attempt_under_lease(
            attempt=attempt,
            training=training,
            source_lock=None,
            readiness=None,
            cache_root=tmp_path,
            plan=plan_identity,
            reserved=resumed,
            backend=backend,
            resume_active=True,
            sampler=None,
            power_provider_identity_sha256=_sha("5"),
            conservative_power_watts=600.0,
            ledger=ledger,
            ledger_path=ledger_path,
        )
    assert not marker_path.exists()
    assert not ledger_path.exists()

    with pytest.raises(
        RuntimeError,
        match="stopped after retry execution start",
    ):
        production_attempt._execute_reserved_attempt_under_lease(
            attempt=attempt,
            training=training,
            source_lock=None,
            readiness=None,
            cache_root=tmp_path,
            plan=plan_identity,
            reserved=resumed,
            backend=backend,
            resume_active=True,
            sampler=None,
            power_provider_identity_sha256=_sha("5"),
            conservative_power_watts=600.0,
            ledger=ledger,
            ledger_path=ledger_path,
        )
    run_directory_module.release_run_execution_lease(resumed)


@pytest.mark.parametrize(
    ("failure_name", "phase"),
    [
        ("experiment-plan.json", "plan"),
        ("reservation.json", "reservation"),
        ("run-manifest.json", "manifest"),
    ],
)
def test_lifecycle_write_failures_never_publish_a_false_terminal_run(
    tmp_path: Path,
    failure_name: str,
    phase: str,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    if phase == "plan":
        backend.fail_name = failure_name
        with pytest.raises(RunLifecycleError, match="experiment plan"):
            publish_experiment_plan(root, _plan(), backend=backend)
        assert not (root / "runs").exists()
        return

    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    backend.fail_name = failure_name
    if phase == "reservation":
        with pytest.raises(RunLifecycleError, match="reservation"):
            reserve_run(
                root,
                "attempt-fail",
                run_role="confirmation",
                started_utc="2026-07-28T01:00:01Z",
                plan=plan_identity,
                backend=backend,
            )
        assert (root / "runs" / ".inprogress" / "attempt-fail").is_dir()
        assert not (root / "runs" / "attempt-fail").exists()
        backend.fail_name = None
        recovered = reserve_run(
            root,
            "attempt-fail",
            run_role="confirmation",
            started_utc="2026-07-28T01:00:00Z",
            plan=plan_identity,
            backend=backend,
        )
        reservation_path = recovered.inprogress_path / "reservation.json"
        original = reservation_path.read_bytes()
        with pytest.raises(RunLifecycleError, match="execution lease"):
            reserve_run(
                root,
                "attempt-fail",
                run_role="confirmation",
                started_utc="2026-07-28T01:00:02Z",
                plan=plan_identity,
                backend=backend,
            )
        assert reservation_path.read_bytes() == original
        return

    reserved = reserve_run(
        root,
        "attempt-fail",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    failure_payload, failure_head = _failure_artifact(
        run_id=reserved.run_id,
    )
    backend.create_exclusive(
        reserved.inprogress_path / "failures.jsonl",
        failure_payload,
    )
    with pytest.raises(RunLifecycleError, match="manifest"):
        finalize_run(
            reserved,
            disposition="failure",
            checkpoints=(),
            checkpoint_artifact_records=(),
            artifact_records=(
                _integrity("failures.jsonl", failure_payload),
            ),
            environment_sha256=_sha("1"),
            provenance_sha256=_sha("2"),
            ended_utc="2026-07-28T03:01:00Z",
            monotonic_duration_seconds=1.0,
            failure_record_sha256=failure_head,
            backend=backend,
        )
    assert reserved.inprogress_path.is_dir()
    assert not (root / "runs" / "attempt-fail").exists()


def test_resume_lease_rejects_duplicate_live_owner(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-race",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(original)
    parent = _terminal_checkpoint(plan_identity.plan)
    events = tuple(
        ResumeLineageEvent.create(
            parent_checkpoint=parent,
            environment_sha256=_sha("6"),
            cursor_sha256=_sha(character),
            reason=f"concurrent resume {ordinal}",
            resumed_utc=f"2026-07-28T03:0{ordinal}:00Z",
        )
        for ordinal, character in ((1, "7"), (2, "8"))
    )

    def attempt(event: ResumeLineageEvent):
        return reserve_run(
            root,
            "attempt-race",
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=event,
            backend=backend,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            future.exception() or future.result()
            for future in (
                executor.submit(attempt, events[0]),
                executor.submit(attempt, events[1]),
            )
        )
    successes = tuple(
        item for item in outcomes if not isinstance(item, BaseException)
    )
    failures = tuple(
        item for item in outcomes if isinstance(item, BaseException)
    )
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RunLifecycleError)
    winner = successes[0]
    assert winner.resume_count == 1
    assert (
        winner.inprogress_path / "resume-leases" / "00000001.json"
    ).is_file()
    lineage_lines = (
        winner.inprogress_path / "resume-lineage.jsonl"
    ).read_bytes().splitlines()
    assert len(lineage_lines) == 1

    winner_event = next(
        event
        for event in events
        if event.lineage_sha256.encode("ascii") in lineage_lines[0]
    )
    run_directory_module.release_run_execution_lease(winner)
    assert attempt(winner_event).resume_count == 1
    assert (
        winner.inprogress_path / "resume-lineage.jsonl"
    ).read_bytes().splitlines() == lineage_lines


def test_resume_owner_survives_orphan_append_without_spending_another_retry(
    tmp_path: Path,
) -> None:
    class _OrphaningBackend(_FilesystemBackend):
        def __init__(self) -> None:
            super().__init__()
            self.orphan_once = True

        def create_exclusive(
            self,
            path: Path,
            payload: bytes,
        ) -> DurableFileIdentity:
            identity = super().create_exclusive(path, payload)
            if (
                self.orphan_once
                and path.parent.name == "resume-leases"
                and path.name == "00000001.json"
            ):
                self.orphan_once = False
                raise OSError("injected process loss after append-CAS create")
            return identity

    backend = _OrphaningBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-owner",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(original)
    parent = _terminal_checkpoint(plan_identity.plan)
    first_event = ResumeLineageEvent.create(
        parent_checkpoint=parent,
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="recover orphan append CAS",
        resumed_utc="2026-07-28T03:00:00Z",
    )
    second_event = ResumeLineageEvent.create(
        parent_checkpoint=parent,
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("8"),
        reason="explicit stale-owner takeover",
        resumed_utc="2026-07-28T03:01:00Z",
    )

    with pytest.raises(RunLifecycleError, match="append-CAS"):
        reserve_run(
            root,
            "attempt-owner",
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=first_event,
            backend=backend,
        )
    marker_path = (
        original.inprogress_path / "resume-execution-started.json"
    )
    assert not marker_path.exists()
    resumed = reserve_run(
        root,
        "attempt-owner",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=first_event,
        backend=backend,
    )
    assert resumed.resume_count == 1
    assert resumed.resume_owner_lineage_sha256 == first_event.lineage_sha256
    assert not marker_path.exists()
    assert len(
        (resumed.inprogress_path / "resume-lineage.jsonl")
        .read_bytes()
        .splitlines()
    ) == 1

    before = (
        (resumed.inprogress_path / "resume-owner.json").read_bytes(),
        (resumed.inprogress_path / "resume-lineage.jsonl").read_bytes(),
    )
    run_directory_module.release_run_execution_lease(resumed)
    with pytest.raises(
        RunLifecycleError,
        match="infrastructure retry budget",
    ):
        reserve_run(
            root,
            "attempt-owner",
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=second_event,
            backend=backend,
        )
    assert (
        (resumed.inprogress_path / "resume-owner.json").read_bytes(),
        (resumed.inprogress_path / "resume-lineage.jsonl").read_bytes(),
    ) == before
    retry = reserve_run(
        root,
        "attempt-owner",
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=first_event,
        backend=backend,
    )
    assert retry.resume_count == 1
    assert retry.resume_owner_lineage_sha256 == first_event.lineage_sha256
    run_directory_module.consume_resume_execution_retry(
        retry,
        backend=backend,
    )

    failure_payload, failure_head = _failure_artifact(
        run_id=retry.run_id,
    )
    backend.create_exclusive(
        retry.inprogress_path / "failures.jsonl",
        failure_payload,
    )
    finalized = finalize_run(
        retry,
        disposition="failure",
        checkpoints=(),
        checkpoint_artifact_records=(),
        artifact_records=(
            _integrity("failures.jsonl", failure_payload),
        ),
        environment_sha256=_sha("8"),
        provenance_sha256=_sha("9"),
        ended_utc="2026-07-28T03:02:00Z",
        monotonic_duration_seconds=120.0,
        failure_record_sha256=failure_head,
        backend=backend,
    )
    owner = json.loads(
        (finalized.run_path / "resume-owner.json").read_text("utf-8")
    )
    assert owner["state"] == "terminal_closed"
    assert owner["lineage_sha256"] == first_event.lineage_sha256
    assert owner["terminal_manifest_sha256"] == finalized.manifest_sha256


def test_manifest_published_before_rename_recovers_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    reserved = reserve_run(
        root,
        "attempt-rename-crash",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    records = _write_run_artifacts(backend, reserved.inprogress_path)
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan_identity.plan,
    )

    def fail_before_rename(_source: Path, _destination: Path) -> None:
        raise OSError("injected crash before terminal rename")

    real_rename = run_directory_module._rename_directory_no_replace
    monkeypatch.setattr(
        run_directory_module,
        "_rename_directory_no_replace",
        fail_before_rename,
    )
    arguments = {
        "disposition": "success",
        "checkpoints": (checkpoint,),
        "checkpoint_artifact_records": (checkpoint_record,),
        "artifact_records": records,
        "environment_sha256": _sha("4"),
        "provenance_sha256": _sha("5"),
        "ended_utc": "2026-07-28T02:00:00Z",
        "monotonic_duration_seconds": 12.5,
        "failure_record_sha256": None,
        "backend": backend,
    }
    with pytest.raises(RunLifecycleError, match="rename"):
        finalize_run(reserved, **arguments)  # type: ignore[arg-type]
    manifest_path = reserved.inprogress_path / "run-manifest.json"
    published = manifest_path.read_bytes()

    monkeypatch.setattr(
        run_directory_module,
        "_rename_directory_no_replace",
        real_rename,
    )
    manifest = finalize_run(
        reserved,
        **arguments,  # type: ignore[arg-type]
    )
    assert manifest.run_path == reserved.final_path
    assert (manifest.run_path / "run-manifest.json").read_bytes() == published
    assert not reserved.inprogress_path.exists()


@pytest.mark.parametrize("tamper", (None, "manifest", "owner"))
def test_terminal_recovery_repairs_only_exact_manifested_active_resume_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str | None,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    original = reserve_run(
        root,
        "attempt-owner-terminalization-crash",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    run_directory_module.release_run_execution_lease(original)
    event = ResumeLineageEvent.create(
        parent_checkpoint=_terminal_checkpoint(plan_identity.plan),
        environment_sha256=_sha("6"),
        cursor_sha256=_sha("7"),
        reason="resume before terminal owner crash fixture",
        resumed_utc="2026-07-28T01:30:00Z",
    )
    reserved = reserve_run(
        root,
        original.run_id,
        run_role="confirmation",
        started_utc=None,
        plan=plan_identity,
        mode="resume",
        resume_lineage=event,
        backend=backend,
    )
    run_directory_module.consume_resume_execution_retry(
        reserved,
        backend=backend,
    )
    records = _write_run_artifacts(backend, reserved.inprogress_path)
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan_identity.plan,
    )
    real_close_owner = run_directory_module._close_resume_owner

    def process_loss_before_owner_close(*_args, **_kwargs) -> None:
        assert (
            reserved.inprogress_path / "run-manifest.json"
        ).is_file()
        raise OSError("process loss before terminal owner closure")

    monkeypatch.setattr(
        run_directory_module,
        "_close_resume_owner",
        process_loss_before_owner_close,
    )
    with pytest.raises(
        RunLifecycleError,
        match="run finalization failed",
    ):
        finalize_run(
            reserved,
            disposition="success",
            checkpoints=(checkpoint,),
            checkpoint_artifact_records=(checkpoint_record,),
            artifact_records=records,
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=12.5,
            failure_record_sha256=None,
            backend=backend,
        )
    monkeypatch.setattr(
        run_directory_module,
        "_close_resume_owner",
        real_close_owner,
    )
    manifest_path = reserved.inprogress_path / "run-manifest.json"
    owner_path = reserved.inprogress_path / "resume-owner.json"
    manifest_payload = manifest_path.read_bytes()
    owner = json.loads(owner_path.read_text("utf-8"))
    assert owner["state"] == "active"
    assert owner["terminal_manifest_sha256"] is None
    run_directory_module.release_run_execution_lease(reserved)

    if tamper == "manifest":
        foreign = json.loads(manifest_payload)
        foreign["environment_sha256"] = _sha("f")
        manifest_path.write_bytes(canonical_json_bytes_generic(foreign))
    elif tamper == "owner":
        foreign_owner = run_directory_module._resume_owner_payload(
            run_id=reserved.run_id,
            experiment_plan_sha256=reserved.experiment_plan_sha256,
            reservation_sha256=reserved.reservation_sha256,
            resume_ordinal=1,
            previous_owner_lineage_sha256=str(
                owner["previous_owner_lineage_sha256"]
            ),
            lineage_sha256=_sha("f"),
            state="active",
            terminal_manifest_sha256=None,
        )
        owner_path.write_bytes(
            canonical_json_bytes_generic(foreign_owner)
        )

    if tamper is not None:
        with pytest.raises(RunLifecycleError):
            run_directory_module.recover_terminal_run(
                root,
                reserved.run_id,
                plan=plan_identity,
                backend=backend,
            )
        assert reserved.inprogress_path.is_dir()
        assert not reserved.final_path.exists()
        return

    real_repair = (
        run_directory_module._repair_manifested_active_resume_owner
    )

    def repair_while_attempt_lease_is_held(*args, **kwargs):  # type: ignore[no-untyped-def]
        with pytest.raises(
            RunLifecycleError,
            match="execution lease is already held",
        ):
            run_directory_module._acquire_attempt_execution_lease(
                root,
                reserved.run_id,
            )
        return real_repair(*args, **kwargs)

    monkeypatch.setattr(
        run_directory_module,
        "_repair_manifested_active_resume_owner",
        repair_while_attempt_lease_is_held,
    )
    recovered = run_directory_module.recover_terminal_run(
        root,
        reserved.run_id,
        plan=plan_identity,
        backend=backend,
    )
    assert recovered.run_path == reserved.final_path
    closed = json.loads(
        (recovered.run_path / "resume-owner.json").read_text("utf-8")
    )
    assert closed["state"] == "terminal_closed"
    assert closed["lineage_sha256"] == event.lineage_sha256
    assert closed["terminal_manifest_sha256"] == recovered.manifest_sha256
    assert validate_run_manifest(
        recovered.run_path / "run-manifest.json",
        expected=recovered,
    ) == recovered


@pytest.mark.parametrize("resume_before_terminal", (False, True))
def test_terminal_rename_recovery_reconstructs_only_from_durable_disk_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_before_terminal: bool,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    initial = reserve_run(
        root,
        "attempt-process-loss",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    reserved = initial
    if resume_before_terminal:
        run_directory_module.release_run_execution_lease(initial)
        event = ResumeLineageEvent.create(
            parent_checkpoint=_terminal_checkpoint(plan_identity.plan),
            environment_sha256=_sha("6"),
            cursor_sha256=_sha("7"),
            reason="explicit process-restart fixture",
            resumed_utc="2026-07-28T01:30:00Z",
        )
        reserved = reserve_run(
            root,
            initial.run_id,
            run_role="confirmation",
            started_utc=None,
            plan=plan_identity,
            mode="resume",
            resume_lineage=event,
            backend=backend,
        )
        run_directory_module.consume_resume_execution_retry(
            reserved,
            backend=backend,
        )
    records = _write_run_artifacts(backend, reserved.inprogress_path)
    checkpoint, checkpoint_record = _write_terminal_checkpoint(
        backend,
        reserved.inprogress_path,
        plan_identity.plan,
    )

    def process_loss_before_rename(_source: Path, _destination: Path) -> None:
        raise OSError("simulated process loss before terminal rename")

    real_rename = run_directory_module._rename_directory_no_replace
    monkeypatch.setattr(
        run_directory_module,
        "_rename_directory_no_replace",
        process_loss_before_rename,
    )
    with pytest.raises(RunLifecycleError, match="rename"):
        finalize_run(
            reserved,
            disposition="success",
            checkpoints=(checkpoint,),
            checkpoint_artifact_records=(checkpoint_record,),
            artifact_records=records,
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=12.5,
            failure_record_sha256=None,
            backend=backend,
        )
    monkeypatch.setattr(
        run_directory_module,
        "_rename_directory_no_replace",
        real_rename,
    )

    del reserved
    del initial
    recovered = run_directory_module.recover_terminal_run(
        root,
        "attempt-process-loss",
        plan=plan_identity,
    )

    assert recovered.run_path == root / "runs" / "attempt-process-loss"
    assert not (
        root / "runs" / ".inprogress" / "attempt-process-loss"
    ).exists()
    assert validate_run_manifest(
        recovered.run_path / "run-manifest.json",
        expected=recovered,
    ) == recovered


def test_final_index_requires_exact_ordered_tuning_inventory_with_failures(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    plan = plan_identity.plan
    tuning = tuple(
        _finalize_tuning(
            root=root,
            plan_identity=plan_identity,
            backend=backend,
            ordinal=ordinal,
            tuning_attempt_key=key,
            disposition="failure" if ordinal == 0 else "success",
        )
        for ordinal, key in enumerate(plan.tuning_attempt_keys)
    )
    confirmations = tuple(
        _finalize_confirmation(
            root=root,
            plan_identity=plan_identity,
            backend=backend,
            ordinal=ordinal,
            logical_key=logical_key,
        )
        for ordinal, logical_key in enumerate(plan.terminal_checkpoint_keys)
    )
    extra_tuning = _finalize_tuning(
        root=root,
        plan_identity=plan_identity,
        backend=backend,
        ordinal=len(tuning),
        tuning_attempt_key=plan.tuning_attempt_keys[0],
    )
    result_table_payload = b'{"rows":[]}'
    backend.create_exclusive(
        root / "result-table.json",
        result_table_payload,
    )
    group_records = (
        _integrity("result-table.json", result_table_payload),
    )

    invalid = (
        tuning[1:] + confirmations,
        (tuning[1], tuning[0], *tuning[2:], *confirmations),
        (tuning[0], tuning[0], *tuning[1:], *confirmations),
        (*tuning, extra_tuning, *confirmations),
    )
    for manifests in invalid:
        with pytest.raises(
            RunLifecycleError,
            match="tuning attempt inventory|run IDs",
        ):
            publish_experiment_index(
                root,
                plan=plan_identity,
                run_manifests=manifests,
                stage="final",
                artifact_records=group_records,
                backend=backend,
            )

    index = publish_experiment_index(
        root,
        plan=plan_identity,
        run_manifests=(*tuning, *confirmations),
        stage="final",
        artifact_records=group_records,
        backend=backend,
    )
    assert validate_experiment_index(index.index_path, expected=index) == index

    original = json.loads(index.index_path.read_text(encoding="utf-8"))
    mutations = []
    missing = json.loads(json.dumps(original))
    del missing["runs"][0]
    mutations.append(missing)
    reordered = json.loads(json.dumps(original))
    reordered["runs"][0], reordered["runs"][1] = (
        reordered["runs"][1],
        reordered["runs"][0],
    )
    mutations.append(reordered)
    duplicated = json.loads(json.dumps(original))
    duplicated["runs"].insert(0, duplicated["runs"][0])
    mutations.append(duplicated)
    extra = json.loads(json.dumps(original))
    extra_entry = {
        "run_id": extra_tuning.run_id,
        "run_role": extra_tuning.run_role,
        "disposition": extra_tuning.disposition,
        "manifest_sha256": extra_tuning.manifest_sha256,
        "manifest_identity_sha256": extra_tuning.identity_sha256,
        "relative_manifest_path": (
            f"runs/{extra_tuning.run_id}/run-manifest.json"
        ),
    }
    extra["runs"].insert(len(tuning), extra_entry)
    mutations.append(extra)
    for document in mutations:
        body = dict(document)
        body.pop("index_sha256")
        document["index_sha256"] = owned_sha256(
            "vfe4.wt103.experiment-index.v1",
            body,
        )
        index.index_path.write_bytes(canonical_json_bytes_generic(document))
        with pytest.raises(
            RunLifecycleError,
            match="tuning attempt inventory|run IDs|manifest hashes|manifest paths",
        ):
            validate_experiment_index(index.index_path)


def test_failed_run_requires_verified_failure_artifact_and_chain_head(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    missing = reserve_run(
        root,
        "failure-missing",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    with pytest.raises(RunLifecycleError, match="failures.jsonl"):
        finalize_run(
            missing,
            disposition="failure",
            checkpoints=(),
            checkpoint_artifact_records=(),
            artifact_records=(),
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=1.0,
            failure_record_sha256=_sha("6"),
            backend=backend,
        )

    wrong = reserve_run(
        root,
        "failure-wrong-head",
        run_role="confirmation",
        started_utc="2026-07-28T01:00:00Z",
        plan=plan_identity,
        backend=backend,
    )
    payload, _head = _failure_artifact(run_id=wrong.run_id)
    backend.create_exclusive(
        wrong.inprogress_path / "failures.jsonl",
        payload,
    )
    with pytest.raises(RunLifecycleError, match="failure ledger head"):
        finalize_run(
            wrong,
            disposition="failure",
            checkpoints=(),
            checkpoint_artifact_records=(),
            artifact_records=(_integrity("failures.jsonl", payload),),
            environment_sha256=_sha("4"),
            provenance_sha256=_sha("5"),
            ended_utc="2026-07-28T02:00:00Z",
            monotonic_duration_seconds=1.0,
            failure_record_sha256=_sha("6"),
            backend=backend,
        )


def test_manifest_revalidation_reenforces_terminal_invariants(
    tmp_path: Path,
) -> None:
    backend = _FilesystemBackend()
    root = tmp_path / "experiment"
    plan_identity = publish_experiment_plan(root, _plan(), backend=backend)
    manifest = _finalize_confirmation(
        root=root,
        plan_identity=plan_identity,
        backend=backend,
        ordinal=0,
        logical_key=plan_identity.plan.terminal_checkpoint_keys[0],
    )
    path = manifest.run_path / "run-manifest.json"
    original = path.read_bytes()

    mutators = (
        lambda document: document.__setitem__(
            "schema_version",
            "wt103-run-manifest-v2",
        ),
        lambda document: document.__setitem__("resume_count", 1),
        lambda document: document.__setitem__("run_role", "tuning"),
        lambda document: document.__setitem__(
            "artifact_records",
            document["artifact_records"][:-1],
        ),
        lambda document: document.__setitem__(
            "checkpoint_artifact_records",
            (),
        ),
    )
    for mutate in mutators:
        document = json.loads(original.decode("utf-8"))
        mutate(document)
        body = dict(document)
        body.pop("manifest_sha256")
        document["manifest_sha256"] = owned_sha256(
            "vfe4.wt103.run-manifest.v1",
            body,
        )
        path.write_bytes(canonical_json_bytes_generic(document))
        with pytest.raises(RunLifecycleError):
            validate_run_manifest(path)
    path.write_bytes(original)
    assert validate_run_manifest(path, expected=manifest) == manifest


def test_environment_lock_and_git_provenance_are_exact_and_dirty_aware() -> None:
    torch_distribution = DistributionIdentity(
        name="torch",
        version="2.10.0.dev20251210+cu128",
        record_sha256=_sha("a"),
    )
    lock_manifest = LockInputManifest.create(
        writer_code_sha256=_sha("b"),
        target_python_version="3.13",
        requirements=(
            LockRequirement(
                name="torch",
                version="2.10.0.dev20251210+cu128",
                environment_marker='python_version >= "3.10"',
                artifact_filename="torch-fixture.whl",
                artifact_url="https://example.invalid/torch-fixture.whl",
                artifact_size_bytes=123,
                artifact_sha256s=(_sha("c"),),
                expected_installed_record_sha256=(
                    torch_distribution.record_sha256
                ),
                task13_obligation=None,
            ),
        ),
    )
    lock_bytes = render_dependency_lock(lock_manifest)
    dependency = DependencyLockIdentity.capture(
        lock_relative_path="requirements-wt103.lock",
        lock_bytes=lock_bytes,
        expected_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        lock_input_manifest=lock_manifest,
        installed_distributions=(torch_distribution,),
    )
    assert tuple(
        (item.name, item.version)
        for item in dependency.locked_distributions
    ) == (("torch", "2.10.0.dev20251210+cu128"),)
    assert dependency.lock_input_manifest_sha256 == (
        lock_manifest.manifest_sha256
    )
    environment = capture_environment(
        EnvironmentObservation(
            captured_utc="2026-07-28T04:00:00Z",
            device_work_started=False,
            python_version="3.13.5",
            pytorch_version="2.10.0.dev20251210+cu128",
            cuda_runtime_version="12.8",
            cudnn_version="91002",
            driver_version="576.80",
            os_name="nt",
            platform_system="Windows",
            platform_release="11",
            cpu_name="fixture-cpu",
            logical_cpu_count=8,
            physical_ram_bytes=64 * 1024**3,
            gpu_names=("fixture-gpu",),
            gpu_device_uuids=("GPU-fixture-0001",),
            gpu_total_bytes=(32 * 1024**3,),
            compute_capabilities=("12.0",),
            blas_identity_sha256=_sha("b"),
            thread_settings_sha256=_sha("c"),
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            locale_name="en_US",
            timezone_name="America/Chicago",
        ),
        dependency_lock=dependency,
    )
    clean_git = capture_git_identity(
        git_head_value="1" * 40,
        dirty_digest_value=_sha("2"),
        status_porcelain=b"",
    )
    dirty_git = capture_git_identity(
        git_head_value="1" * 40,
        dirty_digest_value=_sha("3"),
        status_porcelain=b" M vfe4/example.py\0",
    )
    data_record = _integrity("data/source-manifest.json", b"source")
    evidence_record = _integrity("evidence/h8.json", b"h8")
    inventory_record = _integrity("inventory/endpoints.json", b"inventory")
    provenance = build_training_provenance(
        git_identity=clean_git,
        environment=environment,
        source_record_sha256=_sha("4"),
        tokenizer_spec_sha256=_sha("5"),
        token_cache_set_sha256=_sha("6"),
        schedule_set_sha256=_sha("7"),
        config_sha256=_sha("8"),
        objective_sha256=_sha("9"),
        factory_set_sha256=_sha("a"),
        endpoint_inventory_sha256=_sha("b"),
        data_integrity=data_record,
        evidence_integrity=(evidence_record,),
        inventory_integrity=inventory_record,
        parent_checkpoint=None,
    )

    assert dependency.status is GateStatus.PASS
    assert environment.captured_before_device_work is True
    assert clean_git.is_clean is True
    assert dirty_git.is_clean is False
    assert dirty_git.status_porcelain_sha256 != clean_git.status_porcelain_sha256
    assert provenance.git_identity_sha256 == clean_git.identity_sha256
    assert provenance.artifact_integrity_sha256s == (
        data_record.record_sha256,
        evidence_record.record_sha256,
        inventory_record.record_sha256,
    )

    mismatched_record = DistributionIdentity(
        name="torch",
        version="2.10.0.dev20251210+cu128",
        record_sha256=_sha("f"),
    )
    mismatch = DependencyLockIdentity.capture(
        lock_relative_path="requirements-wt103.lock",
        lock_bytes=lock_bytes,
        expected_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        lock_input_manifest=lock_manifest,
        installed_distributions=(mismatched_record,),
    )
    assert mismatch.status is GateStatus.INCONCLUSIVE
    assert mismatch.obligations == ("installed_distributions_do_not_match_lock",)
    with pytest.raises(ValueError, match="manifest"):
        replace(
            dependency,
            lock_input_manifest_sha256=_sha("0"),
        )
    with pytest.raises(ValueError, match="before device work"):
        capture_environment(
            replace(
                EnvironmentObservation(
                    captured_utc="2026-07-28T04:00:00Z",
                    device_work_started=False,
                    python_version="3.13.5",
                    pytorch_version="2.10",
                    cuda_runtime_version="12.8",
                    cudnn_version="91002",
                    driver_version="576.80",
                    os_name="nt",
                    platform_system="Windows",
                    platform_release="11",
                    cpu_name="fixture-cpu",
                    logical_cpu_count=8,
                    physical_ram_bytes=1024,
                    gpu_names=("fixture-gpu",),
                    gpu_device_uuids=("GPU-fixture-0001",),
                    gpu_total_bytes=(1024,),
                    compute_capabilities=("12.0",),
                    blas_identity_sha256=_sha("b"),
                    thread_settings_sha256=_sha("c"),
                    deterministic_algorithms=True,
                    cudnn_benchmark=False,
                    locale_name="en_US",
                    timezone_name="America/Chicago",
                ),
                device_work_started=True,
            ),
            dependency_lock=dependency,
        )


def _resource_profile() -> ResourceProfile:
    return ResourceProfile(
        maximum_gpu_hours=720.0,
        maximum_wall_hours=840.0,
        maximum_energy_kwh=500.0,
        forecast_headroom_factor=1.25,
        maximum_device_fraction=0.85,
        power_sample_interval_ms=100,
    )


def _execution_identity(
    *,
    config_sha256: str = "e" * 64,
    environment_sha256: str = "5" * 64,
) -> TrainingExecutionIdentity:
    return TrainingExecutionIdentity.create(
        git_identity_sha256=_sha("1"),
        git_head="1" * 40,
        dirty_digest=_sha("2"),
        config_sha256=config_sha256,
        profile_sha256=_sha("3"),
        factory_set_sha256=_sha("4"),
        environment_sha256=environment_sha256,
    )


def _capacity_environment():
    installed = DistributionIdentity(
        name="fixture",
        version="1.0",
        record_sha256=_sha("6"),
    )
    manifest = LockInputManifest.create(
        writer_code_sha256=_sha("7"),
        target_python_version="3.13",
        requirements=(
            LockRequirement(
                name="fixture",
                version="1.0",
                environment_marker='python_version >= "3.10"',
                artifact_filename="fixture-1.0.whl",
                artifact_url="https://example.invalid/fixture-1.0.whl",
                artifact_size_bytes=123,
                artifact_sha256s=(_sha("8"),),
                expected_installed_record_sha256=installed.record_sha256,
                task13_obligation=None,
            ),
        ),
    )
    lock_bytes = render_dependency_lock(manifest)
    dependency = DependencyLockIdentity.capture(
        lock_relative_path="requirements-wt103.lock",
        lock_bytes=lock_bytes,
        expected_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        lock_input_manifest=manifest,
        installed_distributions=(installed,),
    )
    return capture_environment(
        EnvironmentObservation(
            captured_utc="2026-07-28T04:00:00Z",
            device_work_started=False,
            python_version="3.13.5",
            pytorch_version="fixture",
            cuda_runtime_version="12.8",
            cudnn_version="fixture",
            driver_version="fixture",
            os_name="nt",
            platform_system="Windows",
            platform_release="11",
            cpu_name="fixture-cpu",
            logical_cpu_count=8,
            physical_ram_bytes=100_000,
            gpu_names=("fixture-gpu",),
            gpu_device_uuids=("GPU-fixture-0001",),
            gpu_total_bytes=(10_000,),
            compute_capabilities=("12.0",),
            blas_identity_sha256=_sha("9"),
            thread_settings_sha256=_sha("a"),
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            locale_name="en_US",
            timezone_name="America/Chicago",
        ),
        dependency_lock=dependency,
    )


def _passing_allocations(
    inventory: EndpointInventory,
    execution_identity: TrainingExecutionIdentity,
    environment,
) -> tuple[AllocationObservation, ...]:
    return tuple(
        AllocationObservation.shape_identical_for_arm(
            arm,
            execution_identity=execution_identity,
            device_ordinal=0,
            device_uuid="GPU-fixture-0001",
            physical_device_bytes=10_000,
            peak_device_allocated_bytes=8_000,
            peak_device_reserved_bytes=8_500,
            host_available_bytes=5_000,
            checkpoint_duplicate_bytes=2_000,
            disk_available_bytes=5_000,
        )
        for arm in inventory.arms
    )


def test_allocation_preflight_is_inventory_exact_independent_and_enforces_85_percent() -> None:
    inventory = _inventory()
    environment = _capacity_environment()
    execution_identity = _execution_identity(
        environment_sha256=environment.environment_sha256
    )
    passing = _passing_allocations(
        inventory,
        execution_identity,
        environment,
    )
    result = run_allocation_preflight(
        endpoint_inventory=inventory,
        observations=passing,
        execution_identity=execution_identity,
        environment=environment,
        maximum_device_fraction=0.85,
        h8_allocation_evidence=None,
    )
    assert result.status is GateStatus.PASS
    assert result.maximum_peak_reserved_fraction == 0.85
    assert "recognition_proposal" not in passing[0].path_events
    assert "recognition_proposal" in passing[1].path_events

    too_large = AllocationObservation.shape_identical_for_arm(
        inventory.arms[2],
        execution_identity=execution_identity,
        device_ordinal=0,
        device_uuid="GPU-fixture-0001",
        physical_device_bytes=10_000,
        peak_device_allocated_bytes=8_000,
        peak_device_reserved_bytes=8_501,
        host_available_bytes=5_000,
        checkpoint_duplicate_bytes=2_000,
        disk_available_bytes=5_000,
    )
    failed = run_allocation_preflight(
        endpoint_inventory=inventory,
        observations=(*passing[:2], too_large, *passing[3:]),
        execution_identity=execution_identity,
        environment=environment,
        maximum_device_fraction=0.85,
        h8_allocation_evidence=None,
    )
    assert failed.status is GateStatus.FAIL
    assert failed.obligations == (
        f"device_reserved_over_85_percent:{too_large.arm_id}",
    )
    with pytest.raises(ValueError, match="H8"):
        run_allocation_preflight(
            endpoint_inventory=inventory,
            observations=passing,
            execution_identity=execution_identity,
            environment=environment,
            maximum_device_fraction=0.85,
            h8_allocation_evidence=object(),
        )
    mismatched = _passing_allocations(
        inventory,
        _execution_identity(
            config_sha256=_sha("f"),
            environment_sha256=environment.environment_sha256,
        ),
        environment,
    )
    with pytest.raises(ValueError, match="execution identity"):
        run_allocation_preflight(
            endpoint_inventory=inventory,
            observations=mismatched,
            execution_identity=execution_identity,
            environment=environment,
            maximum_device_fraction=0.85,
            h8_allocation_evidence=None,
        )
    forged_denominator = replace(
        passing[0],
        physical_device_bytes=100_000,
        observation_sha256=owned_sha256(
            "vfe4.wt103.allocation-observation.v2",
            {
                name: (
                    100_000
                    if name == "physical_device_bytes"
                    else getattr(passing[0], name)
                )
                for name in tuple(passing[0].__dataclass_fields__)[:-1]
            },
        ),
    )
    with pytest.raises(ValueError, match="captured device"):
        run_allocation_preflight(
            endpoint_inventory=inventory,
            observations=(
                forged_denominator,
                *passing[1:],
            ),
            execution_identity=execution_identity,
            environment=environment,
            maximum_device_fraction=0.85,
            h8_allocation_evidence=None,
        )


def test_resource_forecast_uses_inventory_work_exact_disk_formula_and_hard_ceilings() -> None:
    inventory = _inventory()
    execution_identity = _execution_identity()
    workload = ResourceWorkload(
        train_batches_per_pass=400,
        validation_batches_per_full_evaluation=20,
        test_batches_per_full_evaluation=30,
        preparation_source_work_units=1,
        preparation_tokenizer_work_units=103,
        preparation_window_work_units=101,
    )
    specifications = required_resource_components(inventory, workload)
    by_id = {item.component_id: item for item in specifications}
    assert by_id[
        "tuning/train/WT103-A0-AR-v1"
    ].work_units == 12 * 100
    assert by_id[
        "confirmation/train/WT103-A0-AR-v1"
    ].work_units == 8 * 2 * 400
    assert by_id[
        "validation/WT103-A0-AR-v1"
    ].work_units == (12 + 8 * 2 * 20) * 20
    assert len(
        [
            item
            for item in specifications
            if item.component_id.startswith("figure/")
        ]
    ) == 8
    checkpoint_components = tuple(
        item
        for item in specifications
        if item.component_id.startswith("checkpoint/")
    )
    assert len(checkpoint_components) == 1
    assert checkpoint_components[0].work_units == 40 * (2 * 20 + 1)

    power = PowerProviderIdentity.create(
        provider_kind="nvml",
        provider_version="fixture-1",
        provider_executable_sha256=_sha("d"),
        sample_interval_ms=100,
        reported_power_limit_watts=600.0,
    )
    benchmarks = tuple(
        ComponentBenchmark.observed_for(
            item,
            execution_identity=execution_identity,
            minimum_throughput_per_second=1_000_000.0,
            maximum_duration_seconds=0.001,
            maximum_board_power_watts=725.0 if item.uses_gpu else 0.0,
            power_provider=power if item.uses_gpu else None,
        )
        for item in specifications
    )
    disk = DiskByteForecast.create(
        archive_staging_bytes=100,
        extracted_member_bytes=200,
        int32_token_cache_bytes=300,
        schedule_bytes=400,
        retained_checkpoint_bytes=500,
        jsonl_csv_bytes=600,
        test_record_bytes=700,
        figure_bytes=800,
    )
    expected_payload = 3_600
    assert disk.payload_bytes == expected_payload
    assert disk.temporary_write_overhead_bytes == 900
    assert disk.forecast_bytes == 4_500
    assert disk.required_available_bytes == 2 * 4_500 + 10 * 1024**3

    forecast = forecast_resources(
        endpoint_inventory=inventory,
        workload=workload,
        component_benchmarks=benchmarks,
        execution_identity=execution_identity,
        disk_forecast=disk,
        available_disk_bytes=disk.required_available_bytes,
        resource_profile=_resource_profile(),
        power_provider=power,
    )
    assert forecast.status is GateStatus.PASS
    assert forecast.forecast_gpu_hours == pytest.approx(
        1.25 * forecast.raw_gpu_hours
    )
    assert forecast.forecast_wall_hours == pytest.approx(
        1.25 * forecast.raw_wall_hours
    )
    assert forecast.forecast_energy_kwh == pytest.approx(
        forecast.forecast_gpu_hours * 725.0 / 1000.0
    )
    assert forecast.maximum_observed_board_power_watts == 725.0
    assert forecast.reported_power_limit_watts == 600.0
    assert forecast.conservative_power_watts == 725.0

    missing_power_benchmarks = tuple(
        ComponentBenchmark.observed_for(
            item,
            execution_identity=execution_identity,
            minimum_throughput_per_second=1_000_000.0,
            maximum_duration_seconds=0.001,
            maximum_board_power_watts=(
                None if item.uses_gpu else 0.0
            ),
            power_provider=None,
        )
        for item in specifications
    )
    missing_power = forecast_resources(
        endpoint_inventory=inventory,
        workload=workload,
        component_benchmarks=missing_power_benchmarks,
        execution_identity=execution_identity,
        disk_forecast=disk,
        available_disk_bytes=disk.required_available_bytes,
        resource_profile=_resource_profile(),
        power_provider=None,
    )
    assert missing_power.status is GateStatus.INCONCLUSIVE
    assert "power_provider_missing" in missing_power.obligations
    insufficient_disk = forecast_resources(
        endpoint_inventory=inventory,
        workload=workload,
        component_benchmarks=benchmarks,
        execution_identity=execution_identity,
        disk_forecast=disk,
        available_disk_bytes=disk.required_available_bytes - 1,
        resource_profile=_resource_profile(),
        power_provider=power,
    )
    assert insufficient_disk.status is GateStatus.FAIL
    assert "disk_headroom_insufficient" in insufficient_disk.obligations
    mismatched_benchmarks = tuple(
        ComponentBenchmark.observed_for(
            item,
            execution_identity=_execution_identity(
                config_sha256=_sha("f"),
            ),
            minimum_throughput_per_second=1_000_000.0,
            maximum_duration_seconds=0.001,
            maximum_board_power_watts=500.0 if item.uses_gpu else 0.0,
            power_provider=power if item.uses_gpu else None,
        )
        for item in specifications
    )
    with pytest.raises(ValueError, match="execution identity"):
        forecast_resources(
            endpoint_inventory=inventory,
            workload=workload,
            component_benchmarks=mismatched_benchmarks,
            execution_identity=execution_identity,
            disk_forecast=disk,
            available_disk_bytes=disk.required_available_bytes,
            resource_profile=_resource_profile(),
            power_provider=power,
        )


def _power_authority_forecast(
    *,
    provider: PowerProviderIdentity,
    maximum_observed_board_power_watts: float,
    conservative_power_watts: float,
) -> ResourceForecast:
    component = ComponentForecast(
        component_id="fixture-gpu-component",
        component_spec_sha256=_sha("a"),
        benchmark_sha256=_sha("b"),
        predicted_seconds=3_600.0,
        predicted_gpu_seconds=3_600.0,
    )
    payload = {
        "schema_version": "wt103-resource-forecast-v1",
        "endpoint_inventory_sha256": _sha("c"),
        "execution_identity": _execution_identity(),
        "component_forecasts": (component,),
        "component_benchmark_sha256s": (component.benchmark_sha256,),
        "disk_forecast_sha256": _sha("e"),
        "available_disk_bytes": 1,
        "raw_gpu_hours": 1.0,
        "raw_wall_hours": 1.0,
        "raw_energy_kwh": 0.725,
        "forecast_gpu_hours": 1.25,
        "forecast_wall_hours": 1.25,
        "forecast_energy_kwh": 0.90625,
        "maximum_gpu_hours": 720.0,
        "maximum_wall_hours": 840.0,
        "maximum_energy_kwh": 500.0,
        "forecast_headroom_factor": 1.25,
        "power_provider_identity_sha256": provider.identity_sha256,
        "maximum_observed_board_power_watts": (
            maximum_observed_board_power_watts
        ),
        "reported_power_limit_watts": (
            provider.reported_power_limit_watts
        ),
        "conservative_power_watts": conservative_power_watts,
        "status": GateStatus.PASS,
        "obligations": (),
    }
    return ResourceForecast(
        **payload,
        forecast_sha256=owned_sha256(
            "vfe4.wt103.resource-forecast.v1",
            payload,
        ),
    )


def test_kill_safe_debit_uses_frozen_peak_above_reported_power_limit() -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    provider = PowerProviderIdentity.create(
        provider_kind="nvml",
        provider_version="fixture-1",
        provider_executable_sha256=_sha("d"),
        sample_interval_ms=100,
        reported_power_limit_watts=600.0,
    )
    forecast = _power_authority_forecast(
        provider=provider,
        maximum_observed_board_power_watts=725.0,
        conservative_power_watts=725.0,
    )
    bound_watts = (
        production_attempt._frozen_readiness_conservative_power_watts(
            resource_forecast=forecast,
            live_power_provider=provider,
        )
    )
    training = resolve_training_config(default_training_config_mapping())
    usage = production_attempt._conservative_limit_usage(
        attempt=production_attempt._attempt_inventory(training, None)[0],
        measurement_kind="prepaid_crash_tail_reserve",
        interval_ordinal=0,
        interval_started_ns=0,
        interval_ended_ns=60_000_000_000,
        power_provider_identity_sha256=provider.identity_sha256,
        conservative_power_watts=bound_watts,
    )

    assert bound_watts == pytest.approx(725.0)
    assert usage.sampled_energy_kwh == pytest.approx(
        725.0 * 60.0 / 3_600_000.0
    )
    exact_energy = (
        Fraction.from_float(bound_watts)
        * Fraction(60_000_000_000, 1_000_000_000)
        / 3_600_000
    )
    assert Fraction.from_float(usage.sampled_energy_kwh) >= exact_energy


def test_resource_forecast_rejects_bound_below_measured_peak() -> None:
    provider = PowerProviderIdentity.create(
        provider_kind="nvml",
        provider_version="fixture-1",
        provider_executable_sha256=_sha("d"),
        sample_interval_ms=100,
        reported_power_limit_watts=600.0,
    )

    with pytest.raises(ValueError, match="conservative power"):
        _power_authority_forecast(
            provider=provider,
            maximum_observed_board_power_watts=725.0,
            conservative_power_watts=math.nextafter(725.0, 0.0),
        )


def test_usage_ledger_debits_immutable_plan_and_blocks_underfunded_test_reservation() -> None:
    profile = _resource_profile()
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=_sha("e"),
        resource_profile=profile,
    )
    event = ResourceUsageEvent.create(
        attempt_id="attempt-0001",
        segment_ordinal=0,
        device_seconds=3_600.0,
        wall_seconds=7_200.0,
        sampled_energy_kwh=2.5,
        usage_evidence_sha256=_sha("f"),
    )
    debited = ledger.append(event)

    assert ledger.events == ()
    assert debited.used_gpu_hours == 1.0
    assert debited.remaining_gpu_hours == 719.0
    assert debited.remaining_wall_hours == 838.0
    assert debited.remaining_energy_kwh == 497.5
    assert debited.experiment_plan_sha256 == ledger.experiment_plan_sha256

    authorized = authorize_test_reservation(
        ledger=debited,
        raw_test_gpu_hours=1.0,
        raw_test_wall_hours=2.0,
        raw_test_energy_kwh=3.0,
        raw_test_disk_bytes=100,
        available_disk_bytes=125,
    )
    assert authorized.status is GateStatus.PASS
    blocked = authorize_test_reservation(
        ledger=debited,
        raw_test_gpu_hours=600.0,
        raw_test_wall_hours=800.0,
        raw_test_energy_kwh=450.0,
        raw_test_disk_bytes=101,
        available_disk_bytes=125,
    )
    assert blocked.status is GateStatus.FAIL
    assert set(blocked.obligations) == {
        "remaining_gpu_hours_insufficient",
        "remaining_wall_hours_insufficient",
        "remaining_energy_kwh_insufficient",
        "remaining_disk_bytes_insufficient",
    }


def test_usage_ledger_adds_multiple_segments_for_one_resumed_attempt(
    tmp_path: Path,
) -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt._attempt_inventory(training, None)[0]
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=_sha("e"),
        resource_profile=training.profile.resources,
    )
    ledger_path = tmp_path / "resource-usage-ledger.json"
    backend = production_attempt._backend()
    first = production_attempt._AttemptResourceUsage(
        device_seconds=2.0,
        wall_seconds=3.0,
        sampled_energy_kwh=0.01,
        usage_evidence_sha256=_sha("1"),
    )
    second = production_attempt._AttemptResourceUsage(
        device_seconds=5.0,
        wall_seconds=7.0,
        sampled_energy_kwh=0.02,
        usage_evidence_sha256=_sha("2"),
    )

    after_crash = production_attempt._debit_resource_usage(
        ledger=ledger,
        attempt=attempt,
        usage=first,
        path=ledger_path,
        backend=backend,
    )
    after_resume = production_attempt._debit_resource_usage(
        ledger=after_crash,
        attempt=attempt,
        usage=second,
        path=ledger_path,
        backend=backend,
    )
    reopened = production_attempt._reopen_resource_usage_ledger(
        path=ledger_path,
        experiment_plan_sha256=ledger.experiment_plan_sha256,
    )

    assert after_resume.ledger_sha256 == reopened.ledger_sha256
    assert tuple(
        (event.attempt_id, event.segment_ordinal)
        for event in reopened.events
    ) == (
        (attempt.attempt_id, 0),
        (attempt.attempt_id, 1),
    )
    assert reopened.used_gpu_hours == pytest.approx(7.0 / 3_600.0)
    assert reopened.used_wall_hours == pytest.approx(10.0 / 3_600.0)
    assert reopened.used_energy_kwh == pytest.approx(0.03)


def test_resource_debit_checks_headroom_before_durable_publication(
    tmp_path: Path,
) -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt._attempt_inventory(training, None)[0]
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=_sha("e"),
        resource_profile=training.profile.resources,
    )
    over_ceiling = production_attempt._AttemptResourceUsage(
        device_seconds=600.0 * 3_600.0,
        wall_seconds=600.0 * 3_600.0,
        sampled_energy_kwh=0.01,
        usage_evidence_sha256=_sha("9"),
    )
    ledger_path = tmp_path / "resource-usage-ledger.json"

    with pytest.raises(
        production_attempt.ProductionOperationError,
        match="headroom exceeded",
    ):
        production_attempt._debit_resource_usage(
            ledger=ledger,
            attempt=attempt,
            usage=over_ceiling,
            path=ledger_path,
            backend=production_attempt._backend(),
        )

    assert not ledger_path.exists()


def test_power_sampler_refuses_to_execute_without_an_initial_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import NvidiaSmiPowerSampler

    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"fixture")
    sampler = NvidiaSmiPowerSampler(
        executable=executable,
        device_ordinal=0,
    )
    executed: list[bool] = []

    def fail_power(_self: object) -> float:
        raise RuntimeError("synthetic initial telemetry failure")

    monkeypatch.setattr(
        NvidiaSmiPowerSampler,
        "_read_power",
        fail_power,
    )

    with pytest.raises(RuntimeError, match="power sampling failed"):
        sampler.sample(lambda: executed.append(True))

    assert executed == []


def test_power_sampler_failure_carries_partial_samples_and_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import NvidiaSmiPowerSampler

    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"fixture")
    sampler = NvidiaSmiPowerSampler(
        executable=executable,
        device_ordinal=0,
    )
    second_read_started = threading.Event()
    read_count = 0

    def read_power(_self: object) -> float:
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            return 125.0
        second_read_started.set()
        raise RuntimeError("synthetic worker telemetry failure")

    monkeypatch.setattr(
        NvidiaSmiPowerSampler,
        "_read_power",
        read_power,
    )

    def operation() -> str:
        assert second_read_started.wait(timeout=1.0)
        return "scientific-result"

    with pytest.raises(RuntimeError) as caught:
        sampler.sample(operation)

    assert caught.value.__class__.__name__ == "PowerSampleOperationFailure"
    assert caught.value.operation_completed is True
    assert caught.value.operation_result == "scientific-result"
    assert caught.value.operation_error is None
    assert tuple(item.watts for item in caught.value.observations) == (125.0,)


def test_power_sampler_carries_operation_error_with_partial_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import (
        NvidiaSmiPowerSampler,
        PowerSampleOperationFailure,
    )

    executable = tmp_path / "nvidia-smi.exe"
    executable.write_bytes(b"fixture")
    sampler = NvidiaSmiPowerSampler(
        executable=executable,
        device_ordinal=0,
    )
    monkeypatch.setattr(
        NvidiaSmiPowerSampler,
        "_read_power",
        lambda _self: 75.0,
    )
    error = RuntimeError("synthetic scientific failure")

    def operation() -> None:
        raise error

    with pytest.raises(PowerSampleOperationFailure) as caught:
        sampler.sample(operation)

    assert caught.value.operation_completed is True
    assert caught.value.operation_result is None
    assert caught.value.operation_error is error
    assert caught.value.sampling_error is None
    assert tuple(item.watts for item in caught.value.observations) == (75.0,)


def test_failed_operation_retains_completed_resource_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import PowerObservation
    from vfe4.training import production_attempt

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 500.0

    class FakeSampler:
        sample_interval_seconds = 0.1

        def __post_init__(self) -> None:
            return None

        def sample(self, operation, *, on_observation=None):  # type: ignore[no-untyped-def]
            if on_observation is not None:
                on_observation(
                    PowerObservation(
                        watts=100.0,
                        monotonic_ns=1_100_000_000,
                    )
                )
                on_observation(
                    PowerObservation(
                        watts=120.0,
                        monotonic_ns=1_900_000_000,
                    )
                )
            return operation(), (100.0, 120.0)

    monkeypatch.setattr(
        production_attempt,
        "NvidiaSmiPowerSampler",
        FakeSampler,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "Event",
        FakeEvent,
    )
    clock = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(
        production_attempt.time,
        "perf_counter_ns",
        lambda: next(clock),
    )

    def fail() -> None:
        raise RuntimeError("synthetic interruption")

    with pytest.raises(
        production_attempt._ResourceMeasuredOperationFailure,
    ) as caught:
        production_attempt._measure_attempt_resource_usage(
            fail,
            sampler=FakeSampler(),
            power_provider_identity_sha256=_sha("3"),
            conservative_power_watts=600.0,
        )

    assert isinstance(caught.value.error, RuntimeError)
    assert caught.value.usage.device_seconds == 0.5
    assert caught.value.usage.wall_seconds >= 0.0
    assert caught.value.usage.sampled_energy_kwh >= 0.0


def test_sampler_failure_after_completed_operation_retains_partial_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import (
        PowerObservation,
        PowerSampleOperationFailure,
    )
    from vfe4.training import production_attempt

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 500.0

    class FakeSampler:
        sample_interval_seconds = 0.1

        def __post_init__(self) -> None:
            return None

        def sample(self, operation, *, on_observation=None):  # type: ignore[no-untyped-def]
            observation = PowerObservation(
                watts=125.0,
                monotonic_ns=1_100_000_000,
            )
            if on_observation is not None:
                on_observation(observation)
            result = operation()
            raise PowerSampleOperationFailure(
                sampling_error=RuntimeError(
                    "synthetic post-operation telemetry failure"
                ),
                observations=(observation,),
                operation_completed=True,
                operation_result=result,
                operation_error=None,
            )

    monkeypatch.setattr(
        production_attempt,
        "NvidiaSmiPowerSampler",
        FakeSampler,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "Event",
        FakeEvent,
    )
    clock = iter((1_000_000_000, 2_000_000_000))
    monkeypatch.setattr(
        production_attempt.time,
        "perf_counter_ns",
        lambda: next(clock),
    )

    with pytest.raises(
        production_attempt._ResourceMeasuredOperationFailure,
    ) as caught:
        production_attempt._measure_attempt_resource_usage(
            lambda: "scientific-result",
            sampler=FakeSampler(),
            power_provider_identity_sha256=_sha("3"),
            conservative_power_watts=600.0,
        )

    assert caught.value.operation_completed is True
    assert caught.value.operation_result == "scientific-result"
    assert isinstance(caught.value.error, PowerSampleOperationFailure)
    assert isinstance(caught.value.error.sampling_error, RuntimeError)
    assert "post-operation telemetry" in str(
        caught.value.error.sampling_error
    )
    assert caught.value.usage.device_seconds == 0.5
    assert caught.value.usage.wall_seconds == 1.0
    assert caught.value.usage.sampled_energy_kwh > 0.0


def test_long_measurement_retains_positive_sampled_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.live_environment import PowerObservation
    from vfe4.training import production_attempt

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 1_000.0

    class FakeSampler:
        sample_interval_seconds = 0.1

        def __post_init__(self) -> None:
            return None

        def sample(self, operation, *, on_observation=None):  # type: ignore[no-untyped-def]
            observations = (
                PowerObservation(
                    watts=100.0,
                    monotonic_ns=1_100_000_000,
                ),
                PowerObservation(
                    watts=120.0,
                    monotonic_ns=61_100_000_000,
                ),
                PowerObservation(
                    watts=110.0,
                    monotonic_ns=65_000_000_000,
                ),
            )
            assert on_observation is not None
            for observation in observations:
                on_observation(observation)
            return operation(), tuple(item.watts for item in observations)

    monkeypatch.setattr(
        production_attempt,
        "NvidiaSmiPowerSampler",
        FakeSampler,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        production_attempt.torch.cuda,
        "Event",
        FakeEvent,
    )
    clock = iter((1_000_000_000, 66_000_000_000))
    monkeypatch.setattr(
        production_attempt.time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    result, final = production_attempt._measure_attempt_resource_usage(
        lambda: "scientific-result",
        sampler=FakeSampler(),
        power_provider_identity_sha256=_sha("3"),
        conservative_power_watts=600.0,
    )

    assert result == "scientific-result"
    assert final.wall_seconds == 65.0
    assert final.device_seconds == 1.0
    assert final.sampled_energy_kwh > 0.0


def test_resource_heartbeat_debits_elapsed_time_at_bound_power_limit() -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt._attempt_inventory(training, None)[0]

    class FakeStop:
        def __init__(self) -> None:
            self.outcomes = iter((False, True))

        def wait(self, seconds: float) -> bool:
            assert (
                seconds
                == production_attempt._RESOURCE_USAGE_HEARTBEAT_SECONDS
            )
            return next(self.outcomes)

    debits: list[production_attempt._AttemptResourceUsage] = []
    production_attempt._run_resource_usage_heartbeat(
        stop=FakeStop(),
        attempt=attempt,
        power_provider_identity_sha256=_sha("5"),
        conservative_power_watts=600.0,
        started_ns=1_000_000_000,
        debit=debits.append,
        monotonic_ns=lambda: 31_000_000_000,
    )

    assert len(debits) == 1
    assert debits[0].device_seconds == 30.0
    assert debits[0].wall_seconds == 30.0
    assert debits[0].sampled_energy_kwh == pytest.approx(0.005)


def test_attempt_debits_measured_failure_before_propagating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt._attempt_inventory(training, None)[0]
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=_sha("e"),
        resource_profile=training.profile.resources,
    )
    usage = production_attempt._AttemptResourceUsage(
        device_seconds=4.0,
        wall_seconds=5.0,
        sampled_energy_kwh=0.04,
        usage_evidence_sha256=_sha("4"),
    )
    error = RuntimeError("synthetic measured crash")

    def fail_measurement(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise production_attempt._ResourceMeasuredOperationFailure(
            error=error,
            usage=usage,
        )

    monkeypatch.setattr(
        production_attempt,
        "_measure_attempt_resource_usage",
        fail_measurement,
    )
    ledger_path = tmp_path / "resource-usage-ledger.json"
    backend = production_attempt._backend()

    with pytest.raises(RuntimeError, match="synthetic measured crash"):
        production_attempt._execute_reserved_attempt_under_lease(
            attempt=attempt,
            training=training,
            source_lock=None,
            readiness=None,
            cache_root=tmp_path,
            plan=None,
            reserved=None,
            backend=backend,
            resume_active=False,
            sampler=None,
            power_provider_identity_sha256=_sha("5"),
            conservative_power_watts=600.0,
            ledger=ledger,
            ledger_path=ledger_path,
        )

    reopened = production_attempt._reopen_resource_usage_ledger(
        path=ledger_path,
        experiment_plan_sha256=ledger.experiment_plan_sha256,
    )
    assert len(reopened.events) == 2
    assert reopened.events[0].attempt_id == attempt.attempt_id
    assert reopened.events[0].segment_ordinal == 0
    assert reopened.events[1].segment_ordinal == 1
    assert reopened.used_gpu_hours == pytest.approx(64.0 / 3_600.0)
    assert reopened.used_wall_hours == pytest.approx(65.0 / 3_600.0)
    assert reopened.used_energy_kwh == pytest.approx(0.05)


def test_attempt_predebits_crash_tail_before_measurement_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.config import (
        default_training_config_mapping,
        resolve_training_config,
    )
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    attempt = production_attempt._attempt_inventory(training, None)[0]
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=_sha("e"),
        resource_profile=training.profile.resources,
    )
    final = production_attempt._AttemptResourceUsage(
        device_seconds=4.0,
        wall_seconds=5.0,
        sampled_energy_kwh=0.001,
        usage_evidence_sha256=_sha("7"),
    )
    ledger_path = tmp_path / "resource-usage-ledger.json"
    backend = production_attempt._backend()

    def fail_after_reserve(
        *_args,
        **_kwargs,
    ):  # type: ignore[no-untyped-def]
        reserved = production_attempt._reopen_resource_usage_ledger(
            path=ledger_path,
            experiment_plan_sha256=ledger.experiment_plan_sha256,
        )
        assert len(reserved.events) == 1
        assert reserved.events[0].wall_seconds == 60.0
        assert reserved.events[0].device_seconds == 60.0
        assert reserved.events[0].sampled_energy_kwh == pytest.approx(0.01)
        raise production_attempt._ResourceMeasuredOperationFailure(
            error=RuntimeError("synthetic post-reserve failure"),
            usage=final,
        )

    monkeypatch.setattr(
        production_attempt,
        "_measure_attempt_resource_usage",
        fail_after_reserve,
    )

    with pytest.raises(RuntimeError, match="post-reserve failure"):
        production_attempt._execute_reserved_attempt_under_lease(
            attempt=attempt,
            training=training,
            source_lock=None,
            readiness=None,
            cache_root=tmp_path,
            plan=None,
            reserved=None,
            backend=backend,
            resume_active=False,
            sampler=None,
            power_provider_identity_sha256=_sha("5"),
            conservative_power_watts=600.0,
            ledger=ledger,
            ledger_path=ledger_path,
        )

    reopened = production_attempt._reopen_resource_usage_ledger(
        path=ledger_path,
        experiment_plan_sha256=ledger.experiment_plan_sha256,
    )
    assert tuple(event.segment_ordinal for event in reopened.events) == (0, 1)
    assert reopened.used_gpu_hours == pytest.approx(64.0 / 3_600.0)
    assert reopened.used_wall_hours == pytest.approx(65.0 / 3_600.0)
    assert reopened.used_energy_kwh == pytest.approx(0.011)
