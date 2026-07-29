from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vfe4.config import (
    default_training_config_mapping,
    resolve_training_config,
)
from vfe4.recording.metrics import UpdateControlRecord
from vfe4.training.engine import ProposalEvidence, StepResult
from vfe4.training.production_observability import (
    MemoryObservation,
    PhaseTimingObservation,
)
from vfe4.types import (
    VocabularyIdentity,
    WT103UpdateRecord,
)
from vfe4.types.training import owned_sha256


def test_production_attempt_applies_and_revalidates_exact_precision_runtime_policy_before_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production_attempt

    class _FakeCuda:
        def __init__(self) -> None:
            self.initialized_checks = 0

        def is_initialized(self) -> bool:
            self.initialized_checks += 1
            return False

    class _FakeTorch:
        def __init__(self) -> None:
            self.cuda = _FakeCuda()
            self.backends = SimpleNamespace(
                cudnn=SimpleNamespace(
                    benchmark=True,
                    deterministic=False,
                    allow_tf32=True,
                ),
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(
                        allow_tf32=True,
                        allow_fp16_reduced_precision_reduction=True,
                    )
                ),
            )
            self.deterministic_algorithms = False

        def use_deterministic_algorithms(
            self,
            enabled: bool,
        ) -> None:
            self.deterministic_algorithms = enabled

        def are_deterministic_algorithms_enabled(self) -> bool:
            return self.deterministic_algorithms

    runtime = _FakeTorch()
    environment: dict[str, str] = {}
    training = resolve_training_config(default_training_config_mapping())

    evidence = production_attempt._apply_frozen_precision_runtime_policy(
        training=training,
        torch_runtime=runtime,
        environment=environment,
    )

    assert runtime.cuda.initialized_checks == 1
    assert environment == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert runtime.deterministic_algorithms is True
    assert runtime.backends.cudnn.benchmark is False
    assert runtime.backends.cudnn.deterministic is True
    assert runtime.backends.cuda.matmul.allow_tf32 is False
    assert runtime.backends.cudnn.allow_tf32 is False
    assert (
        runtime.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        is False
    )
    assert evidence.cublas_workspace_config == ":4096:8"
    assert evidence.torch_deterministic_algorithms is True
    assert evidence.cudnn_benchmark is False
    assert evidence.cudnn_deterministic is True
    assert evidence.allow_tf32_matmul is False
    assert evidence.allow_tf32_cudnn is False
    assert evidence.allow_fp16_reduced_precision_reduce is False

    class _DriftingCudnn:
        def __init__(self) -> None:
            self.benchmark = True
            self.deterministic = False
            self._allow_tf32 = True

        @property
        def allow_tf32(self) -> bool:
            return True

        @allow_tf32.setter
        def allow_tf32(self, value: bool) -> None:
            self._allow_tf32 = value

    drifting_runtime = _FakeTorch()
    drifting_runtime.backends.cudnn = _DriftingCudnn()
    with pytest.raises(
        RuntimeError,
        match="effective CUDA precision runtime policy drifted",
    ):
        production_attempt._apply_frozen_precision_runtime_policy(
            training=training,
            torch_runtime=drifting_runtime,
            environment={},
        )

    from vfe4.artifacts.live_environment import (
        LivePrecisionRuntimeEvidence,
    )

    published_evidence = LivePrecisionRuntimeEvidence(
        cublas_workspace_config=":4096:8",
        torch_deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        allow_tf32_matmul=False,
        allow_tf32_cudnn=False,
        allow_fp16_reduced_precision_reduce=False,
    )

    class _UnavailableCuda:
        def is_available(self) -> bool:
            return False

    class _UnavailableTorch:
        cuda = _UnavailableCuda()

        @staticmethod
        def device(value: str) -> SimpleNamespace:
            assert value == "cuda:0"
            return SimpleNamespace(type="cuda")

    monkeypatch.setattr(
        production_attempt,
        "_apply_frozen_precision_runtime_policy",
        lambda *, training: published_evidence,
    )
    monkeypatch.setattr(production_attempt, "torch", _UnavailableTorch())

    with pytest.raises(
        production_attempt.ProductionOperationError,
        match="authorized production training requires CUDA",
    ):
        production_attempt._execute_attempt(
            attempt=None,
            training=training,
            source_lock=None,
            readiness=None,
            cache_root=tmp_path,
            plan=None,
            reserved=SimpleNamespace(inprogress_path=tmp_path),
            backend=production_attempt._backend(),
            resume_active=False,
            resource_abort=lambda: None,
        )

    evidence_path = tmp_path / "live-precision-runtime-evidence.json"
    assert evidence_path.is_file()
    assert production_attempt._canonical_document(evidence_path) == {
        "schema_version": "wt103-live-precision-runtime-evidence-v1",
        "effective_precision_policy": {
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
            "allow_fp16_reduced_precision_reduce": False,
        },
    }


def _production_experiment_plan_fixture(training):
    from vfe4.artifacts.run_directory import ExperimentPlan

    return ExperimentPlan.create(
        experiment_id="wt103-production-attempt-fixture",
        endpoint_inventory=training.endpoint_inventory,
        git_head="1" * 40,
        dirty_digest="2" * 64,
        config_sha256=training.experiment_config_sha256,
        source_record_sha256="3" * 64,
        tokenizer_spec_sha256="4" * 64,
        token_cache_set_sha256="5" * 64,
        window_manifest_sha256s=("6" * 64, "7" * 64, "8" * 64),
        schedule_set_sha256="9" * 64,
        factory_set_sha256="a" * 64,
        objective_sha256="b" * 64,
        checkpoint_schema_sha256="c" * 64,
        resource_forecast_sha256="d" * 64,
        expected_run_artifact_paths=(
            "live-precision-runtime-evidence.json",
            "metrics.csv",
            "metrics.jsonl",
        ),
        expected_group_artifact_paths=("result-table.json",),
    )


class _ResumeSourceLockFixture:
    source_lock_sha256 = "c" * 64
    finalized_source = SimpleNamespace(record_sha256="d" * 64)

    def __post_init__(self) -> None:
        return None


class _ResumeReadinessBundleFixture:
    def __init__(self, *, volume_identity: str) -> None:
        self.durability = SimpleNamespace(volume_identity=volume_identity)
        self.resource_forecast = SimpleNamespace(
            power_provider_identity_sha256="e" * 64
        )
        self.environment = SimpleNamespace(environment_sha256="f" * 64)


class _ResumeReadinessFixture:
    result_sha256 = "1" * 64
    readiness_token = SimpleNamespace(
        finalized_source_record_sha256="d" * 64,
        token_sha256="2" * 64,
    )

    def __init__(self, *, status: object, readiness_bundle: object) -> None:
        self.status = status
        self.readiness_bundle = readiness_bundle

    def __post_init__(self) -> None:
        return None


def _install_resume_authority_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    production_attempt: object,
    *,
    volume_identity: str,
):
    bundle = _ResumeReadinessBundleFixture(
        volume_identity=volume_identity,
    )
    source_lock = _ResumeSourceLockFixture()
    readiness = _ResumeReadinessFixture(
        status=production_attempt.GateStatus.PASS,
        readiness_bundle=bundle,
    )
    monkeypatch.setattr(
        production_attempt,
        "ProductionSourceLock",
        _ResumeSourceLockFixture,
    )
    monkeypatch.setattr(
        production_attempt,
        "ProductionReadinessResult",
        _ResumeReadinessFixture,
    )
    monkeypatch.setattr(
        production_attempt,
        "Task14ReadinessBundle",
        _ResumeReadinessBundleFixture,
    )
    return source_lock, readiness


def test_production_reservations_bind_exact_ordered_tuning_key(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.run_directory import publish_experiment_plan
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    plan_value = _production_experiment_plan_fixture(training)
    backend = production_attempt._backend()
    plan = publish_experiment_plan(
        tmp_path,
        plan_value,
        backend=backend,
    )
    tuning = production_attempt._attempt_inventory(training, None)
    reserved, resumed = production_attempt._reserve_production_attempt(
        experiment_root=tmp_path,
        attempt=tuning[0],
        plan=plan,
        readiness=None,  # not consulted by a new reservation
        backend=backend,
        mode="train",
    )

    assert resumed is False
    assert reserved.tuning_attempt_key == plan.plan.tuning_attempt_keys[0]
    production_attempt.release_run_execution_lease(reserved)


def test_attempt_progress_follows_reservation_and_durable_finalization(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.environment import ResourceUsageLedger
    from vfe4.artifacts.run_directory import publish_experiment_plan
    from vfe4.training import production_attempt
    from vfe4.training.progress import use_progress_reporter

    training = resolve_training_config(default_training_config_mapping())
    plan_value = _production_experiment_plan_fixture(training)
    backend = production_attempt._backend()
    plan = publish_experiment_plan(tmp_path, plan_value, backend=backend)
    attempt = production_attempt._attempt_inventory(training, None)[0]
    reserved, resumed = production_attempt._reserve_production_attempt(
        experiment_root=tmp_path,
        attempt=attempt,
        plan=plan,
        readiness=None,
        backend=backend,
        mode="train",
    )
    usage = production_attempt._AttemptResourceUsage(
        device_seconds=3.0,
        wall_seconds=4.0,
        sampled_energy_kwh=0.01,
        usage_evidence_sha256="a" * 64,
    )
    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=plan_value.experiment_plan_sha256,
        resource_profile=training.profile.resources,
    )
    manifest_path = reserved.final_path / "run-manifest.json"
    manifest = SimpleNamespace(
        run_path=reserved.final_path,
        disposition="success",
        manifest_sha256="b" * 64,
    )
    outcome = SimpleNamespace(terminal_checkpoint_identity_sha256="c" * 64)
    seen: list[tuple[str, object]] = []

    class _Recorder:
        def report(self, event: str, payload: object, /) -> None:
            if event == "attempt_started":
                assert reserved.inprogress_path.is_dir()
            if event == "attempt_finished":
                assert manifest_path.is_file()
            seen.append((event, payload))

    try:
        with use_progress_reporter(_Recorder()):
            production_attempt._emit_attempt_started(
                attempt=attempt,
                reserved=reserved,
                phase_total=plan_value.tuning_attempt_count,
                resume_active=resumed,
            )
            reserved.inprogress_path.rename(reserved.final_path)
            manifest_path.write_bytes(b"durable terminal sentinel")
            production_attempt._emit_attempt_finished(
                attempt=attempt,
                manifest=manifest,
                outcome=outcome,
                usage=usage,
                ledger=ledger,
            )
    finally:
        production_attempt.release_run_execution_lease(reserved)

    assert [event for event, _payload in seen] == [
        "attempt_started",
        "attempt_finished",
    ]
    started = seen[0][1]
    finished = seen[1][1]
    assert started["role"] == "tuning"
    assert started["arm_id"] == attempt.arm_id
    assert started["seed_id"] == attempt.seed_id
    assert started["resume_count"] == 0
    assert finished["observed"] == {
        "gpu_seconds": 3.0,
        "wall_seconds": 4.0,
        "energy_kwh": 0.01,
    }
    assert finished["accounted"]["ledger_sha256"] == ledger.ledger_sha256
    assert finished["observed_disk_bytes"] is None
    assert finished["observed_disk_bytes_status"] == "unavailable"


def test_resume_reopens_declared_plan_ledger_and_sidecar_before_terminal_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.environment import (
        ResourceUsageEvent,
        ResourceUsageLedger,
    )
    from vfe4.artifacts.run_directory import publish_experiment_plan
    from vfe4.training import production_attempt

    raw_training = default_training_config_mapping()
    raw_training["operation"] = "resume"
    training = resolve_training_config(raw_training)
    backend = production_attempt._backend()
    plan_value = _production_experiment_plan_fixture(training)
    run_root = tmp_path / "runs"
    experiment_root = run_root / plan_value.experiment_id
    plan = publish_experiment_plan(
        experiment_root,
        plan_value,
        backend=backend,
    )
    declared_plan = plan.plan_path
    terminal_index = experiment_root / "experiment-index.json"
    assert declared_plan.name == "experiment-plan.json"
    assert not terminal_index.exists()

    tuning = production_attempt._attempt_inventory(training, None)

    def outcome_for(attempt):
        return production_attempt.ProductionAttemptOutcome.create(
            attempt_sha256=attempt.attempt_sha256,
            validation_nll_sum=1.0,
            validation_counted_targets=1,
            validation_nll_per_token=1.0,
            accepted_updates=1,
            terminal_checkpoint_identity_sha256=(
                None if attempt.role == "tuning" else "f" * 64
            ),
            metrics_jsonl_sha256="a" * 64,
            metrics_csv_sha256="b" * 64,
        )

    tuning_outcomes = tuple(outcome_for(attempt) for attempt in tuning)
    selected = production_attempt._select_hyperparameters(
        {
            "completed_outcomes": [
                production_attempt._outcome_document(attempt, outcome)
                for attempt, outcome in zip(
                    tuning,
                    tuning_outcomes,
                    strict=True,
                )
            ]
        },
        training,
    )
    confirmation = production_attempt._attempt_inventory(training, selected)
    confirmation_outcomes = tuple(
        outcome_for(attempt) for attempt in confirmation
    )
    outcome_by_attempt = {
        attempt.attempt_sha256: outcome
        for attempt, outcome in (
            *zip(tuning, tuning_outcomes, strict=True),
            *zip(confirmation, confirmation_outcomes, strict=True),
        )
    }

    ledger = ResourceUsageLedger.create(
        experiment_plan_sha256=plan_value.experiment_plan_sha256,
        resource_profile=training.profile.resources,
    )
    production_attempt._publish_resource_usage_ledger(
        path=experiment_root / "resource-usage-ledger.json",
        ledger=ledger,
        backend=backend,
    )

    original, originally_resumed = (
        production_attempt._reserve_production_attempt(
            experiment_root=experiment_root,
            attempt=tuning[0],
            plan=plan,
            readiness=None,
            backend=backend,
            mode="train",
        )
    )
    assert originally_resumed is False
    (
        _contract,
        _sidecar_identity,
        _cursor,
        _sidecar_checkpoint,
        sidecar_path,
    ) = _resume_sidecar_fixture(
        original.inprogress_path,
        slot=0,
        payload=b"declared-plan-resume-sidecar",
    )
    production_attempt.release_run_execution_lease(original)
    source_lock, readiness = _install_resume_authority_fixtures(
        monkeypatch,
        production_attempt,
        volume_identity=plan.durable_file.volume_identity,
    )
    monkeypatch.setattr(
        production_attempt,
        "_production_experiment_plan",
        lambda **_kwargs: plan_value,
    )
    monkeypatch.setattr(
        production_attempt,
        "discover_nvidia_smi_power_provider",
        lambda: (
            SimpleNamespace(
                identity_sha256="e" * 64,
                sample_interval_ms=(
                    training.profile.resources.power_sample_interval_ms
                ),
            ),
            object(),
        ),
    )
    monkeypatch.setattr(
        production_attempt,
        "_frozen_readiness_conservative_power_watts",
        lambda **_kwargs: 1.0,
    )

    reopened_plan_paths: list[Path] = []
    reopen_plan = production_attempt._reopen_experiment_plan_identity

    def reopen_declared_plan(
        *,
        plan_path: Path,
        expected_plan: object,
        readiness: object,
    ):
        reopened_plan_paths.append(plan_path)
        return reopen_plan(
            plan_path=plan_path,
            expected_plan=expected_plan,
            readiness=readiness,
        )

    monkeypatch.setattr(
        production_attempt,
        "_reopen_experiment_plan_identity",
        reopen_declared_plan,
    )
    reopened_sidecars: list[Path] = []
    checkpoint_sidecar = production_attempt._checkpoint_sidecar

    def reopen_checkpoint_sidecar(path: Path):
        reopened_sidecars.append(path)
        return checkpoint_sidecar(path)

    monkeypatch.setattr(
        production_attempt,
        "_checkpoint_sidecar",
        reopen_checkpoint_sidecar,
    )
    reserve_attempt = production_attempt._reserve_production_attempt
    real_resume_reservations = 0

    def reserve_or_stub(**kwargs):
        nonlocal real_resume_reservations
        attempt = kwargs["attempt"]
        if attempt.attempt_sha256 == tuning[0].attempt_sha256:
            real_resume_reservations += 1
            return reserve_attempt(**kwargs)
        return SimpleNamespace(attempt_id=attempt.attempt_id), False

    monkeypatch.setattr(
        production_attempt,
        "_reserve_production_attempt",
        reserve_or_stub,
    )

    def execute_without_training(**kwargs):
        attempt = kwargs["attempt"]
        reserved = kwargs["reserved"]
        resume_active = kwargs["resume_active"]
        current_ledger = kwargs["ledger"]
        if attempt.attempt_sha256 == tuning[0].attempt_sha256:
            assert resume_active is True
            assert reserved.inprogress_path == original.inprogress_path
            production_attempt.release_run_execution_lease(reserved)
        else:
            assert resume_active is False
        updated_ledger = current_ledger.append(
            ResourceUsageEvent.create(
                attempt_id=attempt.attempt_id,
                segment_ordinal=0,
                device_seconds=0.001,
                wall_seconds=0.002,
                sampled_energy_kwh=0.0,
                usage_evidence_sha256=hashlib.sha256(
                    attempt.attempt_id.encode("ascii")
                ).hexdigest(),
            )
        )
        production_attempt._publish_resource_usage_ledger(
            path=kwargs["ledger_path"],
            ledger=updated_ledger,
            backend=kwargs["backend"],
        )
        return (
            outcome_by_attempt[attempt.attempt_sha256],
            SimpleNamespace(run_id=attempt.attempt_id),
            updated_ledger,
        )

    monkeypatch.setattr(
        production_attempt,
        "_execute_reserved_attempt",
        execute_without_training,
    )

    def publish_terminal_index(
        root: Path,
        *,
        plan: object,
        run_manifests: tuple[object, ...],
        stage: str,
        artifact_records: tuple[object, ...],
        backend: object,
    ) -> object:
        del plan, backend
        assert root == experiment_root
        assert stage == "pretest"
        assert artifact_records == ()
        assert len(run_manifests) == len(tuning) + len(confirmation)
        assert not terminal_index.exists()
        terminal_index.write_bytes(b"terminal pretest index")
        return SimpleNamespace(index_path=terminal_index)

    monkeypatch.setattr(
        production_attempt,
        "publish_experiment_index",
        publish_terminal_index,
    )
    paths = SimpleNamespace(
        cache_root=tmp_path / "cache",
        run_root=run_root,
        resume_experiment_plan_path=declared_plan,
    )

    result = production_attempt.run_production_attempts(
        training=training,
        paths=paths,
        source_lock=source_lock,
        readiness=readiness,
        mode="resume",
    )

    assert result.status == "COMPLETE"
    assert result.experiment_index_path == str(terminal_index)
    assert reopened_plan_paths == [declared_plan]
    assert real_resume_reservations == 1
    assert reopened_sidecars == [sidecar_path]
    assert terminal_index.is_file()


def test_resume_rejects_reparse_experiment_ancestor_before_durable_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production_attempt

    raw_training = default_training_config_mapping()
    raw_training["operation"] = "resume"
    training = resolve_training_config(raw_training)
    run_root = tmp_path / "runs"
    experiment_root = run_root / "declared-experiment"
    experiment_root.mkdir(parents=True)
    declared_plan = experiment_root / "experiment-plan.json"
    declared_plan.write_bytes(b"unopened plan sentinel")
    source_lock, readiness = _install_resume_authority_fixtures(
        monkeypatch,
        production_attempt,
        volume_identity="test-volume",
    )
    original_lstat = Path.lstat

    def lstat_with_reparse_ancestor(path: Path):
        metadata = original_lstat(path)
        if path == experiment_root:
            return SimpleNamespace(
                st_mode=metadata.st_mode,
                st_file_attributes=0x400,
                st_size=metadata.st_size,
            )
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse_ancestor)
    monkeypatch.setattr(
        production_attempt,
        "_backend",
        lambda: (_ for _ in ()).throw(
            AssertionError("durability probe reached")
        ),
    )
    paths = SimpleNamespace(
        cache_root=tmp_path / "cache",
        run_root=run_root,
        resume_experiment_plan_path=declared_plan,
    )

    with pytest.raises(
        production_attempt.ProductionOperationError,
        match="symlink, junction, or reparse point",
    ):
        production_attempt.run_production_attempts(
            training=training,
            paths=paths,
            source_lock=source_lock,
            readiness=readiness,
            mode="resume",
        )


def test_production_reopen_recovers_durable_terminal_before_resume_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.run_directory import publish_experiment_plan
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    backend = production_attempt._backend()
    plan = publish_experiment_plan(
        tmp_path,
        _production_experiment_plan_fixture(training),
        backend=backend,
    )
    attempt = production_attempt._attempt_inventory(training, None)[0]
    inprogress = (
        tmp_path / "runs" / ".inprogress" / attempt.attempt_id
    )
    inprogress.mkdir(parents=True)
    (inprogress / "run-manifest.json").write_bytes(b"durable-terminal")
    calls: list[str] = []

    def recover(root, run_id, *, plan):
        calls.append(run_id)
        final = root / "runs" / run_id
        inprogress.rename(final)
        return SimpleNamespace(run_path=final)

    manifest = SimpleNamespace(
        run_id=attempt.attempt_id,
        run_role="tuning",
        experiment_plan_sha256=plan.plan.experiment_plan_sha256,
        tuning_attempt_key=plan.plan.tuning_attempt_keys[0],
        disposition="success",
    )
    outcome = object()
    monkeypatch.setattr(
        production_attempt,
        "recover_terminal_run",
        recover,
    )
    monkeypatch.setattr(
        production_attempt,
        "validate_run_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        production_attempt,
        "_reopen_attempt_outcome",
        lambda **_kwargs: outcome,
    )

    reopened = production_attempt._reopen_completed_attempt_prefix(
        experiment_root=tmp_path,
        attempts=(attempt,),
        training=training,
        plan=plan,
    )

    assert calls == [attempt.attempt_id]
    assert reopened == ((attempt, outcome, manifest),)
    assert not inprogress.exists()
    assert (tmp_path / "runs" / attempt.attempt_id).is_dir()


def test_validation_scoring_seeds_each_row_and_advances_incremental_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.training import production_attempt

    vocabulary = VocabularyIdentity(
        "production-attempt-prefix-fixture-v1",
        32,
        "a" * 64,
    )
    batch = SimpleNamespace(
        window_ids=(0, 1),
        inputs=torch.tensor(
            (
                (11, 12),
                (21, 0),
            ),
            dtype=torch.int64,
        ),
        targets=torch.tensor(
            (
                (12, 13),
                (22, 0),
            ),
            dtype=torch.int64,
        ),
        attention_mask=torch.tensor(
            (
                (1, 1),
                (1, 0),
            ),
            dtype=torch.int64,
        ),
    )
    calls: list[tuple[tuple[int, ...], object | None]] = []
    resource_abort_checks = 0

    def resource_abort() -> None:
        nonlocal resource_abort_checks
        resource_abort_checks += 1

    expected_target_by_prefix = {
        (11,): (12, -0.25),
        (11, 12): (13, -0.75),
        (21,): (22, -1.5),
    }

    class _Predictor:
        def next_token_log_probs(
            self,
            prefix: object,
            stream: object,
            cache: object | None,
        ) -> object:
            del stream
            token_ids = tuple(
                int(value)
                for value in getattr(prefix, "token_ids").tolist()
            )
            calls.append((token_ids, cache))
            target, selected = expected_target_by_prefix[token_ids]
            values = torch.full(
                (vocabulary.size,),
                -100.0,
                dtype=torch.float64,
            )
            values[target] = selected
            identity = owned_sha256(
                "test.production-attempt-cache-audit.v1",
                token_ids,
            )
            next_cache = SimpleNamespace(cache_sha256=identity)
            return SimpleNamespace(
                log_probs=SimpleNamespace(
                    value=lambda: values,
                    raw_bytes_sha256=identity,
                ),
                cache=next_cache,
                estimator_record=SimpleNamespace(
                    record_sha256=identity,
                ),
            )

    predictor = _Predictor()
    bundle = SimpleNamespace(
        vocabulary=vocabulary,
        make_predictor=lambda: predictor,
    )
    windows = SimpleNamespace(
        manifest=SimpleNamespace(
            counted_targets=3,
            manifest_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        production_attempt,
        "_estimator_stream",
        lambda _bundle: SimpleNamespace(stream_sha256="c" * 64),
    )
    monkeypatch.setattr(
        production_attempt,
        "iter_production_batches",
        lambda **_kwargs: iter((batch,)),
    )

    score = production_attempt._score_validation(
        bundle=bundle,
        windows=windows,
        schedule=SimpleNamespace(schedule_sha256="d" * 64),
        resource_abort=resource_abort,
    )

    assert score.counted_targets == 3
    assert score.summed_nll == pytest.approx(2.5)
    assert score.cache_audit.passed is True
    assert score.cache_audit.summed_nll == score.summed_nll
    assert (
        score.cache_audit.cold_nll_terms_sha256
        == score.cache_audit.warm_nll_terms_sha256
        == score.cache_audit.reverse_nll_terms_sha256
    )
    assert len(score.cache_audit.audit_sha256) == 64
    assert resource_abort_checks >= 9
    assert tuple(prefix for prefix, _cache in calls) == (
        # Cold pass.
        (11,),
        (11, 12),
        (21,),
        # Warm pass.
        (11,),
        (11, 12),
        (21,),
        # Reverse-order cold rebuild.
        (21,),
        (11, 12),
        (11,),
    )
    assert calls[0][1] is None
    assert calls[1][1] is None
    assert calls[2][1] is None
    assert calls[3][1] is None
    assert calls[4][1] is not None
    assert calls[5][1] is None
    assert all(cache is None for _prefix, cache in calls[6:])


def _accepted_a0_step() -> StepResult:
    training = resolve_training_config(default_training_config_mapping())
    arm = training.endpoint_inventory.arms[0]
    update_payload = {
        "schema_version": "wt103-update-record-v1",
        "arm_id": arm.arm_id,
        "phase": "model_ce_adam_proposal",
        "update_label": "adam_proposal",
        "accepted": True,
        "rejection_reason": None,
        "expected_autograd_scope": "m_step",
        "observed_autograd_scope": "m_step",
        "snapshot_sha256": None,
        "optimizer_state_sha256": "b" * 64,
        "scheduler_state_sha256": "c" * 64,
    }
    update = WT103UpdateRecord(
        **update_payload,
        update_sha256=owned_sha256(
            "vfe4.wt103.update-record.v1",
            update_payload,
        ),
    )
    control = UpdateControlRecord.create(
        learning_rate=0.001,
        scheduler_ordinal=7,
        scheduler_state_sha256="c" * 64,
        amp_applicability="applicable",
        amp_scale=65_536.0,
        amp_overflow=False,
        clipping_threshold=5.0,
        gradient_norm_applicability="applicable",
        pre_clip_norm=6.0,
        post_clip_norm=4.0,
        pre_clip_inf_norm=3.5,
        post_clip_inf_norm=2.5,
        clipped=True,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        adamw_epsilon=1.0e-8,
        adamw_weight_decay=0.01,
        adamw_amsgrad=False,
        adamw_maximize=False,
        adamw_capturable=False,
        adamw_differentiable=False,
        adamw_foreach=False,
        adamw_fused=False,
    )
    counted_targets = 3
    before_numerator = 10.0
    after_numerator = 7.0
    rounded_before = float(
        (
            -torch.tensor(before_numerator, dtype=torch.float32)
            / counted_targets
        ).item()
    )
    rounded_after = float(
        (
            -torch.tensor(after_numerator, dtype=torch.float32)
            / counted_targets
        ).item()
    )
    assert rounded_before * counted_targets != -before_numerator
    evidence = ProposalEvidence.create(
        phase="model_ce_adam_proposal",
        affected_block="model",
        expected_autograd_scope="m_step",
        observed_autograd_scope="m_step",
        objective_before_terms=(
            ("cross_entropy_value", before_numerator),
        ),
        objective_after_terms=(
            ("cross_entropy_value", after_numerator),
        ),
        objective_before_value=rounded_before,
        objective_after_value=rounded_after,
        counted_targets=counted_targets,
        estimator_error_bound_before=None,
        estimator_error_bound_after=None,
        support_valid=True,
        spd_valid=True,
        damping_applied=False,
        projection_applied=False,
        rollback_applied=False,
    )
    return StepResult(
        arm_id=arm.arm_id,
        objective_kind="cross_entropy",
        phase_order=("model_ce_adam_proposal",),
        updates=(update,),
        update_controls=(control,),
        proposal_evidence=(evidence,),
        snapshot_sha256=None,
        objective_diagnostics_applicable=True,
        objective_terms={"cross_entropy_value": after_numerator},
        complete_elbo_numerator=None,
        complete_elbo_value=None,
        counted_targets=counted_targets,
        accepted=True,
        failure_kind=None,
        expected_autograd_scope="m_step",
        observed_autograd_scope="m_step",
        reverse_mode_autograd=True,
        monotonicity_claim=False,
    )


def test_train_metric_projection_uses_raw_objective_controls_and_observations() -> None:
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    values = production_attempt._train_metric_values(
        arm_spec=training.endpoint_inventory.arms[0],
        step=_accepted_a0_step(),
        timing=PhaseTimingObservation(
            data_wait_ns=1_000_000,
            forward_ns=2_000_000,
            inference_ns=0,
            backward_ns=3_000_000,
            update_ns=4_000_000,
            evaluation_ns=0,
            checkpoint_ns=0,
            wall_ns=10_000_000,
        ),
        memory=MemoryObservation(
            process_rss_bytes=100,
            process_hwm_bytes=120,
            cuda_allocated_bytes=30,
            cuda_reserved_bytes=50,
            cuda_peak_allocated_bytes=45,
            cuda_peak_reserved_bytes=80,
        ),
        allocation_retries=2,
        oom_count=1,
        source=None,
        numerical=None,
    )
    by_name = {value.name: value for value in values}

    cross_entropy = by_name["train_cross_entropy"]
    assert (
        cross_entropy.numerator,
        cross_entropy.denominator,
        cross_entropy.value,
    ) == (7.0, 3, 7.0 / 3.0)
    assert (
        by_name["objective_before"].numerator,
        by_name["objective_before"].denominator,
        by_name["objective_before"].value,
    ) == (-10.0, 3, -10.0 / 3.0)
    assert by_name["objective_after"].numerator == -7.0
    assert by_name["objective_after"].value == -7.0 / 3.0
    assert by_name["gradient_pre_clip_l2"].value == 6.0
    assert by_name["gradient_post_clip_l2"].value == 4.0
    assert by_name["gradient_l2"].value == 4.0
    assert by_name["gradient_inf"].value == 2.5
    assert by_name["data_wait_seconds"].value == 0.001
    assert by_name["forward_seconds"].value == 0.002
    assert by_name["backward_seconds"].value == 0.003
    assert by_name["update_seconds"].value == 0.004
    assert by_name["wall_seconds"].value == 0.01
    assert (
        by_name["tokens_per_second"].numerator,
        by_name["tokens_per_second"].denominator,
        by_name["tokens_per_second"].value,
    ) == (3.0, 10_000_000, 300.0)
    assert by_name["process_rss_bytes"].value == 100.0
    assert by_name["cuda_peak_reserved_bytes"].value == 80.0
    assert by_name["allocation_retries"].value == 2.0
    assert by_name["oom_count"].value == 1.0
    assert by_name["snapshot_identity_present"].applicability == (
        "not_applicable"
    )
    assert "prior_nll_sum" not in by_name


def test_validation_metric_projection_keeps_corpus_nll_raw_totals() -> None:
    from vfe4.training import production_attempt

    training = resolve_training_config(default_training_config_mapping())
    cache_audit = production_attempt._ValidationCacheAudit.create(
        window_manifest_sha256="1" * 64,
        schedule_sha256="2" * 64,
        estimator_stream_sha256="3" * 64,
        records_sha256="4" * 64,
        nll_terms_sha256="5" * 64,
        summed_nll=5.0,
        prefix_record_count=2,
        counted_targets=2,
    )
    values = production_attempt._validation_metric_values(
        arm_spec=training.endpoint_inventory.arms[0],
        nll_sum=5.0,
        counted_targets=2,
        scorer_kind="exact_autoregressive",
        estimator_stream_id=None,
        particle_count=None,
        cache_audit=cache_audit,
        evaluation_ns=4_000_000,
        wall_ns=5_000_000,
        memory=MemoryObservation(
            process_rss_bytes=200,
            process_hwm_bytes=250,
            cuda_allocated_bytes=40,
            cuda_reserved_bytes=60,
            cuda_peak_allocated_bytes=50,
            cuda_peak_reserved_bytes=90,
        ),
        allocation_retries=3,
        oom_count=0,
    )
    by_name = {value.name: value for value in values}

    assert (
        by_name["prior_nll_sum"].numerator,
        by_name["prior_nll_sum"].denominator,
        by_name["prior_nll_sum"].value,
    ) == (None, None, 5.0)
    assert (
        by_name["prior_nll_per_token"].numerator,
        by_name["prior_nll_per_token"].denominator,
        by_name["prior_nll_per_token"].value,
    ) == (5.0, 2, 2.5)
    assert by_name["perplexity"].numerator == 5.0
    assert by_name["perplexity"].denominator == 2
    assert by_name["perplexity"].value == pytest.approx(
        torch.exp(torch.tensor(2.5, dtype=torch.float64)).item()
    )
    assert by_name["cache_audit_passed"].value == 1.0
    assert cache_audit.audit_sha256 in (
        by_name["cache_audit_passed"].reason
    )
    assert by_name["evaluation_seconds"].value == 0.004
    assert by_name["wall_seconds"].value == 0.005
    assert (
        by_name["tokens_per_second"].numerator,
        by_name["tokens_per_second"].denominator,
        by_name["tokens_per_second"].value,
    ) == (2.0, 5_000_000, 400.0)
    assert by_name["estimator_stream"].applicability == "not_applicable"
    assert by_name["particle_count"].applicability == "not_applicable"


def _resume_sidecar_fixture(
    root: Path,
    *,
    slot: int,
    payload: bytes,
):
    from vfe4.artifacts.durability import canonical_json_bytes_generic
    from vfe4.checkpoint import ResumeContract, make_checkpoint_identity
    from vfe4.types import DataCursor

    digest = "a" * 64
    contract = ResumeContract.create(
        logical_key="wt103-checkpoint-slot-fixture",
        checkpoint_role="resume_only",
        training_complete=False,
        arm_spec_sha256=digest,
        experiment_plan_sha256=digest,
        config_sha256=digest,
        objective_sha256=digest,
        model_schema_sha256=digest,
        recognition_schema_sha256=digest,
        optimizer_schema_sha256=digest,
        scheduler_schema_sha256=digest,
        amp_schema_sha256=digest,
        rng_schema_sha256=digest,
        estimator_schema_sha256=digest,
        cursor_schema_sha256=digest,
        metric_schema_sha256=digest,
        update_trace_schema_sha256=digest,
        precision_profile_sha256=digest,
        dependency_lock_sha256=digest,
        source_sha256=digest,
        tokenizer_sha256=digest,
        data_sha256=digest,
        window_sha256=digest,
        permutation_sha256=digest,
        evidence_sha256=digest,
        environment_sha256=digest,
        maximum_checkpoint_bytes=2 * 1024 * 1024,
        maximum_tensor_bytes=256 * 1024,
        maximum_total_tensor_bytes=512 * 1024,
        maximum_tensor_count=64,
        maximum_container_items=512,
        maximum_recursion_depth=16,
    )
    identity = make_checkpoint_identity(
        logical_key=contract.logical_key,
        checkpoint_role="resume_only",
        scientific_state_sha256="b" * 64,
        checkpoint_payload_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_manifest_body_sha256="c" * 64,
        size_bytes=len(payload),
    )
    cursor = DataCursor.create(
        split="train",
        pass_index=0,
        permutation_sha256="d" * 64,
        next_batch_ordinal=8,
        next_window_ids=(16, 17),
        counted_targets=256,
    )
    checkpoint_path = root / f"resume-only-{slot}.pt"
    checkpoint_path.write_bytes(payload)
    sidecar = root / "resume-sidecar.json"
    sidecar.write_bytes(
        canonical_json_bytes_generic(
            {
                "contract": contract.canonical_payload(),
                "identity": dataclasses.asdict(identity),
                "cursor": dataclasses.asdict(cursor),
                "checkpoint_path": str(checkpoint_path),
            }
        )
    )
    return contract, identity, cursor, checkpoint_path, sidecar


@pytest.mark.parametrize(
    "fault_phase",
    ("before_owner", "after_lease"),
)
def test_fresh_production_resume_recovers_preledger_lineage_intent_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_phase: str,
) -> None:
    from vfe4.artifacts.run_directory import (
        RunLifecycleError,
        publish_experiment_plan,
        release_run_execution_lease,
        reopen_resume_lineage_event,
    )
    from vfe4.training import production_attempt

    class _ReadinessBundle:
        def __init__(self) -> None:
            self.environment = SimpleNamespace(
                environment_sha256="e" * 64,
            )

    class _PreledgerFaultBackend:
        def __init__(self, delegate: object) -> None:
            self.delegate = delegate
            self.fail_once = False

        def __getattr__(self, name: str) -> object:
            return getattr(self.delegate, name)

        def create_exclusive(self, path: Path, payload: bytes):
            if (
                self.fail_once
                and fault_phase == "before_owner"
                and path.name == "resume-owner.json"
            ):
                self.fail_once = False
                raise OSError(
                    "injected process loss before owner publication"
                )
            identity = self.delegate.create_exclusive(path, payload)
            if (
                self.fail_once
                and fault_phase == "after_lease"
                and path.parent.name == "resume-leases"
                and path.name == "00000001.json"
            ):
                self.fail_once = False
                raise OSError(
                    "injected process loss after owner/lease publication"
                )
            return identity

    training = resolve_training_config(default_training_config_mapping())
    backend = _PreledgerFaultBackend(production_attempt._backend())
    plan = publish_experiment_plan(
        tmp_path,
        _production_experiment_plan_fixture(training),
        backend=backend,
    )
    attempt = production_attempt._attempt_inventory(training, None)[0]
    original, resumed = production_attempt._reserve_production_attempt(
        experiment_root=tmp_path,
        attempt=attempt,
        plan=plan,
        readiness=None,
        backend=backend,
        mode="train",
    )
    assert resumed is False
    _resume_sidecar_fixture(
        original.inprogress_path,
        slot=0,
        payload=b"preledger-production-resume-checkpoint",
    )
    release_run_execution_lease(original)

    readiness = SimpleNamespace(readiness_bundle=_ReadinessBundle())
    monkeypatch.setattr(
        production_attempt,
        "Task14ReadinessBundle",
        _ReadinessBundle,
    )
    generated_timestamps: list[str] = []
    timestamps = iter(
        (
            "2026-07-28T18:00:00.000000Z",
            "2026-07-28T18:00:01.000000Z",
        )
    )

    def fresh_timestamp() -> str:
        value = next(timestamps)
        generated_timestamps.append(value)
        return value

    monkeypatch.setattr(
        production_attempt,
        "_canonical_utc_timestamp",
        fresh_timestamp,
    )
    backend.fail_once = True
    with pytest.raises(RunLifecycleError):
        production_attempt._reserve_production_attempt(
            experiment_root=tmp_path,
            attempt=attempt,
            plan=plan,
            readiness=readiness,
            backend=backend,
            mode="resume",
        )

    intent_path = (
        original.inprogress_path / "resume-lineage-intent.json"
    )
    assert intent_path.is_file()
    assert not (
        original.inprogress_path / "resume-lineage.jsonl"
    ).exists()
    assert not (
        original.inprogress_path / "resume-execution-started.json"
    ).exists()
    owner_path = original.inprogress_path / "resume-owner.json"
    leases_path = original.inprogress_path / "resume-leases"
    if fault_phase == "before_owner":
        assert not owner_path.exists()
        assert not leases_path.exists()
    else:
        assert owner_path.is_file()
        assert (
            leases_path / "00000001.json"
        ).is_file()

    recovered, recovered_resume = (
        production_attempt._reserve_production_attempt(
            experiment_root=tmp_path,
            attempt=attempt,
            plan=plan,
            readiness=readiness,
            backend=backend,
            mode="resume",
        )
    )

    assert recovered_resume is True
    assert recovered.resume_count == 1
    assert generated_timestamps == ["2026-07-28T18:00:00.000000Z"]
    reopened = reopen_resume_lineage_event(recovered.inprogress_path)
    assert reopened is not None
    assert (
        reopened.lineage_sha256
        == recovered.resume_owner_lineage_sha256
    )
    assert not (
        recovered.inprogress_path / "resume-execution-started.json"
    ).exists()
    release_run_execution_lease(recovered)


def test_resume_checkpoint_first_save_is_deterministic_and_existing_sidecar_is_authenticated(
    tmp_path: Path,
) -> None:
    from vfe4.training import production_attempt

    assert production_attempt._select_resume_checkpoint_path(tmp_path) == (
        tmp_path / "resume-only-0.pt"
    )

    _contract, _identity, _cursor, active, _sidecar = _resume_sidecar_fixture(
        tmp_path,
        slot=0,
        payload=b"authenticated-resume-checkpoint",
    )
    assert production_attempt._select_resume_checkpoint_path(tmp_path) == (
        tmp_path / "resume-only-1.pt"
    )

    active.write_bytes(b"tampered-resume-checkpointxxxxx")
    with pytest.raises(
        production_attempt.ProductionOperationError,
        match="checkpoint payload differs from its authenticated sidecar",
    ):
        production_attempt._select_resume_checkpoint_path(tmp_path)


def test_resume_checkpoint_crash_before_sidecar_publish_preserves_active_slot_and_retry_reuses_inactive_slot(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import canonical_json_bytes_generic
    from vfe4.checkpoint import make_checkpoint_identity
    from vfe4.training import production_attempt

    contract, old_identity, cursor, old_path, sidecar = _resume_sidecar_fixture(
        tmp_path,
        slot=0,
        payload=b"old-authenticated-resume-checkpoint",
    )
    selected = production_attempt._select_resume_checkpoint_path(tmp_path)
    assert selected == tmp_path / "resume-only-1.pt"

    # Fault injection: durable checkpoint publication succeeds, then the
    # process dies before the atomic sidecar replacement.
    new_payload = b"new-authenticated-resume-checkpoint"
    selected.write_bytes(new_payload)

    reopened = production_attempt._checkpoint_sidecar(sidecar)
    assert reopened[1] == old_identity
    assert reopened[3] == old_path
    assert old_path.read_bytes() == b"old-authenticated-resume-checkpoint"

    # Retry still derives its target from the authenticated predecessor, so
    # the orphaned inactive candidate is safe to replace.
    retry_path = production_attempt._select_resume_checkpoint_path(tmp_path)
    assert retry_path == selected
    retry_path.write_bytes(new_payload)
    new_identity = make_checkpoint_identity(
        logical_key=contract.logical_key,
        checkpoint_role="resume_only",
        scientific_state_sha256="e" * 64,
        checkpoint_payload_sha256=hashlib.sha256(new_payload).hexdigest(),
        checkpoint_manifest_body_sha256="f" * 64,
        size_bytes=len(new_payload),
    )
    sidecar.write_bytes(
        canonical_json_bytes_generic(
            {
                "contract": contract.canonical_payload(),
                "identity": dataclasses.asdict(new_identity),
                "cursor": dataclasses.asdict(cursor),
                "checkpoint_path": str(retry_path),
            }
        )
    )

    committed = production_attempt._checkpoint_sidecar(sidecar)
    assert committed[1] == new_identity
    assert committed[3] == retry_path
    assert production_attempt._reopen_committed_resume_checkpoint(
        sidecar,
        expected_identity=new_identity,
    ) == retry_path
