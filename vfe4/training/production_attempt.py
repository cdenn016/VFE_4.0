"""Authorized WikiText-103 tuning, confirmation, checkpoint, and resume.

The click launcher reaches this module only after exact source-lock reopen,
Task 14 PASS issuance, and the separate production-training authorization.
Training can map only the finalized train and validation caches. The held-out
test cache is not accepted by any function in this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal, Mapping, MutableMapping

import numpy as np
import torch

from vfe4.artifacts.durability import (
    DurableFileIdentity,
    DurabilityBackend,
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
    canonical_json_bytes_generic,
)
from vfe4.artifacts.environment import (
    PowerProviderIdentity,
    ResourceForecast,
    ResourceUsageEvent,
    ResourceUsageLedger,
)
from vfe4.artifacts.live_environment import (
    NvidiaSmiPowerSampler,
    PowerObservation,
    PowerSampleOperationFailure,
    LivePrecisionRuntimeEvidence,
    apply_frozen_precision_runtime_policy,
    discover_nvidia_smi_power_provider,
)
from vfe4.artifacts.live_readiness import Task14ReadinessBundle
from vfe4.artifacts.manifest import ArtifactIntegrityRecord
from vfe4.artifacts.run_directory import (
    ExperimentPlan,
    ExperimentPlanIdentity,
    ReservedRun,
    ResumeLineageEvent,
    RunManifestIdentity,
    consume_resume_execution_retry,
    finalize_run,
    publish_experiment_index,
    publish_experiment_plan,
    recover_terminal_run,
    reopen_resume_lineage_event,
    release_run_execution_lease,
    reserve_run,
    validate_run_manifest,
)
from vfe4.checkpoint import (
    ResumeContract,
    WT103CheckpointIdentity,
    load_checkpoint,
    save_checkpoint,
)
from vfe4.config.schema import TrainingConfig
from vfe4.data.windows import CausalPrefix, WindowSchedule
from vfe4.evaluation.prior_nll import wt103_estimator_stream_seed
from vfe4.predictive.proposal import EstimatorStream
from vfe4.recording.metrics import (
    WT103_REQUIRED_METRIC_FAMILIES,
    WT103_SOURCE_KL_DIAGNOSTIC_REASON,
    WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
    append_metric,
    applicable_metric,
    create_metric_record,
    export_metrics_csv,
    metric_family_applicability,
    metric_family_units,
    not_applicable_metric,
    validate_metric_log,
    validate_required_metric_families,
)
from vfe4.recording.failures import (
    FailureRecord,
    append_failure,
    create_failure_record,
    validate_failure_log,
)
from vfe4.training.engine import StepResult, train_step
from vfe4.training.production_observability import (
    MemoryObservation,
    NumericalObservation,
    PhaseTimer,
    PhaseTimingObservation,
    ProductionObservationError,
    SourceObservation,
    capture_memory_observation,
    project_objective_metrics,
)
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    DataCursor,
    EstimatorProtocol,
    MetricRecord,
    MetricValue,
    WT103ArmSpec,
    owned_sha256,
)

from .production import (
    ProductionOperationError,
    ProductionReadinessResult,
    ProductionSourceLock,
    ProductionWindowSet,
    iter_production_batches,
    open_production_training_split,
    production_cursor_after_batches,
)
from .progress import emit_progress
from .wt103_runtime import (
    WT103ArmRuntimeBundle,
    WT103RuntimeAuthority,
    build_wt103_arm_runtime,
    export_wt103_arm_runtime_state,
    rebuild_wt103_arm_runtime,
)


_ZERO_SHA256 = "0" * 64
_PLAN_NAME = "experiment-plan.json"
_PLAN_SCHEMA = "wt103-production-experiment-plan-v1"
_INDEX_NAME = "production-experiment-index.json"
_INDEX_SCHEMA = "wt103-production-experiment-index-v1"
_TERMINAL_INDEX_NAME = "experiment-index.json"
_RESULT_SCHEMA = "wt103-production-training-result-v1"
_MAXIMUM_INDEX_BYTES = 64 * 1024 * 1024
_CRASH_TAIL_RESERVE_SECONDS = 60.0
_RESOURCE_USAGE_HEARTBEAT_SECONDS = 30.0
_LIVE_PRECISION_RUNTIME_EVIDENCE_NAME = "live-precision-runtime-evidence.json"


@dataclass(frozen=True, slots=True)
class ProductionAttemptSpec:
    attempt_id: str
    ordinal: int
    role: Literal["tuning", "confirmation"]
    arm_id: str
    seed_id: int
    learning_rate: float
    weight_decay: float
    pass_count: int
    quarter_pass: bool
    attempt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        ordinal: int,
        role: Literal["tuning", "confirmation"],
        arm_id: str,
        seed_id: int,
        learning_rate: float,
        weight_decay: float,
        pass_count: int,
        quarter_pass: bool,
    ) -> "ProductionAttemptSpec":
        payload = {
            "ordinal": ordinal,
            "role": role,
            "arm_id": arm_id,
            "seed_id": seed_id,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pass_count": pass_count,
            "quarter_pass": quarter_pass,
        }
        digest = owned_sha256(
            "vfe4.wt103.production-attempt-spec.v1",
            payload,
        )
        return cls(
            attempt_id=f"attempt-{ordinal:04d}-{digest[:16]}",
            **payload,
            attempt_sha256=digest,
        )

    def __post_init__(self) -> None:
        if (
            self.role not in ("tuning", "confirmation")
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or self.attempt_id
            != f"attempt-{self.ordinal:04d}-{self.attempt_sha256[:16]}"
            or type(self.seed_id) is not int
            or not 0 <= self.seed_id < 2**63
            or type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or type(self.weight_decay) is not float
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or type(self.pass_count) is not int
            or self.pass_count not in (1, 2)
            or type(self.quarter_pass) is not bool
            or self.quarter_pass is (self.role != "tuning")
        ):
            raise ProductionOperationError(
                "production attempt specification is invalid"
            )
        expected = owned_sha256(
            "vfe4.wt103.production-attempt-spec.v1",
            {
                "ordinal": self.ordinal,
                "role": self.role,
                "arm_id": self.arm_id,
                "seed_id": self.seed_id,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "pass_count": self.pass_count,
                "quarter_pass": self.quarter_pass,
            },
        )
        if self.attempt_sha256 != expected:
            raise ProductionOperationError(
                "production attempt identity changed"
            )


@dataclass(frozen=True, slots=True)
class ProductionAttemptOutcome:
    attempt_sha256: str
    validation_nll_sum: float
    validation_counted_targets: int
    validation_nll_per_token: float
    accepted_updates: int
    terminal_checkpoint_identity_sha256: str | None
    metrics_jsonl_sha256: str
    metrics_csv_sha256: str
    outcome_sha256: str

    @classmethod
    def create(cls, **values: object) -> "ProductionAttemptOutcome":
        digest = owned_sha256(
            "vfe4.wt103.production-attempt-outcome.v1",
            values,
        )
        return cls(**values, outcome_sha256=digest)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if (
            type(self.validation_nll_sum) is not float
            or not math.isfinite(self.validation_nll_sum)
            or self.validation_nll_sum < 0.0
            or type(self.validation_counted_targets) is not int
            or self.validation_counted_targets <= 0
            or self.validation_nll_per_token
            != self.validation_nll_sum / self.validation_counted_targets
            or type(self.accepted_updates) is not int
            or self.accepted_updates <= 0
        ):
            raise ProductionOperationError(
                "production attempt outcome is invalid"
            )
        expected = owned_sha256(
            "vfe4.wt103.production-attempt-outcome.v1",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "outcome_sha256"
            },
        )
        if self.outcome_sha256 != expected:
            raise ProductionOperationError(
                "production attempt outcome identity changed"
            )


@dataclass(frozen=True, slots=True)
class _ValidationCacheAudit:
    schema_version: Literal["wt103-production-cache-audit-v1"]
    window_manifest_sha256: str
    schedule_sha256: str
    estimator_stream_sha256: str
    cold_records_sha256: str
    warm_records_sha256: str
    reverse_records_sha256: str
    cold_nll_terms_sha256: str
    warm_nll_terms_sha256: str
    reverse_nll_terms_sha256: str
    summed_nll: float
    prefix_record_count: int
    counted_targets: int
    passed: Literal[True]
    audit_sha256: str

    @classmethod
    def create(
        cls,
        *,
        window_manifest_sha256: str,
        schedule_sha256: str,
        estimator_stream_sha256: str,
        records_sha256: str,
        nll_terms_sha256: str,
        summed_nll: float,
        prefix_record_count: int,
        counted_targets: int,
    ) -> "_ValidationCacheAudit":
        payload = {
            "schema_version": "wt103-production-cache-audit-v1",
            "window_manifest_sha256": window_manifest_sha256,
            "schedule_sha256": schedule_sha256,
            "estimator_stream_sha256": estimator_stream_sha256,
            "cold_records_sha256": records_sha256,
            "warm_records_sha256": records_sha256,
            "reverse_records_sha256": records_sha256,
            "cold_nll_terms_sha256": nll_terms_sha256,
            "warm_nll_terms_sha256": nll_terms_sha256,
            "reverse_nll_terms_sha256": nll_terms_sha256,
            "summed_nll": summed_nll,
            "prefix_record_count": prefix_record_count,
            "counted_targets": counted_targets,
            "passed": True,
        }
        return cls(
            **payload,
            audit_sha256=owned_sha256(
                "vfe4.wt103.production-cache-audit.v1",
                payload,
            ),
        )  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        hashes = (
            self.window_manifest_sha256,
            self.schedule_sha256,
            self.estimator_stream_sha256,
            self.cold_records_sha256,
            self.warm_records_sha256,
            self.reverse_records_sha256,
            self.cold_nll_terms_sha256,
            self.warm_nll_terms_sha256,
            self.reverse_nll_terms_sha256,
            self.audit_sha256,
        )
        if (
            self.schema_version
            != "wt103-production-cache-audit-v1"
            or any(
                type(value) is not str
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in hashes
            )
            or not (
                self.cold_records_sha256
                == self.warm_records_sha256
                == self.reverse_records_sha256
            )
            or not (
                self.cold_nll_terms_sha256
                == self.warm_nll_terms_sha256
                == self.reverse_nll_terms_sha256
            )
            or type(self.summed_nll) is not float
            or not math.isfinite(self.summed_nll)
            or self.summed_nll < 0.0
            or type(self.prefix_record_count) is not int
            or self.prefix_record_count <= 0
            or type(self.counted_targets) is not int
            or self.counted_targets <= 0
            or self.prefix_record_count != self.counted_targets
            or self.passed is not True
        ):
            raise ProductionOperationError(
                "production cache audit record is invalid"
            )
        expected = owned_sha256(
            "vfe4.wt103.production-cache-audit.v1",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "audit_sha256"
            },
        )
        if self.audit_sha256 != expected:
            raise ProductionOperationError(
                "production cache audit identity changed"
            )


@dataclass(frozen=True, slots=True)
class _ValidationScore:
    summed_nll: float
    counted_targets: int
    cache_audit: _ValidationCacheAudit

    def __post_init__(self) -> None:
        if (
            type(self.summed_nll) is not float
            or not math.isfinite(self.summed_nll)
            or self.summed_nll < 0.0
            or type(self.counted_targets) is not int
            or self.counted_targets <= 0
            or type(self.cache_audit) is not _ValidationCacheAudit
            or self.cache_audit.counted_targets
            != self.counted_targets
            or self.cache_audit.summed_nll != self.summed_nll
        ):
            raise ProductionOperationError(
                "production validation score is invalid"
            )
        self.cache_audit.__post_init__()


@dataclass(frozen=True, slots=True)
class _AttemptResourceUsage:
    device_seconds: float
    wall_seconds: float
    sampled_energy_kwh: float
    usage_evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not float
                or not math.isfinite(value)
                or value < 0.0
                for value in (
                    self.device_seconds,
                    self.wall_seconds,
                    self.sampled_energy_kwh,
                )
            )
            or self.device_seconds > self.wall_seconds
            or type(self.usage_evidence_sha256) is not str
            or len(self.usage_evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.usage_evidence_sha256
            )
        ):
            raise ProductionOperationError(
                "attempt resource usage observation is invalid"
            )


def _emit_attempt_started(
    *,
    attempt: ProductionAttemptSpec,
    reserved: ReservedRun,
    phase_position: int,
    phase_total: int,
    resume_active: bool,
) -> None:
    """Report only after the run reservation is durably reopened."""

    if (
        type(phase_position) is not int
        or type(phase_total) is not int
        or not 1 <= phase_position <= phase_total
    ):
        raise ProductionOperationError(
            "attempt progress phase position is invalid"
        )
    emit_progress(
        "attempt_started",
        role=attempt.role,
        global_ordinal=attempt.ordinal,
        phase_position=phase_position,
        phase_total=phase_total,
        arm_id=attempt.arm_id,
        seed_id=attempt.seed_id,
        attempt_id=attempt.attempt_id,
        attempt_sha256=attempt.attempt_sha256,
        reservation_sha256=reserved.reservation_sha256,
        in_progress_path=str(reserved.inprogress_path),
        resume_count=reserved.resume_count,
        resume_active=resume_active,
    )


def _emit_attempt_started_or_release(
    *,
    attempt: ProductionAttemptSpec,
    reserved: ReservedRun,
    phase_position: int,
    phase_total: int,
    resume_active: bool,
) -> None:
    """Release the reserved execution lease if progress publication fails."""

    try:
        _emit_attempt_started(
            attempt=attempt,
            reserved=reserved,
            phase_position=phase_position,
            phase_total=phase_total,
            resume_active=resume_active,
        )
    except BaseException:
        release_run_execution_lease(reserved)
        raise


def _emit_attempt_finished(
    *,
    attempt: ProductionAttemptSpec,
    manifest: RunManifestIdentity,
    outcome: object | None,
    usage: _AttemptResourceUsage,
    ledger: ResourceUsageLedger,
) -> None:
    """Report direct measurements separately from durable ledger accounting."""

    emit_progress(
        "attempt_finished",
        role=attempt.role,
        global_ordinal=attempt.ordinal,
        arm_id=attempt.arm_id,
        seed_id=attempt.seed_id,
        attempt_id=attempt.attempt_id,
        disposition=manifest.disposition,
        manifest_path=str(manifest.run_path / "run-manifest.json"),
        manifest_sha256=manifest.manifest_sha256,
        terminal_checkpoint_identity_sha256=getattr(
            outcome,
            "terminal_checkpoint_identity_sha256",
            None,
        ),
        observed={
            "gpu_seconds": usage.device_seconds,
            "wall_seconds": usage.wall_seconds,
            "energy_kwh": usage.sampled_energy_kwh,
        },
        accounted={
            "gpu_hours": ledger.used_gpu_hours,
            "wall_hours": ledger.used_wall_hours,
            "energy_kwh": ledger.used_energy_kwh,
            "ledger_sha256": ledger.ledger_sha256,
        },
        observed_disk_bytes=None,
        observed_disk_bytes_status="unavailable",
    )


class _ResourceMeasuredOperationFailure(BaseException):
    """Carry an operation failure together with its completed usage sample."""

    def __init__(
        self,
        *,
        error: BaseException,
        usage: _AttemptResourceUsage,
        operation_completed: bool = True,
        operation_result: object = None,
        sampling_error: BaseException | None = None,
        operation_error: BaseException | None = None,
    ) -> None:
        if (
            not isinstance(error, BaseException)
            or type(operation_completed) is not bool
            or (
                sampling_error is not None
                and not isinstance(sampling_error, BaseException)
            )
            or (
                operation_error is not None
                and not isinstance(operation_error, BaseException)
            )
        ):
            raise ProductionOperationError(
                "resource-measured failure lost its original exception"
            )
        usage.__post_init__()
        super().__init__(str(error))
        self.error = error
        self.usage = usage
        self.operation_completed = operation_completed
        self.operation_result = operation_result
        self.sampling_error = sampling_error
        self.operation_error = operation_error


class _ResourceHeartbeatFailure(RuntimeError):
    """Abort scientific work without converting accounting loss to a run result."""


class _TerminalAttemptFailure(ProductionOperationError):
    """A scientific attempt failure that must close through failures.jsonl."""

    phase: str
    step: int
    pass_index: int
    scientific_state_advanced: bool

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        step: int,
        pass_index: int,
        scientific_state_advanced: bool,
    ) -> None:
        super().__init__(message)
        if (
            type(phase) is not str
            or not phase
            or type(step) is not int
            or step < 0
            or type(pass_index) is not int
            or pass_index < 0
            or type(scientific_state_advanced) is not bool
        ):
            raise ProductionOperationError(
                "terminal attempt failure context is invalid"
            )
        self.phase = phase
        self.step = step
        self.pass_index = pass_index
        self.scientific_state_advanced = scientific_state_advanced


@dataclass(frozen=True, slots=True)
class _TerminalAttemptResult:
    manifest: RunManifestIdentity
    failure: FailureRecord

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not RunManifestIdentity
            or self.manifest.disposition != "failure"
            or type(self.failure) is not FailureRecord
            or self.failure.record_sha256
            != _manifest_failure_record_sha256(self.manifest)
        ):
            raise ProductionOperationError(
                "terminal attempt result is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class ProductionTrainingResult:
    schema_version: Literal["wt103-production-training-result-v1"]
    mode: Literal["train", "resume"]
    config_sha256: str
    source_lock_sha256: str
    readiness_result_sha256: str
    experiment_plan_sha256: str
    experiment_index_path: str
    tuning_attempt_count: int
    confirmation_attempt_count: int
    completed_attempt_count: int
    selected_hyperparameter_sha256: str
    status: Literal["COMPLETE"]
    heldout_test_opened: Literal[False]
    result_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != _RESULT_SCHEMA
            or self.mode not in ("train", "resume")
            or self.status != "COMPLETE"
            or self.heldout_test_opened is not False
            or self.tuning_attempt_count <= 0
            or self.confirmation_attempt_count <= 0
            or self.completed_attempt_count
            != self.tuning_attempt_count + self.confirmation_attempt_count
        ):
            raise ProductionOperationError(
                "production training result is inconsistent"
            )
        expected = owned_sha256(
            "vfe4.wt103.production-training-result.v1",
            {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name != "result_sha256"
            },
        )
        if self.result_sha256 != expected:
            raise ProductionOperationError(
                "production training result identity changed"
            )


def _backend() -> DurabilityBackend:
    return (
        WindowsDurabilityBackend()
        if os.name == "nt"
        else PosixDurabilityBackend()
    )


def _artifact_record(
    path: Path,
    *,
    relative_path: str,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> ArtifactIntegrityRecord:
    payload = _regular_bytes(path, maximum_bytes=maximum_bytes)
    return ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _production_experiment_plan(
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
) -> ExperimentPlan:
    bundle = readiness.readiness_bundle
    if type(bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "production experiment plan requires exact Task 14 evidence"
        )
    bundle.__post_init__()
    provenance = bundle.provenance
    if (
        bundle.training_config_sha256
        != training.experiment_config_sha256
        or bundle.source_lock_sha256 != source_lock.source_lock_sha256
        or bundle.endpoint_inventory != training.endpoint_inventory
        or provenance.source_record_sha256
        != source_lock.finalized_source.record_sha256
        or provenance.tokenizer_spec_sha256
        != source_lock.tokenizer.spec_sha256
        or provenance.token_cache_set_sha256
        != source_lock.finalized_source.production_token_cache_set_sha256
        or provenance.schedule_set_sha256
        != source_lock.schedules.schedule_set_sha256
    ):
        raise ProductionOperationError(
            "Task 14 evidence differs from production plan authority"
        )
    return ExperimentPlan.create(
        experiment_id=(
            "wt103-production-"
            f"{training.experiment_config_sha256[:12]}-"
            f"{source_lock.finalized_source.record_sha256[:12]}"
        ),
        endpoint_inventory=bundle.endpoint_inventory,
        git_head=provenance.git_head,
        dirty_digest=provenance.dirty_digest,
        config_sha256=training.experiment_config_sha256,
        source_record_sha256=provenance.source_record_sha256,
        tokenizer_spec_sha256=provenance.tokenizer_spec_sha256,
        token_cache_set_sha256=provenance.token_cache_set_sha256,
        window_manifest_sha256s=tuple(
            manifest.manifest_sha256
            for manifest in source_lock.schedules.window_manifests
        ),
        schedule_set_sha256=provenance.schedule_set_sha256,
        factory_set_sha256=provenance.factory_set_sha256,
        objective_sha256=provenance.objective_sha256,
        checkpoint_schema_sha256=owned_sha256(
            "vfe4.wt103.production-checkpoint-schema.v1",
            {
                "schema_version": (
                    training.profile.schemas.checkpoint_schema
                ),
            },
        ),
        resource_forecast_sha256=bundle.resource_forecast.forecast_sha256,
        expected_run_artifact_paths=(
            _LIVE_PRECISION_RUNTIME_EVIDENCE_NAME,
            "metrics.csv",
            "metrics.jsonl",
        ),
        expected_group_artifact_paths=("result-table.json",),
    )


def _reopen_experiment_plan_identity(
    *,
    plan_path: Path,
    expected_plan: ExperimentPlan,
    readiness: ProductionReadinessResult,
) -> ExperimentPlanIdentity:
    bundle = readiness.readiness_bundle
    if type(bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "production plan reopen requires exact Task 14 evidence"
        )
    if (
        not isinstance(plan_path, Path)
        or plan_path.name != _PLAN_NAME
    ):
        raise ProductionOperationError(
            "resume plan path must name the exact experiment-plan.json"
        )
    payload = _regular_bytes(
        plan_path,
        maximum_bytes=_MAXIMUM_INDEX_BYTES,
    )
    expected_payload = canonical_json_bytes_generic(expected_plan)
    if payload != expected_payload:
        raise ProductionOperationError(
            "durable experiment plan differs from current authority"
        )
    durable = DurableFileIdentity.create(
        operation="exclusive_create",
        payload=payload,
        volume_identity=bundle.durability.volume_identity,
    )
    identity = ExperimentPlanIdentity(
        plan_path=plan_path,
        plan=expected_plan,
        durable_file=durable,
        identity_sha256=owned_sha256(
            "vfe4.wt103.experiment-plan-identity.v1",
            {
                "experiment_plan_sha256": (
                    expected_plan.experiment_plan_sha256
                ),
                "durable_file_identity_sha256": durable.identity_sha256,
            },
        ),
    )
    identity.__post_init__()
    return identity


def _allocator_event_counters() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise ProductionOperationError(
            "production allocator counters require CUDA"
        )
    statistics = torch.cuda.memory_stats(torch.cuda.current_device())
    required = ("num_alloc_retries", "num_ooms")
    if any(
        name not in statistics
        or type(statistics[name]) is not int
        or statistics[name] < 0
        for name in required
    ):
        raise ProductionOperationError(
            "CUDA allocator counters are unavailable"
        )
    return statistics["num_alloc_retries"], statistics["num_ooms"]


def _frozen_readiness_conservative_power_watts(
    *,
    resource_forecast: ResourceForecast,
    live_power_provider: PowerProviderIdentity,
) -> float:
    """Read the exact hashed readiness power bound and rebind its live provider."""

    if (
        type(resource_forecast) is not ResourceForecast
        or type(live_power_provider) is not PowerProviderIdentity
    ):
        raise ProductionOperationError(
            "readiness power-bound authority is not exact"
        )
    try:
        resource_forecast.__post_init__()
        live_power_provider.__post_init__()
    except ValueError as exc:
        raise ProductionOperationError(
            "readiness power-bound authority is invalid"
        ) from exc
    conservative_power_watts = (
        resource_forecast.conservative_power_watts
    )
    if (
        resource_forecast.status is not GateStatus.PASS
        or resource_forecast.power_provider_identity_sha256
        != live_power_provider.identity_sha256
        or resource_forecast.reported_power_limit_watts
        != live_power_provider.reported_power_limit_watts
        or type(conservative_power_watts) is not float
        or not math.isfinite(conservative_power_watts)
        or conservative_power_watts <= 0.0
    ):
        raise ProductionOperationError(
            "readiness forecast cannot authorize a conservative power bound"
        )
    return conservative_power_watts


def _ceil_positive_fraction_to_float(
    value: Fraction,
    *,
    name: str,
) -> float:
    if type(value) is not Fraction or value <= 0:
        raise ProductionOperationError(f"{name} exact value is invalid")
    rounded = float(value)
    if not math.isfinite(rounded) or rounded <= 0.0:
        raise ProductionOperationError(f"{name} exceeds binary64")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if (
        not math.isfinite(rounded)
        or Fraction.from_float(rounded) < value
    ):
        raise ProductionOperationError(
            f"{name} cannot be conservatively represented"
        )
    return rounded


def _conservative_limit_usage(
    *,
    attempt: ProductionAttemptSpec,
    measurement_kind: Literal[
        "prepaid_crash_tail_reserve",
        "independent_monotonic_heartbeat",
    ],
    interval_ordinal: int,
    interval_started_ns: int,
    interval_ended_ns: int,
    power_provider_identity_sha256: str,
    conservative_power_watts: float,
) -> _AttemptResourceUsage:
    attempt.__post_init__()
    if (
        measurement_kind
        not in (
            "prepaid_crash_tail_reserve",
            "independent_monotonic_heartbeat",
        )
        or type(interval_ordinal) is not int
        or interval_ordinal < 0
        or type(interval_started_ns) is not int
        or type(interval_ended_ns) is not int
        or interval_started_ns < 0
        or interval_ended_ns <= interval_started_ns
        or type(power_provider_identity_sha256) is not str
        or len(power_provider_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in power_provider_identity_sha256
        )
        or type(conservative_power_watts) is not float
        or not math.isfinite(conservative_power_watts)
        or conservative_power_watts <= 0.0
    ):
        raise ProductionOperationError(
            "conservative resource debit authority is invalid"
        )
    elapsed_ns = interval_ended_ns - interval_started_ns
    seconds = _ceil_positive_fraction_to_float(
        Fraction(elapsed_ns, 1_000_000_000),
        name="conservative elapsed seconds",
    )
    energy_kwh = _ceil_positive_fraction_to_float(
        Fraction.from_float(conservative_power_watts)
        * Fraction(elapsed_ns, 3_600_000_000_000_000),
        name="conservative energy kWh",
    )
    evidence_sha256 = owned_sha256(
        "vfe4.wt103.attempt-resource-usage-evidence.v1",
        {
            "measurement_kind": measurement_kind,
            "attempt_id": attempt.attempt_id,
            "attempt_sha256": attempt.attempt_sha256,
            "interval_ordinal": interval_ordinal,
            "interval_started_ns": interval_started_ns,
            "interval_ended_ns": interval_ended_ns,
            "device_seconds": seconds,
            "wall_seconds": seconds,
            "sampled_energy_kwh": energy_kwh,
            "power_provider_identity_sha256": (
                power_provider_identity_sha256
            ),
            "readiness_conservative_power_watts": (
                conservative_power_watts
            ),
            "accounting_policy": (
                "conservative_no_refund_overaccounting"
            ),
        },
    )
    usage = _AttemptResourceUsage(
        device_seconds=seconds,
        wall_seconds=seconds,
        sampled_energy_kwh=energy_kwh,
        usage_evidence_sha256=evidence_sha256,
    )
    usage.__post_init__()
    return usage


def _run_resource_usage_heartbeat(
    *,
    stop: object,
    attempt: ProductionAttemptSpec,
    power_provider_identity_sha256: str,
    conservative_power_watts: float,
    started_ns: int,
    debit: Callable[[_AttemptResourceUsage], None],
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> None:
    wait = getattr(stop, "wait", None)
    if (
        not callable(wait)
        or not callable(debit)
        or not callable(monotonic_ns)
        or type(started_ns) is not int
        or started_ns < 0
    ):
        raise ProductionOperationError(
            "resource heartbeat authority is invalid"
        )
    prior_ns = started_ns
    ordinal = 0
    while True:
        stopped = wait(_RESOURCE_USAGE_HEARTBEAT_SECONDS)
        if type(stopped) is not bool:
            raise ProductionOperationError(
                "resource heartbeat stop result is invalid"
            )
        if stopped:
            return
        observed_ns = monotonic_ns()
        usage = _conservative_limit_usage(
            attempt=attempt,
            measurement_kind="independent_monotonic_heartbeat",
            interval_ordinal=ordinal,
            interval_started_ns=prior_ns,
            interval_ended_ns=observed_ns,
            power_provider_identity_sha256=(
                power_provider_identity_sha256
            ),
            conservative_power_watts=conservative_power_watts,
        )
        debit(usage)
        prior_ns = observed_ns
        ordinal += 1


def _measure_attempt_resource_usage(
    operation,
    *,
    sampler: NvidiaSmiPowerSampler,
    power_provider_identity_sha256: str,
    conservative_power_watts: float,
) -> tuple[object, _AttemptResourceUsage]:
    if (
        not callable(operation)
        or type(sampler) is not NvidiaSmiPowerSampler
        or type(power_provider_identity_sha256) is not str
        or len(power_provider_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in power_provider_identity_sha256
        )
        or type(conservative_power_watts) is not float
        or not math.isfinite(conservative_power_watts)
        or conservative_power_watts <= 0.0
    ):
        raise ProductionOperationError(
            "attempt resource measurement authority is invalid"
        )
    sampler.__post_init__()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started_ns = time.perf_counter_ns()
    start_event.record()
    failures: list[BaseException] = []
    observations: list[PowerObservation] = []
    integrated_watt_nanoseconds = 0.0
    last_observation: PowerObservation | None = None

    def record_observation(observation: PowerObservation) -> None:
        nonlocal integrated_watt_nanoseconds
        nonlocal last_observation
        if type(observation) is not PowerObservation:
            raise ProductionOperationError(
                "power sampler returned a nonexact observation"
            )
        observation.__post_init__()
        prior_ns = (
            wall_started_ns
            if last_observation is None
            else last_observation.monotonic_ns
        )
        if observation.monotonic_ns < prior_ns:
            raise ProductionOperationError(
                "power observation clock moved backward"
            )
        interval_ns = observation.monotonic_ns - prior_ns
        interval_watts = (
            observation.watts
            if last_observation is None
            else last_observation.watts
        )
        integrated_watt_nanoseconds += interval_watts * interval_ns
        observations.append(observation)
        last_observation = observation

    def measured_operation() -> object:
        try:
            return operation()
        except BaseException as exc:
            failures.append(exc)
            return None

    sample_failure: PowerSampleOperationFailure | None = None
    try:
        result, power_samples = sampler.sample(
            measured_operation,
            on_observation=record_observation,
        )
    except PowerSampleOperationFailure as exc:
        sample_failure = exc
        if tuple(observations) != exc.observations:
            raise ProductionOperationError(
                "power sampler partial observations changed in transit"
            ) from exc
        result = exc.operation_result
        power_samples = tuple(
            observation.watts for observation in exc.observations
        )
    end_event.record()
    torch.cuda.synchronize()
    wall_ended_ns = time.perf_counter_ns()
    if (
        type(power_samples) is not tuple
        or not power_samples
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in power_samples
        )
        or tuple(item.watts for item in observations)
        != power_samples
        or last_observation is None
    ):
        raise ProductionOperationError(
            "attempt power samples are invalid"
        )
    if wall_ended_ns < last_observation.monotonic_ns:
        raise ProductionOperationError(
            "attempt wall clock ended before its final power observation"
        )
    integrated_watt_nanoseconds += last_observation.watts * (
        wall_ended_ns - last_observation.monotonic_ns
    )
    wall_seconds = (wall_ended_ns - wall_started_ns) / 1_000_000_000.0
    observed_device_seconds = (
        float(start_event.elapsed_time(end_event)) / 1_000.0
    )
    energy_kwh = (
        integrated_watt_nanoseconds / 3_600_000_000_000_000.0
    )
    evidence_sha256 = owned_sha256(
        "vfe4.wt103.attempt-resource-usage-evidence.v1",
        {
            "measurement_kind": "final_measured_no_refund_interval",
            "power_provider_identity_sha256": (
                power_provider_identity_sha256
            ),
            "readiness_conservative_power_watts": (
                conservative_power_watts
            ),
            "power_samples_watts": power_samples,
            "power_observations": tuple(
                (item.watts, item.monotonic_ns)
                for item in observations
            ),
            "sample_interval_seconds": sampler.sample_interval_seconds,
            "wall_started_ns": wall_started_ns,
            "wall_ended_ns": wall_ended_ns,
            "observed_device_seconds": observed_device_seconds,
            "device_seconds": observed_device_seconds,
            "sampled_energy_kwh": energy_kwh,
            "safety_accounting": (
                "prepaid_limit_reserve_plus_limit_heartbeat_no_refund"
            ),
        },
    )
    usage = _AttemptResourceUsage(
        device_seconds=observed_device_seconds,
        wall_seconds=wall_seconds,
        sampled_energy_kwh=energy_kwh,
        usage_evidence_sha256=evidence_sha256,
    )
    usage.__post_init__()
    if failures or sample_failure is not None:
        operation_error = failures[0] if failures else (
            sample_failure.operation_error
            if sample_failure is not None
            else None
        )
        sampling_error = (
            sample_failure.sampling_error
            if sample_failure is not None
            else None
        )
        error: BaseException | None = (
            sample_failure
            if sample_failure is not None
            else operation_error
        )
        if error is None:
            raise ProductionOperationError(
                "resource-measured failure lost both failure causes"
            )
        raise _ResourceMeasuredOperationFailure(
            error=error,
            usage=usage,
            operation_completed=(
                sample_failure.operation_completed
                if sample_failure is not None
                else True
            ),
            operation_result=result,
            sampling_error=sampling_error,
            operation_error=operation_error,
        ) from error
    return result, usage


def _publish_resource_usage_ledger(
    *,
    path: Path,
    ledger: ResourceUsageLedger,
    backend: DurabilityBackend,
) -> None:
    ledger.__post_init__()
    payload = canonical_json_bytes_generic(ledger)
    if path.exists():
        backend.replace_durable(path, payload)
    else:
        backend.create_exclusive(path, payload)
    if _regular_bytes(path, maximum_bytes=_MAXIMUM_INDEX_BYTES) != payload:
        raise ProductionOperationError(
            "resource usage ledger changed on durable reopen"
        )


def _reopen_resource_usage_ledger(
    *,
    path: Path,
    experiment_plan_sha256: str,
) -> ResourceUsageLedger:
    document = _canonical_document(path)
    expected_keys = set(ResourceUsageLedger.__dataclass_fields__)
    if set(document) != expected_keys or type(document.get("events")) is not list:
        raise ProductionOperationError(
            "resource usage ledger has an open or malformed schema"
        )
    events = []
    for raw in document["events"]:
        if type(raw) is not dict or set(raw) != set(
            ResourceUsageEvent.__dataclass_fields__
        ):
            raise ProductionOperationError(
                "resource usage event has an open or malformed schema"
            )
        events.append(ResourceUsageEvent(**raw))
    normalized = dict(document)
    normalized["events"] = tuple(events)
    try:
        ledger = ResourceUsageLedger(**normalized)
        ledger.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ProductionOperationError(
            "resource usage ledger failed typed reopen"
        ) from exc
    if ledger.experiment_plan_sha256 != experiment_plan_sha256:
        raise ProductionOperationError(
            "resource usage ledger belongs to another experiment plan"
        )
    return ledger


def _debit_resource_usage(
    *,
    ledger: ResourceUsageLedger,
    attempt: ProductionAttemptSpec,
    usage: _AttemptResourceUsage,
    path: Path,
    backend: DurabilityBackend,
) -> ResourceUsageLedger:
    ledger.__post_init__()
    attempt.__post_init__()
    usage.__post_init__()
    prior_segments = tuple(
        event
        for event in ledger.events
        if event.attempt_id == attempt.attempt_id
    )
    if tuple(event.segment_ordinal for event in prior_segments) != tuple(
        range(len(prior_segments))
    ):
        raise ProductionOperationError(
            "resource usage ledger has a discontinuous attempt segment"
        )
    updated = ledger.append(
        ResourceUsageEvent.create(
            attempt_id=attempt.attempt_id,
            segment_ordinal=len(prior_segments),
            device_seconds=usage.device_seconds,
            wall_seconds=usage.wall_seconds,
            sampled_energy_kwh=usage.sampled_energy_kwh,
            usage_evidence_sha256=usage.usage_evidence_sha256,
        )
    )
    _assert_resource_headroom(updated)
    _publish_resource_usage_ledger(
        path=path,
        ledger=updated,
        backend=backend,
    )
    return updated


def _assert_resource_headroom(ledger: ResourceUsageLedger) -> None:
    ledger.__post_init__()
    factor = ledger.forecast_headroom_factor
    exceeded = tuple(
        name
        for name, used, maximum in (
            (
                "gpu_hours",
                ledger.used_gpu_hours,
                ledger.maximum_gpu_hours,
            ),
            (
                "wall_hours",
                ledger.used_wall_hours,
                ledger.maximum_wall_hours,
            ),
            (
                "energy_kwh",
                ledger.used_energy_kwh,
                ledger.maximum_energy_kwh,
            ),
        )
        if used * factor > maximum
    )
    if exceeded:
        raise ProductionOperationError(
            "resource usage plus frozen headroom exceeded: "
            + ",".join(exceeded)
        )


def _require_nonreparse_path_chain(
    path: Path,
    *,
    name: str,
    final_kind: Literal["directory", "file"],
) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or ".." in path.parts
        or Path(os.path.abspath(path)) != path
    ):
        raise ProductionOperationError(
            f"{name} must be an absolute normalized path"
        )
    current = Path(path.anchor)
    components = [current]
    for part in path.parts[1:]:
        current = current / part
        components.append(current)
    for index, component in enumerate(components):
        try:
            metadata = component.lstat()
            is_junction = getattr(component, "is_junction", None)
            junction = bool(
                is_junction is not None and is_junction()
            )
        except OSError as exc:
            raise ProductionOperationError(
                f"{name} component metadata is unavailable: {component}"
            ) from exc
        reparse = bool(
            getattr(metadata, "st_file_attributes", 0) & 0x400
        )
        if stat.S_ISLNK(metadata.st_mode) or reparse or junction:
            raise ProductionOperationError(
                f"{name} cannot traverse a symlink, junction, or "
                f"reparse point: {component}"
            )
        final = index == len(components) - 1
        expected_directory = not final or final_kind == "directory"
        if expected_directory and not stat.S_ISDIR(metadata.st_mode):
            raise ProductionOperationError(
                f"{name} component is not a regular directory: {component}"
            )
        if (
            final
            and final_kind == "file"
            and not stat.S_ISREG(metadata.st_mode)
        ):
            raise ProductionOperationError(
                f"{name} is not a regular file: {component}"
            )


def _validate_resume_state_boundary(
    *,
    run_root: Path,
    experiment_root: Path,
    declared_plan: Path,
) -> None:
    if (
        declared_plan.name != _PLAN_NAME
        or declared_plan.parent != experiment_root
        or experiment_root.parent != run_root
    ):
        raise ProductionOperationError(
            "resume authority is outside the exact experiment root"
        )
    _require_nonreparse_path_chain(
        run_root,
        name="resume run_root",
        final_kind="directory",
    )
    _require_nonreparse_path_chain(
        experiment_root,
        name="resume experiment_root",
        final_kind="directory",
    )
    _require_nonreparse_path_chain(
        declared_plan,
        name="resume experiment-plan authority",
        final_kind="file",
    )


def _regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionOperationError(
            f"production state artifact is unavailable: {path}"
        ) from exc
    reparse = getattr(metadata, "st_file_attributes", 0) & 0x400
    is_junction = getattr(path, "is_junction", None)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or reparse
        or bool(is_junction is not None and is_junction())
        or not 0 < metadata.st_size <= maximum_bytes
    ):
        raise ProductionOperationError(
            "production state artifact is not a bounded regular file"
        )
    return path.read_bytes()


def _canonical_document(path: Path) -> dict[str, object]:
    raw = _regular_bytes(path, maximum_bytes=_MAXIMUM_INDEX_BYTES)
    try:
        document = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionOperationError(
            "production experiment index is invalid JSON"
        ) from exc
    if type(document) is not dict or raw != canonical_json_bytes_generic(
        document
    ):
        raise ProductionOperationError(
            "production experiment index is not canonical"
        )
    return document


def _publish_document(
    backend: DurabilityBackend,
    path: Path,
    document: Mapping[str, object],
) -> None:
    payload = canonical_json_bytes_generic(dict(document))
    backend.publish_bytes(path, payload)
    if _regular_bytes(path, maximum_bytes=_MAXIMUM_INDEX_BYTES) != payload:
        raise ProductionOperationError(
            "production state publication changed on reopen"
        )


def _attempt_inventory(
    training: TrainingConfig,
    selected: Mapping[str, tuple[float, float]] | None,
) -> tuple[ProductionAttemptSpec, ...]:
    attempts: list[ProductionAttemptSpec] = []
    ordinal = 0
    if selected is None:
        for arm in training.endpoint_inventory.arms:
            for learning_rate in training.profile.statistics.learning_rate_grid:
                for weight_decay in training.profile.statistics.weight_decay_grid:
                    for seed in training.profile.statistics.tuning_seed_ids:
                        attempts.append(
                            ProductionAttemptSpec.create(
                                ordinal=ordinal,
                                role="tuning",
                                arm_id=arm.arm_id,
                                seed_id=seed,
                                learning_rate=float(learning_rate),
                                weight_decay=float(weight_decay),
                                pass_count=1,
                                quarter_pass=True,
                            )
                        )
                        ordinal += 1
    else:
        ordinal = (
            len(training.endpoint_inventory.arms)
            * len(training.profile.statistics.learning_rate_grid)
            * len(training.profile.statistics.weight_decay_grid)
            * len(training.profile.statistics.tuning_seed_ids)
        )
        for arm in training.endpoint_inventory.arms:
            learning_rate, weight_decay = selected[arm.arm_id]
            for seed in training.profile.statistics.confirmatory_seed_ids:
                attempts.append(
                    ProductionAttemptSpec.create(
                        ordinal=ordinal,
                        role="confirmation",
                        arm_id=arm.arm_id,
                        seed_id=seed,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        pass_count=training.profile.cadence.confirmatory_passes,
                        quarter_pass=False,
                    )
                )
                ordinal += 1
    return tuple(attempts)


def _experiment_plan_payload(
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
) -> dict[str, object]:
    tuning_attempts = _attempt_inventory(training, None)
    return {
        "schema_version": _PLAN_SCHEMA,
        "config_sha256": training.experiment_config_sha256,
        "source_lock_sha256": source_lock.source_lock_sha256,
        "readiness_result_sha256": readiness.result_sha256,
        "readiness_token_sha256": getattr(
            readiness.readiness_token,
            "token_sha256",
        ),
        "endpoint_inventory_sha256": (
            training.endpoint_inventory.endpoint_inventory_sha256
        ),
        "tuning_attempts": [
            {
                field.name: getattr(attempt, field.name)
                for field in dataclasses.fields(attempt)
            }
            for attempt in tuning_attempts
        ],
        "confirmatory_seed_ids": list(
            training.profile.statistics.confirmatory_seed_ids
        ),
        "confirmatory_passes": (
            training.profile.cadence.confirmatory_passes
        ),
        "selection_rule": (
            "minimum_mean_full_validation_prior_nll_per_token_then_"
            "lower_learning_rate_then_lower_weight_decay"
        ),
        "test_opening": False,
    }


def _experiment_plan_document(
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
) -> dict[str, object]:
    payload = _experiment_plan_payload(
        training=training,
        source_lock=source_lock,
        readiness=readiness,
    )
    return {
        **payload,
        "experiment_plan_sha256": owned_sha256(
            "vfe4.wt103.production-experiment-plan.v1",
            payload,
        ),
    }


def _validate_experiment_plan(
    document: dict[str, object],
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
) -> str:
    expected = _experiment_plan_document(
        training=training,
        source_lock=source_lock,
        readiness=readiness,
    )
    if document != expected:
        raise ProductionOperationError(
            "durable experiment plan differs from current authority"
        )
    digest = document["experiment_plan_sha256"]
    if type(digest) is not str:
        raise ProductionOperationError(
            "durable experiment-plan identity is malformed"
        )
    return digest


def _new_index(
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    plan_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": _INDEX_SCHEMA,
        "config_sha256": training.experiment_config_sha256,
        "source_lock_sha256": source_lock.source_lock_sha256,
        "readiness_result_sha256": readiness.result_sha256,
        "readiness_token_sha256": getattr(
            readiness.readiness_token,
            "token_sha256",
        ),
        "experiment_plan_sha256": plan_sha256,
        "stage": "tuning",
        "active_attempt_sha256": None,
        "completed_outcomes": [],
        "selected_hyperparameters": {},
        "heldout_test_opened": False,
    }


def _validate_index(
    document: dict[str, object],
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    plan_sha256: str,
) -> None:
    expected = {
        "schema_version",
        "config_sha256",
        "source_lock_sha256",
        "readiness_result_sha256",
        "readiness_token_sha256",
        "experiment_plan_sha256",
        "stage",
        "active_attempt_sha256",
        "completed_outcomes",
        "selected_hyperparameters",
        "heldout_test_opened",
    }
    if (
        set(document) != expected
        or document["schema_version"] != _INDEX_SCHEMA
        or document["config_sha256"]
        != training.experiment_config_sha256
        or document["source_lock_sha256"]
        != source_lock.source_lock_sha256
        or document["readiness_result_sha256"] != readiness.result_sha256
        or document["readiness_token_sha256"]
        != getattr(readiness.readiness_token, "token_sha256")
        or document["experiment_plan_sha256"] != plan_sha256
        or document["stage"] not in ("tuning", "confirmation", "complete")
        or type(document["completed_outcomes"]) is not list
        or type(document["selected_hyperparameters"]) is not dict
        or document["heldout_test_opened"] is not False
    ):
        raise ProductionOperationError(
            "production experiment index differs from current authority"
        )


def _estimator_stream(bundle: WT103ArmRuntimeBundle) -> EstimatorStream:
    logical_stream_id = (
        None if bundle.scorer_kind == "exact_autoregressive" else 0
    )
    protocol = EstimatorProtocol.create()
    seed = wt103_estimator_stream_seed(
        split="validation",
        estimator_protocol_sha256=protocol.protocol_sha256,
        logical_stream_id=logical_stream_id,
    )
    return EstimatorStream.create(
        stream_seed=seed,
        estimator_identity=bundle.estimator_identity,
    )


def _prediction_fingerprint(
    prediction: object,
) -> tuple[str, str, str]:
    log_probs = getattr(prediction, "log_probs", None)
    cache = getattr(prediction, "cache", None)
    estimator_record = getattr(prediction, "estimator_record", None)
    values = (
        getattr(log_probs, "raw_bytes_sha256", None),
        getattr(cache, "cache_sha256", None),
        getattr(estimator_record, "record_sha256", None),
    )
    if any(
        type(value) is not str
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
        for value in values
    ):
        raise ProductionOperationError(
            "validation prediction omitted exact cache-audit identities"
        )
    return values  # type: ignore[return-value]


def _no_resource_abort() -> None:
    return None


def _validation_prediction_pass(
    *,
    predictor: object,
    stream: EstimatorStream,
    bundle: WT103ArmRuntimeBundle,
    batches: tuple[object, ...],
    mode: Literal["cold", "warm", "reverse"],
    resource_abort: Callable[[], None] = _no_resource_abort,
) -> tuple[
    dict[tuple[int, int], tuple[str, str, str]],
    dict[tuple[int, int], float],
]:
    if not callable(resource_abort):
        raise ProductionOperationError(
            "validation resource-abort authority is invalid"
        )
    row_inventory = []
    for batch in batches:
        resource_abort()
        for row, window_id in enumerate(batch.window_ids):
            resource_abort()
            row_inventory.append(
                (
                    int(window_id),
                    batch.inputs[row],
                    batch.targets[row],
                    batch.attention_mask[row],
                )
            )
    rows = tuple(row_inventory)
    traversed = rows if mode != "reverse" else tuple(reversed(rows))
    fingerprints: dict[tuple[int, int], tuple[str, str, str]] = {}
    negative_log_terms: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for window_id, inputs, targets, attention_mask in traversed:
            resource_abort()
            count = int(attention_mask.sum().item())
            positions = tuple(range(count))
            if mode == "reverse":
                positions = tuple(reversed(positions))
            cache = None
            for position in positions:
                resource_abort()
                history = [
                    int(inputs[0].item()),
                    *(
                        int(targets[index].item())
                        for index in range(position)
                    ),
                ]
                prefix = CausalPrefix.create(
                    receiver_t=len(history) + 1,
                    vocabulary=bundle.vocabulary,
                    token_ids=torch.tensor(
                        history,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                )
                prediction = predictor.next_token_log_probs(
                    prefix,
                    stream,
                    cache if mode == "warm" else None,
                )
                if mode == "warm":
                    cache = prediction.cache
                key = (window_id, position)
                if key in fingerprints:
                    raise ProductionOperationError(
                        "validation cache audit repeated a target key"
                    )
                fingerprints[key] = _prediction_fingerprint(prediction)
                target = int(targets[position].item())
                selected = float(
                    prediction.log_probs.value()[target].item()
                )
                if not math.isfinite(selected):
                    raise ProductionOperationError(
                        "validation scorer emitted a nonfinite log probability"
                    )
                negative_log_terms[key] = -selected
    return fingerprints, negative_log_terms


def _score_validation(
    *,
    bundle: WT103ArmRuntimeBundle,
    windows: ProductionWindowSet,
    schedule: object,
    resource_abort: Callable[[], None] = _no_resource_abort,
) -> _ValidationScore:
    if not callable(resource_abort):
        raise ProductionOperationError(
            "validation resource-abort authority is invalid"
        )
    predictor = bundle.make_predictor()
    stream = _estimator_stream(bundle)
    batch_rows = []
    for batch in iter_production_batches(
        windows=windows,
        schedule=schedule,  # type: ignore[arg-type]
    ):
        resource_abort()
        batch_rows.append(batch)
    batches = tuple(batch_rows)
    resource_abort()
    cold, cold_terms = _validation_prediction_pass(
        predictor=predictor,
        stream=stream,
        bundle=bundle,
        batches=batches,
        mode="cold",
        resource_abort=resource_abort,
    )
    resource_abort()
    warm, warm_terms = _validation_prediction_pass(
        predictor=predictor,
        stream=stream,
        bundle=bundle,
        batches=batches,
        mode="warm",
        resource_abort=resource_abort,
    )
    resource_abort()
    reverse, reverse_terms = _validation_prediction_pass(
        predictor=predictor,
        stream=stream,
        bundle=bundle,
        batches=batches,
        mode="reverse",
        resource_abort=resource_abort,
    )
    resource_abort()
    if (
        cold != warm
        or cold != reverse
        or cold_terms != warm_terms
        or cold_terms != reverse_terms
    ):
        raise ProductionOperationError(
            "cold, warm, and reverse validation cache audits differ"
        )
    ordered_keys = tuple(sorted(cold))
    counted = len(ordered_keys)
    if (
        counted != windows.manifest.counted_targets
        or set(ordered_keys) != set(cold_terms)
    ):
        raise ProductionOperationError(
            "validation scorer changed the frozen target denominator"
        )
    record_inventory = tuple(
        (window_id, position, *cold[(window_id, position)])
        for window_id, position in ordered_keys
    )
    records_sha256 = owned_sha256(
        "vfe4.wt103.production-cache-audit-records.v1",
        record_inventory,
    )
    nll_term_inventory = tuple(
        (window_id, position, warm_terms[(window_id, position)])
        for window_id, position in ordered_keys
    )
    nll_terms_sha256 = owned_sha256(
        "vfe4.wt103.production-cache-audit-nll-terms.v1",
        nll_term_inventory,
    )
    summed_nll = float(
        math.fsum(warm_terms[key] for key in ordered_keys)
    )
    manifest_sha256 = getattr(
        windows.manifest,
        "manifest_sha256",
        None,
    )
    schedule_sha256 = getattr(schedule, "schedule_sha256", None)
    if type(manifest_sha256) is not str or type(schedule_sha256) is not str:
        raise ProductionOperationError(
            "validation cache audit lost manifest or schedule identity"
        )
    audit = _ValidationCacheAudit.create(
        window_manifest_sha256=manifest_sha256,
        schedule_sha256=schedule_sha256,
        estimator_stream_sha256=stream.stream_sha256,
        records_sha256=records_sha256,
        nll_terms_sha256=nll_terms_sha256,
        summed_nll=summed_nll,
        prefix_record_count=counted,
        counted_targets=counted,
    )
    score = _ValidationScore(
        summed_nll=summed_nll,
        counted_targets=counted,
        cache_audit=audit,
    )
    score.__post_init__()
    return score


def _canonical_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _measured_metric(
    name: str,
    value: float,
    *,
    numerator: float | None = None,
    denominator: int | None = None,
    reason: str,
) -> MetricValue:
    if type(value) is not float or not math.isfinite(value):
        raise ProductionOperationError(
            f"production metric {name!r} is nonfinite or nonexact"
        )
    return applicable_metric(
        name=name,
        numerator=numerator,
        denominator=denominator,
        value=value,
        units=metric_family_units(name),
        reason=reason,
    )


def _arm_inapplicable_metrics(
    arm_spec: WT103ArmSpec,
) -> dict[str, MetricValue]:
    values: dict[str, MetricValue] = {}
    for name in WT103_REQUIRED_METRIC_FAMILIES:
        applicable, reason = metric_family_applicability(arm_spec, name)
        if not applicable:
            values[name] = not_applicable_metric(
                name=name,
                reason=reason,
                units=metric_family_units(name),
            )
    return values


def _seconds(nanoseconds: int) -> float:
    if type(nanoseconds) is not int or nanoseconds < 0:
        raise ProductionOperationError(
            "production phase duration must be nonnegative nanoseconds"
        )
    return nanoseconds / 1_000_000_000.0


def _proposal_objective_numerator(
    *,
    objective_kind: str,
    terms: tuple[tuple[str, float], ...],
) -> float:
    if (
        type(terms) is not tuple
        or not terms
        or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not float
            or not math.isfinite(row[1])
            for row in terms
        )
    ):
        raise ProductionOperationError(
            "proposal objective terms are not exact raw observations"
        )
    values = dict(terms)
    if len(values) != len(terms):
        raise ProductionOperationError(
            "proposal objective terms contain duplicate names"
        )
    if objective_kind == "cross_entropy":
        numerator = values.get("cross_entropy_value")
        if set(values) != {"cross_entropy_value"}:
            raise ProductionOperationError(
                "cross-entropy proposal terms changed schema"
            )
        assert numerator is not None
        numerator = -numerator
    elif objective_kind == "complete_elbo":
        numerator = values.get("complete_elbo_numerator")
        if numerator is None:
            raise ProductionOperationError(
                "complete-ELBO proposal omitted its raw numerator"
            )
    elif objective_kind == "emission_only_ablation_non_elbo":
        numerator = values.get("emission_only_non_elbo")
        if numerator is None:
            raise ProductionOperationError(
                "emission-only proposal omitted its raw numerator"
            )
    else:
        raise ProductionOperationError(
            "proposal objective kind is outside the frozen inventory"
        )
    if not math.isfinite(numerator):
        raise ProductionOperationError(
            "proposal objective numerator is nonfinite"
        )
    return float(numerator)


def _train_metric_values(
    *,
    arm_spec: WT103ArmSpec,
    step: StepResult,
    timing: PhaseTimingObservation,
    memory: MemoryObservation,
    allocation_retries: int,
    oom_count: int,
    source: SourceObservation | None,
    numerical: NumericalObservation | None,
) -> tuple[MetricValue, ...]:
    """Project one accepted train step from raw, actually observed values."""

    if (
        type(arm_spec) is not WT103ArmSpec
        or type(step) is not StepResult
        or type(timing) is not PhaseTimingObservation
        or type(memory) is not MemoryObservation
        or step.arm_id != arm_spec.arm_id
        or not step.accepted
        or not step.objective_diagnostics_applicable
        or type(step.objective_terms) is not dict
        or type(step.counted_targets) is not int
        or step.counted_targets <= 0
        or type(allocation_retries) is not int
        or allocation_retries < 0
        or type(oom_count) is not int
        or oom_count < 0
    ):
        raise ProductionOperationError(
            "production train metric observation is incomplete"
        )
    arm_spec.__post_init__()
    step.__post_init__()
    timing.__post_init__()
    memory.__post_init__()
    if arm_spec.latent_enabled:
        if (
            type(source) is not SourceObservation
            or type(numerical) is not NumericalObservation
        ):
            raise ProductionOperationError(
                "latent production step omitted source or numerical evidence"
            )
        source.__post_init__()
        numerical.__post_init__()
    elif source is not None or numerical is not None:
        raise ProductionOperationError(
            "nonlatent production step cannot fabricate latent diagnostics"
        )

    try:
        projected = project_objective_metrics(
            objective_kind=step.objective_kind,
            objective_terms=step.objective_terms,
            complete_elbo_numerator=step.complete_elbo_numerator,
            complete_elbo_value=step.complete_elbo_value,
            counted_targets=step.counted_targets,
        )
    except ProductionObservationError as exc:
        raise ProductionOperationError(
            "production objective metric projection failed"
        ) from exc

    values = _arm_inapplicable_metrics(arm_spec)
    for name, metric in projected.items():
        applicable, _reason = metric_family_applicability(arm_spec, name)
        if not applicable:
            raise ProductionOperationError(
                "objective projection disagrees with arm applicability"
            )
        values[name] = _measured_metric(
            name,
            metric.value,
            numerator=metric.numerator,
            denominator=metric.denominator,
            reason=(
                WT103_SOURCE_KL_DIAGNOSTIC_REASON
                if name in {"model_source_kl", "state_source_kl"}
                else "train_objective_projection"
            ),
        )
    if (
        step.objective_kind == "complete_elbo"
        and "estimator_error_bound" not in projected
    ):
        values["estimator_error_bound"] = not_applicable_metric(
            name="estimator_error_bound",
            reason=WT103_UNAVAILABLE_ESTIMATOR_BOUND_REASON,
            units=metric_family_units("estimator_error_bound"),
        )

    proposal_count = len(step.updates)
    accepted_count = sum(item.accepted for item in step.updates)
    rejected_count = proposal_count - accepted_count
    first_evidence = step.proposal_evidence[0]
    last_evidence = step.proposal_evidence[-1]
    if (
        first_evidence.objective_before_terms is None
        or last_evidence.objective_after_terms is None
        or first_evidence.objective_before_value is None
        or last_evidence.objective_after_value is None
        or first_evidence.counted_targets != step.counted_targets
        or last_evidence.counted_targets != step.counted_targets
    ):
        raise ProductionOperationError(
            "proposal evidence omitted objective before/after values"
        )
    counted = step.counted_targets
    before_numerator = _proposal_objective_numerator(
        objective_kind=step.objective_kind,
        terms=first_evidence.objective_before_terms,
    )
    after_numerator = _proposal_objective_numerator(
        objective_kind=step.objective_kind,
        terms=last_evidence.objective_after_terms,
    )
    values.update(
        {
            "accepted_proposals": _measured_metric(
                "accepted_proposals",
                float(accepted_count),
                reason="update_record_count",
            ),
            "rejected_proposals": _measured_metric(
                "rejected_proposals",
                float(rejected_count),
                reason="update_record_count",
            ),
            "acceptance_rate": _measured_metric(
                "acceptance_rate",
                accepted_count / proposal_count,
                numerator=float(accepted_count),
                denominator=proposal_count,
                reason="accepted_proposals/total_proposals",
            ),
            "objective_before": _measured_metric(
                "objective_before",
                before_numerator / counted,
                numerator=before_numerator,
                denominator=counted,
                reason="first_proposal_raw_objective_terms/count",
            ),
            "objective_after": _measured_metric(
                "objective_after",
                after_numerator / counted,
                numerator=after_numerator,
                denominator=counted,
                reason="last_proposal_raw_objective_terms/count",
            ),
        }
    )
    if arm_spec.latent_enabled:
        values["snapshot_identity_present"] = _measured_metric(
            "snapshot_identity_present",
            float(step.snapshot_sha256 is not None),
            reason="step_snapshot_identity_observation",
        )

    controls = step.update_controls
    if (
        any(
            control.gradient_norm_applicability != "applicable"
            for control in controls
        )
        or len({control.learning_rate for control in controls}) != 1
        or len({control.scheduler_ordinal for control in controls}) != 1
    ):
        raise ProductionOperationError(
            "production update controls cannot be projected exactly"
        )
    pre_l2 = math.sqrt(
        math.fsum(
            float(control.pre_clip_norm) ** 2 for control in controls
        )
    )
    post_l2 = math.sqrt(
        math.fsum(
            float(control.post_clip_norm) ** 2 for control in controls
        )
    )
    post_inf = max(
        float(control.post_clip_inf_norm) for control in controls
    )
    control = controls[-1]
    values.update(
        {
            "learning_rate": _measured_metric(
                "learning_rate",
                control.learning_rate,
                reason="update_control_v2",
            ),
            "scheduler_ordinal": _measured_metric(
                "scheduler_ordinal",
                float(control.scheduler_ordinal),
                reason="update_control_v2",
            ),
            "gradient_pre_clip_l2": _measured_metric(
                "gradient_pre_clip_l2",
                pre_l2,
                reason="l2_union_of_disjoint_update_blocks",
            ),
            "gradient_post_clip_l2": _measured_metric(
                "gradient_post_clip_l2",
                post_l2,
                reason="l2_union_of_disjoint_update_blocks",
            ),
            "gradient_l2": _measured_metric(
                "gradient_l2",
                post_l2,
                reason="post_clip_l2_union_of_disjoint_update_blocks",
            ),
            "gradient_inf": _measured_metric(
                "gradient_inf",
                post_inf,
                reason="max_post_clip_linf_across_update_blocks",
            ),
        }
    )

    if source is not None:
        if source.source_row_count == 0:
            if (
                source.entropy_sum != 0.0
                or source.support_size_sum != 0.0
            ):
                raise ProductionOperationError(
                    "zero-row source observation fabricated source totals"
                )
            values.update(
                {
                    name: not_applicable_metric(
                        name=name,
                        reason="source_row_count_is_zero",
                        units=metric_family_units(name),
                    )
                    for name in (
                        "source_entropy",
                        "source_support_size",
                        "effective_source_count",
                    )
                }
            )
        else:
            values.update(
                {
                    "source_entropy": _measured_metric(
                        "source_entropy",
                        source.mean_entropy,
                        numerator=source.entropy_sum,
                        denominator=source.source_row_count,
                        reason="recognition_source_rows",
                    ),
                    "source_support_size": _measured_metric(
                        "source_support_size",
                        source.mean_support_size,
                        numerator=source.support_size_sum,
                        denominator=source.source_row_count,
                        reason="recognition_source_rows",
                    ),
                    "effective_source_count": _measured_metric(
                        "effective_source_count",
                        source.effective_source_count,
                        numerator=source.entropy_sum,
                        denominator=source.source_row_count,
                        reason=(
                            "exp(source_entropy_sum/source_row_count)"
                        ),
                    ),
                }
            )
    if numerical is not None:
        values.update(
            {
                "minimum_cholesky_pivot": _measured_metric(
                    "minimum_cholesky_pivot",
                    numerical.minimum_cholesky_pivot,
                    reason="runtime_numerical_observation",
                ),
                "failed_pivots": _measured_metric(
                    "failed_pivots",
                    float(numerical.failed_pivots),
                    reason="runtime_numerical_observation",
                ),
                "condition_estimate": _measured_metric(
                    "condition_estimate",
                    numerical.condition_estimate,
                    reason="runtime_numerical_observation",
                ),
                "solve_residual": _measured_metric(
                    "solve_residual",
                    numerical.solve_residual,
                    reason="runtime_numerical_observation",
                ),
                "spd_projections": _measured_metric(
                    "spd_projections",
                    float(
                        sum(
                            evidence.projection_applied is True
                            for evidence in step.proposal_evidence
                        )
                    ),
                    reason="proposal_evidence_projection_count",
                ),
                "damping_events": _measured_metric(
                    "damping_events",
                    float(
                        sum(
                            evidence.damping_applied is True
                            for evidence in step.proposal_evidence
                        )
                    ),
                    reason="proposal_evidence_damping_count",
                ),
            }
        )

    wall_seconds = _seconds(timing.wall_ns)
    if wall_seconds <= 0.0:
        raise ProductionOperationError(
            "production train wall duration must be positive"
        )
    timing_values = {
        "data_wait_seconds": _seconds(timing.data_wait_ns),
        "forward_seconds": _seconds(timing.forward_ns),
        "backward_seconds": _seconds(timing.backward_ns),
        "update_seconds": _seconds(timing.update_ns),
        "wall_seconds": wall_seconds,
    }
    if arm_spec.latent_enabled:
        timing_values["inference_seconds"] = _seconds(
            timing.inference_ns
        )
    for name, value in timing_values.items():
        values[name] = _measured_metric(
            name,
            value,
            reason="phase_timer_observation",
        )
    values.update(
        {
            "counted_targets": _measured_metric(
                "counted_targets",
                float(counted),
                reason="step_result_counted_targets",
            ),
            "tokens_per_second": _measured_metric(
                "tokens_per_second",
                counted / wall_seconds,
                numerator=float(counted),
                denominator=timing.wall_ns,
                reason="counted_targets/wall_nanoseconds_scaled_to_seconds",
            ),
        }
    )
    memory_values = {
        "process_rss_bytes": memory.process_rss_bytes,
        "process_hwm_bytes": memory.process_hwm_bytes,
        "cuda_allocated_bytes": memory.cuda_allocated_bytes,
        "cuda_reserved_bytes": memory.cuda_reserved_bytes,
        "cuda_peak_allocated_bytes": memory.cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": memory.cuda_peak_reserved_bytes,
        "allocation_retries": allocation_retries,
        "oom_count": oom_count,
    }
    for name, value in memory_values.items():
        values[name] = _measured_metric(
            name,
            float(value),
            reason="runtime_memory_or_allocator_counter",
        )
    return tuple(
        values[name]
        for name in WT103_REQUIRED_METRIC_FAMILIES
        if name in values
    )


def _validation_metric_values(
    *,
    arm_spec: WT103ArmSpec,
    nll_sum: float,
    counted_targets: int,
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"],
    estimator_stream_id: int | None,
    particle_count: int | None,
    cache_audit: _ValidationCacheAudit,
    evaluation_ns: int,
    wall_ns: int,
    memory: MemoryObservation,
    allocation_retries: int,
    oom_count: int,
) -> tuple[MetricValue, ...]:
    """Project one full-validation boundary without train-value substitution."""

    if (
        type(arm_spec) is not WT103ArmSpec
        or type(nll_sum) is not float
        or not math.isfinite(nll_sum)
        or nll_sum < 0.0
        or type(counted_targets) is not int
        or counted_targets <= 0
        or scorer_kind not in ("exact_autoregressive", "weighted_smc")
        or type(cache_audit) is not _ValidationCacheAudit
        or cache_audit.counted_targets != counted_targets
        or cache_audit.summed_nll != nll_sum
        or cache_audit.passed is not True
        or type(evaluation_ns) is not int
        or evaluation_ns <= 0
        or type(wall_ns) is not int
        or wall_ns < evaluation_ns
        or type(memory) is not MemoryObservation
        or type(allocation_retries) is not int
        or allocation_retries < 0
        or type(oom_count) is not int
        or oom_count < 0
    ):
        raise ProductionOperationError(
            "production validation metric observation is incomplete"
        )
    arm_spec.__post_init__()
    memory.__post_init__()
    cache_audit.__post_init__()
    if scorer_kind == "weighted_smc":
        if (
            type(estimator_stream_id) is not int
            or estimator_stream_id < 0
            or type(particle_count) is not int
            or particle_count <= 0
        ):
            raise ProductionOperationError(
                "weighted validation omitted estimator identity"
            )
    elif estimator_stream_id is not None or particle_count is not None:
        raise ProductionOperationError(
            "exact validation cannot fabricate estimator metrics"
        )
    nll_per_token = nll_sum / counted_targets
    try:
        perplexity = math.exp(nll_per_token)
    except OverflowError as exc:
        raise ProductionOperationError(
            "production validation perplexity overflowed"
        ) from exc
    if not math.isfinite(perplexity):
        raise ProductionOperationError(
            "production validation perplexity is nonfinite"
        )
    wall_seconds = _seconds(wall_ns)
    values = _arm_inapplicable_metrics(arm_spec)
    values.update(
        {
            "prior_nll_sum": _measured_metric(
                "prior_nll_sum",
                nll_sum,
                reason="full_validation_corpus_sum",
            ),
            "prior_nll_per_token": _measured_metric(
                "prior_nll_per_token",
                nll_per_token,
                numerator=nll_sum,
                denominator=counted_targets,
                reason="full_validation_corpus_sum/count",
            ),
            "perplexity": _measured_metric(
                "perplexity",
                perplexity,
                numerator=nll_sum,
                denominator=counted_targets,
                reason="exp(full_validation_corpus_sum/count)",
            ),
            "cache_audit_passed": _measured_metric(
                "cache_audit_passed",
                float(cache_audit.passed),
                reason=(
                    "typed_cold_warm_reverse_cache_audit:"
                    f"{cache_audit.audit_sha256}"
                ),
            ),
            "counted_targets": _measured_metric(
                "counted_targets",
                float(counted_targets),
                reason="full_validation_manifest_denominator",
            ),
            "tokens_per_second": _measured_metric(
                "tokens_per_second",
                counted_targets / wall_seconds,
                numerator=float(counted_targets),
                denominator=wall_ns,
                reason="validation_targets/wall_seconds",
            ),
            "evaluation_seconds": _measured_metric(
                "evaluation_seconds",
                _seconds(evaluation_ns),
                reason="validation_phase_timer",
            ),
            "wall_seconds": _measured_metric(
                "wall_seconds",
                wall_seconds,
                reason="validation_wall_timer",
            ),
        }
    )
    if scorer_kind == "weighted_smc":
        assert estimator_stream_id is not None
        assert particle_count is not None
        values["estimator_stream"] = _measured_metric(
            "estimator_stream",
            float(estimator_stream_id),
            reason="frozen_validation_estimator_stream",
        )
        values["particle_count"] = _measured_metric(
            "particle_count",
            float(particle_count),
            reason="runtime_estimator_spec",
        )
    memory_values = {
        "process_rss_bytes": memory.process_rss_bytes,
        "process_hwm_bytes": memory.process_hwm_bytes,
        "cuda_allocated_bytes": memory.cuda_allocated_bytes,
        "cuda_reserved_bytes": memory.cuda_reserved_bytes,
        "cuda_peak_allocated_bytes": memory.cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": memory.cuda_peak_reserved_bytes,
        "allocation_retries": allocation_retries,
        "oom_count": oom_count,
    }
    for name, value in memory_values.items():
        values[name] = _measured_metric(
            name,
            float(value),
            reason="runtime_memory_or_allocator_counter",
        )
    return tuple(
        values[name]
        for name in WT103_REQUIRED_METRIC_FAMILIES
        if name in values
    )


def _append_metric_values(
    *,
    path: Path,
    backend: DurabilityBackend,
    attempt: ProductionAttemptSpec,
    phase: str,
    split: Literal["train", "validation", "not_applicable"],
    step: int,
    pass_index: int,
    values: tuple[MetricValue, ...],
) -> MetricRecord:
    records = validate_metric_log(path) if path.exists() else ()
    record = create_metric_record(
        ordinal=len(records),
        utc_timestamp=_canonical_utc_timestamp(),
        monotonic_ns=time.monotonic_ns(),
        run_id=attempt.attempt_id,
        arm_id=attempt.arm_id,
        seed_id=attempt.seed_id,
        phase=phase,
        split=split,
        step=step,
        pass_index=pass_index,
        previous_record_sha256=(
            _ZERO_SHA256 if not records else records[-1].record_sha256
        ),
        values=values,
    )
    append_metric(path, record, durability_backend=backend)
    return record


def _append_boundary_metric(
    *,
    path: Path,
    backend: DurabilityBackend,
    attempt: ProductionAttemptSpec,
    bundle: WT103ArmRuntimeBundle,
    nll_sum: float,
    counted_targets: int,
    cache_audit: _ValidationCacheAudit,
    pass_index: int,
    evaluation_ns: int,
    wall_ns: int,
    memory: MemoryObservation,
    allocation_retries: int,
    oom_count: int,
) -> MetricRecord:
    return _append_metric_values(
        path=path,
        backend=backend,
        attempt=attempt,
        phase="validation_prior_scoring",
        split="validation",
        step=bundle.execution_runtime.update_counter,
        pass_index=pass_index,
        values=_validation_metric_values(
            arm_spec=bundle.built_arm.record.spec,
            nll_sum=nll_sum,
            counted_targets=counted_targets,
            scorer_kind=bundle.scorer_kind,
            estimator_stream_id=(
                None
                if bundle.scorer_kind == "exact_autoregressive"
                else 0
            ),
            particle_count=bundle.estimator_spec.particle_count,
            cache_audit=cache_audit,
            evaluation_ns=evaluation_ns,
            wall_ns=wall_ns,
            memory=memory,
            allocation_retries=allocation_retries,
            oom_count=oom_count,
        ),
    )


def _append_checkpoint_metric(
    *,
    path: Path,
    backend: DurabilityBackend,
    attempt: ProductionAttemptSpec,
    step: int,
    pass_index: int,
    checkpoint_ns: int,
) -> MetricRecord:
    if type(checkpoint_ns) is not int or checkpoint_ns <= 0:
        raise ProductionOperationError(
            "checkpoint metric requires a positive measured duration"
        )
    return _append_metric_values(
        path=path,
        backend=backend,
        attempt=attempt,
        phase="checkpoint_serialization",
        split="not_applicable",
        step=step,
        pass_index=pass_index,
        values=(
            _measured_metric(
                "checkpoint_seconds",
                _seconds(checkpoint_ns),
                reason="checkpoint_wall_timer",
            ),
        ),
    )


def _checkpoint_state(
    *,
    bundle: WT103ArmRuntimeBundle,
    cursor: DataCursor,
    schedule: WindowSchedule,
    metric_head: str,
    metric_next_ordinal: int,
    nll_numerator: float,
    nll_denominator: int,
    global_batch_step: int,
    cumulative_counted_targets: int,
) -> dict[str, object]:
    runtime = export_wt103_arm_runtime_state(bundle)
    predictor = bundle.make_predictor()
    stream = _estimator_stream(bundle)
    fixture_prefix = torch.tensor([0], dtype=torch.int64, device="cpu")
    fixture_rng = _capture_process_rng_state()
    try:
        prediction = predictor.next_token_log_probs(
            CausalPrefix.create(
                receiver_t=2,
                vocabulary=bundle.vocabulary,
                token_ids=fixture_prefix,
            ),
            stream,
            None,
        )
    finally:
        _restore_process_rng_state(fixture_rng)
    numpy_state = np.random.get_state()
    permutation_bytes = (
        torch.tensor(
            schedule.window_ids,
            dtype=torch.int64,
            device="cpu",
        )
        .contiguous()
        .view(torch.uint8)
        .clone()
    )
    return {
        "model_state": runtime["model"],
        "recognition_state": runtime["recognition"],
        "optimizer_state": {
            "model": runtime["model_optimizer"],
            "recognition": runtime["recognition_optimizer"],
        },
        "scheduler_state": {
            "model": runtime["model_scheduler"],
            "recognition": runtime["recognition_scheduler"],
        },
        "amp_scaler_state": None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": (
                str(numpy_state[0]),
                torch.from_numpy(
                    np.asarray(numpy_state[1], dtype=np.int64).copy()
                ),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            ),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": tuple(
                item.clone() for item in torch.cuda.get_rng_state_all()
            ),
        },
        "estimator_state": {
            "stream_counters": {
                "validation": bundle.execution_runtime.update_counter,
                "test": 0,
            },
            "runtime_identity_sha256": runtime[
                "runtime_identity_sha256"
            ],
            "estimator_identity_sha256": (
                bundle.estimator_identity.identity_sha256
            ),
        },
        "data_cursor_state": {
            "split": cursor.split,
            "pass_index": cursor.pass_index,
            "batch_index": cursor.next_batch_ordinal,
            "next_batch_ordinal": cursor.next_batch_ordinal,
            "next_window_ids": cursor.next_window_ids,
            "counted_targets": cursor.counted_targets,
            "cursor_sha256": cursor.cursor_sha256,
            "permutation_bytes": permutation_bytes,
            "permutation_sha256": cursor.permutation_sha256,
        },
        "update_trace_state": {
            "global_step": global_batch_step,
            "successful_updates": runtime["update_counter"],
            "rejected_updates": 0,
            "counted_targets": cumulative_counted_targets,
        },
        "metric_state": {
            "next_ordinal": metric_next_ordinal,
            "hash_chain_head": metric_head,
            "nll_numerator": nll_numerator,
            "nll_denominator": nll_denominator,
            "failure_ledger_head": _ZERO_SHA256,
        },
        "next_prediction_fixture": (
            fixture_prefix,
            prediction.log_probs.value(),
        ),
    }


def _checkpoint_contract(
    *,
    attempt: ProductionAttemptSpec,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    bundle: WT103ArmRuntimeBundle,
    plan_sha256: str,
    cursor: DataCursor,
    role: Literal["resume_only", "terminal_scoring"],
) -> ResumeContract:
    spec = bundle.built_arm.record.spec

    def hash_for(domain: str, value: object) -> str:
        return owned_sha256(domain, value)

    return ResumeContract.create(
        logical_key=(
            f"terminal/{attempt.arm_id}/seed={attempt.seed_id}"
            if role == "terminal_scoring"
            else (
                f"{attempt.attempt_id}/pass-{cursor.pass_index}/"
                f"batch-{cursor.next_batch_ordinal}/{role}"
            )
        ),
        checkpoint_role=role,
        training_complete=(role == "terminal_scoring"),
        arm_spec_sha256=spec.arm_spec_sha256,
        experiment_plan_sha256=plan_sha256,
        config_sha256=training.experiment_config_sha256,
        objective_sha256=hash_for(
            "vfe4.wt103.production-objective.v1",
            {
                "arm_spec_sha256": spec.arm_spec_sha256,
                "training_objective": spec.training_objective,
            },
        ),
        model_schema_sha256=bundle.built_arm.record.build_sha256,
        recognition_schema_sha256=hash_for(
            "vfe4.wt103.production-recognition-schema.v1",
            bundle.built_arm.record.recognition_parameter_names,
        ),
        optimizer_schema_sha256=hash_for(
            "vfe4.wt103.production-optimizer-schema.v1",
            training.profile.optimizer,
        ),
        scheduler_schema_sha256=hash_for(
            "vfe4.wt103.production-scheduler-schema.v1",
            training.profile.scheduler,
        ),
        amp_schema_sha256=hash_for(
            "vfe4.wt103.production-amp-schema.v1",
            training.profile.precision,
        ),
        rng_schema_sha256=hash_for(
            "vfe4.wt103.production-rng-schema.v1",
            {
                "python": True,
                "numpy": True,
                "torch_cpu": True,
                "torch_cuda": True,
            },
        ),
        estimator_schema_sha256=bundle.estimator_identity.identity_sha256,
        cursor_schema_sha256=cursor.cursor_sha256,
        metric_schema_sha256=hash_for(
            "vfe4.wt103.production-metric-schema.v1",
            WT103_REQUIRED_METRIC_FAMILIES,
        ),
        update_trace_schema_sha256=hash_for(
            "vfe4.wt103.production-update-trace-schema.v1",
            spec.update_phases,
        ),
        precision_profile_sha256=hash_for(
            "vfe4.wt103.production-precision-profile.v1",
            training.profile.precision,
        ),
        dependency_lock_sha256=(
            source_lock.finalized_source.dependency_lock_sha256
        ),
        source_sha256=source_lock.finalized_source.record_sha256,
        tokenizer_sha256=source_lock.tokenizer.spec_sha256,
        data_sha256=source_lock.token_caches[0].record_sha256,
        window_sha256=source_lock.schedules.window_manifests[
            0
        ].manifest_sha256,
        permutation_sha256=cursor.permutation_sha256,
        evidence_sha256=getattr(
            readiness.readiness_token,
            "token_sha256",
        ),
        environment_sha256=getattr(
            readiness.readiness_token,
            "environment_sha256",
        ),
        maximum_checkpoint_bytes=64 * 1024**3,
        maximum_tensor_bytes=16 * 1024**3,
        maximum_total_tensor_bytes=48 * 1024**3,
        maximum_tensor_count=100_000,
        maximum_container_items=1_000_000,
        maximum_recursion_depth=64,
    )


def _capture_process_rng_state() -> tuple[
    object,
    object,
    torch.Tensor,
    tuple[torch.Tensor, ...],
]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
        tuple(item.clone() for item in torch.cuda.get_rng_state_all()),
    )


def _restore_process_rng_state(
    state: tuple[object, object, torch.Tensor, tuple[torch.Tensor, ...]],
) -> None:
    python_state, numpy_state, torch_cpu, torch_cuda = state
    random.setstate(python_state)  # type: ignore[arg-type]
    np.random.set_state(numpy_state)  # type: ignore[arg-type]
    torch.random.set_rng_state(torch_cpu)
    torch.cuda.set_rng_state_all(list(torch_cuda))


def _restore_checkpoint_rng(state: dict[str, object]) -> None:
    rng = state.get("rng_state")
    if type(rng) is not dict:
        raise ProductionOperationError("checkpoint RNG state is malformed")
    python_state = rng.get("python")
    numpy_state = rng.get("numpy")
    torch_cpu = rng.get("torch_cpu")
    torch_cuda = rng.get("torch_cuda")
    if (
        type(python_state) is not tuple
        or type(numpy_state) is not tuple
        or len(numpy_state) != 5
        or type(numpy_state[0]) is not str
        or type(numpy_state[1]) is not torch.Tensor
        or numpy_state[1].dtype is not torch.int64
        or numpy_state[1].device.type != "cpu"
        or type(numpy_state[2]) is not int
        or type(numpy_state[3]) is not int
        or type(numpy_state[4]) is not float
        or type(torch_cpu) is not torch.Tensor
        or torch_cpu.dtype is not torch.uint8
        or torch_cpu.device.type != "cpu"
        or type(torch_cuda) is not tuple
        or len(torch_cuda) != torch.cuda.device_count()
        or any(
            type(item) is not torch.Tensor
            or item.dtype is not torch.uint8
            or item.device.type != "cpu"
            for item in torch_cuda
        )
    ):
        raise ProductionOperationError(
            "checkpoint RNG streams are not canonical"
        )
    random.setstate(python_state)
    np.random.set_state(
        (
            numpy_state[0],
            numpy_state[1].numpy().astype(np.uint32, copy=True),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.random.set_rng_state(torch_cpu)
    torch.cuda.set_rng_state_all(list(torch_cuda))


class _FreshRuntimeTarget:
    def __init__(
        self,
        *,
        contract: ResumeContract,
        training: TrainingConfig,
        authority: WT103RuntimeAuthority,
        attempt: ProductionAttemptSpec,
        planned_optimizer_steps: int,
        device: torch.device,
        cursor: DataCursor,
        schedule: WindowSchedule,
        metric_head: str,
        metric_next_ordinal: int,
        expected_global_batch_step: int,
        expected_cumulative_counted_targets: int,
    ) -> None:
        self.checkpoint_contract_sha256 = contract.contract_sha256
        self.training = training
        self.authority = authority
        self.attempt = attempt
        self.planned_optimizer_steps = planned_optimizer_steps
        self.device = device
        self.cursor = cursor
        self.schedule = schedule
        self.metric_head = metric_head
        self.metric_next_ordinal = metric_next_ordinal
        self.expected_global_batch_step = expected_global_batch_step
        self.expected_cumulative_counted_targets = (
            expected_cumulative_counted_targets
        )
        self.bundle: WT103ArmRuntimeBundle | None = None
        self.restored_state: dict[str, object] | None = None
        self._validated_bundle: WT103ArmRuntimeBundle | None = None
        self._validated_state: dict[str, object] | None = None

    def is_fresh_checkpoint_target(self) -> bool:
        return self.bundle is None

    def _runtime_state(
        self,
        state: dict[str, object],
    ) -> dict[str, object]:
        estimator = state["estimator_state"]
        optimizer = state["optimizer_state"]
        scheduler = state["scheduler_state"]
        update = state["update_trace_state"]
        if not all(
            type(item) is dict
            for item in (estimator, optimizer, scheduler, update)
        ):
            raise ProductionOperationError(
                "checkpoint runtime state schema changed"
            )
        return {
            "runtime_identity_sha256": estimator[
                "runtime_identity_sha256"
            ],
            "model": state["model_state"],
            "recognition": state["recognition_state"],
            "model_optimizer": optimizer["model"],
            "recognition_optimizer": optimizer["recognition"],
            "model_scheduler": scheduler["model"],
            "recognition_scheduler": scheduler["recognition"],
            "update_counter": update["successful_updates"],
        }

    def _validate_auxiliary_state(self, state: dict[str, object]) -> None:
        cursor = state.get("data_cursor_state")
        metric = state.get("metric_state")
        update = state.get("update_trace_state")
        estimator = state.get("estimator_state")
        if not all(
            type(item) is dict
            for item in (cursor, metric, update, estimator)
        ):
            raise ProductionOperationError(
                "checkpoint auxiliary state is malformed"
            )
        assert type(cursor) is dict
        assert type(metric) is dict
        assert type(update) is dict
        assert type(estimator) is dict
        permutation = (
            torch.tensor(
                self.schedule.window_ids,
                dtype=torch.int64,
                device="cpu",
            )
            .contiguous()
            .view(torch.uint8)
        )
        expected_cursor = {
            "split": self.cursor.split,
            "pass_index": self.cursor.pass_index,
            "batch_index": self.cursor.next_batch_ordinal,
            "next_batch_ordinal": self.cursor.next_batch_ordinal,
            "next_window_ids": self.cursor.next_window_ids,
            "counted_targets": self.cursor.counted_targets,
            "cursor_sha256": self.cursor.cursor_sha256,
            "permutation_sha256": self.cursor.permutation_sha256,
        }
        if any(cursor.get(name) != value for name, value in expected_cursor.items()):
            raise ProductionOperationError(
                "checkpoint cursor differs from its authenticated sidecar"
            )
        observed_permutation = cursor.get("permutation_bytes")
        if (
            type(observed_permutation) is not torch.Tensor
            or observed_permutation.dtype is not torch.uint8
            or observed_permutation.device.type != "cpu"
            or not torch.equal(observed_permutation, permutation)
        ):
            raise ProductionOperationError(
                "checkpoint permutation bytes differ from the frozen schedule"
            )
        if (
            metric.get("hash_chain_head") != self.metric_head
            or metric.get("next_ordinal") != self.metric_next_ordinal
            or type(metric.get("nll_numerator")) is not float
            or not math.isfinite(metric["nll_numerator"])
            or metric["nll_numerator"] < 0.0
            or type(metric.get("nll_denominator")) is not int
            or metric["nll_denominator"] <= 0
            or metric.get("failure_ledger_head") != _ZERO_SHA256
            or update.get("global_step")
            != self.expected_global_batch_step
            or type(update.get("successful_updates")) is not int
            or update["successful_updates"] <= 0
            or update.get("rejected_updates") != 0
            or update.get("counted_targets")
            != self.expected_cumulative_counted_targets
            or estimator.get("stream_counters")
            != {
                "validation": update.get("successful_updates"),
                "test": 0,
            }
        ):
            raise ProductionOperationError(
                "checkpoint metric/update/estimator state is inconsistent"
            )

    def _validate_prediction_fixture(
        self,
        bundle: WT103ArmRuntimeBundle,
        state: dict[str, object],
    ) -> None:
        fixture = state.get("next_prediction_fixture")
        if (
            type(fixture) is not tuple
            or len(fixture) != 2
            or type(fixture[0]) is not torch.Tensor
            or fixture[0].dtype is not torch.int64
            or fixture[0].device.type != "cpu"
            or fixture[0].ndim != 1
            or type(fixture[1]) is not torch.Tensor
            or fixture[1].device.type != "cpu"
        ):
            raise ProductionOperationError(
                "checkpoint next-prediction fixture is malformed"
            )
        prediction = bundle.make_predictor().next_token_log_probs(
            CausalPrefix.create(
                receiver_t=fixture[0].numel() + 1,
                vocabulary=bundle.vocabulary,
                token_ids=fixture[0],
            ),
            _estimator_stream(bundle),
            None,
        )
        if not torch.equal(prediction.log_probs.value(), fixture[1]):
            raise ProductionOperationError(
                "restored runtime changed the next-prediction fixture"
            )

    def validate_checkpoint_state(self, state: dict[str, object]) -> None:
        self._validate_auxiliary_state(state)
        before_rng = _capture_process_rng_state()
        try:
            candidate = rebuild_wt103_arm_runtime(
                self.training,
                arm_id=self.attempt.arm_id,
                authority=self.authority,
                seed=self.attempt.seed_id,
                learning_rate=self.attempt.learning_rate,
                weight_decay=self.attempt.weight_decay,
                planned_optimizer_steps=self.planned_optimizer_steps,
                scientific_state=self._runtime_state(state),
                device=self.device,
                dtype=torch.float32,
            )
            estimator = state["estimator_state"]
            assert type(estimator) is dict
            if (
                estimator.get("estimator_identity_sha256")
                != candidate.estimator_identity.identity_sha256
                or state.get("amp_scaler_state") is not None
            ):
                raise ProductionOperationError(
                    "checkpoint estimator or AMP state changed"
                )
            _restore_checkpoint_rng(state)
            self._validate_prediction_fixture(candidate, state)
            self._validated_bundle = candidate
            self._validated_state = state
        finally:
            _restore_process_rng_state(before_rng)

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if (
            self._validated_bundle is None
            or self._validated_state is not state
        ):
            raise ProductionOperationError(
                "checkpoint restore did not reuse its validated fresh runtime"
            )
        self.bundle = self._validated_bundle
        self._validated_bundle = None
        self._validated_state = None
        self.restored_state = state
        _restore_checkpoint_rng(state)


def _parse_cursor(value: object) -> DataCursor:
    if type(value) is not dict:
        raise ProductionOperationError("checkpoint cursor is malformed")
    raw = dict(value)
    if type(raw.get("next_window_ids")) is list:
        raw["next_window_ids"] = tuple(raw["next_window_ids"])
    return DataCursor(**raw)


def _checkpoint_sidecar(
    path: Path,
) -> tuple[
    ResumeContract,
    WT103CheckpointIdentity,
    DataCursor,
    Path,
]:
    document = _canonical_document(path)
    if set(document) != {
        "contract",
        "identity",
        "cursor",
        "checkpoint_path",
    }:
        raise ProductionOperationError(
            "checkpoint sidecar key inventory changed"
        )
    contract_raw = document.get("contract")
    identity_raw = document.get("identity")
    cursor_raw = document.get("cursor")
    checkpoint_path = document.get("checkpoint_path")
    if (
        type(contract_raw) is not dict
        or type(identity_raw) is not dict
        or type(cursor_raw) is not dict
        or type(checkpoint_path) is not str
    ):
        raise ProductionOperationError(
            "checkpoint sidecar schema is malformed"
        )
    contract = ResumeContract(**contract_raw)
    identity = WT103CheckpointIdentity(**identity_raw)
    cursor = _parse_cursor(cursor_raw)
    checkpoint = Path(checkpoint_path)
    if (
        not checkpoint.is_absolute()
        or checkpoint.absolute().parent != path.absolute().parent
        or checkpoint.name not in ("resume-only-0.pt", "resume-only-1.pt")
        or contract.checkpoint_role != "resume_only"
        or identity.checkpoint_role != "resume_only"
        or identity.logical_key != contract.logical_key
    ):
        raise ProductionOperationError(
            "checkpoint sidecar escaped its resume-only attempt root"
        )
    payload = _regular_bytes(
        checkpoint,
        maximum_bytes=identity.size_bytes,
    )
    if (
        len(payload) != identity.size_bytes
        or hashlib.sha256(payload).hexdigest() != identity.checkpoint_payload_sha256
    ):
        raise ProductionOperationError(
            "checkpoint payload differs from its authenticated sidecar"
        )
    return contract, identity, cursor, checkpoint


def _select_resume_checkpoint_path(attempt_root: Path) -> Path:
    sidecar = attempt_root / "resume-sidecar.json"
    try:
        sidecar.lstat()
    except FileNotFoundError:
        return attempt_root / "resume-only-0.pt"
    except OSError as exc:
        raise ProductionOperationError(
            "checkpoint sidecar metadata is unavailable"
        ) from exc
    _contract, _identity, _cursor, active = _checkpoint_sidecar(sidecar)
    inactive_slot = 1 if active.name == "resume-only-0.pt" else 0
    return attempt_root / f"resume-only-{inactive_slot}.pt"


def _reopen_committed_resume_checkpoint(
    sidecar: Path,
    *,
    expected_identity: WT103CheckpointIdentity,
) -> Path:
    _contract, observed_identity, _cursor, checkpoint = (
        _checkpoint_sidecar(sidecar)
    )
    if observed_identity != expected_identity:
        raise ProductionOperationError(
            "published checkpoint sidecar identity changed on reopen"
        )
    return checkpoint


def _save_attempt_checkpoint(
    *,
    attempt_root: Path,
    backend: DurabilityBackend,
    attempt: ProductionAttemptSpec,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    bundle: WT103ArmRuntimeBundle,
    plan_sha256: str,
    cursor: DataCursor,
    schedule: WindowSchedule,
    metric_head: str,
    metric_next_ordinal: int,
    nll_numerator: float,
    nll_denominator: int,
    global_batch_step: int,
    cumulative_counted_targets: int,
    elapsed_seconds: float,
    role: Literal["resume_only", "terminal_scoring"],
) -> WT103CheckpointIdentity:
    if (
        type(elapsed_seconds) is not float
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
    ):
        raise ProductionOperationError(
            "checkpoint elapsed time must be an actual finite duration"
        )
    contract = _checkpoint_contract(
        attempt=attempt,
        training=training,
        source_lock=source_lock,
        readiness=readiness,
        bundle=bundle,
        plan_sha256=plan_sha256,
        cursor=cursor,
        role=role,
    )
    checkpoint_path = (
        _select_resume_checkpoint_path(attempt_root)
        if role == "resume_only"
        else attempt_root / "terminal-scoring.pt"
    )
    identity = save_checkpoint(
        checkpoint_path,
        contract=contract,
        scientific_state=_checkpoint_state(
            bundle=bundle,
            cursor=cursor,
            schedule=schedule,
            metric_head=metric_head,
            metric_next_ordinal=metric_next_ordinal,
            nll_numerator=nll_numerator,
            nll_denominator=nll_denominator,
            global_batch_step=global_batch_step,
            cumulative_counted_targets=cumulative_counted_targets,
        ),
        durability_backend=backend,
        operational_metadata={
            "process_id": os.getpid(),
            "utc_timestamp": _canonical_utc_timestamp(),
            "monotonic_seconds": time.monotonic(),
            "elapsed_seconds": elapsed_seconds,
            "path_hint": role,
            "write_ordinal": bundle.execution_runtime.update_counter,
        },
    )
    if role == "resume_only":
        _publish_document(
            backend,
            attempt_root / "resume-sidecar.json",
            {
                "contract": contract.canonical_payload(),
                "identity": {
                    field.name: getattr(identity, field.name)
                    for field in dataclasses.fields(identity)
                },
                "cursor": {
                    field.name: getattr(cursor, field.name)
                    for field in dataclasses.fields(cursor)
                },
                "checkpoint_path": str(checkpoint_path),
            },
        )
    return identity


def _maximum_batches_for_attempt(
    attempt: ProductionAttemptSpec,
    *,
    full_batches: int,
) -> int:
    return (
        math.ceil(full_batches / 4)
        if attempt.quarter_pass
        else full_batches
    )


def _attempt_artifacts_exist(attempt_root: Path) -> bool:
    return any(
        path.exists()
        for path in (
            attempt_root / "metrics.jsonl",
            attempt_root / "metrics.csv",
            attempt_root / "resume-sidecar.json",
            attempt_root / "resume-only-0.pt",
            attempt_root / "resume-only-1.pt",
            attempt_root / "terminal-scoring.pt",
        )
    )


def _apply_frozen_precision_runtime_policy(
    *,
    training: TrainingConfig,
    torch_runtime: object | None = None,
    environment: MutableMapping[str, str] | None = None,
) -> LivePrecisionRuntimeEvidence:
    """Apply the frozen CUDA policy before the production attempt touches CUDA."""

    return apply_frozen_precision_runtime_policy(
        precision=training.profile.precision,
        torch_runtime=torch if torch_runtime is None else torch_runtime,
        environment=os.environ if environment is None else environment,
    )


def _publish_live_precision_runtime_evidence(
    *,
    attempt_root: Path,
    evidence: LivePrecisionRuntimeEvidence,
    backend: DurabilityBackend,
) -> Path:
    """Durably bind effective CUDA precision settings to this attempt."""

    if type(evidence) is not LivePrecisionRuntimeEvidence:
        raise ProductionOperationError(
            "live CUDA precision evidence is not exact"
        )
    evidence.__post_init__()
    path = attempt_root / _LIVE_PRECISION_RUNTIME_EVIDENCE_NAME
    document = {
        "schema_version": "wt103-live-precision-runtime-evidence-v1",
        "effective_precision_policy": {
            field.name: getattr(evidence, field.name)
            for field in dataclasses.fields(evidence)
        },
    }
    _publish_document(backend, path, document)
    if _canonical_document(path) != document:
        raise ProductionOperationError(
            "published live CUDA precision evidence changed on reopen"
        )
    return path


def _execute_attempt(
    *,
    attempt: ProductionAttemptSpec,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    cache_root: Path,
    plan: ExperimentPlanIdentity,
    reserved: ReservedRun,
    backend: DurabilityBackend,
    resume_active: bool,
    resource_abort: Callable[[], None],
) -> tuple[ProductionAttemptOutcome, RunManifestIdentity]:
    if not callable(resource_abort):
        raise ProductionOperationError(
            "production resource-abort authority is invalid"
        )
    resource_abort()
    precision_evidence = _apply_frozen_precision_runtime_policy(
        training=training
    )
    _publish_live_precision_runtime_evidence(
        attempt_root=reserved.inprogress_path,
        evidence=precision_evidence,
        backend=backend,
    )
    device = torch.device(training.profile.precision.real_training_device)
    if not torch.cuda.is_available() or device.type != "cuda":
        raise ProductionOperationError(
            "authorized production training requires CUDA"
        )
    authority = WT103RuntimeAuthority.from_production_source_lock(
        source_lock,
        device=device,
        dtype=torch.float32,
    )
    train_windows, first_schedule = open_production_training_split(
        source_lock=source_lock,
        cache_root=cache_root,
        split="train",
        pass_index=0,
    )
    full_batches = math.ceil(
        len(first_schedule.window_ids) / first_schedule.batch_size
    )
    attempted_batches = (
        math.ceil(full_batches / 4)
        if attempt.quarter_pass
        else full_batches * attempt.pass_count
    )
    planned_steps = max(
        training.profile.scheduler.warmup_optimizer_steps + 1,
        attempted_batches,
    )
    if (
        type(plan) is not ExperimentPlanIdentity
        or type(reserved) is not ReservedRun
        or reserved.run_id != attempt.attempt_id
        or reserved.run_role != attempt.role
        or reserved.experiment_plan_sha256
        != plan.plan.experiment_plan_sha256
    ):
        raise ProductionOperationError(
            "production attempt reservation differs from its plan"
        )
    attempt_root = reserved.inprogress_path
    metrics_path = attempt_root / "metrics.jsonl"
    sidecar = attempt_root / "resume-sidecar.json"
    cursor: DataCursor | None = None
    start_pass = 0
    last_nll: tuple[float, int] | None = None
    latest_resume_identity: WT103CheckpointIdentity | None = None
    latest_resume_path: Path | None = None
    attempt_started_ns = time.perf_counter_ns()
    if resume_active and sidecar.is_file():
        contract, identity, cursor, checkpoint_path = _checkpoint_sidecar(
            sidecar
        )
        latest_resume_identity = identity
        latest_resume_path = checkpoint_path
        if (
            cursor.split != "train"
            or not 0 <= cursor.pass_index < attempt.pass_count
        ):
            raise ProductionOperationError(
                "resume cursor is outside the attempt pass inventory"
            )
        resume_windows, resume_schedule = open_production_training_split(
            source_lock=source_lock,
            cache_root=cache_root,
            split="train",
            pass_index=cursor.pass_index,  # type: ignore[arg-type]
        )
        maximum = _maximum_batches_for_attempt(
            attempt,
            full_batches=full_batches,
        )
        if (
            cursor.next_batch_ordinal > maximum
            or cursor.next_batch_ordinal
            not in (
                *source_lock.schedules.validation_boundary_batch_ordinals,
                maximum,
            )
        ):
            raise ProductionOperationError(
                "resume cursor is not an authenticated checkpoint boundary"
            )
        metric_records = validate_metric_log(metrics_path)
        if not metric_records or any(
            record.run_id != attempt.attempt_id
            or record.arm_id != attempt.arm_id
            or record.seed_id != attempt.seed_id
            for record in metric_records
        ):
            raise ProductionOperationError(
                "resume metric log differs from the active attempt"
            )
        checkpoint_metric = (
            metric_records[-1]
            if metric_records[-1].phase == "checkpoint_serialization"
            else None
        )
        checkpoint_metric_head = (
            metric_records[-1].record_sha256
            if checkpoint_metric is None
            else checkpoint_metric.previous_record_sha256
        )
        checkpoint_metric_next_ordinal = (
            len(metric_records)
            if checkpoint_metric is None
            else len(metric_records) - 1
        )
        target = _FreshRuntimeTarget(
            contract=contract,
            training=training,
            authority=authority,
            attempt=attempt,
            planned_optimizer_steps=planned_steps,
            device=device,
            cursor=cursor,
            schedule=resume_schedule,
            metric_head=checkpoint_metric_head,
            metric_next_ordinal=checkpoint_metric_next_ordinal,
            expected_global_batch_step=(
                cursor.pass_index * full_batches
                + cursor.next_batch_ordinal
            ),
            expected_cumulative_counted_targets=(
                cursor.pass_index
                * resume_windows.manifest.counted_targets
                + cursor.counted_targets
            ),
        )
        load_checkpoint(
            checkpoint_path,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=target,
        )
        if target.bundle is None:
            raise ProductionOperationError(
                "production resume did not restore its runtime"
            )
        bundle = target.bundle
        restored = target.restored_state
        if type(restored) is not dict:
            raise ProductionOperationError(
                "production resume omitted its scientific state"
            )
        metric_state = restored.get("metric_state")
        if (
            type(metric_state) is not dict
            or type(metric_state.get("nll_numerator")) is not float
            or type(metric_state.get("nll_denominator")) is not int
            or metric_records[-1].step
            != bundle.execution_runtime.update_counter
        ):
            raise ProductionOperationError(
                "production resume metric projection is inconsistent"
            )
        last_nll = (
            metric_state["nll_numerator"],
            metric_state["nll_denominator"],
        )
        if cursor.next_batch_ordinal == maximum:
            start_pass = cursor.pass_index + 1
            cursor = None
        else:
            start_pass = cursor.pass_index
    else:
        if _attempt_artifacts_exist(attempt_root):
            raise ProductionOperationError(
                "fresh production attempt root already contains artifacts"
            )
        bundle = build_wt103_arm_runtime(
            training,
            arm_id=attempt.arm_id,
            authority=authority,
            seed=attempt.seed_id,
            learning_rate=attempt.learning_rate,
            weight_decay=attempt.weight_decay,
            planned_optimizer_steps=planned_steps,
            device=device,
            dtype=torch.float32,
        )
    resource_abort()
    validation_windows, validation_schedule = (
        open_production_training_split(
            source_lock=source_lock,
            cache_root=cache_root,
            split="validation",
            pass_index=0,
        )
    )
    last_step: StepResult | None = None
    for pass_index in range(start_pass, attempt.pass_count):
        windows, schedule = open_production_training_split(
            source_lock=source_lock,
            cache_root=cache_root,
            split="train",
            pass_index=pass_index,  # type: ignore[arg-type]
        )
        pass_cursor = cursor if cursor is not None and cursor.pass_index == pass_index else None
        maximum = _maximum_batches_for_attempt(
            attempt,
            full_batches=full_batches,
        )
        completed = 0 if pass_cursor is None else pass_cursor.next_batch_ordinal
        batches = iter(
            iter_production_batches(
                windows=windows,
                schedule=schedule,
                cursor=pass_cursor,
            )
        )
        while completed < maximum:
            resource_abort()
            step_wall_started_ns = time.perf_counter_ns()
            try:
                batch = next(batches)
            except StopIteration:
                break
            data_ready_ns = time.perf_counter_ns()
            timer = PhaseTimer(synchronize=torch.cuda.synchronize)
            previous_runner = (
                bundle.execution_runtime.execution_event_runner
            )
            if previous_runner is not None:
                raise ProductionOperationError(
                    "production phase timer cannot replace another event runner"
                )
            bundle.execution_runtime.execution_event_runner = timer.run
            try:
                last_step = train_step(
                    bundle.execution_runtime,
                    batch=batch,
                )
            finally:
                bundle.execution_runtime.execution_event_runner = (
                    previous_runner
                )
            torch.cuda.synchronize()
            resource_abort()
            step_wall_ended_ns = time.perf_counter_ns()
            if not last_step.accepted:
                raise _TerminalAttemptFailure(
                    "production optimizer proposal was rejected: "
                    f"{last_step.failure_kind}",
                    phase="optimizer_proposal",
                    step=bundle.execution_runtime.update_counter,
                    pass_index=pass_index,
                    scientific_state_advanced=(
                        bundle.execution_runtime.update_counter > 0
                    ),
                )
            timing = timer.observation(
                data_wait_ns=data_ready_ns - step_wall_started_ns,
                evaluation_ns=0,
                checkpoint_ns=0,
                wall_ns=step_wall_ended_ns - step_wall_started_ns,
            )
            memory = capture_memory_observation()
            allocation_retries, oom_count = (
                _allocator_event_counters()
            )
            _append_metric_values(
                path=metrics_path,
                backend=backend,
                attempt=attempt,
                phase=last_step.phase_order[-1],
                split="train",
                step=bundle.execution_runtime.update_counter,
                pass_index=pass_index,
                values=_train_metric_values(
                    arm_spec=bundle.built_arm.record.spec,
                    step=last_step,
                    timing=timing,
                    memory=memory,
                    allocation_retries=allocation_retries,
                    oom_count=oom_count,
                    source=bundle.source_observation(),
                    numerical=bundle.numerical_observation(),
                ),
            )
            completed += 1
            boundary = (
                completed in source_lock.schedules.validation_boundary_batch_ordinals
                or completed == maximum
            )
            if boundary:
                try:
                    resource_abort()
                    evaluation_wall_started_ns = time.perf_counter_ns()
                    torch.cuda.synchronize()
                    evaluation_started_ns = time.perf_counter_ns()
                    validation_score = _score_validation(
                        bundle=bundle,
                        windows=validation_windows,
                        schedule=validation_schedule,
                        resource_abort=resource_abort,
                    )
                    last_nll = (
                        validation_score.summed_nll,
                        validation_score.counted_targets,
                    )
                    torch.cuda.synchronize()
                    resource_abort()
                    evaluation_ended_ns = time.perf_counter_ns()
                    boundary_memory = capture_memory_observation()
                    (
                        boundary_allocation_retries,
                        boundary_oom_count,
                    ) = _allocator_event_counters()
                    _append_boundary_metric(
                        path=metrics_path,
                        backend=backend,
                        attempt=attempt,
                        bundle=bundle,
                        nll_sum=last_nll[0],
                        counted_targets=last_nll[1],
                        cache_audit=validation_score.cache_audit,
                        pass_index=pass_index,
                        evaluation_ns=(
                            evaluation_ended_ns - evaluation_started_ns
                        ),
                        wall_ns=(
                            evaluation_ended_ns
                            - evaluation_wall_started_ns
                        ),
                        memory=boundary_memory,
                        allocation_retries=(
                            boundary_allocation_retries
                        ),
                        oom_count=boundary_oom_count,
                    )
                except _TerminalAttemptFailure:
                    raise
                except ProductionOperationError as exc:
                    raise _TerminalAttemptFailure(
                        str(exc),
                        phase="validation_scoring",
                        step=bundle.execution_runtime.update_counter,
                        pass_index=pass_index,
                        scientific_state_advanced=True,
                    ) from exc
                next_cursor = production_cursor_after_batches(
                    windows=windows,
                    schedule=schedule,
                    completed_batch_count=completed,
                )
                metric_records = validate_metric_log(metrics_path)
                resource_abort()
                checkpoint_started_ns = time.perf_counter_ns()
                latest_resume_identity = _save_attempt_checkpoint(
                    attempt_root=attempt_root,
                    backend=backend,
                    attempt=attempt,
                    training=training,
                    source_lock=source_lock,
                    readiness=readiness,
                    bundle=bundle,
                    plan_sha256=plan.plan.experiment_plan_sha256,
                    cursor=next_cursor,
                    schedule=schedule,
                    metric_head=metric_records[-1].record_sha256,
                    metric_next_ordinal=len(metric_records),
                    nll_numerator=last_nll[0],
                    nll_denominator=last_nll[1],
                    global_batch_step=(
                        pass_index * full_batches + completed
                    ),
                    cumulative_counted_targets=(
                        pass_index * windows.manifest.counted_targets
                        + next_cursor.counted_targets
                    ),
                    elapsed_seconds=(
                        time.perf_counter_ns() - attempt_started_ns
                    )
                    / 1_000_000_000.0,
                    role="resume_only",
                )
                latest_resume_path = _reopen_committed_resume_checkpoint(
                    sidecar,
                    expected_identity=latest_resume_identity,
                )
                resource_abort()
                checkpoint_ended_ns = time.perf_counter_ns()
                _append_checkpoint_metric(
                    path=metrics_path,
                    backend=backend,
                    attempt=attempt,
                    step=bundle.execution_runtime.update_counter,
                    pass_index=pass_index,
                    checkpoint_ns=(
                        checkpoint_ended_ns - checkpoint_started_ns
                    ),
                )
        cursor = None
    if (
        last_nll is None
        or bundle.execution_runtime.update_counter <= 0
        or latest_resume_identity is None
        or latest_resume_path is None
    ):
        raise _TerminalAttemptFailure(
            "production attempt omitted updates or its resumable boundary",
            phase="attempt_completion",
            step=bundle.execution_runtime.update_counter,
            pass_index=max(start_pass, attempt.pass_count - 1),
            scientific_state_advanced=(
                bundle.execution_runtime.update_counter > 0
            ),
        )
    records = validate_metric_log(metrics_path)
    resource_abort()
    validate_required_metric_families(
        records,
        arm_spec=bundle.built_arm.record.spec,
    )
    csv_path = attempt_root / "metrics.csv"
    csv_bytes = export_metrics_csv(
        log_path=metrics_path,
        output_path=csv_path,
        durability_backend=backend,
    )
    resource_abort()
    terminal: WT103CheckpointIdentity | None = None
    if attempt.role == "confirmation":
        final_pass_index = attempt.pass_count - 1
        final_windows, final_schedule = open_production_training_split(
            source_lock=source_lock,
            cache_root=cache_root,
            split="train",
            pass_index=final_pass_index,  # type: ignore[arg-type]
        )
        final_cursor = production_cursor_after_batches(
            windows=final_windows,
            schedule=final_schedule,
            completed_batch_count=full_batches,
        )
        terminal = _save_attempt_checkpoint(
            attempt_root=attempt_root,
            backend=backend,
            attempt=attempt,
            training=training,
            source_lock=source_lock,
            readiness=readiness,
            bundle=bundle,
            plan_sha256=plan.plan.experiment_plan_sha256,
            cursor=final_cursor,
            schedule=final_schedule,
            metric_head=records[-1].record_sha256,
            metric_next_ordinal=len(records),
            nll_numerator=last_nll[0],
            nll_denominator=last_nll[1],
            global_batch_step=attempt.pass_count * full_batches,
            cumulative_counted_targets=(
                attempt.pass_count
                * final_windows.manifest.counted_targets
            ),
            elapsed_seconds=(
                time.perf_counter_ns() - attempt_started_ns
            )
            / 1_000_000_000.0,
            role="terminal_scoring",
        )
        resource_abort()
    outcome = ProductionAttemptOutcome.create(
        attempt_sha256=attempt.attempt_sha256,
        validation_nll_sum=last_nll[0],
        validation_counted_targets=last_nll[1],
        validation_nll_per_token=last_nll[0] / last_nll[1],
        accepted_updates=bundle.execution_runtime.update_counter,
        terminal_checkpoint_identity_sha256=(
            None if terminal is None else terminal.checkpoint_identity_sha256
        ),
        metrics_jsonl_sha256=hashlib.sha256(
            metrics_path.read_bytes()
        ).hexdigest(),
        metrics_csv_sha256=hashlib.sha256(csv_bytes).hexdigest(),
    )
    checkpoints = (
        (latest_resume_identity,)
        if terminal is None
        else (latest_resume_identity, terminal)
    )
    checkpoint_paths = (
        (latest_resume_path,)
        if terminal is None
        else (
            latest_resume_path,
            attempt_root / "terminal-scoring.pt",
        )
    )
    checkpoint_artifacts = tuple(
        ArtifactIntegrityRecord.create(
            kind="file",
            relative_path=path.relative_to(attempt_root).as_posix(),
            size_bytes=identity.size_bytes,
            sha256=identity.checkpoint_payload_sha256,
        )
        for identity, path in zip(
            checkpoints,
            checkpoint_paths,
            strict=True,
        )
    )
    artifact_records = tuple(
        sorted(
            (
                _artifact_record(
                    csv_path,
                    relative_path="metrics.csv",
                ),
                _artifact_record(
                    metrics_path,
                    relative_path="metrics.jsonl",
                ),
                _artifact_record(
                    attempt_root / _LIVE_PRECISION_RUNTIME_EVIDENCE_NAME,
                    relative_path=_LIVE_PRECISION_RUNTIME_EVIDENCE_NAME,
                ),
            ),
            key=lambda item: item.relative_path,
        )
    )
    readiness_bundle = readiness.readiness_bundle
    if type(readiness_bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "attempt finalization lost Task 14 evidence"
        )
    manifest = finalize_run(
        reserved,
        disposition="success",
        checkpoints=checkpoints,
        checkpoint_artifact_records=checkpoint_artifacts,
        artifact_records=artifact_records,
        environment_sha256=(
            readiness_bundle.environment.environment_sha256
        ),
        provenance_sha256=(
            readiness_bundle.provenance.provenance_sha256
        ),
        ended_utc=_canonical_utc_timestamp(),
        monotonic_duration_seconds=(
            time.perf_counter_ns() - attempt_started_ns
        )
        / 1_000_000_000.0,
        failure_record_sha256=None,
        backend=backend,
    )
    return outcome, manifest


def _manifest_failure_record_sha256(
    manifest: RunManifestIdentity,
) -> str | None:
    manifest.__post_init__()
    document = _canonical_document(
        manifest.run_path / "run-manifest.json"
    )
    value = document.get("failure_record_sha256")
    if value is not None and (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionOperationError(
            "run manifest failure-record identity is malformed"
        )
    return value


def _failure_checkpoint_inventory(
    attempt_root: Path,
) -> tuple[
    tuple[WT103CheckpointIdentity, ...],
    tuple[ArtifactIntegrityRecord, ...],
    DataCursor | None,
]:
    sidecar = attempt_root / "resume-sidecar.json"
    if not sidecar.exists():
        return (), (), None
    _contract, identity, cursor, checkpoint_path = _checkpoint_sidecar(
        sidecar
    )
    record = ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=checkpoint_path.relative_to(
            attempt_root
        ).as_posix(),
        size_bytes=identity.size_bytes,
        sha256=identity.checkpoint_payload_sha256,
    )
    return (identity,), (record,), cursor


def _failure_artifact_inventory(
    attempt_root: Path,
) -> tuple[ArtifactIntegrityRecord, ...]:
    records: list[ArtifactIntegrityRecord] = []
    metrics_path = attempt_root / "metrics.jsonl"
    csv_path = attempt_root / "metrics.csv"
    failures_path = attempt_root / "failures.jsonl"
    precision_path = attempt_root / _LIVE_PRECISION_RUNTIME_EVIDENCE_NAME
    if metrics_path.exists():
        validate_metric_log(metrics_path)
        records.append(
            _artifact_record(
                metrics_path,
                relative_path="metrics.jsonl",
            )
        )
    if csv_path.exists():
        records.append(
            _artifact_record(
                csv_path,
                relative_path="metrics.csv",
            )
        )
    if precision_path.exists():
        records.append(
            _artifact_record(
                precision_path,
                relative_path=_LIVE_PRECISION_RUNTIME_EVIDENCE_NAME,
            )
        )
    records.append(
        _artifact_record(
            failures_path,
            relative_path="failures.jsonl",
        )
    )
    return tuple(sorted(records, key=lambda item: item.relative_path))


def _finalize_terminal_attempt_failure(
    *,
    reserved: ReservedRun,
    attempt: ProductionAttemptSpec,
    failure: _TerminalAttemptFailure,
    readiness: ProductionReadinessResult,
    backend: DurabilityBackend,
    monotonic_duration_seconds: float,
) -> _TerminalAttemptResult:
    if (
        type(reserved) is not ReservedRun
        or type(attempt) is not ProductionAttemptSpec
        or type(failure) is not _TerminalAttemptFailure
        or reserved.run_id != attempt.attempt_id
        or type(monotonic_duration_seconds) is not float
        or not math.isfinite(monotonic_duration_seconds)
        or monotonic_duration_seconds < 0.0
    ):
        raise ProductionOperationError(
            "terminal attempt finalization authority is invalid"
        )
    attempt_root = reserved.inprogress_path
    (
        checkpoints,
        checkpoint_artifacts,
        cursor,
    ) = _failure_checkpoint_inventory(attempt_root)
    failure_path = attempt_root / "failures.jsonl"
    existing = validate_failure_log(failure_path)
    if existing:
        terminal = existing[-1]
        if (
            len(existing) != 1
            or terminal.run_id != attempt.attempt_id
            or terminal.arm_id != attempt.arm_id
            or terminal.seed_id != attempt.seed_id
        ):
            raise ProductionOperationError(
                "retained terminal failure ledger differs from its attempt"
            )
    else:
        terminal = create_failure_record(
            ordinal=0,
            utc_timestamp=_canonical_utc_timestamp(),
            monotonic_ns=time.monotonic_ns(),
            run_id=attempt.attempt_id,
            arm_id=attempt.arm_id,
            seed_id=attempt.seed_id,
            phase=failure.phase,
            step=failure.step,
            pass_index=failure.pass_index,
            cursor_sha256=(
                None if cursor is None else cursor.cursor_sha256
            ),
            checkpoint_identity_sha256=(
                None
                if not checkpoints
                else checkpoints[0].checkpoint_identity_sha256
            ),
            retry_classification="not_retryable",
            scientific_state_advanced=(
                failure.scientific_state_advanced
            ),
            terminal_disposition="failed",
            exception=failure,
            previous_record_sha256=_ZERO_SHA256,
        )
        append_failure(
            failure_path,
            terminal,
            durability_backend=backend,
        )
        reopened = validate_failure_log(failure_path)
        if reopened != (terminal,):
            raise ProductionOperationError(
                "terminal failure ledger changed on durable reopen"
            )
    readiness_bundle = readiness.readiness_bundle
    if type(readiness_bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "failure finalization lost Task 14 evidence"
        )
    manifest = finalize_run(
        reserved,
        disposition="failure",
        checkpoints=checkpoints,
        checkpoint_artifact_records=checkpoint_artifacts,
        artifact_records=_failure_artifact_inventory(attempt_root),
        environment_sha256=(
            readiness_bundle.environment.environment_sha256
        ),
        provenance_sha256=(
            readiness_bundle.provenance.provenance_sha256
        ),
        ended_utc=_canonical_utc_timestamp(),
        monotonic_duration_seconds=monotonic_duration_seconds,
        failure_record_sha256=terminal.record_sha256,
        backend=backend,
    )
    result = _TerminalAttemptResult(
        manifest=manifest,
        failure=terminal,
    )
    result.__post_init__()
    return result


def _outcome_document(
    attempt: ProductionAttemptSpec,
    outcome: ProductionAttemptOutcome,
) -> dict[str, object]:
    return {
        "attempt": {
            field.name: getattr(attempt, field.name)
            for field in dataclasses.fields(attempt)
        },
        "outcome": {
            field.name: getattr(outcome, field.name)
            for field in dataclasses.fields(outcome)
        },
    }


def _reopen_attempt_outcome(
    *,
    attempt: ProductionAttemptSpec,
    manifest: RunManifestIdentity,
    training: TrainingConfig,
) -> ProductionAttemptOutcome:
    attempt.__post_init__()
    manifest.__post_init__()
    if (
        manifest.run_id != attempt.attempt_id
        or manifest.run_role != attempt.role
        or manifest.disposition != "success"
    ):
        raise ProductionOperationError(
            "terminal run manifest does not describe a successful attempt"
        )
    arm_spec = next(
        (
            arm
            for arm in training.endpoint_inventory.arms
            if arm.arm_id == attempt.arm_id
        ),
        None,
    )
    if arm_spec is None:
        raise ProductionOperationError(
            "attempt arm is outside the endpoint inventory"
        )
    metrics_path = manifest.run_path / "metrics.jsonl"
    csv_path = manifest.run_path / "metrics.csv"
    records = validate_metric_log(metrics_path)
    validate_required_metric_families(records, arm_spec=arm_spec)
    validation = tuple(
        record for record in records if record.split == "validation"
    )
    if not validation:
        raise ProductionOperationError(
            "successful attempt omitted validation metrics"
        )
    final = validation[-1]
    values = {value.name: value for value in final.values}
    required = (
        "prior_nll_sum",
        "prior_nll_per_token",
        "counted_targets",
    )
    if any(
        name not in values
        or values[name].applicability != "applicable"
        or values[name].value is None
        for name in required
    ):
        raise ProductionOperationError(
            "final validation metric projection is incomplete"
        )
    nll_sum = values["prior_nll_sum"].value
    nll_per_token = values["prior_nll_per_token"].value
    counted_float = values["counted_targets"].value
    assert nll_sum is not None
    assert nll_per_token is not None
    assert counted_float is not None
    counted_targets = int(counted_float)
    if (
        counted_float != counted_targets
        or counted_targets <= 0
        or nll_per_token != nll_sum / counted_targets
    ):
        raise ProductionOperationError(
            "final validation metric arithmetic changed"
        )
    document = _canonical_document(
        manifest.run_path / "run-manifest.json"
    )
    checkpoints = document.get("checkpoints")
    if type(checkpoints) is not list:
        raise ProductionOperationError(
            "run manifest checkpoint inventory is malformed"
        )
    terminal_shas = tuple(
        row.get("checkpoint_identity_sha256")
        for row in checkpoints
        if type(row) is dict
        and row.get("checkpoint_role") == "terminal_scoring"
    )
    expected_terminal_count = 1 if attempt.role == "confirmation" else 0
    if (
        len(terminal_shas) != expected_terminal_count
        or any(type(value) is not str for value in terminal_shas)
    ):
        raise ProductionOperationError(
            "run manifest terminal checkpoint inventory changed"
        )
    csv_payload = _regular_bytes(
        csv_path,
        maximum_bytes=512 * 1024 * 1024,
    )
    return ProductionAttemptOutcome.create(
        attempt_sha256=attempt.attempt_sha256,
        validation_nll_sum=nll_sum,
        validation_counted_targets=counted_targets,
        validation_nll_per_token=nll_per_token,
        accepted_updates=max(record.step for record in records),
        terminal_checkpoint_identity_sha256=(
            None if not terminal_shas else terminal_shas[0]
        ),
        metrics_jsonl_sha256=hashlib.sha256(
            _regular_bytes(
                metrics_path,
                maximum_bytes=512 * 1024 * 1024,
            )
        ).hexdigest(),
        metrics_csv_sha256=hashlib.sha256(csv_payload).hexdigest(),
    )


def _select_hyperparameters(
    document: dict[str, object],
    training: TrainingConfig,
) -> dict[str, tuple[float, float]]:
    rows = document["completed_outcomes"]
    if type(rows) is not list:
        raise ProductionOperationError("tuning outcome rows are malformed")
    expected_tuning = {
        attempt.attempt_sha256: attempt
        for attempt in _attempt_inventory(training, None)
    }
    parsed_tuning: list[
        tuple[ProductionAttemptSpec, ProductionAttemptOutcome]
    ] = []
    seen_tuning: set[str] = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"attempt", "outcome"}:
            raise ProductionOperationError(
                "production outcome row is malformed"
            )
        attempt_raw = row.get("attempt")
        outcome_raw = row.get("outcome")
        if type(attempt_raw) is not dict or type(outcome_raw) is not dict:
            raise ProductionOperationError(
                "production outcome payload is malformed"
            )
        try:
            observed_attempt = ProductionAttemptSpec(**attempt_raw)
            observed_outcome = ProductionAttemptOutcome(**outcome_raw)
        except (TypeError, ValueError) as exc:
            raise ProductionOperationError(
                "production outcome payload failed typed reopening"
            ) from exc
        if observed_outcome.attempt_sha256 != observed_attempt.attempt_sha256:
            raise ProductionOperationError(
                "production outcome belongs to another attempt"
            )
        if observed_attempt.role != "tuning":
            continue
        expected_attempt = expected_tuning.get(
            observed_attempt.attempt_sha256
        )
        if expected_attempt != observed_attempt:
            raise ProductionOperationError(
                "tuning outcome is outside the immutable experiment plan"
            )
        if observed_attempt.attempt_sha256 in seen_tuning:
            raise ProductionOperationError(
                "duplicate tuning attempt detected"
            )
        seen_tuning.add(observed_attempt.attempt_sha256)
        parsed_tuning.append((observed_attempt, observed_outcome))
    if seen_tuning != set(expected_tuning):
        raise ProductionOperationError(
            "tuning outcome inventory is incomplete"
        )
    selected: dict[str, tuple[float, float]] = {}
    for arm in training.endpoint_inventory.arms:
        candidates: dict[tuple[float, float], dict[int, float]] = {}
        for attempt, outcome in parsed_tuning:
            if attempt.arm_id == arm.arm_id:
                key = (
                    attempt.learning_rate,
                    attempt.weight_decay,
                )
                by_seed = candidates.setdefault(key, {})
                if attempt.seed_id in by_seed:
                    raise ProductionOperationError(
                        "duplicate tuning replicate detected"
                    )
                by_seed[attempt.seed_id] = (
                    outcome.validation_nll_per_token
                )
        expected_cells = {
            (float(learning_rate), float(weight_decay))
            for learning_rate in (
                training.profile.statistics.learning_rate_grid
            )
            for weight_decay in (
                training.profile.statistics.weight_decay_grid
            )
        }
        expected_seeds = set(
            training.profile.statistics.tuning_seed_ids
        )
        complete = {
            key: tuple(by_seed[seed] for seed in sorted(expected_seeds))
            for key, by_seed in candidates.items()
            if set(by_seed) == expected_seeds
        }
        if set(candidates) != expected_cells or set(complete) != expected_cells:
            raise ProductionOperationError(
                f"tuning inventory is incomplete for {arm.arm_id}"
            )
        selected[arm.arm_id] = min(
            complete,
            key=lambda key: (
                math.fsum(complete[key]) / len(complete[key]),
                key[0],
                key[1],
            ),
        )
    return selected


def _reopen_completed_attempt_prefix(
    *,
    experiment_root: Path,
    attempts: tuple[ProductionAttemptSpec, ...],
    training: TrainingConfig,
    plan: ExperimentPlanIdentity,
) -> tuple[
    tuple[ProductionAttemptSpec, ProductionAttemptOutcome, RunManifestIdentity],
    ...,
]:
    completed: list[
        tuple[
            ProductionAttemptSpec,
            ProductionAttemptOutcome,
            RunManifestIdentity,
        ]
    ] = []
    first_missing: int | None = None
    active_attempt_ids: list[str] = []
    for index, attempt in enumerate(attempts):
        final_path = (
            experiment_root / "runs" / attempt.attempt_id
        )
        inprogress_path = (
            experiment_root
            / "runs"
            / ".inprogress"
            / attempt.attempt_id
        )
        if (
            inprogress_path.is_dir()
            and (inprogress_path / "run-manifest.json").is_file()
        ):
            recover_terminal_run(
                experiment_root,
                attempt.attempt_id,
                plan=plan,
            )
        if inprogress_path.exists():
            active_attempt_ids.append(attempt.attempt_id)
        if not final_path.exists():
            if first_missing is None:
                first_missing = index
            continue
        if first_missing is not None:
            raise ProductionOperationError(
                "terminal attempt inventory is not an ordered prefix"
            )
        manifest = validate_run_manifest(
            final_path / "run-manifest.json"
        )
        if (
            manifest.run_id != attempt.attempt_id
            or manifest.run_role != attempt.role
            or manifest.experiment_plan_sha256
            != plan.plan.experiment_plan_sha256
            or manifest.tuning_attempt_key
            != _planned_tuning_attempt_key(
                attempt=attempt,
                plan=plan,
            )
        ):
            raise ProductionOperationError(
                "terminal run manifest differs from the attempt inventory"
            )
        if manifest.disposition == "failure":
            failures = validate_failure_log(
                manifest.run_path / "failures.jsonl"
            )
            failure_head = _manifest_failure_record_sha256(manifest)
            if (
                not failures
                or failures[-1].record_sha256 != failure_head
            ):
                raise ProductionOperationError(
                    "terminal failure manifest lost its failure ledger"
                )
            raise ProductionOperationError(
                "production experiment contains a terminal failed attempt: "
                f"{attempt.attempt_id}: {failures[-1].message}"
            )
        outcome = _reopen_attempt_outcome(
            attempt=attempt,
            manifest=manifest,
            training=training,
        )
        completed.append((attempt, outcome, manifest))
    if len(active_attempt_ids) > 1:
        raise ProductionOperationError(
            "multiple exact in-progress attempts are retained"
        )
    if active_attempt_ids:
        missing_index = (
            len(attempts) if first_missing is None else first_missing
        )
        if (
            missing_index >= len(attempts)
            or active_attempt_ids[0]
            != attempts[missing_index].attempt_id
        ):
            raise ProductionOperationError(
                "retained in-progress attempt is not the next inventory item"
            )
    return tuple(completed)


def _planned_tuning_attempt_key(
    *,
    attempt: ProductionAttemptSpec,
    plan: ExperimentPlanIdentity,
) -> str | None:
    if (
        type(attempt) is not ProductionAttemptSpec
        or type(plan) is not ExperimentPlanIdentity
    ):
        raise ProductionOperationError(
            "attempt-key projection requires exact typed authority"
        )
    attempt.__post_init__()
    plan.__post_init__()
    if attempt.role == "confirmation":
        return None
    if not 0 <= attempt.ordinal < len(plan.plan.tuning_attempt_keys):
        raise ProductionOperationError(
            "tuning ordinal is outside the immutable attempt inventory"
        )
    key = plan.plan.tuning_attempt_keys[attempt.ordinal]
    if (
        not key.startswith(f"tuning/{attempt.arm_id}/")
        or not key.endswith(f"/seed={attempt.seed_id}")
    ):
        raise ProductionOperationError(
            "tuning attempt identity differs from its ordered plan key"
        )
    return key


def _reserve_production_attempt(
    *,
    experiment_root: Path,
    attempt: ProductionAttemptSpec,
    plan: ExperimentPlanIdentity,
    readiness: ProductionReadinessResult,
    backend: DurabilityBackend,
    mode: Literal["train", "resume"],
) -> tuple[ReservedRun, bool]:
    tuning_attempt_key = _planned_tuning_attempt_key(
        attempt=attempt,
        plan=plan,
    )
    inprogress_path = (
        experiment_root
        / "runs"
        / ".inprogress"
        / attempt.attempt_id
    )
    if not inprogress_path.exists():
        return (
            reserve_run(
                experiment_root,
                attempt.attempt_id,
                run_role=attempt.role,
                started_utc=_canonical_utc_timestamp(),
                plan=plan,
                backend=backend,
                mode="new",
                tuning_attempt_key=tuning_attempt_key,
            ),
            False,
        )
    if mode != "resume":
        raise ProductionOperationError(
            "train mode cannot claim a retained in-progress attempt"
        )
    sidecar = inprogress_path / "resume-sidecar.json"
    if not sidecar.is_file():
        raise ProductionOperationError(
            "retained crash has no authenticated resume boundary"
        )
    _contract, checkpoint, cursor, _checkpoint_path = (
        _checkpoint_sidecar(sidecar)
    )
    readiness_bundle = readiness.readiness_bundle
    if type(readiness_bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "resume reservation lost Task 14 environment evidence"
        )
    owner_path = inprogress_path / "resume-owner.json"
    expected_owner: str | None = None
    if owner_path.exists():
        owner = _canonical_document(owner_path)
        lineage_sha = owner.get("lineage_sha256")
        if (
            owner.get("state") != "active"
            or type(lineage_sha) is not str
        ):
            raise ProductionOperationError(
                "retained resume owner is not active and exact"
            )
        expected_owner = lineage_sha
    lineage = reopen_resume_lineage_event(inprogress_path)
    if lineage is None:
        lineage = ResumeLineageEvent.create(
            parent_checkpoint=checkpoint,
            environment_sha256=(
                readiness_bundle.environment.environment_sha256
            ),
            cursor_sha256=cursor.cursor_sha256,
            reason=(
                "explicit resume of retained crash at authenticated boundary"
            ),
            resumed_utc=_canonical_utc_timestamp(),
        )
    elif expected_owner is None:
        later_resume_paths = (
            inprogress_path / "resume-lineage.jsonl",
            inprogress_path / "resume-leases",
            inprogress_path / "resume-owner-takeovers",
            inprogress_path / "resume-execution-started.json",
        )
        if any(path.exists() for path in later_resume_paths):
            raise ProductionOperationError(
                "ownerless resume intent contains later retry state"
            )
    elif expected_owner != lineage.lineage_sha256:
        raise ProductionOperationError(
            "retained resume lineage differs from its active owner"
        )
    return (
        reserve_run(
            experiment_root,
            attempt.attempt_id,
            run_role=attempt.role,
            started_utc=None,
            plan=plan,
            backend=backend,
            mode="resume",
            resume_lineage=lineage,
            expected_resume_owner_lineage_sha256=expected_owner,
            tuning_attempt_key=tuning_attempt_key,
        ),
        True,
    )


def _execute_reserved_attempt_under_lease(
    *,
    attempt: ProductionAttemptSpec,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    cache_root: Path,
    plan: ExperimentPlanIdentity,
    reserved: ReservedRun,
    backend: DurabilityBackend,
    resume_active: bool,
    sampler: NvidiaSmiPowerSampler,
    power_provider_identity_sha256: str,
    conservative_power_watts: float,
    ledger: ResourceUsageLedger,
    ledger_path: Path,
) -> tuple[
    ProductionAttemptOutcome,
    RunManifestIdentity,
    ResourceUsageLedger,
]:
    if (
        type(conservative_power_watts) is not float
        or not math.isfinite(conservative_power_watts)
        or conservative_power_watts <= 0.0
    ):
        raise ProductionOperationError(
            "attempt power-limit authority is invalid"
        )
    current_ledger = ledger
    ledger_lock = threading.Lock()

    def debit_usage(usage: _AttemptResourceUsage) -> ResourceUsageLedger:
        nonlocal current_ledger
        with ledger_lock:
            try:
                current_ledger = _debit_resource_usage(
                    ledger=current_ledger,
                    attempt=attempt,
                    usage=usage,
                    path=ledger_path,
                    backend=backend,
                )
            except BaseException:
                if ledger_path.exists():
                    current_ledger = _reopen_resource_usage_ledger(
                        path=ledger_path,
                        experiment_plan_sha256=(
                            current_ledger.experiment_plan_sha256
                        ),
                    )
                raise
            return current_ledger

    reserve_declared_ns = time.perf_counter_ns()
    reserve = _conservative_limit_usage(
        attempt=attempt,
        measurement_kind="prepaid_crash_tail_reserve",
        interval_ordinal=0,
        interval_started_ns=reserve_declared_ns,
        interval_ended_ns=(
            reserve_declared_ns
            + int(_CRASH_TAIL_RESERVE_SECONDS * 1_000_000_000)
        ),
        power_provider_identity_sha256=(
            power_provider_identity_sha256
        ),
        conservative_power_watts=conservative_power_watts,
    )
    debit_usage(reserve)
    if resume_active:
        consume_resume_execution_retry(
            reserved,
            backend=backend,
        )
    heartbeat_started_ns = time.perf_counter_ns()
    heartbeat_stop = threading.Event()
    heartbeat_failures: list[BaseException] = []

    def heartbeat_worker() -> None:
        try:
            _run_resource_usage_heartbeat(
                stop=heartbeat_stop,
                attempt=attempt,
                power_provider_identity_sha256=(
                    power_provider_identity_sha256
                ),
                conservative_power_watts=(
                    conservative_power_watts
                ),
                started_ns=heartbeat_started_ns,
                debit=debit_usage,
            )
        except BaseException as exc:
            heartbeat_failures.append(exc)
            heartbeat_stop.set()

    heartbeat = threading.Thread(
        target=heartbeat_worker,
        name="vfe4-resource-usage-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    operation_started_ns = time.perf_counter_ns()

    def assert_heartbeat_healthy() -> None:
        if heartbeat_failures:
            raise _ResourceHeartbeatFailure(
                "independent resource heartbeat failed; prepaid crash-tail "
                "reserve is the only remaining accounting runway"
            ) from heartbeat_failures[0]

    def operation() -> object:
        try:
            return _execute_attempt(
                attempt=attempt,
                training=training,
                source_lock=source_lock,
                readiness=readiness,
                cache_root=cache_root,
                plan=plan,
                reserved=reserved,
                backend=backend,
                resume_active=resume_active,
                resource_abort=assert_heartbeat_healthy,
            )
        except _TerminalAttemptFailure as exc:
            return _finalize_terminal_attempt_failure(
                reserved=reserved,
                attempt=attempt,
                failure=exc,
                readiness=readiness,
                backend=backend,
                monotonic_duration_seconds=(
                    time.perf_counter_ns() - operation_started_ns
                )
                / 1_000_000_000.0,
            )

    measured_failure: _ResourceMeasuredOperationFailure | None = None
    unexpected_error: BaseException | None = None
    result: object = None
    usage: _AttemptResourceUsage | None = None
    try:
        try:
            result, usage = _measure_attempt_resource_usage(
                operation,
                sampler=sampler,
                power_provider_identity_sha256=(
                    power_provider_identity_sha256
                ),
                conservative_power_watts=(
                    conservative_power_watts
                ),
            )
        except _ResourceMeasuredOperationFailure as exc:
            measured_failure = exc
            usage = exc.usage
        except BaseException as exc:
            unexpected_error = exc
    finally:
        heartbeat_stop.set()
        heartbeat.join()
    if unexpected_error is not None:
        if hasattr(unexpected_error, "add_note"):
            unexpected_error.add_note(
                "prepaid crash-tail and completed heartbeat usage remain "
                "durably charged without refund: "
                f"{current_ledger.ledger_sha256}"
            )
            if heartbeat_failures:
                unexpected_error.add_note(
                    "independent resource heartbeat also failed: "
                    f"{heartbeat_failures[0]!r}"
                )
        raise unexpected_error
    if usage is None:
        raise ProductionOperationError(
            "resource measurement returned no terminal usage"
        )
    updated = debit_usage(usage)
    if measured_failure is not None:
        error = measured_failure.error
        if hasattr(error, "add_note"):
            error.add_note(
                "measured usage plus prepaid conservative usage was "
                "durably debited without refund before propagation: "
                f"{updated.ledger_sha256}"
            )
            if heartbeat_failures:
                error.add_note(
                    "independent resource heartbeat also failed: "
                    f"{heartbeat_failures[0]!r}"
                )
        if heartbeat_failures:
            raise error from heartbeat_failures[0]
        raise error
    if heartbeat_failures:
        heartbeat_error = heartbeat_failures[0]
        if hasattr(heartbeat_error, "add_note"):
            heartbeat_error.add_note(
                "measured usage plus prepaid conservative usage was "
                "durably debited without refund before propagation: "
                f"{updated.ledger_sha256}"
            )
        raise heartbeat_error
    if type(result) is _TerminalAttemptResult:
        result.__post_init__()
        _emit_attempt_finished(
            attempt=attempt,
            manifest=result.manifest,
            outcome=None,
            usage=usage,
            ledger=updated,
        )
        raise ProductionOperationError(
            "production attempt terminated with a durable failure record: "
            f"{result.failure.message}; "
            f"resource_ledger_sha256={updated.ledger_sha256}"
        )
    if (
        type(result) is not tuple
        or len(result) != 2
        or type(result[0]) is not ProductionAttemptOutcome
        or type(result[1]) is not RunManifestIdentity
        or result[0].attempt_sha256 != attempt.attempt_sha256
        or result[1].run_id != attempt.attempt_id
        or result[1].disposition != "success"
    ):
        raise ProductionOperationError(
            "resource-measured attempt returned a foreign terminal result; "
            "resource usage was durably debited before rejection: "
            f"{updated.ledger_sha256}"
        )
    _emit_attempt_finished(
        attempt=attempt,
        manifest=result[1],
        outcome=result[0],
        usage=usage,
        ledger=updated,
    )
    return result[0], result[1], updated


def _execute_reserved_attempt(
    *,
    attempt: ProductionAttemptSpec,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    cache_root: Path,
    plan: ExperimentPlanIdentity,
    reserved: ReservedRun,
    backend: DurabilityBackend,
    resume_active: bool,
    sampler: NvidiaSmiPowerSampler,
    power_provider_identity_sha256: str,
    conservative_power_watts: float,
    ledger: ResourceUsageLedger,
    ledger_path: Path,
) -> tuple[
    ProductionAttemptOutcome,
    RunManifestIdentity,
    ResourceUsageLedger,
]:
    try:
        return _execute_reserved_attempt_under_lease(
            attempt=attempt,
            training=training,
            source_lock=source_lock,
            readiness=readiness,
            cache_root=cache_root,
            plan=plan,
            reserved=reserved,
            backend=backend,
            resume_active=resume_active,
            sampler=sampler,
            power_provider_identity_sha256=(
                power_provider_identity_sha256
            ),
            conservative_power_watts=conservative_power_watts,
            ledger=ledger,
            ledger_path=ledger_path,
        )
    finally:
        release_run_execution_lease(reserved)


def _require_resource_event_prefix(
    ledger: ResourceUsageLedger,
    completed: tuple[
        tuple[
            ProductionAttemptSpec,
            ProductionAttemptOutcome,
            RunManifestIdentity,
        ],
        ...,
    ],
    *,
    allow_extension: bool = False,
) -> None:
    ledger.__post_init__()
    expected = tuple(item[0].attempt_id for item in completed)
    observed_groups: list[str] = []
    segment_ordinals: dict[str, list[int]] = {}
    for event in ledger.events:
        if not observed_groups or observed_groups[-1] != event.attempt_id:
            if event.attempt_id in segment_ordinals:
                raise ProductionOperationError(
                    "resource usage segments for one attempt are not contiguous"
                )
            observed_groups.append(event.attempt_id)
            segment_ordinals[event.attempt_id] = []
        segment_ordinals[event.attempt_id].append(event.segment_ordinal)
    for attempt_id, ordinals in segment_ordinals.items():
        if tuple(ordinals) != tuple(range(len(ordinals))):
            raise ProductionOperationError(
                "resource usage segment ordinals are discontinuous for "
                f"{attempt_id}"
            )
    observed = tuple(observed_groups)
    prefix_matches = observed[: len(expected)] == expected
    extension_count = len(observed) - len(expected)
    matches = prefix_matches and (
        extension_count >= 0
        and (
            allow_extension
            or extension_count <= 1
        )
    )
    if (
        type(allow_extension) is not bool
        or not matches
    ):
        raise ProductionOperationError(
            "resource usage ledger differs from finalized successful attempts"
        )


def _emit_run_resolved(
    *,
    training: TrainingConfig,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    expected_plan: ExperimentPlan,
    mode: Literal["train", "resume"],
) -> None:
    finalized = source_lock.finalized_source
    emit_progress(
        "run_resolved",
        mode=mode,
        config_sha256=training.experiment_config_sha256,
        source_lock_sha256=source_lock.source_lock_sha256,
        source_record_sha256=finalized.record_sha256,
        source_member_payload_sha256s=[
            member.payload_sha256 for member in finalized.members
        ],
        readiness_result_sha256=readiness.result_sha256,
        readiness_bundle_sha256=getattr(
            readiness.readiness_bundle,
            "bundle_sha256",
        ),
        readiness_assessment_sha256=getattr(
            readiness.readiness,
            "assessment_sha256",
        ),
        readiness_token_sha256=getattr(
            readiness.readiness_token,
            "token_sha256",
        ),
        tokenizer_spec_sha256=source_lock.tokenizer.spec_sha256,
        tokenizer_tables_sha256=source_lock.tokenizer.tokenizer_tables_sha256,
        token_cache_record_sha256s=[
            cache.record_sha256 for cache in source_lock.token_caches
        ],
        token_cache_set_sha256=(
            finalized.production_token_cache_set_sha256
        ),
        window_manifest_sha256s=[
            manifest.manifest_sha256
            for manifest in source_lock.schedules.window_manifests
        ],
        schedule_sha256s=list(source_lock.schedules.schedule_sha256s),
        schedule_set_sha256=source_lock.schedules.schedule_set_sha256,
        endpoint_inventory_sha256=(
            training.endpoint_inventory.endpoint_inventory_sha256
        ),
        expected_experiment_plan_sha256=(
            expected_plan.experiment_plan_sha256
        ),
    )


def _emit_plan_ready(
    *,
    experiment_root: Path,
    plan: ExperimentPlanIdentity,
    ledger_path: Path,
) -> None:
    emit_progress(
        "plan_ready",
        experiment_id=plan.plan.experiment_id,
        experiment_plan_sha256=plan.plan.experiment_plan_sha256,
        experiment_plan_identity_sha256=plan.identity_sha256,
        plan_path=str(plan.plan_path),
        predicted_artifact_paths={
            "resource_usage_ledger": str(ledger_path),
            "terminal_experiment_index": str(
                experiment_root / _TERMINAL_INDEX_NAME
            ),
            "per_run_relative_paths": list(
                plan.plan.expected_run_artifact_paths
            ),
            "group_relative_paths": list(
                plan.plan.expected_group_artifact_paths
            ),
        },
    )


def _emit_resources_forecast(bundle: Task14ReadinessBundle) -> None:
    forecast = bundle.resource_forecast
    emit_progress(
        "resources_forecast",
        forecast_sha256=forecast.forecast_sha256,
        disk_forecast_sha256=forecast.disk_forecast_sha256,
        available_disk_bytes=forecast.available_disk_bytes,
        raw={
            "gpu_hours": forecast.raw_gpu_hours,
            "wall_hours": forecast.raw_wall_hours,
            "energy_kwh": forecast.raw_energy_kwh,
        },
        forecast={
            "gpu_hours": forecast.forecast_gpu_hours,
            "wall_hours": forecast.forecast_wall_hours,
            "energy_kwh": forecast.forecast_energy_kwh,
        },
        ceilings={
            "gpu_hours": forecast.maximum_gpu_hours,
            "wall_hours": forecast.maximum_wall_hours,
            "energy_kwh": forecast.maximum_energy_kwh,
        },
        forecast_headroom_factor=forecast.forecast_headroom_factor,
        observed_disk_bytes=None,
        observed_disk_bytes_status="unavailable",
    )


def _emit_experiment_finished(
    *,
    result: ProductionTrainingResult,
    experiment_index_path: Path,
) -> None:
    emit_progress(
        "experiment_finished",
        status=result.status,
        tuning_attempt_count=result.tuning_attempt_count,
        confirmation_attempt_count=result.confirmation_attempt_count,
        completed_attempt_count=result.completed_attempt_count,
        selected_hyperparameter_sha256=(
            result.selected_hyperparameter_sha256
        ),
        experiment_index_path=str(experiment_index_path),
        artifact_path=str(experiment_index_path),
    )


def run_production_attempts(
    *,
    training: TrainingConfig,
    paths: object,
    source_lock: ProductionSourceLock,
    readiness: ProductionReadinessResult,
    mode: Literal["train", "resume"],
) -> ProductionTrainingResult:
    """Run or resume the complete equal-tuning and confirmation inventory."""

    if (
        type(training) is not TrainingConfig
        or training.operation != mode
        or type(source_lock) is not ProductionSourceLock
        or type(readiness) is not ProductionReadinessResult
        or readiness.status is not GateStatus.PASS
        or readiness.readiness_token is None
        or mode not in ("train", "resume")
    ):
        raise ProductionOperationError(
            "production attempt authority is incomplete"
        )
    source_lock.__post_init__()
    readiness.__post_init__()
    if (
        getattr(readiness.readiness_token, "finalized_source_record_sha256")
        != source_lock.finalized_source.record_sha256
    ):
        raise ProductionOperationError(
            "readiness token belongs to another finalized source"
        )
    cache_root = getattr(paths, "cache_root", None)
    run_root = getattr(paths, "run_root", None)
    declared_plan = getattr(paths, "resume_experiment_plan_path", None)
    if (
        not isinstance(cache_root, Path)
        or not isinstance(run_root, Path)
        or not isinstance(declared_plan, Path)
    ):
        raise ProductionOperationError(
            "production attempt paths are malformed"
        )
    if mode == "resume":
        experiment_root = declared_plan.parent
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
    else:
        run_root.mkdir(parents=True, exist_ok=True)
    backend = _backend()
    if mode == "resume":
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
    if backend.probe(run_root).status != "pass":
        raise ProductionOperationError(
            "production run-root durability probe failed"
        )
    expected_plan = _production_experiment_plan(
        training=training,
        source_lock=source_lock,
        readiness=readiness,
    )
    _emit_run_resolved(
        training=training,
        source_lock=source_lock,
        readiness=readiness,
        expected_plan=expected_plan,
        mode=mode,
    )
    if mode == "train":
        experiment_root = run_root / expected_plan.experiment_id
        plan = publish_experiment_plan(
            experiment_root,
            expected_plan,
            backend=backend,
        )
        ledger = ResourceUsageLedger.create(
            experiment_plan_sha256=(
                expected_plan.experiment_plan_sha256
            ),
            resource_profile=training.profile.resources,
        )
        ledger_path = experiment_root / "resource-usage-ledger.json"
        _publish_resource_usage_ledger(
            path=ledger_path,
            ledger=ledger,
            backend=backend,
        )
    else:
        if declared_plan.name != _PLAN_NAME:
            raise ProductionOperationError(
                "resume path must name the exact experiment-plan.json"
            )
        if (
            experiment_root.name != expected_plan.experiment_id
            or experiment_root.absolute().parent
            != run_root.absolute()
        ):
            raise ProductionOperationError(
                "resume plan is outside the exact experiment root"
            )
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
        plan = _reopen_experiment_plan_identity(
            plan_path=declared_plan,
            expected_plan=expected_plan,
            readiness=readiness,
        )
        ledger_path = experiment_root / "resource-usage-ledger.json"
        ledger = _reopen_resource_usage_ledger(
            path=ledger_path,
            experiment_plan_sha256=(
                expected_plan.experiment_plan_sha256
            ),
        )
    _assert_resource_headroom(ledger)
    readiness_bundle = readiness.readiness_bundle
    if type(readiness_bundle) is not Task14ReadinessBundle:
        raise ProductionOperationError(
            "production attempts lost exact Task 14 readiness evidence"
        )
    _emit_plan_ready(
        experiment_root=experiment_root,
        plan=plan,
        ledger_path=ledger_path,
    )
    _emit_resources_forecast(readiness_bundle)
    try:
        power_provider, power_sampler = (
            discover_nvidia_smi_power_provider()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProductionOperationError(
            "production power provider discovery failed"
        ) from exc
    expected_power_identity = (
        readiness_bundle.resource_forecast.power_provider_identity_sha256
    )
    if (
        expected_power_identity is None
        or power_provider.identity_sha256 != expected_power_identity
        or power_provider.sample_interval_ms
        != training.profile.resources.power_sample_interval_ms
    ):
        raise ProductionOperationError(
            "live power provider differs from the readiness forecast"
        )
    conservative_power_watts = (
        _frozen_readiness_conservative_power_watts(
            resource_forecast=readiness_bundle.resource_forecast,
            live_power_provider=power_provider,
        )
    )

    tuning = _attempt_inventory(training, None)
    if len(tuning) != expected_plan.tuning_attempt_count:
        raise ProductionOperationError(
            "tuning attempt inventory differs from the immutable plan"
        )
    if mode == "resume":
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
    completed_tuning = list(
        _reopen_completed_attempt_prefix(
            experiment_root=experiment_root,
            attempts=tuning,
            training=training,
            plan=plan,
        )
    )
    _require_resource_event_prefix(
        ledger,
        tuple(completed_tuning),
        allow_extension=(len(completed_tuning) == len(tuning)),
    )
    for phase_position, attempt in enumerate(
        tuning[len(completed_tuning) :],
        start=len(completed_tuning) + 1,
    ):
        if mode == "resume":
            _validate_resume_state_boundary(
                run_root=run_root,
                experiment_root=experiment_root,
                declared_plan=declared_plan,
            )
        reserved, resume_active = _reserve_production_attempt(
            experiment_root=experiment_root,
            attempt=attempt,
            plan=plan,
            readiness=readiness,
            backend=backend,
            mode=mode,
        )
        _emit_attempt_started_or_release(
            attempt=attempt,
            reserved=reserved,
            phase_position=phase_position,
            phase_total=len(tuning),
            resume_active=resume_active,
        )
        outcome, manifest, ledger = _execute_reserved_attempt(
            attempt=attempt,
            training=training,
            source_lock=source_lock,
            readiness=readiness,
            cache_root=cache_root,
            plan=plan,
            reserved=reserved,
            backend=backend,
            resume_active=resume_active,
            sampler=power_sampler,
            power_provider_identity_sha256=(
                power_provider.identity_sha256
            ),
            conservative_power_watts=conservative_power_watts,
            ledger=ledger,
            ledger_path=ledger_path,
        )
        completed_tuning.append((attempt, outcome, manifest))
        _require_resource_event_prefix(ledger, tuple(completed_tuning))

    selection_document: dict[str, object] = {
        "completed_outcomes": [
            _outcome_document(attempt, outcome)
            for attempt, outcome, _manifest in completed_tuning
        ]
    }
    selected = _select_hyperparameters(
        selection_document,
        training,
    )
    selected_document = {
        arm_id: {
            "learning_rate": values[0],
            "weight_decay": values[1],
        }
        for arm_id, values in selected.items()
    }
    confirmation = _attempt_inventory(training, selected)
    if len(confirmation) != expected_plan.terminal_checkpoint_count:
        raise ProductionOperationError(
            "confirmation inventory differs from the immutable plan"
        )
    if mode == "resume":
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
    completed_confirmation = list(
        _reopen_completed_attempt_prefix(
            experiment_root=experiment_root,
            attempts=confirmation,
            training=training,
            plan=plan,
        )
    )
    completed_all = [*completed_tuning, *completed_confirmation]
    _require_resource_event_prefix(ledger, tuple(completed_all))
    for phase_position, attempt in enumerate(
        confirmation[len(completed_confirmation) :],
        start=len(completed_confirmation) + 1,
    ):
        if mode == "resume":
            _validate_resume_state_boundary(
                run_root=run_root,
                experiment_root=experiment_root,
                declared_plan=declared_plan,
            )
        reserved, resume_active = _reserve_production_attempt(
            experiment_root=experiment_root,
            attempt=attempt,
            plan=plan,
            readiness=readiness,
            backend=backend,
            mode=mode,
        )
        _emit_attempt_started_or_release(
            attempt=attempt,
            reserved=reserved,
            phase_position=phase_position,
            phase_total=len(confirmation),
            resume_active=resume_active,
        )
        outcome, manifest, ledger = _execute_reserved_attempt(
            attempt=attempt,
            training=training,
            source_lock=source_lock,
            readiness=readiness,
            cache_root=cache_root,
            plan=plan,
            reserved=reserved,
            backend=backend,
            resume_active=resume_active,
            sampler=power_sampler,
            power_provider_identity_sha256=(
                power_provider.identity_sha256
            ),
            conservative_power_watts=conservative_power_watts,
            ledger=ledger,
            ledger_path=ledger_path,
        )
        completed_confirmation.append((attempt, outcome, manifest))
        completed_all.append((attempt, outcome, manifest))
        _require_resource_event_prefix(ledger, tuple(completed_all))

    _require_resource_event_prefix(ledger, tuple(completed_all))
    run_manifests = tuple(item[2] for item in completed_all)
    if len(run_manifests) != len(tuning) + len(confirmation):
        raise ProductionOperationError(
            "successful manifest inventory is incomplete"
        )
    if mode == "resume":
        _validate_resume_state_boundary(
            run_root=run_root,
            experiment_root=experiment_root,
            declared_plan=declared_plan,
        )
    experiment_index = publish_experiment_index(
        experiment_root,
        plan=plan,
        run_manifests=run_manifests,
        stage="pretest",
        artifact_records=(),
        backend=backend,
    )
    if experiment_index.index_path != experiment_root / _TERMINAL_INDEX_NAME:
        raise ProductionOperationError(
            "published experiment index differs from the exact terminal path"
        )
    selected_sha = owned_sha256(
        "vfe4.wt103.selected-hyperparameters.v1",
        selected_document,
    )
    values = {
        "schema_version": _RESULT_SCHEMA,
        "mode": mode,
        "config_sha256": training.experiment_config_sha256,
        "source_lock_sha256": source_lock.source_lock_sha256,
        "readiness_result_sha256": readiness.result_sha256,
        "experiment_plan_sha256": (
            expected_plan.experiment_plan_sha256
        ),
        "experiment_index_path": str(experiment_index.index_path),
        "tuning_attempt_count": len(tuning),
        "confirmation_attempt_count": len(confirmation),
        "completed_attempt_count": len(completed_all),
        "selected_hyperparameter_sha256": selected_sha,
        "status": "COMPLETE",
        "heldout_test_opened": False,
    }
    result = ProductionTrainingResult(
        **values,
        result_sha256=owned_sha256(
            "vfe4.wt103.production-training-result.v1",
            values,
        ),
    )  # type: ignore[arg-type]
    _emit_experiment_finished(
        result=result,
        experiment_index_path=experiment_index.index_path,
    )
    return result


__all__ = [
    "ProductionAttemptOutcome",
    "ProductionAttemptSpec",
    "ProductionTrainingResult",
    "run_production_attempts",
]
