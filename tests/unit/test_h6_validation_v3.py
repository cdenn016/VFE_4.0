from __future__ import annotations

import hashlib
import inspect
import io
import json
import zipfile
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vfe4.artifacts.h6_prediction_v3 as prediction_artifacts
import vfe4.training.h6_validation_v3 as validation_v3
from vfe4.artifacts.atomic import (
    ArtifactPublicationError,
    canonical_json_bytes as artifact_json_bytes,
)
from vfe4.artifacts.h6_prediction_v3 import (
    H6CheckpointCandidateV3,
    H6CheckpointSelectionV3,
    H6EndpointTuningSelectionV3,
    H6TuningSelectionV3,
    H6ValidationBundleV3,
    H6ValidationRecordV3,
    bind_h6_checkpoint_selection_v3,
    publish_h6_validation_bundle_v3,
    read_h6_validation_bundle_v3,
    select_h6_tuning_v3,
)
from vfe4.config import (
    H6ArchiveMemberExpectation,
    H6DataConfig,
    H6ObservedArchive,
)
from vfe4.data.access import (
    OpeningCapabilityError,
    issue_h6_validation_capability_v3,
)
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.h6_sealed_store_v3 import (
    AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME,
    reopen_authenticated_blinded_store_v3,
)
from vfe4.data.windows import build_causal_windows
from vfe4.data.wikitext2 import (
    WIKITEXT2_RAW_URL,
    H6DataAcquisitionRequest,
    _acquire_wikitext2_blinded,
)
from vfe4.training.arms import build_arm_model
from vfe4.training.checkpoint_v3 import (
    H6CheckpointV3,
    capture_h6_checkpoint_v3,
)
from vfe4.training.h6_experiment_v3 import (
    H6_CONFIRMATORY_SEEDS_V3,
    H6_TUNED_ENDPOINT_CONFIG_IDS_V3,
    H6_TUNING_CELLS_V3,
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    H6TuningCellV3,
    plan_h6_experiment_v3,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_POLICY_V3,
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6MatchingSetV3,
    H6TrainingWorkloadV3,
)
from vfe4.training.matching import (
    A5_REFERENCE_ALLOCATION,
    H6_ADAMW_POLICY,
    endpoint_formula_profile,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    TrainingPhase,
    VocabularyIdentity,
)
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6ObjectiveManifestV3,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
    H6_COUNTER_MAPPING_SHA256,
    H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:Failed to find (cuobjdump|nvdisasm)\\.exe:UserWarning"
)

_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = hashlib.sha256(b"dirty").hexdigest()
_EXPERIMENT_CONFIG_SHA256 = hashlib.sha256(b"experiment").hexdigest()
_DATA_SHA256 = hashlib.sha256(b"data").hexdigest()
_ACCESS_POLICY_SHA256 = hashlib.sha256(b"access").hexdigest()
_MEMBERS = (
    "wikitext-2-raw/wiki.train.raw",
    "wikitext-2-raw/wiki.valid.raw",
    "wikitext-2-raw/wiki.test.raw",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _template(config_id: str, vocabulary: VocabularyIdentity) -> ArmConfig:
    profile = endpoint_formula_profile(config_id)
    if config_id == "h6-a0-transformer-v2":
        allocation = CapacityAllocation.create(
            emission_width=52,
            latent_width=None,
            recognition_width=None,
        )
    elif not profile.latent_enabled:
        allocation = CapacityAllocation.create(
            emission_width=64,
            latent_width=None,
            recognition_width=None,
        )
    elif profile.prior_variant == "parent_specific_pooled_prefix":
        allocation = CapacityAllocation.create(
            emission_width=89,
            latent_width=2,
            recognition_width=113,
            prior_context_width=6,
        )
    else:
        allocation = A5_REFERENCE_ALLOCATION
    return ArmConfig.create(
        arm=ArmId(profile.arm),
        config_id=config_id,
        vocabulary=vocabulary,
        horizon=32,
        latent_enabled=profile.latent_enabled,
        state_channel_enabled=profile.channel_count >= 1,
        model_channel_enabled=profile.channel_count == 2,
        source_mode=profile.source_mode,
        map_mode=profile.map_mode,
        recognition_family=profile.recognition_family,
        recognition_conditioning=profile.recognition_conditioning,
        prior_variant=profile.prior_variant,
        mixture_mode=profile.mixture_mode,
        objective_kind=profile.objective_kind,
        capacity_allocation=allocation,
    )


def _plan_for_data(
    *,
    data_identity_sha256: str,
    access_policy_sha256: str,
    train_token_count: int,
    train_token_sha256: str,
) -> tuple[
    H6ExperimentPlanV3,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
]:
    vocabulary = ByteTokenizerV1().vocabulary_identity
    matching = H6MatchingSetV3.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        workload=H6TrainingWorkloadV3.from_train_tokens(
            train_token_count=train_token_count,
            train_token_sha256=train_token_sha256,
        ),
        endpoint_templates=tuple(
            _template(config_id, vocabulary)
            for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        ),
    )
    runtime = H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version="2.10.0.dev20251210+cu128",
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )
    estimator = H6RecognitionEstimatorSpec.create()
    schedule = H6TrainingScheduleV3.create(
        outer=H6OuterSchedule.create(
            optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256
        ),
        endpoint_phases=tuple(
            H6ArmPhaseSchedule.create(
                endpoint_config_sha256=config.config_sha256,
                latent_enabled=config.latent_enabled,
                phases=(
                    (
                        TrainingPhase.RECOGNITION_ADAMW,
                        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
                        TrainingPhase.MODEL_ADAMW,
                    )
                    if config.latent_enabled
                    else (TrainingPhase.MODEL_CE_ADAMW,)
                ),
            )
            for config in matching.endpoint_configs
        ),
        estimator=estimator,
        runtime=runtime,
    )
    readiness = H6PredictionV3ReadinessToken.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        experiment_config_sha256=_EXPERIMENT_CONFIG_SHA256,
        correctness_manifests=(
            ("H1", _digest("H1")),
            ("H2", _digest("H2")),
            ("H3", _digest("H3")),
            ("H5", _digest("H5")),
        ),
        h1_prefix_prior_manifest_sha256=_digest("h1-manifest"),
        h1_prefix_prior_generative_factor_schema_sha256=_digest("h1-schema"),
        smc_bias_semantics_sha256=_digest("smc-bias"),
        smc_validation_manifest_sha256=_digest("smc-validation"),
        prefix_certificate_set_sha256=_digest("prefix-certificates"),
        h5_update_binding_sha256=_digest("h5-update"),
        critical_values_sha256=_digest("critical-values"),
        endpoint_smc_protocol_sha256=_digest("endpoint-smc"),
        attribution_matrix_sha256=_digest("attribution"),
        objective_gate_spec_sha256=_digest("objective-gate"),
        matching_policy_sha256=H6_MATCHING_POLICY_V3.policy_sha256,
        matching_set_sha256=matching.matching_set_sha256,
        training_schedule_sha256=schedule.schedule_sha256,
        recognition_estimator_sha256=estimator.estimator_sha256,
        runtime_identity_sha256=runtime.runtime_identity_sha256,
        counter_mapping_sha256=H6_COUNTER_MAPPING_SHA256,
        phase_ownership_sha256=H6_PHASE_OWNERSHIP_SHA256,
        objective_manifest_schema_sha256=H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256,
        data_identity_sha256=data_identity_sha256,
        access_policy_sha256=access_policy_sha256,
    )
    return (
        plan_h6_experiment_v3(
            readiness=readiness,
            matching_set=matching,
            training_schedule=schedule,
            runtime_identity=runtime,
        ),
        runtime,
        readiness,
    )


@cache
def _authorities() -> tuple[
    H6ExperimentPlanV3,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
]:
    return _plan_for_data(
        data_identity_sha256=_DATA_SHA256,
        access_policy_sha256=_ACCESS_POLICY_SHA256,
        train_token_count=258,
        train_token_sha256=_digest("train-tokens"),
    )


class _TinyState(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.linspace(-0.25, 0.25, width, dtype=torch.float64)
        )


def _adamw(
    module: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {
                "params": tuple(module.parameters()),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    for parameter in module.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _terminal_checkpoint(
    planned_attempt: H6PlannedAttemptV3,
    *,
    runtime: H6PredictionRuntimeIdentity,
    cell: H6TuningCellV3,
    model: nn.Module | None = None,
    draw_block: int | None = None,
    counter_consumption_sha256: str | None = None,
    permutation_sha256: str | None = None,
) -> H6CheckpointV3:
    if model is None:
        model = _TinyState()
    modules: list[tuple[str, nn.Module]] = [("model", model)]
    optimizers: list[tuple[str, torch.optim.AdamW]] = [
        (
            "model",
            _adamw(
                model,
                learning_rate=cell.learning_rate,
                weight_decay=cell.weight_decay,
            ),
        )
    ]
    latent = planned_attempt.attempt_spec.recognition_factory_sha256 is not None
    if latent:
        recognition = _TinyState()
        modules.append(("recognition", recognition))
        optimizers.append(
            (
                "recognition",
                _adamw(
                    recognition,
                    learning_rate=cell.learning_rate,
                    weight_decay=cell.weight_decay,
                ),
            )
        )
    terminal_phase = (
        TrainingPhase.MODEL_CE_ADAMW
        if planned_attempt.attempt_spec.recognition_factory_sha256 is None
        else TrainingPhase.MODEL_ADAMW
    )
    next_phase = (
        TrainingPhase.RECOGNITION_ADAMW
        if latent
        else TrainingPhase.MODEL_CE_ADAMW
    )
    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=planned_attempt.attempt_spec.attempt_spec_sha256,
        pass_index=planned_attempt.terminal_pass_index,
        batch_index=planned_attempt.terminal_batch_index,
        next_phase=next_phase,
        example_ordinal=planned_attempt.terminal_example_ordinal,
        draw_block=(
            getattr(planned_attempt, "terminal_draw_block", 1)
            if draw_block is None
            else draw_block
        ),
        counter_consumption_sha256=(
            getattr(
                planned_attempt,
                "terminal_counter_consumption_sha256",
                _digest(f"counter-{planned_attempt.planned_attempt_sha256}"),
            )
            if counter_consumption_sha256 is None
            else counter_consumption_sha256
        ),
        permutation_sha256=(
            getattr(
                planned_attempt,
                "terminal_permutation_sha256",
                _digest(f"permutation-{planned_attempt.planned_attempt_sha256}"),
            )
            if permutation_sha256 is None
            else permutation_sha256
        ),
        recognition_update_count=(
            planned_attempt.terminal_recognition_update_count
        ),
        model_update_count=planned_attempt.terminal_model_update_count,
        validation_boundary_count=(
            planned_attempt.terminal_validation_boundary_count
        ),
        checkpoint_boundary_count=(
            planned_attempt.terminal_checkpoint_boundary_count
        ),
    )
    objective = H6ObjectiveManifestV3.create(
        attempt_spec_sha256=planned_attempt.attempt_spec.attempt_spec_sha256,
        endpoint_config_sha256=planned_attempt.endpoint_config_sha256,
        objective_kind=planned_attempt.attempt_spec.objective_kind,
        phase=terminal_phase,
        recognition_estimator_sha256=(
            planned_attempt.attempt_spec.recognition_estimator_sha256
        ),
        counter_consumption_sha256=cursor.counter_consumption_sha256,
        recognition_law_sha256=(
            _digest(f"law-{planned_attempt.planned_attempt_sha256}")
            if latent
            else None
        ),
        detached_snapshot_sha256=(
            _digest(f"snapshot-{planned_attempt.planned_attempt_sha256}")
            if latent
            else None
        ),
        ordered_factor_bindings=(
            (
                "emission",
                0,
                _digest(f"factor-{planned_attempt.planned_attempt_sha256}"),
            ),
        ),
        total_raw_bytes_sha256=_digest(
            f"objective-{planned_attempt.planned_attempt_sha256}"
        ),
    )
    return capture_h6_checkpoint_v3(
        attempt_spec=planned_attempt.attempt_spec,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        named_modules=tuple(modules),
        named_optimizers=tuple(optimizers),
    )


def _authorized_validation(plan: H6ExperimentPlanV3) -> SimpleNamespace:
    return SimpleNamespace(
        windows=build_causal_windows((1, 2, 0), split="validation"),
        vocabulary=ByteTokenizerV1().vocabulary_identity,
        readiness_sha256=plan.readiness_sha256,
        experiment_config_sha256=plan.experiment_config_sha256,
        plan_sha256=plan.plan_sha256,
        matching_set_sha256=plan.matching_set_sha256,
        data_identity_sha256=(
            plan.tuning_attempts[0].attempt_spec.data_identity_sha256
        ),
        runtime_identity_sha256=(
            plan.training_schedule.runtime_identity_sha256
        ),
    )


def _validation_record(
    *,
    plan: H6ExperimentPlanV3,
    attempt: H6PlannedAttemptV3,
    mean_prior_nll: float,
) -> H6ValidationRecordV3:
    assert attempt.tuning_cell is not None
    return prediction_artifacts._create_h6_validation_record_v3(
        experiment_config_sha256=plan.experiment_config_sha256,
        plan_sha256=plan.plan_sha256,
        endpoint_config_id=attempt.endpoint_config_id,
        endpoint_config_sha256=attempt.endpoint_config_sha256,
        tuning_cell=attempt.tuning_cell,
        training_seed=attempt.training_seed,
        attempt_spec_sha256=attempt.attempt_spec.attempt_spec_sha256,
        checkpoint_sha256=_digest(
            f"checkpoint-{attempt.planned_attempt_sha256}"
        ),
        checkpoint_bytes_sha256=_digest(
            f"checkpoint-bytes-{attempt.planned_attempt_sha256}"
        ),
        readiness_sha256=plan.readiness_sha256,
        matching_set_sha256=plan.matching_set_sha256,
        data_identity_sha256=attempt.attempt_spec.data_identity_sha256,
        runtime_identity_sha256=(
            plan.training_schedule.runtime_identity_sha256
        ),
        counted_target_total=10,
        total_prior_nll=10.0 * mean_prior_nll,
    )


@cache
def _tuning_selection() -> H6TuningSelectionV3:
    plan, _, _ = _authorities()
    endpoint_indices = {
        endpoint_id: index
        for index, endpoint_id in enumerate(H6_TUNED_ENDPOINT_CONFIG_IDS_V3)
    }
    cell_indices = {
        H6TuningCellV3.create(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ).cell_sha256: index
        for index, (learning_rate, weight_decay) in enumerate(H6_TUNING_CELLS_V3)
    }
    records = tuple(
        _validation_record(
            plan=plan,
            attempt=attempt,
            mean_prior_nll=(
                1.0
                if endpoint_indices[attempt.endpoint_config_id] == 0
                else float(
                    cell_indices[attempt.tuning_cell.cell_sha256]
                    != endpoint_indices[attempt.endpoint_config_id]
                )
            ),
        )
        for attempt in plan.tuning_attempts
        if attempt.tuning_cell is not None
    )
    return select_h6_tuning_v3(records, plan)


def _rebuild_tuning_selection(
    selection: H6TuningSelectionV3,
    *,
    records: tuple[H6ValidationRecordV3, ...] | None = None,
    endpoints: tuple[H6EndpointTuningSelectionV3, ...] | None = None,
) -> H6TuningSelectionV3:
    owned_records = (
        selection.tuning_validation_records if records is None else records
    )
    owned_endpoints = (
        selection.endpoint_selections if endpoints is None else endpoints
    )
    payload = {
        "selection_schema": selection.selection_schema,
        "experiment_config_sha256": selection.experiment_config_sha256,
        "plan_sha256": selection.plan_sha256,
        "readiness_sha256": selection.readiness_sha256,
        "matching_set_sha256": selection.matching_set_sha256,
        "data_identity_sha256": selection.data_identity_sha256,
        "runtime_identity_sha256": selection.runtime_identity_sha256,
        "tuning_validation_records": tuple(
            record.canonical_payload()
            | {"validation_record_sha256": record.validation_record_sha256}
            for record in owned_records
        ),
        "endpoint_selections": tuple(
            endpoint.canonical_payload()
            | {"endpoint_selection_sha256": endpoint.endpoint_selection_sha256}
            for endpoint in owned_endpoints
        ),
    }
    return H6TuningSelectionV3(
        selection_schema=selection.selection_schema,
        experiment_config_sha256=selection.experiment_config_sha256,
        plan_sha256=selection.plan_sha256,
        readiness_sha256=selection.readiness_sha256,
        matching_set_sha256=selection.matching_set_sha256,
        data_identity_sha256=selection.data_identity_sha256,
        runtime_identity_sha256=selection.runtime_identity_sha256,
        tuning_validation_records=owned_records,
        endpoint_selections=owned_endpoints,
        tuning_selection_sha256=prediction_artifacts._hash(
            "vfe4.h6.tuning-selection.v3",
            payload,
        ),
    )


def _rebuild_checkpoint_candidate(
    candidate: H6CheckpointCandidateV3,
    *,
    checkpoint_sha256: str | None = None,
    tuning_cell: H6TuningCellV3 | None = None,
) -> H6CheckpointCandidateV3:
    owned_checkpoint_sha256 = (
        candidate.checkpoint_sha256
        if checkpoint_sha256 is None
        else checkpoint_sha256
    )
    owned_cell = candidate.tuning_cell if tuning_cell is None else tuning_cell
    values = {
        "endpoint_config_id": candidate.endpoint_config_id,
        "endpoint_config_sha256": candidate.endpoint_config_sha256,
        "tuning_cell": owned_cell,
        "training_seed": candidate.training_seed,
        "planned_attempt_sha256": candidate.planned_attempt_sha256,
        "attempt_spec_sha256": candidate.attempt_spec_sha256,
        "checkpoint_sha256": owned_checkpoint_sha256,
        "checkpoint_bytes_sha256": candidate.checkpoint_bytes_sha256,
        "experiment_config_sha256": candidate.experiment_config_sha256,
        "plan_sha256": candidate.plan_sha256,
        "readiness_sha256": candidate.readiness_sha256,
        "matching_set_sha256": candidate.matching_set_sha256,
        "data_identity_sha256": candidate.data_identity_sha256,
        "runtime_identity_sha256": candidate.runtime_identity_sha256,
    }
    payload = {
        **values,
        "tuning_cell": {
            "learning_rate": owned_cell.learning_rate,
            "weight_decay": owned_cell.weight_decay,
            "cell_sha256": owned_cell.cell_sha256,
        },
    }
    return H6CheckpointCandidateV3(
        **values,
        candidate_sha256=prediction_artifacts._hash(
            "vfe4.h6.checkpoint-candidate.v3",
            payload,
        ),
    )


def _rebuild_checkpoint_selection(
    selection: H6CheckpointSelectionV3,
    checkpoints: tuple[H6CheckpointCandidateV3, ...],
) -> H6CheckpointSelectionV3:
    payload = {
        "selection_schema": selection.selection_schema,
        "experiment_config_sha256": selection.experiment_config_sha256,
        "plan_sha256": selection.plan_sha256,
        "tuning_selection_sha256": selection.tuning_selection_sha256,
        "readiness_sha256": selection.readiness_sha256,
        "matching_set_sha256": selection.matching_set_sha256,
        "data_identity_sha256": selection.data_identity_sha256,
        "runtime_identity_sha256": selection.runtime_identity_sha256,
        "checkpoints": tuple(
            candidate.canonical_payload()
            | {"candidate_sha256": candidate.candidate_sha256}
            for candidate in checkpoints
        ),
    }
    return H6CheckpointSelectionV3(
        selection_schema=selection.selection_schema,
        experiment_config_sha256=selection.experiment_config_sha256,
        plan_sha256=selection.plan_sha256,
        tuning_selection_sha256=selection.tuning_selection_sha256,
        readiness_sha256=selection.readiness_sha256,
        matching_set_sha256=selection.matching_set_sha256,
        data_identity_sha256=selection.data_identity_sha256,
        runtime_identity_sha256=selection.runtime_identity_sha256,
        checkpoints=checkpoints,
        checkpoint_selection_sha256=prediction_artifacts._hash(
            "vfe4.h6.checkpoint-selection.v3",
            payload,
        ),
    )


@cache
def _checkpoint_fixture() -> tuple[
    tuple[tuple[H6PlannedAttemptV3, H6CheckpointV3], ...],
    H6CheckpointSelectionV3,
]:
    plan, runtime, _ = _authorities()
    tuning = _tuning_selection()
    selected = {
        item.endpoint_config_id: item.tuning_cell
        for item in tuning.endpoint_selections
    }
    bindings = tuple(
        (
            attempt,
            _terminal_checkpoint(
                attempt,
                runtime=runtime,
                cell=selected[attempt.endpoint_config_id],
            ),
        )
        for attempt in plan.confirmatory_attempts
    )
    return (
        bindings,
        bind_h6_checkpoint_selection_v3(bindings, plan, tuning),
    )


def _archive_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        directory = zipfile.ZipInfo("wikitext-2-raw/")
        directory.external_attr = (0o40755 << 16) | 0x10
        archive.writestr(directory, b"")
        archive.writestr(_MEMBERS[0], b"train bytes" * 10)
        archive.writestr(_MEMBERS[1], bytes(range(256)) * 513)
        archive.writestr(_MEMBERS[2], b"test bytes")
    return stream.getvalue()


def _acquisition_request(
    archive_bytes: bytes,
    artifact_root: Path,
) -> H6DataAcquisitionRequest:
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        expected_members = tuple(
            H6ArchiveMemberExpectation(
                info.filename,  # type: ignore[arg-type]
                info.compress_size,
                info.file_size,
                info.compress_type,  # type: ignore[arg-type]
                info.CRC,
                hashlib.sha256(archive.read(info)).hexdigest(),
            )
            for info in archive.infolist()
            if not info.is_dir()
        )
    return H6DataAcquisitionRequest(
        data=H6DataConfig(
            "h6-data-config-v1",
            WIKITEXT2_RAW_URL,
            16_777_216,
            ("wikitext-2-raw/", *_MEMBERS),
            (0, 8),
            16_777_216,
            33_554_432,
            100,
            H6ObservedArchive(
                len(archive_bytes),
                hashlib.sha256(archive_bytes).hexdigest(),
                expected_members,
            ),
        ),
        artifact_root=artifact_root,
    )


def _scoring_fixture() -> tuple[
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    H6CheckpointV3,
]:
    plan, runtime, _ = _authorities()
    attempt = plan.tuning_attempts[0]
    assert attempt.tuning_cell is not None
    checkpoint = _terminal_checkpoint(
        attempt,
        runtime=runtime,
        cell=attempt.tuning_cell,
        model=build_arm_model(plan.endpoint_configs[0]),
    )
    return plan, attempt, checkpoint


def test_validation_scores_only_authorized_validation_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, attempt, checkpoint = _scoring_fixture()
    capability = object()
    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        lambda received, *, plan: (
            _authorized_validation(plan)
            if received is capability
            else (_ for _ in ()).throw(OpeningCapabilityError("forged"))
        ),
    )
    record = validation_v3.score_h6_validation_checkpoint_v3(
        capability=capability,
        checkpoint=checkpoint,
        planned_attempt=attempt,
        plan=plan,
    )
    assert record.counted_target_total == 2
    assert record.scoring_method == "target-blind-prefix-prior-nll"
    with pytest.raises(OpeningCapabilityError, match="forged"):
        validation_v3.score_h6_validation_checkpoint_v3(
            capability=object(),
            checkpoint=checkpoint,
            planned_attempt=attempt,
            plan=plan,
        )


def test_tuning_validation_rejects_checkpoint_from_another_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, runtime, _ = _authorities()
    attempt = plan.tuning_attempts[0]
    assert attempt.tuning_cell is not None
    wrong_cell = next(
        cell for cell in plan.tuning_cells if cell != attempt.tuning_cell
    )
    checkpoint = _terminal_checkpoint(
        attempt,
        runtime=runtime,
        cell=wrong_cell,
        model=build_arm_model(plan.endpoint_configs[0]),
    )
    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        lambda _capability, *, plan: _authorized_validation(plan),
    )

    with pytest.raises(ValueError, match="tuning.*cell|cell.*tuning"):
        validation_v3.score_h6_validation_checkpoint_v3(
            capability=object(),
            checkpoint=checkpoint,
            planned_attempt=attempt,
            plan=plan,
        )


def test_validation_uses_fresh_cpu_model_from_checkpoint_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, runtime, _ = _authorities()
    attempt = plan.tuning_attempts[0]
    assert attempt.tuning_cell is not None
    live_model = build_arm_model(plan.endpoint_configs[0])
    windows = build_causal_windows((1, 2, 0), split="validation")
    vocabulary = ByteTokenizerV1().vocabulary_identity
    expected_total = 0.0
    for target_index, target in enumerate((2, 0)):
        prefix = windows.causal_prefix(
            window_index=0,
            receiver_t=target_index + 1,
            vocabulary=vocabulary,
        )
        expected_total -= float(live_model.prefix_log_probs(prefix)[target].item())
    checkpoint = _terminal_checkpoint(
        attempt,
        runtime=runtime,
        cell=attempt.tuning_cell,
        model=live_model,
    )
    with torch.no_grad():
        for parameter in live_model.parameters():
            parameter.add_(100.0)
    built: list[nn.Module] = []
    canonical_builder = build_arm_model

    def observed_builder(config: ArmConfig) -> nn.Module:
        model = canonical_builder(config)
        built.append(model)
        return model

    monkeypatch.setattr(validation_v3, "build_arm_model", observed_builder)
    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        lambda _capability, *, plan: _authorized_validation(plan),
    )
    record = validation_v3.score_h6_validation_checkpoint_v3(
        capability=object(),
        checkpoint=checkpoint,
        planned_attempt=attempt,
        plan=plan,
    )
    assert len(built) == 1
    assert built[0] is not live_model
    assert all(parameter.device.type == "cpu" for parameter in built[0].parameters())
    assert all(parameter.dtype is torch.float64 for parameter in built[0].parameters())
    assert record.total_prior_nll == pytest.approx(expected_total)


def test_tuning_selection_applies_frozen_mean_and_tie_break() -> None:
    plan, _, _ = _authorities()
    selection = _tuning_selection()
    assert len(selection.tuning_validation_records) == 72
    selected = {
        item.endpoint_config_id: item for item in selection.endpoint_selections
    }
    assert set(selected) == set(H6_MATCHING_V3_ENDPOINT_CONFIG_IDS)
    first = selected[H6_TUNED_ENDPOINT_CONFIG_IDS_V3[0]]
    assert (first.tuning_cell.learning_rate, first.tuning_cell.weight_decay) == (
        1.0e-4,
        0.0,
    )
    primary = H6_TUNED_ENDPOINT_CONFIG_IDS_V3[-1]
    for endpoint_id in (
        set(H6_MATCHING_V3_ENDPOINT_CONFIG_IDS)
        - set(H6_TUNED_ENDPOINT_CONFIG_IDS_V3)
    ):
        assert selected[endpoint_id].source_endpoint_config_id == primary
        assert selected[endpoint_id].tuning_cell == selected[primary].tuning_cell
    assert tuple(
        item.endpoint_config_sha256 for item in selection.endpoint_selections
    ) == tuple(config.config_sha256 for config in plan.endpoint_configs)
    duplicate_records = (
        selection.tuning_validation_records[1],
        *selection.tuning_validation_records[1:],
    )
    with pytest.raises(ValueError, match="inventory|duplicate"):
        _rebuild_tuning_selection(
            selection,
            records=duplicate_records,
        )
    wrong_cell = plan.tuning_cells[-1]
    first = selection.endpoint_selections[0]
    wrong_endpoint = H6EndpointTuningSelectionV3.create(
        endpoint_config_id=first.endpoint_config_id,
        endpoint_config_sha256=first.endpoint_config_sha256,
        source_endpoint_config_id=first.source_endpoint_config_id,
        tuning_cell=wrong_cell,
        source_validation_record_sha256s=(
            first.source_validation_record_sha256s
        ),
    )
    with pytest.raises(ValueError, match="winner|selected|cell"):
        _rebuild_tuning_selection(
            selection,
            endpoints=(wrong_endpoint, *selection.endpoint_selections[1:]),
        )


def test_checkpoint_selection_binds_complete_inventory() -> None:
    plan, _, _ = _authorities()
    assert plan.confirmatory_attempts[0].terminal_pass_index == 2
    assert plan.confirmatory_attempts[0].terminal_batch_index == 0
    assert plan.confirmatory_attempts[0].terminal_model_update_count > 1
    bindings, checkpoint_set = _checkpoint_fixture()
    assert len(checkpoint_set.checkpoints) == 12 * 8 == 96
    assert {
        (item.endpoint_config_id, item.training_seed)
        for item in checkpoint_set.checkpoints
    } == {
        (endpoint_id, seed)
        for endpoint_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        for seed in H6_CONFIRMATORY_SEEDS_V3
    }
    with pytest.raises(ValueError, match="96|inventory"):
        bind_h6_checkpoint_selection_v3(
            bindings[:-1],
            plan,
            _tuning_selection(),
        )


def test_validation_cannot_consume_test_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, attempt, checkpoint = _scoring_fixture()
    calls: list[object] = []

    def reject(opening: object, *, plan: object) -> object:
        calls.append((opening, plan))
        raise OpeningCapabilityError(
            "exact access-issued v3 validation capability is required"
        )

    monkeypatch.setattr(
        validation_v3,
        "_consume_h6_validation_capability_v3",
        reject,
    )
    test_opening = SimpleNamespace(
        proof_identity_sha256=_digest("test-opening")
    )
    with pytest.raises(OpeningCapabilityError, match="validation capability"):
        validation_v3.score_h6_validation_checkpoint_v3(
            capability=test_opening,
            checkpoint=checkpoint,
            planned_attempt=attempt,
            plan=plan,
        )
    assert calls == [(test_opening, plan)]


def test_validation_api_has_no_injectable_model_or_prior_callable() -> None:
    parameters = inspect.signature(
        validation_v3.score_h6_validation_checkpoint_v3
    ).parameters
    assert tuple(parameters) == (
        "capability",
        "checkpoint",
        "planned_attempt",
        "plan",
    )
    assert "model_factory" not in parameters
    assert "prior_log_probs" not in parameters
    assert "tuning_cell" not in parameters
    assert "recognition" not in parameters
    assert "target" not in parameters
    assert H6ValidationRecordV3.__dataclass_params__.init is False
    assert "create" not in H6ValidationRecordV3.__dict__


def test_validation_capability_requires_store_readiness_and_plan_authority(
    tmp_path: Path,
) -> None:
    assert tuple(
        inspect.signature(issue_h6_validation_capability_v3).parameters
    ) == ("store", "readiness", "plan")
    archive_bytes = _archive_bytes()
    artifact_root = tmp_path / "store"
    synthetic_store = _acquire_wikitext2_blinded(
        _acquisition_request(archive_bytes, artifact_root),
        lambda _: io.BytesIO(archive_bytes),
    )
    synthetic_identity = synthetic_store.data_identity
    synthetic_plan, _, synthetic_readiness = _plan_for_data(
        data_identity_sha256=synthetic_store.data_identity_sha256,
        access_policy_sha256=synthetic_identity.access_policy_sha256,
        train_token_count=synthetic_identity.train_tokens.token_count,
        train_token_sha256=(
            synthetic_identity.train_tokens.encoded_token_sha256
        ),
    )
    with pytest.raises(OpeningCapabilityError, match="reopened|provenance"):
        issue_h6_validation_capability_v3(
            synthetic_store,
            synthetic_readiness,
            synthetic_plan,
        )
    store = reopen_authenticated_blinded_store_v3(
        artifact_root / AUTHENTICATED_BLINDED_STORE_MANIFEST_V3_FILENAME,
        artifact_root,
    )
    identity = store.data_identity
    plan, _, readiness = _plan_for_data(
        data_identity_sha256=store.data_identity_sha256,
        access_policy_sha256=identity.access_policy_sha256,
        train_token_count=identity.train_tokens.token_count,
        train_token_sha256=identity.train_tokens.encoded_token_sha256,
    )
    capability = issue_h6_validation_capability_v3(store, readiness, plan)
    assert repr(capability) == "<opaque H6 v3 validation capability>"
    with pytest.raises((TypeError, OpeningCapabilityError)):
        issue_h6_validation_capability_v3(  # type: ignore[arg-type]
            build_causal_windows((1, 2), split="test"),
            readiness,
            plan,
        )


def test_checkpoint_candidates_require_actual_planned_checkpoint() -> None:
    parameters = inspect.signature(H6CheckpointCandidateV3.create).parameters
    assert tuple(parameters) == (
        "checkpoint",
        "planned_attempt",
        "plan",
        "tuning_selection",
    )
    assert "checkpoint_sha256" not in parameters
    assert "attempt_spec_sha256" not in parameters
    _, selection = _checkpoint_fixture()
    duplicate = _rebuild_checkpoint_candidate(
        selection.checkpoints[0],
        checkpoint_sha256=selection.checkpoints[1].checkpoint_sha256,
    )
    with pytest.raises(ValueError, match="unique|duplicate"):
        _rebuild_checkpoint_selection(
            selection,
            (duplicate, *selection.checkpoints[1:]),
        )


def test_terminal_checkpoint_requires_exact_counter_and_permutation_binding() -> None:
    plan, runtime, _ = _authorities()
    tuning = _tuning_selection()
    attempt = next(
        item
        for item in plan.confirmatory_attempts
        if item.attempt_spec.recognition_factory_sha256 is not None
    )
    selected_cell = next(
        item.tuning_cell
        for item in tuning.endpoint_selections
        if item.endpoint_config_id == attempt.endpoint_config_id
    )
    forged = _terminal_checkpoint(
        attempt,
        runtime=runtime,
        cell=selected_cell,
        draw_block=999,
        counter_consumption_sha256=_digest("forged-terminal-counter"),
        permutation_sha256=_digest("forged-terminal-permutation"),
    )

    with pytest.raises(
        ValueError,
        match="terminal.*(counter|permutation|boundary)|"
        "(counter|permutation).*terminal",
    ):
        H6CheckpointCandidateV3.create(
            checkpoint=forged,
            planned_attempt=attempt,
            plan=plan,
            tuning_selection=tuning,
        )


def test_validation_reader_uses_single_no_follow_handle_per_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _ = _authorities()
    bundle = H6ValidationBundleV3.create(
        plan=plan,
        tuning_selection=_tuning_selection(),
        checkpoint_selection=_checkpoint_fixture()[1],
    )
    published = publish_h6_validation_bundle_v3(
        run_root=tmp_path / "validation-runs",
        run_name="single-handle",
        bundle=bundle,
    )
    protected_paths = {
        published / "validation_bundle.json",
        published / "manifest.sha256",
    }
    original_read_bytes = Path.read_bytes

    def reject_path_reopen(path: Path) -> bytes:
        if path in protected_paths:
            raise AssertionError("validation payload was reopened by path")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_path_reopen)
    assert read_h6_validation_bundle_v3(
        published.resolve(strict=True),
        expected_plan_sha256=plan.plan_sha256,
        expected_experiment_config_sha256=plan.experiment_config_sha256,
        expected_validation_bundle_sha256=bundle.validation_bundle_sha256,
    ) == bundle


def test_validation_bundle_has_no_replace_publisher_and_authenticated_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _, _ = _authorities()
    _, checkpoint_selection = _checkpoint_fixture()
    bundle = H6ValidationBundleV3.create(
        plan=plan,
        tuning_selection=_tuning_selection(),
        checkpoint_selection=checkpoint_selection,
    )
    run_root = tmp_path / "validation-runs"
    published = publish_h6_validation_bundle_v3(
        run_root=run_root,
        run_name="h6-validation-v3",
        bundle=bundle,
    )
    assert read_h6_validation_bundle_v3(
        published.resolve(strict=True),
        expected_plan_sha256=plan.plan_sha256,
        expected_experiment_config_sha256=plan.experiment_config_sha256,
        expected_validation_bundle_sha256=bundle.validation_bundle_sha256,
    ) == bundle
    with pytest.raises(ArtifactPublicationError, match="already exists"):
        publish_h6_validation_bundle_v3(
            run_root=run_root,
            run_name="h6-validation-v3",
            bundle=bundle,
        )
    bundle_path = published / "validation_bundle.json"
    manifest_path = published / "manifest.sha256"
    original_raw = bundle_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    def write_payload(payload: object) -> None:
        raw = artifact_json_bytes(payload)
        bundle_path.write_bytes(raw)
        manifest_path.write_bytes(
            (
                f"{hashlib.sha256(raw).hexdigest()}  "
                "validation_bundle.json\n"
            ).encode("ascii")
        )

    changed = json.loads(original_raw)
    changed["tuning_selection"]["tuning_validation_records"][0][
        "scoring_method"
    ] = "target-aware"
    write_payload(changed)
    with pytest.raises(ArtifactPublicationError, match="record|identity|field"):
        read_h6_validation_bundle_v3(
            published.resolve(strict=True),
            expected_plan_sha256=plan.plan_sha256,
            expected_experiment_config_sha256=plan.experiment_config_sha256,
            expected_validation_bundle_sha256=(
                bundle.validation_bundle_sha256
            ),
        )

    changed = json.loads(original_raw)
    changed["unexpected"] = True
    write_payload(changed)
    with pytest.raises(ArtifactPublicationError, match="field|object|key"):
        read_h6_validation_bundle_v3(
            published.resolve(strict=True),
            expected_plan_sha256=plan.plan_sha256,
            expected_experiment_config_sha256=plan.experiment_config_sha256,
            expected_validation_bundle_sha256=(
                bundle.validation_bundle_sha256
            ),
        )

    bundle_path.write_bytes(original_raw)
    manifest_path.write_bytes(original_manifest)
    with bundle_path.open("wb") as handle:
        handle.truncate(prediction_artifacts._MAXIMUM_BUNDLE_BYTES + 1)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == bundle_path:
            raise AssertionError("oversized payload was read before size refusal")
        return original_read_bytes(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", guarded_read_bytes)
        with pytest.raises(ArtifactPublicationError, match="size|bound"):
            read_h6_validation_bundle_v3(
                published.resolve(strict=True),
                expected_plan_sha256=plan.plan_sha256,
                expected_experiment_config_sha256=(
                    plan.experiment_config_sha256
                ),
                expected_validation_bundle_sha256=(
                    bundle.validation_bundle_sha256
                ),
            )
