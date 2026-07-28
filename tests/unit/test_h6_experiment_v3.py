from __future__ import annotations

import dataclasses
from functools import cache
from typing import TypeVar

import pytest

from vfe4.training.h6_engine_v3 import H6EngineAuthorityV3
from vfe4.training.h6_experiment_v3 import (
    H6_CONFIRMATORY_SEEDS_V3,
    H6_TUNING_CELLS_V3,
    H6_TUNING_SEEDS_V3,
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
    H6_COUNTER_MAPPING_SHA256,
    H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
)


_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64
_CONFIG_SHA256 = "3" * 64
_DATA_SHA256 = "4" * 64
_RecordT = TypeVar("_RecordT")


def _sha(character: str) -> str:
    return character * 64


def _unsafe_replace(record: _RecordT, **changes: object) -> _RecordT:
    forged = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(
            forged,
            field.name,
            changes.get(field.name, getattr(record, field.name)),
        )
    return forged


def _template(config_id: str) -> ArmConfig:
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
        vocabulary=VocabularyIdentity(
            vocabulary_id="h6-task8-synthetic",
            size=258,
            tokenizer_spec_sha256=_sha("a"),
        ),
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


@cache
def _authorities() -> tuple[
    H6MatchingSetV3,
    H6TrainingScheduleV3,
    H6PredictionV3ReadinessToken,
    H6PredictionRuntimeIdentity,
]:
    templates = tuple(
        _template(config_id) for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
    )
    matching = H6MatchingSetV3.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        workload=H6TrainingWorkloadV3.from_train_tokens(
            train_token_count=258,
            train_token_sha256=_sha("b"),
        ),
        endpoint_templates=templates,
    )
    runtime = H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version="2.8.0+cu128",
        cuda_runtime_version="12.8",
        cuda_device_name="synthetic",
        cuda_compute_capability=(12, 0),
    )
    estimator = H6RecognitionEstimatorSpec.create()
    phases = tuple(
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
    )
    schedule = H6TrainingScheduleV3.create(
        outer=H6OuterSchedule.create(
            optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256
        ),
        endpoint_phases=phases,
        estimator=estimator,
        runtime=runtime,
    )
    readiness = H6PredictionV3ReadinessToken.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        experiment_config_sha256=_CONFIG_SHA256,
        correctness_manifests=(
            ("H1", _sha("1")),
            ("H2", _sha("2")),
            ("H3", _sha("3")),
            ("H5", _sha("5")),
        ),
        h1_prefix_prior_manifest_sha256=_sha("6"),
        h1_prefix_prior_generative_factor_schema_sha256=_sha("7"),
        smc_bias_semantics_sha256=_sha("8"),
        smc_validation_manifest_sha256=_sha("9"),
        prefix_certificate_set_sha256=_sha("a"),
        h5_update_binding_sha256=_sha("b"),
        critical_values_sha256=_sha("c"),
        endpoint_smc_protocol_sha256=_sha("d"),
        attribution_matrix_sha256=_sha("e"),
        objective_gate_spec_sha256=_sha("f"),
        matching_policy_sha256=H6_MATCHING_POLICY_V3.policy_sha256,
        matching_set_sha256=matching.matching_set_sha256,
        training_schedule_sha256=schedule.schedule_sha256,
        recognition_estimator_sha256=estimator.estimator_sha256,
        runtime_identity_sha256=runtime.runtime_identity_sha256,
        counter_mapping_sha256=H6_COUNTER_MAPPING_SHA256,
        phase_ownership_sha256=H6_PHASE_OWNERSHIP_SHA256,
        objective_manifest_schema_sha256=H6_OBJECTIVE_MANIFEST_SCHEMA_SHA256,
        data_identity_sha256=_DATA_SHA256,
        access_policy_sha256=_sha("0"),
    )
    return matching, schedule, readiness, runtime


def test_plan_v3_emits_exact_endpoint_attempt_and_schedule_inventory() -> None:
    matching, schedule, readiness, runtime = _authorities()

    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching,
        training_schedule=schedule,
        runtime_identity=runtime,
    )

    assert plan.endpoint_config_ids == H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
    assert plan.matching_report_sha256s == tuple(
        item.record_sha256 for item in matching.matrix_reports
    )
    assert tuple(
        (cell.learning_rate, cell.weight_decay) for cell in plan.tuning_cells
    ) == (
        (0.0001, 0.0),
        (0.0001, 0.01),
        (0.0003, 0.0),
        (0.0003, 0.01),
        (0.001, 0.0),
        (0.001, 0.01),
    )
    assert H6_TUNING_CELLS_V3 == tuple(
        (cell.learning_rate, cell.weight_decay) for cell in plan.tuning_cells
    )
    assert plan.tuning_seeds == H6_TUNING_SEEDS_V3 == (2026072199, 2026072200)
    assert (
        plan.confirmatory_seeds
        == H6_CONFIRMATORY_SEEDS_V3
        == tuple(range(2026072101, 2026072109))
    )
    assert len(plan.tuning_attempts) == 6 * 6 * 2
    assert len(plan.confirmatory_attempts) == 12 * 8
    assert len(plan.attempts) == 168
    assert {attempt.endpoint_config_id for attempt in plan.tuning_attempts} == {
        *H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[:5],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[7],
    }
    assert {
        (attempt.endpoint_config_id, attempt.tuning_cell, attempt.training_seed)
        for attempt in plan.tuning_attempts
    } == {
        (endpoint, cell, seed)
        for endpoint in (
            *H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[:5],
            H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[7],
        )
        for cell in plan.tuning_cells
        for seed in H6_TUNING_SEEDS_V3
    }
    assert {
        (attempt.endpoint_config_id, attempt.training_seed)
        for attempt in plan.confirmatory_attempts
    } == {
        (endpoint, seed)
        for endpoint in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
        for seed in H6_CONFIRMATORY_SEEDS_V3
    }
    assert all(
        attempt.attempt_spec.readiness_sha256 == readiness.readiness_sha256
        and attempt.attempt_spec.training_schedule_sha256 == schedule.schedule_sha256
        and attempt.attempt_spec.runtime_identity_sha256
        == runtime.runtime_identity_sha256
        and attempt.matching_set_sha256 == matching.matching_set_sha256
        and attempt.matching_policy_sha256 == matching.matching_policy_sha256
        for attempt in plan.attempts
    )
    assert tuple(
        item.endpoint_config_sha256 for item in plan.training_schedule.endpoint_phases
    ) == tuple(config.config_sha256 for config in matching.endpoint_configs)
    assert all(attempt.matching_report_sha256s for attempt in plan.attempts)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.tuning_attempts = ()  # type: ignore[misc]


def test_plan_v3_has_no_corpus_or_outcome_input_and_refuses_identity_drift() -> None:
    matching, schedule, readiness, runtime = _authorities()

    with pytest.raises(TypeError):
        plan_h6_experiment_v3(  # type: ignore[call-arg]
            readiness=readiness,
            matching_set=matching,
            training_schedule=schedule,
            runtime_identity=runtime,
            corpus=b"forbidden",
        )
    with pytest.raises(ValueError, match="matching.*identity|identity.*matching"):
        plan_h6_experiment_v3(
            readiness=_unsafe_replace(
                readiness,
                matching_set_sha256=_sha("f"),
                readiness_sha256=readiness.readiness_sha256,
            ),
            matching_set=matching,
            training_schedule=schedule,
            runtime_identity=runtime,
        )
    with pytest.raises(ValueError, match="schedule.*identity|identity.*schedule"):
        plan_h6_experiment_v3(
            readiness=readiness,
            matching_set=matching,
            training_schedule=_unsafe_replace(
                schedule,
                schedule_sha256=_sha("f"),
            ),
            runtime_identity=runtime,
        )


def test_planned_attempt_binds_matching_evidence_and_exact_adamw_cell() -> None:
    matching, schedule, readiness, runtime = _authorities()
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching,
        training_schedule=schedule,
        runtime_identity=runtime,
    )
    attempt = plan.tuning_attempts[0]
    expected_reports = tuple(record.record_sha256 for record in matching.matrix_reports)

    assert attempt.matching_report_sha256s == expected_reports
    assert attempt.matching_set_sha256 == matching.matching_set_sha256
    assert attempt.matching_policy_sha256 == matching.matching_policy_sha256
    assert attempt.tuning_cell is not None
    authority = H6EngineAuthorityV3.from_planned_attempt(
        planned_attempt=attempt,
    )

    assert authority.planned_attempt_sha256 == attempt.planned_attempt_sha256
    assert authority.matching_set_sha256 == attempt.matching_set_sha256
    assert authority.matching_policy_sha256 == attempt.matching_policy_sha256
    assert authority.endpoint_config_id == attempt.endpoint_config_id
    assert authority.matching_ledger_sha256 == attempt.matching_ledger_sha256
    assert authority.matching_report_sha256s == expected_reports
    assert authority.tuning_cell_sha256 == attempt.tuning_cell.cell_sha256
    assert authority.optimizer_learning_rate == attempt.tuning_cell.learning_rate
    assert authority.optimizer_weight_decay == attempt.tuning_cell.weight_decay

    with pytest.raises(ValueError, match="selected tuning cell"):
        H6EngineAuthorityV3.from_planned_attempt(
            planned_attempt=plan.confirmatory_attempts[0],
        )

    for field in ("matching_set_sha256", "matching_policy_sha256"):
        with pytest.raises(ValueError, match="planned-attempt identity"):
            H6EngineAuthorityV3.from_planned_attempt(
                planned_attempt=_unsafe_replace(
                    attempt,
                    **{field: _sha("f")},
                )
            )
