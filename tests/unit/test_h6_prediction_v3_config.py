from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vfe4.config import (
    H6PredictionV2ResolvedConfig,
    H6PredictionV3ResolvedConfig,
    resolve_config,
    resolve_h6_prediction_config,
    resolve_h6_prediction_v3_config,
)
from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS
from vfe4.types import (
    H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256,
    H6AttemptCursorV3,
    H6AttemptSpecV3,
    H6PredictionRuntimeIdentity,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
    H6_CHECKPOINT_CODEC_SHA256,
    H6_COUNTER_MAPPING_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6_SCORING_INVENTORY_SHA256,
    ObjectiveGateSpec,
    TrainingPhase,
)


def _sha(character: str) -> str:
    return character * 64


def _source() -> dict[str, object]:
    return {
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "source_sha256": _sha("3"),
    }


def _objective_gate() -> dict[str, object]:
    gate = ObjectiveGateSpec.create()
    return {
        "schema_version": gate.schema_version,
        "complete_arm_id": gate.complete_arm_id,
        "emission_arm_id": gate.emission_arm_id,
        "orientation": gate.orientation,
        "delta_obj": gate.delta_obj,
        "opening_policy": gate.opening_policy,
        "evaluation_order": gate.evaluation_order,
        "spec_sha256": gate.spec_sha256,
    }


def _prediction_v2_config() -> dict[str, object]:
    return {
        "schema_version": "h6-prediction-config-v2",
        "operation": "H6-Prediction",
        "source": _source(),
        "data": {
            "schema_version": "h6-data-config-v1",
            "source_url": (
                "https://s3.amazonaws.com/research.metamind.io/"
                "wikitext/wikitext-2-raw-v1.zip"
            ),
            "max_archive_bytes": 16_777_216,
            "member_paths": [
                "wikitext-2-raw/",
                "wikitext-2-raw/wiki.train.raw",
                "wikitext-2-raw/wiki.valid.raw",
                "wikitext-2-raw/wiki.test.raw",
            ],
            "allowed_compression_methods": [0, 8],
            "max_member_bytes": 16_777_216,
            "max_total_uncompressed_bytes": 33_554_432,
            "max_compression_ratio": 100,
            "observed_archive": None,
        },
        "prerequisites": {
            "correctness_manifests": {
                "H1": _sha("1"),
                "H2": _sha("2"),
                "H3": _sha("3"),
                "H5": _sha("5"),
            },
            "h1_prefix_prior_manifest_sha256": _sha("7"),
            "h1_prefix_prior_generative_factor_schema_sha256": (
                H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
            ),
            "smc_validation_manifest_sha256": _sha("8"),
            "prefix_certificate_set_sha256": _sha("9"),
        },
        "h5_update_binding_sha256": _sha("a"),
        "training_schedule": {
            "schedule_schema": "h6-training-schedule-v2",
            "outer": {
                "schedule_schema": "h6-outer-schedule-v1",
                "optimizer_class": "AdamW",
                "optimizer_policy_sha256": _sha("b"),
                "model_updates_per_batch": 1,
                "validation_twentieths_per_pass": 20,
                "full_passes": 2,
            },
            "endpoint_phases": [
                {
                    "endpoint_config_sha256": _sha("c"),
                    "latent_enabled": False,
                    "phases": [TrainingPhase.MODEL_CE_ADAMW.value],
                    "recognition_updates_per_batch": 0,
                    "model_updates_per_batch": 1,
                    "no_op_phases": 0,
                },
                {
                    "endpoint_config_sha256": _sha("d"),
                    "latent_enabled": True,
                    "phases": [
                        TrainingPhase.RECOGNITION_ADAMW.value,
                        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
                        TrainingPhase.MODEL_ADAMW.value,
                    ],
                    "recognition_updates_per_batch": 1,
                    "model_updates_per_batch": 1,
                    "no_op_phases": 0,
                },
            ],
        },
        "critical_values_sha256": _sha("e"),
        "endpoint_smc_protocol": {
            "protocol_schema": "h6-endpoint-smc-v1",
            "particle_counts": [128, 256, 512, 1024],
            "replicate_count": 64,
            "registry_root_seed": 2026072198,
            "common_stream_domain": "h6-wt2-endpoint-mc-v1",
            "simultaneous_interval_count": 352,
            "familywise_alpha": 0.01,
            "critical_value_df63": 4.5144904535377144,
            "remainder_contraction": 0.75,
        },
        "smc_bias_semantics_sha256": SMC_BIAS_SEMANTICS.semantics_sha256,
        "attribution_matrix_sha256": _sha("f"),
        "matching_set_sha256": _sha("0"),
        "objective_gate": _objective_gate(),
        "data_identity_sha256": _sha("1"),
        "access_policy_sha256": _sha("2"),
        "artifact_root": "runs/h6-prediction",
    }


def _prediction_v3_config() -> dict[str, object]:
    raw = _prediction_v2_config()
    raw["schema_version"] = "h6-prediction-config-v3"
    prerequisites = raw["prerequisites"]
    assert isinstance(prerequisites, dict)
    prerequisites["a0_direct_exact_prefix_certificate_sha256"] = _sha("6")

    estimator = H6RecognitionEstimatorSpec.create()
    estimator_payload = estimator.canonical_payload()
    estimator_payload["estimator_sha256"] = estimator.estimator_sha256

    runtime = H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version="2.8.0+cu128",
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )
    runtime_payload = runtime.canonical_payload()
    runtime_payload["cuda_compute_capability"] = [12, 0]
    runtime_payload["runtime_identity_sha256"] = (
        runtime.runtime_identity_sha256
    )

    schedule = raw["training_schedule"]
    assert isinstance(schedule, dict)
    schedule["schedule_schema"] = "h6-training-schedule-v3"
    schedule["recognition_estimator_sha256"] = estimator.estimator_sha256
    schedule["runtime_identity_sha256"] = runtime.runtime_identity_sha256
    schedule["training_noise_domain"] = (
        "vfe4.h6.training-rmc-normal.v1"
    )
    schedule["counter_mapping_sha256"] = H6_COUNTER_MAPPING_SHA256
    schedule["phase_ownership_sha256"] = H6_PHASE_OWNERSHIP_SHA256
    schedule["checkpoint_codec_sha256"] = H6_CHECKPOINT_CODEC_SHA256

    raw["recognition_contract"] = {
        "trajectory_schema": "h6-language-recognition-trajectory-v3",
        "categorical_posterior_schema": (
            "h6-categorical-source-posterior-v3"
        ),
        "terminal_mixture_schema": "h6-terminal-source-mixture-v1",
        "estimator": estimator_payload,
    }
    raw["runtime"] = runtime_payload
    raw["matching_policy_schema"] = "h6-amended-matching-policy-v3"
    raw["matching_policy_sha256"] = _sha("4")
    raw["matching_set_schema"] = "h6-amended-matching-set-v3"
    raw["counter_mapping_sha256"] = H6_COUNTER_MAPPING_SHA256
    raw["phase_ownership_sha256"] = H6_PHASE_OWNERSHIP_SHA256
    raw["checkpoint_codec_sha256"] = H6_CHECKPOINT_CODEC_SHA256
    raw["scoring_inventory_sha256"] = H6_SCORING_INVENTORY_SHA256
    raw["expected_test_row_count"] = 4104
    return raw


def _reverse_mappings(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _reverse_mappings(item)
            for key, item in reversed(tuple(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def test_v3_config_requires_recognition_estimator_runtime_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    raw = _prediction_v3_config()
    before = copy.deepcopy(raw)

    resolved = resolve_h6_prediction_v3_config(raw, repo_root=tmp_path)

    assert raw == before
    assert type(resolved) is H6PredictionV3ResolvedConfig
    assert resolved.schema_version == "h6-prediction-config-v3"
    assert (
        resolved.recognition_trajectory_schema
        == "h6-language-recognition-trajectory-v3"
    )
    assert (
        resolved.categorical_posterior_schema
        == "h6-categorical-source-posterior-v3"
    )
    assert resolved.terminal_mixture_schema == "h6-terminal-source-mixture-v1"
    assert type(resolved.recognition_estimator) is H6RecognitionEstimatorSpec
    assert type(resolved.runtime) is H6PredictionRuntimeIdentity
    assert type(resolved.training_schedule) is H6TrainingScheduleV3
    assert (
        resolved.training_schedule.recognition_estimator_sha256
        == resolved.recognition_estimator.estimator_sha256
    )
    assert (
        resolved.training_schedule.runtime_identity_sha256
        == resolved.runtime.runtime_identity_sha256
    )
    assert resolved.checkpoint_codec_sha256 == H6_CHECKPOINT_CODEC_SHA256
    assert (
        resolved.a0_direct_exact_prefix_certificate_sha256
        == _sha("6")
    )
    assert (
        resolved.training_schedule.checkpoint_codec_sha256
        == H6_CHECKPOINT_CODEC_SHA256
    )

    canonical = json.loads(resolved.canonical_json)
    assert canonical["recognition_contract"]["estimator"][
        "estimator_sha256"
    ] == resolved.recognition_estimator.estimator_sha256
    assert canonical["runtime"]["runtime_identity_sha256"] == (
        resolved.runtime.runtime_identity_sha256
    )
    assert canonical["training_schedule"]["schedule_schema"] == (
        "h6-training-schedule-v3"
    )

    reordered = resolve_h6_prediction_v3_config(
        _reverse_mappings(raw),  # type: ignore[arg-type]
        repo_root=tmp_path,
    )
    assert reordered.canonical_json == resolved.canonical_json
    assert reordered.config_sha256 == resolved.config_sha256
    reopened = resolve_h6_prediction_v3_config(
        json.loads(resolved.canonical_json),
        repo_root=tmp_path,
    )
    assert reopened.canonical_json == resolved.canonical_json
    assert reopened.config_sha256 == resolved.config_sha256

    missing_recognition = copy.deepcopy(raw)
    missing_recognition.pop("recognition_contract")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_recognition,
            repo_root=tmp_path,
        )

    missing_runtime = copy.deepcopy(raw)
    missing_runtime.pop("runtime")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(missing_runtime, repo_root=tmp_path)

    missing_checkpoint = copy.deepcopy(raw)
    missing_checkpoint.pop("checkpoint_codec_sha256")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_checkpoint,
            repo_root=tmp_path,
        )

    schedule_missing_checkpoint = copy.deepcopy(raw)
    broken_schedule = schedule_missing_checkpoint["training_schedule"]
    assert isinstance(broken_schedule, dict)
    broken_schedule.pop("checkpoint_codec_sha256")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            schedule_missing_checkpoint,
            repo_root=tmp_path,
        )

    missing_direct_certificate = copy.deepcopy(raw)
    missing_direct_prerequisites = missing_direct_certificate["prerequisites"]
    assert isinstance(missing_direct_prerequisites, dict)
    missing_direct_prerequisites.pop(
        "a0_direct_exact_prefix_certificate_sha256"
    )
    with pytest.raises(
        ValueError,
        match="a0_direct_exact_prefix_certificate_sha256|keys",
    ):
        resolve_h6_prediction_v3_config(
            missing_direct_certificate,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "schema_version",
    ("h6-prediction-config-v1", "h6-prediction-config-v2"),
)
def test_direct_a0_prerequisite_cannot_be_relabelled_as_v1_or_v2(
    tmp_path: Path,
    schema_version: str,
) -> None:
    relabelled = _prediction_v3_config()
    relabelled["schema_version"] = schema_version
    with pytest.raises(ValueError):
        resolve_h6_prediction_config(relabelled, repo_root=tmp_path)


def test_v3_config_binds_expected_test_row_count_4104(
    tmp_path: Path,
) -> None:
    resolved = resolve_h6_prediction_v3_config(
        _prediction_v3_config(),
        repo_root=tmp_path,
    )

    assert resolved.expected_test_row_count == 4104
    assert json.loads(resolved.canonical_json)["expected_test_row_count"] == 4104

    stale = _prediction_v3_config()
    stale["expected_test_row_count"] = 24_576
    with pytest.raises(ValueError, match="expected_test_row_count|4104"):
        resolve_h6_prediction_v3_config(stale, repo_root=tmp_path)

    bool_count = _prediction_v3_config()
    bool_count["expected_test_row_count"] = True
    with pytest.raises(ValueError, match="expected_test_row_count|4104"):
        resolve_h6_prediction_v3_config(bool_count, repo_root=tmp_path)


def test_v2_config_is_readable_but_cannot_authorize_v3_execution(
    tmp_path: Path,
) -> None:
    raw = _prediction_v2_config()

    generic = resolve_h6_prediction_config(raw, repo_root=tmp_path)
    dispatched = resolve_config(raw, repo_root=tmp_path)

    assert type(generic) is H6PredictionV2ResolvedConfig
    assert type(dispatched) is H6PredictionV2ResolvedConfig
    assert not isinstance(generic, H6PredictionV3ResolvedConfig)
    with pytest.raises(ValueError, match="v3"):
        resolve_h6_prediction_v3_config(raw, repo_root=tmp_path)


def test_v3_resolver_rejects_partial_or_legacy_matching_identity(
    tmp_path: Path,
) -> None:
    missing_policy_schema = _prediction_v3_config()
    missing_policy_schema.pop("matching_policy_schema")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_policy_schema,
            repo_root=tmp_path,
        )

    missing_policy_digest = _prediction_v3_config()
    missing_policy_digest.pop("matching_policy_sha256")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_policy_digest,
            repo_root=tmp_path,
        )

    missing_matching_set = _prediction_v3_config()
    missing_matching_set.pop("matching_set_sha256")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_matching_set,
            repo_root=tmp_path,
        )

    missing_matching_set_schema = _prediction_v3_config()
    missing_matching_set_schema.pop("matching_set_schema")
    with pytest.raises(ValueError):
        resolve_h6_prediction_v3_config(
            missing_matching_set_schema,
            repo_root=tmp_path,
        )

    legacy_policy = _prediction_v3_config()
    legacy_policy["matching_policy_schema"] = "h6-amended-matching-policy-v2"
    with pytest.raises(ValueError, match="matching_policy_schema|v3"):
        resolve_h6_prediction_v3_config(legacy_policy, repo_root=tmp_path)

    legacy_set = _prediction_v3_config()
    legacy_set["matching_set_schema"] = "h6-amended-matching-set-v2"
    with pytest.raises(ValueError, match="matching_set_schema|v3"):
        resolve_h6_prediction_v3_config(legacy_set, repo_root=tmp_path)

    aliased = _prediction_v3_config()
    aliased["matching_policy_sha256"] = aliased["matching_set_sha256"]
    with pytest.raises(ValueError, match="distinct"):
        resolve_h6_prediction_v3_config(aliased, repo_root=tmp_path)


def test_v3_attempt_cursor_binds_next_phase_and_counter_coordinates() -> None:
    attempt = H6AttemptSpecV3.create(
        git_head="1" * 40,
        dirty_digest=_sha("2"),
        readiness_sha256=_sha("3"),
        experiment_config_sha256=_sha("4"),
        endpoint_id="h6-a5-test",
        arm_id="A5",
        endpoint_config_sha256=_sha("5"),
        objective_kind="complete_elbo",
        model_factory_sha256=_sha("6"),
        recognition_factory_sha256=_sha("7"),
        initialization_sha256=_sha("8"),
        optimizer_policy_sha256=_sha("9"),
        training_seed=17,
        data_identity_sha256=_sha("a"),
        window_schedule_sha256=_sha("b"),
        batch_schedule_sha256=_sha("c"),
        phase_schedule_sha256=_sha("d"),
        training_schedule_sha256=_sha("e"),
        recognition_estimator_sha256=_sha("f"),
        runtime_identity_sha256=_sha("0"),
    )
    assert attempt.training_seed == 17
    assert attempt.objective_kind == "complete_elbo"
    with pytest.raises(ValueError, match="identity"):
        replace(attempt, training_seed=18)

    cursor = H6AttemptCursorV3.create(
        attempt_spec_sha256=_sha("1"),
        pass_index=1,
        batch_index=7,
        next_phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        example_ordinal=56,
        draw_block=3,
        counter_consumption_sha256=_sha("2"),
        permutation_sha256=_sha("3"),
        recognition_update_count=8,
        model_update_count=7,
        validation_boundary_count=20,
        checkpoint_boundary_count=2,
    )

    assert cursor.canonical_payload() == {
        "cursor_schema": "h6-attempt-cursor-v3",
        "attempt_spec_sha256": _sha("1"),
        "pass_index": 1,
        "batch_index": 7,
        "next_phase": TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
        "example_ordinal": 56,
        "sample_ordinal": 0,
        "draw_block": 3,
        "counter_consumption_sha256": _sha("2"),
        "permutation_sha256": _sha("3"),
        "recognition_update_count": 8,
        "model_update_count": 7,
        "validation_boundary_count": 20,
        "checkpoint_boundary_count": 2,
    }

    next_phase = H6AttemptCursorV3.create(
        attempt_spec_sha256=_sha("1"),
        pass_index=1,
        batch_index=7,
        next_phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=56,
        draw_block=3,
        counter_consumption_sha256=_sha("2"),
        permutation_sha256=_sha("3"),
        recognition_update_count=8,
        model_update_count=7,
        validation_boundary_count=20,
        checkpoint_boundary_count=2,
    )
    next_block = H6AttemptCursorV3.create(
        attempt_spec_sha256=_sha("1"),
        pass_index=1,
        batch_index=7,
        next_phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
        example_ordinal=56,
        draw_block=4,
        counter_consumption_sha256=_sha("2"),
        permutation_sha256=_sha("3"),
        recognition_update_count=8,
        model_update_count=7,
        validation_boundary_count=20,
        checkpoint_boundary_count=2,
    )

    assert next_phase.cursor_sha256 != cursor.cursor_sha256
    assert next_block.cursor_sha256 != cursor.cursor_sha256
    with pytest.raises(ValueError, match="cursor identity"):
        replace(cursor, next_phase=TrainingPhase.MODEL_ADAMW)
    with pytest.raises(ValueError, match="cursor identity"):
        replace(cursor, draw_block=4)
