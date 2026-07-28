from __future__ import annotations

import dataclasses
import json
import shutil
from functools import cache
from pathlib import Path

import pytest

from vfe4.artifacts.atomic import ArtifactPublicationError
from vfe4.artifacts.h6_prediction_v3 import (
    publish_h6_prediction_v3_authorities,
    read_h6_prediction_v3_authorities,
)
from vfe4.artifacts.h6_matching import H6MatchingSetRecord
from vfe4.config import (
    H6PredictionV3ResolvedConfig,
    resolve_h6_prediction_v3_config,
)
from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS
from vfe4.numerics.critical_values import (
    CRITICAL_VALUES_PROTOCOL_SHA256,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_POLICY_V3,
    H6_MATCHING_V3_ENDPOINT_CONFIG_IDS,
    H6MatchingSetV3,
    H6TrainingWorkloadV3,
)
from vfe4.training.h6_experiment_v3 import plan_h6_experiment_v3
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3 as validate_h6_prediction_readiness_v3,
)
from vfe4.training.matching import (
    A5_REFERENCE_ALLOCATION,
    ARM_MATRIX_ROWS,
    H6_ADAMW_POLICY,
    arm_matrix_sha256,
    endpoint_formula_profile,
)
from vfe4.types import (
    H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    EndpointSmcProtocol,
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    ObjectiveGateSpec,
    TrainingPhase,
    VocabularyIdentity,
)
from vfe4.types.h6_prediction_v3 import (
    H6_CHECKPOINT_CODEC_SHA256,
    H6_COUNTER_MAPPING_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6_SCORING_INVENTORY_SHA256,
    H6PredictionRuntimeIdentity,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
)


_GIT_HEAD = "1" * 40
_DIRTY_DIGEST = "2" * 64
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _unsafe_record_mutation(record: object, **changes: object) -> object:
    """Forge an invalid typed record so the readiness seam must reject it."""

    mutated = object.__new__(type(record))
    for field in dataclasses.fields(record):
        object.__setattr__(
            mutated,
            field.name,
            changes.get(field.name, getattr(record, field.name)),
        )
    return mutated


def _endpoint_config(
    config_id: str,
    allocation: CapacityAllocation,
) -> ArmConfig:
    profile = endpoint_formula_profile(config_id)
    return ArmConfig.create(
        arm=ArmId(profile.arm),
        config_id=config_id,
        vocabulary=VocabularyIdentity(
            vocabulary_id="h6-readiness-v3-test",
            size=258,
            tokenizer_spec_sha256="3" * 64,
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
def _endpoint_templates() -> tuple[ArmConfig, ...]:
    templates: list[ArmConfig] = []
    for config_id in H6_MATCHING_V3_ENDPOINT_CONFIG_IDS:
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
        templates.append(_endpoint_config(config_id, allocation))
    return tuple(templates)


@cache
def _matching_set() -> H6MatchingSetV3:
    workload = H6TrainingWorkloadV3.from_train_tokens(
        train_token_count=258,
        train_token_sha256="4" * 64,
    )
    return H6MatchingSetV3.create(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        workload=workload,
        endpoint_templates=_endpoint_templates(),
    )


def _runtime(
    *,
    torch_full_version: str = "2.10.0.dev20251210+cu128",
) -> H6PredictionRuntimeIdentity:
    return H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version=torch_full_version,
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )


def _schedule(
    *,
    estimator: H6RecognitionEstimatorSpec,
    runtime: H6PredictionRuntimeIdentity,
    matching_set: H6MatchingSetV3,
) -> H6TrainingScheduleV3:
    endpoint_phases = tuple(
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
        for config in matching_set.endpoint_configs
    )
    return H6TrainingScheduleV3.create(
        outer=H6OuterSchedule.create(
            optimizer_policy_sha256=(H6_ADAMW_POLICY.optimizer_policy_sha256),
        ),
        endpoint_phases=endpoint_phases,
        estimator=estimator,
        runtime=runtime,
    )


def _config(
    *,
    matching_set: H6MatchingSetV3,
    artifact_root: Path,
) -> H6PredictionV3ResolvedConfig:
    estimator = H6RecognitionEstimatorSpec.create()
    runtime = _runtime()
    schedule = _schedule(
        estimator=estimator,
        runtime=runtime,
        matching_set=matching_set,
    )
    protocol = EndpointSmcProtocol.create(
        particle_counts=(128, 256, 512, 1024),
        replicate_count=64,
        registry_root_seed=2026072198,
        common_stream_domain="h6-wt2-endpoint-mc-v1",
        simultaneous_interval_count=352,
        familywise_alpha=0.01,
        critical_value_df63=4.5144904535377144,
        remainder_contraction=0.75,
    )
    objective_gate = ObjectiveGateSpec.create()
    estimator_payload = estimator.canonical_payload()
    estimator_payload["estimator_sha256"] = estimator.estimator_sha256
    runtime_payload = runtime.canonical_payload()
    runtime_payload["cuda_compute_capability"] = list(runtime.cuda_compute_capability)
    runtime_payload["runtime_identity_sha256"] = runtime.runtime_identity_sha256
    raw = {
        "schema_version": "h6-prediction-config-v3",
        "operation": "H6-Prediction",
        "source": {
            "git_head": _GIT_HEAD,
            "dirty_digest": _DIRTY_DIGEST,
            "source_sha256": "5" * 64,
        },
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
                "H1": "6" * 64,
                "H2": "7" * 64,
                "H3": "8" * 64,
                "H5": "9" * 64,
            },
            "h1_prefix_prior_manifest_sha256": "a" * 64,
            "h1_prefix_prior_generative_factor_schema_sha256": (
                H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
            ),
            "smc_validation_manifest_sha256": "b" * 64,
            "prefix_certificate_set_sha256": "c" * 64,
            "a0_direct_exact_prefix_certificate_sha256": "1" * 64,
        },
        "h5_update_binding_sha256": "d" * 64,
        "training_schedule": {
            "schedule_schema": "h6-training-schedule-v3",
            "outer": {
                "schedule_schema": schedule.outer.schedule_schema,
                "optimizer_class": schedule.outer.optimizer_class,
                "optimizer_policy_sha256": (schedule.outer.optimizer_policy_sha256),
                "model_updates_per_batch": (schedule.outer.model_updates_per_batch),
                "validation_twentieths_per_pass": (
                    schedule.outer.validation_twentieths_per_pass
                ),
                "full_passes": schedule.outer.full_passes,
            },
            "endpoint_phases": [
                {
                    "endpoint_config_sha256": phase.endpoint_config_sha256,
                    "latent_enabled": phase.latent_enabled,
                    "phases": [item.value for item in phase.phases],
                    "recognition_updates_per_batch": phase.recognition_updates_per_batch,
                    "model_updates_per_batch": phase.model_updates_per_batch,
                    "no_op_phases": phase.no_op_phases,
                }
                for phase in schedule.endpoint_phases
            ],
            "recognition_estimator_sha256": estimator.estimator_sha256,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "training_noise_domain": schedule.training_noise_domain,
            "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
            "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        },
        "critical_values_sha256": CRITICAL_VALUES_PROTOCOL_SHA256,
        "endpoint_smc_protocol": {
            "protocol_schema": protocol.protocol_schema,
            "particle_counts": list(protocol.particle_counts),
            "replicate_count": protocol.replicate_count,
            "registry_root_seed": protocol.registry_root_seed,
            "common_stream_domain": protocol.common_stream_domain,
            "simultaneous_interval_count": (protocol.simultaneous_interval_count),
            "familywise_alpha": protocol.familywise_alpha,
            "critical_value_df63": protocol.critical_value_df63,
            "remainder_contraction": protocol.remainder_contraction,
        },
        "smc_bias_semantics_sha256": (SMC_BIAS_SEMANTICS.semantics_sha256),
        "attribution_matrix_sha256": arm_matrix_sha256(ARM_MATRIX_ROWS),
        "matching_policy_schema": "h6-amended-matching-policy-v3",
        "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        "matching_set_schema": "h6-amended-matching-set-v3",
        "matching_set_sha256": matching_set.matching_set_sha256,
        "objective_gate": {
            "schema_version": objective_gate.schema_version,
            "complete_arm_id": objective_gate.complete_arm_id,
            "emission_arm_id": objective_gate.emission_arm_id,
            "orientation": objective_gate.orientation,
            "delta_obj": objective_gate.delta_obj,
            "opening_policy": objective_gate.opening_policy,
            "evaluation_order": objective_gate.evaluation_order,
            "spec_sha256": objective_gate.spec_sha256,
        },
        "data_identity_sha256": "e" * 64,
        "access_policy_sha256": "f" * 64,
        "recognition_contract": {
            "trajectory_schema": ("h6-language-recognition-trajectory-v3"),
            "categorical_posterior_schema": ("h6-categorical-source-posterior-v3"),
            "terminal_mixture_schema": "h6-terminal-source-mixture-v1",
            "estimator": estimator_payload,
        },
        "runtime": runtime_payload,
        "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
        "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
        "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        "scoring_inventory_sha256": H6_SCORING_INVENTORY_SHA256,
        "expected_test_row_count": 4104,
        "artifact_root": str(
            Path("C:/tmp") / "vfe4-h6-readiness-v3" / artifact_root.name
        ),
    }
    return resolve_h6_prediction_v3_config(raw, repo_root=_REPO_ROOT)


def test_prediction_v3_authorities_round_trip_and_reject_drift(
    tmp_path: Path,
) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    readiness = validate_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )

    published = publish_h6_prediction_v3_authorities(
        run_root=tmp_path,
        run_name="authorities",
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )
    reopened = read_h6_prediction_v3_authorities(published)
    assert reopened.config == config
    assert reopened.matching_set == matching_set
    assert reopened.readiness == readiness
    assert reopened.plan == plan
    assert (
        read_h6_prediction_v3_authorities(
            published,
            expected_authority_sha256=reopened.authority_sha256,
        )
        == reopened
    )
    with pytest.raises(ArtifactPublicationError, match="authority|digest"):
        read_h6_prediction_v3_authorities(
            published,
            expected_authority_sha256="0" * 64,
        )

    corrupt = tmp_path / "corrupt-authorities"
    shutil.copytree(published, corrupt)
    (corrupt / "authorities.json").write_bytes(b"{}")
    with pytest.raises(ArtifactPublicationError, match="manifest|authority"):
        read_h6_prediction_v3_authorities(corrupt)


def test_readiness_v3_rejects_v2_matching_set(tmp_path: Path) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    relabeled_v2 = object.__new__(H6MatchingSetRecord)
    object.__setattr__(
        relabeled_v2,
        "schema_version",
        "h6-amended-matching-set-v3",
    )

    with pytest.raises(ValueError, match="exact H6MatchingSetV3|v2"):
        validate_h6_prediction_readiness_v3(
            config=config,
            matching_set=relabeled_v2,  # type: ignore[arg-type]
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        )


def test_readiness_v3_public_issuer_requires_reopened_prerequisite_evidence(
    tmp_path: Path,
) -> None:
    from vfe4.training.h6_readiness import (
        validate_h6_prediction_readiness_v3 as issue_readiness,
    )

    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    with pytest.raises(ValueError, match="mechanically reopened"):
        issue_readiness(
            config=config,
            matching_set=matching_set,
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
            prerequisite_evidence=object(),  # type: ignore[arg-type]
        )


def test_readiness_v3_authorizes_primary_with_bound_component_disclosures(
    tmp_path: Path,
) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )

    assert (
        config.runtime.runtime_identity_sha256
        == config.training_schedule.runtime_identity_sha256
    )
    assert (
        config.recognition_estimator.estimator_sha256
        == config.training_schedule.recognition_estimator_sha256
    )
    assert (
        config.checkpoint_codec_sha256
        == config.training_schedule.checkpoint_codec_sha256
    )
    assert matching_set.status == "ELIGIBLE"
    assert matching_set.primary_selection.status == "ELIGIBLE"
    assert any(
        selection.status == "INCONCLUSIVE" and selection.obligations
        for selection in matching_set.component_selections
    )

    readiness = validate_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
    )

    assert readiness.status == "PASS"
    assert readiness.matching_policy_sha256 == H6_MATCHING_POLICY_V3.policy_sha256
    assert readiness.matching_set_sha256 == matching_set.matching_set_sha256
    assert readiness.recognition_estimator_sha256 == (
        config.recognition_estimator.estimator_sha256
    )
    assert readiness.runtime_identity_sha256 == config.runtime.runtime_identity_sha256
    assert readiness.checkpoint_codec_sha256 == H6_CHECKPOINT_CODEC_SHA256


def test_readiness_v3_requires_exact_canonical_reresolution(
    tmp_path: Path,
) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    assert json.loads(config.canonical_json)["matching_set_sha256"] == (
        matching_set.matching_set_sha256
    )

    with pytest.raises(ValueError, match="canonical v3"):
        validate_h6_prediction_readiness_v3(
            config=dataclasses.replace(
                config,
                config_sha256="0" * 64,
            ),
            matching_set=matching_set,
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        )
    with pytest.raises(ValueError, match="canonical v3"):
        validate_h6_prediction_readiness_v3(
            config=dataclasses.replace(
                config,
                canonical_json=json.dumps(
                    {
                        **json.loads(config.canonical_json),
                        "matching_set_sha256": "0" * 64,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
            matching_set=matching_set,
            git_head=_GIT_HEAD,
            dirty_digest=_DIRTY_DIGEST,
        )


def test_readiness_v3_refuses_typed_identity_mutation_before_dispatch(
    tmp_path: Path,
) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    for mutation in (
        dataclasses.replace(
            config,
            runtime=_runtime(torch_full_version="99.0+drift"),
        ),
        dataclasses.replace(
            config,
            checkpoint_codec_sha256="0" * 64,
        ),
    ):
        with pytest.raises(ValueError, match="canonical v3"):
            validate_h6_prediction_readiness_v3(
                config=mutation,
                matching_set=matching_set,
                git_head=_GIT_HEAD,
                dirty_digest=_DIRTY_DIGEST,
            )


def test_readiness_v3_refuses_matching_authority_mutations(
    tmp_path: Path,
) -> None:
    matching_set = _matching_set()
    config = _config(
        matching_set=matching_set,
        artifact_root=tmp_path,
    )
    second_eligible = matching_set.primary_selection.candidates[89]
    mutated_primary = _unsafe_record_mutation(
        matching_set.primary_selection,
        selected_candidate_sha256=second_eligible.candidate_sha256,
    )
    first_component = matching_set.component_selections[0]
    assert first_component.status == "INCONCLUSIVE"
    mutated_component = _unsafe_record_mutation(
        first_component,
        obligations=(),
    )
    mutated_components = (
        mutated_component,
        *matching_set.component_selections[1:],
    )

    for mutation in (
        _unsafe_record_mutation(
            matching_set,
            matching_policy_sha256="0" * 64,
        ),
        _unsafe_record_mutation(
            matching_set,
            primary_selection=mutated_primary,
        ),
        _unsafe_record_mutation(
            matching_set,
            component_selections=mutated_components,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="policy|first|eligible|selection|component|digest",
        ):
            validate_h6_prediction_readiness_v3(
                config=config,
                matching_set=mutation,  # type: ignore[arg-type]
                git_head=_GIT_HEAD,
                dirty_digest=_DIRTY_DIGEST,
            )
