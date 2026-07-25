from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vfe4.config import (
    H6PredictionV2ResolvedConfig,
    resolve_config,
    resolve_h6_prediction_config,
    resolve_h6_prediction_v2_config,
    resolve_h6_prefix_config,
)
from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS
from vfe4.training.arms import build_arm
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    H6LanguageStructure,
    H6PrefixProfilePair,
    ObjectiveGateSpec,
    TrainingPhase,
    VocabularyIdentity,
    ZeroDimensionalBase,
    canonical_json_bytes,
)


def _sha(character: str) -> str:
    return character * 64


FULL_PREFIX_AUTHORIZATION_SHA256 = hashlib.sha256(
    b"AUTHORIZE_VFE4_H6_PREFIX_FULL_INVENTORIES_V1"
).hexdigest()


def _data_safety_sha256() -> str:
    return hashlib.sha256(
        b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
    ).hexdigest()


def _source() -> dict[str, object]:
    return {
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "source_sha256": _sha("3"),
    }


def _structure(horizon: int) -> dict[str, object]:
    return {
        "base": {"base_id": "C0", "points": ["*"], "dimension": 0},
        "dag": {
            "labeling": "zero_based",
            "node_labels": list(range(horizon + 1)),
            "rows": [
                {"receiver_t": receiver, "parents": list(range(receiver))}
                for receiver in range(1, horizon + 1)
            ],
        },
        "receiver_labels": list(range(1, horizon + 1)),
    }


def _typed_structure(horizon: int) -> H6LanguageStructure:
    return H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=CausalDag.create(
            node_labels=tuple(range(horizon + 1)),
            rows=tuple(
                CausalDagRow(receiver, tuple(range(receiver)))
                for receiver in range(1, horizon + 1)
            ),
        ),
        receiver_labels=tuple(range(1, horizon + 1)),
    )


def _arm_config(*, vocabulary_size: int, horizon: int) -> ArmConfig:
    vocabulary = VocabularyIdentity(
        "h6-prefix-small-v1"
        if vocabulary_size == 3
        else "wikitext-2-byte-v1",
        vocabulary_size,
        _sha("5"),
    )
    return ArmConfig.create(
        arm=ArmId.A0,
        config_id="h6-a0-transformer-v2",
        vocabulary=vocabulary,
        horizon=horizon,
        latent_enabled=False,
        state_channel_enabled=False,
        model_channel_enabled=False,
        source_mode="absent",
        map_mode="absent",
        recognition_family="absent",
        recognition_conditioning="absent",
        prior_variant="absent",
        mixture_mode="absent",
        objective_kind="cross_entropy",
        capacity_allocation=CapacityAllocation.create(
            emission_width=48 if vocabulary_size == 3 else 52,
            latent_width=None,
            recognition_width=None,
        ),
    )


def _raw_arm_config(config: ArmConfig) -> dict[str, object]:
    payload = config.canonical_payload()
    payload.pop("capacity_allocation_sha256")
    return payload


def _profile(particle_count: int) -> dict[str, object]:
    small = _arm_config(vocabulary_size=3, horizon=4)
    production = _arm_config(vocabulary_size=258, horizon=32)
    estimator = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=particle_count,
        resampling="systematic_ess_half",
    )
    small_model_family_sha256 = hashlib.sha256(
        b"vfe4.h6.arm-model-family.v1\x00"
        + canonical_json_bytes(
            {
                "config_sha256": small.config_sha256,
                "factory": "build_a0@h6-arm-v2",
            }
        )
    ).hexdigest()
    production_model_family_sha256 = hashlib.sha256(
        b"vfe4.h6.arm-model-family.v1\x00"
        + canonical_json_bytes(
            {
                "config_sha256": production.config_sha256,
                "factory": "build_a0@h6-arm-v2",
            }
        )
    ).hexdigest()
    pair = H6PrefixProfilePair.create(
        profile_id=f"h6-a0-smc-{particle_count}-v1",
        small_arm_config=small,
        production_arm_config=production,
        estimator=estimator,
        small_structure=_typed_structure(4),
        production_structure=_typed_structure(32),
        data_safety_sha256=_data_safety_sha256(),
        small_model_family_sha256=small_model_family_sha256,
        production_model_family_sha256=production_model_family_sha256,
    )
    return {
        "profile_id": pair.profile_id,
        "small_arm_config": _raw_arm_config(small),
        "production_arm_config": _raw_arm_config(production),
        "estimator": {
            "schema_version": estimator.schema_version,
            "kind": estimator.kind,
            "particle_count": estimator.particle_count,
            "resampling": estimator.resampling,
            "dtype": estimator.dtype,
            "device": estimator.device,
        },
        "small_structure": _structure(4),
        "production_structure": _structure(32),
        "data_safety_sha256": pair.data_safety_sha256,
        "small_model_family_sha256": pair.small_model_family_sha256,
        "production_model_family_sha256": (
            pair.production_model_family_sha256
        ),
        "profile_pair_sha256": pair.profile_pair_sha256,
    }


def _prefix_config(
    *,
    execution_mode: str = "focused_subset",
    particle_counts: tuple[int, ...] = (4,),
    authorization_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "h6-prefix-config-v1",
        "operation": "H6-Prefix",
        "source": _source(),
        "execution_mode": execution_mode,
        "profiles": [_profile(count) for count in particle_counts],
        "authorization_sha256": authorization_sha256,
        "artifact_root": "runs/h6-prefix",
    }


def _prediction_config() -> dict[str, object]:
    return {
        "schema_version": "h6-prediction-config-v1",
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
        "attribution_matrix_sha256": _sha("f"),
        "matching_set_sha256": _sha("0"),
        "data_identity_sha256": _sha("1"),
        "access_policy_sha256": _sha("2"),
        "artifact_root": "runs/h6-prediction",
    }


def _prediction_v2_config() -> dict[str, object]:
    raw = copy.deepcopy(_prediction_config())
    raw["schema_version"] = "h6-prediction-config-v2"
    raw["smc_bias_semantics_sha256"] = (
        SMC_BIAS_SEMANTICS.semantics_sha256
    )
    prerequisites = raw["prerequisites"]
    assert isinstance(prerequisites, dict)
    prerequisites[
        "h1_prefix_prior_generative_factor_schema_sha256"
    ] = "0ab33d1cc790711eee82c598bb853d46ab52662eb31e9433e973978e77d9e375"
    objective = ObjectiveGateSpec.create()
    raw["objective_gate"] = {
        "schema_version": objective.schema_version,
        "complete_arm_id": objective.complete_arm_id,
        "emission_arm_id": objective.emission_arm_id,
        "orientation": objective.orientation,
        "delta_obj": objective.delta_obj,
        "opening_policy": objective.opening_policy,
        "evaluation_order": objective.evaluation_order,
        "spec_sha256": objective.spec_sha256,
    }
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


def test_prefix_config_is_independent_strict_and_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    raw = _prefix_config()
    before = copy.deepcopy(raw)
    resolved = resolve_h6_prefix_config(raw, repo_root=tmp_path)

    assert raw == before
    assert resolved.operation == "H6-Prefix"
    assert resolved.source.git_head == "1" * 40
    assert resolved.source.dirty_digest == _sha("2")
    assert resolved.source.source_sha256 == _sha("3")
    assert resolved.execution_mode == "focused_subset"
    assert resolved.authorization_sha256 is None
    assert len(resolved.profiles) == 1
    assert resolved.profiles[0].small_structure.receiver_labels == (1, 2, 3, 4)
    assert resolved.profiles[0].production_structure.receiver_labels == tuple(
        range(1, 33)
    )
    assert resolved.profiles[0].estimator.kind == "weighted_smc"
    assert resolved.profiles[0].estimator.particle_count == 4
    assert resolved.artifact_root == (tmp_path / "runs/h6-prefix").resolve()
    assert "prerequisite" not in resolved.canonical_json
    assert "H1" not in resolved.canonical_json
    assert json.loads(resolved.canonical_json)["operation"] == "H6-Prefix"
    assert json.loads(resolved.canonical_json)["source"] == _source()

    reordered = resolve_h6_prefix_config(
        _reverse_mappings(raw), repo_root=tmp_path  # type: ignore[arg-type]
    )
    assert reordered.canonical_json == resolved.canonical_json
    assert reordered.config_sha256 == resolved.config_sha256

    explicit = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=128,
        resampling="systematic_ess_half",
    )
    built = build_arm(ArmId.A0, resolved.profiles[0].small_arm_config)
    assert built.predictor.estimator_spec.particle_count == 4
    _, rebuilt = built.rebuild_predictive_boundary(explicit)
    assert rebuilt.estimator_spec == explicit

    deterministic = _prefix_config()
    deterministic["profiles"][0]["estimator"] = {  # type: ignore[index]
        "schema_version": "h6-estimator-v1",
        "kind": "deterministic_exact",
        "particle_count": None,
        "resampling": "none",
        "dtype": "float64",
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="weighted_smc"):
        resolve_h6_prefix_config(deterministic, repo_root=tmp_path)

    with pytest.raises(ValueError, match="authorization"):
        resolve_h6_prefix_config(
            _prefix_config(authorization_sha256=_sha("a")),
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="128.*256.*512.*1024|ladder"):
        resolve_h6_prefix_config(
            _prefix_config(
                execution_mode="authorized_full",
                particle_counts=(128, 256, 512),
                authorization_sha256=FULL_PREFIX_AUTHORIZATION_SHA256,
            ),
            repo_root=tmp_path,
        )

    authorized = resolve_h6_prefix_config(
        _prefix_config(
            execution_mode="authorized_full",
            particle_counts=(128, 256, 512, 1024),
            authorization_sha256=FULL_PREFIX_AUTHORIZATION_SHA256,
        ),
        repo_root=tmp_path,
    )
    assert tuple(
        profile.estimator.particle_count for profile in authorized.profiles
    ) == (128, 256, 512, 1024)
    assert (
        authorized.authorization_sha256
        == FULL_PREFIX_AUTHORIZATION_SHA256
    )

    stale = _prefix_config()
    stale["profiles"][0]["small_model_family_sha256"] = _sha("f")  # type: ignore[index]
    with pytest.raises(ValueError, match="model.family|profile"):
        resolve_h6_prefix_config(stale, repo_root=tmp_path)

    wrong_safety = _prefix_config()
    wrong_safety["profiles"][0]["data_safety_sha256"] = _sha("6")  # type: ignore[index]
    with pytest.raises(ValueError, match="implemented.*safety"):
        resolve_h6_prefix_config(wrong_safety, repo_root=tmp_path)


def test_h6_helpers_delegate_to_the_single_public_resolver(tmp_path: Path) -> None:
    prefix_from_public = resolve_config(_prefix_config(), repo_root=tmp_path)
    prefix_from_helper = resolve_h6_prefix_config(_prefix_config(), repo_root=tmp_path)
    prediction_from_public = resolve_config(_prediction_config(), repo_root=tmp_path)
    prediction_from_helper = resolve_h6_prediction_config(
        _prediction_config(), repo_root=tmp_path
    )
    assert prefix_from_public == prefix_from_helper
    assert prediction_from_public == prediction_from_helper


def test_prefix_config_rejects_predecessor_or_h1_h5_fields(tmp_path: Path) -> None:
    raw = _prefix_config()
    raw["predecessor_refs"] = {}
    with pytest.raises(ValueError, match="unknown keys|predecessor"):
        resolve_h6_prefix_config(raw, repo_root=tmp_path)


def test_prediction_config_requires_exact_non_h4_prerequisites(tmp_path: Path) -> None:
    raw = _prediction_config()
    before = copy.deepcopy(raw)
    resolved = resolve_h6_prediction_config(raw, repo_root=tmp_path)

    assert raw == before
    assert resolved.operation == "H6-Prediction"
    assert tuple(gate for gate, _ in resolved.correctness_manifests) == (
        "H1",
        "H2",
        "H3",
        "H5",
    )
    assert resolved.h1_prefix_prior_manifest_sha256 == _sha("7")
    assert resolved.prefix_certificate_set_sha256 == _sha("9")
    assert resolved.training_schedule.endpoint_phases[0].latent_enabled is False
    assert resolved.training_schedule.endpoint_phases[1].latent_enabled is True
    assert resolved.data.observed_archive is None
    assert resolved.data.max_archive_bytes == 16_777_216
    assert "H4" not in resolved.canonical_json
    assert "smc_bias_semantics_sha256" not in resolved.canonical_json


def test_prediction_v2_binds_scorer_and_objective_without_upgrading_v1(
    tmp_path: Path,
) -> None:
    raw = _prediction_v2_config()
    before = copy.deepcopy(raw)
    resolved = resolve_h6_prediction_v2_config(raw, repo_root=tmp_path)
    assert type(resolved) is H6PredictionV2ResolvedConfig
    assert raw == before
    assert resolved.h1_prefix_prior_generative_factor_schema_sha256 == (
        "0ab33d1cc790711eee82c598bb853d46ab52662eb31e9433e973978e77d9e375"
    )
    assert (
        resolved.smc_bias_semantics_sha256
        == SMC_BIAS_SEMANTICS.semantics_sha256
    )
    assert resolved.objective_gate == ObjectiveGateSpec.create()
    assert json.loads(resolved.canonical_json)[
        "smc_bias_semantics_sha256"
    ] == SMC_BIAS_SEMANTICS.semantics_sha256
    assert json.loads(resolved.canonical_json)["objective_gate"][
        "spec_sha256"
    ] == resolved.objective_gate.spec_sha256
    assert (
        resolve_h6_prediction_config(raw, repo_root=tmp_path)
        == resolved
    )
    with pytest.raises(ValueError, match="not an H6 Prediction v2"):
        resolve_h6_prediction_v2_config(
            _prediction_config(),
            repo_root=tmp_path,
        )

    tampered = copy.deepcopy(raw)
    tampered["objective_gate"]["delta_obj"] = 0.02  # type: ignore[index]
    with pytest.raises(ValueError, match="delta_obj"):
        resolve_h6_prediction_v2_config(tampered, repo_root=tmp_path)

    tampered = copy.deepcopy(raw)
    tampered["smc_bias_semantics_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="smc_bias_semantics_sha256"):
        resolve_h6_prediction_v2_config(tampered, repo_root=tmp_path)


def test_prediction_data_config_accepts_bound_observations_and_rejects_mirrors(
    tmp_path: Path,
) -> None:
    raw = _prediction_config()
    raw["data"]["observed_archive"] = {  # type: ignore[index]
        "archive_byte_length": 1234,
        "archive_sha256": _sha("a"),
        "members": [
            {
                "path": path,
                "compressed_size": 10 + index,
                "uncompressed_size": 20 + index,
                "compression_method": 8,
                "crc32": 100 + index,
                "raw_sha256": _sha(character),
            }
            for index, (path, character) in enumerate(
                (
                    ("wikitext-2-raw/wiki.train.raw", "b"),
                    ("wikitext-2-raw/wiki.valid.raw", "c"),
                    ("wikitext-2-raw/wiki.test.raw", "d"),
                )
            )
        ],
    }
    resolved = resolve_config(raw, repo_root=tmp_path)
    assert resolved.data.observed_archive is not None
    assert resolved.data.observed_archive.members[1].crc32 == 101

    raw["data"]["source_url"] = "https://example.invalid/mirror.zip"  # type: ignore[index]
    with pytest.raises(ValueError, match="source_url"):
        resolve_config(raw, repo_root=tmp_path)


def test_prediction_config_is_blocked_without_exact_prefix_set(tmp_path: Path) -> None:
    raw = _prediction_config()
    raw["prerequisites"]["prefix_certificate_set_sha256"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="exact H6-Prefix certificate set"):
        resolve_h6_prediction_config(raw, repo_root=tmp_path)


def test_prediction_config_rejects_h4_even_when_other_prerequisites_are_exact(
    tmp_path: Path,
) -> None:
    raw = _prediction_config()
    raw["prerequisites"]["correctness_manifests"]["H4"] = _sha("4")  # type: ignore[index]
    with pytest.raises(ValueError, match="H1.*H2.*H3.*H5|H4|unknown"):
        resolve_h6_prediction_config(raw, repo_root=tmp_path)


def test_prediction_config_hash_is_order_independent_and_input_is_immutable(
    tmp_path: Path,
) -> None:
    raw = _prediction_config()
    before = copy.deepcopy(raw)
    resolved = resolve_h6_prediction_config(raw, repo_root=tmp_path)
    reordered = resolve_h6_prediction_config(
        _reverse_mappings(raw), repo_root=tmp_path  # type: ignore[arg-type]
    )
    assert raw == before
    assert reordered.canonical_json == resolved.canonical_json
    assert reordered.config_sha256 == resolved.config_sha256
