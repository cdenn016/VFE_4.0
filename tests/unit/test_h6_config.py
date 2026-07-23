from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vfe4.config import resolve_config, resolve_h6_prediction_config, resolve_h6_prefix_config
from vfe4.types.h6 import TrainingPhase


def _sha(character: str) -> str:
    return character * 64


def _source() -> dict[str, object]:
    return {
        "git_head": "1" * 40,
        "dirty_digest": _sha("2"),
        "source_sha256": _sha("3"),
    }


def _prefix_config() -> dict[str, object]:
    return {
        "schema_version": "h6-prefix-config-v1",
        "operation": "H6-Prefix",
        "source": _source(),
        "structure": {
            "base": {"base_id": "C0", "points": ["*"], "dimension": 0},
            "dag": {
                "labeling": "zero_based",
                "node_labels": [0, 1, 2, 3, 4],
                "rows": [
                    {"receiver_t": 1, "parents": [0]},
                    {"receiver_t": 2, "parents": [0, 1]},
                    {"receiver_t": 3, "parents": [0, 1, 2]},
                    {"receiver_t": 4, "parents": [0, 1, 2, 3]},
                ],
            },
            "receiver_labels": [1, 2, 3, 4],
        },
        "model_family_sha256": _sha("4"),
        "vocabulary": {
            "vocabulary_id": "h6-prefix-small-v1",
            "size": 3,
            "tokenizer_spec_sha256": _sha("5"),
        },
        "estimator": {
            "schema_version": "h6-estimator-v1",
            "kind": "deterministic_exact",
            "particle_count": None,
            "resampling": "none",
            "dtype": "float64",
            "device": "cpu",
        },
        "data_safety_sha256": _sha("6"),
        "artifact_root": "runs/h6-prefix",
    }


def _prediction_config() -> dict[str, object]:
    return {
        "schema_version": "h6-prediction-config-v1",
        "operation": "H6-Prediction",
        "source": _source(),
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


def _reverse_mappings(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_mappings(item) for key, item in reversed(tuple(value.items()))}
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
    assert resolved.structure.base.dimension == 0
    assert resolved.structure.receiver_labels == (1, 2, 3, 4)
    assert resolved.estimator.kind == "deterministic_exact"
    assert resolved.artifact_root == (tmp_path / "runs/h6-prefix").resolve()
    assert "prerequisite" not in resolved.canonical_json
    assert "H1" not in resolved.canonical_json
    assert json.loads(resolved.canonical_json)["operation"] == "H6-Prefix"

    reordered = resolve_h6_prefix_config(
        _reverse_mappings(raw), repo_root=tmp_path  # type: ignore[arg-type]
    )
    assert reordered.canonical_json == resolved.canonical_json
    assert reordered.config_sha256 == resolved.config_sha256


def test_h6_helpers_delegate_to_the_single_public_resolver(tmp_path: Path) -> None:
    prefix_from_public = resolve_config(_prefix_config(), repo_root=tmp_path)
    prefix_from_helper = resolve_h6_prefix_config(_prefix_config(), repo_root=tmp_path)
    prediction_from_public = resolve_config(_prediction_config(), repo_root=tmp_path)
    prediction_from_helper = resolve_h6_prediction_config(
        _prediction_config(), repo_root=tmp_path
    )
    assert prefix_from_public == prefix_from_helper
    assert prediction_from_public == prediction_from_helper


def test_prefix_resolver_accepts_only_frozen_small_and_production_vocabularies(
    tmp_path: Path,
) -> None:
    production = _prefix_config()
    production["vocabulary"] = {
        "vocabulary_id": "wikitext-2-byte-v1",
        "size": 258,
        "tokenizer_spec_sha256": _sha("5"),
    }
    resolved = resolve_config(production, repo_root=tmp_path)
    assert resolved.vocabulary.size == 258
    assert resolved.vocabulary.vocabulary_id == "wikitext-2-byte-v1"

    invalid = _prefix_config()
    invalid["vocabulary"] = {
        "vocabulary_id": "h6-prefix-small-v1",
        "size": 258,
        "tokenizer_spec_sha256": _sha("5"),
    }
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_config(invalid, repo_root=tmp_path)


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
    assert "H4" not in resolved.canonical_json


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
