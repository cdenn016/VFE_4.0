from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from typing import get_args, get_type_hints

import pytest

from vfe4.config import (
    FigureConfig,
    TrainingConfig,
    default_figure_config_mapping,
    default_training_config_mapping,
    resolve_config,
    resolve_figure_config,
    resolve_training_config,
)
from vfe4.types.figures import FigureSpec
from vfe4.types.training import (
    A0ArchitectureProfile,
    CandidateTokenizerContract,
    DataCursor,
    EndpointInventory,
    PermutationManifest,
    ProductionTokenCacheIdentity,
    ProductionTokenizerSpec,
    SyntheticFixtureTokenCacheIdentity,
    SyntheticFixtureTokenizerSpec,
    WT103ExperimentProfile,
    validate_endpoint_inventory,
)


EXPECTED_ARM_IDS = (
    "WT103-A0-AR-v1",
    "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
    "WT103-A5-FIXED-COMPLETE-v1",
    "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1",
    "WT103-A5-NOLATENT-v1",
)
EXPECTED_GATE_IDS = (
    "SOURCE_LOCK",
    "H8_EXACT_REVISION",
    "POST_H8_READINESS",
    "OBJECTIVE",
    "PRIMARY",
    "PRIOR_CONTROL",
    "LATENT_PATH_CONTROL",
)


def _resolved() -> TrainingConfig:
    return resolve_training_config(default_training_config_mapping())


def test_default_training_config_freezes_the_shared_profile_and_arm_semantics() -> None:
    resolved = _resolved()

    assert type(resolved) is TrainingConfig
    assert type(resolved.profile) is WT103ExperimentProfile
    profile = resolved.profile
    assert (
        profile.schema_version,
        profile.dataset_schema,
        profile.tokenizer_schema,
        profile.vocabulary_size,
        profile.sequence_length,
        profile.stride,
        profile.batch_size,
        profile.gradient_accumulation_steps,
        profile.num_workers,
        profile.pin_memory,
        profile.drop_last,
    ) == (
        "wt103-experiment-profile-v1",
        "wikitext-103-raw-v1",
        "gpt2-tiktoken-v1",
        50_257,
        128,
        128,
        128,
        1,
        0,
        True,
        False,
    )
    assert (
        profile.model_depth,
        profile.d_z,
        profile.d_m,
        profile.K,
        profile.combined_latent_block,
        profile.source_lookback,
    ) == (1, 20, 20, 20, 40, 20)
    assert profile.state_parent_rule == "range(max(0,t-20),t)"
    assert profile.model_parent_rule == "range(max(0,t-20),t)"
    assert (
        profile.decoder_train_token_chunk,
        profile.decoder_eval_token_chunk,
        profile.smc_particle_chunk,
    ) == (512, 256, 32)
    assert profile.optimizer.betas == (0.9, 0.999)
    assert profile.optimizer.epsilon == 1.0e-8
    assert profile.optimizer.amsgrad is False
    assert profile.optimizer.foreach is False
    assert profile.optimizer.fused is False
    assert profile.optimizer.gradient_clip_max_norm == 1.0
    assert profile.scheduler.warmup_optimizer_steps == 100
    assert profile.scheduler.minimum_lr_ratio == 0.1
    assert profile.precision.real_training_device == "cuda:0"
    assert profile.precision.autocast_dtype == "bfloat16"
    assert profile.precision.smc_log_weight_dtype == "float64"
    assert profile.cadence.validation_boundaries_per_pass == 20
    assert profile.cadence.confirmatory_passes == 2
    assert profile.cadence.early_stopping is False
    assert profile.checkpoints.rolling_role == "resume_only"
    assert profile.checkpoints.terminal_role == "terminal_scoring"
    assert profile.checkpoints.best_checkpoint_selection is False
    assert profile.statistics.tuning_seed_ids == (2026072199, 2026072200)
    assert profile.statistics.confirmatory_seed_ids == tuple(
        range(2026072101, 2026072109)
    )
    assert profile.statistics.learning_rate_grid == (1.0e-4, 3.0e-4, 1.0e-3)
    assert profile.statistics.weight_decay_grid == (0.0, 1.0e-2)
    assert profile.statistics.validation_stream_ids == tuple(range(8))
    assert profile.statistics.test_stream_ids == tuple(range(64))
    assert profile.statistics.particle_counts == (128, 256, 512, 1024)
    assert profile.statistics.simultaneous_constant == 4.5144904535377144
    assert profile.statistics.practical_threshold == 0.01005033585350145
    assert profile.resources.maximum_gpu_hours == 720.0
    assert profile.resources.maximum_wall_hours == 840.0
    assert profile.resources.maximum_energy_kwh == 500.0
    assert profile.resources.forecast_headroom_factor == 1.25
    assert profile.schemas.h8_schema == "h8-sparse-scale-v5"
    assert profile.schemas.h8_config_schema == "h8-validation-config-v3"
    assert (
        profile.schemas.h8_parent_child_protocol == "vfe4.h8.parent-child-protocol.v3"
    )
    assert profile.schemas.h6_prediction_schema == "h6-prediction-result-v3"
    assert profile.schemas.training_sparsity_schema == "wt103-training-sparsity-v1"
    assert profile.schemas.metric_schema == "wt103-metric-record-v1"
    assert profile.schemas.figure_schema == "wt103-figure-spec-v1"
    assert profile.nonclaims.backprop_free is False
    assert profile.nonclaims.h8_training_memory_transfer is False
    assert profile.nonclaims.h6_byte_vocabulary_transfer is False
    assert profile.nonclaims.v3_checkpoint_or_config_reuse is False

    arms = resolved.endpoint_inventory.arms
    assert tuple(arm.arm_id for arm in arms) == EXPECTED_ARM_IDS
    assert tuple(arm.result_role for arm in arms) == (
        "PRIMARY_REFERENCE",
        "PRIMARY_ENDPOINT",
        "PRIOR_CONTROL",
        "OBJECTIVE_GATE",
        "LATENT_PATH_CONTROL",
    )
    assert tuple(arm.training_objective for arm in arms) == (
        "cross_entropy",
        "complete_elbo",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
        "cross_entropy",
    )
    assert tuple(arm.prior_variant for arm in arms) == (
        "absent",
        "parent_specific_pooled_prefix",
        "fixed",
        "parent_specific_pooled_prefix",
        "absent",
    )
    assert tuple(arm.scorer_kind for arm in arms) == (
        "exact_autoregressive",
        "weighted_smc",
        "weighted_smc",
        "weighted_smc",
        "exact_autoregressive",
    )
    for arm in (arms[1], arms[2], arms[3]):
        assert arm.source_mixture == "exact"
        assert arm.latent_enabled is True
        assert arm.recognition_enabled is True
        assert arm.recognition_family == "structured_block_tridiagonal_smoothing"
        assert arm.recognition_iterations_per_batch == 1
        assert arm.update_phases == (
            "recognition_adam_proposal",
            "immutable_detached_snapshot",
            "model_adam_proposal",
        )
    for arm in (arms[0], arms[4]):
        assert arm.source_mixture == "absent"
        assert arm.latent_enabled is False
        assert arm.recognition_enabled is False
        assert arm.recognition_family == "absent"
        assert arm.recognition_iterations_per_batch == 0
        assert arm.update_phases == ("model_ce_adam_proposal",)

    profile_fields = {field.name for field in dataclasses.fields(profile)}
    assert (
        not {
            "training_objective",
            "prior_variant",
            "source_mixture",
            "recognition_family",
            "scorer_kind",
            "update_phases",
            "result_role",
        }
        & profile_fields
    )


def test_endpoint_inventory_is_the_only_source_of_derived_keys_and_counts() -> None:
    inventory = _resolved().endpoint_inventory

    assert type(inventory) is EndpointInventory
    assert tuple(gate.gate_id for gate in inventory.gates) == EXPECTED_GATE_IDS
    objective = inventory.gates[3]
    primary = inventory.gates[4]
    assert objective.ordinal < primary.ordinal
    assert "OBJECTIVE" in primary.prerequisite_gate_ids
    assert inventory.tuning_attempt_count == 60
    assert inventory.terminal_checkpoint_count == 40
    assert inventory.validation_endpoint_count == 1_660
    assert inventory.test_endpoint_count == 40
    assert inventory.raw_score_record_count == 14_792
    assert inventory.result_row_count == 5
    assert inventory.figure_panel_count == 8
    assert inventory.figure_series_count == 47
    assert len(set(inventory.tuning_attempt_keys)) == 60
    assert len(set(inventory.terminal_checkpoint_keys)) == 40
    assert len(set(inventory.raw_score_record_keys)) == 14_792
    validate_endpoint_inventory(
        inventory,
        expected_sha256=inventory.endpoint_inventory_sha256,
    )
    with pytest.raises(ValueError, match="endpoint_inventory_sha256"):
        validate_endpoint_inventory(inventory, expected_sha256="0" * 64)
    with pytest.raises(
        (AttributeError, TypeError, dataclasses.FrozenInstanceError),
    ):
        inventory.tuning_attempt_count = 0  # type: ignore[misc]


def test_a0_architecture_is_flash_only_and_source_lock_identity_is_explicit() -> None:
    architecture = _resolved().a0_architecture

    assert type(architecture) is A0ArchitectureProfile
    assert architecture.block_count == 1
    assert architecture.attention_heads == 2
    assert architecture.attention_context == "full_causal_inclusive_self"
    assert architecture.attention_backend_policy == ("flash_attention_only_no_fallback")
    assert architecture.pytorch_sdpa_api_binding == (
        "torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])"
    )
    assert architecture.enabled_backends == ("FLASH_ATTENTION",)
    assert architecture.alternative_backends_disabled is True
    assert architecture.backend_fallback_allowed is False
    assert architecture.attention_mask_argument is None
    assert architecture.attention_returns_weights is False
    assert architecture.fused_attention_materialization == "forbidden"
    assert architecture.positional_encoding == "learned_absolute"
    assert architecture.normalization_placement == "pre_norm_with_final_norm"
    assert architecture.residual_topology == (
        "x=x+attn(ln1(x));x=x+mlp(ln2(x));y=ln_f(x)"
    )
    assert architecture.activation == "gelu_tanh_approximation"
    assert architecture.decoder_projection.startswith("untied_Linear")
    assert architecture.candidate_hidden_widths == (
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64,
        72,
        80,
        96,
        112,
        128,
        160,
    )
    assert architecture.parameter_relative_tolerance == 0.01
    assert architecture.flop_relative_tolerance == 0.05
    assert architecture.source_lock_scope == "candidate_unverified"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update({"unknown": 1}),
        lambda raw: raw["profile"]["optimizer"].update({"momentum": 0.9}),
        lambda raw: raw["profile"].update({"batch_size": True}),
        lambda raw: raw["profile"]["statistics"].update(
            {"confirmatory_seed_ids": [2026072101]}
        ),
        lambda raw: raw["a0_architecture"].update(
            {"enabled_backends": ["FLASH_ATTENTION", "MATH"]}
        ),
        lambda raw: raw["a0_architecture"].update({"backend_fallback_allowed": True}),
        lambda raw: raw["a0_architecture"].update(
            {"fused_attention_materialization": "allowed"}
        ),
        lambda raw: raw["arms"].reverse(),
        lambda raw: raw["gates"].reverse(),
        lambda raw: raw["arms"][1].pop("scorer_kind"),
        lambda raw: raw.update({"endpoint_inventory_sha256": "0" * 64}),
        lambda raw: raw["profile"]["nonclaims"].update({"backprop_free": True}),
        lambda raw: raw["profile"]["nonclaims"].update(
            {"h8_training_memory_transfer": True}
        ),
        lambda raw: raw["profile"]["schemas"].update(
            {"h8_schema": "h8-sparse-scale-v4"}
        ),
        lambda raw: raw["profile"]["schemas"].update(
            {"h6_prediction_schema": "h6-prediction-result-v2"}
        ),
    ],
)
def test_training_resolver_rejects_recursive_drift(mutate) -> None:  # type: ignore[no-untyped-def]
    raw = default_training_config_mapping()
    mutate(raw)

    with pytest.raises(ValueError):
        resolve_training_config(raw)


def test_training_resolution_is_pure_frozen_and_canonical() -> None:
    raw = default_training_config_mapping()
    before = copy.deepcopy(raw)
    reordered = {key: raw[key] for key in reversed(tuple(raw))}

    resolved = resolve_training_config(raw)
    second = resolve_training_config(reordered)

    assert raw == before
    assert resolved.canonical_json == second.canonical_json
    assert resolved.config_sha256 == second.config_sha256
    assert (
        hashlib.sha256(resolved.canonical_json.encode("utf-8")).hexdigest()
        == resolved.config_sha256
    )
    assert json.loads(resolved.canonical_json)["schema_version"] == (
        "wt103-training-config-v1"
    )
    assert type(resolved.endpoint_inventory.arms) is tuple
    assert type(resolved.endpoint_inventory.arms[0].update_phases) is tuple
    assert dataclasses.is_dataclass(resolved.profile)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.profile.batch_size = 64  # type: ignore[misc]


def test_launcher_operation_identity_is_separate_from_experiment_identity() -> None:
    resolved = []
    for operation in (
        "readiness",
        "train",
        "resume",
    ):
        raw = default_training_config_mapping()
        raw["operation"] = operation
        resolved.append(resolve_training_config(raw))

    assert len({item.config_sha256 for item in resolved}) == 3
    assert len(
        {item.experiment_config_sha256 for item in resolved}
    ) == 1

    experiment_document = json.loads(resolved[0].canonical_json)
    experiment_document.pop("operation")
    expected = hashlib.sha256(
        json.dumps(
            experiment_document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert resolved[0].experiment_config_sha256 == expected
    experiment_document["profile"]["sequence_length"] = 64
    changed = hashlib.sha256(
        json.dumps(
            experiment_document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert changed != resolved[0].experiment_config_sha256


def test_public_resolver_dispatches_without_weakening_existing_discriminator() -> None:
    raw = default_training_config_mapping()

    assert resolve_config(raw, repo_root=None) == resolve_training_config(raw)
    changed = copy.deepcopy(raw)
    changed["schema_version"] = "h6-prediction-config-v2"
    with pytest.raises(ValueError):
        resolve_training_config(changed)


def test_tokenizer_and_cache_domains_are_structurally_disjoint() -> None:
    candidate = CandidateTokenizerContract()
    synthetic = SyntheticFixtureTokenizerSpec.create(
        adapter_sha256="1" * 64,
        fixture_sha256="2" * 64,
    )
    synthetic_cache = SyntheticFixtureTokenCacheIdentity.create(
        tokenizer=synthetic,
        payload_sha256="3" * 64,
    )
    production_domain = get_args(get_type_hints(ProductionTokenizerSpec)["hash_domain"])

    assert (candidate.distribution, candidate.version, candidate.encoding_name) == (
        "tiktoken",
        "0.12.0",
        "gpt2",
    )
    assert production_domain == ("vfe4.wt103.production-tokenizer-spec.v1\x00",)
    assert synthetic.hash_domain not in production_domain
    assert synthetic_cache.schema_version == ("wt103-synthetic-fixture-token-cache-v1")
    with pytest.raises(ValueError, match="ProductionTokenizerSpec"):
        ProductionTokenCacheIdentity.create(  # type: ignore[arg-type]
            tokenizer=synthetic,
            split="train",
            payload_sha256="9" * 64,
        )
    with pytest.raises(ValueError, match="SyntheticFixtureTokenizerSpec"):
        SyntheticFixtureTokenCacheIdentity.create(  # type: ignore[arg-type]
            tokenizer=candidate,
            payload_sha256="3" * 64,
        )


def test_permutation_manifest_and_data_cursor_own_their_hashes() -> None:
    permutation = PermutationManifest.create(
        pass_index=0,
        numpy_version="2.3.2",
        window_manifest_sha256="1" * 64,
        payload_sha256="2" * 64,
    )
    second_pass = PermutationManifest.create(
        pass_index=1,
        numpy_version="2.3.2",
        window_manifest_sha256="1" * 64,
        payload_sha256="2" * 64,
    )
    cursor = DataCursor.create(
        split="train",
        pass_index=0,
        permutation_sha256=permutation.manifest_sha256,
        next_batch_ordinal=3,
        next_window_ids=(384, 25, 901),
        counted_targets=49_152,
    )
    advanced = DataCursor.create(
        split="train",
        pass_index=0,
        permutation_sha256=permutation.manifest_sha256,
        next_batch_ordinal=4,
        next_window_ids=(17, 63),
        counted_targets=65_536,
    )

    assert permutation.schema_version == "wt103-permutation-manifest-v1"
    assert permutation.split == "train"
    assert permutation.data_order_seed == 2026072199
    assert permutation.bit_generator == "PCG64"
    assert permutation.manifest_sha256 != second_pass.manifest_sha256
    assert cursor.schema_version == "wt103-data-cursor-v1"
    assert cursor.cursor_sha256 != advanced.cursor_sha256
    with pytest.raises(ValueError, match="manifest_sha256"):
        dataclasses.replace(permutation, manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="cursor_sha256"):
        dataclasses.replace(cursor, cursor_sha256="0" * 64)


def test_types_and_config_do_not_load_live_tokenizer_or_metadata_discovery() -> None:
    before = set(sys.modules)
    _resolved()
    newly_loaded = set(sys.modules) - before

    assert "tiktoken" not in newly_loaded
    assert "importlib.metadata" not in newly_loaded


def test_figure_config_is_strict_frozen_and_inventory_bound() -> None:
    training = _resolved()
    raw = default_figure_config_mapping(training.endpoint_inventory)
    resolved = resolve_figure_config(raw)

    assert type(resolved) is FigureConfig
    assert resolved.endpoint_inventory_sha256 == (
        training.endpoint_inventory.endpoint_inventory_sha256
    )
    assert len(resolved.specs) == 8
    assert all(type(spec) is FigureSpec for spec in resolved.specs)
    assert tuple(spec.figure_id for spec in resolved.specs) == (
        "training-objective-and-validation",
        "terminal-prior-nll-ppl",
        "complete-elbo-decomposition",
        "source-entropy-effective-count",
        "update-acceptance",
        "spd-health",
        "throughput-memory",
        "seed-variability",
    )
    assert all(spec.formats == ("svg", "png", "pdf") for spec in resolved.specs)
    assert all(spec.data_sidecars == ("csv", "json") for spec in resolved.specs)
    assert all(spec.caption and spec.alt_text for spec in resolved.specs)
    changed = copy.deepcopy(raw)
    changed["rendering"]["backend"] = "interactive"
    with pytest.raises(ValueError):
        resolve_figure_config(changed)
    changed = copy.deepcopy(raw)
    changed["specs"][0]["unknown"] = True
    with pytest.raises(ValueError):
        resolve_figure_config(changed)
