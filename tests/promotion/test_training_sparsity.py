from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch

from vfe4.artifacts.manifest import (
    ArtifactIntegrityRecord,
    ClosedManifestIdentity,
)
from vfe4.config import (
    default_training_config_mapping,
    resolve_training_config,
)
from vfe4.training.factories import WT103FactorySetIdentity
from vfe4.training.formulas import (
    A0FlopWorkload,
    build_a0_architecture_profile,
    build_a0_formula_record,
    reconstruct_a0_flops,
    reconstruct_a0_parameters,
)
from vfe4.training.readiness import (
    PredictorPerturbationObservation,
    StaticScientificArtifactRef,
    certify_wt103_predictor_safety,
    validate_static_scientific_preconditions,
)
from vfe4.training.sparsity import (
    ArmPathTrace,
    FlashAttentionObservation,
    ForbiddenStorageRequest,
    TensorStorageObservation,
    certify_training_sparsity,
    guard_tensor_request,
    run_sparsity_negative_controls,
)
from vfe4.training.wt103_models import WT103A0Model
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    ProductionTokenizerSpec,
    SyntheticFixtureTokenCacheIdentity,
    SyntheticFixtureTokenizerSpec,
    production_tokenizer_tables_sha256,
)


_HEAD = "a" * 40
_DIRTY = "b" * 64
_HASH = "c" * 64


def _resolved():
    return resolve_training_config(default_training_config_mapping())


def _architecture():
    resolved = _resolved()
    model = WT103A0Model(
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
        attention_heads=2,
        layer_norm_epsilon=1.0e-5,
        device=torch.device("meta"),
        dtype=torch.float32,
    )
    inventory = reconstruct_a0_parameters(
        model,
        vocabulary_size=50_257,
        positional_capacity=128,
        hidden_width=20,
    )
    ledger = reconstruct_a0_flops(
        A0FlopWorkload(
            batch_size=1,
            sequence_length=2,
            vocabulary_size=50_257,
            hidden_width=20,
            parameter_count=inventory.parameter_count,
            decoder_chunk_size=2,
            optimizer_steps=1,
            validation_batches=0,
        )
    )
    formula = build_a0_formula_record(inventory=inventory, ledger=ledger)
    architecture = build_a0_architecture_profile(
        hidden_width=20,
        formula=formula,
        source_lock_scope="candidate_unverified",
        pytorch_version="unresolved_until_task13_source_lock",
        sdpa_api_sha256="unresolved_until_task13_source_lock",
        flash_backend_sha256="unresolved_until_task13_source_lock",
    )
    factory_set = WT103FactorySetIdentity.bind(
        arm_specs=resolved.endpoint_inventory.arms,
        constructor_ids=(
            "build_wt103_a0",
            "build_wt103_a5_parent_specific",
            "build_wt103_a5_fixed",
            "build_wt103_a5_parent_specific",
            "build_wt103_a5_nolatent",
        ),
        build_sha256s=tuple(str(index) * 64 for index in range(1, 6)),
    )
    return resolved, architecture, formula, factory_set


def _storage(
    event_id: str,
    arm_id: str,
    path_event: str,
    storage_class: str,
    shape: tuple[int, ...],
    *,
    phase: str = "train",
) -> TensorStorageObservation:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return TensorStorageObservation(
        event_id=event_id,
        arm_id=arm_id,
        path_event=path_event,
        phase=phase,
        storage_id=event_id,
        storage_class=storage_class,
        shape=shape,
        logical_axes=tuple(f"axis_{index}" for index in range(len(shape))),
        numel=numel,
        element_size_bytes=4,
        logical_bytes=numel * 4,
        storage_span_bytes=numel * 4,
    )


def _passing_sparsity_audit():
    resolved, architecture, _, factory_set = _architecture()
    profile = resolved.profile
    arms = resolved.endpoint_inventory.arms
    a0_observations = (
        _storage(
            "a0_vocab",
            arms[0].arm_id,
            "forward",
            "vocabulary_parameter",
            (profile.vocabulary_size, architecture.hidden_width),
        ),
        _storage(
            "a0_qkv",
            arms[0].arm_id,
            "forward",
            "a0_qkv_or_result",
            (
                profile.batch_size,
                profile.sequence_length,
                3 * architecture.hidden_width,
            ),
        ),
        _storage(
            "a0_train_logits",
            arms[0].arm_id,
            "cross_entropy",
            "decoder_chunk",
            (profile.decoder_train_token_chunk, profile.vocabulary_size),
        ),
        _storage(
            "a0_eval_logits",
            arms[0].arm_id,
            "exact_autoregressive_scorer",
            "decoder_chunk",
            (profile.decoder_eval_token_chunk, profile.vocabulary_size),
            phase="evaluation",
        ),
    )
    latent_observations = (
        _storage(
            "latent_mean",
            arms[1].arm_id,
            "forward",
            "latent_block",
            (profile.batch_size, profile.sequence_length, profile.combined_latent_block),
        ),
        _storage(
            "latent_diag",
            arms[1].arm_id,
            "recognition_adam_proposal",
            "latent_block",
            (
                profile.batch_size,
                profile.sequence_length,
                profile.combined_latent_block,
                profile.combined_latent_block,
            ),
        ),
        _storage(
            "latent_lower",
            arms[1].arm_id,
            "recognition_adam_proposal",
            "lower_adjacent_block",
            (
                profile.batch_size,
                profile.sequence_length - 1,
                profile.combined_latent_block,
                profile.combined_latent_block,
            ),
        ),
        _storage(
            "source_band",
            arms[1].arm_id,
            "forward",
            "banded_source",
            (
                profile.batch_size,
                profile.sequence_length,
                profile.source_lookback,
            ),
        ),
        _storage(
            "frames",
            arms[1].arm_id,
            "forward",
            "primary_frame",
            (profile.sequence_length, profile.K, profile.K),
        ),
        _storage(
            "particle_chunk",
            arms[1].arm_id,
            "weighted_smc_scorer",
            "particle_chunk",
            (
                profile.smc_particle_chunk,
                profile.batch_size,
                profile.combined_latent_block,
            ),
            phase="evaluation",
        ),
    )
    traces = tuple(
        ArmPathTrace.exact_for_arm(
            arm,
            observations=(
                a0_observations
                if index == 0
                else latent_observations
                if index == 1
                else (
                    _storage(
                        f"scalar_{index}",
                        arm.arm_id,
                        "metric_failure_write",
                        "scalar_or_row",
                        (1,),
                        phase="evaluation",
                    ),
                )
            ),
        )
        for index, arm in enumerate(arms)
    )
    controls = run_sparsity_negative_controls(
        profile=profile,
        architecture=architecture,
    )
    unique_bytes = sum(
        observation.storage_span_bytes
        for trace in traces
        for observation in trace.observations
    )
    return certify_training_sparsity(
        git_head=_HEAD,
        dirty_digest=_DIRTY,
        profile=profile,
        architecture=architecture,
        factory_set=factory_set,
        endpoint_inventory=resolved.endpoint_inventory,
        traces=traces,
        flash_attention=FlashAttentionObservation.exact(
            arm_id=arms[0].arm_id,
            sequence_length=profile.sequence_length,
        ),
        negative_controls=controls,
        allocator_allocated_bytes=unique_bytes + 4_096,
        allocator_overhead_bytes=4_096,
        profiler_dispatch_agree=True,
        serializer_inventory_complete=True,
        h8_evidence=None,
        capacity_evidence=None,
    )


def test_training_sparsity_is_revision_bound_reconciled_and_fail_closed() -> None:
    audit = _passing_sparsity_audit()
    assert audit.certificate.status is GateStatus.PASS
    assert audit.certificate.git_head == _HEAD
    assert audit.allocator_allocated_bytes == (
        audit.classified_unique_storage_bytes + audit.allocator_overhead_bytes
    )
    assert tuple(trace.arm_id for trace in audit.traces) == tuple(
        arm.arm_id for arm in _resolved().endpoint_inventory.arms
    )
    assert all(
        set(trace.observability_views)
        == {
            "dispatch",
            "profiler",
            "cuda_allocator_unique_storage",
            "backend_checkpoint_inventory",
        }
        for trace in audit.traces
    )
    assert len(audit.negative_controls) == 10
    assert all(
        control.fired_pre_allocation
        and not control.allocation_or_serialization_attempted
        for control in audit.negative_controls
    )

    resolved, architecture, _, factory_set = _architecture()
    forbidden = _storage(
        "forbidden_pair",
        resolved.endpoint_inventory.arms[0].arm_id,
        "forward",
        "banded_source",
        (
            resolved.profile.batch_size,
            resolved.profile.sequence_length,
            resolved.profile.sequence_length,
        ),
    )
    bad_traces = tuple(
        dataclasses.replace(
            trace,
            observations=(forbidden,)
            if index == 0
            else trace.observations,
        )
        for index, trace in enumerate(audit.traces)
    )
    bad_bytes = sum(
        observation.storage_span_bytes
        for trace in bad_traces
        for observation in trace.observations
    )
    failed = certify_training_sparsity(
        git_head=_HEAD,
        dirty_digest=_DIRTY,
        profile=resolved.profile,
        architecture=architecture,
        factory_set=factory_set,
        endpoint_inventory=resolved.endpoint_inventory,
        traces=bad_traces,
        flash_attention=audit.flash_attention,
        negative_controls=audit.negative_controls,
        allocator_allocated_bytes=bad_bytes,
        allocator_overhead_bytes=0,
        profiler_dispatch_agree=True,
        serializer_inventory_complete=True,
        h8_evidence=None,
        capacity_evidence=None,
    )
    assert failed.certificate.status is GateStatus.FAIL
    assert any("forbidden" in item for item in failed.certificate.obligations)

    with pytest.raises(ForbiddenStorageRequest):
        guard_tensor_request(
            profile=resolved.profile,
            architecture=architecture,
            storage_class="decoder_chunk",
            shape=(
                resolved.profile.batch_size
                * resolved.profile.sequence_length,
                resolved.profile.vocabulary_size,
            ),
            phase="train",
        )
    with pytest.raises(ValueError, match="H8"):
        certify_training_sparsity(
            git_head=_HEAD,
            dirty_digest=_DIRTY,
            profile=resolved.profile,
            architecture=architecture,
            factory_set=factory_set,
            endpoint_inventory=resolved.endpoint_inventory,
            traces=audit.traces,
            flash_attention=audit.flash_attention,
            negative_controls=audit.negative_controls,
            allocator_allocated_bytes=audit.allocator_allocated_bytes,
            allocator_overhead_bytes=audit.allocator_overhead_bytes,
            profiler_dispatch_agree=True,
            serializer_inventory_complete=True,
            h8_evidence=object(),
            capacity_evidence=None,
        )


class _SafePriorPredictor:
    def next_token_log_probs(
        self,
        prefix_tokens,
        estimator_rng,
        cache=None,
    ):
        return prefix_tokens, estimator_rng, cache


def _closed_manifest(kind: str, result_sha256: str) -> ClosedManifestIdentity:
    result = ArtifactIntegrityRecord.create(
        kind="file",
        relative_path=f"{kind}/result.json",
        size_bytes=2,
        sha256=result_sha256,
    )
    manifest = ArtifactIntegrityRecord.create(
        kind="manifest",
        relative_path=f"{kind}/manifest.sha256",
        size_bytes=64,
        sha256=_HASH,
    )
    return ClosedManifestIdentity.create(
        manifest=manifest,
        entries=(result,),
    )


def _artifact_ref(kind: str, schema_version: str) -> StaticScientificArtifactRef:
    result_sha = _HASH
    return StaticScientificArtifactRef.create(
        kind=kind,
        result_schema=schema_version,
        git_head=_HEAD,
        dirty_digest=_DIRTY,
        result_sha256=result_sha,
        manifest=_closed_manifest(kind, result_sha),
        status=GateStatus.PASS,
    )


def test_static_preconditions_accept_only_native_current_nontransfer_evidence() -> None:
    tokenizer = SyntheticFixtureTokenizerSpec.create(
        adapter_sha256="1" * 64,
        fixture_sha256="2" * 64,
    )
    cache = SyntheticFixtureTokenCacheIdentity.create(
        tokenizer=tokenizer,
        payload_sha256="3" * 64,
    )
    prefix = (17, 23)
    prediction_sha = "4" * 64
    cache_key_sha = "5" * 64
    observations = tuple(
        PredictorPerturbationObservation(
            prefix_tokens=prefix,
            current_target=target,
            suffix_tokens=suffix,
            cache_traversal=traversal,
            prediction_sha256=prediction_sha,
            cache_key_sha256=cache_key_sha,
        )
        for target, suffix, traversal in itertools.product(
            (0, 1, 50_256),
            ((), (7,), (50_256, 9)),
            ("cold", "forward", "reverse"),
        )
    )
    safety = certify_wt103_predictor_safety(
        predictor_type=_SafePriorPredictor,
        tokenizer=tokenizer,
        cache=cache,
        observations=observations,
    )
    assert safety.status is GateStatus.PASS
    assert safety.vocabulary_size == 50_257
    assert safety.production_token_authorized is False
    assert not hasattr(safety, "readiness_token")

    production_table_facts = {
        "regex_pattern_sha256": "7" * 64,
        "regex_engine_distribution_name": "regex",
        "regex_engine_distribution_version": "2026.1.1",
        "regex_engine_distribution_record_sha256": "8" * 64,
        "mergeable_ranks_sha256": "8" * 64,
        "special_tokens_sha256": "9" * 64,
        "golden_vectors_sha256": "a" * 64,
    }
    production = ProductionTokenizerSpec.create_verified(
        distribution_record_sha256="6" * 64,
        **production_table_facts,
        tokenizer_tables_sha256=production_tokenizer_tables_sha256(
            **production_table_facts
        ),
    )
    with pytest.raises(ValueError, match="synthetic"):
        certify_wt103_predictor_safety(
            predictor_type=_SafePriorPredictor,
            tokenizer=production,
            cache=cache,
            observations=observations,
        )

    resolved, architecture, formula, factory_set = _architecture()
    sparsity = _passing_sparsity_audit().certificate
    references = (
        _artifact_ref("h5", "h5-update-result-v1"),
        _artifact_ref("h6_prefix", "h6-prefix-certificate-set-v2"),
        _artifact_ref("h6_prediction", "h6-prediction-result-v3"),
        _artifact_ref("h7", "h7-gate-result-v1"),
        _artifact_ref("h8", "h8-sparse-scale-v5"),
    )
    record = validate_static_scientific_preconditions(
        profile=resolved.profile,
        scientific_profile=resolved.scientific_preconditions,
        architecture=architecture,
        formula=formula,
        factory_set=factory_set,
        endpoint_inventory=resolved.endpoint_inventory,
        objective_sha256="d" * 64,
        update_policy_sha256="e" * 64,
        snapshot_policy_sha256="f" * 64,
        estimator_protocol_sha256=(
            resolved.endpoint_inventory.estimator_protocol_sha256
        ),
        predecessor_references=references,
        predictor_safety=safety,
        training_sparsity=sparsity,
        h6_byte_evidence=None,
        h8_allocation_evidence=None,
        capacity_evidence=None,
    )
    assert record.status is GateStatus.PASS
    assert record.h6_prediction_schema == "h6-prediction-result-v3"
    assert record.h8_schema == "h8-sparse-scale-v5"
    assert record.production_readiness_token_issued is False
    assert not hasattr(record, "post_h8_readiness_token")

    with pytest.raises(ValueError, match="H6 byte"):
        validate_static_scientific_preconditions(
            profile=resolved.profile,
            scientific_profile=resolved.scientific_preconditions,
            architecture=architecture,
            formula=formula,
            factory_set=factory_set,
            endpoint_inventory=resolved.endpoint_inventory,
            objective_sha256="d" * 64,
            update_policy_sha256="e" * 64,
            snapshot_policy_sha256="f" * 64,
            estimator_protocol_sha256=(
                resolved.endpoint_inventory.estimator_protocol_sha256
            ),
            predecessor_references=references,
            predictor_safety=safety,
            training_sparsity=sparsity,
            h6_byte_evidence=object(),
            h8_allocation_evidence=None,
            capacity_evidence=None,
        )
    with pytest.raises(ValueError):
        _artifact_ref("h6_prediction", "h6-prediction-result-v2")
