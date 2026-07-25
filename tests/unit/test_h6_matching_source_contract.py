from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from vfe4.training.matching import (
    A5_REFERENCE_ALLOCATION,
    AMENDED_MATCHING_SCHEDULE_POLICY,
    H6TrainingWorkload,
    analytical_training_flop_ledger,
    dense_matmul_flops,
    endpoint_formula_profile,
    matrix_exp_pade13_flops,
    matrix_solve_lu_flops,
    select_outcome_blind_allocation,
    stable_parameter_key,
)
from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    CapacityAllocation,
    OptimizerBinding,
    ParameterRoleRecord,
    TrainingPhase,
    VocabularyIdentity,
)


SHA_A = "a" * 64


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity(
        vocabulary_id="h6-static-contract-v1",
        size=258,
        tokenizer_spec_sha256="b" * 64,
    )


def _config(
    config_id: str,
    allocation: CapacityAllocation,
) -> ArmConfig:
    profile = endpoint_formula_profile(config_id)
    return ArmConfig.create(
        arm=ArmId(profile.arm),
        config_id=config_id,
        vocabulary=_vocabulary(),
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


def test_h6_parameter_ownership_keys_are_stable_serializable_names() -> None:
    phase = TrainingPhase.MODEL_ADAMW.value
    qualified_name = "model.normalized_emission_head"
    key = stable_parameter_key(
        qualified_name=qualified_name,
        phase=phase,
    )
    assert key == stable_parameter_key(
        qualified_name=qualified_name,
        phase=phase,
    )
    assert len(key) == 64
    assert "parameter_id" not in inspect.signature(
        ParameterRoleRecord.create
    ).parameters
    assert "parameter_ids" not in inspect.signature(
        OptimizerBinding.create
    ).parameters

    role = ParameterRoleRecord.create(
        qualified_name=qualified_name,
        parameter_key=key,
        role="normalized_emission",
        phase=phase,
        scalar_count=12,
    )
    binding = OptimizerBinding.create(
        phase=phase,
        optimizer_class="AdamW",
        optimizer_policy_sha256=SHA_A,
        parameter_keys=(key,),
    )
    assert role.parameter_key == binding.parameter_keys[0]
    with pytest.raises(ValueError, match="parameter_key"):
        ParameterRoleRecord.create(
            qualified_name=qualified_name,
            parameter_key="process-local-address",
            role="normalized_emission",
            phase=phase,
            scalar_count=12,
        )


def test_h6_workload_binds_every_batch_and_both_frozen_passes() -> None:
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=258,
        train_token_sha256=SHA_A,
    )
    assert workload.sequence_length == 32
    assert workload.window_stride == 32
    assert workload.batch_size == 8
    assert workload.full_passes == 2
    assert workload.window_count == 9
    assert workload.full_batches_per_pass == 1
    assert workload.tail_batch_size == 1
    assert workload.batches_per_pass == 2
    assert workload.model_update_opportunities == 4
    assert workload.validation_boundaries_per_pass == (1, 2)
    assert (
        workload.matching_schedule_policy_sha256
        == AMENDED_MATCHING_SCHEDULE_POLICY.policy_sha256
    )


def test_h6_analytical_ledger_is_operator_complete_and_workload_scaled() -> None:
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=258,
        train_token_sha256=SHA_A,
    )
    allocation = CapacityAllocation.create(
        emission_width=64,
        latent_width=16,
        recognition_width=64,
    )
    profile = endpoint_formula_profile(
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
    )
    ledger = analytical_training_flop_ledger(
        endpoint_config=_config(profile.config_id, allocation),
        workload=workload,
    )
    assert ledger.status == "COMPLETE"
    assert ledger.obligations == ()
    assert ledger.total_arithmetic_flops == sum(
        term.total_arithmetic_flops for term in ledger.terms
    )
    assert ledger.total_bytes_copied == sum(
        term.total_bytes_copied for term in ledger.terms
    )
    assert all(
        not term.operation.startswith("INCOMPLETE_")
        for term in ledger.terms
    )
    assert {
        term.phase for term in ledger.terms
    } == {
        TrainingPhase.RECOGNITION_ADAMW.value,
        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
        TrainingPhase.MODEL_ADAMW.value,
    }
    operations = {term.operation for term in ledger.terms}
    assert any("matrix_exp_pade13" in item for item in operations)
    assert any("source_inverse_lu" in item for item in operations)
    assert any("full_batch" in item for item in operations)
    assert any("tail_batch" in item for item in operations)
    assert ledger.workload_sha256 == workload.workload_sha256


def test_h6_live_coboundary_map_call_multiplicities_are_counted_exactly() -> None:
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=32,
        train_token_sha256=SHA_A,
    )
    config = _config(
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        A5_REFERENCE_ALLOCATION,
    )
    ledger = analytical_training_flop_ledger(
        endpoint_config=config,
        workload=workload,
    )
    edge_count = 32 * 33 // 2
    channels = 2
    dimension = 16
    operation_costs = {
        term.operation.split("::")[1]: term.arithmetic_flops_per_repetition
        for term in ledger.terms
        if term.phase == TrainingPhase.MODEL_ADAMW.value
        and term.operation.startswith("forward::")
        and term.operation.endswith("::tail_batch")
    }
    assert operation_costs[
        "shared_coboundary_graph_cached_matrix_exp_pade13"
    ] == channels * 32 * matrix_exp_pade13_flops(dimension)
    assert operation_costs[
        "shared_coboundary_graph_cached_source_inverse_lu"
    ] == channels * 32 * matrix_solve_lu_flops(dimension)
    assert operation_costs[
        "shared_coboundary_edge_frame_product_dense_matmul"
    ] == channels * edge_count * dense_matmul_flops(
        dimension, dimension, dimension
    )

    generic = analytical_training_flop_ledger(
        endpoint_config=_config(
            "h6-a2-generic-map-v1",
            A5_REFERENCE_ALLOCATION,
        ),
        workload=workload,
    )
    assert not any(
        "coboundary" in term.operation for term in generic.terms
    )


def test_h6_formula_selection_derives_all_identity_cross_fields() -> None:
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=258,
        train_token_sha256=SHA_A,
    )
    reference = _config(
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        A5_REFERENCE_ALLOCATION,
    )
    selection = select_outcome_blind_allocation(
        endpoint_template=reference,
        reference_config=reference,
        workload=workload,
    )
    assert selection.endpoint_template_config_sha256 == reference.config_sha256
    assert selection.reference_config_sha256 == reference.config_sha256
    assert selection.workload_sha256 == workload.workload_sha256
    assert (
        selection.reference_ledger.endpoint_config_sha256
        == reference.config_sha256
    )
    assert (
        selection.reference_ledger.endpoint_profile_sha256
        == selection.reference_profile_sha256
    )
    assert (
        "endpoint_config_sha256"
        not in inspect.signature(select_outcome_blind_allocation).parameters
    )
    with pytest.raises(ValueError, match="profile"):
        replace(selection, endpoint_profile_sha256="0" * 64)


def test_h6_amended_matching_policy_is_outcome_blind_and_frozen() -> None:
    policy = AMENDED_MATCHING_SCHEDULE_POLICY
    assert policy.emission_width_candidates == (48, 64, 80, 96, 123)
    assert policy.latent_width_candidates == (2, 8, 16, 24, 32)
    assert policy.recognition_width_candidates == (32, 64, 96)
    assert policy.prior_context_width == 6
    assert policy.selection_rule == "first_lexicographic_hard_eligible"
    assert policy.forbidden_inputs == (
        "corpus_bytes",
        "loss",
        "gradients",
        "validation_metrics",
        "test_metrics",
        "prediction_flops",
    )
