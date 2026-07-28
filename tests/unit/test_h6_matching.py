from __future__ import annotations

from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest
import torch

from vfe4.config import (
    H6ArmMatchingResolvedConfig,
    resolve_h6_arm_matching_config,
    resolve_h6_primary_matching_config,
)
from vfe4.training.arms import (
    ArmConfig,
    CapacityAllocation,
    MatchingReport,
    OptimizerBinding,
    build_a0,
    build_a1,
    build_a2,
    build_a3,
    build_a4,
    build_a5,
)
from vfe4.training.matching import (
    A5_REFERENCE_ALLOCATION,
    H6_ADAMW_POLICY,
    H6TrainingWorkload,
    MATCHING_SCHEDULE_POLICY,
    FlopTerm,
    adamw_flops,
    analytical_training_flop_ledger,
    audit_arm_matching,
    audit_parameter_ownership,
    backward_flops,
    candidate_allocations,
    capacity_candidate_count,
    dense_matmul_flops,
    dense_matvec_flops,
    endpoint_formula_profile,
    immutable_snapshot_flop_term,
    l2_clip_scale_flops,
    log_softmax_flops,
    matrix_exp_pade13_flops,
    matrix_solve_lu_flops,
    scalar_flops,
    select_parent_specific_primary_allocation,
    stable_parameter_key,
)
from vfe4.types import ArmId, TrainingPhase, VocabularyIdentity


SHA_A = "a" * 64


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity("h6-task7-small-v1", 3, SHA_A)


def _allocation(
    *,
    latent: bool,
    recognition: bool | None = None,
    emission_width: int = 64,
) -> CapacityAllocation:
    recognition_enabled = latent if recognition is None else recognition
    return CapacityAllocation.create(
        emission_width=emission_width,
        latent_width=16 if latent else None,
        recognition_width=64 if recognition_enabled else None,
    )


def _config(arm: ArmId, *, emission_width: int = 64) -> ArmConfig:
    if arm is ArmId.A0:
        return ArmConfig.create(
            arm=arm,
            config_id="h6-a0-transformer-v2",
            vocabulary=_vocabulary(),
            horizon=4,
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
            capacity_allocation=_allocation(
                latent=False,
                emission_width=(
                    48 if emission_width == 64 else emission_width
                ),
            ),
        )
    return ArmConfig.create(
        arm=arm,
        config_id={
            ArmId.A1: "h6-a1-ordinary-latent-v1",
            ArmId.A2: "h6-a2-generic-map-v1",
            ArmId.A3: "h6-a3-immediate-predecessor-v1",
            ArmId.A4: "h6-a4-state-only-v1",
            ArmId.A5: (
                "h6-a5-structured-fixed-exact-complete-"
                "latent-smoothing-v1"
            ),
        }[arm],
        vocabulary=_vocabulary(),
        horizon=2,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=arm not in (ArmId.A1, ArmId.A4),
        source_mode={
            ArmId.A1: "absent",
            ArmId.A2: "categorical",
            ArmId.A3: "immediate_predecessor",
            ArmId.A4: "categorical",
            ArmId.A5: "categorical",
        }[arm],
        map_mode={
            ArmId.A1: "absent",
            ArmId.A2: "generic_fixed_frame_non_coboundary",
            ArmId.A3: "shared_vertex_coboundary",
            ArmId.A4: "shared_vertex_coboundary",
            ArmId.A5: "shared_vertex_coboundary",
        }[arm],
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant=(
            "fixed"
            if arm in (ArmId.A2, ArmId.A4, ArmId.A5)
            else "absent"
        ),
        mixture_mode=(
            "exact"
            if arm in (ArmId.A2, ArmId.A4, ArmId.A5)
            else "absent"
        ),
        objective_kind="complete_elbo",
        capacity_allocation=_allocation(
            latent=True,
            emission_width=emission_width,
        ),
    )


def _no_latent_config() -> ArmConfig:
    return ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            "h6-a5-structured-fixed-exact-complete-"
            "nolatent-norecognition-v1"
        ),
        vocabulary=_vocabulary(),
        horizon=2,
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
        capacity_allocation=_allocation(latent=False),
    )


def _matching_config() -> H6ArmMatchingResolvedConfig:
    arm_configs = tuple(_config(arm) for arm in ArmId)
    raw = {
        "schema_version": "h6-arm-matching-config-v1",
        "operation": "H6-Arm-Matching",
        "arm_configs": arm_configs,
        "adamw_policy": H6_ADAMW_POLICY,
        "reference_allocation": A5_REFERENCE_ALLOCATION,
        "emission_width_candidates": (48, 64, 80, 96),
        "latent_width_candidates": (8, 16, 24, 32),
        "recognition_width_candidates": (32, 64, 96),
        "parameter_relative_tolerance": 0.01,
        "flop_relative_tolerance": 0.05,
        "matching_schedule_sha256": MATCHING_SCHEDULE_POLICY.policy_sha256,
        "expected_arm_config_sha256": tuple(
            config.config_sha256 for config in arm_configs
        ),
        "expected_optimizer_policy_sha256": (
            H6_ADAMW_POLICY.optimizer_policy_sha256
        ),
        "expected_reference_allocation_sha256": (
            A5_REFERENCE_ALLOCATION.allocation_sha256
        ),
    }
    return resolve_h6_arm_matching_config(raw, repo_root=Path.cwd())


def test_candidate_search_is_lazy_bounded_and_lexicographic() -> None:
    full = _config(ArmId.A2, emission_width=48)
    a0 = _config(ArmId.A0, emission_width=48)
    a3 = _config(ArmId.A3, emission_width=48)
    no_latent = _no_latent_config()
    matching_config = _matching_config()
    no_latent_profile = endpoint_formula_profile(no_latent.config_id)

    assert no_latent_profile.objective_kind == no_latent.objective_kind
    assert no_latent.objective_kind == "cross_entropy"
    assert (
        A5_REFERENCE_ALLOCATION.emission_width,
        A5_REFERENCE_ALLOCATION.latent_width,
        A5_REFERENCE_ALLOCATION.recognition_width,
    ) == (64, 16, 64)
    assert capacity_candidate_count(
        full, matching_config=matching_config
    ) == 4 * 4 * 3 == 48
    assert capacity_candidate_count(
        a3, matching_config=matching_config
    ) == 4 * 4 * 3 == 48
    assert capacity_candidate_count(
        a0, matching_config=matching_config
    ) == 4
    assert capacity_candidate_count(
        no_latent, matching_config=matching_config
    ) == 4

    candidates = candidate_allocations(
        full, matching_config=matching_config
    )
    assert iter(candidates) is candidates
    first = tuple(islice(candidates, 4))
    assert tuple(
        (
            item.emission_width,
            item.latent_width,
            item.recognition_width,
        )
        for item in first
    ) == (
        (48, 8, 32),
        (48, 8, 64),
        (48, 8, 96),
        (48, 16, 32),
    )
    assert tuple(
        item.emission_width
        for item in candidate_allocations(
            a0, matching_config=matching_config
        )
    ) == (48, 64, 80, 96)


def test_parameter_ownership_is_exact_and_rejects_every_forbidden_case() -> None:
    arm = build_a5(_config(ArmId.A5))
    assert audit_parameter_ownership(arm) is None

    active = {
        **{
            f"model.{name}": parameter
            for name, parameter in arm.model.named_parameters()
        },
        **{
            f"recognition_store.{name}": parameter
            for name, parameter in arm.recognition_store.named_parameters()
        },
    }
    active_keys = {
        stable_parameter_key(
            qualified_name=name,
            phase=(
                TrainingPhase.RECOGNITION_ADAMW.value
                if name.startswith("recognition_store.")
                else TrainingPhase.MODEL_ADAMW.value
            ),
        )
        for name in active
    }
    assert active_keys == {
        record.parameter_key for record in arm.parameter_roles
    }
    assert all(
        record.scalar_count
        == active[record.qualified_name].numel()
        for record in arm.parameter_roles
    )
    bound_keys = [
        parameter_key
        for binding in arm.optimizer_bindings
        for parameter_key in binding.parameter_keys
    ]
    assert len(bound_keys) == len(set(bound_keys))
    assert set(bound_keys) == active_keys

    model_binding = next(
        binding
        for binding in arm.optimizer_bindings
        if binding.phase == TrainingPhase.MODEL_ADAMW.value
    )
    recognition_binding = next(
        binding
        for binding in arm.optimizer_bindings
        if binding.phase == TrainingPhase.RECOGNITION_ADAMW.value
    )
    removed_key = model_binding.parameter_keys[0]
    shortened = OptimizerBinding.create(
        phase=model_binding.phase,
        optimizer_class="AdamW",
        optimizer_policy_sha256=model_binding.optimizer_policy_sha256,
        parameter_keys=model_binding.parameter_keys[1:],
    )
    unbound = replace(
        arm,
        optimizer_bindings=tuple(
            shortened if binding is model_binding else binding
            for binding in arm.optimizer_bindings
        ),
    )
    with pytest.raises(ValueError, match=f"unbound|{removed_key}"):
        audit_parameter_ownership(unbound)

    duplicated = OptimizerBinding.create(
        phase=recognition_binding.phase,
        optimizer_class="AdamW",
        optimizer_policy_sha256=recognition_binding.optimizer_policy_sha256,
        parameter_keys=recognition_binding.parameter_keys + (removed_key,),
    )
    duplicate = replace(
        arm,
        optimizer_bindings=tuple(
            duplicated if binding is recognition_binding else binding
            for binding in arm.optimizer_bindings
        ),
    )
    with pytest.raises(ValueError, match="duplicate|more than one"):
        audit_parameter_ownership(duplicate)

    alias_arm = build_a5(_config(ArmId.A5))
    alias_arm.model.register_parameter(
        "duplicate_alias", next(alias_arm.model.parameters())
    )
    with pytest.raises(ValueError, match="more than one|duplicate"):
        audit_parameter_ownership(alias_arm)

    filler_arm = build_a5(_config(ArmId.A5))
    filler_arm.model.register_parameter(
        "frozen_filler",
        torch.nn.Parameter(torch.zeros(1), requires_grad=False),
    )
    with pytest.raises(ValueError, match="filler|frozen|dormant"):
        audit_parameter_ownership(filler_arm)

    dormant_arm = build_a5(_config(ArmId.A5))
    dormant_arm.model.register_parameter(
        "dormant_parameter",
        torch.nn.Parameter(torch.zeros(1), requires_grad=True),
    )
    with pytest.raises(ValueError, match="dormant|unbound"):
        audit_parameter_ownership(dormant_arm)

    with pytest.raises(ValueError, match="phase|snapshot|optimizer"):
        OptimizerBinding.create(
            phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
            optimizer_class="AdamW",
            optimizer_policy_sha256=model_binding.optimizer_policy_sha256,
            parameter_keys=(removed_key,),
        )


def test_shared_frame_cache_preserves_values_and_gradients() -> None:
    base = _config(ArmId.A5)
    config = ArmConfig.create(
        arm=base.arm,
        config_id=base.config_id,
        vocabulary=base.vocabulary,
        horizon=3,
        latent_enabled=base.latent_enabled,
        state_channel_enabled=base.state_channel_enabled,
        model_channel_enabled=base.model_channel_enabled,
        source_mode=base.source_mode,
        map_mode=base.map_mode,
        recognition_family=base.recognition_family,
        recognition_conditioning=base.recognition_conditioning,
        prior_variant=base.prior_variant,
        mixture_mode=base.mixture_mode,
        objective_kind=base.objective_kind,
        capacity_allocation=base.capacity_allocation,
    )
    model = build_a5(config).model
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    source_support = model.source_prior.structure.dag.rows[2].parents

    def evaluate(frame_cache=None):
        model.zero_grad(set_to_none=True)
        values = torch.stack(
            (
                model.edge_map("state", 2, 0, frame_cache=frame_cache),
                model.edge_map("state", 2, 1, frame_cache=frame_cache),
                model.edge_map("state", 3, 0, frame_cache=frame_cache),
                model.edge_map("state", 3, 2, frame_cache=frame_cache),
                model.edge_map("model", 2, 0, frame_cache=frame_cache),
                model.edge_map("model", 2, 1, frame_cache=frame_cache),
                model.edge_map("model", 3, 0, frame_cache=frame_cache),
                model.edge_map("model", 3, 2, frame_cache=frame_cache),
            )
        )
        values.square().sum().backward()
        gradients = tuple(
            parameter.grad.detach().clone()
            for parameter in (
                tuple(model.state_vertex_phi)
                + tuple(model.model_vertex_phi)
            )
        )
        return values.detach().clone(), gradients

    uncached_values, uncached_gradients = evaluate()
    with model.shared_frame_evaluation() as cache:
        cached_values, cached_gradients = evaluate(cache)
        frame_count = cache.frame_count
        source_inverse_count = cache.source_inverse_count

    assert torch.equal(cached_values, uncached_values)
    assert all(
        torch.equal(cached, uncached)
        for cached, uncached in zip(
            cached_gradients, uncached_gradients, strict=True
        )
    )
    assert frame_count == 8
    assert source_inverse_count == 6
    assert model.source_prior.structure.dag.rows[2].parents == source_support
    assert sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) == parameter_bytes
    with pytest.raises(ValueError, match="closed"):
        model.edge_map(
            "state",
            2,
            0,
            frame_cache=cache,
        )

    with model.shared_frame_evaluation() as update_cache:
        before_update = model.edge_map(
            "state",
            2,
            0,
            frame_cache=update_cache,
        )
        detached_snapshot = before_update.detach().clone()
        with torch.no_grad():
            model.state_vertex_phi[1].add_(0.125)
        with pytest.raises(ValueError, match="parameters changed"):
            model.edge_map(
                "state",
                2,
                0,
                frame_cache=update_cache,
            )
    with model.shared_frame_evaluation() as fresh_cache:
        after_update = model.edge_map(
            "state",
            2,
            0,
            frame_cache=fresh_cache,
        )
        assert after_update is not before_update
        assert not torch.equal(after_update, detached_snapshot)

    state_only = build_a4(_config(ArmId.A4)).model
    with state_only.shared_frame_evaluation() as state_only_cache:
        state_only.edge_map(
            "state",
            2,
            0,
            frame_cache=state_only_cache,
        )


def test_parent_specific_primary_joint_search_closes_both_gates_or_fails_closed() -> None:
    vocabulary = VocabularyIdentity("wikitext-2-byte-v1", 258, SHA_A)
    a0 = ArmConfig.create(
        arm=ArmId.A0,
        config_id="h6-a0-transformer-v2",
        vocabulary=vocabulary,
        horizon=32,
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
            emission_width=52,
            latent_width=None,
            recognition_width=None,
        ),
    )
    a5 = ArmConfig.create(
        arm=ArmId.A5,
        config_id=(
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2"
        ),
        vocabulary=vocabulary,
        horizon=32,
        latent_enabled=True,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="shared_vertex_coboundary",
        recognition_family="structured",
        recognition_conditioning="smoothing",
        prior_variant="parent_specific_pooled_prefix",
        mixture_mode="exact",
        objective_kind="complete_elbo",
        capacity_allocation=CapacityAllocation.create(
            emission_width=89,
            latent_width=2,
            recognition_width=113,
            prior_context_width=6,
        ),
    )
    workload = H6TrainingWorkload.from_train_tokens(
        train_token_count=258,
        train_token_sha256="b" * 64,
    )
    matching_config = resolve_h6_primary_matching_config(
        {
            "schema_version": "h6-primary-matching-config-v1",
            "operation": "H6-Primary-Matching",
            "a0_config": a0,
            "a5_template": a5,
            "latent_width_candidates": (2, 4, 8),
            "prior_context_width_candidates": (4, 6, 8),
            "emission_width_candidates": (84, 85, 86, 87, 88, 89),
            "recognition_width_candidates": (
                113,
                114,
                115,
                116,
                117,
                118,
            ),
            "parameter_relative_tolerance": 0.01,
            "flop_relative_tolerance": 0.05,
        },
        repo_root=Path.cwd(),
    )
    matching_raw = {
        "schema_version": "h6-primary-matching-config-v1",
        "operation": "H6-Primary-Matching",
        "a0_config": a0,
        "a5_template": a5,
        "latent_width_candidates": (2, 4, 8),
        "prior_context_width_candidates": (4, 6, 8),
        "emission_width_candidates": (84, 85, 86, 87, 88, 89),
        "recognition_width_candidates": (113, 114, 115, 116, 117, 118),
        "parameter_relative_tolerance": 0.01,
        "flop_relative_tolerance": 0.05,
    }

    def clone_endpoint(
        config: ArmConfig,
        *,
        vocabulary: VocabularyIdentity,
        horizon: int,
    ) -> ArmConfig:
        return ArmConfig.create(
            arm=config.arm,
            config_id=config.config_id,
            vocabulary=vocabulary,
            horizon=horizon,
            latent_enabled=config.latent_enabled,
            state_channel_enabled=config.state_channel_enabled,
            model_channel_enabled=config.model_channel_enabled,
            source_mode=config.source_mode,
            map_mode=config.map_mode,
            recognition_family=config.recognition_family,
            recognition_conditioning=config.recognition_conditioning,
            prior_variant=config.prior_variant,
            mixture_mode=config.mixture_mode,
            objective_kind=config.objective_kind,
            capacity_allocation=config.capacity_allocation,
        )

    for impostor_vocabulary, impostor_horizon in (
        (VocabularyIdentity("same-count-impostor-v1", 258, SHA_A), 32),
        (VocabularyIdentity("wikitext-2-byte-v1", 257, SHA_A), 32),
        (vocabulary, 31),
    ):
        with pytest.raises(
            ValueError,
            match="vocabulary_id='wikitext-2-byte-v1', V=258, and T=32",
        ):
            resolve_h6_primary_matching_config(
                {
                    **matching_raw,
                    "a0_config": clone_endpoint(
                        a0,
                        vocabulary=impostor_vocabulary,
                        horizon=impostor_horizon,
                    ),
                    "a5_template": clone_endpoint(
                        a5,
                        vocabulary=impostor_vocabulary,
                        horizon=impostor_horizon,
                    ),
                },
                repo_root=Path.cwd(),
            )

    selection = select_parent_specific_primary_allocation(
        matching_config=matching_config,
        a0_config=a0,
        a5_template=a5,
        workload=workload,
    )

    assert 53 % 2 != 0
    assert len(selection.candidates) == 324
    assert (
        selection.candidates[0].latent_width,
        selection.candidates[0].prior_context_width,
        selection.candidates[0].emission_width,
        selection.candidates[0].recognition_width,
    ) == (2, 4, 84, 113)
    assert (
        selection.candidates[-1].latent_width,
        selection.candidates[-1].prior_context_width,
        selection.candidates[-1].emission_width,
        selection.candidates[-1].recognition_width,
    ) == (8, 8, 89, 118)
    provisional = next(
        candidate
        for candidate in selection.candidates
        if (
            candidate.latent_width,
            candidate.prior_context_width,
            candidate.emission_width,
            candidate.recognition_width,
        )
        == (2, 6, 89, 113)
    )
    assert provisional.a0_parameter_count == 61_982
    assert provisional.a5_parameter_count == 62_112

    a0_ledger = analytical_training_flop_ledger(
        endpoint_config=a0,
        workload=workload,
    )
    a5_ledger = analytical_training_flop_ledger(
        endpoint_config=a5,
        workload=workload,
    )
    assert a0_ledger.status == a5_ledger.status == "COMPLETE"
    assert a0_ledger.obligations == a5_ledger.obligations == ()

    a0_tail_costs = {
        term.operation.split("::")[1]: term.arithmetic_flops_per_repetition
        for term in a0_ledger.terms
        if term.phase == TrainingPhase.MODEL_CE_ADAMW.value
        and term.operation.endswith("::tail_batch")
    }
    causal_pairs = 32 * 33 // 2
    assert a0_tail_costs["a0_sdpa_score_scale"] == 2 * causal_pairs
    assert a0_tail_costs["a0_sdpa_qk_vjp"] == 4 * causal_pairs * 52
    assert a0_tail_costs["a0_sdpa_av_vjp"] == 4 * causal_pairs * 52
    assert a0_tail_costs["a0_sdpa_softmax_vjp"] == (
        2 * (4 * causal_pairs - 32)
    )
    assert a0_tail_costs["a0_embedding_scatter_add_vjp"] == (
        (2 * 32 - 1) * 52
    )

    a5_tail_costs = {
        term.operation.split("::")[1]: term.arithmetic_flops_per_repetition
        for term in a5_ledger.terms
        if term.phase == TrainingPhase.MODEL_ADAMW.value
        and term.operation.startswith("forward::")
        and term.operation.endswith("::tail_batch")
    }
    assert a5_tail_costs[
        "shared_coboundary_graph_cached_matrix_exp_pade13"
    ] == 2 * 32 * matrix_exp_pade13_flops(2)
    assert a5_tail_costs[
        "shared_coboundary_graph_cached_source_inverse_lu"
    ] == 2 * 32 * matrix_solve_lu_flops(2)
    assert a5_tail_costs[
        "shared_coboundary_edge_frame_product_dense_matmul"
    ] == 2 * causal_pairs * dense_matmul_flops(2, 2, 2)

    eligible = tuple(
        candidate
        for candidate in selection.candidates
        if candidate.hard_gate_eligible and candidate.formula_complete
    )
    assert eligible
    expected = min(eligible, key=lambda candidate: candidate.selection_key)
    assert selection.status == "ELIGIBLE"
    assert selection.selected_candidate == expected
    assert selection.obligations == ()
    assert all(candidate.formula_complete for candidate in selection.candidates)
    assert any(
        candidate.hard_gate_eligible and candidate.formula_complete
        for candidate in selection.candidates
    )


def test_flop_ledger_uses_only_the_frozen_arithmetic_formulas() -> None:
    assert dense_matmul_flops(2, 3, 4) == 2 * 2 * 3 * 4 == 48
    assert dense_matvec_flops(3, 4) == 2 * 3 * 4 == 24
    assert scalar_flops(7) == 7
    assert log_softmax_flops(3) == 5 * 3 - 1 == 14
    assert backward_flops(50) == 100
    assert l2_clip_scale_flops(11) == 3 * 11 + 3 == 36
    assert adamw_flops(11) == 18 * 11 == 198

    term = FlopTerm.create(
        phase=TrainingPhase.MODEL_ADAMW.value,
        operation="dense_matmul",
        repetitions=3,
        arithmetic_flops_per_repetition=dense_matmul_flops(2, 3, 4),
        bytes_copied_per_repetition=0,
    )
    assert term.total_arithmetic_flops == 144
    assert len(term.term_sha256) == 64

    snapshot = immutable_snapshot_flop_term(
        repetitions=2,
        bytes_copied_per_repetition=64,
    )
    assert snapshot.phase == TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value
    assert snapshot.arithmetic_flops_per_repetition == 0
    assert snapshot.total_arithmetic_flops == 0
    assert snapshot.bytes_copied_per_repetition == 64

    for helper, args in (
        (dense_matmul_flops, (0, 3, 4)),
        (dense_matvec_flops, (3, 0)),
        (log_softmax_flops, (0,)),
        (backward_flops, (-1,)),
        (l2_clip_scale_flops, (-1,)),
        (adamw_flops, (-1,)),
    ):
        with pytest.raises(ValueError):
            helper(*args)


def test_matching_enforces_hard_tolerances_schedule_and_nuisance_rules() -> None:
    builders = (build_a0, build_a1, build_a2, build_a3, build_a4, build_a5)
    reference = build_a5(_config(ArmId.A5))
    matching_config = _matching_config()

    assert MATCHING_SCHEDULE_POLICY.full_passes == 2
    assert MATCHING_SCHEDULE_POLICY.model_updates_per_batch == 1
    assert MATCHING_SCHEDULE_POLICY.validation_boundary_policy == (
        "twentieths_of_each_pass_v1"
    )
    assert MATCHING_SCHEDULE_POLICY.checkpoint_boundary_policy == (
        "terminal_only_v1"
    )
    assert MATCHING_SCHEDULE_POLICY.excluded_operations == (
        "data_io",
        "validation",
        "checkpoint_serialization",
        "test_scoring",
    )

    for builder, arm_id in zip(builders, ArmId, strict=True):
        endpoint = builder(_config(arm_id))
        report = audit_arm_matching(
            endpoint,
            reference,
            matching_config=matching_config,
            named_factor=(
                "whole_declared_architecture"
                if arm_id is ArmId.A0
                else "arm_semantics"
            ),
            nuisance_capacity_fields=tuple(
                field
                for field in (
                    "emission_width",
                    "latent_width",
                    "recognition_width",
                )
                if getattr(endpoint.config.capacity_allocation, field)
                != getattr(reference.config.capacity_allocation, field)
            ),
        )
        assert report.training_flop_ledger_complete is False
        assert report.status == "INCONCLUSIVE"
        assert any(
            "whole-schedule" in item
            or "batches-times-passes" in item
            for item in report.obligations
        )
        assert report.status in ("ELIGIBLE", "INCONCLUSIVE")
        assert report.eligible is (report.status == "ELIGIBLE")
        if report.eligible:
            assert report.obligations == ()
            assert report.parameter_relative_difference <= 0.01
            assert report.flop_relative_difference <= 0.05
        else:
            assert report.obligations
            assert all(
                "filler" not in record.role.lower()
                and "dormant" not in record.role.lower()
                for record in endpoint.parameter_roles
            )
        assert report.capacity_allocation_policy == (
            "outcome_blind_nuisance_reallocation"
        )
        assert report.common_schedule_sha256 == (
            MATCHING_SCHEDULE_POLICY.policy_sha256
        )

    boundary = MatchingReport.from_totals(
        matching_config_sha256=matching_config.config_sha256,
        endpoint_config_sha256="b" * 64,
        reference_config_sha256="c" * 64,
        endpoint_parameter_count=101,
        reference_parameter_count=100,
        endpoint_training_flops=105,
        reference_training_flops=100,
        parameter_relative_tolerance=(
            matching_config.parameter_relative_tolerance
        ),
        flop_relative_tolerance=matching_config.flop_relative_tolerance,
        ownership_valid=True,
        common_schedule=True,
        optimizer_policy_match=True,
        training_flop_ledger_complete=True,
        training_flop_obligations=(),
        semantic_interventions=("map_mode",),
        named_factor="map_mode",
        nuisance_capacity_fields=("emission_width",),
        common_schedule_sha256=MATCHING_SCHEDULE_POLICY.policy_sha256,
    )
    assert boundary.parameter_relative_difference == pytest.approx(0.01)
    assert boundary.flop_relative_difference == pytest.approx(0.05)
    assert boundary.eligible is True

    outside = MatchingReport.from_totals(
        matching_config_sha256=matching_config.config_sha256,
        endpoint_config_sha256="d" * 64,
        reference_config_sha256="e" * 64,
        endpoint_parameter_count=102,
        reference_parameter_count=100,
        endpoint_training_flops=106,
        reference_training_flops=100,
        parameter_relative_tolerance=(
            matching_config.parameter_relative_tolerance
        ),
        flop_relative_tolerance=matching_config.flop_relative_tolerance,
        ownership_valid=True,
        common_schedule=True,
        optimizer_policy_match=True,
        training_flop_ledger_complete=True,
        training_flop_obligations=(),
        semantic_interventions=("map_mode", "prior_variant"),
        named_factor="map_mode",
        nuisance_capacity_fields=("emission_width",),
        common_schedule_sha256=MATCHING_SCHEDULE_POLICY.policy_sha256,
    )
    assert outside.status == "INCONCLUSIVE"
    assert outside.eligible is False
    assert outside.parameter_relative_difference > 0.01
    assert outside.flop_relative_difference > 0.05
    assert any("semantic intervention" in item for item in outside.obligations)

    not_outcome_blind = MatchingReport.from_totals(
        matching_config_sha256=matching_config.config_sha256,
        endpoint_config_sha256="f" * 64,
        reference_config_sha256="0" * 64,
        endpoint_parameter_count=100,
        reference_parameter_count=100,
        endpoint_training_flops=100,
        reference_training_flops=100,
        parameter_relative_tolerance=(
            matching_config.parameter_relative_tolerance
        ),
        flop_relative_tolerance=matching_config.flop_relative_tolerance,
        ownership_valid=True,
        common_schedule=True,
        optimizer_policy_match=True,
        training_flop_ledger_complete=True,
        training_flop_obligations=(),
        semantic_interventions=("map_mode",),
        named_factor="map_mode",
        nuisance_capacity_fields=("validation_nll",),
        common_schedule_sha256=MATCHING_SCHEDULE_POLICY.policy_sha256,
    )
    assert not_outcome_blind.status == "INCONCLUSIVE"
    assert not_outcome_blind.eligible is False
    assert any("nuisance" in item for item in not_outcome_blind.obligations)
