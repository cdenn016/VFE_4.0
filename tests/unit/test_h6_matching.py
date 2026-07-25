from __future__ import annotations

from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest
import torch

from vfe4.config import (
    H6ArmMatchingResolvedConfig,
    resolve_h6_arm_matching_config,
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
    MATCHING_SCHEDULE_POLICY,
    FlopTerm,
    adamw_flops,
    audit_arm_matching,
    audit_parameter_ownership,
    backward_flops,
    candidate_allocations,
    capacity_candidate_count,
    dense_matmul_flops,
    dense_matvec_flops,
    immutable_snapshot_flop_term,
    l2_clip_scale_flops,
    log_softmax_flops,
    scalar_flops,
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
            config_id="h6-a0-ar-v1",
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
            capacity_allocation=_allocation(
                latent=False,
                emission_width=emission_width,
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
        objective_kind="complete_elbo",
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
