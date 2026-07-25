from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.generative import (
    FixedSourcePrior,
    ParentSpecificPooledPrefixSourcePrior,
)
from vfe4.predictive import (
    EstimatorStream,
    PrefixCache,
    PriorPrediction,
    TargetFreeProposalAdapter,
)
from vfe4.recognition import (
    FactorizedLanguageRecognition,
    RecognitionConditioning,
    StructuredLanguageRecognition,
)
from vfe4.recognition.parameter_store import LanguageRecognitionParameterStore
from vfe4.training.arms import (
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    ArmConfig,
    ArmMatrixRow,
    BuiltArm,
    CapacityAllocation,
    arm_matrix_sha256,
    build_a0,
    build_a1,
    build_a2,
    build_a3,
    build_a4,
    build_a5,
    build_arm,
    shared_a2_a5_semantic_payload,
)
from vfe4.training.h6_transformer import (
    H6A0ArchitectureProfile,
    H6CausalTransformer,
    H6ScaledDotProductAttention,
)
from vfe4.training.matching import stable_parameter_key
from vfe4.types import ArmId, TrainingPhase, VocabularyIdentity


_SHA = "a" * 64
_BUILDERS = (build_a0, build_a1, build_a2, build_a3, build_a4, build_a5)
_FIELDS = (
    "config_id",
    "latent_enabled",
    "state_channel_enabled",
    "model_channel_enabled",
    "source_mode",
    "map_mode",
    "recognition_family",
    "recognition_conditioning",
    "prior_variant",
    "mixture_mode",
    "objective_kind",
)
_SEMANTICS = {
    ArmId.A0: (
        "h6-a0-transformer-v2",
        False,
        False,
        False,
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
        "cross_entropy",
    ),
    ArmId.A1: (
        "h6-a1-ordinary-latent-v1",
        True,
        True,
        False,
        "absent",
        "absent",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    ArmId.A2: (
        "h6-a2-generic-map-v1",
        True,
        True,
        True,
        "categorical",
        "generic_fixed_frame_non_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    ArmId.A3: (
        "h6-a3-immediate-predecessor-v1",
        True,
        True,
        True,
        "immediate_predecessor",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "absent",
        "absent",
        "complete_elbo",
    ),
    ArmId.A4: (
        "h6-a4-state-only-v1",
        True,
        True,
        False,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
    ArmId.A5: (
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
        True,
        True,
        True,
        "categorical",
        "shared_vertex_coboundary",
        "structured",
        "smoothing",
        "fixed",
        "exact",
        "complete_elbo",
    ),
}


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity("h6-task7-small-v1", 3, _SHA)


def _transformer_vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity("h6-byte-v1", 258, _SHA)


def test_a0_transformer_is_order_sensitive_and_future_blind() -> None:
    model = H6CausalTransformer(
        vocabulary=_transformer_vocabulary(),
        profile=H6A0ArchitectureProfile.create(),
    )
    model.eval()

    first = CausalPrefix.create(
        receiver_t=4,
        vocabulary=_transformer_vocabulary(),
        token_ids=torch.tensor([1, 2, 1], dtype=torch.int64),
    )
    reordered = CausalPrefix.create(
        receiver_t=4,
        vocabulary=_transformer_vocabulary(),
        token_ids=torch.tensor([1, 1, 2], dtype=torch.int64),
    )
    order_delta = torch.max(
        torch.abs(
            model.prefix_log_probs(first)
            - model.prefix_log_probs(reordered)
        )
    ).item()
    assert order_delta > 1.0e-8

    left = model.sequence_log_probs(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    )
    right = model.sequence_log_probs(
        torch.tensor([1, 2, 9, 8, 7], dtype=torch.int64)
    )
    assert torch.equal(left[:3], right[:3])


def test_a0_transformer_inventory_is_exact() -> None:
    profile = H6A0ArchitectureProfile.create()
    model = H6CausalTransformer(
        vocabulary=_transformer_vocabulary(),
        profile=profile,
    )

    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == 61_982
    assert profile.hidden_width % profile.attention_heads == 0
    assert sum(
        isinstance(module, H6ScaledDotProductAttention)
        for module in model.modules()
    ) == 1
    assert sum(
        isinstance(module, torch.nn.LayerNorm)
        for module in model.modules()
    ) == 3
    assert model.token_embedding.weight.shape == (258, 52)
    assert model.position_embedding.weight.shape == (32, 52)
    assert model.qkv_projection.weight.shape == (156, 52)
    assert model.attention_output.weight.shape == (52, 52)
    assert model.mlp_input.weight.shape == (208, 52)
    assert model.mlp_output.weight.shape == (52, 208)
    assert model.decoder.weight.shape == (258, 52)
    assert (
        model.token_embedding.weight.untyped_storage().data_ptr()
        != model.decoder.weight.untyped_storage().data_ptr()
    )


def _config(arm: ArmId, **changes: object) -> ArmConfig:
    values = dict(zip(_FIELDS, _SEMANTICS[arm], strict=True))
    values.update(changes)
    latent = values["latent_enabled"] is True
    recognized = values["recognition_family"] != "absent"
    a0 = arm is ArmId.A0
    allocation = CapacityAllocation.create(
        emission_width=52 if a0 else 48,
        latent_width=8 if latent else None,
        recognition_width=32 if recognized else None,
        prior_context_width=(
            2
            if values["prior_variant"] == "parent_specific_pooled_prefix"
            else None
        ),
    )
    return ArmConfig.create(
        arm=arm,
        vocabulary=_transformer_vocabulary() if a0 else _vocabulary(),
        horizon=32 if a0 else 2,
        capacity_allocation=allocation,
        **values,
    )


@pytest.fixture(scope="module")
def arms() -> dict[ArmId, BuiltArm]:
    return {
        arm: _BUILDERS[index](_config(arm))
        for index, arm in enumerate(ArmId)
    }


def _role_text(arm: BuiltArm) -> str:
    return " ".join(
        f"{row.qualified_name} {row.role}" for row in arm.parameter_roles
    ).lower()


def test_explicit_factories_construct_the_six_literal_families(
    arms: dict[ArmId, BuiltArm],
) -> None:
    assert all(type(arm) is BuiltArm for arm in arms.values())
    dispatched = build_arm(ArmId.A5, _config(ArmId.A5))
    assert dispatched.config == arms[ArmId.A5].config
    assert dispatched.model_family_sha256 == arms[ArmId.A5].model_family_sha256

    source = inspect.getsource(build_arm)
    assert all(f"if arm is ArmId.{item.name}" in source for item in ArmId)
    assert all(
        word not in source for word in ("getattr(", "inspect.signature", "registry")
    )
    with pytest.raises(ValueError, match="config.*arm|arm.*config"):
        build_a0(_config(ArmId.A1))

    a0, a1, a2, a3, a4, a5 = (arms[arm] for arm in ArmId)
    assert a0.recognition_store is None
    assert not a0.config.latent_enabled
    assert "causal_transformer" in _role_text(a0)
    assert all(
        marker not in _role_text(a0)
        for marker in ("latent", "source_bank", "recognition", "edge_map")
    )

    assert a1.model.model_channel_enabled is False
    assert a1.model.map_mode == "absent"
    assert a1.model.source_mode == "absent"
    assert hasattr(a1.model, "ordinary_latent_transition_weight")
    assert not hasattr(a1.model, "state_source_free_logits")
    assert not hasattr(a1.model, "full_same_receiver_b")

    for arm in (a2, a5):
        assert arm.model.model_channel_enabled
        assert type(arm.model.source_prior) is FixedSourcePrior
        assert hasattr(arm.model.source_prior, "state_source_free_logits")
        assert hasattr(arm.model.source_prior, "model_source_free_logits")
        assert hasattr(arm.model, "full_same_receiver_b")
        assert arm.config.prior_variant == "fixed"
        assert arm.config.mixture_mode == "exact"
        assert arm.config.recognition_family == "structured"
        assert arm.config.objective_kind == "complete_elbo"
    assert hasattr(a2.model, "generic_fixed_frame_state_edge_maps")
    assert not hasattr(a2.model, "state_vertex_phi")
    assert hasattr(a5.model, "state_vertex_phi")
    assert not hasattr(a5.model, "generic_fixed_frame_state_edge_maps")
    assert shared_a2_a5_semantic_payload(a2.config) == (
        shared_a2_a5_semantic_payload(a5.config)
    )

    assert a3.model.model_channel_enabled
    assert a3.model.source_mode == "immediate_predecessor"
    assert hasattr(a3.model, "state_vertex_phi")
    assert hasattr(a3.model, "model_vertex_phi")
    assert not hasattr(a3.model, "state_source_free_logits")
    assert not hasattr(a3.model, "model_source_free_logits")
    assert not hasattr(a3.model, "full_same_receiver_b")

    assert not a4.model.model_channel_enabled
    assert hasattr(a4.model, "state_vertex_phi")
    assert type(a4.model.source_prior) is FixedSourcePrior
    assert hasattr(a4.model.source_prior, "state_source_free_logits")
    assert not hasattr(a4.model, "model_vertex_phi")
    assert a4.model.source_prior.model_source_free_logits is None
    assert not hasattr(a4.model, "full_same_receiver_b")
    assert len(a3.model.state_vertex_phi) == a3.config.horizon
    assert len(a5.model.model_vertex_phi) == a5.config.horizon
    assert torch.equal(
        a5.model.vertex_frame("state", 0),
        torch.eye(8, dtype=torch.float64),
    )
    assert "masked_log_softmax_from_parents" in inspect.getsource(
        type(a5.model.source_prior)._normalized
    )
    assert "torch.log_softmax" not in inspect.getsource(
        type(a5.model.source_prior)._normalized
    )

    prefix_a5 = build_a5(
        _config(
            ArmId.A5,
            config_id=(
                "h6-a5-structured-parent-specific-prefix-exact-complete-"
                "latent-smoothing-v2"
            ),
            prior_variant="parent_specific_pooled_prefix",
        )
    )
    assert (
        type(prefix_a5.model.source_prior)
        is ParentSpecificPooledPrefixSourcePrior
    )
    assert prefix_a5.model.source_prior.context_dim == 2

    for builder, arm_id, change in (
        (build_a0, ArmId.A0, {"latent_enabled": True}),
        (build_a2, ArmId.A2, {"model_channel_enabled": False}),
        (build_a4, ArmId.A4, {"model_channel_enabled": True}),
    ):
        with pytest.raises(
            ValueError,
            match="literal semantic profile|canonical H6 arm profile",
        ):
            builder(_config(arm_id, **change))

    # Present capacity fields determine live tensor shapes.
    assert a0.model.token_embedding.weight.shape[1] == 52
    assert a5.model.emission_state_projection.shape == (48, 8)
    assert a5.recognition_store.token_embedding.weight.shape[1] == 32
    assert tuple(a1.elbo_factor_inventory) != tuple(a5.elbo_factor_inventory)
    assert tuple(a3.elbo_factor_inventory) != tuple(a4.elbo_factor_inventory)
    assert all(not arm.training_flop_ledger_complete for arm in arms.values())
    assert all(
        arm.training_flop_obligations for arm in arms.values()
    )
    assert any(
        "matrix_exp" in obligation
        for obligation in a5.training_flop_obligations
    )

    prefix = CausalPrefix.create(
        receiver_t=2,
        vocabulary=_transformer_vocabulary(),
        token_ids=torch.tensor([1], dtype=torch.int64),
    )
    log_probs = a0.model.prefix_log_probs(prefix)
    assert log_probs.shape == (258,)
    assert log_probs.dtype is torch.float64
    assert torch.logsumexp(log_probs, 0).item() == pytest.approx(
        0.0, abs=1e-14
    )


def test_recognition_store_owns_parameters_and_emits_normalized_task4_laws(
    arms: dict[ArmId, BuiltArm],
) -> None:
    conditioning = RecognitionConditioning.create(
        mode="smoothing",
        horizon=2,
        observed_tokens=torch.tensor([0, 2], dtype=torch.int64),
    )
    cases = (
        (arms[ArmId.A5], StructuredLanguageRecognition),
        (
            build_a5(
                _config(
                    ArmId.A5,
                    config_id=(
                        "h6-a5-factorized-fixed-exact-complete-"
                        "latent-smoothing-v1"
                    ),
                    recognition_family="factorized",
                )
            ),
            FactorizedLanguageRecognition,
        ),
    )
    for arm, law_type in cases:
        store = arm.recognition_store
        assert isinstance(store, LanguageRecognitionParameterStore)
        store_names = {
            f"recognition_store.{name}"
            for name, _ in store.named_parameters()
        }
        assert store_names
        role_keys = {
            row.parameter_key
            for row in arm.parameter_roles
            if row.phase == TrainingPhase.RECOGNITION_ADAMW.value
        }
        binding_keys = {
            parameter_key
            for binding in arm.optimizer_bindings
            if binding.phase == TrainingPhase.RECOGNITION_ADAMW.value
            for parameter_key in binding.parameter_keys
        }
        expected_keys = {
            stable_parameter_key(
                qualified_name=name,
                phase=TrainingPhase.RECOGNITION_ADAMW.value,
            )
            for name in store_names
        }
        assert role_keys == binding_keys == expected_keys

        law = store.recognition_law(conditioning)
        assert type(law) is law_type
        assert not hasattr(law, "parameters")
        assert law.mean_value().requires_grad
        assert law.precision_cholesky_value().requires_grad
        diagonal = torch.diagonal(law.precision_cholesky_value())
        assert bool(torch.isfinite(diagonal).all() and torch.all(diagonal > 0.0))
        assert bool(torch.isfinite(law.entropy()))


def test_predictors_are_target_free_and_state_identities_are_rebuilt(
    arms: dict[ArmId, BuiltArm],
) -> None:
    for arm in arms.values():
        assert isinstance(arm.proposal, TargetFreeProposalAdapter)
        method = arm.predictor.next_token_log_probs
        assert tuple(inspect.signature(method).parameters) == (
            "prefix_tokens",
            "estimator_rng",
            "cache",
        )
        hints = get_type_hints(type(arm.predictor).next_token_log_probs)
        assert hints == {
            "prefix_tokens": CausalPrefix,
            "estimator_rng": EstimatorStream,
            "cache": PrefixCache | None,
            "return": PriorPrediction,
        }
        stream = EstimatorStream.create(
            stream_seed=2026072300,
            estimator_identity=arm.predictor.estimator_identity,
        )
        with pytest.raises(ValueError, match="CausalPrefix|causal prefix"):
            method(torch.tensor([0], dtype=torch.int64), stream)

    a0 = arms[ArmId.A0]
    prior_identity = a0.proposal.proposal_identity_sha256
    with torch.no_grad():
        next(a0.model.parameters()).add_(1e-6)
    with pytest.raises(ValueError, match="changed|rebuild"):
        a0.proposal.assert_current_state()
    rebuilt_proposal, rebuilt_predictor = a0.rebuild_predictive_boundary()
    assert rebuilt_proposal.proposal_identity_sha256 != prior_identity
    assert rebuilt_predictor.proposal is rebuilt_proposal

    no_latent = build_a5(
        ArmConfig.create(
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
            capacity_allocation=CapacityAllocation.create(
                emission_width=48,
                latent_width=None,
                recognition_width=None,
            ),
        )
    )
    assert no_latent.recognition_store is None
    assert {row.phase for row in no_latent.parameter_roles} == {
        TrainingPhase.MODEL_CE_ADAMW.value
    }
    assert {row.phase for row in no_latent.optimizer_bindings} == {
        TrainingPhase.MODEL_CE_ADAMW.value
    }
    assert all(
        row.phase == TrainingPhase.MODEL_CE_ADAMW.value
        for row in no_latent.flop_terms
    )


def test_matrix_rows_freeze_literal_interventions_and_nonclaims() -> None:
    expected = (
        (
            "PRIMARY",
            "h6-a0-transformer-v2",
            "whole_declared_architecture",
            "equal_grid",
            "primary",
            "not_component_attribution",
        ),
        (
            "MAP",
            "h6-a2-generic-map-v1",
            "map_mode",
            "equal_grid",
            "conditional",
            "generic_fixed_frame_non_coboundary_not_h7_covariance",
        ),
        (
            "STRUCTURE",
            "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1",
            "recognition_family",
            "shared_a5",
            "conditional",
            "conditional_on_a5_tuning",
        ),
        (
            "PRIOR",
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
            "prior_variant",
            "shared_a5",
            "descriptive",
            "changed_joint_descriptive",
        ),
        (
            "MIXTURE",
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
            "mixture_mode",
            "shared_a5",
            "descriptive",
            "projection_not_exact",
        ),
        (
            "OBJECTIVE",
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
            "objective_kind",
            "shared_a5",
            "conditional",
            "emission_not_elbo",
        ),
        (
            "LATENT",
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
            "latent_channel",
            "shared_a5",
            "descriptive",
            "latent_capacity_descriptive",
        ),
        (
            "RECOGNITION",
            "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1",
            "recognition_conditioning",
            "shared_a5",
            "conditional",
            "recognition_not_used_for_scoring",
        ),
    )
    assert tuple(row.row_id for row in ARM_MATRIX_ROWS) == tuple(
        item[0] for item in expected
    )
    assert all(type(row) is ArmMatrixRow for row in ARM_MATRIX_ROWS)
    assert len({row.row_sha256 for row in ARM_MATRIX_ROWS}) == 8
    assert arm_matrix_sha256(ARM_MATRIX_ROWS) == ARM_MATRIX_SHA256
    for row, item in zip(ARM_MATRIX_ROWS, expected, strict=True):
        row_id, left_id, factor, tuning, interpretation, nonclaim = item
        assert (row.row_id, row.left_config_id) == (row_id, left_id)
        assert (row.named_factor, row.tuning_estimand, row.interpretation) == (
            factor,
            tuning,
            interpretation,
        )
        assert row.semantic_interventions == (factor,)
        assert (
            row.capacity_allocation_policy
            == "outcome_blind_nuisance_reallocation"
        )
        assert set(row.nuisance_capacity_fields) <= {
            "emission_width",
            "latent_width",
            "recognition_width",
            "prior_context_width",
        }
        assert row.confirmatory_seeds == tuple(range(2026072101, 2026072109))
        assert "{config_sha256}" in row.checkpoint_template
        assert "{prefix_case_key_sha256}" in row.certificate_key_template
        assert row.opening_group == "h6-prediction-global-test-opening-v1"
        assert nonclaim in row.nonclaims
