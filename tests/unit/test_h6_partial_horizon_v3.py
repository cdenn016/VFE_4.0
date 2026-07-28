from __future__ import annotations

import hashlib
import math

import pytest
import torch

from vfe4.objective import (
    evaluate_h6_no_latent_cross_entropy_v3,
    evaluate_h6_prediction_elbo_v3,
)
from vfe4.recognition import (
    H6ActiveHorizonV3,
    H6ActiveRecognitionTopologyV3,
    LanguageRecognitionParameterStore,
    RecognitionConditioning,
    project_active_recognition_topology_v3,
)
from vfe4.training.arms import LatentLanguageArmModel
from vfe4.training.h6_training_attempt_v3 import (
    H6GenerativePriorFeatureProviderV3,
)
from vfe4.types.h6 import ArmId, VocabularyIdentity


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="h6-partial-horizon-v3-tests",
        size=11,
        tokenizer_spec_bytes=b"h6-partial-horizon-v3-tests",
    )


def _model(*, horizon: int = 3) -> LatentLanguageArmModel:
    return LatentLanguageArmModel(
        arm=ArmId.A2,
        vocabulary=_vocabulary(),
        horizon=horizon,
        emission_width=3,
        latent_width=1,
        state_channel_enabled=True,
        model_channel_enabled=True,
        source_mode="categorical",
        map_mode="generic_fixed_frame_non_coboundary",
        prior_variant="fixed",
        prior_context_width=None,
        predictor_config_sha256=hashlib.sha256(b"predictor").hexdigest(),
        model_family_sha256=hashlib.sha256(b"family").hexdigest(),
    )


def _store(*, horizon: int = 3) -> LanguageRecognitionParameterStore:
    return LanguageRecognitionParameterStore(
        vocabulary=_vocabulary(),
        horizon=horizon,
        latent_width=1,
        recognition_width=3,
        channel_count=2,
        family="structured",
        conditioning_mode="smoothing",
        trainable_source_banks=("state", "model"),
    )


def _conditioning(tokens: tuple[int, ...]) -> RecognitionConditioning:
    return RecognitionConditioning.create(
        mode="smoothing",
        horizon=len(tokens),
        observed_tokens=torch.tensor(tokens, dtype=torch.int64),
    )


def _freeze_model(model: LatentLanguageArmModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def test_active_horizon_is_exact_prefix_bound_to_maximum_horizon() -> None:
    binding = H6ActiveHorizonV3.create(
        maximum_horizon=3,
        active_horizon=1,
        active_receiver_mask=(True, True, False, False),
    )

    assert binding.maximum_horizon == 3
    assert binding.active_horizon == 1
    assert binding.active_receiver_mask == (True, True, False, False)
    assert len(binding.evaluation_identity_sha256) == 64

    invalid = (
        dict(maximum_horizon=3, active_horizon=0),
        dict(
            maximum_horizon=3,
            active_horizon=1,
            active_receiver_mask=(True, False, True, False),
        ),
        dict(
            maximum_horizon=3,
            active_horizon=2,
            active_receiver_mask=(True, True, False, False),
        ),
        dict(maximum_horizon=3, active_horizon=4),
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            H6ActiveHorizonV3.create(**kwargs)


def test_partial_trajectory_uses_only_real_targets_and_reanchors_terminal() -> (
    None
):
    model = _model()
    store = _store()
    binding = H6ActiveHorizonV3.create(
        maximum_horizon=3,
        active_horizon=1,
        active_receiver_mask=(True, True, False, False),
    )
    provider = H6GenerativePriorFeatureProviderV3(model=model)

    trajectory = store.recognition_trajectory(
        _conditioning((7,)),
        prior_feature_provider=provider,
        active_horizon=binding,
    )

    assert store.horizon == model.horizon == 3
    assert trajectory.conditioning.horizon == 1
    assert trajectory.receiver_labels == (0, 1)
    assert tuple(row.receiver_t for row in trajectory.state_source.rows) == (
        1,
    )
    assert tuple(row.receiver_t for row in trajectory.model_source.rows) == (
        1,
    )
    assert all(
        component.receiver_t == 1
        for component in trajectory.terminal_components
    )
    assert torch.equal(
        trajectory.conditioning.observed_tokens.value(),
        torch.tensor([7], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="active horizon"):
        store.recognition_trajectory(
            _conditioning((7, 8, 9)),
            prior_feature_provider=provider,
            active_horizon=binding,
        )


def test_partial_trajectory_projects_exact_ragged_engine_topology() -> None:
    model = _model()
    store = _store()
    binding = H6ActiveHorizonV3.create(
        maximum_horizon=3,
        active_horizon=2,
    )
    trajectory = store.recognition_trajectory(
        _conditioning((5, 7)),
        prior_feature_provider=H6GenerativePriorFeatureProviderV3(model=model),
        active_horizon=binding,
    )

    topology = project_active_recognition_topology_v3(
        trajectory=trajectory,
        active_horizon=binding,
    )

    assert type(topology) is H6ActiveRecognitionTopologyV3
    assert topology.receiver_count == 3
    assert topology.state_categorical_enabled
    assert topology.model_categorical_enabled
    assert topology.state_categorical_supports == ((0,), (0, 1))
    assert topology.model_categorical_supports == ((0,), (0, 1))
    assert tuple(item[0] for item in topology.receiver_components) == (0, 1, 2)
    assert topology.receiver_components == tuple(
        (
            receiver_t,
            tuple(
                component.component_identity_sha256 for component in components
            ),
        )
        for receiver_t, components in enumerate(trajectory.receiver_components)
    )
    assert topology.trajectory_identity_sha256 == (
        trajectory.trajectory_identity_sha256
    )
    assert topology.active_horizon_evaluation_sha256 == (
        binding.evaluation_identity_sha256
    )

    with pytest.raises(ValueError, match="active horizon"):
        project_active_recognition_topology_v3(
            trajectory=trajectory,
            active_horizon=H6ActiveHorizonV3.create(
                maximum_horizon=3,
                active_horizon=1,
            ),
        )
    with pytest.raises(ValueError, match="maximum horizon"):
        project_active_recognition_topology_v3(
            trajectory=trajectory,
            active_horizon=H6ActiveHorizonV3.create(
                maximum_horizon=4,
                active_horizon=2,
            ),
        )


def test_partial_elbo_evaluates_exact_active_receivers_and_noise() -> None:
    model = _model()
    store = _store()
    binding = H6ActiveHorizonV3.create(
        maximum_horizon=3,
        active_horizon=1,
    )
    trajectory = store.recognition_trajectory(
        _conditioning((7,)),
        prior_feature_provider=H6GenerativePriorFeatureProviderV3(model=model),
        active_horizon=binding,
    )
    _freeze_model(model)
    noise = torch.zeros((2, 2), dtype=torch.float64)

    estimate = evaluate_h6_prediction_elbo_v3(
        model=model,
        trajectory=trajectory,
        observed_tokens=torch.tensor([7], dtype=torch.int64),
        base_noise=noise,
        mixture_mode="exact",
        active_parameter_block="recognition",
        active_horizon=binding,
    )

    assert estimate.active_horizon == binding
    assert max(term.receiver_t for term in estimate.ordered_terms) == 1
    assert tuple(
        term.receiver_t
        for term in estimate.ordered_terms
        if term.partition == "emission"
    ) == (1,)
    assert len(estimate.source_law.source_independent_samples) == 1

    with pytest.raises(ValueError, match="base_noise"):
        evaluate_h6_prediction_elbo_v3(
            model=model,
            trajectory=trajectory,
            observed_tokens=torch.tensor([7], dtype=torch.int64),
            base_noise=torch.zeros((4, 2), dtype=torch.float64),
            mixture_mode="exact",
            active_parameter_block="recognition",
            active_horizon=binding,
        )


def test_explicit_full_horizon_is_identical_to_omitted_full_horizon() -> None:
    model = _model()
    store = _store()
    conditioning = _conditioning((3, 5, 7))
    provider = H6GenerativePriorFeatureProviderV3(model=model)
    default_trajectory = store.recognition_trajectory(
        conditioning,
        prior_feature_provider=provider,
    )
    full = H6ActiveHorizonV3.create(
        maximum_horizon=3,
        active_horizon=3,
    )
    explicit_trajectory = store.recognition_trajectory(
        conditioning,
        prior_feature_provider=provider,
        active_horizon=full,
    )
    _freeze_model(model)
    noise = torch.arange(8, dtype=torch.float64).reshape(4, 2) / 10.0
    tokens = torch.tensor([3, 5, 7], dtype=torch.int64)

    default = evaluate_h6_prediction_elbo_v3(
        model=model,
        trajectory=default_trajectory,
        observed_tokens=tokens,
        base_noise=noise,
        mixture_mode="exact",
        active_parameter_block="recognition",
    )
    explicit = evaluate_h6_prediction_elbo_v3(
        model=model,
        trajectory=explicit_trajectory,
        observed_tokens=tokens,
        base_noise=noise,
        mixture_mode="exact",
        active_parameter_block="recognition",
        active_horizon=full,
    )

    assert (
        default_trajectory.trajectory_identity_sha256
        == explicit_trajectory.trajectory_identity_sha256
    )
    assert (
        default.estimate_identity_sha256 == explicit.estimate_identity_sha256
    )
    assert default.active_horizon == explicit.active_horizon == full
    assert torch.equal(default.elbo, explicit.elbo)
    assert tuple(
        (term.partition, term.receiver_t, term.value.detach().clone())
        for term in default.ordered_terms
    ) == tuple(
        (term.partition, term.receiver_t, term.value.detach().clone())
        for term in explicit.ordered_terms
    )


def test_no_latent_ce_counts_only_concatenated_mixed_horizon_rows() -> None:
    logits = torch.zeros(
        (5, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor(
        (0, 1, 2, 1, 0),
        dtype=torch.int64,
    )
    bindings = (
        H6ActiveHorizonV3.create(maximum_horizon=4, active_horizon=2),
        H6ActiveHorizonV3.create(maximum_horizon=4, active_horizon=3),
    )

    loss = evaluate_h6_no_latent_cross_entropy_v3(
        logits=logits,
        targets=targets,
        active_horizons=bindings,
    )
    loss.backward()

    assert loss.shape == ()
    assert loss.item() == pytest.approx(math.log(3.0), rel=0.0, abs=1e-15)
    assert logits.grad is not None
    assert logits.grad.shape == (5, 3)
    assert torch.count_nonzero(logits.grad).item() == 15


def test_no_latent_ce_rejects_padding_count_drift_and_legacy_bindings() -> None:
    logits = torch.zeros((2, 3), dtype=torch.float64)
    binding = H6ActiveHorizonV3.create(
        maximum_horizon=4,
        active_horizon=2,
    )

    invalid_targets = (
        torch.tensor((0, -100), dtype=torch.int64),
        torch.tensor((0, 1, 2), dtype=torch.int64),
        torch.tensor(((0, 1, -100, -100),), dtype=torch.int64),
    )
    for targets in invalid_targets:
        with pytest.raises(ValueError, match="target|active"):
            evaluate_h6_no_latent_cross_entropy_v3(
                logits=logits,
                targets=targets,
                active_horizons=(binding,),
            )

    with pytest.raises(ValueError, match="exact H6ActiveHorizonV3"):
        evaluate_h6_no_latent_cross_entropy_v3(
            logits=logits,
            targets=torch.tensor((0, 1), dtype=torch.int64),
            active_horizons=(2,),  # type: ignore[arg-type]
        )
