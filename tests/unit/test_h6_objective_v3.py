from __future__ import annotations

import hashlib
import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from vfe4.objective.h6_prediction_v3 import (
    ExactSourceMixtureEvaluationV3,
    evaluate_h6_prediction_elbo_v3,
    project_terminal_mixture_v3,
)
from vfe4.predictive.identities import canonical_model_state_sha256
from vfe4.recognition.h6_prediction_v3 import (
    CategoricalSourceBank,
    CategoricalSourceRow,
    GaussianReceiverComponent,
    LanguageRecognitionTrajectory,
    ReceiverRecognitionContext,
)
from vfe4.recognition.language import RecognitionConditioning
from vfe4.training.arms import LatentLanguageArmModel
from vfe4.training.h6_noise_v3 import (
    H6TrainingCounterKeyV3,
    training_normal_tensor_v3,
    training_normal_values_v3,
)
from vfe4.types.h6 import (
    ArmId,
    FrozenTensorSnapshot,
    TrainingPhase,
    VocabularyIdentity,
)


_RECOGNITION_STATE_IDENTITY = hashlib.sha256(b"recognition-state").hexdigest()


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="h6-objective-v3-tests",
        size=11,
        tokenizer_spec_bytes=b"h6-objective-v3-tests",
    )


def _model(*, horizon: int = 2) -> LatentLanguageArmModel:
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


def _source_row(
    *,
    bank: str,
    receiver_t: int,
    support: tuple[int, ...],
    log_probabilities: Tensor,
    source_model_state_sha256: str,
) -> CategoricalSourceRow:
    zeros = torch.zeros_like(log_probabilities)
    probabilities = log_probabilities.exp()
    return CategoricalSourceRow.create(
        bank=bank,  # type: ignore[arg-type]
        receiver_t=receiver_t,
        support=support,
        causal_prefix_sha256=hashlib.sha256(
            f"prefix:{receiver_t}".encode()
        ).hexdigest(),
        source_model_state_sha256=source_model_state_sha256,
        log_prior_baseline=zeros,
        residual_scores=zeros,
        log_probabilities=log_probabilities,
        entropy=-(probabilities * log_probabilities).sum(),
    )


def _trajectory(
    *,
    model: LatentLanguageArmModel,
    requires_grad: bool,
) -> tuple[LanguageRecognitionTrajectory, tuple[Tensor, ...]]:
    source_model_state_sha256 = canonical_model_state_sha256(model)
    base_means = torch.tensor(
        [[-0.20, 0.10], [0.35, -0.15], [0.05, 0.30]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    precision_cholesky = torch.tensor(
        [[1.20, 0.00], [0.15, 0.90]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    state_logits = torch.tensor(
        [-0.40, 0.60],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    model_logits = torch.tensor(
        [0.25, -0.15],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    singleton = base_means.new_zeros(1)
    state_log_probabilities = F.log_softmax(state_logits, dim=0)
    model_log_probabilities = F.log_softmax(model_logits, dim=0)
    state_rows = (
        _source_row(
            bank="state",
            receiver_t=1,
            support=(0,),
            log_probabilities=singleton,
            source_model_state_sha256=source_model_state_sha256,
        ),
        _source_row(
            bank="state",
            receiver_t=2,
            support=(0, 1),
            log_probabilities=state_log_probabilities,
            source_model_state_sha256=source_model_state_sha256,
        ),
    )
    model_rows = (
        _source_row(
            bank="model",
            receiver_t=1,
            support=(0,),
            log_probabilities=singleton,
            source_model_state_sha256=source_model_state_sha256,
        ),
        _source_row(
            bank="model",
            receiver_t=2,
            support=(0, 1),
            log_probabilities=model_log_probabilities,
            source_model_state_sha256=source_model_state_sha256,
        ),
    )
    state_bank = CategoricalSourceBank.create(bank="state", rows=state_rows)
    model_bank = CategoricalSourceBank.create(bank="model", rows=model_rows)
    zero = base_means.new_zeros(())
    precision_snapshot = FrozenTensorSnapshot.capture(precision_cholesky)
    components: list[tuple[GaussianReceiverComponent, ...]] = [
        (
            GaussianReceiverComponent.create(
                receiver_t=receiver_t,
                state_source_j=None,
                model_source_j=None,
                mean=base_means[receiver_t],
                precision_cholesky_snapshot=precision_snapshot,
                log_probability=zero,
            ),
        )
        for receiver_t in range(2)
    ]
    terminal: list[GaussianReceiverComponent] = []
    for state_index, state_source_j in enumerate(state_rows[-1].support):
        for model_index, model_source_j in enumerate(model_rows[-1].support):
            shift = base_means.new_tensor(
                [0.30 * state_source_j, -0.20 * model_source_j]
            )
            terminal.append(
                GaussianReceiverComponent.create(
                    receiver_t=2,
                    state_source_j=state_source_j,
                    model_source_j=model_source_j,
                    mean=base_means[2] + shift,
                    precision_cholesky_snapshot=precision_snapshot,
                    log_probability=(
                        state_log_probabilities[state_index]
                        + model_log_probabilities[model_index]
                    ),
                )
            )
    components.append(tuple(terminal))
    conditioning = RecognitionConditioning.create(
        mode="smoothing",
        horizon=2,
        observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
    )
    receiver_contexts = tuple(
        ReceiverRecognitionContext.create(
            receiver_t=receiver_t,
            conditioning_mode="smoothing",
            context=torch.zeros(4, dtype=torch.float64),
        )
        for receiver_t in range(3)
    )
    trajectory = LanguageRecognitionTrajectory.create(
        conditioning=conditioning,
        family="structured",
        block_sizes=(2,),
        receiver_contexts=receiver_contexts,
        base_means=base_means,
        precision_cholesky_snapshot=precision_snapshot,
        state_source=state_bank,
        model_source=model_bank,
        receiver_components=tuple(components),
        recognition_store_state_sha256=_RECOGNITION_STATE_IDENTITY,
        source_model_state_sha256=source_model_state_sha256,
    )
    return trajectory, (
        base_means,
        precision_cholesky,
        state_logits,
        model_logits,
    )


def _freeze_model(model: LatentLanguageArmModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _diagonal_log_prob(value: Tensor, mean: Tensor, log_scale: Tensor) -> Tensor:
    standardized = (value - mean) * torch.exp(-log_scale)
    return -0.5 * torch.sum(
        standardized.square() + 2.0 * log_scale + math.log(2.0 * math.pi)
    )


def _projection_components(
    *,
    means: tuple[tuple[float, ...], ...],
    probabilities: tuple[float, ...],
) -> tuple[GaussianReceiverComponent, ...]:
    dimension = len(means[0])
    precision = torch.eye(dimension, dtype=torch.float64)
    precision_snapshot = FrozenTensorSnapshot.capture(precision)
    return tuple(
        GaussianReceiverComponent.create(
            receiver_t=2,
            state_source_j=index,
            model_source_j=None,
            mean=torch.tensor(mean, dtype=torch.float64, requires_grad=True),
            precision_cholesky_snapshot=precision_snapshot,
            log_probability=torch.tensor(math.log(probability), dtype=torch.float64),
        )
        for index, (mean, probability) in enumerate(
            zip(means, probabilities, strict=True)
        )
    )


def test_counter_normal_is_stable_and_execution_order_independent() -> None:
    key = H6TrainingCounterKeyV3(
        attempt_spec_sha256=hashlib.sha256(b"attempt").hexdigest(),
        pass_index=1,
        batch_index=4,
        phase=TrainingPhase.RECOGNITION_ADAMW,
        example_ordinal=3,
        sample_ordinal=0,
        draw_block=0,
    )
    other = H6TrainingCounterKeyV3(
        attempt_spec_sha256=key.attempt_spec_sha256,
        pass_index=1,
        batch_index=4,
        phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=3,
        sample_ordinal=0,
        draw_block=0,
    )

    first = training_normal_values_v3(key, count=6)
    _ = training_normal_values_v3(other, count=31)
    second = training_normal_values_v3(key, count=6)
    tensor, consumption = training_normal_tensor_v3(
        key,
        receiver_count=3,
        latent_dimension=2,
        device="cpu",
    )

    assert first == second
    assert first != training_normal_values_v3(other, count=6)
    assert tensor.device.type == "cpu"
    assert tensor.dtype is torch.float64
    assert tensor.shape == (3, 2)
    assert tuple(tensor.reshape(-1).tolist()) == first
    assert len(consumption) == 64


def test_terminal_beta_gamma_sum_matches_monolithic_log_ratio_oracle() -> None:
    model = _model()
    _freeze_model(model)
    trajectory, _leaves = _trajectory(
        model=model,
        requires_grad=True,
    )
    base_noise = torch.zeros((3, 2), dtype=torch.float64)

    estimate = evaluate_h6_prediction_elbo_v3(
        model=model,
        trajectory=trajectory,
        observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
        base_noise=base_noise,
        mixture_mode="exact",
        active_parameter_block="recognition",
    )

    state_row = trajectory.state_source.rows[-1]
    model_row = trajectory.model_source.rows[-1]
    earlier = estimate.source_law.source_independent_samples
    expected = base_noise.new_zeros(())
    for component in trajectory.terminal_components:
        assert component.state_source_j is not None
        assert component.model_source_j is not None
        state_index = state_row.support.index(component.state_source_j)
        model_index = model_row.support.index(component.model_source_j)
        state_log_q = state_row.log_probabilities[state_index]
        model_log_q = model_row.log_probabilities[model_index]
        value = component.mean
        state = value[:1]
        model_value = value[1:]
        state_mean = model.state_transition_mean(
            receiver_t=2,
            source_j=component.state_source_j,
            source_state=earlier[component.state_source_j][:1],
            current_model=model_value,
        )
        model_mean = model.model_transition_mean(
            receiver_t=2,
            source_j=component.model_source_j,
            source_model=earlier[component.model_source_j][1:],
        )
        bracket = (
            -math.log(2.0)
            + _diagonal_log_prob(
                state,
                state_mean,
                model.state_transition_log_scale,
            )
            - math.log(2.0)
            + _diagonal_log_prob(
                model_value,
                model_mean,
                model.model_transition_log_scale,
            )
            + model.emission_log_probs(
                state=state,
                model=model_value,
            )[7]
            - state_log_q
            - model_log_q
        )
        expected = expected + torch.exp(state_log_q + model_log_q) * bracket

    torch.testing.assert_close(
        estimate.terminal_joint_contribution,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(estimate.source_law.terminal_components) == 4


def test_exact_source_law_is_an_evaluated_record_not_a_marker() -> None:
    model = _model()
    _freeze_model(model)
    trajectory, _leaves = _trajectory(
        model=model,
        requires_grad=True,
    )

    estimate = evaluate_h6_prediction_elbo_v3(
        model=model,
        trajectory=trajectory,
        observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
        base_noise=torch.tensor(
            [[0.25, -0.5], [0.75, 0.125], [-0.2, 0.4]],
            dtype=torch.float64,
        ),
        mixture_mode="exact",
        active_parameter_block="recognition",
    )

    law = estimate.source_law
    assert isinstance(law, ExactSourceMixtureEvaluationV3)
    assert law.source_rows
    assert law.terminal_components
    assert len(law.source_independent_samples) == 2
    assert all(row.generative_log_prior.numel() for row in law.source_rows)
    assert all(row.transition_log_probabilities.numel() for row in law.source_rows)
    assert all(
        row.sampled_earlier_latents.shape[0] == row.receiver_t
        for row in law.source_rows
    )
    assert law.law_identity_sha256 != trajectory.trajectory_identity_sha256
    assert estimate.canonical_ordered_total is estimate.independently_accumulated_total
    assert estimate.elbo is estimate.canonical_ordered_total
    torch.testing.assert_close(estimate.loss, -estimate.elbo)

    with torch.no_grad():
        model.normalized_emission_bias[0].add_(0.125)
    with pytest.raises(ValueError, match="source-model state"):
        evaluate_h6_prediction_elbo_v3(
            model=model,
            trajectory=trajectory,
            observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
            base_noise=torch.zeros((3, 2), dtype=torch.float64),
            mixture_mode="exact",
            active_parameter_block="recognition",
        )


def test_projected_terminal_sampler_includes_between_component_covariance() -> None:
    components = _projection_components(
        means=((-1.0, -2.0), (1.0, 2.0)),
        probabilities=(0.5, 0.5),
    )

    projection = project_terminal_mixture_v3(
        terminal_components=components,
        base_noise=torch.zeros(2, dtype=torch.float64),
    )

    torch.testing.assert_close(
        projection.projected_mean,
        torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        projection.projected_covariance,
        torch.tensor([[2.0, 2.0], [2.0, 5.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        projection.sample,
        projection.projected_mean,
    )
    assert projection.projected_covariance[0, 1] != 0.0


def test_projected_entropy_and_component_kl_bound_match_hand_oracle() -> None:
    components = _projection_components(
        means=((-1.0,), (1.0,)),
        probabilities=(0.5, 0.5),
    )

    projection = project_terminal_mixture_v3(
        terminal_components=components,
        base_noise=torch.zeros(1, dtype=torch.float64),
    )

    expected_entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + math.log(2.0))
    expected_upper_bound = 0.5 * math.log(2.0)
    torch.testing.assert_close(
        projection.analytic_entropy,
        torch.tensor(expected_entropy, dtype=torch.float64),
    )
    torch.testing.assert_close(
        projection.component_kl_upper_bound,
        torch.tensor(expected_upper_bound, dtype=torch.float64),
    )
    assert projection.bound_label == "weighted_component_kl_upper_bound"
    assert projection.bound_label != "mixture_kl"


def test_elbo_gradients_reach_only_the_requested_parameter_block() -> None:
    recognition_model = _model()
    _freeze_model(recognition_model)
    live_trajectory, recognition_leaves = _trajectory(
        model=recognition_model,
        requires_grad=True,
    )
    recognition_estimate = evaluate_h6_prediction_elbo_v3(
        model=recognition_model,
        trajectory=live_trajectory,
        observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
        base_noise=torch.zeros((3, 2), dtype=torch.float64),
        mixture_mode="exact",
        active_parameter_block="recognition",
    )
    recognition_estimate.loss.backward()

    assert all(leaf.grad is not None for leaf in recognition_leaves)
    assert any(bool(torch.any(leaf.grad != 0.0)) for leaf in recognition_leaves)
    assert all(parameter.grad is None for parameter in recognition_model.parameters())

    model_phase_model = _model()
    detached_trajectory, detached_leaves = _trajectory(
        model=model_phase_model,
        requires_grad=False,
    )
    model_estimate = evaluate_h6_prediction_elbo_v3(
        model=model_phase_model,
        trajectory=detached_trajectory,
        observed_tokens=torch.tensor([3, 7], dtype=torch.int64),
        base_noise=torch.zeros((3, 2), dtype=torch.float64),
        mixture_mode="moment_projection",
        active_parameter_block="model",
    )
    model_estimate.loss.backward()

    assert all(leaf.grad is None for leaf in detached_leaves)
    model_gradients = tuple(
        parameter.grad
        for parameter in model_phase_model.parameters()
        if parameter.requires_grad
    )
    assert any(gradient is not None for gradient in model_gradients)
    assert any(
        gradient is not None and bool(torch.any(gradient != 0.0))
        for gradient in model_gradients
    )
