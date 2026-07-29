from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import pytest
import torch
from torch.nn import functional as F

from vfe4.data.windows import CausalPrefix
from vfe4.predictive.identities import EstimatorIdentity
from vfe4.predictive.proposal import EstimatorStream
from vfe4.training.engine import (
    WT103_STRUCTURED_FACTOR_ELBO_SCHEMA,
    WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256,
    ForwardTerms,
    RecognitionSnapshot,
)
from vfe4.training.wt103_adapters import (
    ExactAutoregressivePriorPredictor,
    WT103ChunkedSmcPriorPredictor,
)
from vfe4.training.wt103_runtime import (
    BandedRecognitionState,
    WT103LatentGenerativeModel,
    WT103NoLatentModel,
    WT103StructuredRecognition,
    build_adamw,
    build_warmup_cosine_scheduler,
    causal_observation_history,
    compute_wt103_forward_terms,
)
from vfe4.types.h5_schema import (
    H5_OBJECTIVE_SCHEMA_SHA256,
    H5_SIGNED_TERM_IDS,
)
from vfe4.types.h6 import EstimatorSpec, VocabularyIdentity
from vfe4.types.training import AdamWProfile, SchedulerProfile


@dataclass(frozen=True)
class _Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor
    counted_targets: int


def _batch(*, vocabulary_size: int, length: int) -> _Batch:
    inputs = torch.tensor(
        [[index % vocabulary_size for index in range(length)]],
        dtype=torch.int64,
    )
    targets = torch.tensor(
        [[(index + 1) % vocabulary_size for index in range(length)]],
        dtype=torch.int64,
    )
    return _Batch(
        inputs=inputs,
        targets=targets,
        attention_mask=torch.ones_like(inputs, dtype=torch.bool),
        counted_targets=length,
    )


@dataclass(frozen=True)
class _IndependentStructuredElbo:
    objective_numerator: torch.Tensor
    expected_log_emission: tuple[torch.Tensor, ...]
    initial_model_cross_entropy: torch.Tensor
    initial_state_cross_entropy: torch.Tensor
    model_source_cross_entropy: tuple[torch.Tensor, ...]
    model_transition_cross_entropy: tuple[torch.Tensor, ...]
    state_source_cross_entropy: tuple[torch.Tensor, ...]
    state_transition_cross_entropy: tuple[torch.Tensor, ...]
    continuous_recognition_entropy: torch.Tensor
    conditional_source_entropy_estimate: torch.Tensor
    joint_recognition_entropy_estimate: torch.Tensor


def _independent_zero_noise_structured_elbo(
    *,
    model: WT103LatentGenerativeModel,
    recognition: WT103StructuredRecognition,
    batch: _Batch,
) -> _IndependentStructuredElbo:
    """Reconstruct the structured-factor ELBO without ForwardTerms helpers."""

    inputs = batch.inputs.to(device=model.device)
    targets = batch.targets.to(device=model.device)
    mask = batch.attention_mask.to(device=model.device)
    state = recognition(batch)
    latent = state.mean
    log_two_pi = math.log(2.0 * math.pi)

    logits = F.linear(
        latent,
        model.decoder.projection.weight,
        model.decoder.projection.bias,
    )
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    selected = F.log_softmax(logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    selected = torch.where(mask, selected, 0.0)
    expected_log_emission = tuple(
        selected[:, position].sum()
        for position in range(selected.shape[1])
    )

    initial_scale = torch.exp(model.initial_log_scale)
    initial_negative_log_density = (
        0.5
        * ((latent[:, 0] - model.initial_mean) / initial_scale).square()
        + torch.log(initial_scale)
        + 0.5 * log_two_pi
    )
    initial_active = mask[:, 0].to(dtype=latent.dtype)
    initial_state_cross_entropy = (
        initial_negative_log_density[:, : model.d_z].sum(dim=-1)
        * initial_active
    ).sum()
    initial_model_cross_entropy = (
        initial_negative_log_density[:, model.d_z :].sum(dim=-1)
        * initial_active
    ).sum()

    dimensions = mask.sum(dim=1) * model.latent_width
    continuous_recognition_entropy = (
        0.5
        * dimensions.to(dtype=latent.dtype)
        * (1.0 + log_two_pi)
        - torch.log(state.diagonal_factor)
        .sum(dim=-1)
        .masked_fill(~mask, 0.0)
        .sum(dim=1)
    ).sum()
    state_source_log_probs, model_source_log_probs = (
        model.source_log_probs(inputs, latent)
    )
    zero = latent.new_zeros(())
    state_source_rows = [zero]
    model_source_rows = [zero]
    state_transition_rows = [zero]
    model_transition_rows = [zero]
    source_entropy_rows: list[torch.Tensor] = []
    for receiver in range(1, model.sequence_length):
        indices = model.parent_indices[receiver]
        valid = indices >= 0
        parent_rows = latent[:, indices[valid]]
        state_parents = parent_rows[..., : model.d_z]
        model_parents = parent_rows[..., model.d_z :]
        current_state = latent[:, receiver, : model.d_z]
        current_model = latent[:, receiver, model.d_z :]
        state_q_log = F.log_softmax(
            -(
                state_parents - current_state.unsqueeze(1)
            ).square().sum(dim=-1),
            dim=-1,
        )
        model_q_log = F.log_softmax(
            -(
                model_parents - current_model.unsqueeze(1)
            ).square().sum(dim=-1),
            dim=-1,
        )
        state_q = state_q_log.exp()
        model_q = model_q_log.exp()
        active = mask[:, receiver].to(dtype=latent.dtype)
        source_entropy_rows.append(
            (
                -(state_q * state_q_log).sum(dim=-1)
                -(model_q * model_q_log).sum(dim=-1)
            ).mul(active).sum()
        )
        state_source_rows.append(
            (
                state_q
                * -state_source_log_probs[:, receiver, valid]
            ).sum(dim=-1).mul(active).sum()
        )
        model_source_rows.append(
            (
                model_q
                * -model_source_log_probs[:, receiver, valid]
            ).sum(dim=-1).mul(active).sum()
        )

        model_mean = model.model_transition_mean(
            input_ids=inputs[:, receiver],
            receiver_position=receiver,
            source_positions=indices[valid],
            model_parent=model_parents,
        )
        model_scale = torch.exp(
            model.transition_log_scale[model.d_z :]
        )
        model_component_log = -(
            0.5
            * (
                (current_model.unsqueeze(1) - model_mean)
                / model_scale
            ).square()
            + torch.log(model_scale)
            + 0.5 * log_two_pi
        ).sum(dim=-1)
        expanded_model = current_model.unsqueeze(1).expand(
            -1, state_parents.shape[1], -1
        )
        state_mean = model.state_transition_mean(
            input_ids=inputs[:, receiver],
            receiver_position=receiver,
            source_positions=indices[valid],
            state_parent=state_parents,
            current_model=expanded_model,
        )
        state_scale = torch.exp(
            model.transition_log_scale[: model.d_z]
        )
        state_component_log = -(
            0.5
            * (
                (current_state.unsqueeze(1) - state_mean)
                / state_scale
            ).square()
            + torch.log(state_scale)
            + 0.5 * log_two_pi
        ).sum(dim=-1)
        model_transition_rows.append(
            (model_q * -model_component_log)
            .sum(dim=-1)
            .mul(active)
            .sum()
        )
        state_transition_rows.append(
            (state_q * -state_component_log)
            .sum(dim=-1)
            .mul(active)
            .sum()
        )

    state_source_cross_entropy = tuple(state_source_rows)
    model_source_cross_entropy = tuple(model_source_rows)
    state_transition_cross_entropy = tuple(state_transition_rows)
    model_transition_cross_entropy = tuple(model_transition_rows)
    conditional_source_entropy_estimate = sum(
        source_entropy_rows, zero
    )
    joint_recognition_entropy_estimate = (
        continuous_recognition_entropy
        + conditional_source_entropy_estimate
    )
    objective_numerator = (
        sum(expected_log_emission)
        - initial_model_cross_entropy
        - initial_state_cross_entropy
        - sum(model_source_cross_entropy)
        - sum(model_transition_cross_entropy)
        - sum(state_source_cross_entropy)
        - sum(state_transition_cross_entropy)
        + joint_recognition_entropy_estimate
    )
    return _IndependentStructuredElbo(
        objective_numerator=objective_numerator,
        expected_log_emission=expected_log_emission,
        initial_model_cross_entropy=initial_model_cross_entropy,
        initial_state_cross_entropy=initial_state_cross_entropy,
        model_source_cross_entropy=model_source_cross_entropy,
        model_transition_cross_entropy=model_transition_cross_entropy,
        state_source_cross_entropy=state_source_cross_entropy,
        state_transition_cross_entropy=state_transition_cross_entropy,
        continuous_recognition_entropy=continuous_recognition_entropy,
        conditional_source_entropy_estimate=(
            conditional_source_entropy_estimate
        ),
        joint_recognition_entropy_estimate=(
            joint_recognition_entropy_estimate
        ),
    )


def _optimizer_profile() -> AdamWProfile:
    return AdamWProfile(
        optimizer="AdamW",
        betas=(0.9, 0.999),
        epsilon=1.0e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
        gradient_clip="per_active_block_global_l2",
        gradient_clip_max_norm=1.0,
        proposal_acceptance="validity_only_no_monotonicity_claim",
        reject_on=(
            "nonfinite_objective",
            "nonfinite_gradient",
            "amp_overflow",
            "invalid_support",
            "non_spd",
            "scope_mismatch",
            "snapshot_alias",
            "optimizer_access_mismatch",
        ),
    )


def _scheduler_profile() -> SchedulerProfile:
    return SchedulerProfile(
        scheduler="linear_warmup_then_cosine",
        warmup_optimizer_steps=100,
        minimum_lr_ratio=0.1,
        restart_count=0,
        horizon="planned_active_optimizer_steps_for_attempt",
    )


def _estimator(
    *,
    kind: str,
    particle_count: int | None,
) -> tuple[EstimatorSpec, EstimatorIdentity, EstimatorStream]:
    spec = EstimatorSpec.create(
        kind=kind,  # type: ignore[arg-type]
        particle_count=particle_count,
        resampling=(
            "none" if particle_count is None else "systematic_ess_half"
        ),
    )
    identity = EstimatorIdentity.from_spec(spec)
    return spec, identity, EstimatorStream.create(
        stream_seed=17,
        estimator_identity=identity,
    )


def _vocabulary(size: int) -> VocabularyIdentity:
    return VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="wt103-test",
        size=size,
        tokenizer_spec_bytes=b"synthetic-wt103-tokenizer",
    )


def test_structured_recognition_is_block_banded_and_differentiable() -> None:
    recognition = WT103StructuredRecognition(
        vocabulary_size=31,
        sequence_length=8,
        latent_width=4,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = recognition(_batch(vocabulary_size=31, length=8))

    assert type(state) is BandedRecognitionState
    assert state.mean.shape == (1, 8, 4)
    assert state.diagonal_factor.shape == (1, 8, 4)
    assert state.lower_blocks.shape == (1, 7, 4, 4)
    assert state.parent_indices.shape == (8, 3)
    assert all(
        not (
            tensor.ndim >= 2
            and tensor.shape[-2:] == (8, 8)
        )
        for tensor in state.tensor_inventory()
    )

    sample = state.rsample(torch.ones_like(state.mean))
    loss = sample.square().mean() - state.entropy().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in recognition.parameters())


def test_complete_and_emission_forward_terms_use_live_arm_modules() -> None:
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=8,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=31,
        sequence_length=8,
        latent_width=4,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    batch = _batch(vocabulary_size=31, length=8)

    complete = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="complete_elbo",
        phase="recognition_adam_proposal",
        batch=batch,
        snapshot=None,
    )
    assert type(complete) is ForwardTerms
    assert complete.objective_kind == "complete_elbo"
    assert len(complete.expected_log_emission or ()) == 8
    complete.loss().backward()
    assert any(parameter.grad is not None for parameter in recognition.parameters())

    recognition.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    snapshot = RecognitionSnapshot.capture(recognition)
    emission = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="emission_only_ablation_non_elbo",
        phase="model_adam_proposal",
        batch=batch,
        snapshot=snapshot,
    )
    assert emission.objective_kind == "emission_only_ablation_non_elbo"
    assert emission.initial_state_kl is None
    source = model.source_observation()
    assert source is not None
    assert source.source_row_count == 2 * (model.sequence_length - 1)
    emission.loss().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_parent_specific_source_offsets_are_ragged_and_anchor_every_full_band() -> None:
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=7,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    history = torch.arange(24, dtype=torch.float64).reshape(1, 6, 4) / 10.0

    state_rows, model_rows = model.source_log_probs(inputs, history)
    for receiver in range(1, inputs.shape[1]):
        width = min(model.source_lookback, receiver)
        assert torch.allclose(
            torch.logsumexp(state_rows[:, receiver, -width:], dim=-1),
            torch.zeros(1, dtype=torch.float64),
            atol=1.0e-13,
            rtol=0.0,
        )
        assert torch.allclose(
            torch.logsumexp(model_rows[:, receiver, -width:], dim=-1),
            torch.zeros(1, dtype=torch.float64),
            atol=1.0e-13,
            rtol=0.0,
        )

    expected_key_elements = sum(
        (min(model.source_lookback, receiver) - 1) * model.latent_width
        for receiver in range(2, model.sequence_length)
    )
    expected_bias_elements = sum(
        min(model.source_lookback, receiver) - 1
        for receiver in range(2, model.sequence_length)
    )
    named = dict(model.named_parameters())
    for bank in ("state", "model"):
        assert sum(
            parameter.numel()
            for name, parameter in named.items()
            if name.startswith(f"{bank}_source_free_keys.")
        ) == expected_key_elements
        assert sum(
            parameter.numel()
            for name, parameter in named.items()
            if name.startswith(f"{bank}_source_free_biases.")
        ) == expected_bias_elements
        assert all(
            parameter.shape[0]
            == min(model.source_lookback, int(name.rsplit(".", 1)[-1])) - 1
            for name, parameter in named.items()
            if name.startswith(
                (
                    f"{bank}_source_free_keys.",
                    f"{bank}_source_free_biases.",
                )
            )
        )


def test_training_and_smc_source_rows_share_the_same_observation_history() -> None:
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=7,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    inputs = torch.tensor([[7, 11, 13, 17, 19, 23]], dtype=torch.int64)
    history = torch.arange(24, dtype=torch.float64).reshape(1, 6, 4) / 11.0
    receiver = 4

    full_state, full_model = model.source_log_probs(inputs, history)
    observed = causal_observation_history(
        inputs,
        receiver_position=receiver,
    )
    next_state, next_model = model.next_source_log_probs(
        observed,
        history[:, :receiver],
    )
    width = min(model.source_lookback, receiver)

    assert torch.equal(observed, inputs[:, :receiver])
    assert torch.equal(full_state[:, receiver, -width:], next_state)
    assert torch.equal(full_model[:, receiver, -width:], next_model)


def test_transition_factor_oracle_uses_two_banks_and_shared_kernels() -> None:
    torch.manual_seed(7)
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=4,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="fixed",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    inputs = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    history = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
            ]
        ],
        dtype=torch.float64,
    )
    state_rows, model_rows = model.source_log_probs(inputs, history)
    next_state, next_model = model.next_source_log_probs(
        inputs[:, :2],
        history[:, :2],
    )
    assert torch.equal(state_rows[:, 2, -2:], next_state)
    assert torch.equal(model_rows[:, 2, -2:], next_model)
    assert set(model.state_source_fixed_logits) == {"2", "3"}
    assert set(model.model_source_fixed_logits) == {"2", "3"}
    assert model.state_source_fixed_logits["2"].data_ptr() != (
        model.model_source_fixed_logits["2"].data_ptr()
    )

    state_parents = history[:, :2, :2]
    model_parents = history[:, :2, 2:]
    current_state = history[:, 2, :2]
    current_model = history[:, 2, 2:]
    state_log, model_log = model.transition_component_log_probs(
        input_ids=inputs[:, 2],
        receiver_position=2,
        state_source_positions=torch.tensor([0, 1], dtype=torch.int64),
        model_source_positions=torch.tensor([0, 1], dtype=torch.int64),
        current_state=current_state,
        current_model=current_model,
        state_parents=state_parents,
        model_parents=model_parents,
    )
    model_mean = model.model_transition_mean(
        input_ids=inputs[:, 2],
        receiver_position=2,
        source_positions=torch.tensor([0, 1], dtype=torch.int64),
        model_parent=model_parents,
    )
    model_scale = torch.exp(model.transition_log_scale[2:])
    direct_model = -(
        0.5
        * ((current_model[:, None] - model_mean) / model_scale).square()
        + torch.log(model_scale)
        + 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    expanded_model = current_model[:, None].expand(-1, 2, -1)
    state_mean = model.state_transition_mean(
        input_ids=inputs[:, 2],
        receiver_position=2,
        source_positions=torch.tensor([0, 1], dtype=torch.int64),
        state_parent=state_parents,
        current_model=expanded_model,
    )
    state_scale = torch.exp(model.transition_log_scale[:2])
    direct_state = -(
        0.5
        * ((current_state[:, None] - state_mean) / state_scale).square()
        + torch.log(state_scale)
        + 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    assert torch.allclose(model_log, direct_model, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(state_log, direct_state, atol=1.0e-12, rtol=0.0)


def test_adamw_and_warmup_cosine_scheduler_are_exact_and_resumable() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = build_adamw(
        (parameter,),
        profile=_optimizer_profile(),
        learning_rate=3.0e-4,
        weight_decay=1.0e-2,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        profile=_scheduler_profile(),
        planned_optimizer_steps=200,
    )

    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1.0e-8
    assert optimizer.defaults["amsgrad"] is False
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False
    initial_state = scheduler.state_dict()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-6)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] <= 3.0e-4

    replacement = build_warmup_cosine_scheduler(
        optimizer,
        profile=_scheduler_profile(),
        planned_optimizer_steps=200,
    )
    replacement.load_state_dict(initial_state)
    assert replacement.state_dict() == initial_state


def test_exact_and_chunked_smc_predictors_are_target_blind_and_normalized(
    monkeypatch,
) -> None:
    autocast_calls: list[dict[str, object]] = []
    real_autocast = torch.autocast

    def injected_autocast(*args, **kwargs):
        autocast_calls.append(dict(kwargs))
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", injected_autocast)
    vocabulary = _vocabulary(31)
    exact_spec, exact_identity, exact_stream = _estimator(
        kind="deterministic_exact",
        particle_count=None,
    )
    no_latent = WT103NoLatentModel(
        vocabulary_size=31,
        sequence_length=8,
        hidden_width=4,
        decoder_chunk_size=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    exact = ExactAutoregressivePriorPredictor(
        model=no_latent,
        vocabulary=vocabulary,
        estimator_spec=exact_spec,
        estimator_identity=exact_identity,
        predictor_config_sha256="1" * 64,
        data_safety_sha256="2" * 64,
    )
    prefix = CausalPrefix.create(
        receiver_t=4,
        vocabulary=vocabulary,
        token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    first = exact.next_token_log_probs(prefix, exact_stream)
    second = exact.next_token_log_probs(prefix, exact_stream)
    assert first.log_probs.raw_bytes_sha256 == second.log_probs.raw_bytes_sha256
    assert torch.allclose(
        torch.logsumexp(first.log_probs.value(), dim=0),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1.0e-13,
        rtol=0.0,
    )

    latent = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=8,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    smc_spec, smc_identity, smc_stream = _estimator(
        kind="weighted_smc",
        particle_count=4,
    )
    smc = WT103ChunkedSmcPriorPredictor(
        model=latent,
        vocabulary=vocabulary,
        estimator_spec=smc_spec,
        estimator_identity=smc_identity,
        predictor_config_sha256="3" * 64,
        data_safety_sha256="4" * 64,
    )
    kernel_calls = {"model": 0, "state": 0}
    real_model_kernel = latent.model_transition_mean
    real_state_kernel = latent.state_transition_mean

    def model_kernel(**kwargs):
        kernel_calls["model"] += 1
        return real_model_kernel(**kwargs)

    def state_kernel(**kwargs):
        kernel_calls["state"] += 1
        return real_state_kernel(**kwargs)

    monkeypatch.setattr(latent, "model_transition_mean", model_kernel)
    monkeypatch.setattr(latent, "state_transition_mean", state_kernel)
    smc_prediction = smc.next_token_log_probs(prefix, smc_stream)
    assert smc.max_observed_particle_decoder_chunk <= 2
    assert smc_prediction.log_probs.shape == (31,)
    assert torch.allclose(
        torch.logsumexp(smc_prediction.log_probs.value(), dim=0),
        torch.tensor(0.0, dtype=torch.float64),
        atol=1.0e-13,
        rtol=0.0,
    )
    assert any(
        call == {
            "device_type": "cpu",
            "dtype": torch.bfloat16,
            "enabled": False,
        }
        for call in autocast_calls
    )
    assert kernel_calls == {"model": 2, "state": 2}


def test_chunked_smc_cache_advances_only_the_new_suffix_and_matches_cold(
    monkeypatch,
) -> None:
    from vfe4.predictive.cache import (
        MarginalPendingPrediction,
        PrefixCache,
    )
    from vfe4.predictive.proposal import ProposalPopulation

    vocabulary = _vocabulary(31)
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=8,
        d_z=2,
        d_m=2,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    spec, identity, stream = _estimator(
        kind="weighted_smc",
        particle_count=4,
    )
    predictor = WT103ChunkedSmcPriorPredictor(
        model=model,
        vocabulary=vocabulary,
        estimator_spec=spec,
        estimator_identity=identity,
        predictor_config_sha256="5" * 64,
        data_safety_sha256="6" * 64,
    )
    short = CausalPrefix.create(
        receiver_t=3,
        vocabulary=vocabulary,
        token_ids=torch.tensor([1, 2], dtype=torch.int64),
    )
    extended = CausalPrefix.create(
        receiver_t=5,
        vocabulary=vocabulary,
        token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
    )
    calls = {"model": 0, "state": 0}
    real_model = model.model_transition_mean
    real_state = model.state_transition_mean

    def counted_model(**kwargs):
        calls["model"] += 1
        return real_model(**kwargs)

    def counted_state(**kwargs):
        calls["state"] += 1
        return real_state(**kwargs)

    monkeypatch.setattr(model, "model_transition_mean", counted_model)
    monkeypatch.setattr(model, "state_transition_mean", counted_state)
    warm_seed = predictor.next_token_log_probs(short, stream)
    calls.update(model=0, state=0)
    warm = predictor.next_token_log_probs(
        extended,
        stream,
        warm_seed.cache,
    )
    warm_calls = dict(calls)
    calls.update(model=0, state=0)
    cold = predictor.next_token_log_probs(extended, stream, None)
    cold_calls = dict(calls)

    assert warm_calls == {"model": 2, "state": 2}
    assert cold_calls == {"model": 3, "state": 3}
    assert warm.log_probs.raw_bytes_sha256 == cold.log_probs.raw_bytes_sha256
    assert warm.cache.cache_sha256 == cold.cache.cache_sha256
    assert warm.cache.counter_consumption == cold.cache.counter_consumption
    assert warm.cache.assimilations == cold.cache.assimilations
    assert warm.cache.filtered_population.particle_count == 4
    assert tuple(
        component.name
        for component in warm.cache.filtered_population.components
    ) == ("ensemble", "history")
    assert warm.cache.filtered_log_weights.shape == (4,)
    assert type(warm.cache.pending) is MarginalPendingPrediction
    assert warm.cache.pending.proposed_population.particle_count == 4
    assert warm.cache.pending.parent_log_weights.shape == (4,)
    assert warm.cache.pending.prediction_log_probs.shape == (
        vocabulary.size,
    )
    assert all(
        component.values.shape[-1] != vocabulary.size
        for component in warm.cache.filtered_population.components
    )

    mismatched_population = ProposalPopulation.create(
        {
            component.name: component.values.value() + 1.0
            for component in warm.cache.filtered_population.components
        }
    )
    mismatched_pending = MarginalPendingPrediction.create(
        prefix_sha256=warm.cache.pending.prefix_sha256,
        proposal_identity_sha256=(
            warm.cache.pending.proposal_identity_sha256
        ),
        proposed_population=mismatched_population,
        parent_log_weights=warm.cache.pending.parent_log_weights.value(),
        prediction_log_probs=warm.cache.pending.prediction_log_probs.value(),
        counter_consumption=warm.cache.pending.counter_consumption,
    )
    with pytest.raises(
        ValueError,
        match="pending prediction does not match the filtered cache key/state",
    ):
        PrefixCache.create(
            key=warm.cache.key,
            filtered_population=warm.cache.filtered_population,
            filtered_log_weights=warm.cache.filtered_log_weights.value(),
            cumulative_log_normalizer=(
                warm.cache.cumulative_log_normalizer
            ),
            pending=mismatched_pending,
            assimilations=warm.cache.assimilations,
            counter_consumption=warm.cache.counter_consumption,
        )


def test_forward_terms_are_raw_batch_sums_and_diagnostics_are_measured(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=4,
        d_z=2,
        d_m=1,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=31,
        sequence_length=4,
        latent_width=3,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    single = _batch(vocabulary_size=31, length=4)
    doubled = _Batch(
        inputs=single.inputs.repeat(2, 1),
        targets=single.targets.repeat(2, 1),
        attention_mask=single.attention_mask.repeat(2, 1),
        counted_targets=2 * single.counted_targets,
    )

    one = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="complete_elbo",
        phase="recognition_adam_proposal",
        batch=single,
        snapshot=None,
    )
    two = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="complete_elbo",
        phase="recognition_adam_proposal",
        batch=doubled,
        snapshot=None,
    )
    one_values = one.detached_values()
    two_values = two.detached_values()
    assert one.partition_schema == "wt103-structured-factor-elbo-v1"
    assert one.estimator_error_bound is None
    assert {
        "initial_model_cross_entropy",
        "initial_state_cross_entropy",
        "model_source_cross_entropy[0]",
        "model_transition_cross_entropy[0]",
        "state_source_cross_entropy[0]",
        "state_transition_cross_entropy[0]",
        "joint_recognition_entropy_estimate",
        "continuous_recognition_entropy",
        "conditional_source_entropy_estimate",
        "model_source_kl[0]",
        "state_source_kl[0]",
    } <= set(one_values)
    assert {
        "initial_model_kl",
        "initial_state_kl",
        "model_transition_kl[0]",
        "state_transition_kl[0]",
        "joint_recognition_entropy",
        "estimator_error_bound",
    }.isdisjoint(one_values)
    reconstructed = math.fsum(
        (
            *(
                value
                for name, value in one_values.items()
                if name.startswith("expected_log_emission[")
            ),
            -one_values["initial_model_cross_entropy"],
            -one_values["initial_state_cross_entropy"],
            *(
                -value
                for name, value in one_values.items()
                if "_source_cross_entropy[" in name
                or "_transition_cross_entropy[" in name
            ),
            one_values["joint_recognition_entropy_estimate"],
        )
    )
    assert reconstructed == pytest.approx(
        one_values["complete_elbo_numerator"]
    )
    assert math.fsum(
        value
        for name, value in one_values.items()
        if name.startswith(("model_source_kl[", "state_source_kl["))
    ) == pytest.approx(
        math.fsum(
            value
            for name, value in one_values.items()
            if "_source_cross_entropy[" in name
        )
        - one_values["conditional_source_entropy_estimate"]
    )
    assert one_values.keys() == two_values.keys()
    for name in one_values:
        assert two_values[name] == pytest.approx(2.0 * one_values[name])
    assert float(two.objective().item()) == pytest.approx(
        float(one.objective().item())
    )

    source = model.source_observation()
    assert source is not None
    assert source.source_row_count == 2 * 2 * (model.sequence_length - 1)
    assert source.support_size_sum == float(
        2
        * 2
        * sum(
            min(model.source_lookback, receiver)
            for receiver in range(1, model.sequence_length)
        )
    )
    state = recognition(doubled)
    assert two_values["joint_recognition_entropy_estimate"] == pytest.approx(
        float(state.entropy(doubled.attention_mask).sum().item())
        + source.entropy_sum
    )
    numerical = state.numerical_observation()
    assert numerical.minimum_cholesky_pivot == pytest.approx(
        float(state.diagonal_factor.amin().item())
    )
    assert numerical.failed_pivots == 0
    assert numerical.nonfinite_count == 0
    assert numerical.solve_residual >= 0.0

    no_latent = WT103NoLatentModel(
        vocabulary_size=31,
        sequence_length=4,
        hidden_width=4,
        decoder_chunk_size=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    one_ce, one_count = no_latent.cross_entropy(single)
    two_ce, two_count = no_latent.cross_entropy(doubled)
    assert two_count == 2 * one_count
    assert float(two_ce.item()) == pytest.approx(2.0 * float(one_ce.item()))


def test_structured_factor_elbo_matches_independent_tensor_and_gradients(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = WT103LatentGenerativeModel(
        vocabulary_size=23,
        sequence_length=4,
        d_z=2,
        d_m=1,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=3,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=23,
        sequence_length=4,
        latent_width=3,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    batch = _batch(vocabulary_size=23, length=4)

    terms = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="complete_elbo",
        phase="recognition_adam_proposal",
        batch=batch,
        snapshot=None,
    )
    reconstructed = _independent_zero_noise_structured_elbo(
        model=model,
        recognition=recognition,
        batch=batch,
    )

    assert terms.expected_log_emission is not None
    for actual, expected in zip(
        terms.expected_log_emission,
        reconstructed.expected_log_emission,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    for name in (
        "model_source_cross_entropy",
        "model_transition_cross_entropy",
        "state_source_cross_entropy",
        "state_transition_cross_entropy",
    ):
        actual_rows = getattr(terms, name)
        expected_rows = getattr(reconstructed, name)
        assert actual_rows is not None
        for actual, expected in zip(
            actual_rows,
            expected_rows,
            strict=True,
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
    for name in (
        "initial_model_cross_entropy",
        "initial_state_cross_entropy",
        "continuous_recognition_entropy",
        "conditional_source_entropy_estimate",
        "joint_recognition_entropy_estimate",
    ):
        actual = getattr(terms, name)
        expected = getattr(reconstructed, name)
        assert actual is not None
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    torch.testing.assert_close(
        terms.objective_numerator(),
        reconstructed.objective_numerator,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    parameters = tuple(model.parameters()) + tuple(
        recognition.parameters()
    )
    actual_gradients = torch.autograd.grad(
        terms.objective_numerator(),
        parameters,
        allow_unused=True,
    )
    reconstructed_gradients = torch.autograd.grad(
        reconstructed.objective_numerator,
        parameters,
        allow_unused=True,
    )
    assert any(gradient is not None for gradient in actual_gradients)
    for parameter, actual, expected in zip(
        parameters,
        actual_gradients,
        reconstructed_gradients,
        strict=True,
    ):
        actual_or_zero = (
            torch.zeros_like(parameter) if actual is None else actual
        )
        expected_or_zero = (
            torch.zeros_like(parameter) if expected is None else expected
        )
        torch.testing.assert_close(
            actual_or_zero,
            expected_or_zero,
            rtol=1.0e-11,
            atol=1.0e-11,
        )


def test_old_correlated_gaussian_local_allocation_can_be_negative() -> None:
    correlation = 0.5
    old_initial_allocation = -0.5 * math.log1p(
        -(correlation * correlation)
    )
    old_transition_allocation = 0.5 * math.log1p(
        -(correlation * correlation)
    )

    assert old_initial_allocation > 0.0
    assert old_transition_allocation < 0.0
    assert old_initial_allocation + old_transition_allocation == pytest.approx(
        0.0,
        abs=1.0e-15,
    )


def test_structured_factor_elbo_handles_mixed_prefix_masks_and_widths(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = WT103LatentGenerativeModel(
        vocabulary_size=29,
        sequence_length=5,
        d_z=2,
        d_m=1,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=29,
        sequence_length=5,
        latent_width=3,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    batch = _Batch(
        inputs=torch.tensor(
            ((1, 2, 3, 4, 5), (6, 7, 0, 0, 0)),
            dtype=torch.int64,
        ),
        targets=torch.tensor(
            ((2, 3, 4, 5, 6), (7, 8, -100, -100, -100)),
            dtype=torch.int64,
        ),
        attention_mask=torch.tensor(
            (
                (True, True, True, True, True),
                (True, True, False, False, False),
            ),
            dtype=torch.bool,
        ),
        counted_targets=7,
    )

    terms = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind="complete_elbo",
        phase="recognition_adam_proposal",
        batch=batch,
        snapshot=None,
    )
    reconstructed = _independent_zero_noise_structured_elbo(
        model=model,
        recognition=recognition,
        batch=batch,
    )

    assert model.d_z != model.d_m
    assert terms.counted_targets == 7
    assert terms.expected_log_emission is not None
    assert len(terms.expected_log_emission) == 5
    torch.testing.assert_close(
        terms.objective_numerator(),
        reconstructed.objective_numerator,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        terms.continuous_recognition_entropy,
        reconstructed.continuous_recognition_entropy,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        terms.joint_recognition_entropy_estimate,
        reconstructed.joint_recognition_entropy_estimate,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    source = model.source_observation()
    assert source is not None
    assert source.source_row_count == 10
    assert source.support_size_sum == 20.0


def test_wt103_structured_schema_is_separate_from_h5_kl_schema() -> None:
    assert (
        WT103_STRUCTURED_FACTOR_ELBO_SCHEMA
        == "wt103-structured-factor-elbo-v1"
    )
    assert (
        WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256
        != H5_OBJECTIVE_SCHEMA_SHA256
    )
    assert {
        "initial_model_kl",
        "initial_state_kl",
        "model_transition_kl[1]",
        "state_transition_kl[1]",
    } <= set(H5_SIGNED_TERM_IDS)

    model_source_cross_entropy = torch.tensor(
        0.5, dtype=torch.float64, requires_grad=True
    )
    state_source_cross_entropy = torch.tensor(
        0.25, dtype=torch.float64, requires_grad=True
    )
    model_source_kl = torch.tensor(
        0.125, dtype=torch.float64, requires_grad=True
    )
    state_source_kl = torch.tensor(
        0.375, dtype=torch.float64, requires_grad=True
    )

    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)

    terms = ForwardTerms.complete_elbo(
        expected_log_emission=(scalar(1.0),),
        initial_model_cross_entropy=scalar(0.25),
        initial_state_cross_entropy=scalar(0.25),
        model_source_cross_entropy=(model_source_cross_entropy,),
        model_transition_cross_entropy=(scalar(0.125),),
        state_source_cross_entropy=(state_source_cross_entropy,),
        state_transition_cross_entropy=(scalar(0.125),),
        model_source_kl=(model_source_kl,),
        state_source_kl=(state_source_kl,),
        continuous_recognition_entropy=scalar(0.5),
        conditional_source_entropy_estimate=scalar(0.25),
        joint_recognition_entropy_estimate=scalar(0.75),
        estimator_error_bound=None,
        counted_targets=1,
    )
    values = terms.detached_values()
    assert terms.partition_schema == WT103_STRUCTURED_FACTOR_ELBO_SCHEMA
    assert {
        "initial_model_kl",
        "initial_state_kl",
        "model_transition_kl[1]",
        "state_transition_kl[1]",
    }.isdisjoint(values)
    gradients = torch.autograd.grad(
        terms.objective_numerator(),
        (
            model_source_kl,
            state_source_kl,
            model_source_cross_entropy,
            state_source_cross_entropy,
        ),
        allow_unused=True,
    )
    assert gradients[:2] == (None, None)
    assert gradients[2] is not None
    assert gradients[3] is not None
    torch.testing.assert_close(gradients[2], scalar(-1.0))
    torch.testing.assert_close(gradients[3], scalar(-1.0))


@pytest.mark.parametrize(
    "objective_kind",
    ("complete_elbo", "emission_only_ablation_non_elbo"),
)
def test_one_target_latent_batch_records_canonical_zero_source_rows(
    objective_kind,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=1,
        d_z=2,
        d_m=1,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=1,
        particle_chunk_size=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=31,
        sequence_length=1,
        latent_width=3,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    terms = compute_wt103_forward_terms(
        model=model,
        recognition=recognition,
        objective_kind=objective_kind,
        phase="recognition_adam_proposal",
        batch=_batch(vocabulary_size=31, length=1),
        snapshot=None,
    )

    assert terms.counted_targets == 1
    assert torch.isfinite(terms.objective())
    source = model.source_observation()
    assert source is not None
    assert source.entropy_sum == 0.0
    assert source.source_row_count == 0
    assert source.support_size_sum == 0.0


def test_initializer_provenance_names_and_binds_every_parameter() -> None:
    from vfe4.training.wt103_runtime import _initialize_named_substreams

    model = WT103LatentGenerativeModel(
        vocabulary_size=31,
        sequence_length=4,
        d_z=2,
        d_m=1,
        source_lookback=3,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=4,
        particle_chunk_size=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=31,
        sequence_length=4,
        latent_width=3,
        source_lookback=3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    provenance = _initialize_named_substreams(
        root_seed=2026072101,
        arm_id="WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1",
        modules={"model": model, "recognition": recognition},
    )
    parameters = {
        **{
            f"model.{name}": parameter
            for name, parameter in model.named_parameters()
        },
        **{
            f"recognition.{name}": parameter
            for name, parameter in recognition.named_parameters()
        },
    }
    rows = {row.name: row for row in provenance.substreams}

    assert set(rows) == {*parameters, "training_rng"}
    assert len(rows) == len(provenance.substreams)
    for name, parameter in parameters.items():
        row = rows[name]
        assert row.target_shape == tuple(parameter.shape)
        assert row.target_dtype == str(parameter.dtype)
        assert row.initialized_value_sha256 == hashlib.sha256(
            parameter.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes(order="C")
        ).hexdigest()
        assert row.terminal_counter == (
            parameter.numel()
            if row.initializer_class == "xavier_uniform_gain_1"
            else 0
        )

    assert rows["model.state_frame_generator"].initializer_class == (
        "identity_frame_via_zero_generator"
    )
    assert rows["recognition.raw_diagonal"].initializer_class == (
        "identity_block_precision_diagonal_factor"
    )
    assert rows["recognition.raw_lower"].initializer_class == (
        "zero_block_precision_lower_factor"
    )
    assert rows[
        "model.state_source_free_keys.2"
    ].initializer_class == "zero_source_parameter"
    assert rows[
        "model.decoder.projection.bias"
    ].initializer_class == "zero_linear_bias"
    assert rows[
        "model.token_embedding.weight"
    ].initializer_class == "xavier_uniform_gain_1"
    assert rows["training_rng"].initializer_class == "training_rng_seed"
    assert torch.equal(model.state_frame_generator, torch.zeros_like(
        model.state_frame_generator
    ))
    assert torch.equal(recognition.raw_lower, torch.zeros_like(
        recognition.raw_lower
    ))
    assert torch.equal(
        F.softplus(recognition.raw_diagonal) + 1.0e-4,
        torch.ones_like(recognition.raw_diagonal),
    )
