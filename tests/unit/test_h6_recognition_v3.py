from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from vfe4.data.windows import CausalPrefix
from vfe4.predictive import canonical_model_state_sha256
from vfe4.recognition import (
    AbsentSourceBank,
    CategoricalSourceBank,
    LanguageRecognitionParameterStore,
    RecognitionConditioning,
    RecognitionPriorFeature,
    RecognitionPriorFeatureProvider,
)
from vfe4.types.h6 import VocabularyIdentity


STATE_SUPPORTS = ((0,), (0, 1), (0, 2))
MODEL_SUPPORTS = ((0,), (1,), (0, 1, 2))


class _TestPriorFeatureProvider(RecognitionPriorFeatureProvider):
    def __init__(
        self,
        *,
        state_supports: tuple[tuple[int, ...], ...] = STATE_SUPPORTS,
        model_supports: tuple[tuple[int, ...], ...] = MODEL_SUPPORTS,
        tilt: float = 0.0,
        inject_zero_mass: bool = False,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor((0.75,), dtype=torch.float64))
        self.state_supports = state_supports
        self.model_supports = model_supports
        self.tilt = tilt
        self.inject_zero_mass = inject_zero_mass
        self.seen_prefixes: list[tuple[str, CausalPrefix]] = []

    def forward(
        self,
        *,
        bank: str,
        causal_prefix: CausalPrefix,
        earlier_recognition_means: Tensor,
    ) -> RecognitionPriorFeature:
        self.seen_prefixes.append((bank, causal_prefix))
        rows = (
            self.state_supports if bank == "state" else self.model_supports
        )
        support = rows[causal_prefix.receiver_t - 1]
        safe_support = tuple(
            source_j
            if 0 <= source_j < earlier_recognition_means.shape[0]
            else 0
            for source_j in support
        )
        indices = torch.tensor(
            safe_support,
            dtype=torch.int64,
            device=earlier_recognition_means.device,
        )
        channel = 0 if bank == "state" else 2
        values = (
            self.scale
            * earlier_recognition_means.index_select(0, indices)[:, channel]
        )
        values = values + self.tilt * torch.arange(
            len(support),
            dtype=torch.float64,
            device=values.device,
        )
        if self.inject_zero_mass and len(support) > 1:
            values = values.clone()
            values[-1] = -torch.inf
        return RecognitionPriorFeature(
            bank=bank,
            causal_prefix_sha256=causal_prefix.prefix_sha256,
            support=support,
            log_prior_features=values,
        )


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity.from_tokenizer_spec(
        vocabulary_id="h6-v3-tests",
        size=19,
        tokenizer_spec_bytes=b"h6-recognition-v3",
    )


def _conditioning(
    tokens: tuple[int, ...], *, mode: str
) -> RecognitionConditioning:
    return RecognitionConditioning.create(
        mode=mode,
        horizon=len(tokens),
        observed_tokens=torch.tensor(tokens, dtype=torch.int64),
    )


def _store(
    *,
    mode: str = "filtering",
    channel_count: int = 2,
    family: str = "structured",
    trainable_source_banks: tuple[str, ...] = ("state", "model"),
) -> LanguageRecognitionParameterStore:
    return LanguageRecognitionParameterStore(
        vocabulary=_vocabulary(),
        horizon=3,
        latent_width=2,
        recognition_width=5,
        channel_count=channel_count,
        family=family,
        conditioning_mode=mode,
        trainable_source_banks=trainable_source_banks,
    )


def _trajectory(
    store: LanguageRecognitionParameterStore,
    conditioning: RecognitionConditioning,
    *,
    provider: RecognitionPriorFeatureProvider | None = None,
):
    return store.recognition_trajectory(
        conditioning,
        prior_feature_provider=provider or _TestPriorFeatureProvider(),
    )


def test_receiver_trajectory_filtering_never_reads_future_tokens() -> None:
    store = _store(mode="filtering")
    provider = _TestPriorFeatureProvider()
    baseline = _trajectory(
        store,
        _conditioning((2, 5, 7), mode="filtering"),
        provider=provider,
    )
    changed_future = _trajectory(
        store, _conditioning((2, 5, 13), mode="filtering")
    )

    assert baseline.receiver_labels == (0, 1, 2, 3)
    torch.testing.assert_close(
        baseline.contexts[:3], changed_future.contexts[:3]
    )
    assert not torch.equal(baseline.contexts[3], changed_future.contexts[3])

    expected_prefix = torch.tensor((2, 5, 7), dtype=torch.int64)
    assert len(provider.seen_prefixes) == 6
    for bank, prefix in provider.seen_prefixes:
        assert bank in ("state", "model")
        assert type(prefix) is CausalPrefix
        torch.testing.assert_close(
            prefix.token_ids, expected_prefix[: prefix.receiver_t - 1]
        )

    if torch.cuda.is_available():
        cuda_store = _store(mode="filtering").to(device="cuda")
        cuda_provider = _TestPriorFeatureProvider().to(device="cuda")
        cuda_trajectory = _trajectory(
            cuda_store,
            _conditioning((2, 5, 7), mode="filtering"),
            provider=cuda_provider,
        )
        assert cuda_trajectory.contexts.device.type == "cuda"


def test_smoothing_contexts_are_receiver_distinct() -> None:
    store = _store(mode="smoothing", family="factorized")
    provider = _TestPriorFeatureProvider()
    trajectory = _trajectory(
        store,
        _conditioning((3, 4, 8), mode="smoothing"),
        provider=provider,
    )

    assert trajectory.position_descriptor_schema == (
        "frozen_float64_sinusoidal_receiver_position_v1"
    )
    assert trajectory.family == "factorized"
    assert trajectory.block_sizes == (2, 2)
    assert trajectory.contexts.shape == (4, 5)
    assert torch.unique(trajectory.contexts, dim=0).shape[0] == 4
    context_identities = tuple(
        context.context_identity_sha256
        for context in trajectory.receiver_contexts
    )
    assert len(set(context_identities)) == 4
    assert trajectory.recognition_store_state_sha256 == (
        canonical_model_state_sha256(store)
    )
    assert trajectory.source_model_state_sha256 == (
        canonical_model_state_sha256(provider)
    )

    original_contexts = trajectory.contexts
    exported_contexts = trajectory.contexts
    with torch.no_grad():
        exported_contexts.zero_()
    torch.testing.assert_close(trajectory.contexts, original_contexts)
    with pytest.raises(ValueError, match="trajectory identity"):
        replace(
            trajectory,
            recognition_store_state_sha256="0" * 64,
        )


def test_terminal_component_rows_are_normalized_and_non_degenerate() -> None:
    store = _store()
    trajectory = _trajectory(
        store, _conditioning((1, 6, 9), mode="filtering")
    )

    assert isinstance(trajectory.state_source, CategoricalSourceBank)
    assert isinstance(trajectory.model_source, CategoricalSourceBank)
    for bank in (trajectory.state_source, trajectory.model_source):
        for row in bank.rows:
            torch.testing.assert_close(
                torch.logsumexp(row.log_probabilities, dim=0),
                row.log_probabilities.new_zeros(()),
            )

    terminal = trajectory.terminal_components
    assert len(terminal) == (
        len(STATE_SUPPORTS[-1]) * len(MODEL_SUPPORTS[-1])
    )
    weights = torch.stack([component.probability for component in terminal])
    torch.testing.assert_close(weights.sum(), weights.new_ones(()))
    means = torch.stack([component.mean for component in terminal])
    assert torch.unique(means, dim=0).shape[0] > 1
    assert all(
        component.precision_identity_sha256
        == trajectory.shared_precision_identity_sha256
        for component in terminal
    )

    original_mean = terminal[0].mean
    exported_mean = terminal[0].mean
    with torch.no_grad():
        exported_mean.add_(100.0)
    torch.testing.assert_close(terminal[0].mean, original_mean)
    original_row = trajectory.state_source.rows[-1].log_probabilities
    exported_row = trajectory.state_source.rows[-1].log_probabilities
    with torch.no_grad():
        exported_row.zero_()
    torch.testing.assert_close(
        trajectory.state_source.rows[-1].log_probabilities, original_row
    )


def test_earlier_receivers_are_source_independent() -> None:
    store = _store()
    conditioning = _conditioning((4, 5, 6), mode="filtering")
    baseline = _trajectory(
        store,
        conditioning,
        provider=_TestPriorFeatureProvider(),
    )
    tilted = _trajectory(
        store,
        conditioning,
        provider=_TestPriorFeatureProvider(tilt=3.0),
    )
    for receiver_t in range(store.horizon):
        baseline_components = baseline.receiver_components[receiver_t]
        tilted_components = tilted.receiver_components[receiver_t]
        assert len(baseline_components) == len(tilted_components) == 1
        assert baseline_components[0].state_source_j is None
        assert baseline_components[0].model_source_j is None
        torch.testing.assert_close(
            baseline_components[0].mean, tilted_components[0].mean
        )

    baseline_weights = torch.stack(
        [component.probability for component in baseline.terminal_components]
    )
    tilted_weights = torch.stack(
        [component.probability for component in tilted.terminal_components]
    )
    assert not torch.allclose(baseline_weights, tilted_weights)


def test_source_rows_reject_zero_prior_mass_and_future_parents() -> None:
    store = _store()
    conditioning = _conditioning((1, 2, 3), mode="filtering")

    future_provider = _TestPriorFeatureProvider(
        state_supports=((0,), (0, 2), (0, 1))
    )
    with pytest.raises(ValueError, match="strictly earlier"):
        _trajectory(store, conditioning, provider=future_provider)

    zero_mass_provider = _TestPriorFeatureProvider(inject_zero_mass=True)
    with pytest.raises(ValueError, match="positive prior mass"):
        _trajectory(store, conditioning, provider=zero_mass_provider)


def test_recognition_parameter_inventory_has_one_owner_per_live_bank() -> None:
    store = _store()
    source_parameters = {
        name: parameter
        for name, parameter in store.named_parameters()
        if name.startswith("source_")
    }
    expected_names = {
        "source_residual_vectors.state",
        "source_residual_vectors.model",
        "source_lag_scalars.state",
        "source_lag_scalars.model",
        "source_shift_vectors.state",
        "source_shift_vectors.model",
    }

    assert source_parameters.keys() == expected_names
    assert sum(parameter.numel() for parameter in source_parameters.values()) == (
        2 * (store.recognition_width + 1 + store.latent_width)
    )
    assert len({parameter.data_ptr() for parameter in source_parameters.values()}) == 6
    assert all(
        bool(torch.any(parameter.detach() != 0.0))
        for parameter in source_parameters.values()
    )

    provider = _TestPriorFeatureProvider()
    trajectory = _trajectory(
        store,
        _conditioning((2, 7, 11), mode="filtering"),
        provider=provider,
    )
    loss = sum(
        row.entropy
        for bank in (trajectory.state_source, trajectory.model_source)
        if isinstance(bank, CategoricalSourceBank)
        for row in bank.rows
    )
    loss.backward()
    assert provider.scale.grad is None
    assert store.mean_weight.grad is not None
    assert bool(torch.any(store.mean_weight.grad != 0.0))


def test_absent_source_bank_is_parameter_free_entropy_free_singleton() -> None:
    store = _store(
        channel_count=1, trainable_source_banks=("state",)
    )
    trajectory = _trajectory(
        store, _conditioning((2, 3, 5), mode="filtering")
    )

    absent = trajectory.model_source
    assert isinstance(absent, AbsentSourceBank)
    assert absent.support == (None,)
    torch.testing.assert_close(absent.probability, absent.probability.new_ones(()))
    torch.testing.assert_close(absent.entropy, absent.entropy.new_zeros(()))
    assert absent.parameter_count == 0
    assert not any(
        "model" in name
        for name, _parameter in store.named_parameters()
        if name.startswith("source_")
    )
    assert len(trajectory.terminal_components) == len(STATE_SUPPORTS[-1])
