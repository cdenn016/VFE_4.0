from __future__ import annotations

import math

import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.generative import (
    FixedSourcePrior,
    MaskCaseKey,
    NormalizedSourceFactor,
    PrefixConditionedSourcePrior,
)
from vfe4.numerics import (
    AllInvalidSourceRowError,
    MaskedLogProbabilities,
    masked_log_softmax_from_parents,
)
from vfe4.types import (
    CausalDag,
    CausalDagRow,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
    FrozenTensorSnapshot,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _structure() -> H6LanguageStructure:
    dag = CausalDag.create(
        node_labels=(0, 1, 2, 3, 4),
        rows=tuple(
            CausalDagRow(receiver_t, tuple(range(receiver_t)))
            for receiver_t in range(1, 5)
        ),
    )
    return H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2, 3, 4),
    )


def _vocabulary(size: int) -> VocabularyIdentity:
    vocabulary_id = "h6-prefix-small-v1" if size == 3 else "wikitext-2-byte-v1"
    return VocabularyIdentity(vocabulary_id, size, SHA_A)


def _rows() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.linspace(-0.3, 0.4, receiver_t, dtype=torch.float64)
        for receiver_t in range(1, 5)
    )


def _fixed(vocabulary: VocabularyIdentity) -> FixedSourcePrior:
    return FixedSourcePrior(
        structure=_structure(),
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        state_logits=_rows(),
        model_logits=tuple(row + 0.2 for row in _rows()),
    )


def test_masked_log_softmax_masks_before_normalization_and_preserves_gradients() -> None:
    logits = torch.tensor([0.0, 1_000.0, 1.0], dtype=torch.float64, requires_grad=True)

    result = masked_log_softmax_from_parents(logits, (0, 2), 3)

    assert type(result) is MaskedLogProbabilities
    assert result.support_mask.tolist() == [True, False, True]
    assert torch.isneginf(result.log_probs[1])
    assert result.log_probs.exp()[1].item() == 0.0
    assert torch.logsumexp(result.log_probs, dim=0).item() == pytest.approx(0.0, abs=1e-15)
    assert result.log_probs[[0, 2]].tolist() == pytest.approx(
        torch.log_softmax(torch.tensor([0.0, 1.0], dtype=torch.float64), dim=0).tolist()
    )

    result.log_probs[0].backward()
    assert logits.grad is not None
    assert logits.grad[1].item() == 0.0


@pytest.mark.parametrize(
    ("parents", "receiver_t", "error"),
    [
        ((), 2, AllInvalidSourceRowError),
        ((0, 2), 2, ValueError),
        ((0, 1, 1), 3, ValueError),
        ((1, 0), 3, ValueError),
    ],
)
def test_masked_log_softmax_rejects_empty_or_noncausal_support(
    parents: tuple[int, ...], receiver_t: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        masked_log_softmax_from_parents(
            torch.zeros(receiver_t, dtype=torch.float64), parents, receiver_t
        )


def test_fixed_source_prior_normalizes_both_banks_and_returns_bound_identities() -> None:
    prior = _fixed(_vocabulary(3))

    state = prior.state_source_log_probs(receiver_t=3)
    model = prior.model_source_log_probs(receiver_t=3)

    assert type(state) is NormalizedSourceFactor
    assert type(state.mask_case_key) is MaskCaseKey
    assert state.mask_case_key.prior_variant == "fixed"
    assert state.mask_case_key.bank == "state"
    assert model.mask_case_key.bank == "model"
    assert state.mask_case_key.context_sha256 == model.mask_case_key.context_sha256
    assert state.factor_identity_sha256 != model.factor_identity_sha256
    assert state.support_mask == (True, True, True)
    assert len(prior.state_source_free_logits) == 3
    assert prior.state_source_free_logits[1].shape == (2,)
    assert prior.model_source_free_logits is not None
    assert len(prior.model_source_free_logits) == 3
    assert prior.model_source_free_logits[1].shape == (2,)
    assert all(
        parameter.numel() > 0
        for parameter in prior.parameters()
    )
    assert torch.logsumexp(state.log_probs.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-15
    )
    assert torch.logsumexp(model.log_probs.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-15
    )


def test_normalized_source_factor_rejects_forgery_drift_and_invalid_rows() -> None:
    factor = _fixed(_vocabulary(3)).state_source_log_probs(receiver_t=3)
    changed = FrozenTensorSnapshot.capture(
        torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))
    )

    with pytest.raises(ValueError, match="factor identity"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            factor.support_mask,
            factor.log_probs,
            "0" * 64,
        )
    with pytest.raises(ValueError, match="factor identity"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            factor.support_mask,
            changed,
            factor.factor_identity_sha256,
        )
    with pytest.raises(ValueError, match="off-support"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            (True, False, True),
            FrozenTensorSnapshot.capture(
                torch.log(torch.tensor([0.4, 0.2, 0.4], dtype=torch.float64))
            ),
            "0" * 64,
        )
    with pytest.raises(ValueError, match="normalized"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            (True, True, True),
            FrozenTensorSnapshot.capture(torch.zeros(3, dtype=torch.float64)),
            "0" * 64,
        )
    with pytest.raises(ValueError, match="float64"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            factor.support_mask,
            FrozenTensorSnapshot.capture(torch.zeros(3, dtype=torch.float32)),
            "0" * 64,
        )
    with pytest.raises(ValueError, match="finite"):
        NormalizedSourceFactor(
            factor.mask_case_key,
            factor.support_mask,
            FrozenTensorSnapshot.capture(
                torch.tensor([float("nan"), -1.0, -1.0], dtype=torch.float64)
            ),
            "0" * 64,
        )


def test_normalized_source_factor_clone_preserves_autograd() -> None:
    prior = _fixed(_vocabulary(3))
    factor = prior.state_source_log_probs(receiver_t=3)

    factor.log_probs.value()[0].backward()

    assert prior.state_source_free_logits[1].grad is not None
    assert torch.isfinite(prior.state_source_free_logits[1].grad).all()


def test_prefix_prior_uses_only_typed_prior_tokens_and_earlier_latents() -> None:
    vocabulary = _vocabulary(3)
    prior = PrefixConditionedSourcePrior(
        structure=_structure(),
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        latent_dim=2,
        context_dim=2,
    )
    with torch.no_grad():
        prior.token_embedding.weight.copy_(
            torch.tensor([[0.0, 0.0], [1.0, -1.0], [-0.5, 0.75]], dtype=torch.float64)
        )
        prior.state_source_free_parent_keys[1].copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        )
        prior.model_source_free_parent_keys[1].copy_(
            torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)
        )

    first = CausalPrefix.create(
        receiver_t=3,
        vocabulary=vocabulary,
        token_ids=torch.tensor([0, 1], dtype=torch.int64),
    )
    second = CausalPrefix.create(
        receiver_t=3,
        vocabulary=vocabulary,
        token_ids=torch.tensor([0, 2], dtype=torch.int64),
    )
    history = torch.tensor([[0.1, 0.2], [0.3, -0.1], [0.0, 0.4]], dtype=torch.float64)

    first_state = prior.state_source_log_probs(prefix=first, earlier_latents=history)
    second_state = prior.state_source_log_probs(prefix=second, earlier_latents=history)
    first_model = prior.model_source_log_probs(prefix=first, earlier_latents=history)

    assert first_state.mask_case_key.receiver_t == 3
    assert first_state.mask_case_key.context_sha256 != second_state.mask_case_key.context_sha256
    assert not torch.equal(first_state.log_probs.value(), second_state.log_probs.value())
    assert torch.logsumexp(first_state.log_probs.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-15
    )
    assert torch.logsumexp(first_model.log_probs.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-15
    )

    with pytest.raises(ValueError, match="CausalPrefix"):
        prior.state_source_log_probs(  # type: ignore[arg-type]
            prefix=torch.tensor([0, 1, 2]), earlier_latents=history
        )
    with pytest.raises(ValueError, match="earlier_latents"):
        prior.state_source_log_probs(
            prefix=first, earlier_latents=torch.zeros((4, 2), dtype=torch.float64)
        )


def test_frozen_mask_inventory_arithmetic_is_exact_without_large_enumeration() -> None:
    fixed_per_bank = 4
    prefix_per_bank = 2 * sum(3 ** (receiver_t - 1) for receiver_t in range(1, 5))

    assert 2 * fixed_per_bank + 2 * prefix_per_bank == 168
    assert 2 * 2 * 4_096 == 16_384
    assert math.fsum([1.0, 3.0, 9.0, 27.0]) == 40.0
