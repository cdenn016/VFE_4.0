from __future__ import annotations

import inspect
import math

import pytest
import torch

from vfe4.generative import (
    FixedSourcePrior,
    LanguageGenerativeModel,
    NormalizedLanguageFactor,
    NormalizedSourceFactor,
    ParentSpecificPooledPrefixSourcePrior,
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


def _prior(structure: H6LanguageStructure, vocabulary: VocabularyIdentity) -> FixedSourcePrior:
    rows = tuple(
        torch.linspace(-0.2, 0.2, receiver_t, dtype=torch.float64)
        for receiver_t in range(1, 5)
    )
    return FixedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        state_logits=rows,
        model_logits=tuple(row + 0.1 for row in rows),
    )


def _model(vocabulary_size: int, *, source_free: bool = False) -> LanguageGenerativeModel:
    structure = _structure()
    vocabulary = _vocabulary(vocabulary_size)
    return LanguageGenerativeModel(
        structure=structure,
        vocabulary=vocabulary,
        model_family_sha256=SHA_C,
        latent_dim=2,
        source_prior=None if source_free else _prior(structure, vocabulary),
    )


@pytest.mark.parametrize("vocabulary_size", [3, 258])
def test_language_emission_is_a_vocabulary_sized_normalized_factor(
    vocabulary_size: int,
) -> None:
    model = _model(vocabulary_size)

    factor = model.emission_log_probs(
        receiver_t=2,
        current_state=torch.tensor([0.2, -0.1], dtype=torch.float64),
        current_model=torch.tensor([-0.3, 0.4], dtype=torch.float64),
    )

    assert type(factor) is NormalizedLanguageFactor
    assert factor.partition == "emission"
    assert factor.log_values.shape == (vocabulary_size,)
    assert torch.logsumexp(factor.log_values.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-15
    )
    assert len(factor.factor_identity_sha256) == 64


def test_initial_and_transition_factors_are_normalized_gaussian_log_densities() -> None:
    model = _model(3)
    zero_pair = torch.zeros((2, 2), dtype=torch.float64)
    zero_history = torch.zeros((3, 2), dtype=torch.float64)

    initial = model.initial_log_prob(initial_latents=zero_pair)
    model_transition = model.model_transition_log_prob(
        receiver_t=3,
        current_model=torch.zeros(2, dtype=torch.float64),
        earlier_models=zero_history,
        source_index=1,
    )
    state_transition = model.state_transition_log_prob(
        receiver_t=3,
        current_state=torch.zeros(2, dtype=torch.float64),
        current_model=torch.zeros(2, dtype=torch.float64),
        earlier_states=zero_history,
        source_index=1,
    )

    assert initial.partition == "initial"
    assert initial.receiver_t == 0
    assert initial.log_values.value().item() == pytest.approx(
        -0.5 * 4 * math.log(2.0 * math.pi), abs=1e-12
    )
    for factor in (model_transition, state_transition):
        assert factor.log_values.shape == ()
        assert factor.log_values.value().item() == pytest.approx(
            -0.5 * 2 * math.log(2.0 * math.pi), abs=1e-12
        )


def test_normalized_language_factor_rejects_forgery_drift_and_wrong_semantics() -> None:
    factor = _model(3).emission_log_probs(
        receiver_t=2,
        current_state=torch.zeros(2, dtype=torch.float64),
        current_model=torch.zeros(2, dtype=torch.float64),
    )
    changed = FrozenTensorSnapshot.capture(
        torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))
    )

    with pytest.raises(ValueError, match="factor identity"):
        NormalizedLanguageFactor(
            factor.receiver_t,
            factor.partition,
            factor.vocabulary_size,
            factor.factor_context_sha256,
            "0" * 64,
            factor.log_values,
        )
    with pytest.raises(ValueError, match="factor identity"):
        NormalizedLanguageFactor(
            factor.receiver_t,
            factor.partition,
            factor.vocabulary_size,
            factor.factor_context_sha256,
            factor.factor_identity_sha256,
            changed,
        )
    with pytest.raises(ValueError, match="normalized"):
        NormalizedLanguageFactor(
            1,
            "emission",
            3,
            SHA_A,
            "0" * 64,
            FrozenTensorSnapshot.capture(torch.zeros(3, dtype=torch.float64)),
        )
    with pytest.raises(ValueError, match="scalar"):
        NormalizedLanguageFactor(
            1,
            "state_transition",
            3,
            SHA_A,
            "0" * 64,
            FrozenTensorSnapshot.capture(torch.zeros(1, dtype=torch.float64)),
        )
    with pytest.raises(ValueError, match="vocabulary shape"):
        NormalizedLanguageFactor(
            1,
            "emission",
            3,
            SHA_A,
            "0" * 64,
            FrozenTensorSnapshot.capture(
                torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        NormalizedLanguageFactor(
            1,
            "model_transition",
            3,
            SHA_A,
            "0" * 64,
            FrozenTensorSnapshot.capture(
                torch.tensor(float("inf"), dtype=torch.float64)
            ),
        )


def test_normalized_language_factor_clone_preserves_autograd() -> None:
    model = _model(3)
    factor = model.emission_log_probs(
        receiver_t=2,
        current_state=torch.tensor([0.2, -0.1], dtype=torch.float64),
        current_model=torch.tensor([-0.3, 0.4], dtype=torch.float64),
    )

    factor.log_values.value()[0].backward()

    assert model.emission_state_weight.grad is not None
    assert torch.isfinite(model.emission_state_weight.grad).all()


def test_language_model_delegates_both_normalized_source_banks() -> None:
    model = _model(3)

    state = model.state_source_log_probs(receiver_t=4)
    model_bank = model.model_source_log_probs(receiver_t=4)

    assert type(state) is NormalizedSourceFactor
    assert type(model_bank) is NormalizedSourceFactor
    assert state.mask_case_key.bank == "state"
    assert model_bank.mask_case_key.bank == "model"


def test_a3_source_free_model_has_no_source_parameters_and_uses_predecessor() -> None:
    model = _model(3, source_free=True)

    assert model.source_prior is None
    assert all("source_prior" not in name for name, _ in model.named_parameters())
    with pytest.raises(ValueError, match="structurally absent"):
        model.state_source_log_probs(receiver_t=2)
    with pytest.raises(ValueError, match="immediate predecessor"):
        model.model_transition_log_prob(
            receiver_t=3,
            current_model=torch.zeros(2, dtype=torch.float64),
            earlier_models=torch.zeros((3, 2), dtype=torch.float64),
            source_index=1,
        )

    factor = model.model_transition_log_prob(
        receiver_t=3,
        current_model=torch.zeros(2, dtype=torch.float64),
        earlier_models=torch.zeros((3, 2), dtype=torch.float64),
        source_index=None,
    )
    assert factor.partition == "model_transition"


def test_language_model_rejects_source_prior_identity_and_latent_mismatches() -> None:
    structure = _structure()
    vocabulary = _vocabulary(3)
    wrong_family = FixedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_B,
        state_logits=tuple(
            torch.zeros(receiver_t, dtype=torch.float64)
            for receiver_t in range(1, 5)
        ),
        model_logits=tuple(
            torch.zeros(receiver_t, dtype=torch.float64)
            for receiver_t in range(1, 5)
        ),
    )
    with pytest.raises(ValueError, match="model_family_sha256"):
        LanguageGenerativeModel(
            structure=structure,
            vocabulary=vocabulary,
            model_family_sha256=SHA_C,
            latent_dim=2,
            source_prior=wrong_family,
        )

    wrong_latent = ParentSpecificPooledPrefixSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        latent_dim=3,
        context_dim=2,
    )
    with pytest.raises(ValueError, match="latent_dim"):
        LanguageGenerativeModel(
            structure=structure,
            vocabulary=vocabulary,
            model_family_sha256=SHA_C,
            latent_dim=2,
            source_prior=wrong_latent,
        )


def test_generative_surface_is_target_free_and_keeps_base_separate_from_dag() -> None:
    model = _model(3)
    forbidden = {"target", "targets", "suffix", "recognition", "full_window"}
    methods = (
        model.state_source_log_probs,
        model.model_source_log_probs,
        model.initial_log_prob,
        model.model_transition_log_prob,
        model.state_transition_log_prob,
        model.emission_log_probs,
    )

    for method in methods:
        assert forbidden.isdisjoint(inspect.signature(method).parameters)
    assert model.structure.base is not model.structure.dag
    assert model.structure.base.dimension == 0
    assert model.structure.dag.rows[0] == CausalDagRow(1, (0,))
