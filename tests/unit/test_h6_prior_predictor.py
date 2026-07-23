from __future__ import annotations

import inspect

import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.generative import FixedSourcePrior, LanguageGenerativeModel
from vfe4.predictive import (
    BootstrapSmcPredictor,
    EstimatorIdentity,
    EstimatorStream,
    LanguageGenerativeProposalAdapter,
    PriorPrediction,
    canonical_model_state_sha256,
    vocabulary_identity_sha256,
)
from vfe4.types import (
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    FrozenTensorSnapshot,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _model() -> LanguageGenerativeModel:
    dag = CausalDag.create(
        node_labels=(0, 1, 2, 3),
        rows=tuple(
            CausalDagRow(receiver_t, tuple(range(receiver_t)))
            for receiver_t in range(1, 4)
        ),
    )
    structure = H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2, 3),
    )
    vocabulary = VocabularyIdentity("h6-prefix-small-v1", 3, SHA_A)
    rows = tuple(
        torch.linspace(-0.2, 0.2, receiver_t, dtype=torch.float64)
        for receiver_t in range(1, 4)
    )
    prior = FixedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        state_logits=rows,
        model_logits=tuple(row.flip(0) for row in rows),
    )
    model = LanguageGenerativeModel(
        structure=structure,
        vocabulary=vocabulary,
        model_family_sha256=SHA_C,
        latent_dim=2,
        source_prior=prior,
    )
    with torch.no_grad():
        model.initial_log_scale.fill_(-0.4)
        model.model_transition_weight.copy_(
            torch.tensor([[0.65, 0.05], [-0.1, 0.55]], dtype=torch.float64)
        )
        model.state_transition_weight.copy_(
            torch.tensor([[0.5, 0.1], [0.05, 0.6]], dtype=torch.float64)
        )
        model.state_model_weight.copy_(
            torch.tensor([[0.2, 0.0], [0.0, -0.15]], dtype=torch.float64)
        )
        model.model_transition_log_scale.fill_(-0.7)
        model.state_transition_log_scale.fill_(-0.6)
        model.emission_state_weight.copy_(
            torch.tensor([[0.25, -0.1], [-0.15, 0.2], [0.05, 0.05]], dtype=torch.float64)
        )
        model.emission_model_weight.copy_(
            torch.tensor([[0.1, 0.05], [0.0, -0.1], [-0.1, 0.05]], dtype=torch.float64)
        )
        model.emission_bias.copy_(
            torch.tensor([0.08, -0.03, -0.05], dtype=torch.float64)
        )
    return model


def _predictor(
    model: LanguageGenerativeModel,
) -> tuple[BootstrapSmcPredictor, EstimatorIdentity, EstimatorStream]:
    spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=8,
        resampling="systematic_ess_half",
    )
    identity = EstimatorIdentity.from_spec(spec)
    predictor = BootstrapSmcPredictor(
        proposal=LanguageGenerativeProposalAdapter(model),
        estimator_spec=spec,
        estimator_identity=identity,
        predictor_config_sha256=SHA_B,
        data_safety_sha256=SHA_D,
    )
    return predictor, identity, EstimatorStream.create(
        stream_seed=2026072300, estimator_identity=identity
    )


def test_prior_predictor_is_target_blind_identity_bound_and_snapshot_safe() -> None:
    model = _model()
    predictor, identity, stream = _predictor(model)
    prefix = CausalPrefix.create(
        receiver_t=1,
        vocabulary=model.vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64),
    )

    assert tuple(
        inspect.signature(predictor.next_token_log_probs).parameters
    ) == ("prefix_tokens", "estimator_rng", "cache")
    forbidden = {"target", "targets", "suffix", "full_window", "recognition", "posterior"}
    for method in (
        predictor.next_token_log_probs,
        predictor.proposal.initialize,
        predictor.proposal.propagate,
    ):
        assert forbidden.isdisjoint(inspect.signature(method).parameters)

    first = predictor.next_token_log_probs(prefix, stream)
    second = predictor.next_token_log_probs(
        prefix,
        EstimatorStream.create(
            stream_seed=2026072300, estimator_identity=identity
        ),
    )

    assert type(first) is PriorPrediction
    assert type(first.log_probs) is FrozenTensorSnapshot
    assert first.log_probs.shape == (3,)
    assert first.log_probs.dtype == "float64"
    assert torch.logsumexp(first.log_probs.value(), dim=0).item() == pytest.approx(
        0.0, abs=1e-14
    )
    assert first.log_probs.raw_bytes_sha256 == second.log_probs.raw_bytes_sha256
    assert first.prediction_sha256 == second.prediction_sha256
    assert first.estimator_record.estimator_semantic_sha256 == identity.semantic_sha256
    assert (
        first.estimator_record.estimator_artifact_bytes_sha256
        == identity.artifact_bytes_sha256
    )
    assert identity.semantic_sha256 != identity.artifact_bytes_sha256
    assert vocabulary_identity_sha256(model.vocabulary) == model.vocabulary_sha256
    assert canonical_model_state_sha256(model) == predictor.model_state_sha256

    returned = first.log_probs.value()
    before = first.log_probs.raw_bytes_sha256
    returned.add_(100.0)
    assert first.log_probs.raw_bytes_sha256 == before
    assert not torch.equal(returned, first.log_probs.value())
    first.log_probs.value()[0].backward()
    assert model.emission_state_weight.grad is not None
    assert bool(torch.isfinite(model.emission_state_weight.grad).all())

    owned = first.log_probs._FrozenTensorSnapshot__owned
    with torch.no_grad():
        owned.add_(1.0)
    with pytest.raises(ValueError, match="integrity"):
        first.log_probs.value()


def test_predictor_rejects_a_model_state_changed_after_adapter_freeze() -> None:
    model = _model()
    predictor, _, stream = _predictor(model)
    prefix = CausalPrefix.create(
        receiver_t=1,
        vocabulary=model.vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64),
    )
    frozen_digest = predictor.model_state_sha256

    with torch.no_grad():
        model.emission_bias[0].add_(0.125)

    assert canonical_model_state_sha256(model) != frozen_digest
    with pytest.raises(ValueError, match="model state"):
        predictor.next_token_log_probs(prefix, stream)
