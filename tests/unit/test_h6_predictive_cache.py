from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.generative import FixedSourcePrior, LanguageGenerativeModel
from vfe4.predictive import (
    AssimilationRecord,
    BootstrapSmcPredictor,
    EstimatorIdentity,
    EstimatorStream,
    LanguageGenerativeProposalAdapter,
    PrefixCache,
)
from vfe4.predictive.smc import WeightUpdate
from vfe4.types import (
    CausalDag,
    CausalDagRow,
    EstimatorSpec,
    H6LanguageStructure,
    VocabularyIdentity,
    ZeroDimensionalBase,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _model(*, vocabulary: VocabularyIdentity | None = None) -> LanguageGenerativeModel:
    vocabulary = vocabulary or VocabularyIdentity("h6-prefix-small-v1", 3, SHA_A)
    structure = H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=CausalDag.create(
            node_labels=(0, 1, 2, 3),
            rows=tuple(
                CausalDagRow(receiver_t, tuple(range(receiver_t)))
                for receiver_t in range(1, 4)
            ),
        ),
        receiver_labels=(1, 2, 3),
    )
    rows = tuple(
        torch.zeros(receiver_t, dtype=torch.float64)
        for receiver_t in range(1, 4)
    )
    prior = FixedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=SHA_A,
        predictor_config_sha256=SHA_B,
        model_family_sha256=SHA_C,
        state_logits=rows,
        model_logits=rows,
    )
    model = LanguageGenerativeModel(
        structure=structure,
        vocabulary=vocabulary,
        model_family_sha256=SHA_C,
        latent_dim=1,
        source_prior=prior,
    )
    with torch.no_grad():
        model.initial_log_scale.fill_(-0.2)
        model.model_transition_log_scale.fill_(-0.5)
        model.state_transition_log_scale.fill_(-0.5)
        model.emission_state_weight.copy_(
            torch.tensor([[0.35], [-0.2], [0.05]], dtype=torch.float64)
        )
        model.emission_model_weight.copy_(
            torch.tensor([[0.15], [0.05], [-0.2]], dtype=torch.float64)
        )
    return model


def _build(
    model: LanguageGenerativeModel,
    *,
    config_sha256: str = SHA_B,
    data_safety_sha256: str = SHA_D,
    particles: int = 8,
) -> tuple[BootstrapSmcPredictor, EstimatorIdentity]:
    spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=particles,
        resampling="systematic_ess_half",
    )
    identity = EstimatorIdentity.from_spec(spec)
    return (
        BootstrapSmcPredictor(
            proposal=LanguageGenerativeProposalAdapter(model),
            estimator_spec=spec,
            estimator_identity=identity,
            predictor_config_sha256=config_sha256,
            data_safety_sha256=data_safety_sha256,
        ),
        identity,
    )


def _stream(identity: EstimatorIdentity) -> EstimatorStream:
    return EstimatorStream.create(stream_seed=17, estimator_identity=identity)


def _prefix(
    vocabulary: VocabularyIdentity, token_ids: tuple[int, ...]
) -> CausalPrefix:
    return CausalPrefix.create(
        receiver_t=len(token_ids) + 1,
        vocabulary=vocabulary,
        token_ids=torch.tensor(token_ids, dtype=torch.int64),
    )


def test_prefix_cache_replays_exactly_and_carries_weighted_filter_state() -> None:
    model = _model()
    predictor, identity = _build(model)
    stream = _stream(identity)

    at_start = predictor.next_token_log_probs(
        _prefix(model.vocabulary, ()), stream
    )
    warm = predictor.next_token_log_probs(
        _prefix(model.vocabulary, (1,)), stream, at_start.cache
    )
    repeated = predictor.next_token_log_probs(
        _prefix(model.vocabulary, (1,)), stream, warm.cache
    )
    cold = predictor.next_token_log_probs(
        _prefix(model.vocabulary, (1,)), _stream(identity)
    )

    assert type(warm.cache) is PrefixCache
    assert warm.log_probs.raw_bytes_sha256 == repeated.log_probs.raw_bytes_sha256
    assert warm.log_probs.raw_bytes_sha256 == cold.log_probs.raw_bytes_sha256
    assert warm.cache.cache_sha256 == cold.cache.cache_sha256
    assert warm.cache.key.prefix_tokens == (1,)
    assert warm.cache.key.prefix_sha256 == _prefix(
        model.vocabulary, (1,)
    ).prefix_sha256
    assert warm.cache.key.estimator_semantic_sha256 == identity.semantic_sha256
    assert (
        warm.cache.key.estimator_artifact_bytes_sha256
        == identity.artifact_bytes_sha256
    )
    assert torch.logsumexp(
        warm.cache.filtered_log_weights.value(), dim=0
    ).item() == pytest.approx(0.0, abs=1e-14)
    assert warm.cache.cumulative_log_normalizer == pytest.approx(
        at_start.log_probs.value()[1].item(), abs=1e-14
    )
    assert warm.cache.pending.prefix_sha256 == warm.cache.key.prefix_sha256
    assert warm.cache.pending.parent_log_weights.shape == (8,)
    with pytest.raises(FrozenInstanceError):
        warm.cache.cumulative_log_normalizer = 0.0  # type: ignore[misc]


def test_cache_and_stream_identity_mismatches_fail_closed() -> None:
    model = _model()
    predictor, identity = _build(model)
    at_start = predictor.next_token_log_probs(
        _prefix(model.vocabulary, ()), _stream(identity)
    )
    cache = predictor.next_token_log_probs(
        _prefix(model.vocabulary, (1,)), _stream(identity), at_start.cache
    ).cache

    with pytest.raises(ValueError, match="cache prefix"):
        predictor.next_token_log_probs(
            _prefix(model.vocabulary, (2,)), _stream(identity), cache
        )

    wrong_config, wrong_config_identity = _build(model, config_sha256=SHA_E)
    with pytest.raises(ValueError, match="predictor config"):
        wrong_config.next_token_log_probs(
            _prefix(model.vocabulary, ()), _stream(wrong_config_identity), cache
        )

    wrong_safety, wrong_safety_identity = _build(
        model, data_safety_sha256=SHA_E
    )
    with pytest.raises(ValueError, match="data safety"):
        wrong_safety.next_token_log_probs(
            _prefix(model.vocabulary, ()), _stream(wrong_safety_identity), cache
        )

    changed_model = _model()
    with torch.no_grad():
        changed_model.emission_bias[0].add_(0.1)
    wrong_state, wrong_state_identity = _build(changed_model)
    with pytest.raises(ValueError, match="model state"):
        wrong_state.next_token_log_probs(
            _prefix(changed_model.vocabulary, ()), _stream(wrong_state_identity), cache
        )

    wrong_estimator, wrong_estimator_identity = _build(model, particles=6)
    with pytest.raises(ValueError, match="estimator"):
        wrong_estimator.next_token_log_probs(
            _prefix(model.vocabulary, ()),
            _stream(wrong_estimator_identity),
            cache,
        )

    other_vocabulary = VocabularyIdentity("other-vocabulary-v1", 3, SHA_E)
    other_model = _model(vocabulary=other_vocabulary)
    other_predictor, other_identity = _build(other_model)
    with pytest.raises(ValueError, match="vocabulary"):
        other_predictor.next_token_log_probs(
            _prefix(other_vocabulary, ()), _stream(other_identity), cache
        )

    with pytest.raises(ValueError, match="estimator"):
        predictor.next_token_log_probs(
            _prefix(model.vocabulary, ()),
            _stream(wrong_estimator_identity),
            cache,
        )


def test_prefix_cache_rejects_inconsistent_assimilation_history() -> None:
    model = _model()
    predictor, identity = _build(model)
    stream = _stream(identity)
    initial = predictor.next_token_log_probs(
        _prefix(model.vocabulary, ()), stream
    )
    cache = predictor.next_token_log_probs(
        _prefix(model.vocabulary, (1,)), stream, initial.cache
    ).cache
    original = cache.assimilations[0]

    def recreate(
        assimilation: AssimilationRecord,
        *,
        cumulative: float = cache.cumulative_log_normalizer,
    ) -> PrefixCache:
        return PrefixCache.create(
            key=cache.key,
            filtered_population=cache.filtered_population,
            filtered_log_weights=cache.filtered_log_weights.value(),
            cumulative_log_normalizer=cumulative,
            pending=cache.pending,
            assimilations=(assimilation,),
            counter_consumption=cache.counter_consumption,
        )

    with pytest.raises(ValueError, match="sequential prefix tokens"):
        recreate(
            AssimilationRecord.create(
                position=2,
                observed_token=original.observed_token,
                incremental_log_normalizer=original.incremental_log_normalizer,
                ess=original.ess,
                ancestors=original.ancestors,
                resampling_consumption=original.resampling_consumption,
            )
        )
    with pytest.raises(ValueError, match="sequential prefix tokens"):
        recreate(
            AssimilationRecord.create(
                position=original.position,
                observed_token=2,
                incremental_log_normalizer=original.incremental_log_normalizer,
                ess=original.ess,
                ancestors=original.ancestors,
                resampling_consumption=original.resampling_consumption,
            )
        )
    with pytest.raises(ValueError, match="cumulative log normalizer"):
        recreate(
            original,
            cumulative=math.nextafter(
                cache.cumulative_log_normalizer, math.inf
            ),
        )


def test_resampling_counter_binds_post_assimilation_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    predictor, identity = _build(model, particles=4)
    stream = _stream(identity)
    initial = predictor.next_token_log_probs(
        _prefix(model.vocabulary, ()), stream
    )
    next_prefix = _prefix(model.vocabulary, (1,))
    uniform = torch.full((4,), -math.log(4), dtype=torch.float64)
    captured: dict[str, str] = {}

    def forced_update(
        _parent: torch.Tensor, _selected: torch.Tensor
    ) -> WeightUpdate:
        return WeightUpdate(uniform, -0.25, 1.0)

    def capture_offset(
        _stream: EstimatorStream,
        key: object,
        *,
        particle_count: int,
    ) -> float:
        assert particle_count == 4
        captured["prefix_sha256"] = key.prefix_sha256  # type: ignore[attr-defined]
        return 0.0

    monkeypatch.setattr(
        "vfe4.predictive.smc.assimilate_log_weights", forced_update
    )
    monkeypatch.setattr(EstimatorStream, "systematic_offset", capture_offset)

    predictor.next_token_log_probs(next_prefix, stream, initial.cache)

    assert captured["prefix_sha256"] == next_prefix.prefix_sha256
