"""Frozen carried-weight bootstrap SMC recursion for H6 prior prediction."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import EstimatorSpec

from .cache import (
    AssimilationRecord,
    PendingPrediction,
    PrefixCache,
    PrefixCacheKey,
)
from .identities import EstimatorIdentity
from .prior import EstimatorRecord, PriorPrediction
from .proposal import (
    CounterConsumption,
    CounterKey,
    CounterPurpose,
    EstimatorStream,
    ManagedForwardGraphProposal,
    ProposalPopulation,
    TargetFreeProposalAdapter,
)


def _normalized_log_weights(value: Tensor) -> Tensor:
    if (
        type(value) is not Tensor
        or value.device.type != "cpu"
        or value.dtype is not torch.float64
        or value.ndim != 1
        or value.numel() <= 0
        or bool(torch.isnan(value).any())
        or bool(torch.isposinf(value).any())
    ):
        raise ValueError(
            "log weights must be one nonempty CPU float64 row"
        )
    if not math.isclose(
        float(torch.logsumexp(value, dim=0).item()),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-13,
    ):
        raise ValueError("carried log weights must be normalized")
    return value


def weighted_mixture_log_probs(
    normalized_log_weights: Tensor, emission_log_probs: Tensor
) -> Tensor:
    """Compute the public weighted pre-observation vocabulary mixture."""

    weights = _normalized_log_weights(normalized_log_weights)
    if (
        type(emission_log_probs) is not Tensor
        or emission_log_probs.device.type != "cpu"
        or emission_log_probs.dtype is not torch.float64
        or emission_log_probs.ndim != 2
        or emission_log_probs.shape[0] != weights.numel()
        or emission_log_probs.shape[1] <= 0
        or not bool(torch.isfinite(emission_log_probs).all())
    ):
        raise ValueError(
            "emissions must be finite CPU float64 shape (particles, vocabulary)"
        )
    if not bool(
        torch.allclose(
            torch.logsumexp(emission_log_probs, dim=1),
            torch.zeros(emission_log_probs.shape[0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-13,
        )
    ):
        raise ValueError("every emission row must be normalized")
    mixture = torch.logsumexp(
        weights[:, None] + emission_log_probs, dim=0
    )
    if not math.isclose(
        float(torch.logsumexp(mixture, dim=0).item()),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-13,
    ):
        raise ValueError("weighted emission mixture failed normalization")
    return mixture


@dataclass(frozen=True)
class WeightUpdate:
    normalized_log_weights: Tensor
    log_normalizer: float
    ess: float


def assimilate_log_weights(
    parent_log_weights: Tensor, selected_emission_log_probs: Tensor
) -> WeightUpdate:
    """Assimilate exactly one formerly predicted token after public scoring."""

    parent = _normalized_log_weights(parent_log_weights)
    if (
        type(selected_emission_log_probs) is not Tensor
        or selected_emission_log_probs.device.type != "cpu"
        or selected_emission_log_probs.dtype is not torch.float64
        or selected_emission_log_probs.shape != parent.shape
        or not bool(torch.isfinite(selected_emission_log_probs).all())
    ):
        raise ValueError(
            "selected emissions must be one finite float64 value per particle"
        )
    unnormalized = parent + selected_emission_log_probs
    log_normalizer_tensor = torch.logsumexp(unnormalized, dim=0)
    if not bool(torch.isfinite(log_normalizer_tensor)):
        raise ValueError("incremental SMC log normalizer must be finite")
    normalized = unnormalized - log_normalizer_tensor
    probabilities = torch.exp(normalized)
    ess = 1.0 / float(torch.sum(probabilities.square()).item())
    return WeightUpdate(
        normalized,
        float(log_normalizer_tensor.item()),
        ess,
    )


def systematic_ancestors(
    normalized_log_weights: Tensor, *, offset: float
) -> Tensor:
    """Apply the frozen left-search systematic-resampling rule."""

    weights = _normalized_log_weights(normalized_log_weights)
    particle_count = weights.numel()
    if (
        type(offset) is not float
        or not math.isfinite(offset)
        or not 0.0 <= offset < 1.0 / particle_count
    ):
        raise ValueError(
            "systematic offset must lie in [0, 1/particle_count)"
        )
    cumulative = torch.cumsum(torch.exp(weights), dim=0)
    cumulative[-1] = 1.0
    points = (
        torch.arange(particle_count, dtype=torch.float64) / particle_count
        + offset
    )
    return torch.searchsorted(cumulative, points, right=False).to(
        dtype=torch.int64
    )


class BootstrapSmcPredictor:
    """The sole target-blind carried-weight bootstrap predictor."""

    def __init__(
        self,
        *,
        proposal: TargetFreeProposalAdapter,
        estimator_spec: EstimatorSpec,
        estimator_identity: EstimatorIdentity,
        predictor_config_sha256: str,
        data_safety_sha256: str,
    ) -> None:
        required = (
            "vocabulary",
            "vocabulary_sha256",
            "model_family_sha256",
            "model_state_sha256",
            "proposal_identity_sha256",
            "assert_current_state",
            "initialize",
            "propagate",
        )
        if any(not hasattr(proposal, name) for name in required):
            raise ValueError(
                "proposal must implement the explicit target-free proposal adapter"
            )
        if getattr(proposal, "proposal_mode", None) != "generative_bootstrap":
            raise ValueError(
                "only the declared generative bootstrap proposal is supported"
            )
        if type(estimator_spec) is not EstimatorSpec:
            raise ValueError("estimator_spec must be an exact EstimatorSpec")
        estimator_spec.__post_init__()
        if (
            estimator_spec.kind != "weighted_smc"
            or estimator_spec.resampling != "systematic_ess_half"
            or type(estimator_spec.particle_count) is not int
        ):
            raise ValueError(
                "BootstrapSmcPredictor requires weighted systematic-ESS-half SMC"
            )
        if type(estimator_identity) is not EstimatorIdentity:
            raise ValueError(
                "estimator_identity must be an exact EstimatorIdentity"
            )
        estimator_identity.__post_init__()
        if (
            estimator_identity.semantic_sha256
            != estimator_spec.estimator_sha256
        ):
            raise ValueError(
                "estimator semantic identity does not match its specification"
            )
        for value, name in (
            (predictor_config_sha256, "predictor_config_sha256"),
            (data_safety_sha256, "data_safety_sha256"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        proposal.assert_current_state()
        self.proposal = proposal
        self.estimator_spec = estimator_spec
        self.estimator_identity = estimator_identity
        self.predictor_config_sha256 = predictor_config_sha256
        self.data_safety_sha256 = data_safety_sha256
        self.vocabulary = proposal.vocabulary
        self.vocabulary_sha256 = proposal.vocabulary_sha256
        self.model_family_sha256 = proposal.model_family_sha256
        self.model_state_sha256 = proposal.model_state_sha256
        self.proposal_identity_sha256 = proposal.proposal_identity_sha256
        self.particle_count = estimator_spec.particle_count

    def _validate_stream(self, stream: EstimatorStream) -> None:
        if type(stream) is not EstimatorStream:
            raise ValueError("estimator_rng must be an exact EstimatorStream")
        stream.__post_init__()
        if (
            stream.estimator_semantic_sha256
            != self.estimator_identity.semantic_sha256
            or stream.estimator_artifact_bytes_sha256
            != self.estimator_identity.artifact_bytes_sha256
            or stream.estimator_identity_sha256
            != self.estimator_identity.identity_sha256
        ):
            raise ValueError(
                "estimator stream identity does not match the predictor"
            )

    def _validate_prefix(self, prefix: CausalPrefix) -> None:
        if type(prefix) is not CausalPrefix:
            raise ValueError(
                "prefix_tokens must be an exact target-free CausalPrefix"
            )
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("prefix vocabulary does not match the predictor")

    def _cache_key(
        self, prefix: CausalPrefix, stream: EstimatorStream
    ) -> PrefixCacheKey:
        return PrefixCacheKey.create(
            prefix=prefix,
            vocabulary_sha256=self.vocabulary_sha256,
            predictor_config_sha256=self.predictor_config_sha256,
            model_family_sha256=self.model_family_sha256,
            model_state_sha256=self.model_state_sha256,
            proposal_identity_sha256=self.proposal_identity_sha256,
            estimator_semantic_sha256=(
                self.estimator_identity.semantic_sha256
            ),
            estimator_artifact_bytes_sha256=(
                self.estimator_identity.artifact_bytes_sha256
            ),
            estimator_stream_sha256=stream.stream_sha256,
            data_safety_sha256=self.data_safety_sha256,
        )

    def _validate_cache(
        self,
        cache: PrefixCache,
        requested_prefix: CausalPrefix,
        stream: EstimatorStream,
    ) -> None:
        if type(cache) is not PrefixCache:
            raise ValueError("cache must be an exact immutable PrefixCache")
        cache.__post_init__()
        key = cache.key
        checks = (
            (
                key.vocabulary_sha256,
                self.vocabulary_sha256,
                "cache vocabulary identity does not match the predictor",
            ),
            (
                key.predictor_config_sha256,
                self.predictor_config_sha256,
                "cache predictor config identity does not match the predictor",
            ),
            (
                key.model_family_sha256,
                self.model_family_sha256,
                "cache model family identity does not match the predictor",
            ),
            (
                key.model_state_sha256,
                self.model_state_sha256,
                "cache model state identity does not match the predictor",
            ),
            (
                key.proposal_identity_sha256,
                self.proposal_identity_sha256,
                "cache proposal identity does not match the predictor",
            ),
            (
                key.estimator_semantic_sha256,
                self.estimator_identity.semantic_sha256,
                "cache estimator semantic identity does not match the predictor",
            ),
            (
                key.estimator_artifact_bytes_sha256,
                self.estimator_identity.artifact_bytes_sha256,
                "cache estimator artifact identity does not match the predictor",
            ),
            (
                key.estimator_stream_sha256,
                stream.stream_sha256,
                "cache estimator stream identity does not match the call",
            ),
            (
                key.data_safety_sha256,
                self.data_safety_sha256,
                "cache data safety identity does not match the predictor",
            ),
        )
        for observed, expected, message in checks:
            if observed != expected:
                raise ValueError(message)
        requested_tokens = tuple(
            int(value) for value in requested_prefix.token_ids.tolist()
        )
        if (
            len(key.prefix_tokens) > len(requested_tokens)
            or requested_tokens[: len(key.prefix_tokens)] != key.prefix_tokens
        ):
            raise ValueError(
                "cache prefix is not an exact causal prefix of the requested call"
            )
        reconstructed = CausalPrefix.create(
            receiver_t=len(key.prefix_tokens) + 1,
            vocabulary=self.vocabulary,
            token_ids=torch.tensor(key.prefix_tokens, dtype=torch.int64),
        )
        if reconstructed.prefix_sha256 != key.prefix_sha256:
            raise ValueError("cache prefix identity is stale")

    def _predict_cache(
        self,
        *,
        prefix: CausalPrefix,
        stream: EstimatorStream,
        filtered_population: ProposalPopulation,
        filtered_log_weights: Tensor,
        cumulative_log_normalizer: float,
        assimilations: tuple[AssimilationRecord, ...],
        prior_consumption: tuple[CounterConsumption, ...],
    ) -> PrefixCache:
        step = self.proposal.propagate(
            filtered_population, prefix, stream
        )
        if (
            step.proposal_identity_sha256
            != self.proposal_identity_sha256
        ):
            raise ValueError("proposal step identity does not match predictor")
        mixture = weighted_mixture_log_probs(
            filtered_log_weights, step.emission_log_probs.value()
        )
        pending = PendingPrediction.create(
            prefix_sha256=prefix.prefix_sha256,
            step=step,
            parent_log_weights=filtered_log_weights,
            prediction_log_probs=mixture,
        )
        all_consumption = prior_consumption + step.counter_consumption
        return PrefixCache.create(
            key=self._cache_key(prefix, stream),
            filtered_population=step.population,
            filtered_log_weights=filtered_log_weights,
            cumulative_log_normalizer=cumulative_log_normalizer,
            pending=pending,
            assimilations=assimilations,
            counter_consumption=all_consumption,
        )

    def _initial_cache(
        self, stream: EstimatorStream
    ) -> PrefixCache:
        empty = CausalPrefix.create(
            receiver_t=1,
            vocabulary=self.vocabulary,
            token_ids=torch.empty(0, dtype=torch.int64),
        )
        population, consumption = self.proposal.initialize(
            empty, stream, self.particle_count
        )
        log_weights = torch.full(
            (self.particle_count,),
            -math.log(self.particle_count),
            dtype=torch.float64,
        )
        return self._predict_cache(
            prefix=empty,
            stream=stream,
            filtered_population=population,
            filtered_log_weights=log_weights,
            cumulative_log_normalizer=0.0,
            assimilations=(),
            prior_consumption=consumption,
        )

    def _advance(
        self,
        cache: PrefixCache,
        next_prefix: CausalPrefix,
        stream: EstimatorStream,
    ) -> PrefixCache:
        previous_count = len(cache.key.prefix_tokens)
        next_tokens = tuple(
            int(value) for value in next_prefix.token_ids.tolist()
        )
        if (
            len(next_tokens) != previous_count + 1
            or next_tokens[:-1] != cache.key.prefix_tokens
        ):
            raise ValueError(
                "cache advancement requires exactly one newly appended token"
            )
        observed_token = next_tokens[-1]
        pending = cache.pending
        update = assimilate_log_weights(
            pending.parent_log_weights.value(),
            pending.emission_log_probs.value()[:, observed_token],
        )
        position = previous_count + 1
        ancestors: tuple[int, ...] = ()
        resampling_consumption: CounterConsumption | None = None
        extra_consumption: tuple[CounterConsumption, ...] = ()
        if update.ess < self.particle_count / 2.0:
            key = CounterKey(
                stream.stream_seed,
                next_prefix.prefix_sha256,
                position,
                CounterPurpose.SYSTEMATIC_RESAMPLING,
                0,
            )
            offset = stream.systematic_offset(
                key, particle_count=self.particle_count
            )
            ancestor_tensor = systematic_ancestors(
                update.normalized_log_weights, offset=offset
            )
            ancestors = tuple(int(value) for value in ancestor_tensor.tolist())
            filtered_population = pending.proposed_population.select(
                ancestor_tensor
            )
            filtered_weights = torch.full(
                (self.particle_count,),
                -math.log(self.particle_count),
                dtype=torch.float64,
            )
            resampling_consumption = CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.SYSTEMATIC_RESAMPLING,
                particle_count=1,
                draws_per_particle=1,
            )
            extra_consumption = (resampling_consumption,)
        else:
            filtered_population = pending.proposed_population
            filtered_weights = update.normalized_log_weights
        assimilation = AssimilationRecord.create(
            position=position,
            observed_token=observed_token,
            incremental_log_normalizer=update.log_normalizer,
            ess=update.ess,
            ancestors=ancestors,
            resampling_consumption=resampling_consumption,
        )
        cumulative = math.fsum(
            [
                cache.cumulative_log_normalizer,
                update.log_normalizer,
            ]
        )
        return self._predict_cache(
            prefix=next_prefix,
            stream=stream,
            filtered_population=filtered_population,
            filtered_log_weights=filtered_weights,
            cumulative_log_normalizer=cumulative,
            assimilations=cache.assimilations + (assimilation,),
            prior_consumption=cache.counter_consumption + extra_consumption,
        )

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction:
        scope = (
            self.proposal.live_forward_graph()
            if isinstance(self.proposal, ManagedForwardGraphProposal)
            else nullcontext()
        )
        with scope:
            return self._next_token_log_probs(
                prefix_tokens,
                estimator_rng,
                cache,
            )

    def _next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None,
    ) -> PriorPrediction:
        self.proposal.assert_current_state()
        self._validate_prefix(prefix_tokens)
        self._validate_stream(estimator_rng)
        requested_tokens = tuple(
            int(value) for value in prefix_tokens.token_ids.tolist()
        )
        if cache is None:
            current = self._initial_cache(estimator_rng)
            start = 0
        else:
            self._validate_cache(cache, prefix_tokens, estimator_rng)
            current = cache
            start = len(cache.key.prefix_tokens)
        for length in range(start + 1, len(requested_tokens) + 1):
            intermediate = CausalPrefix.create(
                receiver_t=length + 1,
                vocabulary=self.vocabulary,
                token_ids=torch.tensor(
                    requested_tokens[:length], dtype=torch.int64
                ),
            )
            current = self._advance(current, intermediate, estimator_rng)
        record = EstimatorRecord.from_cache(
            stream=estimator_rng, cache=current
        )
        return PriorPrediction.create(
            vocabulary=self.vocabulary,
            log_probs=current.pending.prediction_log_probs.value(),
            cache=current,
            estimator_record=record,
        )


__all__ = [
    "BootstrapSmcPredictor",
    "WeightUpdate",
    "assimilate_log_weights",
    "systematic_ancestors",
    "weighted_mixture_log_probs",
]
