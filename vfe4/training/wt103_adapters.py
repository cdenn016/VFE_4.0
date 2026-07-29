"""Target-blind WikiText-103 ``PriorPredictor`` adapters."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn import functional as F

from vfe4.data.windows import CausalPrefix
from vfe4.predictive.cache import (
    AssimilationRecord,
    MarginalPendingPrediction,
    PendingPrediction,
    PrefixCache,
    PrefixCacheKey,
)
from vfe4.predictive.identities import (
    EstimatorIdentity,
    canonical_model_state_sha256,
    vocabulary_identity_sha256,
)
from vfe4.predictive.prior import EstimatorRecord, PriorPrediction
from vfe4.predictive.proposal import (
    CounterConsumption,
    CounterKey,
    CounterPurpose,
    EstimatorStream,
    ProposalPopulation,
    ProposalStep,
)
from vfe4.predictive.smc import (
    assimilate_log_weights,
    systematic_ancestors,
)
from vfe4.types.h6 import EstimatorSpec, VocabularyIdentity
from vfe4.types.training import owned_sha256

from .wt103_models import WT103A0Model
from .wt103_runtime import (
    WT103LatentGenerativeModel,
    WT103NoLatentModel,
    causal_observation_history,
)


def _bf16_autocast(model: torch.nn.Module):
    parameter = next(model.parameters())
    return torch.autocast(
        device_type=parameter.device.type,
        dtype=torch.bfloat16,
        enabled=(
            parameter.device.type == "cuda"
            and parameter.dtype is torch.float32
        ),
    )


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_common(
    *,
    model: torch.nn.Module,
    vocabulary: VocabularyIdentity,
    estimator_spec: EstimatorSpec,
    estimator_identity: EstimatorIdentity,
    predictor_config_sha256: str,
    data_safety_sha256: str,
) -> None:
    if not isinstance(model, torch.nn.Module):
        raise ValueError("predictor model must be a torch module")
    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("predictor vocabulary must be exact")
    vocabulary.__post_init__()
    if type(estimator_spec) is not EstimatorSpec:
        raise ValueError("predictor estimator specification must be exact")
    estimator_spec.__post_init__()
    if type(estimator_identity) is not EstimatorIdentity:
        raise ValueError("predictor estimator identity must be exact")
    estimator_identity.__post_init__()
    if estimator_spec.estimator_sha256 != estimator_identity.semantic_sha256:
        raise ValueError("estimator identity does not bind its specification")
    _sha256(predictor_config_sha256, "predictor_config_sha256")
    _sha256(data_safety_sha256, "data_safety_sha256")


class _PredictorBase:
    proposal_mode = "generative_bootstrap"

    def _initialize_common(
        self,
        *,
        model: torch.nn.Module,
        vocabulary: VocabularyIdentity,
        estimator_spec: EstimatorSpec,
        estimator_identity: EstimatorIdentity,
        predictor_config_sha256: str,
        data_safety_sha256: str,
        adapter_kind: str,
    ) -> None:
        _validate_common(
            model=model,
            vocabulary=vocabulary,
            estimator_spec=estimator_spec,
            estimator_identity=estimator_identity,
            predictor_config_sha256=predictor_config_sha256,
            data_safety_sha256=data_safety_sha256,
        )
        if getattr(model, "vocabulary_size", None) != vocabulary.size:
            raise ValueError("model and vocabulary sizes differ")
        self.model = model
        self.vocabulary = vocabulary
        self.vocabulary_sha256 = vocabulary_identity_sha256(vocabulary)
        self.estimator_spec = estimator_spec
        self.estimator_identity = estimator_identity
        self.predictor_config_sha256 = predictor_config_sha256
        self.data_safety_sha256 = data_safety_sha256
        self.model_family_sha256 = owned_sha256(
            "vfe4.wt103.predictor-model-family.v1",
            {
                "adapter_kind": adapter_kind,
                "model_type": type(model).__name__,
                "vocabulary_size": vocabulary.size,
                "sequence_length": getattr(
                    model,
                    "sequence_length",
                    getattr(model, "positional_capacity", None),
                ),
            },
        )
        self.model_state_sha256 = canonical_model_state_sha256(model)
        self.proposal_identity_sha256 = owned_sha256(
            "vfe4.wt103.prior-proposal-adapter.v1",
            {
                "adapter_kind": adapter_kind,
                "model_family_sha256": self.model_family_sha256,
                "model_state_sha256": self.model_state_sha256,
                "estimator_sha256": estimator_spec.estimator_sha256,
            },
        )

    def assert_current_state(self) -> None:
        if canonical_model_state_sha256(self.model) != self.model_state_sha256:
            raise ValueError(
                "predictor model changed; construct a new state-bound adapter"
            )

    def _validate_call(
        self,
        prefix: CausalPrefix,
        stream: EstimatorStream,
        cache: PrefixCache | None,
    ) -> tuple[int, ...]:
        self.assert_current_state()
        if type(prefix) is not CausalPrefix:
            raise ValueError("predictor requires an exact CausalPrefix")
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("prefix vocabulary differs from predictor")
        if type(stream) is not EstimatorStream:
            raise ValueError("predictor requires an exact EstimatorStream")
        stream.__post_init__()
        if (
            stream.estimator_semantic_sha256
            != self.estimator_identity.semantic_sha256
            or stream.estimator_artifact_bytes_sha256
            != self.estimator_identity.artifact_bytes_sha256
            or stream.estimator_identity_sha256
            != self.estimator_identity.identity_sha256
        ):
            raise ValueError("estimator stream differs from predictor")
        tokens = tuple(int(value) for value in prefix.token_ids.tolist())
        if cache is not None:
            if type(cache) is not PrefixCache:
                raise ValueError("prefix cache must be exact")
            cache.__post_init__()
            key = cache.key
            if (
                key.vocabulary_sha256 != self.vocabulary_sha256
                or key.predictor_config_sha256
                != self.predictor_config_sha256
                or key.model_family_sha256 != self.model_family_sha256
                or key.model_state_sha256 != self.model_state_sha256
                or key.proposal_identity_sha256
                != self.proposal_identity_sha256
                or key.estimator_semantic_sha256
                != self.estimator_identity.semantic_sha256
                or key.estimator_artifact_bytes_sha256
                != self.estimator_identity.artifact_bytes_sha256
                or key.estimator_stream_sha256 != stream.stream_sha256
                or key.data_safety_sha256 != self.data_safety_sha256
                or len(key.prefix_tokens) > len(tokens)
                or tokens[: len(key.prefix_tokens)] != key.prefix_tokens
            ):
                raise ValueError("cache does not bind this causal predictor call")
        return tokens

    def _cache_key(
        self,
        *,
        prefix: CausalPrefix,
        stream: EstimatorStream,
    ) -> PrefixCacheKey:
        return PrefixCacheKey.create(
            prefix=prefix,
            vocabulary_sha256=self.vocabulary_sha256,
            predictor_config_sha256=self.predictor_config_sha256,
            model_family_sha256=self.model_family_sha256,
            model_state_sha256=self.model_state_sha256,
            proposal_identity_sha256=self.proposal_identity_sha256,
            estimator_semantic_sha256=self.estimator_identity.semantic_sha256,
            estimator_artifact_bytes_sha256=(
                self.estimator_identity.artifact_bytes_sha256
            ),
            estimator_stream_sha256=stream.stream_sha256,
            data_safety_sha256=self.data_safety_sha256,
        )

    def _prediction(
        self,
        *,
        prefix: CausalPrefix,
        stream: EstimatorStream,
        population: ProposalPopulation,
        prediction_log_probs: Tensor,
        assimilations: tuple[AssimilationRecord, ...],
        cumulative_log_normalizer: float,
        counter_consumption: tuple[CounterConsumption, ...],
        filtered_log_weights: Tensor | None = None,
    ) -> PriorPrediction:
        normalized = (
            prediction_log_probs.to(device="cpu", dtype=torch.float64)
            .contiguous()
        )
        normalized = normalized - torch.logsumexp(normalized, dim=0)
        if self.estimator_spec.kind == "deterministic_exact":
            if (
                filtered_log_weights is not None
                or population.particle_count != 1
            ):
                raise ValueError(
                    "exact predictor cache must contain one exact state"
                )
            step = ProposalStep.create(
                position=max(1, len(assimilations) + 1),
                population=population,
                emission_log_probs=normalized.unsqueeze(0),
                counter_consumption=counter_consumption,
                proposal_identity_sha256=self.proposal_identity_sha256,
            )
            weights = torch.zeros(1, dtype=torch.float64)
            pending: PendingPrediction | MarginalPendingPrediction = (
                PendingPrediction.create(
                    prefix_sha256=prefix.prefix_sha256,
                    step=step,
                    parent_log_weights=weights,
                    prediction_log_probs=normalized,
                )
            )
        else:
            if (
                type(filtered_log_weights) is not Tensor
                or self.estimator_spec.particle_count
                != population.particle_count
            ):
                raise ValueError(
                    "SMC predictor cache must retain its real particle axis"
                )
            weights = (
                filtered_log_weights.detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
            pending = MarginalPendingPrediction.create(
                prefix_sha256=prefix.prefix_sha256,
                proposal_identity_sha256=self.proposal_identity_sha256,
                proposed_population=population,
                parent_log_weights=weights,
                prediction_log_probs=normalized,
                counter_consumption=counter_consumption,
            )
        cache = PrefixCache.create(
            key=self._cache_key(prefix=prefix, stream=stream),
            filtered_population=population,
            filtered_log_weights=weights,
            cumulative_log_normalizer=cumulative_log_normalizer,
            pending=pending,
            assimilations=assimilations,
            counter_consumption=counter_consumption,
        )
        estimator_record = EstimatorRecord.from_cache(
            stream=stream,
            cache=cache,
        )
        return PriorPrediction.create(
            vocabulary=self.vocabulary,
            log_probs=normalized,
            cache=cache,
            estimator_record=estimator_record,
        )


class ExactAutoregressivePriorPredictor(_PredictorBase):
    """Deterministic target-blind adapter for A0 and the no-latent control."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        vocabulary: VocabularyIdentity,
        estimator_spec: EstimatorSpec,
        estimator_identity: EstimatorIdentity,
        predictor_config_sha256: str,
        data_safety_sha256: str,
    ) -> None:
        if type(model) not in (WT103A0Model, WT103NoLatentModel):
            raise ValueError("exact WT103 adapter requires A0 or no-latent model")
        if (
            estimator_spec.kind != "deterministic_exact"
            or estimator_spec.particle_count is not None
        ):
            raise ValueError("exact adapter requires deterministic estimator")
        self._initialize_common(
            model=model,
            vocabulary=vocabulary,
            estimator_spec=estimator_spec,
            estimator_identity=estimator_identity,
            predictor_config_sha256=predictor_config_sha256,
            data_safety_sha256=data_safety_sha256,
            adapter_kind="exact_autoregressive",
        )
        self.particle_count = None

    def _log_probs(self, token_ids: tuple[int, ...]) -> Tensor:
        model = self.model
        device = next(model.parameters()).device
        prefix = torch.tensor(
            token_ids,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)
        inference_scope = (
            torch.no_grad()
            if not torch.is_grad_enabled()
            else nullcontext()
        )
        with inference_scope, _bf16_autocast(model):
            if type(model) is WT103NoLatentModel:
                return model.next_token_log_probs(prefix)
            assert type(model) is WT103A0Model
            if prefix.shape[1] == 0:
                hidden = model.position_embedding.weight[:1].unsqueeze(0)
                hidden = model.final_norm(
                    model.block(
                        hidden,
                        attention_heads=model.attention_heads,
                    )
                )
            else:
                hidden = model.encode(prefix)
            logits = model.decoder(hidden[:, -1])
            observer = model.live_observer
            if observer is not None:
                observer.observe_tensor(
                    logits,
                    "decoder_chunk",
                    ("token_or_particle_chunk", "vocabulary"),
                    model.live_phase,
                    f"{model.live_event_prefix}:exact",
                )
            return F.log_softmax(logits, dim=-1).squeeze(0)

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction:
        tokens = self._validate_call(prefix_tokens, estimator_rng, cache)
        assimilations = [] if cache is None else list(cache.assimilations)
        cumulative = (
            0.0 if cache is None else cache.cumulative_log_normalizer
        )
        start = 0 if cache is None else len(cache.key.prefix_tokens)
        for position in range(start + 1, len(tokens) + 1):
            token = tokens[position - 1]
            log_probs = (
                self._log_probs(tokens[: position - 1])
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            log_probs = log_probs - torch.logsumexp(log_probs, dim=0)
            increment = float(log_probs[token].item())
            cumulative = math.fsum((cumulative, increment))
            assimilations.append(
                AssimilationRecord.create(
                    position=position,
                    observed_token=token,
                    incremental_log_normalizer=increment,
                    ess=1.0,
                    ancestors=(),
                    resampling_consumption=None,
                )
            )
        final = self._log_probs(tokens).detach()
        state = torch.tensor(
            (float(len(tokens)),),
            dtype=torch.float64,
        ).view(1, 1)
        population = ProposalPopulation.create({"exact_state": state})
        return self._prediction(
            prefix=prefix_tokens,
            stream=estimator_rng,
            population=population,
            prediction_log_probs=final,
            assimilations=tuple(assimilations),
            cumulative_log_normalizer=cumulative,
            counter_consumption=(),
        )


class WT103ChunkedSmcPriorPredictor(_PredictorBase):
    """Weighted bootstrap SMC with vocabulary decoding in particle chunks."""

    def __init__(
        self,
        *,
        model: WT103LatentGenerativeModel,
        vocabulary: VocabularyIdentity,
        estimator_spec: EstimatorSpec,
        estimator_identity: EstimatorIdentity,
        predictor_config_sha256: str,
        data_safety_sha256: str,
    ) -> None:
        if type(model) is not WT103LatentGenerativeModel:
            raise ValueError("SMC adapter requires exact WT103 latent model")
        if (
            estimator_spec.kind != "weighted_smc"
            or type(estimator_spec.particle_count) is not int
        ):
            raise ValueError("SMC adapter requires weighted particle estimator")
        self._initialize_common(
            model=model,
            vocabulary=vocabulary,
            estimator_spec=estimator_spec,
            estimator_identity=estimator_identity,
            predictor_config_sha256=predictor_config_sha256,
            data_safety_sha256=data_safety_sha256,
            adapter_kind="chunked_weighted_smc",
        )
        self.particle_count = estimator_spec.particle_count
        self.max_observed_particle_decoder_chunk = 0

    @property
    def latent_model(self) -> WT103LatentGenerativeModel:
        return self.model  # type: ignore[return-value]

    def _gaussian(
        self,
        stream: EstimatorStream,
        *,
        prefix_sha256: str,
        position: int,
        purpose: CounterPurpose,
        width: int,
    ) -> Tensor:
        rows = tuple(
            stream.gaussian(
                CounterKey(
                    stream.stream_seed,
                    prefix_sha256,
                    position,
                    purpose,
                    particle,
                ),
                count=width,
            )
            for particle in range(self.particle_count)
        )
        return torch.tensor(
            rows,
            device=self.latent_model.device,
            dtype=self.latent_model.dtype,
        )

    def _initial_particles(
        self,
        stream: EstimatorStream,
        *,
        prefix_sha256: str,
    ) -> tuple[Tensor, tuple[CounterConsumption, ...]]:
        model = self.latent_model
        state_noise = self._gaussian(
            stream,
            prefix_sha256=prefix_sha256,
            position=0,
            purpose=CounterPurpose.INITIAL_STATE_GAUSSIAN,
            width=model.d_z,
        )
        model_noise = self._gaussian(
            stream,
            prefix_sha256=prefix_sha256,
            position=0,
            purpose=CounterPurpose.INITIAL_MODEL_GAUSSIAN,
            width=model.d_m,
        )
        noise = torch.cat((state_noise, model_noise), dim=-1)
        particles = model.initial_mean + torch.exp(
            model.initial_log_scale
        ) * noise
        return particles, (
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.INITIAL_STATE_GAUSSIAN,
                particle_count=self.particle_count,
                draws_per_particle=model.d_z,
            ),
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.INITIAL_MODEL_GAUSSIAN,
                particle_count=self.particle_count,
                draws_per_particle=model.d_m,
            ),
        )

    def _mixture_log_probs(self, particles: Tensor, log_weights: Tensor) -> Tensor:
        model = self.latent_model
        mixture: Tensor | None = None
        for start in range(0, self.particle_count, model.particle_chunk_size):
            end = min(start + model.particle_chunk_size, self.particle_count)
            self.max_observed_particle_decoder_chunk = max(
                self.max_observed_particle_decoder_chunk,
                end - start,
            )
            with _bf16_autocast(model):
                rows = model.particle_emission_log_probs(
                    particles[start:end]
                )
            contribution = torch.logsumexp(
                log_weights[start:end, None].to(rows.dtype) + rows,
                dim=0,
            )
            mixture = (
                contribution
                if mixture is None
                else torch.logaddexp(mixture, contribution)
            )
        assert mixture is not None
        mixture = mixture.to(device="cpu", dtype=torch.float64)
        return mixture - torch.logsumexp(mixture, dim=0)

    def _selected_log_probs(self, particles: Tensor, token: int) -> Tensor:
        model = self.latent_model
        rows: list[Tensor] = []
        for start in range(0, self.particle_count, model.particle_chunk_size):
            end = min(start + model.particle_chunk_size, self.particle_count)
            self.max_observed_particle_decoder_chunk = max(
                self.max_observed_particle_decoder_chunk,
                end - start,
            )
            with _bf16_autocast(model):
                log_probs = model.particle_emission_log_probs(
                    particles[start:end]
                )
            rows.append(log_probs[:, token].to(device="cpu", dtype=torch.float64))
        return torch.cat(tuple(rows))

    def _next_source_log_probs(
        self,
        prefix: tuple[int, ...],
        history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        model = self.latent_model
        if not prefix:
            raise ValueError("source selection requires one causal parent")
        tokens = torch.tensor(
            prefix,
            dtype=torch.int64,
            device=model.device,
        ).unsqueeze(0).expand(history.shape[0], -1)
        tokens = causal_observation_history(
            tokens,
            receiver_position=len(prefix),
        )
        with _bf16_autocast(model):
            state_row, model_row = model.next_source_log_probs(
                tokens,
                history,
            )
        return (
            state_row.to(device="cpu", dtype=torch.float64),
            model_row.to(device="cpu", dtype=torch.float64),
        )

    def _propagate(
        self,
        *,
        particles: Tensor,
        history: Tensor,
        source_history: tuple[int, ...],
        transition_token: int,
        stream: EstimatorStream,
        prefix_sha256: str,
        position: int,
    ) -> tuple[Tensor, tuple[CounterConsumption, ...]]:
        model = self.latent_model
        state_source_log_probs, model_source_log_probs = (
            self._next_source_log_probs(source_history, history)
        )
        if (
            state_source_log_probs.ndim != 2
            or model_source_log_probs.ndim != 2
            or state_source_log_probs.shape[0] != self.particle_count
            or model_source_log_probs.shape[0] != self.particle_count
            or state_source_log_probs.shape != model_source_log_probs.shape
        ):
            raise ValueError("state/model source rows require exact [N,W] shape")
        valid_count = state_source_log_probs.shape[-1]
        if not 1 <= valid_count <= history.shape[1]:
            raise ValueError("state/model source row widths disagree")
        z_sources: list[int] = []
        m_sources: list[int] = []
        for particle in range(self.particle_count):
            z_sources.append(
                stream.categorical(
                    CounterKey(
                        stream.stream_seed,
                        prefix_sha256,
                        position,
                        CounterPurpose.STATE_SOURCE_CATEGORICAL,
                        particle,
                    ),
                    state_source_log_probs[particle],
                )
            )
            m_sources.append(
                stream.categorical(
                    CounterKey(
                        stream.stream_seed,
                        prefix_sha256,
                        position,
                        CounterPurpose.MODEL_SOURCE_CATEGORICAL,
                        particle,
                    ),
                    model_source_log_probs[particle],
                )
            )
        offset = history.shape[1] - valid_count
        particle_ids = torch.arange(
            self.particle_count, device=model.device
        )
        z_index = torch.tensor(z_sources, device=model.device) + offset
        m_index = torch.tensor(m_sources, device=model.device) + offset
        if (
            z_index.shape != (self.particle_count,)
            or m_index.shape != (self.particle_count,)
            or bool(torch.any(z_index < 0))
            or bool(torch.any(m_index < 0))
            or bool(torch.any(z_index >= history.shape[1]))
            or bool(torch.any(m_index >= history.shape[1]))
            or bool(torch.any(z_index >= position))
            or bool(torch.any(m_index >= position))
        ):
            raise ValueError("sampled source indices left the causal global bank")
        z_parent = history[particle_ids, z_index, : model.d_z]
        m_parent = history[particle_ids, m_index, model.d_z :]
        token = torch.tensor(
            transition_token, dtype=torch.int64, device=model.device
        )
        m_noise = self._gaussian(
            stream,
            prefix_sha256=prefix_sha256,
            position=position,
            purpose=CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
            width=model.d_m,
        )
        with _bf16_autocast(model):
            model_mean = model.model_transition_mean(
                input_ids=token,
                receiver_position=position,
                source_positions=m_index,
                model_parent=m_parent,
            )
        current_model = (
            model_mean
            + torch.exp(model.transition_log_scale[model.d_z :])
            * m_noise
        )
        with _bf16_autocast(model):
            state_mean = model.state_transition_mean(
                input_ids=token,
                receiver_position=position,
                source_positions=z_index,
                state_parent=z_parent,
                current_model=current_model,
            )
        z_noise = self._gaussian(
            stream,
            prefix_sha256=prefix_sha256,
            position=position,
            purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
            width=model.d_z,
        )
        current_state = (
            state_mean
            + torch.exp(model.transition_log_scale[: model.d_z])
            * z_noise
        )
        next_particles = torch.cat((current_state, current_model), dim=-1)
        consumption = (
            CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.STATE_SOURCE_CATEGORICAL,
                particle_count=self.particle_count,
                draws_per_particle=1,
            ),
            CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.MODEL_SOURCE_CATEGORICAL,
                particle_count=self.particle_count,
                draws_per_particle=1,
            ),
            CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
                particle_count=self.particle_count,
                draws_per_particle=model.d_z,
            ),
            CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
                particle_count=self.particle_count,
                draws_per_particle=model.d_m,
            ),
        )
        return next_particles, consumption

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction:
        tokens = self._validate_call(prefix_tokens, estimator_rng, cache)
        if cache is None:
            empty = CausalPrefix.create(
                receiver_t=1,
                vocabulary=self.vocabulary,
                token_ids=torch.empty(0, dtype=torch.int64),
            )
            particles, initial_consumption = self._initial_particles(
                estimator_rng,
                prefix_sha256=empty.prefix_sha256,
            )
            history = particles.unsqueeze(1)
            log_weights = torch.full(
                (self.particle_count,),
                -math.log(self.particle_count),
                dtype=torch.float64,
            )
            assimilations: list[AssimilationRecord] = []
            consumption: list[CounterConsumption] = list(initial_consumption)
            cumulative = 0.0
            start = 0
        else:
            population = cache.filtered_population
            if tuple(component.name for component in population.components) != (
                "ensemble",
                "history",
            ):
                raise ValueError("SMC cache scientific components changed")
            particles = population.component("ensemble").to(
                device=self.latent_model.device,
                dtype=self.latent_model.dtype,
            )
            history = population.component("history").to(
                device=self.latent_model.device,
                dtype=self.latent_model.dtype,
            )
            log_weights = cache.filtered_log_weights.value().to(
                device="cpu",
                dtype=torch.float64,
            )
            start = len(cache.key.prefix_tokens)
            expected_history = max(1, start)
            if (
                particles.shape
                != (self.particle_count, self.latent_model.latent_width)
                or history.shape
                != (
                    self.particle_count,
                    expected_history,
                    self.latent_model.latent_width,
                )
                or log_weights.shape != (self.particle_count,)
                or not torch.allclose(
                    particles,
                    history[:, -1],
                    atol=0.0,
                    rtol=0.0,
                )
                or not math.isclose(
                    float(torch.logsumexp(log_weights, dim=0).item()),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-13,
                )
            ):
                raise ValueError("SMC cache scientific state is malformed")
            assimilations = list(cache.assimilations)
            consumption = list(cache.counter_consumption)
            cumulative = cache.cumulative_log_normalizer

        if start == 0 and tokens:
            # The first token is the left-window causal input x_s, not a target
            # emitted by z_0.  It is retained as a neutral boundary record so
            # PrefixCache still has exactly one record per public prefix token.
            assimilations.append(
                AssimilationRecord.create(
                    position=1,
                    observed_token=tokens[0],
                    incremental_log_normalizer=0.0,
                    ess=float(self.particle_count),
                    ancestors=(),
                    resampling_consumption=None,
                )
            )
            start = 1

        for public_position in range(start + 1, len(tokens) + 1):
            token = tokens[public_position - 1]
            selected = self._selected_log_probs(particles, token)
            update = assimilate_log_weights(log_weights, selected)
            filtered_particles = particles
            filtered_history = history
            ancestors: tuple[int, ...] = ()
            resampling: CounterConsumption | None = None
            if update.ess < self.particle_count / 2.0:
                intermediate = CausalPrefix.create(
                    receiver_t=public_position + 1,
                    vocabulary=self.vocabulary,
                    token_ids=torch.tensor(
                        tokens[:public_position],
                        dtype=torch.int64,
                    ),
                )
                key = CounterKey(
                    estimator_rng.stream_seed,
                    intermediate.prefix_sha256,
                    public_position,
                    CounterPurpose.SYSTEMATIC_RESAMPLING,
                    0,
                )
                offset = estimator_rng.systematic_offset(
                    key, particle_count=self.particle_count
                )
                ancestor_tensor = systematic_ancestors(
                    update.normalized_log_weights,
                    offset=offset,
                )
                ancestors = tuple(int(item) for item in ancestor_tensor.tolist())
                device_ancestors = ancestor_tensor.to(
                    device=self.latent_model.device
                )
                filtered_particles = particles[device_ancestors]
                filtered_history = history[device_ancestors]
                log_weights = torch.full_like(
                    log_weights, -math.log(self.particle_count)
                )
                resampling = CounterConsumption.create(
                    position=public_position,
                    purpose=CounterPurpose.SYSTEMATIC_RESAMPLING,
                    particle_count=1,
                    draws_per_particle=1,
                )
                consumption.append(resampling)
            else:
                log_weights = update.normalized_log_weights
            cumulative = math.fsum((cumulative, update.log_normalizer))
            assimilations.append(
                AssimilationRecord.create(
                    position=public_position,
                    observed_token=token,
                    incremental_log_normalizer=update.log_normalizer,
                    ess=update.ess,
                    ancestors=ancestors,
                    resampling_consumption=resampling,
                )
            )
            receiver_position = public_position - 1
            if not 1 <= receiver_position < self.latent_model.sequence_length:
                raise ValueError("SMC causal receiver exceeds model capacity")
            source_tokens = torch.tensor(
                tokens,
                dtype=torch.int64,
                device=self.latent_model.device,
            ).unsqueeze(0)
            source_history = tuple(
                int(value)
                for value in causal_observation_history(
                    source_tokens,
                    receiver_position=receiver_position,
                )
                .squeeze(0)
                .tolist()
            )
            intermediate = CausalPrefix.create(
                receiver_t=public_position + 1,
                vocabulary=self.vocabulary,
                token_ids=torch.tensor(
                    tokens[:public_position],
                    dtype=torch.int64,
                ),
            )
            particles, step_consumption = self._propagate(
                particles=filtered_particles,
                history=filtered_history,
                source_history=source_history,
                transition_token=token,
                stream=estimator_rng,
                prefix_sha256=intermediate.prefix_sha256,
                position=receiver_position,
            )
            history = torch.cat(
                (filtered_history, particles.unsqueeze(1)), dim=1
            )
            consumption.extend(step_consumption)
        mixture = self._mixture_log_probs(particles, log_weights)
        population = ProposalPopulation.create(
            {
                "ensemble": particles.detach().to(device="cpu"),
                "history": history.detach().to(device="cpu"),
            }
        )
        return self._prediction(
            prefix=prefix_tokens,
            stream=estimator_rng,
            population=population,
            prediction_log_probs=mixture,
            assimilations=tuple(assimilations),
            cumulative_log_normalizer=cumulative,
            counter_consumption=tuple(consumption),
            filtered_log_weights=log_weights,
        )


__all__ = [
    "ExactAutoregressivePriorPredictor",
    "WT103ChunkedSmcPriorPredictor",
]
