"""Target-free proposal adapters and frozen counter-based estimator streams."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from vfe4.data.windows import CausalPrefix
from vfe4.generative import LanguageGenerativeModel, PrefixConditionedSourcePrior
from vfe4.types.h6 import (
    FrozenTensorSnapshot,
    VocabularyIdentity,
    canonical_json_bytes,
)

from .identities import (
    EstimatorIdentity,
    canonical_model_state_sha256,
    vocabulary_identity_sha256,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_COUNTER_BLOCK_DOMAIN = b"VFE4-H6-SMC-COUNTER-BLOCK-V1\x00"


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _snapshot_payload(snapshot: FrozenTensorSnapshot) -> dict[str, object]:
    snapshot.assert_intact()
    return {
        "dtype": snapshot.dtype,
        "shape": snapshot.shape,
        "device": snapshot.device,
        "contiguous": snapshot.contiguous,
        "requires_grad": snapshot.requires_grad,
        "storage_version": snapshot.storage_version,
        "raw_bytes_sha256": snapshot.raw_bytes_sha256,
    }


class CounterPurpose(str, Enum):
    """Closed purpose vocabulary for every H6 estimator draw."""

    INITIAL_STATE_GAUSSIAN = "initial_state_gaussian"
    INITIAL_MODEL_GAUSSIAN = "initial_model_gaussian"
    FINITE_INITIAL_CATEGORICAL = "finite_initial_categorical"
    STATE_SOURCE_CATEGORICAL = "state_source_categorical"
    MODEL_SOURCE_CATEGORICAL = "model_source_categorical"
    MODEL_TRANSITION_GAUSSIAN = "model_transition_gaussian"
    STATE_TRANSITION_GAUSSIAN = "state_transition_gaussian"
    FINITE_TRANSITION_CATEGORICAL = "finite_transition_categorical"
    SYSTEMATIC_RESAMPLING = "systematic_resampling"


COUNTER_PURPOSE_VOCABULARY = tuple(purpose.value for purpose in CounterPurpose)
_PURPOSE_INDEX = {
    purpose: index for index, purpose in enumerate(CounterPurpose)
}
_GAUSSIAN_PURPOSES = frozenset(
    {
        CounterPurpose.INITIAL_STATE_GAUSSIAN,
        CounterPurpose.INITIAL_MODEL_GAUSSIAN,
        CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
        CounterPurpose.STATE_TRANSITION_GAUSSIAN,
    }
)
_CATEGORICAL_PURPOSES = frozenset(
    {
        CounterPurpose.FINITE_INITIAL_CATEGORICAL,
        CounterPurpose.STATE_SOURCE_CATEGORICAL,
        CounterPurpose.MODEL_SOURCE_CATEGORICAL,
        CounterPurpose.FINITE_TRANSITION_CATEGORICAL,
    }
)


@dataclass(frozen=True)
class CounterKey:
    stream_seed: int
    prefix_sha256: str
    position: int
    purpose: CounterPurpose
    particle_index: int

    def __post_init__(self) -> None:
        if (
            type(self.stream_seed) is not int
            or not 0 <= self.stream_seed < 2**64
        ):
            raise ValueError("stream_seed must be an unsigned 64-bit integer")
        _require_sha256(self.prefix_sha256, "prefix_sha256")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("position must be a nonnegative integer")
        if type(self.purpose) is not CounterPurpose:
            raise ValueError("purpose must be one frozen CounterPurpose")
        if type(self.particle_index) is not int or self.particle_index < 0:
            raise ValueError("particle_index must be a nonnegative integer")


@dataclass(frozen=True)
class EstimatorStream:
    """A stateless SHA-256 counter stream with explicit draw expansion rules."""

    stream_seed: int
    estimator_semantic_sha256: str
    estimator_artifact_bytes_sha256: str
    estimator_identity_sha256: str
    stream_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.stream_seed) is not int
            or not 0 <= self.stream_seed < 2**64
        ):
            raise ValueError("stream_seed must be an unsigned 64-bit integer")
        for name in (
            "estimator_semantic_sha256",
            "estimator_artifact_bytes_sha256",
            "estimator_identity_sha256",
            "stream_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.estimator-stream.v1",
            {
                "stream_seed": self.stream_seed,
                "estimator_semantic_sha256": self.estimator_semantic_sha256,
                "estimator_artifact_bytes_sha256": (
                    self.estimator_artifact_bytes_sha256
                ),
                "estimator_identity_sha256": self.estimator_identity_sha256,
                "purpose_vocabulary": COUNTER_PURPOSE_VOCABULARY,
                "draw_expansion": "sha256_blocks_four_uint64_le_v1",
                "open_uniform": "uint64_plus_half_clamped_open_v1",
                "gaussian": "box_muller_pair_v1",
                "categorical": "cdf_left_final_one_v1",
            },
        )
        if self.stream_sha256 != expected:
            raise ValueError("stream_sha256 does not match the frozen stream rules")

    @classmethod
    def create(
        cls, *, stream_seed: int, estimator_identity: EstimatorIdentity
    ) -> "EstimatorStream":
        if type(estimator_identity) is not EstimatorIdentity:
            raise ValueError(
                "estimator_identity must be an exact EstimatorIdentity"
            )
        estimator_identity.__post_init__()
        payload = {
            "stream_seed": stream_seed,
            "estimator_semantic_sha256": estimator_identity.semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                estimator_identity.artifact_bytes_sha256
            ),
            "estimator_identity_sha256": estimator_identity.identity_sha256,
            "purpose_vocabulary": COUNTER_PURPOSE_VOCABULARY,
            "draw_expansion": "sha256_blocks_four_uint64_le_v1",
            "open_uniform": "uint64_plus_half_clamped_open_v1",
            "gaussian": "box_muller_pair_v1",
            "categorical": "cdf_left_final_one_v1",
        }
        return cls(
            stream_seed,
            estimator_identity.semantic_sha256,
            estimator_identity.artifact_bytes_sha256,
            estimator_identity.identity_sha256,
            _owned_hash("vfe4.h6.estimator-stream.v1", payload),
        )

    def _word(self, key: CounterKey, draw_index: int) -> int:
        self.__post_init__()
        if type(key) is not CounterKey:
            raise ValueError("counter key must be an exact CounterKey")
        key.__post_init__()
        if key.stream_seed != self.stream_seed:
            raise ValueError("counter key stream seed does not match the stream")
        if type(draw_index) is not int or draw_index < 0:
            raise ValueError("draw_index must be a nonnegative integer")
        block_index, word_index = divmod(draw_index, 4)
        digest = hashlib.sha256(
            _COUNTER_BLOCK_DOMAIN
            + self.stream_seed.to_bytes(8, "little")
            + bytes.fromhex(key.prefix_sha256)
            + key.position.to_bytes(8, "little")
            + _PURPOSE_INDEX[key.purpose].to_bytes(2, "little")
            + key.particle_index.to_bytes(8, "little")
            + block_index.to_bytes(8, "little")
        ).digest()
        offset = 8 * word_index
        return int.from_bytes(digest[offset : offset + 8], "little")

    def open_uniform(self, key: CounterKey, *, draw_index: int = 0) -> float:
        """Map one uint64 counter word deterministically into the open interval."""

        word = self._word(key, draw_index)
        value = (float(word) + 0.5) / float(2**64)
        if value <= 0.0:
            return math.nextafter(0.0, 1.0)
        if value >= 1.0:
            return math.nextafter(1.0, 0.0)
        return value

    def gaussian(self, key: CounterKey, *, count: int) -> tuple[float, ...]:
        """Expand open uniforms into standard normals by paired Box-Muller."""

        if key.purpose not in _GAUSSIAN_PURPOSES:
            raise ValueError("Gaussian draws require a frozen Gaussian purpose")
        if type(count) is not int or count <= 0:
            raise ValueError("Gaussian count must be a positive integer")
        values: list[float] = []
        pair_count = (count + 1) // 2
        for pair_index in range(pair_count):
            first = self.open_uniform(key, draw_index=2 * pair_index)
            second = self.open_uniform(key, draw_index=2 * pair_index + 1)
            radius = math.sqrt(-2.0 * math.log(first))
            angle = 2.0 * math.pi * second
            values.append(radius * math.cos(angle))
            if len(values) < count:
                values.append(radius * math.sin(angle))
        return tuple(values)

    def categorical(self, key: CounterKey, log_probs: Tensor) -> int:
        """Inverse-CDF categorical sampling in index order with final CDF one."""

        if key.purpose not in _CATEGORICAL_PURPOSES:
            raise ValueError(
                "categorical draws require a frozen categorical purpose"
            )
        if (
            type(log_probs) is not Tensor
            or log_probs.device.type != "cpu"
            or log_probs.dtype is not torch.float64
            or log_probs.ndim != 1
            or log_probs.numel() <= 0
            or bool(torch.isnan(log_probs).any())
            or bool(torch.isposinf(log_probs).any())
        ):
            raise ValueError(
                "categorical log probabilities must be a normalized CPU float64 row"
            )
        normalizer = torch.logsumexp(log_probs, dim=0)
        if not math.isclose(
            float(normalizer.item()), 0.0, rel_tol=0.0, abs_tol=1e-13
        ):
            raise ValueError("categorical log probabilities must be normalized")
        cumulative = torch.cumsum(torch.exp(log_probs), dim=0)
        cumulative[-1] = 1.0
        uniform = self.open_uniform(key)
        return int(
            torch.searchsorted(
                cumulative,
                torch.tensor(uniform, dtype=torch.float64),
                right=False,
            ).item()
        )

    def systematic_offset(self, key: CounterKey, *, particle_count: int) -> float:
        if (
            key.purpose is not CounterPurpose.SYSTEMATIC_RESAMPLING
            or key.particle_index != 0
        ):
            raise ValueError(
                "systematic resampling requires its named purpose and particle zero"
            )
        if type(particle_count) is not int or particle_count <= 0:
            raise ValueError("particle_count must be a positive integer")
        return self.open_uniform(key) / particle_count


@dataclass(frozen=True)
class CounterConsumption:
    position: int
    purpose: CounterPurpose
    particle_count: int
    draws_per_particle: int
    consumption_sha256: str

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ValueError("counter-consumption position must be nonnegative")
        if type(self.purpose) is not CounterPurpose:
            raise ValueError("counter consumption requires a frozen purpose")
        if type(self.particle_count) is not int or self.particle_count <= 0:
            raise ValueError("counter consumption requires positive particles")
        if (
            type(self.draws_per_particle) is not int
            or self.draws_per_particle <= 0
        ):
            raise ValueError("counter consumption requires positive draw counts")
        expected = _owned_hash(
            "vfe4.h6.counter-consumption.v1",
            {
                "position": self.position,
                "purpose": self.purpose,
                "particle_count": self.particle_count,
                "draws_per_particle": self.draws_per_particle,
            },
        )
        if self.consumption_sha256 != expected:
            raise ValueError("counter consumption identity is stale")

    @classmethod
    def create(
        cls,
        *,
        position: int,
        purpose: CounterPurpose,
        particle_count: int,
        draws_per_particle: int,
    ) -> "CounterConsumption":
        payload = {
            "position": position,
            "purpose": purpose,
            "particle_count": particle_count,
            "draws_per_particle": draws_per_particle,
        }
        return cls(
            position,
            purpose,
            particle_count,
            draws_per_particle,
            _owned_hash("vfe4.h6.counter-consumption.v1", payload),
        )


@dataclass(frozen=True)
class PopulationComponent:
    name: str
    values: FrozenTensorSnapshot

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("population component names must be nonempty")
        if type(self.values) is not FrozenTensorSnapshot:
            raise ValueError("population components require frozen tensor values")
        self.values.assert_intact()
        if not self.values.shape:
            raise ValueError("population components require a particle axis")


@dataclass(frozen=True)
class ProposalPopulation:
    particle_count: int
    components: tuple[PopulationComponent, ...]
    population_sha256: str

    def __post_init__(self) -> None:
        if type(self.particle_count) is not int or self.particle_count <= 0:
            raise ValueError("proposal populations require positive particles")
        if (
            type(self.components) is not tuple
            or not self.components
            or tuple(component.name for component in self.components)
            != tuple(sorted(component.name for component in self.components))
            or len({component.name for component in self.components})
            != len(self.components)
        ):
            raise ValueError(
                "proposal population components must be unique and sorted"
            )
        for component in self.components:
            component.__post_init__()
            if component.values.shape[0] != self.particle_count:
                raise ValueError(
                    "every population component must share the particle axis"
                )
        expected = _owned_hash(
            "vfe4.h6.proposal-population.v1",
            {
                "particle_count": self.particle_count,
                "components": tuple(
                    {
                        "name": component.name,
                        "values": _snapshot_payload(component.values),
                    }
                    for component in self.components
                ),
            },
        )
        if self.population_sha256 != expected:
            raise ValueError("proposal population identity is stale")

    @classmethod
    def create(
        cls, components: Mapping[str, Tensor]
    ) -> "ProposalPopulation":
        if (
            not isinstance(components, Mapping)
            or not components
            or any(type(name) is not str or not name for name in components)
        ):
            raise ValueError("proposal components must be a nonempty mapping")
        ordered = tuple(
            PopulationComponent(name, FrozenTensorSnapshot.capture(components[name]))
            for name in sorted(components)
        )
        particle_count = ordered[0].values.shape[0]
        payload = {
            "particle_count": particle_count,
            "components": tuple(
                {
                    "name": component.name,
                    "values": _snapshot_payload(component.values),
                }
                for component in ordered
            ),
        }
        return cls(
            particle_count,
            ordered,
            _owned_hash("vfe4.h6.proposal-population.v1", payload),
        )

    def component(self, name: str) -> Tensor:
        self.__post_init__()
        matches = tuple(
            component for component in self.components if component.name == name
        )
        if len(matches) != 1:
            raise ValueError(f"proposal population lacks component {name!r}")
        return matches[0].values.value()

    def select(self, ancestors: Tensor) -> "ProposalPopulation":
        self.__post_init__()
        if (
            type(ancestors) is not Tensor
            or ancestors.device.type != "cpu"
            or ancestors.dtype is not torch.int64
            or ancestors.shape != (self.particle_count,)
            or bool(torch.any(ancestors < 0))
            or bool(torch.any(ancestors >= self.particle_count))
        ):
            raise ValueError(
                "ancestor indices must be CPU int64 with one valid index per particle"
            )
        return ProposalPopulation.create(
            {
                component.name: component.values.value()[ancestors]
                for component in self.components
            }
        )


@dataclass(frozen=True)
class ProposalStep:
    position: int
    population: ProposalPopulation
    emission_log_probs: FrozenTensorSnapshot
    counter_consumption: tuple[CounterConsumption, ...]
    proposal_identity_sha256: str
    step_sha256: str

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position <= 0:
            raise ValueError("proposal position must be positive")
        if type(self.population) is not ProposalPopulation:
            raise ValueError("proposal step requires an exact population")
        self.population.__post_init__()
        if type(self.emission_log_probs) is not FrozenTensorSnapshot:
            raise ValueError("proposal emissions must be frozen")
        self.emission_log_probs.assert_intact()
        if (
            self.emission_log_probs.dtype != "float64"
            or len(self.emission_log_probs.shape) != 2
            or self.emission_log_probs.shape[0]
            != self.population.particle_count
        ):
            raise ValueError(
                "proposal emissions must be float64 shape (particles, vocabulary)"
            )
        values = self.emission_log_probs.value()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("proposal emissions must be finite")
        if not bool(
            torch.allclose(
                torch.logsumexp(values, dim=1),
                torch.zeros(values.shape[0], dtype=torch.float64),
                rtol=0.0,
                atol=1e-13,
            )
        ):
            raise ValueError("each proposal emission row must be normalized")
        if (
            type(self.counter_consumption) is not tuple
            or any(
                type(record) is not CounterConsumption
                for record in self.counter_consumption
            )
        ):
            raise ValueError(
                "proposal counter consumption must be immutable exact records"
            )
        for record in self.counter_consumption:
            record.__post_init__()
        _require_sha256(
            self.proposal_identity_sha256, "proposal_identity_sha256"
        )
        expected = _owned_hash(
            "vfe4.h6.proposal-step.v1",
            {
                "position": self.position,
                "population_sha256": self.population.population_sha256,
                "emission_log_probs": _snapshot_payload(
                    self.emission_log_probs
                ),
                "counter_consumption": tuple(
                    record.consumption_sha256
                    for record in self.counter_consumption
                ),
                "proposal_identity_sha256": self.proposal_identity_sha256,
            },
        )
        if self.step_sha256 != expected:
            raise ValueError("proposal step identity is stale")

    @classmethod
    def create(
        cls,
        *,
        position: int,
        population: ProposalPopulation,
        emission_log_probs: Tensor,
        counter_consumption: tuple[CounterConsumption, ...],
        proposal_identity_sha256: str,
    ) -> "ProposalStep":
        snapshot = FrozenTensorSnapshot.capture(emission_log_probs)
        payload = {
            "position": position,
            "population_sha256": population.population_sha256,
            "emission_log_probs": _snapshot_payload(snapshot),
            "counter_consumption": tuple(
                record.consumption_sha256 for record in counter_consumption
            ),
            "proposal_identity_sha256": proposal_identity_sha256,
        }
        return cls(
            position,
            population,
            snapshot,
            counter_consumption,
            proposal_identity_sha256,
            _owned_hash("vfe4.h6.proposal-step.v1", payload),
        )


@runtime_checkable
class TargetFreeProposalAdapter(Protocol):
    vocabulary: VocabularyIdentity
    vocabulary_sha256: str
    model_family_sha256: str
    model_state_sha256: str
    proposal_identity_sha256: str

    def assert_current_state(self) -> None: ...

    def initialize(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        particle_count: int,
    ) -> tuple[ProposalPopulation, tuple[CounterConsumption, ...]]: ...

    def propagate(
        self,
        population: ProposalPopulation,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
    ) -> ProposalStep: ...


class LanguageGenerativeProposalAdapter:
    """Explicit bootstrap sampler for Task-3 normalized generative factors."""

    proposal_mode = "generative_bootstrap"

    def __init__(
        self,
        model: LanguageGenerativeModel,
        *,
        proposal_mode: str = "generative_bootstrap",
    ) -> None:
        if type(model) is not LanguageGenerativeModel:
            raise ValueError(
                "model must be an exact Task-3 LanguageGenerativeModel"
            )
        if proposal_mode != "generative_bootstrap":
            raise ValueError(
                "only the declared generative bootstrap proposal is supported"
            )
        for parameter in model.parameters():
            if parameter.device.type != "cpu" or parameter.dtype is not torch.float64:
                raise ValueError(
                    "H6 proposal parameters must be CPU float64 tensors"
                )
        self.model = model
        self.vocabulary = model.vocabulary
        self.vocabulary_sha256 = vocabulary_identity_sha256(model.vocabulary)
        self.model_family_sha256 = model.model_family_sha256
        self.model_state_sha256 = canonical_model_state_sha256(model)
        self.proposal_identity_sha256 = _owned_hash(
            "vfe4.h6.language-generative-proposal.v1",
            {
                "proposal_mode": self.proposal_mode,
                "vocabulary_sha256": self.vocabulary_sha256,
                "model_family_sha256": self.model_family_sha256,
                "model_state_sha256": self.model_state_sha256,
            },
        )

    def assert_current_state(self) -> None:
        if canonical_model_state_sha256(self.model) != self.model_state_sha256:
            raise ValueError(
                "language proposal model state changed after identity freeze"
            )

    def _key(
        self,
        *,
        stream: EstimatorStream,
        prefix: CausalPrefix,
        purpose: CounterPurpose,
        particle_index: int,
    ) -> CounterKey:
        return CounterKey(
            stream.stream_seed,
            prefix.prefix_sha256,
            prefix.receiver_t,
            purpose,
            particle_index,
        )

    def initialize(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        particle_count: int,
    ) -> tuple[ProposalPopulation, tuple[CounterConsumption, ...]]:
        self.assert_current_state()
        if (
            type(prefix_tokens) is not CausalPrefix
            or prefix_tokens.vocabulary != self.vocabulary
            or prefix_tokens.receiver_t != 1
            or prefix_tokens.token_ids.numel() != 0
        ):
            raise ValueError(
                "proposal initialization requires the exact empty target-free CausalPrefix"
            )
        if type(estimator_rng) is not EstimatorStream:
            raise ValueError("estimator_rng must be an exact EstimatorStream")
        if type(particle_count) is not int or particle_count <= 0:
            raise ValueError("particle_count must be positive")
        latent_dim = self.model.latent_dim
        means = self.model.initial_mean.reshape(2, latent_dim)
        scales = torch.exp(self.model.initial_log_scale.reshape(2, latent_dim))
        states: list[Tensor] = []
        models: list[Tensor] = []
        for particle_index in range(particle_count):
            state_noise = torch.tensor(
                estimator_rng.gaussian(
                    CounterKey(
                        estimator_rng.stream_seed,
                        prefix_tokens.prefix_sha256,
                        0,
                        CounterPurpose.INITIAL_STATE_GAUSSIAN,
                        particle_index,
                    ),
                    count=latent_dim,
                ),
                dtype=torch.float64,
            )
            model_noise = torch.tensor(
                estimator_rng.gaussian(
                    CounterKey(
                        estimator_rng.stream_seed,
                        prefix_tokens.prefix_sha256,
                        0,
                        CounterPurpose.INITIAL_MODEL_GAUSSIAN,
                        particle_index,
                    ),
                    count=latent_dim,
                ),
                dtype=torch.float64,
            )
            states.append(means[0] + scales[0] * state_noise)
            models.append(means[1] + scales[1] * model_noise)
        population = ProposalPopulation.create(
            {
                "model_history": torch.stack(models, dim=0).unsqueeze(1),
                "state_history": torch.stack(states, dim=0).unsqueeze(1),
            }
        )
        draws = 2 * ((latent_dim + 1) // 2)
        return population, (
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.INITIAL_STATE_GAUSSIAN,
                particle_count=particle_count,
                draws_per_particle=draws,
            ),
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.INITIAL_MODEL_GAUSSIAN,
                particle_count=particle_count,
                draws_per_particle=draws,
            ),
        )

    def _source_log_probs(
        self,
        *,
        bank: str,
        prefix: CausalPrefix,
        earlier_latents: Tensor,
    ) -> Tensor:
        kwargs: dict[str, object] = {"receiver_t": prefix.receiver_t}
        if type(self.model.source_prior) is PrefixConditionedSourcePrior:
            kwargs.update(
                {"prefix": prefix, "earlier_latents": earlier_latents}
            )
        if bank == "state":
            return self.model.state_source_log_probs(**kwargs).log_probs.value()
        return self.model.model_source_log_probs(**kwargs).log_probs.value()

    def propagate(
        self,
        population: ProposalPopulation,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
    ) -> ProposalStep:
        self.assert_current_state()
        if type(population) is not ProposalPopulation:
            raise ValueError("population must be an exact ProposalPopulation")
        population.__post_init__()
        if (
            type(prefix_tokens) is not CausalPrefix
            or prefix_tokens.vocabulary != self.vocabulary
        ):
            raise ValueError(
                "proposal propagation requires a matching target-free CausalPrefix"
            )
        if type(estimator_rng) is not EstimatorStream:
            raise ValueError("estimator_rng must be an exact EstimatorStream")
        position = prefix_tokens.receiver_t
        state_history = population.component("state_history")
        model_history = population.component("model_history")
        expected_shape = (
            population.particle_count,
            position,
            self.model.latent_dim,
        )
        if (
            state_history.shape != expected_shape
            or model_history.shape != expected_shape
            or state_history.dtype is not torch.float64
            or model_history.dtype is not torch.float64
        ):
            raise ValueError(
                "proposal histories must match the causal receiver and latent dimension"
            )

        current_states: list[Tensor] = []
        current_models: list[Tensor] = []
        emission_rows: list[Tensor] = []
        for particle_index in range(population.particle_count):
            states = state_history[particle_index]
            models = model_history[particle_index]
            if self.model.source_prior is None:
                model_source = position - 1
                state_source = position - 1
            else:
                model_source = estimator_rng.categorical(
                    self._key(
                        stream=estimator_rng,
                        prefix=prefix_tokens,
                        purpose=CounterPurpose.MODEL_SOURCE_CATEGORICAL,
                        particle_index=particle_index,
                    ),
                    self._source_log_probs(
                        bank="model",
                        prefix=prefix_tokens,
                        earlier_latents=models,
                    ),
                )
                state_source = estimator_rng.categorical(
                    self._key(
                        stream=estimator_rng,
                        prefix=prefix_tokens,
                        purpose=CounterPurpose.STATE_SOURCE_CATEGORICAL,
                        particle_index=particle_index,
                    ),
                    self._source_log_probs(
                        bank="state",
                        prefix=prefix_tokens,
                        earlier_latents=states,
                    ),
                )

            model_noise = torch.tensor(
                estimator_rng.gaussian(
                    self._key(
                        stream=estimator_rng,
                        prefix=prefix_tokens,
                        purpose=CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
                        particle_index=particle_index,
                    ),
                    count=self.model.latent_dim,
                ),
                dtype=torch.float64,
            )
            current_model = (
                self.model.model_transition_weight @ models[model_source]
                + self.model.model_transition_bias
                + torch.exp(self.model.model_transition_log_scale)
                * model_noise
            )
            state_noise = torch.tensor(
                estimator_rng.gaussian(
                    self._key(
                        stream=estimator_rng,
                        prefix=prefix_tokens,
                        purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
                        particle_index=particle_index,
                    ),
                    count=self.model.latent_dim,
                ),
                dtype=torch.float64,
            )
            current_state = (
                self.model.state_transition_weight @ states[state_source]
                + self.model.state_model_weight @ current_model
                + self.model.state_transition_bias
                + torch.exp(self.model.state_transition_log_scale)
                * state_noise
            )
            emission = self.model.emission_log_probs(
                receiver_t=position,
                current_state=current_state,
                current_model=current_model,
            ).log_values.value()
            current_models.append(current_model)
            current_states.append(current_state)
            emission_rows.append(emission)

        next_population = ProposalPopulation.create(
            {
                "model_history": torch.cat(
                    [model_history, torch.stack(current_models).unsqueeze(1)],
                    dim=1,
                ),
                "state_history": torch.cat(
                    [state_history, torch.stack(current_states).unsqueeze(1)],
                    dim=1,
                ),
            }
        )
        draws = 2 * ((self.model.latent_dim + 1) // 2)
        consumption: list[CounterConsumption] = []
        if self.model.source_prior is not None:
            consumption.extend(
                [
                    CounterConsumption.create(
                        position=position,
                        purpose=CounterPurpose.MODEL_SOURCE_CATEGORICAL,
                        particle_count=population.particle_count,
                        draws_per_particle=1,
                    ),
                    CounterConsumption.create(
                        position=position,
                        purpose=CounterPurpose.STATE_SOURCE_CATEGORICAL,
                        particle_count=population.particle_count,
                        draws_per_particle=1,
                    ),
                ]
            )
        consumption.extend(
            [
                CounterConsumption.create(
                    position=position,
                    purpose=CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
                    particle_count=population.particle_count,
                    draws_per_particle=draws,
                ),
                CounterConsumption.create(
                    position=position,
                    purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
                    particle_count=population.particle_count,
                    draws_per_particle=draws,
                ),
            ]
        )
        return ProposalStep.create(
            position=position,
            population=next_population,
            emission_log_probs=torch.stack(emission_rows, dim=0),
            counter_consumption=tuple(consumption),
            proposal_identity_sha256=self.proposal_identity_sha256,
        )


__all__ = [
    "COUNTER_PURPOSE_VOCABULARY",
    "CounterConsumption",
    "CounterKey",
    "CounterPurpose",
    "EstimatorStream",
    "LanguageGenerativeProposalAdapter",
    "PopulationComponent",
    "ProposalPopulation",
    "ProposalStep",
    "TargetFreeProposalAdapter",
]
