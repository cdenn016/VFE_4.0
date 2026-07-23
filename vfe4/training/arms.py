"""Explicit H6 arm factories and their target-free predictive boundaries."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.data.windows import CausalPrefix
from vfe4.numerics.categorical import masked_log_softmax_from_parents
from vfe4.predictive import (
    BootstrapSmcPredictor,
    CounterConsumption,
    CounterKey,
    CounterPurpose,
    EstimatorIdentity,
    EstimatorStream,
    PrefixCache,
    PriorPrediction,
    ProposalPopulation,
    ProposalStep,
    TargetFreeProposalAdapter,
    canonical_model_state_sha256,
    vocabulary_identity_sha256,
)
from vfe4.recognition.parameter_store import LanguageRecognitionParameterStore
from vfe4.types.h6 import (
    ArmId,
    EstimatorSpec,
    TrainingPhase,
    VocabularyIdentity,
    canonical_json_bytes,
)

from .matching import (
    ARM_MATRIX_ROWS,
    ARM_MATRIX_SHA256,
    H6_ADAMW_POLICY,
    ArmConfig,
    ArmMatrixRow,
    CapacityAllocation,
    FlopTerm,
    MatchingReport,
    OptimizerBinding,
    ParameterRoleRecord,
    arm_matrix_sha256,
)


Channel = Literal["state", "model"]
ElboPartition = Literal[
    "initial",
    "state_source",
    "model_source",
    "state_transition",
    "model_transition",
    "emission",
    "entropy",
]

H6_TARGET_FREE_DATA_SAFETY_SHA256 = hashlib.sha256(
    b"VFE4-H6-TARGET-FREE-PREDICTIVE-BOUNDARY-V1"
).hexdigest()

_SEMANTIC_FIELDS = (
    "latent_enabled",
    "state_channel_enabled",
    "model_channel_enabled",
    "source_mode",
    "map_mode",
    "recognition_family",
    "recognition_conditioning",
    "prior_variant",
    "mixture_mode",
    "objective_kind",
)
_BASE_PROFILE_IDS = {
    ArmId.A0: "h6-a0-ar-v1",
    ArmId.A1: "h6-a1-ordinary-latent-v1",
    ArmId.A2: "h6-a2-generic-map-v1",
    ArmId.A3: "h6-a3-immediate-predecessor-v1",
    ArmId.A4: "h6-a4-state-only-v1",
}
_BASE_PROFILES: dict[ArmId, tuple[object, ...]] = {
    ArmId.A0: (
        False, False, False, "absent", "absent", "absent", "absent",
        "absent", "absent", "cross_entropy",
    ),
    ArmId.A1: (
        True, True, False, "absent", "absent", "structured", "smoothing",
        "absent", "absent", "complete_elbo",
    ),
    ArmId.A2: (
        True, True, True, "categorical",
        "generic_fixed_frame_non_coboundary", "structured", "smoothing",
        "fixed", "exact", "complete_elbo",
    ),
    ArmId.A3: (
        True, True, True, "immediate_predecessor",
        "shared_vertex_coboundary", "structured", "smoothing", "absent",
        "absent", "complete_elbo",
    ),
    ArmId.A4: (
        True, True, False, "categorical", "shared_vertex_coboundary",
        "structured", "smoothing", "fixed", "exact", "complete_elbo",
    ),
}
_A5_BASE = (
    True, True, True, "categorical", "shared_vertex_coboundary",
    "structured", "smoothing", "fixed", "exact", "complete_elbo",
)
_A5_PROFILES: dict[str, tuple[object, ...]] = {
    "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1": _A5_BASE,
    "h6-a5-factorized-fixed-exact-complete-latent-smoothing-v1": (
        *_A5_BASE[:5], "factorized", *_A5_BASE[6:]
    ),
    "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1": (
        *_A5_BASE[:7], "learned", *_A5_BASE[8:]
    ),
    "h6-a5-structured-fixed-projection-complete-latent-smoothing-v1": (
        *_A5_BASE[:8], "moment_projected", _A5_BASE[9]
    ),
    "h6-a5-structured-fixed-exact-emission-latent-smoothing-v1": (
        *_A5_BASE[:9], "emission_only_ablation_non_elbo"
    ),
    "h6-a5-structured-fixed-exact-complete-latent-filtering-v1": (
        *_A5_BASE[:6], "filtering", *_A5_BASE[7:]
    ),
    (
        "h6-a5-structured-fixed-exact-complete-"
        "nolatent-norecognition-v1"
    ): (
        False, False, False, "absent", "absent", "absent", "absent",
        "absent", "absent", "complete_elbo",
    ),
}


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _deterministic_matrix(rows: int, columns: int, *, scale: float) -> Tensor:
    values = torch.arange(rows * columns, dtype=torch.float64)
    values = values.reshape(rows, columns)
    centered = values - 0.5 * max(0, rows * columns - 1)
    return scale * centered / max(1, rows * columns)


def _identity_perturbation(dimension: int, ordinal: int) -> Tensor:
    return torch.eye(dimension, dtype=torch.float64) + _deterministic_matrix(
        dimension, dimension, scale=0.01 * (ordinal + 1)
    )


def _exact_prefix(
    prefix: object,
    *,
    vocabulary: VocabularyIdentity,
    maximum_receiver: int | None,
) -> CausalPrefix:
    if type(prefix) is not CausalPrefix:
        raise ValueError("prefix must be an exact target-free CausalPrefix")
    prefix.__post_init__()
    if prefix.vocabulary != vocabulary:
        raise ValueError("causal prefix vocabulary does not match the arm")
    if maximum_receiver is not None and prefix.receiver_t > maximum_receiver:
        raise ValueError("causal prefix exceeds the configured arm horizon")
    return prefix


class CausalAutoregressiveModel(nn.Module):
    """Normalized conventional AR categorical model used by A0/no-latent."""

    def __init__(
        self,
        *,
        vocabulary: VocabularyIdentity,
        emission_width: int,
        family_label: str,
    ) -> None:
        super().__init__()
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        vocabulary.__post_init__()
        if type(emission_width) is not int or emission_width <= 0:
            raise ValueError("emission_width must be a positive integer")
        self.vocabulary = vocabulary
        self.emission_width = emission_width
        self.family_label = family_label
        self.token_embedding = nn.Embedding(
            vocabulary.size, emission_width, dtype=torch.float64
        )
        with torch.no_grad():
            self.token_embedding.weight.copy_(
                _deterministic_matrix(
                    vocabulary.size, emission_width, scale=0.125
                )
            )
        self.autoregressive_bos_context = nn.Parameter(
            torch.linspace(
                -0.05, 0.05, emission_width, dtype=torch.float64
            )
        )
        self.autoregressive_emission_weight = nn.Parameter(
            _deterministic_matrix(
                vocabulary.size, emission_width, scale=0.25
            )
        )
        self.autoregressive_emission_bias = nn.Parameter(
            torch.zeros(vocabulary.size, dtype=torch.float64)
        )
        self.elbo_factor_inventory: tuple[ElboPartition, ...] = ("emission",)
        self.elbo_inventory_sha256 = _owned_hash(
            "vfe4.h6.arm-elbo-inventory.v1",
            {
                "family": family_label,
                "partitions": self.elbo_factor_inventory,
            },
        )

    def prefix_log_probs(self, prefix: CausalPrefix) -> Tensor:
        checked = _exact_prefix(
            prefix, vocabulary=self.vocabulary, maximum_receiver=None
        )
        tokens = checked.token_ids
        context = (
            self.token_embedding(tokens).mean(dim=0)
            if tokens.numel()
            else self.autoregressive_bos_context
        )
        logits = (
            self.autoregressive_emission_weight @ context
            + self.autoregressive_emission_bias
        )
        return torch.log_softmax(logits, dim=0)


class LatentLanguageArmModel(nn.Module):
    """Live normalized latent family for the literal A1--A5 arm semantics."""

    def __init__(
        self,
        *,
        arm: ArmId,
        vocabulary: VocabularyIdentity,
        horizon: int,
        emission_width: int,
        latent_width: int,
        state_channel_enabled: bool,
        model_channel_enabled: bool,
        source_mode: Literal["absent", "immediate_predecessor", "categorical"],
        map_mode: Literal[
            "absent",
            "generic_fixed_frame_non_coboundary",
            "shared_vertex_coboundary",
        ],
    ) -> None:
        super().__init__()
        if arm not in (ArmId.A1, ArmId.A2, ArmId.A3, ArmId.A4, ArmId.A5):
            raise ValueError("latent model requires one of A1 through A5")
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        vocabulary.__post_init__()
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(emission_width) is not int or emission_width <= 0:
            raise ValueError("emission_width must be a positive integer")
        if type(latent_width) is not int or latent_width <= 0:
            raise ValueError("latent_width must be a positive integer")
        if not state_channel_enabled:
            raise ValueError("every latent H6 arm requires the state channel")
        if source_mode not in ("absent", "immediate_predecessor", "categorical"):
            raise ValueError("unsupported source mode")
        if map_mode not in (
            "absent",
            "generic_fixed_frame_non_coboundary",
            "shared_vertex_coboundary",
        ):
            raise ValueError("unsupported map mode")

        self.arm = arm
        self.vocabulary = vocabulary
        self.horizon = horizon
        self.emission_width = emission_width
        self.latent_width = latent_width
        self.state_channel_enabled = state_channel_enabled
        self.model_channel_enabled = model_channel_enabled
        self.source_mode = source_mode
        self.map_mode = map_mode

        self.initial_state_mean = nn.Parameter(
            torch.zeros(latent_width, dtype=torch.float64)
        )
        self.initial_state_log_scale = nn.Parameter(
            torch.zeros(latent_width, dtype=torch.float64)
        )
        self.state_transition_bias = nn.Parameter(
            torch.zeros(latent_width, dtype=torch.float64)
        )
        self.state_transition_log_scale = nn.Parameter(
            torch.zeros(latent_width, dtype=torch.float64)
        )

        if model_channel_enabled:
            self.initial_model_mean = nn.Parameter(
                torch.zeros(latent_width, dtype=torch.float64)
            )
            self.initial_model_log_scale = nn.Parameter(
                torch.zeros(latent_width, dtype=torch.float64)
            )
            self.model_transition_bias = nn.Parameter(
                torch.zeros(latent_width, dtype=torch.float64)
            )
            self.model_transition_log_scale = nn.Parameter(
                torch.zeros(latent_width, dtype=torch.float64)
            )

        if arm is ArmId.A1:
            self.ordinary_latent_transition_weight = nn.Parameter(
                _identity_perturbation(latent_width, 0)
            )

        edge_count = horizon * (horizon + 1) // 2
        if map_mode == "generic_fixed_frame_non_coboundary":
            self.generic_fixed_frame_state_edge_maps = nn.ParameterList(
                [
                    nn.Parameter(_identity_perturbation(latent_width, index))
                    for index in range(edge_count)
                ]
            )
            self.generic_fixed_frame_model_edge_maps = nn.ParameterList(
                [
                    nn.Parameter(
                        _identity_perturbation(latent_width, edge_count + index)
                    )
                    for index in range(edge_count)
                ]
            )
        elif map_mode == "shared_vertex_coboundary":
            self.state_vertex_phi = nn.ParameterList(
                [
                    nn.Parameter(
                        _deterministic_matrix(
                            latent_width,
                            latent_width,
                            scale=0.005 * vertex,
                        )
                    )
                    for vertex in range(1, horizon + 1)
                ]
            )
            if model_channel_enabled:
                self.model_vertex_phi = nn.ParameterList(
                    [
                        nn.Parameter(
                            _deterministic_matrix(
                                latent_width,
                                latent_width,
                                scale=0.0075 * vertex,
                            )
                        )
                        for vertex in range(1, horizon + 1)
                    ]
                )

        if model_channel_enabled and arm in (ArmId.A2, ArmId.A5):
            self.full_same_receiver_b = nn.ParameterList(
                [
                    nn.Parameter(
                        _deterministic_matrix(
                            latent_width,
                            latent_width,
                            scale=0.025 * receiver_t,
                        )
                    )
                    for receiver_t in range(1, horizon + 1)
                ]
            )

        if source_mode == "categorical":
            # One fixed zero anchor removes the categorical additive-shift
            # null direction. Receiver one has no trainable source scalar.
            self.state_source_free_logits = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(receiver_t - 1, dtype=torch.float64)
                    )
                    for receiver_t in range(2, horizon + 1)
                ]
            )
            if model_channel_enabled:
                self.model_source_free_logits = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.zeros(receiver_t - 1, dtype=torch.float64)
                        )
                        for receiver_t in range(2, horizon + 1)
                    ]
                )

        self.emission_state_projection = nn.Parameter(
            _deterministic_matrix(
                emission_width, latent_width, scale=0.125
            )
        )
        if model_channel_enabled:
            self.emission_model_projection = nn.Parameter(
                _deterministic_matrix(
                    emission_width, latent_width, scale=0.1
                )
            )
        self.normalized_emission_head = nn.Parameter(
            _deterministic_matrix(
                vocabulary.size, emission_width, scale=0.2
            )
        )
        self.normalized_emission_bias = nn.Parameter(
            torch.zeros(vocabulary.size, dtype=torch.float64)
        )

        self.elbo_factor_inventory = self._inventory()
        self.elbo_inventory_sha256 = _owned_hash(
            "vfe4.h6.arm-elbo-inventory.v1",
            {
                "arm": arm.value,
                "source_mode": source_mode,
                "model_channel_enabled": model_channel_enabled,
                "partitions": self.elbo_factor_inventory,
            },
        )

    def _inventory(self) -> tuple[ElboPartition, ...]:
        if self.arm is ArmId.A1:
            return (
                "initial",
                "state_transition",
                "emission",
                "entropy",
            )
        if self.arm is ArmId.A3:
            return (
                "initial",
                "state_transition",
                "model_transition",
                "emission",
                "entropy",
            )
        if self.arm is ArmId.A4:
            return (
                "initial",
                "state_source",
                "state_transition",
                "emission",
                "entropy",
            )
        return (
            "initial",
            "state_source",
            "model_source",
            "state_transition",
            "model_transition",
            "emission",
            "entropy",
        )

    def _vector(self, value: Tensor, name: str) -> Tensor:
        if (
            type(value) is not Tensor
            or value.dtype is not torch.float64
            or value.shape != (self.latent_width,)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"{name} must be a finite float64 latent-width vector"
            )
        return value

    def _receiver_source(self, receiver_t: int, source_j: int) -> None:
        if (
            type(receiver_t) is not int
            or not 1 <= receiver_t <= self.horizon
        ):
            raise ValueError("receiver_t lies outside the configured horizon")
        if type(source_j) is not int or not 0 <= source_j < receiver_t:
            raise ValueError("source_j must be an earlier declared vertex")

    @staticmethod
    def _edge_index(receiver_t: int, source_j: int) -> int:
        return receiver_t * (receiver_t - 1) // 2 + source_j

    def vertex_frame(self, channel: Channel, vertex_t: int) -> Tensor:
        if self.map_mode != "shared_vertex_coboundary":
            raise ValueError("vertex frames exist only for coboundary arms")
        if type(vertex_t) is not int or not 0 <= vertex_t <= self.horizon:
            raise ValueError("vertex_t lies outside the configured horizon")
        if channel == "state":
            parameters = self.state_vertex_phi
        elif channel == "model" and self.model_channel_enabled:
            parameters = self.model_vertex_phi
        else:
            raise ValueError("requested channel is structurally absent")
        if vertex_t == 0:
            return torch.eye(
                self.latent_width,
                dtype=parameters[0].dtype,
                device=parameters[0].device,
            )
        return torch.matrix_exp(parameters[vertex_t - 1])

    def edge_map(
        self, channel: Channel, receiver_t: int, source_j: int
    ) -> Tensor:
        self._receiver_source(receiver_t, source_j)
        if self.map_mode == "generic_fixed_frame_non_coboundary":
            index = self._edge_index(receiver_t, source_j)
            if channel == "state":
                return self.generic_fixed_frame_state_edge_maps[index]
            if channel == "model" and self.model_channel_enabled:
                return self.generic_fixed_frame_model_edge_maps[index]
            raise ValueError("requested channel is structurally absent")
        if self.map_mode == "shared_vertex_coboundary":
            receiver_frame = self.vertex_frame(channel, receiver_t)
            source_frame = self.vertex_frame(channel, source_j)
            return receiver_frame @ torch.linalg.inv(source_frame)
        raise ValueError("ordinary A1 has no internal map sector")

    def state_source_log_probs(self, receiver_t: int) -> Tensor:
        if self.source_mode != "categorical":
            raise ValueError("state source categorical variable is absent")
        self._receiver_source(receiver_t, 0)
        logits = self._canonical_source_logits(
            receiver_t=receiver_t,
            free_rows=self.state_source_free_logits,
        )
        return masked_log_softmax_from_parents(
            logits, tuple(range(receiver_t)), receiver_t
        ).log_probs

    def model_source_log_probs(self, receiver_t: int) -> Tensor:
        if self.source_mode != "categorical" or not self.model_channel_enabled:
            raise ValueError("model source categorical variable is absent")
        self._receiver_source(receiver_t, 0)
        logits = self._canonical_source_logits(
            receiver_t=receiver_t,
            free_rows=self.model_source_free_logits,
        )
        return masked_log_softmax_from_parents(
            logits, tuple(range(receiver_t)), receiver_t
        ).log_probs

    @staticmethod
    def _canonical_source_logits(
        *,
        receiver_t: int,
        free_rows: nn.ParameterList,
    ) -> Tensor:
        if receiver_t == 1:
            if len(free_rows):
                exemplar = free_rows[0]
                return torch.zeros(
                    1, dtype=exemplar.dtype, device=exemplar.device
                )
            return torch.zeros(1, dtype=torch.float64)
        free = free_rows[receiver_t - 2]
        anchor = torch.zeros(1, dtype=free.dtype, device=free.device)
        return torch.cat((free, anchor))

    @staticmethod
    def _diagonal_gaussian_log_prob(
        value: Tensor, mean: Tensor, log_scale: Tensor
    ) -> Tensor:
        standardized = (value - mean) * torch.exp(-log_scale)
        return -0.5 * torch.sum(
            standardized.square()
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        )

    def initial_log_prob(
        self, *, state: Tensor, model: Tensor | None = None
    ) -> Tensor:
        state_value = self._vector(state, "state")
        result = self._diagonal_gaussian_log_prob(
            state_value, self.initial_state_mean, self.initial_state_log_scale
        )
        if self.model_channel_enabled:
            if model is None:
                raise ValueError("model initial state is required")
            model_value = self._vector(model, "model")
            result = result + self._diagonal_gaussian_log_prob(
                model_value,
                self.initial_model_mean,
                self.initial_model_log_scale,
            )
        elif model is not None:
            raise ValueError("model channel is structurally absent")
        return result

    def model_transition_mean(
        self,
        *,
        receiver_t: int,
        source_j: int,
        source_model: Tensor,
    ) -> Tensor:
        if not self.model_channel_enabled:
            raise ValueError("model channel is structurally absent")
        source = self._vector(source_model, "source_model")
        transport = self.edge_map("model", receiver_t, source_j)
        return transport @ source + self.model_transition_bias

    def state_transition_mean(
        self,
        *,
        receiver_t: int,
        source_j: int,
        source_state: Tensor,
        current_model: Tensor | None,
    ) -> Tensor:
        source = self._vector(source_state, "source_state")
        if self.arm is ArmId.A1:
            if source_j != receiver_t - 1:
                raise ValueError("A1 uses only the immediate predecessor")
            mean = (
                self.ordinary_latent_transition_weight @ source
                + self.state_transition_bias
            )
        else:
            mean = (
                self.edge_map("state", receiver_t, source_j) @ source
                + self.state_transition_bias
            )
        if self.arm in (ArmId.A2, ArmId.A5):
            if current_model is None:
                raise ValueError("same-receiver B_t requires current model")
            mean = (
                mean
                + self.full_same_receiver_b[receiver_t - 1]
                @ self._vector(current_model, "current_model")
            )
        elif current_model is not None and not self.model_channel_enabled:
            raise ValueError("model channel is structurally absent")
        return mean

    def emission_log_probs(
        self, *, state: Tensor, model: Tensor | None = None
    ) -> Tensor:
        state_value = self._vector(state, "state")
        hidden = self.emission_state_projection @ state_value
        if self.model_channel_enabled:
            if model is None:
                raise ValueError("model channel value is required")
            hidden = (
                hidden
                + self.emission_model_projection
                @ self._vector(model, "model")
            )
        elif model is not None:
            raise ValueError("model channel is structurally absent")
        logits = (
            self.normalized_emission_head @ torch.tanh(hidden)
            + self.normalized_emission_bias
        )
        return torch.log_softmax(logits, dim=0)


ArmModel = CausalAutoregressiveModel | LatentLanguageArmModel


class ArmTargetFreeProposalAdapter:
    """Bootstrap proposal specialized by one immutable arm/model-state pair."""

    proposal_mode = "generative_bootstrap"

    def __init__(
        self,
        *,
        model: ArmModel,
        model_family_sha256: str,
    ) -> None:
        if type(model) not in (
            CausalAutoregressiveModel,
            LatentLanguageArmModel,
        ):
            raise ValueError("unsupported exact arm model")
        self.model = model
        self.vocabulary = model.vocabulary
        self.vocabulary_sha256 = vocabulary_identity_sha256(
            model.vocabulary
        )
        self.model_family_sha256 = model_family_sha256
        self.model_state_sha256 = canonical_model_state_sha256(model)
        self.proposal_identity_sha256 = _owned_hash(
            "vfe4.h6.arm-target-free-proposal.v1",
            {
                "model_family_sha256": model_family_sha256,
                "model_state_sha256": self.model_state_sha256,
                "vocabulary_sha256": self.vocabulary_sha256,
                "proposal_mode": self.proposal_mode,
            },
        )

    def assert_current_state(self) -> None:
        if canonical_model_state_sha256(self.model) != self.model_state_sha256:
            raise ValueError(
                "arm proposal model state changed; rebuild the predictive boundary"
            )

    @staticmethod
    def _key(
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
        checked = _exact_prefix(
            prefix_tokens,
            vocabulary=self.vocabulary,
            maximum_receiver=(
                self.model.horizon
                if type(self.model) is LatentLanguageArmModel
                else None
            ),
        )
        if checked.receiver_t != 1 or checked.token_ids.numel() != 0:
            raise ValueError("proposal initialization requires the empty prefix")
        if type(estimator_rng) is not EstimatorStream:
            raise ValueError("estimator_rng must be an exact EstimatorStream")
        if type(particle_count) is not int or particle_count <= 0:
            raise ValueError("particle_count must be a positive integer")

        if type(self.model) is CausalAutoregressiveModel:
            return (
                ProposalPopulation.create(
                    {
                        "autoregressive_history_marker": torch.zeros(
                            (particle_count, 1, 1), dtype=torch.float64
                        )
                    }
                ),
                (),
            )

        states: list[Tensor] = []
        models: list[Tensor] = []
        for particle_index in range(particle_count):
            state_noise = torch.tensor(
                estimator_rng.gaussian(
                    CounterKey(
                        estimator_rng.stream_seed,
                        checked.prefix_sha256,
                        0,
                        CounterPurpose.INITIAL_STATE_GAUSSIAN,
                        particle_index,
                    ),
                    count=self.model.latent_width,
                ),
                dtype=torch.float64,
            )
            states.append(
                self.model.initial_state_mean
                + torch.exp(self.model.initial_state_log_scale) * state_noise
            )
            if self.model.model_channel_enabled:
                model_noise = torch.tensor(
                    estimator_rng.gaussian(
                        CounterKey(
                            estimator_rng.stream_seed,
                            checked.prefix_sha256,
                            0,
                            CounterPurpose.INITIAL_MODEL_GAUSSIAN,
                            particle_index,
                        ),
                        count=self.model.latent_width,
                    ),
                    dtype=torch.float64,
                )
                models.append(
                    self.model.initial_model_mean
                    + torch.exp(self.model.initial_model_log_scale)
                    * model_noise
                )
        components = {
            "state_history": torch.stack(states, dim=0).unsqueeze(1)
        }
        if self.model.model_channel_enabled:
            components["model_history"] = (
                torch.stack(models, dim=0).unsqueeze(1)
            )
        draws = 2 * ((self.model.latent_width + 1) // 2)
        consumption = [
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.INITIAL_STATE_GAUSSIAN,
                particle_count=particle_count,
                draws_per_particle=draws,
            )
        ]
        if self.model.model_channel_enabled:
            consumption.append(
                CounterConsumption.create(
                    position=0,
                    purpose=CounterPurpose.INITIAL_MODEL_GAUSSIAN,
                    particle_count=particle_count,
                    draws_per_particle=draws,
                )
            )
        return ProposalPopulation.create(components), tuple(consumption)

    def _categorical_source(
        self,
        *,
        bank: Channel,
        prefix: CausalPrefix,
        stream: EstimatorStream,
        particle_index: int,
    ) -> int:
        if type(self.model) is not LatentLanguageArmModel:
            raise ValueError("autoregressive arm has no source variables")
        purpose = (
            CounterPurpose.STATE_SOURCE_CATEGORICAL
            if bank == "state"
            else CounterPurpose.MODEL_SOURCE_CATEGORICAL
        )
        log_probs = (
            self.model.state_source_log_probs(prefix.receiver_t)
            if bank == "state"
            else self.model.model_source_log_probs(prefix.receiver_t)
        )
        return stream.categorical(
            self._key(stream, prefix, purpose, particle_index), log_probs
        )

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
        checked = _exact_prefix(
            prefix_tokens,
            vocabulary=self.vocabulary,
            maximum_receiver=(
                self.model.horizon
                if type(self.model) is LatentLanguageArmModel
                else None
            ),
        )
        if type(estimator_rng) is not EstimatorStream:
            raise ValueError("estimator_rng must be an exact EstimatorStream")

        if type(self.model) is CausalAutoregressiveModel:
            marker = population.component("autoregressive_history_marker")
            emissions = self.model.prefix_log_probs(checked).repeat(
                population.particle_count, 1
            )
            next_population = ProposalPopulation.create(
                {
                    "autoregressive_history_marker": torch.cat(
                        [
                            marker,
                            torch.zeros(
                                (population.particle_count, 1, 1),
                                dtype=torch.float64,
                            ),
                        ],
                        dim=1,
                    )
                }
            )
            return ProposalStep.create(
                position=checked.receiver_t,
                population=next_population,
                emission_log_probs=emissions,
                counter_consumption=(),
                proposal_identity_sha256=self.proposal_identity_sha256,
            )

        position = checked.receiver_t
        state_history = population.component("state_history")
        expected = (
            population.particle_count,
            position,
            self.model.latent_width,
        )
        if state_history.shape != expected:
            raise ValueError("state proposal history shape is stale")
        model_history = (
            population.component("model_history")
            if self.model.model_channel_enabled
            else None
        )
        if model_history is not None and model_history.shape != expected:
            raise ValueError("model proposal history shape is stale")

        current_states: list[Tensor] = []
        current_models: list[Tensor] = []
        emission_rows: list[Tensor] = []
        for particle_index in range(population.particle_count):
            state_source = (
                self._categorical_source(
                    bank="state",
                    prefix=checked,
                    stream=estimator_rng,
                    particle_index=particle_index,
                )
                if self.model.source_mode == "categorical"
                else position - 1
            )
            current_model: Tensor | None = None
            if self.model.model_channel_enabled:
                model_source = (
                    self._categorical_source(
                        bank="model",
                        prefix=checked,
                        stream=estimator_rng,
                        particle_index=particle_index,
                    )
                    if self.model.source_mode == "categorical"
                    else position - 1
                )
                model_noise = torch.tensor(
                    estimator_rng.gaussian(
                        self._key(
                            estimator_rng,
                            checked,
                            CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
                            particle_index,
                        ),
                        count=self.model.latent_width,
                    ),
                    dtype=torch.float64,
                )
                current_model = self.model.model_transition_mean(
                    receiver_t=position,
                    source_j=model_source,
                    source_model=model_history[particle_index, model_source],
                ) + torch.exp(self.model.model_transition_log_scale) * model_noise
                current_models.append(current_model)

            state_noise = torch.tensor(
                estimator_rng.gaussian(
                    self._key(
                        estimator_rng,
                        checked,
                        CounterPurpose.STATE_TRANSITION_GAUSSIAN,
                        particle_index,
                    ),
                    count=self.model.latent_width,
                ),
                dtype=torch.float64,
            )
            current_state = self.model.state_transition_mean(
                receiver_t=position,
                source_j=state_source,
                source_state=state_history[particle_index, state_source],
                current_model=current_model,
            ) + torch.exp(self.model.state_transition_log_scale) * state_noise
            current_states.append(current_state)
            emission_rows.append(
                self.model.emission_log_probs(
                    state=current_state, model=current_model
                )
            )

        components = {
            "state_history": torch.cat(
                [state_history, torch.stack(current_states).unsqueeze(1)],
                dim=1,
            )
        }
        if model_history is not None:
            components["model_history"] = torch.cat(
                [model_history, torch.stack(current_models).unsqueeze(1)],
                dim=1,
            )

        consumption: list[CounterConsumption] = []
        if self.model.source_mode == "categorical":
            consumption.append(
                CounterConsumption.create(
                    position=position,
                    purpose=CounterPurpose.STATE_SOURCE_CATEGORICAL,
                    particle_count=population.particle_count,
                    draws_per_particle=1,
                )
            )
            if self.model.model_channel_enabled:
                consumption.append(
                    CounterConsumption.create(
                        position=position,
                        purpose=CounterPurpose.MODEL_SOURCE_CATEGORICAL,
                        particle_count=population.particle_count,
                        draws_per_particle=1,
                    )
                )
        draws = 2 * ((self.model.latent_width + 1) // 2)
        consumption.append(
            CounterConsumption.create(
                position=position,
                purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
                particle_count=population.particle_count,
                draws_per_particle=draws,
            )
        )
        if self.model.model_channel_enabled:
            consumption.append(
                CounterConsumption.create(
                    position=position,
                    purpose=CounterPurpose.MODEL_TRANSITION_GAUSSIAN,
                    particle_count=population.particle_count,
                    draws_per_particle=draws,
                )
            )
        return ProposalStep.create(
            position=position,
            population=ProposalPopulation.create(components),
            emission_log_probs=torch.stack(emission_rows),
            counter_consumption=tuple(consumption),
            proposal_identity_sha256=self.proposal_identity_sha256,
        )


def _predictive_boundary(
    *,
    config: ArmConfig,
    model: ArmModel,
    model_family_sha256: str,
    estimator_spec: EstimatorSpec | None = None,
) -> tuple[ArmTargetFreeProposalAdapter, BootstrapSmcPredictor]:
    proposal = ArmTargetFreeProposalAdapter(
        model=model, model_family_sha256=model_family_sha256
    )
    if estimator_spec is None:
        estimator_spec = EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=4,
            resampling="systematic_ess_half",
        )
    elif type(estimator_spec) is not EstimatorSpec:
        raise ValueError("estimator_spec must be an exact EstimatorSpec")
    estimator_spec.__post_init__()
    estimator_identity = EstimatorIdentity.from_spec(estimator_spec)
    predictor = BootstrapSmcPredictor(
        proposal=proposal,
        estimator_spec=estimator_spec,
        estimator_identity=estimator_identity,
        predictor_config_sha256=config.config_sha256,
        data_safety_sha256=H6_TARGET_FREE_DATA_SAFETY_SHA256,
    )
    return proposal, predictor


@dataclass(frozen=True)
class BuiltArm:
    config: ArmConfig
    model: ArmModel
    recognition_store: LanguageRecognitionParameterStore | None
    proposal: TargetFreeProposalAdapter
    predictor: BootstrapSmcPredictor
    parameter_roles: tuple[ParameterRoleRecord, ...]
    optimizer_bindings: tuple[OptimizerBinding, ...]
    flop_terms: tuple[FlopTerm, ...]
    model_family_sha256: str
    elbo_factor_inventory: tuple[ElboPartition, ...]
    elbo_inventory_sha256: str
    training_flop_ledger_complete: Literal[False]
    training_flop_obligations: tuple[str, ...]

    def rebuild_predictive_boundary(
        self,
        estimator_spec: EstimatorSpec | None = None,
    ) -> tuple[ArmTargetFreeProposalAdapter, BootstrapSmcPredictor]:
        """Freeze a fresh adapter/predictor identity around current parameters."""

        return _predictive_boundary(
            config=self.config,
            model=self.model,
            model_family_sha256=self.model_family_sha256,
            estimator_spec=estimator_spec,
        )


def _semantic_role(config: ArmConfig, qualified_name: str) -> str:
    if "generic_fixed_frame" in qualified_name:
        return "generic_fixed_frame_non_coboundary_edge_map"
    if "vertex_phi" in qualified_name:
        return "shared_vertex_coboundary_frame"
    if "full_same_receiver_b" in qualified_name:
        return "full_same_receiver_b"
    if "state_source" in qualified_name:
        return "categorical_state_source_bank"
    if "model_source" in qualified_name:
        return "categorical_model_source_bank"
    if "ordinary_latent" in qualified_name:
        return "ordinary_latent_state_chain"
    if "autoregressive" in qualified_name or "token_embedding" in qualified_name:
        return "autoregressive_categorical_emission"
    if "recognition" in qualified_name:
        return "recognition_parameter_store"
    if "emission" in qualified_name:
        return "normalized_emission"
    return f"{config.arm.value.lower()}_active_latent_factor"


def _parameter_records(
    *,
    config: ArmConfig,
    model: ArmModel,
    recognition_store: LanguageRecognitionParameterStore | None,
) -> tuple[
    tuple[ParameterRoleRecord, ...],
    tuple[OptimizerBinding, ...],
    tuple[FlopTerm, ...],
]:
    model_phase = (
        TrainingPhase.MODEL_ADAMW.value
        if config.latent_enabled
        else TrainingPhase.MODEL_CE_ADAMW.value
    )
    roles: list[ParameterRoleRecord] = []
    model_ids: list[int] = []
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        model_ids.append(parameter_id)
        roles.append(
            ParameterRoleRecord.create(
                qualified_name=f"model.{name}",
                parameter_id=parameter_id,
                role=_semantic_role(config, name),
                phase=model_phase,
                scalar_count=parameter.numel(),
            )
        )

    recognition_ids: list[int] = []
    if recognition_store is not None:
        for name, parameter in recognition_store.named_parameters():
            parameter_id = id(parameter)
            recognition_ids.append(parameter_id)
            roles.append(
                ParameterRoleRecord.create(
                    qualified_name=f"recognition_store.{name}",
                    parameter_id=parameter_id,
                    role="recognition_parameter_store",
                    phase=TrainingPhase.RECOGNITION_ADAMW.value,
                    scalar_count=parameter.numel(),
                )
            )

    bindings: list[OptimizerBinding] = []
    if recognition_ids:
        bindings.append(
            OptimizerBinding.create(
                phase=TrainingPhase.RECOGNITION_ADAMW.value,
                optimizer_class="AdamW",
                optimizer_policy_sha256=(
                    H6_ADAMW_POLICY.optimizer_policy_sha256
                ),
                parameter_ids=tuple(recognition_ids),
            )
        )
    bindings.append(
        OptimizerBinding.create(
            phase=model_phase,
            optimizer_class="AdamW",
            optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256,
            parameter_ids=tuple(model_ids),
        )
    )

    model_scalars = sum(parameter.numel() for parameter in model.parameters())
    flop_terms: list[FlopTerm] = []
    if recognition_store is not None:
        recognition_scalars = sum(
            parameter.numel() for parameter in recognition_store.parameters()
        )
        flop_terms.append(
            FlopTerm.create(
                phase=TrainingPhase.RECOGNITION_ADAMW.value,
                operation=(
                    "INCOMPLETE_LOWER_BOUND_"
                    "recognition_parameter_update_arithmetic"
                ),
                repetitions=1,
                arithmetic_flops_per_repetition=24 * recognition_scalars + 3,
                bytes_copied_per_repetition=0,
            )
        )
        copied_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in recognition_store.parameters()
        )
        flop_terms.append(
            FlopTerm.create(
                phase=TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT.value,
                operation="immutable_detached_recognition_snapshot",
                repetitions=1,
                arithmetic_flops_per_repetition=0,
                bytes_copied_per_repetition=copied_bytes,
            )
        )
    flop_terms.append(
        FlopTerm.create(
            phase=model_phase,
            operation="INCOMPLETE_LOWER_BOUND_model_parameter_update_arithmetic",
            repetitions=1,
            arithmetic_flops_per_repetition=24 * model_scalars + 3,
            bytes_copied_per_repetition=0,
        )
    )
    return tuple(roles), tuple(bindings), tuple(flop_terms)


def _training_flop_obligations(
    model: ArmModel,
    recognition_store: LanguageRecognitionParameterStore | None,
) -> tuple[str, ...]:
    obligations = [
        "forward and backward operator-level arithmetic is not fully counted",
    ]
    if (
        type(model) is LatentLanguageArmModel
        and model.map_mode == "shared_vertex_coboundary"
    ):
        obligations.append(
            "matrix_exp and matrix inverse/solve arithmetic is not counted"
        )
    if recognition_store is not None:
        obligations.append(
            "recognition Cholesky assembly and Gaussian-law arithmetic is not counted"
        )
    return tuple(obligations)


def _model_family_sha256(config: ArmConfig) -> str:
    return _owned_hash(
        "vfe4.h6.arm-model-family.v1",
        {
            "config_sha256": config.config_sha256,
            "factory": f"build_{config.arm.value.lower()}@h6-arm-v1",
        },
    )


def literal_arm_semantic_payload(config: ArmConfig) -> dict[str, object]:
    """Return the validated literal semantic profile, excluding capacity."""

    if type(config) is not ArmConfig:
        raise ValueError("config must be an exact ArmConfig")
    config.__post_init__()
    if config.arm is ArmId.A5:
        expected = _A5_PROFILES.get(config.config_id)
    else:
        expected = _BASE_PROFILES.get(config.arm)
        if config.config_id != _BASE_PROFILE_IDS.get(config.arm):
            expected = None
    observed = tuple(getattr(config, name) for name in _SEMANTIC_FIELDS)
    if expected is None or observed != expected:
        raise ValueError(
            "config does not match the exact literal semantic profile for its arm"
        )
    return dict(zip(_SEMANTIC_FIELDS, observed, strict=True))


def shared_a2_a5_semantic_payload(config: ArmConfig) -> dict[str, object]:
    """Return the fields that the frozen A2/A5 map contrast must share."""

    if type(config) is not ArmConfig or config.arm not in (ArmId.A2, ArmId.A5):
        raise ValueError("shared map payload requires an exact A2 or A5 config")
    payload = literal_arm_semantic_payload(config)
    if config.arm is ArmId.A5 and config.config_id != (
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
    ):
        raise ValueError("the MAP contrast uses only the primary A5 profile")
    payload.pop("map_mode")
    return payload


def _require_builder_arm(config: ArmConfig, arm: ArmId) -> None:
    if type(config) is not ArmConfig:
        raise ValueError("config must be an exact ArmConfig")
    config.__post_init__()
    if config.arm is not arm:
        raise ValueError("config arm does not match the explicit builder arm")
    literal_arm_semantic_payload(config)


def _construct(config: ArmConfig) -> BuiltArm:
    allocation = config.capacity_allocation
    if not config.latent_enabled:
        model: ArmModel = CausalAutoregressiveModel(
            vocabulary=config.vocabulary,
            emission_width=allocation.emission_width,
            family_label=(
                "a0_autoregressive"
                if config.arm is ArmId.A0
                else "a5_nolatent_norecognition"
            ),
        )
        recognition_store = None
    else:
        if allocation.latent_width is None:
            raise ValueError("latent arm requires latent_width")
        model = LatentLanguageArmModel(
            arm=config.arm,
            vocabulary=config.vocabulary,
            horizon=config.horizon,
            emission_width=allocation.emission_width,
            latent_width=allocation.latent_width,
            state_channel_enabled=config.state_channel_enabled,
            model_channel_enabled=config.model_channel_enabled,
            source_mode=config.source_mode,
            map_mode=config.map_mode,
        )
        if (
            allocation.recognition_width is None
            or config.recognition_family == "absent"
        ):
            raise ValueError("latent arm requires a recognition allocation/family")
        recognition_store = LanguageRecognitionParameterStore(
            vocabulary=config.vocabulary,
            horizon=config.horizon,
            latent_width=allocation.latent_width,
            recognition_width=allocation.recognition_width,
            channel_count=2 if config.model_channel_enabled else 1,
            family=config.recognition_family,
            conditioning_mode=config.recognition_conditioning,
        )

    family_sha256 = _model_family_sha256(config)
    proposal, predictor = _predictive_boundary(
        config=config, model=model, model_family_sha256=family_sha256
    )
    roles, bindings, flop_terms = _parameter_records(
        config=config, model=model, recognition_store=recognition_store
    )
    return BuiltArm(
        config,
        model,
        recognition_store,
        proposal,
        predictor,
        roles,
        bindings,
        flop_terms,
        family_sha256,
        model.elbo_factor_inventory,
        model.elbo_inventory_sha256,
        False,
        _training_flop_obligations(model, recognition_store),
    )


def build_a0(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A0)
    return _construct(config)


def build_a1(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A1)
    return _construct(config)


def build_a2(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A2)
    return _construct(config)


def build_a3(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A3)
    return _construct(config)


def build_a4(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A4)
    return _construct(config)


def build_a5(config: ArmConfig) -> BuiltArm:
    _require_builder_arm(config, ArmId.A5)
    return _construct(config)


def build_arm(arm: ArmId, config: ArmConfig) -> BuiltArm:
    if not isinstance(arm, ArmId):
        raise ValueError("arm must be an ArmId")
    if type(config) is not ArmConfig or config.arm is not arm:
        raise ValueError("arm and config arm must match exactly")
    if arm is ArmId.A0:
        return build_a0(config)
    if arm is ArmId.A1:
        return build_a1(config)
    if arm is ArmId.A2:
        return build_a2(config)
    if arm is ArmId.A3:
        return build_a3(config)
    if arm is ArmId.A4:
        return build_a4(config)
    if arm is ArmId.A5:
        return build_a5(config)
    raise ValueError("unsupported H6 arm")


__all__ = [
    "ARM_MATRIX_ROWS",
    "ARM_MATRIX_SHA256",
    "ArmConfig",
    "ArmMatrixRow",
    "ArmTargetFreeProposalAdapter",
    "BuiltArm",
    "CapacityAllocation",
    "CausalAutoregressiveModel",
    "H6_TARGET_FREE_DATA_SAFETY_SHA256",
    "LatentLanguageArmModel",
    "MatchingReport",
    "arm_matrix_sha256",
    "build_a0",
    "build_a1",
    "build_a2",
    "build_a3",
    "build_a4",
    "build_a5",
    "build_arm",
    "literal_arm_semantic_payload",
    "shared_a2_a5_semantic_payload",
]
