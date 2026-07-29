"""Production WikiText-103 arm modules and executable runtime construction.

The latent path stores a block-bidiagonal Cholesky factor of the recognition
precision.  Consequently its precision is block tridiagonal without ever
materializing a population ``[L,L]`` or ``[Lb,Lb]`` tensor.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vfe4.config.schema import TrainingConfig
from vfe4.data.windows import CausalPrefix
from vfe4.predictive.identities import EstimatorIdentity
from vfe4.predictive.prior import PriorPredictor
from vfe4.predictive.proposal import EstimatorStream
from vfe4.types.h6 import EstimatorSpec, VocabularyIdentity
from vfe4.types.training import (
    AdamWProfile,
    FinalizedWikiText103SourceRecord,
    GateStatus,
    ProductionTokenizerSpec,
    SchedulerProfile,
    SyntheticFixtureTokenizerSpec,
    WT103ArmSpec,
    owned_sha256,
)

from .engine import (
    ArmExecutionRuntime,
    ForwardTerms,
    RecognitionSnapshot,
    StepResult,
    WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256,
    train_step,
)
from .factories import (
    A0FactoryInputs,
    ArmMatchingReport,
    WT103FactorySetIdentity,
    build_wt103_a0,
    build_wt103_a5_fixed,
    build_wt103_a5_nolatent,
    build_wt103_a5_parent_specific,
)
from .production_observability import NumericalObservation, SourceObservation
from .wt103_models import (
    BuiltWT103Arm,
    ExecutionScope,
    OptimizerParameterBinding,
    WT103ArmRuntimeComponents,
)


ObjectiveKind = Literal[
    "cross_entropy",
    "complete_elbo",
    "emission_only_ablation_non_elbo",
]
PriorVariant = Literal["fixed", "parent_specific_pooled_prefix"]
InitializerClass = Literal[
    "xavier_uniform_gain_1",
    "zero_linear_bias",
    "identity_layer_norm_affine",
    "zero_layer_norm_bias",
    "identity_frame_via_zero_generator",
    "identity_block_precision_diagonal_factor",
    "zero_block_precision_lower_factor",
    "zero_source_parameter",
    "standard_normal_latent_parameter",
    "training_rng_seed",
]
_LOG_2PI = math.log(2.0 * math.pi)
_SOURCE_RECOGNITION_LAW_PAYLOAD = {
    "schema_version": "wt103-restricted-source-recognition-law-v1",
    "family": "euclidean_squared_distance_softmax",
    "variational_scope": "restricted_amortized_source_family",
    "exact_coordinate_update": False,
    "state_model_banks": "separate",
}
SOURCE_RECOGNITION_LAW_SHA256 = owned_sha256(
    "vfe4.wt103.restricted-source-recognition-law.v1",
    _SOURCE_RECOGNITION_LAW_PAYLOAD,
)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
    return value


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite exact float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _bf16_autocast(module: nn.Module):
    parameter = next(module.parameters())
    return torch.autocast(
        device_type=parameter.device.type,
        dtype=torch.bfloat16,
        enabled=(
            parameter.device.type == "cuda"
            and parameter.dtype is torch.float32
        ),
    )


def banded_parent_indices(
    sequence_length: int,
    source_lookback: int,
    *,
    device: torch.device,
) -> Tensor:
    """Return the closed left-parent band as ``[L,W]`` int64 indices."""

    _positive_int(sequence_length, "sequence_length")
    _positive_int(source_lookback, "source_lookback")
    rows = torch.full(
        (sequence_length, source_lookback),
        -1,
        dtype=torch.int64,
        device=device,
    )
    for receiver in range(1, sequence_length):
        start = max(0, receiver - source_lookback)
        parents = torch.arange(start, receiver, device=device)
        rows[receiver, -parents.numel() :] = parents
    return rows


def causal_observation_history(
    input_ids: Tensor,
    *,
    receiver_position: int,
) -> Tensor:
    """Return exactly the emitted observations available before one receiver."""

    if (
        type(input_ids) is not Tensor
        or input_ids.ndim != 2
        or input_ids.dtype is not torch.int64
        or type(receiver_position) is not int
        or not 1 <= receiver_position <= input_ids.shape[1]
    ):
        raise ValueError(
            "causal observation history requires [B,L] int64 and 1 <= r <= L"
        )
    return input_ids[:, :receiver_position]


@dataclass(slots=True)
class BandedRecognitionState:
    """Live smoothing law in block-bidiagonal precision-factor form."""

    mean: Tensor
    diagonal_factor: Tensor
    lower_blocks: Tensor
    parent_indices: Tensor

    def validate(self) -> None:
        if (
            type(self.mean) is not Tensor
            or self.mean.ndim != 3
            or type(self.diagonal_factor) is not Tensor
            or self.diagonal_factor.shape != self.mean.shape
            or type(self.lower_blocks) is not Tensor
            or self.lower_blocks.ndim != 4
            or self.lower_blocks.shape[:2]
            != (self.mean.shape[0], self.mean.shape[1] - 1)
            or self.lower_blocks.shape[2:]
            != (self.mean.shape[2], self.mean.shape[2])
            or type(self.parent_indices) is not Tensor
            or self.parent_indices.ndim != 2
            or self.parent_indices.shape[0] != self.mean.shape[1]
            or self.parent_indices.dtype is not torch.int64
            or not bool(torch.isfinite(self.mean).all())
            or not bool(torch.isfinite(self.diagonal_factor).all())
            or not bool(torch.isfinite(self.lower_blocks).all())
            or not bool(torch.all(self.diagonal_factor > 0.0))
        ):
            raise ValueError("recognition state is not a finite block-banded law")
        length = self.mean.shape[1]
        if any(
            tensor.ndim >= 2 and tuple(tensor.shape[-2:]) == (length, length)
            for tensor in self.tensor_inventory()
        ):
            raise ValueError("recognition state materialized a dense position pair")

    def tensor_inventory(self) -> tuple[Tensor, ...]:
        return (
            self.mean,
            self.diagonal_factor,
            self.lower_blocks,
            self.parent_indices,
        )

    def rsample(
        self,
        epsilon: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Solve ``L.T x=epsilon`` by adjacent-block back substitution."""

        self.validate()
        if (
            type(epsilon) is not Tensor
            or epsilon.shape != self.mean.shape
            or epsilon.dtype != self.mean.dtype
            or epsilon.device != self.mean.device
        ):
            raise ValueError("epsilon must match the recognition mean exactly")
        if attention_mask is None:
            attention_mask = torch.ones(
                self.mean.shape[:2],
                dtype=torch.bool,
                device=self.mean.device,
            )
        if (
            type(attention_mask) is not Tensor
            or attention_mask.dtype is not torch.bool
            or attention_mask.device != self.mean.device
            or attention_mask.shape != self.mean.shape[:2]
            or bool(torch.any(attention_mask.sum(dim=1) <= 0))
            or any(
                bool(torch.any(row[int(row.sum().item()) :]))
                for row in attention_mask
            )
        ):
            raise ValueError("recognition sample mask must be a nonempty prefix")
        with torch.autocast(
            device_type=self.mean.device.type,
            enabled=False,
        ):
            solve_dtype = (
                torch.float32
                if self.mean.device.type == "cuda"
                else self.mean.dtype
            )
            centered_rows: list[Tensor] = []
            for row in range(self.mean.shape[0]):
                valid_length = int(attention_mask[row].sum().item())
                solved_reversed = [
                    epsilon[row, valid_length - 1].to(dtype=solve_dtype)
                    / self.diagonal_factor[row, valid_length - 1].to(
                        dtype=solve_dtype
                    )
                ]
                for position in range(valid_length - 2, -1, -1):
                    coupling = torch.matmul(
                        self.lower_blocks[row, position]
                        .to(dtype=solve_dtype)
                        .transpose(-1, -2),
                        solved_reversed[-1].unsqueeze(-1),
                    ).squeeze(-1)
                    solved_reversed.append(
                        (
                            epsilon[row, position].to(dtype=solve_dtype)
                            - coupling
                        )
                        / self.diagonal_factor[row, position].to(
                            dtype=solve_dtype
                        )
                    )
                active = torch.stack(tuple(reversed(solved_reversed)), dim=0)
                if valid_length < self.mean.shape[1]:
                    active = torch.cat(
                        (
                            active,
                            torch.zeros_like(
                                self.mean[row, valid_length:],
                                dtype=solve_dtype,
                            ),
                        ),
                        dim=0,
                    )
                centered_rows.append(active)
            centered = torch.stack(centered_rows, dim=0)
            return self.mean.to(dtype=solve_dtype) + centered

    def entropy(self, attention_mask: Tensor | None = None) -> Tensor:
        """Exact entropy of the represented joint Gaussian, one per batch row."""

        self.validate()
        if attention_mask is None:
            attention_mask = torch.ones(
                self.mean.shape[:2],
                dtype=torch.bool,
                device=self.mean.device,
            )
        if (
            type(attention_mask) is not Tensor
            or attention_mask.dtype is not torch.bool
            or attention_mask.device != self.mean.device
            or attention_mask.shape != self.mean.shape[:2]
            or bool(torch.any(attention_mask.sum(dim=1) <= 0))
            or any(
                bool(torch.any(row[int(row.sum().item()) :]))
                for row in attention_mask
            )
        ):
            raise ValueError("recognition entropy mask must be a nonempty prefix")
        dimensions = attention_mask.sum(dim=1) * self.mean.shape[2]
        with torch.autocast(
            device_type=self.mean.device.type,
            enabled=False,
        ):
            solve_dtype = (
                torch.float32
                if self.mean.device.type == "cuda"
                else self.mean.dtype
            )
            return (
                0.5
                * dimensions.to(dtype=solve_dtype)
                * (1.0 + _LOG_2PI)
                - torch.log(
                    self.diagonal_factor.to(dtype=solve_dtype)
                )
                .sum(dim=-1)
                .masked_fill(~attention_mask, 0.0)
                .sum(dim=1)
            )

    def solve_residual(self) -> Tensor:
        """Return a current mechanical residual for the block solve."""

        self.validate()
        epsilon = torch.ones_like(self.mean)
        centered = self.rsample(epsilon) - self.mean
        reconstructed: list[Tensor] = []
        for position in range(self.mean.shape[1]):
            value = self.diagonal_factor[:, position] * centered[:, position]
            if position + 1 < self.mean.shape[1]:
                value = value + torch.matmul(
                    self.lower_blocks[:, position].transpose(-1, -2),
                    centered[:, position + 1].unsqueeze(-1),
                ).squeeze(-1)
            reconstructed.append(value)
        return (
            torch.stack(tuple(reconstructed), dim=1) - epsilon
        ).abs().amax()

    def numerical_observation(self) -> NumericalObservation:
        """Project the live banded factor into measured scalar diagnostics."""

        diagonal = self.diagonal_factor.detach()
        inventory = tuple(tensor.detach() for tensor in self.tensor_inventory())
        nonfinite_count = sum(
            int((~torch.isfinite(tensor)).sum().item())
            for tensor in inventory
            if tensor.is_floating_point()
        )
        failed_pivots = int(
            ((~torch.isfinite(diagonal)) | (diagonal <= 0.0)).sum().item()
        )
        finite_diagonal = diagonal[torch.isfinite(diagonal) & (diagonal > 0.0)]
        if finite_diagonal.numel() == 0:
            raise ValueError("recognition state has no finite positive pivot")
        minimum = float(finite_diagonal.amin().item())
        maximum = float(finite_diagonal.amax().item())
        residual = float(self.solve_residual().detach().cpu().item())
        return NumericalObservation(
            minimum_cholesky_pivot=minimum,
            failed_pivots=failed_pivots,
            condition_estimate=maximum / minimum,
            solve_residual=residual,
            nonfinite_count=nonfinite_count,
        )


class WT103StructuredRecognition(nn.Module):
    """Target-conditioned block-tridiagonal smoothing recognition module."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        sequence_length: int,
        latent_width: int,
        source_lookback: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        _positive_int(vocabulary_size, "vocabulary_size")
        _positive_int(sequence_length, "sequence_length")
        _positive_int(latent_width, "latent_width")
        _positive_int(source_lookback, "source_lookback")
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.latent_width = latent_width
        self.source_lookback = source_lookback
        self.input_embedding = nn.Embedding(
            vocabulary_size, latent_width, device=device, dtype=dtype
        )
        self.observation_embedding = nn.Embedding(
            vocabulary_size, latent_width, device=device, dtype=dtype
        )
        self.mean_projection = nn.Linear(
            2 * latent_width, latent_width, device=device, dtype=dtype
        )
        self.raw_diagonal = nn.Parameter(
            torch.full(
                (sequence_length, latent_width),
                math.log(math.expm1(1.0 - 1.0e-4)),
                device=device,
                dtype=dtype,
            )
        )
        self.raw_lower = nn.Parameter(
            torch.zeros(
                sequence_length - 1,
                latent_width,
                latent_width,
                device=device,
                dtype=dtype,
            )
        )
        self.register_buffer(
            "parent_indices",
            banded_parent_indices(
                sequence_length, source_lookback, device=device
            ),
            persistent=True,
        )

    def forward(self, batch: object) -> BandedRecognitionState:
        inputs, targets, mask, _ = _batch_tensors(
            batch,
            sequence_length=self.sequence_length,
            vocabulary_size=self.vocabulary_size,
            device=self.raw_diagonal.device,
        )
        safe_targets = torch.where(mask, targets, inputs)
        features = torch.cat(
            (
                self.input_embedding(inputs),
                self.observation_embedding(safe_targets),
            ),
            dim=-1,
        )
        mean = self.mean_projection(features).to(dtype=self.raw_diagonal.dtype)
        diagonal = F.softplus(self.raw_diagonal) + 1.0e-4
        diagonal = diagonal.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        lower = 0.05 * torch.tanh(self.raw_lower)
        lower = lower.unsqueeze(0).expand(inputs.shape[0], -1, -1, -1)
        state = BandedRecognitionState(
            mean=mean,
            diagonal_factor=diagonal,
            lower_blocks=lower,
            parent_indices=self.parent_indices,
        )
        state.validate()
        return state


def _batch_tensors(
    batch: object,
    *,
    sequence_length: int,
    vocabulary_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    inputs = getattr(batch, "inputs", None)
    targets = getattr(batch, "targets", None)
    mask = getattr(batch, "attention_mask", None)
    counted_targets = getattr(batch, "counted_targets", None)
    if (
        type(inputs) is not Tensor
        or type(targets) is not Tensor
        or type(mask) is not Tensor
        or inputs.ndim != 2
        or inputs.shape != targets.shape
        or inputs.shape != mask.shape
        or inputs.shape[1] != sequence_length
        or inputs.dtype is not torch.int64
        or targets.dtype is not torch.int64
        or mask.dtype is not torch.bool
        or type(counted_targets) is not int
        or counted_targets <= 0
    ):
        raise ValueError("batch is not a typed causal [B,L] batch")
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)
    mask = mask.to(device=device)
    valid_targets = targets[mask]
    if (
        inputs.numel() == 0
        or bool(torch.any(inputs < 0))
        or bool(torch.any(inputs >= vocabulary_size))
        or bool(torch.any(valid_targets < 0))
        or bool(torch.any(valid_targets >= vocabulary_size))
        or int(mask.sum().item()) != counted_targets
        or bool(torch.any(targets[~mask] != -100))
        or any(
            bool(torch.any(mask[row, int(mask[row].sum().item()) :]))
            for row in range(mask.shape[0])
        )
    ):
        raise ValueError("batch token support or counted target total is invalid")
    return inputs, targets, mask, counted_targets


class _ChunkedCategoricalDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int,
        vocabulary_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.vocabulary_size = vocabulary_size
        self.projection = nn.Linear(
            hidden_width,
            vocabulary_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.live_observer: object | None = None
        self.live_phase: Literal["train", "evaluation"] = "train"
        self.live_event_prefix = "decoder"

    def _observe_logits(self, logits: Tensor, ordinal: int) -> None:
        observer = self.live_observer
        if observer is None:
            return
        observer.observe_tensor(
            logits,
            "decoder_chunk",
            ("token_or_particle_chunk", "vocabulary"),
            self.live_phase,
            f"{self.live_event_prefix}:{ordinal}",
        )

    def selected_log_probs(
        self,
        hidden: Tensor,
        targets: Tensor,
        mask: Tensor,
        *,
        token_chunk_size: int,
    ) -> Tensor:
        """Decode at most ``token_chunk_size`` rows and return ``[B,L]``."""

        _positive_int(token_chunk_size, "token_chunk_size")
        if (
            hidden.ndim != 3
            or targets.shape != hidden.shape[:2]
            or mask.shape != targets.shape
        ):
            raise ValueError("decoder selected-log-prob shapes disagree")
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        rows: list[Tensor] = []
        for ordinal, start in enumerate(
            range(0, flat_hidden.shape[0], token_chunk_size)
        ):
            end = min(start + token_chunk_size, flat_hidden.shape[0])
            logits = self.projection(flat_hidden[start:end])
            self._observe_logits(logits, ordinal)
            safe = torch.where(
                flat_mask[start:end],
                flat_targets[start:end],
                torch.zeros_like(flat_targets[start:end]),
            )
            selected = F.log_softmax(logits, dim=-1).gather(
                -1, safe.unsqueeze(-1)
            ).squeeze(-1)
            rows.append(torch.where(flat_mask[start:end], selected, 0.0))
        return torch.cat(tuple(rows)).reshape(targets.shape)

    def log_probs(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.hidden_width:
            raise ValueError("decoder hidden width changed")
        logits = self.projection(hidden)
        self._observe_logits(logits, 0)
        return F.log_softmax(logits, dim=-1)


class WT103LatentGenerativeModel(nn.Module):
    """Normalized band-source Gaussian latent model with chunked emissions."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        sequence_length: int,
        d_z: int,
        d_m: int,
        source_lookback: int,
        prior_variant: PriorVariant,
        decoder_chunk_size: int,
        particle_chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        for value, name in (
            (vocabulary_size, "vocabulary_size"),
            (sequence_length, "sequence_length"),
            (d_z, "d_z"),
            (d_m, "d_m"),
            (source_lookback, "source_lookback"),
            (decoder_chunk_size, "decoder_chunk_size"),
            (particle_chunk_size, "particle_chunk_size"),
        ):
            _positive_int(value, name)
        if prior_variant not in ("fixed", "parent_specific_pooled_prefix"):
            raise ValueError("latent prior must be fixed or parent-specific")
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.d_z = d_z
        self.d_m = d_m
        self.latent_width = d_z + d_m
        self.source_lookback = source_lookback
        self.prior_variant = prior_variant
        self.decoder_chunk_size = decoder_chunk_size
        self.particle_chunk_size = particle_chunk_size
        self._source_observation: SourceObservation | None = None
        self.token_embedding = nn.Embedding(
            vocabulary_size, self.latent_width, device=device, dtype=dtype
        )
        self.position_embedding = nn.Embedding(
            sequence_length, self.latent_width, device=device, dtype=dtype
        )
        self.model_input_projection = nn.Linear(
            self.latent_width,
            d_m,
            device=device,
            dtype=dtype,
        )
        self.state_input_projection = nn.Linear(
            self.latent_width,
            d_z,
            device=device,
            dtype=dtype,
        )
        self.state_model_projection = nn.Linear(
            d_m,
            d_z,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.initial_mean = nn.Parameter(
            torch.zeros(self.latent_width, device=device, dtype=dtype)
        )
        self.initial_log_scale = nn.Parameter(
            torch.zeros(self.latent_width, device=device, dtype=dtype)
        )
        self.transition_log_scale = nn.Parameter(
            torch.zeros(self.latent_width, device=device, dtype=dtype)
        )
        self.state_frame_generator = nn.Parameter(
            torch.zeros(sequence_length, d_z, d_z, device=device, dtype=dtype)
        )
        self.model_frame_generator = nn.Parameter(
            torch.zeros(sequence_length, d_m, d_m, device=device, dtype=dtype)
        )
        active_receivers = tuple(
            receiver
            for receiver in range(2, sequence_length)
            if min(source_lookback, receiver) > 1
        )
        if prior_variant == "fixed":
            self.state_source_fixed_logits = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
            self.model_source_fixed_logits = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
            self.state_source_query = None
            self.state_source_key = None
            self.model_source_query = None
            self.model_source_key = None
            self.state_source_free_keys = None
            self.state_source_free_biases = None
            self.model_source_free_keys = None
            self.model_source_free_biases = None
        else:
            self.state_source_fixed_logits = None
            self.model_source_fixed_logits = None
            self.state_source_query = nn.Linear(
                self.latent_width,
                self.latent_width,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.state_source_key = nn.Linear(
                self.d_z,
                self.latent_width,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.model_source_query = nn.Linear(
                self.latent_width,
                self.latent_width,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.model_source_key = nn.Linear(
                self.d_m,
                self.latent_width,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.state_source_free_keys = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            self.latent_width,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
            self.state_source_free_biases = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
            self.model_source_free_keys = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            self.latent_width,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
            self.model_source_free_biases = nn.ParameterDict(
                {
                    str(receiver): nn.Parameter(
                        torch.zeros(
                            min(source_lookback, receiver) - 1,
                            device=device,
                            dtype=dtype,
                        )
                    )
                    for receiver in active_receivers
                }
            )
        self.decoder = _ChunkedCategoricalDecoder(
            hidden_width=self.latent_width,
            vocabulary_size=vocabulary_size,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "parent_indices",
            banded_parent_indices(
                sequence_length, source_lookback, device=device
            ),
            persistent=True,
        )

    @property
    def device(self) -> torch.device:
        return self.initial_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.initial_mean.dtype

    def _source_row_logits(
        self,
        embedded: Tensor,
        latent_history: Tensor,
        *,
        receiver: int,
        state_row: bool,
    ) -> Tensor:
        valid_count = min(self.source_lookback, receiver)
        if (
            embedded.ndim != 3
            or embedded.shape[1] != receiver
            or latent_history.ndim != 3
            or latent_history.shape[:2] != embedded.shape[:2]
            or latent_history.shape[-1] != self.latent_width
        ):
            raise ValueError("source row history does not end at its receiver")
        if valid_count <= 0:
            return embedded.new_zeros((embedded.shape[0], 1))
        if valid_count == 1:
            return embedded.new_zeros((latent_history.shape[0], 1))
        if self.prior_variant == "fixed":
            parameter = (
                self.state_source_fixed_logits
                if state_row
                else self.model_source_fixed_logits
            )
            assert parameter is not None
            free = parameter[str(receiver)]
            anchored = torch.cat((free, free.new_zeros(1)))
            return anchored.unsqueeze(0).expand(
                latent_history.shape[0], -1
            )
        query_layer = (
            self.state_source_query
            if state_row
            else self.model_source_query
        )
        key_layer = (
            self.state_source_key
            if state_row
            else self.model_source_key
        )
        assert query_layer is not None and key_layer is not None
        free_keys = (
            self.state_source_free_keys
            if state_row
            else self.model_source_free_keys
        )
        free_biases = (
            self.state_source_free_biases
            if state_row
            else self.model_source_free_biases
        )
        assert free_keys is not None and free_biases is not None
        query = query_layer(embedded.mean(dim=1))
        bank_history = (
            latent_history[..., : self.d_z]
            if state_row
            else latent_history[..., self.d_z :]
        )
        key = key_layer(bank_history[:, -valid_count:])
        free_key = free_keys[str(receiver)]
        anchored_key = torch.cat(
            (
                free_key,
                free_key.new_zeros((1, self.latent_width)),
            ),
            dim=0,
        )
        free_bias = free_biases[str(receiver)]
        anchored_bias = torch.cat((free_bias, free_bias.new_zeros(1)))
        projected = key + anchored_key.unsqueeze(0)
        raw = torch.sum(
            projected * query.unsqueeze(1),
            dim=-1,
        ) / math.sqrt(self.latent_width) + anchored_bias.unsqueeze(0)
        return raw - raw[:, -1:]

    def source_log_probs(
        self,
        inputs: Tensor,
        latent_history: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            inputs.ndim != 2
            or inputs.shape[1] > self.sequence_length
            or latent_history.shape != (
                inputs.shape[0],
                inputs.shape[1],
                self.latent_width,
            )
        ):
            raise ValueError("source inputs/history must be aligned [B,L]")
        length = inputs.shape[1]
        state_rows = latent_history.new_full(
            (inputs.shape[0], length, self.source_lookback),
            -torch.inf,
        )
        model_rows = torch.full_like(state_rows, -torch.inf)
        state_rows[:, 0, -1] = 0.0
        model_rows[:, 0, -1] = 0.0
        for receiver in range(1, length):
            count = min(self.source_lookback, receiver)
            observed = causal_observation_history(
                inputs,
                receiver_position=receiver,
            )
            embedded = self.token_embedding(observed)
            state_rows[:, receiver, -count:] = self._source_row_logits(
                embedded,
                latent_history[:, :receiver],
                receiver=receiver,
                state_row=True,
            )
            model_rows[:, receiver, -count:] = self._source_row_logits(
                embedded,
                latent_history[:, :receiver],
                receiver=receiver,
                state_row=False,
            )
        return (
            F.log_softmax(state_rows, dim=-1),
            F.log_softmax(model_rows, dim=-1),
        )

    def next_source_log_probs(
        self,
        prefix_inputs: Tensor,
        earlier_latents: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return distinct state/model rows for the next latent receiver."""

        if (
            prefix_inputs.ndim != 2
            or prefix_inputs.shape[1] <= 0
            or prefix_inputs.shape[1] > self.sequence_length
            or earlier_latents.ndim != 3
            or earlier_latents.shape[1:] != (
                prefix_inputs.shape[1],
                self.latent_width,
            )
        ):
            raise ValueError("source prefix/history must be aligned")
        receiver = prefix_inputs.shape[1]
        observed = causal_observation_history(
            prefix_inputs,
            receiver_position=receiver,
        )
        embedded = self.token_embedding(observed)
        return (
            F.log_softmax(
                self._source_row_logits(
                    embedded,
                    earlier_latents,
                    receiver=receiver,
                    state_row=True,
                ),
                dim=-1,
            ),
            F.log_softmax(
                self._source_row_logits(
                    embedded,
                    earlier_latents,
                    receiver=receiver,
                    state_row=False,
                ),
                dim=-1,
            ),
        )

    def _transport_frame(
        self,
        *,
        receiver_position: int,
        source_positions: Tensor,
        parent: Tensor,
        state_bank: bool,
    ) -> Tensor:
        if (
            type(source_positions) is not Tensor
            or source_positions.dtype is not torch.int64
            or source_positions.device != parent.device
            or source_positions.shape != parent.shape[:-1][-1:]
            and source_positions.shape != parent.shape[:-1]
            or bool(torch.any(source_positions < 0))
            or bool(torch.any(source_positions >= receiver_position))
        ):
            raise ValueError("source frame positions do not match parents")
        frame_dtype = (
            torch.float32
            if self.device.type == "cuda"
            else self.state_frame_generator.dtype
        )
        with torch.autocast(device_type=self.device.type, enabled=False):
            generator = (
                self.state_frame_generator
                if state_bank
                else self.model_frame_generator
            )
            receiver_frame = torch.matrix_exp(
                generator[receiver_position].to(
                    dtype=frame_dtype
                )
            )
            source_frames = torch.matrix_exp(
                generator[source_positions].to(dtype=frame_dtype)
            )
            omega = torch.matmul(
                receiver_frame,
                torch.linalg.inv(source_frames),
            ).to(dtype=parent.dtype)
        return torch.einsum("...j,...ij->...i", parent, omega)

    def _transition_feature(
        self,
        input_ids: Tensor,
        receiver_position: int,
        target_ndim: int,
    ) -> Tensor:
        feature = self.token_embedding(input_ids)
        feature = feature + self.position_embedding.weight[receiver_position]
        while feature.ndim < target_ndim:
            feature = feature.unsqueeze(-2)
        return feature

    def model_transition_mean(
        self,
        *,
        input_ids: Tensor,
        receiver_position: int,
        source_positions: Tensor,
        model_parent: Tensor,
    ) -> Tensor:
        """K_m location shared by exact ELBO enumeration and SMC."""

        if (
            type(receiver_position) is not int
            or not 1 <= receiver_position < self.sequence_length
            or model_parent.shape[-1] != self.d_m
        ):
            raise ValueError("model transition contract changed")
        transported = self._transport_frame(
            receiver_position=receiver_position,
            source_positions=source_positions,
            parent=model_parent,
            state_bank=False,
        )
        feature = self._transition_feature(
            input_ids,
            receiver_position,
            transported.ndim,
        )
        return transported + self.model_input_projection(feature)

    def state_transition_mean(
        self,
        *,
        input_ids: Tensor,
        receiver_position: int,
        source_positions: Tensor,
        state_parent: Tensor,
        current_model: Tensor,
    ) -> Tensor:
        """K_z location conditioned on current m_t, shared by both paths."""

        if (
            type(receiver_position) is not int
            or not 1 <= receiver_position < self.sequence_length
            or state_parent.shape[:-1] != current_model.shape[:-1]
            or state_parent.shape[-1] != self.d_z
            or current_model.shape[-1] != self.d_m
        ):
            raise ValueError("state transition contract changed")
        transported = self._transport_frame(
            receiver_position=receiver_position,
            source_positions=source_positions,
            parent=state_parent,
            state_bank=True,
        )
        feature = self._transition_feature(
            input_ids,
            receiver_position,
            transported.ndim,
        )
        return (
            transported
            + self.state_model_projection(current_model)
            + self.state_input_projection(feature)
        )

    def transition_component_log_probs(
        self,
        *,
        input_ids: Tensor,
        receiver_position: int,
        state_source_positions: Tensor,
        model_source_positions: Tensor,
        current_state: Tensor,
        current_model: Tensor,
        state_parents: Tensor,
        model_parents: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Enumerate normalized K_z and K_m logs over their source banks."""

        model_mean = self.model_transition_mean(
            input_ids=input_ids,
            receiver_position=receiver_position,
            source_positions=model_source_positions,
            model_parent=model_parents,
        )
        model_scale = torch.exp(
            self.transition_log_scale[self.d_z :]
        )
        model_log = -(
            0.5
            * (
                (current_model.unsqueeze(1) - model_mean)
                / model_scale
            ).square()
            + torch.log(model_scale)
            + 0.5 * _LOG_2PI
        ).sum(dim=-1)
        expanded_model = current_model.unsqueeze(1).expand(
            -1, state_parents.shape[1], -1
        )
        state_mean = self.state_transition_mean(
            input_ids=input_ids,
            receiver_position=receiver_position,
            source_positions=state_source_positions,
            state_parent=state_parents,
            current_model=expanded_model,
        )
        state_scale = torch.exp(
            self.transition_log_scale[: self.d_z]
        )
        state_log = -(
            0.5
            * (
                (current_state.unsqueeze(1) - state_mean)
                / state_scale
            ).square()
            + torch.log(state_scale)
            + 0.5 * _LOG_2PI
        ).sum(dim=-1)
        return state_log, model_log

    def expected_log_emission_terms(
        self,
        latent: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, ...]:
        selected = self.decoder.selected_log_probs(
            latent,
            targets,
            mask,
            token_chunk_size=self.decoder_chunk_size,
        )
        return tuple(
            selected[:, position].sum()
            for position in range(selected.shape[1])
        )

    def particle_emission_log_probs(self, particles: Tensor) -> Tensor:
        return self.decoder.log_probs(particles)

    def source_observation(self) -> SourceObservation | None:
        """Return the latest actual q-source totals, if the objective used them."""

        return self._source_observation


class WT103NoLatentModel(nn.Module):
    """Causal pooled-prefix exact-autoregressive no-latent control."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        sequence_length: int,
        hidden_width: int,
        decoder_chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        for value, name in (
            (vocabulary_size, "vocabulary_size"),
            (sequence_length, "sequence_length"),
            (hidden_width, "hidden_width"),
            (decoder_chunk_size, "decoder_chunk_size"),
        ):
            _positive_int(value, name)
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.hidden_width = hidden_width
        self.decoder_chunk_size = decoder_chunk_size
        self.token_embedding = nn.Embedding(
            vocabulary_size, hidden_width, device=device, dtype=dtype
        )
        self.position_embedding = nn.Embedding(
            sequence_length, hidden_width, device=device, dtype=dtype
        )
        self.prefix_projection = nn.Linear(
            hidden_width, hidden_width, device=device, dtype=dtype
        )
        self.decoder = _ChunkedCategoricalDecoder(
            hidden_width=hidden_width,
            vocabulary_size=vocabulary_size,
            device=device,
            dtype=dtype,
        )

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def encode(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.sequence_length:
            raise ValueError("input IDs must be [B,L] within capacity")
        length = input_ids.shape[1]
        if length == 0:
            return torch.tanh(
                self.prefix_projection(self.position_embedding.weight[:1])
            ).unsqueeze(0)
        embedded = self.token_embedding(input_ids)
        cumulative = embedded.cumsum(dim=1)
        denominator = torch.arange(
            1, length + 1, device=input_ids.device, dtype=embedded.dtype
        ).view(1, length, 1)
        pooled = cumulative / denominator
        positions = self.position_embedding(
            torch.arange(length, device=input_ids.device)
        ).unsqueeze(0)
        return torch.tanh(self.prefix_projection(pooled + positions))

    def next_token_log_probs(self, prefix_ids: Tensor) -> Tensor:
        hidden = self.encode(prefix_ids)
        return self.decoder.log_probs(hidden[:, -1]).squeeze(0)

    def cross_entropy(self, batch: object) -> tuple[Tensor, int]:
        inputs, targets, mask, count = _batch_tensors(
            batch,
            sequence_length=self.sequence_length,
            vocabulary_size=self.vocabulary_size,
            device=self.device,
        )
        selected = self.decoder.selected_log_probs(
            self.encode(inputs),
            targets,
            mask,
            token_chunk_size=self.decoder_chunk_size,
        )
        return -selected.sum(), count


class WT103LatentArmContainer(nn.Module):
    """One inventory module retaining disjoint model/recognition namespaces."""

    def __init__(
        self,
        generative: WT103LatentGenerativeModel,
        recognition: WT103StructuredRecognition,
    ) -> None:
        super().__init__()
        self.generative = generative
        self.recognition = recognition


def _recognition_state(
    recognition: WT103StructuredRecognition,
    batch: object,
    snapshot: RecognitionSnapshot | None,
) -> BandedRecognitionState:
    if snapshot is None:
        return recognition(batch)
    snapshot.assert_intact()
    snapshot.assert_nonaliasing(recognition)
    parameters = {
        name: snapshot.tensor(name)
        for name, _ in recognition.named_parameters()
    }
    state = torch.func.functional_call(recognition, parameters, (batch,))
    if type(state) is not BandedRecognitionState:
        raise ValueError("functional recognition call changed its result type")
    return state


def _source_posterior_log_probs(child: Tensor, parents: Tensor) -> Tensor:
    if (
        child.ndim != 2
        or parents.ndim != 3
        or child.shape[0] != parents.shape[0]
        or child.shape[-1] != parents.shape[-1]
    ):
        raise ValueError("source posterior child/parent shapes disagree")
    logits = -(
        parents - child.unsqueeze(1)
    ).square().sum(dim=-1)
    return F.log_softmax(logits, dim=-1)


def _observe_source_posterior(
    *,
    model: WT103LatentGenerativeModel,
    latent: Tensor,
    mask: Tensor,
) -> SourceObservation:
    """Measure q-source diagnostics without adding them to an ablation loss."""

    if (
        latent.ndim != 3
        or mask.ndim != 2
        or latent.shape[:2] != mask.shape
        or latent.shape[-1] != model.latent_width
    ):
        raise ValueError("source diagnostic tensors are malformed")
    density_dtype = (
        torch.float32 if model.device.type == "cuda" else latent.dtype
    )
    entropy_sum = 0.0
    row_count = 0
    support_size_sum = 0.0
    with torch.no_grad():
        owned = latent.detach().to(dtype=density_dtype)
        for receiver in range(1, owned.shape[1]):
            indices = model.parent_indices[receiver]
            valid = indices >= 0
            parents = owned[:, indices[valid]]
            state_log = _source_posterior_log_probs(
                owned[:, receiver, : model.d_z],
                parents[..., : model.d_z],
            )
            model_log = _source_posterior_log_probs(
                owned[:, receiver, model.d_z :],
                parents[..., model.d_z :],
            )
            active = mask[:, receiver]
            active_count = int(active.sum().item())
            entropy_sum += float(
                (
                    -(state_log.exp() * state_log).sum(dim=-1)
                    -(model_log.exp() * model_log).sum(dim=-1)
                )[active]
                .sum()
                .cpu()
                .item()
            )
            row_count += 2 * active_count
            support_size_sum += float(
                2 * active_count * int(valid.sum().item())
            )
    return SourceObservation(
        entropy_sum=entropy_sum,
        source_row_count=row_count,
        support_size_sum=support_size_sum,
    )


def compute_wt103_forward_terms(
    *,
    model: WT103LatentGenerativeModel,
    recognition: WT103StructuredRecognition,
    objective_kind: Literal[
        "complete_elbo", "emission_only_ablation_non_elbo"
    ],
    phase: str,
    batch: object,
    snapshot: RecognitionSnapshot | None,
    recognition_state: BandedRecognitionState | None = None,
) -> ForwardTerms:
    """Evaluate a live latent-arm objective with no dense population matrix."""

    if phase not in ("recognition_adam_proposal", "model_adam_proposal"):
        raise ValueError("latent forward phase is not frozen")
    state = (
        _recognition_state(recognition, batch, snapshot)
        if recognition_state is None
        else recognition_state
    )
    state.validate()
    inputs, targets, mask, counted = _batch_tensors(
        batch,
        sequence_length=model.sequence_length,
        vocabulary_size=model.vocabulary_size,
        device=model.device,
    )
    # One reparameterized estimator; reverse-mode autograd is intentionally
    # retained for both the recognition and generative proposal phases.
    latent = state.rsample(torch.randn_like(state.mean), mask)
    emissions = model.expected_log_emission_terms(latent, targets, mask)
    if objective_kind == "emission_only_ablation_non_elbo":
        model._source_observation = _observe_source_posterior(
            model=model,
            latent=latent,
            mask=mask,
        )
        return ForwardTerms.emission_only(
            expected_log_emission=emissions,
            counted_targets=counted,
        )
    if objective_kind != "complete_elbo":
        raise ValueError("latent model received an unknown objective")
    state_source_log_probs, model_source_log_probs = (
        model.source_log_probs(inputs, latent)
    )
    density_dtype = (
        torch.float32 if model.device.type == "cuda" else latent.dtype
    )
    with torch.autocast(device_type=model.device.type, enabled=False):
        density_latent = latent.to(dtype=density_dtype)
        initial_mean = model.initial_mean.to(dtype=density_dtype)
        initial_scale = torch.exp(
            model.initial_log_scale.to(dtype=density_dtype)
        )
        initial_negative_log_density = (
            0.5
            * (
                (density_latent[:, 0] - initial_mean)
                / initial_scale
            ).square()
            + torch.log(initial_scale)
            + 0.5 * _LOG_2PI
        )
        state_source_log_probs = state_source_log_probs.to(
            dtype=density_dtype
        )
        model_source_log_probs = model_source_log_probs.to(
            dtype=density_dtype
        )
    continuous_recognition_entropy = state.entropy(mask).sum()
    initial_active = mask[:, 0].to(dtype=density_dtype)
    initial_state_cross_entropy = (
        initial_negative_log_density[:, : model.d_z].sum(dim=-1)
        * initial_active
    ).sum()
    initial_model_cross_entropy = (
        initial_negative_log_density[:, model.d_z :].sum(dim=-1)
        * initial_active
    ).sum()
    zero = latent.new_zeros(())
    state_transition_cross_entropy_rows: list[Tensor] = [zero]
    model_transition_cross_entropy_rows: list[Tensor] = [zero]
    state_source_cross_entropy_rows: list[Tensor] = [zero]
    model_source_cross_entropy_rows: list[Tensor] = [zero]
    state_source_kl_rows: list[Tensor] = [zero]
    model_source_kl_rows: list[Tensor] = [zero]
    source_entropy_rows: list[Tensor] = []
    source_row_count = 0
    source_support_size_sum = 0.0
    for receiver in range(1, latent.shape[1]):
        indices = model.parent_indices[receiver]
        valid = indices >= 0
        parent_rows = density_latent[:, indices[valid]]
        state_parents = parent_rows[..., : model.d_z]
        model_parents = parent_rows[..., model.d_z :]
        state_q_log = _source_posterior_log_probs(
            density_latent[:, receiver, : model.d_z],
            state_parents,
        )
        model_q_log = _source_posterior_log_probs(
            density_latent[:, receiver, model.d_z :],
            model_parents,
        )
        state_q = state_q_log.exp()
        model_q = model_q_log.exp()
        active = mask[:, receiver]
        active_density = active.to(dtype=density_dtype)
        active_count = int(active.sum().item())
        source_entropy_rows.append(
            (
                (
                    -(state_q * state_q_log).sum(dim=-1)
                    -(model_q * model_q_log).sum(dim=-1)
                )
                * active_density
            ).sum()
        )
        source_row_count += 2 * active_count
        source_support_size_sum += float(
            2 * active_count * int(valid.sum().item())
        )
        state_source_cross_entropy_rows.append(
            (
                torch.sum(
                    state_q
                    * -state_source_log_probs[:, receiver, valid],
                    dim=-1,
                )
                * active_density
            ).sum()
        )
        model_source_cross_entropy_rows.append(
            (
                torch.sum(
                    model_q
                    * -model_source_log_probs[:, receiver, valid],
                    dim=-1,
                )
                * active_density
            ).sum()
        )
        state_source_kl_rows.append(
            (
                torch.sum(
                    state_q
                    * (
                        state_q_log
                        - state_source_log_probs[:, receiver, valid]
                    ),
                    dim=-1,
                )
                * active_density
            ).sum()
        )
        model_source_kl_rows.append(
            (
                torch.sum(
                    model_q
                    * (
                        model_q_log
                        - model_source_log_probs[:, receiver, valid]
                    ),
                    dim=-1,
                )
                * active_density
            ).sum()
        )
        state_component_log, model_component_log = (
            model.transition_component_log_probs(
                input_ids=inputs[:, receiver],
                receiver_position=receiver,
                state_source_positions=indices[valid],
                model_source_positions=indices[valid],
                current_state=density_latent[
                    :, receiver, : model.d_z
                ],
                current_model=density_latent[
                    :, receiver, model.d_z :
                ],
                state_parents=state_parents,
                model_parents=model_parents,
            )
        )
        model_transition_cross_entropy_rows.append(
            (
                torch.sum(
                    model_q * -model_component_log.to(dtype=density_dtype),
                    dim=1,
                )
                * active_density
            ).sum()
        )
        state_transition_cross_entropy_rows.append(
            (
                torch.sum(
                    state_q * -state_component_log.to(dtype=density_dtype),
                    dim=1,
                )
                * active_density
            ).sum()
        )
    state_transition_cross_entropy = tuple(
        state_transition_cross_entropy_rows
    )
    model_transition_cross_entropy = tuple(
        model_transition_cross_entropy_rows
    )
    state_source_cross_entropy = tuple(
        state_source_cross_entropy_rows
    )
    model_source_cross_entropy = tuple(
        model_source_cross_entropy_rows
    )
    state_source_kl = tuple(state_source_kl_rows)
    model_source_kl = tuple(model_source_kl_rows)
    source_entropy = sum(source_entropy_rows, zero)
    joint_recognition_entropy_estimate = (
        continuous_recognition_entropy + source_entropy
    )
    source_entropy_sum = float(
        source_entropy.detach().cpu().item()
    )
    model._source_observation = SourceObservation(
        entropy_sum=source_entropy_sum,
        source_row_count=source_row_count,
        support_size_sum=source_support_size_sum,
    )
    return ForwardTerms.complete_elbo(
        expected_log_emission=emissions,
        initial_model_cross_entropy=initial_model_cross_entropy,
        initial_state_cross_entropy=initial_state_cross_entropy,
        model_source_cross_entropy=model_source_cross_entropy,
        model_transition_cross_entropy=model_transition_cross_entropy,
        state_source_cross_entropy=state_source_cross_entropy,
        state_transition_cross_entropy=state_transition_cross_entropy,
        model_source_kl=model_source_kl,
        state_source_kl=state_source_kl,
        continuous_recognition_entropy=continuous_recognition_entropy,
        conditional_source_entropy_estimate=source_entropy,
        joint_recognition_entropy_estimate=(
            joint_recognition_entropy_estimate
        ),
        estimator_error_bound=None,
        counted_targets=counted,
    )


def _compute_nonlatent_terms(model: nn.Module, batch: object) -> ForwardTerms:
    if hasattr(model, "cross_entropy"):
        value, count = model.cross_entropy(batch)  # type: ignore[attr-defined]
        return ForwardTerms.cross_entropy(value=value, counted_targets=count)
    inputs, targets, mask, count = _batch_tensors(
        batch,
        sequence_length=getattr(model, "positional_capacity"),
        vocabulary_size=getattr(model, "vocabulary_size"),
        device=next(model.parameters()).device,
    )
    hidden = model.encode(inputs)  # type: ignore[attr-defined]
    selected_sum = hidden.new_zeros(())
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    for start, end, logits in model.iter_decoder_logits(  # type: ignore[attr-defined]
        hidden,
        decoder_chunk_size=512,
    ):
        safe = torch.where(
            flat_mask[start:end],
            flat_targets[start:end],
            torch.zeros_like(flat_targets[start:end]),
        )
        selected = F.log_softmax(logits, dim=-1).gather(
            -1, safe.unsqueeze(-1)
        ).squeeze(-1)
        selected_sum = selected_sum + torch.where(
            flat_mask[start:end], selected, 0.0
        ).sum()
    return ForwardTerms.cross_entropy(
        value=-selected_sum,
        counted_targets=count,
    )


def build_adamw(
    parameters: tuple[nn.Parameter, ...],
    *,
    profile: AdamWProfile,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    if type(profile) is not AdamWProfile:
        raise ValueError("optimizer profile must be exact")
    profile.__post_init__()
    _finite_float(learning_rate, "learning_rate", positive=True)
    if (
        type(weight_decay) is not float
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
    ):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (
        type(parameters) is not tuple
        or not parameters
        or len({id(item) for item in parameters}) != len(parameters)
        or any(type(item) is not nn.Parameter for item in parameters)
    ):
        raise ValueError("optimizer parameters must be one unique exact tuple")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=profile.betas,
        eps=profile.epsilon,
        weight_decay=weight_decay,
        amsgrad=profile.amsgrad,
        foreach=profile.foreach,
        fused=profile.fused,
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.AdamW,
    *,
    profile: SchedulerProfile,
    planned_optimizer_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("scheduler requires the exact AdamW optimizer")
    if type(profile) is not SchedulerProfile:
        raise ValueError("scheduler profile must be exact")
    profile.__post_init__()
    _positive_int(planned_optimizer_steps, "planned_optimizer_steps")
    if planned_optimizer_steps <= profile.warmup_optimizer_steps:
        raise ValueError("scheduler horizon must extend beyond warmup")

    def multiplier(step: int) -> float:
        if step < profile.warmup_optimizer_steps:
            return (step + 1) / profile.warmup_optimizer_steps
        progress = min(
            1.0,
            (step - profile.warmup_optimizer_steps)
            / (planned_optimizer_steps - profile.warmup_optimizer_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return profile.minimum_lr_ratio + (
            1.0 - profile.minimum_lr_ratio
        ) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _tokenizer_digest(
    tokenizer: ProductionTokenizerSpec | SyntheticFixtureTokenizerSpec,
) -> str:
    if type(tokenizer) is ProductionTokenizerSpec:
        tokenizer.__post_init__()
        return tokenizer.spec_sha256
    if type(tokenizer) is SyntheticFixtureTokenizerSpec:
        tokenizer.__post_init__()
        return tokenizer.spec_sha256
    raise ValueError("runtime tokenizer identity is not typed")


@dataclass(frozen=True, slots=True)
class WT103RuntimeAuthority:
    """Frozen source/tokenizer/A0 identities admitted to runtime construction."""

    execution_scope: ExecutionScope
    source_record: FinalizedWikiText103SourceRecord | None
    tokenizer_spec: ProductionTokenizerSpec | SyntheticFixtureTokenizerSpec
    a0_factory_inputs: A0FactoryInputs
    authority_sha256: str

    @classmethod
    def from_production_source_lock(
        cls,
        source_lock: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WT103RuntimeAuthority":
        """Bind the production lock directly to the runtime constructor."""

        from .production import ProductionSourceLock

        if type(source_lock) is not ProductionSourceLock:
            raise ValueError("source_lock must be exact ProductionSourceLock")
        source_lock.__post_init__()
        if type(device) is not torch.device or type(dtype) is not torch.dtype:
            raise ValueError("authority device/dtype must be explicit")
        if (
            type(source_lock.a0_matching) is not ArmMatchingReport
            or source_lock.a0_matching.status is not GateStatus.PASS
        ):
            raise ValueError(
                "production runtime requires a closed PASS A0 matching report"
            )
        return cls.create(
            execution_scope="production_source_lock_verified",
            source_record=source_lock.finalized_source,
            tokenizer_spec=source_lock.tokenizer,
            a0_factory_inputs=A0FactoryInputs(
                architecture=source_lock.a0_architecture,
                formula=source_lock.a0_formula,
                flop_ledger=source_lock.a0_flop_ledger,
                matching=source_lock.a0_matching,
                device=device,
                dtype=dtype,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        execution_scope: ExecutionScope,
        source_record: FinalizedWikiText103SourceRecord | None,
        tokenizer_spec: ProductionTokenizerSpec | SyntheticFixtureTokenizerSpec,
        a0_factory_inputs: A0FactoryInputs,
    ) -> "WT103RuntimeAuthority":
        if execution_scope == "production_source_lock_verified":
            if (
                type(source_record) is not FinalizedWikiText103SourceRecord
                or type(tokenizer_spec) is not ProductionTokenizerSpec
            ):
                raise ValueError(
                    "production runtime requires finalized source and tokenizer"
                )
            source_record.__post_init__()
        elif execution_scope == "nonproduction_synthetic_smoke":
            if (
                source_record is not None
                or type(tokenizer_spec) is not SyntheticFixtureTokenizerSpec
            ):
                raise ValueError(
                    "synthetic runtime accepts only synthetic tokenizer authority"
                )
        else:
            raise ValueError("runtime execution scope is unknown")
        if type(a0_factory_inputs) is not A0FactoryInputs:
            raise ValueError("runtime authority requires exact A0 inputs")
        a0_factory_inputs.__post_init__()
        payload = {
            "execution_scope": execution_scope,
            "source_record_sha256": (
                None if source_record is None else source_record.record_sha256
            ),
            "tokenizer_spec_sha256": _tokenizer_digest(tokenizer_spec),
            "a0_architecture_sha256": (
                a0_factory_inputs.architecture.architecture_sha256
            ),
            "a0_formula_sha256": a0_factory_inputs.formula.formula_sha256,
            "a0_matching_sha256": a0_factory_inputs.matching.matching_sha256,
        }
        return cls(
            execution_scope=execution_scope,
            source_record=source_record,
            tokenizer_spec=tokenizer_spec,
            a0_factory_inputs=a0_factory_inputs,
            authority_sha256=owned_sha256(
                "vfe4.wt103.runtime-authority.v1", payload
            ),
        )

    def __post_init__(self) -> None:
        expected = WT103RuntimeAuthority.create(
            execution_scope=self.execution_scope,
            source_record=self.source_record,
            tokenizer_spec=self.tokenizer_spec,
            a0_factory_inputs=self.a0_factory_inputs,
        )
        if self.authority_sha256 != expected.authority_sha256:
            raise ValueError("runtime authority identity is stale")


class _LatentForward:
    def __init__(
        self,
        model: WT103LatentGenerativeModel,
        recognition: WT103StructuredRecognition,
        objective_kind: str,
    ) -> None:
        self.model = model
        self.recognition = recognition
        self.objective_kind = objective_kind
        self.last_state: BandedRecognitionState | None = None
        self.last_support_valid = False
        self.last_damping_applied = False
        self.last_projection_applied = False

    def __call__(
        self,
        phase: str,
        batch: object,
        snapshot: RecognitionSnapshot | None,
    ) -> ForwardTerms:
        with _bf16_autocast(self.model):
            state = _recognition_state(self.recognition, batch, snapshot)
        self.last_state = state
        self.last_support_valid = bool(
            torch.isfinite(state.mean).all()
            and torch.isfinite(state.diagonal_factor).all()
            and torch.all(state.diagonal_factor > 0.0)
            and torch.all(
                (state.parent_indices < self.model.sequence_length)
                & (state.parent_indices >= -1)
            )
        )
        self.last_damping_applied = True
        self.last_projection_applied = False
        with _bf16_autocast(self.model):
            terms = compute_wt103_forward_terms(
                model=self.model,
                recognition=self.recognition,
                objective_kind=self.objective_kind,  # type: ignore[arg-type]
                phase=phase,
                batch=batch,
                snapshot=snapshot,
                recognition_state=state,
            )
        return terms

    def support_valid(self) -> bool:
        return self.last_state is not None and self.last_support_valid

    def spd_valid(self) -> bool:
        if self.last_state is None:
            return False
        residual = self.last_state.solve_residual()
        diagonal = self.last_state.diagonal_factor
        condition_proxy = diagonal.amax() / diagonal.amin()
        return bool(
            torch.isfinite(residual)
            and residual <= 1.0e-5
            and torch.isfinite(condition_proxy)
            and condition_proxy <= 1.0e8
        )

    def damping_applied(self) -> bool:
        return self.last_damping_applied

    def projection_applied(self) -> bool:
        return self.last_projection_applied


@dataclass(frozen=True, slots=True)
class WT103InitializerSubstream:
    name: str
    initializer_class: InitializerClass
    target_shape: tuple[int, ...]
    target_dtype: str
    seed: int
    terminal_counter: int
    terminal_state_sha256: str
    initialized_value_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.initializer_class
            not in (
                "xavier_uniform_gain_1",
                "zero_linear_bias",
                "identity_layer_norm_affine",
                "zero_layer_norm_bias",
                "identity_frame_via_zero_generator",
                "identity_block_precision_diagonal_factor",
                "zero_block_precision_lower_factor",
                "zero_source_parameter",
                "standard_normal_latent_parameter",
                "training_rng_seed",
            )
            or type(self.target_shape) is not tuple
            or any(
                type(dimension) is not int or dimension < 0
                for dimension in self.target_shape
            )
            or type(self.target_dtype) is not str
            or not self.target_dtype
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.terminal_counter) is not int
            or self.terminal_counter < 0
            or type(self.terminal_state_sha256) is not str
            or len(self.terminal_state_sha256) != 64
            or type(self.initialized_value_sha256) is not str
            or len(self.initialized_value_sha256) != 64
        ):
            raise ValueError("initializer substream provenance is malformed")


@dataclass(frozen=True, slots=True)
class WT103InitializerProvenance:
    schema_version: Literal["wt103-named-initializer-substreams-v1"]
    root_seed: int
    arm_id: str
    substreams: tuple[WT103InitializerSubstream, ...]
    provenance_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_seed": self.root_seed,
            "arm_id": self.arm_id,
            "substreams": self.substreams,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-named-initializer-substreams-v1"
            or type(self.root_seed) is not int
            or not 0 <= self.root_seed < 2**63
            or type(self.arm_id) is not str
            or not self.arm_id
            or type(self.substreams) is not tuple
            or not self.substreams
            or tuple(row.name for row in self.substreams)
            != tuple(sorted(row.name for row in self.substreams))
            or any(
                type(row) is not WT103InitializerSubstream
                for row in self.substreams
            )
            or self.provenance_sha256
            != owned_sha256(
                "vfe4.wt103.named-initializer-substreams.v1",
                self.semantic_payload(),
            )
        ):
            raise ValueError("initializer provenance is inconsistent")


def _initialize_named_substreams(
    *,
    root_seed: int,
    arm_id: str,
    modules: Mapping[str, nn.Module],
) -> WT103InitializerProvenance:
    def raw_sha256(value: Tensor) -> str:
        owned = (
            value.detach()
            .contiguous()
            .view(torch.uint8)
            .cpu()
            .numpy()
            .tobytes(order="C")
        )
        return hashlib.sha256(owned).hexdigest()

    def initializer_class(
        *,
        parameter_name: str,
        owner: nn.Module,
        leaf_name: str,
    ) -> InitializerClass:
        if type(owner) is nn.Embedding and leaf_name == "weight":
            return "xavier_uniform_gain_1"
        if type(owner) is nn.Linear:
            if leaf_name == "weight":
                return "xavier_uniform_gain_1"
            if leaf_name == "bias":
                return "zero_linear_bias"
        if type(owner) is nn.LayerNorm:
            if leaf_name == "weight":
                return "identity_layer_norm_affine"
            if leaf_name == "bias":
                return "zero_layer_norm_bias"
        if "frame_generator" in parameter_name:
            return "identity_frame_via_zero_generator"
        if parameter_name == "raw_diagonal":
            return "identity_block_precision_diagonal_factor"
        if parameter_name == "raw_lower":
            return "zero_block_precision_lower_factor"
        if "_source_" in parameter_name:
            return "zero_source_parameter"
        if parameter_name in (
            "initial_mean",
            "initial_log_scale",
            "transition_log_scale",
        ):
            return "standard_normal_latent_parameter"
        raise ValueError(
            f"parameter {parameter_name!r} has no frozen initializer class"
        )

    def initialize(
        parameter: nn.Parameter,
        *,
        initializer: InitializerClass,
        generator: torch.Generator,
    ) -> int:
        with torch.no_grad():
            if initializer == "xavier_uniform_gain_1":
                nn.init.xavier_uniform_(
                    parameter,
                    gain=1.0,
                    generator=generator,
                )
                return parameter.numel()
            if initializer in (
                "zero_linear_bias",
                "zero_layer_norm_bias",
                "identity_frame_via_zero_generator",
                "zero_block_precision_lower_factor",
                "zero_source_parameter",
                "standard_normal_latent_parameter",
            ):
                nn.init.zeros_(parameter)
                return 0
            if initializer == "identity_layer_norm_affine":
                nn.init.ones_(parameter)
                return 0
            if initializer == "identity_block_precision_diagonal_factor":
                parameter.fill_(math.log(math.expm1(1.0 - 1.0e-4)))
                return 0
        raise ValueError(f"unknown initializer class {initializer!r}")

    rows: list[WT103InitializerSubstream] = []
    for module_name in sorted(modules):
        module = modules[module_name]
        owners = dict(module.named_modules())
        for parameter_name, parameter in sorted(
            module.named_parameters(),
            key=lambda item: item[0],
        ):
            owner_name, _, leaf_name = parameter_name.rpartition(".")
            owner = owners[owner_name]
            target_name = f"{module_name}.{parameter_name}"
            policy = initializer_class(
                parameter_name=parameter_name,
                owner=owner,
                leaf_name=leaf_name or parameter_name,
            )
            seed = int(
                owned_sha256(
                    "vfe4.wt103.initializer-substream-seed.v1",
                    {
                        "root_seed": root_seed,
                        "arm_id": arm_id,
                        "substream": target_name,
                        "initializer_class": policy,
                    },
                )[:15],
                16,
            )
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(seed)
            counter = initialize(
                parameter,
                initializer=policy,
                generator=generator,
            )
            state = generator.get_state().to(device="cpu").contiguous()
            rows.append(
                WT103InitializerSubstream(
                    name=target_name,
                    initializer_class=policy,
                    target_shape=tuple(parameter.shape),
                    target_dtype=str(parameter.dtype),
                    seed=seed,
                    terminal_counter=counter,
                    terminal_state_sha256=hashlib.sha256(
                        state.numpy().tobytes(order="C")
                    ).hexdigest(),
                    initialized_value_sha256=raw_sha256(parameter),
                )
            )
    training_seed = int(
        owned_sha256(
            "vfe4.wt103.initializer-substream-seed.v1",
            {
                "root_seed": root_seed,
                "arm_id": arm_id,
                "substream": "training_rng",
            },
        )[:15],
        16,
    )
    torch.manual_seed(training_seed)
    training_states = [torch.random.get_rng_state().cpu().contiguous()]
    if torch.cuda.is_available():
        training_states.extend(
            state.cpu().contiguous() for state in torch.cuda.get_rng_state_all()
        )
    training_state_bytes = b"".join(
        state.numpy().tobytes(order="C")
        for state in training_states
    )
    rows.append(
        WT103InitializerSubstream(
            name="training_rng",
            initializer_class="training_rng_seed",
            target_shape=(len(training_state_bytes),),
            target_dtype="torch.uint8",
            seed=training_seed,
            terminal_counter=0,
            terminal_state_sha256=hashlib.sha256(
                training_state_bytes
            ).hexdigest(),
            initialized_value_sha256=hashlib.sha256(
                training_state_bytes
            ).hexdigest(),
        )
    )
    rows.sort(key=lambda row: row.name)
    payload = {
        "schema_version": "wt103-named-initializer-substreams-v1",
        "root_seed": root_seed,
        "arm_id": arm_id,
        "substreams": tuple(rows),
    }
    return WT103InitializerProvenance(
        **payload,
        provenance_sha256=owned_sha256(
            "vfe4.wt103.named-initializer-substreams.v1",
            payload,
        ),
    )


@dataclass(slots=True)
class WT103ArmRuntimeBundle:
    """One build, engine runtime, scorer binding, and runtime identity."""

    built_arm: BuiltWT103Arm
    execution_runtime: ArmExecutionRuntime
    predictor: PriorPredictor
    estimator_spec: EstimatorSpec
    estimator_identity: EstimatorIdentity
    vocabulary: VocabularyIdentity
    authority_sha256: str
    initializer_provenance: WT103InitializerProvenance
    initializer_provenance_sha256: str
    source_recognition_law_sha256: str | None
    runtime_identity_sha256: str

    @property
    def scorer_kind(self) -> str:
        return self.built_arm.record.spec.scorer_kind

    def source_observation(self) -> SourceObservation | None:
        model = self.execution_runtime.model
        if not isinstance(model, WT103LatentGenerativeModel):
            return None
        return model.source_observation()

    def numerical_observation(self) -> NumericalObservation | None:
        compute = self.execution_runtime.compute_terms
        state = getattr(compute, "last_state", None)
        if state is None:
            return None
        if type(state) is not BandedRecognitionState:
            raise ValueError("latent forward retained an unknown recognition state")
        return state.numerical_observation()

    def make_predictor(
        self,
        particle_count: int | None = None,
    ) -> PriorPredictor:
        from .wt103_adapters import (
            ExactAutoregressivePriorPredictor,
            WT103ChunkedSmcPriorPredictor,
        )

        spec = self.built_arm.record.spec
        model = self.execution_runtime.model
        if spec.scorer_kind == "exact_autoregressive":
            if particle_count is not None:
                raise ValueError("exact predictor cannot accept particles")
            return ExactAutoregressivePriorPredictor(
                model=model,
                vocabulary=self.vocabulary,
                estimator_spec=self.estimator_spec,
                estimator_identity=self.estimator_identity,
                predictor_config_sha256=self.runtime_identity_sha256,
                data_safety_sha256=self.authority_sha256,
            )
        count = (
            self.estimator_spec.particle_count
            if particle_count is None
            else particle_count
        )
        if type(count) is not int or count <= 0:
            raise ValueError("weighted predictor requires positive particles")
        estimator_spec = EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=count,
            resampling="systematic_ess_half",
        )
        estimator_identity = EstimatorIdentity.from_spec(estimator_spec)
        return WT103ChunkedSmcPriorPredictor(
            model=model,  # type: ignore[arg-type]
            vocabulary=self.vocabulary,
            estimator_spec=estimator_spec,
            estimator_identity=estimator_identity,
            predictor_config_sha256=self.runtime_identity_sha256,
            data_safety_sha256=self.authority_sha256,
        )


@dataclass(frozen=True, slots=True)
class WT103RuntimeSetIdentity:
    schema_version: Literal["wt103-runtime-set-v1"]
    config_sha256: str
    authority_sha256: str
    factory_set_sha256: str
    ordered_runtime_sha256s: tuple[str, ...]
    runtime_set_sha256: str


@dataclass(slots=True)
class WT103RuntimeSet:
    arms: tuple[WT103ArmRuntimeBundle, ...]
    factory_set: WT103FactorySetIdentity
    identity: WT103RuntimeSetIdentity

    def for_arm(self, arm_id: str) -> WT103ArmRuntimeBundle:
        matches = tuple(
            bundle
            for bundle in self.arms
            if bundle.built_arm.record.spec.arm_id == arm_id
        )
        if len(matches) != 1:
            raise KeyError(arm_id)
        return matches[0]


@dataclass(frozen=True, slots=True)
class WT103PrimaryParameterRow:
    name: str
    shape: tuple[int, ...]
    numel: int
    optimizer_id: Literal["model_adamw", "recognition_adamw"]

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or type(self.shape) is not tuple
            or not self.shape
            or any(type(value) is not int or value <= 0 for value in self.shape)
            or type(self.numel) is not int
            or self.numel != math.prod(self.shape)
            or self.optimizer_id not in (
                "model_adamw",
                "recognition_adamw",
            )
        ):
            raise ValueError("PRIMARY parameter row is malformed")


@dataclass(frozen=True, slots=True)
class WT103PrimaryParameterInventory:
    schema_version: Literal["wt103-primary-parameter-inventory-v1"]
    arm_spec_sha256: str
    rows: tuple[WT103PrimaryParameterRow, ...]
    model_parameter_count: int
    recognition_parameter_count: int
    parameter_count: int
    optimizer_access_exact: bool
    inventory_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in tuple(self.__dataclass_fields__)[:-1]
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "wt103-primary-parameter-inventory-v1"
            or type(self.arm_spec_sha256) is not str
            or len(self.arm_spec_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.arm_spec_sha256
            )
            or type(self.rows) is not tuple
            or not self.rows
            or any(type(row) is not WT103PrimaryParameterRow for row in self.rows)
            or len({row.name for row in self.rows}) != len(self.rows)
            or {
                row.optimizer_id for row in self.rows
            }
            != {"model_adamw", "recognition_adamw"}
            or type(self.model_parameter_count) is not int
            or self.model_parameter_count
            != sum(
                row.numel
                for row in self.rows
                if row.optimizer_id == "model_adamw"
            )
            or type(self.recognition_parameter_count) is not int
            or self.recognition_parameter_count
            != sum(
                row.numel
                for row in self.rows
                if row.optimizer_id == "recognition_adamw"
            )
            or type(self.parameter_count) is not int
            or self.parameter_count
            != self.model_parameter_count + self.recognition_parameter_count
            or self.optimizer_access_exact is not True
            or type(self.inventory_sha256) is not str
            or len(self.inventory_sha256) != 64
            or self.inventory_sha256
            != owned_sha256(
                "vfe4.wt103.primary-parameter-inventory.v1",
                self.semantic_payload(),
            )
        ):
            raise ValueError("PRIMARY parameter inventory is inconsistent")


def _runtime_components(
    container: WT103LatentArmContainer,
) -> WT103ArmRuntimeComponents:
    named = tuple(container.named_parameters())
    recognition = tuple(
        name for name, _ in named if name.startswith("recognition.")
    )
    frame = tuple(name for name, _ in named if "frame_generator" in name)
    source = tuple(name for name, _ in named if ".source_" in name)
    latent = tuple(
        name
        for name, _ in named
        if name.startswith("generative.")
        and (
            "initial_" in name
            or "transition_" in name
            or "input_projection" in name
            or "state_model_projection" in name
        )
        and name not in frame
    )
    assigned = set((*recognition, *frame, *source, *latent))
    model = tuple(name for name, _ in named if name not in assigned)
    generative_names = tuple(
        name for name, _ in named if name.startswith("generative.")
    )
    return WT103ArmRuntimeComponents.create(
        model=container,
        model_parameter_names=model,
        latent_parameter_names=latent,
        source_parameter_names=source,
        frame_parameter_names=frame,
        recognition_parameter_names=recognition,
        optimizer_bindings=(
            OptimizerParameterBinding(
                optimizer_id="model_adamw",
                parameter_names=generative_names,
            ),
            OptimizerParameterBinding(
                optimizer_id="recognition_adamw",
                parameter_names=recognition,
            ),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def reconstruct_wt103_primary_parameters(
    config: TrainingConfig,
    *,
    device: torch.device = torch.device("meta"),
    dtype: torch.dtype = torch.float32,
) -> WT103PrimaryParameterInventory:
    """Reconstruct the live PRIMARY A5 named parameter inventory exactly."""

    if type(config) is not TrainingConfig:
        raise ValueError("config must be exact TrainingConfig")
    if type(device) is not torch.device or type(dtype) is not torch.dtype:
        raise ValueError("inventory device/dtype must be explicit")
    primary = tuple(
        spec
        for spec in config.endpoint_inventory.arms
        if spec.arm_id
        == "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1"
    )
    if len(primary) != 1:
        raise ValueError("endpoint inventory lacks the unique PRIMARY arm")
    profile = config.profile
    model = WT103LatentGenerativeModel(
        vocabulary_size=profile.vocabulary_size,
        sequence_length=profile.sequence_length,
        d_z=profile.d_z,
        d_m=profile.d_m,
        source_lookback=profile.source_lookback,
        prior_variant="parent_specific_pooled_prefix",
        decoder_chunk_size=profile.decoder_train_token_chunk,
        particle_chunk_size=profile.smc_particle_chunk,
        device=device,
        dtype=dtype,
    )
    recognition = WT103StructuredRecognition(
        vocabulary_size=profile.vocabulary_size,
        sequence_length=profile.sequence_length,
        latent_width=profile.combined_latent_block,
        source_lookback=profile.source_lookback,
        device=device,
        dtype=dtype,
    )
    container = WT103LatentArmContainer(model, recognition)
    components = _runtime_components(container)
    bound_names = tuple(
        name
        for binding in components.optimizer_bindings
        for name in binding.parameter_names
    )
    bindings = {
        name: binding.optimizer_id
        for binding in components.optimizer_bindings
        for name in binding.parameter_names
    }
    named = tuple(container.named_parameters())
    if (
        len(bound_names) != len(set(bound_names))
        or set(bindings) != {name for name, _ in named}
    ):
        raise ValueError("PRIMARY optimizer access does not cover parameters")
    rows = tuple(
        WT103PrimaryParameterRow(
            name=name,
            shape=tuple(parameter.shape),
            numel=parameter.numel(),
            optimizer_id=bindings[name],  # type: ignore[arg-type]
        )
        for name, parameter in named
    )
    model_count = sum(
        row.numel for row in rows if row.optimizer_id == "model_adamw"
    )
    recognition_count = sum(
        row.numel
        for row in rows
        if row.optimizer_id == "recognition_adamw"
    )
    payload = {
        "schema_version": "wt103-primary-parameter-inventory-v1",
        "arm_spec_sha256": primary[0].arm_spec_sha256,
        "rows": rows,
        "model_parameter_count": model_count,
        "recognition_parameter_count": recognition_count,
        "parameter_count": model_count + recognition_count,
        "optimizer_access_exact": True,
    }
    return WT103PrimaryParameterInventory(
        **payload,
        inventory_sha256=owned_sha256(
            "vfe4.wt103.primary-parameter-inventory.v1",
            payload,
        ),
    )


def _no_latent_components(
    model: WT103NoLatentModel,
) -> WT103ArmRuntimeComponents:
    names = tuple(name for name, _ in model.named_parameters())
    return WT103ArmRuntimeComponents.create(
        model=model,
        model_parameter_names=names,
        latent_parameter_names=(),
        source_parameter_names=(),
        frame_parameter_names=(),
        recognition_parameter_names=(),
        optimizer_bindings=(
            OptimizerParameterBinding("model_adamw", names),
        ),
        filler_parameter_names=(),
        dormant_parameter_names=(),
    )


def _make_vocabulary(
    config: TrainingConfig,
    authority: WT103RuntimeAuthority,
) -> VocabularyIdentity:
    return VocabularyIdentity(
        vocabulary_id="gpt2-wt103-v1",
        size=config.profile.vocabulary_size,
        tokenizer_spec_sha256=_tokenizer_digest(authority.tokenizer_spec),
    )


def _make_execution_runtime(
    *,
    spec: WT103ArmSpec,
    model: nn.Module,
    recognition: nn.Module | None,
    profile: object,
    learning_rate: float,
    weight_decay: float,
    planned_optimizer_steps: int,
) -> ArmExecutionRuntime:
    optimizer_profile = getattr(profile, "optimizer")
    scheduler_profile = getattr(profile, "scheduler")
    model_optimizer = build_adamw(
        tuple(model.parameters()),
        profile=optimizer_profile,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model_scheduler = build_warmup_cosine_scheduler(
        model_optimizer,
        profile=scheduler_profile,
        planned_optimizer_steps=planned_optimizer_steps,
    )
    if recognition is None:
        recognition_optimizer = None
        recognition_scheduler = None

        def compute(
            _phase: str,
            batch: object,
            _snapshot: RecognitionSnapshot | None,
        ) -> ForwardTerms:
            with _bf16_autocast(model):
                return _compute_nonlatent_terms(model, batch)

        def support() -> bool:
            return True

        def spd() -> bool:
            return True

        def damping() -> bool:
            return False

        def projection() -> bool:
            return False
    else:
        recognition_optimizer = build_adamw(
            tuple(recognition.parameters()),
            profile=optimizer_profile,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        recognition_scheduler = build_warmup_cosine_scheduler(
            recognition_optimizer,
            profile=scheduler_profile,
            planned_optimizer_steps=planned_optimizer_steps,
        )
        forward = _LatentForward(
            model,  # type: ignore[arg-type]
            recognition,  # type: ignore[arg-type]
            spec.training_objective,
        )
        compute = forward
        support = forward.support_valid
        spd = forward.spd_valid
        damping = forward.damping_applied
        projection = forward.projection_applied
    runtime = ArmExecutionRuntime(
        arm_spec=spec,
        model=model,
        recognition=recognition,
        model_optimizer=model_optimizer,
        recognition_optimizer=recognition_optimizer,
        model_scheduler=model_scheduler,
        recognition_scheduler=recognition_scheduler,
        grad_scaler=None,
        compute_terms=compute,
        support_validator=support,
        spd_validator=spd,
        damping_observer=damping,
        projection_observer=projection,
        state_participants=(),
        gradient_clip_norm=optimizer_profile.gradient_clip_max_norm,
    )
    runtime.validate()
    return runtime


def build_wt103_arm_runtime(
    config: TrainingConfig,
    *,
    arm_id: str,
    authority: WT103RuntimeAuthority,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    planned_optimizer_steps: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> WT103ArmRuntimeBundle:
    """Build one exact arm without materializing the other four runtimes."""

    if type(config) is not TrainingConfig:
        raise ValueError("config must be an exact TrainingConfig")
    if type(authority) is not WT103RuntimeAuthority:
        raise ValueError("authority must be exact")
    authority.__post_init__()
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("runtime seed must be a nonnegative exact int")
    torch.manual_seed(seed)
    selected_device = (
        torch.device(config.profile.precision.real_training_device)
        if device is None
        else device
    )
    selected_dtype = torch.float32 if dtype is None else dtype
    if type(selected_device) is not torch.device or type(selected_dtype) is not torch.dtype:
        raise ValueError("runtime device/dtype must be explicit")
    matches = tuple(
        spec
        for spec in config.endpoint_inventory.arms
        if spec.arm_id == arm_id
    )
    if len(matches) != 1:
        raise ValueError("arm_id must name one exact inventory row")
    spec = matches[0]
    profile = config.profile
    vocabulary = _make_vocabulary(config, authority)
    if spec.arm_id == "WT103-A0-AR-v1":
        built = build_wt103_a0(
            spec=spec,
            profile=profile,
            inputs=dataclasses.replace(
                authority.a0_factory_inputs,
                device=selected_device,
                dtype=selected_dtype,
            ),
            execution_scope=authority.execution_scope,
        )
        model = built.runtime.model
        recognition = None
    elif spec.arm_id == "WT103-A5-NOLATENT-v1":
        model = WT103NoLatentModel(
            vocabulary_size=profile.vocabulary_size,
            sequence_length=profile.sequence_length,
            hidden_width=profile.combined_latent_block,
            decoder_chunk_size=profile.decoder_train_token_chunk,
            device=selected_device,
            dtype=selected_dtype,
        )
        built = build_wt103_a5_nolatent(
            spec=spec,
            profile=profile,
            runtime=_no_latent_components(model),
            execution_scope=authority.execution_scope,
        )
        recognition = None
    else:
        prior_variant: PriorVariant = (
            "fixed"
            if spec.prior_variant == "fixed"
            else "parent_specific_pooled_prefix"
        )
        model = WT103LatentGenerativeModel(
            vocabulary_size=profile.vocabulary_size,
            sequence_length=profile.sequence_length,
            d_z=profile.d_z,
            d_m=profile.d_m,
            source_lookback=profile.source_lookback,
            prior_variant=prior_variant,
            decoder_chunk_size=profile.decoder_train_token_chunk,
            particle_chunk_size=profile.smc_particle_chunk,
            device=selected_device,
            dtype=selected_dtype,
        )
        recognition = WT103StructuredRecognition(
            vocabulary_size=profile.vocabulary_size,
            sequence_length=profile.sequence_length,
            latent_width=profile.combined_latent_block,
            source_lookback=profile.source_lookback,
            device=selected_device,
            dtype=selected_dtype,
        )
        container = WT103LatentArmContainer(model, recognition)
        components = _runtime_components(container)
        if spec.prior_variant == "fixed":
            built = build_wt103_a5_fixed(
                spec=spec,
                profile=profile,
                runtime=components,
                execution_scope=authority.execution_scope,
            )
        else:
            built = build_wt103_a5_parent_specific(
                spec=spec,
                profile=profile,
                runtime=components,
                execution_scope=authority.execution_scope,
            )
    initialization_modules = {"model": model}
    if recognition is not None:
        initialization_modules["recognition"] = recognition
    initializer_provenance = _initialize_named_substreams(
        root_seed=seed,
        arm_id=spec.arm_id,
        modules=initialization_modules,
    )
    source_recognition_law_sha256 = (
        SOURCE_RECOGNITION_LAW_SHA256 if recognition is not None else None
    )
    execution = _make_execution_runtime(
        spec=spec,
        model=model,
        recognition=recognition,
        profile=profile,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        planned_optimizer_steps=planned_optimizer_steps,
    )
    estimator_spec = EstimatorSpec.create(
        kind=(
            "deterministic_exact"
            if spec.scorer_kind == "exact_autoregressive"
            else "weighted_smc"
        ),
        particle_count=(
            None
            if spec.scorer_kind == "exact_autoregressive"
            else profile.statistics.validation_particle_count
        ),
        resampling=(
            "none"
            if spec.scorer_kind == "exact_autoregressive"
            else "systematic_ess_half"
        ),
    )
    estimator_identity = EstimatorIdentity.from_spec(estimator_spec)
    runtime_payload = {
        "config_sha256": config.config_sha256,
        "authority_sha256": authority.authority_sha256,
        "arm_build_sha256": built.record.build_sha256,
        "estimator_identity_sha256": estimator_identity.identity_sha256,
        "initializer_provenance_sha256": (
            initializer_provenance.provenance_sha256
        ),
        "source_recognition_law_sha256": source_recognition_law_sha256,
        "structured_factor_elbo_schema_sha256": (
            WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256
        ),
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "planned_optimizer_steps": planned_optimizer_steps,
    }
    runtime_sha = owned_sha256(
        "vfe4.wt103.arm-execution-runtime.v1", runtime_payload
    )
    bundle = WT103ArmRuntimeBundle(
        built_arm=built,
        execution_runtime=execution,
        predictor=None,  # type: ignore[arg-type]
        estimator_spec=estimator_spec,
        estimator_identity=estimator_identity,
        vocabulary=vocabulary,
        authority_sha256=authority.authority_sha256,
        initializer_provenance=initializer_provenance,
        initializer_provenance_sha256=(
            initializer_provenance.provenance_sha256
        ),
        source_recognition_law_sha256=source_recognition_law_sha256,
        runtime_identity_sha256=runtime_sha,
    )
    bundle.predictor = bundle.make_predictor()
    return bundle


def build_wt103_runtime_set(
    config: TrainingConfig,
    *,
    authority: WT103RuntimeAuthority,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    planned_optimizer_steps: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> WT103RuntimeSet:
    """Build the exact ordered five-arm runtime through the per-arm API."""

    bundles = tuple(
        build_wt103_arm_runtime(
            config,
            arm_id=spec.arm_id,
            authority=authority,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            planned_optimizer_steps=planned_optimizer_steps,
            device=device,
            dtype=dtype,
        )
        for spec in config.endpoint_inventory.arms
    )
    builds = tuple(bundle.built_arm for bundle in bundles)
    factory_set = WT103FactorySetIdentity.create(tuple(builds))
    identity_payload = {
        "schema_version": "wt103-runtime-set-v1",
        "config_sha256": config.config_sha256,
        "authority_sha256": authority.authority_sha256,
        "factory_set_sha256": factory_set.factory_set_sha256,
        "ordered_runtime_sha256s": tuple(
            item.runtime_identity_sha256 for item in bundles
        ),
    }
    identity = WT103RuntimeSetIdentity(
        **identity_payload,
        runtime_set_sha256=owned_sha256(
            "vfe4.wt103.runtime-set.v1", identity_payload
        ),
    )
    return WT103RuntimeSet(bundles, factory_set, identity)


def export_wt103_runtime_state(
    runtime_set: WT103RuntimeSet,
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for bundle in runtime_set.arms:
        rows[bundle.built_arm.record.spec.arm_id] = (
            export_wt103_arm_runtime_state(bundle)
        )
    return {
        "runtime_set_sha256": runtime_set.identity.runtime_set_sha256,
        "arms": rows,
    }


def export_wt103_arm_runtime_state(
    bundle: WT103ArmRuntimeBundle,
) -> dict[str, object]:
    """Copy one arm's complete scientific optimizer/resume state."""

    if type(bundle) is not WT103ArmRuntimeBundle:
        raise ValueError("arm runtime bundle must be exact")
    runtime = bundle.execution_runtime
    return {
        "runtime_identity_sha256": bundle.runtime_identity_sha256,
        "model": copy.deepcopy(runtime.model.state_dict()),
        "recognition": (
            None
            if runtime.recognition is None
            else copy.deepcopy(runtime.recognition.state_dict())
        ),
        "model_optimizer": copy.deepcopy(
            runtime.model_optimizer.state_dict()
        ),
        "recognition_optimizer": (
            None
            if runtime.recognition_optimizer is None
            else copy.deepcopy(runtime.recognition_optimizer.state_dict())
        ),
        "model_scheduler": copy.deepcopy(
            runtime.model_scheduler.state_dict()
        ),
        "recognition_scheduler": (
            None
            if runtime.recognition_scheduler is None
            else copy.deepcopy(runtime.recognition_scheduler.state_dict())
        ),
        "update_counter": runtime.update_counter,
    }


def _validate_module_state(
    module: nn.Module,
    state: object,
    *,
    name: str,
) -> Mapping[str, Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{name} state must be a mapping")
    expected = module.state_dict()
    if tuple(state) != tuple(expected):
        raise ValueError(f"{name} state inventory changed")
    for key, expected_tensor in expected.items():
        observed = state[key]
        if (
            type(observed) is not Tensor
            or observed.shape != expected_tensor.shape
            or observed.dtype != expected_tensor.dtype
        ):
            raise ValueError(f"{name}.{key} tensor contract changed")
    return state  # type: ignore[return-value]


def _validate_arm_runtime_state(
    bundle: WT103ArmRuntimeBundle,
    row: Mapping[str, object],
) -> None:
    arm_id = bundle.built_arm.record.spec.arm_id
    runtime = bundle.execution_runtime
    if (
        tuple(row)
        != (
            "runtime_identity_sha256",
            "model",
            "recognition",
            "model_optimizer",
            "recognition_optimizer",
            "model_scheduler",
            "recognition_scheduler",
            "update_counter",
        )
        or row["runtime_identity_sha256"] != bundle.runtime_identity_sha256
        or type(row["update_counter"]) is not int
    ):
        raise ValueError(f"resume row {arm_id} identity/schema changed")
    _validate_module_state(runtime.model, row["model"], name=f"{arm_id}.model")
    if runtime.recognition is None:
        if (
            row["recognition"] is not None
            or row["recognition_optimizer"] is not None
            or row["recognition_scheduler"] is not None
        ):
            raise ValueError(f"{arm_id} fabricated recognition state")
    else:
        _validate_module_state(
            runtime.recognition,
            row["recognition"],
            name=f"{arm_id}.recognition",
        )
    for target, key in (
        (runtime.model_optimizer, "model_optimizer"),
        (runtime.model_scheduler, "model_scheduler"),
        (runtime.recognition_optimizer, "recognition_optimizer"),
        (runtime.recognition_scheduler, "recognition_scheduler"),
    ):
        if target is None:
            continue
        probe = copy.deepcopy(target)
        probe.load_state_dict(copy.deepcopy(row[key]))


def _load_arm_runtime_state(
    bundle: WT103ArmRuntimeBundle,
    row: Mapping[str, object],
) -> None:
    runtime = bundle.execution_runtime
    runtime.model.load_state_dict(row["model"], strict=True)
    if runtime.recognition is not None:
        runtime.recognition.load_state_dict(row["recognition"], strict=True)
    runtime.model_optimizer.load_state_dict(row["model_optimizer"])
    runtime.model_scheduler.load_state_dict(row["model_scheduler"])
    if runtime.recognition_optimizer is not None:
        runtime.recognition_optimizer.load_state_dict(
            row["recognition_optimizer"]
        )
        runtime.recognition_scheduler.load_state_dict(
            row["recognition_scheduler"]
        )
    runtime.update_counter = row["update_counter"]
    bundle.predictor = bundle.make_predictor()


def rebuild_wt103_arm_runtime(
    config: TrainingConfig,
    *,
    arm_id: str,
    authority: WT103RuntimeAuthority,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    planned_optimizer_steps: int,
    scientific_state: Mapping[str, object],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> WT103ArmRuntimeBundle:
    """Reconstruct one arm and fail closed before loading its state."""

    bundle = build_wt103_arm_runtime(
        config,
        arm_id=arm_id,
        authority=authority,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        planned_optimizer_steps=planned_optimizer_steps,
        device=device,
        dtype=dtype,
    )
    if not isinstance(scientific_state, Mapping):
        raise ValueError("arm resume state must be a mapping")
    _validate_arm_runtime_state(bundle, scientific_state)
    _load_arm_runtime_state(bundle, scientific_state)
    return bundle


def rebuild_wt103_runtime_set(
    config: TrainingConfig,
    *,
    authority: WT103RuntimeAuthority,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    planned_optimizer_steps: int,
    scientific_state: Mapping[str, object],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> WT103RuntimeSet:
    """Reconstruct and fail closed before loading any checkpoint state."""

    rebuilt = build_wt103_runtime_set(
        config,
        authority=authority,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        planned_optimizer_steps=planned_optimizer_steps,
        device=device,
        dtype=dtype,
    )
    if (
        not isinstance(scientific_state, Mapping)
        or tuple(scientific_state)
        != ("runtime_set_sha256", "arms")
        or scientific_state["runtime_set_sha256"]
        != rebuilt.identity.runtime_set_sha256
        or not isinstance(scientific_state["arms"], Mapping)
    ):
        raise ValueError("resume state does not bind the reconstructed set")
    rows = scientific_state["arms"]
    expected_ids = tuple(
        bundle.built_arm.record.spec.arm_id for bundle in rebuilt.arms
    )
    if tuple(rows) != expected_ids:
        raise ValueError("resume arm inventory/order changed")
    validated: list[tuple[WT103ArmRuntimeBundle, Mapping[str, object]]] = []
    for bundle in rebuilt.arms:
        arm_id = bundle.built_arm.record.spec.arm_id
        row = rows[arm_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"resume row {arm_id} is not a mapping")
        runtime = bundle.execution_runtime
        if (
            tuple(row)
            != (
                "runtime_identity_sha256",
                "model",
                "recognition",
                "model_optimizer",
                "recognition_optimizer",
                "model_scheduler",
                "recognition_scheduler",
                "update_counter",
            )
            or row["runtime_identity_sha256"]
            != bundle.runtime_identity_sha256
            or type(row["update_counter"]) is not int
        ):
            raise ValueError(f"resume row {arm_id} identity/schema changed")
        _validate_module_state(runtime.model, row["model"], name=f"{arm_id}.model")
        if runtime.recognition is None:
            if (
                row["recognition"] is not None
                or row["recognition_optimizer"] is not None
                or row["recognition_scheduler"] is not None
            ):
                raise ValueError(f"{arm_id} fabricated recognition state")
        else:
            _validate_module_state(
                runtime.recognition,
                row["recognition"],
                name=f"{arm_id}.recognition",
            )
        # Load optimizer/scheduler states into isolated copies first.  No live
        # reconstructed runtime mutates until every arm has validated.
        for target, key in (
            (runtime.model_optimizer, "model_optimizer"),
            (runtime.model_scheduler, "model_scheduler"),
            (runtime.recognition_optimizer, "recognition_optimizer"),
            (runtime.recognition_scheduler, "recognition_scheduler"),
        ):
            if target is None:
                continue
            probe = copy.deepcopy(target)
            probe.load_state_dict(copy.deepcopy(row[key]))
        validated.append((bundle, row))
    for bundle, row in validated:
        runtime = bundle.execution_runtime
        runtime.model.load_state_dict(row["model"], strict=True)
        if runtime.recognition is not None:
            runtime.recognition.load_state_dict(row["recognition"], strict=True)
        runtime.model_optimizer.load_state_dict(row["model_optimizer"])
        runtime.model_scheduler.load_state_dict(row["model_scheduler"])
        if runtime.recognition_optimizer is not None:
            runtime.recognition_optimizer.load_state_dict(
                row["recognition_optimizer"]
            )
            runtime.recognition_scheduler.load_state_dict(
                row["recognition_scheduler"]
            )
        runtime.update_counter = row["update_counter"]
        bundle.predictor = bundle.make_predictor()
    return rebuilt


@runtime_checkable
class WT103LiveProbeObserver(Protocol):
    def mark_path_event(self, name: str) -> None: ...

    def observe_tensor(
        self,
        tensor: Tensor,
        storage_class: str,
        logical_axes: tuple[str, ...],
        phase: str,
        event_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WT103LiveProbeRequest:
    bundle: WT103ArmRuntimeBundle
    batch: object
    stream: EstimatorStream
    prefix_length: int


@dataclass(frozen=True, slots=True)
class WT103LiveProbeResult:
    arm_id: str
    step: StepResult
    prediction_sha256: str
    checkpoint_duplicate_bytes: int
    serializer_inventory_complete: bool
    serializer_unique_tensor_count: int
    serializer_inventory_sha256: str


def _state_tensors(value: object):
    if type(value) is Tensor:
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _state_tensors(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from _state_tensors(item)


def _named_state_tensors(
    value: object,
    *,
    prefix: str,
):
    if type(value) is Tensor:
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _named_state_tensors(
                item,
                prefix=f"{prefix}.{key}",
            )
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            yield from _named_state_tensors(
                item,
                prefix=f"{prefix}.{index}",
            )


@dataclass(frozen=True, slots=True)
class _TransferredProbeBatch:
    inputs: Tensor
    targets: Tensor
    attention_mask: Tensor
    counted_targets: int


def _run_observed_event(
    observer: WT103LiveProbeObserver,
    name: str,
    operation,
):
    path_event = getattr(observer, "path_event", None)
    if callable(path_event):
        with path_event(name):
            return operation()
    observer.mark_path_event(name)
    return operation()


def execute_live_probe(
    request: WT103LiveProbeRequest,
    observer: WT103LiveProbeObserver,
) -> WT103LiveProbeResult:
    """Execute the exact train, scorer, and serializer path for one arm."""

    if type(request) is not WT103LiveProbeRequest:
        raise ValueError("live probe request must be exact")
    if not isinstance(observer, WT103LiveProbeObserver):
        raise ValueError("live probe observer lacks the exact callback protocol")
    bundle = request.bundle
    runtime = bundle.execution_runtime
    spec = bundle.built_arm.record.spec
    arm_id = spec.arm_id

    def transfer_batch() -> _TransferredProbeBatch:
        device = next(runtime.model.parameters()).device
        transferred = _TransferredProbeBatch(
            inputs=getattr(request.batch, "inputs").to(device=device),
            targets=getattr(request.batch, "targets").to(device=device),
            attention_mask=getattr(
                request.batch, "attention_mask"
            ).to(device=device),
            counted_targets=getattr(request.batch, "counted_targets"),
        )
        for name in ("inputs", "targets", "attention_mask"):
            observer.observe_tensor(
                getattr(transferred, name),
                "token_ids_or_mask",
                ("batch", "position"),
                "train",
                f"{arm_id}:{name}",
            )
        return transferred

    batch = _run_observed_event(observer, "data_transfer", transfer_batch)
    if type(batch) is not _TransferredProbeBatch:
        raise ValueError("data-transfer instrumentation changed the batch")

    # Bind instrumentation to the real decoder allocation sites.  The A0
    # attention callback is attached only for the full training window: scorer
    # replay uses shorter causal prefixes and is not the Flash-shape witness.
    decoder = getattr(runtime.model, "decoder", None)
    if hasattr(decoder, "live_observer"):
        decoder.live_observer = observer
        decoder.live_phase = "train"
        decoder.live_event_prefix = f"{arm_id}:train_decoder"
    if hasattr(runtime.model, "live_observer"):
        runtime.model.live_observer = observer
        runtime.model.live_phase = "train"
        runtime.model.live_event_prefix = f"{arm_id}:train_decoder"
    block = getattr(runtime.model, "block", None)
    if hasattr(block, "live_observer"):
        block.live_observer = observer

    state_observed = False

    def engine_event_runner(name: str, operation):
        nonlocal state_observed

        def observed_operation():
            nonlocal state_observed
            result = operation()
            if (
                spec.latent_enabled
                and name == "forward"
                and not state_observed
            ):
                forward = runtime.compute_terms
                state = getattr(forward, "last_state", None)
                if type(state) is BandedRecognitionState:
                    observer.observe_tensor(
                        state.mean,
                        "latent_block",
                        ("batch", "position", "latent_block"),
                        "train",
                        f"{arm_id}:recognition_mean",
                    )
                    observer.observe_tensor(
                        state.diagonal_factor,
                        "latent_block",
                        ("batch", "position", "latent_block"),
                        "train",
                        f"{arm_id}:recognition_diagonal",
                    )
                    observer.observe_tensor(
                        state.lower_blocks,
                        "lower_adjacent_block",
                        (
                            "batch",
                            "position_minus_one",
                            "latent_block_row",
                            "latent_block_column",
                        ),
                        "train",
                        f"{arm_id}:recognition_lower",
                    )
                    band = state.parent_indices.unsqueeze(0).repeat(
                        state.mean.shape[0], 1, 1
                    )
                    observer.observe_tensor(
                        band,
                        "banded_source",
                        ("batch", "receiver", "source_slot"),
                        "train",
                        f"{arm_id}:source_band",
                    )
                    state_observed = True
            return result

        return _run_observed_event(observer, name, observed_operation)

    previous_runner = runtime.execution_event_runner
    runtime.execution_event_runner = engine_event_runner
    try:
        step = train_step(runtime, batch=batch)
    finally:
        runtime.execution_event_runner = previous_runner
        if hasattr(block, "live_observer"):
            block.live_observer = None

    inputs = batch.inputs
    if (
        type(request.prefix_length) is not int
        or not 1 <= request.prefix_length <= inputs.shape[1]
    ):
        raise ValueError("live probe prefix length is outside the batch")
    prefix = CausalPrefix.create(
        receiver_t=request.prefix_length + 1,
        vocabulary=bundle.vocabulary,
        token_ids=inputs[0, : request.prefix_length]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
    )
    if hasattr(decoder, "live_phase"):
        decoder.live_phase = "evaluation"
        decoder.live_event_prefix = f"{arm_id}:eval_decoder"
    if hasattr(runtime.model, "live_phase"):
        runtime.model.live_phase = "evaluation"
        runtime.model.live_event_prefix = f"{arm_id}:eval_decoder"
    predictor = bundle.make_predictor()
    scorer_event = (
        "weighted_smc_scorer"
        if spec.latent_enabled
        else "exact_autoregressive_scorer"
    )
    prediction = _run_observed_event(
        observer,
        scorer_event,
        lambda: predictor.next_token_log_probs(
            prefix, request.stream, None
        ),
    )

    metric_record = _run_observed_event(
        observer,
        "metric_failure_write",
        lambda: {
            "arm_id": step.arm_id,
            "accepted": step.accepted,
            "failure_kind": step.failure_kind,
            "prediction_sha256": prediction.prediction_sha256,
        },
    )
    if not isinstance(metric_record, Mapping):
        raise ValueError("metric instrumentation changed the record")

    def serialize_checkpoint():
        scientific = export_wt103_arm_runtime_state(bundle)
        inventory_sources: dict[str, object] = {
            "model": scientific["model"],
            "recognition": scientific["recognition"],
            "model_optimizer": scientific["model_optimizer"],
            "recognition_optimizer": scientific["recognition_optimizer"],
            "model_scheduler": scientific["model_scheduler"],
            "recognition_scheduler": scientific["recognition_scheduler"],
            "rng_cpu": torch.random.get_rng_state().clone(),
        }
        if next(runtime.model.parameters()).device.type == "cuda":
            inventory_sources["rng_cuda"] = torch.cuda.get_rng_state(
                next(runtime.model.parameters()).device
            ).clone()
        seen: set[tuple[str, int, int]] = set()
        inventory_rows: list[dict[str, object]] = []
        duplicate_bytes = 0
        for path, tensor in _named_state_tensors(
            inventory_sources,
            prefix="checkpoint",
        ):
            observed = tensor.detach()
            if observed.ndim == 0:
                observed = observed.reshape(1)
            storage = observed.untyped_storage()
            storage_key = (
                str(observed.device),
                int(storage.data_ptr()),
                int(storage.nbytes()),
            )
            if storage_key in seen:
                continue
            seen.add(storage_key)
            if ".recognition_optimizer" in path or ".model_optimizer" in path:
                storage_class = "checkpoint_optimizer_state"
            elif ".recognition_scheduler" in path or ".model_scheduler" in path:
                storage_class = "checkpoint_scheduler_state"
            elif ".recognition" in path:
                storage_class = "checkpoint_recognition_parameter"
            elif ".model" in path:
                storage_class = "checkpoint_model_parameter"
            elif ".rng_" in path:
                storage_class = "checkpoint_rng_state"
            else:
                raise ValueError(f"unclassified checkpoint tensor: {path}")
            observer.observe_tensor(
                observed,
                storage_class,
                tuple(f"axis_{index}" for index in range(observed.ndim)),
                "checkpoint",
                f"{step.arm_id}:{path}",
            )
            raw = (
                observed.to(device="cpu")
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            nbytes = observed.numel() * observed.element_size()
            duplicate_bytes += nbytes
            inventory_rows.append(
                {
                    "path": path,
                    "shape": tuple(observed.shape),
                    "dtype": str(observed.dtype),
                    "nbytes": nbytes,
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        metadata = {
            "runtime_identity_sha256": bundle.runtime_identity_sha256,
            "update_counter": runtime.update_counter,
            "estimator_stream_sha256": request.stream.stream_sha256,
            "metric_record": dict(metric_record),
            "tensor_inventory": tuple(inventory_rows),
        }
        return (
            duplicate_bytes,
            len(seen),
            owned_sha256(
                "vfe4.wt103.serializer-inventory.v1",
                metadata,
            ),
        )

    duplicate, unique_count, inventory_sha256 = _run_observed_event(
        observer,
        "checkpoint_serialization",
        serialize_checkpoint,
    )
    record_checkpoint = getattr(observer, "record_checkpoint_duplicate_bytes", None)
    if callable(record_checkpoint):
        record_checkpoint(duplicate)

    if hasattr(decoder, "live_observer"):
        decoder.live_observer = None
    if hasattr(runtime.model, "live_observer"):
        runtime.model.live_observer = None
    return WT103LiveProbeResult(
        arm_id=step.arm_id,
        step=step,
        prediction_sha256=prediction.prediction_sha256,
        checkpoint_duplicate_bytes=duplicate,
        serializer_inventory_complete=(unique_count > 0),
        serializer_unique_tensor_count=unique_count,
        serializer_inventory_sha256=inventory_sha256,
    )


__all__ = [
    "BandedRecognitionState",
    "SOURCE_RECOGNITION_LAW_SHA256",
    "WT103ArmRuntimeBundle",
    "WT103InitializerProvenance",
    "WT103InitializerSubstream",
    "WT103LatentGenerativeModel",
    "WT103LiveProbeObserver",
    "WT103LiveProbeRequest",
    "WT103LiveProbeResult",
    "WT103NoLatentModel",
    "WT103PrimaryParameterInventory",
    "WT103PrimaryParameterRow",
    "WT103RuntimeAuthority",
    "WT103RuntimeSet",
    "WT103RuntimeSetIdentity",
    "WT103_STRUCTURED_FACTOR_ELBO_SCHEMA_SHA256",
    "WT103StructuredRecognition",
    "banded_parent_indices",
    "build_adamw",
    "build_warmup_cosine_scheduler",
    "build_wt103_arm_runtime",
    "build_wt103_runtime_set",
    "causal_observation_history",
    "compute_wt103_forward_terms",
    "execute_live_probe",
    "export_wt103_arm_runtime_state",
    "export_wt103_runtime_state",
    "rebuild_wt103_arm_runtime",
    "rebuild_wt103_runtime_set",
    "reconstruct_wt103_primary_parameters",
]
