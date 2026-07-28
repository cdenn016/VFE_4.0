"""Trainable parameter ownership for H6 language-recognition laws."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vfe4.predictive.identities import canonical_model_state_sha256
from vfe4.types.h6 import VocabularyIdentity

from .h6_prediction_v3 import (
    LanguageRecognitionTrajectory,
    RecognitionPriorFeatureProvider,
    SourceBankName,
    SourceRecognitionParameters,
    build_language_recognition_trajectory,
    build_receiver_contexts,
)
from .language import (
    FactorizedLanguageRecognition,
    RecognitionConditioning,
    StructuredLanguageRecognition,
)


RecognitionStoreFamily = Literal["structured", "factorized"]
RecognitionStoreConditioning = Literal["filtering", "smoothing"]
LanguageRecognitionLaw = (
    StructuredLanguageRecognition | FactorizedLanguageRecognition
)


def _deterministic_matrix(rows: int, columns: int, *, scale: float) -> Tensor:
    """Return a small nonzero initialization without consuming an RNG stream."""

    values = torch.arange(rows * columns, dtype=torch.float64)
    values = values.reshape(rows, columns)
    centered = values - 0.5 * max(0, rows * columns - 1)
    return scale * centered / max(1, rows * columns)


def _deterministic_nonzero_vector(length: int, *, scale: float) -> Tensor:
    """Return a nonzero deterministic vector, including at width one."""

    values = torch.arange(1, length + 1, dtype=torch.float64)
    return scale * values / length


class LanguageRecognitionParameterStore(nn.Module):
    """Own every trainable parameter for legacy laws and v3 trajectories."""

    def __init__(
        self,
        *,
        vocabulary: VocabularyIdentity,
        horizon: int,
        latent_width: int,
        recognition_width: int,
        channel_count: Literal[1, 2],
        family: RecognitionStoreFamily,
        conditioning_mode: RecognitionStoreConditioning,
        trainable_source_banks: tuple[SourceBankName, ...] = (),
    ) -> None:
        super().__init__()
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        vocabulary.__post_init__()
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if type(latent_width) is not int or latent_width <= 0:
            raise ValueError("latent_width must be a positive integer")
        if type(recognition_width) is not int or recognition_width <= 0:
            raise ValueError("recognition_width must be a positive integer")
        if channel_count not in (1, 2):
            raise ValueError("channel_count must be exactly one or two")
        if family not in ("structured", "factorized"):
            raise ValueError("family must be structured or factorized")
        if conditioning_mode not in ("filtering", "smoothing"):
            raise ValueError(
                "conditioning_mode must be filtering or smoothing"
            )
        if (
            type(trainable_source_banks) is not tuple
            or any(
                bank not in ("state", "model")
                for bank in trainable_source_banks
            )
            or len(set(trainable_source_banks))
            != len(trainable_source_banks)
            or trainable_source_banks
            != tuple(
                bank
                for bank in ("state", "model")
                if bank in trainable_source_banks
            )
        ):
            raise ValueError(
                "trainable_source_banks must be a canonical unique tuple"
            )
        if "model" in trainable_source_banks and channel_count != 2:
            raise ValueError(
                "a live model source bank requires two Gaussian channels"
            )

        self.vocabulary = vocabulary
        self.horizon = horizon
        self.latent_width = latent_width
        self.recognition_width = recognition_width
        self.channel_count = channel_count
        self.family = family
        self.conditioning_mode = conditioning_mode
        self.trainable_source_banks = trainable_source_banks
        self.gaussian_dimension = channel_count * latent_width
        self.block_sizes = (latent_width,) * channel_count

        self.token_embedding = nn.Embedding(
            vocabulary.size, recognition_width, dtype=torch.float64
        )
        with torch.no_grad():
            self.token_embedding.weight.copy_(
                _deterministic_matrix(
                    vocabulary.size, recognition_width, scale=0.125
                )
            )
        self.mean_weight = nn.Parameter(
            _deterministic_matrix(
                self.gaussian_dimension, recognition_width, scale=0.25
            )
        )
        self.mean_bias = nn.Parameter(
            torch.zeros(self.gaussian_dimension, dtype=torch.float64)
        )

        packed_count = (
            self.gaussian_dimension * (self.gaussian_dimension + 1) // 2
            if family == "structured"
            else channel_count * latent_width * (latent_width + 1) // 2
        )
        # Factorized storage contains only within-block entries. There are no
        # trainable off-block scalars that would be permanently masked.
        self.packed_precision_cholesky = nn.Parameter(
            torch.zeros(packed_count, dtype=torch.float64)
        )
        self.source_residual_vectors = nn.ParameterDict()
        self.source_lag_scalars = nn.ParameterDict()
        self.source_shift_vectors = nn.ParameterDict()
        for bank_index, bank in enumerate(
            self.trainable_source_banks, start=1
        ):
            bank_scale = 0.03125 * bank_index
            self.source_residual_vectors[bank] = nn.Parameter(
                _deterministic_nonzero_vector(
                    recognition_width, scale=bank_scale
                )
            )
            self.source_lag_scalars[bank] = nn.Parameter(
                torch.tensor((bank_scale,), dtype=torch.float64)
            )
            self.source_shift_vectors[bank] = nn.Parameter(
                _deterministic_nonzero_vector(
                    latent_width, scale=0.5 * bank_scale
                )
            )

    def _context(self, conditioning: RecognitionConditioning) -> Tensor:
        if type(conditioning) is not RecognitionConditioning:
            raise ValueError(
                "conditioning must be an exact RecognitionConditioning"
            )
        conditioning.__post_init__()
        if conditioning.horizon != self.horizon:
            raise ValueError("recognition conditioning horizon does not match")
        if conditioning.mode != self.conditioning_mode:
            raise ValueError(
                "recognition conditioning mode does not match the store profile"
            )
        tokens = conditioning.observed_tokens.value()
        if bool(torch.any(tokens < 0)) or bool(
            torch.any(tokens >= self.vocabulary.size)
        ):
            raise ValueError("recognition tokens fall outside the vocabulary")
        embedded = self.token_embedding(tokens)
        if conditioning.mode == "smoothing":
            return embedded.mean(dim=0)

        # Filtering encodes the ordered causal contexts rather than silently
        # reusing the smoothing statistic at the terminal receiver.
        denominators = torch.arange(
            1,
            self.horizon + 1,
            dtype=torch.float64,
            device=embedded.device,
        ).unsqueeze(1)
        causal_means = torch.cumsum(embedded, dim=0) / denominators
        return causal_means.mean(dim=0)

    @staticmethod
    def _fill_block(packed: Tensor, *, dimension: int) -> Tensor:
        row, column = torch.tril_indices(
            dimension, dimension, device=packed.device
        )
        result = torch.zeros(
            (dimension, dimension),
            dtype=packed.dtype,
            device=packed.device,
        )
        diagonal = row == column
        values = packed.clone()
        values[diagonal] = F.softplus(values[diagonal]) + 1e-6
        return result.index_put((row, column), values)

    def _precision_cholesky(self) -> Tensor:
        if self.family == "structured":
            return self._fill_block(
                self.packed_precision_cholesky,
                dimension=self.gaussian_dimension,
            )

        blocks: list[Tensor] = []
        packed_per_block = self.latent_width * (self.latent_width + 1) // 2
        for block_index in range(self.channel_count):
            start = block_index * packed_per_block
            stop = start + packed_per_block
            blocks.append(
                self._fill_block(
                    self.packed_precision_cholesky[start:stop],
                    dimension=self.latent_width,
                )
            )
        return torch.block_diag(*blocks)

    def recognition_law(
        self, conditioning: RecognitionConditioning
    ) -> LanguageRecognitionLaw:
        """Emit one normalized Gaussian law with live autograd connectivity."""

        context = self._context(conditioning)
        mean = self.mean_weight @ context + self.mean_bias
        precision_cholesky = self._precision_cholesky()
        if self.family == "structured":
            return StructuredLanguageRecognition.create(
                conditioning=conditioning,
                mean=mean,
                precision_cholesky=precision_cholesky,
            )
        return FactorizedLanguageRecognition.create(
            conditioning=conditioning,
            mean=mean,
            precision_cholesky=precision_cholesky,
            block_sizes=self.block_sizes,
        )

    def recognition_trajectory(
        self,
        conditioning: RecognitionConditioning,
        *,
        prior_feature_provider: RecognitionPriorFeatureProvider,
    ) -> LanguageRecognitionTrajectory:
        """Emit the distinct receiver-indexed H6 v3 recognition trajectory."""

        if type(conditioning) is not RecognitionConditioning:
            raise ValueError(
                "conditioning must be an exact RecognitionConditioning"
            )
        conditioning.__post_init__()
        if conditioning.horizon != self.horizon:
            raise ValueError("recognition conditioning horizon does not match")
        if conditioning.mode != self.conditioning_mode:
            raise ValueError(
                "recognition conditioning mode does not match the store profile"
            )
        owned_tokens = conditioning.observed_tokens.value()
        if bool(torch.any(owned_tokens < 0)) or bool(
            torch.any(owned_tokens >= self.vocabulary.size)
        ):
            raise ValueError("recognition tokens fall outside the vocabulary")
        tokens = owned_tokens.to(device=self.token_embedding.weight.device)

        contexts = build_receiver_contexts(
            conditioning=conditioning,
            token_embeddings=self.token_embedding(tokens),
        )
        base_means = F.linear(contexts, self.mean_weight, self.mean_bias)
        source_parameters = {
            bank: SourceRecognitionParameters(
                bank,
                self.source_residual_vectors[bank],
                self.source_lag_scalars[bank],
                self.source_shift_vectors[bank],
            )
            for bank in self.trainable_source_banks
        }
        store_state_sha256 = canonical_model_state_sha256(self)
        trajectory = build_language_recognition_trajectory(
            conditioning=conditioning,
            vocabulary=self.vocabulary,
            family=self.family,
            block_sizes=self.block_sizes,
            contexts=contexts,
            base_means=base_means,
            shared_precision_cholesky=self._precision_cholesky(),
            latent_width=self.latent_width,
            channel_count=self.channel_count,
            source_parameters=source_parameters,
            prior_feature_provider=prior_feature_provider,
            recognition_store_state_sha256=store_state_sha256,
        )
        if canonical_model_state_sha256(self) != store_state_sha256:
            raise ValueError(
                "recognition store mutated while emitting its trajectory"
            )
        return trajectory


__all__ = [
    "LanguageRecognitionLaw",
    "LanguageRecognitionParameterStore",
    "RecognitionStoreConditioning",
    "RecognitionStoreFamily",
]
