"""Trainable parameter ownership for H6 language-recognition laws."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vfe4.types.h6 import VocabularyIdentity

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


class LanguageRecognitionParameterStore(nn.Module):
    """Own parameters and emit ephemeral normalized Task-4 Gaussian laws."""

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

        self.vocabulary = vocabulary
        self.horizon = horizon
        self.latent_width = latent_width
        self.recognition_width = recognition_width
        self.channel_count = channel_count
        self.family = family
        self.conditioning_mode = conditioning_mode
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


__all__ = [
    "LanguageRecognitionLaw",
    "LanguageRecognitionParameterStore",
    "RecognitionStoreConditioning",
    "RecognitionStoreFamily",
]
