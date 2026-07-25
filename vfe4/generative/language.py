"""Minimal normalized H6 language-generative factors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import (
    FrozenTensorSnapshot,
    H6LanguageStructure,
    VocabularyIdentity,
    canonical_json_bytes,
)

from .source_priors import (
    FixedSourcePrior,
    NormalizedSourceFactor,
    PrefixConditionedSourcePrior,
)


LanguagePartition = Literal[
    "initial", "state_transition", "model_transition", "emission"
]


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _vocabulary_sha256(vocabulary: VocabularyIdentity) -> str:
    return _owned_hash(
        "vfe4.h6.vocabulary-identity.v1",
        {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
    )


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


@dataclass(frozen=True)
class NormalizedLanguageFactor:
    """One typed normalized language factor and its immutable log value(s)."""

    receiver_t: int
    partition: LanguagePartition
    vocabulary_size: int
    factor_context_sha256: str
    factor_identity_sha256: str
    log_values: FrozenTensorSnapshot

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("receiver_t must be a nonnegative integer")
        if self.partition not in (
            "initial",
            "state_transition",
            "model_transition",
            "emission",
        ):
            raise ValueError("unsupported language factor partition")
        if type(self.vocabulary_size) is not int or self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be a positive integer")
        _require_sha256(self.factor_context_sha256, "factor_context_sha256")
        _require_sha256(self.factor_identity_sha256, "factor_identity_sha256")
        if type(self.log_values) is not FrozenTensorSnapshot:
            raise ValueError("log_values must be a FrozenTensorSnapshot")
        self.log_values.assert_intact()
        values = self.log_values.value()
        if values.dtype is not torch.float64:
            raise ValueError("language factor values must use float64")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("language factor values must be finite")
        if self.partition == "initial":
            if self.receiver_t != 0:
                raise ValueError("initial factors must use receiver_t=0")
        elif self.receiver_t <= 0:
            raise ValueError("noninitial factors require a positive receiver_t")
        if self.partition == "emission":
            if values.shape != (self.vocabulary_size,):
                raise ValueError("emission log-probabilities must have vocabulary shape")
            allowance = 128.0 * math.ulp(1.0) * max(1, self.vocabulary_size)
            if abs(float(torch.logsumexp(values, dim=0).item())) > allowance:
                raise ValueError("emission log-probabilities must be normalized")
        elif values.shape != ():
            raise ValueError("continuous language log-probabilities must be scalar")
        expected = _owned_hash(
            "vfe4.h6.normalized-language-factor.v1",
            {
                "receiver_t": self.receiver_t,
                "partition": self.partition,
                "vocabulary_size": self.vocabulary_size,
                "factor_context_sha256": self.factor_context_sha256,
                "log_values": _snapshot_payload(self.log_values),
            },
        )
        if self.factor_identity_sha256 != expected:
            raise ValueError("language factor identity does not match the normalized record")

    @classmethod
    def create(
        cls,
        *,
        receiver_t: int,
        partition: LanguagePartition,
        vocabulary_size: int,
        factor_context_sha256: str,
        log_values: Tensor,
    ) -> "NormalizedLanguageFactor":
        snapshot = FrozenTensorSnapshot.capture(log_values)
        identity = _owned_hash(
            "vfe4.h6.normalized-language-factor.v1",
            {
                "receiver_t": receiver_t,
                "partition": partition,
                "vocabulary_size": vocabulary_size,
                "factor_context_sha256": factor_context_sha256,
                "log_values": _snapshot_payload(snapshot),
            },
        )
        return cls(
            receiver_t,
            partition,
            vocabulary_size,
            factor_context_sha256,
            identity,
            snapshot,
        )


SourcePrior = FixedSourcePrior | PrefixConditionedSourcePrior


@dataclass(frozen=True, eq=False)
class H7LanguageGenerativeGeometry:
    """Additive live H7 geometry absent from the H6 training arithmetic."""

    frames: tuple[Tensor, Tensor, Tensor]
    support_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.support_sha256, "support_sha256")
        if type(self.frames) is not tuple or len(self.frames) != 3:
            raise ValueError("H7 geometry requires exactly three frames")
        first = self.frames[0]
        if (
            not isinstance(first, Tensor)
            or first.dtype is not torch.float64
            or first.ndim != 2
            or first.shape[0] != first.shape[1]
            or first.shape[0] not in (1, 2)
        ):
            raise ValueError("H7 frames must be float64 GL+(1) or GL+(2)")
        for frame in self.frames:
            if (
                not isinstance(frame, Tensor)
                or frame.dtype is not torch.float64
                or tuple(frame.shape) != tuple(first.shape)
                or not bool(torch.isfinite(frame).all())
            ):
                raise ValueError("H7 frames must share one finite shape")
            sign, logabsdet = torch.linalg.slogdet(frame)
            if not bool(torch.isfinite(logabsdet)) or not bool(sign > 0):
                raise ValueError("H7 frames must have positive determinant")


@dataclass(frozen=True, eq=False)
class H7LanguageGenerativeTrace:
    """Immutable references to the exact live tensors used by H6 factors."""

    frames: tuple[Tensor, Tensor, Tensor]
    receiver_labels: tuple[int, int]
    support_sha256: str
    initial_mean: Tensor
    initial_log_scale: Tensor
    model_transition_weight: Tensor
    model_transition_bias: Tensor
    model_transition_log_scale: Tensor
    state_transition_weight: Tensor
    state_model_weight: Tensor
    state_transition_bias: Tensor
    state_transition_log_scale: Tensor
    emission_state_weight: Tensor
    emission_model_weight: Tensor
    emission_bias: Tensor

    def __post_init__(self) -> None:
        geometry = H7LanguageGenerativeGeometry(
            self.frames, self.support_sha256
        )
        dimension = geometry.frames[0].shape[0]
        if self.receiver_labels != (1, 2):
            raise ValueError("H7 live trace requires receiver labels (1, 2)")
        if (
            not isinstance(self.emission_bias, Tensor)
            or self.emission_bias.ndim != 1
            or self.emission_bias.numel() < 2
        ):
            raise ValueError("H7 live trace emission bias is invalid")
        expected_shapes = {
            "initial_mean": (2 * dimension,),
            "initial_log_scale": (2 * dimension,),
            "model_transition_weight": (dimension, dimension),
            "model_transition_bias": (dimension,),
            "model_transition_log_scale": (dimension,),
            "state_transition_weight": (dimension, dimension),
            "state_model_weight": (dimension, dimension),
            "state_transition_bias": (dimension,),
            "state_transition_log_scale": (dimension,),
            "emission_state_weight": (self.emission_bias.numel(), dimension),
            "emission_model_weight": (self.emission_bias.numel(), dimension),
            "emission_bias": (self.emission_bias.numel(),),
        }
        for name, shape in expected_shapes.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.dtype is not torch.float64
                or tuple(value.shape) != shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"H7 live generative tensor {name} is invalid"
                )


class LanguageGenerativeModel(nn.Module):
    """Normalized initial, transition, source, and categorical emission factors."""

    def __init__(
        self,
        *,
        structure: H6LanguageStructure,
        vocabulary: VocabularyIdentity,
        model_family_sha256: str,
        latent_dim: int,
        source_prior: SourcePrior | None,
        h7_geometry: H7LanguageGenerativeGeometry | None = None,
    ) -> None:
        super().__init__()
        if type(structure) is not H6LanguageStructure:
            raise ValueError("structure must be an exact H6LanguageStructure")
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if source_prior is not None and type(source_prior) not in (
            FixedSourcePrior,
            PrefixConditionedSourcePrior,
        ):
            raise ValueError("source_prior must be a supported exact source-prior type")
        if h7_geometry is not None and type(
            h7_geometry
        ) is not H7LanguageGenerativeGeometry:
            raise ValueError(
                "h7_geometry must be an exact H7LanguageGenerativeGeometry"
            )
        structure.__post_init__()
        vocabulary.__post_init__()
        checked_model_family_sha256 = _require_sha256(
            model_family_sha256, "model_family_sha256"
        )
        if source_prior is not None and (
            source_prior.structure != structure or source_prior.vocabulary != vocabulary
        ):
            raise ValueError("source prior identities do not match the language model")
        if (
            source_prior is not None
            and source_prior.model_family_sha256 != checked_model_family_sha256
        ):
            raise ValueError(
                "source prior model_family_sha256 does not match the language model"
            )
        if (
            type(source_prior) is PrefixConditionedSourcePrior
            and source_prior.latent_dim != latent_dim
        ):
            raise ValueError("prefix source-prior latent_dim does not match the language model")
        self.structure = structure
        self.vocabulary = vocabulary
        self.model_family_sha256 = checked_model_family_sha256
        self.vocabulary_sha256 = _vocabulary_sha256(vocabulary)
        self.latent_dim = latent_dim
        self.source_prior = source_prior
        self.h7_geometry = h7_geometry
        if h7_geometry is not None:
            h7_geometry.__post_init__()
            if h7_geometry.frames[0].shape != (latent_dim, latent_dim):
                raise ValueError(
                    "h7_geometry frame width must equal latent_dim"
                )

        pair_dim = 2 * latent_dim
        self.initial_mean = nn.Parameter(torch.zeros(pair_dim, dtype=torch.float64))
        self.initial_log_scale = nn.Parameter(torch.zeros(pair_dim, dtype=torch.float64))
        self.model_transition_weight = nn.Parameter(
            torch.eye(latent_dim, dtype=torch.float64)
        )
        self.model_transition_bias = nn.Parameter(
            torch.zeros(latent_dim, dtype=torch.float64)
        )
        self.model_transition_log_scale = nn.Parameter(
            torch.zeros(latent_dim, dtype=torch.float64)
        )
        self.state_transition_weight = nn.Parameter(
            torch.eye(latent_dim, dtype=torch.float64)
        )
        self.state_model_weight = nn.Parameter(
            torch.zeros((latent_dim, latent_dim), dtype=torch.float64)
        )
        self.state_transition_bias = nn.Parameter(
            torch.zeros(latent_dim, dtype=torch.float64)
        )
        self.state_transition_log_scale = nn.Parameter(
            torch.zeros(latent_dim, dtype=torch.float64)
        )
        self.emission_state_weight = nn.Parameter(
            torch.zeros((vocabulary.size, latent_dim), dtype=torch.float64)
        )
        self.emission_model_weight = nn.Parameter(
            torch.zeros((vocabulary.size, latent_dim), dtype=torch.float64)
        )
        self.emission_bias = nn.Parameter(
            torch.zeros(vocabulary.size, dtype=torch.float64)
        )

    def export_h7_trace(self) -> H7LanguageGenerativeTrace:
        """Return direct live references; no H7 tensor is cloned here."""

        if type(self.h7_geometry) is not H7LanguageGenerativeGeometry:
            raise ValueError(
                "LanguageGenerativeModel has no complete H7 geometry trace"
            )
        self.h7_geometry.__post_init__()
        return H7LanguageGenerativeTrace(
            frames=self.h7_geometry.frames,
            receiver_labels=self.structure.receiver_labels,  # type: ignore[arg-type]
            support_sha256=self.h7_geometry.support_sha256,
            initial_mean=self.initial_mean,
            initial_log_scale=self.initial_log_scale,
            model_transition_weight=self.model_transition_weight,
            model_transition_bias=self.model_transition_bias,
            model_transition_log_scale=self.model_transition_log_scale,
            state_transition_weight=self.state_transition_weight,
            state_model_weight=self.state_model_weight,
            state_transition_bias=self.state_transition_bias,
            state_transition_log_scale=self.state_transition_log_scale,
            emission_state_weight=self.emission_state_weight,
            emission_model_weight=self.emission_model_weight,
            emission_bias=self.emission_bias,
        )

    def _receiver_row(self, receiver_t: int) -> tuple[int, ...]:
        if type(receiver_t) is not int or receiver_t not in self.structure.receiver_labels:
            raise ValueError("receiver_t is not declared by the language DAG")
        index = self.structure.receiver_labels.index(receiver_t)
        return self.structure.dag.rows[index].parents

    def _vector(self, value: Tensor, name: str) -> Tensor:
        if (
            type(value) is not Tensor
            or value.dtype is not torch.float64
            or value.shape != (self.latent_dim,)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} must be a finite float64 latent vector")
        return value

    def _history(self, value: Tensor, receiver_t: int, name: str) -> Tensor:
        if (
            type(value) is not Tensor
            or value.dtype is not torch.float64
            or value.shape != (receiver_t, self.latent_dim)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"{name} must be finite float64 shape (receiver_t, latent_dim)"
            )
        return value

    def _source_index(self, receiver_t: int, source_index: int | None) -> int:
        parents = self._receiver_row(receiver_t)
        if self.source_prior is None:
            if source_index is not None:
                raise ValueError(
                    "source-free A3 transitions require the fixed immediate predecessor"
                )
            return receiver_t - 1
        if type(source_index) is not int or source_index not in parents:
            raise ValueError("source_index must be one declared causal parent")
        return source_index

    def _factor(
        self,
        *,
        receiver_t: int,
        partition: LanguagePartition,
        log_values: Tensor,
        source_index: int | None = None,
    ) -> NormalizedLanguageFactor:
        context_identity = _owned_hash(
            "vfe4.h6.language-factor-context.v1",
            {
                "structure_sha256": self.structure.structure_sha256,
                "model_family_sha256": self.model_family_sha256,
                "vocabulary_sha256": self.vocabulary_sha256,
                "receiver_t": receiver_t,
                "partition": partition,
                "source_index": source_index,
            },
        )
        return NormalizedLanguageFactor.create(
            receiver_t=receiver_t,
            partition=partition,
            vocabulary_size=self.vocabulary.size,
            factor_context_sha256=context_identity,
            log_values=log_values,
        )

    @staticmethod
    def _diagonal_gaussian_log_prob(
        value: Tensor, mean: Tensor, log_scale: Tensor
    ) -> Tensor:
        if not bool(torch.isfinite(log_scale).all()):
            raise ValueError("Gaussian log scales must remain finite")
        standardized = (value - mean) * torch.exp(-log_scale)
        return -0.5 * torch.sum(
            standardized.square() + 2.0 * log_scale + math.log(2.0 * math.pi)
        )

    def initial_log_prob(self, *, initial_latents: Tensor) -> NormalizedLanguageFactor:
        if (
            type(initial_latents) is not Tensor
            or initial_latents.dtype is not torch.float64
            or initial_latents.shape != (2, self.latent_dim)
            or not bool(torch.isfinite(initial_latents).all())
        ):
            raise ValueError("initial_latents must be finite float64 shape (2, latent_dim)")
        value = initial_latents.reshape(-1)
        log_prob = self._diagonal_gaussian_log_prob(
            value, self.initial_mean, self.initial_log_scale
        )
        return self._factor(
            receiver_t=0, partition="initial", log_values=log_prob
        )

    def state_source_log_probs(
        self,
        *,
        receiver_t: int,
        prefix: CausalPrefix | None = None,
        earlier_latents: Tensor | None = None,
    ) -> NormalizedSourceFactor:
        if self.source_prior is None:
            raise ValueError("source variables are structurally absent for A3")
        if type(self.source_prior) is FixedSourcePrior:
            if prefix is not None or earlier_latents is not None:
                raise ValueError("fixed source priors do not accept a prefix context")
            return self.source_prior.state_source_log_probs(receiver_t=receiver_t)
        if prefix is None or earlier_latents is None or prefix.receiver_t != receiver_t:
            raise ValueError("prefix-conditioned source priors require the exact receiver context")
        return self.source_prior.state_source_log_probs(
            prefix=prefix, earlier_latents=earlier_latents
        )

    def model_source_log_probs(
        self,
        *,
        receiver_t: int,
        prefix: CausalPrefix | None = None,
        earlier_latents: Tensor | None = None,
    ) -> NormalizedSourceFactor:
        if self.source_prior is None:
            raise ValueError("source variables are structurally absent for A3")
        if type(self.source_prior) is FixedSourcePrior:
            if prefix is not None or earlier_latents is not None:
                raise ValueError("fixed source priors do not accept a prefix context")
            return self.source_prior.model_source_log_probs(receiver_t=receiver_t)
        if prefix is None or earlier_latents is None or prefix.receiver_t != receiver_t:
            raise ValueError("prefix-conditioned source priors require the exact receiver context")
        return self.source_prior.model_source_log_probs(
            prefix=prefix, earlier_latents=earlier_latents
        )

    def model_transition_log_prob(
        self,
        *,
        receiver_t: int,
        current_model: Tensor,
        earlier_models: Tensor,
        source_index: int | None,
    ) -> NormalizedLanguageFactor:
        current = self._vector(current_model, "current_model")
        history = self._history(earlier_models, receiver_t, "earlier_models")
        source = self._source_index(receiver_t, source_index)
        mean = self.model_transition_weight @ history[source] + self.model_transition_bias
        log_prob = self._diagonal_gaussian_log_prob(
            current, mean, self.model_transition_log_scale
        )
        return self._factor(
            receiver_t=receiver_t,
            partition="model_transition",
            log_values=log_prob,
            source_index=source,
        )

    def state_transition_log_prob(
        self,
        *,
        receiver_t: int,
        current_state: Tensor,
        current_model: Tensor,
        earlier_states: Tensor,
        source_index: int | None,
    ) -> NormalizedLanguageFactor:
        current = self._vector(current_state, "current_state")
        model = self._vector(current_model, "current_model")
        history = self._history(earlier_states, receiver_t, "earlier_states")
        source = self._source_index(receiver_t, source_index)
        mean = (
            self.state_transition_weight @ history[source]
            + self.state_model_weight @ model
            + self.state_transition_bias
        )
        log_prob = self._diagonal_gaussian_log_prob(
            current, mean, self.state_transition_log_scale
        )
        return self._factor(
            receiver_t=receiver_t,
            partition="state_transition",
            log_values=log_prob,
            source_index=source,
        )

    def emission_log_probs(
        self,
        *,
        receiver_t: int,
        current_state: Tensor,
        current_model: Tensor,
    ) -> NormalizedLanguageFactor:
        self._receiver_row(receiver_t)
        state = self._vector(current_state, "current_state")
        model = self._vector(current_model, "current_model")
        logits = (
            self.emission_state_weight @ state
            + self.emission_model_weight @ model
            + self.emission_bias
        )
        return self._factor(
            receiver_t=receiver_t,
            partition="emission",
            log_values=torch.log_softmax(logits, dim=0),
        )


__all__ = [
    "H7LanguageGenerativeGeometry",
    "H7LanguageGenerativeTrace",
    "LanguageGenerativeModel",
    "NormalizedLanguageFactor",
]
