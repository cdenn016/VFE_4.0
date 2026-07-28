"""Receiver-indexed H6 prediction-recognition trajectories."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vfe4.data.windows import CausalPrefix
from vfe4.predictive.identities import canonical_model_state_sha256
from vfe4.types.h6 import (
    FrozenTensorSnapshot,
    VocabularyIdentity,
    canonical_json_bytes,
)

from .language import RecognitionConditioning


SourceBankName = Literal["state", "model"]
RecognitionFamily = Literal["structured", "factorized"]
H6_RECOGNITION_POSITION_DESCRIPTOR_SCHEMA = (
    "frozen_float64_sinusoidal_receiver_position_v1"
)
_LOWER_HEX = frozenset("0123456789abcdef")


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
    if type(snapshot) is not FrozenTensorSnapshot:
        raise ValueError("tensor record must be an exact FrozenTensorSnapshot")
    snapshot.assert_intact()
    return {
        "dtype": snapshot.dtype,
        "shape": snapshot.shape,
        "device": snapshot.device,
        "contiguous": snapshot.contiguous,
        "requires_grad": snapshot.requires_grad,
        "raw_bytes_sha256": snapshot.raw_bytes_sha256,
    }


def _require_live_float64(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    finite: bool = True,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype is not torch.float64
        or (ndim is not None and value.ndim != ndim)
        or (shape is not None and tuple(value.shape) != shape)
    ):
        raise ValueError(f"{name} must be a float64 tensor with the required shape")
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class H6ActiveHorizonV3:
    """One prefix-closed evaluation horizon under a fixed maximum model."""

    maximum_horizon: int
    active_horizon: int
    active_receiver_mask: tuple[bool, ...]
    evaluation_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "maximum_horizon": self.maximum_horizon,
            "active_horizon": self.active_horizon,
            "active_receiver_mask": self.active_receiver_mask,
        }

    def __post_init__(self) -> None:
        if (
            type(self.maximum_horizon) is not int
            or self.maximum_horizon <= 0
            or type(self.active_horizon) is not int
            or not 1 <= self.active_horizon <= self.maximum_horizon
        ):
            raise ValueError(
                "active horizon must be an exact integer in 1..maximum_horizon"
            )
        expected_mask = (True,) * (self.active_horizon + 1) + (False,) * (
            self.maximum_horizon - self.active_horizon
        )
        if (
            type(self.active_receiver_mask) is not tuple
            or any(type(active) is not bool for active in self.active_receiver_mask)
            or self.active_receiver_mask != expected_mask
        ):
            raise ValueError(
                "active receiver mask must be the exact prefix 0..active_horizon"
            )
        _require_sha256(
            self.evaluation_identity_sha256,
            "evaluation_identity_sha256",
        )
        if self.evaluation_identity_sha256 != _owned_hash(
            "vfe4.h6.active-horizon-evaluation.v3",
            self.canonical_payload(),
        ):
            raise ValueError("active-horizon evaluation identity is stale")

    @classmethod
    def create(
        cls,
        *,
        maximum_horizon: int,
        active_horizon: int,
        active_receiver_mask: tuple[bool, ...] | None = None,
    ) -> "H6ActiveHorizonV3":
        if (
            type(maximum_horizon) is not int
            or maximum_horizon <= 0
            or type(active_horizon) is not int
            or not 1 <= active_horizon <= maximum_horizon
        ):
            raise ValueError(
                "active horizon must be an exact integer in 1..maximum_horizon"
            )
        expected_mask = (True,) * (active_horizon + 1) + (False,) * (
            maximum_horizon - active_horizon
        )
        mask = expected_mask if active_receiver_mask is None else active_receiver_mask
        payload = {
            "maximum_horizon": maximum_horizon,
            "active_horizon": active_horizon,
            "active_receiver_mask": mask,
        }
        return cls(
            maximum_horizon=maximum_horizon,
            active_horizon=active_horizon,
            active_receiver_mask=mask,
            evaluation_identity_sha256=_owned_hash(
                "vfe4.h6.active-horizon-evaluation.v3",
                payload,
            ),
        )


def _capture_float64(
    value: Tensor,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> FrozenTensorSnapshot:
    checked = _require_live_float64(value, name=name, ndim=ndim, shape=shape)
    return FrozenTensorSnapshot.capture(checked)


def _require_causal_support(support: object, *, receiver_t: int) -> tuple[int, ...]:
    if (
        type(receiver_t) is not int
        or receiver_t <= 0
        or type(support) is not tuple
        or not support
        or any(type(source_j) is not int for source_j in support)
        or tuple(sorted(set(support))) != support
        or any(source_j < 0 or source_j >= receiver_t for source_j in support)
    ):
        raise ValueError(
            "each source support must be nonempty, unique, and strictly earlier"
        )
    return support


class RecognitionPriorFeatureProvider(nn.Module, ABC):
    """Typed source-model boundary evaluated with detached module state."""

    @property
    def source_model(self) -> nn.Module:
        """Return the model whose current state the proposal feature uses."""

        return self

    @abstractmethod
    def forward(
        self,
        *,
        bank: SourceBankName,
        causal_prefix: CausalPrefix,
        earlier_recognition_means: Tensor,
    ) -> "RecognitionPriorFeature":
        """Derive positive support and mean-evaluated prior features."""

        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class RecognitionPriorFeature:
    bank: SourceBankName
    causal_prefix_sha256: str
    support: tuple[int, ...]
    log_prior_features: Tensor

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("recognition prior feature bank is invalid")
        _require_sha256(self.causal_prefix_sha256, "causal_prefix_sha256")
        if (
            type(self.support) is not tuple
            or not self.support
            or any(type(source_j) is not int for source_j in self.support)
            or not isinstance(self.log_prior_features, Tensor)
            or self.log_prior_features.dtype is not torch.float64
            or self.log_prior_features.ndim != 1
            or self.log_prior_features.shape != (len(self.support),)
        ):
            raise ValueError(
                "recognition prior feature must carry one compact float64 support row"
            )


@dataclass(frozen=True, eq=False)
class SourceRecognitionParameters:
    bank: SourceBankName
    residual_vector: Tensor
    lag_scalar: Tensor
    shift_vector: Tensor

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("source parameter bank must be state or model")
        residual = _require_live_float64(
            self.residual_vector, name="residual_vector", ndim=1
        )
        lag = _require_live_float64(self.lag_scalar, name="lag_scalar", shape=(1,))
        shift = _require_live_float64(self.shift_vector, name="shift_vector", ndim=1)
        if (
            residual.numel() == 0
            or shift.numel() == 0
            or residual.device != lag.device
            or residual.device != shift.device
        ):
            raise ValueError("source parameter triple is inconsistent")


@dataclass(frozen=True, eq=False)
class ReceiverRecognitionContext:
    receiver_t: int
    conditioning_mode: Literal["filtering", "smoothing"]
    context_snapshot: FrozenTensorSnapshot
    context_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "receiver_t": self.receiver_t,
            "conditioning_mode": self.conditioning_mode,
            "context": _snapshot_payload(self.context_snapshot),
        }

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("receiver context label must be nonnegative")
        if self.conditioning_mode not in ("filtering", "smoothing"):
            raise ValueError("receiver context conditioning mode is invalid")
        context = self.context
        if context.ndim != 1 or context.numel() == 0:
            raise ValueError("receiver context must be a nonempty vector")
        _require_sha256(self.context_identity_sha256, "context_identity_sha256")
        expected = _owned_hash(
            "vfe4.h6.receiver-recognition-context.v1",
            self.canonical_payload(),
        )
        if self.context_identity_sha256 != expected:
            raise ValueError("receiver context identity is stale")

    @classmethod
    def create(
        cls,
        *,
        receiver_t: int,
        conditioning_mode: Literal["filtering", "smoothing"],
        context: Tensor,
    ) -> "ReceiverRecognitionContext":
        snapshot = _capture_float64(context, name="receiver context", ndim=1)
        payload = {
            "receiver_t": receiver_t,
            "conditioning_mode": conditioning_mode,
            "context": _snapshot_payload(snapshot),
        }
        return cls(
            receiver_t,
            conditioning_mode,
            snapshot,
            _owned_hash("vfe4.h6.receiver-recognition-context.v1", payload),
        )

    @property
    def context(self) -> Tensor:
        self.context_snapshot.assert_intact()
        return self.context_snapshot.value()


@dataclass(frozen=True, eq=False)
class CategoricalSourceRow:
    bank: SourceBankName
    receiver_t: int
    support: tuple[int, ...]
    causal_prefix_sha256: str
    source_model_state_sha256: str
    log_prior_baseline_snapshot: FrozenTensorSnapshot
    residual_scores_snapshot: FrozenTensorSnapshot
    log_probabilities_snapshot: FrozenTensorSnapshot
    entropy_snapshot: FrozenTensorSnapshot
    row_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bank": self.bank,
            "receiver_t": self.receiver_t,
            "support": self.support,
            "causal_prefix_sha256": self.causal_prefix_sha256,
            "source_model_state_sha256": self.source_model_state_sha256,
            "log_prior_baseline": _snapshot_payload(self.log_prior_baseline_snapshot),
            "residual_scores": _snapshot_payload(self.residual_scores_snapshot),
            "log_probabilities": _snapshot_payload(self.log_probabilities_snapshot),
            "entropy": _snapshot_payload(self.entropy_snapshot),
        }

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("categorical source bank is invalid")
        _require_causal_support(self.support, receiver_t=self.receiver_t)
        _require_sha256(self.causal_prefix_sha256, "causal_prefix_sha256")
        _require_sha256(
            self.source_model_state_sha256,
            "source_model_state_sha256",
        )
        shape = (len(self.support),)
        tensors = (
            _require_live_float64(
                self.log_prior_baseline,
                name="log_prior_baseline",
                shape=shape,
            ),
            _require_live_float64(
                self.residual_scores,
                name="residual_scores",
                shape=shape,
            ),
            _require_live_float64(
                self.log_probabilities,
                name="log_probabilities",
                shape=shape,
            ),
            _require_live_float64(self.entropy, name="categorical entropy", shape=()),
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("categorical source tensors must share one device")
        if not bool(
            torch.allclose(
                torch.logsumexp(tensors[2], dim=0),
                tensors[2].new_zeros(()),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError("categorical source row must be normalized")
        _require_sha256(self.row_identity_sha256, "row_identity_sha256")
        expected = _owned_hash(
            "vfe4.h6.categorical-source-row.v3",
            self.canonical_payload(),
        )
        if self.row_identity_sha256 != expected:
            raise ValueError("categorical source row identity is stale")

    @classmethod
    def create(
        cls,
        *,
        bank: SourceBankName,
        receiver_t: int,
        support: tuple[int, ...],
        causal_prefix_sha256: str,
        source_model_state_sha256: str,
        log_prior_baseline: Tensor,
        residual_scores: Tensor,
        log_probabilities: Tensor,
        entropy: Tensor,
    ) -> "CategoricalSourceRow":
        snapshots = (
            _capture_float64(
                log_prior_baseline,
                name="log_prior_baseline",
                shape=(len(support),),
            ),
            _capture_float64(
                residual_scores,
                name="residual_scores",
                shape=(len(support),),
            ),
            _capture_float64(
                log_probabilities,
                name="log_probabilities",
                shape=(len(support),),
            ),
            _capture_float64(entropy, name="categorical entropy", shape=()),
        )
        payload = {
            "bank": bank,
            "receiver_t": receiver_t,
            "support": support,
            "causal_prefix_sha256": causal_prefix_sha256,
            "source_model_state_sha256": source_model_state_sha256,
            "log_prior_baseline": _snapshot_payload(snapshots[0]),
            "residual_scores": _snapshot_payload(snapshots[1]),
            "log_probabilities": _snapshot_payload(snapshots[2]),
            "entropy": _snapshot_payload(snapshots[3]),
        }
        return cls(
            bank,
            receiver_t,
            support,
            causal_prefix_sha256,
            source_model_state_sha256,
            *snapshots,
            _owned_hash("vfe4.h6.categorical-source-row.v3", payload),
        )

    @property
    def log_prior_baseline(self) -> Tensor:
        return self.log_prior_baseline_snapshot.value()

    @property
    def residual_scores(self) -> Tensor:
        return self.residual_scores_snapshot.value()

    @property
    def log_probabilities(self) -> Tensor:
        return self.log_probabilities_snapshot.value()

    @property
    def probabilities(self) -> Tensor:
        return self.log_probabilities.exp()

    @property
    def entropy(self) -> Tensor:
        return self.entropy_snapshot.value()


@dataclass(frozen=True, eq=False)
class CategoricalSourceBank:
    bank: SourceBankName
    rows: tuple[CategoricalSourceRow, ...]
    bank_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bank": self.bank,
            "row_identities": tuple(row.row_identity_sha256 for row in self.rows),
        }

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("categorical source bank is invalid")
        if (
            type(self.rows) is not tuple
            or not self.rows
            or tuple(row.receiver_t for row in self.rows)
            != tuple(range(1, len(self.rows) + 1))
            or any(
                type(row) is not CategoricalSourceRow or row.bank != self.bank
                for row in self.rows
            )
        ):
            raise ValueError("categorical source bank rows must cover receivers 1..T")
        for row in self.rows:
            row.__post_init__()
        _require_sha256(self.bank_identity_sha256, "bank_identity_sha256")
        expected = _owned_hash(
            "vfe4.h6.categorical-source-bank.v3",
            self.canonical_payload(),
        )
        if self.bank_identity_sha256 != expected:
            raise ValueError("categorical source bank identity is stale")

    @classmethod
    def create(
        cls,
        *,
        bank: SourceBankName,
        rows: tuple[CategoricalSourceRow, ...],
    ) -> "CategoricalSourceBank":
        payload = {
            "bank": bank,
            "row_identities": tuple(row.row_identity_sha256 for row in rows),
        }
        return cls(
            bank,
            rows,
            _owned_hash("vfe4.h6.categorical-source-bank.v3", payload),
        )


@dataclass(frozen=True, eq=False)
class AbsentSourceBank:
    bank: SourceBankName
    probability_snapshot: FrozenTensorSnapshot
    log_probability_snapshot: FrozenTensorSnapshot
    entropy_snapshot: FrozenTensorSnapshot
    bank_identity_sha256: str
    support: tuple[None, ...] = field(default=(None,), init=False)
    parameter_count: Literal[0] = field(default=0, init=False)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bank": self.bank,
            "support": self.support,
            "probability": _snapshot_payload(self.probability_snapshot),
            "log_probability": _snapshot_payload(self.log_probability_snapshot),
            "entropy": _snapshot_payload(self.entropy_snapshot),
            "parameter_count": self.parameter_count,
        }

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("absent source bank is invalid")
        probability = self.probability
        log_probability = self.log_probability
        entropy = self.entropy
        if (
            probability.shape != ()
            or log_probability.shape != ()
            or entropy.shape != ()
            or probability.dtype is not torch.float64
            or log_probability.dtype is not torch.float64
            or entropy.dtype is not torch.float64
            or probability.device != log_probability.device
            or probability.device != entropy.device
            or not bool(probability == 1.0)
            or not bool(log_probability == 0.0)
            or not bool(entropy == 0.0)
        ):
            raise ValueError(
                "absent source bank must be probability-one and entropy-free"
            )
        _require_sha256(self.bank_identity_sha256, "bank_identity_sha256")
        expected = _owned_hash(
            "vfe4.h6.absent-source-bank.v1", self.canonical_payload()
        )
        if self.bank_identity_sha256 != expected:
            raise ValueError("absent source bank identity is stale")

    @classmethod
    def create(cls, *, bank: SourceBankName, reference: Tensor) -> "AbsentSourceBank":
        probability = FrozenTensorSnapshot.capture(reference.new_ones(()))
        log_probability = FrozenTensorSnapshot.capture(reference.new_zeros(()))
        entropy = FrozenTensorSnapshot.capture(reference.new_zeros(()))
        temporary_payload = {
            "bank": bank,
            "support": (None,),
            "probability": _snapshot_payload(probability),
            "log_probability": _snapshot_payload(log_probability),
            "entropy": _snapshot_payload(entropy),
            "parameter_count": 0,
        }
        return cls(
            bank,
            probability,
            log_probability,
            entropy,
            _owned_hash("vfe4.h6.absent-source-bank.v1", temporary_payload),
        )

    @property
    def probability(self) -> Tensor:
        return self.probability_snapshot.value()

    @property
    def log_probability(self) -> Tensor:
        return self.log_probability_snapshot.value()

    @property
    def entropy(self) -> Tensor:
        return self.entropy_snapshot.value()


SourceBankTrajectory = CategoricalSourceBank | AbsentSourceBank


@dataclass(frozen=True, eq=False)
class GaussianReceiverComponent:
    receiver_t: int
    state_source_j: int | None
    model_source_j: int | None
    mean_snapshot: FrozenTensorSnapshot
    precision_cholesky_snapshot: FrozenTensorSnapshot
    log_probability_snapshot: FrozenTensorSnapshot
    precision_identity_sha256: str
    component_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "receiver_t": self.receiver_t,
            "state_source_j": self.state_source_j,
            "model_source_j": self.model_source_j,
            "mean": _snapshot_payload(self.mean_snapshot),
            "precision_identity_sha256": self.precision_identity_sha256,
            "log_probability": _snapshot_payload(self.log_probability_snapshot),
        }

    def __post_init__(self) -> None:
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("receiver_t must be nonnegative")
        for source_j in (self.state_source_j, self.model_source_j):
            if source_j is not None and (
                type(source_j) is not int or source_j < 0 or source_j >= self.receiver_t
            ):
                raise ValueError("component sources must be strictly causal")
        mean = self.mean
        cholesky = self.precision_cholesky
        log_probability = self.log_probability
        if (
            mean.ndim != 1
            or mean.dtype is not torch.float64
            or cholesky.shape != (mean.numel(), mean.numel())
            or cholesky.dtype is not torch.float64
            or log_probability.shape != ()
            or log_probability.dtype is not torch.float64
            or mean.device != cholesky.device
            or mean.device != log_probability.device
            or not torch.equal(cholesky, torch.tril(cholesky))
            or not bool(torch.all(torch.diagonal(cholesky) > 0.0))
        ):
            raise ValueError("Gaussian receiver component is inconsistent")
        _require_sha256(
            self.precision_identity_sha256,
            "precision_identity_sha256",
        )
        expected_precision = _owned_hash(
            "vfe4.h6.shared-recognition-precision.v1",
            _snapshot_payload(self.precision_cholesky_snapshot),
        )
        if self.precision_identity_sha256 != expected_precision:
            raise ValueError("component precision identity is stale")
        _require_sha256(
            self.component_identity_sha256,
            "component_identity_sha256",
        )
        expected = _owned_hash(
            "vfe4.h6.gaussian-receiver-component.v3",
            self.canonical_payload(),
        )
        if self.component_identity_sha256 != expected:
            raise ValueError("Gaussian component identity is stale")

    @classmethod
    def create(
        cls,
        *,
        receiver_t: int,
        state_source_j: int | None,
        model_source_j: int | None,
        mean: Tensor,
        precision_cholesky_snapshot: FrozenTensorSnapshot,
        log_probability: Tensor,
    ) -> "GaussianReceiverComponent":
        mean_snapshot = _capture_float64(mean, name="component mean", ndim=1)
        log_probability_snapshot = _capture_float64(
            log_probability,
            name="component log probability",
            shape=(),
        )
        precision_identity = _owned_hash(
            "vfe4.h6.shared-recognition-precision.v1",
            _snapshot_payload(precision_cholesky_snapshot),
        )
        payload = {
            "receiver_t": receiver_t,
            "state_source_j": state_source_j,
            "model_source_j": model_source_j,
            "mean": _snapshot_payload(mean_snapshot),
            "precision_identity_sha256": precision_identity,
            "log_probability": _snapshot_payload(log_probability_snapshot),
        }
        return cls(
            receiver_t,
            state_source_j,
            model_source_j,
            mean_snapshot,
            precision_cholesky_snapshot,
            log_probability_snapshot,
            precision_identity,
            _owned_hash("vfe4.h6.gaussian-receiver-component.v3", payload),
        )

    @property
    def mean(self) -> Tensor:
        return self.mean_snapshot.value()

    @property
    def precision_cholesky(self) -> Tensor:
        return self.precision_cholesky_snapshot.value()

    @property
    def log_probability(self) -> Tensor:
        return self.log_probability_snapshot.value()

    @property
    def probability(self) -> Tensor:
        return self.log_probability.exp()


@dataclass(frozen=True, eq=False)
class LanguageRecognitionTrajectory:
    conditioning: RecognitionConditioning
    family: RecognitionFamily
    block_sizes: tuple[int, ...]
    receiver_labels: tuple[int, ...]
    receiver_contexts: tuple[ReceiverRecognitionContext, ...]
    base_mean_snapshots: tuple[FrozenTensorSnapshot, ...]
    precision_cholesky_snapshot: FrozenTensorSnapshot
    state_source: SourceBankTrajectory
    model_source: SourceBankTrajectory
    receiver_components: tuple[tuple[GaussianReceiverComponent, ...], ...]
    recognition_store_state_sha256: str
    source_model_state_sha256: str
    trajectory_identity_sha256: str
    active_horizon_binding: H6ActiveHorizonV3
    horizon_scope_identity_sha256: str
    position_descriptor_schema: str = field(
        default=H6_RECOGNITION_POSITION_DESCRIPTOR_SCHEMA,
        init=False,
    )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema": "h6-language-recognition-trajectory-v3",
            "conditioning_mode": self.conditioning.mode,
            "observed_tokens_sha256": (
                self.conditioning.observed_tokens.raw_bytes_sha256
            ),
            "family": self.family,
            "block_sizes": self.block_sizes,
            "receiver_labels": self.receiver_labels,
            "position_descriptor_schema": self.position_descriptor_schema,
            "context_identities": tuple(
                context.context_identity_sha256 for context in self.receiver_contexts
            ),
            "base_means": tuple(
                _snapshot_payload(snapshot) for snapshot in self.base_mean_snapshots
            ),
            "shared_precision": _snapshot_payload(self.precision_cholesky_snapshot),
            "state_source_identity": self.state_source.bank_identity_sha256,
            "model_source_identity": self.model_source.bank_identity_sha256,
            "component_identities": tuple(
                tuple(component.component_identity_sha256 for component in components)
                for components in self.receiver_components
            ),
            "recognition_store_state_sha256": (self.recognition_store_state_sha256),
            "source_model_state_sha256": self.source_model_state_sha256,
        }

    def __post_init__(self) -> None:
        if type(self.conditioning) is not RecognitionConditioning:
            raise ValueError("trajectory conditioning type is invalid")
        self.conditioning.__post_init__()
        if self.family not in ("structured", "factorized"):
            raise ValueError("trajectory family is invalid")
        expected_labels = tuple(range(self.conditioning.horizon + 1))
        if self.receiver_labels != expected_labels:
            raise ValueError("receiver labels must be exactly 0..T")
        if (
            type(self.receiver_contexts) is not tuple
            or len(self.receiver_contexts) != len(expected_labels)
            or tuple(context.receiver_t for context in self.receiver_contexts)
            != expected_labels
            or any(
                context.conditioning_mode != self.conditioning.mode
                for context in self.receiver_contexts
            )
        ):
            raise ValueError("receiver-context inventory is incomplete")
        for context in self.receiver_contexts:
            context.__post_init__()
        contexts = self.contexts
        if type(self.base_mean_snapshots) is not tuple or len(
            self.base_mean_snapshots
        ) != len(expected_labels):
            raise ValueError("base means must cover receivers 0..T")
        base_means = self.base_means
        cholesky = self.shared_precision_cholesky
        if (
            contexts.ndim != 2
            or base_means.ndim != 2
            or contexts.shape[0] != len(expected_labels)
            or base_means.shape[0] != len(expected_labels)
            or cholesky.shape != (base_means.shape[1], base_means.shape[1])
            or contexts.device != base_means.device
            or contexts.device != cholesky.device
            or type(self.block_sizes) is not tuple
            or not self.block_sizes
            or any(type(size) is not int or size <= 0 for size in self.block_sizes)
            or sum(self.block_sizes) != base_means.shape[1]
        ):
            raise ValueError("trajectory Gaussian inventory is inconsistent")
        if self.family == "factorized":
            allowed = torch.zeros_like(cholesky, dtype=torch.bool)
            start = 0
            for size in self.block_sizes:
                stop = start + size
                allowed[start:stop, start:stop] = True
                start = stop
            if not bool(torch.all(cholesky.masked_select(~allowed) == 0.0)):
                raise ValueError(
                    "factorized trajectory precision must be block diagonal"
                )
        for bank_name, bank in (
            ("state", self.state_source),
            ("model", self.model_source),
        ):
            if (
                type(bank) not in (CategoricalSourceBank, AbsentSourceBank)
                or bank.bank != bank_name
            ):
                raise ValueError("trajectory source-bank typing is invalid")
            bank.__post_init__()
        if (
            type(self.receiver_components) is not tuple
            or len(self.receiver_components) != len(expected_labels)
            or any(not components for components in self.receiver_components)
        ):
            raise ValueError("components must cover every receiver")
        for receiver_t, components in enumerate(self.receiver_components):
            for component in components:
                if (
                    type(component) is not GaussianReceiverComponent
                    or component.receiver_t != receiver_t
                    or component.precision_cholesky_snapshot
                    is not self.precision_cholesky_snapshot
                ):
                    raise ValueError(
                        "components must share the trajectory precision snapshot"
                    )
                component.__post_init__()
        terminal_log_probabilities = torch.stack(
            tuple(
                component.log_probability for component in self.receiver_components[-1]
            )
        )
        if not bool(
            torch.allclose(
                torch.logsumexp(terminal_log_probabilities, dim=0),
                terminal_log_probabilities.new_zeros(()),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError("terminal source mixture must be normalized")
        _require_sha256(
            self.recognition_store_state_sha256,
            "recognition_store_state_sha256",
        )
        _require_sha256(
            self.source_model_state_sha256,
            "source_model_state_sha256",
        )
        _require_sha256(
            self.trajectory_identity_sha256,
            "trajectory_identity_sha256",
        )
        if (
            type(self.active_horizon_binding) is not H6ActiveHorizonV3
            or self.active_horizon_binding.active_horizon != self.conditioning.horizon
        ):
            raise ValueError(
                "trajectory active-horizon binding does not match conditioning"
            )
        self.active_horizon_binding.__post_init__()
        expected = _owned_hash(
            "vfe4.h6.language-recognition-trajectory.v3",
            self.canonical_payload(),
        )
        if self.trajectory_identity_sha256 != expected:
            raise ValueError("trajectory identity is stale")
        _require_sha256(
            self.horizon_scope_identity_sha256,
            "horizon_scope_identity_sha256",
        )
        if self.horizon_scope_identity_sha256 != _owned_hash(
            "vfe4.h6.language-recognition-horizon-scope.v3",
            {
                "trajectory_identity_sha256": (self.trajectory_identity_sha256),
                "active_horizon_evaluation_sha256": (
                    self.active_horizon_binding.evaluation_identity_sha256
                ),
            },
        ):
            raise ValueError("trajectory horizon-scope identity is stale")

    @classmethod
    def create(
        cls,
        *,
        conditioning: RecognitionConditioning,
        family: RecognitionFamily,
        block_sizes: tuple[int, ...],
        receiver_contexts: tuple[ReceiverRecognitionContext, ...],
        base_means: Tensor,
        precision_cholesky_snapshot: FrozenTensorSnapshot,
        state_source: SourceBankTrajectory,
        model_source: SourceBankTrajectory,
        receiver_components: tuple[tuple[GaussianReceiverComponent, ...], ...],
        recognition_store_state_sha256: str,
        source_model_state_sha256: str,
        active_horizon_binding: H6ActiveHorizonV3 | None = None,
    ) -> "LanguageRecognitionTrajectory":
        mean_snapshots = tuple(
            _capture_float64(mean, name="base receiver mean", ndim=1)
            for mean in base_means.unbind()
        )
        receiver_labels = tuple(range(conditioning.horizon + 1))
        payload = {
            "schema": "h6-language-recognition-trajectory-v3",
            "conditioning_mode": conditioning.mode,
            "observed_tokens_sha256": (conditioning.observed_tokens.raw_bytes_sha256),
            "family": family,
            "block_sizes": block_sizes,
            "receiver_labels": receiver_labels,
            "position_descriptor_schema": (H6_RECOGNITION_POSITION_DESCRIPTOR_SCHEMA),
            "context_identities": tuple(
                context.context_identity_sha256 for context in receiver_contexts
            ),
            "base_means": tuple(
                _snapshot_payload(snapshot) for snapshot in mean_snapshots
            ),
            "shared_precision": _snapshot_payload(precision_cholesky_snapshot),
            "state_source_identity": state_source.bank_identity_sha256,
            "model_source_identity": model_source.bank_identity_sha256,
            "component_identities": tuple(
                tuple(component.component_identity_sha256 for component in components)
                for components in receiver_components
            ),
            "recognition_store_state_sha256": (recognition_store_state_sha256),
            "source_model_state_sha256": source_model_state_sha256,
        }
        trajectory_identity = _owned_hash(
            "vfe4.h6.language-recognition-trajectory.v3", payload
        )
        binding = (
            H6ActiveHorizonV3.create(
                maximum_horizon=conditioning.horizon,
                active_horizon=conditioning.horizon,
            )
            if active_horizon_binding is None
            else active_horizon_binding
        )
        if type(binding) is not H6ActiveHorizonV3:
            raise ValueError(
                "active_horizon_binding must be an exact H6ActiveHorizonV3"
            )
        binding.__post_init__()
        return cls(
            conditioning=conditioning,
            family=family,
            block_sizes=block_sizes,
            receiver_labels=receiver_labels,
            receiver_contexts=receiver_contexts,
            base_mean_snapshots=mean_snapshots,
            precision_cholesky_snapshot=precision_cholesky_snapshot,
            state_source=state_source,
            model_source=model_source,
            receiver_components=receiver_components,
            recognition_store_state_sha256=recognition_store_state_sha256,
            source_model_state_sha256=source_model_state_sha256,
            trajectory_identity_sha256=trajectory_identity,
            active_horizon_binding=binding,
            horizon_scope_identity_sha256=_owned_hash(
                "vfe4.h6.language-recognition-horizon-scope.v3",
                {
                    "trajectory_identity_sha256": trajectory_identity,
                    "active_horizon_evaluation_sha256": (
                        binding.evaluation_identity_sha256
                    ),
                },
            ),
        )

    @property
    def contexts(self) -> Tensor:
        return torch.stack(tuple(context.context for context in self.receiver_contexts))

    @property
    def base_means(self) -> Tensor:
        return torch.stack(
            tuple(snapshot.value() for snapshot in self.base_mean_snapshots)
        )

    @property
    def shared_precision_cholesky(self) -> Tensor:
        return self.precision_cholesky_snapshot.value()

    @property
    def shared_precision_identity_sha256(self) -> str:
        return _owned_hash(
            "vfe4.h6.shared-recognition-precision.v1",
            _snapshot_payload(self.precision_cholesky_snapshot),
        )

    @property
    def terminal_components(self) -> tuple[GaussianReceiverComponent, ...]:
        return self.receiver_components[-1]


H6ActiveCategoricalSupportsV3 = tuple[tuple[int, ...], ...] | tuple[None]


@dataclass(frozen=True, slots=True)
class H6ActiveRecognitionTopologyV3:
    """Ragged recognition topology for one active-horizon example."""

    maximum_horizon: int
    active_horizon: int
    receiver_count: int
    state_categorical_enabled: bool
    model_categorical_enabled: bool
    state_categorical_supports: H6ActiveCategoricalSupportsV3
    model_categorical_supports: H6ActiveCategoricalSupportsV3
    receiver_components: tuple[tuple[int, tuple[str, ...]], ...]
    trajectory_identity_sha256: str
    active_horizon_evaluation_sha256: str
    topology_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "maximum_horizon": self.maximum_horizon,
            "active_horizon": self.active_horizon,
            "receiver_count": self.receiver_count,
            "state_categorical_enabled": self.state_categorical_enabled,
            "model_categorical_enabled": self.model_categorical_enabled,
            "state_categorical_supports": self.state_categorical_supports,
            "model_categorical_supports": self.model_categorical_supports,
            "receiver_components": self.receiver_components,
            "trajectory_identity_sha256": self.trajectory_identity_sha256,
            "active_horizon_evaluation_sha256": (self.active_horizon_evaluation_sha256),
        }

    def __post_init__(self) -> None:
        if (
            type(self.maximum_horizon) is not int
            or self.maximum_horizon <= 0
            or type(self.active_horizon) is not int
            or not 1 <= self.active_horizon <= self.maximum_horizon
            or type(self.receiver_count) is not int
            or self.receiver_count != self.active_horizon + 1
        ):
            raise ValueError("active recognition topology horizon is invalid")
        for name in (
            "trajectory_identity_sha256",
            "active_horizon_evaluation_sha256",
            "topology_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for bank, enabled, supports in (
            (
                "state",
                self.state_categorical_enabled,
                self.state_categorical_supports,
            ),
            (
                "model",
                self.model_categorical_enabled,
                self.model_categorical_supports,
            ),
        ):
            if type(enabled) is not bool:
                raise ValueError(
                    "active recognition categorical flags must be exact bools"
                )
            if enabled:
                if type(supports) is not tuple or len(supports) != self.active_horizon:
                    raise ValueError(
                        f"active {bank} supports must cover receivers 1..T"
                    )
                for receiver_t, support in enumerate(supports, start=1):
                    _require_causal_support(
                        support,
                        receiver_t=receiver_t,
                    )
            elif supports != (None,):
                raise ValueError(f"absent active {bank} support must be (None,)")
        if (
            type(self.receiver_components) is not tuple
            or len(self.receiver_components) != self.receiver_count
            or tuple(item[0] for item in self.receiver_components)
            != tuple(range(self.receiver_count))
        ):
            raise ValueError("active receiver components must cover exactly 0..T")
        for receiver_t, component_ids in self.receiver_components:
            if (
                type(receiver_t) is not int
                or type(component_ids) is not tuple
                or not component_ids
                or len(set(component_ids)) != len(component_ids)
            ):
                raise ValueError("active receiver component inventory is malformed")
            for component_id in component_ids:
                _require_sha256(component_id, "component identity")
        if self.topology_identity_sha256 != _owned_hash(
            "vfe4.h6.active-recognition-topology.v3",
            self.canonical_payload(),
        ):
            raise ValueError("active recognition topology identity is stale")


def project_active_recognition_topology_v3(
    *,
    trajectory: LanguageRecognitionTrajectory,
    active_horizon: H6ActiveHorizonV3,
) -> H6ActiveRecognitionTopologyV3:
    """Project one active trajectory into the engine's ragged topology."""

    if type(trajectory) is not LanguageRecognitionTrajectory:
        raise ValueError("trajectory must be an exact LanguageRecognitionTrajectory")
    if type(active_horizon) is not H6ActiveHorizonV3:
        raise ValueError("active_horizon must be an exact H6ActiveHorizonV3")
    trajectory.__post_init__()
    active_horizon.__post_init__()
    if trajectory.conditioning.horizon != active_horizon.active_horizon:
        raise ValueError("trajectory does not match the requested active horizon")
    if (
        trajectory.active_horizon_binding.maximum_horizon
        != active_horizon.maximum_horizon
    ):
        raise ValueError(
            "trajectory and projection binding have different maximum horizons"
        )
    if (
        trajectory.active_horizon_binding.evaluation_identity_sha256
        != active_horizon.evaluation_identity_sha256
    ):
        raise ValueError(
            "trajectory does not match the requested active-horizon identity"
        )
    state_enabled = type(trajectory.state_source) is CategoricalSourceBank
    model_enabled = type(trajectory.model_source) is CategoricalSourceBank
    state_supports: H6ActiveCategoricalSupportsV3 = (
        tuple(row.support for row in trajectory.state_source.rows)
        if state_enabled
        else (None,)
    )
    model_supports: H6ActiveCategoricalSupportsV3 = (
        tuple(row.support for row in trajectory.model_source.rows)
        if model_enabled
        else (None,)
    )
    components = tuple(
        (
            receiver_t,
            tuple(
                component.component_identity_sha256 for component in receiver_components
            ),
        )
        for receiver_t, receiver_components in enumerate(trajectory.receiver_components)
    )
    values = {
        "maximum_horizon": active_horizon.maximum_horizon,
        "active_horizon": active_horizon.active_horizon,
        "receiver_count": active_horizon.active_horizon + 1,
        "state_categorical_enabled": state_enabled,
        "model_categorical_enabled": model_enabled,
        "state_categorical_supports": state_supports,
        "model_categorical_supports": model_supports,
        "receiver_components": components,
        "trajectory_identity_sha256": trajectory.trajectory_identity_sha256,
        "active_horizon_evaluation_sha256": (active_horizon.evaluation_identity_sha256),
    }
    provisional = object.__new__(H6ActiveRecognitionTopologyV3)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return H6ActiveRecognitionTopologyV3(
        **values,
        topology_identity_sha256=_owned_hash(
            "vfe4.h6.active-recognition-topology.v3",
            provisional.canonical_payload(),
        ),
    )


def frozen_sinusoidal_receiver_positions(
    *,
    horizon: int,
    recognition_width: int,
    device: torch.device,
) -> Tensor:
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if type(recognition_width) is not int or recognition_width <= 0:
        raise ValueError("recognition_width must be a positive integer")
    receiver = torch.arange(horizon + 1, dtype=torch.float64, device=device).unsqueeze(
        1
    )
    coordinate = torch.arange(recognition_width, dtype=torch.float64, device=device)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * (2.0 * torch.div(coordinate, 2, rounding_mode="floor"))
        / recognition_width
    )
    angles = receiver * frequencies.unsqueeze(0)
    positions = torch.empty_like(angles)
    positions[:, 0::2] = torch.sin(angles[:, 0::2])
    positions[:, 1::2] = torch.cos(angles[:, 1::2])
    return positions.detach()


def build_receiver_contexts(
    *,
    conditioning: RecognitionConditioning,
    token_embeddings: Tensor,
) -> Tensor:
    if type(conditioning) is not RecognitionConditioning:
        raise ValueError("conditioning must be an exact RecognitionConditioning")
    conditioning.__post_init__()
    embeddings = _require_live_float64(
        token_embeddings, name="token_embeddings", ndim=2
    )
    if embeddings.shape[0] != conditioning.horizon or embeddings.shape[1] <= 0:
        raise ValueError(
            "token_embeddings must have shape (conditioning horizon, width)"
        )
    positions = frozen_sinusoidal_receiver_positions(
        horizon=conditioning.horizon,
        recognition_width=embeddings.shape[1],
        device=embeddings.device,
    )
    if conditioning.mode == "smoothing":
        return positions + embeddings.mean(dim=0).unsqueeze(0)
    denominators = torch.arange(
        1,
        conditioning.horizon + 1,
        dtype=torch.float64,
        device=embeddings.device,
    ).unsqueeze(1)
    causal_means = torch.cumsum(embeddings, dim=0) / denominators
    return torch.cat((positions[:1], positions[1:] + causal_means), dim=0)


def _source_residual(
    *,
    context: Tensor,
    parent_contexts: Tensor,
    receiver_t: int,
    support: tuple[int, ...],
    parameters: SourceRecognitionParameters,
) -> Tensor:
    indices = torch.tensor(support, dtype=torch.int64, device=context.device)
    differences = context.unsqueeze(0) - parent_contexts.index_select(0, indices)
    lags = torch.tensor(
        tuple(receiver_t - source_j for source_j in support),
        dtype=torch.float64,
        device=context.device,
    )
    return differences @ parameters.residual_vector + parameters.lag_scalar[
        0
    ] * torch.log1p(lags)


def _stopped_provider_state(
    provider: RecognitionPriorFeatureProvider,
) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, parameter in provider.named_parameters():
        state[name] = parameter.detach().clone(memory_format=torch.preserve_format)
    for name, buffer in provider.named_buffers():
        if name in state:
            raise ValueError("provider parameter and buffer names overlap")
        state[name] = buffer.detach().clone(memory_format=torch.preserve_format)
    return state


def _evaluate_prior_feature(
    *,
    provider: RecognitionPriorFeatureProvider,
    stopped_state: Mapping[str, Tensor],
    bank: SourceBankName,
    causal_prefix: CausalPrefix,
    earlier_recognition_means: Tensor,
) -> RecognitionPriorFeature:
    internally_stopped = getattr(
        provider,
        "uses_internal_stopped_state_v3",
        False,
    )
    if internally_stopped is True:
        result = provider(
            bank=bank,
            causal_prefix=causal_prefix,
            earlier_recognition_means=earlier_recognition_means,
        )
    else:
        result = torch.func.functional_call(
            provider,
            dict(stopped_state),
            (),
            {
                "bank": bank,
                "causal_prefix": causal_prefix,
                "earlier_recognition_means": earlier_recognition_means,
            },
            strict=True,
        )
    if type(result) is not RecognitionPriorFeature:
        raise ValueError("prior feature provider must return RecognitionPriorFeature")
    result.__post_init__()
    if result.bank != bank:
        raise ValueError("prior feature provider changed the requested bank")
    if result.causal_prefix_sha256 != causal_prefix.prefix_sha256:
        raise ValueError("prior feature provider did not bind the exact CausalPrefix")
    original_parameters = tuple(
        parameter for parameter in provider.parameters() if parameter.requires_grad
    )
    if result.log_prior_features.requires_grad and original_parameters:
        gradients = torch.autograd.grad(
            result.log_prior_features.sum(),
            original_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        if any(gradient is not None for gradient in gradients):
            raise ValueError("prior feature retained a source-model gradient path")
    return result


def _build_source_bank(
    *,
    bank: SourceBankName,
    conditioning: RecognitionConditioning,
    vocabulary: VocabularyIdentity,
    contexts: Tensor,
    base_means: Tensor,
    parameters: SourceRecognitionParameters,
    provider: RecognitionPriorFeatureProvider,
    stopped_state: Mapping[str, Tensor],
    source_model_state_sha256: str,
) -> CategoricalSourceBank:
    observed_tokens = (
        conditioning.observed_tokens.value()
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
    )
    rows: list[CategoricalSourceRow] = []
    for receiver_t in range(1, conditioning.horizon + 1):
        causal_prefix = CausalPrefix.create(
            receiver_t=receiver_t,
            vocabulary=vocabulary,
            token_ids=observed_tokens[: receiver_t - 1].contiguous(),
        )
        feature = _evaluate_prior_feature(
            provider=provider,
            stopped_state=stopped_state,
            bank=bank,
            causal_prefix=causal_prefix,
            earlier_recognition_means=base_means[:receiver_t],
        )
        support = _require_causal_support(feature.support, receiver_t=receiver_t)
        baseline = _require_live_float64(
            feature.log_prior_features,
            name="mean-evaluated source prior",
            shape=(len(support),),
            finite=False,
        )
        if baseline.device != contexts.device:
            raise ValueError(
                "mean-evaluated source prior must share the recognition device"
            )
        if not bool(torch.isfinite(baseline).all()):
            raise ValueError(
                "every declared source parent must have positive prior mass"
            )
        residual = _source_residual(
            context=contexts[receiver_t],
            parent_contexts=contexts,
            receiver_t=receiver_t,
            support=support,
            parameters=parameters,
        )
        log_probabilities = F.log_softmax(baseline + residual, dim=0)
        entropy = _categorical_entropy(log_probabilities)
        rows.append(
            CategoricalSourceRow.create(
                bank=bank,
                receiver_t=receiver_t,
                support=support,
                causal_prefix_sha256=causal_prefix.prefix_sha256,
                source_model_state_sha256=source_model_state_sha256,
                log_prior_baseline=baseline,
                residual_scores=residual,
                log_probabilities=log_probabilities,
                entropy=entropy,
            )
        )
    return CategoricalSourceBank.create(bank=bank, rows=tuple(rows))


def _categorical_entropy(log_probabilities: Tensor) -> Tensor:
    """Evaluate the categorical entropy used only by complete-ELBO rows."""

    checked = _require_live_float64(
        log_probabilities,
        name="categorical log probabilities",
        ndim=1,
    )
    probabilities = checked.exp()
    return -(probabilities * checked).sum()


def _build_source_prior_free_bank(
    *,
    bank: SourceBankName,
    conditioning: RecognitionConditioning,
    vocabulary: VocabularyIdentity,
    contexts: Tensor,
    base_means: Tensor,
    parameters: SourceRecognitionParameters,
    source_model_state_sha256: str,
) -> CategoricalSourceBank:
    """Build the recognition-only source law without a generative prior call."""

    observed_tokens = (
        conditioning.observed_tokens.value()
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
    )
    rows: list[CategoricalSourceRow] = []
    for receiver_t in range(1, conditioning.horizon + 1):
        causal_prefix = CausalPrefix.create(
            receiver_t=receiver_t,
            vocabulary=vocabulary,
            token_ids=observed_tokens[: receiver_t - 1].contiguous(),
        )
        support = tuple(range(receiver_t))
        baseline = base_means.new_zeros((receiver_t,))
        residual = _source_residual(
            context=contexts[receiver_t],
            parent_contexts=contexts,
            receiver_t=receiver_t,
            support=support,
            parameters=parameters,
        )
        log_probabilities = F.log_softmax(residual, dim=0)
        rows.append(
            CategoricalSourceRow.create(
                bank=bank,
                receiver_t=receiver_t,
                support=support,
                causal_prefix_sha256=causal_prefix.prefix_sha256,
                source_model_state_sha256=source_model_state_sha256,
                log_prior_baseline=baseline,
                residual_scores=residual,
                log_probabilities=log_probabilities,
                entropy=log_probabilities.new_zeros(()),
            )
        )
    return CategoricalSourceBank.create(bank=bank, rows=tuple(rows))


def _source_options(
    bank: SourceBankTrajectory,
) -> tuple[tuple[int | None, Tensor], ...]:
    if type(bank) is AbsentSourceBank:
        return ((None, bank.log_probability),)
    terminal_row = bank.rows[-1]
    return tuple(
        (source_j, log_probability)
        for source_j, log_probability in zip(
            terminal_row.support,
            terminal_row.log_probabilities.unbind(),
            strict=True,
        )
    )


def _source_shift(
    *,
    source_j: int,
    receiver_t: int,
    contexts: Tensor,
    parameters: SourceRecognitionParameters,
) -> Tensor:
    scalar = _source_residual(
        context=contexts[receiver_t],
        parent_contexts=contexts,
        receiver_t=receiver_t,
        support=(source_j,),
        parameters=parameters,
    )[0]
    return parameters.shift_vector * scalar


def build_language_recognition_trajectory(
    *,
    conditioning: RecognitionConditioning,
    vocabulary: VocabularyIdentity,
    family: RecognitionFamily,
    block_sizes: tuple[int, ...],
    contexts: Tensor,
    base_means: Tensor,
    shared_precision_cholesky: Tensor,
    latent_width: int,
    channel_count: Literal[1, 2],
    source_parameters: Mapping[SourceBankName, SourceRecognitionParameters],
    prior_feature_provider: RecognitionPriorFeatureProvider,
    recognition_store_state_sha256: str,
    active_horizon_binding: H6ActiveHorizonV3 | None = None,
    source_prior_free: bool = False,
) -> LanguageRecognitionTrajectory:
    """Build v3 without exposing mutable aliases or the legacy law API."""

    if type(conditioning) is not RecognitionConditioning:
        raise ValueError("conditioning must be an exact RecognitionConditioning")
    conditioning.__post_init__()
    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("vocabulary must be an exact VocabularyIdentity")
    vocabulary.__post_init__()
    if family not in ("structured", "factorized"):
        raise ValueError("recognition family is invalid")
    if channel_count not in (1, 2):
        raise ValueError("channel_count must be one or two")
    if type(source_prior_free) is not bool:
        raise ValueError("source_prior_free must be an exact boolean")
    if (
        type(latent_width) is not int
        or latent_width <= 0
        or block_sizes != (latent_width,) * channel_count
    ):
        raise ValueError("recognition block structure is inconsistent")
    contexts = _require_live_float64(contexts, name="contexts", ndim=2)
    if contexts.shape[0] != conditioning.horizon + 1:
        raise ValueError("contexts must cover receivers 0..T")
    base_means = _require_live_float64(
        base_means,
        name="base_means",
        shape=(conditioning.horizon + 1, channel_count * latent_width),
    )
    shared_precision_cholesky = _require_live_float64(
        shared_precision_cholesky,
        name="shared_precision_cholesky",
        shape=(channel_count * latent_width, channel_count * latent_width),
    )
    if (
        contexts.device != base_means.device
        or contexts.device != shared_precision_cholesky.device
    ):
        raise ValueError("trajectory tensors must share one device")
    _require_sha256(
        recognition_store_state_sha256,
        "recognition_store_state_sha256",
    )
    if not isinstance(prior_feature_provider, RecognitionPriorFeatureProvider):
        raise ValueError(
            "prior_feature_provider must be a RecognitionPriorFeatureProvider"
        )
    expected_banks = frozenset(source_parameters)
    if any(bank not in ("state", "model") for bank in expected_banks) or (
        "model" in expected_banks and channel_count != 2
    ):
        raise ValueError("source banks do not match Gaussian channels")
    for bank, parameters in source_parameters.items():
        if (
            type(parameters) is not SourceRecognitionParameters
            or parameters.bank != bank
            or parameters.residual_vector.shape != (contexts.shape[1],)
            or parameters.shift_vector.shape != (latent_width,)
        ):
            raise ValueError("source parameter dimensions are inconsistent")
        parameters.__post_init__()

    source_model = prior_feature_provider.source_model
    if not isinstance(source_model, nn.Module):
        raise ValueError("prior feature provider source_model must be an nn.Module")
    cached_state = getattr(
        prior_feature_provider,
        "stopped_state_v3",
        None,
    )
    cached_sha256 = getattr(
        prior_feature_provider,
        "stopped_state_sha256_v3",
        None,
    )
    assert_intact = getattr(
        prior_feature_provider,
        "assert_stopped_source_intact_v3",
        None,
    )
    if (
        callable(cached_state)
        and type(cached_sha256) is str
        and callable(assert_intact)
    ):
        stopped_state = cached_state()
        source_model_state_sha256 = cached_sha256
        assert_intact()
    else:
        source_model_state_sha256 = canonical_model_state_sha256(source_model)
        stopped_state = _stopped_provider_state(prior_feature_provider)
    receiver_contexts = tuple(
        ReceiverRecognitionContext.create(
            receiver_t=receiver_t,
            conditioning_mode=conditioning.mode,
            context=context,
        )
        for receiver_t, context in enumerate(contexts.unbind())
    )
    built_banks: dict[SourceBankName, SourceBankTrajectory] = {}
    for bank in ("state", "model"):
        if bank not in expected_banks:
            built_banks[bank] = AbsentSourceBank.create(bank=bank, reference=base_means)
        elif source_prior_free:
            built_banks[bank] = _build_source_prior_free_bank(
                bank=bank,
                conditioning=conditioning,
                vocabulary=vocabulary,
                contexts=contexts,
                base_means=base_means,
                parameters=source_parameters[bank],
                source_model_state_sha256=source_model_state_sha256,
            )
        else:
            built_banks[bank] = _build_source_bank(
                bank=bank,
                conditioning=conditioning,
                vocabulary=vocabulary,
                contexts=contexts,
                base_means=base_means,
                parameters=source_parameters[bank],
                provider=prior_feature_provider,
                stopped_state=stopped_state,
                source_model_state_sha256=source_model_state_sha256,
            )
    if callable(assert_intact):
        assert_intact()
    elif canonical_model_state_sha256(source_model) != source_model_state_sha256:
        raise ValueError("prior feature provider mutated its source-model state")

    precision_snapshot = _capture_float64(
        shared_precision_cholesky,
        name="shared precision Cholesky",
        shape=(channel_count * latent_width, channel_count * latent_width),
    )
    receiver_components: list[tuple[GaussianReceiverComponent, ...]] = []
    zero_log_probability = base_means.new_zeros(())
    for receiver_t in range(conditioning.horizon):
        receiver_components.append(
            (
                GaussianReceiverComponent.create(
                    receiver_t=receiver_t,
                    state_source_j=None,
                    model_source_j=None,
                    mean=base_means[receiver_t],
                    precision_cholesky_snapshot=precision_snapshot,
                    log_probability=zero_log_probability,
                ),
            )
        )

    terminal_t = conditioning.horizon
    terminal_components: list[GaussianReceiverComponent] = []
    for state_source_j, state_log_probability in _source_options(built_banks["state"]):
        for model_source_j, model_log_probability in _source_options(
            built_banks["model"]
        ):
            mean = base_means[terminal_t]
            if state_source_j is not None:
                state_shift = _source_shift(
                    source_j=state_source_j,
                    receiver_t=terminal_t,
                    contexts=contexts,
                    parameters=source_parameters["state"],
                )
                mean = torch.cat(
                    (
                        mean[:latent_width] + state_shift,
                        mean[latent_width:],
                    )
                )
            if model_source_j is not None:
                model_shift = _source_shift(
                    source_j=model_source_j,
                    receiver_t=terminal_t,
                    contexts=contexts,
                    parameters=source_parameters["model"],
                )
                mean = torch.cat(
                    (
                        mean[:latent_width],
                        mean[latent_width : 2 * latent_width] + model_shift,
                    )
                )
            terminal_components.append(
                GaussianReceiverComponent.create(
                    receiver_t=terminal_t,
                    state_source_j=state_source_j,
                    model_source_j=model_source_j,
                    mean=mean,
                    precision_cholesky_snapshot=precision_snapshot,
                    log_probability=(state_log_probability + model_log_probability),
                )
            )
    receiver_components.append(tuple(terminal_components))
    return LanguageRecognitionTrajectory.create(
        conditioning=conditioning,
        family=family,
        block_sizes=block_sizes,
        receiver_contexts=receiver_contexts,
        base_means=base_means,
        precision_cholesky_snapshot=precision_snapshot,
        state_source=built_banks["state"],
        model_source=built_banks["model"],
        receiver_components=tuple(receiver_components),
        recognition_store_state_sha256=recognition_store_state_sha256,
        source_model_state_sha256=source_model_state_sha256,
        active_horizon_binding=active_horizon_binding,
    )


__all__ = [
    "AbsentSourceBank",
    "CategoricalSourceBank",
    "CategoricalSourceRow",
    "GaussianReceiverComponent",
    "H6ActiveHorizonV3",
    "H6ActiveRecognitionTopologyV3",
    "H6ActiveCategoricalSupportsV3",
    "H6_RECOGNITION_POSITION_DESCRIPTOR_SCHEMA",
    "LanguageRecognitionTrajectory",
    "ReceiverRecognitionContext",
    "RecognitionFamily",
    "RecognitionPriorFeature",
    "RecognitionPriorFeatureProvider",
    "SourceBankName",
    "SourceRecognitionParameters",
    "build_language_recognition_trajectory",
    "build_receiver_contexts",
    "frozen_sinusoidal_receiver_positions",
    "project_active_recognition_topology_v3",
]
