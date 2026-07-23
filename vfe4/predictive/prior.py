"""Public target-blind prior-prediction boundary and immutable result records."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import (
    FrozenTensorSnapshot,
    VocabularyIdentity,
    canonical_json_bytes,
)

from .cache import PrefixCache
from .identities import vocabulary_identity_sha256
from .proposal import CounterConsumption, EstimatorStream


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
class EstimatorRecord:
    estimator_semantic_sha256: str
    estimator_artifact_bytes_sha256: str
    estimator_stream_sha256: str
    stream_seed: int
    prefix_sha256: str
    counter_trace_sha256: str
    counter_draw_count: int
    cumulative_log_normalizer: float
    record_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "estimator_semantic_sha256",
            "estimator_artifact_bytes_sha256",
            "estimator_stream_sha256",
            "prefix_sha256",
            "counter_trace_sha256",
            "record_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.stream_seed) is not int
            or not 0 <= self.stream_seed < 2**64
        ):
            raise ValueError("stream seed must be an unsigned 64-bit integer")
        if (
            type(self.counter_draw_count) is not int
            or self.counter_draw_count < 0
        ):
            raise ValueError("counter draw count must be nonnegative")
        if not math.isfinite(self.cumulative_log_normalizer):
            raise ValueError("cumulative log normalizer must be finite")
        expected = _owned_hash(
            "vfe4.h6.estimator-record.v1", self._payload()
        )
        if self.record_sha256 != expected:
            raise ValueError("estimator record identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "estimator_semantic_sha256": self.estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                self.estimator_artifact_bytes_sha256
            ),
            "estimator_stream_sha256": self.estimator_stream_sha256,
            "stream_seed": self.stream_seed,
            "prefix_sha256": self.prefix_sha256,
            "counter_trace_sha256": self.counter_trace_sha256,
            "counter_draw_count": self.counter_draw_count,
            "cumulative_log_normalizer": self.cumulative_log_normalizer,
        }

    @classmethod
    def from_cache(
        cls,
        *,
        stream: EstimatorStream,
        cache: PrefixCache,
    ) -> "EstimatorRecord":
        if type(stream) is not EstimatorStream:
            raise ValueError("stream must be an exact EstimatorStream")
        stream.__post_init__()
        if type(cache) is not PrefixCache:
            raise ValueError("cache must be an exact PrefixCache")
        cache.__post_init__()
        records: tuple[CounterConsumption, ...] = cache.counter_consumption
        trace_sha256 = _owned_hash(
            "vfe4.h6.counter-trace.v1",
            tuple(record.consumption_sha256 for record in records),
        )
        draw_count = sum(
            record.particle_count * record.draws_per_particle
            for record in records
        )
        payload = {
            "estimator_semantic_sha256": (
                stream.estimator_semantic_sha256
            ),
            "estimator_artifact_bytes_sha256": (
                stream.estimator_artifact_bytes_sha256
            ),
            "estimator_stream_sha256": stream.stream_sha256,
            "stream_seed": stream.stream_seed,
            "prefix_sha256": cache.key.prefix_sha256,
            "counter_trace_sha256": trace_sha256,
            "counter_draw_count": draw_count,
            "cumulative_log_normalizer": (
                cache.cumulative_log_normalizer
            ),
        }
        return cls(
            **payload,
            record_sha256=_owned_hash(
                "vfe4.h6.estimator-record.v1", payload
            ),
        )


@dataclass(frozen=True)
class PriorPrediction:
    vocabulary: VocabularyIdentity
    log_probs: FrozenTensorSnapshot
    cache: PrefixCache
    estimator_record: EstimatorRecord
    prediction_sha256: str

    def __post_init__(self) -> None:
        if type(self.vocabulary) is not VocabularyIdentity:
            raise ValueError(
                "prior prediction requires an exact VocabularyIdentity"
            )
        self.vocabulary.__post_init__()
        if type(self.log_probs) is not FrozenTensorSnapshot:
            raise ValueError("prior log probabilities must be frozen")
        self.log_probs.assert_intact()
        if (
            self.log_probs.dtype != "float64"
            or self.log_probs.shape != (self.vocabulary.size,)
        ):
            raise ValueError(
                "prior log probabilities must have exact vocabulary shape"
            )
        values = self.log_probs.value()
        if (
            not bool(torch.isfinite(values).all())
            or not math.isclose(
                float(torch.logsumexp(values, dim=0).item()),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-13,
            )
        ):
            raise ValueError(
                "prior log probabilities must be finite and normalized"
            )
        if type(self.cache) is not PrefixCache:
            raise ValueError("prior prediction requires an exact PrefixCache")
        self.cache.__post_init__()
        if (
            self.cache.key.vocabulary_sha256
            != vocabulary_identity_sha256(self.vocabulary)
            or self.cache.pending.prediction_log_probs.raw_bytes_sha256
            != self.log_probs.raw_bytes_sha256
        ):
            raise ValueError(
                "prior prediction does not match its vocabulary/cache state"
            )
        if type(self.estimator_record) is not EstimatorRecord:
            raise ValueError(
                "prior prediction requires an exact EstimatorRecord"
            )
        self.estimator_record.__post_init__()
        if (
            self.estimator_record.prefix_sha256
            != self.cache.key.prefix_sha256
            or self.estimator_record.estimator_semantic_sha256
            != self.cache.key.estimator_semantic_sha256
            or self.estimator_record.estimator_artifact_bytes_sha256
            != self.cache.key.estimator_artifact_bytes_sha256
            or self.estimator_record.estimator_stream_sha256
            != self.cache.key.estimator_stream_sha256
        ):
            raise ValueError(
                "prior prediction estimator record does not match its cache"
            )
        expected = _owned_hash(
            "vfe4.h6.prior-prediction.v1", self._payload()
        )
        if self.prediction_sha256 != expected:
            raise ValueError("prior prediction identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "vocabulary_sha256": vocabulary_identity_sha256(
                self.vocabulary
            ),
            "log_probs": _snapshot_payload(self.log_probs),
            "cache_sha256": self.cache.cache_sha256,
            "estimator_record_sha256": self.estimator_record.record_sha256,
        }

    @classmethod
    def create(
        cls,
        *,
        vocabulary: VocabularyIdentity,
        log_probs: torch.Tensor,
        cache: PrefixCache,
        estimator_record: EstimatorRecord,
    ) -> "PriorPrediction":
        snapshot = FrozenTensorSnapshot.capture(log_probs)
        payload = {
            "vocabulary_sha256": vocabulary_identity_sha256(vocabulary),
            "log_probs": _snapshot_payload(snapshot),
            "cache_sha256": cache.cache_sha256,
            "estimator_record_sha256": estimator_record.record_sha256,
        }
        return cls(
            vocabulary,
            snapshot,
            cache,
            estimator_record,
            _owned_hash("vfe4.h6.prior-prediction.v1", payload),
        )


@runtime_checkable
class PriorPredictor(Protocol):
    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction: ...


__all__ = ["EstimatorRecord", "PriorPrediction", "PriorPredictor"]
