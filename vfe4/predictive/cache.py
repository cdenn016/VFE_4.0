"""Immutable identity-bound cache records for H6 weighted prior prediction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from vfe4.data.windows import CausalPrefix
from vfe4.types.h6 import FrozenTensorSnapshot, canonical_json_bytes

from .proposal import (
    CounterConsumption,
    ProposalPopulation,
    ProposalStep,
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
class PrefixCacheKey:
    prefix_tokens: tuple[int, ...]
    prefix_sha256: str
    vocabulary_sha256: str
    predictor_config_sha256: str
    model_family_sha256: str
    model_state_sha256: str
    proposal_identity_sha256: str
    estimator_semantic_sha256: str
    estimator_artifact_bytes_sha256: str
    estimator_stream_sha256: str
    data_safety_sha256: str
    key_sha256: str

    _SHA_FIELDS = (
        "prefix_sha256",
        "vocabulary_sha256",
        "predictor_config_sha256",
        "model_family_sha256",
        "model_state_sha256",
        "proposal_identity_sha256",
        "estimator_semantic_sha256",
        "estimator_artifact_bytes_sha256",
        "estimator_stream_sha256",
        "data_safety_sha256",
    )

    def __post_init__(self) -> None:
        if (
            type(self.prefix_tokens) is not tuple
            or any(
                type(token) is not int or not 0 <= token <= 65535
                for token in self.prefix_tokens
            )
        ):
            raise ValueError(
                "cache prefix tokens must be immutable canonical uint16 values"
            )
        for name in (*self._SHA_FIELDS, "key_sha256"):
            _require_sha256(getattr(self, name), name)
        expected = _owned_hash(
            "vfe4.h6.prefix-cache-key.v1", self._payload()
        )
        if self.key_sha256 != expected:
            raise ValueError("prefix cache key identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "prefix_tokens": self.prefix_tokens,
            **{name: getattr(self, name) for name in self._SHA_FIELDS},
        }

    @classmethod
    def create(
        cls,
        *,
        prefix: CausalPrefix,
        vocabulary_sha256: str,
        predictor_config_sha256: str,
        model_family_sha256: str,
        model_state_sha256: str,
        proposal_identity_sha256: str,
        estimator_semantic_sha256: str,
        estimator_artifact_bytes_sha256: str,
        estimator_stream_sha256: str,
        data_safety_sha256: str,
    ) -> "PrefixCacheKey":
        if type(prefix) is not CausalPrefix:
            raise ValueError("prefix must be an exact target-free CausalPrefix")
        prefix.__post_init__()
        values = {
            "prefix_tokens": tuple(int(value) for value in prefix.token_ids.tolist()),
            "prefix_sha256": prefix.prefix_sha256,
            "vocabulary_sha256": vocabulary_sha256,
            "predictor_config_sha256": predictor_config_sha256,
            "model_family_sha256": model_family_sha256,
            "model_state_sha256": model_state_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "estimator_semantic_sha256": estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                estimator_artifact_bytes_sha256
            ),
            "estimator_stream_sha256": estimator_stream_sha256,
            "data_safety_sha256": data_safety_sha256,
        }
        return cls(
            **values,
            key_sha256=_owned_hash(
                "vfe4.h6.prefix-cache-key.v1", values
            ),
        )


@dataclass(frozen=True)
class PendingPrediction:
    prefix_sha256: str
    proposal_identity_sha256: str
    proposed_population: ProposalPopulation
    emission_log_probs: FrozenTensorSnapshot
    parent_log_weights: FrozenTensorSnapshot
    prediction_log_probs: FrozenTensorSnapshot
    counter_consumption: tuple[CounterConsumption, ...]
    pending_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.prefix_sha256, "prefix_sha256")
        _require_sha256(
            self.proposal_identity_sha256, "proposal_identity_sha256"
        )
        if type(self.proposed_population) is not ProposalPopulation:
            raise ValueError(
                "pending prediction requires an exact proposal population"
            )
        self.proposed_population.__post_init__()
        for name in (
            "emission_log_probs",
            "parent_log_weights",
            "prediction_log_probs",
        ):
            snapshot = getattr(self, name)
            if type(snapshot) is not FrozenTensorSnapshot:
                raise ValueError(f"{name} must be a FrozenTensorSnapshot")
            snapshot.assert_intact()
        particle_count = self.proposed_population.particle_count
        if (
            self.emission_log_probs.dtype != "float64"
            or len(self.emission_log_probs.shape) != 2
            or self.emission_log_probs.shape[0] != particle_count
            or self.parent_log_weights.dtype != "float64"
            or self.parent_log_weights.shape != (particle_count,)
            or self.prediction_log_probs.dtype != "float64"
            or self.prediction_log_probs.shape
            != (self.emission_log_probs.shape[1],)
        ):
            raise ValueError("pending prediction tensor shapes do not align")
        parent = self.parent_log_weights.value()
        emission = self.emission_log_probs.value()
        prediction = self.prediction_log_probs.value()
        if not math.isclose(
            float(torch.logsumexp(parent, dim=0).item()),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-13,
        ):
            raise ValueError("pending parent weights must be normalized")
        expected_prediction = torch.logsumexp(
            parent[:, None] + emission, dim=0
        )
        if not torch.equal(prediction, expected_prediction):
            raise ValueError(
                "pending prediction must be the exact carried-weight mixture"
            )
        if (
            type(self.counter_consumption) is not tuple
            or any(
                type(record) is not CounterConsumption
                for record in self.counter_consumption
            )
        ):
            raise ValueError(
                "pending counter consumption must be immutable exact records"
            )
        for record in self.counter_consumption:
            record.__post_init__()
        expected = _owned_hash(
            "vfe4.h6.pending-prediction.v1", self._payload()
        )
        if self.pending_sha256 != expected:
            raise ValueError("pending prediction identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "prefix_sha256": self.prefix_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "proposed_population_sha256": (
                self.proposed_population.population_sha256
            ),
            "emission_log_probs": _snapshot_payload(
                self.emission_log_probs
            ),
            "parent_log_weights": _snapshot_payload(
                self.parent_log_weights
            ),
            "prediction_log_probs": _snapshot_payload(
                self.prediction_log_probs
            ),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in self.counter_consumption
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        prefix_sha256: str,
        step: ProposalStep,
        parent_log_weights: Tensor,
        prediction_log_probs: Tensor,
    ) -> "PendingPrediction":
        if type(step) is not ProposalStep:
            raise ValueError("step must be an exact ProposalStep")
        step.__post_init__()
        emission = FrozenTensorSnapshot.capture(
            step.emission_log_probs.value()
        )
        parent = FrozenTensorSnapshot.capture(parent_log_weights)
        prediction = FrozenTensorSnapshot.capture(prediction_log_probs)
        provisional = {
            "prefix_sha256": prefix_sha256,
            "proposal_identity_sha256": step.proposal_identity_sha256,
            "proposed_population_sha256": step.population.population_sha256,
            "emission_log_probs": _snapshot_payload(emission),
            "parent_log_weights": _snapshot_payload(parent),
            "prediction_log_probs": _snapshot_payload(prediction),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in step.counter_consumption
            ),
        }
        return cls(
            prefix_sha256,
            step.proposal_identity_sha256,
            step.population,
            emission,
            parent,
            prediction,
            step.counter_consumption,
            _owned_hash("vfe4.h6.pending-prediction.v1", provisional),
        )


@dataclass(frozen=True)
class MarginalPendingPrediction:
    """Pending prediction with real particles and no cached ``[N,V]`` slab."""

    prefix_sha256: str
    proposal_identity_sha256: str
    proposed_population: ProposalPopulation
    parent_log_weights: FrozenTensorSnapshot
    prediction_log_probs: FrozenTensorSnapshot
    counter_consumption: tuple[CounterConsumption, ...]
    marginalization_evidence_sha256: str
    pending_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.prefix_sha256, "prefix_sha256")
        _require_sha256(
            self.proposal_identity_sha256,
            "proposal_identity_sha256",
        )
        if type(self.proposed_population) is not ProposalPopulation:
            raise ValueError(
                "marginal pending prediction requires an exact population"
            )
        self.proposed_population.__post_init__()
        for name in ("parent_log_weights", "prediction_log_probs"):
            snapshot = getattr(self, name)
            if type(snapshot) is not FrozenTensorSnapshot:
                raise ValueError(f"{name} must be a FrozenTensorSnapshot")
            snapshot.assert_intact()
        if (
            self.parent_log_weights.dtype != "float64"
            or self.parent_log_weights.shape
            != (self.proposed_population.particle_count,)
            or self.prediction_log_probs.dtype != "float64"
            or len(self.prediction_log_probs.shape) != 1
            or self.prediction_log_probs.shape[0] <= 0
        ):
            raise ValueError(
                "marginal pending prediction tensor shapes do not align"
            )
        parent = self.parent_log_weights.value()
        prediction = self.prediction_log_probs.value()
        if (
            not math.isclose(
                float(torch.logsumexp(parent, dim=0).item()),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-13,
            )
            or not bool(torch.isfinite(prediction).all())
            or not math.isclose(
                float(torch.logsumexp(prediction, dim=0).item()),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-13,
            )
        ):
            raise ValueError(
                "marginal pending weights/prediction must be normalized"
            )
        if (
            type(self.counter_consumption) is not tuple
            or any(
                type(record) is not CounterConsumption
                for record in self.counter_consumption
            )
        ):
            raise ValueError(
                "marginal pending counters must be immutable exact records"
            )
        for record in self.counter_consumption:
            record.__post_init__()
        expected_evidence = _owned_hash(
            "vfe4.wt103.streamed-marginalization-evidence.v1",
            self._evidence_payload(),
        )
        _require_sha256(
            self.marginalization_evidence_sha256,
            "marginalization_evidence_sha256",
        )
        if self.marginalization_evidence_sha256 != expected_evidence:
            raise ValueError(
                "streamed marginalization evidence identity is stale"
            )
        expected = _owned_hash(
            "vfe4.wt103.marginal-pending-prediction.v1",
            self._payload(),
        )
        _require_sha256(self.pending_sha256, "pending_sha256")
        if self.pending_sha256 != expected:
            raise ValueError("marginal pending prediction identity is stale")

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "algorithm": "particle_chunk_logsumexp_without_cached_N_by_V",
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "proposed_population_sha256": (
                self.proposed_population.population_sha256
            ),
            "parent_log_weights": _snapshot_payload(
                self.parent_log_weights
            ),
            "prediction_log_probs": _snapshot_payload(
                self.prediction_log_probs
            ),
        }

    def _payload(self) -> dict[str, object]:
        return {
            "prefix_sha256": self.prefix_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "proposed_population_sha256": (
                self.proposed_population.population_sha256
            ),
            "parent_log_weights": _snapshot_payload(
                self.parent_log_weights
            ),
            "prediction_log_probs": _snapshot_payload(
                self.prediction_log_probs
            ),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in self.counter_consumption
            ),
            "marginalization_evidence_sha256": (
                self.marginalization_evidence_sha256
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        prefix_sha256: str,
        proposal_identity_sha256: str,
        proposed_population: ProposalPopulation,
        parent_log_weights: Tensor,
        prediction_log_probs: Tensor,
        counter_consumption: tuple[CounterConsumption, ...],
    ) -> "MarginalPendingPrediction":
        parent = FrozenTensorSnapshot.capture(parent_log_weights)
        prediction = FrozenTensorSnapshot.capture(prediction_log_probs)
        evidence_payload = {
            "algorithm": "particle_chunk_logsumexp_without_cached_N_by_V",
            "proposal_identity_sha256": proposal_identity_sha256,
            "proposed_population_sha256": (
                proposed_population.population_sha256
            ),
            "parent_log_weights": _snapshot_payload(parent),
            "prediction_log_probs": _snapshot_payload(prediction),
        }
        evidence_sha256 = _owned_hash(
            "vfe4.wt103.streamed-marginalization-evidence.v1",
            evidence_payload,
        )
        payload = {
            "prefix_sha256": prefix_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "proposed_population_sha256": (
                proposed_population.population_sha256
            ),
            "parent_log_weights": _snapshot_payload(parent),
            "prediction_log_probs": _snapshot_payload(prediction),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in counter_consumption
            ),
            "marginalization_evidence_sha256": evidence_sha256,
        }
        return cls(
            prefix_sha256=prefix_sha256,
            proposal_identity_sha256=proposal_identity_sha256,
            proposed_population=proposed_population,
            parent_log_weights=parent,
            prediction_log_probs=prediction,
            counter_consumption=counter_consumption,
            marginalization_evidence_sha256=evidence_sha256,
            pending_sha256=_owned_hash(
                "vfe4.wt103.marginal-pending-prediction.v1",
                payload,
            ),
        )


@dataclass(frozen=True)
class AssimilationRecord:
    position: int
    observed_token: int
    incremental_log_normalizer: float
    ess: float
    resampled: bool
    ancestors: tuple[int, ...]
    resampling_consumption: CounterConsumption | None
    record_sha256: str

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position <= 0:
            raise ValueError("assimilation position must be positive")
        if (
            type(self.observed_token) is not int
            or self.observed_token < 0
        ):
            raise ValueError("observed token must be nonnegative")
        if not math.isfinite(self.incremental_log_normalizer):
            raise ValueError("incremental log normalizer must be finite")
        if not math.isfinite(self.ess) or self.ess <= 0.0:
            raise ValueError("ESS must be finite and positive")
        if type(self.resampled) is not bool:
            raise ValueError("resampled must be an exact Boolean")
        if (
            type(self.ancestors) is not tuple
            or any(
                type(ancestor) is not int or ancestor < 0
                for ancestor in self.ancestors
            )
        ):
            raise ValueError("ancestor indices must be immutable integers")
        if self.resampled:
            if (
                not self.ancestors
                or type(self.resampling_consumption)
                is not CounterConsumption
            ):
                raise ValueError(
                    "resampling requires ancestors and one counter record"
                )
            self.resampling_consumption.__post_init__()
        elif self.ancestors or self.resampling_consumption is not None:
            raise ValueError(
                "non-resampled assimilation cannot consume a resampling counter"
            )
        expected = _owned_hash(
            "vfe4.h6.assimilation-record.v1", self._payload()
        )
        if self.record_sha256 != expected:
            raise ValueError("assimilation record identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "position": self.position,
            "observed_token": self.observed_token,
            "incremental_log_normalizer": (
                self.incremental_log_normalizer
            ),
            "ess": self.ess,
            "resampled": self.resampled,
            "ancestors": self.ancestors,
            "resampling_consumption_sha256": (
                None
                if self.resampling_consumption is None
                else self.resampling_consumption.consumption_sha256
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        position: int,
        observed_token: int,
        incremental_log_normalizer: float,
        ess: float,
        ancestors: tuple[int, ...],
        resampling_consumption: CounterConsumption | None,
    ) -> "AssimilationRecord":
        payload = {
            "position": position,
            "observed_token": observed_token,
            "incremental_log_normalizer": incremental_log_normalizer,
            "ess": ess,
            "resampled": bool(ancestors),
            "ancestors": ancestors,
            "resampling_consumption_sha256": (
                None
                if resampling_consumption is None
                else resampling_consumption.consumption_sha256
            ),
        }
        return cls(
            position,
            observed_token,
            incremental_log_normalizer,
            ess,
            bool(ancestors),
            ancestors,
            resampling_consumption,
            _owned_hash("vfe4.h6.assimilation-record.v1", payload),
        )


@dataclass(frozen=True)
class PrefixCache:
    key: PrefixCacheKey
    filtered_population: ProposalPopulation
    filtered_log_weights: FrozenTensorSnapshot
    cumulative_log_normalizer: float
    pending: PendingPrediction | MarginalPendingPrediction
    assimilations: tuple[AssimilationRecord, ...]
    counter_consumption: tuple[CounterConsumption, ...]
    cache_sha256: str

    def __post_init__(self) -> None:
        if type(self.key) is not PrefixCacheKey:
            raise ValueError("cache key must be an exact PrefixCacheKey")
        self.key.__post_init__()
        if type(self.filtered_population) is not ProposalPopulation:
            raise ValueError("cache requires an exact filtered population")
        self.filtered_population.__post_init__()
        if type(self.filtered_log_weights) is not FrozenTensorSnapshot:
            raise ValueError("cache weights must be frozen")
        self.filtered_log_weights.assert_intact()
        if (
            self.filtered_log_weights.dtype != "float64"
            or self.filtered_log_weights.shape
            != (self.filtered_population.particle_count,)
            or not math.isclose(
                float(
                    torch.logsumexp(
                        self.filtered_log_weights.value(), dim=0
                    ).item()
                ),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-13,
            )
        ):
            raise ValueError(
                "cache weights must be one normalized float64 row"
            )
        if not math.isfinite(self.cumulative_log_normalizer):
            raise ValueError("cumulative log normalizer must be finite")
        if type(self.pending) not in (
            PendingPrediction,
            MarginalPendingPrediction,
        ):
            raise ValueError("cache requires one immutable pending prediction")
        self.pending.__post_init__()
        if (
            self.pending.prefix_sha256 != self.key.prefix_sha256
            or self.pending.proposal_identity_sha256
            != self.key.proposal_identity_sha256
            or self.pending.proposed_population.population_sha256
            != self.filtered_population.population_sha256
            or self.pending.parent_log_weights.raw_bytes_sha256
            != self.filtered_log_weights.raw_bytes_sha256
        ):
            raise ValueError(
                "pending prediction does not match the filtered cache key/state"
            )
        if (
            type(self.assimilations) is not tuple
            or any(
                type(record) is not AssimilationRecord
                for record in self.assimilations
            )
            or len(self.assimilations) != len(self.key.prefix_tokens)
        ):
            raise ValueError(
                "cache must retain one immutable assimilation per prefix token"
            )
        expected_cumulative_log_normalizer = 0.0
        for position, (record, token) in enumerate(
            zip(self.assimilations, self.key.prefix_tokens, strict=True),
            start=1,
        ):
            record.__post_init__()
            if record.position != position or record.observed_token != token:
                raise ValueError(
                    "cache assimilations must exactly match sequential prefix tokens"
                )
            expected_cumulative_log_normalizer = math.fsum(
                [
                    expected_cumulative_log_normalizer,
                    record.incremental_log_normalizer,
                ]
            )
        if (
            self.cumulative_log_normalizer
            != expected_cumulative_log_normalizer
        ):
            raise ValueError(
                "cache cumulative log normalizer does not match assimilations"
            )
        if (
            type(self.counter_consumption) is not tuple
            or any(
                type(record) is not CounterConsumption
                for record in self.counter_consumption
            )
        ):
            raise ValueError(
                "cache counter consumption must be immutable exact records"
            )
        for record in self.counter_consumption:
            record.__post_init__()
        expected = _owned_hash("vfe4.h6.prefix-cache.v1", self._payload())
        if self.cache_sha256 != expected:
            raise ValueError("prefix cache identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "key_sha256": self.key.key_sha256,
            "filtered_population_sha256": (
                self.filtered_population.population_sha256
            ),
            "filtered_log_weights": _snapshot_payload(
                self.filtered_log_weights
            ),
            "cumulative_log_normalizer": self.cumulative_log_normalizer,
            "pending_sha256": self.pending.pending_sha256,
            "assimilations": tuple(
                record.record_sha256 for record in self.assimilations
            ),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in self.counter_consumption
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        key: PrefixCacheKey,
        filtered_population: ProposalPopulation,
        filtered_log_weights: Tensor,
        cumulative_log_normalizer: float,
        pending: PendingPrediction | MarginalPendingPrediction,
        assimilations: tuple[AssimilationRecord, ...],
        counter_consumption: tuple[CounterConsumption, ...],
    ) -> "PrefixCache":
        weights = FrozenTensorSnapshot.capture(filtered_log_weights)
        provisional = {
            "key_sha256": key.key_sha256,
            "filtered_population_sha256": (
                filtered_population.population_sha256
            ),
            "filtered_log_weights": _snapshot_payload(weights),
            "cumulative_log_normalizer": cumulative_log_normalizer,
            "pending_sha256": pending.pending_sha256,
            "assimilations": tuple(
                record.record_sha256 for record in assimilations
            ),
            "counter_consumption": tuple(
                record.consumption_sha256
                for record in counter_consumption
            ),
        }
        return cls(
            key,
            filtered_population,
            weights,
            cumulative_log_normalizer,
            pending,
            assimilations,
            counter_consumption,
            _owned_hash("vfe4.h6.prefix-cache.v1", provisional),
        )


__all__ = [
    "AssimilationRecord",
    "MarginalPendingPrediction",
    "PendingPrediction",
    "PrefixCache",
    "PrefixCacheKey",
]
