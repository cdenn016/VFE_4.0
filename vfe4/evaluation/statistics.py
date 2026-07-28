"""Estimator-aware WikiText-103 aggregation and frozen paired decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from vfe4.artifacts.durability import DurableFileIdentity
from vfe4.numerics.critical_values import ENDPOINT_T_DF63
from vfe4.predictive import EstimatorStream
from vfe4.types import (
    EndpointInventory,
    WT103CheckpointIdentity,
    WT103EvaluationRecord,
    validate_endpoint_inventory,
)
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    WT103_ARM_IDS,
    WT103_PARTICLE_COUNTS,
    owned_sha256,
)

from .smc_uncertainty import (
    PAIRED_CORNER_COUNT,
    PREDICTION_DELTA,
    EndpointSmcAggregate,
    EndpointSmcObservation,
    InflatedPairedInterval,
    aggregate_endpoint_smc,
    inflate_paired_interval,
)
from .prior_nll import (
    WT103EstimatorStreamBinding,
    WT103EvaluationBatches,
    WT103ScoreTrace,
    wt103_common_stream_registry_sha256,
    wt103_estimator_stream_seed,
)
from .test_opening import (
    DurableTestOpeningCapability,
    _require_issued_capability,
)


_OBJECTIVE_COMPLETE_ARM = (
    "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1"
)
_OBJECTIVE_EMISSION_ARM = (
    "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1"
)
_PRIMARY_REFERENCE_ARM = "WT103-A0-AR-v1"
_PRIMARY_ENDPOINT_ARM = _OBJECTIVE_COMPLETE_ARM
_HEX = frozenset("0123456789abcdef")
_EXPECTED_TEST_RECORD_CACHE: dict[
    str,
    tuple[EndpointInventory, dict[str, tuple[object, ...]]],
] = {}
_RAW_UPSTREAM_CACHE: dict[
    tuple[int, int, int],
    tuple[
        EndpointInventory,
        DurableTestOpeningCapability,
        WT103EvaluationBatches,
    ],
] = {}


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class WT103ArmApplicability:
    arm_id: str
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    estimator_statistics: Literal["applicable", "not_applicable"]
    estimator_reason: str
    objective_gate: Literal["applicable", "not_applicable"]
    objective_reason: str
    primary_gate: Literal["applicable", "not_applicable"]
    primary_reason: str
    result_role: str


@dataclass(frozen=True, slots=True)
class WT103SeedEstimate:
    arm_id: str
    seed_id: int
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    checkpoint_identity_sha256: str
    nll_per_token: float
    perplexity: float
    estimator_applicability: Literal["applicable", "not_applicable"]
    applicability_reason: str
    smc: EndpointSmcAggregate | None

    def __post_init__(self) -> None:
        if self.arm_id not in WT103_ARM_IDS:
            raise ValueError("seed estimate arm is outside the frozen inventory")
        if (
            type(self.seed_id) is not int
            or type(self.nll_per_token) is not float
            or not math.isfinite(self.nll_per_token)
            or self.nll_per_token < 0.0
            or type(self.perplexity) is not float
            or self.perplexity != math.exp(self.nll_per_token)
        ):
            raise ValueError("seed estimate NLL/perplexity is not canonical")
        if self.scorer_kind == "exact_autoregressive":
            if (
                self.estimator_applicability != "not_applicable"
                or self.applicability_reason
                != "exact_autoregressive_has_no_monte_carlo_estimator"
                or self.smc is not None
            ):
                raise ValueError("exact estimates cannot fabricate SMC statistics")
        elif (
            self.scorer_kind != "weighted_smc"
            or self.estimator_applicability != "applicable"
            or self.applicability_reason
            != "weighted_smc_requires_q2_remainder_and_random_error"
            or type(self.smc) is not EndpointSmcAggregate
        ):
            raise ValueError("weighted estimates require exact SMC statistics")


@dataclass(frozen=True, slots=True)
class WT103EstimatorAggregation:
    schema_version: Literal["wt103-estimator-aggregation-v1"]
    status: GateStatus
    complete: bool
    raw_record_count: int
    expected_raw_record_count: int
    seed_estimates: tuple[WT103SeedEstimate, ...]
    arm_applicability: tuple[WT103ArmApplicability, ...]
    obligations: tuple[str, ...]
    aggregation_sha256: str


@dataclass(frozen=True, slots=True)
class WT103CornerInterval:
    effects: tuple[float, ...]
    error_radii: tuple[float, ...]
    uninflated_mean: float
    uninflated_lower: float
    uninflated_upper: float
    corner_bounds: tuple[tuple[float, float], ...]
    lower: float
    upper: float
    eligible: bool
    corner_count: Literal[256]
    interval_sha256: str

    @classmethod
    def from_h6(
        cls,
        interval: InflatedPairedInterval,
    ) -> "WT103CornerInterval":
        interval.__post_init__()
        corner_bounds = tuple(
            (item.lower, item.upper) for item in interval.corner_intervals
        )
        payload = {
            "effects": interval.values,
            "error_radii": interval.error_radii,
            "uninflated_mean": interval.uninflated.mean,
            "uninflated_lower": interval.uninflated.lower,
            "uninflated_upper": interval.uninflated.upper,
            "corner_bounds": corner_bounds,
            "lower": interval.lower,
            "upper": interval.upper,
            "eligible": interval.eligible,
            "corner_count": PAIRED_CORNER_COUNT,
        }
        return cls(
            **payload,
            interval_sha256=owned_sha256(
                "vfe4.wt103.corner-interval.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class WT103ResultRow:
    result_row_key: str
    arm_id: str
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    applicability: Literal["decision_bearing", "descriptive_only"]
    applicability_reason: str
    result_role: str
    seed_nll_per_token: tuple[float, ...]
    seed_perplexity: tuple[float, ...]
    mean_nll_per_token: float
    mean_perplexity: float
    status: GateStatus
    row_sha256: str


@dataclass(frozen=True, slots=True)
class WT103PredictionStatistics:
    schema_version: Literal["wt103-prediction-statistics-v1"]
    complete: bool
    raw_record_count: int
    expected_raw_record_count: int
    objective_status: GateStatus
    primary_status: GateStatus
    objective_interval: WT103CornerInterval | None
    primary_interval: WT103CornerInterval | None
    delta: Literal[0.01005033585350145]
    arm_applicability: tuple[WT103ArmApplicability, ...]
    result_rows: tuple[WT103ResultRow, ...]
    figure_series_keys: tuple[str, ...]
    obligations: tuple[str, ...]
    statistics_sha256: str


def _applicability(
    inventory: EndpointInventory,
) -> tuple[WT103ArmApplicability, ...]:
    return tuple(
        WT103ArmApplicability(
            arm_id=arm.arm_id,
            scorer_kind=arm.scorer_kind,  # type: ignore[arg-type]
            estimator_statistics=(
                "applicable"
                if arm.scorer_kind == "weighted_smc"
                else "not_applicable"
            ),
            estimator_reason=(
                "weighted_smc_requires_q2_remainder_and_random_error"
                if arm.scorer_kind == "weighted_smc"
                else "exact_autoregressive_has_no_monte_carlo_estimator"
            ),
            objective_gate=(
                "applicable"
                if arm.arm_id
                in (_OBJECTIVE_COMPLETE_ARM, _OBJECTIVE_EMISSION_ARM)
                else "not_applicable"
            ),
            objective_reason=(
                "complete_minus_emission_objective_gate"
                if arm.arm_id
                in (_OBJECTIVE_COMPLETE_ARM, _OBJECTIVE_EMISSION_ARM)
                else "arm_is_not_an_objective_gate_endpoint"
            ),
            primary_gate=(
                "applicable"
                if arm.arm_id in (_PRIMARY_REFERENCE_ARM, _PRIMARY_ENDPOINT_ARM)
                else "not_applicable"
            ),
            primary_reason=(
                "a0_minus_parent_specific_complete_primary_pair"
                if arm.arm_id in (_PRIMARY_REFERENCE_ARM, _PRIMARY_ENDPOINT_ARM)
                else "arm_is_descriptive_and_cannot_rescue_primary"
            ),
            result_role=arm.result_role,
        )
        for arm in inventory.arms
    )


def _expected_test_records(
    inventory: EndpointInventory,
) -> dict[str, tuple[object, ...]]:
    cached = _EXPECTED_TEST_RECORD_CACHE.get(
        inventory.endpoint_inventory_sha256
    )
    if cached is not None and cached[0] == inventory:
        return cached[1]
    expected: dict[str, tuple[object, ...]] = {}
    for arm in inventory.arms:
        for seed in inventory.confirmatory_seed_ids:
            checkpoint_key = f"terminal/{arm.arm_id}/seed={seed}"
            endpoint = f"test/{checkpoint_key}"
            if arm.scorer_kind == "exact_autoregressive":
                key = f"raw-score/test/{endpoint}/exact"
                expected[key] = (
                    arm.arm_id,
                    seed,
                    checkpoint_key,
                    arm.scorer_kind,
                    None,
                    None,
                )
                continue
            for particle_count in inventory.particle_counts:
                for stream_id in inventory.test_stream_ids:
                    key = (
                        f"raw-score/test/{endpoint}/particles="
                        f"{particle_count}/stream={stream_id}"
                    )
                    expected[key] = (
                        arm.arm_id,
                        seed,
                        checkpoint_key,
                        arm.scorer_kind,
                        particle_count,
                        stream_id,
                    )
    expected_inventory = tuple(
        key
        for key in inventory.raw_score_record_keys
        if key.startswith("raw-score/test/")
    )
    if tuple(expected) != expected_inventory:
        raise ValueError("derived test score keys differ from EndpointInventory")
    _EXPECTED_TEST_RECORD_CACHE[inventory.endpoint_inventory_sha256] = (
        inventory,
        expected,
    )
    return expected


def _common_test_stream_registry_sha256(
    *,
    estimator_protocol_sha256: str,
    stream_id: int,
) -> str:
    """Derive the frozen logical common-stream registry identity.

    This identity authenticates the preregistered logical stream shared across
    particle counts.  The raw scorer cache audit separately binds the concrete
    estimator stream used to produce each corpus total.
    """

    if type(stream_id) is not int or not 0 <= stream_id < 64:
        raise ValueError("test stream ID is outside the frozen registry")
    return wt103_common_stream_registry_sha256(
        split="test",
        estimator_protocol_sha256=estimator_protocol_sha256,
        logical_stream_id=stream_id,
    )


def _validate_raw_upstream(
    *,
    inventory: EndpointInventory,
    opening_capability: DurableTestOpeningCapability,
    evaluation_batches: WT103EvaluationBatches,
) -> None:
    if type(inventory) is not EndpointInventory:
        raise ValueError("raw score requires an exact EndpointInventory")
    cache_key = (
        id(inventory),
        id(opening_capability),
        id(evaluation_batches),
    )
    cached = _RAW_UPSTREAM_CACHE.get(cache_key)
    if (
        cached is not None
        and cached[0] is inventory
        and cached[1] is opening_capability
        and cached[2] is evaluation_batches
    ):
        return
    validate_endpoint_inventory(
        inventory,
        expected_sha256=inventory.endpoint_inventory_sha256,
    )
    issued = _require_issued_capability(opening_capability)
    if not issued.consumed:
        raise ValueError(
            "raw score requires a consumed test-opening capability"
        )
    if type(evaluation_batches) is not WT103EvaluationBatches:
        raise ValueError(
            "raw score requires exact typed evaluation batches"
        )
    evaluation_batches.__post_init__()
    if (
        evaluation_batches.manifest.split != "test"
        or opening_capability.endpoint_inventory_sha256
        != inventory.endpoint_inventory_sha256
        or opening_capability.estimator_protocol_sha256
        != inventory.estimator_protocol_sha256
        or opening_capability.test_window_manifest_sha256
        != evaluation_batches.manifest.manifest_sha256
        or opening_capability.test_schedule_sha256
        != evaluation_batches.schedule.schedule_sha256
        or opening_capability.tokenizer_identity_sha256
        != evaluation_batches.windows.tokenizer_spec.spec_sha256
        or opening_capability.data_identity_sha256
        != evaluation_batches.windows.cache_record.raw_parent_sha256
    ):
        raise ValueError(
            "opening capability, inventory, and typed test batches disagree"
        )
    _RAW_UPSTREAM_CACHE[cache_key] = (
        inventory,
        opening_capability,
        evaluation_batches,
    )


def _validate_raw_endpoint(
    *,
    inventory: EndpointInventory,
    raw_record_key: str,
    checkpoint: WT103CheckpointIdentity,
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"],
    particle_count: int | None,
    logical_stream_id: int | None,
) -> None:
    expected = _expected_test_records(inventory).get(raw_record_key)
    if expected is None:
        raise ValueError("raw score key is outside EndpointInventory")
    (
        _arm_id,
        _seed,
        checkpoint_key,
        expected_scorer,
        expected_particles,
        expected_stream,
    ) = expected
    if (
        checkpoint.logical_key != checkpoint_key
        or checkpoint.checkpoint_role != "terminal_scoring"
        or scorer_kind != expected_scorer
        or particle_count != expected_particles
        or logical_stream_id != expected_stream
    ):
        raise ValueError(
            "raw score endpoint evidence differs from EndpointInventory"
        )


@dataclass(frozen=True, init=False, slots=True)
class WT103RawScoreRecord:
    """Integrity envelope for one finalized or failed raw test endpoint."""

    schema_version: Literal["wt103-raw-score-record-v1"]
    raw_record_key: str
    endpoint_inventory_sha256: str
    opening_capability_sha256: str
    reservation_identity_sha256: str
    estimator_protocol_sha256: str
    checkpoint: WT103CheckpointIdentity
    data_identity_sha256: str
    tokenizer_identity_sha256: str
    window_manifest_sha256: str
    schedule_sha256: str
    window_source_sha256: str
    batches_sha256: str
    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    logical_stream_id: int | None
    particle_count: int | None
    estimator_stream_binding_sha256: str | None
    estimator_stream_sha256: str | None
    stream_seed: int | None
    common_stream_registry_sha256: str | None
    counter_trace_sha256: str | None
    counter_draw_count: int | None
    score_trace_sha256: str | None
    disposition: Literal["finalized", "failed"]
    evaluation: WT103EvaluationRecord | None
    failure_identity_sha256: str | None
    failure_reason: str | None
    raw_record_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_record_key": self.raw_record_key,
            "endpoint_inventory_sha256": self.endpoint_inventory_sha256,
            "opening_capability_sha256": self.opening_capability_sha256,
            "reservation_identity_sha256": self.reservation_identity_sha256,
            "estimator_protocol_sha256": self.estimator_protocol_sha256,
            "checkpoint": self.checkpoint,
            "data_identity_sha256": self.data_identity_sha256,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "window_manifest_sha256": self.window_manifest_sha256,
            "schedule_sha256": self.schedule_sha256,
            "window_source_sha256": self.window_source_sha256,
            "batches_sha256": self.batches_sha256,
            "scorer_kind": self.scorer_kind,
            "logical_stream_id": self.logical_stream_id,
            "particle_count": self.particle_count,
            "estimator_stream_binding_sha256": (
                self.estimator_stream_binding_sha256
            ),
            "estimator_stream_sha256": self.estimator_stream_sha256,
            "stream_seed": self.stream_seed,
            "common_stream_registry_sha256": (
                self.common_stream_registry_sha256
            ),
            "counter_trace_sha256": self.counter_trace_sha256,
            "counter_draw_count": self.counter_draw_count,
            "score_trace_sha256": self.score_trace_sha256,
            "disposition": self.disposition,
            "evaluation": self.evaluation,
            "failure_identity_sha256": self.failure_identity_sha256,
            "failure_reason": self.failure_reason,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version != "wt103-raw-score-record-v1"
            or type(self.raw_record_key) is not str
            or not self.raw_record_key.startswith("raw-score/test/")
            or self.scorer_kind
            not in ("exact_autoregressive", "weighted_smc")
            or self.disposition not in ("finalized", "failed")
        ):
            raise ValueError("raw score record literals are invalid")
        for name in (
            "endpoint_inventory_sha256",
            "opening_capability_sha256",
            "reservation_identity_sha256",
            "estimator_protocol_sha256",
            "data_identity_sha256",
            "tokenizer_identity_sha256",
            "window_manifest_sha256",
            "schedule_sha256",
            "window_source_sha256",
            "batches_sha256",
            "raw_record_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.checkpoint) is not WT103CheckpointIdentity:
            raise ValueError("raw score checkpoint has the wrong type")
        self.checkpoint.__post_init__()
        if self.checkpoint.checkpoint_role != "terminal_scoring":
            raise ValueError("raw scoring requires a terminal checkpoint")

        if self.scorer_kind == "exact_autoregressive":
            if any(
                value is not None
                for value in (
                    self.logical_stream_id,
                    self.particle_count,
                    self.estimator_stream_binding_sha256,
                    self.estimator_stream_sha256,
                    self.stream_seed,
                    self.common_stream_registry_sha256,
                    self.counter_trace_sha256,
                    self.counter_draw_count,
                )
            ):
                raise ValueError(
                    "exact raw score cannot fabricate estimator evidence"
                )
        else:
            if (
                type(self.logical_stream_id) is not int
                or not 0 <= self.logical_stream_id < 64
                or type(self.particle_count) is not int
                or self.particle_count not in WT103_PARTICLE_COUNTS
                or type(self.stream_seed) is not int
            ):
                raise ValueError("weighted raw score identities are invalid")
            for name in (
                "estimator_stream_binding_sha256",
                "estimator_stream_sha256",
                "common_stream_registry_sha256",
            ):
                _require_sha256(getattr(self, name), name)
            expected_common = _common_test_stream_registry_sha256(
                estimator_protocol_sha256=self.estimator_protocol_sha256,
                stream_id=self.logical_stream_id,
            )
            if self.common_stream_registry_sha256 != expected_common:
                raise ValueError(
                    "common stream registry identity is not canonical"
                )
            expected_seed = wt103_estimator_stream_seed(
                split="test",
                estimator_protocol_sha256=self.estimator_protocol_sha256,
                logical_stream_id=self.logical_stream_id,
            )
            if self.stream_seed != expected_seed:
                raise ValueError(
                    "weighted raw score stream seed is not canonical"
                )

        if self.disposition == "finalized":
            if (
                type(self.evaluation) is not WT103EvaluationRecord
                or self.failure_identity_sha256 is not None
                or self.failure_reason is not None
            ):
                raise ValueError(
                    "finalized raw score requires only evaluation evidence"
                )
            self.evaluation.__post_init__()
            _require_sha256(
                self.score_trace_sha256,
                "score_trace_sha256",
            )
            if (
                self.evaluation.endpoint_key != self.raw_record_key
                or self.evaluation.checkpoint != self.checkpoint
                or self.evaluation.totals.scorer_kind != self.scorer_kind
                or self.evaluation.totals.estimator_stream_id
                != self.logical_stream_id
                or self.evaluation.totals.particle_count
                != self.particle_count
                or (
                    self.scorer_kind == "weighted_smc"
                    and (
                        type(self.counter_draw_count) is not int
                        or self.counter_draw_count <= 0
                        or _require_sha256(
                            self.counter_trace_sha256,
                            "counter_trace_sha256",
                        )
                        != self.counter_trace_sha256
                    )
                )
                or (
                    self.scorer_kind == "exact_autoregressive"
                    and (
                        self.counter_trace_sha256 is not None
                        or self.counter_draw_count is not None
                    )
                )
            ):
                raise ValueError(
                    "finalized raw score envelope/evaluation disagree"
                )
        else:
            if (
                self.evaluation is not None
                or self.score_trace_sha256 is not None
                or self.counter_trace_sha256 is not None
                or self.counter_draw_count is not None
                or type(self.failure_reason) is not str
                or not self.failure_reason.strip()
            ):
                raise ValueError(
                    "failed raw score requires only explicit failure evidence"
                )
            _require_sha256(
                self.failure_identity_sha256,
                "failure_identity_sha256",
            )

        expected = owned_sha256(
            "vfe4.wt103.raw-score-record.v1",
            self._payload(),
        )
        if self.raw_record_sha256 != expected:
            raise ValueError(
                "raw_record_sha256 does not match raw score evidence"
            )

    @classmethod
    def create_finalized(
        cls,
        *,
        inventory: EndpointInventory,
        evaluation: WT103EvaluationRecord,
        opening_capability: DurableTestOpeningCapability,
        evaluation_batches: WT103EvaluationBatches,
        score_trace: WT103ScoreTrace,
    ) -> "WT103RawScoreRecord":
        _validate_raw_upstream(
            inventory=inventory,
            opening_capability=opening_capability,
            evaluation_batches=evaluation_batches,
        )
        if type(evaluation) is not WT103EvaluationRecord:
            raise ValueError("raw score requires an exact evaluation record")
        evaluation.__post_init__()
        if type(score_trace) is not WT103ScoreTrace:
            raise ValueError("raw score requires an exact WT103ScoreTrace")
        score_trace.__post_init__()
        totals = evaluation.totals
        if (
            score_trace.evaluation_batches is not evaluation_batches
            or score_trace.totals is not totals
            or score_trace.binding.split != "test"
            or score_trace.binding.estimator_protocol_sha256
            != inventory.estimator_protocol_sha256
        ):
            raise ValueError(
                "raw score trace differs from evaluation/opening evidence"
            )
        _validate_raw_endpoint(
            inventory=inventory,
            raw_record_key=evaluation.endpoint_key,
            checkpoint=evaluation.checkpoint,
            scorer_kind=totals.scorer_kind,
            particle_count=totals.particle_count,
            logical_stream_id=totals.estimator_stream_id,
        )
        weighted = totals.scorer_kind == "weighted_smc"
        common_stream = (
            None
            if not weighted
            else score_trace.binding.common_stream_registry_sha256
        )
        payload = {
            "schema_version": "wt103-raw-score-record-v1",
            "raw_record_key": evaluation.endpoint_key,
            "endpoint_inventory_sha256": (
                inventory.endpoint_inventory_sha256
            ),
            "opening_capability_sha256": (
                opening_capability.capability_sha256
            ),
            "reservation_identity_sha256": (
                opening_capability.reservation_identity_sha256
            ),
            "estimator_protocol_sha256": (
                inventory.estimator_protocol_sha256
            ),
            "checkpoint": evaluation.checkpoint,
            "data_identity_sha256": (
                opening_capability.data_identity_sha256
            ),
            "tokenizer_identity_sha256": (
                opening_capability.tokenizer_identity_sha256
            ),
            "window_manifest_sha256": (
                evaluation_batches.manifest.manifest_sha256
            ),
            "schedule_sha256": (
                evaluation_batches.schedule.schedule_sha256
            ),
            "window_source_sha256": (
                evaluation_batches.window_source_sha256
            ),
            "batches_sha256": evaluation_batches.batches_sha256,
            "scorer_kind": totals.scorer_kind,
            "logical_stream_id": totals.estimator_stream_id,
            "particle_count": totals.particle_count,
            "estimator_stream_binding_sha256": (
                score_trace.binding.binding_sha256 if weighted else None
            ),
            "estimator_stream_sha256": (
                score_trace.stream.stream_sha256 if weighted else None
            ),
            "stream_seed": (
                score_trace.stream.stream_seed if weighted else None
            ),
            "common_stream_registry_sha256": common_stream,
            "counter_trace_sha256": score_trace.counter_trace_sha256,
            "counter_draw_count": score_trace.counter_draw_count,
            "score_trace_sha256": score_trace.trace_sha256,
            "disposition": "finalized",
            "evaluation": evaluation,
            "failure_identity_sha256": None,
            "failure_reason": None,
        }
        return cls._from_payload(payload)

    @classmethod
    def create_failed(
        cls,
        *,
        inventory: EndpointInventory,
        raw_record_key: str,
        checkpoint: WT103CheckpointIdentity,
        scorer_kind: Literal["exact_autoregressive", "weighted_smc"],
        logical_stream_id: int | None,
        particle_count: int | None,
        opening_capability: DurableTestOpeningCapability,
        evaluation_batches: WT103EvaluationBatches,
        failure_identity: DurableFileIdentity,
        failure_reason: str,
        estimator_stream_binding: WT103EstimatorStreamBinding | None = None,
        estimator_stream: EstimatorStream | None = None,
    ) -> "WT103RawScoreRecord":
        _validate_raw_upstream(
            inventory=inventory,
            opening_capability=opening_capability,
            evaluation_batches=evaluation_batches,
        )
        if type(failure_identity) is not DurableFileIdentity:
            raise ValueError(
                "raw failure requires a typed durable failure identity"
            )
        _require_sha256(
            failure_identity.identity_sha256,
            "failure identity",
        )
        _validate_raw_endpoint(
            inventory=inventory,
            raw_record_key=raw_record_key,
            checkpoint=checkpoint,
            scorer_kind=scorer_kind,
            particle_count=particle_count,
            logical_stream_id=logical_stream_id,
        )
        weighted = scorer_kind == "weighted_smc"
        if weighted:
            if (
                type(estimator_stream_binding)
                is not WT103EstimatorStreamBinding
                or type(estimator_stream) is not EstimatorStream
            ):
                raise ValueError(
                    "weighted failure requires typed binding/stream evidence"
                )
            estimator_stream_binding.__post_init__()
            estimator_stream.__post_init__()
            if (
                estimator_stream_binding.split != "test"
                or estimator_stream_binding.logical_stream_id
                != logical_stream_id
                or estimator_stream_binding.estimator_protocol_sha256
                != inventory.estimator_protocol_sha256
                or estimator_stream_binding.estimator_stream_sha256
                != estimator_stream.stream_sha256
                or estimator_stream_binding.stream_seed
                != estimator_stream.stream_seed
            ):
                raise ValueError(
                    "weighted failure binding/stream evidence disagrees"
                )
        elif (
            estimator_stream_binding is not None
            or estimator_stream is not None
        ):
            raise ValueError("exact failure cannot fabricate estimator evidence")
        common_stream = (
            None
            if not weighted
            else estimator_stream_binding.common_stream_registry_sha256
        )
        payload = {
            "schema_version": "wt103-raw-score-record-v1",
            "raw_record_key": raw_record_key,
            "endpoint_inventory_sha256": (
                inventory.endpoint_inventory_sha256
            ),
            "opening_capability_sha256": (
                opening_capability.capability_sha256
            ),
            "reservation_identity_sha256": (
                opening_capability.reservation_identity_sha256
            ),
            "estimator_protocol_sha256": (
                inventory.estimator_protocol_sha256
            ),
            "checkpoint": checkpoint,
            "data_identity_sha256": (
                opening_capability.data_identity_sha256
            ),
            "tokenizer_identity_sha256": (
                opening_capability.tokenizer_identity_sha256
            ),
            "window_manifest_sha256": (
                evaluation_batches.manifest.manifest_sha256
            ),
            "schedule_sha256": (
                evaluation_batches.schedule.schedule_sha256
            ),
            "window_source_sha256": (
                evaluation_batches.window_source_sha256
            ),
            "batches_sha256": evaluation_batches.batches_sha256,
            "scorer_kind": scorer_kind,
            "logical_stream_id": logical_stream_id,
            "particle_count": particle_count,
            "estimator_stream_binding_sha256": (
                None
                if estimator_stream_binding is None
                else estimator_stream_binding.binding_sha256
            ),
            "estimator_stream_sha256": (
                None
                if estimator_stream is None
                else estimator_stream.stream_sha256
            ),
            "stream_seed": (
                None
                if estimator_stream is None
                else estimator_stream.stream_seed
            ),
            "common_stream_registry_sha256": common_stream,
            "counter_trace_sha256": None,
            "counter_draw_count": None,
            "score_trace_sha256": None,
            "disposition": "failed",
            "evaluation": None,
            "failure_identity_sha256": failure_identity.identity_sha256,
            "failure_reason": failure_reason,
        }
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, object],
    ) -> "WT103RawScoreRecord":
        record = object.__new__(cls)
        for name, value in payload.items():
            object.__setattr__(record, name, value)
        object.__setattr__(
            record,
            "raw_record_sha256",
            owned_sha256(
                "vfe4.wt103.raw-score-record.v1",
                payload,
            ),
        )
        record.__post_init__()
        return record


def _incomplete_aggregation(
    *,
    inventory: EndpointInventory,
    raw_count: int,
    expected_count: int,
    obligations: tuple[str, ...],
) -> WT103EstimatorAggregation:
    payload = {
        "schema_version": "wt103-estimator-aggregation-v1",
        "status": GateStatus.INCONCLUSIVE.value,
        "complete": False,
        "raw_record_count": raw_count,
        "expected_raw_record_count": expected_count,
        "seed_estimate_keys": (),
        "arm_applicability": tuple(
            (
                item.arm_id,
                item.estimator_statistics,
                item.objective_gate,
                item.primary_gate,
            )
            for item in _applicability(inventory)
        ),
        "obligations": obligations,
    }
    return WT103EstimatorAggregation(
        schema_version="wt103-estimator-aggregation-v1",
        status=GateStatus.INCONCLUSIVE,
        complete=False,
        raw_record_count=raw_count,
        expected_raw_record_count=expected_count,
        seed_estimates=(),
        arm_applicability=_applicability(inventory),
        obligations=obligations,
        aggregation_sha256=owned_sha256(
            "vfe4.wt103.estimator-aggregation.v1",
            payload,
        ),
    )


def aggregate_a5_smc(
    records: tuple[WT103RawScoreRecord, ...],
    *,
    inventory: EndpointInventory,
) -> WT103EstimatorAggregation:
    """Aggregate the exact scorer-kind-derived five-arm test inventory."""

    if type(inventory) is not EndpointInventory:
        raise ValueError("inventory must be an exact EndpointInventory")
    validate_endpoint_inventory(
        inventory,
        expected_sha256=inventory.endpoint_inventory_sha256,
    )
    if type(records) is not tuple or any(
        type(record) is not WT103RawScoreRecord for record in records
    ):
        raise ValueError(
            "records must be an immutable enveloped WT103RawScoreRecord tuple"
        )
    expected = _expected_test_records(inventory)
    keys = tuple(record.raw_record_key for record in records)
    duplicate_keys = len(set(keys)) != len(keys)
    missing_keys = tuple(key for key in expected if key not in set(keys))
    unexpected_keys = tuple(key for key in keys if key not in expected)
    invalid: list[str] = []
    for record in records:
        try:
            record.__post_init__()
        except ValueError:
            invalid.append(record.raw_record_key)
    failed = tuple(
        record.raw_record_key
        for record in records
        if record.disposition != "finalized"
    )
    obligations: list[str] = []
    if duplicate_keys:
        obligations.append("duplicate raw test records prevent aggregation")
    if missing_keys:
        obligations.append("missing raw test records prevent aggregation")
    if unexpected_keys:
        obligations.append("unexpected raw test records prevent aggregation")
    if invalid:
        obligations.append("invalid or nonfinite raw test records prevent aggregation")
    if failed:
        obligations.append("failed raw test records prevent aggregation")
    if any(
        record.endpoint_inventory_sha256
        != inventory.endpoint_inventory_sha256
        or record.estimator_protocol_sha256
        != inventory.estimator_protocol_sha256
        for record in records
    ):
        obligations.append(
            "raw test envelope differs from EndpointInventory"
        )
    for name in (
        "opening_capability_sha256",
        "reservation_identity_sha256",
        "data_identity_sha256",
        "tokenizer_identity_sha256",
        "window_manifest_sha256",
        "schedule_sha256",
        "window_source_sha256",
        "batches_sha256",
    ):
        if len({getattr(record, name) for record in records}) != 1:
            obligations.append(
                f"raw test records disagree on {name}"
            )
    if obligations:
        return _incomplete_aggregation(
            inventory=inventory,
            raw_count=len(records),
            expected_count=len(expected),
            obligations=tuple(obligations),
        )

    by_key = {record.raw_record_key: record for record in records}
    counted_targets: set[int] = set()
    checkpoint_ids: dict[tuple[str, int], str] = {}
    for key, metadata in expected.items():
        arm_id, seed, checkpoint_key, scorer_kind, particles, stream_id = metadata
        raw_record = by_key[key]
        record = raw_record.evaluation
        if type(record) is not WT103EvaluationRecord:
            obligations.append(
                "finalized raw test envelope lost evaluation evidence"
            )
            break
        totals = record.totals
        counted_targets.add(totals.counted_targets)
        checkpoint_pair = (str(arm_id), int(seed))
        previous = checkpoint_ids.setdefault(
            checkpoint_pair,
            record.checkpoint.checkpoint_identity_sha256,
        )
        if (
            record.checkpoint.logical_key != checkpoint_key
            or record.checkpoint.checkpoint_role != "terminal_scoring"
            or previous != record.checkpoint.checkpoint_identity_sha256
            or totals.scorer_kind != scorer_kind
            or totals.particle_count != particles
            or raw_record.logical_stream_id != stream_id
            or raw_record.particle_count != particles
            or (
                scorer_kind == "weighted_smc"
                and totals.estimator_stream_id != stream_id
            )
            or (
                scorer_kind == "exact_autoregressive"
                and totals.estimator_stream_id is not None
            )
        ):
            obligations.append(
                "raw test record metadata differs from EndpointInventory"
            )
            break
    if len(counted_targets) != 1:
        obligations.append("raw test records disagree on counted targets")
    if obligations:
        return _incomplete_aggregation(
            inventory=inventory,
            raw_count=len(records),
            expected_count=len(expected),
            obligations=tuple(dict.fromkeys(obligations)),
        )

    estimates: list[WT103SeedEstimate] = []
    estimator_inconclusive = False
    estimator_obligations: list[str] = []
    for arm in inventory.arms:
        for seed in inventory.confirmatory_seed_ids:
            checkpoint_key = f"terminal/{arm.arm_id}/seed={seed}"
            endpoint = f"test/{checkpoint_key}"
            if arm.scorer_kind == "exact_autoregressive":
                raw_record = by_key[f"raw-score/test/{endpoint}/exact"]
                record = raw_record.evaluation
                if type(record) is not WT103EvaluationRecord:
                    raise RuntimeError("finalized exact raw score lost evaluation")
                estimates.append(
                    WT103SeedEstimate(
                        arm_id=arm.arm_id,
                        seed_id=seed,
                        scorer_kind="exact_autoregressive",
                        checkpoint_identity_sha256=(
                            record.checkpoint.checkpoint_identity_sha256
                        ),
                        nll_per_token=record.totals.nll_per_token,
                        perplexity=record.totals.perplexity,
                        estimator_applicability="not_applicable",
                        applicability_reason=(
                            "exact_autoregressive_has_no_monte_carlo_estimator"
                        ),
                        smc=None,
                    )
                )
                continue
            observations = tuple(
                EndpointSmcObservation(
                    checkpoint_sha256=(
                        by_key[key].checkpoint.checkpoint_identity_sha256
                    ),
                    replicate_id=stream_id,
                    particle_count=particle_count,
                    common_stream_sha256=(
                        by_key[key].common_stream_registry_sha256
                    ),
                    negative_log_likelihood_sum=(
                        by_key[key].evaluation.totals.summed_nll
                    ),
                    counted_targets=(
                        by_key[key].evaluation.totals.counted_targets
                    ),
                )
                for particle_count in WT103_PARTICLE_COUNTS
                for stream_id in inventory.test_stream_ids
                for key in (
                    f"raw-score/test/{endpoint}/particles="
                    f"{particle_count}/stream={stream_id}",
                )
            )
            aggregate = aggregate_endpoint_smc(observations)
            if type(aggregate) is not EndpointSmcAggregate:
                return _incomplete_aggregation(
                    inventory=inventory,
                    raw_count=len(records),
                    expected_count=len(expected),
                    obligations=(
                        "weighted SMC endpoint aggregation rejected raw records",
                    ),
                )
            if not aggregate.eligible:
                estimator_inconclusive = True
                estimator_obligations.extend(aggregate.obligations)
            estimates.append(
                WT103SeedEstimate(
                    arm_id=arm.arm_id,
                    seed_id=seed,
                    scorer_kind="weighted_smc",
                    checkpoint_identity_sha256=aggregate.checkpoint_sha256,
                    nll_per_token=aggregate.reported_nll,
                    perplexity=math.exp(aggregate.reported_nll),
                    estimator_applicability="applicable",
                    applicability_reason=(
                        "weighted_smc_requires_q2_remainder_and_random_error"
                    ),
                    smc=aggregate,
                )
            )
    status = (
        GateStatus.INCONCLUSIVE
        if estimator_inconclusive
        else GateStatus.PASS
    )
    final_obligations = tuple(dict.fromkeys(estimator_obligations))
    payload = {
        "schema_version": "wt103-estimator-aggregation-v1",
        "status": status.value,
        "complete": True,
        "raw_record_count": len(records),
        "expected_raw_record_count": len(expected),
        "seed_estimate_keys": tuple(
            (
                item.arm_id,
                item.seed_id,
                item.checkpoint_identity_sha256,
                item.nll_per_token,
                None if item.smc is None else item.smc.reported_nll,
            )
            for item in estimates
        ),
        "arm_applicability": tuple(
            (
                item.arm_id,
                item.estimator_statistics,
                item.objective_gate,
                item.primary_gate,
            )
            for item in _applicability(inventory)
        ),
        "obligations": final_obligations,
    }
    return WT103EstimatorAggregation(
        schema_version="wt103-estimator-aggregation-v1",
        status=status,
        complete=True,
        raw_record_count=len(records),
        expected_raw_record_count=len(expected),
        seed_estimates=tuple(estimates),
        arm_applicability=_applicability(inventory),
        obligations=final_obligations,
        aggregation_sha256=owned_sha256(
            "vfe4.wt103.estimator-aggregation.v1",
            payload,
        ),
    )


def _half_width(values: tuple[float, ...]) -> float:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    return ENDPOINT_T_DF63 * math.sqrt(variance / len(values))


def _corner_interval(
    *,
    effects: tuple[float, ...],
    paired_half_widths: tuple[float, ...],
    left_bias_bounds: tuple[float, ...],
    right_bias_bounds: tuple[float, ...],
) -> WT103CornerInterval:
    return WT103CornerInterval.from_h6(
        inflate_paired_interval(
            effects,
            paired_half_widths,
            left_bias_bounds,
            right_bias_bounds,
        )
    )


def _result_rows(
    *,
    inventory: EndpointInventory,
    estimates: tuple[WT103SeedEstimate, ...],
    objective_status: GateStatus,
    primary_status: GateStatus,
) -> tuple[WT103ResultRow, ...]:
    rows: list[WT103ResultRow] = []
    for arm, result_row_key in zip(
        inventory.arms,
        inventory.result_row_keys,
        strict=True,
    ):
        arm_estimates = tuple(
            item for item in estimates if item.arm_id == arm.arm_id
        )
        seed_nll = tuple(item.nll_per_token for item in arm_estimates)
        seed_ppl = tuple(item.perplexity for item in arm_estimates)
        mean_nll = math.fsum(seed_nll) / len(seed_nll)
        mean_ppl = math.exp(mean_nll)
        if arm.result_role in ("PRIMARY_REFERENCE", "PRIMARY_ENDPOINT"):
            applicability = "decision_bearing"
            reason = "a0_minus_parent_specific_complete_primary_pair"
            status = primary_status
        elif arm.result_role == "OBJECTIVE_GATE":
            applicability = "decision_bearing"
            reason = "complete_minus_emission_objective_gate"
            status = objective_status
        else:
            applicability = "descriptive_only"
            reason = "control_is_reported_but_cannot_rescue_or_reverse_primary"
            status = (
                GateStatus.PASS
                if all(
                    item.smc is None or item.smc.eligible
                    for item in arm_estimates
                )
                else GateStatus.INCONCLUSIVE
            )
        payload = {
            "result_row_key": result_row_key,
            "arm_id": arm.arm_id,
            "scorer_kind": arm.scorer_kind,
            "applicability": applicability,
            "applicability_reason": reason,
            "result_role": arm.result_role,
            "seed_nll_per_token": seed_nll,
            "seed_perplexity": seed_ppl,
            "mean_nll_per_token": mean_nll,
            "mean_perplexity": mean_ppl,
            "status": status.value,
        }
        rows.append(
            WT103ResultRow(
                result_row_key=result_row_key,
                arm_id=arm.arm_id,
                scorer_kind=arm.scorer_kind,  # type: ignore[arg-type]
                applicability=applicability,  # type: ignore[arg-type]
                applicability_reason=reason,
                result_role=arm.result_role,
                seed_nll_per_token=seed_nll,
                seed_perplexity=seed_ppl,
                mean_nll_per_token=mean_nll,
                mean_perplexity=mean_ppl,
                status=status,
                row_sha256=owned_sha256(
                    "vfe4.wt103.result-row.v1",
                    payload,
                ),
            )
        )
    return tuple(rows)


def _incomplete_statistics(
    *,
    aggregation: WT103EstimatorAggregation,
    inventory: EndpointInventory,
) -> WT103PredictionStatistics:
    obligations = aggregation.obligations or (
        "estimator statistics are not eligible for a scientific decision",
    )
    payload = {
        "schema_version": "wt103-prediction-statistics-v1",
        "complete": False,
        "raw_record_count": aggregation.raw_record_count,
        "expected_raw_record_count": aggregation.expected_raw_record_count,
        "objective_status": GateStatus.INCONCLUSIVE.value,
        "primary_status": GateStatus.INCONCLUSIVE.value,
        "objective_interval_sha256": None,
        "primary_interval_sha256": None,
        "delta": PREDICTION_DELTA,
        "result_row_sha256s": (),
        "figure_series_keys": inventory.figure_series_keys,
        "obligations": obligations,
    }
    return WT103PredictionStatistics(
        schema_version="wt103-prediction-statistics-v1",
        complete=False,
        raw_record_count=aggregation.raw_record_count,
        expected_raw_record_count=aggregation.expected_raw_record_count,
        objective_status=GateStatus.INCONCLUSIVE,
        primary_status=GateStatus.INCONCLUSIVE,
        objective_interval=None,
        primary_interval=None,
        delta=PREDICTION_DELTA,
        arm_applicability=aggregation.arm_applicability,
        result_rows=(),
        figure_series_keys=inventory.figure_series_keys,
        obligations=obligations,
        statistics_sha256=owned_sha256(
            "vfe4.wt103.prediction-statistics.v1",
            payload,
        ),
    )


def paired_prediction_decision(
    records: tuple[WT103RawScoreRecord, ...],
    *,
    inventory: EndpointInventory,
) -> WT103PredictionStatistics:
    """Apply OBJECTIVE then PRIMARY using all eight 256-corner error boxes."""

    aggregation = aggregate_a5_smc(records, inventory=inventory)
    if not aggregation.complete or not aggregation.seed_estimates:
        return _incomplete_statistics(
            aggregation=aggregation,
            inventory=inventory,
        )
    by_arm_seed = {
        (item.arm_id, item.seed_id): item
        for item in aggregation.seed_estimates
    }
    objective_effects: list[float] = []
    objective_half_widths: list[float] = []
    objective_left_bias: list[float] = []
    objective_right_bias: list[float] = []
    primary_effects: list[float] = []
    primary_half_widths: list[float] = []
    primary_left_bias: list[float] = []
    primary_right_bias: list[float] = []
    # Descriptive control eligibility cannot rescue or block the two
    # decision-bearing gates.  Only each gate's associated endpoints enter its
    # eligibility conjunction.
    endpoint_eligible = True
    for seed in inventory.confirmatory_seed_ids:
        complete = by_arm_seed[(_OBJECTIVE_COMPLETE_ARM, seed)]
        emission = by_arm_seed[(_OBJECTIVE_EMISSION_ARM, seed)]
        reference = by_arm_seed[(_PRIMARY_REFERENCE_ARM, seed)]
        if complete.smc is None or emission.smc is None:
            raise RuntimeError("weighted decision arms lost their SMC records")
        objective_effects.append(
            complete.nll_per_token - emission.nll_per_token
        )
        objective_half_widths.append(
            _half_width(
                tuple(
                    left - right
                    for left, right in zip(
                        complete.smc.q2,
                        emission.smc.q2,
                        strict=True,
                    )
                )
            )
        )
        objective_left_bias.append(complete.smc.bias_bound)
        objective_right_bias.append(emission.smc.bias_bound)
        primary_effects.append(
            reference.nll_per_token - complete.nll_per_token
        )
        primary_half_widths.append(
            _half_width(
                tuple(
                    reference.nll_per_token - value
                    for value in complete.smc.q2
                )
            )
        )
        primary_left_bias.append(0.0)
        primary_right_bias.append(complete.smc.bias_bound)
        endpoint_eligible = (
            endpoint_eligible
            and complete.smc.eligible
            and emission.smc.eligible
        )

    objective_interval = _corner_interval(
        effects=tuple(objective_effects),
        paired_half_widths=tuple(objective_half_widths),
        left_bias_bounds=tuple(objective_left_bias),
        right_bias_bounds=tuple(objective_right_bias),
    )
    primary_interval = _corner_interval(
        effects=tuple(primary_effects),
        paired_half_widths=tuple(primary_half_widths),
        left_bias_bounds=tuple(primary_left_bias),
        right_bias_bounds=tuple(primary_right_bias),
    )
    objective_eligible = endpoint_eligible and objective_interval.eligible
    if not objective_eligible:
        objective_status = GateStatus.INCONCLUSIVE
        obligations = ["OBJECTIVE estimator interval is ineligible"]
    elif objective_interval.upper <= PREDICTION_DELTA:
        objective_status = GateStatus.PASS
        obligations = []
    elif objective_interval.lower > PREDICTION_DELTA:
        objective_status = GateStatus.FAIL
        obligations = []
    else:
        objective_status = GateStatus.INCONCLUSIVE
        obligations = [
            "OBJECTIVE interval does not cross its frozen decision boundary"
        ]

    primary_eligible = endpoint_eligible and primary_interval.eligible
    if objective_status is not GateStatus.PASS:
        primary_status = GateStatus.INCONCLUSIVE
        obligations.append("OBJECTIVE must PASS before PRIMARY")
    elif not primary_eligible:
        primary_status = GateStatus.INCONCLUSIVE
        obligations.append("PRIMARY estimator interval is ineligible")
    elif primary_interval.lower > PREDICTION_DELTA:
        primary_status = GateStatus.PASS
    elif primary_interval.upper <= 0.0:
        primary_status = GateStatus.FAIL
    else:
        primary_status = GateStatus.INCONCLUSIVE
        obligations.append(
            "PRIMARY interval does not cross a frozen decision boundary"
        )
    rows = _result_rows(
        inventory=inventory,
        estimates=aggregation.seed_estimates,
        objective_status=objective_status,
        primary_status=primary_status,
    )
    final_obligations = tuple(dict.fromkeys(obligations))
    payload = {
        "schema_version": "wt103-prediction-statistics-v1",
        "complete": True,
        "raw_record_count": aggregation.raw_record_count,
        "expected_raw_record_count": aggregation.expected_raw_record_count,
        "objective_status": objective_status.value,
        "primary_status": primary_status.value,
        "objective_interval_sha256": objective_interval.interval_sha256,
        "primary_interval_sha256": primary_interval.interval_sha256,
        "delta": PREDICTION_DELTA,
        "result_row_sha256s": tuple(row.row_sha256 for row in rows),
        "figure_series_keys": inventory.figure_series_keys,
        "obligations": final_obligations,
    }
    return WT103PredictionStatistics(
        schema_version="wt103-prediction-statistics-v1",
        complete=True,
        raw_record_count=aggregation.raw_record_count,
        expected_raw_record_count=aggregation.expected_raw_record_count,
        objective_status=objective_status,
        primary_status=primary_status,
        objective_interval=objective_interval,
        primary_interval=primary_interval,
        delta=PREDICTION_DELTA,
        arm_applicability=aggregation.arm_applicability,
        result_rows=rows,
        figure_series_keys=inventory.figure_series_keys,
        obligations=final_obligations,
        statistics_sha256=owned_sha256(
            "vfe4.wt103.prediction-statistics.v1",
            payload,
        ),
    )


__all__ = [
    "WT103ArmApplicability",
    "WT103CornerInterval",
    "WT103EstimatorAggregation",
    "WT103PredictionStatistics",
    "WT103RawScoreRecord",
    "WT103ResultRow",
    "WT103SeedEstimate",
    "aggregate_a5_smc",
    "paired_prediction_decision",
]
