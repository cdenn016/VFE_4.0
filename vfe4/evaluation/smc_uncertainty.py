"""Bounded H6 endpoint-SMC and paired-seed uncertainty arithmetic."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from vfe4.numerics.critical_values import (
    ENDPOINT_T_DF63,
    TRAINING_T_DF7,
)
from vfe4.types import EvidenceStatus, PredictionDecision


ENDPOINT_PARTICLE_COUNTS = (128, 256, 512, 1024)
ENDPOINT_REPLICATE_COUNT = 64
ENDPOINT_DEGREES_OF_FREEDOM = 63
ENDPOINT_REMAINDER_CONTRACTION = 0.75
PREDICTION_DELTA = 0.01005033585350145
ENDPOINT_BIAS_LIMIT = 0.00025125839633753625
ENDPOINT_HALF_WIDTH_LIMIT = 0.0005025167926750725
PAIRED_ERROR_RADIUS_LIMIT = 0.001005033585350145
PAIRED_SEED_COUNT = 8
PAIRED_DEGREES_OF_FREEDOM = 7
PAIRED_CORNER_COUNT = 256


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite binary64 float")
    return value


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(
    values: tuple[float, ...],
    *,
    mean: float,
    degrees_of_freedom: int,
) -> float:
    return math.fsum((value - mean) ** 2 for value in values) / (
        degrees_of_freedom
    )


def _sample_covariance(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    left_mean = _mean(left)
    right_mean = _mean(right)
    return math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / ENDPOINT_DEGREES_OF_FREEDOM


def _endpoint_half_width(values: tuple[float, ...]) -> float:
    mean = _mean(values)
    variance = _sample_variance(
        values,
        mean=mean,
        degrees_of_freedom=ENDPOINT_DEGREES_OF_FREEDOM,
    )
    return ENDPOINT_T_DF63 * math.sqrt(
        variance / ENDPOINT_REPLICATE_COUNT
    )


@dataclass(frozen=True)
class EndpointSmcObservation:
    checkpoint_sha256: str
    replicate_id: int
    particle_count: int
    nats_per_token: float

    def __post_init__(self) -> None:
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
        if (
            type(self.replicate_id) is not int
            or not 0 <= self.replicate_id < ENDPOINT_REPLICATE_COUNT
        ):
            raise ValueError("replicate_id must be an exact integer in 0..63")
        if (
            type(self.particle_count) is not int
            or self.particle_count not in ENDPOINT_PARTICLE_COUNTS
        ):
            raise ValueError("particle_count is outside the frozen ladder")
        if type(self.nats_per_token) is not float:
            raise ValueError("nats_per_token must be a binary64 float")


@dataclass(frozen=True)
class EndpointSmcAggregate:
    checkpoint_sha256: str
    y: Mapping[int, tuple[float, ...]]
    q0: tuple[float, ...]
    q1: tuple[float, ...]
    q2: tuple[float, ...]
    r1: tuple[float, ...]
    r2: tuple[float, ...]
    q2_sample_variance: float
    r1_r2_sample_covariance: float
    u1: float
    u2: float
    bias_bound: float
    half_width: float
    reported_nll: float
    contraction_eligible: bool
    bias_eligible: bool
    half_width_eligible: bool
    eligible: bool
    status: EvidenceStatus
    obligations: tuple[str, ...]


@dataclass(frozen=True)
class EndpointSmcFailure:
    status: EvidenceStatus
    eligible: bool
    failure_kinds: tuple[str, ...]
    checkpoint_sha256s: tuple[str, ...]
    observation_count: int
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in (
            EvidenceStatus.FAIL,
            EvidenceStatus.INCONCLUSIVE,
        ):
            raise ValueError("endpoint-SMC failure must be FAIL or INCONCLUSIVE")
        if self.eligible is not False:
            raise ValueError("endpoint-SMC failure cannot be eligible")
        if (
            type(self.failure_kinds) is not tuple
            or not self.failure_kinds
            or any(type(item) is not str or not item for item in self.failure_kinds)
        ):
            raise ValueError("endpoint-SMC failure kinds must be nonempty")
        if (
            type(self.checkpoint_sha256s) is not tuple
            or any(not _is_sha256(item) for item in self.checkpoint_sha256s)
        ):
            raise ValueError("checkpoint identities must be lowercase SHA-256")
        if type(self.observation_count) is not int or self.observation_count < 0:
            raise ValueError("observation_count must be nonnegative")
        if self.status is EvidenceStatus.FAIL:
            if self.obligations:
                raise ValueError(
                    "witnessed endpoint identity failure has no obligation"
                )
        elif not self.obligations:
            raise ValueError("INCONCLUSIVE endpoint failure needs an obligation")


def _endpoint_failure(
    *,
    status: EvidenceStatus,
    failure_kinds: tuple[str, ...],
    checkpoint_sha256s: tuple[str, ...],
    observation_count: int,
) -> EndpointSmcFailure:
    obligations = (
        ()
        if status is EvidenceStatus.FAIL
        else tuple(
            {
                "missing_observations": (
                    "complete the exact 64-by-4 endpoint inventory"
                ),
                "duplicate_observations": (
                    "resolve duplicate stream/particle observations"
                ),
                "nonfinite_observations": (
                    "replace nonfinite endpoint observations"
                ),
                "nonfinite_derived_values": (
                    "resolve nonfinite endpoint uncertainty arithmetic"
                ),
            }[kind]
            for kind in failure_kinds
        )
    )
    return EndpointSmcFailure(
        status=status,
        eligible=False,
        failure_kinds=failure_kinds,
        checkpoint_sha256s=checkpoint_sha256s,
        observation_count=observation_count,
        obligations=obligations,
    )


def aggregate_endpoint_smc(
    observations: Iterable[EndpointSmcObservation],
) -> EndpointSmcAggregate | EndpointSmcFailure:
    """Aggregate exactly one 64-stream by four-particle checkpoint table."""

    records = tuple(observations)
    if any(type(record) is not EndpointSmcObservation for record in records):
        raise ValueError("all observations must be exact EndpointSmcObservation")
    checkpoint_ids = tuple(
        sorted({record.checkpoint_sha256 for record in records})
    )
    if len(checkpoint_ids) > 1:
        return _endpoint_failure(
            status=EvidenceStatus.FAIL,
            failure_kinds=("mixed_checkpoint_identity",),
            checkpoint_sha256s=checkpoint_ids,
            observation_count=len(records),
        )
    keyed: dict[tuple[int, int], float] = {}
    duplicate_keys: set[tuple[int, int]] = set()
    nonfinite_keys: set[tuple[int, int]] = set()
    for record in records:
        key = (record.replicate_id, record.particle_count)
        if key in keyed:
            duplicate_keys.add(key)
        else:
            keyed[key] = record.nats_per_token
        if not math.isfinite(record.nats_per_token):
            nonfinite_keys.add(key)
    expected_keys = {
        (replicate_id, particle_count)
        for replicate_id in range(ENDPOINT_REPLICATE_COUNT)
        for particle_count in ENDPOINT_PARTICLE_COUNTS
    }
    failure_kinds: list[str] = []
    if set(keyed) != expected_keys:
        failure_kinds.append("missing_observations")
    if duplicate_keys:
        failure_kinds.append("duplicate_observations")
    if nonfinite_keys:
        failure_kinds.append("nonfinite_observations")
    if failure_kinds:
        return _endpoint_failure(
            status=EvidenceStatus.INCONCLUSIVE,
            failure_kinds=tuple(failure_kinds),
            checkpoint_sha256s=checkpoint_ids,
            observation_count=len(records),
        )

    y_mutable = {
        particle_count: tuple(
            keyed[(replicate_id, particle_count)]
            for replicate_id in range(ENDPOINT_REPLICATE_COUNT)
        )
        for particle_count in ENDPOINT_PARTICLE_COUNTS
    }
    y: Mapping[int, tuple[float, ...]] = MappingProxyType(y_mutable)
    q0 = tuple(
        2.0 * y256 - y128
        for y128, y256 in zip(y[128], y[256], strict=True)
    )
    q1 = tuple(
        2.0 * y512 - y256
        for y256, y512 in zip(y[256], y[512], strict=True)
    )
    q2 = tuple(
        2.0 * y1024 - y512
        for y512, y1024 in zip(y[512], y[1024], strict=True)
    )
    r1 = tuple(
        right - left for left, right in zip(q0, q1, strict=True)
    )
    r2 = tuple(
        right - left for left, right in zip(q1, q2, strict=True)
    )
    derived = q0 + q1 + q2 + r1 + r2
    if not all(math.isfinite(value) for value in derived):
        return _endpoint_failure(
            status=EvidenceStatus.INCONCLUSIVE,
            failure_kinds=("nonfinite_derived_values",),
            checkpoint_sha256s=checkpoint_ids,
            observation_count=len(records),
        )

    q2_mean = _mean(q2)
    q2_variance = _sample_variance(
        q2,
        mean=q2_mean,
        degrees_of_freedom=ENDPOINT_DEGREES_OF_FREEDOM,
    )
    r1_mean = _mean(r1)
    r2_mean = _mean(r2)
    u1 = abs(r1_mean) + _endpoint_half_width(r1)
    u2 = abs(r2_mean) + _endpoint_half_width(r2)
    bias_bound = u2 / (1.0 - ENDPOINT_REMAINDER_CONTRACTION)
    half_width = ENDPOINT_T_DF63 * math.sqrt(
        q2_variance / ENDPOINT_REPLICATE_COUNT
    )
    contraction_eligible = (
        u2 <= ENDPOINT_REMAINDER_CONTRACTION * u1
    )
    bias_eligible = bias_bound <= ENDPOINT_BIAS_LIMIT
    half_width_eligible = half_width <= ENDPOINT_HALF_WIDTH_LIMIT
    eligible = (
        contraction_eligible and bias_eligible and half_width_eligible
    )
    obligations: list[str] = []
    if not contraction_eligible:
        obligations.append("endpoint remainder contraction did not close")
    if not bias_eligible:
        obligations.append("endpoint conditional bias bound exceeded")
    if not half_width_eligible:
        obligations.append("endpoint random half-width exceeded")

    return EndpointSmcAggregate(
        checkpoint_sha256=checkpoint_ids[0],
        y=y,
        q0=q0,
        q1=q1,
        q2=q2,
        r1=r1,
        r2=r2,
        q2_sample_variance=q2_variance,
        r1_r2_sample_covariance=_sample_covariance(r1, r2),
        u1=u1,
        u2=u2,
        bias_bound=bias_bound,
        half_width=half_width,
        reported_nll=q2_mean,
        contraction_eligible=contraction_eligible,
        bias_eligible=bias_eligible,
        half_width_eligible=half_width_eligible,
        eligible=eligible,
        status=(
            EvidenceStatus.PASS
            if eligible
            else EvidenceStatus.INCONCLUSIVE
        ),
        obligations=tuple(obligations),
    )


@dataclass(frozen=True)
class PairedInterval:
    values: tuple[float, ...]
    mean: float
    sample_variance: float
    half_width: float
    lower: float
    upper: float


def paired_t_interval(values: Iterable[float]) -> PairedInterval:
    """Return the frozen ordinary df=7 interval for exactly eight seeds."""

    owned = tuple(values)
    if len(owned) != PAIRED_SEED_COUNT:
        raise ValueError("paired interval requires exactly eight seed values")
    for value in owned:
        _finite_float(value, "paired seed value")
    mean = _mean(owned)
    variance = _sample_variance(
        owned,
        mean=mean,
        degrees_of_freedom=PAIRED_DEGREES_OF_FREEDOM,
    )
    half_width = TRAINING_T_DF7 * math.sqrt(
        variance / PAIRED_SEED_COUNT
    )
    return PairedInterval(
        values=owned,
        mean=mean,
        sample_variance=variance,
        half_width=half_width,
        lower=mean - half_width,
        upper=mean + half_width,
    )


@dataclass(frozen=True)
class InflatedPairedInterval:
    values: tuple[float, ...]
    half_widths: tuple[float, ...]
    error_radii: tuple[float, ...]
    uninflated: PairedInterval
    corner_intervals: tuple[PairedInterval, ...]
    lower: float
    upper: float
    eligible: bool
    status: EvidenceStatus
    obligations: tuple[str, ...]


def inflate_paired_interval(
    values: Iterable[float],
    half_widths: Iterable[float],
    error_radii: Iterable[float],
) -> InflatedPairedInterval:
    """Envelope all 256 df=7 intervals over the eight error-box corners."""

    owned_values = tuple(values)
    owned_half_widths = tuple(half_widths)
    owned_radii = tuple(error_radii)
    if (
        len(owned_values) != PAIRED_SEED_COUNT
        or len(owned_half_widths) != PAIRED_SEED_COUNT
        or len(owned_radii) != PAIRED_SEED_COUNT
    ):
        raise ValueError(
            "interval inflation requires eight values, half-widths, and radii"
        )
    for value in owned_values:
        _finite_float(value, "paired seed value")
    for half_width in owned_half_widths:
        _finite_float(half_width, "paired estimator half-width")
        if half_width < 0.0:
            raise ValueError("paired estimator half-widths must be nonnegative")
    for radius in owned_radii:
        _finite_float(radius, "paired error radius")
        if radius < 0.0:
            raise ValueError("paired error radii must be nonnegative")

    corner_intervals = tuple(
        paired_t_interval(
            tuple(
                value + sign * radius
                for value, radius, sign in zip(
                    owned_values,
                    owned_radii,
                    signs,
                    strict=True,
                )
            )
        )
        for signs in itertools.product((-1.0, 1.0), repeat=PAIRED_SEED_COUNT)
    )
    if len(corner_intervals) != PAIRED_CORNER_COUNT:
        raise RuntimeError("paired corner enumeration is incomplete")
    half_width_eligible = all(
        half_width <= ENDPOINT_HALF_WIDTH_LIMIT
        for half_width in owned_half_widths
    )
    radius_eligible = all(
        radius <= PAIRED_ERROR_RADIUS_LIMIT for radius in owned_radii
    )
    eligible = half_width_eligible and radius_eligible
    obligations: list[str] = []
    if not half_width_eligible:
        obligations.append("paired estimator half-width exceeded")
    if not radius_eligible:
        obligations.append("paired estimator error radius exceeded")
    return InflatedPairedInterval(
        values=owned_values,
        half_widths=owned_half_widths,
        error_radii=owned_radii,
        uninflated=paired_t_interval(owned_values),
        corner_intervals=corner_intervals,
        lower=min(interval.lower for interval in corner_intervals),
        upper=max(interval.upper for interval in corner_intervals),
        eligible=eligible,
        status=(
            EvidenceStatus.PASS
            if eligible
            else EvidenceStatus.INCONCLUSIVE
        ),
        obligations=tuple(obligations),
    )


def decide_primary_prediction(
    inflated_interval: InflatedPairedInterval,
    *,
    estimator_complete: bool,
) -> PredictionDecision:
    """Apply the frozen PRIMARY boundaries only to an inflated interval."""

    if type(inflated_interval) is not InflatedPairedInterval:
        raise ValueError("PRIMARY decision requires an inflated interval")
    if type(estimator_complete) is not bool:
        raise ValueError("estimator_complete must be boolean")
    complete = estimator_complete and inflated_interval.eligible
    return PredictionDecision.classify(
        primary_interval=(
            float(inflated_interval.lower),
            float(inflated_interval.upper),
        ),
        estimator_complete=complete,
    )


__all__ = [
    "ENDPOINT_BIAS_LIMIT",
    "ENDPOINT_DEGREES_OF_FREEDOM",
    "ENDPOINT_HALF_WIDTH_LIMIT",
    "ENDPOINT_PARTICLE_COUNTS",
    "ENDPOINT_REMAINDER_CONTRACTION",
    "ENDPOINT_REPLICATE_COUNT",
    "EndpointSmcAggregate",
    "EndpointSmcFailure",
    "EndpointSmcObservation",
    "InflatedPairedInterval",
    "PAIRED_CORNER_COUNT",
    "PAIRED_DEGREES_OF_FREEDOM",
    "PAIRED_ERROR_RADIUS_LIMIT",
    "PAIRED_SEED_COUNT",
    "PREDICTION_DELTA",
    "PairedInterval",
    "aggregate_endpoint_smc",
    "decide_primary_prediction",
    "inflate_paired_interval",
    "paired_t_interval",
]
