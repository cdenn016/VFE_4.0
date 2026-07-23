from __future__ import annotations

import math

import pytest

from vfe4.evaluation.smc_uncertainty import (
    EndpointSmcFailure,
    EndpointSmcObservation,
    aggregate_endpoint_smc,
)
from vfe4.types import EvidenceStatus


SHA_A = "a" * 64
PARTICLES = (128, 256, 512, 1024)
DF63_CRITICAL_VALUE = 4.5144904535377144


def _sample_variance(values: tuple[float, ...]) -> float:
    mean = math.fsum(values) / 64
    return math.fsum((value - mean) ** 2 for value in values) / 63


def _sample_covariance(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float:
    left_mean = math.fsum(left) / 64
    right_mean = math.fsum(right) / 64
    return math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    ) / 63


def test_one_checkpoint_64_by_4_aggregate_uses_frozen_arithmetic() -> None:
    observations = []
    expected_y = {particle_count: [] for particle_count in PARTICLES}
    offsets = {
        128: 8e-6,
        256: 6e-6,
        512: 3.5e-6,
        1024: 1.875e-6,
    }
    for replicate_id in range(64):
        q = 1.0 + (replicate_id - 31.5) * 1e-6
        for particle_count in PARTICLES:
            value = q + offsets[particle_count]
            expected_y[particle_count].append(value)
            observations.append(
                EndpointSmcObservation(
                    checkpoint_sha256=SHA_A,
                    replicate_id=replicate_id,
                    particle_count=particle_count,
                    nats_per_token=value,
                )
            )

    aggregate = aggregate_endpoint_smc(tuple(observations))
    q0 = tuple(
        2 * y256 - y128
        for y128, y256 in zip(
            expected_y[128], expected_y[256], strict=True
        )
    )
    q1 = tuple(
        2 * y512 - y256
        for y256, y512 in zip(
            expected_y[256], expected_y[512], strict=True
        )
    )
    q2 = tuple(
        2 * y1024 - y512
        for y512, y1024 in zip(
            expected_y[512], expected_y[1024], strict=True
        )
    )
    r1 = tuple(right - left for left, right in zip(q0, q1, strict=True))
    r2 = tuple(right - left for left, right in zip(q1, q2, strict=True))
    h = DF63_CRITICAL_VALUE * math.sqrt(_sample_variance(q2) / 64)
    u1 = abs(math.fsum(r1) / 64) + (
        DF63_CRITICAL_VALUE * math.sqrt(_sample_variance(r1) / 64)
    )
    u2 = abs(math.fsum(r2) / 64) + (
        DF63_CRITICAL_VALUE * math.sqrt(_sample_variance(r2) / 64)
    )

    assert aggregate.y[128] == pytest.approx(expected_y[128])
    assert aggregate.q0 == pytest.approx(q0)
    assert aggregate.q1 == pytest.approx(q1)
    assert aggregate.q2 == pytest.approx(q2)
    assert aggregate.r1 == pytest.approx(r1)
    assert aggregate.r2 == pytest.approx(r2)
    assert aggregate.q2_sample_variance == pytest.approx(
        _sample_variance(q2)
    )
    assert aggregate.r1_r2_sample_covariance == pytest.approx(
        _sample_covariance(r1, r2)
    )
    assert aggregate.u1 == pytest.approx(u1)
    assert aggregate.u2 == pytest.approx(u2)
    assert aggregate.bias_bound == pytest.approx(4 * u2)
    assert aggregate.half_width == pytest.approx(h)
    assert aggregate.reported_nll == pytest.approx(math.fsum(q2) / 64)
    assert aggregate.status is EvidenceStatus.PASS

    incomplete = aggregate_endpoint_smc(tuple(observations[:-1]))
    assert type(incomplete) is EndpointSmcFailure
    assert incomplete.status is EvidenceStatus.INCONCLUSIVE
    assert incomplete.failure_kinds == ("missing_observations",)
