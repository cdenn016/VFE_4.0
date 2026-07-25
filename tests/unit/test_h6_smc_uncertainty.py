from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest

from vfe4.evaluation.smc_uncertainty import (
    EndpointSmcFailure,
    EndpointSmcObservation,
    SmcBiasSemantics,
    aggregate_endpoint_smc,
)
from vfe4.types import EvidenceStatus
from verification.numpy_oracles.h6_linear_gaussian_smc import (
    LinearGaussianSmcRawRecord,
    build_continuous_sensitivity_report,
    deterministic_linear_gaussian_fixture,
    kalman_log_likelihood,
)
from verification.h6_continuous_smc_gate import (
    run_tiny_continuous_smc_gate,
)


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


def _stream_sha256(replicate_id: int) -> str:
    return hashlib.sha256(
        f"h6-test-common-stream-{replicate_id}".encode("ascii")
    ).hexdigest()


def test_raw_jensen_sign_does_not_claim_a_q2_sign() -> None:
    semantics = SmcBiasSemantics()

    assert (
        semantics.raw_log_normalizer_direction
        == "downward_or_equal_in_expectation"
    )
    assert semantics.raw_nll_direction == "upward_or_equal_in_expectation"
    assert semantics.reported_endpoint == "Q2=2Y_1024-Y_512"
    assert (
        semantics.q2_remainder_direction
        == "unknown_without_signed_expansion"
    )
    assert (
        semantics.q2_bound_kind
        == "two_sided_conditional_geometric_remainder"
    )
    assert len(semantics.semantics_sha256) == 64

    with pytest.raises(ValueError, match="Q2 remainder direction"):
        SmcBiasSemantics(q2_remainder_direction="downward")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Q2 bound kind"):
        SmcBiasSemantics(q2_bound_kind="one_sided")  # type: ignore[arg-type]

    fixture = deterministic_linear_gaussian_fixture(
        time_steps=3,
        dimension=2,
    )
    exact = kalman_log_likelihood(fixture)
    raw_records = (
        LinearGaussianSmcRawRecord(0, 8, exact - 0.04),
        LinearGaussianSmcRawRecord(0, 16, exact - 0.02),
        LinearGaussianSmcRawRecord(1, 8, exact - 0.02),
        LinearGaussianSmcRawRecord(1, 16, exact - 0.01),
    )
    report = build_continuous_sensitivity_report(
        fixture=fixture,
        raw_records=raw_records,
        lower_particle_count=8,
        upper_particle_count=16,
        q2_two_sided_radius=1e-12,
    )

    assert report.q2_estimates == pytest.approx((exact, exact))
    assert report.q2_interval_contains_exact is True
    assert (
        report.q2_remainder_direction
        == "unknown_without_signed_expansion"
    )
    assert report.q2_bound_kind == "two_sided_sensitivity_envelope"
    assert (
        report.transfer_claim
        == "sensitivity_control_not_trained_endpoint_bound"
    )
    forged_lower = exact - 0.08
    forged_upper = exact - 0.04
    forged_q2 = replace(
        report.q2_records[0],
        lower_raw_log_likelihood=forged_lower,
        upper_raw_log_likelihood=forged_upper,
        q2_log_likelihood=2.0 * forged_upper - forged_lower,
    )
    with pytest.raises(ValueError, match="derived from raw_records"):
        replace(
            report,
            q2_records=(forged_q2, report.q2_records[1]),
        )

    gate = run_tiny_continuous_smc_gate(fixture)
    assert gate.config.particle_counts == (8, 16)
    assert len(gate.config.stream_seeds) == 2
    assert gate.config.rng_identity == "numpy.random.Philox"
    assert (
        gate.config.algorithm_identity
        == "bootstrap-carried-weight-systematic-ess-half-common-philox-v1"
    )
    assert len(gate.config.gate_source_sha256) == 64
    assert len(gate.config.oracle_source_sha256) == 64
    assert len(gate.raw_records) == 4
    assert gate.contraction_evaluated is False
    assert gate.contraction_eligible is None
    assert gate.q2_bound_kind == "two_sided_sensitivity_envelope"
    assert (
        gate.transfer_claim
        == "sensitivity_control_not_trained_endpoint_bound"
    )
    assert gate.status == "INCONCLUSIVE"


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
                    common_stream_sha256=_stream_sha256(replicate_id),
                    negative_log_likelihood_sum=value * 100.0,
                    counted_targets=100,
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
    assert aggregate.y_means[128] == pytest.approx(
        math.fsum(expected_y[128]) / 64
    )
    assert aggregate.y_sample_variances[128] == pytest.approx(
        _sample_variance(tuple(expected_y[128]))
    )
    assert aggregate.y_cross_level_sample_covariances[(128, 256)] == (
        pytest.approx(
            _sample_covariance(
                tuple(expected_y[128]), tuple(expected_y[256])
            )
        )
    )
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

    wrong_stream = list(observations)
    wrong_stream[-1] = replace(
        wrong_stream[-1], common_stream_sha256="f" * 64
    )
    identity_failure = aggregate_endpoint_smc(tuple(wrong_stream))
    assert type(identity_failure) is EndpointSmcFailure
    assert identity_failure.status is EvidenceStatus.FAIL
    assert identity_failure.failure_kinds == (
        "common_stream_identity_mismatch",
    )
