from __future__ import annotations

import itertools
import math

import pytest

from vfe4.evaluation.smc_uncertainty import (
    H6_OBJECTIVE_GATE_SPEC,
    decide_objective_gate,
    decide_primary_prediction,
    inflate_paired_interval,
    paired_t_interval,
)
from vfe4.types import EvidenceStatus


DF7_CRITICAL_VALUE = 2.364624251592784


def _expected_interval(values: tuple[float, ...]) -> tuple[float, float]:
    mean = math.fsum(values) / 8
    variance = math.fsum((value - mean) ** 2 for value in values) / 7
    half_width = DF7_CRITICAL_VALUE * math.sqrt(variance / 8)
    return mean - half_width, mean + half_width


def test_eight_seed_interval_inflates_over_all_256_scalar_corners() -> None:
    values = tuple(
        0.01205033585350145 + (index - 3.5) * 0.0001
        for index in range(8)
    )
    half_widths = (0.00004,) * 8
    left_bias_bounds = (0.00002,) * 8
    right_bias_bounds = (0.00004,) * 8
    error_radii = tuple(
        math.fsum(items)
        for items in zip(
            half_widths,
            left_bias_bounds,
            right_bias_bounds,
            strict=True,
        )
    )

    interval = paired_t_interval(values)
    expected_lower, expected_upper = _expected_interval(values)
    assert interval.mean == pytest.approx(math.fsum(values) / 8)
    assert interval.lower == pytest.approx(expected_lower)
    assert interval.upper == pytest.approx(expected_upper)

    expected_corners = tuple(
        tuple(
            value + sign * radius
            for value, radius, sign in zip(
                values, error_radii, signs, strict=True
            )
        )
        for signs in itertools.product((-1.0, 1.0), repeat=8)
    )
    expected_intervals = tuple(
        _expected_interval(corner) for corner in expected_corners
    )
    inflated = inflate_paired_interval(
        values,
        half_widths,
        left_bias_bounds,
        right_bias_bounds,
    )

    assert len(inflated.corner_intervals) == 2**8 == 256
    assert inflated.lower == pytest.approx(
        min(lower for lower, _ in expected_intervals)
    )
    assert inflated.upper == pytest.approx(
        max(upper for _, upper in expected_intervals)
    )
    assert inflated.paired_half_widths == half_widths
    assert inflated.error_radii == error_radii
    decision = decide_primary_prediction(
        inflated,
        estimator_complete=True,
    )
    assert decision.status is EvidenceStatus.PASS

    object.__setattr__(inflated, "lower", inflated.lower + 1.0)
    with pytest.raises(ValueError, match="inconsistent"):
        decide_primary_prediction(inflated, estimator_complete=True)


def test_objective_gate_direction_and_exact_margin_boundaries() -> None:
    delta = 0.01005033585350145
    spec = H6_OBJECTIVE_GATE_SPEC
    assert (
        spec.schema_version,
        spec.complete_arm_id,
        spec.emission_arm_id,
        spec.orientation,
        spec.delta_obj,
        spec.opening_policy,
        spec.evaluation_order,
    ) == (
        "h6-objective-gate-v1",
        (
            "h6-a5-structured-parent-specific-prefix-exact-complete-"
            "latent-smoothing-v2"
        ),
        (
            "h6-a5-structured-parent-specific-prefix-exact-emission-"
            "latent-smoothing-v2"
        ),
        "nll_complete_minus_nll_emission",
        delta,
        "single_all_or_none",
        "OBJECTIVE_then_PRIMARY",
    )
    assert len(spec.spec_sha256) == 64

    def interval(
        value: float,
        *,
        error_radius: float = 0.0,
    ):
        return inflate_paired_interval(
            (value,) * 8,
            (error_radius,) * 8,
            (0.0,) * 8,
            (0.0,) * 8,
        )

    boundary = decide_objective_gate(
        interval(delta),
        spec,
        estimator_complete=True,
    )
    assert boundary.status is EvidenceStatus.PASS
    assert boundary.objective_interval == (delta, delta)

    failed = decide_objective_gate(
        interval(math.nextafter(delta, math.inf)),
        spec,
        estimator_complete=True,
    )
    assert failed.status is EvidenceStatus.FAIL

    crossing = decide_objective_gate(
        interval(delta, error_radius=1e-4),
        spec,
        estimator_complete=True,
    )
    assert crossing.status is EvidenceStatus.INCONCLUSIVE
    assert crossing.obligations == (
        "objective interval does not cross the frozen upper-bound decision",
    )

    incomplete = decide_objective_gate(
        interval(delta),
        spec,
        estimator_complete=False,
    )
    assert incomplete.status is EvidenceStatus.INCONCLUSIVE
    assert incomplete.obligations == (
        "objective actual-endpoint estimator evidence incomplete",
    )

    ineligible = decide_objective_gate(
        interval(delta, error_radius=0.02),
        spec,
        estimator_complete=True,
    )
    assert ineligible.status is EvidenceStatus.INCONCLUSIVE
    assert ineligible.obligations == (
        "objective estimator interval is ineligible",
    )
