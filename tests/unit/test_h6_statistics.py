from __future__ import annotations

import itertools
import math

import pytest

from vfe4.evaluation.smc_uncertainty import (
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
    error_radii = (0.0001,) * 8

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
        error_radii,
    )

    assert len(inflated.corner_intervals) == 2**8 == 256
    assert inflated.lower == pytest.approx(
        min(lower for lower, _ in expected_intervals)
    )
    assert inflated.upper == pytest.approx(
        max(upper for _, upper in expected_intervals)
    )
    assert inflated.half_widths == half_widths
    decision = decide_primary_prediction(
        inflated,
        estimator_complete=True,
    )
    assert decision.status is EvidenceStatus.PASS
