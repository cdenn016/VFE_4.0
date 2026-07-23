from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import pytest

import vfe4.numerics.critical_values as critical_values
from vfe4.numerics.critical_values import (
    CHI2_LOWER_DF511,
    CHI2_UPPER_DF511,
    ENDPOINT_T_DF63,
    FINITE_SMC_BIAS_LIMIT,
    FINITE_SMC_CELL_COUNT,
    FINITE_SMC_REPLICATE_COUNT,
    FINITE_SMC_SD_LIMIT,
    FINITE_SMC_TAIL_ALPHA,
    FINITE_T_DF511,
    TRAINING_T_DF7,
    finite_smc_error_bounds,
    validate_critical_constants,
)
from verification.h6_smc_gate import (
    SMC_VALIDATION_RELATIVE_PATH,
    classify_smc_bounds,
    finite_gate_inventory,
)


FIXTURE_SHA256 = "97d323a3d020d5f54d250b2d3569344f1348da9bb16c2bed4e79d07b82119db0"


def test_frozen_critical_values_and_inventory_are_literal_and_scipy_free() -> None:
    assert TRAINING_T_DF7 == 2.364624251592784
    assert FINITE_T_DF511 == 4.0243186150882195
    assert CHI2_LOWER_DF511 == 393.23185025997486
    assert CHI2_UPPER_DF511 == 648.65591595794933
    assert ENDPOINT_T_DF63 == 4.5144904535377144
    assert FINITE_SMC_TAIL_ALPHA == pytest.approx(0.01 / 304.0, rel=0, abs=0)
    assert FINITE_SMC_REPLICATE_COUNT == 512
    assert FINITE_SMC_CELL_COUNT == 4 * 6 * 3 + 4 == 76
    assert finite_gate_inventory() == {
        "fixture_count": 4,
        "horizon": 6,
        "vocabulary_size": 3,
        "token_cells": 72,
        "normalizer_cells": 4,
        "cell_count": 76,
        "replicate_count": 512,
        "degrees_of_freedom": 511,
        "tail_count": 304,
    }
    assert SMC_VALIDATION_RELATIVE_PATH == "validation/h6_smc_accuracy.json"
    assert "scipy" not in inspect.getsource(critical_values).lower()

    fixture = (
        Path(__file__).resolve().parents[2]
        / "verification"
        / "fixtures"
        / "h6_critical_values_v1.json"
    )
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == FIXTURE_SHA256
    validate_critical_constants()
    with pytest.raises(ValueError, match="critical constants"):
        validate_critical_constants(
            finite_t_df511=math.nextafter(FINITE_T_DF511, math.inf)
        )


def test_simultaneous_bounds_use_df511_and_frozen_status_boundaries() -> None:
    exact_zero = finite_smc_error_bounds((0.0,) * 512)
    assert exact_zero.sample_variance == 0.0
    assert exact_zero.upper_absolute_bias == 0.0
    assert exact_zero.upper_sd == 0.0
    assert (
        classify_smc_bounds((exact_zero,) * FINITE_SMC_CELL_COUNT) == "PASS"
    )

    definite_bias = finite_smc_error_bounds(
        (2.0 * FINITE_SMC_BIAS_LIMIT,) * 512
    )
    assert definite_bias.lower_absolute_bias > FINITE_SMC_BIAS_LIMIT
    assert (
        classify_smc_bounds((definite_bias,) * FINITE_SMC_CELL_COUNT)
        == "FAIL"
    )

    alternating = tuple(
        FINITE_SMC_SD_LIMIT if index % 2 else -FINITE_SMC_SD_LIMIT
        for index in range(512)
    )
    uncertain = finite_smc_error_bounds(alternating)
    assert uncertain.lower_sd < FINITE_SMC_SD_LIMIT < uncertain.upper_sd
    assert (
        classify_smc_bounds((uncertain,) * FINITE_SMC_CELL_COUNT)
        == "INCONCLUSIVE"
    )
