"""Literal H6 critical values and simultaneous finite-SMC error bounds."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

from vfe4.types.h6 import canonical_json_bytes


TRAINING_T_DF7 = 2.364624251592784
FINITE_T_DF511 = 4.0243186150882195
CHI2_LOWER_DF511 = 393.23185025997486
CHI2_UPPER_DF511 = 648.65591595794933
ENDPOINT_T_DF63 = 4.5144904535377144

FINITE_SMC_FIXTURE_COUNT = 4
FINITE_SMC_HORIZON = 6
FINITE_SMC_VOCABULARY_SIZE = 3
FINITE_SMC_TOKEN_CELL_COUNT = (
    FINITE_SMC_FIXTURE_COUNT
    * FINITE_SMC_HORIZON
    * FINITE_SMC_VOCABULARY_SIZE
)
FINITE_SMC_NORMALIZER_CELL_COUNT = FINITE_SMC_FIXTURE_COUNT
FINITE_SMC_CELL_COUNT = (
    FINITE_SMC_TOKEN_CELL_COUNT + FINITE_SMC_NORMALIZER_CELL_COUNT
)
FINITE_SMC_REPLICATE_COUNT = 512
FINITE_SMC_DEGREES_OF_FREEDOM = FINITE_SMC_REPLICATE_COUNT - 1
FINITE_SMC_TAIL_COUNT = 2 * FINITE_SMC_CELL_COUNT * 2
FINITE_SMC_TAIL_ALPHA = 0.01 / FINITE_SMC_TAIL_COUNT
FINITE_SMC_DELTA = 0.01005033585350145
FINITE_SMC_BIAS_LIMIT = 0.001005033585350145
FINITE_SMC_SD_LIMIT = 0.0025125839633753625


_CONSTANTS = (
    TRAINING_T_DF7,
    FINITE_T_DF511,
    CHI2_LOWER_DF511,
    CHI2_UPPER_DF511,
    ENDPOINT_T_DF63,
)

CRITICAL_VALUES_PROTOCOL = {
    "schema_version": "h6-critical-values-v1",
    "training_t_df7": TRAINING_T_DF7,
    "finite_t_df511": FINITE_T_DF511,
    "chi2_lower_df511": CHI2_LOWER_DF511,
    "chi2_upper_df511": CHI2_UPPER_DF511,
    "endpoint_t_df63": ENDPOINT_T_DF63,
    "finite_cells": FINITE_SMC_CELL_COUNT,
    "finite_replicates": FINITE_SMC_REPLICATE_COUNT,
    "finite_degrees_of_freedom": FINITE_SMC_DEGREES_OF_FREEDOM,
    "finite_tail_count": FINITE_SMC_TAIL_COUNT,
    "finite_tail_alpha": FINITE_SMC_TAIL_ALPHA,
    "unbiased_variance": True,
}
CRITICAL_VALUES_PROTOCOL_SHA256 = hashlib.sha256(
    b"VFE4-H6-CRITICAL-VALUES-PROTOCOL-V1\x00"
    + canonical_json_bytes(CRITICAL_VALUES_PROTOCOL)
).hexdigest()


def validate_critical_constants(
    *,
    training_t_df7: float = TRAINING_T_DF7,
    finite_t_df511: float = FINITE_T_DF511,
    chi2_lower_df511: float = CHI2_LOWER_DF511,
    chi2_upper_df511: float = CHI2_UPPER_DF511,
    endpoint_t_df63: float = ENDPOINT_T_DF63,
) -> None:
    observed = (
        training_t_df7,
        finite_t_df511,
        chi2_lower_df511,
        chi2_upper_df511,
        endpoint_t_df63,
    )
    if any(type(value) is not float for value in observed) or observed != _CONSTANTS:
        raise ValueError("critical constants differ from the frozen H6 literals")


@dataclass(frozen=True)
class SmcErrorBounds:
    replicate_count: int
    degrees_of_freedom: int
    mean_error: float
    sample_variance: float
    sample_sd: float
    mean_interval_lower: float
    mean_interval_upper: float
    lower_absolute_bias: float
    upper_absolute_bias: float
    variance_interval_lower: float
    variance_interval_upper: float
    lower_sd: float
    upper_sd: float

    def __post_init__(self) -> None:
        if (
            self.replicate_count != FINITE_SMC_REPLICATE_COUNT
            or self.degrees_of_freedom != FINITE_SMC_DEGREES_OF_FREEDOM
        ):
            raise ValueError("SMC bounds require the frozen 512-replicate inventory")
        if any(
            not math.isfinite(value)
            for value in (
                self.mean_error,
                self.sample_variance,
                self.sample_sd,
                self.mean_interval_lower,
                self.mean_interval_upper,
                self.lower_absolute_bias,
                self.upper_absolute_bias,
                self.variance_interval_lower,
                self.variance_interval_upper,
                self.lower_sd,
                self.upper_sd,
            )
        ):
            raise ValueError("SMC error bounds must be finite")
        if (
            self.sample_variance < 0.0
            or self.sample_sd < 0.0
            or self.lower_absolute_bias < 0.0
            or self.upper_absolute_bias < self.lower_absolute_bias
            or self.variance_interval_lower < 0.0
            or self.variance_interval_upper < self.variance_interval_lower
            or self.lower_sd < 0.0
            or self.upper_sd < self.lower_sd
        ):
            raise ValueError("SMC error bounds are not ordered")


def finite_smc_error_bounds(errors: Iterable[float]) -> SmcErrorBounds:
    """Apply the frozen joint t/chi-square construction to one 512-error cell."""

    values = tuple(errors)
    if (
        len(values) != FINITE_SMC_REPLICATE_COUNT
        or any(type(value) is not float or not math.isfinite(value) for value in values)
    ):
        raise ValueError("one SMC cell requires exactly 512 finite float errors")
    mean = math.fsum(values) / FINITE_SMC_REPLICATE_COUNT
    squared = math.fsum((value - mean) ** 2 for value in values)
    variance = squared / FINITE_SMC_DEGREES_OF_FREEDOM
    sample_sd = math.sqrt(variance)
    half_width = (
        FINITE_T_DF511
        * sample_sd
        / math.sqrt(FINITE_SMC_REPLICATE_COUNT)
    )
    mean_lower = mean - half_width
    mean_upper = mean + half_width
    upper_absolute_bias = max(abs(mean_lower), abs(mean_upper))
    if mean_lower <= 0.0 <= mean_upper:
        lower_absolute_bias = 0.0
    else:
        lower_absolute_bias = min(abs(mean_lower), abs(mean_upper))
    variance_lower = (
        FINITE_SMC_DEGREES_OF_FREEDOM
        * variance
        / CHI2_UPPER_DF511
    )
    variance_upper = (
        FINITE_SMC_DEGREES_OF_FREEDOM
        * variance
        / CHI2_LOWER_DF511
    )
    result = SmcErrorBounds(
        FINITE_SMC_REPLICATE_COUNT,
        FINITE_SMC_DEGREES_OF_FREEDOM,
        mean,
        variance,
        sample_sd,
        mean_lower,
        mean_upper,
        lower_absolute_bias,
        upper_absolute_bias,
        variance_lower,
        variance_upper,
        math.sqrt(variance_lower),
        math.sqrt(variance_upper),
    )
    result.__post_init__()
    return result


__all__ = [
    "CHI2_LOWER_DF511",
    "CHI2_UPPER_DF511",
    "CRITICAL_VALUES_PROTOCOL",
    "CRITICAL_VALUES_PROTOCOL_SHA256",
    "ENDPOINT_T_DF63",
    "FINITE_SMC_BIAS_LIMIT",
    "FINITE_SMC_CELL_COUNT",
    "FINITE_SMC_DEGREES_OF_FREEDOM",
    "FINITE_SMC_DELTA",
    "FINITE_SMC_FIXTURE_COUNT",
    "FINITE_SMC_HORIZON",
    "FINITE_SMC_NORMALIZER_CELL_COUNT",
    "FINITE_SMC_REPLICATE_COUNT",
    "FINITE_SMC_SD_LIMIT",
    "FINITE_SMC_TAIL_ALPHA",
    "FINITE_SMC_TAIL_COUNT",
    "FINITE_SMC_TOKEN_CELL_COUNT",
    "FINITE_SMC_VOCABULARY_SIZE",
    "FINITE_T_DF511",
    "SmcErrorBounds",
    "TRAINING_T_DF7",
    "finite_smc_error_bounds",
    "validate_critical_constants",
]
