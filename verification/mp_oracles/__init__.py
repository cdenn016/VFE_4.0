"""Independent arbitrary-precision verification oracles."""

from .h7_covariance import (
    H7_COMPLETE_LOCAL_TERM_IDS,
    H7_REQUIRED_TRIAL_IDS,
    H7MPOracleResult,
    H7OracleInconclusive,
    MPGaussHermiteRule,
    MPLawValues,
    MPProbeEvaluationRecord,
    MPScorerRowRecord,
    MPSourcePathRecord,
    MPTask5OracleComparison,
    MPTask5WiringResult,
    MPTrialResult,
    MPValueRecord,
    evaluate_h7_from_raw_bytes,
    evaluate_h7_task5_wiring,
    standard_normal_gauss_hermite,
)

__all__ = [
    "H7_COMPLETE_LOCAL_TERM_IDS",
    "H7_REQUIRED_TRIAL_IDS",
    "H7MPOracleResult",
    "H7OracleInconclusive",
    "MPGaussHermiteRule",
    "MPLawValues",
    "MPProbeEvaluationRecord",
    "MPScorerRowRecord",
    "MPSourcePathRecord",
    "MPTask5OracleComparison",
    "MPTask5WiringResult",
    "MPTrialResult",
    "MPValueRecord",
    "evaluate_h7_from_raw_bytes",
    "evaluate_h7_task5_wiring",
    "standard_normal_gauss_hermite",
]
