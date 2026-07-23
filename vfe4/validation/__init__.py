"""Public H1 and H3 fixture validation interfaces."""

from .h3_fixture import (
    H3_COUPLED_FIXTURE_PATH,
    H3_COUPLED_SHA256,
    H3_EXPECTED_SHA256_BY_FIXTURE_ID,
    H3_ZERO_CONTROL_FIXTURE_PATH,
    H3_ZERO_CONTROL_SHA256,
    parse_h3_fixture_bytes,
    validate_independent_control,
)
from .h1_fixture import enumerate_source_paths, label_to_index, load_h1_fixture
from .h6_prefix import (
    AllInvalidSourceObservation,
    DynamicCheckResult,
    DynamicExecutionPlan,
    DynamicPrefixCase,
    DynamicPrefixReport,
    FrozenValidationPerturbations,
    MAX_FOCUSED_CASES,
    PairSideHarness,
    SourceMaskObservation,
    ValidationPerturbationRecord,
    load_frozen_validation_perturbations,
    observe_all_invalid_source_rejection,
    run_dynamic_prefix_checks,
)

__all__ = [
    "AllInvalidSourceObservation",
    "DynamicCheckResult",
    "DynamicExecutionPlan",
    "DynamicPrefixCase",
    "DynamicPrefixReport",
    "FrozenValidationPerturbations",
    "H3_COUPLED_FIXTURE_PATH",
    "H3_COUPLED_SHA256",
    "H3_EXPECTED_SHA256_BY_FIXTURE_ID",
    "H3_ZERO_CONTROL_FIXTURE_PATH",
    "H3_ZERO_CONTROL_SHA256",
    "MAX_FOCUSED_CASES",
    "PairSideHarness",
    "SourceMaskObservation",
    "ValidationPerturbationRecord",
    "enumerate_source_paths",
    "label_to_index",
    "load_h1_fixture",
    "load_frozen_validation_perturbations",
    "observe_all_invalid_source_rejection",
    "parse_h3_fixture_bytes",
    "validate_independent_control",
    "run_dynamic_prefix_checks",
]
