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
from .h7_fixture import (
    H1_FIXTURE_RAW_SHA256,
    H7_DENSITY_PROBE_EXPANSION,
    H7_DENSITY_PROBE_SET_SHA256,
    H7_DENSITY_PROBE_TABLE_PATH,
    H7_DENSITY_PROBE_TABLE_RAW_SHA256,
    H7_FIXTURE_PATH,
    H7_FIXTURE_RAW_SHA256,
    adapt_optional_h1_fixture_bytes,
    h7_scalar_trial_specs,
    h7_trial_specs_from_config,
    h7_validation_config_mapping,
    parse_h7_fixture_bytes,
)

_H6_PREFIX_EXPORTS = frozenset(
    {
        "AllInvalidSourceObservation",
        "DynamicCheckResult",
        "DynamicExecutionPlan",
        "DynamicPrefixCase",
        "DynamicPrefixReport",
        "FrozenValidationPerturbations",
        "MAX_FOCUSED_CASES",
        "PairSideHarness",
        "SourceMaskObservation",
        "ValidationPerturbationRecord",
        "load_frozen_validation_perturbations",
        "observe_all_invalid_source_rejection",
        "run_dynamic_prefix_checks",
    }
)


def __getattr__(name: str) -> object:
    """Load H6 validation surfaces lazily to keep config imports acyclic."""

    if name in _H6_PREFIX_EXPORTS:
        from . import h6_prefix

        return getattr(h6_prefix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "H1_FIXTURE_RAW_SHA256",
    "H7_DENSITY_PROBE_EXPANSION",
    "H7_DENSITY_PROBE_SET_SHA256",
    "H7_DENSITY_PROBE_TABLE_PATH",
    "H7_DENSITY_PROBE_TABLE_RAW_SHA256",
    "H7_FIXTURE_PATH",
    "H7_FIXTURE_RAW_SHA256",
    "MAX_FOCUSED_CASES",
    "PairSideHarness",
    "SourceMaskObservation",
    "ValidationPerturbationRecord",
    "enumerate_source_paths",
    "adapt_optional_h1_fixture_bytes",
    "h7_scalar_trial_specs",
    "h7_trial_specs_from_config",
    "h7_validation_config_mapping",
    "label_to_index",
    "load_h1_fixture",
    "load_frozen_validation_perturbations",
    "observe_all_invalid_source_rejection",
    "parse_h3_fixture_bytes",
    "parse_h7_fixture_bytes",
    "validate_independent_control",
    "run_dynamic_prefix_checks",
]
