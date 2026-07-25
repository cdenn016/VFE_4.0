"""Pure symbolic parameter counts for the frozen H6 language-arm constructors.

This module performs integer arithmetic only.  It does not construct a model,
enumerate a capacity grid, inspect data, or use training or evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ArmName = Literal["A0", "A1", "A2", "A3", "A4", "A5"]
RecognitionFamily = Literal["structured", "factorized"]
ParameterCountStatus = Literal["AVAILABLE", "UNAVAILABLE"]

A5_REFERENCE_PARAMETER_COUNT = 63_634
AMENDED_EMISSION_WIDTH_CANDIDATES = (48, 64, 80, 96, 123)
AMENDED_LATENT_WIDTH_CANDIDATES = (2, 8, 16, 24, 32)
AMENDED_RECOGNITION_WIDTH_CANDIDATES = (32, 64, 96)
PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH = 6

_PREFIX_CONFIG_ID = (
    "h6-a5-structured-prefix-exact-complete-latent-smoothing-v1"
)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def recognition_parameter_count(
    *,
    vocabulary_size: int,
    latent_width: int,
    recognition_width: int,
    channel_count: Literal[1, 2],
    family: RecognitionFamily,
) -> int:
    """Count the recognition embedding, mean head, and packed precision."""

    vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
    latent_width = _positive_int(latent_width, "latent_width")
    recognition_width = _positive_int(
        recognition_width, "recognition_width"
    )
    if type(channel_count) is not int or channel_count not in (1, 2):
        raise ValueError("channel_count must be exactly one or two")
    if family not in ("structured", "factorized"):
        raise ValueError("family must be structured or factorized")

    gaussian_dimension = channel_count * latent_width
    packed_precision = (
        gaussian_dimension * (gaussian_dimension + 1) // 2
        if family == "structured"
        else channel_count * latent_width * (latent_width + 1) // 2
    )
    return (
        vocabulary_size * recognition_width
        + gaussian_dimension * recognition_width
        + gaussian_dimension
        + packed_precision
    )


def fixed_source_prior_parameter_count(
    *,
    horizon: int,
    bank_count: Literal[1, 2],
    gauge_anchored: bool = True,
) -> int:
    """Count trainable prefix-independent source logits.

    Both the arm-local and standalone constructors remove one
    additive-softmax null scalar per receiver and bank.
    """

    horizon = _positive_int(horizon, "horizon")
    if type(bank_count) is not int or bank_count not in (1, 2):
        raise ValueError("bank_count must be exactly one or two")
    if type(gauge_anchored) is not bool:
        raise ValueError("gauge_anchored must be a bool")
    row_scalars = (
        horizon * (horizon - 1) // 2
        if gauge_anchored
        else horizon * (horizon + 1) // 2
    )
    return bank_count * row_scalars


def prefix_conditioned_source_prior_parameter_count(
    *,
    vocabulary_size: int,
    horizon: int,
    latent_width: int,
    context_width: int,
    gauge_anchored: bool,
) -> int:
    """Count both banks of the prefix-conditioned source-prior constructor."""

    vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
    horizon = _positive_int(horizon, "horizon")
    latent_width = _positive_int(latent_width, "latent_width")
    context_width = _positive_int(context_width, "context_width")
    if type(gauge_anchored) is not bool:
        raise ValueError("gauge_anchored must be a bool")

    parent_entries = (
        horizon * (horizon - 1)
        if gauge_anchored
        else horizon * (horizon + 1)
    )
    return (
        vocabulary_size * context_width
        + 2 * latent_width * context_width
        + parent_entries * context_width
        + parent_entries
    )


def h6_a0_parameter_count(
    *,
    vocabulary_size: int,
    position_capacity: int,
    hidden_width: int,
) -> int:
    """Count the amended one-block, two-equal-head H6 A0 Transformer."""

    vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
    position_capacity = _positive_int(
        position_capacity, "position_capacity"
    )
    hidden_width = _positive_int(hidden_width, "hidden_width")
    if hidden_width % 2:
        raise ValueError("H6 A0 hidden width must split into two equal heads")
    return (
        2 * vocabulary_size * hidden_width
        + position_capacity * hidden_width
        + 12 * hidden_width * hidden_width
        + 15 * hidden_width
        + vocabulary_size
    )


def mean_pooled_no_latent_parameter_count(
    *,
    vocabulary_size: int,
    emission_width: int,
) -> int:
    """Count the descriptive mean-pooled no-latent floor."""

    vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
    emission_width = _positive_int(emission_width, "emission_width")
    return (
        2 * vocabulary_size * emission_width
        + emission_width
        + vocabulary_size
    )


def arm_parameter_count(
    arm: ArmName,
    *,
    vocabulary_size: int,
    horizon: int,
    emission_width: int,
    latent_width: int | None = None,
    recognition_width: int | None = None,
    recognition_family: RecognitionFamily = "structured",
) -> int:
    """Return the exact trainable-scalar count of one current arm shape."""

    if arm not in ("A0", "A1", "A2", "A3", "A4", "A5"):
        raise ValueError("arm must be A0 through A5")
    vocabulary_size = _positive_int(vocabulary_size, "vocabulary_size")
    horizon = _positive_int(horizon, "horizon")
    emission_width = _positive_int(emission_width, "emission_width")
    latent_width = _optional_positive_int(latent_width, "latent_width")
    recognition_width = _optional_positive_int(
        recognition_width, "recognition_width"
    )

    no_latent_a5 = (
        arm == "A5"
        and latent_width is None
        and recognition_width is None
    )
    if arm == "A0":
        if latent_width is not None or recognition_width is not None:
            raise ValueError(
                "no-latent arms have no latent or recognition width"
            )
        return h6_a0_parameter_count(
            vocabulary_size=vocabulary_size,
            position_capacity=horizon,
            hidden_width=emission_width,
        )
    if no_latent_a5:
        return mean_pooled_no_latent_parameter_count(
            vocabulary_size=vocabulary_size,
            emission_width=emission_width,
        )
    if latent_width is None or recognition_width is None:
        raise ValueError("latent arms require latent and recognition widths")

    d = latent_width
    e = emission_width
    v = vocabulary_size
    t = horizon
    if arm == "A1":
        model_count = d * d + 4 * d + e * d + v * e + v
        channels: Literal[1, 2] = 1
    elif arm == "A2":
        model_count = (
            (t * t + 2 * t) * d * d
            + 8 * d
            + fixed_source_prior_parameter_count(
                horizon=t, bank_count=2
            )
            + 2 * e * d
            + v * e
            + v
        )
        channels = 2
    elif arm == "A3":
        model_count = (
            2 * t * d * d + 8 * d + 2 * e * d + v * e + v
        )
        channels = 2
    elif arm == "A4":
        model_count = (
            t * d * d
            + 4 * d
            + fixed_source_prior_parameter_count(
                horizon=t, bank_count=1
            )
            + e * d
            + v * e
            + v
        )
        channels = 1
    else:
        model_count = (
            3 * t * d * d
            + 8 * d
            + fixed_source_prior_parameter_count(
                horizon=t, bank_count=2
            )
            + 2 * e * d
            + v * e
            + v
        )
        channels = 2
    return model_count + recognition_parameter_count(
        vocabulary_size=v,
        latent_width=d,
        recognition_width=recognition_width,
        channel_count=channels,
        family=recognition_family,
    )


def parameter_count_within_tolerance(
    parameter_count: int,
    *,
    reference_parameter_count: int = A5_REFERENCE_PARAMETER_COUNT,
) -> bool:
    """Apply the frozen one-percent gate with exact integer arithmetic."""

    parameter_count = _positive_int(parameter_count, "parameter_count")
    reference_parameter_count = _positive_int(
        reference_parameter_count, "reference_parameter_count"
    )
    return (
        100 * abs(parameter_count - reference_parameter_count)
        <= reference_parameter_count
    )


@dataclass(frozen=True, slots=True)
class ParameterCountAssessment:
    """One predeclared, metric-blind endpoint capacity assessment."""

    config_id: str
    arm: ArmName
    emission_width: int
    latent_width: int | None
    recognition_width: int | None
    prior_context_width: int | None
    status: ParameterCountStatus
    parameter_count: int | None
    planned_parameter_count: int | None
    obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.config_id) is not str or not self.config_id:
            raise ValueError("config_id must be a nonempty string")
        _positive_int(self.emission_width, "emission_width")
        _optional_positive_int(self.latent_width, "latent_width")
        _optional_positive_int(self.recognition_width, "recognition_width")
        _optional_positive_int(
            self.prior_context_width, "prior_context_width"
        )
        _optional_positive_int(self.parameter_count, "parameter_count")
        _optional_positive_int(
            self.planned_parameter_count, "planned_parameter_count"
        )
        if self.status == "AVAILABLE":
            if (
                self.parameter_count is None
                or self.planned_parameter_count is not None
                or self.obligations
            ):
                raise ValueError(
                    "available counts require one count and no obligations"
                )
        elif self.status == "UNAVAILABLE":
            if self.parameter_count is not None or not self.obligations:
                raise ValueError(
                    "unavailable counts require obligations and no current count"
                )
        else:
            raise ValueError("status must be AVAILABLE or UNAVAILABLE")

    @property
    def parameter_within_tolerance(self) -> bool:
        return (
            self.parameter_count is not None
            and parameter_count_within_tolerance(self.parameter_count)
        )


def _available_assessment(
    *,
    config_id: str,
    arm: ArmName,
    emission_width: int,
    latent_width: int | None,
    recognition_width: int | None,
    recognition_family: RecognitionFamily = "structured",
) -> ParameterCountAssessment:
    return ParameterCountAssessment(
        config_id=config_id,
        arm=arm,
        emission_width=emission_width,
        latent_width=latent_width,
        recognition_width=recognition_width,
        prior_context_width=None,
        status="AVAILABLE",
        parameter_count=arm_parameter_count(
            arm,
            vocabulary_size=258,
            horizon=32,
            emission_width=emission_width,
            latent_width=latent_width,
            recognition_width=recognition_width,
            recognition_family=recognition_family,
        ),
        planned_parameter_count=None,
        obligations=(),
    )


def _prefix_a5_parameter_count() -> int:
    emission_width = 80
    latent_width = 8
    recognition_width = 96
    current_fixed_count = arm_parameter_count(
        "A5",
        vocabulary_size=258,
        horizon=32,
        emission_width=emission_width,
        latent_width=latent_width,
        recognition_width=recognition_width,
    )
    return (
        current_fixed_count
        - fixed_source_prior_parameter_count(horizon=32, bank_count=2)
        + prefix_conditioned_source_prior_parameter_count(
            vocabulary_size=258,
            horizon=32,
            latent_width=latent_width,
            context_width=PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
            gauge_anchored=True,
        )
    )


def outcome_blind_feasibility_assessments(
) -> tuple[ParameterCountAssessment, ...]:
    """Return the twelve static witnesses without enumerating a grid."""

    base_a5_id = (
        "h6-a5-structured-fixed-exact-complete-latent-smoothing-v1"
    )
    return (
        _available_assessment(
            config_id="h6-a0-transformer-v2",
            arm="A0",
            emission_width=52,
            latent_width=None,
            recognition_width=None,
        ),
        _available_assessment(
            config_id="h6-a1-ordinary-latent-v1",
            arm="A1",
            emission_width=123,
            latent_width=24,
            recognition_width=96,
        ),
        _available_assessment(
            config_id="h6-a2-generic-map-v1",
            arm="A2",
            emission_width=123,
            latent_width=2,
            recognition_width=96,
        ),
        _available_assessment(
            config_id="h6-a3-immediate-predecessor-v1",
            arm="A3",
            emission_width=64,
            latent_width=16,
            recognition_width=96,
        ),
        _available_assessment(
            config_id="h6-a4-state-only-v1",
            arm="A4",
            emission_width=123,
            latent_width=24,
            recognition_width=32,
        ),
        _available_assessment(
            config_id=base_a5_id,
            arm="A5",
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        ),
        _available_assessment(
            config_id=(
                "h6-a5-factorized-fixed-exact-complete-"
                "latent-smoothing-v1"
            ),
            arm="A5",
            emission_width=64,
            latent_width=16,
            recognition_width=64,
            recognition_family="factorized",
        ),
        ParameterCountAssessment(
            config_id=_PREFIX_CONFIG_ID,
            arm="A5",
            emission_width=80,
            latent_width=8,
            recognition_width=96,
            prior_context_width=PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH,
            status="AVAILABLE",
            parameter_count=_prefix_a5_parameter_count(),
            planned_parameter_count=None,
            obligations=(),
        ),
        _available_assessment(
            config_id=(
                "h6-a5-structured-fixed-projection-complete-"
                "latent-smoothing-v1"
            ),
            arm="A5",
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        ),
        _available_assessment(
            config_id=(
                "h6-a5-structured-fixed-exact-emission-"
                "latent-smoothing-v1"
            ),
            arm="A5",
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        ),
        _available_assessment(
            config_id=(
                "h6-a5-structured-fixed-exact-complete-"
                "nolatent-norecognition-v1"
            ),
            arm="A5",
            emission_width=123,
            latent_width=None,
            recognition_width=None,
        ),
        _available_assessment(
            config_id=(
                "h6-a5-structured-fixed-exact-complete-"
                "latent-filtering-v1"
            ),
            arm="A5",
            emission_width=64,
            latent_width=16,
            recognition_width=64,
        ),
    )


__all__ = [
    "A5_REFERENCE_PARAMETER_COUNT",
    "AMENDED_EMISSION_WIDTH_CANDIDATES",
    "AMENDED_LATENT_WIDTH_CANDIDATES",
    "AMENDED_RECOGNITION_WIDTH_CANDIDATES",
    "PROPOSED_PREFIX_PRIOR_CONTEXT_WIDTH",
    "ParameterCountAssessment",
    "arm_parameter_count",
    "fixed_source_prior_parameter_count",
    "h6_a0_parameter_count",
    "mean_pooled_no_latent_parameter_count",
    "outcome_blind_feasibility_assessments",
    "parameter_count_within_tolerance",
    "prefix_conditioned_source_prior_parameter_count",
    "recognition_parameter_count",
]
