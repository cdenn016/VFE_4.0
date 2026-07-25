"""Normalized generative law for the nonblocking H6 depth-2 probe."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from vfe4.types.h6_depth import (
    Depth2CascadeSpec,
    Depth2SourceBanks,
    ScalarGaussianRegression,
)


@dataclass(frozen=True, slots=True)
class Depth2SourcePath:
    """One complete assignment of the four source sequences."""

    layer1_state: tuple[int, ...]
    layer1_model: tuple[int, ...]
    layer2_state: tuple[int, ...]
    layer2_model: tuple[int, ...]

    @property
    def horizon(self) -> int:
        lengths = {
            len(self.layer1_state),
            len(self.layer1_model),
            len(self.layer2_state),
            len(self.layer2_model),
        }
        if len(lengths) != 1:
            raise ValueError("all source-path channels must share one horizon")
        return lengths.pop()


@dataclass(frozen=True, slots=True)
class Depth2NormalizationReport:
    """Exhaustive discrete and analytic-continuous normalization witness."""

    probe_law_id: str
    horizon: int
    vocabulary_size: int
    source_path_count: int
    token_sequence_count: int
    emission_region_count: int
    gaussian_factor_count: int
    source_mass: float
    maximum_emission_mass_error: float
    total_mass: float

    def __post_init__(self) -> None:
        if self.probe_law_id != "H6-DEPTH2-CASCADE-v1":
            raise ValueError("normalization report has the wrong law identity")
        for name in (
            "horizon",
            "vocabulary_size",
            "source_path_count",
            "token_sequence_count",
            "emission_region_count",
            "gaussian_factor_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "source_mass",
            "maximum_emission_mass_error",
            "total_mass",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if self.maximum_emission_mass_error < 0.0:
            raise ValueError("maximum emission mass error must be nonnegative")


def _ordered_banks(source_banks: Depth2SourceBanks) -> tuple[object, ...]:
    return (
        source_banks.layer1_state,
        source_banks.layer1_model,
        source_banks.layer2_state,
        source_banks.layer2_model,
    )


def enumerate_depth2_source_paths(
    cascade: Depth2CascadeSpec,
) -> tuple[Depth2SourcePath, ...]:
    """Enumerate every explicit source assignment in deterministic order."""

    if type(cascade) is not Depth2CascadeSpec:
        raise ValueError("cascade must be a Depth2CascadeSpec")
    supports = tuple(
        row.parents
        for bank in _ordered_banks(cascade.source_banks)
        for row in bank.rows
    )
    paths: list[Depth2SourcePath] = []
    horizon = cascade.horizon
    for assignment in itertools.product(*supports):
        paths.append(
            Depth2SourcePath(
                layer1_state=tuple(assignment[0:horizon]),
                layer1_model=tuple(assignment[horizon : 2 * horizon]),
                layer2_state=tuple(
                    assignment[2 * horizon : 3 * horizon]
                ),
                layer2_model=tuple(
                    assignment[3 * horizon : 4 * horizon]
                ),
            )
        )
    return tuple(paths)


def depth2_source_path_probability(
    cascade: Depth2CascadeSpec,
    path: Depth2SourcePath,
) -> float:
    """Return the product of all normalized conditional source rows."""

    if (
        type(cascade) is not Depth2CascadeSpec
        or type(path) is not Depth2SourcePath
        or path.horizon != cascade.horizon
    ):
        raise ValueError("source path does not match the cascade")
    sequences = (
        path.layer1_state,
        path.layer1_model,
        path.layer2_state,
        path.layer2_model,
    )
    probability = 1.0
    for bank, sequence in zip(
        _ordered_banks(cascade.source_banks),
        sequences,
        strict=True,
    ):
        for receiver_t, parent in enumerate(sequence, start=1):
            probability *= bank.row(receiver_t).probability(parent)
    return probability


def _regression_mean(
    regression: ScalarGaussianRegression,
    predictors: dict[str, float],
) -> float:
    if set(predictors) != set(regression.predictor_labels):
        raise ValueError("predictors do not match the Gaussian regression")
    return regression.intercept + math.fsum(
        coefficient * predictors[label]
        for label, coefficient in regression.coefficients
    )


def _gaussian_log_density(value: float, mean: float, variance: float) -> float:
    return -0.5 * (
        math.log(2.0 * math.pi * variance)
        + (value - mean) * (value - mean) / variance
    )


def top_layer_emission_probabilities(
    cascade: Depth2CascadeSpec,
    *,
    state: float,
    model: float,
) -> tuple[float, ...]:
    """Return the normalized V-way row at one top-layer state."""

    boundary = (
        cascade.emission.state_weight * state
        + cascade.emission.model_weight * model
        + cascade.emission.offset
    )
    return (
        cascade.emission.nonnegative_probabilities
        if boundary >= 0.0
        else cascade.emission.negative_probabilities
    )


def depth2_complete_log_joint(
    cascade: Depth2CascadeSpec,
    path: Depth2SourcePath,
    *,
    state_values: tuple[tuple[float, ...], tuple[float, ...]],
    model_values: tuple[tuple[float, ...], tuple[float, ...]],
    tokens: tuple[int, ...],
) -> float:
    """Evaluate the complete normalized joint at one latent/source assignment."""

    if (
        type(cascade) is not Depth2CascadeSpec
        or type(path) is not Depth2SourcePath
        or path.horizon != cascade.horizon
        or type(state_values) is not tuple
        or type(model_values) is not tuple
        or len(state_values) != 2
        or len(model_values) != 2
        or any(len(series) != cascade.horizon + 1 for series in state_values)
        or any(len(series) != cascade.horizon + 1 for series in model_values)
        or type(tokens) is not tuple
        or len(tokens) != cascade.horizon
        or any(
            type(token) is not int
            or not 0 <= token < cascade.vocabulary_size
            for token in tokens
        )
    ):
        raise ValueError("complete-joint values do not match the cascade")
    z1, z2 = state_values
    m1, m2 = model_values
    total = math.log(depth2_source_path_probability(cascade, path))
    total += _gaussian_log_density(
        m1[0],
        _regression_mean(cascade.layer1_initial_model, {}),
        cascade.layer1_initial_model.variance,
    )
    total += _gaussian_log_density(
        z1[0],
        _regression_mean(
            cascade.layer1_initial_state,
            {"m1_0": m1[0]},
        ),
        cascade.layer1_initial_state.variance,
    )
    total += _gaussian_log_density(
        m2[0],
        _regression_mean(
            cascade.layer2_initial_model,
            {"m1_0": m1[0], "z1_0": z1[0]},
        ),
        cascade.layer2_initial_model.variance,
    )
    total += _gaussian_log_density(
        z2[0],
        _regression_mean(
            cascade.layer2_initial_state,
            {"m1_0": m1[0], "m2_0": m2[0], "z1_0": z1[0]},
        ),
        cascade.layer2_initial_state.variance,
    )
    for receiver_t in range(1, cascade.horizon + 1):
        index = receiver_t - 1
        total += _gaussian_log_density(
            m1[receiver_t],
            _regression_mean(
                cascade.layer1_model_transition,
                {"m1_parent": m1[path.layer1_model[index]]},
            ),
            cascade.layer1_model_transition.variance,
        )
        total += _gaussian_log_density(
            z1[receiver_t],
            _regression_mean(
                cascade.layer1_state_transition,
                {
                    "m1_t": m1[receiver_t],
                    "z1_parent": z1[path.layer1_state[index]],
                },
            ),
            cascade.layer1_state_transition.variance,
        )
        total += _gaussian_log_density(
            m2[receiver_t],
            _regression_mean(
                cascade.layer2_model_transition,
                {
                    "m1_t": m1[receiver_t],
                    "m2_parent": m2[path.layer2_model[index]],
                },
            ),
            cascade.layer2_model_transition.variance,
        )
        total += _gaussian_log_density(
            z2[receiver_t],
            _regression_mean(
                cascade.layer2_state_transition,
                {
                    "m2_t": m2[receiver_t],
                    "z1_t": z1[receiver_t],
                    "z2_parent": z2[path.layer2_state[index]],
                },
            ),
            cascade.layer2_state_transition.variance,
        )
        probabilities = top_layer_emission_probabilities(
            cascade,
            state=z2[receiver_t],
            model=m2[receiver_t],
        )
        total += math.log(probabilities[tokens[index]])
    return total


def evaluate_depth2_normalization(
    cascade: Depth2CascadeSpec,
) -> Depth2NormalizationReport:
    """Exhaustively verify discrete rows and analytic Gaussian normalization."""

    paths = enumerate_depth2_source_paths(cascade)
    source_mass = math.fsum(
        depth2_source_path_probability(cascade, path) for path in paths
    )
    token_sequences = tuple(
        itertools.product(
            range(cascade.vocabulary_size),
            repeat=cascade.horizon,
        )
    )
    region_rows = (
        cascade.emission.negative_probabilities,
        cascade.emission.nonnegative_probabilities,
    )
    maximum_emission_error = 0.0
    for region_assignment in itertools.product(
        range(2), repeat=cascade.horizon
    ):
        token_mass = math.fsum(
            math.prod(
                region_rows[region_assignment[index]][token]
                for index, token in enumerate(sequence)
            )
            for sequence in token_sequences
        )
        maximum_emission_error = max(
            maximum_emission_error,
            abs(token_mass - 1.0),
        )
    return Depth2NormalizationReport(
        probe_law_id=cascade.probe_law_id,
        horizon=cascade.horizon,
        vocabulary_size=cascade.vocabulary_size,
        source_path_count=len(paths),
        token_sequence_count=len(token_sequences),
        emission_region_count=2**cascade.horizon,
        gaussian_factor_count=4 * (cascade.horizon + 1),
        source_mass=float(source_mass),
        maximum_emission_mass_error=float(maximum_emission_error),
        total_mass=float(source_mass),
    )


__all__ = [
    "Depth2NormalizationReport",
    "Depth2SourcePath",
    "depth2_complete_log_joint",
    "depth2_source_path_probability",
    "enumerate_depth2_source_paths",
    "evaluate_depth2_normalization",
    "top_layer_emission_probabilities",
]
