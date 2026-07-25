"""Complete ELBO for the nonblocking normalized H6 depth-2 cascade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from vfe4.types.h6_depth import (
    ConditionalSourceRow,
    Depth2CascadeProbe,
    GaussianMarginal,
    ScalarGaussianRegression,
)


Depth2ObjectivePartition = Literal[
    "initial",
    "source",
    "transition",
    "emission",
    "recognition_entropy",
]


@dataclass(frozen=True, slots=True)
class Depth2ObjectiveTerm:
    """One uniquely named local term in the complete depth-2 ELBO."""

    name: str
    partition: Depth2ObjectivePartition
    value: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("objective term name must be nonempty")
        if self.partition not in (
            "initial",
            "source",
            "transition",
            "emission",
            "recognition_entropy",
        ):
            raise ValueError("unsupported depth-2 objective partition")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError("objective term value must be finite")


@dataclass(frozen=True, slots=True)
class Depth2ObjectiveBreakdown:
    """Canonical local inventory and its stable sum."""

    probe_id: str
    terms: tuple[Depth2ObjectiveTerm, ...]
    total: float

    def __post_init__(self) -> None:
        if self.probe_id != "h6-depth2-t2-scalar-v3-v1":
            raise ValueError("objective breakdown has the wrong probe identity")
        if type(self.terms) is not tuple or not self.terms:
            raise ValueError("objective breakdown must contain local terms")
        names = tuple(term.name for term in self.terms)
        if any(type(term) is not Depth2ObjectiveTerm for term in self.terms):
            raise ValueError("objective breakdown contains an invalid term")
        if len(names) != len(set(names)):
            raise ValueError("objective term names must be unique")
        if type(self.total) is not float or not math.isfinite(self.total):
            raise ValueError("objective total must be finite")
        if not math.isclose(
            self.total,
            math.fsum(term.value for term in self.terms),
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ):
            raise ValueError("objective total must equal the stable local sum")


def _expected_log_regression(
    target: GaussianMarginal,
    regression: ScalarGaussianRegression,
    predictors: dict[str, GaussianMarginal],
) -> float:
    if set(predictors) != set(regression.predictor_labels):
        raise ValueError("predictors do not match the Gaussian regression")
    predicted_mean = regression.intercept + math.fsum(
        coefficient * predictors[label].mean
        for label, coefficient in regression.coefficients
    )
    residual_variance = target.variance + math.fsum(
        coefficient * coefficient * predictors[label].variance
        for label, coefficient in regression.coefficients
    )
    mean_residual = target.mean - predicted_mean
    return -0.5 * (
        math.log(2.0 * math.pi * regression.variance)
        + (residual_variance + mean_residual * mean_residual)
        / regression.variance
    )


def _expected_log_source(
    posterior: ConditionalSourceRow,
    prior: ConditionalSourceRow,
) -> float:
    if posterior.parents != prior.parents:
        raise ValueError("source posterior/prior supports must match")
    return math.fsum(
        probability * math.log(prior.probability(parent))
        for parent, probability in zip(
            posterior.parents,
            posterior.probabilities,
            strict=True,
        )
    )


def _source_entropy(row: ConditionalSourceRow) -> float:
    return -math.fsum(
        probability * math.log(probability)
        for probability in row.probabilities
    )


def _gaussian_entropy(marginal: GaussianMarginal) -> float:
    return 0.5 * math.log(2.0 * math.pi * math.e * marginal.variance)


def _expected_transition(
    target: GaussianMarginal,
    regression: ScalarGaussianRegression,
    posterior: ConditionalSourceRow,
    predictor_for_parent: object,
) -> float:
    return math.fsum(
        probability
        * _expected_log_regression(
            target,
            regression,
            predictor_for_parent(parent),  # type: ignore[operator]
        )
        for parent, probability in zip(
            posterior.parents,
            posterior.probabilities,
            strict=True,
        )
    )


def _expected_emission(
    probe: Depth2CascadeProbe,
    receiver_t: int,
) -> float:
    cascade = probe.cascade
    recognition = probe.recognition
    state = recognition.marginal(2, "state", receiver_t)
    model = recognition.marginal(2, "model", receiver_t)
    emission = cascade.emission
    boundary_mean = (
        emission.state_weight * state.mean
        + emission.model_weight * model.mean
        + emission.offset
    )
    boundary_variance = (
        emission.state_weight * emission.state_weight * state.variance
        + emission.model_weight * emission.model_weight * model.variance
    )
    nonnegative_probability = 0.5 * (
        1.0
        + math.erf(
            boundary_mean / math.sqrt(2.0 * boundary_variance)
        )
    )
    token = probe.observed_tokens[receiver_t - 1]
    return (
        nonnegative_probability
        * math.log(emission.nonnegative_probabilities[token])
        + (1.0 - nonnegative_probability)
        * math.log(emission.negative_probabilities[token])
    )


def _initial_terms(probe: Depth2CascadeProbe) -> tuple[Depth2ObjectiveTerm, ...]:
    q = probe.recognition
    p = probe.cascade
    m1 = q.marginal(1, "model", 0)
    z1 = q.marginal(1, "state", 0)
    m2 = q.marginal(2, "model", 0)
    z2 = q.marginal(2, "state", 0)
    return (
        Depth2ObjectiveTerm(
            "initial.layer1.model",
            "initial",
            float(_expected_log_regression(m1, p.layer1_initial_model, {})),
        ),
        Depth2ObjectiveTerm(
            "initial.layer1.state",
            "initial",
            float(
                _expected_log_regression(
                    z1,
                    p.layer1_initial_state,
                    {"m1_0": m1},
                )
            ),
        ),
        Depth2ObjectiveTerm(
            "initial.layer2.model",
            "initial",
            float(
                _expected_log_regression(
                    m2,
                    p.layer2_initial_model,
                    {"m1_0": m1, "z1_0": z1},
                )
            ),
        ),
        Depth2ObjectiveTerm(
            "initial.layer2.state",
            "initial",
            float(
                _expected_log_regression(
                    z2,
                    p.layer2_initial_state,
                    {"m1_0": m1, "m2_0": m2, "z1_0": z1},
                )
            ),
        ),
    )


def _receiver_terms(
    probe: Depth2CascadeProbe,
    receiver_t: int,
) -> tuple[Depth2ObjectiveTerm, ...]:
    q = probe.recognition
    p = probe.cascade
    source_terms: list[Depth2ObjectiveTerm] = []
    for layer in (1, 2):
        for channel in ("state", "model"):
            q_row = q.source_posteriors.bank(layer, channel).row(receiver_t)
            p_row = p.source_banks.bank(layer, channel).row(receiver_t)
            source_terms.append(
                Depth2ObjectiveTerm(
                    f"source.layer{layer}.{channel}.t{receiver_t}",
                    "source",
                    float(_expected_log_source(q_row, p_row)),
                )
            )

    m1_t = q.marginal(1, "model", receiver_t)
    z1_t = q.marginal(1, "state", receiver_t)
    m2_t = q.marginal(2, "model", receiver_t)
    z2_t = q.marginal(2, "state", receiver_t)
    transition_terms = (
        Depth2ObjectiveTerm(
            f"transition.layer1.model.t{receiver_t}",
            "transition",
            float(
                _expected_transition(
                    m1_t,
                    p.layer1_model_transition,
                    q.source_posteriors.layer1_model.row(receiver_t),
                    lambda parent: {
                        "m1_parent": q.marginal(1, "model", parent)
                    },
                )
            ),
        ),
        Depth2ObjectiveTerm(
            f"transition.layer1.state.t{receiver_t}",
            "transition",
            float(
                _expected_transition(
                    z1_t,
                    p.layer1_state_transition,
                    q.source_posteriors.layer1_state.row(receiver_t),
                    lambda parent: {
                        "m1_t": m1_t,
                        "z1_parent": q.marginal(1, "state", parent),
                    },
                )
            ),
        ),
        Depth2ObjectiveTerm(
            f"transition.layer2.model.t{receiver_t}",
            "transition",
            float(
                _expected_transition(
                    m2_t,
                    p.layer2_model_transition,
                    q.source_posteriors.layer2_model.row(receiver_t),
                    lambda parent: {
                        "m1_t": m1_t,
                        "m2_parent": q.marginal(2, "model", parent),
                    },
                )
            ),
        ),
        Depth2ObjectiveTerm(
            f"transition.layer2.state.t{receiver_t}",
            "transition",
            float(
                _expected_transition(
                    z2_t,
                    p.layer2_state_transition,
                    q.source_posteriors.layer2_state.row(receiver_t),
                    lambda parent: {
                        "m2_t": m2_t,
                        "z1_t": z1_t,
                        "z2_parent": q.marginal(2, "state", parent),
                    },
                )
            ),
        ),
    )
    emission_term = Depth2ObjectiveTerm(
        f"emission.top.t{receiver_t}",
        "emission",
        float(_expected_emission(probe, receiver_t)),
    )
    return tuple(source_terms) + transition_terms + (emission_term,)


def _entropy_terms(
    probe: Depth2CascadeProbe,
) -> tuple[Depth2ObjectiveTerm, ...]:
    q = probe.recognition
    terms: list[Depth2ObjectiveTerm] = []
    for layer in (1, 2):
        for channel in ("state", "model"):
            for receiver_t in range(q.horizon + 1):
                terms.append(
                    Depth2ObjectiveTerm(
                        (
                            f"entropy.gaussian.layer{layer}.{channel}."
                            f"t{receiver_t}"
                        ),
                        "recognition_entropy",
                        float(
                            _gaussian_entropy(
                                q.marginal(layer, channel, receiver_t)
                            )
                        ),
                    )
                )
            for receiver_t in range(1, q.horizon + 1):
                terms.append(
                    Depth2ObjectiveTerm(
                        (
                            f"entropy.source.layer{layer}.{channel}."
                            f"t{receiver_t}"
                        ),
                        "recognition_entropy",
                        float(
                            _source_entropy(
                                q.source_posteriors.bank(
                                    layer, channel
                                ).row(receiver_t)
                            )
                        ),
                    )
                )
    return tuple(terms)


def evaluate_depth2_local_objective(
    probe: Depth2CascadeProbe,
) -> Depth2ObjectiveBreakdown:
    """Assemble every local initial/source/transition/emission/entropy term."""

    if type(probe) is not Depth2CascadeProbe:
        raise ValueError("probe must be a Depth2CascadeProbe")
    terms = (
        _initial_terms(probe)
        + tuple(
            term
            for receiver_t in range(1, probe.cascade.horizon + 1)
            for term in _receiver_terms(probe, receiver_t)
        )
        + _entropy_terms(probe)
    )
    return Depth2ObjectiveBreakdown(
        probe_id=probe.probe_id,
        terms=terms,
        total=float(math.fsum(term.value for term in terms)),
    )


def evaluate_depth2_monolithic_objective(
    probe: Depth2CascadeProbe,
) -> float:
    """Evaluate the complete ELBO directly without consuming local records."""

    if type(probe) is not Depth2CascadeProbe:
        raise ValueError("probe must be a Depth2CascadeProbe")
    q = probe.recognition
    p = probe.cascade
    m1_0 = q.marginal(1, "model", 0)
    z1_0 = q.marginal(1, "state", 0)
    m2_0 = q.marginal(2, "model", 0)
    z2_0 = q.marginal(2, "state", 0)
    contributions = [
        _expected_log_regression(m1_0, p.layer1_initial_model, {}),
        _expected_log_regression(
            z1_0, p.layer1_initial_state, {"m1_0": m1_0}
        ),
        _expected_log_regression(
            m2_0,
            p.layer2_initial_model,
            {"m1_0": m1_0, "z1_0": z1_0},
        ),
        _expected_log_regression(
            z2_0,
            p.layer2_initial_state,
            {"m1_0": m1_0, "m2_0": m2_0, "z1_0": z1_0},
        ),
    ]
    for receiver_t in range(1, p.horizon + 1):
        for layer in (1, 2):
            for channel in ("state", "model"):
                contributions.append(
                    _expected_log_source(
                        q.source_posteriors.bank(
                            layer, channel
                        ).row(receiver_t),
                        p.source_banks.bank(layer, channel).row(receiver_t),
                    )
                )
        m1_t = q.marginal(1, "model", receiver_t)
        z1_t = q.marginal(1, "state", receiver_t)
        m2_t = q.marginal(2, "model", receiver_t)
        z2_t = q.marginal(2, "state", receiver_t)
        contributions.extend(
            (
                _expected_transition(
                    m1_t,
                    p.layer1_model_transition,
                    q.source_posteriors.layer1_model.row(receiver_t),
                    lambda parent: {
                        "m1_parent": q.marginal(1, "model", parent)
                    },
                ),
                _expected_transition(
                    z1_t,
                    p.layer1_state_transition,
                    q.source_posteriors.layer1_state.row(receiver_t),
                    lambda parent: {
                        "m1_t": m1_t,
                        "z1_parent": q.marginal(1, "state", parent),
                    },
                ),
                _expected_transition(
                    m2_t,
                    p.layer2_model_transition,
                    q.source_posteriors.layer2_model.row(receiver_t),
                    lambda parent: {
                        "m1_t": m1_t,
                        "m2_parent": q.marginal(2, "model", parent),
                    },
                ),
                _expected_transition(
                    z2_t,
                    p.layer2_state_transition,
                    q.source_posteriors.layer2_state.row(receiver_t),
                    lambda parent: {
                        "m2_t": m2_t,
                        "z1_t": z1_t,
                        "z2_parent": q.marginal(2, "state", parent),
                    },
                ),
                _expected_emission(probe, receiver_t),
            )
        )
    for layer in (1, 2):
        for channel in ("state", "model"):
            contributions.extend(
                _gaussian_entropy(q.marginal(layer, channel, receiver_t))
                for receiver_t in range(q.horizon + 1)
            )
            contributions.extend(
                _source_entropy(
                    q.source_posteriors.bank(layer, channel).row(receiver_t)
                )
                for receiver_t in range(1, q.horizon + 1)
            )
    return float(math.fsum(contributions))


__all__ = [
    "Depth2ObjectiveBreakdown",
    "Depth2ObjectivePartition",
    "Depth2ObjectiveTerm",
    "evaluate_depth2_local_objective",
    "evaluate_depth2_monolithic_objective",
]
