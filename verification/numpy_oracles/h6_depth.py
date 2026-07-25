"""Independent scalar oracle for the H6 two-layer cascade amendment.

This module intentionally imports neither ``vfe4`` nor Torch/NumPy.  It
exhaustively enumerates the frozen source and token supports and reduces every
Gaussian recognition expectation with scalar closed forms.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field


_REPORT_DOMAIN = b"vfe4.h6-depth2-independent-oracle.v1\x00"


@dataclass(frozen=True, slots=True)
class Depth2OracleReport:
    schema_version: str
    probe_id: str
    source_path_count: int
    token_sequence_count: int
    emission_region_count: int
    gaussian_factor_count: int
    source_mass: float
    maximum_emission_mass_error: float
    objective_term_names: tuple[str, ...]
    objective_term_values: tuple[float, ...]
    objective_total: float
    report_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "h6-depth2-independent-oracle-v1":
            raise ValueError("unsupported depth-2 oracle schema")
        if self.probe_id != "h6-depth2-t2-scalar-v3-v1":
            raise ValueError("unsupported depth-2 oracle probe")
        for name in (
            "source_path_count",
            "token_sequence_count",
            "emission_region_count",
            "gaussian_factor_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "source_mass",
            "maximum_emission_mass_error",
            "objective_total",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            type(self.objective_term_names) is not tuple
            or type(self.objective_term_values) is not tuple
            or len(self.objective_term_names) != len(self.objective_term_values)
            or len(self.objective_term_names)
            != len(set(self.objective_term_names))
            or any(
                type(name) is not str or not name
                for name in self.objective_term_names
            )
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.objective_term_values
            )
        ):
            raise ValueError("oracle objective inventory is malformed")
        if not math.isclose(
            self.objective_total,
            math.fsum(self.objective_term_values),
            rel_tol=0.0,
            abs_tol=2.0e-14,
        ):
            raise ValueError("oracle objective total is not its stable local sum")
        object.__setattr__(
            self,
            "report_sha256",
            hashlib.sha256(
                _REPORT_DOMAIN + _canonical_report_bytes(self)
            ).hexdigest(),
        )


def _bank(banks: object, layer: int, channel: str) -> object:
    return getattr(banks, f"layer{layer}_{channel}")


def _row(banks: object, layer: int, channel: str, receiver_t: int) -> object:
    return _bank(banks, layer, channel).rows[receiver_t - 1]


def _marginal(
    recognition: object,
    layer: int,
    channel: str,
    receiver_t: int,
) -> object:
    field = (
        recognition.state_marginals
        if channel == "state"
        else recognition.model_marginals
    )
    return field[layer - 1][receiver_t]


def _expected_log_regression(
    target: object,
    regression: object,
    predictors: dict[str, object],
) -> float:
    coefficients = tuple(regression.coefficients)
    labels = tuple(label for label, _ in coefficients)
    if set(predictors) != set(labels):
        raise ValueError("oracle regression predictors do not align")
    predicted_mean = float(regression.intercept) + math.fsum(
        float(coefficient) * float(predictors[label].mean)
        for label, coefficient in coefficients
    )
    residual_variance = float(target.variance) + math.fsum(
        float(coefficient)
        * float(coefficient)
        * float(predictors[label].variance)
        for label, coefficient in coefficients
    )
    mean_residual = float(target.mean) - predicted_mean
    variance = float(regression.variance)
    return -0.5 * (
        math.log(2.0 * math.pi * variance)
        + (residual_variance + mean_residual * mean_residual) / variance
    )


def _expected_log_source(posterior: object, prior: object) -> float:
    if tuple(posterior.parents) != tuple(prior.parents):
        raise ValueError("oracle source supports do not align")
    prior_probabilities = dict(
        zip(prior.parents, prior.probabilities, strict=True)
    )
    return math.fsum(
        float(probability) * math.log(float(prior_probabilities[parent]))
        for parent, probability in zip(
            posterior.parents,
            posterior.probabilities,
            strict=True,
        )
    )


def _source_entropy(row: object) -> float:
    return -math.fsum(
        float(probability) * math.log(float(probability))
        for probability in row.probabilities
    )


def _gaussian_entropy(marginal: object) -> float:
    return 0.5 * math.log(
        2.0 * math.pi * math.e * float(marginal.variance)
    )


def _expected_transition(
    target: object,
    regression: object,
    posterior: object,
    predictor_for_parent: object,
) -> float:
    return math.fsum(
        float(probability)
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


def _expected_emission(probe: object, receiver_t: int) -> float:
    cascade = probe.cascade
    recognition = probe.recognition
    emission = cascade.emission
    state = _marginal(recognition, 2, "state", receiver_t)
    model = _marginal(recognition, 2, "model", receiver_t)
    boundary_mean = (
        float(emission.state_weight) * float(state.mean)
        + float(emission.model_weight) * float(model.mean)
        + float(emission.offset)
    )
    boundary_variance = (
        float(emission.state_weight) ** 2 * float(state.variance)
        + float(emission.model_weight) ** 2 * float(model.variance)
    )
    p_nonnegative = 0.5 * (
        1.0
        + math.erf(
            boundary_mean / math.sqrt(2.0 * boundary_variance)
        )
    )
    token = probe.observed_tokens[receiver_t - 1]
    return (
        p_nonnegative
        * math.log(float(emission.nonnegative_probabilities[token]))
        + (1.0 - p_nonnegative)
        * math.log(float(emission.negative_probabilities[token]))
    )


def _normalization(probe: object) -> tuple[int, int, int, float, float]:
    cascade = probe.cascade
    banks = cascade.source_banks
    ordered_banks = tuple(
        _bank(banks, layer, channel)
        for layer, channel in (
            (1, "state"),
            (1, "model"),
            (2, "state"),
            (2, "model"),
        )
    )
    rows = tuple(row for bank in ordered_banks for row in bank.rows)
    source_mass = 0.0
    source_count = 0
    for assignment in itertools.product(
        *(tuple(row.parents) for row in rows)
    ):
        source_count += 1
        source_mass += math.prod(
            float(row.probabilities[row.parents.index(parent)])
            for row, parent in zip(rows, assignment, strict=True)
        )
    vocabulary_size = len(cascade.emission.negative_probabilities)
    horizon = int(cascade.horizon)
    token_sequences = tuple(
        itertools.product(range(vocabulary_size), repeat=horizon)
    )
    region_rows = (
        cascade.emission.negative_probabilities,
        cascade.emission.nonnegative_probabilities,
    )
    maximum_error = 0.0
    region_count = 0
    for regions in itertools.product(range(2), repeat=horizon):
        region_count += 1
        token_mass = math.fsum(
            math.prod(
                float(region_rows[regions[index]][token])
                for index, token in enumerate(sequence)
            )
            for sequence in token_sequences
        )
        maximum_error = max(maximum_error, abs(token_mass - 1.0))
    return (
        source_count,
        len(token_sequences),
        region_count,
        float(source_mass),
        float(maximum_error),
    )


def _objective_terms(probe: object) -> tuple[tuple[str, float], ...]:
    cascade = probe.cascade
    recognition = probe.recognition
    m1_0 = _marginal(recognition, 1, "model", 0)
    z1_0 = _marginal(recognition, 1, "state", 0)
    m2_0 = _marginal(recognition, 2, "model", 0)
    z2_0 = _marginal(recognition, 2, "state", 0)
    terms: list[tuple[str, float]] = [
        (
            "initial.layer1.model",
            _expected_log_regression(
                m1_0, cascade.layer1_initial_model, {}
            ),
        ),
        (
            "initial.layer1.state",
            _expected_log_regression(
                z1_0,
                cascade.layer1_initial_state,
                {"m1_0": m1_0},
            ),
        ),
        (
            "initial.layer2.model",
            _expected_log_regression(
                m2_0,
                cascade.layer2_initial_model,
                {"m1_0": m1_0, "z1_0": z1_0},
            ),
        ),
        (
            "initial.layer2.state",
            _expected_log_regression(
                z2_0,
                cascade.layer2_initial_state,
                {"m1_0": m1_0, "m2_0": m2_0, "z1_0": z1_0},
            ),
        ),
    ]
    for receiver_t in range(1, int(cascade.horizon) + 1):
        for layer in (1, 2):
            for channel in ("state", "model"):
                terms.append(
                    (
                        f"source.layer{layer}.{channel}.t{receiver_t}",
                        _expected_log_source(
                            _row(
                                recognition.source_posteriors,
                                layer,
                                channel,
                                receiver_t,
                            ),
                            _row(
                                cascade.source_banks,
                                layer,
                                channel,
                                receiver_t,
                            ),
                        ),
                    )
                )
        m1_t = _marginal(recognition, 1, "model", receiver_t)
        z1_t = _marginal(recognition, 1, "state", receiver_t)
        m2_t = _marginal(recognition, 2, "model", receiver_t)
        z2_t = _marginal(recognition, 2, "state", receiver_t)
        terms.extend(
            (
                (
                    f"transition.layer1.model.t{receiver_t}",
                    _expected_transition(
                        m1_t,
                        cascade.layer1_model_transition,
                        _row(
                            recognition.source_posteriors,
                            1,
                            "model",
                            receiver_t,
                        ),
                        lambda parent: {
                            "m1_parent": _marginal(
                                recognition, 1, "model", parent
                            )
                        },
                    ),
                ),
                (
                    f"transition.layer1.state.t{receiver_t}",
                    _expected_transition(
                        z1_t,
                        cascade.layer1_state_transition,
                        _row(
                            recognition.source_posteriors,
                            1,
                            "state",
                            receiver_t,
                        ),
                        lambda parent: {
                            "m1_t": m1_t,
                            "z1_parent": _marginal(
                                recognition, 1, "state", parent
                            ),
                        },
                    ),
                ),
                (
                    f"transition.layer2.model.t{receiver_t}",
                    _expected_transition(
                        m2_t,
                        cascade.layer2_model_transition,
                        _row(
                            recognition.source_posteriors,
                            2,
                            "model",
                            receiver_t,
                        ),
                        lambda parent: {
                            "m1_t": m1_t,
                            "m2_parent": _marginal(
                                recognition, 2, "model", parent
                            ),
                        },
                    ),
                ),
                (
                    f"transition.layer2.state.t{receiver_t}",
                    _expected_transition(
                        z2_t,
                        cascade.layer2_state_transition,
                        _row(
                            recognition.source_posteriors,
                            2,
                            "state",
                            receiver_t,
                        ),
                        lambda parent: {
                            "m2_t": m2_t,
                            "z1_t": z1_t,
                            "z2_parent": _marginal(
                                recognition, 2, "state", parent
                            ),
                        },
                    ),
                ),
                (
                    f"emission.top.t{receiver_t}",
                    _expected_emission(probe, receiver_t),
                ),
            )
        )
    for layer in (1, 2):
        for channel in ("state", "model"):
            for receiver_t in range(int(recognition.horizon) + 1):
                terms.append(
                    (
                        (
                            f"entropy.gaussian.layer{layer}.{channel}."
                            f"t{receiver_t}"
                        ),
                        _gaussian_entropy(
                            _marginal(
                                recognition,
                                layer,
                                channel,
                                receiver_t,
                            )
                        ),
                    )
                )
            for receiver_t in range(1, int(recognition.horizon) + 1):
                terms.append(
                    (
                        (
                            f"entropy.source.layer{layer}.{channel}."
                            f"t{receiver_t}"
                        ),
                        _source_entropy(
                            _row(
                                recognition.source_posteriors,
                                layer,
                                channel,
                                receiver_t,
                            )
                        ),
                    )
                )
    return tuple((name, float(value)) for name, value in terms)


def evaluate_depth2_oracle(probe: object) -> Depth2OracleReport:
    """Evaluate the exact tiny normalization and complete-objective oracle."""

    identity = (
        getattr(probe, "probe_id", None),
        getattr(getattr(probe, "cascade", None), "probe_law_id", None),
        getattr(getattr(probe, "cascade", None), "horizon", None),
        getattr(getattr(probe, "cascade", None), "vocabulary_size", None),
        getattr(getattr(probe, "recognition", None), "horizon", None),
    )
    if identity != (
        "h6-depth2-t2-scalar-v3-v1",
        "H6-DEPTH2-CASCADE-v1",
        2,
        3,
        2,
    ):
        raise ValueError("probe does not match the frozen T=2 scalar V=3 law")
    (
        source_path_count,
        token_sequence_count,
        emission_region_count,
        source_mass,
        maximum_emission_mass_error,
    ) = _normalization(probe)
    terms = _objective_terms(probe)
    return Depth2OracleReport(
        schema_version="h6-depth2-independent-oracle-v1",
        probe_id=probe.probe_id,
        source_path_count=source_path_count,
        token_sequence_count=token_sequence_count,
        emission_region_count=emission_region_count,
        gaussian_factor_count=4 * (probe.cascade.horizon + 1),
        source_mass=source_mass,
        maximum_emission_mass_error=maximum_emission_mass_error,
        objective_term_names=tuple(name for name, _ in terms),
        objective_term_values=tuple(value for _, value in terms),
        objective_total=float(math.fsum(value for _, value in terms)),
    )


def _canonical(value: object) -> object:
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("oracle report floats must be finite")
        return value.hex()
    if type(value) in (str, int):
        return value
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    raise ValueError(f"unsupported oracle report value {type(value).__name__}")


def _canonical_report_bytes(report: Depth2OracleReport) -> bytes:
    core = {
        name: _canonical(getattr(report, name))
        for name in (
            "schema_version",
            "probe_id",
            "source_path_count",
            "token_sequence_count",
            "emission_region_count",
            "gaussian_factor_count",
            "source_mass",
            "maximum_emission_mass_error",
            "objective_term_names",
            "objective_term_values",
            "objective_total",
        )
    }
    return json.dumps(
        core,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


__all__ = ["Depth2OracleReport", "evaluate_depth2_oracle"]
