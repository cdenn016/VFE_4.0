"""Independent NumPy dense-moment oracle for the frozen H2 fixture."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


_DIMENSION = 6
_PATHS = (
    ((0, 0), (0, 0)),
    ((0, 1), (0, 0)),
    ((0, 0), (0, 1)),
    ((0, 1), (0, 1)),
)


@dataclass(frozen=True)
class H2OracleGaussian:
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    h: np.ndarray
    log_normalizer: float
    entropy: float
    minimum_cholesky_pivot: float
    lambda_min: float
    lambda_max: float
    kappa_2: float
    absolute_summand_accumulation: Mapping[str, float] = field(compare=False)


@dataclass(frozen=True)
class H2OracleMarginal:
    indices: tuple[int, ...]
    mean: np.ndarray
    covariance: np.ndarray
    absolute_summand_accumulation: float


@dataclass(frozen=True)
class H2MomentOracleComponent:
    path: tuple[tuple[int, int], tuple[int, int]]
    weight: float
    q: H2OracleGaussian
    p: H2OracleGaussian
    emission_marginals: tuple[H2OracleMarginal, H2OracleMarginal]
    gaussian_kl: float
    gaussian_log_ratio: float
    source_log_ratio: float
    expected_log_emission: tuple[float, float]
    complete_value: float
    absolute_summand_accumulation: Mapping[str, float] = field(compare=False)


@dataclass(frozen=True)
class H2OracleLocalTerms:
    expected_log_emission: tuple[float, float]
    initial_model_kl: float
    initial_state_kl: float
    model_source_kl: tuple[float, float]
    model_transition_kl: tuple[float, float]
    state_source_kl: tuple[float, float]
    state_transition_kl: tuple[float, float]
    joint_recognition_entropy: float
    complete_elbo: float
    absolute_summand_accumulation: Mapping[str, float] = field(compare=False)


@dataclass(frozen=True)
class H2MomentOracleEvaluation:
    components: tuple[
        H2MomentOracleComponent,
        H2MomentOracleComponent,
        H2MomentOracleComponent,
        H2MomentOracleComponent,
    ]
    local_terms: H2OracleLocalTerms
    signed_local_terms: tuple[float, ...]
    source_entropy: float
    weighted_component_entropy: float
    joint_recognition_entropy: float
    complete_elbo: float
    absolute_summand_accumulation: Mapping[str, float] = field(compare=False)


@dataclass(frozen=True)
class _Reduction:
    value: float
    absolute_sum: float


def evaluate_h2_moment_oracle(
    fixture_path: Path, *, quadrature_order: int = 21
) -> H2MomentOracleEvaluation:
    """Parse and evaluate ``h1-v1`` without production or H1-oracle imports."""

    if not isinstance(fixture_path, Path):
        raise ValueError("fixture_path must be a pathlib.Path")
    if type(quadrature_order) is not int or quadrature_order != 21:
        raise ValueError("quadrature_order must equal the frozen order 21")
    root = _read_fixture(fixture_path)
    q_data = _mapping(root["recognition"], "recognition")
    q_components: list[H2MomentOracleComponent] = []
    for path in _PATHS:
        q_mean, q_covariance = _assemble_recognition(root, q_data, path)
        p_mean, p_covariance = _assemble_generative(root, path)
        q = _gaussian(q_mean, q_covariance)
        p = _gaussian(p_mean, p_covariance)
        weight = _recognition_weight(q_data, path)
        source_ratio = _generative_source_log_probability(root, path) - math.log(weight)
        marginals = tuple(
            _marginal(q, indices) for indices in ((2, 3), (4, 5))
        )
        emission_reductions = tuple(
            _expected_log_emission(
                _mapping(_sequence(root["decoder"], 2, "decoder")[time], f"decoder[{time}]"),
                marginal,
                int(_sequence(root["observation_labels"], 2, "observation_labels")[time]) - 1,
                quadrature_order,
            )
            for time, marginal in enumerate(marginals)
        )
        gaussian_kl = _oriented_kl(q, p)
        gaussian_log_ratio = -gaussian_kl.value
        complete_value = math.fsum(
            (
                gaussian_log_ratio,
                source_ratio,
                emission_reductions[0].value,
                emission_reductions[1].value,
            )
        )
        accumulations = MappingProxyType(
            {
                "gaussian_kl": gaussian_kl.absolute_sum,
                "gaussian_log_ratio": gaussian_kl.absolute_sum,
                "source_log_ratio": abs(source_ratio),
                "expected_log_emission[0]": emission_reductions[0].absolute_sum,
                "expected_log_emission[1]": emission_reductions[1].absolute_sum,
                "complete_value": math.fsum(
                    (
                        gaussian_kl.absolute_sum,
                        abs(source_ratio),
                        emission_reductions[0].absolute_sum,
                        emission_reductions[1].absolute_sum,
                    )
                ),
            }
        )
        q_components.append(
            H2MomentOracleComponent(
                path=path,
                weight=weight,
                q=q,
                p=p,
                emission_marginals=marginals,  # type: ignore[arg-type]
                gaussian_kl=gaussian_kl.value,
                gaussian_log_ratio=gaussian_log_ratio,
                source_log_ratio=source_ratio,
                expected_log_emission=(
                    emission_reductions[0].value,
                    emission_reductions[1].value,
                ),
                complete_value=complete_value,
                absolute_summand_accumulation=accumulations,
            )
        )

    components = tuple(q_components)
    weights = tuple(component.weight for component in components)
    source_entropy_parts = tuple(-weight * math.log(weight) for weight in weights)
    weighted_entropy_parts = tuple(
        component.weight * component.q.entropy for component in components
    )
    source_entropy = math.fsum(source_entropy_parts)
    weighted_component_entropy = math.fsum(weighted_entropy_parts)
    joint_entropy = math.fsum((source_entropy, weighted_component_entropy))
    local_terms, signed_terms = _local_terms(root, q_data, components, joint_entropy)
    complete_parts = tuple(
        component.weight * component.complete_value for component in components
    )
    complete_elbo = math.fsum(complete_parts)
    return H2MomentOracleEvaluation(
        components=components,  # type: ignore[arg-type]
        local_terms=local_terms,
        signed_local_terms=signed_terms,
        source_entropy=source_entropy,
        weighted_component_entropy=weighted_component_entropy,
        joint_recognition_entropy=joint_entropy,
        complete_elbo=complete_elbo,
        absolute_summand_accumulation=MappingProxyType(
            {
                "source_entropy": math.fsum(abs(value) for value in source_entropy_parts),
                "weighted_component_entropy": math.fsum(
                    abs(value) for value in weighted_entropy_parts
                ),
                "joint_recognition_entropy": math.fsum(
                    (math.fsum(abs(value) for value in source_entropy_parts),
                     math.fsum(abs(value) for value in weighted_entropy_parts))
                ),
                "complete_elbo": math.fsum(abs(value) for value in complete_parts),
            }
        ),
    )


def _read_fixture(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fixture must contain readable JSON bytes") from error
    root = _mapping(parsed, "fixture")
    if root.get("fixture_schema_version") != 1 or root.get("fixture_id") != "h1-v1":
        raise ValueError("unsupported fixture identity")
    if root.get("continuous_order") != ["z0", "m0", "z1", "m1", "z2", "m2"]:
        raise ValueError("continuous_order must match h1-v1")
    return root


def _assemble_recognition(
    root: Mapping[str, object],
    recognition: Mapping[str, object],
    path: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    initial_mean = _vector(recognition["initial_mean"], 2, "recognition.initial_mean")
    initial_covariance = _spd(
        _matrix(recognition["initial_covariance"], 2, "recognition.initial_covariance"),
        "recognition.initial_covariance",
    )
    transform, mean, noise = _initial_affine(initial_mean, initial_covariance)
    state_path, model_path = path
    model_kernels = _sequence(recognition["model_kernels"], 2, "model_kernels")
    state_kernels = _sequence(recognition["state_kernels"], 2, "state_kernels")
    for time in (1, 2):
        a, b = state_path[time - 1], model_path[time - 1]
        model_rows = _sequence(model_kernels[time - 1], time, f"model_kernels[{time - 1}]")
        state_rows = _sequence(
            state_kernels[time - 1], 1 if time == 1 else 4, f"state_kernels[{time - 1}]"
        )
        model = _mapping(model_rows[b], "recognition model kernel")
        state = _mapping(state_rows[0 if time == 1 else a + 2 * b], "recognition state kernel")
        _append_affine(
            transform,
            mean,
            noise,
            time,
            a,
            b,
            _finite(model["slope"], "model slope"),
            _finite(model["offset"], "model offset"),
            _positive(model["variance"], "model variance"),
            _finite(state["z_slope"], "state z slope"),
            _finite(state["m_slope"], "state m slope"),
            _finite(state["offset"], "state offset"),
            _positive(state["variance"], "state variance"),
        )
    return _finish_affine(transform, mean, noise)


def _assemble_generative(
    root: Mapping[str, object], path: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    initial_mean = _vector(
        _mapping(root["initial_joint"], "initial_joint")["mean"], 2, "initial mean"
    )
    initial_covariance = _spd(
        _matrix(
            _mapping(root["initial_joint"], "initial_joint")["covariance"],
            2,
            "initial covariance",
        ),
        "initial covariance",
    )
    transform, mean, noise = _initial_affine(initial_mean, initial_covariance)
    model_offsets = _vector(root["model_offsets"], 2, "model_offsets")
    model_variances = _vector(root["model_variances"], 2, "model_variances", positive=True)
    state_offsets = _vector(root["state_offsets"], 2, "state_offsets")
    state_variances = _vector(root["state_variances"], 2, "state_variances", positive=True)
    state_model_slopes = _vector(root["state_model_slopes"], 2, "state_model_slopes")
    frames = _vector(root["frames"], 3, "frames", positive=True)
    state_path, model_path = path
    for time in (1, 2):
        a, b = state_path[time - 1], model_path[time - 1]
        _append_affine(
            transform,
            mean,
            noise,
            time,
            a,
            b,
            frames[time] / frames[b],
            model_offsets[time - 1],
            model_variances[time - 1],
            frames[time] / frames[a],
            state_model_slopes[time - 1],
            state_offsets[time - 1],
            state_variances[time - 1],
        )
    return _finish_affine(transform, mean, noise)


def _initial_affine(
    mean: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.zeros((_DIMENSION, _DIMENSION), dtype=np.float64)
    transform[0, 0] = transform[1, 1] = 1.0
    result_mean = np.zeros(_DIMENSION, dtype=np.float64)
    result_mean[:2] = mean
    noise = np.zeros((_DIMENSION, _DIMENSION), dtype=np.float64)
    noise[:2, :2] = covariance
    return transform, result_mean, noise


def _append_affine(
    transform: np.ndarray,
    mean: np.ndarray,
    noise: np.ndarray,
    time: int,
    a: int,
    b: int,
    model_slope: float,
    model_offset: float,
    model_variance: float,
    state_slope: float,
    state_model_slope: float,
    state_offset: float,
    state_variance: float,
) -> None:
    m_index, z_index = 2 * time + 1, 2 * time
    m_source, z_source = 2 * b + 1, 2 * a
    transform[m_index] = model_slope * transform[m_source]
    transform[m_index, m_index] += 1.0
    mean[m_index] = model_slope * mean[m_source] + model_offset
    transform[z_index] = (
        state_slope * transform[z_source] + state_model_slope * transform[m_index]
    )
    transform[z_index, z_index] += 1.0
    mean[z_index] = (
        state_slope * mean[z_source] + state_model_slope * mean[m_index] + state_offset
    )
    noise[m_index, m_index] = model_variance
    noise[z_index, z_index] = state_variance


def _finish_affine(
    transform: np.ndarray, mean: np.ndarray, noise: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    covariance = transform @ noise @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    _spd(covariance, "assembled covariance")
    return mean, covariance


def _gaussian(mean: np.ndarray, covariance: np.ndarray) -> H2OracleGaussian:
    chol = np.linalg.cholesky(covariance)
    identity = np.eye(mean.size, dtype=np.float64)
    precision = np.linalg.solve(chol.T, np.linalg.solve(chol, identity))
    precision = 0.5 * (precision + precision.T)
    h = precision @ mean
    precision_chol = np.linalg.cholesky(precision)
    eigenvalues = np.linalg.eigvalsh(precision)
    logdet_precision = 2.0 * math.fsum(math.log(value) for value in np.diag(precision_chol))
    h_mu = float(h @ mean)
    log_normalizer_parts = (h_mu, -logdet_precision, mean.size * math.log(2.0 * math.pi))
    log_normalizer = 0.5 * math.fsum(log_normalizer_parts)
    entropy_parts = (
        mean.size * (1.0 + math.log(2.0 * math.pi)),
        -logdet_precision,
    )
    entropy = 0.5 * math.fsum(entropy_parts)
    return H2OracleGaussian(
        mean=mean.copy(),
        covariance=covariance.copy(),
        precision=precision,
        h=h,
        log_normalizer=log_normalizer,
        entropy=entropy,
        minimum_cholesky_pivot=float(np.min(np.diag(precision_chol))),
        lambda_min=float(eigenvalues[0]),
        lambda_max=float(eigenvalues[-1]),
        kappa_2=float(eigenvalues[-1] / eigenvalues[0]),
        absolute_summand_accumulation=MappingProxyType(
            {
                "mean": float(np.sum(np.abs(mean))),
                "covariance": float(np.sum(np.abs(covariance))),
                "precision": float(np.sum(np.abs(precision))),
                "h": float(np.sum(np.abs(h))),
                "log_normalizer": 0.5 * math.fsum(abs(value) for value in log_normalizer_parts),
                "entropy": 0.5 * math.fsum(abs(value) for value in entropy_parts),
            }
        ),
    )


def _marginal(law: H2OracleGaussian, indices: tuple[int, ...]) -> H2OracleMarginal:
    selected = list(indices)
    mean = law.mean[selected].copy()
    covariance = law.covariance[np.ix_(selected, selected)].copy()
    return H2OracleMarginal(
        indices,
        mean,
        covariance,
        math.fsum((float(np.sum(np.abs(mean))), float(np.sum(np.abs(covariance))))),
    )


def _oriented_kl(q: H2OracleGaussian, p: H2OracleGaussian) -> _Reduction:
    delta = p.mean - q.mean
    trace = float(np.trace(p.precision @ q.covariance))
    quadratic = float(delta @ p.precision @ delta)
    q_logdet = float(np.linalg.slogdet(q.precision)[1])
    p_logdet = float(np.linalg.slogdet(p.precision)[1])
    parts = (trace, quadratic, -float(_DIMENSION), q_logdet, -p_logdet)
    value = 0.5 * math.fsum(parts)
    if value < -1.0e-14:
        raise ValueError("oriented Gaussian KL must be nonnegative")
    return _Reduction(max(0.0, value), 0.5 * math.fsum(abs(value) for value in parts))


def _expected_log_emission(
    decoder: Mapping[str, object],
    marginal: H2OracleMarginal,
    selected_index: int,
    order: int,
) -> _Reduction:
    w_z = _vector(decoder["w_z"], 3, "decoder.w_z")
    w_m = _vector(decoder["w_m"], 3, "decoder.w_m")
    bias = _vector(decoder["bias"], 3, "decoder.bias")
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    normalized = weights / math.sqrt(2.0 * math.pi)
    chol = np.linalg.cholesky(marginal.covariance)
    contributions: list[float] = []
    for first in range(order):
        for second in range(order):
            value = marginal.mean + chol @ np.array((nodes[first], nodes[second]))
            logits = w_z * value[0] + w_m * value[1] + bias
            maximum = float(np.max(logits))
            logsumexp = maximum + math.log(math.fsum(math.exp(float(x - maximum)) for x in logits))
            selected = float(logits[selected_index]) - logsumexp
            contributions.append(float(normalized[first] * normalized[second]) * selected)
    return _Reduction(
        math.fsum(contributions), math.fsum(abs(value) for value in contributions)
    )


def _local_terms(
    root: Mapping[str, object],
    recognition: Mapping[str, object],
    components: tuple[H2MomentOracleComponent, ...],
    joint_entropy: float,
) -> tuple[H2OracleLocalTerms, tuple[float, ...]]:
    p_initial = _mapping(root["initial_joint"], "initial_joint")
    p_mean = _vector(p_initial["mean"], 2, "initial mean")
    p_cov = _matrix(p_initial["covariance"], 2, "initial covariance")
    q_mean = _vector(recognition["initial_mean"], 2, "recognition initial mean")
    q_cov = _matrix(recognition["initial_covariance"], 2, "recognition initial covariance")
    initial_model = _conditional_kl(
        q_cov[1, 1], p_cov[1, 1], (q_mean[1] - p_mean[1]) ** 2
    )
    q_slope, q_offset, q_variance = _initial_conditional(q_mean, q_cov)
    p_slope, p_offset, p_variance = _initial_conditional(p_mean, p_cov)
    difference = q_slope - p_slope
    initial_state = _conditional_kl(
        q_variance,
        p_variance,
        difference * difference * q_cov[1, 1]
        + (difference * q_mean[1] + q_offset - p_offset) ** 2,
    )
    q_b_rows = _sequence(recognition["model_source_probabilities"], 2, "q_b")
    q_a_tables = _sequence(
        recognition["state_source_probabilities_given_model_source"], 2, "q_a"
    )
    p_b_rows = _sequence(root["model_source_priors"], 2, "p_b")
    p_a_rows = _sequence(root["state_source_priors"], 2, "p_a")
    model_source = tuple(
        _categorical_kl(
            _vector(q_b_rows[t], t + 1, "q_b"),
            _vector(p_b_rows[t], t + 1, "p_b"),
        )
        for t in range(2)
    )
    state_source = tuple(
        _state_source_kl(
            _vector(q_b_rows[t], t + 1, "q_b"),
            _matrix(q_a_tables[t], t + 1, "q_a"),
            _vector(p_a_rows[t], t + 1, "p_a"),
        )
        for t in range(2)
    )
    model_transition = tuple(
        _weighted_transition(root, recognition, components, time, state=False)
        for time in (1, 2)
    )
    state_transition = tuple(
        _weighted_transition(root, recognition, components, time, state=True)
        for time in (1, 2)
    )
    emissions = tuple(
        _Reduction(
            math.fsum(component.weight * component.expected_log_emission[t] for component in components),
            math.fsum(component.weight * component.absolute_summand_accumulation[f"expected_log_emission[{t}]"] for component in components),
        )
        for t in range(2)
    )
    signed = (
        emissions[0].value,
        emissions[1].value,
        -initial_model.value,
        -initial_state.value,
        -model_source[0].value,
        -model_transition[0].value,
        -model_source[1].value,
        -model_transition[1].value,
        -state_source[0].value,
        -state_transition[0].value,
        -state_source[1].value,
        -state_transition[1].value,
    )
    absolute = {
        "expected_log_emission[0]": emissions[0].absolute_sum,
        "expected_log_emission[1]": emissions[1].absolute_sum,
        "initial_model_kl": initial_model.absolute_sum,
        "initial_state_kl": initial_state.absolute_sum,
        "model_source_kl[0]": model_source[0].absolute_sum,
        "model_transition_kl[0]": model_transition[0].absolute_sum,
        "model_source_kl[1]": model_source[1].absolute_sum,
        "model_transition_kl[1]": model_transition[1].absolute_sum,
        "state_source_kl[0]": state_source[0].absolute_sum,
        "state_transition_kl[0]": state_transition[0].absolute_sum,
        "state_source_kl[1]": state_source[1].absolute_sum,
        "state_transition_kl[1]": state_transition[1].absolute_sum,
        "joint_recognition_entropy": abs(joint_entropy),
        "complete_elbo": math.fsum(abs(value) for value in signed),
    }
    return (
        H2OracleLocalTerms(
            expected_log_emission=(emissions[0].value, emissions[1].value),
            initial_model_kl=initial_model.value,
            initial_state_kl=initial_state.value,
            model_source_kl=(model_source[0].value, model_source[1].value),
            model_transition_kl=(model_transition[0].value, model_transition[1].value),
            state_source_kl=(state_source[0].value, state_source[1].value),
            state_transition_kl=(state_transition[0].value, state_transition[1].value),
            joint_recognition_entropy=joint_entropy,
            complete_elbo=math.fsum(signed),
            absolute_summand_accumulation=MappingProxyType(absolute),
        ),
        signed,
    )


def _weighted_transition(
    root: Mapping[str, object],
    recognition: Mapping[str, object],
    components: tuple[H2MomentOracleComponent, ...],
    time: int,
    *,
    state: bool,
) -> _Reduction:
    frames = _vector(root["frames"], 3, "frames")
    q_model_rows = _sequence(recognition["model_kernels"], 2, "model_kernels")
    q_state_rows = _sequence(recognition["state_kernels"], 2, "state_kernels")
    contributions: list[float] = []
    absolute: list[float] = []
    for component in components:
        state_path, model_path = component.path
        a, b = state_path[time - 1], model_path[time - 1]
        if state:
            rows = _sequence(q_state_rows[time - 1], 1 if time == 1 else 4, "state kernels")
            q = _mapping(rows[0 if time == 1 else a + 2 * b], "state kernel")
            difference = np.array(
                (
                    _finite(q["z_slope"], "z slope") - frames[time] / frames[a],
                    _finite(q["m_slope"], "m slope")
                    - _vector(root["state_model_slopes"], 2, "state_model_slopes")[time - 1],
                ),
                dtype=np.float64,
            )
            parent_indices = (2 * a, 2 * time + 1)
            mean = component.q.mean[list(parent_indices)]
            covariance = component.q.covariance[np.ix_(parent_indices, parent_indices)]
            offset = _finite(q["offset"], "state offset") - _vector(
                root["state_offsets"], 2, "state_offsets"
            )[time - 1]
            mean_square = float(difference @ covariance @ difference) + float(
                difference @ mean + offset
            ) ** 2
            reduction = _conditional_kl(
                _positive(q["variance"], "state variance"),
                _vector(root["state_variances"], 2, "state_variances")[time - 1],
                mean_square,
            )
        else:
            rows = _sequence(q_model_rows[time - 1], time, "model kernels")
            q = _mapping(rows[b], "model kernel")
            difference = _finite(q["slope"], "model slope") - frames[time] / frames[b]
            parent_index = 2 * b + 1
            offset = _finite(q["offset"], "model offset") - _vector(
                root["model_offsets"], 2, "model_offsets"
            )[time - 1]
            mean_square = (
                difference * difference * component.q.covariance[parent_index, parent_index]
                + (difference * component.q.mean[parent_index] + offset) ** 2
            )
            reduction = _conditional_kl(
                _positive(q["variance"], "model variance"),
                _vector(root["model_variances"], 2, "model_variances")[time - 1],
                float(mean_square),
            )
        contributions.append(component.weight * reduction.value)
        absolute.append(component.weight * reduction.absolute_sum)
    return _Reduction(math.fsum(contributions), math.fsum(absolute))


def _initial_conditional(mean: np.ndarray, covariance: np.ndarray) -> tuple[float, float, float]:
    variance = float(covariance[1, 1])
    slope = float(covariance[0, 1]) / variance
    offset = float(mean[0]) - slope * float(mean[1])
    conditional = float(covariance[0, 0]) - float(covariance[0, 1]) ** 2 / variance
    return slope, offset, _positive(conditional, "initial conditional variance")


def _conditional_kl(q_variance: float, p_variance: float, mean_square: float) -> _Reduction:
    q = _positive(q_variance, "q variance")
    p = _positive(p_variance, "p variance")
    square = _nonnegative(mean_square, "mean square")
    log_ratio = math.log(p / q)
    variance_ratio = (q + square) / p
    return _Reduction(
        max(0.0, 0.5 * math.fsum((log_ratio, variance_ratio, -1.0))),
        0.5 * math.fsum((abs(log_ratio), abs(variance_ratio), 1.0)),
    )


def _categorical_kl(q: np.ndarray, p: np.ndarray) -> _Reduction:
    values = tuple(float(qv) * (math.log(float(qv)) - math.log(float(pv))) for qv, pv in zip(q, p) if qv > 0.0)
    return _Reduction(max(0.0, math.fsum(values)), math.fsum(abs(value) for value in values))


def _state_source_kl(q_b: np.ndarray, q_a: np.ndarray, p_a: np.ndarray) -> _Reduction:
    values = tuple(
        float(q_b[b] * q_a[b, a])
        * (math.log(float(q_a[b, a])) - math.log(float(p_a[a])))
        for b in range(q_b.size)
        for a in range(p_a.size)
        if q_b[b] > 0.0 and q_a[b, a] > 0.0
    )
    return _Reduction(max(0.0, math.fsum(values)), math.fsum(abs(value) for value in values))


def _recognition_weight(
    recognition: Mapping[str, object], path: tuple[tuple[int, int], tuple[int, int]]
) -> float:
    state_path, model_path = path
    model = _sequence(recognition["model_source_probabilities"], 2, "model probabilities")
    state = _sequence(
        recognition["state_source_probabilities_given_model_source"], 2, "state probabilities"
    )
    values = []
    for time in range(2):
        b, a = model_path[time], state_path[time]
        values.append(_vector(model[time], time + 1, "model probability")[b])
        values.append(_matrix(state[time], time + 1, "state probability")[b, a])
    # Preserve the fixture's exact declared decimal weights instead of the
    # incidental last-bit product of their conditional factorization.
    return float(f"{_positive(math.prod(values), 'recognition source weight'):.15g}")


def _generative_source_log_probability(
    root: Mapping[str, object], path: tuple[tuple[int, int], tuple[int, int]]
) -> float:
    state_path, model_path = path
    model = _sequence(root["model_source_priors"], 2, "model priors")
    state = _sequence(root["state_source_priors"], 2, "state priors")
    return math.fsum(
        math.log(_vector(model[t], t + 1, "model prior")[model_path[t]])
        + math.log(_vector(state[t], t + 1, "state prior")[state_path[t]])
        for t in range(2)
    )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, size: int, name: str) -> list[object]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return value


def _vector(
    value: object, size: int, name: str, *, positive: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if positive and not bool(np.all(array > 0.0)):
        raise ValueError(f"{name} must be positive")
    return array


def _matrix(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size, size) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    return array


def _spd(value: np.ndarray, name: str) -> np.ndarray:
    if not np.array_equal(value, value.T):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return value


def _positive(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _nonnegative(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float, np.float64) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "H2MomentOracleComponent",
    "H2MomentOracleEvaluation",
    "H2OracleGaussian",
    "H2OracleLocalTerms",
    "H2OracleMarginal",
    "evaluate_h2_moment_oracle",
]
