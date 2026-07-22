"""Complete-component production ELBO for the frozen H1 reference law."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from vfe4.generative import H1GenerativeModel
from vfe4.numerics import probabilists_gauss_hermite
from vfe4.recognition import H1RecognitionLaw
from vfe4.types import NumericalAllowance, SourcePath
from vfe4.types.h1 import EmissionRecord, GaussianLaw

_OBSERVATION_INDICES = (0, 1)
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class MonolithicElboResult:
    value: float
    component_values: tuple[float, float, float, float]
    component_gaussian_log_ratios: tuple[float, float, float, float]
    component_source_log_ratios: tuple[float, float, float, float]
    component_emission_values: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    expected_log_emission: tuple[float, float]
    quadrature_order: Literal[21]
    convergence_check_order: Literal[17]
    numerical_allowance: NumericalAllowance

    def __post_init__(self) -> None:
        _finite(self.value, "value")
        for name, values, size in (
            ("component_values", self.component_values, 4),
            ("component_gaussian_log_ratios", self.component_gaussian_log_ratios, 4),
            ("component_source_log_ratios", self.component_source_log_ratios, 4),
            ("expected_log_emission", self.expected_log_emission, 2),
        ):
            if type(values) is not tuple or len(values) != size:
                raise ValueError(f"{name} must be a tuple of length {size}")
            for index, value in enumerate(values):
                _finite(value, f"{name}[{index}]")
        if type(self.component_emission_values) is not tuple or len(
            self.component_emission_values
        ) != 4:
            raise ValueError("component_emission_values must be a tuple of length 4")
        for index, values in enumerate(self.component_emission_values):
            if type(values) is not tuple or len(values) != 2:
                raise ValueError(
                    f"component_emission_values[{index}] must be a tuple of length 2"
                )
            for time, value in enumerate(values):
                _finite(value, f"component_emission_values[{index}][{time}]")
            reconstructed = math.fsum(
                (
                    self.component_gaussian_log_ratios[index],
                    self.component_source_log_ratios[index],
                    values[0],
                    values[1],
                )
            )
            allowance = 64.0 * _FLOAT64_EPSILON * max(
                1.0, abs(reconstructed), abs(self.component_values[index])
            )
            if abs(self.component_values[index] - reconstructed) > allowance:
                raise ValueError("component value does not match its path contributions")
        if type(self.quadrature_order) is not int or self.quadrature_order != 21:
            raise ValueError("quadrature_order must equal 21")
        if (
            type(self.convergence_check_order) is not int
            or self.convergence_check_order != 17
        ):
            raise ValueError("convergence_check_order must equal 17")
        if not isinstance(self.numerical_allowance, NumericalAllowance):
            raise ValueError("numerical_allowance must be a NumericalAllowance")


@dataclass(frozen=True)
class _Reduction:
    value: float
    absolute_sum: float

    def __post_init__(self) -> None:
        _finite(self.value, "reduction value")
        _nonnegative_finite(self.absolute_sum, "reduction absolute sum")


@dataclass(frozen=True)
class _OrderEvaluation:
    value: float
    absolute_sum: float
    component_values: tuple[float, float, float, float]
    gaussian_log_ratios: tuple[float, float, float, float]
    source_log_ratios: tuple[float, float, float, float]
    emission_values: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    expected_log_emission: tuple[float, float]


def evaluate_monolithic_elbo(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> MonolithicElboResult:
    """Evaluate the H1 ELBO from four complete six-dimensional components."""
    _validate_inputs(model, recognition, quadrature_order, convergence_check_order)
    reported = _evaluate_order(model, recognition, quadrature_order)
    check = _evaluate_order(model, recognition, convergence_check_order)
    allowance = NumericalAllowance(
        convergence_estimate=abs(reported.value - check.value),
        rounding_allowance=32.0 * _FLOAT64_EPSILON * reported.absolute_sum,
    )
    return MonolithicElboResult(
        value=reported.value,
        component_values=reported.component_values,
        component_gaussian_log_ratios=reported.gaussian_log_ratios,
        component_source_log_ratios=reported.source_log_ratios,
        component_emission_values=reported.emission_values,
        expected_log_emission=reported.expected_log_emission,
        quadrature_order=21,
        convergence_check_order=17,
        numerical_allowance=allowance,
    )


def _evaluate_order(
    model: H1GenerativeModel, recognition: H1RecognitionLaw, order: int
) -> _OrderEvaluation:
    path_values: list[float] = []
    gaussian_ratios: list[float] = []
    source_ratios: list[float] = []
    emission_values: list[tuple[float, float]] = []
    weights: list[float] = []
    weighted_contributions: list[float] = []
    weighted_absolute_contributions: list[float] = []

    emissions = model.factors.emissions
    for path in _source_paths():
        weight = _tensor_scalar(recognition.source_probability(path), "source probability")
        if weight < 0.0:
            raise ValueError("recognition source probabilities must be nonnegative")
        weights.append(weight)
        if weight == 0.0:
            path_values.append(0.0)
            gaussian_ratios.append(0.0)
            source_ratios.append(0.0)
            emission_values.append((0.0, 0.0))
            continue
        q_component = recognition.joint_component(path)
        p_component = model.joint_component(path)
        gaussian_ratio = _complete_gaussian_log_ratio(q_component, p_component)
        source_ratio = _tensor_scalar(model.source_log_prob(path), "source log probability") - math.log(
            weight
        )
        per_time = tuple(
            _expected_emission(
                q_component,
                emission,
                time=time,
                selected_index=selected_index,
                order=order,
            )
            for time, (emission, selected_index) in enumerate(
                zip(emissions, _OBSERVATION_INDICES), start=1
            )
        )
        component_value = math.fsum(
            (gaussian_ratio, source_ratio, per_time[0].value, per_time[1].value)
        )

        gaussian_ratios.append(gaussian_ratio)
        source_ratios.append(source_ratio)
        emission_values.append((per_time[0].value, per_time[1].value))
        path_values.append(component_value)
        for contribution in (gaussian_ratio, source_ratio):
            weighted = weight * contribution
            weighted_contributions.append(weighted)
            weighted_absolute_contributions.append(abs(weighted))
        for reduction in per_time:
            weighted_contributions.append(weight * reduction.value)
            weighted_absolute_contributions.append(weight * reduction.absolute_sum)

    if abs(math.fsum(weights) - 1.0) > 64.0 * _FLOAT64_EPSILON:
        raise ValueError("recognition path weights must sum to one")
    value = math.fsum(weighted_contributions)
    absolute_sum = math.fsum(weighted_absolute_contributions)
    expected_emission = tuple(
        math.fsum(weight * values[time] for weight, values in zip(weights, emission_values))
        for time in range(2)
    )
    return _OrderEvaluation(
        value=_finite(value, "monolithic ELBO"),
        absolute_sum=_nonnegative_finite(absolute_sum, "monolithic weighted absolute sum"),
        component_values=_tuple4(path_values, "component values"),
        gaussian_log_ratios=_tuple4(gaussian_ratios, "Gaussian log ratios"),
        source_log_ratios=_tuple4(source_ratios, "source log ratios"),
        emission_values=tuple(emission_values),  # type: ignore[arg-type]
        expected_log_emission=(
            _finite(expected_emission[0], "expected emission 1"),
            _finite(expected_emission[1], "expected emission 2"),
        ),
    )


def _complete_gaussian_log_ratio(q: GaussianLaw, p: GaussianLaw) -> float:
    q_mean = q.mean
    p_mean = p.mean
    q_covariance = q.covariance
    p_covariance = p.covariance
    if q_mean.shape != (6,) or p_mean.shape != (6,):
        raise ValueError("complete Gaussian components must be six-dimensional")
    q_chol = torch.linalg.cholesky(q_covariance)
    p_chol = torch.linalg.cholesky(p_covariance)
    solved_covariance = torch.cholesky_solve(q_covariance, p_chol)
    displacement = (q_mean - p_mean).unsqueeze(1)
    solved_displacement = torch.cholesky_solve(displacement, p_chol)
    trace_term = torch.trace(solved_covariance)
    quadratic_term = torch.sum(displacement * solved_displacement)
    q_log_determinant = 2.0 * torch.log(torch.diagonal(q_chol)).sum()
    p_log_determinant = 2.0 * torch.log(torch.diagonal(p_chol)).sum()
    derived = (
        q_chol,
        p_chol,
        solved_covariance,
        solved_displacement,
        trace_term,
        quadratic_term,
        q_log_determinant,
        p_log_determinant,
    )
    if not all(bool(torch.isfinite(value).all()) for value in derived):
        raise ValueError("complete Gaussian KL produced a nonfinite intermediate")
    kl = 0.5 * (
        trace_term
        + quadratic_term
        - q_mean.numel()
        + p_log_determinant
        - q_log_determinant
    )
    return -_tensor_scalar(kl, "complete Gaussian KL")


def _expected_emission(
    component: GaussianLaw,
    emission: EmissionRecord,
    *,
    time: int,
    selected_index: int,
    order: int,
) -> _Reduction:
    indices = (2 * time, 2 * time + 1)
    mean = component.mean[list(indices)]
    covariance = component.covariance[list(indices)][:, list(indices)]
    chol = torch.linalg.cholesky(covariance)
    if not bool(torch.isfinite(chol).all()):
        raise ValueError("emission marginal Cholesky factor must be finite")
    nodes, weights = probabilists_gauss_hermite(order, dtype=torch.float64)
    contributions: list[float] = []
    absolute_contributions: list[float] = []
    for first, second in product(range(order), repeat=2):
        standard = torch.stack((nodes[first], nodes[second]))
        value = mean + chol @ standard
        logits = emission.w_z * value[0] + emission.w_m * value[1] + emission.bias
        selected = torch.log_softmax(logits, dim=0)[selected_index]
        contribution = (
            _tensor_scalar(weights[first], "quadrature weight")
            * _tensor_scalar(weights[second], "quadrature weight")
            * _tensor_scalar(selected, "selected log-softmax")
        )
        contributions.append(contribution)
        absolute_contributions.append(abs(contribution))
    return _Reduction(math.fsum(contributions), math.fsum(absolute_contributions))


def _source_paths() -> tuple[SourcePath, SourcePath, SourcePath, SourcePath]:
    return tuple(
        SourcePath((0, state_source), (0, model_source))
        for model_source, state_source in product(range(2), repeat=2)
    )  # type: ignore[return-value]


def _validate_inputs(
    model: object,
    recognition: object,
    quadrature_order: object,
    convergence_check_order: object,
) -> None:
    if not isinstance(model, H1GenerativeModel):
        raise ValueError("model must be an H1GenerativeModel")
    if not isinstance(recognition, H1RecognitionLaw):
        raise ValueError("recognition must be an H1RecognitionLaw")
    if type(quadrature_order) is not int or quadrature_order != 21:
        raise ValueError("quadrature_order must equal the frozen order 21")
    if type(convergence_check_order) is not int or convergence_check_order != 17:
        raise ValueError("quadrature convergence_check_order must equal the frozen order 17")


def _tuple4(values: list[float], name: str) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values")
    return tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _tensor_scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64 or value.shape != ():
        raise ValueError(f"{name} must be a float64 scalar tensor")
    return _finite(float(value.item()), name)


def _nonnegative_finite(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)
