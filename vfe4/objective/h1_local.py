"""Conditional-factor production ELBO for the frozen H1 reference law."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor

from vfe4.generative import H1GenerativeModel
from vfe4.numerics import probabilists_gauss_hermite
from vfe4.recognition import H1RecognitionLaw
from vfe4.types import ElboTermAllowances, ElboTerms, NumericalAllowance, SourcePath
from vfe4.types.h1 import EmissionRecord, H1RecognitionFactorRecord

_OBSERVATION_INDICES = (0, 1)
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class _Reduction:
    value: float
    absolute_sum: float

    def __post_init__(self) -> None:
        _finite(self.value, "reduction value")
        _nonnegative_finite(self.absolute_sum, "reduction absolute sum")


@dataclass(frozen=True)
class _Moments:
    mean: Tensor
    covariance: Tensor

    def __post_init__(self) -> None:
        if self.mean.dtype is not torch.float64 or self.mean.shape != (6,):
            raise ValueError("component mean must be a float64 six-vector")
        if self.covariance.dtype is not torch.float64 or self.covariance.shape != (6, 6):
            raise ValueError("component covariance must be a float64 6x6 matrix")
        if not bool(torch.isfinite(self.mean).all()) or not bool(torch.isfinite(self.covariance).all()):
            raise ValueError("component moments must be finite")
        torch.linalg.cholesky(self.covariance)


@dataclass(frozen=True)
class _LocalEvaluation:
    expected_log_emission: tuple[_Reduction, _Reduction]
    initial_model_kl: _Reduction
    initial_state_kl: _Reduction
    model_source_kl: tuple[_Reduction, _Reduction]
    model_transition_kl: tuple[_Reduction, _Reduction]
    state_source_kl: tuple[_Reduction, _Reduction]
    state_transition_kl: tuple[_Reduction, _Reduction]
    joint_recognition_entropy: _Reduction
    complete_elbo: _Reduction


def evaluate_local_elbo(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    quadrature_order: int,
    convergence_check_order: int,
) -> ElboTerms:
    """Evaluate the H1 ELBO in manuscript conditional-factor order."""
    _validate_inputs(model, recognition, quadrature_order, convergence_check_order)
    reported = _evaluate_order(model, recognition, quadrature_order)
    check = _evaluate_order(model, recognition, convergence_check_order)
    allowances = ElboTermAllowances(
        expected_log_emission=tuple(
            _allowance(reported.expected_log_emission[index], check.expected_log_emission[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        initial_model_kl=_allowance(reported.initial_model_kl, check.initial_model_kl),
        initial_state_kl=_allowance(reported.initial_state_kl, check.initial_state_kl),
        model_source_kl=tuple(
            _allowance(reported.model_source_kl[index], check.model_source_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        model_transition_kl=tuple(
            _allowance(reported.model_transition_kl[index], check.model_transition_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        state_source_kl=tuple(
            _allowance(reported.state_source_kl[index], check.state_source_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        state_transition_kl=tuple(
            _allowance(reported.state_transition_kl[index], check.state_transition_kl[index])
            for index in range(2)
        ),  # type: ignore[arg-type]
        joint_recognition_entropy=_allowance(
            reported.joint_recognition_entropy, check.joint_recognition_entropy
        ),
        complete_elbo=_allowance(reported.complete_elbo, check.complete_elbo),
    )
    return ElboTerms(
        expected_log_emission=tuple(
            item.value for item in reported.expected_log_emission
        ),  # type: ignore[arg-type]
        initial_model_kl=reported.initial_model_kl.value,
        initial_state_kl=reported.initial_state_kl.value,
        model_source_kl=tuple(item.value for item in reported.model_source_kl),  # type: ignore[arg-type]
        model_transition_kl=tuple(
            item.value for item in reported.model_transition_kl
        ),  # type: ignore[arg-type]
        state_source_kl=tuple(item.value for item in reported.state_source_kl),  # type: ignore[arg-type]
        state_transition_kl=tuple(
            item.value for item in reported.state_transition_kl
        ),  # type: ignore[arg-type]
        joint_recognition_entropy=reported.joint_recognition_entropy.value,
        allowances=allowances,
        complete_elbo=reported.complete_elbo.value,
    )


def _evaluate_order(
    model: H1GenerativeModel, recognition: H1RecognitionLaw, order: int
) -> _LocalEvaluation:
    generative = model.factors
    recognized = recognition.factors
    paths = _source_paths()
    weights = tuple(_source_weight(recognized, path) for path in paths)
    if abs(math.fsum(weights) - 1.0) > 64.0 * _FLOAT64_EPSILON:
        raise ValueError("recognition path weights must sum to one")
    moments = tuple(_recognition_moments(recognized, path) for path in paths)

    expected_log_emission = tuple(
        _mixture_emission_expectation(
            generative.emissions[time - 1],
            selected_index=_OBSERVATION_INDICES[time - 1],
            time=time,
            order=order,
            weights=weights,
            moments=moments,
        )
        for time in (1, 2)
    )
    initial_model_kl, initial_state_kl = _initial_kls(
        recognized.initial_joint.mean,
        recognized.initial_joint.covariance,
        generative.initial_joint.mean,
        generative.initial_joint.covariance,
    )
    model_source_kl = tuple(
        _categorical_kl_reduction(
            recognized.model_source_probabilities[time],
            generative.model_source_priors[time],
        )
        for time in range(2)
    )
    model_transition_kl = tuple(
        _model_transition_kl_reduction(
            time=time,
            weights=weights,
            paths=paths,
            moments=moments,
            recognized=recognized,
            model=model,
        )
        for time in (1, 2)
    )
    state_source_kl = tuple(
        _state_source_kl_reduction(
            recognized.model_source_probabilities[time],
            recognized.state_source_probabilities_given_model_source[time],
            generative.state_source_priors[time],
        )
        for time in range(2)
    )
    state_transition_kl = tuple(
        _state_transition_kl_reduction(
            time=time,
            weights=weights,
            paths=paths,
            moments=moments,
            recognized=recognized,
            model=model,
        )
        for time in (1, 2)
    )
    entropy = _recognition_entropy(recognized, weights, paths)

    objective_terms = (
        *expected_log_emission,
        initial_model_kl,
        initial_state_kl,
        *model_source_kl,
        *model_transition_kl,
        *state_source_kl,
        *state_transition_kl,
    )
    complete_value = math.fsum(
        (
            expected_log_emission[0].value,
            expected_log_emission[1].value,
            -initial_model_kl.value,
            -initial_state_kl.value,
            -model_source_kl[0].value,
            -model_transition_kl[0].value,
            -model_source_kl[1].value,
            -model_transition_kl[1].value,
            -state_source_kl[0].value,
            -state_transition_kl[0].value,
            -state_source_kl[1].value,
            -state_transition_kl[1].value,
        )
    )
    complete_absolute_sum = math.fsum(item.absolute_sum for item in objective_terms)
    return _LocalEvaluation(
        expected_log_emission=expected_log_emission,  # type: ignore[arg-type]
        initial_model_kl=initial_model_kl,
        initial_state_kl=initial_state_kl,
        model_source_kl=model_source_kl,  # type: ignore[arg-type]
        model_transition_kl=model_transition_kl,  # type: ignore[arg-type]
        state_source_kl=state_source_kl,  # type: ignore[arg-type]
        state_transition_kl=state_transition_kl,  # type: ignore[arg-type]
        joint_recognition_entropy=entropy,
        complete_elbo=_Reduction(complete_value, complete_absolute_sum),
    )


def _recognition_moments(factors: H1RecognitionFactorRecord, path: SourcePath) -> _Moments:
    transform = torch.zeros((6, 6), dtype=torch.float64)
    transform[0, 0] = 1.0
    transform[1, 1] = 1.0
    mean = torch.zeros(6, dtype=torch.float64)
    mean[:2] = factors.initial_joint.mean
    noise_covariance = torch.zeros((6, 6), dtype=torch.float64)
    noise_covariance[:2, :2] = factors.initial_joint.covariance
    for time in (1, 2):
        a = path.a[time - 1]
        b = path.b[time - 1]
        model_kernel = factors.model_kernels[time - 1]
        state_kernel = factors.state_kernels[time - 1]
        state_slot = 0 if time == 1 else a + 2 * b
        m_index, z_index = 2 * time + 1, 2 * time
        m_source, z_source = 2 * b + 1, 2 * a
        m_slope = model_kernel.slopes[b]
        z_slope = state_kernel.z_slopes[state_slot]
        state_m_slope = state_kernel.m_slopes[state_slot]
        transform[m_index] = m_slope * transform[m_source]
        transform[m_index, m_index] += 1.0
        mean[m_index] = m_slope * mean[m_source] + model_kernel.offsets[b]
        transform[z_index] = z_slope * transform[z_source] + state_m_slope * transform[m_index]
        transform[z_index, z_index] += 1.0
        mean[z_index] = (
            z_slope * mean[z_source]
            + state_m_slope * mean[m_index]
            + state_kernel.offsets[state_slot]
        )
        noise_covariance[m_index, m_index] = model_kernel.variances[b]
        noise_covariance[z_index, z_index] = state_kernel.variances[state_slot]
    covariance = transform @ noise_covariance @ transform.transpose(0, 1)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return _Moments(mean, covariance)


def _initial_kls(
    q_mean: Tensor, q_covariance: Tensor, p_mean: Tensor, p_covariance: Tensor
) -> tuple[_Reduction, _Reduction]:
    model_kl = _conditional_gaussian_kl(
        q_variance=_scalar(q_covariance[1, 1], "q m0 variance"),
        p_variance=_scalar(p_covariance[1, 1], "p m0 variance"),
        mean_square=(_scalar(q_mean[1], "q m0 mean") - _scalar(p_mean[1], "p m0 mean")) ** 2,
    )
    q_slope, q_offset, q_variance = _initial_state_conditional(q_mean, q_covariance)
    p_slope, p_offset, p_variance = _initial_state_conditional(p_mean, p_covariance)
    slope_difference = q_slope - p_slope
    offset_difference = q_offset - p_offset
    q_model_mean = _scalar(q_mean[1], "q m0 mean")
    q_model_variance = _scalar(q_covariance[1, 1], "q m0 variance")
    mean_square = (
        slope_difference * slope_difference * q_model_variance
        + (slope_difference * q_model_mean + offset_difference) ** 2
    )
    state_kl = _conditional_gaussian_kl(q_variance, p_variance, mean_square)
    return _Reduction(model_kl, abs(model_kl)), _Reduction(state_kl, abs(state_kl))


def _initial_state_conditional(mean: Tensor, covariance: Tensor) -> tuple[float, float, float]:
    model_variance = _positive(_scalar(covariance[1, 1], "m0 variance"), "m0 variance")
    slope = _scalar(covariance[0, 1], "z0-m0 covariance") / model_variance
    offset = _scalar(mean[0], "z0 mean") - slope * _scalar(mean[1], "m0 mean")
    variance = _scalar(covariance[0, 0], "z0 variance") - (
        _scalar(covariance[0, 1], "z0-m0 covariance") ** 2 / model_variance
    )
    return _finite(slope, "conditional slope"), _finite(offset, "conditional offset"), _positive(
        variance, "conditional variance"
    )


def _model_transition_kl_reduction(
    *,
    time: int,
    weights: tuple[float, float, float, float],
    paths: tuple[SourcePath, SourcePath, SourcePath, SourcePath],
    moments: tuple[_Moments, _Moments, _Moments, _Moments],
    recognized: H1RecognitionFactorRecord,
    model: H1GenerativeModel,
) -> _Reduction:
    contributions: list[float] = []
    for weight, path, component in zip(weights, paths, moments):
        b = path.b[time - 1]
        parent_index = 2 * b + 1
        q_kernel = recognized.model_kernels[time - 1]
        p_kernel = model.factors.model_transitions[time - 1]
        slope_difference = _scalar(q_kernel.slopes[b], "q model slope") - _scalar(
            p_kernel.source_slopes[b], "p model slope"
        )
        offset_difference = _scalar(q_kernel.offsets[b], "q model offset") - _scalar(
            p_kernel.offset, "p model offset"
        )
        parent_mean = _scalar(component.mean[parent_index], "model parent mean")
        parent_variance = _scalar(
            component.covariance[parent_index, parent_index], "model parent variance"
        )
        mean_square = (
            slope_difference * slope_difference * parent_variance
            + (slope_difference * parent_mean + offset_difference) ** 2
        )
        kl = _conditional_gaussian_kl(
            _scalar(q_kernel.variances[b], "q model variance"),
            _scalar(p_kernel.variance, "p model variance"),
            mean_square,
        )
        contributions.append(weight * kl)
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _state_transition_kl_reduction(
    *,
    time: int,
    weights: tuple[float, float, float, float],
    paths: tuple[SourcePath, SourcePath, SourcePath, SourcePath],
    moments: tuple[_Moments, _Moments, _Moments, _Moments],
    recognized: H1RecognitionFactorRecord,
    model: H1GenerativeModel,
) -> _Reduction:
    contributions: list[float] = []
    for weight, path, component in zip(weights, paths, moments):
        a = path.a[time - 1]
        b = path.b[time - 1]
        slot = 0 if time == 1 else a + 2 * b
        q_kernel = recognized.state_kernels[time - 1]
        p_kernel = model.factors.state_transitions[time - 1]
        slope_difference = torch.tensor(
            [
                _scalar(q_kernel.z_slopes[slot], "q state slope")
                - _scalar(p_kernel.source_slopes[a], "p state slope"),
                _scalar(q_kernel.m_slopes[slot], "q state-model slope")
                - _scalar(p_kernel.model_slope, "p state-model slope"),
            ],
            dtype=torch.float64,
        )
        offset_difference = _scalar(q_kernel.offsets[slot], "q state offset") - _scalar(
            p_kernel.offset, "p state offset"
        )
        parent_indices = [2 * a, 2 * time + 1]
        parent_mean = component.mean[parent_indices]
        parent_covariance = component.covariance[parent_indices][:, parent_indices]
        centered_mean = torch.dot(slope_difference, parent_mean) + offset_difference
        mean_square_tensor = (
            torch.dot(slope_difference, parent_covariance @ slope_difference)
            + centered_mean * centered_mean
        )
        mean_square = _scalar(mean_square_tensor, "state conditional mean square")
        kl = _conditional_gaussian_kl(
            _scalar(q_kernel.variances[slot], "q state variance"),
            _scalar(p_kernel.variance, "p state variance"),
            mean_square,
        )
        contributions.append(weight * kl)
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _conditional_gaussian_kl(q_variance: float, p_variance: float, mean_square: float) -> float:
    q_checked = _positive(q_variance, "q conditional variance")
    p_checked = _positive(p_variance, "p conditional variance")
    mean_checked = _nonnegative_finite(mean_square, "conditional mean square")
    value = 0.5 * (
        math.log(p_checked / q_checked)
        + (q_checked + mean_checked) / p_checked
        - 1.0
    )
    return _nonnegative_finite(value, "conditional Gaussian KL")


def _categorical_kl_reduction(q: Tensor, p: Tensor) -> _Reduction:
    if q.shape != p.shape:
        raise ValueError("categorical q and p must have the same shape")
    contributions: list[float] = []
    for q_value_tensor, p_value_tensor in zip(q, p):
        q_value = _nonnegative_finite(_scalar(q_value_tensor, "q source probability"), "q source probability")
        p_value = _nonnegative_finite(_scalar(p_value_tensor, "p source probability"), "p source probability")
        if q_value > 0.0:
            if p_value <= 0.0:
                raise ValueError("recognition source mass lies outside generative support")
            contributions.append(q_value * (math.log(q_value) - math.log(p_value)))
    value = math.fsum(contributions)
    return _Reduction(_nonnegative_finite(value, "categorical KL"), math.fsum(abs(item) for item in contributions))


def _state_source_kl_reduction(q_b: Tensor, q_a_given_b: Tensor, p_a: Tensor) -> _Reduction:
    if q_a_given_b.shape != (q_b.numel(), p_a.numel()):
        raise ValueError("state source table shape does not match source vectors")
    contributions: list[float] = []
    for b in range(q_b.numel()):
        q_b_value = _scalar(q_b[b], "model source probability")
        for a in range(p_a.numel()):
            q_a_value = _scalar(q_a_given_b[b, a], "state source probability")
            p_a_value = _scalar(p_a[a], "state source prior")
            if q_b_value > 0.0 and q_a_value > 0.0:
                if p_a_value <= 0.0:
                    raise ValueError("recognition state-source mass lies outside generative support")
                contributions.append(
                    q_b_value * q_a_value * (math.log(q_a_value) - math.log(p_a_value))
                )
    value = math.fsum(contributions)
    return _Reduction(_nonnegative_finite(value, "expected state-source KL"), math.fsum(abs(item) for item in contributions))


def _mixture_emission_expectation(
    emission: EmissionRecord,
    *,
    selected_index: int,
    time: int,
    order: int,
    weights: tuple[float, float, float, float],
    moments: tuple[_Moments, _Moments, _Moments, _Moments],
) -> _Reduction:
    nodes, quadrature_weights = probabilists_gauss_hermite(order, dtype=torch.float64)
    contributions: list[float] = []
    for path_weight, component in zip(weights, moments):
        indices = [2 * time, 2 * time + 1]
        mean = component.mean[indices]
        covariance = component.covariance[indices][:, indices]
        chol = torch.linalg.cholesky(covariance)
        if not bool(torch.isfinite(chol).all()):
            raise ValueError("emission marginal Cholesky factor must be finite")
        for first, second in product(range(order), repeat=2):
            standard = torch.stack((nodes[first], nodes[second]))
            value = mean + chol @ standard
            logits = emission.w_z * value[0] + emission.w_m * value[1] + emission.bias
            selected = torch.log_softmax(logits, dim=0)[selected_index]
            contributions.append(
                path_weight
                * _scalar(quadrature_weights[first], "quadrature weight")
                * _scalar(quadrature_weights[second], "quadrature weight")
                * _scalar(selected, "selected log-softmax")
            )
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _recognition_entropy(
    factors: H1RecognitionFactorRecord,
    weights: tuple[float, float, float, float],
    paths: tuple[SourcePath, SourcePath, SourcePath, SourcePath],
) -> _Reduction:
    contributions: list[float] = []
    for weight in weights:
        if weight <= 0.0:
            raise ValueError("recognition path weights must be positive")
        contributions.append(-weight * math.log(weight))
    initial_covariance = factors.initial_joint.covariance
    initial_chol = torch.linalg.cholesky(initial_covariance)
    initial_log_determinant = 2.0 * torch.log(torch.diagonal(initial_chol)).sum()
    initial_entropy = 0.5 * (
        2.0 * (1.0 + math.log(2.0 * math.pi))
        + _scalar(initial_log_determinant, "initial log determinant")
    )
    contributions.append(initial_entropy)
    for weight, path in zip(weights, paths):
        for time in (1, 2):
            b = path.b[time - 1]
            a = path.a[time - 1]
            slot = 0 if time == 1 else a + 2 * b
            model_variance = _scalar(
                factors.model_kernels[time - 1].variances[b], "recognition model variance"
            )
            state_variance = _scalar(
                factors.state_kernels[time - 1].variances[slot], "recognition state variance"
            )
            contributions.append(weight * _scalar_gaussian_entropy(model_variance))
            contributions.append(weight * _scalar_gaussian_entropy(state_variance))
    return _Reduction(math.fsum(contributions), math.fsum(abs(item) for item in contributions))


def _scalar_gaussian_entropy(variance: float) -> float:
    return _finite(
        0.5 * math.log(2.0 * math.pi * math.e * _positive(variance, "entropy variance")),
        "scalar Gaussian entropy",
    )


def _source_weight(factors: H1RecognitionFactorRecord, path: SourcePath) -> float:
    probabilities: list[float] = []
    model_probabilities = factors.model_source_probabilities
    state_probabilities = factors.state_source_probabilities_given_model_source
    for time in range(2):
        b = path.b[time]
        a = path.a[time]
        probabilities.append(_scalar(model_probabilities[time][b], "model source probability"))
        probabilities.append(_scalar(state_probabilities[time][b, a], "state source probability"))
    return _positive(math.prod(probabilities), "recognition source weight")


def _source_paths() -> tuple[SourcePath, SourcePath, SourcePath, SourcePath]:
    return tuple(
        SourcePath((0, state_source), (0, model_source))
        for model_source, state_source in product(range(2), repeat=2)
    )  # type: ignore[return-value]


def _allowance(reported: _Reduction, check: _Reduction) -> NumericalAllowance:
    return NumericalAllowance(
        convergence_estimate=abs(reported.value - check.value),
        rounding_allowance=32.0 * _FLOAT64_EPSILON * reported.absolute_sum,
    )


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


def _scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64 or value.shape != ():
        raise ValueError(f"{name} must be a float64 scalar tensor")
    return _finite(float(value.item()), name)


def _positive(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _nonnegative_finite(value: object, name: str) -> float:
    checked = _finite(value, name)
    if checked < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)
