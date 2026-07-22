"""Direct information-form evaluation of the frozen H1 complete ELBO."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from vfe4.generative import H1GenerativeModel, assemble_generative_information
from vfe4.numerics import probabilists_gauss_hermite
from vfe4.numerics.information import InformationGaussian
from vfe4.recognition import H1RecognitionLaw, assemble_recognition_information
from vfe4.types import (
    ElboTermAllowances,
    ElboTerms,
    MatrixBlock,
    NumericalAllowance,
    PrecisionDiagnostics,
    SourcePath,
)
from vfe4.types.h1 import EmissionRecord, H1RecognitionFactorRecord

_DIMENSION = 6
_OBSERVATION_INDICES = (0, 1)
_SOURCE_PATHS = tuple(
    SourcePath((0, state_source), (0, model_source))
    for model_source, state_source in product(range(2), repeat=2)
)
_FLOAT64_EPSILON = float(torch.finfo(torch.float64).eps)


@dataclass(frozen=True)
class RoundingInputs:
    """Scale and conditioning inputs retained for a later H2 error budget."""

    output_inf_norm: float
    absolute_summand_accumulation_inf: float
    spd_kappa2: float

    def __post_init__(self) -> None:
        _nonnegative(self.output_inf_norm, "output_inf_norm")
        _nonnegative(
            self.absolute_summand_accumulation_inf,
            "absolute_summand_accumulation_inf",
        )
        _positive(self.spd_kappa2, "spd_kappa2")


@dataclass(frozen=True)
class H2ComponentTerms:
    """Every contribution from one positive-weight fixed-source component."""

    path: SourcePath
    weight: float
    q_log_normalizer: float
    p_log_normalizer: float
    q_entropy: float
    gaussian_kl: float
    gaussian_log_ratio: float
    source_log_ratio: float
    expected_log_emission: tuple[float, float]
    complete_value: float
    rounding_inputs: Mapping[str, RoundingInputs] = field(compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, SourcePath):
            raise ValueError("path must be a SourcePath")
        _positive(self.weight, "weight")
        for name in (
            "q_log_normalizer",
            "p_log_normalizer",
            "q_entropy",
            "gaussian_kl",
            "gaussian_log_ratio",
            "source_log_ratio",
            "complete_value",
        ):
            _finite(getattr(self, name), name)
        if self.gaussian_kl < 0.0:
            raise ValueError("gaussian_kl must be nonnegative")
        if self.gaussian_log_ratio != -self.gaussian_kl:
            raise ValueError("gaussian_log_ratio must be the negative oriented KL")
        _finite_pair(self.expected_log_emission, "expected_log_emission")
        reconstructed = math.fsum(
            (
                self.gaussian_log_ratio,
                self.source_log_ratio,
                *self.expected_log_emission,
            )
        )
        if self.complete_value != reconstructed:
            raise ValueError("complete_value must use the component contribution partition")
        object.__setattr__(
            self,
            "rounding_inputs",
            _immutable_rounding_mapping(self.rounding_inputs),
        )


@dataclass(frozen=True)
class H2ComponentDiagnostics:
    """Precision and mean-envelope diagnostics for one q/p component pair."""

    path: SourcePath
    q_precision: PrecisionDiagnostics
    p_precision: PrecisionDiagnostics
    q_mean_inf_norm: float
    p_mean_inf_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.path, SourcePath):
            raise ValueError("path must be a SourcePath")
        if not isinstance(self.q_precision, PrecisionDiagnostics) or not isinstance(
            self.p_precision, PrecisionDiagnostics
        ):
            raise ValueError("component precisions must have diagnostics")
        _nonnegative(self.q_mean_inf_norm, "q_mean_inf_norm")
        _nonnegative(self.p_mean_inf_norm, "p_mean_inf_norm")


@dataclass(frozen=True)
class H2InformationEvaluation:
    """Complete direct-information evaluation and its local H1 partition."""

    components: tuple[
        H2ComponentTerms,
        H2ComponentTerms,
        H2ComponentTerms,
        H2ComponentTerms,
    ]
    local_terms: ElboTerms
    source_entropy: float
    weighted_component_entropy: float
    joint_recognition_entropy: float
    complete_elbo: float
    rounding_inputs: Mapping[str, RoundingInputs] = field(compare=False)
    component_diagnostics: tuple[
        H2ComponentDiagnostics,
        H2ComponentDiagnostics,
        H2ComponentDiagnostics,
        H2ComponentDiagnostics,
    ]

    def __post_init__(self) -> None:
        if type(self.components) is not tuple or len(self.components) != 4 or not all(
            isinstance(item, H2ComponentTerms) for item in self.components
        ):
            raise ValueError("components must contain four H2ComponentTerms")
        if not isinstance(self.local_terms, ElboTerms):
            raise ValueError("local_terms must be ElboTerms")
        for name in (
            "source_entropy",
            "weighted_component_entropy",
            "joint_recognition_entropy",
            "complete_elbo",
        ):
            _finite(getattr(self, name), name)
        if self.source_entropy < 0.0:
            raise ValueError("source_entropy must be nonnegative")
        if self.joint_recognition_entropy != math.fsum(
            (self.source_entropy, self.weighted_component_entropy)
        ):
            raise ValueError("joint recognition entropy must reconstruct")
        weighted = math.fsum(
            component.weight * component.complete_value
            for component in self.components
        )
        if self.complete_elbo != weighted:
            raise ValueError("complete_elbo must use the source-weighted components")
        object.__setattr__(
            self,
            "rounding_inputs",
            _immutable_rounding_mapping(self.rounding_inputs),
        )
        if (
            type(self.component_diagnostics) is not tuple
            or len(self.component_diagnostics) != 4
            or not all(
                isinstance(item, H2ComponentDiagnostics)
                for item in self.component_diagnostics
            )
        ):
            raise ValueError("component_diagnostics must contain four records")


@dataclass(frozen=True)
class _Reduction:
    value: float
    absolute_sum: float
    spd_kappa2: float = 1.0

    def __post_init__(self) -> None:
        _finite(self.value, "reduction value")
        _nonnegative(self.absolute_sum, "reduction absolute sum")
        _positive(self.spd_kappa2, "reduction SPD condition number")

    @property
    def rounding(self) -> RoundingInputs:
        return RoundingInputs(abs(self.value), self.absolute_sum, self.spd_kappa2)


@dataclass(frozen=True)
class _SelectedMoments:
    mean: Tensor
    covariance: Tensor
    second_moment: Tensor
    spd_kappa2: float


@dataclass(frozen=True)
class _ComponentWork:
    terms: H2ComponentTerms
    q_info: InformationGaussian
    p_info: InformationGaussian
    initial: _SelectedMoments
    model_pairs: tuple[_SelectedMoments, _SelectedMoments]
    state_triples: tuple[_SelectedMoments, _SelectedMoments]
    emissions: tuple[_SelectedMoments, _SelectedMoments]
    emission_reductions: tuple[_Reduction, _Reduction]


def evaluate_information_elbo(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    quadrature_order: int,
) -> H2InformationEvaluation:
    """Evaluate all four positive-weight paths through information-form factors."""

    _validate_inputs(model, recognition, quadrature_order)
    weights = tuple(_source_weight(recognition.factors, path) for path in _SOURCE_PATHS)
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("every recognition path weight must be positive")
    if abs(math.fsum(weights) - 1.0) > 64.0 * _FLOAT64_EPSILON:
        raise ValueError("recognition path weights must sum to one")

    work = tuple(
        _evaluate_component(
            model,
            recognition,
            path=path,
            weight=weight,
            quadrature_order=quadrature_order,
        )
        for path, weight in zip(_SOURCE_PATHS, weights)
    )
    components = tuple(item.terms for item in work)

    source_entropy_contributions = tuple(-weight * math.log(weight) for weight in weights)
    source_entropy = _Reduction(
        math.fsum(source_entropy_contributions),
        math.fsum(abs(value) for value in source_entropy_contributions),
    )
    weighted_entropy_contributions = tuple(
        item.terms.weight * item.terms.q_entropy for item in work
    )
    entropy_kappa = max(
        item.q_info.factor.diagnostics.kappa_2 for item in work
    )
    weighted_component_entropy = _Reduction(
        math.fsum(weighted_entropy_contributions),
        math.fsum(abs(value) for value in weighted_entropy_contributions),
        entropy_kappa,
    )
    joint_recognition_entropy = _Reduction(
        math.fsum((source_entropy.value, weighted_component_entropy.value)),
        math.fsum((source_entropy.absolute_sum, weighted_component_entropy.absolute_sum)),
        entropy_kappa,
    )

    local_terms, local_rounding = _evaluate_local_partition(
        model,
        recognition,
        work,
        joint_recognition_entropy,
    )
    weighted_component_values = tuple(
        component.weight * component.complete_value for component in components
    )
    complete_elbo = math.fsum(weighted_component_values)
    complete_reduction = _Reduction(
        complete_elbo,
        math.fsum(abs(value) for value in weighted_component_values),
        max(
            item.terms.rounding_inputs["complete_value"].spd_kappa2
            for item in work
        ),
    )
    rounding_inputs = dict(local_rounding)
    rounding_inputs.update(
        {
            "source_entropy": source_entropy.rounding,
            "weighted_component_entropy": weighted_component_entropy.rounding,
            "joint_recognition_entropy": joint_recognition_entropy.rounding,
            "complete_elbo": complete_reduction.rounding,
        }
    )
    diagnostics = tuple(
        H2ComponentDiagnostics(
            path=item.terms.path,
            q_precision=item.q_info.factor.diagnostics,
            p_precision=item.p_info.factor.diagnostics,
            q_mean_inf_norm=_tensor_inf_norm(item.q_info.mean()),
            p_mean_inf_norm=_tensor_inf_norm(item.p_info.mean()),
        )
        for item in work
    )
    return H2InformationEvaluation(
        components=components,  # type: ignore[arg-type]
        local_terms=local_terms,
        source_entropy=source_entropy.value,
        weighted_component_entropy=weighted_component_entropy.value,
        joint_recognition_entropy=joint_recognition_entropy.value,
        complete_elbo=complete_elbo,
        rounding_inputs=rounding_inputs,
        component_diagnostics=diagnostics,  # type: ignore[arg-type]
    )


def _evaluate_component(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    *,
    path: SourcePath,
    weight: float,
    quadrature_order: int,
) -> _ComponentWork:
    q_info = assemble_recognition_information(recognition.factors, path)
    p_info = assemble_generative_information(model.factors, path)
    mu_q = q_info.mean()

    initial = _selected_moments(q_info, mu_q, (0, 1))
    model_pairs: list[_SelectedMoments] = []
    state_triples: list[_SelectedMoments] = []
    for time in (1, 2):
        a = path.a[time - 1]
        b = path.b[time - 1]
        model_index = 2 * time + 1
        state_index = 2 * time
        model_pairs.append(
            _selected_moments(q_info, mu_q, (2 * b + 1, model_index))
        )
        state_triples.append(
            _selected_moments(q_info, mu_q, (2 * a, model_index, state_index))
        )
    emission_moments = tuple(
        _selected_moments(q_info, mu_q, (2 * time, 2 * time + 1))
        for time in (1, 2)
    )

    q_log_normalizer = _log_normalizer_reduction(q_info, mu_q)
    p_mean = p_info.mean()
    p_log_normalizer = _log_normalizer_reduction(p_info, p_mean)
    q_entropy = _entropy_reduction(q_info)
    gaussian_kl = _gaussian_kl_reduction(q_info, p_info, mu_q, p_mean)
    gaussian_log_ratio = _Reduction(
        -gaussian_kl.value,
        gaussian_kl.absolute_sum,
        gaussian_kl.spd_kappa2,
    )
    source_log_ratio = _source_log_ratio(model, path, weight)
    emission_reductions = tuple(
        _emission_expectation(
            emission,
            selected_index=selected_index,
            moments=moments,
            order=quadrature_order,
        )
        for emission, selected_index, moments in zip(
            model.factors.emissions,
            _OBSERVATION_INDICES,
            emission_moments,
        )
    )
    complete_value = math.fsum(
        (
            gaussian_log_ratio.value,
            source_log_ratio.value,
            emission_reductions[0].value,
            emission_reductions[1].value,
        )
    )
    complete_reduction = _Reduction(
        complete_value,
        math.fsum(
            (
                gaussian_log_ratio.absolute_sum,
                source_log_ratio.absolute_sum,
                emission_reductions[0].absolute_sum,
                emission_reductions[1].absolute_sum,
            )
        ),
        max(
            gaussian_log_ratio.spd_kappa2,
            emission_reductions[0].spd_kappa2,
            emission_reductions[1].spd_kappa2,
        ),
    )
    rounding = {
        "q_log_normalizer": q_log_normalizer.rounding,
        "p_log_normalizer": p_log_normalizer.rounding,
        "q_entropy": q_entropy.rounding,
        "gaussian_kl": gaussian_kl.rounding,
        "gaussian_log_ratio": gaussian_log_ratio.rounding,
        "source_log_ratio": source_log_ratio.rounding,
        "expected_log_emission[0]": emission_reductions[0].rounding,
        "expected_log_emission[1]": emission_reductions[1].rounding,
        "complete_value": complete_reduction.rounding,
    }
    terms = H2ComponentTerms(
        path=path,
        weight=weight,
        q_log_normalizer=q_log_normalizer.value,
        p_log_normalizer=p_log_normalizer.value,
        q_entropy=q_entropy.value,
        gaussian_kl=gaussian_kl.value,
        gaussian_log_ratio=gaussian_log_ratio.value,
        source_log_ratio=source_log_ratio.value,
        expected_log_emission=(
            emission_reductions[0].value,
            emission_reductions[1].value,
        ),
        complete_value=complete_value,
        rounding_inputs=rounding,
    )
    return _ComponentWork(
        terms=terms,
        q_info=q_info,
        p_info=p_info,
        initial=initial,
        model_pairs=tuple(model_pairs),  # type: ignore[arg-type]
        state_triples=tuple(state_triples),  # type: ignore[arg-type]
        emissions=emission_moments,  # type: ignore[arg-type]
        emission_reductions=emission_reductions,  # type: ignore[arg-type]
    )


def _evaluate_local_partition(
    model: H1GenerativeModel,
    recognition: H1RecognitionLaw,
    work: tuple[_ComponentWork, ...],
    joint_entropy: _Reduction,
) -> tuple[ElboTerms, Mapping[str, RoundingInputs]]:
    initial_model_kl, initial_state_kl = _initial_kls(
        work[0].initial,
        model.factors.initial_joint.mean,
        model.factors.initial_joint.covariance,
    )
    model_source_kl = tuple(
        _categorical_kl(q, p)
        for q, p in zip(
            recognition.factors.model_source_probabilities,
            model.factors.model_source_priors,
        )
    )
    state_source_kl = tuple(
        _state_source_kl(q_b, q_a_given_b, p_a)
        for q_b, q_a_given_b, p_a in zip(
            recognition.factors.model_source_probabilities,
            recognition.factors.state_source_probabilities_given_model_source,
            model.factors.state_source_priors,
        )
    )
    model_transition_kl = tuple(
        _weighted_transition_kl(
            work,
            time=time,
            state=False,
            recognition=recognition,
            model=model,
        )
        for time in (1, 2)
    )
    state_transition_kl = tuple(
        _weighted_transition_kl(
            work,
            time=time,
            state=True,
            recognition=recognition,
            model=model,
        )
        for time in (1, 2)
    )
    expected_log_emission = tuple(
        _weighted_component_reduction(
            tuple(item.emission_reductions[time] for item in work),
            tuple(item.terms.weight for item in work),
        )
        for time in range(2)
    )
    signed_values = (
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
    local_complete = _Reduction(
        math.fsum(signed_values),
        math.fsum(abs(value) for value in signed_values),
        max(
            reduction.spd_kappa2
            for reduction in (
                *expected_log_emission,
                initial_model_kl,
                initial_state_kl,
                *model_source_kl,
                *model_transition_kl,
                *state_source_kl,
                *state_transition_kl,
            )
        ),
    )
    terms = ElboTerms(
        expected_log_emission=tuple(item.value for item in expected_log_emission),  # type: ignore[arg-type]
        initial_model_kl=initial_model_kl.value,
        initial_state_kl=initial_state_kl.value,
        model_source_kl=tuple(item.value for item in model_source_kl),  # type: ignore[arg-type]
        model_transition_kl=tuple(item.value for item in model_transition_kl),  # type: ignore[arg-type]
        state_source_kl=tuple(item.value for item in state_source_kl),  # type: ignore[arg-type]
        state_transition_kl=tuple(item.value for item in state_transition_kl),  # type: ignore[arg-type]
        joint_recognition_entropy=joint_entropy.value,
        allowances=_local_allowances(
            expected_log_emission,
            initial_model_kl,
            initial_state_kl,
            model_source_kl,
            model_transition_kl,
            state_source_kl,
            state_transition_kl,
            joint_entropy,
            local_complete,
        ),
        complete_elbo=local_complete.value,
    )
    rounding = {
        "local.expected_log_emission[0]": expected_log_emission[0].rounding,
        "local.expected_log_emission[1]": expected_log_emission[1].rounding,
        "local.initial_model_kl": initial_model_kl.rounding,
        "local.initial_state_kl": initial_state_kl.rounding,
        "local.model_source_kl[0]": model_source_kl[0].rounding,
        "local.model_transition_kl[0]": model_transition_kl[0].rounding,
        "local.model_source_kl[1]": model_source_kl[1].rounding,
        "local.model_transition_kl[1]": model_transition_kl[1].rounding,
        "local.state_source_kl[0]": state_source_kl[0].rounding,
        "local.state_transition_kl[0]": state_transition_kl[0].rounding,
        "local.state_source_kl[1]": state_source_kl[1].rounding,
        "local.state_transition_kl[1]": state_transition_kl[1].rounding,
        "local.joint_recognition_entropy": joint_entropy.rounding,
        "local.complete_elbo": local_complete.rounding,
    }
    return terms, MappingProxyType(rounding)


def _selected_moments(
    info: InformationGaussian, mean: Tensor, indices: tuple[int, ...]
) -> _SelectedMoments:
    block = MatrixBlock(indices, indices)
    second_moment = info.selected_moment_blocks((block,))[block]
    selector = torch.tensor(indices, dtype=torch.int64, device=mean.device)
    selected_mean = mean.index_select(0, selector)
    covariance = second_moment - torch.outer(selected_mean, selected_mean)
    _require_finite_tensor(covariance, "selected covariance")
    eigenvalues = torch.linalg.eigvalsh(covariance)
    _require_finite_tensor(eigenvalues, "selected covariance eigenvalues")
    lambda_min = float(eigenvalues[0].item())
    lambda_max = float(eigenvalues[-1].item())
    if lambda_min <= 0.0:
        raise ValueError("selected covariance must be positive definite")
    return _SelectedMoments(
        mean=selected_mean,
        covariance=covariance,
        second_moment=second_moment,
        spd_kappa2=lambda_max / lambda_min,
    )


def _log_normalizer_reduction(
    info: InformationGaussian, mean: Tensor
) -> _Reduction:
    dot = _tensor_scalar(torch.dot(info.h, mean), "information linear term")
    logdet = _tensor_scalar(info.factor.logdet(), "precision log determinant")
    dimension_term = info.dimension * math.log(2.0 * math.pi)
    value = _tensor_scalar(info.log_normalizer(), "log normalizer")
    return _Reduction(
        value,
        0.5 * math.fsum((abs(dot), abs(logdet), abs(dimension_term))),
        info.factor.diagnostics.kappa_2,
    )


def _entropy_reduction(info: InformationGaussian) -> _Reduction:
    dimension_term = info.dimension * (1.0 + math.log(2.0 * math.pi))
    logdet = _tensor_scalar(info.factor.logdet(), "precision log determinant")
    value = _tensor_scalar(info.entropy(), "Gaussian entropy")
    return _Reduction(
        value,
        0.5 * math.fsum((abs(dimension_term), abs(logdet))),
        info.factor.diagnostics.kappa_2,
    )


def _gaussian_kl_reduction(
    q: InformationGaussian,
    p: InformationGaussian,
    q_mean: Tensor,
    p_mean: Tensor,
) -> _Reduction:
    delta = p_mean - q_mean
    trace = _tensor_scalar(
        q.factor.trace_inverse_product(p.factor), "Gaussian KL trace"
    )
    quadratic = _tensor_scalar(p.factor.quadratic(delta), "Gaussian KL quadratic")
    q_logdet = _tensor_scalar(q.factor.logdet(), "q precision log determinant")
    p_logdet = _tensor_scalar(p.factor.logdet(), "p precision log determinant")
    value = _tensor_scalar(q.oriented_kl(p), "oriented Gaussian KL")
    if value < 0.0:
        raise ValueError("oriented Gaussian KL must be nonnegative")
    return _Reduction(
        value,
        0.5
        * math.fsum(
            (abs(trace), abs(quadratic), float(q.dimension), abs(q_logdet), abs(p_logdet))
        ),
        max(q.factor.diagnostics.kappa_2, p.factor.diagnostics.kappa_2),
    )


def _source_log_ratio(
    model: H1GenerativeModel, path: SourcePath, weight: float
) -> _Reduction:
    log_prior_terms: list[float] = []
    model_priors = model.factors.model_source_priors
    state_priors = model.factors.state_source_priors
    for time in range(2):
        model_prior = _positive(
            _tensor_scalar(model_priors[time][path.b[time]], "model source prior"),
            "model source prior",
        )
        state_prior = _positive(
            _tensor_scalar(state_priors[time][path.a[time]], "state source prior"),
            "state source prior",
        )
        log_prior_terms.extend((math.log(model_prior), math.log(state_prior)))
    recognition_log_weight = math.log(_positive(weight, "recognition path weight"))
    value = math.fsum((*log_prior_terms, -recognition_log_weight))
    return _Reduction(
        value,
        math.fsum(abs(item) for item in (*log_prior_terms, recognition_log_weight)),
    )


def _emission_expectation(
    emission: EmissionRecord,
    *,
    selected_index: int,
    moments: _SelectedMoments,
    order: int,
) -> _Reduction:
    chol, info = torch.linalg.cholesky_ex(moments.covariance, check_errors=False)
    if int(info.item()) != 0 or not bool(torch.isfinite(chol).all()):
        raise ValueError("emission covariance must have a finite Cholesky factor")
    nodes, weights = probabilists_gauss_hermite(order, dtype=torch.float64)
    contributions: list[float] = []
    for first, second in product(range(order), repeat=2):
        standard = torch.stack((nodes[first], nodes[second]))
        value = moments.mean + chol @ standard
        logits = emission.w_z * value[0] + emission.w_m * value[1] + emission.bias
        selected = torch.log_softmax(logits, dim=0)[selected_index]
        try:
            selected_value = _tensor_scalar(selected, "emission selected log-softmax")
        except ValueError as error:
            raise ValueError("emission output must be finite") from error
        contributions.append(
            _tensor_scalar(weights[first], "quadrature weight")
            * _tensor_scalar(weights[second], "quadrature weight")
            * selected_value
        )
    return _Reduction(
        math.fsum(contributions),
        math.fsum(abs(value) for value in contributions),
        moments.spd_kappa2,
    )


def _initial_kls(
    q: _SelectedMoments, p_mean: Tensor, p_covariance: Tensor
) -> tuple[_Reduction, _Reduction]:
    p_kappa = _spd_kappa2(p_covariance, "generative initial covariance")
    model_mean_square = (
        _tensor_scalar(q.mean[1], "q initial model mean")
        - _tensor_scalar(p_mean[1], "p initial model mean")
    ) ** 2
    model_kl = _conditional_gaussian_kl(
        _tensor_scalar(q.covariance[1, 1], "q initial model variance"),
        _tensor_scalar(p_covariance[1, 1], "p initial model variance"),
        model_mean_square,
    )
    q_slope, q_offset, q_variance = _initial_state_conditional(
        q.mean, q.covariance
    )
    p_slope, p_offset, p_variance = _initial_state_conditional(
        p_mean, p_covariance
    )
    slope_difference = q_slope - p_slope
    offset_difference = q_offset - p_offset
    q_model_mean = _tensor_scalar(q.mean[1], "q initial model mean")
    q_model_variance = _tensor_scalar(
        q.covariance[1, 1], "q initial model variance"
    )
    state_mean_square = (
        slope_difference * slope_difference * q_model_variance
        + (slope_difference * q_model_mean + offset_difference) ** 2
    )
    state_kl = _conditional_gaussian_kl(
        q_variance, p_variance, state_mean_square
    )
    kappa = max(q.spd_kappa2, p_kappa)
    return (
        _Reduction(model_kl.value, model_kl.absolute_sum, kappa),
        _Reduction(state_kl.value, state_kl.absolute_sum, kappa),
    )


def _initial_state_conditional(
    mean: Tensor, covariance: Tensor
) -> tuple[float, float, float]:
    model_variance = _positive(
        _tensor_scalar(covariance[1, 1], "initial model variance"),
        "initial model variance",
    )
    cross = _tensor_scalar(covariance[0, 1], "initial cross covariance")
    slope = cross / model_variance
    offset = _tensor_scalar(mean[0], "initial state mean") - slope * _tensor_scalar(
        mean[1], "initial model mean"
    )
    variance = _tensor_scalar(covariance[0, 0], "initial state variance") - (
        cross * cross / model_variance
    )
    return slope, offset, _positive(variance, "initial conditional variance")


def _weighted_transition_kl(
    work: tuple[_ComponentWork, ...],
    *,
    time: int,
    state: bool,
    recognition: H1RecognitionLaw,
    model: H1GenerativeModel,
) -> _Reduction:
    contributions: list[float] = []
    absolute_sums: list[float] = []
    kappas: list[float] = []
    for item in work:
        path = item.terms.path
        if state:
            selected = item.state_triples[time - 1]
            a = path.a[time - 1]
            b = path.b[time - 1]
            slot = 0 if time == 1 else a + 2 * b
            q_kernel = recognition.factors.state_kernels[time - 1]
            p_kernel = model.factors.state_transitions[time - 1]
            slope_difference = torch.tensor(
                [
                    _tensor_scalar(q_kernel.z_slopes[slot], "q state slope")
                    - _tensor_scalar(p_kernel.source_slopes[a], "p state slope"),
                    _tensor_scalar(q_kernel.m_slopes[slot], "q state model slope")
                    - _tensor_scalar(p_kernel.model_slope, "p state model slope"),
                ],
                dtype=torch.float64,
            )
            offset_difference = _tensor_scalar(
                q_kernel.offsets[slot], "q state offset"
            ) - _tensor_scalar(p_kernel.offset, "p state offset")
            parent_mean = selected.mean[:2]
            parent_covariance = selected.covariance[:2, :2]
            centered = torch.dot(slope_difference, parent_mean) + offset_difference
            mean_square = _tensor_scalar(
                torch.dot(
                    slope_difference, parent_covariance @ slope_difference
                )
                + centered * centered,
                "state transition mean square",
            )
            reduction = _conditional_gaussian_kl(
                _tensor_scalar(q_kernel.variances[slot], "q state variance"),
                _tensor_scalar(p_kernel.variance, "p state variance"),
                mean_square,
            )
        else:
            selected = item.model_pairs[time - 1]
            b = path.b[time - 1]
            q_kernel = recognition.factors.model_kernels[time - 1]
            p_kernel = model.factors.model_transitions[time - 1]
            slope_difference = _tensor_scalar(
                q_kernel.slopes[b], "q model slope"
            ) - _tensor_scalar(p_kernel.source_slopes[b], "p model slope")
            offset_difference = _tensor_scalar(
                q_kernel.offsets[b], "q model offset"
            ) - _tensor_scalar(p_kernel.offset, "p model offset")
            parent_mean = _tensor_scalar(selected.mean[0], "model parent mean")
            parent_variance = _tensor_scalar(
                selected.covariance[0, 0], "model parent variance"
            )
            mean_square = (
                slope_difference * slope_difference * parent_variance
                + (slope_difference * parent_mean + offset_difference) ** 2
            )
            reduction = _conditional_gaussian_kl(
                _tensor_scalar(q_kernel.variances[b], "q model variance"),
                _tensor_scalar(p_kernel.variance, "p model variance"),
                mean_square,
            )
        weighted = item.terms.weight * reduction.value
        contributions.append(weighted)
        absolute_sums.append(item.terms.weight * reduction.absolute_sum)
        kappas.append(selected.spd_kappa2)
    return _Reduction(
        math.fsum(contributions),
        math.fsum(absolute_sums),
        max(kappas),
    )


def _conditional_gaussian_kl(
    q_variance: float, p_variance: float, mean_square: float
) -> _Reduction:
    q_checked = _positive(q_variance, "q conditional variance")
    p_checked = _positive(p_variance, "p conditional variance")
    square_checked = _nonnegative(mean_square, "conditional mean square")
    log_ratio = math.log(p_checked / q_checked)
    variance_ratio = (q_checked + square_checked) / p_checked
    value = _nonnegative(
        0.5 * (log_ratio + variance_ratio - 1.0),
        "conditional Gaussian KL",
    )
    return _Reduction(
        value,
        0.5 * math.fsum((abs(log_ratio), abs(variance_ratio), 1.0)),
    )


def _categorical_kl(q: Tensor, p: Tensor) -> _Reduction:
    if q.shape != p.shape:
        raise ValueError("categorical probability shapes must match")
    contributions: list[float] = []
    for q_tensor, p_tensor in zip(q, p):
        q_value = _nonnegative(
            _tensor_scalar(q_tensor, "q source probability"),
            "q source probability",
        )
        p_value = _nonnegative(
            _tensor_scalar(p_tensor, "p source probability"),
            "p source probability",
        )
        if q_value > 0.0:
            if p_value <= 0.0:
                raise ValueError("recognition source mass is outside generative support")
            contributions.append(q_value * (math.log(q_value) - math.log(p_value)))
    value = math.fsum(contributions)
    return _Reduction(
        _nonnegative(value, "categorical KL"),
        math.fsum(abs(item) for item in contributions),
    )


def _state_source_kl(q_b: Tensor, q_a_given_b: Tensor, p_a: Tensor) -> _Reduction:
    if q_a_given_b.shape != (q_b.numel(), p_a.numel()):
        raise ValueError("state-source table shape must match source vectors")
    contributions: list[float] = []
    for b in range(q_b.numel()):
        q_b_value = _tensor_scalar(q_b[b], "model source probability")
        for a in range(p_a.numel()):
            q_a_value = _tensor_scalar(
                q_a_given_b[b, a], "state source probability"
            )
            p_a_value = _tensor_scalar(p_a[a], "state source prior")
            if q_b_value > 0.0 and q_a_value > 0.0:
                if p_a_value <= 0.0:
                    raise ValueError(
                        "recognition state-source mass is outside generative support"
                    )
                contributions.append(
                    q_b_value
                    * q_a_value
                    * (math.log(q_a_value) - math.log(p_a_value))
                )
    value = math.fsum(contributions)
    return _Reduction(
        _nonnegative(value, "expected state-source KL"),
        math.fsum(abs(item) for item in contributions),
    )


def _weighted_component_reduction(
    reductions: tuple[_Reduction, ...], weights: tuple[float, ...]
) -> _Reduction:
    values = tuple(weight * item.value for weight, item in zip(weights, reductions))
    return _Reduction(
        math.fsum(values),
        math.fsum(
            weight * item.absolute_sum for weight, item in zip(weights, reductions)
        ),
        max(item.spd_kappa2 for item in reductions),
    )


def _source_weight(factors: H1RecognitionFactorRecord, path: SourcePath) -> float:
    probabilities: list[float] = []
    model_probabilities = factors.model_source_probabilities
    state_probabilities = factors.state_source_probabilities_given_model_source
    for time in range(2):
        probabilities.extend(
            (
                _tensor_scalar(
                    model_probabilities[time][path.b[time]],
                    "model source probability",
                ),
                _tensor_scalar(
                    state_probabilities[time][path.b[time], path.a[time]],
                    "state source probability",
                ),
            )
        )
    return _finite(math.prod(probabilities), "recognition path weight")


def _local_allowances(
    expected_log_emission: tuple[_Reduction, ...],
    initial_model_kl: _Reduction,
    initial_state_kl: _Reduction,
    model_source_kl: tuple[_Reduction, ...],
    model_transition_kl: tuple[_Reduction, ...],
    state_source_kl: tuple[_Reduction, ...],
    state_transition_kl: tuple[_Reduction, ...],
    joint_entropy: _Reduction,
    complete_elbo: _Reduction,
) -> ElboTermAllowances:
    def allowance(item: _Reduction) -> NumericalAllowance:
        return NumericalAllowance(
            convergence_estimate=0.0,
            rounding_allowance=256.0
            * _FLOAT64_EPSILON
            * max(1.0, item.absolute_sum),
        )

    return ElboTermAllowances(
        expected_log_emission=tuple(allowance(item) for item in expected_log_emission),  # type: ignore[arg-type]
        initial_model_kl=allowance(initial_model_kl),
        initial_state_kl=allowance(initial_state_kl),
        model_source_kl=tuple(allowance(item) for item in model_source_kl),  # type: ignore[arg-type]
        model_transition_kl=tuple(allowance(item) for item in model_transition_kl),  # type: ignore[arg-type]
        state_source_kl=tuple(allowance(item) for item in state_source_kl),  # type: ignore[arg-type]
        state_transition_kl=tuple(allowance(item) for item in state_transition_kl),  # type: ignore[arg-type]
        joint_recognition_entropy=allowance(joint_entropy),
        complete_elbo=allowance(complete_elbo),
    )


def _validate_inputs(
    model: object, recognition: object, quadrature_order: object
) -> None:
    if not isinstance(model, H1GenerativeModel):
        raise ValueError("model must be an H1GenerativeModel")
    if not isinstance(recognition, H1RecognitionLaw):
        raise ValueError("recognition must be an H1RecognitionLaw")
    if type(quadrature_order) is not int or quadrature_order != 21:
        raise ValueError("quadrature_order must equal the frozen order 21")


def _immutable_rounding_mapping(
    value: Mapping[str, RoundingInputs]
) -> Mapping[str, RoundingInputs]:
    if not isinstance(value, Mapping):
        raise ValueError("rounding_inputs must be a mapping")
    copied = dict(value)
    if not copied:
        raise ValueError("rounding_inputs must not be empty")
    if any(type(name) is not str or not name for name in copied):
        raise ValueError("rounding input names must be nonempty strings")
    if not all(isinstance(item, RoundingInputs) for item in copied.values()):
        raise ValueError("rounding_inputs must contain RoundingInputs")
    return MappingProxyType(copied)


def _spd_kappa2(value: Tensor, name: str) -> float:
    eigenvalues = torch.linalg.eigvalsh(value)
    _require_finite_tensor(eigenvalues, f"{name} eigenvalues")
    lambda_min = float(eigenvalues[0].item())
    lambda_max = float(eigenvalues[-1].item())
    if lambda_min <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return lambda_max / lambda_min


def _tensor_inf_norm(value: Tensor) -> float:
    _require_finite_tensor(value, "tensor infinity norm input")
    return float(torch.max(torch.abs(value)).item())


def _tensor_scalar(value: Tensor, name: str) -> float:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64 or value.shape != ():
        raise ValueError(f"{name} must be a float64 scalar tensor")
    return _finite(float(value.item()), name)


def _require_finite_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.float64:
        raise ValueError(f"{name} must be a float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _finite_pair(value: object, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    for index, item in enumerate(value):
        _finite(item, f"{name}[{index}]")


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
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


__all__ = [
    "H2ComponentDiagnostics",
    "H2ComponentTerms",
    "H2InformationEvaluation",
    "RoundingInputs",
    "evaluate_information_elbo",
]
