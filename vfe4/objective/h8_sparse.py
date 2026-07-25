"""Complete normalized, block-local objective for the H8 synthetic chain."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from vfe4.generative.reference_h8 import H8Problem, validate_h8_problem
from vfe4.numerics.quadrature import probabilists_gauss_hermite
from vfe4.numerics.sparse_information import FactorBackedInformationGaussian
from vfe4.types.h8 import H8ObjectiveTerm, H8ObjectiveTerms


@torch.no_grad()
def evaluate_h8_sparse_objective(
    problem: H8Problem,
    recognition: FactorBackedInformationGaussian,
) -> H8ObjectiveTerms:
    """Evaluate every normalized H8 factor using only selected block moments."""

    problem = validate_h8_problem(problem)
    if type(recognition) is not FactorBackedInformationGaussian:
        raise ValueError("recognition must use the H8 production type")
    if recognition.layout != problem.layout:
        raise ValueError("problem and recognition layouts must match")
    layout = problem.layout
    mean = recognition.mean()
    moments = recognition.selected_moment_blocks()
    diagonal, lower = moments._block_refs()
    initial_value = _normal_log_expectation(
        mean[0],
        _covariance(diagonal[0], mean[0]),
        _tensor(problem.initial_mean),
        _tensor(problem.initial_covariance),
    )
    initial = _term("initial_joint", "initial_joint", None, initial_value)
    model_terms: list[H8ObjectiveTerm] = []
    state_terms: list[H8ObjectiveTerm] = []
    emission21: list[H8ObjectiveTerm] = []
    emission17: list[H8ObjectiveTerm] = []
    z_width = layout.d_z
    for receiver_t in range(1, layout.population_size):
        model = problem.model_transitions[receiver_t - 1]
        state = problem.state_transitions[receiver_t - 1]
        if model.source_support != (receiver_t - 1,) or state.source_support != (receiver_t - 1,):
            raise ValueError("H8 source supports must be exactly singleton parents")
        # ``mean[receiver_t]`` and each selected moment are already local
        # ``[b]``/``[b,b]`` blocks.  Global layout slices are invalid here.
        local_model = slice(layout.d_z, layout.block_size)
        model_value = _transition_log_expectation(
            mean[receiver_t, local_model],
            mean[receiver_t - 1, local_model],
            diagonal[receiver_t, local_model, local_model],
            diagonal[receiver_t - 1, local_model, local_model],
            lower[receiver_t - 1, local_model, local_model],
            _tensor(model.matrix),
            _tensor(model.offset),
            _tensor(model.covariance),
        )
        model_terms.append(_term(f"model_transition:{receiver_t:04d}", "model_transition", receiver_t, model_value))
        z = slice(0, z_width)
        m = slice(z_width, layout.block_size)
        parent_mean = torch.cat((mean[receiver_t - 1, z], mean[receiver_t, m]))
        parent_second = torch.empty((layout.block_size, layout.block_size), dtype=torch.float64)
        parent_second[:z_width, :z_width] = diagonal[receiver_t - 1, z, z]
        parent_second[z_width:, z_width:] = diagonal[receiver_t, m, m]
        lower_m_z = lower[receiver_t - 1, m, z]
        parent_second[z_width:, :z_width] = lower_m_z
        parent_second[:z_width, z_width:] = lower_m_z.T
        target_second = diagonal[receiver_t, z, z]
        target_parent_second = torch.cat(
            (lower[receiver_t - 1, z, z], diagonal[receiver_t, z, m]), dim=1
        )
        state_value = _transition_log_expectation(
            mean[receiver_t, z],
            parent_mean,
            target_second,
            parent_second,
            target_parent_second,
            torch.cat((_tensor(state.state_matrix), _tensor(state.model_matrix)), dim=1),
            _tensor(state.offset),
            _tensor(state.covariance),
        )
        state_terms.append(_term(f"state_transition:{receiver_t:04d}", "state_transition", receiver_t, state_value))
        emission = problem.emissions[receiver_t - 1]
        u_mean = torch.dot(_tensor(emission.weight), mean[receiver_t])
        local_covariance = _covariance(diagonal[receiver_t], mean[receiver_t])
        u_variance = _tensor(emission.weight) @ local_covariance @ _tensor(emission.weight)
        value21 = h8_emission_expectation(u_mean, u_variance, _tensor(problem.alpha), _tensor(emission.bias), emission.observation, order=21)
        value17 = h8_emission_expectation(u_mean, u_variance, _tensor(problem.alpha), _tensor(emission.bias), emission.observation, order=17)
        emission21.append(_term(f"emission_order21:{receiver_t:04d}", "emission_order21", receiver_t, value21))
        emission17.append(_term(f"emission_order17:{receiver_t:04d}", "emission_order17", receiver_t, value17))
    entropy = float(recognition.entropy().item())
    log_normalizer = float(recognition.log_normalizer().item())
    complete_values = (initial.value, *(term.value for term in model_terms), *(term.value for term in state_terms), *(term.value for term in emission21), entropy)
    return H8ObjectiveTerms(
        horizon=layout.horizon,
        initial_joint=initial,
        model_transitions=tuple(model_terms),
        state_transitions=tuple(state_terms),
        emissions_order21=tuple(emission21),
        emissions_order17=tuple(emission17),
        recognition_entropy=entropy,
        log_normalizer=log_normalizer,
        model_source_kl=0.0,
        state_source_kl=0.0,
        source_entropy=0.0,
        quadrature_absolute_difference=math.fsum(abs(left.value - right.value) for left, right in zip(emission21, emission17, strict=True)),
        complete_order21=math.fsum(complete_values),
        absolute_term_sum=math.fsum(abs(value) for value in complete_values),
    )


@torch.no_grad()
def h8_emission_expectation(
    mean: Tensor,
    variance: Tensor,
    alpha: Tensor,
    bias: Tensor,
    observation: int,
    *,
    order: int,
) -> float:
    """Stable GH expectation for one normalized categorical observation."""

    if order not in (17, 21):
        raise ValueError("H8 emission quadrature order must be 17 or 21")
    if type(observation) is not int or not 0 <= observation < 3:
        raise ValueError("observation must be an H8 vocabulary index")
    for value, shape, name in ((mean, (), "mean"), (variance, (), "variance"), (alpha, (3,), "alpha"), (bias, (3,), "bias")):
        if type(value) is not Tensor or value.dtype is not torch.float64 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite float64")
    if float(variance.item()) < 0.0:
        raise ValueError("emission variance must be nonnegative")
    nodes, weights = probabilists_gauss_hermite(order, dtype=torch.float64)
    u = mean + torch.sqrt(variance) * nodes
    logits = u.unsqueeze(1) * alpha.unsqueeze(0) + bias.unsqueeze(0)
    return float(torch.sum(weights * torch.log_softmax(logits, dim=1)[:, observation]).item())


def _transition_log_expectation(
    target_mean: Tensor,
    parent_mean: Tensor,
    target_second: Tensor,
    parent_second: Tensor,
    target_parent_second: Tensor,
    matrix: Tensor,
    offset: Tensor,
    covariance: Tensor,
) -> float:
    target_covariance = _covariance(target_second, target_mean)
    parent_covariance = _covariance(parent_second, parent_mean)
    cross_covariance = target_parent_second - torch.outer(target_mean, parent_mean)
    residual_mean = target_mean - matrix @ parent_mean - offset
    residual_covariance = target_covariance - cross_covariance @ matrix.T - matrix @ cross_covariance.T + matrix @ parent_covariance @ matrix.T
    return _normal_log_expectation(residual_mean, residual_covariance, torch.zeros_like(residual_mean), covariance)


def _normal_log_expectation(mean: Tensor, covariance: Tensor, location: Tensor, factor_covariance: Tensor) -> float:
    residual = mean - location
    chol = torch.linalg.cholesky(factor_covariance)
    precision = torch.cholesky_solve(torch.eye(mean.numel(), dtype=torch.float64), chol)
    quadratic = torch.trace(precision @ covariance) + residual @ precision @ residual
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return float((-0.5 * (mean.numel() * math.log(2.0 * math.pi) + logdet + quadratic)).item())


def _covariance(second: Tensor, mean: Tensor) -> Tensor:
    return second - torch.outer(mean, mean)


def _term(factor_id: str, role: str, receiver_t: int | None, value: float) -> H8ObjectiveTerm:
    return H8ObjectiveTerm(factor_id=factor_id, role=role, receiver_t=receiver_t, value=value, absolute_sum_bound=abs(value))


def _tensor(value: object) -> Tensor:
    return torch.tensor(value, dtype=torch.float64, device="cpu")


__all__ = ["evaluate_h8_sparse_objective", "h8_emission_expectation"]
