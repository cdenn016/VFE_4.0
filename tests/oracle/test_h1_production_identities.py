from __future__ import annotations

import inspect
import math

import pytest
import torch

from vfe4.generative import H1GenerativeModel
from vfe4.objective import evaluate_local_elbo, evaluate_monolithic_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.validation import enumerate_source_paths, load_h1_fixture


def _models() -> tuple[object, H1GenerativeModel, H1RecognitionLaw]:
    fixture = load_h1_fixture()
    return (
        fixture,
        H1GenerativeModel.from_fixture(fixture),
        H1RecognitionLaw.from_fixture(fixture),
    )


def _comparison_roundoff(*values: float) -> float:
    return 32.0 * math.ulp(1.0) * math.fsum(abs(value) for value in values)


def _joint_gaussian_entropy(covariance: torch.Tensor) -> float:
    chol = torch.linalg.cholesky(covariance)
    log_determinant = 2.0 * torch.log(torch.diagonal(chol)).sum().item()
    return 0.5 * (covariance.shape[0] * (1.0 + math.log(2.0 * math.pi)) + log_determinant)


def test_production_elbos_agree_with_paired_calibrated_allowance() -> None:
    fixture, model, recognition = _models()

    monolithic = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    allowance = (
        monolithic.numerical_allowance.total
        + local.allowances.complete_elbo.total
        + _comparison_roundoff(monolithic.value, local.complete_elbo)
    )

    assert abs(monolithic.value - local.complete_elbo) <= allowance
    assert monolithic.quadrature_order == 21
    assert monolithic.convergence_check_order == 17
    assert monolithic.numerical_allowance.convergence_estimate < fixture.maximum_convergence_estimate


def test_complete_component_gaussian_ratio_matches_conditional_kl_chain_rule() -> None:
    fixture, model, recognition = _models()
    monolithic = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    weights = tuple(
        recognition.source_probability(path).item()
        for path in enumerate_source_paths(fixture)
    )
    complete_ratio = math.fsum(
        weight * ratio
        for weight, ratio in zip(weights, monolithic.component_gaussian_log_ratios)
    )
    local_gaussian_kl = math.fsum(
        (
            local.initial_model_kl,
            local.initial_state_kl,
            *local.model_transition_kl,
            *local.state_transition_kl,
        )
    )
    allowance = (
        local.allowances.initial_model_kl.total
        + local.allowances.initial_state_kl.total
        + math.fsum(item.total for item in local.allowances.model_transition_kl)
        + math.fsum(item.total for item in local.allowances.state_transition_kl)
        + _comparison_roundoff(complete_ratio, local_gaussian_kl)
    )

    assert abs(complete_ratio + local_gaussian_kl) <= allowance


def test_joint_recognition_entropy_matches_direct_component_entropy() -> None:
    fixture, model, recognition = _models()
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    direct_contributions: list[float] = []
    for path in enumerate_source_paths(fixture):
        weight = recognition.source_probability(path).item()
        direct_contributions.append(-weight * math.log(weight))
        direct_contributions.append(
            weight * _joint_gaussian_entropy(recognition.joint_component(path).covariance)
        )
    direct_entropy = math.fsum(direct_contributions)
    allowance = local.allowances.joint_recognition_entropy.total + _comparison_roundoff(
        direct_entropy, local.joint_recognition_entropy
    )

    assert abs(direct_entropy - local.joint_recognition_entropy) <= allowance


def test_omitting_recognition_source_entropy_exceeds_paired_allowance() -> None:
    fixture, model, recognition = _models()
    monolithic = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    weights = tuple(
        recognition.source_probability(path).item()
        for path in enumerate_source_paths(fixture)
    )
    source_entropy = -math.fsum(weight * math.log(weight) for weight in weights)
    injected_without_source_entropy = monolithic.value - source_entropy
    allowance = (
        monolithic.numerical_allowance.total
        + local.allowances.complete_elbo.total
        + _comparison_roundoff(injected_without_source_entropy, local.complete_elbo)
    )

    assert abs(injected_without_source_entropy - local.complete_elbo) > allowance


def test_replacing_log_softmax_with_raw_selected_logits_exceeds_paired_allowance() -> None:
    fixture, model, recognition = _models()
    monolithic = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    local = evaluate_local_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    factors = model.factors
    raw_expectations = [0.0, 0.0]
    for path in enumerate_source_paths(fixture):
        weight = recognition.source_probability(path).item()
        component = recognition.joint_component(path)
        for time, selected_index in enumerate((0, 1), start=1):
            emission = factors.emissions[time - 1]
            raw_selected_logit = (
                emission.w_z[selected_index] * component.mean[2 * time]
                + emission.w_m[selected_index] * component.mean[2 * time + 1]
                + emission.bias[selected_index]
            ).item()
            raw_expectations[time - 1] += weight * raw_selected_logit
    injected_raw_logits = monolithic.value + math.fsum(
        raw - correct
        for raw, correct in zip(raw_expectations, monolithic.expected_log_emission)
    )
    allowance = (
        monolithic.numerical_allowance.total
        + local.allowances.complete_elbo.total
        + _comparison_roundoff(injected_raw_logits, local.complete_elbo)
    )

    assert abs(injected_raw_logits - local.complete_elbo) > allowance


def test_monolithic_path_does_not_use_pointwise_or_local_objective_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, model, recognition = _models()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("complete pointwise density method was called")

    monkeypatch.setattr(model, "log_joint", forbidden)
    monkeypatch.setattr(recognition, "log_prob", forbidden)
    result = evaluate_monolithic_elbo(
        model,
        recognition,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    source = inspect.getsource(inspect.getmodule(evaluate_monolithic_elbo))

    assert math.isfinite(result.value)
    assert "h1_local" not in source


@pytest.mark.parametrize(
    ("quadrature_order", "convergence_check_order"),
    [(17, 17), (21, 21), (19, 17), (21, False)],
)
def test_monolithic_rejects_nonfrozen_quadrature_orders(
    quadrature_order: int, convergence_check_order: int
) -> None:
    _, model, recognition = _models()

    with pytest.raises(ValueError, match="quadrature"):
        evaluate_monolithic_elbo(
            model,
            recognition,
            quadrature_order=quadrature_order,
            convergence_check_order=convergence_check_order,
        )
