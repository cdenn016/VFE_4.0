from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import vfe4.objective as objective
from vfe4.generative import H1GenerativeModel
from vfe4.numerics.precision import DenseCholeskyPrecision
from vfe4.objective import (
    H2ComponentTerms,
    H2InformationEvaluation,
    RoundingInputs,
    evaluate_information_elbo,
)
from vfe4.recognition import H1RecognitionLaw
from vfe4.types import SourcePath
from vfe4.validation import load_h1_fixture


def _inputs() -> tuple[H1GenerativeModel, H1RecognitionLaw]:
    fixture = load_h1_fixture()
    return (
        H1GenerativeModel.from_fixture(fixture),
        H1RecognitionLaw.from_fixture(fixture),
    )


def _evaluate() -> H2InformationEvaluation:
    model, recognition = _inputs()
    return evaluate_information_elbo(model, recognition, quadrature_order=21)


def test_evaluates_all_components_in_exact_source_order_and_reconstructs_each() -> None:
    evaluation = _evaluate()
    expected_paths = (
        SourcePath(a=(0, 0), b=(0, 0)),
        SourcePath(a=(0, 1), b=(0, 0)),
        SourcePath(a=(0, 0), b=(0, 1)),
        SourcePath(a=(0, 1), b=(0, 1)),
    )

    assert tuple(component.path for component in evaluation.components) == expected_paths
    assert tuple(component.weight for component in evaluation.components) == pytest.approx(
        (0.30, 0.10, 0.12, 0.48)
    )
    for component in evaluation.components:
        assert component.complete_value == math.fsum(
            (
                component.gaussian_log_ratio,
                component.source_log_ratio,
                *component.expected_log_emission,
            )
        )
        assert component.gaussian_log_ratio == -component.gaussian_kl
        assert math.isfinite(component.q_log_normalizer)
        assert math.isfinite(component.p_log_normalizer)

    weighted = math.fsum(
        component.weight * component.complete_value
        for component in evaluation.components
    )
    assert evaluation.complete_elbo == weighted


def test_joint_entropy_and_local_partition_reconstruct_independently() -> None:
    evaluation = _evaluate()
    expected_source_entropy = -math.fsum(
        component.weight * math.log(component.weight)
        for component in evaluation.components
    )
    expected_weighted_component_entropy = math.fsum(
        component.weight * component.q_entropy
        for component in evaluation.components
    )
    assert evaluation.source_entropy == expected_source_entropy
    assert evaluation.weighted_component_entropy == expected_weighted_component_entropy
    assert evaluation.joint_recognition_entropy == math.fsum(
        (expected_source_entropy, expected_weighted_component_entropy)
    )

    local = evaluation.local_terms
    signed_local_terms = (
        local.expected_log_emission[0],
        local.expected_log_emission[1],
        -local.initial_model_kl,
        -local.initial_state_kl,
        -local.model_source_kl[0],
        -local.model_transition_kl[0],
        -local.model_source_kl[1],
        -local.model_transition_kl[1],
        -local.state_source_kl[0],
        -local.state_transition_kl[0],
        -local.state_source_kl[1],
        -local.state_transition_kl[1],
    )
    assert len(signed_local_terms) == 12
    assert local.complete_elbo == math.fsum(signed_local_terms)
    assert evaluation.complete_elbo == pytest.approx(local.complete_elbo, abs=1e-12)


def test_uses_only_factor_surfaces_and_small_selected_moment_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, recognition = _inputs()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forbidden H1 or dense-inverse surface was called")

    monkeypatch.setattr(H1GenerativeModel, "joint_component", forbidden)
    monkeypatch.setattr(H1RecognitionLaw, "joint_component", forbidden)
    monkeypatch.setattr(objective, "evaluate_local_elbo", forbidden)
    monkeypatch.setattr(objective, "evaluate_monolithic_elbo", forbidden)
    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch.linalg, "pinv", forbidden)
    monkeypatch.setattr(torch, "cholesky_inverse", forbidden)

    requests: list[tuple[int, ...]] = []
    original = DenseCholeskyPrecision.selected_inverse

    def instrumented_selected_inverse(
        self: DenseCholeskyPrecision, blocks: object
    ) -> object:
        checked = tuple(blocks)  # type: ignore[arg-type]
        assert len(checked) == 1
        columns = tuple(dict.fromkeys(checked[0].columns))
        assert set(columns) != set(range(6))
        requests.append(columns)
        return original(self, checked)

    monkeypatch.setattr(
        DenseCholeskyPrecision,
        "selected_inverse",
        instrumented_selected_inverse,
    )

    evaluation = evaluate_information_elbo(model, recognition, quadrature_order=21)

    assert len(evaluation.components) == 4
    assert len(requests) == 7 * 4
    assert all(1 < len(columns) <= 3 for columns in requests)


def test_result_records_and_rounding_metadata_are_immutable() -> None:
    evaluation = _evaluate()

    assert isinstance(evaluation, H2InformationEvaluation)
    assert isinstance(evaluation.components[0], H2ComponentTerms)
    assert isinstance(evaluation.rounding_inputs["complete_elbo"], RoundingInputs)
    assert evaluation.rounding_inputs["complete_elbo"].spd_kappa2 > 0.0
    assert len(evaluation.component_diagnostics) == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluation.complete_elbo = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        evaluation.rounding_inputs["complete_elbo"] = RoundingInputs(0.0, 0.0, 1.0)  # type: ignore[index]


@pytest.mark.parametrize("order", [17, True])
def test_rejects_any_quadrature_order_other_than_integer_21(order: object) -> None:
    model, recognition = _inputs()

    with pytest.raises(ValueError, match="quadrature_order"):
        evaluate_information_elbo(
            model,
            recognition,
            quadrature_order=order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("probabilities", [(0.0, 1.0), (-0.2, 1.2)])
def test_rejects_nonpositive_path_weight(probabilities: tuple[float, float]) -> None:
    model, recognition = _inputs()
    recognition.factors._model_source_probabilities[1].copy_(  # type: ignore[attr-defined]
        torch.tensor(probabilities, dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="positive"):
        evaluate_information_elbo(model, recognition, quadrature_order=21)


def test_rejects_nonfinite_emission_output() -> None:
    model, recognition = _inputs()
    model.factors.emissions[0]._bias[0] = float("inf")  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="emission"):
        evaluate_information_elbo(model, recognition, quadrature_order=21)
