from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import vfe4.objective as objective
import vfe4.objective.h2_information as h2_information
import verification.numpy_oracles.h1_elbo as independent_oracle
from vfe4.generative import H1GenerativeModel
from vfe4.numerics.information import InformationGaussian
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


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vfe4"
    / "validation"
    / "fixtures"
    / "h1_v1.json"
)


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


def _independent_log_normalizer(component: object) -> float:
    mean = component.mean  # type: ignore[attr-defined]
    covariance = component.covariance  # type: ignore[attr-defined]
    precision = np.linalg.solve(covariance, np.eye(6, dtype=np.float64))
    sign, covariance_logdet = np.linalg.slogdet(covariance)
    assert sign == 1.0
    return 0.5 * (
        float(mean @ precision @ mean)
        + float(covariance_logdet)
        + 6.0 * math.log(2.0 * math.pi)
    )


def test_component_and_local_fields_match_independent_numpy_fixture_values() -> None:
    evaluation = _evaluate()
    fixture = independent_oracle._load_complete_fixture(FIXTURE_PATH)
    expected_components: list[tuple[float, ...]] = []
    for path in independent_oracle._PATHS:
        q_component = independent_oracle._assemble_recognition_component(
            fixture.recognition, path
        )
        p_component = independent_oracle._assemble_generative_component(
            fixture.generative, path
        )
        q_weight = independent_oracle._recognition_source_weight(
            fixture.recognition, path
        )
        p_weight = independent_oracle._generative_source_weight(
            fixture.generative, path
        )
        emissions = tuple(
            independent_oracle._expected_log_emission_component(
                q_component,
                fixture.generative.emissions[time - 1],
                time=time,
                selected_index=time - 1,
                order=21,
            ).value
            for time in (1, 2)
        )
        gaussian_log_ratio = -independent_oracle._gaussian_kl(
            q_component, p_component
        )
        source_log_ratio = math.log(p_weight) - math.log(q_weight)
        expected_components.append(
            (
                _independent_log_normalizer(q_component),
                _independent_log_normalizer(p_component),
                gaussian_log_ratio,
                source_log_ratio,
                emissions[0],
                emissions[1],
                math.fsum((gaussian_log_ratio, source_log_ratio, *emissions)),
            )
        )

    assert any(abs(values[0] - values[1]) > 1.0e-3 for values in expected_components)
    assert any(abs(values[4] - values[5]) > 1.0e-3 for values in expected_components)
    for component, expected in zip(evaluation.components, expected_components):
        actual = (
            component.q_log_normalizer,
            component.p_log_normalizer,
            component.gaussian_log_ratio,
            component.source_log_ratio,
            component.expected_log_emission[0],
            component.expected_log_emission[1],
            component.complete_value,
        )
        assert actual == pytest.approx(expected, abs=2.0e-13)

    independent_local = independent_oracle._local_order(fixture, 21)
    local = evaluation.local_terms
    actual_local = (
        *local.expected_log_emission,
        local.initial_model_kl,
        local.initial_state_kl,
        *local.model_source_kl,
        *local.model_transition_kl,
        *local.state_source_kl,
        *local.state_transition_kl,
        local.joint_recognition_entropy,
        local.complete_elbo,
    )
    expected_local = (
        *(item.value for item in independent_local.expected_log_emission),
        independent_local.initial_model_kl.value,
        independent_local.initial_state_kl.value,
        *(item.value for item in independent_local.model_source_kl),
        *(item.value for item in independent_local.model_transition_kl),
        *(item.value for item in independent_local.state_source_kl),
        *(item.value for item in independent_local.state_transition_kl),
        independent_local.joint_recognition_entropy.value,
        independent_local.complete_elbo.value,
    )
    assert abs(expected_local[0] - expected_local[1]) > 1.0e-3
    assert actual_local == pytest.approx(expected_local, abs=2.0e-13)


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
    assert requests == [
        (0, 1), (1, 3), (0, 3, 2), (1, 5), (0, 5, 4), (2, 3), (4, 5),
        (0, 1), (1, 3), (0, 3, 2), (1, 5), (2, 5, 4), (2, 3), (4, 5),
        (0, 1), (1, 3), (0, 3, 2), (3, 5), (0, 5, 4), (2, 3), (4, 5),
        (0, 1), (1, 3), (0, 3, 2), (3, 5), (2, 5, 4), (2, 3), (4, 5),
    ]


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


def test_conditional_kl_retains_nonzero_absolute_summands_when_q_equals_p() -> None:
    reduction = h2_information._conditional_gaussian_kl(0.25, 0.25, 0.0)

    assert reduction.value == 0.0
    assert reduction.absolute_sum == 1.0


def test_local_conditional_rounding_uses_pre_reduction_absolute_summands() -> None:
    evaluation = _evaluate()
    conditional_fields = {
        "local.initial_model_kl": evaluation.local_terms.initial_model_kl,
        "local.initial_state_kl": evaluation.local_terms.initial_state_kl,
        "local.model_transition_kl[0]": evaluation.local_terms.model_transition_kl[0],
        "local.model_transition_kl[1]": evaluation.local_terms.model_transition_kl[1],
        "local.state_transition_kl[0]": evaluation.local_terms.state_transition_kl[0],
        "local.state_transition_kl[1]": evaluation.local_terms.state_transition_kl[1],
    }

    for name, value in conditional_fields.items():
        assert (
            evaluation.rounding_inputs[name].absolute_summand_accumulation_inf
            > abs(value)
        )


def test_negative_gaussian_and_joint_differential_entropy_records_are_valid() -> None:
    evaluation = _evaluate()
    concentrated = InformationGaussian.from_information(
        torch.zeros(6, dtype=torch.float64),
        1.0e4 * torch.eye(6, dtype=torch.float64),
    )
    negative_entropy = float(concentrated.entropy().item())
    assert negative_entropy < 0.0
    components = tuple(
        dataclasses.replace(component, q_entropy=negative_entropy)
        for component in evaluation.components
    )
    joint_entropy = math.fsum((evaluation.source_entropy, negative_entropy))

    changed = dataclasses.replace(
        evaluation,
        components=components,
        weighted_component_entropy=negative_entropy,
        joint_recognition_entropy=joint_entropy,
    )

    assert changed.weighted_component_entropy < 0.0
    assert changed.joint_recognition_entropy < 0.0


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
