from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import verification.numpy_oracles.h1_elbo as oracle_module
from verification.numpy_oracles.h1_elbo import (
    IndependentTermAllowances,
    IndependentNumericalAllowance,
    h1_evidence_and_posterior_kl,
    h1_evidence_enumeration_pair,
    h1_local_diagnostics,
)
from vfe4.generative import H1GenerativeModel
from vfe4.objective import evaluate_local_elbo, evaluate_monolithic_elbo
from vfe4.recognition import H1RecognitionLaw
from vfe4.validation import load_h1_fixture


FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "vfe4"
    / "validation"
    / "fixtures"
    / "h1_v1.json"
)


def _comparison_roundoff(*values: float) -> float:
    return 32.0 * math.ulp(1.0) * math.fsum(abs(value) for value in values)


def _paired_limit(
    independent: IndependentNumericalAllowance,
    production: object,
    *values: float,
) -> float:
    return independent.total + production.total + _comparison_roundoff(*values)


def _production_results() -> tuple[object, object, object]:
    fixture = load_h1_fixture(FIXTURE_PATH)
    model = H1GenerativeModel.from_fixture(fixture)
    recognition = H1RecognitionLaw.from_fixture(fixture)
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
    return fixture, monolithic, local


def test_oracle_module_imports_only_standard_library_and_numpy() -> None:
    source = inspect.getsource(oracle_module)
    tree = ast.parse(source)
    roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert roots <= {
        "__future__",
        "dataclasses",
        "functools",
        "itertools",
        "json",
        "math",
        "pathlib",
        "typing",
        "numpy",
    }
    assert "vfe4" not in roots


def test_independent_identity_agrees_with_both_production_elbos() -> None:
    fixture, monolithic, local = _production_results()
    identity = h1_evidence_and_posterior_kl(
        FIXTURE_PATH,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )
    monolithic_limit = (
        identity.identity_allowance.total
        + monolithic.numerical_allowance.total
        + _comparison_roundoff(identity.elbo_from_identity, monolithic.value)
    )
    local_limit = (
        identity.identity_allowance.total
        + local.allowances.complete_elbo.total
        + _comparison_roundoff(identity.elbo_from_identity, local.complete_elbo)
    )

    assert abs(identity.elbo_from_identity - monolithic.value) <= monolithic_limit
    assert abs(identity.elbo_from_identity - local.complete_elbo) <= local_limit
    assert identity.posterior_kl >= -identity.posterior_kl_allowance.total
    assert identity.elbo_from_identity <= (
        identity.evidence.log_probability
        + identity.identity_allowance.total
        + identity.evidence.log_probability_allowance.total
    )
    assert identity.quadrature_order == 21
    assert identity.convergence_check_order == 17


def test_independent_local_diagnostics_match_every_production_term() -> None:
    fixture, _, production = _production_results()
    independent = h1_local_diagnostics(
        FIXTURE_PATH,
        quadrature_order=fixture.quadrature_order,
        convergence_check_order=fixture.convergence_check_order,
    )

    paired_fields = (
        "expected_log_emission",
        "model_source_kl",
        "model_transition_kl",
        "state_source_kl",
        "state_transition_kl",
    )
    for field in paired_fields:
        independent_values = getattr(independent, field)
        production_values = getattr(production, field)
        independent_allowances = getattr(independent.allowances, field)
        production_allowances = getattr(production.allowances, field)
        for index in range(2):
            limit = _paired_limit(
                independent_allowances[index],
                production_allowances[index],
                independent_values[index],
                production_values[index],
            )
            assert abs(independent_values[index] - production_values[index]) <= limit

    for field in (
        "initial_model_kl",
        "initial_state_kl",
        "joint_recognition_entropy",
        "complete_elbo",
    ):
        independent_value = getattr(independent, field)
        production_value = getattr(production, field)
        limit = _paired_limit(
            getattr(independent.allowances, field),
            getattr(production.allowances, field),
            independent_value,
            production_value,
        )
        assert abs(independent_value - production_value) <= limit


def test_direct_source_loop_and_prebuilt_component_enumerations_agree() -> None:
    direct, table = h1_evidence_enumeration_pair(
        FIXTURE_PATH,
        (1, 2),
        quadrature_order=21,
    )

    assert direct == table
    assert 0.0 < direct <= 1.0


def test_all_reported_convergence_estimates_are_finite_and_preregistered() -> None:
    identity = h1_evidence_and_posterior_kl(
        FIXTURE_PATH,
        quadrature_order=21,
        convergence_check_order=17,
    )
    local = h1_local_diagnostics(
        FIXTURE_PATH,
        quadrature_order=21,
        convergence_check_order=17,
    )
    allowances = [
        identity.evidence.probability_allowance,
        identity.evidence.log_probability_allowance,
        identity.posterior_kl_allowance,
        identity.identity_allowance,
        local.allowances.initial_model_kl,
        local.allowances.initial_state_kl,
        local.allowances.joint_recognition_entropy,
        local.allowances.complete_elbo,
        *local.allowances.expected_log_emission,
        *local.allowances.model_source_kl,
        *local.allowances.model_transition_kl,
        *local.allowances.state_source_kl,
        *local.allowances.state_transition_kl,
    ]

    assert all(math.isfinite(item.convergence_estimate) for item in allowances)
    assert max(item.convergence_estimate for item in allowances) <= 1.0e-9
    assert all(math.isfinite(item.total) for item in allowances)


def test_identity_uses_independently_calculated_probability_and_log_allowances() -> None:
    identity = h1_evidence_and_posterior_kl(
        FIXTURE_PATH,
        quadrature_order=21,
        convergence_check_order=17,
    )
    evidence = identity.evidence

    assert evidence.probability_allowance is not evidence.log_probability_allowance
    assert evidence.probability_allowance.total != evidence.log_probability_allowance.total
    assert np.isclose(math.log(evidence.probability), evidence.log_probability, rtol=0.0, atol=0.0)
    propagated_floor = (
        evidence.probability_allowance.rounding_allowance / evidence.probability
        + 32.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(evidence.log_probability))
    )
    assert evidence.log_probability_allowance.rounding_allowance >= propagated_floor


def _weighted_moment(values: np.ndarray, weights: np.ndarray, power: int) -> float:
    return math.fsum(float(weight * value**power) for value, weight in zip(values, weights))


def _recover_conditional(
    component: object, target: int, parents: tuple[int, ...]
) -> tuple[np.ndarray, float, float]:
    parent_indices = np.asarray(parents, dtype=np.int64)
    parent_covariance = component.covariance[np.ix_(parent_indices, parent_indices)]
    target_parent_covariance = component.covariance[target, parent_indices]
    slopes = np.linalg.solve(parent_covariance, target_parent_covariance)
    offset = float(component.mean[target] - slopes @ component.mean[parent_indices])
    variance = float(
        component.covariance[target, target]
        - target_parent_covariance @ np.linalg.solve(
            parent_covariance, target_parent_covariance
        )
    )
    return slopes, offset, variance


def test_physicists_hermite_scaling_calibrates_standard_normal_moments() -> None:
    for order in (17, 21):
        points, weights = oracle_module._standard_normal_grid(order, 4)
        assert abs(math.fsum(float(item) for item in weights) - 1.0) <= 64.0 * np.finfo(np.float64).eps
        for dimension in range(4):
            values = points[:, dimension]
            assert abs(_weighted_moment(values, weights, 1)) <= 2.0e-15
            assert abs(_weighted_moment(values, weights, 2) - 1.0) <= 2.0e-14
            assert abs(_weighted_moment(values, weights, 4) - 3.0) <= 8.0e-14


def test_quadrature_transform_recovers_every_component_emission_marginal() -> None:
    complete = oracle_module._load_complete_fixture(FIXTURE_PATH)
    standard, weights = oracle_module._standard_normal_grid(17, 4)
    components = [
        *(oracle_module._assemble_generative_component(complete.generative, path) for path in oracle_module._PATHS),
        *(oracle_module._assemble_recognition_component(complete.recognition, path) for path in oracle_module._PATHS),
    ]
    indices = np.asarray((2, 3, 4, 5), dtype=np.int64)
    for component in components:
        expected_mean = component.mean[indices]
        expected_covariance = component.covariance[np.ix_(indices, indices)]
        chol = np.linalg.cholesky(expected_covariance)
        transformed = expected_mean + standard @ chol.T
        recovered_mean = np.asarray(
            [
                math.fsum(float(weight * value) for weight, value in zip(weights, transformed[:, index]))
                for index in range(4)
            ]
        )
        centered = transformed - recovered_mean
        recovered_covariance = np.asarray(
            [
                [
                    math.fsum(
                        float(weight * first * second)
                        for weight, first, second in zip(
                            weights, centered[:, row], centered[:, column]
                        )
                    )
                    for column in range(4)
                ]
                for row in range(4)
            ]
        )
        assert np.allclose(recovered_mean, expected_mean, rtol=0.0, atol=2.0e-14)
        assert np.allclose(recovered_covariance, expected_covariance, rtol=0.0, atol=1.0e-13)


def test_full_component_assemblies_recover_declared_directed_conditionals() -> None:
    complete = oracle_module._load_complete_fixture(FIXTURE_PATH)
    generative = complete.generative
    recognition = complete.recognition
    for path in oracle_module._PATHS:
        a_second, b_second = path
        p_component = oracle_module._assemble_generative_component(generative, path)
        q_component = oracle_module._assemble_recognition_component(recognition, path)
        assert np.array_equal(p_component.mean[:2], generative.initial.mean)
        assert np.array_equal(p_component.covariance[:2, :2], generative.initial.covariance)
        assert np.array_equal(q_component.mean[:2], recognition.initial.mean)
        assert np.array_equal(q_component.covariance[:2, :2], recognition.initial.covariance)
        for time in (1, 2):
            a = 0 if time == 1 else a_second
            b = 0 if time == 1 else b_second
            m_index, z_index = 2 * time + 1, 2 * time
            p_m_slopes, p_m_offset, p_m_variance = _recover_conditional(
                p_component, m_index, (2 * b + 1,)
            )
            assert np.allclose(
                p_m_slopes,
                (generative.frames[time] / generative.frames[b],),
                rtol=0.0,
                atol=2.0e-14,
            )
            assert math.isclose(p_m_offset, generative.model_offsets[time - 1], rel_tol=0.0, abs_tol=2.0e-14)
            assert math.isclose(p_m_variance, generative.model_variances[time - 1], rel_tol=0.0, abs_tol=2.0e-14)
            p_z_slopes, p_z_offset, p_z_variance = _recover_conditional(
                p_component, z_index, (2 * a, m_index)
            )
            assert np.allclose(
                p_z_slopes,
                (
                    generative.frames[time] / generative.frames[a],
                    generative.state_model_slopes[time - 1],
                ),
                rtol=0.0,
                atol=3.0e-14,
            )
            assert math.isclose(p_z_offset, generative.state_offsets[time - 1], rel_tol=0.0, abs_tol=3.0e-14)
            assert math.isclose(p_z_variance, generative.state_variances[time - 1], rel_tol=0.0, abs_tol=3.0e-14)

            slot = 0 if time == 1 else a + 2 * b
            q_m_slopes, q_m_offset, q_m_variance = _recover_conditional(
                q_component, m_index, (2 * b + 1,)
            )
            assert np.allclose(q_m_slopes, (recognition.model_kernels[time - 1].slopes[b],), rtol=0.0, atol=2.0e-14)
            assert math.isclose(q_m_offset, recognition.model_kernels[time - 1].offsets[b], rel_tol=0.0, abs_tol=2.0e-14)
            assert math.isclose(q_m_variance, recognition.model_kernels[time - 1].variances[b], rel_tol=0.0, abs_tol=2.0e-14)
            q_z_slopes, q_z_offset, q_z_variance = _recover_conditional(
                q_component, z_index, (2 * a, m_index)
            )
            assert np.allclose(
                q_z_slopes,
                (
                    recognition.state_kernels[time - 1].z_slopes[slot],
                    recognition.state_kernels[time - 1].m_slopes[slot],
                ),
                rtol=0.0,
                atol=3.0e-14,
            )
            assert math.isclose(q_z_offset, recognition.state_kernels[time - 1].offsets[slot], rel_tol=0.0, abs_tol=3.0e-14)
            assert math.isclose(q_z_variance, recognition.state_kernels[time - 1].variances[slot], rel_tol=0.0, abs_tol=3.0e-14)


def test_complete_component_identity_does_not_delegate_to_local_diagnostics(
    monkeypatch: object,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("local diagnostics were called")

    monkeypatch.setattr(oracle_module, "h1_local_diagnostics", forbidden)
    identity = h1_evidence_and_posterior_kl(
        FIXTURE_PATH, quadrature_order=21, convergence_check_order=17
    )
    source = inspect.getsource(oracle_module._identity_order)

    assert math.isfinite(identity.posterior_kl)
    assert "h1_local_diagnostics" not in source
    assert "_gaussian_kl" in source


def test_stable_softmax_and_log_softmax_ignore_common_large_logit_shift() -> None:
    logits = np.asarray(((1001.0, 999.0, 1003.0), (-1000.0, -999.0, -998.0)))
    shifted = logits + 1000.0
    assert np.allclose(
        oracle_module._softmax(logits),
        oracle_module._softmax(shifted),
        rtol=0.0,
        atol=2.0e-14,
    )

    complete = oracle_module._load_complete_fixture(FIXTURE_PATH)
    component = oracle_module._assemble_recognition_component(
        complete.recognition, oracle_module._PATHS[0]
    )
    emission = complete.generative.emissions[0]
    shifted_emission = oracle_module._Emission(
        emission.w_z, emission.w_m, emission.bias + 1000.0
    )
    baseline = oracle_module._expected_log_emission_component(
        component, emission, time=1, selected_index=0, order=21
    )
    shifted_result = oracle_module._expected_log_emission_component(
        component, shifted_emission, time=1, selected_index=0, order=21
    )
    assert math.isclose(baseline.value, shifted_result.value, rel_tol=0.0, abs_tol=4.0e-14)


@pytest.mark.parametrize(
    ("convergence", "rounding"),
    [
        (-1.0, 0.0),
        (math.nan, 0.0),
        (0.0, math.inf),
        (2.0e-9, 0.0),
        (
            float.fromhex("0x1.fffffffffffffp+1023"),
            float.fromhex("0x1.fffffffffffffp+1023"),
        ),
    ],
)
def test_independent_allowances_fail_closed(convergence: float, rounding: float) -> None:
    with pytest.raises(ValueError):
        IndependentNumericalAllowance(convergence, rounding)


def test_term_allowances_reject_malformed_pairs() -> None:
    zero = IndependentNumericalAllowance(0.0, 0.0)
    with pytest.raises(ValueError, match="pair"):
        IndependentTermAllowances(
            expected_log_emission=(zero,),
            initial_model_kl=zero,
            initial_state_kl=zero,
            model_source_kl=(zero, zero),
            model_transition_kl=(zero, zero),
            state_source_kl=(zero, zero),
            state_transition_kl=(zero, zero),
            joint_recognition_entropy=zero,
            complete_elbo=zero,
        )


def test_state_source_diagnostic_is_expected_conditional_categorical_kl() -> None:
    complete = oracle_module._load_complete_fixture(FIXTURE_PATH)
    result = h1_local_diagnostics(
        FIXTURE_PATH, quadrature_order=21, convergence_check_order=17
    )
    q_b = complete.recognition.model_probabilities[1]
    q_a = complete.recognition.state_probabilities[1]
    p_a = complete.generative.state_priors[1]
    expected = math.fsum(
        float(q_b[b] * q_a[b, a] * (math.log(q_a[b, a]) - math.log(p_a[a])))
        for b in range(2)
        for a in range(2)
    )
    recomposed_without_entropy = math.fsum(
        (
            *result.expected_log_emission,
            -result.initial_model_kl,
            -result.initial_state_kl,
            *(-value for value in result.model_source_kl),
            *(-value for value in result.model_transition_kl),
            *(-value for value in result.state_source_kl),
            *(-value for value in result.state_transition_kl),
        )
    )

    assert math.isclose(result.state_source_kl[1], expected, rel_tol=0.0, abs_tol=2.0e-15)
    assert math.isclose(result.complete_elbo, recomposed_without_entropy, rel_tol=0.0, abs_tol=2.0e-15)
    assert abs(result.complete_elbo - (recomposed_without_entropy + result.joint_recognition_entropy)) > 1.0


def test_oracle_numerics_use_raw_spd_factorizations_without_fallbacks() -> None:
    source = inspect.getsource(oracle_module)

    assert "np.linalg.cholesky" in source
    assert "np.linalg.slogdet" in source
    assert "np.linalg.solve" in source
    assert "np.linalg.pinv" not in source
    assert "jitter" not in source
    assert "np.linalg.eig" not in source
