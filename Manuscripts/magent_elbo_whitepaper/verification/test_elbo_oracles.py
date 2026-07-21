from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from manuscripts.magent_elbo_whitepaper.verification.elbo_oracles import (
    block_mean_field_covariance,
    determinant_gap,
    directed_gaussian_information,
    gaussian_cavi_block,
    gaussian_information_log_normalizer,
    gaussian_kl,
    linear_gaussian_elbo,
    linear_gaussian_posterior,
    nested_identity_terms,
    source_row_envelope,
    transported_covariance,
    transported_precision,
)


BLOCKS = ((0, 1), (2,), (3, 4))


def test_gaussian_block_mean_field_gap_matches_reverse_kl() -> None:
    rng = np.random.default_rng(20260718)
    factor = rng.normal(size=(5, 5))
    precision = factor.T @ factor + np.eye(5)
    information = rng.normal(size=5)
    mean = np.linalg.solve(precision, information)
    covariance_mf = block_mean_field_covariance(precision, BLOCKS)
    covariance_exact = np.linalg.inv(precision)

    observed = gaussian_kl(mean, covariance_mf, mean, covariance_exact)
    expected = determinant_gap(precision, BLOCKS)

    assert abs(observed - expected) < 1e-10
    assert expected > 0.0


def test_partition_specific_zero_coupling_controls() -> None:
    precision = np.array(
        [
            [2.0, 0.4, 0.0, 0.0, 0.0],
            [0.4, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 7.0, 0.6],
            [0.0, 0.0, 0.0, 0.6, 11.0],
        ]
    )
    mean = np.array([0.5, -0.25, 0.75, -1.0, 1.25])
    covariance_agent_block = block_mean_field_covariance(precision, BLOCKS)
    covariance_exact = np.linalg.inv(precision)
    fully_factorized_blocks = ((0,), (1,), (2,), (3,), (4,))

    assert gaussian_kl(
        mean,
        covariance_agent_block,
        mean,
        covariance_exact,
    ) < 1e-12
    assert determinant_gap(precision, BLOCKS) < 1e-12
    assert determinant_gap(precision, fully_factorized_blocks) > 0.0

    diagonal_precision = np.diag([2.0, 3.0, 5.0, 7.0, 11.0])
    assert determinant_gap(diagonal_precision, fully_factorized_blocks) < 1e-12


def test_nested_state_configuration_identity_at_nonunit_common_scale() -> None:
    p0 = np.array([0.3, 0.7])
    p_y_o_given_x = np.array([[0.12, 0.08], [0.07, 0.28]])
    r = np.array([0.4, 0.6])
    q_y_given_x = np.array([[0.25, 0.75], [0.6, 0.4]])

    left, right, residual = nested_identity_terms(
        p0,
        p_y_o_given_x,
        r,
        q_y_given_x,
        temperature=3.7,
    )
    unit_left, unit_right, unit_residual = nested_identity_terms(
        p0,
        p_y_o_given_x,
        r,
        q_y_given_x,
    )

    assert np.isfinite(left)
    assert np.isfinite(right)
    assert abs(residual) < 1e-12
    assert abs(unit_residual) < 1e-12
    assert abs(left / 3.7 - unit_left) < 1e-12
    assert abs(right / 3.7 - unit_right) < 1e-12


def test_source_row_objective_equals_log_sum_exp_envelope() -> None:
    """The fixture detects a missing nonunit-temperature response coefficient."""
    prior = np.array([0.2, 0.3, 0.5])
    energies = np.array([1.4, 0.2, 0.9])
    temperature = 2.3

    beta, objective, envelope = source_row_envelope(
        prior,
        energies,
        temperature=temperature,
    )
    suppressed_response = float(beta @ energies)
    categorical_kl = float(np.sum(beta * np.log(beta / prior)))

    assert np.isclose(beta.sum(), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(beta > 0.0)
    assert abs(objective - envelope) < 1e-12
    assert abs(
        envelope - suppressed_response - temperature * categorical_kl
    ) < 1e-12
    assert envelope >= suppressed_response


def test_source_row_equal_active_energies_saturate_response_inequality() -> None:
    """An inactive component imposes no energy condition for scalar equality."""
    prior = np.array([0.0, 0.3, 0.7])
    energies = np.array([8.0, 0.7, 0.7])

    beta, objective, envelope = source_row_envelope(
        prior,
        energies,
        temperature=2.3,
    )
    suppressed_response = float(beta @ energies)

    assert np.allclose(beta, prior, rtol=0.0, atol=1e-12)
    assert abs(objective - envelope) < 1e-12
    assert abs(envelope - suppressed_response) < 1e-12


def test_source_row_envelope_support_and_nonunit_temperature() -> None:
    prior = np.array([0.0, 0.3, 0.7])
    energies = np.array([8.0, 1.4, 0.2])

    beta, objective, envelope = source_row_envelope(
        prior,
        energies,
        temperature=2.3,
    )

    assert beta[0] == 0.0
    assert np.isclose(beta.sum(), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(beta[1:] > 0.0)
    assert abs(objective - envelope) < 1e-12


def test_covariance_transport_is_gauge_equivariant() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 1.5]])
    omega = np.array([[1.1, 0.2], [-0.1, 0.9]])
    sender_frame = np.array([[1.2, 0.1], [0.0, 0.8]])
    receiver_frame = np.array([[0.9, -0.2], [0.1, 1.1]])
    transformed_omega = (
        receiver_frame @ omega @ np.linalg.inv(sender_frame)
    )
    transformed_sender_covariance = (
        sender_frame @ covariance @ sender_frame.T
    )

    observed = transported_covariance(
        transformed_omega,
        transformed_sender_covariance,
    )
    expected = (
        receiver_frame
        @ transported_covariance(omega, covariance)
        @ receiver_frame.T
    )

    assert np.allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_symbolic_two_scalar_block_gap_under_spd_domain() -> None:
    """The Gaussian domain also requires a * b - c**2 > 0."""
    a, b, c = sp.symbols("a b c", positive=True)
    determinant = a * b - c**2
    gap = sp.Rational(1, 2) * (
        sp.log(a) + sp.log(b) - sp.log(determinant)
    )

    assert sp.simplify(
        sp.exp(2 * gap) - a * b / determinant
    ) == 0
    assert determinant.subs({a: 2, b: 3, c: 1}) > 0


def test_symbolic_peer_kl_mixed_third_derivative_is_not_identically_zero() -> None:
    """The identity-transport Bernoulli path assumes 0 < receiver,sender < 1."""
    receiver, sender = sp.symbols("receiver sender", positive=True)
    peer_kl = (
        receiver * sp.log(receiver / sender)
        + (1 - receiver)
        * sp.log((1 - receiver) / (1 - sender))
    )
    mixed = sp.simplify(sp.diff(peer_kl, receiver, 1, sender, 2))
    expected = 1 / sender**2 - 1 / (1 - sender) ** 2

    assert sp.simplify(mixed - expected) == 0
    assert mixed != 0
    assert mixed.subs(sender, sp.Rational(1, 3)) != 0


def test_directed_gaussian_joint_is_normalized_in_information_form() -> None:
    coefficients = np.array(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.2, 0.3, 0.0]]
    )
    offset = np.array([0.5, -0.25, 0.75])
    noise_covariance = np.diag([1.2, 0.8, 1.5])
    information, precision, log_constant = directed_gaussian_information(
        coefficients,
        offset,
        noise_covariance,
    )
    residual_map = np.eye(3) - coefficients
    inverse_residual = np.linalg.solve(residual_map, np.eye(3))
    structural_mean = np.linalg.solve(residual_map, offset)
    structural_covariance = (
        inverse_residual @ noise_covariance @ inverse_residual.T
    )

    assert abs(
        log_constant
        + gaussian_information_log_normalizer(information, precision)
    ) < 1e-12
    assert np.allclose(
        np.linalg.solve(precision, information),
        structural_mean,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        np.linalg.inv(precision),
        structural_covariance,
        rtol=0.0,
        atol=1e-12,
    )


def test_directed_gaussian_requires_exact_strict_lower_structure() -> None:
    coefficients = np.array([[1e-13, 0.0], [0.4, 0.0]])

    with pytest.raises(ValueError, match="strictly lower triangular"):
        directed_gaussian_information(
            coefficients,
            np.zeros(2),
            np.eye(2),
        )


def test_state_evidence_identity_and_analytic_posterior_moments() -> None:
    prior_mean = np.array([0.2, -0.4])
    prior_covariance = np.array([[1.4, 0.2], [0.2, 0.9]])
    observation_matrix = np.array([[1.0, -0.3]])
    noise_covariance = np.array([[0.6]])
    observation = np.array([0.75])
    q_mean = np.array([-0.1, 0.3])
    q_covariance = np.array([[0.8, 0.1], [0.1, 0.7]])
    log_evidence, posterior_mean, posterior_covariance = (
        linear_gaussian_posterior(
            prior_mean,
            prior_covariance,
            observation_matrix,
            noise_covariance,
            observation,
        )
    )
    elbo = linear_gaussian_elbo(
        q_mean,
        q_covariance,
        prior_mean,
        prior_covariance,
        observation_matrix,
        noise_covariance,
        observation,
    )
    posterior_gap = gaussian_kl(
        q_mean,
        q_covariance,
        posterior_mean,
        posterior_covariance,
    )
    predictive_covariance = (
        observation_matrix @ prior_covariance @ observation_matrix.T
        + noise_covariance
    )
    gain = (
        prior_covariance
        @ observation_matrix.T
        @ np.linalg.inv(predictive_covariance)
    )
    expected_posterior_mean = (
        prior_mean + gain @ (observation - observation_matrix @ prior_mean)
    )
    expected_posterior_covariance = (
        prior_covariance - gain @ observation_matrix @ prior_covariance
    )

    assert abs(log_evidence - elbo - posterior_gap) < 1e-10
    assert np.allclose(
        posterior_mean,
        expected_posterior_mean,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        posterior_covariance,
        expected_posterior_covariance,
        rtol=0.0,
        atol=1e-12,
    )


def test_gaussian_cavi_uses_complete_markov_blanket_and_converges() -> None:
    precision = np.array(
        [[3.0, -0.5, 0.2], [-0.5, 2.5, -0.4], [0.2, -0.4, 2.0]]
    )
    information = np.array([0.7, -0.2, 0.5])
    mean = np.zeros(3)
    for _ in range(100):
        for block in ((0,), (1,), (2,)):
            block_mean, block_covariance = gaussian_cavi_block(
                precision,
                information,
                mean,
                block,
            )
            mean[np.asarray(block)] = block_mean
            assert np.allclose(
                block_covariance,
                np.linalg.inv(precision[np.ix_(block, block)]),
                rtol=0.0,
                atol=1e-12,
            )

    assert np.allclose(
        mean,
        np.linalg.solve(precision, information),
        rtol=0.0,
        atol=1e-10,
    )


def test_gaussian_cavi_rejects_indices_outside_the_precision() -> None:
    precision = np.eye(3)
    information = np.zeros(3)
    mean = np.zeros(3)

    with pytest.raises(ValueError, match="block index"):
        gaussian_cavi_block(precision, information, mean, (-1,))
    with pytest.raises(ValueError, match="block index"):
        gaussian_cavi_block(precision, information, mean, (3,))


def test_precision_uses_dual_inverse_congruence() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 1.5]])
    precision = np.linalg.inv(covariance)
    omega = np.array([[1.1, 0.2], [-0.1, 0.9]])

    observed = transported_precision(omega, precision)
    expected = np.linalg.inv(transported_covariance(omega, covariance))

    assert np.allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_precision_transport_rejects_singular_frame() -> None:
    singular_frame = np.array([[1.0, 2.0], [0.5, 1.0]])

    with pytest.raises(ValueError, match="transport must be invertible"):
        transported_precision(singular_frame, np.eye(2))


def test_complete_linear_gaussian_elbo_is_gauge_invariant() -> None:
    prior_mean = np.array([0.2, -0.4])
    prior_covariance = np.array([[1.4, 0.2], [0.2, 0.9]])
    q_mean = np.array([-0.1, 0.3])
    q_covariance = np.array([[0.8, 0.1], [0.1, 0.7]])
    observation_matrix = np.array([[1.0, -0.3]])
    noise_covariance = np.array([[0.6]])
    observation = np.array([0.75])
    frame = np.array([[1.2, 0.1], [-0.2, 0.9]])
    transformed_observation_matrix = observation_matrix @ np.linalg.inv(frame)

    original_elbo = linear_gaussian_elbo(
        q_mean,
        q_covariance,
        prior_mean,
        prior_covariance,
        observation_matrix,
        noise_covariance,
        observation,
    )
    transformed_elbo = linear_gaussian_elbo(
        frame @ q_mean,
        frame @ q_covariance @ frame.T,
        frame @ prior_mean,
        frame @ prior_covariance @ frame.T,
        transformed_observation_matrix,
        noise_covariance,
        observation,
    )
    original_evidence = linear_gaussian_posterior(
        prior_mean,
        prior_covariance,
        observation_matrix,
        noise_covariance,
        observation,
    )[0]
    transformed_evidence = linear_gaussian_posterior(
        frame @ prior_mean,
        frame @ prior_covariance @ frame.T,
        transformed_observation_matrix,
        noise_covariance,
        observation,
    )[0]

    assert abs(original_elbo - transformed_elbo) < 1e-10
    assert abs(original_evidence - transformed_evidence) < 1e-10
