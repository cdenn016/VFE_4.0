from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_spd(matrix: np.ndarray, name: str) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return value


def _validated_blocks(
    blocks: Sequence[Sequence[int]],
    dimension: int,
) -> Sequence[np.ndarray]:
    arrays = tuple(np.asarray(block, dtype=np.int64) for block in blocks)
    if not arrays or any(block.ndim != 1 or block.size == 0 for block in arrays):
        raise ValueError("blocks must be nonempty one-dimensional index sets")
    joined = np.concatenate(arrays)
    if np.any(joined < 0) or np.any(joined >= dimension):
        raise ValueError("block index is outside the precision matrix")
    if not np.array_equal(np.sort(joined), np.arange(dimension)):
        raise ValueError("blocks must form a disjoint complete partition")
    return arrays


def gaussian_kl(
    mu_q: np.ndarray,
    cov_q: np.ndarray,
    mu_p: np.ndarray,
    cov_p: np.ndarray,
) -> float:
    covariance_q = _as_spd(cov_q, "cov_q")
    covariance_p = _as_spd(cov_p, "cov_p")
    mean_q = np.asarray(mu_q, dtype=np.float64)
    mean_p = np.asarray(mu_p, dtype=np.float64)
    dimension = covariance_q.shape[0]
    if covariance_p.shape != covariance_q.shape:
        raise ValueError("covariances must have the same shape")
    if mean_q.shape != (dimension,) or mean_p.shape != (dimension,):
        raise ValueError("means must match the covariance dimension")
    sign_q, logdet_q = np.linalg.slogdet(covariance_q)
    sign_p, logdet_p = np.linalg.slogdet(covariance_p)
    if sign_q <= 0.0 or sign_p <= 0.0:
        raise ValueError("covariance determinants must be positive")
    difference = mean_p - mean_q
    trace_term = np.trace(np.linalg.solve(covariance_p, covariance_q))
    quadratic = difference @ np.linalg.solve(covariance_p, difference)
    return float(
        0.5 * (trace_term + quadratic - dimension + logdet_p - logdet_q)
    )


def block_mean_field_covariance(
    precision: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> FloatArray:
    value = _as_spd(precision, "precision")
    partition = _validated_blocks(blocks, value.shape[0])
    covariance = np.zeros_like(value)
    for indices in partition:
        block = value[np.ix_(indices, indices)]
        covariance[np.ix_(indices, indices)] = np.linalg.solve(
            block,
            np.eye(indices.size),
        )
    return covariance


def determinant_gap(
    precision: np.ndarray,
    blocks: Sequence[Sequence[int]],
) -> float:
    value = _as_spd(precision, "precision")
    partition = _validated_blocks(blocks, value.shape[0])
    sign_full, logdet_full = np.linalg.slogdet(value)
    if sign_full <= 0.0:
        raise ValueError("precision determinant must be positive")
    logdet_blocks = 0.0
    for indices in partition:
        sign_block, logdet_block = np.linalg.slogdet(
            value[np.ix_(indices, indices)]
        )
        if sign_block <= 0.0:
            raise ValueError("precision block determinant must be positive")
        logdet_blocks += logdet_block
    return float(0.5 * (logdet_blocks - logdet_full))


def nested_identity_terms(
    p0: np.ndarray,
    p_y_o_given_x: np.ndarray,
    r: np.ndarray,
    q_y_given_x: np.ndarray,
    temperature: float = 1.0,
) -> tuple[float, float, float]:
    prior = np.asarray(p0, dtype=np.float64)
    joint_slice = np.asarray(p_y_o_given_x, dtype=np.float64)
    posterior_x = np.asarray(r, dtype=np.float64)
    conditional_q = np.asarray(q_y_given_x, dtype=np.float64)
    if joint_slice.ndim != 2 or conditional_q.shape != joint_slice.shape:
        raise ValueError("state joint and recognition kernel must have one common shape")
    if prior.shape != (joint_slice.shape[0],) or posterior_x.shape != prior.shape:
        raise ValueError("configuration vectors must match the number of rows")
    if np.any(prior <= 0.0) or np.any(posterior_x <= 0.0):
        raise ValueError("configuration probabilities must be strictly positive")
    if np.any(joint_slice <= 0.0) or np.any(conditional_q <= 0.0):
        raise ValueError("state probabilities must be strictly positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not np.isclose(prior.sum(), 1.0) or not np.isclose(posterior_x.sum(), 1.0):
        raise ValueError("configuration probabilities must be normalized")
    if not np.allclose(conditional_q.sum(axis=1), 1.0):
        raise ValueError("recognition rows must be normalized")

    state_free_energy = np.sum(
        conditional_q * (np.log(conditional_q) - np.log(joint_slice)),
        axis=1,
    )
    configuration_kl = np.sum(posterior_x * np.log(posterior_x / prior))
    nested_free_energy_nats = float(
        configuration_kl + posterior_x @ state_free_energy
    )

    evidence = float(prior @ joint_slice.sum(axis=1))
    exact_posterior = prior[:, None] * joint_slice / evidence
    joint_q = posterior_x[:, None] * conditional_q
    joint_kl = float(np.sum(joint_q * np.log(joint_q / exact_posterior)))
    right_hand_side_nats = float(-np.log(evidence) + joint_kl)
    nested_free_energy = temperature * nested_free_energy_nats
    right_hand_side = temperature * right_hand_side_nats
    return (
        nested_free_energy,
        right_hand_side,
        nested_free_energy - right_hand_side,
    )


def source_row_envelope(
    prior: np.ndarray,
    energies: np.ndarray,
    temperature: float = 1.0,
) -> tuple[FloatArray, float, float]:
    prior_row = np.asarray(prior, dtype=np.float64)
    energy_row = np.asarray(energies, dtype=np.float64)
    if prior_row.ndim != 1 or energy_row.shape != prior_row.shape:
        raise ValueError("prior and energies must be equal-length vectors")
    if np.any(prior_row < 0.0) or not np.isclose(prior_row.sum(), 1.0):
        raise ValueError("prior must be a probability vector")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    active = prior_row > 0.0
    log_weights = np.log(prior_row[active]) - energy_row[active] / temperature
    maximum = float(np.max(log_weights))
    log_normalizer = maximum + float(
        np.log(np.exp(log_weights - maximum).sum())
    )
    beta = np.zeros_like(prior_row)
    beta[active] = np.exp(log_weights - log_normalizer)
    objective = float(
        beta @ energy_row
        + temperature
        * np.sum(beta[active] * np.log(beta[active] / prior_row[active]))
    )
    envelope = -temperature * log_normalizer
    return beta, objective, envelope


def transported_covariance(
    omega: np.ndarray,
    covariance: np.ndarray,
) -> FloatArray:
    transport = np.asarray(omega, dtype=np.float64)
    value = _as_spd(covariance, "covariance")
    if transport.ndim != 2 or transport.shape[1] != value.shape[0]:
        raise ValueError("transport domain must match the covariance dimension")
    if transport.shape[0] != transport.shape[1]:
        raise ValueError("transport must be square in the reference model")
    if not np.isfinite(np.linalg.slogdet(transport)[1]):
        raise ValueError("transport must be invertible")
    return transport @ value @ transport.T


def gaussian_information_log_normalizer(
    information: np.ndarray,
    precision: np.ndarray,
) -> float:
    value = _as_spd(precision, "precision")
    vector = np.asarray(information, dtype=np.float64)
    if vector.shape != (value.shape[0],):
        raise ValueError("information vector must match the precision dimension")
    sign, logdet = np.linalg.slogdet(value)
    if sign <= 0.0:
        raise ValueError("precision determinant must be positive")
    dimension = value.shape[0]
    return float(
        0.5 * vector @ np.linalg.solve(value, vector)
        - 0.5 * logdet
        + 0.5 * dimension * np.log(2.0 * np.pi)
    )


def _gaussian_logpdf(
    value: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    cov = _as_spd(covariance, "covariance")
    point = np.asarray(value, dtype=np.float64)
    location = np.asarray(mean, dtype=np.float64)
    if point.shape != location.shape or point.shape != (cov.shape[0],):
        raise ValueError("point and mean must match the covariance dimension")
    difference = point - location
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0.0:
        raise ValueError("covariance determinant must be positive")
    return float(
        -0.5
        * (
            cov.shape[0] * np.log(2.0 * np.pi)
            + logdet
            + difference @ np.linalg.solve(cov, difference)
        )
    )


def directed_gaussian_information(
    coefficients: np.ndarray,
    offset: np.ndarray,
    noise_covariance: np.ndarray,
) -> tuple[FloatArray, FloatArray, float]:
    matrix = np.asarray(coefficients, dtype=np.float64)
    shift = np.asarray(offset, dtype=np.float64)
    noise = _as_spd(noise_covariance, "noise_covariance")
    dimension = noise.shape[0]
    if matrix.shape != noise.shape or shift.shape != (dimension,):
        raise ValueError("directed Gaussian inputs must share one dimension")
    if not np.array_equal(matrix, np.tril(matrix, k=-1)):
        raise ValueError("coefficients must be strictly lower triangular")
    residual_map = np.eye(dimension) - matrix
    precision = residual_map.T @ np.linalg.solve(noise, residual_map)
    information = residual_map.T @ np.linalg.solve(noise, shift)
    sign, logdet_noise = np.linalg.slogdet(noise)
    if sign <= 0.0:
        raise ValueError("noise determinant must be positive")
    log_constant = float(
        -0.5 * shift @ np.linalg.solve(noise, shift)
        - 0.5 * logdet_noise
        - 0.5 * dimension * np.log(2.0 * np.pi)
    )
    return information, precision, log_constant


def linear_gaussian_posterior(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    observation: np.ndarray,
) -> tuple[float, FloatArray, FloatArray]:
    mean = np.asarray(prior_mean, dtype=np.float64)
    prior_cov = _as_spd(prior_covariance, "prior_covariance")
    matrix = np.asarray(observation_matrix, dtype=np.float64)
    noise_cov = _as_spd(noise_covariance, "noise_covariance")
    observed = np.asarray(observation, dtype=np.float64)
    if mean.shape != (prior_cov.shape[0],):
        raise ValueError("prior mean must match prior covariance")
    if matrix.shape != (noise_cov.shape[0], prior_cov.shape[0]):
        raise ValueError("observation matrix has incompatible shape")
    if observed.shape != (noise_cov.shape[0],):
        raise ValueError("observation must match noise covariance")
    predictive_covariance = matrix @ prior_cov @ matrix.T + noise_cov
    log_evidence = _gaussian_logpdf(
        observed,
        matrix @ mean,
        predictive_covariance,
    )
    prior_precision = np.linalg.solve(prior_cov, np.eye(prior_cov.shape[0]))
    posterior_precision = (
        prior_precision
        + matrix.T @ np.linalg.solve(noise_cov, matrix)
    )
    posterior_covariance = np.linalg.solve(
        posterior_precision,
        np.eye(posterior_precision.shape[0]),
    )
    posterior_information = (
        prior_precision @ mean
        + matrix.T @ np.linalg.solve(noise_cov, observed)
    )
    posterior_mean = posterior_covariance @ posterior_information
    return log_evidence, posterior_mean, posterior_covariance


def linear_gaussian_elbo(
    q_mean: np.ndarray,
    q_covariance: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    observation: np.ndarray,
) -> float:
    mean_q = np.asarray(q_mean, dtype=np.float64)
    covariance_q = _as_spd(q_covariance, "q_covariance")
    matrix = np.asarray(observation_matrix, dtype=np.float64)
    noise_cov = _as_spd(noise_covariance, "noise_covariance")
    observed = np.asarray(observation, dtype=np.float64)
    residual = observed - matrix @ mean_q
    sign, logdet_noise = np.linalg.slogdet(noise_cov)
    if sign <= 0.0:
        raise ValueError("noise determinant must be positive")
    expected_log_likelihood = float(
        -0.5
        * (
            observed.size * np.log(2.0 * np.pi)
            + logdet_noise
            + residual @ np.linalg.solve(noise_cov, residual)
            + np.trace(
                np.linalg.solve(
                    noise_cov,
                    matrix @ covariance_q @ matrix.T,
                )
            )
        )
    )
    return expected_log_likelihood - gaussian_kl(
        mean_q,
        covariance_q,
        prior_mean,
        prior_covariance,
    )


def gaussian_cavi_block(
    precision: np.ndarray,
    information: np.ndarray,
    current_mean: np.ndarray,
    block: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    value = _as_spd(precision, "precision")
    vector = np.asarray(information, dtype=np.float64)
    mean = np.asarray(current_mean, dtype=np.float64)
    indices = np.asarray(block, dtype=np.int64)
    if vector.shape != (value.shape[0],) or mean.shape != vector.shape:
        raise ValueError("information and mean must match precision")
    if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("block must contain distinct indices")
    if np.any(indices < 0) or np.any(indices >= value.shape[0]):
        raise ValueError("block index is outside the precision matrix")
    others = np.setdiff1d(np.arange(value.shape[0]), indices)
    block_precision = value[np.ix_(indices, indices)]
    block_covariance = np.linalg.solve(
        block_precision,
        np.eye(indices.size),
    )
    block_information = vector[indices] - value[np.ix_(indices, others)] @ mean[others]
    block_mean = block_covariance @ block_information
    return block_mean, block_covariance


def transported_precision(
    omega: np.ndarray,
    precision: np.ndarray,
) -> FloatArray:
    transport = np.asarray(omega, dtype=np.float64)
    value = _as_spd(precision, "precision")
    if transport.shape != value.shape:
        raise ValueError("transport and precision must have the same square shape")
    try:
        inverse = np.linalg.inv(transport)
    except np.linalg.LinAlgError as exc:
        raise ValueError("transport must be invertible") from exc
    return inverse.T @ value @ inverse
