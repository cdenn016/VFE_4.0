"""Tiny CPU checks of the coupled Gaussian derivatives and dissipation identity."""

from __future__ import annotations

import json
import os
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "4"

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
RNG = np.random.default_rng(20261026)
EDGES = ((0, 1), (0, 2), (1, 0))  # (receiver, source)
TEMPERATURE = 0.7
ROW_PRIOR = np.array([0.35, 0.65])


def spd(size: int) -> Array:
    """Return a small non-diagonal positive-definite matrix."""
    factor = RNG.normal(size=(size, size)) / np.sqrt(size)
    return factor @ factor.T + np.eye(size)


P = [spd(2) for _ in range(3)]
M = RNG.normal(size=(3, 2))
CODES = [np.array([[1.0, 0.3], [-0.2, 1.1]]),
         np.array([[0.8, -0.4], [0.1, 1.2]]),
         np.array([[1.1, 0.2], [0.4, 0.9]])]
NOISE = [0.3 * spd(2) for _ in EDGES]
OPERATING_CONTENT = [np.array([1.0, 0.4]), np.array([-0.3, 0.8]),
                     np.array([0.2, -0.5])]
KAPPA = [np.kron(np.eye(2), value.reshape(1, 2)) for value in OPERATING_CONTENT]
B = np.concatenate((np.eye(4), np.zeros((4, 4)), np.eye(4)), axis=1)
A = 1.2 * np.eye(12) + 0.3 * B.T @ spd(4) @ B
PRIOR_LINEAR = 0.1 * RNG.normal(size=12)
A0 = np.linalg.solve(A, PRIOR_LINEAR)
BLOCKS = [slice(4 * edge, 4 * edge + 4) for edge in range(3)]


def logdet(matrix: Array) -> float:
    """Return log determinant, checking positivity in the tested SPD domain."""
    sign, result = np.linalg.slogdet(matrix)
    assert sign > 0
    return float(result)


def pack(mu: Array, sigma: list[Array], a: Array, variance: list[Array]) -> Array:
    """Pack mean vectors and independent symmetric covariance coordinates."""
    return np.concatenate([mu.ravel(), *[s[np.triu_indices(2)] for s in sigma],
                           a.ravel(), *[v[np.triu_indices(4)] for v in variance]])


def unpack(coordinates: Array) -> tuple[Array, list[Array], Array, list[Array]]:
    """Recover all 57 matter coordinates with symmetric off-diagonal entries."""
    mu = coordinates[:6].reshape(3, 2)
    cursor = 6
    sigma = []
    for _ in range(3):
        matrix = np.zeros((2, 2))
        matrix[np.triu_indices(2)] = coordinates[cursor:cursor + 3]
        matrix += np.triu(matrix, 1).T
        sigma.append(matrix)
        cursor += 3
    a = coordinates[cursor:cursor + 12].reshape(3, 4)
    cursor += 12
    variance = []
    for _ in range(3):
        matrix = np.zeros((4, 4))
        matrix[np.triu_indices(4)] = coordinates[cursor:cursor + 10]
        matrix += np.triu(matrix, 1).T
        variance.append(matrix)
        cursor += 10
    assert cursor == len(coordinates) == 57
    return mu, sigma, a, variance


def matter_data(coordinates: Array) -> tuple[float, Array, list[tuple]]:
    r"""Evaluate F_0 and C_e = E_{r_e} KL(A_e || B_e) directly."""
    mu, sigma, a, variance = unpack(coordinates)
    base = 0.0
    for i in range(3):
        displacement = mu[i] - M[i]
        base += 0.5 * (displacement @ P[i] @ displacement
                       + np.trace(P[i] @ sigma[i]) - logdet(P[i])
                       - logdet(sigma[i]) - 2)
    displacement = a.ravel() - A0
    base += 0.5 * (displacement @ A @ displacement
                   + sum(np.trace(A[block, block] @ v)
                         for block, v in zip(BLOCKS, variance))
                   - logdet(A) - sum(logdet(v) for v in variance) - 12)
    costs, edge_data = [], []
    for e, (i, j) in enumerate(EDGES):
        code = CODES[e]
        h = code @ KAPPA[i]  # message x dictionary: (2, 4)
        si = code @ sigma[i] @ code.T + NOISE[e]
        sj = code @ sigma[j] @ code.T + NOISE[e]
        inverse_source = np.linalg.inv(sj)
        d = code @ (mu[i] - mu[j]) - h @ a[e]
        q = si + np.outer(d, d) + h @ variance[e] @ h.T
        costs.append(0.5 * (np.trace(inverse_source @ q) - 2
                            + logdet(sj) - logdet(si)))
        edge_data.append((h, si, inverse_source, d, q))
    return float(base), np.asarray(costs), edge_data


def energy(coordinates: Array, row: float | None) -> float:
    """Evaluate the full potential or the stable Gibbs-reduced potential."""
    base, costs, _ = matter_data(coordinates)
    if row is None:
        logits = np.log(ROW_PRIOR) - costs[:2] / TEMPERATURE
        maximum = np.max(logits)
        return float(base - TEMPERATURE * (maximum + np.log(np.exp(logits - maximum).sum()))
                     + costs[2])
    beta = np.array([row, 1.0 - row])
    return float(base + beta @ costs[:2] + costs[2]
                 + TEMPERATURE * np.sum(beta * np.log(beta / ROW_PRIOR)))


def derivatives(coordinates: Array, row: float | None) -> tuple[Array, Array, float, float]:
    r"""Return dF, Fisher velocity, row derivative, and predicted dissipation.

    Covariance velocity is -2 Sigma G_Sigma Sigma, with trace-pairing gradients.
    """
    mu, sigma, a, variance = unpack(coordinates)
    _, costs, edge_data = matter_data(coordinates)
    if row is None:
        logits = np.log(ROW_PRIOR) - costs[:2] / TEMPERATURE
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
    else:
        weights = np.array([row, 1.0 - row])
    beta = np.concatenate((weights, [1.0]))
    g_mu = np.asarray([P[i] @ (mu[i] - M[i]) for i in range(3)])
    g_sigma = [0.5 * (P[i] - np.linalg.inv(sigma[i])) for i in range(3)]
    g_a = (A @ a.ravel() - PRIOR_LINEAR).reshape(3, 4)
    g_variance = [0.5 * (A[block, block] - np.linalg.inv(variance[e]))
                  for e, block in enumerate(BLOCKS)]
    for e, (i, j) in enumerate(EDGES):
        h, si, j_inv, d, q = edge_data[e]
        code = CODES[e]
        mean_force = beta[e] * code.T @ j_inv @ d
        g_mu[i] += mean_force
        g_mu[j] -= mean_force
        g_a[e] -= beta[e] * h.T @ j_inv @ d
        g_sigma[i] += 0.5 * beta[e] * code.T @ (j_inv - np.linalg.inv(si)) @ code
        g_sigma[j] += 0.5 * beta[e] * code.T @ (j_inv - j_inv @ q @ j_inv) @ code
        g_variance[e] += 0.5 * beta[e] * h.T @ j_inv @ h
    velocity_mu = np.asarray([-s @ g for s, g in zip(sigma, g_mu)])
    velocity_sigma = [-2 * s @ g @ s for s, g in zip(sigma, g_sigma)]
    velocity_a = np.asarray([-v @ g for v, g in zip(variance, g_a)])
    velocity_variance = [-2 * v @ g @ v for v, g in zip(variance, g_variance)]
    velocity = pack(velocity_mu, velocity_sigma, velocity_a, velocity_variance)
    # Independent off-diagonal coordinates affect two matrix entries.
    coordinate_sigma = [2 * g - np.diag(np.diag(g)) for g in g_sigma]
    coordinate_variance = [2 * g - np.diag(np.diag(g)) for g in g_variance]
    gradient = pack(g_mu, coordinate_sigma, g_a, coordinate_variance)
    dissipation = sum(float(g @ s @ g) for s, g in zip(sigma, g_mu))
    dissipation += sum(float(2 * np.trace(s @ g @ s @ g)) for s, g in zip(sigma, g_sigma))
    dissipation += sum(float(g @ v @ g) for v, g in zip(variance, g_a))
    dissipation += sum(float(2 * np.trace(v @ g @ v @ g)) for v, g in zip(variance, g_variance))
    row_derivative = 0.0
    if row is not None:
        row_derivative = float(costs[0] - costs[1] + TEMPERATURE * np.log(
            row * ROW_PRIOR[1] / ((1 - row) * ROW_PRIOR[0])))
        dissipation += row * (1 - row) * row_derivative**2
    return gradient, velocity, row_derivative, dissipation


def main() -> None:
    """Check all coordinates at one noncommuting SPD point, with no simulation."""
    coordinates = pack(RNG.normal(size=(3, 2)), [spd(2) for _ in range(3)],
                       RNG.normal(size=(3, 4)), [spd(4) for _ in range(3)])
    step = 2e-6
    results = {}
    for name, row in (("full", 0.42), ("reduced", None)):
        gradient, velocity, row_derivative, dissipation = derivatives(coordinates, row)
        numerical = np.empty_like(coordinates)
        for index in range(len(coordinates)):
            displacement = np.zeros_like(coordinates)
            displacement[index] = step
            numerical[index] = (energy(coordinates + displacement, row)
                                - energy(coordinates - displacement, row)) / (2 * step)
        error = float(np.max(np.abs(gradient - numerical)))
        assert error < 5e-8, (name, error)
        contraction = float(gradient @ velocity)
        row_error = None
        if row is not None:
            numerical_row = (energy(coordinates, row + step) - energy(coordinates, row - step)) / (2 * step)
            row_error = abs(numerical_row - row_derivative)
            assert row_error < 5e-8
            contraction -= row * (1 - row) * row_derivative**2
        residual = abs(contraction + dissipation)
        assert residual < 1e-9
        assert dissipation > 0
        results[name] = {"coordinates_checked": 57 + (row is not None),
                         "maximum_matter_derivative_error": error,
                         "row_derivative_error": row_error,
                         "dissipation_identity_residual": residual,
                         "negative_energy_derivative": dissipation}
    result = {"seed": 20261026, "step": step, "dimension": 2,
              "dictionary_dimension": 4, "agents": 3, "edges": 3,
              "scope": "one finite-difference point; correlated prior; noncommuting matrices",
              "checks": results}
    output = Path(__file__).with_name("matrix-flow-checks.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
