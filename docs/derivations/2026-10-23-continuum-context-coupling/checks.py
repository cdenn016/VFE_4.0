"""Small CPU formula checks for the standalone continuum derivation.

These checks do not prove the Sobolev existence theorem. No model is trained,
and no torch, CUDA, process pool, or background computation is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
J = np.array([[0.0, -1.0], [1.0, 0.0]])
H = np.array([[3.0, -1.0, -2.0, 0.0], [-1.0, 4.0, 0.0, -2.0],
              [-2.0, 0.0, 3.0, -1.0], [0.0, -2.0, -1.0, 4.0]])


def inner(a: Array, b: Array) -> Array:
    """Return pointwise Frobenius inner products over the final two axes."""
    return np.sum(a * b, axis=(-2, -1))


def derivative(a: Array, order: int = 1) -> Array:
    r"""Evaluate the periodic spectral derivative on a 2*pi grid."""
    n = a.shape[0]
    frequency = np.fft.fftfreq(n, d=1.0 / n)
    multiplier = (1j * frequency) ** order
    multiplier = multiplier.reshape((n,) + (1,) * (a.ndim - 1))
    return np.fft.ifft(multiplier * np.fft.fft(a, axis=0), axis=0).real


def rotate(phi: Array, p: Array) -> Array:
    r"""Compute R_phi P, with P shaped (context, 2, 4)."""
    return np.cos(phi)[:, None, None] * p + np.sin(phi)[:, None, None] * (J @ p)


def auxiliary(p: Array) -> tuple[Array, Array]:
    r"""Compute alpha=-<P_x,JP>/||P||^2 and the horizontal residual."""
    p_x = derivative(p)
    alpha = -inner(p_x, J @ p) / inner(p, p)
    residual = p_x + alpha[:, None, None] * (J @ p)
    return alpha, residual


def local_energy(p: Array) -> Array:
    r"""Evaluate V=2+<P,PH>/2 pointwise."""
    return 2.0 + 0.5 * inner(p, p @ H)


def total_energy(p: Array, coupling: float) -> float:
    r"""Evaluate the normalized contextual energy using its eliminated field."""
    _, residual = auxiliary(p)
    return float(np.mean(local_energy(p) + 0.5 * coupling * inner(residual, residual)))


def field_gradient(p: Array, coupling: float) -> Array:
    r"""Return PH-lambda*D_x(D_xP), using periodic integration by parts."""
    alpha, residual = auxiliary(p)
    tension = derivative(residual) + alpha[:, None, None] * (J @ residual)
    return p @ H - coupling * tension


def central_difference(f: Callable[[Array], float], x: Array, step: float = 1e-6) -> Array:
    """Differentiate a small scalar function by centered coordinate differences."""
    output = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        direction = np.zeros_like(x)
        direction[index] = step
        output[index] = (f(x + direction) - f(x - direction)) / (2.0 * step)
    return output


def assert_error(name: str, actual: Array | float, expected: Array | float,
                 tolerance: float, results: list[dict[str, Any]]) -> None:
    """Record and enforce a maximum absolute formula error."""
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    results.append({"name": name, "max_absolute_error": error,
                    "tolerance": tolerance, "passed": error <= tolerance})
    if not error <= tolerance:
        raise AssertionError(f"{name}: {error} > {tolerance}")


def run_checks() -> dict[str, Any]:
    """Execute bounded formula checks, with no PDE simulation or existence claim."""
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    rng = np.random.default_rng(20261023)
    n_grid = 256
    x = 2.0 * np.pi * np.arange(n_grid) / n_grid
    base = np.array([[1.7, 0.6, -0.9, 0.3], [0.2, -0.8, 0.5, 1.1]])
    p = (base[None, :, :] + 0.08 * np.sin(x)[:, None, None] * rng.normal(size=(2, 4))
         + 0.06 * np.cos(2.0 * x)[:, None, None] * rng.normal(size=(2, 4)))
    coupling = 0.7

    # Evaluate the reference-joint KL with its actual normalized covariance.
    covariance = np.block([[2.0 * np.eye(2), np.eye(2)], [np.eye(2), np.eye(2)]])
    precision = np.linalg.inv(covariance)
    _, logdet = np.linalg.slogdet(covariance)
    direct = np.zeros(n_grid)
    for belief_column, model_column in ((0, 1), (2, 3)):
        mean = np.concatenate((p[:, :, belief_column], p[:, :, model_column]), axis=1)
        direct += 0.5 * (np.trace(precision) - 4.0 + logdet
                         + np.einsum("ni,ij,nj->n", mean, precision, mean))
    direct += np.sum((p[:, :, 0] - p[:, :, 2]) ** 2, axis=1)
    direct += np.sum((p[:, :, 1] - p[:, :, 3]) ** 2, axis=1)
    assert_error("normalized_reference_KL_and_peer_potential", local_energy(p), direct, 2e-13, results)
    numerical_local = central_difference(lambda z: float(local_energy(z)), base)
    assert_error("eight_component_local_potential_gradient", numerical_local, base @ H, 1e-8, results)

    alpha, residual = auxiliary(p)
    assert_error("auxiliary_normal_equation", inner(residual, J @ p), 0.0, 2e-14, results)

    angle = 0.2 * np.sin(x) + 0.05 * np.cos(3.0 * x)
    rotated = rotate(angle, p)
    alpha_new, residual_new = auxiliary(rotated)
    assert_error("local_gauge_connection_law", alpha_new, alpha - derivative(angle), 3e-13, results)
    assert_error("local_gauge_residual_covariance", residual_new, rotate(angle, residual), 4e-13, results)
    assert_error("local_gauge_energy_invariance", total_energy(rotated, coupling),
                 total_energy(p, coupling), 2e-13, results)

    gradient = field_gradient(p, coupling)
    directional_errors = []
    for _ in range(6):
        variation = (np.sin(x)[:, None, None] * rng.normal(size=(2, 4))
                     + np.cos(3.0 * x)[:, None, None] * rng.normal(size=(2, 4)))
        eps = 2e-6
        numerical = (total_energy(p + eps * variation, coupling)
                     - total_energy(p - eps * variation, coupling)) / (2.0 * eps)
        predicted = float(np.mean(inner(gradient, variation)))
        directional_errors.append(numerical - predicted)
    assert_error("six_full_functional_directional_derivatives", np.array(directional_errors),
                 0.0, 2e-8, results)
    assert_error("local_gauge_flow_covariance", field_gradient(rotated, coupling),
                 rotate(angle, gradient), 3e-10, results)

    # The adapted-coordinate metric must reproduce the entire block gradient.
    h = np.array([[1.4, 0.2], [0.2, 1.1]])
    k = np.array([[1.3]])
    c = np.array([[0.3, -0.5]])
    v_u, v_r, tension = np.array([0.7, -0.8]), np.array([0.4]), np.array([-0.2, 0.6])
    metric = np.block([[h + c.T @ k @ c, c.T @ k], [k @ c, k]])
    euler = np.concatenate((v_u - coupling * h @ tension, v_r))
    u_t = coupling * tension - np.linalg.solve(h, v_u - c.T @ v_r)
    r_t = -c @ u_t - np.linalg.solve(k, v_r)
    velocity = np.concatenate((u_t, r_t))
    assert_error("nonbasic_adapted_metric_gradient", velocity, -np.linalg.solve(metric, euler), 1e-14, results)
    assert_error("nonbasic_block_dissipation", euler @ velocity,
                 -(u_t @ h @ u_t + v_r @ np.linalg.solve(k, v_r)), 1e-14, results)

    # Check the single Gaussian example against its Cartesian natural gradient.
    u, r, eta, reference = 1.6, 0.4, 0.7, 0.9
    phi = r + eta * u
    e_radial = np.array([np.cos(phi), np.sin(phi)])
    e_tangent = J @ e_radial
    p_single = u * e_radial
    u_t_single = -u + reference * np.cos(phi)
    r_t_single = -eta * u_t_single - reference / u * np.sin(phi)
    p_t_single = e_radial * u_t_single + u * e_tangent * (r_t_single + eta * u_t_single)
    assert_error("fixed_reference_cartesian_reconstruction", p_t_single,
                 -p_single + np.array([reference, 0.0]), 1e-14, results)

    # Nontrivial three-source row: natural gradient, normalization and variance.
    b = np.array([0.3, -0.4])
    prior = np.array([0.2, 0.35, 0.45])
    costs = np.array([0.7, 1.2, 0.3])
    temperature = 0.8

    def row_values(logits: Array) -> Array:
        """Convert two log odds into a normalized three-source row."""
        weight = np.exp(np.concatenate((logits, np.array([0.0]))))
        return weight / weight.sum()

    def row_energy(logits: Array) -> float:
        r"""Evaluate beta dot D + T KL(beta || prior)."""
        beta = row_values(logits)
        return float(beta @ costs + temperature * np.sum(beta * np.log(beta / prior)))

    beta = row_values(b)
    fisher = np.diag(beta[:2]) - np.outer(beta[:2], beta[:2])
    b_t = -(costs[:2] - costs[2]) - temperature * (b - np.log(prior[:2] / prior[2]))
    row_gradient = central_difference(row_energy, b)
    assert_error("three_source_logodds_natural_gradient", fisher @ b_t, -row_gradient, 2e-10, results)
    row_h = costs + temperature * np.log(beta / prior)
    beta_t = -beta * (row_h - beta @ row_h)
    assert_error("row_normalization", beta_t.sum(), 0.0, 2e-16, results)
    assert_error("row_fisher_dissipation", b_t @ fisher @ b_t,
                 np.sum(beta_t ** 2 / beta), 1e-14, results)

    tau = (np.sqrt(5.0) - 1.0) / 2.0
    decay = 5.0 - tau
    vector = np.array([1.0, tau, -1.0, -tau])
    v0 = np.outer(np.array([1.0, 0.0]), vector)
    w = float(np.sum(v0 ** 2))
    assert_error("witness_exact_eigenvector", H @ vector, decay * vector, 2e-15, results)
    a0, a1, mode = 1.3, 0.35, 2
    phase = 0.2 * np.sin(x)
    pde_errors, energy_errors, dissipation_errors, alpha_errors = [], [], [], []
    min_amplitude = float("inf")
    for t in (0.0, 0.07, 0.3, 0.8):
        rate = decay + coupling * mode ** 2
        amplitude = a0 * np.exp(-decay * t) + a1 * np.exp(-rate * t) * np.cos(mode * x)
        amplitude_t = -decay * a0 * np.exp(-decay * t) - rate * a1 * np.exp(-rate * t) * np.cos(mode * x)
        exact_p = rotate(phase, amplitude[:, None, None] * v0)
        exact_p_t = rotate(phase, amplitude_t[:, None, None] * v0)
        exact_alpha, _ = auxiliary(exact_p)
        pde_errors.append(float(np.max(np.abs(exact_p_t + field_gradient(exact_p, coupling)))))
        alpha_errors.append(float(np.max(np.abs(exact_alpha + 0.2 * np.cos(x)))))
        energy = (2.0 + 0.5 * w * decay * a0 ** 2 * np.exp(-2.0 * decay * t)
                  + 0.25 * w * rate * a1 ** 2 * np.exp(-2.0 * rate * t))
        energy_errors.append(total_energy(exact_p, coupling) - energy)
        energy_t = (-w * decay ** 2 * a0 ** 2 * np.exp(-2.0 * decay * t)
                    - 0.5 * w * rate ** 2 * a1 ** 2 * np.exp(-2.0 * rate * t))
        dissipation_errors.append(energy_t + float(np.mean(inner(exact_p_t, exact_p_t))))
        min_amplitude = min(min_amplitude, float(amplitude.min()))
    assert_error("exact_continuum_PDE_at_four_times", np.array(pde_errors), 0.0, 4e-10, results)
    assert_error("exact_auxiliary_connection_at_four_times", np.array(alpha_errors), 0.0, 1e-13, results)
    assert_error("exact_integrated_energy_at_four_times", np.array(energy_errors), 0.0, 1e-13, results)
    assert_error("exact_integrated_dissipation_at_four_times", np.array(dissipation_errors), 0.0, 2e-13, results)
    assert min_amplitude > 0.0

    velocity_mode_one = (-decay * (a0 + a1) - coupling * a1) * v0
    velocity_mode_two = (-decay * (a0 + a1) - 4.0 * coupling * a1) * v0
    assert_error("same_local_state_different_context_velocity", velocity_mode_two - velocity_mode_one,
                 -3.0 * coupling * a1 * v0, 2e-15, results)
    rotation_tangent = (J @ base).ravel()
    projection = np.eye(8) - np.outer(rotation_tangent, rotation_tangent) / (rotation_tangent @ rotation_tangent)
    assert_error("horizontal_principal_symbol_one_zero_mode", np.linalg.eigvalsh(projection),
                 np.array([0.0] + [1.0] * 7), 1e-14, results)

    return {"status": "passed", "check_count": len(results), "checks": results,
            "runtime_seconds": time.perf_counter() - started,
            "python": platform.python_version(), "numpy": np.__version__,
            "threads": {name: os.environ[name] for name in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "periodic_grid_points": n_grid, "seed": 20261023,
            "minimum_sampled_exact_amplitude": min_amplitude,
            "scope": "Formula checks only; the local Sobolev theorem is established by the written proof.",
            "not_run": ["PDE convergence experiment", "global existence experiment", "GPU", "training", "Lean"]}


if __name__ == "__main__":
    record = run_checks()
    path = Path(__file__).with_name("check-results.json")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
