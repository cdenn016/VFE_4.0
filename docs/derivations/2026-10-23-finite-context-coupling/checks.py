"""Small CPU checks of the explicit Gaussian witness, not a general theorem test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rotation(theta: float) -> Array:
    """Return the left SO(2) action in the declared planar coordinates."""
    return np.cos(theta) * np.eye(2) + np.sin(theta) * J


def alignment(X: Array, Y: Array) -> tuple[Array, float, float, float]:
    r"""Minimize c = ||Y - R X||_F^2 / 2 on the regular rho > 0 domain."""
    C = float(np.sum(Y * X))
    S = float(np.sum(Y * (J @ X)))
    rho = float(np.hypot(C, S))
    if rho <= 1e-12:
        raise ValueError("The auxiliary minimizing branch is singular.")
    return (C * np.eye(2) + S * J) / rho, C, S, rho


def contextual(X: Array, Y: Array) -> float:
    """Evaluate the original edge cost at its exact minimizing rotation."""
    R, _, _, _ = alignment(X, Y)
    return float(0.5 * np.sum((Y - R @ X) ** 2))


def contextual_gradient(X: Array, Y: Array) -> tuple[Array, Array]:
    """Return both endpoint envelope gradients in the location Fisher metric."""
    R, _, _, _ = alignment(X, Y)
    return X - R.T @ Y, Y - R @ X


def local_energy(X: Array) -> float:
    r"""Evaluate KL(q_i tensor s_i || r_i G_i) and both directed peer terms."""
    value = 0.0
    for q_col, s_col in ((0, 1), (2, 3)):
        mu_q, mu_s = X[:, q_col], X[:, s_col]
        value += 0.5 * np.sum((mu_q - mu_s) ** 2)
        value += 0.5 * np.sum(mu_s**2) + 1.0
    value += np.sum((X[:, 0] - X[:, 2]) ** 2)
    value += np.sum((X[:, 1] - X[:, 3]) ** 2)
    return float(value)


def local_gradient(X: Array) -> Array:
    """Differentiate every local source and receiver role for two agents."""
    result = np.zeros_like(X)
    for q_col, s_col, other_q, other_s in ((0, 1, 2, 3), (2, 3, 0, 1)):
        result[:, q_col] = X[:, q_col] - X[:, s_col]
        result[:, q_col] += 2 * (X[:, q_col] - X[:, other_q])
        result[:, s_col] = 2 * X[:, s_col] - X[:, q_col]
        result[:, s_col] += 2 * (X[:, s_col] - X[:, other_s])
    return result


def total_energy(Z: Array, lam: float = 1.0) -> float:
    """Evaluate the reduced two-context reciprocal potential."""
    X, Y = Z[:8].reshape(2, 4), Z[8:].reshape(2, 4)
    return local_energy(X) + local_energy(Y) + lam * contextual(X, Y)


def total_gradient(Z: Array, lam: float = 1.0) -> Array:
    """Return the gradient in the unit product Fisher metric."""
    X, Y = Z[:8].reshape(2, 4), Z[8:].reshape(2, 4)
    CX, CY = contextual_gradient(X, Y)
    return np.concatenate(((local_gradient(X) + lam * CX).ravel(),
                           (local_gradient(Y) + lam * CY).ravel()))


def central_gradient(function: Callable[[Array], float], value: Array) -> Array:
    """Estimate a scalar derivative independently by centered differences."""
    result = np.zeros_like(value)
    step = 1e-6
    for index in np.ndindex(value.shape):
        delta = np.zeros_like(value)
        delta[index] = step
        result[index] = (function(value + delta) - function(value - delta)) / (2 * step)
    return result


def main() -> None:
    """Run deterministic finite checks and save actual numeric measurements."""
    results: dict[str, object] = {}
    X = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    Y = np.array([[2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]])
    R, C, S, rho = alignment(X, Y)
    Z = np.concatenate((X.ravel(), Y.ravel()))
    velocity = -total_gradient(Z)
    assert np.allclose(R, np.eye(2))
    assert contextual(X, Y) == 3.0
    assert total_energy(Z) == 37.5
    assert np.dot(velocity, velocity) == 323.0
    results["exact_witness"] = {
        "C": C, "S": S, "rho": rho, "context_energy": contextual(X, Y),
        "total_energy": total_energy(Z), "energy_derivative": -float(velocity @ velocity),
        "source_velocity": velocity[:8].reshape(2, 4).tolist(),
        "target_velocity": velocity[8:].reshape(2, 4).tolist(),
    }

    # Nonzero S exercises the rotation response in every endpoint derivative.
    X_test = X + np.array([[0.1, -0.2, 0.15, 0.0], [0.05, 0.1, -0.1, 0.2]])
    Y_test = rotation(0.47) @ Y
    Z_test = np.concatenate((X_test.ravel(), Y_test.ravel()))
    g = total_gradient(Z_test)
    fd = central_gradient(total_energy, Z_test)
    gradient_error = float(np.max(np.abs(g - fd)))
    assert gradient_error < 2e-7
    results["all_16_total_gradient_components"] = {"max_abs_error": gradient_error}

    R_test, C_test, S_test, rho_test = alignment(X_test, Y_test)
    theta = float(np.arctan2(S_test, C_test))
    angle_step = 1e-4
    angle_cost = lambda a: float(0.5 * np.sum((Y_test - rotation(a) @ X_test) ** 2))
    first = (angle_cost(theta + angle_step) - angle_cost(theta - angle_step)) / (2 * angle_step)
    second = (angle_cost(theta + angle_step) - 2 * angle_cost(theta)
              + angle_cost(theta - angle_step)) / angle_step**2
    assert abs(first) < 1e-9 and abs(second - rho_test) < 1e-6
    results["stationarity_and_group_hessian"] = {
        "stationarity_residual": first, "finite_difference_hessian": second,
        "analytic_hessian": rho_test,
    }

    HX, HY = rotation(0.31), rotation(-0.62)
    X_gauge, Y_gauge = HX @ X_test, HY @ Y_test
    R_gauge, _, _, _ = alignment(X_gauge, Y_gauge)
    CX, CY = contextual_gradient(X_test, Y_test)
    CX_gauge, CY_gauge = contextual_gradient(X_gauge, Y_gauge)
    gauge_error = max(float(np.max(np.abs(R_gauge - HY @ R_test @ HX.T))),
                      float(np.max(np.abs(CX_gauge - HX @ CX))),
                      float(np.max(np.abs(CY_gauge - HY @ CY))))
    assert gauge_error < 1e-12
    assert abs(contextual(X_gauge, Y_gauge) - contextual(X_test, Y_test)) < 1e-12
    assert abs(local_energy(X_gauge) - local_energy(X_test)) < 1e-12
    assert contextual(X, rotation(0.43) @ X) < 1e-28
    results["endpoint_gauge_covariance_and_pure_frame_zero_cost"] = {"max_abs_error": gauge_error}

    Y_changed = Y.copy()
    Y_changed[0, 0] += 1.0
    changed = np.concatenate((X.ravel(), Y_changed.ravel()))
    delta = (-total_gradient(changed) - velocity)[:8].reshape(2, 4)
    expected = np.zeros_like(X)
    expected[0, 0] = 1.0
    assert np.allclose(delta, expected)
    assert np.allclose(total_gradient(Z, 0.0),
                       np.concatenate((local_gradient(X).ravel(), local_gradient(Y).ravel())))
    results["neighbor_influence_and_zero_coupling"] = {"local_velocity_change": delta.tolist()}

    step = 1e-6
    numerical_rate = (total_energy(Z + step * velocity) - total_energy(Z - step * velocity)) / (2 * step)
    assert abs(numerical_rate + 323.0) < 1e-7
    results["dissipation_directional_derivative"] = {"measured": numerical_rate, "exact": -323.0}

    beta = np.array([0.2, 0.5, 0.3])
    pi = np.array([0.4, 0.35, 0.25])
    costs = np.array([0.3, 1.2, 0.7])
    temperature = 0.6
    h = costs + temperature * np.log(beta / pi)
    beta_dot = -beta * (h - beta @ h)
    row_energy = lambda b: float(b @ costs + temperature * np.sum(b * np.log(b / pi)))
    row_rate = (row_energy(beta + step * beta_dot) - row_energy(beta - step * beta_dot)) / (2 * step)
    row_norm = float(np.sum(beta_dot**2 / beta))
    assert abs(np.sum(beta_dot)) < 1e-15 and abs(row_rate + row_norm) < 1e-9
    results["nontrivial_row_fisher_dissipation"] = {"rate": row_rate, "minus_norm": -row_norm}

    V = np.array([[0.1, 0.2, -0.3, 0.2], [0.4, -0.2, 0.3, 0.1]])
    b = float(np.sum(V * (J @ X)))
    N = float(np.sum(X**2))
    a = -b / N
    eps = 1e-5
    R_eps, _, _, _ = alignment(X, X + eps * V)
    angle_slope = float(np.arctan2(R_eps[1, 0], R_eps[0, 0]) / eps)
    scaled_cost = contextual(X, X + eps * V) / eps**2
    expected_cost = 0.5 * (float(np.sum(V**2)) - b**2 / N)
    assert abs(angle_slope + a) < 1e-6 and abs(scaled_cost - expected_cost) < 1e-6
    assert abs(N * a + b) < 1e-15
    results["infinitesimal_normal_equation"] = {
        "A_scalar": a, "transport_angle_over_epsilon": angle_slope,
        "scaled_edge_energy": scaled_cost, "projected_tangent_energy": expected_cost,
    }

    x_cusp = np.eye(2)
    y_cusp = np.diag([1.0, -1.0])
    flat_costs = [float(0.5 * np.sum((x_cusp - rotation(t) @ y_cusp) ** 2))
                  for t in (0.0, 0.3, 1.2, np.pi)]
    assert np.allclose(flat_costs, 2.0)
    cusp_eps = 1e-5
    cusp_values: dict[str, float] = {}
    for sign in (-1, 1):
        epsilon = sign * cusp_eps
        x_eps = np.diag([1.0, 1.0 + epsilon])
        value = contextual(y_cusp, x_eps)
        formula = 2.0 + epsilon + 0.5 * epsilon**2 - abs(epsilon)
        assert abs(value - formula) < 1e-14
        cusp_values[str(sign)] = (value - 2.0) / epsilon
    assert abs(cusp_values["-1"] - 2.0) < 1e-4
    assert abs(cusp_values["1"]) < 1e-4
    results["free_action_minimizer_cusp"] = {
        "flat_rotation_costs": flat_costs, "left_slope": cusp_values["-1"],
        "right_slope": cusp_values["1"],
    }
    report = {"status": "passed", "check_groups": len(results), "numpy": np.__version__,
              "compute": "single-thread CPU; no torch, GPU, process pools, or trajectories",
              "scope": "explicit finite examples only; analytical theorem in main.tex",
              "results": results}
    output = Path(__file__).with_name("check-results.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
