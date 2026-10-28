from __future__ import annotations

import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parent


def gaussian_kl(mu: sp.Expr, variance: sp.Expr, reference_mu: sp.Expr, reference_variance: sp.Expr) -> sp.Expr:
    r"""Return KL(N(mu, variance) || N(reference_mu, reference_variance))."""
    return (variance / reference_variance + (mu - reference_mu) ** 2 / reference_variance
            - 1 + sp.log(reference_variance / variance)) / 2


def main() -> None:
    mu0, mu1, mu2, a1, a2 = sp.symbols("mu0 mu1 mu2 a1 a2", real=True)
    v0, v1, v2, V1, V2 = sp.symbols("v0 v1 v2 V1 V2", positive=True)
    x = sp.Matrix([mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2])
    beta, temperature = sp.symbols("beta T", positive=True)
    r = sp.Rational
    costs = []
    for mu_j, v_j, a_j, V_j in ((mu1, v1, a1, V1), (mu2, v2, a2, V2)):
        discrepancy = mu0 - mu_j - a_j
        costs.append(((v0 + 1 + discrepancy**2 + V_j) / (v_j + 1)
                      - 1 + sp.log((v_j + 1) / (v0 + 1))) / 2)
    local = (gaussian_kl(mu0, v0, 0, 1)
             + gaussian_kl(mu1, v1, -r(23, 15), r(4, 5))
             + gaussian_kl(mu2, v2, r(23, 15), r(4, 5))
             + gaussian_kl(a1, V1, 0, r(4, 3))
             + gaussian_kl(a2, V2, 0, r(4, 3)))
    fixed = local + (costs[0] + costs[1]) / 2
    full = local + beta * costs[0] + (1 - beta) * costs[1]
    full += temperature * (beta * sp.log(2 * beta) + (1 - beta) * sp.log(2 * (1 - beta)))
    point_values = [0, -r(4, 3), r(4, 3), r(1, 3), -r(1, 3), 1, 1, 1, 1, 1]
    point = dict(zip(x, point_values))
    gradient = fixed.diff(x).subs(point).applyfunc(sp.simplify)
    assert gradient == sp.zeros(10, 1)
    H = sp.hessian(fixed, x).subs(point).applyfunc(sp.simplify)
    g = (costs[0] - costs[1]).diff(x).subs(point).applyfunc(sp.simplify)
    y = H.inv() * g
    y = y.applyfunc(sp.simplify)
    T_c = sp.factor((g.T * y)[0] / 4)
    assert T_c == r(3269, 10118), T_c
    H_mean, g_mean = H[:5, :5], g[:5, :]
    T_mean = sp.factor((g_mean.T * H_mean.inv() * g_mean)[0] / 4)
    assert T_mean == r(31, 122), T_mean
    assert T_mean < r(3, 10) < T_c
    perturbations = sp.symbols("x0 x1 x2 b1 b2 h0 h1 h2 k1 k2", real=True)
    x0, x1, x2, b1, b2, h0, h1, h2, k1, k2 = perturbations
    z = sp.Matrix(perturbations)
    certificate = x0**2 + r(5, 4) * (x1**2 + x2**2) + r(3, 4) * (b1**2 + b2**2) + h0**2 / 2
    for x_j, b_j, h_j, k_j, sign in ((x1, b1, h1, k1, 1), (x2, b2, h2, k2, -1)):
        d_j = x0 - x_j - b_j
        certificate += (r(1, 4) * (d_j - sign * h_j / 2)**2
                        + r(1, 16) * (h0 - h_j)**2 + r(71, 128) * h_j**2
                        + r(1, 2) * (k_j - h_j / 8)**2)
    certificate_residual = sp.expand((z.T * H * z)[0] - certificate)
    assert certificate_residual == 0, certificate_residual
    principal_minors = [sp.factor(H[:n, :n].det()) for n in range(1, 11)]
    assert all(value > 0 for value in principal_minors)
    joint_x = sp.Matrix([*x, beta])
    full_gradient = full.diff(joint_x).subs(point | {beta: r(1, 2)}).applyfunc(sp.simplify)
    assert full_gradient == sp.zeros(11, 1)
    full_H = sp.hessian(full, joint_x).subs(point | {beta: r(1, 2)}).applyfunc(sp.simplify)
    assert full_H[:10, :10] == H
    assert full_H[:10, 10] == g
    assert full_H[10, 10] == 4 * temperature
    # Compute the reduced Hessian directly, independently of the block identity.
    reduced = local - temperature * sp.log((sp.exp(-costs[0] / temperature) + sp.exp(-costs[1] / temperature)) / 2)
    reduced_H = sp.hessian(reduced, x).subs(point).applyfunc(sp.simplify)
    assert (reduced_H - H + g * g.T / (4 * temperature)).applyfunc(sp.simplify) == sp.zeros(10)
    mobility = np.diag([1.0] * 5 + [2.0] * 5)
    joint_mobility = np.diag([1.0] * 5 + [2.0] * 5 + [0.25])
    numerical = []
    for value in (r(1, 4), r(3, 10), r(1, 2)):
        reduced_numeric = np.array(reduced_H.subs(temperature, value), dtype=float)
        full_numeric = np.array(full_H.subs(temperature, value), dtype=float)
        eigen_reduced = np.linalg.eigvalsh(-np.sqrt(mobility) @ reduced_numeric @ np.sqrt(mobility))
        eigen_full = np.linalg.eigvalsh(-np.sqrt(joint_mobility) @ full_numeric @ np.sqrt(joint_mobility))
        numerical.append({"temperature": str(value), "reduced_jacobian_eigenvalues": eigen_reduced.tolist(),
                          "full_jacobian_eigenvalues": eigen_full.tolist(),
                          "frozen_covariance_mean_jacobian_eigenvalues":
                          np.linalg.eigvalsh(-reduced_numeric[:5, :5]).tolist()})
    result = {
        "coordinate_order": [str(var) for var in x], "stationary_point": [str(value) for value in point_values],
        "costs_at_stationary_point": [str(sp.simplify(cost.subs(point))) for cost in costs],
        "fixed_row_gradient": [str(value) for value in gradient], "full_gradient": [str(value) for value in full_gradient],
        "fixed_row_hessian": [[str(value) for value in row] for row in H.tolist()],
        "cost_difference_gradient": [str(value) for value in g], "inverse_hessian_gradient": [str(value) for value in y],
        "critical_temperature": str(T_c), "critical_temperature_decimal": float(T_c),
        "frozen_covariance_critical_temperature": str(T_mean),
        "strict_threshold_separation_at_three_tenths": True,
        "sum_of_squares_residual": str(certificate_residual),
        "leading_principal_minors": [str(value) for value in principal_minors],
        "direct_reduced_hessian_identity": True, "full_block_hessian_identity": True,
        "eigenvalue_spot_checks": numerical,
        "scope": "Exact algebra for one scalar Gaussian restricted family; eigenvalues are spot checks, not the proof of the temperature criterion.",
    }
    (OUT / "stability-checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stationary_gradient": "zero", "fixed_row_hessian": "positive definite",
                      "critical_temperature": str(T_c), "direct_reduced_hessian_identity": True,
                      "sum_of_squares_residual": str(certificate_residual), "eigenvalue_spot_checks": numerical}, indent=2))


if __name__ == "__main__":
    main()
