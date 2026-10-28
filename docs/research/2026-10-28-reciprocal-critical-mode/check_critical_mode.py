"""Exact local jets for the fully coupled three-agent Gaussian example."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"

import sympy as sp

OUT = Path(__file__).resolve().parent


def gaussian_kl(mu: sp.Expr, variance: sp.Expr, mean: sp.Expr, precision: sp.Expr) -> sp.Expr:
    r"""Return $D_{KL}(N(\mu,v)\Vert N(m,P^{-1}))$."""
    return (precision * variance + precision * (mu - mean)**2 - 1
            - sp.log(precision * variance)) / 2


def main() -> None:
    """Compute rational stationary jets without integrating either ODE."""
    started = time.perf_counter()
    r = sp.Rational
    mu0, mu1, mu2, a1, a2 = sp.symbols("mu0 mu1 mu2 a1 a2", real=True)
    v0, v1, v2, V1, V2 = sp.symbols("v0 v1 v2 V1 V2", positive=True)
    x = sp.Matrix([mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2])
    x0 = sp.Matrix([0, -r(4, 3), r(4, 3), r(1, 3), -r(1, 3), 1, 1, 1, 1, 1])
    point = dict(zip(x, x0))
    costs = []
    for mu, variance, a, V in ((mu1, v1, a1, V1), (mu2, v2, a2, V2)):
        costs.append(((v0 + 1 + (mu0 - mu - a)**2 + V) / (variance + 1)
                      - 1 + sp.log((variance + 1) / (v0 + 1))) / 2)
    F = (gaussian_kl(mu0, v0, 0, 1)
         + gaussian_kl(mu1, v1, -r(23, 15), r(5, 4))
         + gaussian_kl(mu2, v2, r(23, 15), r(5, 4))
         + gaussian_kl(a1, V1, 0, r(3, 4))
         + gaussian_kl(a2, V2, 0, r(3, 4)))
    A = F + (costs[0] + costs[1]) / 2
    delta = costs[0] - costs[1]
    gradient_A = A.diff(x)
    assert gradient_A.subs(point) == sp.zeros(10, 1)
    hessian_A = sp.hessian(A, x)
    hessian_delta = sp.hessian(delta, x)
    H = hessian_A.subs(point)
    g = delta.diff(x).subs(point)
    p = -H.inv() * g
    Tc = sp.factor(-(g.T * p)[0] / 4)
    assert Tc == r(3269, 10118)

    def directional(expr: sp.Expr, order: int = 1) -> sp.Expr:
        """Contract coordinate derivatives with the constant vector p."""
        for _ in range(order):
            expr = sum(p[i] * sp.diff(expr, x[i]) for i in range(10))
        return expr

    k = sp.Matrix([sp.factor((directional(gradient_A[i], 2)
                             + 2 * directional(sp.diff(delta, x[i]))).subs(point))
                   for i in range(10)])
    x2 = (-H.inv() * k).applyfunc(sp.factor)
    A4 = sp.factor(directional(A, 4).subs(point))
    delta3 = sp.factor(directional(delta, 3).subs(point))
    correction = sp.factor(3 * (k.T * H.inv() * k)[0])
    matter_fourth = sp.factor(A4 + 4 * delta3 - correction)
    fourth = sp.factor(matter_fourth + 32 * Tc)
    cubic_force = sp.factor(fourth / 6)
    metric = sp.diag(*([1] * 5 + [r(1, 2)] * 5))
    mobility = metric.inv()
    D = sp.factor((p.T * metric * p)[0])

    # Reconstruct the force coefficient from X''' rather than the fourth-envelope identity.
    third_rhs = sp.Matrix([
        sp.factor((directional(gradient_A[i], 3)
                   + 3 * directional(sp.diff(delta, x[i]), 2)
                   + 3 * sum((directional(hessian_A[i, j]) + hessian_delta[i, j])
                             * x2[j] for j in range(10))).subs(point))
        for i in range(10)
    ])
    x3 = (-H.inv() * third_rhs).applyfunc(sp.factor)
    direct_cubic_force = sp.factor(
        (g.T * x3)[0] / 6 + (p.T * hessian_delta.subs(point) * x2)[0] / 2
        + delta3 / 6 + r(16, 3) * Tc
    )
    assert direct_cubic_force == cubic_force
    assert cubic_force < 0
    assert D > 0

    # The reflection/exchange symmetry is exact on the complete matter space.
    reflected = sp.Matrix([-mu0, -mu2, -mu1, -a2, -a1, v0, v2, v1, V2, V1])
    symmetry = dict(zip(x, reflected))
    assert sp.simplify(A.subs(symmetry, simultaneous=True) - A) == 0
    assert sp.simplify(delta.subs(symmetry, simultaneous=True) + delta) == 0
    assert reflected.subs(point) == x0

    # Leading center-manifold invariance, in the physical row coordinate.
    B = H.inv() * metric * p
    reduced_hessian = H - g * g.T / (4 * Tc)
    full_hessian = H.row_join(g).col_join(g.T.row_join(sp.Matrix([[4 * Tc]])))
    full_mobility = sp.diag(mobility, r(1, 4))
    full_tangent = p.col_join(sp.Matrix([1]))
    assert reduced_hessian * p == sp.zeros(10, 1)
    assert full_hessian * full_tangent == sp.zeros(11, 1)
    assert sp.factor((g.T * B)[0]) == -D
    kinetics = {}
    for kind, norm in (("reduced", D), ("full", D + 4)):
        alpha = sp.factor(-4 / norm)
        beta = sp.factor(-cubic_force / norm)
        cubic_shift = -B * beta
        parameter_shift = -B * alpha
        # At orders s^3 and (T-Tc)s, xdot = p * sdot.
        assert (-mobility * H * cubic_shift - p * beta).applyfunc(sp.factor) == sp.zeros(10, 1)
        assert (-mobility * H * parameter_shift - p * alpha).applyfunc(sp.factor) == sp.zeros(10, 1)
        row_cubic = sp.factor(cubic_force + (g.T * cubic_shift)[0])
        row_parameter = sp.factor(4 + (g.T * parameter_shift)[0])
        if kind == "full":
            assert sp.factor(-row_cubic / 4) == beta
            assert sp.factor(-row_parameter / 4) == alpha
            left = full_mobility.inv() * full_tangent / norm
            jacobian = -full_mobility * full_hessian
            jacobian_T = sp.zeros(11)
            jacobian_T[10, 10] = -1
            tangent = full_tangent
        else:
            assert row_cubic == 0
            assert row_parameter == 0
            left = metric * p / norm
            jacobian = -mobility * reduced_hessian
            jacobian_T = -mobility * g * g.T / (4 * Tc**2)
            tangent = p
        assert (left.T * jacobian).applyfunc(sp.factor) == sp.zeros(1, len(tangent))
        assert sp.factor((left.T * tangent)[0]) == 1
        assert sp.factor((left.T * jacobian_T * tangent)[0]) == alpha
        kinetics[kind] = {
            "linear_parameter_coefficient": str(alpha),
            "cubic_coefficient": str(beta),
            "linear_parameter_decimal": float(alpha),
            "cubic_decimal": float(beta),
            "offshoot_eigenvalue_slope": str(sp.factor(8 / norm)),
            "center_manifold_invariance_at_leading_orders": True,
            "linear_coefficient_from_left_null_projection": True,
        }
    result = {
        "coordinate_order": [str(z) for z in x],
        "critical_temperature": str(Tc),
        "X_prime": [str(z) for z in p],
        "k": [str(z) for z in k],
        "X_second": [str(z) for z in x2],
        "X_third": [str(z) for z in x3],
        "A_fourth_along_p": str(A4),
        "delta_third_along_p": str(delta3),
        "stable_direction_fourth_correction": str(correction),
        "matter_envelope_fourth": str(matter_fourth),
        "total_envelope_fourth_at_critical": str(fourth),
        "cubic_potential_derivative_coefficient": str(cubic_force),
        "cubic_potential_derivative_decimal": float(cubic_force),
        "matter_metric_norm_D": str(D),
        "inverse_metric_norm_reduced": str(sp.factor(1 / D)),
        "inverse_metric_norm_full": str(sp.factor(1 / (D + 4))),
        "branch_squared_amplitude_slope": str(sp.factor(-4 / cubic_force)),
        "branch_amplitude_prefactor_decimal": float(sp.sqrt(-4 / cubic_force)),
        "full_to_reduced_leading_speed_ratio": str(sp.factor(D / (D + 4))),
        "fourth_envelope_checked_by_direct_stationary_third_jet": True,
        "reflection_exchange_symmetry_checked": True,
        "kinetics": kinetics,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "scope": "Exact rational jets at one specified stationary point; no trajectory or parameter scan.",
    }
    (OUT / "critical-mode-checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
