"""Small exact algebra checks for the scoped causal-memory derivations.

These checks establish the displayed algebraic identities, not the universal
memory claims, history sufficiency, or a model simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def run_checks() -> dict[str, bool]:
    """Check rotation, jet, score and angular-velocity identities exactly."""
    theta = sp.symbols("theta", real=True)
    omega = sp.symbols("omega", real=True)
    r = sp.Matrix(sp.symbols("r0:2", real=True))
    v = sp.Matrix(sp.symbols("v0:2", real=True))
    j2 = sp.Matrix([[0, -1], [1, 0]])
    rot = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                     [sp.sin(theta), sp.cos(theta)]])
    zero = sp.zeros(2, 1)
    checks: dict[str, bool] = {}
    checks["native_field_cancellation"] = sp.simplify(
        rot * (v - omega * j2 * r) + omega * j2 * rot * r - rot * v
    ) == zero

    c, t = sp.symbols("c t", real=True)
    th = sp.Function("theta")(c, t)
    rr = sp.Matrix([sp.Function("r0")(c, t), sp.Function("r1")(c, t)])
    rt = rot.subs(theta, th)
    zz = rt * rr
    first = rt * (rr.diff(c) + th.diff(c) * j2 * rr)
    second = rt * (rr.diff(c, 2) + 2 * th.diff(c) * j2 * rr.diff(c)
                   + th.diff(c, 2) * j2 * rr - th.diff(c) ** 2 * rr)
    checks["contextual_first_jet"] = sp.simplify(zz.diff(c) - first) == zero
    checks["contextual_second_jet"] = sp.simplify(zz.diff(c, 2) - second) == zero

    alpha = sp.Function("alpha")(c, t)
    checks["spatial_temporal_compatibility"] = sp.simplify(
        sp.diff(sp.diff(alpha, c), t) - sp.diff(sp.diff(alpha, t), c)
    ) == 0
    psi, alpha0, b = sp.symbols("psi alpha0 b", real=True)
    alpha_fn = sp.Function("alpha0")(c)
    b_fn = sp.Function("b")(c)
    fine_log = b_fn * sp.cos(psi)
    transformed_log = b_fn * sp.cos(psi + alpha_fn)
    fine_context_at_image = fine_log.diff(c).subs(psi, psi + alpha_fn)
    ell_phys_at_image = b_fn * sp.sin(psi + alpha_fn)
    checks["initial_phase_score_defect_sign"] = sp.simplify(
        transformed_log.diff(c) - fine_context_at_image
        + alpha_fn.diff(c) * ell_phys_at_image
    ) == 0

    lam, h0, cross, inertia = sp.symbols("lam h0 cross inertia", real=True)
    poly = h0 + 2 * lam * cross + lam ** 2 * inertia
    checks["physical_score_positive_quadratic_coefficient"] = (
        sp.expand(poly).coeff(lam, 2) == inertia
    )
    ss, kk, ee, uu = sp.symbols("s k E u", positive=True)
    dd = ss + kk * ee + kk * uu
    om = ((4 * kk * uu - 2 * kk * ee) / dd - 2 * kk / (ss + 2 * kk)) / 5
    checks["native_angular_velocity_depends_on_public_weight"] = sp.simplify(
        sp.diff(om, uu) - (4 * kk * ss + 6 * kk ** 2 * ee) / (5 * dd ** 2)
    ) == 0
    return checks


if __name__ == "__main__":
    result = {"sympy_version": sp.__version__, "checks": run_checks()}
    result["passed"] = all(result["checks"].values())
    Path(__file__).with_name("symbolic-results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
