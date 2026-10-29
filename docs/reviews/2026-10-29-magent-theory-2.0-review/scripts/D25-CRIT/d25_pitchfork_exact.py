r"""D25-CRIT check 2 (Part A only): exact stationary elimination for it2:critical-pitchfork
(chapters/18_link_dynamics.tex:519-623). Parts B and C are in d25_cm_flow.py.

Part A (stationary elimination): solve G_x(X(s), s) = 0 order by order by exact series, obtain
X'(0) = p, X''(0), X'''(0); compute Phi_ssss(0, T_c) two ways:
  (i) the manuscript's formula A4[p^4] + 4 Delta3[p^3] - 3 k^T H^{-1} k + 32 T_c;
  (ii) direct series of Phi(s) = G(X(s), s) + T_c h(s) to order s^4;
and the straight-line value A4 + 4 Delta3 + 32 T_c (no stable response).

Part B (dynamics, in d25_cm_flow.py): for the actual vector fields
  full:     xdot = -M(x) grad_x E(x,s),   sdot = -(1/4 - s^2) dE/ds,
  reduced:  xdot = -M(x) grad E_red(x),   E_red = A - T log cosh(Delta/(2T)),
  pullback: xdot = -g_pb(x)^{-1} grad E_red(x),  g_pb = M^{-1} + grad b* grad b*^T/(b*(1-b*)),
compute the standard center-manifold projection coefficients at (x_*, T_c):
  u' = a1 (T - T_c) u + a3 u^3 + ...,   a1 = l^T (dJ/dT) q,
  a3 = (1/6) l^T D^3F[q,q,q] + (1/2) l^T D^2F[q,h2],  J h2 = -D^2F[q,q],  l^T h2 = 0,
which include every derivative of the mobility. Compare with -4/d and -c/d, d = D or D + 4.
Part C (in d25_cm_flow.py): third-order center-manifold graph versus X(s), checking Y(s,0) - X(s) = (c/d) B s^3.
Output: d25_pitchfork_exact.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sympy as sp

from d25_model import (A_FUN, DELTA, ECAL, E_RED, H_ENT, MOB, POINT, R, T, T_C_CLAIM, XS, XSTAR,
                       at_point, grad, s)

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
res: dict = {}
eps = sp.Symbol("eps")
Tc = T_C_CLAIM
c_claim = R(-657534393590529472, 6223404588394626861)
D_claim = R(19189456, 25593481)


def series_coeffs(expr: sp.Expr, var: sp.Symbol, order: int) -> list:
    """Exact Taylor coefficients of expr in var up to var**order."""
    ser = sp.series(expr, var, 0, order + 1).removeO()
    poly = sp.Poly(sp.expand(ser), var)
    return [sp.nsimplify(poly.coeff_monomial(var ** k)) for k in range(order + 1)]


def tlog(msg: str) -> None:
    print(f"[{time.perf_counter() - t0:6.2f}s] {msg}", flush=True)


# ---------- Part A: stationary elimination ----------
GX = grad(A_FUN, XS) + s * grad(DELTA, XS)          # G_x(x, s)
H = at_point(sp.hessian(A_FUN, XS)).applyfunc(sp.nsimplify)
g = at_point(grad(DELTA, XS)).applyfunc(sp.nsimplify)
cs = []                                               # coefficients c_k of X(s) = x_* + sum c_k s^k
for n in (1, 2, 3):
    curve = {z: XSTAR[i] + sum(cs[k][i] * s ** (k + 1) for k in range(len(cs))) for i, z in enumerate(XS)}
    r_n = sp.Matrix([series_coeffs(GX[i].subs(curve), s, n)[n] for i in range(10)])
    cs.append(-H.LUsolve(r_n))
    tlog(f"stationary curve order {n} solved")
p, X2, X3 = cs[0], 2 * cs[1], 6 * cs[2]               # X'(0), X''(0), X'''(0)
res["p"] = [str(z) for z in p]
res["p_equals_minus_Hinv_g"] = bool(p == -H.LUsolve(g))
res["X2"] = [str(z) for z in X2]
res["X3"] = [str(z) for z in X3]

# formula ingredients via directional derivatives along p
line = {z: XSTAR[i] + eps * p[i] for i, z in enumerate(XS)}
A4 = series_coeffs(A_FUN.subs(line), eps, 4)[4] * 24
D3 = series_coeffs(DELTA.subs(line), eps, 3)[3] * 6
k = sp.Matrix([series_coeffs(grad(A_FUN, XS)[i].subs(line), eps, 2)[2] * 2
               + 2 * series_coeffs(grad(DELTA, XS)[i].subs(line), eps, 1)[1] for i in range(10)])
kHk3 = 3 * (k.T * H.LUsolve(k))[0]
res["A4"] = str(A4)
res["Delta3"] = str(D3)
res["3kHk"] = str(kHk3)
res["A4_claim_match"] = bool(A4 == R(236914859731152, 655026269697361))
res["Delta3_claim_match"] = bool(D3 == R(-31633266640, 129477420379))
res["3kHk_claim_match"] = bool(kHk3 == R(21485600170644766720, 2074468196131542287))
res["X2_equals_minus_Hinv_k"] = bool(X2 == -H.LUsolve(k))
phi4_formula = A4 + 4 * D3 - kHk3 + 32 * Tc
tlog("formula ingredients done")

# direct series of Phi(s) = G(X(s), s) + T_c h(s)
curve3 = {z: XSTAR[i] + p[i] * s + X2[i] * s ** 2 / 2 + X3[i] * s ** 3 / 6 for i, z in enumerate(XS)}
Gs = (A_FUN + s * DELTA).subs(curve3) + Tc * H_ENT
phi_coeffs = series_coeffs(Gs, s, 4)
phi4_direct = phi_coeffs[4] * 24
curve2 = {z: XSTAR[i] + p[i] * s + X2[i] * s ** 2 / 2 for i, z in enumerate(XS)}
phi4_direct_X2only = series_coeffs((A_FUN + s * DELTA).subs(curve2) + Tc * H_ENT, s, 4)[4] * 24
res["Phi_coeffs_s0..s4_at_Tc"] = [str(z) for z in phi_coeffs]
res["Phi_ssss_formula"] = str(phi4_formula)
res["Phi_ssss_direct"] = str(phi4_direct)
res["Phi_ssss_direct_with_X_truncated_at_s2"] = str(phi4_direct_X2only)
res["formula_equals_direct"] = bool(phi4_formula == phi4_direct == phi4_direct_X2only)
c = phi4_direct / 6
res["c"] = str(c)
res["c_float"] = float(c)
res["c_matches_claim"] = bool(c == c_claim)
straight = A4 + 4 * D3 + 32 * Tc
res["straight_line_Phi_ssss"] = str(straight)
res["straight_line_c_float"] = float(straight / 6)
Minv = sp.diag(*([1] * 5 + [R(1, 2)] * 5))
Mstar = Minv.inv()
D = (p.T * Minv * p)[0]
res["D"] = str(D)
res["D_matches_claim"] = bool(D == D_claim)
res["branch_prefactor_sqrt(-4/c)"] = float(sp.sqrt(-4 / c))
tlog(f"Part A done: c = {float(c):.12f}, straight-line c = {float(straight/6):.6f}")


res["H"] = [[str(z) for z in H.row(i)] for i in range(10)]
res["g"] = [str(z) for z in g]
res["k"] = [str(z) for z in k]
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_pitchfork_exact.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps({kk: v for kk, v in res.items() if kk not in ("X2", "X3", "H", "k", "p", "g")}, indent=1))
