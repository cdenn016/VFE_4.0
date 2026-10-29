r"""D25-CRIT check 1: exact recomputation of it2:coupled-threshold (chapters/18_link_dynamics.tex:445-517).

Checks, all in exact rational arithmetic unless labeled float:
  (1) stationarity of (x_*, b = 1/2) for the full potential and of x_* for the reduced potential;
  (2) g = grad(C1 - C2)(x_*) against it2:threshold-g;
  (3) the one-edge Hessian contribution and the sum-of-squares certificate it2:threshold-positive;
  (4) H^{-1} g and g^T H^{-1} g against it2:threshold-inverse, T_c = g^T H^{-1} g / 4;
  (5) the reduced Hessian H_T = H - g g^T/(4T), by direct differentiation of the log-sum-exp potential;
  (6) the full 11x11 Hessian and its completion of squares it2:threshold-full;
  (7) the mean-only threshold 31/122, the symmetric minimizer (15,-6,-10)t/31 and value 61 t^2/62;
  (8) inertia of H_T on both sides of T_c (exact determinant sign plus leading minors), float Jacobian spectra.
Output: d25_threshold_exact.json in this directory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from d25_model import (A_FUN, C1, C2, DELTA, ECAL, E_RED, E_RED_LSE, F0, MOB, POINT, R, T, T_C_CLAIM,
                       T_MEAN_CLAIM, X, XS, XSTAR, at_point, grad, s)

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
res: dict = {}

# (1) stationarity
gx_full = at_point(grad(ECAL, XS), {s: 0}).applyfunc(sp.simplify)
gs_full = sp.simplify(at_point(sp.diff(ECAL, s), {s: 0}))
gx_red = at_point(grad(E_RED, XS)).applyfunc(sp.simplify)
res["full_matter_gradient_at_xstar"] = [str(z) for z in gx_full]
res["full_row_gradient_at_xstar"] = str(gs_full)
res["reduced_gradient_at_xstar"] = [str(z) for z in gx_red]
res["costs_at_xstar"] = [str(sp.simplify(at_point(C1))), str(sp.simplify(at_point(C2)))]
assert gx_full == sp.zeros(10, 1) and gs_full == 0 and gx_red == sp.zeros(10, 1)

# rewriting check of E_RED vs log-sum-exp at a random interior point (float)
rng = np.random.default_rng(0)
pt = {z: (rng.normal() if i < 5 else 0.5 + rng.random()) for i, z in enumerate(XS)}
pt[T] = 0.37
res["lse_vs_logcosh_rewrite_abs_diff"] = float(abs((E_RED_LSE - E_RED).subs(pt).evalf(30)))

# (2) g and H
H = at_point(sp.hessian(A_FUN, XS)).applyfunc(sp.nsimplify)
g = at_point(grad(DELTA, XS)).applyfunc(sp.nsimplify)
g_claim = sp.Matrix([1, R(-1, 2), R(-1, 2), R(-1, 2), R(-1, 2), 0, R(-1, 4), R(1, 4), R(1, 4), R(-1, 4)])
res["g"] = [str(z) for z in g]
res["g_matches_claim"] = bool(g == g_claim)

# (3) one-edge contribution and SOS
u0, u1, u2, b1, b2, h0, h1, h2, k1, k2 = sp.symbols("u0 u1 u2 b1 b2 h0 h1 h2 k1 k2", real=True)
xi = sp.Matrix([u0, u1, u2, b1, b2, h0, h1, h2, k1, k2])
z1, z2 = u0 - u1 - b1, u0 - u2 - b2
HC1 = at_point(sp.hessian(C1, XS)).applyfunc(sp.nsimplify)
edge1 = sp.expand((xi.T * HC1 * xi)[0] / 2)
edge1_claim = sp.expand(z1 ** 2 / 4 - z1 * h1 / 4 + h0 ** 2 / 16 - h0 * h1 / 8 + 3 * h1 ** 2 / 16 - h1 * k1 / 8)
res["one_edge_contribution_residual"] = str(sp.expand(edge1 - edge1_claim))
sos = u0 ** 2 + R(5, 4) * (u1 ** 2 + u2 ** 2) + R(3, 4) * (b1 ** 2 + b2 ** 2) + h0 ** 2 / 2
for zj, hj, kj, eps in ((z1, h1, k1, 1), (z2, h2, k2, -1)):
    sos += (zj - eps * hj / 2) ** 2 / 4 + (h0 - hj) ** 2 / 16 + R(71, 128) * hj ** 2 + (kj - hj / 8) ** 2 / 2
res["sos_residual"] = str(sp.expand((xi.T * H * xi)[0] - sos))
minors = [H[:n, :n].det() for n in range(1, 11)]
res["leading_minors_H"] = [str(z) for z in minors]
assert all(z > 0 for z in minors)

# (4) inverse action and T_c
y = H.LUsolve(g)
y_claim = sp.Matrix([2400, -960, -960, -1600, -1600, 0, -716, 716, 2440, -2440]) / 5059
res["Hinv_g"] = [str(z) for z in y]
res["Hinv_g_matches_claim"] = bool(y == y_claim)
gHg = (g.T * y)[0]
res["gT_Hinv_g"] = str(gHg)
Tc = sp.nsimplify(gHg / 4)
res["T_c"] = str(Tc)
res["T_c_matches_claim"] = bool(Tc == T_C_CLAIM)
res["det_H"] = str(H.det())

# (5) reduced Hessian identity by direct differentiation of the log-sum-exp form
H_red_direct = at_point(sp.hessian(E_RED_LSE, XS)).applyfunc(sp.simplify)
diff_red = (H_red_direct - (H - g * g.T / (4 * T))).applyfunc(sp.simplify)
res["reduced_hessian_identity_residual_zero"] = bool(diff_red == sp.zeros(10, 10))

# (6) full Hessian (coordinates x, s; s = b - 1/2 so d/ds = d/db)
XF = XS + [s]
H_full = at_point(sp.hessian(ECAL, XF), {s: 0}).applyfunc(sp.simplify)
res["full_hessian_blocks_ok"] = bool(H_full[:10, :10] == H and H_full[:10, 10] == g and sp.simplify(H_full[10, 10] - 4 * T) == 0)
db = sp.Symbol("db")
xiF = sp.Matrix(list(xi) + [db])
lhs = sp.expand((xiF.T * H_full * xiF)[0])
rhs = sp.expand((xi.T * (H - g * g.T / (4 * T)) * xi)[0] + 4 * T * (db + (g.T * xi)[0] / (4 * T)) ** 2)
res["full_completion_of_squares_residual"] = str(sp.simplify(lhs - rhs))

# (7) mean-only threshold
Hm, gm = H[:5, :5], g[:5, :]
ym = Hm.LUsolve(gm)
gHg_m = (gm.T * ym)[0]
res["mean_only_gT_Hinv_g"] = str(gHg_m)
Tm = gHg_m / 4
res["T_mean"] = str(Tm)
res["T_mean_matches_claim"] = bool(Tm == T_MEAN_CLAIM)
tt = sp.Symbol("t")
xi_min = tt * ym / gHg_m
res["mean_minimizer_over_t"] = [str(sp.simplify(z / tt)) for z in xi_min]
res["mean_minimizer_matches_(15,-6,-6,-10,-10)/31"] = bool(
    (xi_min / tt).applyfunc(sp.simplify) == sp.Matrix([15, -6, -6, -10, -10]) / 31)
res["mean_min_value_over_t2"] = str(sp.simplify((xi_min.T * Hm * xi_min)[0] / tt ** 2))
res["ordering_31/122<3/10<Tc"] = bool(Tm < R(3, 10) < Tc)

# (8) inertia and float Jacobian spectra
def det_sign(Tv):
    return sp.sign(H.det() * (1 - gHg / (4 * Tv)))  # matrix determinant lemma

res["det_sign_H_T"] = {str(Tv): int(det_sign(Tv)) for Tv in (R(1, 4), R(3, 10), R(1, 2))}
Mstar = np.diag([1.0] * 5 + [2.0] * 5)
Mfull = np.diag([1.0] * 5 + [2.0] * 5 + [0.25])
spectra = {}
for Tv in (R(1, 4), R(3, 10), Tc, R(1, 2)):
    HT = np.array((H - g * g.T / (4 * Tv)).evalf(), dtype=float)
    HF = np.array(H_full.subs(T, Tv).evalf(), dtype=float)
    ev_red = np.linalg.eigvals(-Mstar @ HT)
    ev_full = np.linalg.eigvals(-Mfull @ HF)
    spectra[str(Tv)] = {
        "reduced_max_imag": float(np.max(np.abs(ev_red.imag))),
        "reduced_sorted_real": sorted(float(z) for z in ev_red.real),
        "full_max_imag": float(np.max(np.abs(ev_full.imag))),
        "full_sorted_real": sorted(float(z) for z in ev_full.real),
        "hessian_eigs_HT": sorted(float(z) for z in np.linalg.eigvalsh(HT)),
    }
res["float_spectra"] = spectra

# frozen-variance mean Jacobian at T = 3/10 (mobility I at v = 1)
HTm = np.array((Hm - gm * gm.T / (4 * R(3, 10))).evalf(), dtype=float)
res["frozen_mean_hessian_eigs_T=3/10"] = sorted(float(z) for z in np.linalg.eigvalsh(HTm))

# D = p^T M_*^{-1} p
p = -y
Minv = sp.diag(*([1] * 5 + [R(1, 2)] * 5))
D = (p.T * Minv * p)[0]
res["D"] = str(D)
res["D_matches_claim"] = bool(D == R(19189456, 25593481))
res["D_over_D_plus_4"] = str(D / (D + 4))
res["(D+4)/D_float"] = float((D + 4) / D)
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_threshold_exact.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("float_spectra", "leading_minors_H")}, indent=1))
print("float spectra:", json.dumps(spectra, indent=1))
