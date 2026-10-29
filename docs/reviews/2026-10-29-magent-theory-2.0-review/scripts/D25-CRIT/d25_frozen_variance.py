r"""D25-CRIT check 6 (extension): bifurcation type of the frozen-variance mean system of
it2:coupled-threshold (all five variances held at one, T_mean = 31/122).

With variances fixed, A and Delta are quadratic in the five means, so G_x is affine in x and the
stationary curve is exact: X(s) - x_* = -s (H_m + s Delta_xx)^{-1} g_m, and
    Phi(s) = A(x_*) - (s^2/2) g_m^T (H_m + s Delta_xx)^{-1} g_m + T h(s)   (exact).
Phi_ssss(0, T_mean) = -12 p^T Delta_xx H_m^{-1} Delta_xx p + 32 T_mean, p = -H_m^{-1} g_m.
Also the exact branch temperature T_b(s) = -Delta(X(s))/h'(s) and its fold, if any.
Output: d25_frozen_variance.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from d25_model import A_FUN, DELTA, H_ENT, POINT, R, XS, XSTAR, grad, s

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
mu_vars = XS[:5]
fix = {z: 1 for z in XS[5:]}
A_m = sp.expand(A_FUN.subs(fix))
D_m = sp.expand(DELTA.subs(fix))
pt = {z: POINT[z] for z in mu_vars}
Hm = sp.hessian(A_m, mu_vars).subs(pt)
gm = grad(D_m, mu_vars).subs(pt)
Dxx = sp.hessian(D_m, mu_vars).subs(pt)
res = {"A_mean_is_quadratic": bool(sp.Poly(A_m - sp.log(sp.Integer(1)), *mu_vars).total_degree() <= 2),
       "Delta_mean_is_quadratic": bool(sp.Poly(D_m, *mu_vars).total_degree() <= 2)}
Tm = (gm.T * Hm.LUsolve(gm))[0] / 4
res["T_mean"] = str(Tm)
p = -Hm.LUsolve(gm)
quart = -12 * (p.T * Dxx * Hm.LUsolve(Dxx * p))[0] + 32 * Tm
res["Phi_ssss_mean"] = str(sp.nsimplify(quart))
res["c_mean"] = str(sp.nsimplify(quart / 6))
res["c_mean_float"] = float(quart / 6)
res["type"] = "supercritical (stable small branches below T_mean)" if quart > 0 else "subcritical"
# exact Phi(s) check by series
Phi = A_m.subs(pt) - s ** 2 / 2 * (gm.T * (Hm + s * Dxx).inv() * gm)[0] + Tm * H_ENT
ser = sp.series(Phi, s, 0, 6).removeO()
res["Phi_series_s4_coeff_times_24"] = str(sp.nsimplify(sp.expand(ser).coeff(s, 4) * 24))
# branch temperature and fold (numeric)
Hn, gn, Dn = (np.array(M.evalf(), dtype=float) for M in (Hm, gm, Dxx))
gn = gn.ravel()
f_delta = sp.lambdify([mu_vars], D_m, "numpy")
xs0 = np.array([float(POINT[z]) for z in mu_vars])
svals = np.linspace(1e-4, 0.4999, 4000)
Tb = []
for sv in svals:
    x = xs0 - sv * np.linalg.solve(Hn + sv * Dn, gn)
    Tb.append(-f_delta(x) / np.log((1 + 2 * sv) / (1 - 2 * sv)))
Tb = np.array(Tb)
res["Tb_initial_trend"] = "decreasing" if Tb[10] < Tb[0] else "increasing"
res["Tb_max"] = float(Tb.max())
res["Tb_monotone_decreasing"] = bool(np.all(np.diff(Tb) < 0))
res["T_mean_float"] = float(Tm)
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_frozen_variance.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
