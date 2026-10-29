r"""D25-CRIT check 5 (numerical observation, float64): continuation of the asymmetric branch of the
three-agent example beyond the local pitchfork (it2:open-reciprocal asks for distant branches).

Every full stationary point with matter coordinates on the continued curve X(s) (G_x(X(s), s) = 0,
G = A + s Delta, temperature-independent) is stationary exactly at the temperature
    T_b(s) = -Delta(X(s)) / h'(s),  h'(s) = log((1+2s)/(1-2s)),
and, when K(s) = G_xx(X(s), s) > 0, its full Hessian inertia is that of K plus sign(Phi_ss), with
Phi_ss = -h'(s) T_b'(s).  So the branch is a saddle where T_b increases and is a strict local
minimum (locally exponentially stable for both flows) where T_b decreases and K > 0.
Checks: small-s agreement with T_c - c s^2/4; Jacobian eigenvalues at nearby branch points against
8 delta/d; positivity of K and of the variances along the branch; location of the fold.
Output: d25_branches_numeric.json.  This is numerics on one example, not a theorem.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from d25_model import A_FUN, DELTA, ECAL, E_RED, MOB, T, T_C_CLAIM, XS, XSTAR, grad, s

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
A = json.loads((OUT / "d25_pitchfork_exact.json").read_text(encoding="utf-8"))
c = float(sp.Rational(A["c"]))
D = float(sp.Rational(A["D"]))
Tc = float(T_C_CLAIM)
GX = grad(A_FUN, XS) + s * grad(DELTA, XS)
GXX = GX.jacobian(XS)
f_gx = sp.lambdify([XS, s], GX, "numpy")
f_gxx = sp.lambdify([XS, s], GXX, "numpy")
f_delta = sp.lambdify([XS], DELTA, "numpy")
f_dx = sp.lambdify([XS], grad(DELTA, XS), "numpy")
XF = XS + [s]
F_full = (-MOB * grad(ECAL, XS)).col_join(sp.Matrix([-(sp.Rational(1, 4) - s ** 2) * sp.diff(ECAL, s)]))
f_Jfull = sp.lambdify([XF, T], F_full.jacobian(XF), "numpy")
F_red = -MOB * grad(E_RED, XS)
f_Jred = sp.lambdify([XS, T], F_red.jacobian(XS), "numpy")
print(f"[{time.perf_counter()-t0:5.2f}s] lambdified", flush=True)


def solve_X(svals: np.ndarray, x_init: np.ndarray):
    xs, ok = [], True
    x = x_init.copy()
    for sv in svals:
        for _ in range(50):
            r = np.asarray(f_gx(x, sv), dtype=float).ravel()
            Jm = np.asarray(f_gxx(x, sv), dtype=float)
            dx = np.linalg.solve(Jm, -r)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-14:
                break
        if np.min(x[5:]) <= 0 or np.linalg.norm(np.asarray(f_gx(x, sv), dtype=float)) > 1e-10:
            ok = False
            break
        xs.append(x.copy())
    return np.array(xs), ok


xstar = np.array([float(z) for z in XSTAR])
svals = np.linspace(1e-4, 0.4999, 5000)
Xs, ok = solve_X(svals, xstar)
svals = svals[: len(Xs)]
L = np.log((1 + 2 * svals) / (1 - 2 * svals))
Tb = -np.array([f_delta(x) for x in Xs]) / L
Kmin = np.array([np.linalg.eigvalsh(np.asarray(f_gxx(x, sv), dtype=float)).min() for x, sv in zip(Xs, svals)])
dTb = np.gradient(Tb, svals)
res = {"continuation_reached_s": float(svals[-1]), "continuation_ok": bool(ok),
       "min_variance_along_branch": float(Xs[:, 5:].min()),
       "min_eig_K_along_branch": float(Kmin.min())}
# small-s check: T_b(s) - T_c ~ -c s^2 / 4
for sv in (1e-3, 1e-2, 3e-2):
    i = int(np.argmin(np.abs(svals - sv)))
    res[f"Tb_minus_Tc_over_(-c s^2/4)_at_s={svals[i]:.4f}"] = float((Tb[i] - Tc) / (-c * svals[i] ** 2 / 4))
imax = int(np.argmax(Tb))
res["fold_s"] = float(svals[imax])
res["fold_T"] = float(Tb[imax])
res["T_c"] = Tc
res["Tb_at_s_end"] = float(Tb[-1])
res["Tb_crosses_Tc_at_s"] = None
below = np.where((Tb < Tc) & (svals > svals[imax]))[0]
if len(below):
    res["Tb_crosses_Tc_at_s"] = float(svals[below[0]])
res["Tb_monotone_decreasing_after_fold"] = bool(np.all(np.diff(Tb[imax:]) < 0))
res["stable_part_K_positive"] = bool(np.all(Kmin[imax:] > 0))
res["T_at_s=0.3,0.4,0.45,0.49"] = [float(np.interp(z, svals, Tb)) for z in (0.3, 0.4, 0.45, 0.49)]
# a stable asymmetric state at T = 3/10 (below T_c): solve T_b(s) = 0.3 on the decreasing part
for Ttarget in (0.30, Tc, 0.33):
    j = np.where((Tb[imax:-1] - Ttarget) * (Tb[imax + 1:] - Ttarget) <= 0)[0]
    if len(j):
        jj = imax + j[0]
        sv = svals[jj]
        x = Xs[jj]
        Jf = np.asarray(f_Jfull(np.concatenate([x, [sv]]), Tb[jj]), dtype=float)
        Jr = np.asarray(f_Jred(x, Tb[jj]), dtype=float)
        evf, evr = np.linalg.eigvals(Jf), np.linalg.eigvals(Jr)
        res[f"stable_branch_point_near_T={Ttarget:.5f}"] = {
            "s": float(sv), "T": float(Tb[jj]), "x": [float(z) for z in x],
            "full_max_real_eig": float(evf.real.max()), "reduced_max_real_eig": float(evr.real.max()),
            "max_imag": float(max(np.abs(evf.imag).max(), np.abs(evr.imag).max()))}
# nearby saddle check: delta small, s_+ ~ sqrt(-4 delta / c); unstable eigenvalue ~ 8 delta / d
near = []
for sv in (0.01, 0.02, 0.04):
    xs_, _ = solve_X(np.linspace(1e-4, sv, 200), xstar)
    x = xs_[-1]
    Tv = -f_delta(x) / np.log((1 + 2 * sv) / (1 - 2 * sv))
    delta = Tv - Tc
    Jf = np.asarray(f_Jfull(np.concatenate([x, [sv]]), Tv), dtype=float)
    Jr = np.asarray(f_Jred(x, Tv), dtype=float)
    evf, evr = np.linalg.eigvals(Jf), np.linalg.eigvals(Jr)
    near.append({"s": sv, "delta": float(delta), "s_pred": float(np.sqrt(-4 * delta / c)),
                 "full_unstable_eig": float(evf.real.max()), "pred_8delta/(D+4)": 8 * delta / (D + 4),
                 "reduced_unstable_eig": float(evr.real.max()), "pred_8delta/D": 8 * delta / D,
                 "n_positive_full": int(np.sum(evf.real > 0)), "n_positive_reduced": int(np.sum(evr.real > 0))})
res["nearby_saddles"] = near
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_branches_numeric.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
