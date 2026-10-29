r"""D25-CRIT check 7 (numerical observation): short integrations of both Fisher flows of the
three-agent example (it2:coupled-threshold), to test the destinations suggested by
d25_branches_numeric.json.  scipy solve_ivp (BDF, analytic Jacobian), single thread.
  (a) T = 0.30 < T_c, start x_* + 1e-3 p (full flow also s = 1e-3): where do trajectories go?
  (b) T = 0.3234 in (T_c, T_fold): start near x_* -> back to x_*?  start at the stable branch -> stays?
  (c) T = T_c: center speed ratio. Starting on the leading center-manifold approximation, compare
      ds/dt / s^3 with -c/d for both flows (d = D or D + 4).
Output: d25_trajectories.json.  Finite runs; not a proof of global attraction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from d25_model import DELTA, ECAL, E_RED, MOB, T, T_C_CLAIM, XS, XSTAR, grad, s

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
A = json.loads((OUT / "d25_pitchfork_exact.json").read_text(encoding="utf-8"))
Bn = json.loads((OUT / "d25_branches_numeric.json").read_text(encoding="utf-8"))
p = np.array([float(sp.Rational(z)) for z in A["p"]])
X2 = np.array([float(sp.Rational(z)) for z in A["X2"]])
c = float(sp.Rational(A["c"]))
D = float(sp.Rational(A["D"]))
Tc = float(T_C_CLAIM)
XF = XS + [s]
F_full = (-MOB * grad(ECAL, XS)).col_join(sp.Matrix([-(sp.Rational(1, 4) - s ** 2) * sp.diff(ECAL, s)]))
F_red = -MOB * grad(E_RED, XS)
ff = sp.lambdify([XF, T], F_full, "numpy")
fJ = sp.lambdify([XF, T], F_full.jacobian(XF), "numpy")
fr = sp.lambdify([XS, T], F_red, "numpy")
rJ = sp.lambdify([XS, T], F_red.jacobian(XS), "numpy")
sstar = sp.lambdify([XS, T], (-sp.tanh(DELTA / (2 * T)) / 2), "numpy")
xstar = np.array([float(z) for z in XSTAR])
print(f"[{time.perf_counter()-t0:5.2f}s] lambdified", flush=True)


def run(kind: str, y0: np.ndarray, Tv: float, tmax: float):
    if kind == "full":
        fun = lambda t, y: np.asarray(ff(y, Tv), dtype=float).ravel()
        jac = lambda t, y: np.asarray(fJ(y, Tv), dtype=float)
    else:
        fun = lambda t, y: np.asarray(fr(y, Tv), dtype=float).ravel()
        jac = lambda t, y: np.asarray(rJ(y, Tv), dtype=float)
    return solve_ivp(fun, (0, tmax), y0, method="BDF", jac=jac, rtol=1e-10, atol=1e-12, dense_output=False)


res = {}
# (a) below threshold
for kind in ("reduced", "full"):
    y0 = xstar + 1e-3 * p if kind == "reduced" else np.concatenate([xstar + 1e-3 * p, [1e-3]])
    sol = run(kind, y0, 0.30, 4000.0)
    yend = sol.y[:, -1]
    s_end = yend[10] if kind == "full" else float(sstar(yend[:10], 0.30))
    speed = np.linalg.norm(np.asarray(ff(yend, 0.30) if kind == "full" else fr(yend, 0.30), dtype=float))
    res[f"(a)_{kind}_T=0.30"] = {"status": sol.status, "t_end": float(sol.t[-1]), "s_end": float(s_end),
                                 "x_end": [float(z) for z in yend[:10]], "speed_end": float(speed)}
    print(f"[{time.perf_counter()-t0:5.2f}s] (a) {kind}: s_end={s_end:.6f} speed={speed:.2e}", flush=True)
br = Bn.get("stable_branch_point_near_T=0.30000")
if br:
    res["(a)_branch_prediction"] = {"s": br["s"], "T": br["T"]}
# (b) bistability window
Tw = 0.3234
assert Tc < Tw < Bn["fold_T"]
for label, y0x in (("near_symmetric", xstar + 1e-2 * p),):
    for kind in ("reduced", "full"):
        y0 = y0x if kind == "reduced" else np.concatenate([y0x, [1e-2]])
        sol = run(kind, y0, Tw, 20000.0)
        yend = sol.y[:, -1]
        s_end = yend[10] if kind == "full" else float(sstar(yend[:10], Tw))
        res[f"(b)_{kind}_{label}_T={Tw}"] = {"status": sol.status, "s_end": float(s_end),
                                            "dist_to_xstar": float(np.linalg.norm(yend[:10] - xstar))}
        print(f"[{time.perf_counter()-t0:5.2f}s] (b) {kind} {label}: s_end={s_end:.6f}", flush=True)
# start from the T=0.30 asymmetric endpoint and raise T to Tw
ya = np.array(res["(a)_reduced_T=0.30"]["x_end"])
for kind in ("reduced", "full"):
    y0 = ya if kind == "reduced" else np.concatenate([ya, [float(sstar(ya, Tw))]])
    sol = run(kind, y0, Tw, 20000.0)
    yend = sol.y[:, -1]
    s_end = yend[10] if kind == "full" else float(sstar(yend[:10], Tw))
    res[f"(b)_{kind}_from_asymmetric_T={Tw}"] = {"status": sol.status, "s_end": float(s_end),
                                                "dist_to_xstar": float(np.linalg.norm(yend[:10] - xstar))}
    print(f"[{time.perf_counter()-t0:5.2f}s] (b) {kind} from asymmetric: s_end={s_end:.6f}", flush=True)
# (c) center speed at T_c: start on X(s0) + O(s^3) graph, s0 = 0.03; measure ds/dt / s^3 after transients
for kind, d in (("reduced", D), ("full", D + 4)):
    s0 = 0.03
    x0 = xstar + p * s0 + 0.5 * X2 * s0 ** 2
    y0 = x0 if kind == "reduced" else np.concatenate([x0, [s0]])
    fun = (lambda t, y: np.asarray(fr(y, Tc), dtype=float).ravel()) if kind == "reduced" else \
          (lambda t, y: np.asarray(ff(y, Tc), dtype=float).ravel())
    sol = solve_ivp(fun, (0, 400.0), y0, method="BDF", rtol=1e-11, atol=1e-13,
                    jac=(lambda t, y: np.asarray(rJ(y, Tc) if kind == "reduced" else fJ(y, Tc), dtype=float)),
                    t_eval=np.linspace(0, 400.0, 401))
    S = np.array([float(sstar(y[:10], Tc)) if kind == "reduced" else y[10] for y in sol.y.T])
    dS = np.gradient(S, sol.t)
    ratios = dS[50:] / S[50:] ** 3
    res[f"(c)_{kind}_center_speed"] = {"s_start": float(S[0]), "s_at_t=400": float(S[-1]),
                                       "median_dsdt_over_s3_after_t=50": float(np.median(ratios)),
                                       "predicted_-c/d": float(-c / d)}
    print(f"[{time.perf_counter()-t0:5.2f}s] (c) {kind}: ratio {np.median(ratios):.5f} vs {-c/d:.5f}", flush=True)
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_trajectories.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
