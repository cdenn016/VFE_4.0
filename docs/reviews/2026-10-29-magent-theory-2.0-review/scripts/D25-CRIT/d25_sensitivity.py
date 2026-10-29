r"""D25-CRIT check 9 (numerical observation, float64): robustness of the subcritical sign of
it2:critical-pitchfork under perturbations of the symmetric data of it2:coupled-threshold.

For symmetric data (P_1 = P_2 = P_src, m_2 = -m_1 = m_src, equal dictionary precisions A_w,
receiver precision P_0, m_0 = 0), the symmetric state is located by Newton on the full matter
gradient at b = 1/2.  Then T_c = g^T H^{-1} g / 4 and, on the stationary curve X(s),
  Phi_s(s, T_c) = Delta(X(s)) + T_c h'(s) = c s^3 + O(s^5),
so c is estimated by Richardson extrapolation of Phi_s / s^3 at s = 0.01, 0.02.
The published data are reproduced first as a control (exact c = -0.1056550934864).
Output: d25_sensitivity.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
mu0, mu1, mu2, a1, a2 = sp.symbols("mu0 mu1 mu2 a1 a2", real=True)
v0, v1, v2, V1, V2 = sp.symbols("v0 v1 v2 V1 V2", positive=True)
XS = [mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2]
s = sp.Symbol("s", real=True)
P0, Ps, ms, Aw = sp.symbols("P0 Ps ms Aw", positive=True)


def kl(mu, v, mean, prec):
    return (prec * v + prec * (mu - mean) ** 2 - 1 - sp.log(prec * v)) / 2


def cost(mi, vi, mj, vj, a, V):
    Si, Sj, d = vi + 1, vj + 1, mi - mj - a
    return ((Si + d ** 2 + V) / Sj - 1 + sp.log(Sj) - sp.log(Si)) / 2


F0 = kl(mu0, v0, 0, P0) + kl(mu1, v1, -ms, Ps) + kl(mu2, v2, ms, Ps) + kl(a1, V1, 0, Aw) + kl(a2, V2, 0, Aw)
C1, C2 = cost(mu0, v0, mu1, v1, a1, V1), cost(mu0, v0, mu2, v2, a2, V2)
A = F0 + (C1 + C2) / 2
Dl = C1 - C2
GX = sp.Matrix([sp.diff(A + s * Dl, z) for z in XS])
GXX = GX.jacobian(XS)
args = [XS, s, P0, Ps, ms, Aw]
f_gx = sp.lambdify(args, GX, "numpy")
f_gxx = sp.lambdify(args, GXX, "numpy")
f_dl = sp.lambdify([XS, P0, Ps, ms, Aw], Dl, "numpy")
f_g = sp.lambdify([XS, P0, Ps, ms, Aw], sp.Matrix([sp.diff(Dl, z) for z in XS]), "numpy")
print(f"[{time.perf_counter()-t0:5.2f}s] lambdified", flush=True)


def newton(x, sv, prm):
    for _ in range(100):
        r = np.asarray(f_gx(x, sv, *prm), dtype=float).ravel()
        dx = np.linalg.solve(np.asarray(f_gxx(x, sv, *prm), dtype=float), -r)
        x = x + dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    return x, float(np.linalg.norm(np.asarray(f_gx(x, sv, *prm), dtype=float)))


def analyse(prm, x_init):
    x, r0 = newton(x_init.copy(), 0.0, prm)
    if r0 > 1e-10 or np.min(x[5:]) <= 0:
        return None
    H = np.asarray(f_gxx(x, 0.0, *prm), dtype=float)
    if np.linalg.eigvalsh(H).min() <= 0:
        return {"H_positive": False}
    g = np.asarray(f_g(x, *prm), dtype=float).ravel()
    Tc = float(g @ np.linalg.solve(H, g) / 4)
    est = []
    for sv in (0.01, 0.02):
        xs_, xx = np.linspace(0, sv, 21)[1:], x.copy()
        for z in xs_:
            xx, _ = newton(xx, z, prm)
        phis = f_dl(xx, *prm) + Tc * np.log((1 + 2 * sv) / (1 - 2 * sv))
        est.append(phis / sv ** 3)
    c = (4 * est[0] - est[1]) / 3  # Richardson (error O(s^2))
    return {"x_sym": [float(z) for z in x], "T_c": Tc, "c": float(c), "c_raw_s=0.01": float(est[0]),
            "H_min_eig": float(np.linalg.eigvalsh(H).min())}


x_pub = np.array([0, -4 / 3, 4 / 3, 1 / 3, -1 / 3, 1, 1, 1, 1, 1], dtype=float)
base = (1.0, 1.25, 23 / 15, 0.75)
res = {"published": analyse(base, x_pub)}
print(f"[{time.perf_counter()-t0:5.2f}s] published: {res['published']['c']:.10f} T_c {res['published']['T_c']:.10f}", flush=True)
scan = []
names = ["P0", "P_src", "m_src", "A_omega"]
grids = [[0.5, 0.75, 1.0, 1.5, 2.0, 4.0], [0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
         [0.75, 1.0, 1.25, 23 / 15, 2.0, 3.0], [0.25, 0.5, 0.75, 1.0, 1.5, 3.0]]
for k, grid in enumerate(grids):
    xprev = x_pub.copy()
    for val in grid:
        prm = list(base)
        prm[k] = val
        out = analyse(tuple(prm), xprev)
        if out and "c" in out:
            xprev = np.array(out["x_sym"])
            scan.append({"param": names[k], "value": val, "T_c": out["T_c"], "c": out["c"],
                         "type": "subcritical" if out["c"] < 0 else "supercritical"})
        else:
            scan.append({"param": names[k], "value": val, "result": out})
    print(f"[{time.perf_counter()-t0:5.2f}s] scanned {names[k]}", flush=True)
res["scan"] = scan
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_sensitivity.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
for row in scan:
    print(row)
