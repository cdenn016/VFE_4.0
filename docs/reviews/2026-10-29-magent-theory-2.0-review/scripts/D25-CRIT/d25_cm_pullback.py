r"""D25-CRIT check 4 (extension test): Gibbs reduction with the pullback of the full metric to the
Gibbs graph x -> (x, b*(x)).  Hypothesis under test: its leading center coefficients equal those
of full row relaxation, a1 = -4/(D+4), a3 = -c/(D+4).

Pullback metric g_pb = M^{-1} + grad b* grad b*^T / (b*(1-b*)).  With sigma = sech^2(Delta/(2T)),
grad b* = -(sigma/(4T)) Delta_x and b*(1-b*) = sigma/4, Sherman-Morrison gives
    F_pb = -M grad E + sigma (M Delta_x)(Delta_x^T M grad E) / (4 T^2 + sigma Delta_x^T M Delta_x).
(The algebraic reduction is re-verified numerically at a random point below.)  Exact jets of
grad E, Delta_x and Delta along lines through x_* are computed by differentiation, and F_pb's jets
are assembled by exact truncated-polynomial arithmetic.  Same projection formula as d25_cm_flow.py.
Output: d25_cm_pullback.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from d25_model import DELTA, E_RED, MOB, R, T, T_C_CLAIM, XS, XSTAR, grad

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
eps = sp.Symbol("eps")
Tc = T_C_CLAIM
A = json.loads((OUT / "d25_pitchfork_exact.json").read_text(encoding="utf-8"))
p = sp.Matrix([sp.Rational(z) for z in A["p"]])
c = sp.Rational(A["c"])
D = sp.Rational(A["D"])
N = 3  # jet order


def tlog(msg: str) -> None:
    print(f"[{time.perf_counter() - t0:6.2f}s] {msg}", flush=True)


gradE = grad(E_RED, XS)
DX = grad(DELTA, XS)
mob = [MOB[i, i] for i in range(10)]

# numerical check of the Sherman-Morrison reduction at a random interior point
bstar = (1 - sp.tanh(DELTA / (2 * T))) / 2
gb = grad(bstar, XS)
rng = np.random.default_rng(1)
pt = {z: (0.3 * rng.normal() if i < 5 else 0.6 + 0.8 * rng.random()) for i, z in enumerate(XS)}
pt[T] = 0.41
Mn = np.diag([float(m.subs(pt)) for m in mob])
gbn = np.array([float(z.subs(pt)) for z in gb])
bn = float(bstar.subs(pt))
gEn = np.array([float(z.subs(pt)) for z in gradE])
DXn = np.array([float(z.subs(pt)) for z in DX])
Dn = float(DELTA.subs(pt))
gpb = np.linalg.inv(Mn) + np.outer(gbn, gbn) / (bn * (1 - bn))
F_direct = -np.linalg.solve(gpb, gEn)
sig = 1 / np.cosh(Dn / (2 * 0.41)) ** 2
F_sm = -Mn @ gEn + sig * (Mn @ DXn) * (DXn @ Mn @ gEn) / (4 * 0.41 ** 2 + sig * DXn @ Mn @ DXn)
res = {"sherman_morrison_reduction_max_abs_error": float(np.max(np.abs(F_direct - F_sm)))}
tlog(f"Sherman-Morrison check {res['sherman_morrison_reduction_max_abs_error']:.2e}")


def jet(expr: sp.Expr, sub: dict, order: int, Tval) -> list:
    e = expr.subs(T, Tval).subs(sub)
    out = []
    for k in range(order + 1):
        v = e.subs(eps, 0)
        if not v.is_Rational:
            v = sp.nsimplify(sp.simplify(v))
        out.append(v / sp.factorial(k))
        e = sp.diff(e, eps)
    return out


def pmul(a: list, b: list) -> list:
    return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(N + 1)]


def padd(*xs: list) -> list:
    return [sum(x[k] for x in xs) for k in range(N + 1)]


def pscale(a: list, sc) -> list:
    return [sc * z for z in a]


def pinv(a: list) -> list:
    assert a[0] != 0
    b = [1 / a[0]]
    for k in range(1, N + 1):
        b.append(-sum(a[i] * b[k - i] for i in range(1, k + 1)) / a[0])
    return b


def F_pb_jet(v: sp.Matrix, Tval, order: int = N) -> list:
    """Truncated Taylor polynomials (in eps) of the 10 components of F_pb along x_* + eps v."""
    sub = {z: XSTAR[i] + eps * v[i] for i, z in enumerate(XS)}
    gE = [jet(gradE[i], sub, order, Tval) + [0] * (N - order) for i in range(10)]
    dx = [jet(DX[i], sub, order, Tval) + [0] * (N - order) for i in range(10)]
    dl = jet(DELTA, sub, order, Tval) + [0] * (N - order)
    mj = [None] * 10
    for i in range(5, 10):
        x0, x1 = XSTAR[i], v[i]
        mj[i] = [2 * x0 ** 2, 4 * x0 * x1, 2 * x1 ** 2, 0]
    for i in range(5):
        mj[i] = [XSTAR[i + 5], v[i + 5], 0, 0]  # mobility of mean i is the variance v_i / V_j
    assert dl[0] == 0
    z = pscale(dl, 1 / (2 * Tval))
    z2 = pmul(z, z)
    sigma = padd([1, 0, 0, 0], pscale(z2, -1))          # sech^2 z = 1 - z^2 + O(z^4)
    MgE = [pmul(mj[i], gE[i]) for i in range(10)]
    Mdx = [pmul(mj[i], dx[i]) for i in range(10)]
    num_s = padd(*[pmul(dx[i], MgE[i]) for i in range(10)])
    q_s = padd(*[pmul(dx[i], Mdx[i]) for i in range(10)])
    den = padd([4 * Tval ** 2, 0, 0, 0], pmul(sigma, q_s))
    fac = pmul(pmul(sigma, num_s), pinv(den))
    return [padd(pscale(MgE[i], -1), pmul(Mdx[i], fac)) for i in range(10)]


# Jacobian columns from first-order jets along basis vectors
cols = []
for j in range(10):
    ej = sp.zeros(10, 1)
    ej[j] = 1
    jt = F_pb_jet(ej, Tc, order=1)
    cols.append(sp.Matrix([jt[i][1] for i in range(10)]))
J = sp.Matrix.hstack(*cols)
tlog("Jacobian done")
q = p
res["Jq_zero"] = bool(J * q == sp.zeros(10, 1))
lnull = J.T.nullspace()
res["left_null_dim"] = len(lnull)
l = lnull[0] / (lnull[0].T * q)[0]
jq = F_pb_jet(q, Tc)
B2 = sp.Matrix([2 * jq[i][2] for i in range(10)])
C3 = sp.Matrix([6 * jq[i][3] for i in range(10)])
res["lT_D2F_qq"] = str((l.T * B2)[0])
border = J.row_join(q).col_join(l.T.row_join(sp.zeros(1, 1)))
sol = border.LUsolve((-B2).col_join(sp.zeros(1, 1)))
h2 = sol[:10, 0]
jqh = F_pb_jet(q + h2, Tc, order=2)
jh = F_pb_jet(h2, Tc, order=2)
Bqh = sp.Matrix([(2 * jqh[i][2] - B2[i] - 2 * jh[i][2]) / 2 for i in range(10)])
a3 = (l.T * C3)[0] / 6 + (l.T * Bqh)[0] / 2
tlog("cubic coefficient done")
# parameter coefficient: at the equilibrium grad E = 0 for every T, so J(T) = -g_pb^{-1}(x_*, T) H_red(T);
# this closed form is checked against the jet Jacobian at T_c before differentiating in T.
H = sp.Matrix([[sp.Rational(z) for z in row] for row in A["H"]])
g = sp.Matrix([sp.Rational(z) for z in A["g"]])
Mstar = sp.diag(1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
Tv = sp.Symbol("Tv", positive=True)
gpb_inv = Mstar - (Mstar * g) * (Mstar * g).T / (4 * Tv ** 2 + (g.T * Mstar * g)[0])
J_T = -gpb_inv * (H - g * g.T / (4 * Tv))
res["J_formula_equals_jet_Jacobian_at_Tc"] = bool(J_T.subs(Tv, Tc).applyfunc(sp.nsimplify) == J)
a1 = sp.nsimplify((l.T * J_T.diff(Tv).subs(Tv, Tc) * q)[0])
pull_len = (q.T * (Mstar.inv() + g * g.T / (16 * Tc ** 2) * 4) * q)[0]
res["pullback_squared_length_of_p"] = str(pull_len)
res["pullback_length_equals_D_plus_4"] = bool(pull_len == D + 4)
res["a1"] = str(a1)
res["a3"] = str(sp.nsimplify(a3))
res["a1_float"] = float(a1)
res["a3_float"] = float(a3)
res["a1_expected_-4/(D+4)"] = str(-4 / (D + 4))
res["a3_expected_-c/(D+4)"] = str(-c / (D + 4))
res["a1_match"] = bool(a1 == -4 / (D + 4))
res["a3_match"] = bool(sp.nsimplify(a3) == -c / (D + 4))
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_cm_pullback.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
