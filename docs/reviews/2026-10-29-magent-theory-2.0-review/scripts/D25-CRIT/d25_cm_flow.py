r"""D25-CRIT check 3: center-manifold coefficients of the actual Fisher vector fields at (x_*, T_c)
for it2:critical-pitchfork, computed by the standard projection formula (independent of the
manuscript's w = Y - X argument). Usage: python d25_cm_flow.py {reduced|full|pullback}

  full:     xdot = -M(x) grad_x E(x,s),   sdot = -(1/4 - s^2) dE/ds          (q = (p, 1))
  reduced:  xdot = -M(x) grad E_red(x),   E_red = A - T log cosh(Delta/(2T))  (q = p)
  pullback: xdot = -g_pb^{-1} grad E_red, g_pb = M^{-1} + grad b* grad b*^T / (b*(1-b*))  (q = p)

With J the Jacobian at T_c, l the left null vector (l^T q = 1), h2 solving J h2 = -D^2F[q,q],
l^T h2 = 0, the reduced field in u = l^T(y - y_*) is
  u' = a1 (T - T_c) u + a3 u^3 + ...,  a1 = l^T (dJ/dT) q,
  a3 = (1/6) l^T D^3F[q,q,q] + (1/2) l^T D^2F[q,h2].
All mobility derivatives are included because F itself is differentiated.
Part C (reduced, full): third-order graph h3 and the comparison Y(s,0) - X(s) = (c/d) B s^3.
Output: d25_cm_<flow>.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import sympy as sp

from d25_model import DELTA, ECAL, E_RED, MOB, POINT, R, T, T_C_CLAIM, XS, XSTAR, grad, s

OUT = Path(__file__).resolve().parent
flow = sys.argv[1]
t0 = time.perf_counter()
eps, eta = sp.symbols("eps eta")
Tc = T_C_CLAIM
A = json.loads((OUT / "d25_pitchfork_exact.json").read_text(encoding="utf-8"))
p = sp.Matrix([sp.Rational(z) for z in A["p"]])
X3 = sp.Matrix([sp.Rational(z) for z in A["X3"]])
X2 = sp.Matrix([sp.Rational(z) for z in A["X2"]])
H = sp.Matrix([[sp.Rational(z) for z in row] for row in A["H"]])
c = sp.Rational(A["c"])
D = sp.Rational(A["D"])
Minv = sp.diag(*([1] * 5 + [R(1, 2)] * 5))


def tlog(msg: str) -> None:
    print(f"[{time.perf_counter() - t0:6.2f}s] {msg}", flush=True)


def to_rational(z: sp.Expr) -> sp.Expr:
    if z.is_Rational:
        return z
    z = sp.nsimplify(sp.simplify(z))
    assert z.is_Rational, z
    return z


bstar = (1 - sp.tanh(DELTA / (2 * T))) / 2
if flow == "full":
    vars_ = XS + [s]
    base = XSTAR + [R(0)]
    F = (-MOB * grad(ECAL, XS)).col_join(sp.Matrix([-(R(1, 4) - s ** 2) * sp.diff(ECAL, s)]))
    q = p.col_join(sp.Matrix([1]))
    d = D + 4
else:
    vars_ = XS
    base = XSTAR
    gradE = grad(E_RED, XS)
    if flow == "reduced":
        F = -MOB * gradE
        d = D
    elif flow == "pullback":
        gb = grad(bstar, XS)
        den = bstar * (1 - bstar) + (gb.T * MOB * gb)[0]
        F = -(MOB - (MOB * gb) * (MOB * gb).T / den) * gradE
        d = D + 4
    else:
        raise SystemExit("flow must be reduced, full or pullback")
    q = p
n = len(vars_)
FT = F.subs(T, Tc)
pt = dict(zip(vars_, base))

# Jacobian at T_c and its T-derivative applied to q
Jsym = F.jacobian(vars_)
Jat = Jsym.subs(pt)
J = Jat.subs(T, Tc).applyfunc(to_rational)
dJq = (Jat.applyfunc(lambda z: sp.diff(z, T)).subs(T, Tc) * q).applyfunc(to_rational)
tlog("Jacobian done")
res = {"flow": flow, "Jq_zero": bool(J * q == sp.zeros(n, 1))}
lnull = J.T.nullspace()
res["left_null_dim"] = len(lnull)
l = lnull[0] / (lnull[0].T * q)[0]
res["l_equals_Mfisher_q_over_norm"] = None
if flow in ("reduced", "full"):
    G = Minv if flow == "reduced" else sp.diag(Minv, 4)
    res["l_equals_Mfisher_q_over_norm"] = bool(l == G * q / (q.T * G * q)[0])
a1 = (l.T * dJq)[0]

# directional derivatives along q from one substituted expression per component
sub_q = {z: base[i] + eps * q[i] for i, z in enumerate(vars_)}
B2, C3, Eq = [], [], []
for i in range(n):
    e = FT[i].subs(sub_q)
    Eq.append(e)
    e1 = sp.diff(e, eps)
    e2 = sp.diff(e1, eps)
    B2.append(to_rational(e2.subs(eps, 0)))
    C3.append(to_rational(sp.diff(e2, eps).subs(eps, 0)))
B2, C3 = sp.Matrix(B2), sp.Matrix(C3)
tlog("second and third directional derivatives done")
res["lT_D2F_qq"] = str((l.T * B2)[0])
border = J.row_join(q).col_join(l.T.row_join(sp.zeros(1, 1)))
sol = border.LUsolve((-B2).col_join(sp.zeros(1, 1)))
h2 = sol[:n, 0]
res["bordered_multiplier"] = str(sol[n, 0])
sub_qh = {z: base[i] + eps * q[i] + eta * h2[i] for i, z in enumerate(vars_)}
Bqh = sp.Matrix([to_rational(sp.diff(FT[i].subs(sub_qh), eps, eta).subs({eps: 0, eta: 0})) for i in range(n)])
tlog("mixed derivative done")
a3 = (l.T * C3)[0] / 6 + (l.T * Bqh)[0] / 2
res["a1"] = str(a1)
res["a3"] = str(a3)
res["a1_float"] = float(a1)
res["a3_float"] = float(a3)
res["a1_expected_-4/d"] = str(-4 / d)
res["a3_expected_-c/d"] = str(-c / d)
res["a1_match"] = bool(a1 == -4 / d)
res["a3_match"] = bool(a3 == -c / d)
res["d_used"] = str(d)
tlog(f"a1 = {float(a1):.12f} (expected {float(-4/d):.12f}); a3 = {float(a3):.12f} (expected {float(-c/d):.12f})")

# Part C: third-order graph and Y(s,0) - X(s)
if flow in ("reduced", "full"):
    rhs = 6 * (q * a3 - Bqh / 2 - C3 / 6)
    sol3 = border.LUsolve(rhs.col_join(sp.zeros(1, 1)))
    h3 = sol3[:n, 0]
    if flow == "full":
        h2x, h3x = h2[:10, 0], h3[:10, 0]
        h2s = h2[10, 0]
        kappa = h3[10, 0] / 6
    else:
        h2x, h3x = h2, h3
        u = sp.Symbol("u")
        yu = {z: XSTAR[i] + q[i] * u + h2[i] * u ** 2 / 2 + h3[i] * u ** 3 / 6 for i, z in enumerate(XS)}
        sst = (bstar - R(1, 2)).subs(T, Tc).subs(yu)
        d1 = sp.diff(sst, u)
        d2 = sp.diff(d1, u)
        h2s = to_rational(d2.subs(u, 0)) / 2
        kappa = to_rational(sp.diff(d2, u).subs(u, 0)) / 6
        res["sstar_linear_coefficient"] = str(to_rational(d1.subs(u, 0)))
    Bvec = H.LUsolve(Minv * p)
    Y3 = h3x / 6 - kappa * p
    diff3 = Y3 - X3 / 6
    res["h2_matter_equals_X''(0)"] = bool(h2x == X2)
    res["second_order_row_coefficient"] = str(h2s)
    res["kappa_s_of_u"] = str(kappa)
    res["Y_minus_X_s3_equals_(c/d)B"] = bool(diff3 == (c / d) * Bvec)
    res["Y_minus_X_s3_over_B_first_nonzero"] = str(next(diff3[i] / Bvec[i] for i in range(10) if Bvec[i] != 0))
    tlog("Part C done")
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / f"d25_cm_{flow}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
