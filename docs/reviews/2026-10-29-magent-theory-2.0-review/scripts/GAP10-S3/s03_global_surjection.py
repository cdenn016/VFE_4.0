r"""GAP10-S3 s03: the global statement behind chapters/16_link_base_transport.tex:160.

Claim tested: a global C^2 surjection X: C x R^r -> E (columns = global sections, fixed
message metric, X_beta = h^{-1} X_alpha on overlaps) yields a global connection
a = -dX Q_X on E, and E can be nontrivial when r > K.

Witness: E = TS^2, K = 2, r = 3, columns = P_n e_alpha (three agents whose means are the
tangential projections of the constant vectors e_1, e_2, e_3).  In each stereographic
chart the local coefficients are X = g^{-1} f^T (f = coordinate frame, g = f^T f).
Checks (exact sympy):
  (1) f X = I - n n^T (columns are the projected constant vectors), rank 2 everywhere;
  (2) Q_X = f and a = -dX Q_X equals the Levi-Civita form g^{-1} f^T df;
  (3) gluing: X_S = h^{-1} X_N on the overlap and a_S = h^{-1} a_N h + h^{-1} dh (ibt:eq:affine);
  (4) metric compatibility d g = g a + a^T g with g = H_X^{-1};
  (5) Euler density Pf = (g F_uv)_{12}/sqrt(det g) = +- sqrt(det g) (Gauss curvature 1),
      integral 4 pi, Euler number 2 != 0, so E is not trivial (Chern-Weil / hairy ball).
Also: the Moebius line bundle over S^1 with r = 2 (nonorientable E, holonomy -1).
"""
from __future__ import annotations

import sympy as sp

u, v = sp.symbols('u v', real=True)
rho2 = u ** 2 + v ** 2
nN = sp.Matrix([2 * u, 2 * v, rho2 - 1]) / (1 + rho2)
fN = nN.jacobian([u, v])                     # 3x2 coordinate frame
gN = sp.simplify(fN.T * fN)
print("g_N =", gN)
XN = sp.simplify(gN.inv() * fN.T)            # 2x3 local coefficients of P_n e_alpha
P = sp.simplify(fN * XN)
print("(1) f X - (I - n n^T) =", sp.simplify(P - (sp.eye(3) - nN * nN.T)))
HN = sp.simplify(XN * XN.T)
QN = sp.simplify(XN.T * HN.inv())
print("    H_X - g^{-1} =", sp.simplify(HN - gN.inv()), "; det H_X =", sp.factor(HN.det()), "(> 0: rank 2 everywhere)")
print("(2) Q_X - f =", sp.simplify(QN - fN))
aN = [sp.simplify(-sp.diff(XN, s) * QN) for s in (u, v)]
LC = [sp.simplify(gN.inv() * fN.T * sp.diff(fN, s)) for s in (u, v)]
print("    a - Levi-Civita form:", [sp.simplify(aN[k] - LC[k]) for k in range(2)])

# (4) metric compatibility with g = H_X^{-1}
Mg = sp.simplify(HN.inv())
comp = [sp.simplify(sp.diff(Mg, s) - (Mg * aN[k] + aN[k].T * Mg)) for k, s in enumerate((u, v))]
print("(4) dM - (M a + a^T M):", comp)

# (5) curvature and Euler density
Fuv = sp.simplify(sp.diff(aN[1], u) - sp.diff(aN[0], v) + aN[0] * aN[1] - aN[1] * aN[0])
print("    F_uv =", Fuv)
MF = sp.simplify(Mg * Fuv)
print("    M F skew:", sp.simplify(MF + MF.T))
pf = sp.simplify(MF[0, 1] / sp.sqrt(Mg.det()))
area = sp.simplify(sp.sqrt(Mg.det()))
print("(5) Pfaffian density =", pf, "; area density =", area, "; ratio =", sp.simplify(pf / area))
r_, ph_ = sp.symbols('r phi', positive=True)
closed = 4 / (1 + rho2) ** 2
print("    Pfaffian density - 4/(1+u^2+v^2)^2 =", sp.simplify(sp.factor(pf ** 2) - closed ** 2), "(squares; both positive)")
integral = sp.integrate(sp.integrate(closed.subs({u: r_ * sp.cos(ph_), v: r_ * sp.sin(ph_)}).simplify() * r_, (r_, 0, sp.oo)), (ph_, 0, 2 * sp.pi))
print("    integral of Pfaffian density over the chart (= S^2 minus a point):", sp.simplify(integral),
      "; Euler number = integral/(2 pi) =", sp.simplify(integral / (2 * sp.pi)))

# (3) south chart and gluing on the overlap
up, vp = sp.symbols('up vp', real=True)
rp2 = up ** 2 + vp ** 2
nS = sp.Matrix([2 * up, 2 * vp, 1 - rp2]) / (1 + rp2)
# overlap map: the same point: (up, vp) = (u, v)/(u^2+v^2)
sub = {up: u / rho2, vp: v / rho2}
print("(3) same point on overlap:", sp.simplify(nS.subs(sub) - nN))
fS_p = nS.jacobian([up, vp])                 # frame of the S chart, in its own coordinates
fS = sp.simplify(fS_p.subs(sub))             # as a function of (u, v)
gS = sp.simplify(fS.T * fS)
XS = sp.simplify(gS.inv() * fS.T)
h = sp.simplify(gN.inv() * fN.T * fS)        # w_N = h w_S, i.e. X_S = h^{-1} X_N
print("    X_S - h^{-1} X_N =", sp.simplify(XS - h.inv() * XN))
QS = sp.simplify(XS.T * (XS * XS.T).inv())
aS = [sp.simplify(-sp.diff(XS, s) * QS) for s in (u, v)]   # components in (u, v) coordinates
glue = [sp.simplify(aS[k] - (h.inv() * aN[k] * h + h.inv() * sp.diff(h, s))) for k, s in enumerate((u, v))]
print("    a_S - (h^{-1} a_N h + h^{-1} dh):", glue)
print("    det h =", sp.factor(sp.simplify(h.det())), "(orientation-reversing chart change; gluing is unaffected)")

# Moebius line bundle over S^1: K = 1, r = 2
th = sp.symbols('theta', real=True)
Xm = sp.Matrix([[sp.cos(th / 2), sp.sin(th / 2)]])
am = sp.simplify(-sp.diff(Xm, th) * Xm.T * (Xm * Xm.T).inv())
print("\nMoebius: X(theta) = (cos theta/2, sin theta/2), X(2 pi) = -X(0) (transition h = -1);",
      "a =", am, "; holonomy = transition sign =", (Xm.subs(th, 2 * sp.pi) * Xm.subs(th, 0).T)[0])
