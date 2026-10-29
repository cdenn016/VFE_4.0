r"""D21-BT s01: exact checks of the rectangular message construction (ch16).

Checks, for chapters/16_link_base_transport.tex:
  (1) affine law  a' = h^{-1} a h + h^{-1} dh  for X' = h^{-1} X, general GL(2) h(c);
  (2) the three curvature forms of ibt:curvature, plus the Berry form X dP ^ dP Q;
  (3) metric compatibility dH + aH + H a^T = 0, Tr a = -1/2 d log det H, Tr F = 0;
  (4) the witness (ibt:eq:witnessA, ibt:eq:witnessF) and the Laplace pullback;
  (5) the scalar backtrack example;
  (6) a two-dimensional backtrack-only witness with nonzero (gauge-invariant) curvature;
  (7) the M-skew property of F (eigenvalues purely imaginary) at sample points.
Exact rational arithmetic; derivatives are symbolic, evaluation at rational points.
"""
from __future__ import annotations

import sympy as sp

x, y = sp.symbols("x y", real=True)
coords = (x, y)


def conn(X: sp.Matrix) -> list[sp.Matrix]:
    r"""a_mu = -\partial_mu X Q_X with Q_X = X^T (X X^T)^{-1}."""
    H = X * X.T
    Q = X.T * H.inv()
    return [-X.diff(c) * Q for c in coords]


def curv(a: list[sp.Matrix]) -> sp.Matrix:
    r"""F_xy = \partial_x a_y - \partial_y a_x + [a_x, a_y]."""
    return a[1].diff(x) - a[0].diff(y) + a[0] * a[1] - a[1] * a[0]


def wedge(A: list[sp.Matrix], B: list[sp.Matrix]) -> sp.Matrix:
    """(A ^ B)_xy = A_x B_y - A_y B_x for matrix-valued one-forms."""
    return A[0] * B[1] - A[1] * B[0]


def at(M: sp.Matrix, pt: dict) -> sp.Matrix:
    return sp.simplify(M.subs(pt))


results: dict[str, object] = {}

# (1)-(3): generic polynomial 2x4 message matrix (r = 4 > K = 2)
X = sp.Matrix([[1 + x, y, x * y, 2],
               [x**2, 1 - y, 3, x + y**2]])
H = X * X.T
Q = X.T * H.inv()
P = Q * X
a = conn(X)
F = curv(a)
dX = [X.diff(c) for c in coords]
dP = [P.diff(c) for c in coords]
dQ = [Q.diff(c) for c in coords]
I4 = sp.eye(4)
forms = {
    "dX^dP Q": wedge(dX, [dp * Q for dp in dP]),
    "dX^(I-P)dQ": wedge(dX, [(I4 - P) * dq for dq in dQ]),
    "dX^(I-P)dX^T H^-1": wedge(dX, [(I4 - P) * d.T * H.inv() for d in dX]),
    "X dP^dP Q (Berry)": wedge([X * dp for dp in dP], [dp * Q for dp in dP]),
}
pts = [{x: sp.Rational(1, 3), y: sp.Rational(-2, 5)},
       {x: sp.Rational(-3, 7), y: sp.Rational(5, 4)},
       {x: sp.Rational(2, 1), y: sp.Rational(1, 9)}]
for name, form in forms.items():
    res = [at(F - form, p) for p in pts]
    results[f"curvature form {name}"] = all(r == sp.zeros(2) for r in res)
for i, c in enumerate(coords):
    res = [at(H.diff(c) + a[i] * H + H * a[i].T, p) for p in pts]
    results[f"metric compatibility d_{c}"] = all(r == sp.zeros(2) for r in res)
    res = [sp.simplify((a[i].trace() + sp.Rational(1, 2) * sp.diff(sp.log(H.det()), c)).subs(p)) for p in pts]
    results[f"trace identity d_{c}"] = all(r == 0 for r in res)
results["Tr F = 0"] = all(sp.simplify(F.trace().subs(p)) == 0 for p in pts)
# M-skewness of F with M = H^{-1}: (M F)^T = -(M F)
results["H^-1 F skew"] = all(at((H.inv() * F) + (H.inv() * F).T, p) == sp.zeros(2) for p in pts)

# affine law under a general (non-orthogonal, nonconstant) GL(2) h
h = sp.Matrix([[1 + x**2, y], [x * y, 2 + y]])
Xp = h.inv() * X
ap = conn(Xp)
for i, c in enumerate(coords):
    target = h.inv() * a[i] * h + h.inv() * h.diff(c)
    res = [at(ap[i] - target, p) for p in pts]
    results[f"affine law d_{c}"] = all(r == sp.zeros(2) for r in res)

# (4) witness
Xw = sp.Matrix([[1, 0, x], [0, 1, y]])
aw = [sp.simplify(m) for m in conn(Xw)]
s = 1 + x**2 + y**2
v = sp.Matrix([x, y])
aw_claim = [-sp.Matrix([1, 0]) * v.T / s, -sp.Matrix([0, 1]) * v.T / s]
results["witness a"] = all(sp.simplify(aw[i] - aw_claim[i]) == sp.zeros(2) for i in range(2))
Fw = sp.simplify(curv(aw))
Fw_claim = sp.Matrix([[-x * y, 1 + x**2], [-(1 + y**2), x * y]]) / s**2
results["witness F"] = sp.simplify(Fw - Fw_claim) == sp.zeros(2)
results["witness F(0)"] = str(Fw.subs({x: 0, y: 0}))

# Laplace realization: third column (x, 1+xy)
Xl = sp.Matrix([[1, 0, x], [0, 1, 1 + x * y]])
Fl = sp.simplify(curv(conn(Xl)))
pull = sp.simplify(Fw_claim.subs({y: 1 + x * y}, simultaneous=True) * x)
results["Laplace pullback F"] = sp.simplify(Fl - pull) == sp.zeros(2)
results["Laplace F at x=0 vanishes"] = sp.simplify(Fl.subs(x, 0)) == sp.zeros(2)
results["Laplace F entry (1,2) / x positive factor"] = str(sp.factor(sp.simplify(Fl[0, 1] / x)))

# (5) scalar backtrack example: X = [1, s(c)]
sc = sp.Function("s")(x, y)
Xs = sp.Matrix([[1, sc]])
a_s = conn(Xs)
results["scalar a = -s ds/(1+s^2)"] = all(sp.simplify(a_s[i][0] + sc * sc.diff(c) / (1 + sc**2)) == 0
                                          for i, c in enumerate(coords))
# the scalar connection is exact: a = d(-1/2 log(1+s^2)), hence gauge-equivalent to a = 0
results["scalar a exact (gauge-trivial)"] = all(
    sp.simplify(a_s[i][0] - sp.diff(-sp.Rational(1, 2) * sp.log(1 + sc**2), c)) == 0
    for i, c in enumerate(coords))

# (6) backtrack-only 2D witness: Theta_ij = Theta_ji = S(c) SPD, so Omega = I, N = S^2.
S = sp.diag(sp.exp(x), sp.exp(y))
mu = sp.Matrix([1, 1])
X_raw = sp.Matrix.hstack(sp.eye(2), S * mu)
X_rec = sp.Matrix.hstack(sp.eye(2), sp.eye(2) * mu)
F_raw = sp.simplify(curv(conn(X_raw)))
F_rec = sp.simplify(curv(conn(X_rec)))
results["backtrack-only F_raw at (0,0)"] = str(F_raw.subs({x: 0, y: 0}))
results["backtrack-only F_raw nonzero entry (1,2)"] = str(sp.simplify(F_raw[0, 1]))
results["reciprocal-only F_rec = 0"] = F_rec == sp.zeros(2)

for k, val in results.items():
    print(f"{k}: {val}")
assert all(v is True for v in results.values() if isinstance(v, bool)), "a check failed"
print("ALL BOOLEAN CHECKS TRUE")
