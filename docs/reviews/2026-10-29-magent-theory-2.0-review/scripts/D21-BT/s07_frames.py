r"""D21-BT s07: marked square frames and affine averages (ch16, ibt:frame, ibt:eq:average-curvature).

(1) a_Q = -dQ Q^{-1} is flat, D^{a_Q} Q = 0, and T_Q composes; affine law for Q' = h^{-1}Q.
(2) For flat a1 = a_{Q1}, a2 = a_{Q2}, Delta = a1 - a2 and a scalar weight lambda(c),
    F^{a2 + lambda Delta} = d lambda ^ Delta - lambda (1 - lambda) Delta ^ Delta (exact).
(3) Averaging the frames instead, Q = lambda Q1 + (1 - lambda) Q2, gives a flat connection.
"""
from __future__ import annotations
import sympy as sp

x, y = sp.symbols("x y", real=True)
co = (x, y)


def curv(a):
    return a[1].diff(x) - a[0].diff(y) + a[0] * a[1] - a[1] * a[0]


def frame_conn(Q):
    Qi = Q.inv()
    return [-Q.diff(c) * Qi for c in co]


Q1 = sp.Matrix([[1 + x**2, y], [x, 2 + y**2]])
Q2 = sp.Matrix([[sp.exp(x), x * y], [0, 1 + y]])
pts = [{x: sp.Rational(1, 3), y: sp.Rational(-1, 4)}, {x: sp.Rational(-2, 5), y: sp.Rational(3, 7)}]
z2 = sp.zeros(2)
a1, a2 = frame_conn(Q1), frame_conn(Q2)
print("(1) F^{a_Q1} = 0:", all(sp.simplify(curv(a1).subs(p)) == z2 for p in pts))
print("(1) D^{a_Q1} Q1 = 0:", all(sp.simplify((Q1.diff(c) + a1[i] * Q1).subs(p)) == z2 for p in pts for i, c in enumerate(co)))
h = sp.Matrix([[2 + x, y**2], [x * y, 1 + y]])
a1p = frame_conn(h.inv() * Q1)
print("(1) affine law for Q' = h^{-1}Q:", all(sp.simplify((a1p[i] - (h.inv() * a1[i] * h + h.inv() * h.diff(c))).subs(p)) == z2
                                           for p in pts for i, c in enumerate(co)))
lam = sp.Rational(1, 3) + x * y / 5
Delta = [a1[i] - a2[i] for i in range(2)]
a = [a2[i] + lam * Delta[i] for i in range(2)]
lhs = curv(a)
rhs = (lam.diff(x) * Delta[1] - lam.diff(y) * Delta[0]) - lam * (1 - lam) * (Delta[0] * Delta[1] - Delta[1] * Delta[0])
print("(2) average-curvature formula:", all(sp.simplify((lhs - rhs).subs(p)) == z2 for p in pts))
lam0 = sp.Rational(1, 3)
a_const = [a2[i] + lam0 * Delta[i] for i in range(2)]
Fc = curv(a_const).subs(pts[0])
print("(2) constant weight, nonzero curvature from noncommuting Delta: F =", sp.N(sp.simplify(Fc), 6).tolist())
Qavg = lam * Q1 + (1 - lam) * Q2
print("(3) frame average is flat:", all(sp.simplify(curv(frame_conn(Qavg)).subs(p)) == z2 for p in pts))
