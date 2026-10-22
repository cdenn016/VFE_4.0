r"""L02-GEOM s06: appendices/base_connection.tex (eq:conn-infinitesimal, prop:conn-curvature,
prop:conn-patches, eq:conn-reference-affine), with D = d + B and dT/dt = -B(gamma') T.

1. first-order expansion of U(c+dc) exp(-a dc) U(c)^{-1} = I - B(dc)
2. F_B = dB + B^B = U (da + a^a) U^{-1}  for random 2x2 polynomial U, a on R^2
3. B' = g B g^{-1} - dg g^{-1} under U' = g U; B unchanged under U' = U h, a' = h^{-1} a h + h^{-1} dh
4. constant example a_x = E12, a_y = E21: F_xy = diag(1,-1); holonomy of b dtheta on the circle.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(1)
x, y, e = sp.symbols('x y epsilon', real=True)
def rmat():
    return sp.Matrix(2, 2, lambda i, j: sum(sp.Rational(int(rng.integers(-3, 4)), 2) * m
                                            for m in (1, x, y, x * y)))
U = sp.eye(2) * 3 + rmat()
ax, ay = rmat(), rmat()
Ui = U.inv()
Bx = U * ax * Ui - sp.diff(U, x) * Ui
By = U * ay * Ui - sp.diff(U, y) * Ui
pt = {x: sp.Rational(1, 5), y: sp.Rational(-1, 7)}
# 1. expansion along direction (1, 2)
dx, dy = 1, 2
Theta = U.subs({x: x + e * dx, y: y + e * dy}) * (sp.eye(2) - e * (ax * dx + ay * dy)) * Ui
lin = sp.diff(Theta, e).subs(e, 0)
print('1. d/de Theta + B(dc) at point:', sp.simplify((lin + Bx * dx + By * dy).subs(pt)))
# 2. curvature
FB = sp.diff(By, x) - sp.diff(Bx, y) + Bx * By - By * Bx
Fa = sp.diff(ay, x) - sp.diff(ax, y) + ax * ay - ay * ax
print('2. F_B - U F_a U^-1 at point:', sp.simplify((FB - U * Fa * Ui).subs(pt)))
# 3. patch laws
g = sp.eye(2) * 2 + rmat(); gi = g.inv()
U2 = g * U; U2i = U2.inv()
Bx2 = U2 * ax * U2i - sp.diff(U2, x) * U2i
print("3. B2 - (g B g^-1 - dg g^-1):", sp.simplify((Bx2 - (g * Bx * gi - sp.diff(g, x) * gi)).subs(pt)))
h = sp.eye(2) * 2 + rmat(); hi = h.inv()
U3 = U * h; U3i = U3.inv()
ax3 = hi * ax * h + hi * sp.diff(h, x)
Bx3 = U3 * ax3 * U3i - sp.diff(U3, x) * U3i
print('3. reference change leaves B fixed:', sp.simplify((Bx3 - Bx).subs(pt)))
# 4. constant example and holonomy
E12 = sp.Matrix([[0, 1], [0, 0]]); E21 = sp.Matrix([[0, 0], [1, 0]])
print('4. [E12,E21] =', (E12 * E21 - E21 * E12).tolist())
b, th = sp.symbols('b theta', real=True)
T = sp.Function('T')
sol = sp.dsolve(sp.Eq(T(th).diff(th), -b * T(th)), T(th), ics={T(0): 1})
print('4. holonomy of b dtheta:', sp.simplify(sol.rhs.subs(th, 2 * sp.pi)))
