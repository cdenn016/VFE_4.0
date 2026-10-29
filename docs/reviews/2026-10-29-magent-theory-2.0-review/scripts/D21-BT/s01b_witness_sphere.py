r"""D21-BT s01b: the ch16 witness is the Levi-Civita connection of the unit sphere pulled back
by inverse central (gnomonic) projection n(v) = (v, -1)/sqrt(1 + |v|^2).

Checks: det F_xy (a gauge invariant of the 2x2 traceless metric curvature) equals the square
of the pulled-back sphere area density |d_x n x d_y n|, i.e. Gauss curvature 1 times area.
Also: ker X(c) is spanned by (v, -1), so W(c) = im X^T = n(c)^perp = T_{n(c)} S^2.
"""
import sympy as sp
x, y = sp.symbols("x y", real=True)
v = sp.Matrix([x, y]); s = 1 + x**2 + y**2
F = sp.Matrix([[-x * y, 1 + x**2], [-(1 + y**2), x * y]]) / s**2
n = sp.Matrix([x, y, -1]) / sp.sqrt(s)
area = sp.simplify(n.diff(x).cross(n.diff(y)).norm())
print("det F_xy =", sp.simplify(F.det()), "; area density =", area, "; det F - area^2 =", sp.simplify(F.det() - area**2))
X = sp.Matrix([[1, 0, x], [0, 1, y]])
print("X (v,-1) =", (X * sp.Matrix([x, y, -1])).T)
