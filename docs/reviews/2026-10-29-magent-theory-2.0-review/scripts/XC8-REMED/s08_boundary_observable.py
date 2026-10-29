r"""Check eq:boundary-response-derivatives and eq:boundary-observability-det (10_boundary_closure.tex:320-350).

x_i' = -(1-x_i^2)[2 atanh x_i - atanh x_j - J y_i] - (x_i - x_j), and the exchanged equation for y.
With x_{1,2} = m +- a, y_{1,2} = n +- e, F_x = (x_1'+x_2')/2. Claimed:
  F_x = -(1-m^2)(atanh m - J n) + (6m - J n) a^2 - 2 J m a e + O(5).
Also the Jacobian determinant of (m,n,F_x,F_y) at m=n=eps c1, a=e=eps c2:
  4 eps^4 c1^2 c2^2 (6-3J)(6-J) + O(eps^6).
atanh is replaced by its Taylor polynomial to degree 5, which is exact through total degree 4 here.
"""
import sympy as sp

m, n, a, e, J, t = sp.symbols("m n a e J t", real=True)


def atanh_p(u):
    return u + u**3 / 3 + u**5 / 5


def xdot(xi, xj, yi):
    return -(1 - xi**2) * (2 * atanh_p(xi) - atanh_p(xj) - J * yi) - (xi - xj)


def trunc(expr, deg):
    p = sp.Poly(sp.expand(expr), m, n, a, e)
    return sum(cf * m**i * n**j * a**k * e**l for (i, j, k, l), cf in p.terms() if i + j + k + l <= deg)


x1, x2, y1, y2 = m + a, m - a, n + e, n - e
Fx = trunc((xdot(x1, x2, y1) + xdot(x2, x1, y2)) / 2, 4)
claim = trunc(-(1 - m**2) * (atanh_p(m) - J * n) + (6 * m - J * n) * a**2 - 2 * J * m * a * e, 4)
r1 = sp.simplify(Fx - claim)
print("F_x - claim through total degree 4:", r1)
Fy = trunc(Fx.subs({m: n, n: m, a: e, e: a}, simultaneous=True), 4)
H = sp.Matrix([m, n, Fx, Fy])
Jac = H.jacobian([m, n, a, e])
c1, c2, eps = sp.symbols("c1 c2 epsilon", positive=True)
detJ = sp.expand(Jac.det().subs({m: eps * c1, n: eps * c1, a: eps * c2, e: eps * c2}))
lead = sp.expand(sum(detJ.coeff(eps, k) * eps**k for k in range(0, 6)))
claimed = sp.expand(4 * eps**4 * c1**2 * c2**2 * (6 - 3 * J) * (6 - J))
print("det through eps^5 minus claim:", sp.simplify(lead - claimed))
with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
    fh.write("F_x - claim through total degree 4: %s\n" % r1)
    fh.write("det through eps^5 minus claim: %s\n" % sp.simplify(lead - claimed))
