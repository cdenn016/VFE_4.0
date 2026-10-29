"""XC3-SCOPE sympy check of eq:public-phase-binary-expansion and eq:public-phase-binary-parent
(chapter 09): for xdot_i = -(1-x_i^2)[2 atanh x_i - atanh x_j - J x_i] - (x_i - x_j),
x_{1,2} = m +- a, verify mdot = f_b(m) + (6-3J) m a^2 + O(5) and adot = -(5-J) a + O(3),
and the invariance coefficient c = (6-3J)/(2(5-J))."""
import sympy as sp
m, a, J, e = sp.symbols('m a J epsilon')
def fld(xi, xj):
    return -(1-xi**2)*(2*sp.atanh(xi) - sp.atanh(xj) - J*xi) - (xi - xj)
x1, x2 = m + a, m - a
mdot = (fld(x1, x2) + fld(x2, x1))/2
adot = (fld(x1, x2) - fld(x2, x1))/2
fb = -(1-m**2)*(sp.atanh(m) - J*m)
s = sp.series((mdot - fb).subs({m: e*m, a: e*a}), e, 0, 5).removeO()
print("mdot - f_b(m) through total degree 4:", sp.factor(sp.expand(s.subs(e, 1))))
s2 = sp.series(adot.subs({m: e*m, a: e*a}), e, 0, 3).removeO()
print("adot through total degree 2:", sp.factor(sp.expand(s2.subs(e, 1))))
c = sp.symbols('c')
print("invariance coefficient:", sp.solve(sp.Eq((6-3*J) - 2*(5-J)*c, 0), c)[0], " vs 3(2-J)/(2(5-J)) ->",
      sp.simplify(sp.solve(sp.Eq((6-3*J) - 2*(5-J)*c, 0), c)[0] - 3*(2-J)/(2*(5-J))))
