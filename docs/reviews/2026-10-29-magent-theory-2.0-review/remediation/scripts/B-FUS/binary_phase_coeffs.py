"""Check eq:public-phase-binary-expansion and the next-order phase coefficients
c12, c32, c14 by solving the invariance equation D Pi(V) = f_b(Pi) as a series."""
import sympy as sp
m, a, J = sp.symbols('m a J')
def xdot(xi, xj):
    return -(1 - xi**2)*(2*sp.atanh(xi) - sp.atanh(xj) - J*xi) - (xi - xj)
x1, x2 = m + a, m - a
md = (xdot(x1, x2) + xdot(x2, x1))/2
ad = (xdot(x1, x2) - xdot(x2, x1))/2
fb = lambda x: -(1 - x**2)*(sp.atanh(x) - J*x)
t = sp.symbols('t')
def trunc(expr, order):
    s = sp.series(expr.subs({m: t*m, a: t*a}), t, 0, order).removeO()
    return sp.expand(s.subs(t, 1))
g = trunc(md - fb(m), 7)
print('mdot - f_b(m) to degree 5:', sp.factor(sp.expand(g)))
print('adot to degree 5:', sp.factor(trunc(ad, 6)))
c12, c32, c14 = sp.symbols('c12 c32 c14')
Pi = m + c12*m*a**2 + c32*m**3*a**2 + c14*m*a**4
lhs = sp.diff(Pi, m)*md + sp.diff(Pi, a)*ad
res = trunc(lhs - fb(Pi), 6)
eqs = [sp.expand(res).coeff(m, p).coeff(a, q) for (p, q) in [(1, 2), (3, 2), (1, 4)]]
sol = sp.solve(eqs, [c12, c32, c14], dict=True)[0]
for k, v in sol.items():
    print(k, '=', sp.factor(v))
print('c12 check:', sp.simplify(sol[c12] - 3*(2 - J)/(2*(5 - J))))
print('c32 check:', sp.simplify(sol[c32] - (9*J**2 - 32*J + 52)/(6*(5 - J)*(3 - J))))
print('c14 check:', sp.simplify(sol[c14] - (15*J**2 - 64*J + 80)/(8*(5 - J)**2)))
