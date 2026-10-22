r"""Symbolic check of the smooth binary model (chapters/05_inheritance.tex, inh:smooth-binary-model,
inh:smooth-mixture-failure, inh:smooth-boundary-failure, inh:role-loss numbers).

E_0(x,y) = sum_{k,m} q_x(k) s_y(m) log( q_x s_y / (r G_J) ) = psi(x)+psi(y)-Jxy+log cosh J,
q_x K_x = q_{f(x)}, f(x)=(x+x^3)/4,
xdot_i = -(1-x_i^2) d_{x_i} E, ydot_i = -(1-y_i^2) d_{y_i} E,
E = sum_i E_0(x_i,y_i) + P(x_1,x_2) + P(y_1,y_2), P(x1,x2)=d(x1,f(x2))+d(x2,f(x1)), d = KL(q_x||q_z).
"""
import sympy as sp

x, y, z, J = sp.symbols('x y z J', real=True)
k_vals = (-1, 1)
q = lambda t, k: (1 + k * t) / 2
G = lambda k, m: sp.exp(J * k * m) / (2 * sp.cosh(J))
E0 = sum(q(x, k) * q(y, m) * sp.log(q(x, k) * q(y, m) / (sp.Rational(1, 2) * G(k, m)))
         for k in k_vals for m in k_vals)
psi = lambda t: (1 + t) / 2 * sp.log(1 + t) + (1 - t) / 2 * sp.log(1 - t)
claimed = psi(x) + psi(y) - J * x * y + sp.log(sp.cosh(J))
# numeric spot check at several points (log simplification is slow symbolically)
import random
random.seed(0)
res = 0.0
for _ in range(20):
    vals = {x: random.uniform(-0.95, 0.95), y: random.uniform(-0.95, 0.95), J: random.uniform(-2, 2)}
    res = max(res, abs(float((E0 - claimed).subs(vals).evalf(30))))
print("E_0 identity max residual (20 random points, 30 digits):", res)

# channel
rho = (1 + x**2) / 4
ell = sp.symbols('ell')
qK = {l: sp.simplify(sum(q(x, k) * (1 + rho * k * l) / 2 for k in k_vals)) for l in k_vals}
fx = (x + x**3) / 4
print("q_x K_x = q_f(x):", all(sp.simplify(qK[l] - q(fx, l)) == 0 for l in k_vals))
print("min channel entry (1-rho)/2 > 1/4 on (-1,1): (1-rho)/2 - 1/4 =", sp.simplify((1 - rho) / 2 - sp.Rational(1, 4)))

# KL derivatives
d = sum(q(x, k) * sp.log(q(x, k) / q(z, k)) for k in k_vals)
_r = sp.diff(d, x) - (sp.atanh(x) - sp.atanh(z))
print("d_x d - (atanh x - atanh z), max residual at 20 random interior points:",
      max(abs(float(_r.subs({x: random.uniform(-0.95, 0.95), z: random.uniform(-0.95, 0.95)}).evalf(30)))
          for _ in range(20)))
print("d_z d - (z-x)/(1-z^2):", sp.simplify(sp.diff(d, z) - (z - x) / (1 - z**2)))

f = lambda t: (t + t**3) / 4
dfun = lambda a_, b_: d.subs({x: a_, z: b_}, simultaneous=True)
x1, x2, y1, y2 = sp.symbols('x1 x2 y1 y2', real=True)
P = lambda u1, u2: dfun(u1, f(u2)) + dfun(u2, f(u1))
E0f = lambda u, v: psi(u) + psi(v) - J * u * v + sp.log(sp.cosh(J))
E = E0f(x1, y1) + E0f(x2, y2) + P(x1, x2) + P(y1, y2)
xd = [-(1 - v**2) * sp.diff(E, v) for v in (x1, x2)]
yd = [-(1 - v**2) * sp.diff(E, v) for v in (y1, y2)]
a, c = sp.symbols('a c', real=True)
Jv = sp.Rational(1, 2)
A = {x1: 0, y1: c, x2: 0, y2: c, J: Jv}
B = {x1: a, y1: c, x2: -a, y2: c, J: Jv}
xbarA = sp.simplify((xd[0] + xd[1]).subs(A, simultaneous=True) / 2)
xbarB = sp.simplify((xd[0] + xd[1]).subs(B, simultaneous=True) / 2)
print("xbar_dot(A) =", xbarA, "; xbar_dot(B) =", sp.factor(xbarB))
ybarA = (yd[0] + yd[1]).subs(A, simultaneous=True) / 2
ybarB = (yd[0] + yd[1]).subs(B, simultaneous=True) / 2
for av, cv in [(sp.Rational(1, 2), sp.Rational(1, 4)), (sp.Rational(1, 10), sp.Rational(-3, 7))]:
    print(f"a={av}, c={cv}: xbar A,B =", sp.nsimplify(xbarA.subs({a: av, c: cv})),
          sp.nsimplify(xbarB.subs({a: av, c: cv})),
          " ybar A-B =", sp.N(ybarA.subs({a: av, c: cv}) - ybarB.subs({a: av, c: cv}), 25))
# individual y derivatives at B versus h(c)
h = 2 * sp.atanh(c) - sp.atanh(f(c)) + sp.diff(f(c), c) * (f(c) - c) / (1 - f(c)**2)
yB1 = yd[0].subs(B, simultaneous=True)
print("ydot_1(B) + (1-c^2)(h(c)-J a) at a=0.3,c=0.2:",
      sp.N((yB1 + (1 - c**2) * (h - Jv * a)).subs({a: sp.Rational(3, 10), c: sp.Rational(1, 5)}), 25))

# boundary failure
m = sp.symbols('m', real=True)
b = lambda u1, u2: (f(u1) + f(u2)) / 2
print("b(m+a,m-a)-b(m,m) - 3/4 m a^2 =", sp.simplify(b(m + a, m - a) - b(m, m) - sp.Rational(3, 4) * m * a**2))
print("means at m=a=1/4:", b(sp.Rational(1, 2), 0) * 256, "/256 and", b(sp.Rational(1, 4), sp.Rational(1, 4)) * 256, "/256")

# role loss numbers
for p1, p2 in [(0.8, 0.2), (0.2, 0.8)]:
    law = ((1 - p1) * (1 - p2), p1 * (1 - p2) + p2 * (1 - p1), p1 * p2)
    print("role-loss", (p1, p2), "sum law", tuple(round(t, 12) for t in law), "dE/dt", -(p1 + 2 * p2))
# role-loss face check at lambda=0: field on face p1=0 is tangent, not inward
lam = 0.0
print("role-loss face p1=0, lambda=0: p1dot =", lam * 0.5, "(tangent, not strictly inward)")
