r"""X2-FRAME spot checks of stated results (sympy/numpy, single thread).
1. inh:smooth-mixture-failure: xbar_dot at A=(0,c,0,c), B=(a,c,-a,c) equals Jc and Jc(1-a^2).
2. inh:smooth-boundary-failure: b(m+a,m-a)-b(m,m) = 3 m a^2/4.
3. eq:public-phase-binary-expansion: mdot = f_b(m) + (6-3J) m a^2 + O(5), adot = -(5-J) a + O(3),
   and parent coefficient c = 3(2-J)/(2(5-J)).
4. eq:boundary-response-derivatives and eq:boundary-observability-det.
5. fus:two-stages collision times; form:selection event times.
"""
import sympy as sp

x1, x2, y1, y2, J, a, c, m, n, b, eps, c1, c2 = sp.symbols('x1 x2 y1 y2 J a c m n b eps c1 c2', real=True)
psi = lambda x: (1 + x) / 2 * sp.log(1 + x) + (1 - x) / 2 * sp.log(1 - x)
f = lambda x: (x + x**3) / 4
d = lambda x, z: (1 + x) / 2 * sp.log((1 + x) / (1 + z)) + (1 - x) / 2 * sp.log((1 - x) / (1 - z))
E0 = lambda x, y: psi(x) + psi(y) - J * x * y
P = lambda u, v: d(u, f(v)) + d(v, f(u))
E = E0(x1, y1) + E0(x2, y2) + P(x1, x2) + P(y1, y2)
xd1 = -(1 - x1**2) * sp.diff(E, x1); xd2 = -(1 - x2**2) * sp.diff(E, x2)
xbar = (xd1 + xd2) / 2
vA = sp.simplify(xbar.subs({x1: 0, x2: 0, y1: c, y2: c}))
vB = sp.simplify(xbar.subs({x1: a, x2: -a, y1: c, y2: c}))
print("1. xbar(A) - Jc =", sp.simplify(vA - J * c), "; xbar(B) - Jc(1-a^2) =", sp.simplify(vB - J * c * (1 - a**2)))
print("   numeric at a=1/2,c=1/4,J=1/2:", [float(v.subs({a: sp.Rational(1, 2), c: sp.Rational(1, 4), J: sp.Rational(1, 2)})) for v in (vA, vB)])

bb = lambda u, v: (f(u) + f(v)) / 2
print("2. boundary defect - 3ma^2/4 =", sp.simplify(sp.expand(bb(m + a, m - a) - bb(m, m) - sp.Rational(3, 4) * m * a**2)))

# 3. binary identity-peer pair with q_i=s_i (ch. 09)
X1, X2 = sp.symbols('X1 X2')
fine = lambda xi, xj: -(1 - xi**2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * xi) - (xi - xj)
mdot = ((fine(X1, X2) + fine(X2, X1)) / 2).subs({X1: m + a, X2: m - a})
adot = ((fine(X1, X2) - fine(X2, X1)) / 2).subs({X1: m + a, X2: m - a})
t = sp.symbols('t')
ser_m = sp.series(mdot.subs({m: t * m, a: t * a}), t, 0, 5).removeO()
fb = -(1 - m**2) * (sp.atanh(m) - J * m)
ser_fb = sp.series(fb.subs(m, t * m), t, 0, 5).removeO()
print("3. mdot - f_b(m) - (6-3J) m a^2 through order 4:", sp.simplify(sp.expand(ser_m - ser_fb - (6 - 3 * J) * (t * m) * (t * a)**2)))
ser_a = sp.series(adot.subs({m: t * m, a: t * a}), t, 0, 2).removeO()
print("   adot + (5-J) a at order 1:", sp.simplify(ser_a + (5 - J) * t * a))
cc = sp.symbols('cc')
print("   parent coefficient solve (6-3J)-2(5-J)c=0:", sp.solve((6 - 3 * J) - 2 * (5 - J) * cc, cc), " vs 3(2-J)/(2(5-J))")

# 4. ch. 10 observability: F_x, F_y with general (x,y)
fx = lambda xi, xj, yi: -(1 - xi**2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * yi) - (xi - xj)
Fx = ((fx(X1, X2, sp.Symbol('Y1')) + fx(X2, X1, sp.Symbol('Y2'))) / 2).subs({X1: m + a, X2: m - a, sp.Symbol('Y1'): n + b, sp.Symbol('Y2'): n - b})
Fy = Fx.subs({m: n, n: m, a: b, b: a}, simultaneous=True)
claimed_Fx = -(1 - m**2) * (sp.atanh(m) - J * n) + (6 * m - J * n) * a**2 - 2 * J * m * a * b
diff = sp.series((Fx - claimed_Fx).subs({m: t * m, n: t * n, a: t * a, b: t * b}), t, 0, 5).removeO()
print("4. F_x - claimed through order 4:", sp.simplify(sp.expand(diff)))
# Jacobian of (m,n,Fx,Fy) in (m,n,a,b) is block lower triangular with identity
# upper-left block, so det = det d(Fx,Fy)/d(a,b). Use the verified degree-4 expansion.
Fxp = sp.expand(sp.series(Fx.subs({m: t * m, n: t * n, a: t * a, b: t * b}), t, 0, 5).removeO().subs(t, 1))
Fyp = Fxp.subs({m: n, n: m, a: b, b: a}, simultaneous=True)
blk = sp.Matrix([[sp.diff(Fxp, a), sp.diff(Fxp, b)], [sp.diff(Fyp, a), sp.diff(Fyp, b)]])
detJ = sp.expand(blk.det().subs({m: eps * c1, n: eps * c1, a: eps * c2, b: eps * c2}))
lead = sum(detJ.coeff(eps, k) * eps**k for k in range(0, 6))
print("   det - 4 eps^4 c1^2 c2^2 (6-3J)(6-J) at order <=5:", sp.simplify(sp.expand(lead - 4 * eps**4 * c1**2 * c2**2 * (6 - 3 * J) * (6 - J))))

# 5. event times
import math
kap, lam, dd = 0.7, 1.3, 0.9
z = [-3 * dd, -2 * dd, 2 * dd, 3 * dd]
tau1 = dd / (2 * lam); tau2 = tau1 + 3 * dd / (4 * lam)
print("5. fusion t1,t2:", math.log(1 + kap * tau1) / kap, math.log(1 + kap * 5 * dd / (4 * lam)) / kap,
      " formula:", math.log(1 + kap * dd / (2 * lam)) / kap, math.log(1 + 5 * kap * dd / (4 * lam)) / kap)
