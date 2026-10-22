r"""L06-ERA script 4: two-agent binary phase example (eq:public-phase-binary*).

(a) Derive the fine equation from the original reduced potential (one-source rows)
    and compare with eq:public-phase-binary-fine on the invariant set q_i = s_i.
(b) Series of \dot m and \dot a (eq:public-phase-binary-expansion).
(c) Solve the order-3 invariance equation Dp(V) = f_b(p) (eq:public-phase-binary-parent).
(d) Compute the phase numerically, Pi = lim_t phi_{-t}(mean chart coordinate of Phi_t x),
    in the log-density chart theta = atanh x, and compare p - m with c m a^2.
(e) Response defect: uniform-child bit mean minus phase-law mean.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

J, m, a, x1, x2, y1, y2, eps = sp.symbols('J m a x1 x2 y1 y2 epsilon', real=True)


def d(x, z):
    return (1 + x) / 2 * sp.log((1 + x) / (1 + z)) + (1 - x) / 2 * sp.log((1 - x) / (1 - z))


psi = lambda t: d(t, 0)
E = (psi(x1) + psi(y1) - J * x1 * y1 + psi(x2) + psi(y2) - J * x2 * y2
     + d(x1, x2) + d(x2, x1) + d(y1, y2) + d(y2, y1))
xd1 = -(1 - x1**2) * sp.diff(E, x1)
claim1 = -(1 - x1**2) * (2 * sp.atanh(x1) - sp.atanh(x2) - J * x1) - (x1 - x2)
f1 = sp.lambdify((x1, x2, J), (xd1.subs({y1: x1, y2: x2}) - claim1))
res = max(abs(f1(u, v, 0.2)) for u in np.linspace(-.8, .8, 9) for v in np.linspace(-.8, .8, 9))
print('(a) fine equation residual on q_i=s_i:', f'{res:.2e}')
# invariance of q_i = s_i: dE/dx1 = dE/dy1 at y=x
inv = sp.simplify((sp.diff(E, x1) - sp.diff(E, y1)).subs({y1: x1, y2: x2}))
print('(a) dE/dx1 - dE/dy1 on q=s:', inv)

# (b) series
xdot = lambda u, v: -(1 - u**2) * (2 * sp.atanh(u) - sp.atanh(v) - J * u) - (u - v)
mdot = (xdot(m + a, m - a) + xdot(m - a, m + a)) / 2
adot = (xdot(m + a, m - a) - xdot(m - a, m + a)) / 2
fb = lambda t: -(1 - t**2) * (sp.atanh(t) - J * t)
s = sp.symbols('s')
ser_m = sp.expand(sp.series((mdot - fb(m)).subs({m: s * m, a: s * a}), s, 0, 5).removeO().subs(s, 1))
ser_a = sp.expand(sp.series(adot.subs({m: s * m, a: s * a}), s, 0, 3).removeO().subs(s, 1))
print('(b) mdot - f_b(m) through order 4:', sp.factor(ser_m))
print('(b) adot through order 2:', sp.factor(ser_a))

# (c) invariance equation at order 3
c = sp.symbols('c')
p = m + c * m * a**2
lhs = sp.diff(p, m) * mdot + sp.diff(p, a) * adot
rhs = fb(p)
diff3 = sp.expand(sp.series((lhs - rhs).subs({m: s * m, a: s * a}), s, 0, 4).removeO().subs(s, 1))
coef = sp.Poly(diff3, m, a).coeff_monomial(m * a**2)
csol = sp.solve(sp.Eq(coef, 0), c)[0]
print('(c) c solved:', sp.simplify(csol), ' claimed:', sp.simplify(3 * (2 - J) / (2 * (5 - J))),
      ' diff:', sp.simplify(csol - 3 * (2 - J) / (2 * (5 - J))))
print('(c) remaining order-3 residual after c:', sp.simplify(diff3.subs(c, csol)))

# (d) numerics
Jn = 0.2
cn = 3 * (2 - Jn) / (2 * (5 - Jn))


def fine(t, X):
    u, v = X
    return [-(1 - u**2) * (2 * np.arctanh(u) - np.arctanh(v) - Jn * u) - (u - v),
            -(1 - v**2) * (2 * np.arctanh(v) - np.arctanh(u) - Jn * v) - (v - u)]


def diag_back(t, th):
    return [th[0] - Jn * np.tanh(th[0])]  # backward-time diagonal flow in theta chart


def phase(mm, aa, tf=6.0):
    sol = solve_ivp(fine, (0, tf), [mm + aa, mm - aa], method='DOP853', rtol=1e-13, atol=1e-18)
    xe = sol.y[:, -1]
    thbar = np.mean(np.arctanh(xe))
    back = solve_ivp(diag_back, (0, tf), [thbar], method='DOP853', rtol=1e-13, atol=1e-20)
    return np.tanh(back.y[0, -1])


for mm in (0.08, 0.04, 0.02, 0.01):
    pp = phase(mm, mm)
    pp2 = phase(mm, mm, tf=8.0)
    print(f'(d) m=a={mm}: p={pp:.12e}, (p-m)/(m a^2)={(pp-mm)/mm**3:.6f}, c={cn:.6f}, '
          f'tf 6 vs 8 change {abs(pp-pp2):.1e}')
# (e) response defect
mm = 0.04
pp = phase(mm, mm)
print(f'(e) uniform-child bit mean m={mm}, phase-law mean p={pp:.8f}, defect={mm-pp:.3e}')
