"""Ch.09 eq:public-phase-binary-expansion and eq:public-phase-binary-parent.

(1) sympy: expansion of the two-agent binary fine field at the reference.
(2) numerics: phase parent p = Pi(x) via the defining limit, computed in the
    log-density (logit) chart, compared with m + c m a^2.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# (1) symbolic expansion
eps, M, A, J = sp.symbols('epsilon M A J', real=True)
def xdot(xi, xj):
    return -(1 - xi**2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * xi) - (xi - xj)
fb = lambda x: -(1 - x**2) * (sp.atanh(x) - J * x)
x1, x2 = eps * (M + A), eps * (M - A)
mdot = (xdot(x1, x2) + xdot(x2, x1)) / 2
adot = (xdot(x1, x2) - xdot(x2, x1)) / 2
res_m = sp.series(mdot - fb(eps * M), eps, 0, 5).removeO()
res_a = sp.series(adot, eps, 0, 3).removeO()
print('mdot - f_b(m) to order 4:', sp.factor(sp.expand(res_m)))
print('adot to order 2:', sp.factor(sp.expand(res_a)))
print('check (6-3J) m a^2 residual:', sp.simplify(sp.expand(res_m) - (6 - 3 * J) * M * A**2 * eps**3))

# (2) numerical phase parent
def phase_parent(m, a, Jv, T=6.0):
    def fine(t, th):
        x = np.tanh(th)
        out = np.empty(2)
        for i, j in ((0, 1), (1, 0)):
            xd = -(1 - x[i]**2) * (2 * th[i] - th[j] - Jv * x[i]) - (x[i] - x[j])
            out[i] = xd / (1 - x[i]**2)          # logit chart
        return out
    th0 = np.arctanh([m + a, m - a])
    sol = solve_ivp(fine, (0, T), th0, method='DOP853', rtol=1e-13, atol=1e-16)
    zbar = sol.y[:, -1].mean()
    diag = lambda t, th: -(th - Jv * np.tanh(th))   # diagonal flow in logit chart
    back = solve_ivp(lambda t, th: -diag(t, th), (0, T), [zbar], method='DOP853', rtol=1e-13, atol=1e-16)
    return np.tanh(back.y[0, -1])

for Jv in (0.1, 0.2):
    c = 3 * (2 - Jv) / (2 * (5 - Jv))
    for s in (0.04, 0.02, 0.01):
        m = a = s
        p = phase_parent(m, a, Jv)
        p2 = phase_parent(m, a, Jv, T=8.0)
        print(f'J={Jv} m=a={s}: (p-m)/(m a^2)={(p - m) / (m * a * a):.6f}  claimed c={c:.6f}  T-stability={abs(p - p2):.2e}')
