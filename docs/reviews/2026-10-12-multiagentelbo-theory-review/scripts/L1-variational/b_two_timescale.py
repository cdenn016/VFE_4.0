r"""L1-variational check B: c08 two-timescale statements on a nonlinear Gaussian instance.

Generative: m ~ N(theta, 1), k | m ~ N(m, e^{-m}), o | k ~ N(k, 0.25), o = 1.3.
Recognition: Q = N(f1, e^{f2}) (x) N(s1, vm), vm = 0.3 fixed.
Fast f = (f1, f2) (belief mean, log variance); slow s = (s1, theta).
F(f, s) = -E_Q log p(o,k,m) + E_Q log q = -log p_theta(o) + KL(Q || Pi_theta).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize

f1, f2, s1, th = sp.symbols('f1 f2 s1 th', real=True)
o = sp.Rational(13, 10); so2 = sp.Rational(1, 4); vm = sp.Rational(3, 10); L2P = sp.log(2 * sp.pi)
vf = sp.exp(f2); Eem = sp.exp(s1 + vm / 2)
Fs = (L2P / 2 + ((s1 - th) ** 2 + vm) / 2
      + L2P / 2 - s1 / 2 + (Eem * ((f1 - s1 - vm) ** 2 + vm) + Eem * vf) / 2
      + sp.log(2 * sp.pi * so2) / 2 + ((o - f1) ** 2 + vf) / (2 * so2)
      - (1 + L2P + f2) / 2 - (1 + L2P + sp.log(vm)) / 2)
# sanity: E[e^m (k-m)^2] closed form against direct symbolic expectation is checked numerically below
fv, sv = [f1, f2], [s1, th]
F = sp.lambdify((fv, sv), Fs, 'numpy')
gf = sp.lambdify((fv, sv), [sp.diff(Fs, v) for v in fv], 'numpy')
gs = sp.lambdify((fv, sv), [sp.diff(Fs, v) for v in sv], 'numpy')
Hff = sp.lambdify((fv, sv), sp.hessian(Fs, fv), 'numpy')
Hfs = sp.lambdify((fv, sv), [[sp.diff(Fs, a, b) for b in sv] for a in fv], 'numpy')

# Monte Carlo sanity check of the closed form F at one point (independent of the algebra)
rng = np.random.default_rng(0)
fp, spt = np.array([0.4, -1.2]), np.array([0.2, -0.1])
k = rng.normal(fp[0], np.exp(fp[1] / 2), 2_000_000); m = rng.normal(spt[0], np.sqrt(0.3), 2_000_000)
logp = (-0.5 * np.log(2 * np.pi) - 0.5 * (m - spt[1]) ** 2
        - 0.5 * np.log(2 * np.pi) + 0.5 * m - 0.5 * np.exp(m) * (k - m) ** 2
        - 0.5 * np.log(2 * np.pi * 0.25) - (1.3 - k) ** 2 / 0.5)
logq = (-0.5 * np.log(2 * np.pi * np.exp(fp[1])) - (k - fp[0]) ** 2 / (2 * np.exp(fp[1]))
        - 0.5 * np.log(2 * np.pi * 0.3) - (m - spt[0]) ** 2 / 0.6)
mc = np.mean(logq - logp); se = np.std(logq - logp) / np.sqrt(len(k))
print("[0] closed-form F vs Monte Carlo:", float(F(fp, spt)), mc, "+/-", se)


def fstar(s, f0=None):
    f = np.array([1.0, -1.0]) if f0 is None else np.array(f0, float)
    for _ in range(100):
        step = np.linalg.solve(np.array(Hff(f, s), float), np.array(gf(f, s), float))
        f = f - step
        if np.abs(step).max() < 1e-14:
            break
    return f


def Fred(s):
    return F(fstar(s), s)


emax = 0.0; hmin = np.inf
for _ in range(20):
    s = rng.normal(size=2) * 0.7
    fs = fstar(s)
    h = 1e-5
    d1 = np.array([(Fred(s + h * e) - Fred(s - h * e)) / (2 * h) for e in np.eye(2)])
    d2 = np.array(gs(fs, s), float)
    emax = max(emax, np.abs(d1 - d2).max())
    hmin = min(hmin, np.linalg.eigvalsh(np.array(Hff(fs, s), float)).min())
print("[1] envelope |dF_red - d_sF(f*(s),s)| max over 20 points:", emax, "; min eig d2_ff F at f*:", hmin)


def rhs_eps(t, z, eps):
    f, s = z[:2], z[2:]
    return np.concatenate([-np.array(gf(f, s), float) / eps, -np.array(gs(f, s), float)])


def rhs_red(t, s):
    return -np.array(gs(fstar(s), s), float)


s0 = np.array([-0.5, 0.8]); f0 = np.array([0.0, 0.0]); T1 = 3.0
tt = np.linspace(0.0, T1, 61)
red = solve_ivp(rhs_red, (0, T1), s0, t_eval=tt, rtol=1e-10, atol=1e-12)
print("    reduced flow: s(T1) =", red.y[:, -1], "; F_red(s0) =", Fred(s0), "; F_red(s(T1)) =", Fred(red.y[:, -1]))
for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
    sol = solve_ivp(rhs_eps, (0, T1), np.concatenate([f0, s0]), t_eval=tt, args=(eps,),
                    method='Radau', rtol=1e-10, atol=1e-12)
    err_s = np.abs(sol.y[2:] - red.y).max()
    late = np.where(tt >= 0.5)[0]
    off = max(np.abs(sol.y[:2, j] - fstar(sol.y[2:, j])).max() for j in late)
    Fv = np.array([F(sol.y[:2, j], sol.y[2:, j]) for j in range(len(tt))])
    # predicted O(eps) lag of the fast variable: f - f* ~ -eps Hff^{-1} Df*(s) sdot ... report ratio off/eps
    print(f"[2] eps={eps:7.4f}: max|s_eps-s_red|={err_s:.3e} (/eps={err_s/eps:.3f}); "
          f"max_(t>=0.5)|f_eps-f*(s_eps)|={off:.3e} (/eps={off/eps:.3f}); F nonincreasing: {bool(np.all(np.diff(Fv) <= 1e-12))}")


def logev(t):
    g = lambda mm: (np.exp(-0.5 * (mm - t) ** 2) / np.sqrt(2 * np.pi)
                    * np.exp(-0.5 * (1.3 - mm) ** 2 / (np.exp(-mm) + 0.25)) / np.sqrt(2 * np.pi * (np.exp(-mm) + 0.25)))
    return np.log(quad(g, -15, 15, limit=400, epsabs=1e-14)[0])


r = minimize(Fred, s0, method='Nelder-Mead', options=dict(xatol=1e-11, fatol=1e-15, maxiter=5000))
rml = minimize(lambda t: -logev(t[0]), [0.5], method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14))
print("[3] product-family reduced minimizer (s1, theta) =", r.x, "; theta_ML =", rml.x[0],
      "; min F_red - min(-log p) =", r.fun - rml.fun)
