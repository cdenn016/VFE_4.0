r"""GAP01-S3 s1: explicit constants for pa:persistent-grouping (chapters/13_per_agent_inheritance.tex:138-172).

Exact Gibbs-reduced per-agent log-odds flow (pa:bit-flow) for three Bernoulli agents,
    dl_i/dt = -l_i + \sum_{j\ne i} \beta_{ij}(l) l_j + \epsilon o_i,   o = (1,1,-1),
rows \beta_{ij} \propto \pi_{ij} e^{-D_{ij}}, D_{ij} = KL(Bern(l_i)||Bern(l_j)), D_{ii} = KL(Bern(l_i)||Bern(0)),
\pi_{ii} = 1/5, \pi_{ij} = 2/5.  Scaled u = l/\epsilon satisfies du/dt = M u + o + R_\epsilon(u),
R_{\epsilon,i}(u) = \sum_{j\ne i}(\beta_{ij}-\pi_{ij}) u_j.

Rigorous ingredients (proved in the report, evaluated here):
  KL(Bern(l)||Bern(l')) <= (l-l')^2/8;  |beta_ij - pi_ij| <= pi_ij (e^{Dmax_i} - 1);
  M symmetric with largest eigenvalue -1/5, so d|u-u_*|/dt <= -|u-u_*|/5 + |R|.
Outputs: (a) exact M, u_*, spectrum and the O(eps^2) coefficient w of u_eps - u_*;
(b) numerical u_eps; (c) explicit eps_0 and entrance time T from the Lyapunov/shadowing bound;
(d) exact-flow entrance times for comparison (floating point).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

np.set_printoptions(precision=6, suppress=False)
A_PR, P_PR = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])

# (a) exact constants -------------------------------------------------------------------------
Ms = -sp.Rational(7, 5) * sp.eye(3) + sp.Rational(2, 5) * sp.ones(3, 3)
os_ = sp.Matrix([1, 1, -1])
us = -Ms.inv() * os_
print("(a) u_* =", list(us), " eigenvalues of M:", Ms.eigenvals())
# second-order expansion: D_ij = eps^2 (u_i-u_j)^2/8, D_ii = eps^2 u_i^2/8
u = sp.symbols("u1:4")
pi = [[sp.Rational(1, 5) if i == j else sp.Rational(2, 5) for j in range(3)] for i in range(3)]
D2 = [[(u[i] ** 2 / 8 if i == j else (u[i] - u[j]) ** 2 / 8) for j in range(3)] for i in range(3)]
R2 = []
for i in range(3):
    mean_i = sum(pi[i][k] * D2[i][k] for k in range(3))
    R2.append(sum(pi[i][j] * (-D2[i][j] + mean_i) * u[j] for j in range(3) if j != i))
R2v = sp.Matrix(R2).subs({u[k]: us[k] for k in range(3)})
w = sp.simplify(-Ms.inv() * R2v)
print("    u_eps = u_* + eps^2 w + O(eps^4), w =", list(w), "=", [float(x) for x in w],
      " |w| =", float(sp.sqrt(sum(x ** 2 for x in w))))

def kl_bern(li, lj):
    return (li - lj) * np.tanh(li / 2) / 2 - np.log(np.cosh(li / 2)) + np.log(np.cosh(lj / 2))

def rows(l):
    D = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            D[i, j] = kl_bern(l[i], 0.0) if i == j else kl_bern(l[i], l[j])
    W = np.array([[A_PR if i == j else P_PR for j in range(3)] for i in range(3)]) * np.exp(-D)
    return W / W.sum(axis=1, keepdims=True)

def field_u(uu, eps):
    l = eps * uu
    b = rows(l)
    return np.array([-uu[i] + sum(b[i, j] * uu[j] for j in range(3) if j != i) + o[i] for i in range(3)])

# (b) numerical equilibria ---------------------------------------------------------------------
ustar = np.array([15 / 7, 15 / 7, 5 / 7])
wf = np.array([float(x) for x in w])
print("(b) numerical u_eps (fsolve on the exact scaled field):")
for eps in (0.2, 0.1, 0.05, 0.025):
    ue = fsolve(lambda x: field_u(x, eps), ustar, xtol=1e-14)
    print("    eps=%.3f |u_eps-u_*|=%.4e  (u_eps-u_*)/eps^2=%s  vs w=%s  res=%.1e"
          % (eps, np.linalg.norm(ue - ustar), (ue - ustar) / eps ** 2, wf, np.abs(field_u(ue, eps)).max()))

# (c) explicit constants ----------------------------------------------------------------------
uin = np.array([4.0, 0.5, -1.0])
e0 = uin - ustar
c1 = np.full(3, e0.mean()); cp = e0 - c1
print("(c) |u_in-u_*| = %.6f, common part %s, difference part %s" % (np.linalg.norm(e0), c1, cp))

def Rbound(rho, eps):
    """Rigorous sup of |R_eps(u)|_2 over the Euclidean ball B(u_*, rho)."""
    Rs = []
    for i in range(3):
        Ui = abs(ustar[i]) + rho
        Di = max([Ui] + [abs(ustar[i] - ustar[j]) + np.sqrt(2) * rho for j in range(3) if j != i])
        Dmax = eps ** 2 * Di ** 2 / 8
        sum_other = sum(abs(ustar[j]) for j in range(3) if j != i) + np.sqrt(2) * rho
        Rs.append(P_PR * np.expm1(Dmax) * sum_other)
    return float(np.sqrt(np.sum(np.square(Rs))))

# constraints for edges (1,2) and (2,3): n.u > 0
NS = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [4, -1, 0], [-1, 4, 0], [0, 4, -1], [0, -1, 4])]

def margin(n, t):
    ub = ustar + np.exp(-t / 5) * c1 + np.exp(-7 * t / 5) * cp
    return n @ ub / np.linalg.norm(n)

def entrance_time(r0, sh):
    """Smallest T such that for all t >= T every constraint margin exceeds e^{-t/5} r0 + sh."""
    T = 0.0
    for n in NS:
        a1, ap = n @ c1, n @ cp
        if a1 <= 0 and ap <= 0:  # margin increasing in t, bound decreasing: single crossing
            f = lambda t: margin(n, t) - (np.exp(-t / 5) * r0 + sh)
            if f(200.0) <= 0:
                return np.inf
            lo, hi = 0.0, 200.0
            if f(0.0) > 0:
                continue
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) <= 0 else (lo, mid)
            T = max(T, hi)
        else:  # rigorous lower bound of the margin for all t >= 0
            lb = (n @ ustar + min(0, a1) + min(0, ap)) / np.linalg.norm(n)
            if lb <= r0 + sh:
                return np.inf
    return T

asym = (4 * 5 / 7 - 15 / 7) / np.sqrt(17)
print("    binding asymptotic margin of 4u_3-u_2>0 at u_*: %.6f" % asym)
print("    initial no-edge margin of 4u_2-u_1<0 at u_in: %.6f (r0 must be smaller)" % (2 / np.sqrt(17)))
for r0 in (0.05, 0.2):
    rho0 = np.linalg.norm(e0) + r0
    # largest eps with ball invariance and 5 Rbar <= asym/2
    eps_inv = max(e for e in np.linspace(1e-4, 0.5, 50000) if Rbound(rho0, e) < rho0 / 5)
    eps_sel = max(e for e in np.linspace(1e-4, 0.5, 50000) if 5 * Rbound(rho0, e) <= asym / 2)
    print("  r0=%.2f rho0=%.4f: ball B(u_*,rho0) invariant for eps<=%.4f; 5*Rbar<=asym/2 for eps<=%.4f"
          % (r0, rho0, eps_inv, eps_sel))
    for eps in (0.001, 0.005, 0.01, 0.02, eps_sel):
        Rb = Rbound(rho0, eps)
        T = entrance_time(r0, 5 * Rb)
        print("      eps=%.4f Rbar=%.3e shadow=5Rbar=%.3e  explicit entrance time T=%.4f" % (eps, Rb, 5 * Rb, T))

# (d) exact-flow entrance times ---------------------------------------------------------------
def full_block(uu):
    def edge(i, j):
        if uu[i] * uu[j] <= 0:
            return False
        return min(abs(uu[i]), abs(uu[j])) / max(abs(uu[i]), abs(uu[j])) > 0.25
    E = [edge(0, 1), edge(0, 2), edge(1, 2)]
    return sum(E) >= 2

print("(d) exact flow from u_in (DOP853, rtol 1e-11): last time the full block is absent, t in [0,60]")
tt = np.linspace(0, 60, 60001)
for eps in (0.01, 0.02, 0.05, 0.1, 0.3):
    sol = solve_ivp(lambda t, x: field_u(x, eps), (0, 60), uin, t_eval=tt, method="DOP853", rtol=1e-11, atol=1e-13)
    fb = np.array([full_block(sol.y[:, k]) for k in range(len(tt))])
    last_absent = tt[np.where(~fb)[0].max()] if (~fb).any() else 0.0
    dev = np.abs(sol.y[:, -1] - ustar).max()
    print("    eps=%.2f: full block from t=%.3f on (persistent to t=60: %s); |u(60)-u_*|_inf=%.3e"
          % (eps, last_absent, bool(fb[np.where(tt > last_absent)[0]].all()), dev))
# limit-flow entrance time (eps=0)
lim = [t for t in tt if full_block(ustar + np.exp(-t / 5) * c1 + np.exp(-7 * t / 5) * cp)]
print("    eps=0 limit trajectory: full block from t=%.4f" % lim[0])
