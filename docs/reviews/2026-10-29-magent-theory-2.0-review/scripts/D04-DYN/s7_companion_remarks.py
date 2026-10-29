r"""D04-DYN s7: rem:dyn-product, eq:comp-bernoulli-flow, product Fisher additivity, and the directed three-cycle.

(1) rem:dyn-product: under the joint Fisher flow with h = log(zeta/R) + h^q(k) + h^s(m), d/dt LOR = -(LOR - LOR(R)),
    for arbitrary (even time-dependent) marginal terms; integrate from a product law.
(2) eq:comp-bernoulli-flow (sympy).
(3) Fisher metric of a product law restricted to product directions is the sum of factor metrics.
(4) prop:dyn-nonreciprocal / companion c03:10: eigenvalues of the directed three-cycle, and whether the flow has a
    global Lyapunov function (symmetric part of the linear map).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

rng = np.random.default_rng(4)
ks = np.array([-1, -1, 1, 1])
ms = np.array([-1, 1, -1, 1])
R = rng.dirichlet(np.ones(4))
LOR = lambda z: float(np.sum(ks * ms * np.log(z)))


def rhs(t, y):
    z = np.exp(y)
    z /= z.sum()
    hq = np.sin(3 * t) * np.array([1.3, -0.4])[(ks + 1) // 2] + z[(ks == 1)].sum()          # arbitrary marginal terms
    hs = np.cos(2 * t) * np.array([-0.7, 2.1])[(ms + 1) // 2] + np.log(z[(ms == 1)].sum())
    h = np.log(z / R) + hq + hs
    return -(h - z @ h)


z0 = np.outer([0.3, 0.7], [0.6, 0.4]).ravel()          # product law, LOR = 0 (k major)
sol = solve_ivp(rhs, (0, 4), np.log(z0), rtol=1e-12, atol=1e-13, t_eval=[0, 0.5, 1, 2, 4])
dev = [abs(LOR(np.exp(sol.y[:, k]) / np.exp(sol.y[:, k]).sum()) - LOR(R) * (1 - np.exp(-t))) for k, t in enumerate(sol.t)]
print(f"(1) LOR(R) = {LOR(R):.6f}; max abs(LOR(zeta_t) - LOR(R)(1 - e^-t)) = {max(dev):.2e}")

p, p0 = sp.symbols("p p0", positive=True)
E = p * sp.log(p / p0) + (1 - p) * sp.log((1 - p) / (1 - p0))
l = sp.log(p / (1 - p))
l0 = sp.log(p0 / (1 - p0))
pdot = -p * (1 - p) * sp.diff(E, p)
print(f"(2) pdot + p(1-p)(l - l0) = {sp.simplify(pdot + p * (1 - p) * (l - l0))}; ldot + (l - l0) = {sp.simplify(sp.diff(l, p) * pdot + (l - l0))}")

q = rng.dirichlet(np.ones(3))
s = rng.dirichlet(np.ones(4))
vq = rng.normal(size=3)
vq -= vq.mean()
vs = rng.normal(size=4)
vs -= vs.mean()
P = np.outer(q, s)
dP = np.outer(vq, s) + np.outer(q, vs)
print(f"(3) joint Fisher norm {np.sum(dP**2 / P):.12f} vs sum of factor norms {np.sum(vq**2 / q) + np.sum(vs**2 / s):.12f}")

print("(4) directed three-cycle, (1+w) l_i' = -(l_i - rho_i) - w (l_i - l_{i+1})")
Sh = np.roll(np.eye(3), 1, axis=1)          # (S l)_i = l_{i+1}
for w in (0.5, 2.0, 10.0):
    Amat = (-(1 + w) * np.eye(3) + w * Sh) / (1 + w)
    ev = np.linalg.eigvals(Amat)
    pred = np.array([-1 / (1 + w), (-1 - 1.5 * w + 1j * np.sqrt(3) * w / 2) / (1 + w), (-1 - 1.5 * w - 1j * np.sqrt(3) * w / 2) / (1 + w)])
    err = max(min(abs(e - pp) for e in ev) for pp in pred)
    sym = np.linalg.eigvalsh(0.5 * (Amat + Amat.T))
    print(f"  w = {w}: eigenvalues {np.round(ev, 6)} (error vs stated {err:.1e}); eigenvalues of the symmetric part {np.round(sym, 6)}")
    print(f"     => d/dt |l - l*|^2 <= 2 * ({sym.max():.4f}) |l - l*|^2: |l - l*|^2 is a strict global Lyapunov function")
f1 = sp.lambdify((p, p0), pdot + p * (1 - p) * (l - l0), "numpy")
f2 = sp.lambdify((p, p0), sp.diff(l, p) * pdot + (l - l0), "numpy")
grid = np.linspace(0.01, 0.99, 50)
P_, P0_ = np.meshgrid(grid, grid)
print(f"(2, numeric on a 50x50 grid in (0,1)^2) max residuals {np.max(np.abs(f1(P_, P0_))):.1e}, {np.max(np.abs(f2(P_, P0_))):.1e}")

print("(5) per-agent Bernoulli dynamics with fixed rows and identity channels on random directed graphs")
from scipy.linalg import solve_continuous_lyapunov
worst_mu, worst_P, maxim = -np.inf, np.inf, 0.0
for trial in range(2000):
    N = rng.integers(2, 8)
    beta = np.zeros((N, N))
    for i in range(N):
        k = rng.integers(1, N)
        js = rng.choice([j for j in range(N) if j != i], size=min(k, N - 1), replace=False)
        w = rng.dirichlet(np.ones(len(js) + 1))
        beta[i, i] = w[0]                      # self (evaluator) weight, positive
        beta[i, js] = w[1:]
    # l_i' = -beta_ii (l_i - rho_i) - sum_j beta_ij (l_i - l_j)   (Bernoulli Fisher flow in log odds)
    A = -np.diag(beta.sum(axis=1)) + (beta - np.diag(np.diag(beta)))
    mu_inf = max(A[i, i] + np.sum(np.abs(A[i])) - abs(A[i, i]) for i in range(N))
    P = solve_continuous_lyapunov(A.T, -np.eye(N))
    worst_mu = max(worst_mu, mu_inf)
    worst_P = min(worst_P, np.linalg.eigvalsh(P).min())
    maxim = max(maxim, np.max(np.abs(np.linalg.eigvals(A).imag)))
print(f"  2000 graphs: max log-norm mu_inf(A) = {worst_mu:.4f} (< 0 means max_i |l_i - l*_i| is a strict Lyapunov function)")
print(f"  min eigenvalue of the Lyapunov solution P (A'P + PA = -I) = {worst_P:.4f} (> 0: quadratic Lyapunov function); max imaginary part of spectrum {maxim:.3f}")
