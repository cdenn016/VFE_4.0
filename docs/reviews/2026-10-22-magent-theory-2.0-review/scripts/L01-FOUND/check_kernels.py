r"""L01-FOUND finite-alphabet checks of kernel covariance and record-kernel consistency.

1. prop:int-kl-covariance on finite alphabets: with K'(g_j y, A) = K(y, g_i^{-1} A) (eq:int-kernel-covariance)
   and passive relabelings g_i, g_j (permutations), nu_j' K' = (g_i)_# (nu_j K) and D_ij is unchanged.
2. prop:record-law vs def:registered-event: the record kernel induced (through the readout) by the
   state marginal of J equals the event kernel R when the registration condition holds.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python check_kernels.py
"""
from __future__ import annotations

import itertools

import numpy as np

rng = np.random.default_rng(11)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m]))) if np.all(q[m] > 0) else np.inf


def simplex(*shape: int) -> np.ndarray:
    x = rng.gamma(1.0, size=shape)
    return x / x.sum(-1, keepdims=True)


nj, ni = 4, 5
K = simplex(nj, ni)                 # K[y, z] = K(y, {z})
nu_j, nu_i = simplex(nj), simplex(ni)
pj, pi = rng.permutation(nj), rng.permutation(ni)   # g_j(y) = pj[y], g_i(z) = pi[z]
# K'(g_j y, {g_i z}) = K(y, {z})
Kp = np.zeros_like(K)
for y in range(nj):
    for z in range(ni):
        Kp[pj[y], pi[z]] = K[y, z]
nu_jp = np.zeros(nj); nu_jp[pj] = nu_j
nu_ip = np.zeros(ni); nu_ip[pi] = nu_i
lhs = nu_jp @ Kp
rhs = np.zeros(ni); rhs[pi] = nu_j @ K
print(f"prop:int-kl-covariance  nu_j' K' = (g_i)_# (nu_j K)   residual = {np.abs(lhs - rhs).max():.3e}")
print(f"prop:int-kl-covariance  D' - D                        residual = {abs(kl(nu_ip, lhs) - kl(nu_i, nu_j @ K)):.3e}")

# deterministic link kernel delta_{Theta y} with Theta' = g_i Theta g_j^{-1} satisfies eq:int-kernel-covariance
Theta = rng.integers(0, ni, size=nj)            # a (not necessarily injective) map E_j -> E_i
Thetap = np.empty(nj, dtype=int)
for y in range(nj):
    Thetap[pj[y]] = pi[Theta[y]]
Kd = np.zeros((nj, ni)); Kd[np.arange(nj), Theta] = 1
Kdp = np.zeros((nj, ni)); Kdp[np.arange(nj), Thetap] = 1
ok = all(Kdp[pj[y], pi[z]] == Kd[y, z] for y in range(nj) for z in range(ni))
print(f"link kernel delta_(Theta y) satisfies eq:int-kernel-covariance: {ok}")

# record kernel consistency: R(x, B) = Gamma(x, rho^{-1} B) with Gamma the state marginal of J
cores, obs = range(3), range(3)
states = list(itertools.product(cores, obs))
R = {x: simplex(3) for x in states}
A = {(x, o): ((x[0] * 2 + o) % 3, o) for x in states for o in obs}
worst = 0.0
for x in states:
    Gamma = {}
    for o in obs:
        xp = A[(x, o)]
        Gamma[xp] = Gamma.get(xp, 0.0) + R[x][o]
    induced = np.zeros(3)
    for xp, p in Gamma.items():
        induced[xp[1]] += p        # readout rho(x') = register
    worst = max(worst, np.abs(induced - R[x]).max())
print(f"prop:record-law induced kernel = event kernel R under registration  residual = {worst:.3e}")
