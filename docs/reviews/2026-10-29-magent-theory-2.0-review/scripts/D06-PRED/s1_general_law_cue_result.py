r"""D06-PRED s1: exact finite (non-Gaussian) instance of the general-law cue/result model.

Checks, on finite alphabets with random normalized kernels (no Gaussian structure):
  cmem:model / cmem:joint       normalization of P = r mu K L
  cmem:posterior                w^o(m) Pi_m^o(u,v) equals the Bayes posterior
  cmem:free-energy              F(Q;o) = KL(s||w) + sum_m s_m KL(q_m||Pi_m) - log Z(o)
                                for random correlated Q = s(m) q_m(u,v); the separated
                                ELBO form KL(s||r) + sum_m s_m[KL(q_m||G_m) - E log l] agrees
  minimum statement             F >= -log Z, equality at the posterior
  cmem:gibbs                    s_a propto r_a exp(-f_a) minimizes F at fixed q_m; equals w at q_m = Pi_m
  cmem:product-label            structured family s_a q_X(u) K_a(u,v): result conditional cancels,
                                label update and q_X update are coordinate minimizers
  cmem:result-model             result-only representation l_Y is normalized, gives the same
                                (M,Y,O) marginal and the same result posterior
Run: OMP_NUM_THREADS=1 ... timeout 30 python s1_general_law_cue_result.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(20261029)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL(p||q) = sum p log(p/q) on a finite alphabet, +inf if p is not << q."""
    p = np.asarray(p, float).ravel()
    q = np.asarray(q, float).ravel()
    m = p > 0
    if np.any(q[m] <= 0):
        return float("inf")
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def rand_simplex(*shape: int, alpha: float = 1.0) -> np.ndarray:
    """Random normalized rows along the last axis."""
    x = rng.gamma(alpha, size=shape)
    return x / x.sum(axis=-1, keepdims=True)


nM, nX, nY, nO = 3, 4, 3, 5
r = rand_simplex(nM)                      # proper presentation reference r(m)
mu = rand_simplex(nM, nX)                 # cue kernel mu_m(u)
K = rand_simplex(nM, nX, nY)              # result kernel K_m(u, v)
Lrec = rand_simplex(nX, nO)               # record kernel L(u, o) = l(o|u) (counting measure)

# joint P(m,u,v,o) = r(m) mu_m(u) K_m(u,v) L(u,o)      shape (nM, nX, nY, nO)
P = r[:, None, None, None] * mu[:, :, None, None] * K[:, :, :, None] * Lrec[None, :, None, :]
print("normalization |sum P - 1| =", abs(P.sum() - 1.0))

o = 2
l_o = Lrec[:, o]                                           # l(o|u), shape (nX,)
Zm = (mu * l_o[None, :]).sum(axis=1)                       # Z_m(o)
Z = float((r * Zm).sum())                                  # Z(o)
w = r * Zm / Z                                             # w^o(m)
Pi = (l_o[None, :, None] * mu[:, :, None] * K) / Zm[:, None, None]   # Pi_m^o(u,v)
zeta = w[:, None, None] * Pi                                # w Pi
bayes = P[:, :, :, o] / P[:, :, :, o].sum()
print("posterior formula residual max|w Pi - Bayes| =", np.abs(zeta - bayes).max())
print("evidence residual |Z - P(o)| =", abs(Z - P[:, :, :, o].sum()))

# generative factor rG(m,u,v) = r(m) mu_m(u) K_m(u,v)
rG = r[:, None, None] * mu[:, :, None] * K
G = mu[:, :, None] * K                                       # G_m(u,v)


def free_energy_def(Q: np.ndarray) -> float:
    r"""F(Q;o) = E_Q[log dQ/d(rG) - log l(o|u)] (logarithmic density ratio to the unnormalized joint)."""
    m = Q > 0
    unnorm = rG * l_o[None, :, None]
    if np.any(unnorm[m] <= 0):
        return float("inf")
    return float(np.sum(Q[m] * (np.log(Q[m]) - np.log(unnorm[m]))))


def free_energy_split(s: np.ndarray, q: np.ndarray) -> float:
    """KL(s||w) + sum_m s_m KL(q_m||Pi_m) - log Z."""
    return kl(s, w) + sum(s[a] * kl(q[a], Pi[a]) for a in range(nM) if s[a] > 0) - np.log(Z)


def free_energy_elbo(s: np.ndarray, q: np.ndarray) -> float:
    """KL(s||r) + sum_m s_m [KL(q_m||G_m) - E_{q_m} log l(o|u)]."""
    tot = kl(s, r)
    for a in range(nM):
        if s[a] > 0:
            Elog = float(np.sum(q[a] * np.log(l_o)[:, None]))
            tot += s[a] * (kl(q[a], G[a]) - Elog)
    return tot


res_split, res_elbo, gaps = [], [], []
for _ in range(200):
    s = rand_simplex(nM)
    q = rand_simplex(nM, nX * nY).reshape(nM, nX, nY)       # correlated: q depends on m
    Q = s[:, None, None] * q
    F0 = free_energy_def(Q)
    res_split.append(abs(F0 - free_energy_split(s, q)))
    res_elbo.append(abs(F0 - free_energy_elbo(s, q)))
    gaps.append(F0 + np.log(Z))
print("cmem:free-energy-split max residual over 200 random correlated Q =", max(res_split))
print("separated ELBO form max residual =", max(res_elbo))
print("min over random Q of F + log Z (must be > 0) =", min(gaps))
print("F(posterior) + log Z =", free_energy_def(zeta) + np.log(Z))

# cmem:gibbs: fix random content conditionals q_a, optimize s
q = rand_simplex(nM, nX * nY).reshape(nM, nX, nY)
f = np.array([kl(q[a], G[a]) - float(np.sum(q[a] * np.log(l_o)[:, None])) for a in range(nM)])
s_star = r * np.exp(-f)
s_star /= s_star.sum()
F_star = free_energy_elbo(s_star, q)
worse = min(free_energy_elbo(rand_simplex(nM), q) - F_star for _ in range(500))
print("cmem:gibbs: min over 500 random s of F(s) - F(s*) (must be >= 0) =", worse)
f_exact = np.array([kl(Pi[a], G[a]) - float(np.sum(Pi[a] * np.log(l_o)[:, None])) for a in range(nM)])
print("cmem:gibbs: max|f_a(Pi_a) + log Z_a| =", np.abs(f_exact + np.log(Zm)).max())
s_exact = r * np.exp(-f_exact)
s_exact /= s_exact.sum()
print("cmem:gibbs: Gibbs weights at exact conditionals vs w, max diff =", np.abs(s_exact - w).max())

# cmem:product-label: structured family s_a q_X(u) K_a(u,v)
qX = rand_simplex(nX)
s = rand_simplex(nM)
Qs = s[:, None, None] * qX[None, :, None] * K
F_struct = free_energy_def(Qs)
F_struct_formula = kl(s, r) + sum(s[a] * kl(qX, mu[a]) for a in range(nM)) - float(np.sum(qX * np.log(l_o)))
print("structured family: F residual vs KL(s||r)+sum s KL(qX||mu_a) - E log l =", abs(F_struct - F_struct_formula))
s_lab = r * np.exp(-np.array([kl(qX, mu[a]) for a in range(nM)]))
s_lab /= s_lab.sum()
F_lab = free_energy_def(s_lab[:, None, None] * qX[None, :, None] * K)
worse = min(free_energy_def(ss[:, None, None] * qX[None, :, None] * K) - F_lab
            for ss in (rand_simplex(nM) for _ in range(500)))
print("cmem:product-label: min over random s of F - F(label update) (>= 0) =", worse)
qX_new = l_o * np.exp((s_lab[:, None] * np.log(mu)).sum(axis=0))
qX_new /= qX_new.sum()
F_q = free_energy_def(s_lab[:, None, None] * qX_new[None, :, None] * K)
worse = min(free_energy_def(s_lab[:, None, None] * qq[None, :, None] * K) - F_q
            for qq in (rand_simplex(nX) for _ in range(500)))
print("q_X update: min over random q_X of F - F(update) (>= 0) =", worse)
print("structured optimum vs exact minimum (F_struct - (-log Z), >= 0) =", F_q + np.log(Z))

# cmem:result-model: result-only representation
GY = (mu[:, :, None] * K).sum(axis=1)                       # G_Y(m, v)
R = (mu[:, :, None] * K) / GY[:, None, :]                   # R_m(v, u) stored as [m, u, v]
lY = np.einsum("muv,uo->mvo", R, Lrec)                      # l_Y(v, m; o) stored [m, v, o]
print("l_Y normalized: max|sum_o l_Y - 1| =", np.abs(lY.sum(axis=2) - 1).max())
marg_full = P.sum(axis=1)                                   # (m, v, o)
marg_res = r[:, None, None] * GY[:, :, None] * lY
print("(M,Y,O) marginal residual =", np.abs(marg_full - marg_res).max())
post_res = marg_res[:, :, o] / marg_res[:, :, o].sum()
print("result posterior (M,Y) residual =", np.abs(post_res - zeta.sum(axis=1)).max())

# alternate the structured label and q_X updates to a fixed point (coordinate descent on the structured family)
s_c, q_c = rand_simplex(nM), rand_simplex(nX)
F_hist = []
for _ in range(500):
    s_c = r * np.exp(-np.array([kl(q_c, mu[a]) for a in range(nM)]))
    s_c /= s_c.sum()
    q_c = l_o * np.exp((s_c[:, None] * np.log(mu)).sum(axis=0))
    q_c /= q_c.sum()
    F_hist.append(free_energy_def(s_c[:, None, None] * q_c[None, :, None] * K))
print("structured coordinate descent: monotone =", bool(np.all(np.diff(F_hist) <= 1e-13)),
      "; converged gap F - (-log Z) =", F_hist[-1] + np.log(Z), "; last change =", abs(F_hist[-1] - F_hist[-2]))
