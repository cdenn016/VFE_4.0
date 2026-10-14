"""E7: check the information-form Gaussian evidence identity against the moment form,
using the log normalizer A(h, Lam) = 1/2 h^T Lam^{-1} h - 1/2 logdet Lam + n/2 log 2pi
(eq:gauss-lognorm), and check that the bound at the block optimum equals log p(o) minus
the determinant gap (thm:restrict-determinant-gap)."""
import numpy as np
from scipy.stats import multivariate_normal

rng = np.random.default_rng(0)

def A(h, L):
    n = len(h)
    return 0.5 * h @ np.linalg.solve(L, h) - 0.5 * np.linalg.slogdet(L)[1] + 0.5 * n * np.log(2 * np.pi)

n, m = 4, 3
M = rng.normal(size=(n + m, n + m))
Lh = M @ M.T + 0.5 * np.eye(n + m)           # joint precision, ordering (Y, o)
hh = rng.normal(size=n + m)
LYY, LYo, Loo = Lh[:n, :n], Lh[:n, n:], Lh[n:, n:]
hY, ho = hh[:n], hh[n:]
o = rng.normal(size=m)

logp_info = ho @ o - 0.5 * o @ Loo @ o + A(hY - LYo @ o, LYY) - A(hh, Lh)
Sig = np.linalg.inv(Lh); mu = Sig @ hh
logp_moment = multivariate_normal(mu[n:], Sig[n:, n:]).logpdf(o)
print("evidence identity residual", abs(logp_info - logp_moment), "scale", abs(logp_moment))

# block optimum bound: blocks {0},{1,2},{3} of Y
L = LYY; h = hY - LYo @ o; muY = np.linalg.solve(L, h)
blocks = [[0], [1, 2], [3]]
SigQ = np.zeros((n, n))
for b in blocks:
    SigQ[np.ix_(b, b)] = np.linalg.inv(L[np.ix_(b, b)])
# ELBO = E_Q log p(y, o) + H(Q), with log p(y,o) = hh^T x - 1/2 x^T Lh x - A(hh, Lh)
x_mean = np.concatenate([muY, o])
Cx = np.zeros((n + m, n + m)); Cx[:n, :n] = SigQ
E_logjoint = hh @ x_mean - 0.5 * (x_mean @ Lh @ x_mean + np.trace(Lh @ Cx)) - A(hh, Lh)
H_Q = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * SigQ)[1]
elbo = E_logjoint + H_Q
gap = 0.5 * (sum(np.linalg.slogdet(L[np.ix_(b, b)])[1] for b in blocks) - np.linalg.slogdet(L)[1])
print("log p - ELBO* - gap residual", abs(logp_info - elbo - gap), "gap", gap)
