"""D22-COMM s06: extension checks (not claims of the manuscript).

  (a) finite-alphabet softmax code P(h = c | y) = softmax(W theta^{-1} y)_c: the Fisher kernel on gl(K) equals
      {delta : C W delta = 0} with C = I - 11^T/m the centering matrix, so the code behaves like a location code
      of rank rank(CW) <= m - 1.
  (b) code-shaped links: for theta near I with rank(theta - I) <= r, the principal logarithm has rank <= r,
      so near the identity the links e^{OT} are exactly the elements with rank(theta - I) <= r.
"""
import numpy as np
from scipy.linalg import expm, logm, null_space

rng = np.random.default_rng(13)


def softmax_fisher(W, theta, Y):
    m, K = W.shape
    thinv = np.linalg.inv(theta)
    basis = []
    for i in range(K):
        for j in range(K):
            E = np.zeros((K, K)); E[i, j] = 1.0; basis.append(E)
    n = len(basis)
    F = np.zeros((n, n))
    Z = W @ thinv @ Y
    P = np.exp(Z - Z.max(0)); P /= P.sum(0)
    for t in range(Y.shape[1]):
        p = P[:, t]
        Cov = np.diag(p) - np.outer(p, p)
        D = np.array([-(W @ Bm @ thinv @ Y[:, t]) for Bm in basis]).T  # m x n, derivative of logits
        F += D.T @ Cov @ D
    return F / Y.shape[1], basis


K = 3
theta = expm(0.3 * rng.standard_normal((K, K)))
Y = rng.standard_normal((K, 300))
for m in [2, 3, 4]:
    W = rng.standard_normal((m, K))
    F, basis = softmax_fisher(W, theta, Y)
    ev = np.linalg.eigvalsh(F)
    kdim = int(np.sum(ev < 1e-10 * ev.max()))
    Cm = np.eye(m) - np.ones((m, m)) / m
    rc = np.linalg.matrix_rank(Cm @ W)
    kerF = null_space(F, rcond=1e-10)
    resid = max([np.abs(Cm @ W @ kerF[:, k].reshape(K, K)).max() for k in range(kerF.shape[1])] or [0.0])
    print(f"(a) m={m} symbols, K={K}: dim ker F = {kdim}, predicted K(K-rank(CW)) = {K * (K - rc)} with rank(CW) = {rc};"
          f" max abs(C W delta) over kernel = {resid:.1e}")

# (b) rank of principal log for low-rank perturbations of I
K, r = 5, 2
worst = 0
for _ in range(50):
    U = rng.standard_normal((K, r)); V = rng.standard_normal((r, K))
    th = np.eye(K) + 0.2 * U @ V / np.linalg.norm(U @ V, 2)
    L = np.real(logm(th))
    sv = np.linalg.svd(L, compute_uv=False)
    worst = max(worst, sv[r] / sv[0])
print("(b) max over 50 draws of sigma_{r+1}/sigma_1 of log(I + rank-2 perturbation) =", worst)
