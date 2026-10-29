r"""D23-LEARN check 1b: sqrt(n) scaling of the streaming posterior-mean error (illustration).

Single reciprocal pair, K = 2, rank-one codes, isotropic anchor, lam = 1.  For several seeds the
error |mean_n - w_inf| (max norm) is reported with sqrt(n) * error; a bounded sqrt(n)*error is
consistent with the O(n^{-1/2}) fluctuation predicted by the proof of it1:learning(ii).
"""
import numpy as np
from d23_common import invisible_basis, loop_matrix, fisher_edge

K, m = 2, 4
codes = [np.array([[1.0, 0.5]]), np.array([[0.3, 1.0]])]
Z, Y = invisible_basis(codes, K)
B = loop_matrix([[0, 1]], 2, K)
s2, lam = 1.0, 1.0
Lam0 = np.eye(2 * m) / s2 + lam * B.T @ B
w_star = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.1, 0.6])
S = Z.T @ Lam0 @ Z
w_inf = w_star - Z @ np.linalg.solve(S, Z.T @ Lam0 @ w_star)
for seed in range(4):
    rng = np.random.default_rng(seed)
    out = []
    for n in [1000, 10000, 100000]:
        Lam = Lam0.copy(); eta = np.zeros(2 * m)
        for e in range(2):
            T = codes[e]
            yb = rng.standard_normal((n, K))
            we = w_star[e * m:(e + 1) * m].reshape(K, K)
            h = (T @ (np.eye(K) - we) @ yb.T).T + rng.standard_normal((n, 1))
            v = h - (T @ yb.T).T
            Lam[e * m:(e + 1) * m, e * m:(e + 1) * m] += np.kron(T.T @ T, yb.T @ yb)
            eta[e * m:(e + 1) * m] += -(T.T @ v.T @ yb).reshape(-1)
        err = np.abs(np.linalg.solve(Lam, eta) - w_inf).max()
        out.append(f"n={n}: err={err:.2e} sqrt(n)*err={np.sqrt(n) * err:.2f}")
    print(f"seed {seed}: " + "; ".join(out))
