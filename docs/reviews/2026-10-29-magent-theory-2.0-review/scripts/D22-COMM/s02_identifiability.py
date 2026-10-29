"""D22-COMM s02: what messages reveal about the dictionary (chapters/17_communication_links.tex:130-165).

Checks:
  (a) it1:identifiability (ii): Fisher form F(delta,delta) = Tr(R^-1 T delta C delta^T T^T) against a
      finite-difference Jacobian of the mean map delta -> T e^{-delta} theta^{-1} y, averaged over referents;
      rank = K r and kernel = {delta : T delta = 0} for GL(K).
  (b) it1:identifiability (i),(iii): a stabilizer element s (T s = T) leaves the effective code unchanged;
      dim Stab = K(K-r).
  (c) single operating content (M = c c^T) versus a full-rank grounded referent experiment: Fisher ranks,
      and the companion example T = I_2, c = e_1, delta = E_22.
  (d) orthogonal dictionaries: dimension of {delta in so(K) : T delta = 0} = (K-r)(K-r-1)/2,
      so the identifiable dimension in SO(K) is r(2K-r-1)/2 (extension check, not in the manuscript).
  (e) it1:location: location Fisher matrix of Student-t noise exceeds the inverse covariance (Cramer-Rao),
      univariate closed form J = (nu+1)/(nu+3) versus 1/Var = (nu-2)/nu.
  (f) known-code hypothesis: if the reference code is also unknown, (Tbar, theta) and (Tbar g, theta g) give the
      same effective code for every g in GL(K), so only the effective code is identified.
"""
import numpy as np
from scipy.linalg import expm, null_space

rng = np.random.default_rng(7)


def spd(K):
    X = rng.standard_normal((K, K))
    return X @ X.T + 0.5 * np.eye(K)


def basis_gl(K):
    B = []
    for i in range(K):
        for j in range(K):
            E = np.zeros((K, K)); E[i, j] = 1.0; B.append(E)
    return B


def fisher_formula(T, theta, M, R, basis):
    C = np.linalg.inv(theta) @ M @ np.linalg.inv(theta).T
    Ri = np.linalg.inv(R)
    n = len(basis)
    F = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            F[a, b] = np.trace(Ri @ T @ basis[a] @ C @ basis[b].T @ T.T)
    return 0.5 * (F + F.T)


def fisher_fd(T, theta, Y, R, basis, eps=1e-6):
    """Average over referent sample Y (columns) of J^T R^-1 J with J the FD derivative of the mean map."""
    Ri = np.linalg.inv(R)
    thinv = np.linalg.inv(theta)
    n = len(basis)
    F = np.zeros((n, n))
    cols = []
    for Bm in basis:
        plus = T @ expm(-eps * Bm) @ thinv @ Y
        minus = T @ expm(eps * Bm) @ thinv @ Y
        cols.append((plus - minus) / (2 * eps))
    N = Y.shape[1]
    for a in range(n):
        for b in range(n):
            F[a, b] = np.sum(cols[a] * (Ri @ cols[b])) / N
    return F


for (K, r) in [(3, 2), (5, 2)]:
    T = rng.standard_normal((r, K))
    theta = expm(0.4 * rng.standard_normal((K, K)))
    R = spd(r) * 0.5
    L = np.linalg.cholesky(spd(K))
    Y = L @ rng.standard_normal((K, 400))
    M_emp = Y @ Y.T / Y.shape[1]
    basis = basis_gl(K)
    F1 = fisher_formula(T, theta, M_emp, R, basis)
    F2 = fisher_fd(T, theta, Y, R, basis)
    ev = np.linalg.eigvalsh(F1)
    rank = int(np.sum(ev > 1e-9 * ev.max()))
    print(f"(a) K={K} r={r}: max abs(F formula - F finite diff) = {np.abs(F1 - F2).max():.3e} (scale {np.abs(F1).max():.3e});"
          f" rank F = {rank}, K r = {K * r}, K^2 = {K * K}")
    # kernel equals {delta : T delta = 0}
    kerF = null_space(F1, rcond=1e-9)
    ok = max(np.abs(T @ kerF[:, k].reshape(K, K)).max() for k in range(kerF.shape[1]))
    print(f"(a)   dim ker F = {kerF.shape[1]}, K(K-r) = {K * (K - r)}, max abs(T delta) over kernel basis = {ok:.2e}")
    # (b) stabilizer coset
    U = null_space(T)  # K x (K-r)
    s = np.eye(K) + U @ rng.standard_normal((K - r, K))
    print(f"(b)   det s = {np.linalg.det(s):.3f}; max abs(T s - T) = {np.abs(T @ s - T).max():.2e};"
          f" max abs(T (theta s)^-1 - T theta^-1) = {np.abs(T @ np.linalg.inv(theta @ s) - T @ np.linalg.inv(theta)).max():.2e}")
    # (c) single operating content
    c = rng.standard_normal(K)
    F_single = fisher_formula(T, theta, np.outer(c, c), R, basis)
    evs = np.linalg.eigvalsh(F_single)
    print(f"(c)   single operating content: rank F = {int(np.sum(evs > 1e-9 * evs.max()))} (= r = {r}); grounded PD M: rank {rank}")

# companion example
T2 = np.eye(2); cvec = np.array([1.0, 0.0]); E22 = np.array([[0.0, 0.0], [0.0, 1.0]])
print("(c) companion: T delta c =", T2 @ E22 @ cvec, "; T delta =", (T2 @ E22).tolist(),
      "; F with M = I:", np.trace(T2 @ E22 @ np.eye(2) @ E22.T @ T2.T), "; F with M = c c^T:",
      np.trace(T2 @ E22 @ np.outer(cvec, cvec) @ E22.T @ T2.T))

# (d) orthogonal dictionaries
for (K, r) in [(3, 1), (3, 2), (4, 2), (5, 2)]:
    T = rng.standard_normal((r, K))
    basis_so = []
    for i in range(K):
        for j in range(i + 1, K):
            E = np.zeros((K, K)); E[i, j] = 1.0; E[j, i] = -1.0; basis_so.append(E)
    Mlin = np.array([(T @ Bm).ravel() for Bm in basis_so]).T  # map coefficients -> T delta
    kdim = len(basis_so) - np.linalg.matrix_rank(Mlin)
    print(f"(d) SO({K}), r={r}: dim{{delta in so(K): T delta = 0}} = {kdim}, predicted (K-r)(K-r-1)/2 = {(K - r) * (K - r - 1) // 2};"
          f" identifiable dim = {len(basis_so) - kdim}, GL identifiable K r = {K * r}, GL invisible K(K-r) = {K * (K - r)}")

# (e) Student-t location Fisher information versus inverse variance
for nu in [3, 5, 10, 30]:
    J = (nu + 1) / (nu + 3)
    print(f"(e) t_{nu}: J_f = {J:.6f} >= 1/Var = {(nu - 2) / nu:.6f}: {J >= (nu - 2) / nu}")

# (f) unknown code: full right G-action ambiguity of (Tbar, theta)
K, r = 4, 2
T = rng.standard_normal((r, K)); theta = expm(0.3 * rng.standard_normal((K, K)))
g = expm(0.8 * rng.standard_normal((K, K)))
print("(f) max abs(Tbar theta^-1 - (Tbar g)(theta g)^-1) =", np.abs(T @ np.linalg.inv(theta) - (T @ g) @ np.linalg.inv(theta @ g)).max(),
      "; Tbar g - Tbar is not zero:", np.abs(T @ g - T).max() > 1e-3)
