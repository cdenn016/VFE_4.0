r"""D23-LEARN check 3: it1:exact-pair, it1:exact-flat, it1:owned, the dressed norm, BCH order,
and the directed ring it2:ring of chapters/18_link_dynamics.tex (focus question 2).

E1  exact reciprocal identification: finite residual element s = I + k a^T, k in ker(T1; T0 th*),
    reproduces both effective codes and reciprocity; residual Lie algebra dimension.
E2  exact flat identification for the complete exchange of three agents: a nonconstant section
    of the T'-sheaf gives a configuration with identical message laws and identity holonomy on
    all two-cycles and directed triangles (all basings); the pure-gauge parametrization
    k_0 = I, k_i = th_{i0} reproduces every edge.  Dimension of the residual set at the truth.
O1  receiver-factorized Gaussian recognition: CAVI means equal exact means; covariances are the
    inverse diagonal precision blocks and are Loewner-below the exact marginals.
G1  dressed norm invariance ||h^-1 X h||^2_{h^-1 S h^-T} = ||X||^2_S and the diag(t,1,..) scaling.
G2  second-order error of the linearized holonomy log(e^{X3} e^{X2} e^{X1}) - (X1+X2+X3).
R1  ring: per-agent mean-flow Jacobian rebuilt from the energy of it2:ring; spectrum versus
    {-1 (mult l)} U {-(1 - zeta w)}; weak-anchor limits for c != 0 and c = 0; global flow spectrum.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, logm

from d23_common import blockdiag, null_space, rank, spd

rng = np.random.default_rng(11)
inv = np.linalg.inv

print("=== E1: exact reciprocal identification (no linearization) ===")
K = 3
th_star = expm(rng.standard_normal((K, K))) @ np.diag([1.0, 2.0, 0.5])   # far from identity
th_star_10 = inv(th_star)
for (r1, r0) in [(1, 1), (2, 2), (1, 2)]:
    T1 = rng.standard_normal((r1, K)); T0 = rng.standard_normal((r0, K))
    stack = np.vstack([T1, T0 @ th_star])
    ker = null_space(stack)
    print(f"ranks (r1,r0)=({r1},{r0}): dim ker(T1; T0 th*) = {ker.shape[1]} -> residual dimension K*dim = {K * ker.shape[1]}")
    if ker.shape[1]:
        k = ker[:, 0]; a = rng.standard_normal(K) * 2.0
        if abs(1 + a @ k) < 0.2:
            a = -a
        s = np.eye(K) + np.outer(k, a)
        th01 = th_star @ s; th10 = inv(s) @ th_star_10
        print(f"   |T1 th01^-1 - T1 th*01^-1| = {np.abs(T1 @ inv(th01) - T1 @ inv(th_star)).max():.1e}, "
              f"|T0 th10^-1 - T0 th*10^-1| = {np.abs(T0 @ inv(th10) - T0 @ inv(th_star_10)).max():.1e}, "
              f"|th01 th10 - I| = {np.abs(th01 @ th10 - np.eye(K)).max():.1e}, "
              f"max|th01 - th*01| = {np.abs(th01 - th_star).max():.3f}")

print("\n=== E2: exact flat identification, complete exchange of three agents ===")
edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]
g = [expm(0.8 * rng.standard_normal((K, K))) for _ in range(3)]
th_true = {(j, i): g[i] @ inv(g[j]) for (j, i) in edges}       # theta*_{ij} for edge j -> i
kdir = rng.standard_normal(K)
Pk = np.eye(K) - np.outer(kdir, kdir) / (kdir @ kdir)
Tp = {e: (rng.standard_normal((1, K)) @ Pk) for e in edges}      # T'_e kills kdir
T = {(j, i): Tp[(j, i)] @ inv(g[j]) for (j, i) in edges}        # T_e = T'_e g_j^{-1}
# sheaf for T': sections u with T'_e (u_i - u_j) = 0; nonconstant section s_v = c_v kdir
s = [0.0 * kdir, 1.0 * kdir, -0.7 * kdir]
w = rng.standard_normal(K)
t = 0.3
u = [np.eye(K) + t * np.outer(sv, w) for sv in s]
th = {(j, i): g[i] @ u[i] @ inv(u[j]) @ inv(g[j]) for (j, i) in edges}
law_err = max(np.abs(T[e] @ inv(th[e]) - T[e] @ inv(th_true[e])).max() for e in edges)
def hol(path):  # path = [v0, v1, ..., v0]; H = th_{v_k v_{k-1}} ... th_{v1 v0}
    H = np.eye(K)
    for a, b in zip(path[:-1], path[1:]):
        H = th[(a, b)] @ H                                       # edge a -> b carries th_{b a}
    return H
loops = [[0, 1, 0], [1, 0, 1], [1, 2, 1], [2, 1, 2], [0, 2, 0], [2, 0, 2],
         [0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2], [0, 2, 1, 0], [2, 1, 0, 2], [1, 0, 2, 1]]
hol_err = max(np.abs(hol(p) - np.eye(K)).max() for p in loops)
print(f"message-law residual = {law_err:.1e}; max |H - I| over 12 based loops = {hol_err:.1e}; "
      f"max |theta - theta*| = {max(np.abs(th[e] - th_true[e]).max() for e in edges):.3f}")
# pure-gauge parametrization from the proof: k_0 = I, k_i = th_{i0}
kk = [np.eye(K), th[(0, 1)], th[(0, 2)]]
pg_err = max(np.abs(th[(j, i)] - kk[i] @ inv(kk[j])).max() for (j, i) in edges)
print(f"pure-gauge reconstruction k_0 = I, k_i = theta_(i0): max residual = {pg_err:.1e}")
# dimension of the residual set at the truth: {u in GL^V: T'_e(u_i - u_j) = 0} / right GL(K)
rows = []
for (j, i) in edges:
    Te = Tp[(j, i)]
    blk = np.zeros((Te.shape[0] * K, 3 * K * K))
    # vec_row(T u) = (T kron I) vec_row(u)
    blk[:, i * K * K:(i + 1) * K * K] += np.kron(Te, np.eye(K))
    blk[:, j * K * K:(j + 1) * K * K] -= np.kron(Te, np.eye(K))
    rows.append(blk)
sol_dim = null_space(np.vstack(rows)).shape[1]
print(f"dim of linear solution space for u = {sol_dim}; minus K^2 for right multiplication = {sol_dim - K * K}"
      f" (expected K(dim H0 - K) = 3*(5-3) = 6)")

print("\n=== O1: receiver-factorized Gaussian recognition (it1:owned) ===")
# random SPD posterior precision on blocks: receiver 0 owns edges into 0, etc.
sizes = [4, 4, 4]                       # two edges per receiver in K = sqrt(2)? use generic blocks
d = sum(sizes)
Lam = spd(d, rng, cond=30.0)
eta = rng.standard_normal(d)
exact_mean = np.linalg.solve(Lam, eta)
exact_cov = inv(Lam)
mu = np.zeros(d)
cuts = np.cumsum([0] + sizes)
for sweep in range(400):
    for b in range(3):
        sl = slice(cuts[b], cuts[b + 1])
        other = np.ones(d, bool); other[sl] = False
        mu[sl] = np.linalg.solve(Lam[sl, sl], eta[sl] - Lam[sl][:, other] @ mu[other])
gaps = []
for b in range(3):
    sl = slice(cuts[b], cuts[b + 1])
    gaps.append(np.linalg.eigvalsh(exact_cov[sl, sl] - inv(Lam[sl, sl])).min())
print(f"|CAVI mean - exact mean| = {np.abs(mu - exact_mean).max():.1e}; min eig(exact block cov - inv diag block) "
      f"per receiver = {np.round(gaps, 5)}; total variance exact {np.trace(exact_cov):.3f} vs factorized "
      f"{sum(np.trace(inv(Lam[cuts[b]:cuts[b + 1], cuts[b]:cuts[b + 1]])) for b in range(3)):.3f}")

print("\n=== G1: dressed norm and the absence of a conjugation-invariant form on gl(K) ===")
K = 3
X = rng.standard_normal((K, K)); S = spd(K, rng); h = rng.standard_normal((K, K))
def dn(X, S):
    return np.trace(X.T @ inv(S) @ X @ S)
print(f"||X||_S^2 = {dn(X, S):.10f}; ||h^-1 X h||^2 at h^-1 S h^-T = {dn(inv(h) @ X @ h, inv(h) @ S @ inv(h).T):.10f}")
E12 = np.zeros((K, K)); E12[0, 1] = 1.0
tt = 3.0; D = np.diag([tt, 1.0, 1.0])
print("diag(t,1,1) E12 diag(t,1,1)^-1 =", (D @ E12 @ inv(D))[0, 1], "(= t);  h^-1 E12 h =", (inv(D) @ E12 @ D)[0, 1])

print("\n=== G2: second-order error of the linearized holonomy (directed triangle) ===")
Xs = [rng.standard_normal((K, K)) for _ in range(3)]
for eps in [1e-2, 1e-3]:
    H = expm(eps * Xs[2]) @ expm(eps * Xs[1]) @ expm(eps * Xs[0])
    err = np.abs(np.real(logm(H)) - eps * sum(Xs)).max()
    bch = 0.5 * eps ** 2 * (Xs[2] @ Xs[1] - Xs[1] @ Xs[2] + Xs[2] @ Xs[0] - Xs[0] @ Xs[2] + Xs[1] @ Xs[0] - Xs[0] @ Xs[1])
    print(f"eps={eps:.0e}: |log H - eps sum X|/eps^2 = {err / eps ** 2:.4f}; "
          f"|log H - eps sum X - (1/2) sum_{{a>b}} [X_a,X_b]|/eps^3 = {np.abs(np.real(logm(H)) - eps * sum(Xs) - bch).max() / eps ** 3:.3f}")

print("\n=== R1: directed ring of it2:ring (K = 1) ===")
def ring_jacobian(l, p, q, s, c):
    """Per-agent plug-in mean flow mu' = -A^{-1}(grad E_i) from E_i =
    p/2 (y_i - m_i)^2 + s/2 w_i^2 + q/2 (o_i - y_i + y_{i+1} + c w_i)^2, agent i hears i+1."""
    n = 2 * l
    A = np.zeros((n, n)); H = np.zeros((n, n))
    for i in range(l):
        nb = (i + 1) % l
        yi, wi, yn = 2 * i, 2 * i + 1, 2 * nb
        # residual r = o - y_i + y_{i+1} + c w_i ; dr/dy_i = -1, dr/dw_i = c, dr/dy_{i+1} = +1
        grad_r = {yi: -1.0, wi: c, yn: 1.0}
        Ai = np.array([[p + q, -q * c], [-q * c, s + q * c * c]])
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Ai
        for a in (yi, wi):
            for b, gb in grad_r.items():
                H[a, b] += q * grad_r[a] * gb
        H[yi, yi] += p; H[wi, wi] += s
    return -np.linalg.solve(A, H)
def global_jacobian(l, p, q, s, c):
    n = 2 * l
    J = np.zeros((n, n))
    for i in range(l):
        nb = (i + 1) % l
        yi, wi, yn = 2 * i, 2 * i + 1, 2 * nb
        gr = np.zeros(n); gr[yi] = -1.0; gr[wi] = c; gr[yn] = 1.0
        J += q * np.outer(gr, gr)
        J[yi, yi] += p; J[wi, wi] += s
    Dblk = blockdiag([J[2 * i:2 * i + 2, 2 * i:2 * i + 2] for i in range(l)])
    return -np.linalg.solve(Dblk, J)
for (l, p, q, s, c) in [(3, 1.0, 2.0, 0.7, 1.3), (5, 0.5, 1.5, 2.0, -0.8), (4, 1.0, 1.0, 1.0, 0.0)]:
    J = ring_jacobian(l, p, q, s, c)
    ev = np.sort_complex(np.linalg.eigvals(J))
    w_ = q * s / (p * s + p * q * c * c + q * s)
    zetas = np.exp(2j * np.pi * np.arange(l) / l)
    pred = np.sort_complex(np.concatenate([-np.ones(l), -(1 - zetas * w_)]))
    gev = np.linalg.eigvals(global_jacobian(l, p, q, s, c))
    print(f"l={l}, c={c}: |spectrum - predicted| = {np.abs(ev - pred).max():.1e}; w = {w_:.4f}; "
          f"max |Im| per-agent = {np.abs(ev.imag).max():.3f}; max |Im| global = {np.abs(gev.imag).max():.1e}")
p, q = 1.0, 2.0
for c in [1.0, 0.0]:
    vals = [q * s / (p * s + p * q * c * c + q * s) for s in [1e-6, 1e-3, 1.0, 1e3, 1e6]]
    print(f"c={c}: w at s = 1e-6, 1e-3, 1, 1e3, 1e6 -> {np.round(vals, 6)}; q/(p+q) = {q / (p + q):.6f}")
