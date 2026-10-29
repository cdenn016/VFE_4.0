"""D22-COMM s03: several modalities (chapters/17_communication_links.tex:167-220).

Checks:
  (a) it1:modalities: identifiable dimension K dim(sum of row spaces) for overlapping codes; two identical
      full-rank codes saturate K^2.
  (b) it1:query-dictionary: solutions of T theta^{-1} = S form a coset of Stab(T) and meet both determinant signs.
  (c) it1:stack-compatibility: a solution exists iff ker(T*^T) = ker(S*^T); the counterexample T1=T2=(1,0),
      S1=(1,0), S2=(2,0); random rank-deficient stacks with equal and with unequal relation spaces.
  (d) it1:routed: Monte Carlo of the complete-data and latent-label Fisher information for a two-sender,
      two-modality example (K=2, r=1), compared with sum_c beta_c F_c and the Loewner bound.
"""
import numpy as np
from scipy.linalg import expm, null_space

rng = np.random.default_rng(11)


def fisher_rank_sum(Ts, K, M):
    basis = []
    for i in range(K):
        for j in range(K):
            E = np.zeros((K, K)); E[i, j] = 1.0; basis.append(E)
    n = len(basis)
    F = np.zeros((n, n))
    for T in Ts:
        for a in range(n):
            for b in range(n):
                F[a, b] += np.trace(T @ basis[a] @ M @ basis[b].T @ T.T)
    ev = np.linalg.eigvalsh(F)
    return int(np.sum(ev > 1e-9 * ev.max()))


K = 5
shared = rng.standard_normal(K)
T1 = np.vstack([shared, rng.standard_normal(K)])
T2 = np.vstack([shared, rng.standard_normal(K)])
M = np.eye(K)
print("(a) overlapping widths 2+2 sharing one row: Fisher rank =", fisher_rank_sum([T1, T2], K, M),
      " predicted K*dim(row sum) =", K * np.linalg.matrix_rank(np.vstack([T1, T2])), " naive K*(2+2) =", K * 4)
Tf = rng.standard_normal((K, K))
print("(a) two identical full-rank codes: Fisher rank =", fisher_rank_sum([Tf, Tf], K, M), " K^2 =", K * K)

# (b) query dictionary
K, r = 4, 2
T = rng.standard_normal((r, K)); S = rng.standard_normal((r, K))
Pm = np.vstack([T, rng.standard_normal((K - r, K))]); Qm = np.vstack([S, rng.standard_normal((K - r, K))])
th0_inv = np.linalg.inv(Pm) @ Qm
Qm2 = Qm.copy(); Qm2[-1] *= -1
th1_inv = np.linalg.inv(Pm) @ Qm2
print("(b) residuals T th0^-1 - S:", np.abs(T @ th0_inv - S).max(), " T th1^-1 - S:", np.abs(T @ th1_inv - S).max(),
      " det signs:", np.sign(np.linalg.det(th0_inv)), np.sign(np.linalg.det(th1_inv)))
th0 = np.linalg.inv(th0_inv); th1 = np.linalg.inv(th1_inv)
sdiff = np.linalg.inv(th0) @ th1
print("(b) th0^-1 th1 in Stab(T): max abs(T s - T) =", np.abs(T @ sdiff - T).max())


def solve_stack(Ts, Ss):
    """Find invertible A with Ts A = Ss if one exists; report lstsq residual and invertibility."""
    X0, *_ = np.linalg.lstsq(Ts, Ss, rcond=None)
    res = np.abs(Ts @ X0 - Ss).max()
    Nk = null_space(Ts)  # columns span ker Ts (right kernel)
    if Nk.shape[1] > 0:
        X = X0 + Nk @ rng.standard_normal((Nk.shape[1], Ts.shape[1]))
    else:
        X = X0
    return res, abs(np.linalg.det(X)), np.abs(Ts @ X - Ss).max()


def left_kernel_equal(Ts, Ss, tol=1e-9):
    k1 = null_space(Ts.T, rcond=tol); k2 = null_space(Ss.T, rcond=tol)
    if k1.shape[1] != k2.shape[1]:
        return False
    if k1.shape[1] == 0:
        return True
    return np.linalg.matrix_rank(np.hstack([k1, k2]), tol=1e-8) == k1.shape[1]


Ts = np.array([[1.0, 0.0], [1.0, 0.0]]); Ss = np.array([[1.0, 0.0], [2.0, 0.0]])
res, det, _ = solve_stack(Ts, Ss)
print("(c) counterexample: ranks", np.linalg.matrix_rank(Ts), np.linalg.matrix_rank(Ss),
      " equal left kernels:", left_kernel_equal(Ts, Ss), " lstsq residual =", res)
K = 4
a1, a2 = rng.standard_normal(K), rng.standard_normal(K)
Ts = np.vstack([a1, a2, a1 + 2 * a2])  # rank 2, relation (1,2,-1)
A_true = expm(0.5 * rng.standard_normal((K, K)))
Ss_good = Ts @ A_true
b1, b2 = rng.standard_normal(K), rng.standard_normal(K)
Ss_bad = np.vstack([b1, b2, 3 * b1 - b2])  # rank 2, different relation (3,-1,-1)
for lab, Ss in [("equal relations", Ss_good), ("unequal relations, equal rank", Ss_bad)]:
    res, det, res2 = solve_stack(Ts, Ss)
    print(f"(c) {lab}: ranks {np.linalg.matrix_rank(Ts)}, {np.linalg.matrix_rank(Ss)}; equal left kernels: {left_kernel_equal(Ts, Ss)};"
          f" lstsq residual {res:.2e}; generic solution det {det:.3f}, residual {res2:.2e}")

# (d) routed modalities: K=2, r=1, two senders (0 target, 1 other), two modalities
K = 2
beta = np.array([0.7, 0.2])  # probability that modality c carries sender 0
T0 = [np.array([[1.0, 0.3]]), np.array([[-0.4, 1.0]])]   # target sender codes per modality
T1 = [np.array([[0.2, 1.0]]), np.array([[1.0, -0.5]])]   # other sender codes per modality
Rn = [0.5, 0.8]
th0 = expm(np.array([[0.1, 0.3], [-0.2, 0.05]])); th1 = expm(np.array([[0.0, -0.4], [0.3, 0.1]]))
th0i, th1i = np.linalg.inv(th0), np.linalg.inv(th1)
basis = []
for i in range(K):
    for j in range(K):
        E = np.zeros((K, K)); E[i, j] = 1.0; basis.append(E)
Mref = np.diag([1.0, 2.0])
# analytic complete-data information sum_c beta_c F_c
C = th0i @ Mref @ th0i.T
Fan = np.zeros((4, 4))
for c in range(2):
    for a in range(4):
        for b in range(4):
            Fan[a, b] += beta[c] * np.trace(T0[c] @ basis[a] @ C @ basis[b].T @ T0[c].T) / Rn[c]
N = 400000
Y = np.sqrt(np.diag(Mref))[:, None] * rng.standard_normal((K, N))
score_c = np.zeros((4, N)); score_o = np.zeros((4, N))
for c in range(2):
    J = rng.random(N) < beta[c]
    mu0 = (T0[c] @ th0i @ Y)[0]; mu1 = (T1[c] @ th1i @ Y)[0]
    h = np.where(J, mu0, mu1) + np.sqrt(Rn[c]) * rng.standard_normal(N)
    dmu = np.array([-(T0[c] @ Bm @ th0i @ Y)[0] for Bm in basis])  # 4 x N derivative of mu0
    s0 = (h - mu0) / Rn[c] * dmu
    score_c += np.where(J, s0, 0.0)
    l0 = beta[c] * np.exp(-0.5 * (h - mu0) ** 2 / Rn[c]); l1 = (1 - beta[c]) * np.exp(-0.5 * (h - mu1) ** 2 / Rn[c])
    w = l0 / (l0 + l1)
    score_o += w * s0
Fc_mc = score_c @ score_c.T / N
Fo_mc = score_o @ score_o.T / N
print("(d) analytic sum_c beta_c F_c eigenvalues:", np.linalg.eigvalsh(Fan).round(4))
print("(d) complete-data MC eigenvalues:          ", np.linalg.eigvalsh(Fc_mc).round(4),
      " max rel deviation:", (np.abs(Fc_mc - Fan).max() / np.abs(Fan).max()).round(4))
print("(d) latent-label MC eigenvalues:           ", np.linalg.eigvalsh(Fo_mc).round(4))
print("(d) eigenvalues of (sum beta F) - latent MC (Loewner gap, should be >= 0 up to MC error):",
      np.linalg.eigvalsh(Fan - Fo_mc).round(4))
