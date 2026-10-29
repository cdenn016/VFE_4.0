"""D22-COMM s01: reception through a code (chapters/17_communication_links.tex:52-128).

Checks, in float64 at small size:
  (a) it1:gaussian-reconstruction: AG=I, P^2=P, ker P = ker A, range P = range(Sigma_p A^T),
      reconstruction mean/covariance against direct conditioning + mixing, and the
      covariance identity Sigma_p - P Sigma_p = (I-P) Sigma_p (I-P)^T.
  (b) it1:loss: KL(q||qhat) = KL(q||p) - KL(Aq||Ap) for the Gaussian instance.
  (c) it1:noisy: Gaussian qhat formula and the inequality KL(q||qhat) <= KL(q||p) - KL(qK||pK).
  (d) it1:kalman: posterior, precision form, mutual information; also with rank-deficient A
      (full row rank is not needed for these formulas).
  (e) it1:exponential: e^{OT} = I + O phi1(TO) T, det = e^{tr OT}; e^P = I + (e-1) P.
  (f) companion c17 examples (eq c17:loss-example and the correlated-prior example).
  (g) meaning loss along a stabilizer orbit of the dictionary (not identifiable from messages).
"""
import numpy as np
from scipy.linalg import expm, null_space, orth

rng = np.random.default_rng(20261029)


def spd(K):
    X = rng.standard_normal((K, K))
    return X @ X.T + 0.5 * np.eye(K)


def kl_gauss(m0, S0, m1, S1):
    """KL(N(m0,S0) || N(m1,S1))."""
    K = len(m0)
    S1inv = np.linalg.inv(S1)
    d = m1 - m0
    return 0.5 * (np.trace(S1inv @ S0) + d @ S1inv @ d - K
                  + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


def cond_gauss(m, S, A, h, R):
    """Posterior of y ~ N(m,S) given h = A y + eps, eps ~ N(0,R) (R may be 0)."""
    Sh = A @ S @ A.T + R
    Kg = S @ A.T @ np.linalg.inv(Sh)
    return m + Kg @ (h - A @ m), S - Kg @ A @ S, Kg


K, r = 5, 2
A = rng.standard_normal((r, K))
Sp, Sq = spd(K), spd(K)
m, a = rng.standard_normal(K), rng.standard_normal(K)

# (a) projector identities
G = Sp @ A.T @ np.linalg.inv(A @ Sp @ A.T)
P = G @ A
print("(a) max abs(AG-I) =", np.abs(A @ G - np.eye(r)).max())
print("(a) max abs(P^2-P) =", np.abs(P @ P - P).max())
kerA = null_space(A)
print("(a) max abs(P kerA) =", np.abs(P @ kerA).max(), " rank P =", np.linalg.matrix_rank(P))
RA = orth(Sp @ A.T)
print("(a) range P vs range(Sp A^T): max abs((I-RA RA^T) P) =", np.abs((np.eye(K) - RA @ RA.T) @ P).max())
_, Spost, Kg = cond_gauss(m, Sp, A, np.zeros(r), np.zeros((r, r)))
mean_direct = m + Kg @ (A @ a - A @ m)
cov_direct = Spost + Kg @ (A @ Sq @ A.T) @ Kg.T
mean_formula = (np.eye(K) - P) @ m + P @ a
cov_formula = (np.eye(K) - P) @ Sp @ (np.eye(K) - P).T + P @ Sq @ P.T
print("(a) mean residual =", np.abs(mean_direct - mean_formula).max())
print("(a) cov residual  =", np.abs(cov_direct - cov_formula).max())
print("(a) max abs(Sp - P Sp - (I-P)Sp(I-P)^T) =", np.abs(Sp - P @ Sp - (np.eye(K) - P) @ Sp @ (np.eye(K) - P).T).max())
print("(a) min eig of reconstruction covariance =", np.linalg.eigvalsh(cov_formula).min())

lhs = kl_gauss(a, Sq, mean_formula, cov_formula)
rhs = kl_gauss(a, Sq, m, Sp) - kl_gauss(A @ a, A @ Sq @ A.T, A @ m, A @ Sp @ A.T)
print("(b) KL(q||qhat) =", lhs, " KL(q||p)-KL(Aq||Ap) =", rhs, " residual =", abs(lhs - rhs))

R = spd(r) * 0.3
_, Spost_n, O = cond_gauss(m, Sp, A, np.zeros(r), R)
mean_n = m + O @ A @ (a - m)
cov_n = Sp - O @ A @ Sp + O @ (A @ Sq @ A.T + R) @ O.T
mean_n_direct = m + O @ (A @ a - A @ m)
cov_n_direct = Spost_n + O @ (A @ Sq @ A.T + R) @ O.T
print("(c) noisy qhat mean/cov residual =", np.abs(mean_n - mean_n_direct).max(), np.abs(cov_n - cov_n_direct).max())
lhs_n = kl_gauss(a, Sq, mean_n, cov_n)
rhs_n = kl_gauss(a, Sq, m, Sp) - kl_gauss(A @ a, A @ Sq @ A.T + R, A @ m, A @ Sp @ A.T + R)
print("(c) KL(q||qhat) =", lhs_n, " bound =", rhs_n, " slack =", rhs_n - lhs_n)
worst = np.inf
for _ in range(2000):
    A2 = rng.standard_normal((r, K)); Sp2, Sq2 = spd(K), spd(K); R2 = spd(r) * rng.uniform(0.01, 3)
    m2, a2 = rng.standard_normal(K), rng.standard_normal(K)
    _, Spn2, O2 = cond_gauss(m2, Sp2, A2, np.zeros(r), R2)
    mn2 = m2 + O2 @ A2 @ (a2 - m2)
    cn2 = Spn2 + O2 @ (A2 @ Sq2 @ A2.T + R2) @ O2.T
    s = (kl_gauss(a2, Sq2, m2, Sp2) - kl_gauss(A2 @ a2, A2 @ Sq2 @ A2.T + R2, A2 @ m2, A2 @ Sp2 @ A2.T + R2)) - kl_gauss(a2, Sq2, mn2, cn2)
    worst = min(worst, s)
print("(c) minimum slack over 2000 random Gaussian instances =", worst)

for name, Ad in [("full row rank", A), ("rank-deficient", np.vstack([A[0], 2 * A[0]]))]:
    mp, Sx, Og = cond_gauss(m, Sp, Ad, np.zeros(r), R)
    prec = np.linalg.inv(Sp) + Ad.T @ np.linalg.inv(R) @ Ad
    mi_formula = 0.5 * np.linalg.slogdet(np.eye(r) + np.linalg.inv(R) @ Ad @ Sp @ Ad.T)[1]
    mi_entropy = 0.5 * np.linalg.slogdet(Sp)[1] - 0.5 * np.linalg.slogdet(Sx)[1]
    print("(d)", name, ": max abs(inv(post cov) - precision form) =", np.abs(np.linalg.inv(Sx) - prec).max(),
          " MI formula - entropy diff =", mi_formula - mi_entropy,
          " rank of precision correction =", np.linalg.matrix_rank(Ad.T @ np.linalg.inv(R) @ Ad))


def phi1(Z):
    n = Z.shape[0]
    Aug = np.zeros((2 * n, 2 * n)); Aug[:n, :n] = Z; Aug[:n, n:] = np.eye(n)
    return expm(Aug)[:n, n:]


worst_e, worst_d = 0.0, 0.0
for trial in range(200):
    Om = rng.standard_normal((K, r)); Tm = rng.standard_normal((r, K))
    if trial == 0:
        Om[:, 1] = 0.0
    E = expm(Om @ Tm)
    worst_e = max(worst_e, np.abs(E - (np.eye(K) + Om @ phi1(Tm @ Om) @ Tm)).max() / max(1, np.abs(E).max()))
    worst_d = max(worst_d, abs(np.linalg.det(E) - np.exp(np.trace(Om @ Tm))) / np.exp(np.trace(Om @ Tm)))
print("(e) max rel residual e^{OT} - (I + O phi1(TO) T) over 200 draws =", worst_e)
print("(e) max rel residual det e^{OT} - e^{tr OT} =", worst_d)
print("(e) max abs(e^P - (I + (e-1)P)) =", np.abs(expm(P) - (np.eye(K) + (np.e - 1) * P)).max())
print("(e) rank(e^P - I) =", np.linalg.matrix_rank(expm(P) - np.eye(K)), " rank P =", np.linalg.matrix_rank(P))

aa, bb, rr = 0.7, -1.3, 0.4
A1 = np.array([[1.0, 0.0]])
for Sp_c, label in [(np.eye(2), "independent prior"), (np.array([[1, rr], [rr, 1.0]]), "correlated prior r=0.4")]:
    G1 = Sp_c @ A1.T @ np.linalg.inv(A1 @ Sp_c @ A1.T); P1 = G1 @ A1
    mh = (np.eye(2) - P1) @ np.zeros(2) + P1 @ np.array([aa, bb])
    Ch = (np.eye(2) - P1) @ Sp_c @ (np.eye(2) - P1).T + P1 @ np.eye(2) @ P1.T
    print("(f)", label, ": qhat mean =", mh, " cov =", Ch.round(12).tolist(),
          " KL(q||qhat) =", kl_gauss(np.array([aa, bb]), np.eye(2), mh, Ch))
print("(f) companion predicts b^2/2 =", bb ** 2 / 2, " and mean (a, r a) =", (aa, rr * aa))

Tbar = rng.standard_normal((r, K))
theta = expm(0.3 * rng.standard_normal((K, K)))
mu_j, S_j = rng.standard_normal(K), spd(K)
u = null_space(Tbar)[:, 0]
s = np.eye(K) + np.outer(u, rng.standard_normal(K))
print("(g) max abs(Tbar s - Tbar) =", np.abs(Tbar @ s - Tbar).max(), " det s =", np.linalg.det(s))
for lab, th in [("theta", theta), ("theta s", theta @ s)]:
    Aeff = Tbar @ np.linalg.inv(th)
    q_mean, q_cov = th @ mu_j, th @ S_j @ th.T
    Gx = Sp @ Aeff.T @ np.linalg.inv(Aeff @ Sp @ Aeff.T); Px = Gx @ Aeff
    qh_m = (np.eye(K) - Px) @ m + Px @ q_mean
    qh_c = (np.eye(K) - Px) @ Sp @ (np.eye(K) - Px).T + Px @ q_cov @ Px.T
    print("(g) dictionary", lab, ": effective code row0 =", Aeff[0].round(10), " message mean =", (Aeff @ q_mean).round(10),
          " loss KL(q||qhat) =", kl_gauss(q_mean, q_cov, qh_m, qh_c))
