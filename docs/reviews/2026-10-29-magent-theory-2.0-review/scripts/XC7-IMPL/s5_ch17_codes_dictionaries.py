r"""XC7-IMPL s5: conformance checks for chapter 17 (codes, dictionaries, tangent learning).

(a) it1:kalman (posterior, rank, mutual information) and it1:gaussian-reconstruction-eq,
    it1:loss-eq in closed Gaussian form; it1:noisy-eq bound; it1:exponential identity.
(b) it1:learning (ii)-(v) on a 3-agent complete exchange (K=2, rank-1 codes): exact
    information-filter posterior after n grounded messages per edge vs the limit
    formulas; covariance in invisible coordinates vs S^{-1}.
(c) it1:pair dimension counts and detectable dissonance; it1:unreachable sheaf count.
"""
import numpy as np
from scipy.linalg import expm, null_space, orth
rng = np.random.default_rng(11)

def kl(m1, S1, m2, S2):
    d = len(m1); S2i = np.linalg.inv(S2); dm = m2 - m1
    return 0.5 * (np.trace(S2i @ S1) + dm @ S2i @ dm - d + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])
def spd(d): Ar = rng.normal(size=(d, d)); return Ar @ Ar.T + 0.5 * np.eye(d)

K, r = 4, 2
A = rng.normal(size=(r, K)); Sp, Sq, Rn = spd(K), spd(K), spd(r)
m, a = rng.normal(size=K), rng.normal(size=K)
# (a) Kalman
O = Sp @ A.T @ np.linalg.inv(A @ Sp @ A.T + Rn)
post_cov = Sp - O @ A @ Sp
print("(a) precision form residual:", np.abs(np.linalg.inv(post_cov) - (np.linalg.inv(Sp) + A.T @ np.linalg.inv(Rn) @ A)).max(),
      " rank of precision correction:", np.linalg.matrix_rank(A.T @ np.linalg.inv(Rn) @ A))
MI = 0.5 * (np.linalg.slogdet(Sp)[1] - np.linalg.slogdet(post_cov)[1])
print("    I(y;h) entropy difference vs it1:mutual-information:", MI, 0.5 * np.linalg.slogdet(np.eye(r) + np.linalg.inv(Rn) @ A @ Sp @ A.T)[1])
# Gaussian reconstruction through a noiseless code
G = Sp @ A.T @ np.linalg.inv(A @ Sp @ A.T); P = G @ A
qh_m = (np.eye(K) - P) @ m + P @ a
qh_S = (np.eye(K) - P) @ Sp @ (np.eye(K) - P).T + P @ Sq @ P.T
# direct: condition p on Ay=h, mix over h ~ N(Aa, A Sq A^T)
cond_S = Sp - G @ A @ Sp
dir_m = m + G @ (A @ a - A @ m); dir_S = cond_S + G @ A @ Sq @ A.T @ G.T
print("    reconstruction formula vs direct mixing:", np.abs(qh_m - dir_m).max(), np.abs(qh_S - dir_S).max(),
      " P^2-P:", np.abs(P @ P - P).max())
lhs = kl(a, Sq, qh_m, qh_S); rhs = kl(a, Sq, m, Sp) - kl(A @ a, A @ Sq @ A.T, A @ m, A @ Sp @ A.T)
print("    it1:loss-eq KL(q||qhat) vs KL(q||p)-KL(Tq||Tp):", lhs, rhs)
# noisy code bound
On = O
qn_m = m + On @ A @ (a - m); qn_S = Sp - On @ A @ Sp + On @ (A @ Sq @ A.T + Rn) @ On.T
lhs = kl(a, Sq, qn_m, qn_S); rhs = kl(a, Sq, m, Sp) - kl(A @ a, A @ Sq @ A.T + Rn, A @ m, A @ Sp @ A.T + Rn)
print("    it1:noisy-eq: KL(q||qhat) =", round(lhs, 6), "<=", round(rhs, 6), ":", lhs <= rhs + 1e-12)
# exponential identity
Ox, Tx = rng.normal(size=(K, r)), rng.normal(size=(r, K))
Z = Tx @ Ox
phi1 = (expm(Z) - np.eye(r)) @ np.linalg.inv(Z)
print("    it1:exponential residual:", np.abs(expm(Ox @ Tx) - (np.eye(K) + Ox @ phi1 @ Tx)).max(),
      " e^P - (I+(e-1)P):", np.abs(expm(P) - (np.eye(K) + (np.e - 1) * P)).max())

# (b) tangent learning on complete exchange of 3 agents, K=2, rank-1 codes
K2 = 2; agents = [0, 1, 2]
edges = [(i, j) for i in agents for j in agents if i != j]          # (receiver, sender)
eidx = {e: k for k, e in enumerate(edges)}; E = len(edges); D = K2 * K2
codes = {e: rng.normal(size=(1, K2)) for e in edges}
loops = [[(0, 1), (1, 0)], [(0, 2), (2, 0)], [(1, 2), (2, 1)],            # two-cycles: edges (i<-j),(j<-i)
         [(1, 0), (2, 1), (0, 2)], [(2, 0), (1, 2), (0, 1)]]               # directed triangles
B = np.zeros((len(loops) * D, E * D))
for g, lp in enumerate(loops):
    for e in lp:
        B[g * D:(g + 1) * D, eidx[e] * D:(eidx[e] + 1) * D] += np.eye(D)
sigma, lam = 1.0, 3.0
L0 = np.eye(E * D) / sigma ** 2 + lam * B.T @ B
# invisible directions N_e = {delta : T_e delta = 0}; vec(delta) row-major
Nblocks = []
for e in edges:
    Tm = np.kron(codes[e], np.eye(K2))           # T delta as a linear map on vec(delta) (row-major)
    Nblocks.append(null_space(Tm))
Zb = np.zeros((E * D, sum(nb.shape[1] for nb in Nblocks))); col = 0
for k, nb in enumerate(Nblocks):
    Zb[k * D:(k + 1) * D, col:col + nb.shape[1]] = nb; col += nb.shape[1]
Sm = Zb.T @ L0 @ Zb
PiN = Zb @ np.linalg.solve(Sm, Zb.T @ L0)
omega_star = rng.normal(size=E * D) * 0.3
omega_inf = (np.eye(E * D) - PiN) @ omega_star
# exact posterior after n grounded messages per edge (information filter in closed batch form)
Mref = np.eye(K2)
for n, nz in ((10 ** 3, 0.5), (10 ** 5, 0.5), (10 ** 3, 0.0), (10 ** 5, 0.0)):
    Lam, eta = L0.copy(), np.zeros(E * D)
    for e in edges:
        k = eidx[e]
        ys = rng.normal(size=(n, K2))                       # referents with second moment I
        Tm = codes[e]; noise = rng.normal(size=n) * nz
        # J_t omega = -T omega_e y_t ; h_t - T y_t = -T omega*_e y_t + eps
        Jrows = -np.einsum('r,nk->nrk', Tm[0], ys).reshape(n, D)   # d/d vec(omega_e) of -(T omega y)
        obs = Jrows @ omega_star[k * D:(k + 1) * D] + noise
        Lam[k * D:(k + 1) * D, k * D:(k + 1) * D] += Jrows.T @ Jrows / 0.25
        eta[k * D:(k + 1) * D] += Jrows.T @ obs / 0.25
    mean = np.linalg.solve(Lam, eta); cov = np.linalg.inv(Lam)
    print(f"(b) n={n}, noise sd={nz}: |mean - (I-Pi_N)omega*| = {np.abs(mean - omega_inf).max():.2e}; "
          f"|Z^T cov Z - S^-1| = {np.abs(Zb.T @ cov @ Zb - np.linalg.inv(Sm)).max():.2e}")
LN = Zb.T @ B.T @ B @ Zb
print("    (iii) dim N =", Zb.shape[1], " rank L_N =", np.linalg.matrix_rank(LN), " dim(N cap ker B) =",
      Zb.shape[1] - np.linalg.matrix_rank(B @ Zb), " |S^-1 - sigma^2(I+lam sigma^2 L_N)^-1| =",
      np.abs(np.linalg.inv(Sm) - sigma ** 2 * np.linalg.inv(np.eye(LN.shape[0]) + lam * sigma ** 2 * LN)).max())
# (iv) flat truth: omega* = coboundary (a_i - a_j) on edge (i<-j)
avec = rng.normal(size=(3, D)) * 0.3
om_flat = np.concatenate([avec[i] - avec[j] for (i, j) in edges])
zst = Zb.T @ om_flat
lhs = Zb.T @ ((np.eye(E * D) - PiN) @ om_flat - om_flat)
print("    (iv) flat truth |B omega*| =", np.abs(B @ om_flat).max(), "; shrinkage residual:",
      np.abs(lhs + np.linalg.solve(np.eye(LN.shape[0]) + lam * sigma ** 2 * LN, zst)).max())
# (c) it1:pair: two agents, K=3, rank-2 codes
K3 = 3; T1, T0 = rng.normal(size=(2, K3)), rng.normal(size=(2, K3))
K0, K1 = null_space(T0), null_space(T1); R0, R1 = orth(T0.T), orth(T1.T)
dim = lambda Mx: np.linalg.matrix_rank(Mx) if Mx.size else 0
dimsum = dim(np.hstack([K0, K1])); diminter = K0.shape[1] + K1.shape[1] - dimsum
Rint = null_space(np.hstack([R0, -R1]))                     # coefficients of R0 cap R1
Pcap = (R0 @ Rint[:R0.shape[1]]); Pcap = orth(Pcap); Pcap = Pcap @ Pcap.T
# N = {(nu01, nu10): range nu01 in K1, range nu10 in K0}; B nu = nu01 + nu10 (Frobenius W)
BN = np.hstack([np.kron(np.eye(K3), K1), np.kron(np.eye(K3), K0)])   # vec_F(K1 C) = (I kron K1) vec_F(C)
print("(c) it1:pair dim B(N) =", dim(BN), " claimed K dim(K0+K1) =", K3 * dimsum,
      "; rank B - dim B(N) =", K3 ** 2 - dim(BN), " claimed K dim(R0 cap R1) =", K3 * (Rint.shape[1]))
om01, om10 = rng.normal(size=(K3, K3)), rng.normal(size=(K3, K3))
# detectable dissonance by brute minimization over N
Bw = (om01 + om10).reshape(-1, order='F')
coef, *_ = np.linalg.lstsq(BN, -Bw, rcond=None)
Dnum = np.sum((Bw + BN @ coef) ** 2)
print("    detectable dissonance: brute min", round(Dnum, 10), " ||P_cap (om01+om10)||^2", round(np.sum((Pcap @ (om01 + om10)) ** 2), 10))
