"""L8 checks for g02_gaussian_realization.tex."""
import numpy as np
from numpy.linalg import inv, det, slogdet, eigvalsh, matrix_rank
from scipy.linalg import sqrtm, null_space
from common import rng, out, rep, note, spd, gl_plus

# prop:gauss-log-normalizer-moments
n = 3; L = spd(n); h = rng.standard_normal(n)
A = lambda h, L: 0.5 * h @ inv(L) @ h - 0.5 * slogdet(L)[1] + n / 2 * np.log(2 * np.pi)
eps = 1e-6
g_h = np.array([(A(h + eps * e, L) - A(h - eps * e, L)) / (2 * eps) for e in np.eye(n)])
rep("g02 dA/dh = mu (FD)", np.abs(g_h - inv(L) @ h).max(), np.abs(inv(L) @ h).max())
E = np.zeros((n, n)); E[0, 1] = E[1, 0] = 1
dA = (A(h, L + eps * E) - A(h, L - eps * E)) / (2 * eps)
Sig = inv(L); mu = Sig @ h
rep("g02 dA[dLambda] = -1/2 Tr[(Sigma+mu mu^T) dLambda] (FD)", abs(dA + 0.5 * np.trace((Sig + np.outer(mu, mu)) @ E)), abs(dA))

# prop:gauss-conditional-precision (iii) witness
Lbb = np.diag([1., 10.]); W = np.ones((2, 2)); Lcc = 10 * np.eye(2)
J = np.block([[Lbb, -W], [-W, Lcc]])
note(f"g02 cond-precision witness leading minors {[round(det(J[:k, :k]), 9) for k in range(1, 5)]} (claimed 1,10,89,780)")
dc = np.array([1., -2.]); disp = inv(Lbb) @ W @ dc
note(f"g02 displacement {disp}, Euclidean pairing {disp @ dc} (claimed (-1,-0.1), -0.8); Lambda_bb pairing {dc @ Lbb @ disp}")

# lem:gauss-schur (i)-(iv), well and ill conditioned
for cond in [1e1, 1e8]:
    n = 4; L = spd(n, cond); Q, _ = np.linalg.qr(rng.standard_normal((n, n))); E_, F_ = Q[:, :2], Q[:, 2:]
    lhs = inv(E_.T @ inv(L) @ E_)
    rhs = E_.T @ L @ E_ - E_.T @ L @ F_ @ inv(F_.T @ L @ F_) @ F_.T @ L @ E_
    rep(f"g02 Schur (i) cond={cond:.0e} (relative)", np.abs(lhs - rhs).max() / np.abs(lhs).max())
    D = E_.T @ L @ E_ - lhs
    note(f"   (ii) min eig of difference {eigvalsh(D).min():.3e}, rank {matrix_rank(D, tol=1e-9 * np.abs(D).max())} vs rank(E^T L F) {matrix_rank(E_.T @ L @ F_)}")
    dd = slogdet(L)[1] - (slogdet(E_.T @ L @ E_)[1] - slogdet(F_.T @ inv(L) @ F_)[1])
    rep(f"   (iv) log det factorization cond={cond:.0e}", abs(dd), abs(slogdet(L)[1]))
L3 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2.]])
rep("g02 Schur (v) tridiagonal inverse", np.abs(inv(L3) - np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]]) / 4).max())

# cor:ig-full-rank-restriction-pushforward, non-orthogonal S
n, k = 4, 2; L = spd(n); S = rng.standard_normal((n, k)); M = S.T @ S
Lav = M @ inv(S.T @ inv(L) @ S) @ M
Mh = sqrtm(M).real; B = S @ inv(Mh)
rhs = Mh @ (B.T @ L @ B - inv(B.T @ inv(L) @ B)) @ Mh
rep("g02 full-rank restriction identity", np.abs(S.T @ L @ S - Lav - rhs).max(), np.abs(rhs).max())
rep("g02 Fisher of averaged pushforward", np.abs(inv(inv(M) @ S.T @ inv(L) @ S @ inv(M)) - Lav).max(), np.abs(Lav).max())

# prop:gauss-assembled-precision-blocks: chain 0 -> 1 -> 2, nonorthogonal links, noninvertible gains allowed
K, dm, do = 2, 3, 2
Th = [None, gl_plus(K, 50.), gl_plus(K, 50.)]
F = [None, rng.standard_normal((K, K)), np.array([[1., 2.], [0.5, 1.]])]  # second gain singular
Lam = [spd(K), spd(K), spd(K)]
Om = [None, F[1] @ Th[1], F[2] @ Th[2]]
N = 3; P = np.zeros((N * K, N * K))
blk = lambda i: slice(i * K, (i + 1) * K)
for i in range(N):
    R = np.zeros((K, N * K)); R[:, blk(i)] = np.eye(K)
    if i > 0:
        R[:, blk(i - 1)] = -Om[i]
    P += R.T @ Lam[i] @ R
pred = np.zeros_like(P)
for i in range(N):
    pred[blk(i), blk(i)] = Lam[i] + (Om[i + 1].T @ Lam[i + 1] @ Om[i + 1] if i + 1 < N else 0)
    if i > 0:
        pred[blk(i), blk(i - 1)] = -Lam[i] @ Om[i]
        pred[blk(i - 1), blk(i)] = (-Lam[i] @ Om[i]).T
rep("g02 assembled directed precision blocks", np.abs(P - pred).max(), np.abs(P).max())
Bi = rng.standard_normal((K, dm)); Cb = rng.standard_normal((do, K)); Cm = rng.standard_normal((do, dm)); So = spd(do); Li = spd(K)
Rk = np.hstack([np.eye(K), -Bi]); Ro = np.hstack([-Cb, -Cm])
H = Rk.T @ Li @ Rk + Ro.T @ inv(So) @ Ro
rep("g02 k-m block, K=2, d_m=3", np.abs(H[:K, K:] - (-Li @ Bi + Cb.T @ inv(So) @ Cm)).max(), np.abs(H[:K, K:]).max())

# prop:gen-lg-kernel-covariance and information-form frame law
g = gl_plus(K, cond=1e3); Sb = spd(K); mu = rng.standard_normal(K)
L2 = inv(g @ Sb @ g.T); Lp = inv(g).T @ inv(Sb) @ inv(g)
rep("g02 Lambda' = g^-T Lambda g^-1 (cond g = 1e3, relative)", np.abs(L2 - Lp).max() / np.abs(L2).max())
h = inv(Sb) @ mu; h2 = L2 @ (g @ mu)
rep("g02 information vector h' = g^-T h (relative)", np.abs(h2 - inv(g).T @ h).max() / np.abs(h2).max())
# full kernel identity with bridge: mean and cov of transformed kernel = pushforward
gi, gj = gl_plus(K, 20.), gl_plus(K, 20.); gm = gl_plus(dm, 20.)
Fb = rng.standard_normal((K, K)); Tb = gl_plus(K); Bb = rng.standard_normal((K, dm)); xi = rng.standard_normal(K); Sk = spd(K)
kj = rng.standard_normal(K); mi = rng.standard_normal(dm)
mean_new = (gi @ Fb @ inv(gi)) @ (gi @ Tb @ inv(gj)) @ (gj @ kj) + (gi @ Bb @ inv(gm)) @ (gm @ mi) + gi @ xi
rep("g02 kernel covariance mean identity", np.abs(mean_new - gi @ (Fb @ Tb @ kj + Bb @ mi + xi)).max(), np.abs(mean_new).max())

# prop:gen-lg-m-step identity and bound
m, p, T_ = 3, 4, 400
T = rng.standard_normal((T_, m)); Wr = np.hstack([rng.standard_normal((T_, p - 1)), np.ones((T_, 1))])
Mww = Wr.T @ Wr / T_; Fs = (T.T @ Wr / T_) @ inv(Mww)
Ss = (T - Wr @ Fs.T).T @ (T - Wr @ Fs.T) / T_
Ft = rng.standard_normal((m, p))
lhs = (T - Wr @ Ft.T).T @ (T - Wr @ Ft.T) / T_
rep("g02 M-step residual decomposition", np.abs(lhs - Ss - (Ft - Fs) @ Mww @ (Ft - Fs).T).max(), np.abs(lhs).max())
Sg = spd(m)
val = np.mean([-0.5 * (m * np.log(2 * np.pi) + slogdet(Sg)[1] + r @ inv(Sg) @ r) for r in (T - Wr @ Ft.T)])
note(f"g02 M-step bound: E log N = {val:.4f} <= -1/2 logdet(2 pi e Sigma*) = {-0.5 * slogdet(2 * np.pi * np.e * Ss)[1]:.4f}")

# prop:geo-selective-lift (ii),(iv) with nonorthogonal frames
Si = spd(4); SQ = rng.standard_normal((2, 4)); y = rng.standard_normal(2)
R = Si @ SQ.T @ inv(SQ @ Si @ SQ.T)
rep("g02 S^Q R = I", np.abs(SQ @ R - np.eye(2)).max())
gg = gl_plus(4, 1e2); phi = gl_plus(2, 10.)
SQp = phi @ SQ @ inv(gg); Sip = gg @ Si @ gg.T
Rp = Sip @ SQp.T @ inv(SQp @ Sip @ SQp.T)
rep("g02 lift frame law R' = g R phi^-1", np.abs(Rp - gg @ R @ inv(phi)).max(), np.abs(Rp).max())

# prop:obs-selective-gaussian-record (ii)
Lrec = np.array([[1., 0.]]); s2 = 0.37
I_ = 0.5 * np.log(det(np.array([[s2]]) + Lrec @ np.eye(2) @ Lrec.T) / s2)
rep("g02 record MI = 1/2 log(1+1/sigma^2)", abs(I_ - 0.5 * np.log(1 + 1 / s2)))
print("\n".join(out))
