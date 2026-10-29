"""D24-LDYN script 2: numerical checks of it2:gauge, it2:split, it2:grounding,
it2:global-flow, it2:per-agent-flow, it2:separation on a random three-agent tangent model
with two two-cycles and one directed triangle (K=2, codes of rank 1 or 2)."""
import numpy as np
from tangent_lib import Model, kappa, code_kron, null_basis, spd

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(20261029)
edges = [(1, 0), (0, 1), (2, 1), (0, 2), (2, 0)]    # (sender, receiver)
M = Model(3, 2, edges, rng, sigma2=1.3, lam=0.8, ranks=[1, 2, 1, 1, 2])
print("loops (edge indices):", M.loops, " ranks:", [T.shape[0] for T in M.T])
J, eta = M.global_J_eta()
xs = np.linalg.solve(J, eta)
print("J positive definite:", np.linalg.eigvalsh(J).min() > 0)

# it2:gauge
Lall = np.vstack(M.Lmat)
Gb, _ = null_basis(Lall)
ycols = np.concatenate(M.iy)
wcols = np.concatenate(M.iw)
print("\n[it2:gauge] dim G =", Gb.shape[1], " rank of belief projection =",
      np.linalg.matrix_rank(Gb[ycols, :]), "of", len(ycols))
dy = rng.normal(size=(M.N, M.K))
d = np.zeros(M.n)
for a in range(M.N):
    d[M.iy[a]] = dy[a]
for k, (j, i) in enumerate(edges):
    dw = np.outer(dy[i] - dy[j], M.c[i]) / (M.c[i] @ M.c[i])
    d[M.iw[k]] = dw.reshape(-1)
print("  constructed completion: max |L_e d| =", max(np.abs(L @ d).max() for L in M.Lmat))
F = sum(L.T @ np.linalg.inv(R) @ L for L, R in zip(M.Lmat, M.R))
Fb, _ = null_basis(F)
print("  null(F) == G: dims", Fb.shape[1], Gb.shape[1], " rank[G,nullF] =",
      np.linalg.matrix_rank(np.hstack([Gb, Fb]), tol=1e-8))
v = Gb @ rng.normal(size=Gb.shape[1])
print("  for v in G: v'Jv - v'(J-F)v =", v @ J @ v - v @ (J - F) @ v)
Zs, Ys = [], []
for k in range(M.E):
    Zk, Yk = null_basis(code_kron(M.T[k]))
    Zs.append(Zk)
    Ys.append(Yk)
def blockdiag(mats):
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r0 = c0 = 0
    for m in mats:
        out[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] = m
        r0 += m.shape[0]
        c0 += m.shape[1]
    return out
Zbig = blockdiag(Zs)
Ybig = blockdiag(Ys)
Lr = np.hstack([Lall[:, ycols], Lall[:, wcols] @ Zbig])
Nb, _ = null_basis(Lr)
Bproj = Nb[:len(ycols), :]
rb = np.linalg.matrix_rank(Bproj, tol=1e-9)
rowsd = []
for k, (j, i) in enumerate(edges):
    blocks = []
    for a in range(M.N):
        if a == i:
            blocks.append(M.T[k])
        elif a == j:
            blocks.append(-M.T[k])
        else:
            blocks.append(np.zeros_like(M.T[k]))
    rowsd.append(np.hstack(blocks))
delta = np.vstack(rowsd)
H0, _ = null_basis(delta)
same = np.linalg.matrix_rank(np.hstack([Bproj, H0]), tol=1e-8) == H0.shape[1]
print("  (iii) dim belief part with omega in N =", rb, " dim code-sheaf H0 =", H0.shape[1], " same subspace:", same)
shift = np.tile(rng.normal(size=M.K), M.N)
print("  common shift in H0: |delta shift| =", np.abs(delta @ shift).max())

# it2:split
print("\n[it2:split]")
for trial in range(3):
    K = 3
    n = 1 + trial % 3
    T = rng.normal(size=(n, K))
    R = spd(n, rng, 0.3, 2)
    c = rng.normal(size=K)
    Si, Sj, V = spd(K, rng, 0.3, 2), spd(K, rng, 0.3, 2), spd(K * K, rng, 0.3, 2)
    mi, mj, wb = rng.normal(size=K), rng.normal(size=K), rng.normal(size=K * K)
    o = rng.normal(size=n)
    kap = kappa(c)
    Mz = np.hstack([T, -T, -T @ kap])
    Cz = blockdiag([Si, Sj, V])
    mz = np.concatenate([mi, mj, wb])
    r = o - Mz @ mz
    S = Mz @ Cz @ Mz.T + R
    shift = Cz @ Mz.T @ np.linalg.solve(S, r)
    Crho = kap @ V @ kap.T
    S_formula = T @ (Si + Sj + Crho) @ T.T + R
    g_ = np.linalg.solve(S_formula, r)
    e1 = np.abs(shift[:K] - Si @ T.T @ g_).max()
    e2 = np.abs(shift[K:2 * K] + Sj @ T.T @ g_).max()
    e3 = np.abs(kap @ shift[2 * K:] + Crho @ T.T @ g_).max()
    Mp = np.hstack([T, -T @ kap])
    Cp = blockdiag([Si, V])
    Rp = R + T @ Sj @ T.T
    rp = o - (Mp @ np.concatenate([mi, wb]) - T @ mj)
    Sp = Mp @ Cp @ Mp.T + Rp
    shp = Cp @ Mp.T @ np.linalg.solve(Sp, rp)
    Sprime = T @ (Si + Crho) @ T.T + R + T @ Sj @ T.T
    gp = np.linalg.solve(Sprime, r)
    e4 = np.abs(shp[:K] - Si @ T.T @ gp).max()
    e5 = np.abs(kap @ shp[K:] + Crho @ T.T @ gp).max()
    print(f"  trial {trial} (n={n}): global residuals {e1:.1e} {e2:.1e} {e3:.1e}; per-agent {e4:.1e} {e5:.1e}; "
          f"|S'-S| = {np.abs(Sprime - S_formula).max():.1e}; |per-agent dy_i - global dy_i| = {np.abs(shp[:K]-shift[:K]).max():.1e}")
sig2 = 0.7
print("  (iii) isotropic: |sigma^2 kappa kappa^T - sigma^2|c|^2 I| =",
      np.abs(sig2 * kappa(c) @ kappa(c).T - sig2 * (c @ c) * np.eye(K)).max())

# it2:grounding
print("\n[it2:grounding]")
K = 3
T = rng.normal(size=(1, K))
c = rng.normal(size=K)
kap = kappa(c)
TK = code_kron(T)
Nk, Rk = null_basis(TK)
Pi = Rk @ Rk.T
Lam = spd(K * K, rng, 0.5, 2)
Gbar = TK.T @ spd(TK.shape[0], rng, 0.5, 2) @ TK
for g in (1e1, 1e3, 1e5):
    V = np.linalg.inv(Lam + g * Gbar)
    lhs = T @ kap @ V @ kap.T @ T.T
    rhs = T @ kap @ Pi @ V @ Pi @ kap.T @ T.T
    print(f"  g={g:.0e}: |T C T' - T k Pi V Pi k' T'| = {np.abs(lhs - rhs).max():.1e}; g*|Pi V Pi| = {g*np.abs(Pi @ V @ Pi).max():.4f}; "
          f"g*|kappa V kappa' T'| = {g*np.abs(kap @ V @ kap.T @ T.T).max():.4f}")

# it2:global-flow
print("\n[it2:global-flow]")
def FG_direct(mu, Ss):
    Sfull = np.zeros((M.n, M.n))
    for a, bl in enumerate(M.blocks):
        Sfull[np.ix_(bl, bl)] = Ss[a]
    return 0.5 * mu @ J @ mu - eta @ mu + 0.5 * np.trace(J @ Sfull) - 0.5 * sum(np.linalg.slogdet(S)[1] for S in Ss)
def FG_claim(mu, Ss):
    tot = 0.5 * (mu - xs) @ J @ (mu - xs)
    for a, bl in enumerate(M.blocks):
        tot += 0.5 * (np.trace(J[np.ix_(bl, bl)] @ Ss[a]) - np.linalg.slogdet(Ss[a])[1])
    return tot
diffs = []
for t in range(4):
    mu = rng.normal(size=M.n)
    Ss = [spd(len(bl), rng, 0.2, 3) for bl in M.blocks]
    diffs.append(FG_direct(mu, Ss) - FG_claim(mu, Ss))
print("  F_G(direct) - F_G(claimed) over 4 random (mu,S): spread =", np.ptp(diffs))
D = np.zeros_like(J)
for bl in M.blocks:
    D[np.ix_(bl, bl)] = J[np.ix_(bl, bl)]
ev = np.linalg.eigvals(-np.linalg.solve(D, J))
print("  -D^{-1}J: max|Im| =", np.abs(ev.imag).max(), " max Re =", ev.real.max())
Jinv = np.linalg.inv(J)
print("  min eig of (exact marginal cov - J_ii^{-1}) per block:",
      [round(np.linalg.eigvalsh(Jinv[np.ix_(bl, bl)] - np.linalg.inv(J[np.ix_(bl, bl)])).min(), 6) for bl in M.blocks])

# it2:per-agent-flow, plug-in
print("\n[it2:per-agent-flow] plug-in")
Hp, bp, Ap = M.per_agent_H_b('plug')
print("  dictionary rows: |H - J| =", np.abs(Hp[wcols] - J[wcols]).max(), " |b - eta| =", np.abs(bp[wcols] - eta[wcols]).max())
def sender_terms(Rs):
    s1 = np.zeros_like(J)
    s2 = np.zeros_like(eta)
    for a in range(M.N):
        for k in M.out[a]:
            Ri = np.linalg.inv(Rs[k])
            L = M.Lmat[k]
            s1[M.iy[a]] += (L.T @ Ri @ L)[M.iy[a]]
            s2[M.iy[a]] += (L.T @ Ri @ M.o[k])[M.iy[a]]
    return s1, s2
send, sende = sender_terms(M.R)
print("  content rows: |H - (J - sender)| =", np.abs(Hp[ycols] - (J - send)[ycols]).max(),
      " |b - (eta - sender)| =", np.abs(bp[ycols] - (eta - sende)[ycols]).max())
print("  |H - H^T| =", np.abs(Hp - Hp.T).max(), "  |mu_pa - x*| =", np.abs(np.linalg.solve(Hp, bp) - xs).max())
evp = np.linalg.eigvals(-np.linalg.solve(Ap, Hp))
print("  -A^{-1}H: max Re =", evp.real.max(), " max|Im| =", np.abs(evp.imag).max())
k = 0
(j, i) = edges[k]
blk = Hp[np.ix_(M.iy[i], M.iy[j])] + M.T[k].T @ np.linalg.inv(M.R[k]) @ M.T[k]
blk_rev = Hp[np.ix_(M.iy[j], M.iy[i])]
opp = [f for f, (jj, ii) in enumerate(edges) if (jj, ii) == (i, j)]
rev_pred = -sum(M.T[f].T @ np.linalg.inv(M.R[f]) @ M.T[f] for f in opp) if opp else 0 * blk_rev
print("  edge 1->0: |H_{y0 y1} + T'R^{-1}T| =", np.abs(blk).max(), "; H_{y1 y0} equals the opposite edge's own term only:", np.abs(blk_rev - rev_pred).max())

# law-level covariance fixed point: uniqueness from two starts, subhomogeneity, monotonicity
print("\n[law-level covariance fixed point]")
S_lo, V_lo, it_lo = M.law_level_fixed_point(start=0.0)
S_hi, V_hi, it_hi = M.law_level_fixed_point(start=50.0)
dmax = max(max(np.abs(a - b).max() for a, b in zip(S_lo, S_hi)), max(np.abs(a - b).max() for a, b in zip(V_lo, V_hi)))
print("  iterations", it_lo, it_hi, " max diff between limits from low and high starts:", dmax)
worst_sub, worst_mono = np.inf, np.inf
for t in range(20):
    X = [spd(M.K, rng, 0.01, 3) for _ in range(M.N)]
    Xv = [spd(M.d, rng, 0.01, 3) for _ in range(M.E)]
    tt = rng.uniform(0.05, 0.95)
    A1, B1 = M.law_level_cov_map([tt * x for x in X], [tt * x for x in Xv])
    A2, B2 = M.law_level_cov_map(X, Xv)
    worst_sub = min(worst_sub, min(np.linalg.eigvalsh(a - tt * b).min() for a, b in zip(A1 + B1, A2 + B2)))
    Y = [x + spd(M.K, rng, 0.0, 1) for x in X]
    Yv = [x + spd(M.d, rng, 0.0, 1) for x in Xv]
    A3, B3 = M.law_level_cov_map(Y, Yv)
    worst_mono = min(worst_mono, min(np.linalg.eigvalsh(a - b).min() for a, b in zip(A3 + B3, A2 + B2)))
print("  min eig of Phi(tX) - t Phi(X), 20 draws (claim > 0):", worst_sub)
print("  min eig of Phi(Y) - Phi(X) for Y >= X, 20 draws (claim >= 0):", worst_mono)
Hl, bl_, Al = M.per_agent_H_b('law', S_lo, V_lo)
print("  law-level dictionary rows: |H - J| =", np.abs(Hl[wcols] - J[wcols]).max(), "(nonzero expected)")
Rt = [M.R[k] + M.T[k] @ S_lo[edges[k][0]] @ M.T[k].T for k in range(M.E)]
Jt, etat = M.global_J_eta(Rrep=Rt)
sendt, _ = sender_terms(Rt)
print("  law-level content rows: |H - (J[R->R~] - sender)| =", np.abs(Hl[ycols] - (Jt - sendt)[ycols]).max())
evl = np.linalg.eigvals(-np.linalg.solve(Al, Hl))
print("  law-level -A^{-1}H: max Re =", evl.real.max(), " max|Im| =", np.abs(evl.imag).max())

# it2:separation
print("\n[it2:separation]")
def known_system(mode):
    A = np.zeros((M.N * M.K,) * 2)
    bb = np.zeros(M.N * M.K)
    for a in range(M.N):
        A[np.ix_(M.iy[a], M.iy[a])] += M.P[a]
        bb[M.iy[a]] += M.P[a] @ M.m[a]
    for k, (j, i) in enumerate(edges):
        Ri = np.linalg.inv(M.R[k])
        ok = M.o[k] + M.T[k] @ kappa(M.c[i]) @ M.what[k]
        TT = M.T[k].T @ Ri @ M.T[k]
        A[np.ix_(M.iy[i], M.iy[i])] += TT
        A[np.ix_(M.iy[i], M.iy[j])] -= TT
        bb[M.iy[i]] += M.T[k].T @ Ri @ ok
        if mode == 'global':
            A[np.ix_(M.iy[j], M.iy[j])] += TT
            A[np.ix_(M.iy[j], M.iy[i])] -= TT
            bb[M.iy[j]] -= M.T[k].T @ Ri @ ok
    return np.linalg.solve(A, bb)
yG, yP = known_system('global'), known_system('plug')
Lprior = np.eye(len(wcols)) / M.sigma2 + M.lam * sum((B.T @ W @ B)[np.ix_(wcols, wcols)] for B, W in zip(M.Bg, M.W))
wh = np.concatenate(M.what)
nu = -np.linalg.solve(Zbig.T @ Lprior @ Zbig, Zbig.T @ Lprior @ wh)
for g in (1e2, 1e4, 1e6):
    M.gscale = g
    Jg, etag = M.global_J_eta()
    xg = np.linalg.solve(Jg, etag)
    Hg, bg, Ag = M.per_agent_H_b('plug')
    xp = np.linalg.solve(Hg, bg)
    ey_g, ey_p = np.abs(xg[ycols] - yG).max(), np.abs(xp[ycols] - yP).max()
    ea_g, ea_p = np.abs(Ybig.T @ (xg[wcols] - wh)).max(), np.abs(Ybig.T @ (xp[wcols] - wh)).max()
    en_g, en_p = np.abs(Zbig.T @ (xg[wcols] - wh) - nu).max(), np.abs(Zbig.T @ (xp[wcols] - wh) - nu).max()
    print(f"  g={g:.0e}: global |y-yG|={ey_g:.2e} (g*err {g*ey_g:.3f}) |alpha|={ea_g:.2e} |nu-lim|={en_g:.2e}; "
          f"plug-in |y-yP|={ey_p:.2e} (g*err {g*ey_p:.3f}) |alpha|={ea_p:.2e} |nu-lim|={en_p:.2e}")
print("  |yG - yP| =", np.abs(yG - yP).max(), " (limits differ)")
M.gscale = 1.0

print("\n[companion example] scalar edge T=c=R=Sigma_j=1, o=0, mu_i-mu_j=1, a=0")
grad_global = 1.0 * (1.0 / 1.0) * (0 - 1 + 0)
grad_law = 1.0 * (1.0 / (1.0 + 1.0)) * (0 - 1 + 0)
print("  global record derivative in dictionary:", grad_global, " law-level:", grad_law)
