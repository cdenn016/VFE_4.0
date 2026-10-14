"""L8 checks for g07 part A: aggregation, conditioning, Galerkin pushforward, exterior interface."""
import numpy as np
from numpy.linalg import inv, slogdet, eigvalsh, eigh
from scipy.linalg import sqrtm, null_space, eigh as geigh
from common import rng, out, rep, note, spd, kl

def interaction(N, K, edges, A):
    L = np.zeros((N * K, N * K))
    for i in range(N):
        L[i*K:(i+1)*K, i*K:(i+1)*K] += A[i]
    for (i, j), W in edges.items():
        L[i*K:(i+1)*K, i*K:(i+1)*K] += W; L[j*K:(j+1)*K, j*K:(j+1)*K] += W
        L[i*K:(i+1)*K, j*K:(j+1)*K] -= W; L[j*K:(j+1)*K, i*K:(i+1)*K] -= W
    return L

def psd(K, rank=None):
    G = rng.standard_normal((K, rank or K)); return G @ G.T

# prop:cg-gaussian-aggregation
N, K = 5, 2
edges = {(0, 1): psd(K), (1, 2): psd(K, 1), (0, 3): psd(K), (2, 4): psd(K), (3, 4): psd(K), (1, 4): psd(K)}
A = [psd(K, 1) for _ in range(N)]
L = interaction(N, K, edges, A)
part = [[0, 1], [2], [3, 4]]
Sh = np.zeros((N, 3))
for I, c in enumerate(part):
    for i in c: Sh[i, I] = 1
S = np.kron(Sh, np.eye(K)); Lc = S.T @ L @ S
Ac = [sum(A[i] for i in c) for c in part]
Ec = {}
for (i, j), W in edges.items():
    I = [k for k, c in enumerate(part) if i in c][0]; J = [k for k, c in enumerate(part) if j in c][0]
    if I != J:
        key = (min(I, J), max(I, J)); Ec[key] = Ec.get(key, 0) + W
rep("g07 aggregation blocks (A_I, W_IJ sums, internal annihilated)", np.abs(Lc - interaction(3, K, Ec, Ac)).max(), np.abs(Lc).max())

# prop:cg-aggregation-conditioning: conditional at w=0 equals N((B^T L B)^-1 B^T h, (B^T L B)^-1), B nonorthogonal
n, r = 4, 2; Lf = spd(n); hf = rng.standard_normal(n); Bn = rng.standard_normal((n, r)); Bp = null_space(Bn.T)
T = np.hstack([Bn, Bp]); Lyw = T.T @ Lf @ T; hyw = T.T @ hf  # density in (y,w)
Lyy = Lyw[:r, :r]; m_y = inv(Lyy) @ hyw[:r]
rep("g07 conditioning at w=0: precision", np.abs(Lyy - Bn.T @ Lf @ Bn).max())
rep("g07 conditioning at w=0: mean", np.abs(m_y - inv(Bn.T @ Lf @ Bn) @ Bn.T @ hf).max())

# prop:ig-galerkin-pushforward (ii),(iii),(v)
n = 6; Lf = spd(n, 1e3); S1 = rng.standard_normal((n, 4)); S2 = rng.standard_normal((4, 2))
P = lambda L, S: inv(S.T @ L @ S) @ S.T @ L
L1 = S1.T @ Lf @ S1
rep("g07 Galerkin nested composition (iii)", np.abs(P(L1, S2) @ P(Lf, S1) - P(Lf, S1 @ S2)).max(), np.abs(P(Lf, S1 @ S2)).max())
S = S1; Pl = P(Lf, S); Lc = S.T @ Lf @ S
D = Lf - Pl.T @ Lc @ Pl
note(f"g07 Galerkin Fisher defect min eig {eigvalsh(D).min():.3e} (scale {eigvalsh(Lf).max():.1e}); defect on range(S): {np.abs(D @ S).max():.3e}")
Sh = np.kron(np.array([[1., 0], [1, 0], [0, 1]]), np.eye(2))
L_id = np.eye(6); L_3 = np.eye(6); L_3[:2, :2] = 3 * np.eye(2)
note(f"g07 Galerkin (v): P_I = P_3? max diff {np.abs(P(L_id, Sh) - P(L_3, Sh)).max():.3f} (nonzero => no single kernel)")

# prop:cg-exterior-interface
n, m = 4, 3; Lam = spd(n, 50.); h = rng.standard_normal(n); C = rng.standard_normal((m, n)); So = spd(m); o0 = rng.standard_normal(m)
Lrec = C.T @ inv(So) @ C
dref, X = geigh(Lrec, Lam); idx = np.argsort(-dref); dref, X = dref[idx], X[:, idx]
rep("g07 exterior pencil Lambda-orthonormality", np.abs(X.T @ Lam @ X - np.eye(n)).max())
dk = dref / (1 + dref)
# canonical correlations of (Y,o)
SY = inv(Lam); SYo = SY @ C.T; Soo = C @ SY @ C.T + So
Kcc = inv(sqrtm(SY).real) @ SYo @ inv(sqrtm(Soo).real)
cc = np.sort(np.linalg.svd(Kcc, compute_uv=False))[::-1]
rep("g07 exterior canonical correlations rho_k^2 = d_k", np.abs(cc**2 - dk[:min(n, m)]).max())
IoY = 0.5 * np.sum(np.log1p(dref))
IoY_direct = 0.5 * (slogdet(Soo)[1] - slogdet(So)[1])
rep("g07 exterior I(o;Y) = 1/2 sum log(1+d_ref)", abs(IoY - IoY_direct), IoY)
for r in [1, 2]:
    Sr = X[:, :r]; Pt = Sr @ Sr.T @ Lam
    # I(o; Pt Y) via coordinates c_1..c_r = Sr^T Lam Y
    Wst = Sr.T @ Lam
    Sc = Wst @ SY @ Wst.T; Sco = Wst @ SYo
    I_r = 0.5 * (slogdet(Sc)[1] - slogdet(Sc - Sco @ inv(Soo) @ Sco.T)[1])
    rep(f"g07 exterior I(o;P_r Y) r={r}", abs(I_r - 0.5 * np.sum(np.log1p(dref[:r]))), I_r)
    # optimality vs random rank-r W
    best = max(0.5 * (slogdet(Wm @ SY @ Wm.T)[1] - slogdet(Wm @ SY @ Wm.T - Wm @ SYo @ inv(Soo) @ SYo.T @ Wm.T)[1])
               for Wm in [rng.standard_normal((r, n)) for _ in range(2000)])
    note(f"   r={r}: max over 2000 random rank-{r} W of I(o;WY) = {best:.6f} <= {I_r:.6f}")
    # HS tail bounds
    disc = 0.5 * np.sum(np.log1p(dref[r:]))
    Sext = Soo
    HSo = np.trace(inv(Lam) @ (np.eye(n) - Pt).T @ C.T @ inv(So) @ C @ (np.eye(n) - Pt))
    HSe = np.trace(inv(Lam) @ (np.eye(n) - Pt).T @ C.T @ inv(Sext) @ C @ (np.eye(n) - Pt))
    rep(f"   HS_r(So^-1) = sum_{{k>r}} d_ref", abs(HSo - dref[r:].sum()), HSo)
    rep(f"   HS_r(Sext^-1) = sum_{{k>r}} rho^2", abs(HSe - dk[r:].sum()), HSe)
    note(f"   bounds {0.5*HSe:.6f} <= {disc:.6f} <= {0.5*HSo:.6f}")
    # surrogate evidence formula: expected log Zhat/p under p(o) computed in closed form
    # Zhat: o = C Pt Y + o0 + xi, Y ~ N(SY h, SY)
    mY = SY @ h
    m_p = C @ mY + o0; S_p = Soo
    m_s = C @ Pt @ mY + o0; S_s = C @ Pt @ SY @ Pt.T @ C.T + So
    lhs = -kl(m_p, S_p, m_s, S_s)
    rhs = -0.5 * np.sum(dref[r:] - np.log1p(dref[r:]) + dref[r:] * (X[:, r:].T @ h) ** 2)
    rep(f"   surrogate evidence identity (eq:cg-exterior-evidence) r={r}", abs(lhs - rhs), abs(lhs))
    # Galerkin identification P_{Lam+Lrec} = P_Lam = Sr^T Lam
    rep(f"   P_(Lam+Lrec) = Sr^T Lam", np.abs(P(Lam + Lrec, Sr) - Sr.T @ Lam).max())
# counterexample variances
th = 0.7
note(f"g07 exterior counterexample: posterior var {1-0.5*np.cos(th)**2:.6f}, surrogate {1/(1+np.cos(th)**2):.6f}")
print("\n".join(out))
