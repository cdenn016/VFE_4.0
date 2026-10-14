"""L8 checks for g04_restrictions.tex."""
import numpy as np
from numpy.linalg import inv, slogdet
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from common import rng, out, rep, note, spd, kl

for cond in [1e1, 1e10]:
    n = 4; L = spd(n, cond); mu = rng.standard_normal(n); muQ = rng.standard_normal(n); SQ = spd(n)
    v1 = 0.5 * (np.trace(L @ SQ) + (muQ - mu) @ L @ (muQ - mu) - n - slogdet(L)[1] - slogdet(SQ)[1])
    v2 = kl(muQ, SQ, mu, inv(L))
    rep(f"g04 reverse KL precision form vs covariance form cond={cond:.0e} (relative)", abs(v1 - v2) / abs(v1))

n = 4; L = spd(n); mu = rng.standard_normal(n); blocks = [[0, 1], [2, 3]]
def obj(x):
    muQ = x[:4]; SQ = np.zeros((4, 4))
    for b, idx in enumerate(blocks):
        Lc = np.zeros((2, 2)); Lc[np.tril_indices(2)] = x[4 + 3 * b: 7 + 3 * b]
        Lc[np.diag_indices(2)] = np.exp(np.diag(Lc))
        SQ[np.ix_(idx, idx)] = Lc @ Lc.T
    return 0.5 * (np.trace(L @ SQ) + (muQ - mu) @ L @ (muQ - mu) - n - slogdet(L)[1] - slogdet(SQ)[1])
res = minimize(obj, np.zeros(10), method="BFGS", options={"gtol": 1e-11})
gap = 0.5 * (sum(slogdet(L[np.ix_(b, b)])[1] for b in blocks) - slogdet(L)[1])
rep("g04 block optimum value vs determinant gap (BFGS)", abs(res.fun - gap), gap)
L11, L22, L12 = L[:2, :2], L[2:, 2:], L[:2, 2:]
Kc = inv(sqrtm(L11).real) @ L12 @ inv(sqrtm(L22).real)
rep("g04 two-block gap = -1/2 logdet(I - K^T K)", abs(gap + 0.5 * slogdet(np.eye(2) - Kc.T @ Kc)[1]), gap)
Sg = np.full((3, 3), 0.5) + 0.5 * np.eye(3); Lg = inv(Sg)
note(f"g04 TC {0.5*(-slogdet(Sg)[1]):.15f} vs {0.5*np.log(2):.15f}; DTC {0.5*(3*np.log(Lg[0,0]) - slogdet(Lg)[1]):.15f} vs {0.5*np.log(27/16):.15f}")
b4 = [[0], [1, 2], [3]]; inc = 0
for k in range(1, 3):
    prev = sum(b4[:k], []); cur = sum(b4[:k + 1], [])
    inc += 0.5 * (slogdet(L[np.ix_(prev, prev)])[1] + slogdet(L[np.ix_(b4[k], b4[k])])[1] - slogdet(L[np.ix_(cur, cur)])[1])
tot = 0.5 * (sum(slogdet(L[np.ix_(b, b)])[1] for b in b4) - slogdet(L)[1])
rep("g04 telescoping", abs(inc - tot), tot)

Q, _ = np.linalg.qr(rng.standard_normal((4, 4))); B, Bp = Q[:, :2], Q[:, 2:]
mup = Bp.T @ mu; cost = 0.5 * mup @ inv(Bp.T @ inv(L) @ Bp) @ mup
muQs = mu - inv(L) @ Bp @ inv(Bp.T @ inv(L) @ Bp) @ mup
rep("g04 mean-cost optimizer lies in range(B)", np.abs(Bp.T @ muQs).max(), np.abs(mu).max())
res2 = minimize(lambda y: 0.5 * (B @ y - mu) @ L @ (B @ y - mu), np.zeros(2), method="BFGS", options={"gtol": 1e-12})
rep("g04 mean cost vs BFGS", abs(res2.fun - cost), cost)
for e_ in [1e-2, 1e-8]:
    Se = B @ inv(B.T @ L @ B) @ B.T + e_ * Bp @ Bp.T
    direct = kl(muQs, Se, mu, inv(L))
    formula = cost + np.log(1 / e_) - 1 + 0.5 * slogdet(Bp.T @ inv(L) @ Bp)[1] + e_ / 2 * np.trace(Bp.T @ L @ Bp)
    rep(f"g04 epsilon-identification KL eps={e_:.0e}", abs(direct - formula), abs(direct))
note(f"g04 hard-trace Var(o) = {inv(np.array([[3, -1], [-1, 1.]]))[1, 1]} (claimed 1.5)")
n = 4; Lh = spd(n + 2); LY, LYo = Lh[:n, :n], Lh[:n, n:]; hY = rng.standard_normal(n); o = rng.standard_normal(2)
S = np.array([[1, 0], [1, 0], [0, 1], [0, 1.]])
muP = inv(LY) @ (hY - LYo @ o); PL = inv(S.T @ LY @ S) @ S.T @ LY
rep("g04 Galerkin channel mean", np.abs(PL @ muP - inv(S.T @ LY @ S) @ S.T @ (hY - LYo @ o)).max(), np.abs(PL @ muP).max())
rep("g04 Galerkin channel cov", np.abs(PL @ inv(LY) @ PL.T - inv(S.T @ LY @ S)).max())
a, b = 0.8, 0.2
note(f"g04 nonnested gaps {0.5*np.log((1-a*a)/(1-a*a-b*b)):.4f}, {0.5*np.log((1-b*b)/(1-a*a-b*b)):.4f} (claimed 0.0589, 0.5493)")
print("\n".join(out))
