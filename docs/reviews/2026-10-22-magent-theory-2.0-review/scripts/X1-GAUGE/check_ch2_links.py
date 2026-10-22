"""X1-GAUGE: checks of chapter 2 (interactions) link algebra, supplement s02, and the
law-dressed loop defect proposed in the review. float64, small sizes, single thread."""
import numpy as np
from scipy.linalg import expm, null_space, orth

rng = np.random.default_rng(20261022)
d = 3

def rand_gl(d):
    while True:
        A = rng.normal(size=(d, d))
        if abs(np.linalg.det(A)) > 0.3:
            return A

def rand_spd(d):
    A = rng.normal(size=(d, d))
    return A @ A.T + 0.5 * np.eye(d)

# 1. prop:int-moment-equivariance: L = lam mu_i mu_j^T M_j^{-1}, reference change h
mu_i, mu_j = rng.normal(size=d), rng.normal(size=d)
M_j = rand_spd(d)
lam = 0.7
L = lam * np.outer(mu_i, mu_j) @ np.linalg.inv(M_j)
h = rand_gl(d)
hi = np.linalg.inv(h)
mu_i2, mu_j2 = hi @ mu_i, hi @ mu_j
M_j2 = hi @ M_j @ hi.T
L2 = lam * np.outer(mu_i2, mu_j2) @ np.linalg.inv(M_j2)
print("moment-family reference equivariance |L' - h^-1 L h| =", np.abs(L2 - hi @ L @ h).max())

# 2. prop:int-endpoint: Theta' = g_i Theta g_j^-1 with reference laws fixed (generator unchanged)
U_i, U_j, g_i, g_j = rand_gl(d), rand_gl(d), rand_gl(d), rand_gl(d)
Th = U_i @ expm(L) @ np.linalg.inv(U_j)
Th2 = (g_i @ U_i) @ expm(L) @ np.linalg.inv(g_j @ U_j)
print("endpoint covariance |Theta' - g_i Theta g_j^-1| =", np.abs(Th2 - g_i @ Th @ np.linalg.inv(g_j)).max())
# common reference change U -> U h with L -> h^-1 L h leaves Theta unchanged
Th3 = (U_i @ h) @ expm(hi @ L @ h) @ np.linalg.inv(U_j @ h)
print("reference-rechoice invariance |Theta(Uh, h^-1Lh) - Theta| =", np.abs(Th3 - Th).max())

# 3. eq:int-nongaussian-loop
E12 = np.array([[0., 1.], [0., 0.]]); E21 = E12.T
loop = expm(E12) @ expm(E21)
print("non-Gaussian loop Theta12 Theta21 =", loop.tolist(), " trace", np.trace(loop), " det", np.linalg.det(loop))
print("Theta21 == Theta12^-1 ?", np.allclose(expm(E21), np.linalg.inv(expm(E12))))

# 4. prop:int-rank and lem:sgeom-lowrank (including singular TO)
r = 2
O = rng.normal(size=(d, r)); T = rng.normal(size=(r, d))
E = expm(O @ T)
v = null_space(T)[:, 0]
print("e^{OT} v - v on ker T:", np.abs(E @ v - v).max())
P_perp = np.eye(d) - orth(O) @ orth(O).T
print("(I-P_imO)(e^{OT}-I):", np.abs(P_perp @ (E - np.eye(d))).max())
def phi1(Z, n=60):
    S = np.zeros_like(Z); P = np.eye(Z.shape[0]); fact = 1.0
    for k in range(1, n):
        fact *= k
        S = S + P / fact
        P = P @ Z
    return S
print("low-rank identity |e^{OT} - (I + O phi1(TO) T)| =", np.abs(E - (np.eye(d) + O @ phi1(T @ O) @ T)).max())
# singular TO: make T O nilpotent rank-one
O2 = np.array([[1., 0.], [0., 1.], [0., 0.]]); T2 = np.array([[0., 1., 0.], [0., 0., 0.]])
print("singular TO case:", np.abs(expm(O2 @ T2) - (np.eye(3) + O2 @ phi1(T2 @ O2) @ T2)).max())
# rank-one det
u, w = rng.normal(size=d), rng.normal(size=d)
Lr = np.outer(u, w); t = w @ u
print("rank-one closed form:", np.abs(expm(Lr) - (np.eye(d) + (np.expm1(t) / t) * Lr)).max(), " det-e^t:", np.linalg.det(expm(Lr)) - np.exp(t))

# 5. lem:sgeom-frechet: compare quadrature with block-matrix exact derivative
Lm = rng.normal(size=(d, d)) * 0.7; H = rng.normal(size=(d, d))
blk = expm(np.block([[Lm, H], [np.zeros((d, d)), Lm]]))[:d, d:]
ts = np.linspace(0, 1, 2001)
quad = np.trapezoid(np.array([expm((1 - s) * Lm) @ H @ expm(s * Lm) for s in ts]), ts, axis=0)
print("Frechet derivative quadrature vs block exact:", np.abs(quad - blk).max())

# 6. Dressing: reference links are passively invariant, so even NON-loop words are observables
#    under the passive per-agent group; under the common reference change only conjugation acts,
#    so traces of arbitrary words (not only closed loops) are invariant.
L12 = rng.normal(size=(d, d)) * 0.5; L21 = rng.normal(size=(d, d)) * 0.5
W_open = expm(L12)                       # reference link of a single (open) edge
print("tr of an open reference edge under common reference change: before", np.trace(W_open),
      " after", np.trace(expm(hi @ L12 @ h)))
# displayed open edge is not passively invariant, displayed loop trace is
Th12 = U_i @ expm(L12) @ np.linalg.inv(U_j); Th21 = U_j @ expm(L21) @ np.linalg.inv(U_i)
Th12p = g_i @ Th12 @ np.linalg.inv(g_j); Th21p = g_j @ Th21 @ np.linalg.inv(g_i)
print("displayed open-edge trace changes under passive g:", np.trace(Th12), "->", np.trace(Th12p))
print("displayed 2-cycle trace invariant:", np.trace(Th12 @ Th21) - np.trace(Th12p @ Th21p))

# 7. Law-dressed loop defect Q = 1/2 tr(M (I-W)^T M^{-1} (I-W)) with M the receiver covariance
def Q_law(W, M):
    I = np.eye(W.shape[0])
    return 0.5 * np.trace(M @ (I - W).T @ np.linalg.inv(M) @ (I - W))
Mi_ref = rand_spd(d)                     # receiver covariance in reference coordinates
W_ref = expm(L12) @ expm(L21)            # reference 2-cycle at i
# passive change: displayed W = U_i W_ref U_i^-1, displayed M = U_i M_ref U_i^T
Wd = U_i @ W_ref @ np.linalg.inv(U_i); Md = U_i @ Mi_ref @ U_i.T
Wd2 = g_i @ Wd @ np.linalg.inv(g_i); Md2 = g_i @ Md @ g_i.T
print("law-dressed defect: reference", Q_law(W_ref, Mi_ref), " displayed", Q_law(Wd, Md), " after passive g", Q_law(Wd2, Md2))
# common reference change h: W -> h^-1 W h, M -> h^-1 M h^-T
print("law-dressed defect after common reference change:", Q_law(hi @ W_ref @ h, hi @ Mi_ref @ hi.T))
# fixed (background) Frobenius metric is not invariant under non-orthogonal h
fro = lambda W: 0.5 * np.linalg.norm(np.eye(d) - W) ** 2
print("background Frobenius defect: before", fro(W_ref), " after non-orthogonal h", fro(hi @ W_ref @ h))
# separation on the unipotent shear, where every class function equals its identity value
Wsh = np.eye(d); Wsh[0, 1] = 1.0
print("shear: eigenvalues", np.linalg.eigvals(Wsh), " trace", np.trace(Wsh), " law-dressed Q", Q_law(Wsh, Mi_ref))
# conjugating the shear toward identity: class functions constant, law-dressed defect is NOT constant
# because M is conjugated along (it is an orbit invariant of the pair (W, M), not a class function of W)
for tt in [1.0, 0.1, 0.01]:
    S = np.diag([1.0, tt, tt ** 2])
    Wc = np.linalg.inv(S) @ Wsh @ S
    Mc = np.linalg.inv(S) @ Mi_ref @ np.linalg.inv(S).T
    print(f"  t={tt}: |W_t - I| = {np.abs(Wc - np.eye(d)).max():.3e}, Q(W_t, M_t) = {Q_law(Wc, Mc):.6f}, Q(W_t, M fixed) = {Q_law(Wc, Mi_ref):.6f}")
