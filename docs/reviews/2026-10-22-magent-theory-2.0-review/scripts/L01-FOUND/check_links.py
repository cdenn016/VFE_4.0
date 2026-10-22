r"""L01-FOUND checks of the matrix identities in 02_interactions.tex and s02_geometry.tex.

Each block prints a residual. Numerical agreement supports, and does not prove, an identity.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python check_links.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import expm, expm_frechet, sqrtm, logm, eigvals

rng = np.random.default_rng(20261022)


def phi1(Z: np.ndarray) -> np.ndarray:
    r"""\varphi_1(Z)=\sum_{n\ge1} Z^{n-1}/n!, computed from expm([[Z, I],[0, 0]]) = [[e^Z, \varphi_1(Z)],[0, I]]."""
    r = Z.shape[0]
    B = np.zeros((2 * r, 2 * r))
    B[:r, :r] = Z
    B[:r, r:] = np.eye(r)
    return expm(B)[:r, r:]


def gauss_kl(m0, S0, m1, S1) -> float:
    r"""KL(N(m0,S0) || N(m1,S1))."""
    d = len(m0)
    S1i = np.linalg.inv(S1)
    dm = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + dm @ S1i @ dm - d + np.log(np.linalg.det(S1) / np.linalg.det(S0)))


def report(name: str, val: float) -> None:
    print(f"{name:70s} residual = {val:.3e}")


# lem:sgeom-lowrank, eq:sgeom-lowrank, including singular and nilpotent TO
for d, r in [(3, 1), (4, 2), (5, 3), (6, 6)]:
    O = rng.normal(size=(d, r))
    T = rng.normal(size=(r, d))
    report(f"eq:sgeom-lowrank generic d={d} r={r}", np.abs(expm(O @ T) - (np.eye(d) + O @ phi1(T @ O) @ T)).max())
# singular TO: T O = 0 (choose T rows orthogonal to columns of O)
d, r = 4, 2
O = np.zeros((d, r)); O[0, 0] = 1.0; O[1, 1] = 2.0
T = np.zeros((r, d)); T[0, 2] = 1.0; T[1, 3] = -1.5
report("eq:sgeom-lowrank with TO = 0", np.abs(expm(O @ T) - (np.eye(d) + O @ T)).max())
# nilpotent nonzero TO
O = np.zeros((d, r)); O[:, 0] = [1, 0, 0, 0]; O[:, 1] = [0, 1, 0, 0]
T = np.zeros((r, d)); T[0, 1] = 1.0; T[1, 2] = 1.0  # TO = [[0,1],[0,0]]
print("   TO =", (T @ O).tolist())
report("eq:sgeom-lowrank with nilpotent singular TO", np.abs(expm(O @ T) - (np.eye(d) + O @ phi1(T @ O) @ T)).max())

# stacked interfaces: e^{sum O_h T_h} = I + O phi1(TO) T, and block-diagonal replacement fails
d = 4
O1, O2 = rng.normal(size=(d, 1)), rng.normal(size=(d, 2))
T1, T2 = rng.normal(size=(1, d)), rng.normal(size=(2, d))
Ost, Tst = np.hstack([O1, O2]), np.vstack([T1, T2])
full = expm(O1 @ T1 + O2 @ T2)
report("stacked interfaces identity", np.abs(full - (np.eye(d) + Ost @ phi1(Tst @ Ost) @ Tst)).max())
blockdiag = np.eye(d) + O1 @ phi1(T1 @ O1) @ T1 + O2 @ phi1(T2 @ O2) @ T2
print(f"   diagonal-block replacement error with generic cross terms = {np.abs(full - blockdiag).max():.3e} (should be O(1))")
# with T_h O_k = 0 for h != k the diagonal replacement holds
O1 = np.array([[1.0], [0.0], [0.0], [0.0]]); T1 = np.array([[0.7, 0.0, 0.0, 0.0]])
O2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]); T2 = np.array([[0.0, 0.3, -0.2, 0.0], [0.0, 0.1, 0.4, 0.9]])
full = expm(O1 @ T1 + O2 @ T2)
blockdiag = np.eye(d) + O1 @ phi1(T1 @ O1) @ T1 + O2 @ phi1(T2 @ O2) @ T2
report("diagonal-block replacement when T1 O2 = T2 O1 = 0", np.abs(full - blockdiag).max())

# prop:sgeom-rankone
for t_target in [0.8, 0.0, -1.3]:
    u = rng.normal(size=3)
    v = rng.normal(size=3)
    if t_target == 0.0:
        v = v - (v @ u) / (u @ u) * u
    L = np.outer(u, v)
    t = v @ u
    ph = 1.0 if abs(t) < 1e-14 else (np.exp(t) - 1) / t
    report(f"prop:sgeom-rankone closed form t={t:+.3f}", np.abs(expm(L) - (np.eye(3) + ph * L)).max())
    report(f"prop:sgeom-rankone det e^L = e^t t={t:+.3f}", abs(np.linalg.det(expm(L)) - np.exp(t)))
    if t_target == 0.0:
        report("prop:sgeom-rankone L^2 = 0 when t = 0", np.abs(L @ L).max())

# lem:sgeom-frechet: integral formula vs scipy expm_frechet vs central finite differences
L = rng.normal(size=(4, 4))
H = rng.normal(size=(4, 4))
xs, ws = np.polynomial.legendre.leggauss(40)
ts, ws = 0.5 * (xs + 1), 0.5 * ws
integral = sum(w * expm((1 - t) * L) @ H @ expm(t * L) for t, w in zip(ts, ws))
eps = 1e-6
fd = (expm(L + eps * H) - expm(L - eps * H)) / (2 * eps)
report("lem:sgeom-frechet integral vs scipy expm_frechet", np.abs(integral - expm_frechet(L, H, compute_expm=False)).max())
report("lem:sgeom-frechet integral vs central difference", np.abs(integral - fd).max())
Hc = L @ L + 0.5 * L  # commutes with L
report("lem:sgeom-frechet commuting case equals e^L H", np.abs(expm_frechet(L, Hc, compute_expm=False) - expm(L) @ Hc).max())

# eq:sgeom-moment-derivative (symbolic, d = 2)
lam = sp.Symbol("lambda")
mi = sp.Matrix(sp.symbols("a1 a2"))
mj = sp.Matrix(sp.symbols("b1 b2"))
m11, m12, m22 = sp.symbols("m11 m12 m22")
Mj = sp.Matrix([[m11, m12], [m12, m22]])
s = sp.Symbol("s")
dmi = sp.Matrix(sp.symbols("da1 da2"))
dmj = sp.Matrix(sp.symbols("db1 db2"))
dm11, dm12, dm22 = sp.symbols("dm11 dm12 dm22")
dMj = sp.Matrix([[dm11, dm12], [dm12, dm22]])
Lfun = lambda a, b, M: lam * a * b.T * M.inv()
Ls = Lfun(mi + s * dmi, mj + s * dmj, Mj + s * dMj)
dL_exact = Ls.diff(s).subs(s, 0)
Minv = Mj.inv()
dL_formula = lam * (dmi * mj.T * Minv + mi * dmj.T * Minv - mi * mj.T * Minv * dMj * Minv)
print("eq:sgeom-moment-derivative symbolic residual:", sp.simplify(dL_exact - dL_formula))

# prop:int-rank
d, r = 5, 2
O = rng.normal(size=(d, r))
T = rng.normal(size=(r, d))
E = expm(O @ T)
kerT = np.linalg.svd(T)[2][r:].T  # basis of ker T, d x (d-r)
report("prop:int-rank e^{OT} v = v on ker T", np.abs(E @ kerT - kerT).max())
rank_O = np.linalg.matrix_rank(O)
rank_aug = np.linalg.matrix_rank(np.hstack([O, E - np.eye(d)]), tol=1e-9)
print(f"prop:int-rank im(e^OT - I) in im O: rank O = {rank_O}, rank [O, e^OT - I] = {rank_aug}")
print(f"prop:int-rank rank OT = {np.linalg.matrix_rank(O @ T)}, det e^OT = {np.linalg.det(E):.4f} = e^tr = {np.exp(np.trace(O @ T)):.4f}")

# prop:int-moment-equivariance and prop:int-endpoint with the moment family, d = 3
d = 3


def rand_spd(k: int) -> np.ndarray:
    A = rng.normal(size=(k, k))
    return A @ A.T + k * np.eye(k)


lam_e = 0.7
mu_i, mu_j = rng.normal(size=d), rng.normal(size=d)
M_i, M_j = rand_spd(d), rand_spd(d)
Lij = lam_e * np.outer(mu_i, mu_j) @ np.linalg.inv(M_j)
h = rng.normal(size=(d, d)) + 2 * np.eye(d)
hi = np.linalg.inv(h)
mu_i2, mu_j2 = hi @ mu_i, hi @ mu_j
M_j2 = hi @ M_j @ hi.T
Lij2 = lam_e * np.outer(mu_i2, mu_j2) @ np.linalg.inv(M_j2)
report("prop:int-moment-equivariance L' = h^-1 L h", np.abs(Lij2 - hi @ Lij @ h).max())
U_i, U_j = rng.normal(size=(d, d)) + 2 * np.eye(d), rng.normal(size=(d, d)) + 2 * np.eye(d)
Theta = U_i @ expm(Lij) @ np.linalg.inv(U_j)
Theta_ref = (U_i @ h) @ expm(Lij2) @ np.linalg.inv(U_j @ h)
report("eq:int-reference-equivariance => link independent of common h", np.abs(Theta - Theta_ref).max())
g_i, g_j = rng.normal(size=(d, d)) + 2 * np.eye(d), rng.normal(size=(d, d)) + 2 * np.eye(d)
Theta_p = (g_i @ U_i) @ expm(Lij) @ np.linalg.inv(g_j @ U_j)
report("eq:int-endpoint Theta' = g_i Theta g_j^-1", np.abs(Theta_p - g_i @ Theta @ np.linalg.inv(g_j)).max())

# Frame redundancy: D_ij depends only on reference laws (Gaussian closed form)
# reference laws N(mu_i, M_i), N(mu_j, M_j); agent-coordinate laws pushed by U_i, U_j.
eL = expm(Lij)
D_ref = gauss_kl(mu_i, M_i, eL @ mu_j, eL @ M_j @ eL.T)
nu_i_m, nu_i_S = U_i @ mu_i, U_i @ M_i @ U_i.T
nu_j_m, nu_j_S = U_j @ mu_j, U_j @ M_j @ U_j.T
D_agent = gauss_kl(nu_i_m, nu_i_S, Theta @ nu_j_m, Theta @ nu_j_S @ Theta.T)
report("frame redundancy KL(nu_i||Theta# nu_j) = KL(ref_i||e^L# ref_j)", abs(D_ref - D_agent))

# eq:int-nongaussian-loop
E12 = np.array([[0.0, 1.0], [0.0, 0.0]])
E21 = E12.T
loop = expm(E12) @ expm(E21)
report("eq:int-nongaussian-loop (I+E12)(I+E21) = [[2,1],[1,1]]", np.abs(loop - np.array([[2.0, 1.0], [1.0, 1.0]])).max())
# Laplace law with variance one: scale b = 1/sqrt(2); check moment-family generators by exact moments
b = 1 / np.sqrt(2)
print(f"   Laplace(0,b) variance 2b^2 = {2 * b * b:.6f}")
mu1, mu2, Mlap = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.eye(2)
report("Laplace example L12 = E12", np.abs(np.outer(mu1, mu2) @ np.linalg.inv(Mlap) - E12).max())
report("Laplace example L21 = E21", np.abs(np.outer(mu2, mu1) @ np.linalg.inv(Mlap) - E21).max())
print(f"   loop trace {np.trace(loop):.3f}, det {np.linalg.det(loop):.3f}, eigenvalues {np.sort(eigvals(loop).real)}")
report("Theta_21 != Theta_12^-1 (norm of difference, should be O(1))", -np.abs(expm(E21) - np.linalg.inv(expm(E12))).max())

# Extension: reciprocal / backtrack decomposition Omega_ij = Theta_ij (Theta_ji Theta_ij)^{-1/2}
def omega(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""Omega_ij = Theta_ij (Theta_ji Theta_ij)^{-1/2}, A = Theta_ij, B = Theta_ji (principal root)."""
    return A @ np.linalg.inv(sqrtm(B @ A))


worst_sym, worst_cov = 0.0, 0.0
for _ in range(50):
    d = 3
    A = expm(0.4 * rng.normal(size=(d, d)))
    B = expm(0.4 * rng.normal(size=(d, d)))
    Om_ij, Om_ji = omega(A, B), omega(B, A)
    worst_sym = max(worst_sym, np.abs(Om_ij @ Om_ji - np.eye(d)).max())
    gi = expm(0.5 * rng.normal(size=(d, d)))
    gj = expm(0.5 * rng.normal(size=(d, d)))
    A2, B2 = gi @ A @ np.linalg.inv(gj), gj @ B @ np.linalg.inv(gi)
    worst_cov = max(worst_cov, np.abs(omega(A2, B2) - gi @ Om_ij @ np.linalg.inv(gj)).max())
report("reciprocal part: Omega_ij Omega_ji = I (50 random pairs, worst)", worst_sym)
report("reciprocal part: Omega' = g_i Omega g_j^-1 (50 random, worst)", worst_cov)
A, B = expm(E12), expm(E21)
Om = omega(A, B)
print("   Laplace example: backtrack loop at agent 2 Theta_21 Theta_12 =", (B @ A).tolist())
print("   Laplace example: reciprocal part Omega_12 =", np.round(Om.real, 6).tolist())
report("   Laplace Omega_12 Omega_21 = I", np.abs(Om @ omega(B, A) - np.eye(2)).max())
report("   Laplace Theta_12 = Omega_12 (Theta_21 Theta_12)^{1/2}", np.abs(A - Om @ sqrtm(B @ A)).max())

# Non-exponential GL^+ element (Culver): [[-1,1],[0,-1]] has no real logarithm
N = np.array([[-1.0, 1.0], [0.0, -1.0]])
lg = logm(N)
print(f"   logm([[-1,1],[0,-1]]) max |imag| = {np.abs(np.imag(lg)).max():.3f}; det = {np.linalg.det(N):.1f} > 0")
