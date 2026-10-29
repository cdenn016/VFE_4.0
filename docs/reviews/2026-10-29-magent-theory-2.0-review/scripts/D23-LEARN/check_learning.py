r"""D23-LEARN check 1: tangent learning model, it1:tangent-model, it1:learning, it1:complementary.

Independent reconstruction (does not import the manuscript's item1_checks).  Checks:
  P1  Lambda_0 = Pi_0 + lam B^T W B is positive definite; log normalizer depends on lam, W.
  L2  limit mean (I - Pi_N) w*, covariance Z S^{-1} Z^T, n * Y-block -> (Y^T F_kappa Y)^{-1}
      via the exact Gaussian posterior with expected statistics (deterministic) and
      equality with covariance-form Gaussian conditioning of the prior on Y^T w = Y^T w*.
  L2s stochastic streaming posterior with routed observed labels (illustration only).
  L3  isotropic anchor: S^{-1} = s2 (I + lam s2 L_N)^{-1}; ker/rank of L_N; the clause
      "strictly smaller on the orthogonal complement" against (a) s2 and (b) the prior
      marginal covariance Z^T Lambda_0^{-1} Z; the lam = 0 case.
  L4  flat truth shrinkage formula.
  L5  bias limit as lam -> infinity and its O(1/lam) rate.
  C1  D(w) by least squares equals ||Q B w||_W^2 and is constant on cosets of N.
  C2  B w_inf -> Q B w* as lam -> infinity, isotropic anchor and general PD anchor.
  C3  dimension split of B(X).
  OL  order of limits n -> inf, lam -> inf for a dissonant truth (constraint form vs penalty).
Outputs residuals; agreement supports the identities at these instances only.
"""
from __future__ import annotations

import numpy as np

from d23_common import (W_projection_complement, blockdiag, detectable_lstsq, fisher_edge,
                        gaussian_condition, invisible_basis, loop_matrix, null_space, orth,
                        rank, spd)

rng = np.random.default_rng(20261029)
np.set_printoptions(precision=4, suppress=True)


def full_rank_code(r: int, K: int) -> np.ndarray:
    while True:
        T = rng.standard_normal((r, K))
        if rank(T) == r:
            return T


def setup_complete3(K: int, ranks: list[int]):
    # complete exchange of agents 0,1,2: edges (j, i) = j -> i
    edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]
    idx = {e: n for n, e in enumerate(edges)}
    # loops as lists of edge indices: three two-cycles and both directed triangles
    loops = [[idx[(1, 0)], idx[(0, 1)]], [idx[(2, 1)], idx[(1, 2)]], [idx[(0, 2)], idx[(2, 0)]],
             [idx[(0, 1)], idx[(1, 2)], idx[(2, 0)]],   # 0 -> 1 -> 2 -> 0
             [idx[(0, 2)], idx[(2, 1)], idx[(1, 0)]]]   # 0 -> 2 -> 1 -> 0
    codes = [full_rank_code(r, K) for r in ranks]
    return edges, loops, codes


def model_mats(K, edges, loops, codes, lam, Pi0, Wforms, Rs, Ms, kappas):
    m = K * K
    nE = len(edges)
    Z, Y = invisible_basis(codes, K)
    B = loop_matrix(loops, nE, K)
    W = blockdiag(Wforms)
    Lam0 = Pi0 + lam * B.T @ W @ B
    Fk = blockdiag([kappas[e] * fisher_edge(codes[e], Rs[e], Ms[e]) for e in range(nE)])
    return Z, Y, B, W, Lam0, Fk


def limit_mean(Z, Lam0, w):
    S = Z.T @ Lam0 @ Z
    return w - Z @ np.linalg.solve(S, Z.T @ Lam0 @ w), S


print("=== P1: normalizability of the anchored loop prior ===")
K = 2
edges, loops, codes = setup_complete3(K, [1, 1, 1, 1, 1, 1])
nE, m = len(edges), K * K
d = nE * m
B = loop_matrix(loops, nE, K)
Wforms = [spd(m, rng) for _ in loops]
W = blockdiag(Wforms)
Pi0 = spd(d, rng)
for lam in [0.0, 1.0, 1e3]:
    Lam0 = Pi0 + lam * B.T @ W @ B
    ev = np.linalg.eigvalsh(Lam0)
    sign, logdet = np.linalg.slogdet(Lam0)
    print(f"lam={lam:8.1f}: min eig Lambda0 = {ev.min():.4f} (>0), "
          f"log normalizer = d/2 log 2pi - 1/2 logdet = {0.5 * d * np.log(2 * np.pi) - 0.5 * logdet:.4f}")
print("rank B^T W B =", rank(B.T @ W @ B), "of", d, "-> loop term alone is improper (rank-deficient);"
      " the anchor makes the prior proper.")

print("\n=== L2: limit of the recognition law (deterministic expected statistics) ===")
K = 3
edges, loops, codes = setup_complete3(K, [1, 2, 1, 2, 2, 1])
nE, m = len(edges), K * K
d = nE * m
Rs = [spd(T.shape[0], rng) for T in codes]
Ms = [spd(K, rng) for _ in codes]
kappas = rng.uniform(0.5, 2.0, nE)
Wforms = [spd(m, rng) for _ in loops]
Pi0 = spd(d, rng)          # general (non-isotropic) anchor
lam = 0.7
Z, Y, B, W, Lam0, Fk = model_mats(K, edges, loops, codes, lam, Pi0, Wforms, Rs, Ms, kappas)
w_star = rng.standard_normal(d)
w_inf, S = limit_mean(Z, Lam0, w_star)
print("dim N =", Z.shape[1], " dim X =", d, " ker(F_kappa) dim =", d - rank(Fk),
      " |F_kappa Z| =", np.abs(Fk @ Z).max())
for n in [1e4, 1e6, 1e8]:
    Lam_n = Lam0 + n * Fk
    mean_n = np.linalg.solve(Lam_n, n * Fk @ w_star)
    cov_n = np.linalg.inv(Lam_n)
    nYcov = n * Y.T @ cov_n @ Y
    target = np.linalg.inv(Y.T @ Fk @ Y)
    print(f"n={n:.0e}: |mean-w_inf|={np.abs(mean_n - w_inf).max():.2e}  "
          f"|cov - Z S^-1 Z^T|={np.abs(cov_n - Z @ np.linalg.solve(S, Z.T)).max():.2e}  "
          f"rel|n Ycov Y - (Y^T F Y)^-1|={np.abs(nYcov - target).max() / np.abs(target).max():.2e}")
cm, cc = gaussian_condition(np.linalg.inv(Lam0), Y, Y.T @ w_star)
print("conditioned prior on {Y^T w = Y^T w*}: |mean - w_inf| =", f"{np.abs(cm - w_inf).max():.2e}",
      " |cov - Z S^-1 Z^T| =", f"{np.abs(cc - Z @ np.linalg.solve(S, Z.T)).max():.2e}")
print("w_inf - w* in N:  |Y^T (w_inf - w*)| =", f"{np.abs(Y.T @ (w_inf - w_star)).max():.2e}")

print("\n=== L2s: streaming posterior with routed observed labels (illustration) ===")
# edge 0 has two modalities with routing weights; others one modality with beta = 1.
K = 2
edges, loops, codes1 = setup_complete3(K, [1, 1, 1, 1, 1, 1])
nE, m = len(edges), K * K
d = nE * m
mods = []
for e in range(nE):
    if e == 0:
        mods.append([(codes1[0], spd(1, rng), 0.6), (full_rank_code(1, K), spd(1, rng), 0.3)])
    else:
        mods.append([(codes1[e], spd(1, rng), 1.0)])
stacked = [np.vstack([T for (T, R, b) in mods[e] if b > 0]) for e in range(nE)]
Ms = [spd(K, rng) for _ in range(nE)]
Lchol = [np.linalg.cholesky(M) for M in Ms]
kap = np.array([1.0, 0.5, 1.5, 1.0, 2.0, 0.8])
Wforms = [np.eye(m) for _ in loops]
s2 = 0.8
Pi0 = np.eye(d) / s2
lam = 2.0
Z, Y = invisible_basis(stacked, K)
B = loop_matrix(loops, nE, K)
W = blockdiag(Wforms)
Lam0 = Pi0 + lam * B.T @ W @ B
Fbar = [sum(b * fisher_edge(T, R, Ms[e]) for (T, R, b) in mods[e]) for e in range(nE)]
Fk = blockdiag([kap[e] * Fbar[e] for e in range(nE)])
w_star = rng.standard_normal(d)
w_inf, S = limit_mean(Z, Lam0, w_star)
for n in [2000, 200000]:
    Lam = Lam0.copy()
    eta = np.zeros(d)
    for e in range(nE):
        ne = int(np.ceil(kap[e] * n))
        ybar = rng.standard_normal((ne, K)) @ Lchol[e].T          # referents ~ N(0, M_e)
        we = w_star[e * m:(e + 1) * m].reshape(K, K)
        for (T, R, b) in mods[e]:
            sel = rng.random(ne) < b                               # observed label J_c = j
            yb = ybar[sel]
            eps = rng.standard_normal((yb.shape[0], T.shape[0])) @ np.linalg.cholesky(R).T
            h = (T @ (np.eye(K) - we) @ yb.T).T + eps              # tangent message model
            v = h - (T @ yb.T).T                                   # h_t - T ybar_t
            Ri = np.linalg.inv(R)
            # sum_t J_t^T R^-1 J_t = (T^T R^-1 T) kron (sum ybar ybar^T); J_t w = -T w_e ybar_t
            Lam[e * m:(e + 1) * m, e * m:(e + 1) * m] += np.kron(T.T @ Ri @ T, yb.T @ yb)
            eta[e * m:(e + 1) * m] += -(T.T @ Ri @ v.T @ yb).reshape(-1)
    mean = np.linalg.solve(Lam, eta)
    cov = np.linalg.inv(Lam)
    print(f"n={n}: |mean - w_inf| = {np.abs(mean - w_inf).max():.3e}   "
          f"|cov - Z S^-1 Z^T| = {np.abs(cov - Z @ np.linalg.solve(S, Z.T)).max():.3e}   "
          f"rel|n Ycov Y - (Y^T F Y)^-1| = "
          f"{np.abs(n * Y.T @ cov @ Y - np.linalg.inv(Y.T @ Fk @ Y)).max() / np.abs(np.linalg.inv(Y.T @ Fk @ Y)).max():.3e}")
print("kernel of routed F_kappa equals N (stacked attending codes):", d - rank(Fk) == Z.shape[1],
      " |F_kappa Z| =", f"{np.abs(Fk @ Z).max():.1e}")

print("\n=== L3: isotropic anchor covariance on the invisible coordinates ===")
K = 3
edges, loops, codes = setup_complete3(K, [1, 2, 1, 2, 2, 1])
nE, m = len(edges), K * K
d = nE * m
Wforms = [spd(m, rng) for _ in loops]
s2 = 1.3
for lam in [0.0, 0.9]:
    Z, Y = invisible_basis(codes, K)
    B = loop_matrix(loops, nE, K)
    W = blockdiag(Wforms)
    Lam0 = np.eye(d) / s2 + lam * B.T @ W @ B
    S = Z.T @ Lam0 @ Z
    LN = Z.T @ B.T @ W @ B @ Z
    formula = s2 * np.linalg.inv(np.eye(LN.shape[0]) + lam * s2 * LN)
    kerL = null_space(LN)
    NkerB = null_space(np.vstack([Y.T, B]))                       # N cap ker B in X
    print(f"lam={lam}: |S^-1 - s2(I+lam s2 L_N)^-1| = {np.abs(np.linalg.inv(S) - formula).max():.2e}; "
          f"dim ker L_N = {kerL.shape[1]}, dim(N cap ker B) = {NkerB.shape[1]}, "
          f"rank L_N = {rank(LN)}, dim N - dim(N cap ker B) = {Z.shape[1] - NkerB.shape[1]}")
    comp = orth(LN)                                               # orthogonal complement of ker L_N
    lim_cov = np.linalg.inv(S)
    prior_marg = Z.T @ np.linalg.inv(Lam0) @ Z
    if comp.shape[1]:
        a = np.linalg.eigvalsh(comp.T @ lim_cov @ comp).max()
        b = np.linalg.eigvalsh(comp.T @ (prior_marg - lim_cov) @ comp).min()
        print(f"   on complement: max eig(limit cov) = {a:.4f} vs s2 = {s2}; "
              f"min eig(prior marginal - limit cov) = {b:.3e}")
    if kerL.shape[1]:
        print(f"   on ker L_N: |limit cov - s2 I| = {np.abs(kerL.T @ lim_cov @ kerL - s2 * np.eye(kerL.shape[1])).max():.2e}, "
              f"|prior marginal - s2 I| = {np.abs(kerL.T @ prior_marg @ kerL - s2 * np.eye(kerL.shape[1])).max():.2e}")

print("   counterexample to 'strictly smaller than the prior covariance': reciprocal pair, K=2, "
      "both codes (1,0)")
K = 2
edges = [(1, 0), (0, 1)]
loops = [[0, 1]]
codes = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]
m = K * K
d = 2 * m
Z, Y = invisible_basis(codes, K)
B = loop_matrix(loops, 2, K)
W = np.eye(m)
lam, s2 = 1.0, 1.0
Lam0 = np.eye(d) / s2 + lam * B.T @ W @ B
S = Z.T @ Lam0 @ Z
LN = Z.T @ B.T @ W @ B @ Z
comp = orth(LN)
lim_cov = np.linalg.inv(S)
prior_marg = Z.T @ np.linalg.inv(Lam0) @ Z
D = comp.T @ (prior_marg - lim_cov) @ comp
print(f"   dim N = {Z.shape[1]}, rank L_N = {comp.shape[1]}, eig(prior marginal - limit cov) on complement = "
      f"{np.linalg.eigvalsh(D)}; limit cov on complement eig = {np.linalg.eigvalsh(comp.T @ lim_cov @ comp)} (< s2 = 1)")

print("\n=== L4/L5: shrinkage for flat truth, bias limit for general truth (isotropic anchor) ===")
K = 3
edges, loops, codes = setup_complete3(K, [1, 1, 2, 1, 1, 2])
nE, m = len(edges), K * K
d = nE * m
Wforms = [spd(m, rng) for _ in loops]
s2 = 0.6
Z, Y = invisible_basis(codes, K)
B = loop_matrix(loops, nE, K)
W = blockdiag(Wforms)
LN = Z.T @ B.T @ W @ B @ Z
kerB = null_space(B)
w_flat = kerB @ rng.standard_normal(kerB.shape[1])
w_gen = rng.standard_normal(d)
P_ker = null_space(LN) @ null_space(LN).T
for lam in [0.5, 50.0]:
    Lam0 = np.eye(d) / s2 + lam * B.T @ W @ B
    w_inf, S = limit_mean(Z, Lam0, w_flat)
    pred = -np.linalg.solve(np.eye(LN.shape[0]) + lam * s2 * LN, Z.T @ w_flat)
    print(f"lam={lam}: |B w*| = {np.abs(B @ w_flat).max():.1e}; |Z^T(w_inf - w*) - shrinkage formula| = "
          f"{np.abs(Z.T @ (w_inf - w_flat) - pred).max():.2e}")
lim = -P_ker @ (Z.T @ w_gen) - np.linalg.pinv(LN) @ (Z.T @ B.T @ W @ B @ w_gen)
for lam in [1e2, 1e4, 1e6]:
    Lam0 = np.eye(d) / s2 + lam * B.T @ W @ B
    w_inf, S = limit_mean(Z, Lam0, w_gen)
    err = np.abs(Z.T @ (w_inf - w_gen) - lim).max()
    print(f"lam={lam:.0e}: |Z^T(w_inf - w*) - bias limit| = {err:.3e}   lam*err = {lam * err:.3f}")

print("\n=== C1-C3: detectable dissonance and the split of the loop space ===")
for (anchor, label) in [(None, "isotropic"), ("general", "general PD anchor")]:
    if anchor is None:
        Pi0 = np.eye(d) / s2
    else:
        Pi0 = spd(d, rng, cond=20.0)
    Q = W_projection_complement(B, W, Z)
    Dls = detectable_lstsq(w_gen, B, W, Z)
    Dq = float((Q @ B @ w_gen) @ W @ (Q @ B @ w_gen))
    nu = Z @ rng.standard_normal(Z.shape[1])
    print(f"[{label}] D lstsq = {Dls:.10f}, ||QBw||_W^2 = {Dq:.10f}, D(w + nu) = {detectable_lstsq(w_gen + nu, B, W, Z):.10f}")
    for lam in [1e2, 1e4, 1e6]:
        Lam0 = Pi0 + lam * B.T @ W @ B
        w_inf, S = limit_mean(Z, Lam0, w_gen)
        print(f"   lam={lam:.0e}: |B w_inf - Q B w*| = {np.abs(B @ w_inf - Q @ B @ w_gen).max():.3e}, "
              f"loop energy {float((B @ w_inf) @ W @ (B @ w_inf)):.6f} vs D(w*) = {Dq:.6f}")
dimBN = rank(B @ Z)
NkerB = null_space(np.vstack([Y.T, B])).shape[1]
print("dim B(N) =", dimBN, "= dim N - dim(N cap ker B) =", Z.shape[1] - NkerB,
      "; rank B - dim B(N) =", rank(B) - dimBN, "; dim of W-complement within B(X) =",
      rank(Q @ B))

print("\n=== OL: order of limits for a dissonant truth ===")
Pi0 = np.eye(d) / s2
Q = W_projection_complement(B, W, Z)
Fk = blockdiag([fisher_edge(codes[e], np.eye(codes[e].shape[0]), np.eye(K)) for e in range(nE)])
# (A) n -> inf first, then lam -> inf: the limit mean lies in w* + N, loop energy -> D(w*)
lamA = 1e7
w_A, _ = limit_mean(Z, Pi0 + lamA * B.T @ W @ B, w_gen)
# (B) lam = inf first (prior supported on ker B, anchor restricted), then n -> inf
Kb = null_space(B)
nB = 1e9
P = Kb.T @ Pi0 @ Kb + nB * Kb.T @ Fk @ Kb
w_B = Kb @ np.linalg.solve(P, nB * Kb.T @ Fk @ w_gen)
print(f"D(w*) = {float((Q @ B @ w_gen) @ W @ (Q @ B @ w_gen)):.5f}")
print(f"(A) n->inf then lam->inf: |Y^T(w_A - w*)| = {np.abs(Y.T @ (w_A - w_gen)).max():.2e}, "
      f"loop energy = {float((B @ w_A) @ W @ (B @ w_A)):.5f}")
print(f"(B) lam=inf then n->inf : |Y^T(w_B - w*)| = {np.abs(Y.T @ (w_B - w_gen)).max():.3f}, "
      f"loop energy = {float((B @ w_B) @ W @ (B @ w_B)):.2e}, "
      f"expected neg-loglik excess (w_B - w*)^T F (w_B - w*) = {float((w_B - w_gen) @ Fk @ (w_B - w_gen)):.4f}")
