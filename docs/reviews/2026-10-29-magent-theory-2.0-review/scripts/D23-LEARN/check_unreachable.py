r"""D23-LEARN check 2: it1:unreachable, it1:complete-exchange, the K = 3 example after it, it1:pair.

U1  ker B = coboundaries {(a_i - a_j)_e} when the loop chains span the cycle space; failure
    of that equality when they do not (complete exchange with two-cycles only).
U2  dim(N cap ker B) = K (dim H^0 - K) on random instances (several graphs, ranks, K).
U3  The K = 3 example: three agents, pairwise kernel intersections three independent lines.
    With two-cycles and triangles: N cap ker B = 0.  With two-cycles only: count per pair.
    Variant with three coplanar (dependent) lines: N cap ker B has dimension K.
U4  Detection exponent: D(w*) = 0 iff min_{w in ker B} (w - w*)^T F (w - w*) = 0.
P1  it1:pair split (filled, unreachable, detectable) and D formula; doubling when both
    basings of the two-cycle are listed in the loop family.
"""
from __future__ import annotations

import numpy as np

from d23_common import (W_projection_complement, blockdiag, coboundary_gl, detectable_lstsq,
                        dim_H0, fisher_edge, invisible_basis, loop_matrix, null_space, orth, rank,
                        spd)

rng = np.random.default_rng(7)


def subspace_equal(A: np.ndarray, Bm: np.ndarray) -> bool:
    return rank(A) == rank(Bm) == rank(np.hstack([A, Bm]))


def complete3():
    edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]
    idx = {e: n for n, e in enumerate(edges)}
    two = [[idx[(1, 0)], idx[(0, 1)]], [idx[(2, 1)], idx[(1, 2)]], [idx[(0, 2)], idx[(2, 0)]]]
    tri = [[idx[(0, 1)], idx[(1, 2)], idx[(2, 0)]], [idx[(0, 2)], idx[(2, 1)], idx[(1, 0)]]]
    return edges, idx, two, tri


def unreach_dim(edges, loops, codes, K):
    Z, Y = invisible_basis(codes, K)
    B = loop_matrix(loops, len(edges), K)
    return null_space(np.vstack([Y.T, B])).shape[1], Z.shape[1]


print("=== U1: ker B versus coboundaries ===")
K = 2
edges, idx, two, tri = complete3()
B_full = loop_matrix(two + tri, len(edges), K)
B_two = loop_matrix(two, len(edges), K)
Dcob = coboundary_gl(edges, 3, K)
print("complete exchange, two-cycles + triangles: dim ker B =", null_space(B_full).shape[1],
      " dim im(delta) =", rank(Dcob), " equal subspaces:", subspace_equal(null_space(B_full), orth(Dcob)))
print("complete exchange, two-cycles only        : dim ker B =", null_space(B_two).shape[1],
      " dim im(delta) =", rank(Dcob), " (cycle space has dim 6-3+1 = 4, two-cycles span 3)")
# directed 4-cycle with a chord whose only directed cycles are the 4-cycle and one triangle
edges4 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]           # 0->1->2->3->0 and chord 0->2
loops4 = [[0, 1, 2, 3], [4, 2, 3]]                          # 0->1->2->3->0, 0->2->3->0
B4 = loop_matrix(loops4, 5, K)
print("4-cycle with chord (strongly connected): dim ker B =", null_space(B4).shape[1],
      " dim im(delta) =", rank(coboundary_gl(edges4, 4, K)),
      " equal:", subspace_equal(null_space(B4), orth(coboundary_gl(edges4, 4, K))))

print("\n=== U2: dim(N cap ker B) = K (dim H^0 - K) on random instances ===")
worst = 0
for trial in range(30):
    K = int(rng.integers(2, 4))
    choice = trial % 3
    if choice == 0:
        edges, idx, two, tri = complete3(); loops = two + tri; nV = 3
    elif choice == 1:
        edges, loops, nV = edges4, loops4, 4
    else:
        edges = [(0, 1), (1, 2), (2, 0)]; loops = [[0, 1, 2]]; nV = 3
    codes = [rng.standard_normal((int(rng.integers(1, K + 1)), K)) for _ in edges]
    # occasionally force a common unread direction
    if trial % 5 == 0:
        k = rng.standard_normal(K)
        P = np.eye(K) - np.outer(k, k) / (k @ k)
        codes = [T @ P if T.shape[0] < K else (T @ P)[: K - 1] for T in codes]
    u, dN = unreach_dim(edges, loops, codes, K)
    pred = K * (dim_H0(edges, codes, nV, K) - K)
    worst = max(worst, abs(u - pred))
print("max |dim(N cap ker B) - K(dim H0 - K)| over 30 random instances =", worst)

print("\n=== U3: the K = 3 example after it1:complete-exchange ===")
K = 3
edges, idx, two, tri = complete3()
e1, e2, e3 = np.eye(3)
# pair {0,1}: codes rows e1 (on 1->0) and e2 (on 0->1) -> kernel intersection span(e3)
# pair {1,2}: rows e2, e3 -> span(e1);  pair {0,2}: rows e1, e3 -> span(e2)
code_of = {(1, 0): e1, (0, 1): e2, (2, 1): e2, (1, 2): e3, (0, 2): e1, (2, 0): e3}
codes = [code_of[e][None, :] for e in edges]
for (a, b) in [(0, 1), (1, 2), (0, 2)]:
    kin = null_space(np.vstack([code_of[(a, b)], code_of[(b, a)]]))
    print(f"pair {{{a},{b}}}: dim(ker cap ker) = {kin.shape[1]}, line = {np.round(kin[:, 0], 3)}")
u_full, dN = unreach_dim(edges, two + tri, codes, K)
u_two, _ = unreach_dim(edges, two, codes, K)
print("dim N =", dN, "; two-cycles + triangles: dim(N cap ker B) =", u_full,
      " K(dim H0 - K) =", K * (dim_H0(edges, codes, 3, K) - K))
print("two-cycles only: dim(N cap ker B) =", u_two, " (sum over pairs of K*1 = 9)")
# dependent variant: three distinct coplanar lines (all in the plane x3 = 0)
lines = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 0])]
def two_rows_with_kernel_line(l):
    # two rank-one codes whose kernels meet exactly in span(l)
    comp = null_space(l[None, :])          # 3 x 2 basis of l^perp
    return comp[:, 0][None, :], comp[:, 1][None, :]
code_dep = {}
for (a, b), l in zip([(0, 1), (1, 2), (0, 2)], lines):
    r1, r2 = two_rows_with_kernel_line(l)
    code_dep[(b, a)] = r1; code_dep[(a, b)] = r2
codes_dep = [code_dep[e] for e in edges]
u_dep, _ = unreach_dim(edges, two + tri, codes_dep, K)
print("coplanar (dependent) lines with two-cycles + triangles: dim(N cap ker B) =", u_dep,
      " K(dim H0 - K) =", K * (dim_H0(edges, codes_dep, 3, K) - K))

print("\n=== U4: detectability equals rejectability of the flat hypothesis ===")
K = 2
edges, idx, two, tri = complete3()
loops = two + tri
codes = [rng.standard_normal((1, K)) for _ in edges]
codes[0] = codes[1] = np.array([[1.0, 0.0]])            # pair {0,1} shares an unread direction
Z, Y = invisible_basis(codes, K)
B = loop_matrix(loops, len(edges), K)
W = blockdiag([spd(K * K, rng) for _ in loops])
F = blockdiag([fisher_edge(T, np.eye(1), np.eye(K)) for T in codes])
Kb = null_space(B)
Q = W_projection_complement(B, W, Z)
def flat_excess(w):
    # min over flat w' of (w' - w)^T F (w' - w)
    A = np.linalg.cholesky(F + 1e-14 * np.eye(F.shape[0])).T @ Kb
    b = np.linalg.cholesky(F + 1e-14 * np.eye(F.shape[0])).T @ w
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = A @ c - b
    return float(r @ r)
w_gen = rng.standard_normal(B.shape[1])
w_BN = Kb @ rng.standard_normal(Kb.shape[1]) + Z @ rng.standard_normal(Z.shape[1])  # B w in B(N)
for lab, w in [("generic truth", w_gen), ("truth with B w* in B(N)", w_BN)]:
    print(f"{lab}: D(w*) = {detectable_lstsq(w, B, W, Z):.3e}, flat-fit Fisher excess = {flat_excess(w):.3e}")

print("\n=== P1: reciprocal pair split and detectable defect ===")
K = 3
for (r0, r1) in [(2, 2), (1, 1), (1, 2), (2, 1)]:
    T1 = rng.standard_normal((r1, K)); T0 = rng.standard_normal((r0, K))
    codes = [T1, T0]                      # edge 0: 1 -> 0 carries T1; edge 1: 0 -> 1 carries T0
    edges = [(1, 0), (0, 1)]
    Z, Y = invisible_basis(codes, K)
    B = loop_matrix([[0, 1]], 2, K)
    W = np.eye(K * K)
    filled = rank(B @ Z)
    unreach = null_space(np.vstack([Y.T, B])).shape[1]
    detect = rank(B) - filled
    K0 = null_space(T0); K1 = null_space(T1)
    sumK = rank(np.hstack([K0, K1]))
    capK = K0.shape[1] + K1.shape[1] - sumK
    capR = K - sumK
    w = rng.standard_normal(2 * K * K)
    Rcap = null_space(np.hstack([K0, K1]).T)      # (K0 + K1)^perp = R0 cap R1
    Pcap = Rcap @ Rcap.T
    Dformula = float(np.sum((Pcap @ (w[:9].reshape(3, 3) + w[9:].reshape(3, 3))) ** 2))
    Dls = detectable_lstsq(w, B, W, Z)
    B2 = loop_matrix([[0, 1], [1, 0]], 2, K)     # both basings of the two-cycle
    D2 = detectable_lstsq(w, B2, np.eye(2 * K * K), Z)
    print(f"ranks (r0,r1)=({r0},{r1}): (filled,unreach,detect)=({filled},{unreach},{detect}) "
          f"predicted ({K * sumK},{K * capK},{K * capR}); D lstsq={Dls:.6f} formula={Dformula:.6f}; "
          f"both basings listed: D={D2:.6f} (ratio {D2 / Dls if Dls > 1e-12 else float('nan'):.3f})")
