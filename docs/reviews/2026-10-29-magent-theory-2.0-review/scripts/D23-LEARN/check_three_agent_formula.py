r"""D23-LEARN check 5.
(a) Three-agent complete exchange: closed formula
    dim(N cap ker B) = K [ dim I01 + dim I12 + dim I20 - dim(I01 + I12 + I20) ],
    I_ij = ker T_{j->i} cap ker T_{i->j}, with L = three two-cycles and both directed triangles,
    tested against direct computation on random code ranks (some with planted common kernels).
(b) The real span of loop chains (enough for it1:unreachable) does not give the nonlinear
    hypothesis of it1:exact-flat: directed triangle 0->1->2->0 with L = {gamma^2}; a reflection
    s in Stab(T) with s^2 = I on one edge reproduces every message law, has identity holonomy on L,
    and is not pure gauge (holonomy on gamma is s != I).
"""
import numpy as np
from d23_common import invisible_basis, loop_matrix, null_space, rank

rng = np.random.default_rng(3)
edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]
idx = {e: n for n, e in enumerate(edges)}
loops = [[idx[(1, 0)], idx[(0, 1)]], [idx[(2, 1)], idx[(1, 2)]], [idx[(0, 2)], idx[(2, 0)]],
         [idx[(0, 1)], idx[(1, 2)], idx[(2, 0)]], [idx[(0, 2)], idx[(2, 1)], idx[(1, 0)]]]
worst = 0
for trial in range(40):
    K = int(rng.integers(2, 5))
    codes = {}
    common = rng.standard_normal((K, int(rng.integers(0, 2))))
    for e in edges:
        r = int(rng.integers(1, K + 1))
        T = rng.standard_normal((r, K))
        if common.shape[1] and r < K and rng.random() < 0.6:
            P = np.eye(K) - common @ np.linalg.pinv(common)
            T = T @ P
        codes[e] = T
    Z, Y = invisible_basis([codes[e] for e in edges], K)
    B = loop_matrix(loops, 6, K)
    direct = null_space(np.vstack([Y.T, B])).shape[1]
    I = {}
    for (a, b) in [(0, 1), (1, 2), (2, 0)]:
        I[(a, b)] = null_space(np.vstack([codes[(a, b)], codes[(b, a)]]))
    dims = sum(v.shape[1] for v in I.values())
    span = rank(np.hstack(list(I.values()))) if dims else 0
    formula = K * (dims - span)
    worst = max(worst, abs(direct - formula))
print("(a) max |direct - K(sum dim I - dim sum I)| over 40 random three-agent instances =", worst)

K = 3
k = np.array([0.0, 0.0, 1.0])
T = [np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 1.0, 0.0]])]  # all kill k
a = np.array([0.3, -0.5, 0.0]); a[2] = -2.0 / k[2]          # a^T k = -2 -> s^2 = I
s = np.eye(K) + np.outer(k, a)
th = [s, np.eye(K), np.eye(K)]                               # edge 0->1 carries s; truth is identity
law = max(np.abs(T[e] @ np.linalg.inv(th[e]) - T[e]).max() for e in range(3))
H = th[2] @ th[1] @ th[0]                                    # gamma: 0->1->2->0
print(f"(b) det s = {np.linalg.det(s):.1f}; message-law residual vs truth = {law:.1e}; "
      f"|H_gamma - I| = {np.abs(H - np.eye(K)).max():.2f}; |H_gamma^2 - I| = {np.abs(H @ H - np.eye(K)).max():.1e}; "
      f"tangent: chains of gamma^2 span the cycle space over R (rank of B = {rank(loop_matrix([[0, 1, 2, 0, 1, 2]], 3, K))} = K^2)")
