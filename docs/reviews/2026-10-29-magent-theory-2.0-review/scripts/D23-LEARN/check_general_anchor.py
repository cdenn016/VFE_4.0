r"""D23-LEARN check 7: general positive-definite anchor, lam -> infinity limit of z = Z^T(w_inf - w*).
Claimed closed form (report D23-LEARN-04):
  z_inf = z0 - K_L (K_L^T Pt K_L)^{-1} K_L^T (Z^T Pi0 w* + Pt z0),
  z0 = -L_N^+ Z^T B^T W B w*,  Pt = Z^T Pi0 Z,  K_L orthonormal basis of ker L_N.
Instance: three-agent complete exchange, K = 2, codes chosen with a shared unread direction on
pair {0,1} so that ker L_N is nontrivial.
"""
import numpy as np
from d23_common import blockdiag, invisible_basis, loop_matrix, null_space, spd

rng = np.random.default_rng(13)
K = 2
edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]
idx = {e: n for n, e in enumerate(edges)}
loops = [[idx[(1, 0)], idx[(0, 1)]], [idx[(2, 1)], idx[(1, 2)]], [idx[(0, 2)], idx[(2, 0)]],
         [idx[(0, 1)], idx[(1, 2)], idx[(2, 0)]], [idx[(0, 2)], idx[(2, 1)], idx[(1, 0)]]]
k = np.array([0.0, 1.0])
codes = [np.array([[1.0, 0.0]]) for _ in edges]       # every code kills e2: nontrivial ker L_N
Z, Y = invisible_basis(codes, K)
B = loop_matrix(loops, 6, K)
W = blockdiag([spd(4, rng) for _ in loops])
Pi0 = spd(24, rng, cond=15.0)
w = rng.standard_normal(24)
LN = Z.T @ B.T @ W @ B @ Z
KL = null_space(LN)
Pt = Z.T @ Pi0 @ Z
z0 = -np.linalg.pinv(LN) @ (Z.T @ B.T @ W @ B @ w)
zinf = z0 - KL @ np.linalg.solve(KL.T @ Pt @ KL, KL.T @ (Z.T @ Pi0 @ w + Pt @ z0))
print("dim N =", Z.shape[1], " dim ker L_N =", KL.shape[1])
for lam in [1e2, 1e4, 1e6]:
    Lam0 = Pi0 + lam * B.T @ W @ B
    S = Z.T @ Lam0 @ Z
    z = -np.linalg.solve(S, Z.T @ Lam0 @ w)            # Z^T(w_inf - w*)
    print(f"lam={lam:.0e}: |z - z_inf| = {np.abs(z - zinf).max():.3e}")
iso = -KL @ KL.T @ (Z.T @ w) - np.linalg.pinv(LN) @ (Z.T @ B.T @ W @ B @ w)
z0i = z0; zi = z0i - KL @ np.linalg.solve(KL.T @ KL, KL.T @ (Z.T @ w + z0i))
print("isotropic reduction to it1:bias: |formula(Pi0 = I) - (-P_ker z* - L_N^+ Z^T B^T W B w*)| =",
      f"{np.abs(zi - iso).max():.2e}")
