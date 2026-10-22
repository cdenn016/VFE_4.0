r"""Checks for 02_interactions and 02_geometry additions.

(a) Reciprocal/backtrack decomposition Omega_ij = Theta_ij (Theta_ji Theta_ij)^{-1/2}.
(b) Law-dressed loop defect Q = (1/2) tr(Sigma (I-W)^T Sigma^{-1} (I-W)) invariance; value on the
    two-cycle [[2,1],[1,1]] and on the shear.
(c) Agreement singularity: SO(2) on N((eps, c), I): A^F = -eps/(c^2+eps^2), integral -pi.
(d) Contraction identity h^min - hbar^min = Delta(r,r) + kbar(shift) on the manuscript example and random cases.
(e) Labeled-weight Fisher term.
"""
import numpy as np
from scipy.linalg import sqrtm, expm

rng = np.random.default_rng(1)

# (a)
worst1 = worst2 = 0.0
for _ in range(50):
    T1 = expm(0.4 * rng.normal(size=(3, 3))); T2 = expm(0.4 * rng.normal(size=(3, 3)))
    N = T2 @ T1
    if np.any((np.abs(np.linalg.eigvals(N).imag) < 1e-12) & (np.linalg.eigvals(N).real <= 0)):
        continue
    O12 = T1 @ np.linalg.inv(sqrtm(T2 @ T1)); O21 = T2 @ np.linalg.inv(sqrtm(T1 @ T2))
    worst1 = max(worst1, np.abs(O21 @ O12 - np.eye(3)).max())
    gi, gj = rng.normal(size=(3, 3)) + 3 * np.eye(3), rng.normal(size=(3, 3)) + 3 * np.eye(3)
    T1p = gi @ T1 @ np.linalg.inv(gj); T2p = gj @ T2 @ np.linalg.inv(gi)
    O12p = T1p @ np.linalg.inv(sqrtm(T2p @ T1p))
    worst2 = max(worst2, np.abs(O12p - gi @ O12 @ np.linalg.inv(gj)).max())
print('(a) Omega inverse residual', worst1, ' covariance residual', worst2)
T12 = np.array([[1., 1], [0, 1]]); T21 = T12.T
O = T12 @ np.linalg.inv(sqrtm(T21 @ T12))
print('    Laplace example Omega_12 =', np.round(O.real, 6), 'det', np.linalg.det(O).real, 'O O^T - I', np.abs(O @ O.T - np.eye(2)).max())

# (b)
def Q(W, S):
    I = np.eye(len(W)); D = I - W
    return 0.5 * np.trace(S @ D.T @ np.linalg.solve(S, D))
worst = 0
for _ in range(100):
    W = expm(0.5 * rng.normal(size=(3, 3))); X = rng.normal(size=(3, 3)); S = X @ X.T + 0.2 * np.eye(3)
    g = rng.normal(size=(3, 3)) + 2 * np.eye(3)
    worst = max(worst, abs(Q(W, S) - Q(g @ W @ np.linalg.inv(g), g @ S @ g.T)) / Q(W, S))
print('(b) relative invariance error', worst, ' Q(two-cycle, I) =', Q(np.array([[2., 1], [1, 1]]), np.eye(2)), ' Q(shear, I) =', Q(T12, np.eye(2)))

# (c)
for eps in [1e-1, 1e-2, 1e-3]:
    cs = np.linspace(-50, 50, 2000001)
    AF = -eps / (cs**2 + eps**2)
    print('(c) eps', eps, ' A^F(0) =', -1 / eps, ' integral', np.trapezoid(AF, cs))

# (d)
def hmin(Gm, dp, Jv):
    k = Jv @ Gm @ Jv; a = -(Jv @ Gm @ dp) / k; r = dp + Jv * a
    return r @ Gm @ r, a, r
G = np.eye(2); Gb = np.diag([0.5, 1.0]); Jv = np.array([1., 1.]); dp = np.array([1., -1.])
h, a, r = hmin(G, dp, Jv); hb, ab, _ = hmin(Gb, dp, Jv)
D = G - Gb
print('(d) example', h, hb, r @ D @ r, (Jv @ Gb @ Jv) * (a - ab) ** 2, ' residual', h - hb - r @ D @ r - (Jv @ Gb @ Jv) * (a - ab) ** 2)
worst = 0
for _ in range(200):
    n = 4; tau = rng.uniform(0.1, 2, n)
    G = np.eye(n); Gb = np.diag(1 / (1 + tau**2)); Jv = np.ones(n); dp = rng.normal(size=n)
    h, a, r = hmin(G, dp, Jv); hb, ab, _ = hmin(Gb, dp, Jv)
    worst = max(worst, abs(h - hb - r @ (G - Gb) @ r - (Jv @ Gb @ Jv) * (a - ab) ** 2))
print('    random residual', worst)

# horizontal-defect witness: location family, noise variance s = 1, connection offset c
for cc in [0.1, -0.6, 0.5]:
    print('    witness c =', cc, ' fine - coarse =', 1 - (1 + cc) ** 2 / 2)
