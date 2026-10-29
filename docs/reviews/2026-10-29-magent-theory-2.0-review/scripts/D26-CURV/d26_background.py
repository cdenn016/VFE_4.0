r"""D26-CURV check 5: it3:observation examples, and an extension beyond the identity background.

(a) E_22 with c = e_1: single-topic record blind (E_22 c = 0), grounded test T A = E_22 nonzero.
(b) Quarter turn H: KL(N(0,I) || H#N(0,I)) = 0 while Q(H; I_2) = 2.
(c) Extension: linearize the exact record T_e(theta_e^-1 y_i - y_j) at a background theta0 instead of
    the identity. The coboundary becomes the twisted one, (delta0 y)_e = T_e(theta0_e^-1 y_i - y_j), and
    (V0 omega)_e = T_e theta0_e^-1 omega_e c_i. Predicted rank of the weak-anchor penalty with injective
    codes: K(|E| - |V|) + dim H^0, where H^0 = sections y_i = theta0_e y_j (holonomy-fixed vectors).
"""
from d26_common import *
from scipy.linalg import expm

rng = np.random.default_rng(99)
E22 = np.array([[0., 0.], [0., 1.]]); c = np.array([1., 0.])
print("(a) E22 c =", E22 @ c, " T E22 with T = I_2 nonzero:", np.abs(np.eye(2) @ E22).max() > 0)
Hq = np.array([[0., -1.], [1., 0.]])
Sigma = np.eye(2)
kl0 = 0.5 * (np.trace(np.linalg.solve(Hq @ Hq.T, Sigma)) - 2 + np.log(np.linalg.det(Hq @ Hq.T) / np.linalg.det(Sigma)))
Q = 0.5 * np.linalg.norm(np.eye(2) - Hq, "fro") ** 2
print("(b) KL(q || H#q) =", kl0, " Q(H; I_2) =", Q)


def twisted(K, nV, edges, codes, c, th0):
    nH = sum(T.shape[0] for T in codes)
    D = np.zeros((nH, nV * K)); V = np.zeros((nH, len(edges) * K * K)); r = 0
    for e, (j, i) in enumerate(edges):
        T = codes[e]; n = T.shape[0]; Ti = T @ np.linalg.inv(th0[e])
        D[r:r + n, i * K:(i + 1) * K] += Ti
        D[r:r + n, j * K:(j + 1) * K] -= T
        V[r:r + n, e * K * K:(e + 1) * K * K] = Ti @ kap(c[i])
        r += n
    return D, V


K = 2
cases = {}
ring = [(0, 1), (1, 2), (2, 0)]
codes = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in ring]
cs = [rng.standard_normal(K) for _ in range(3)]
g = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in range(3)]
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
cases["identity background"] = [np.eye(K)] * 3
cases["pure gauge g_i g_j^-1"] = [g[i] @ np.linalg.inv(g[j]) for (j, i) in ring]
cases["holonomy rotation 0.7 (no fixed vector)"] = [rot(0.7), np.eye(K), np.eye(K)]
cases["holonomy diag(1, 2) (one fixed vector)"] = [np.diag([1.0, 2.0]), np.eye(K), np.eye(K)]
cases["holonomy unipotent I + E12 (one fixed vector)"] = [np.eye(K) + np.array([[0., 1.], [0., 0.]]), np.eye(K), np.eye(K)]
R = blockdiag([spd(rng, K, 0.5) for _ in ring])
for name, th0 in cases.items():
    D, V = twisted(K, 3, ring, codes, cs, th0)
    lim = V.T @ harmonic_form(D, R) @ V
    h0 = 3 * K - np.linalg.matrix_rank(D, tol=1e-9)
    Hol = th0[2] @ th0[1] @ th0[0]
    print("(c) ring,", name, ": rank =", np.linalg.matrix_rank(lim, tol=1e-9), " predicted K(|E|-|V|) + dim H^0 =",
          K * 0 + h0, " dim ker(Hol - I) =", 2 - np.linalg.matrix_rank(Hol - np.eye(K), tol=1e-9))
edges = [(0, 1), (1, 0), (1, 2), (2, 0), (0, 2)]
codes = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in edges]
R = blockdiag([spd(rng, K, 0.5) for _ in edges])
for name, th0 in {"identity": [np.eye(K)] * 5,
                  "generic nonflat": [expm(0.5 * rng.standard_normal((K, K))) for _ in edges]}.items():
    D, V = twisted(K, 3, edges, codes, cs, th0)
    lim = V.T @ harmonic_form(D, R) @ V
    h0 = 3 * K - np.linalg.matrix_rank(D, tol=1e-9)
    print("    five-edge graph,", name, ": rank =", np.linalg.matrix_rank(lim, tol=1e-9),
          " predicted K(|E|-|V|) + dim H^0 =", K * (5 - 3) + h0)

# (d) exact-record determinant at weak anchors: (1/2) log det(eps Pbar + delta0^T R^-1 delta0) ~ (h0/2) log eps
print("(d) slope of (1/2) log det J in log eps, ring with K = 2")
codes = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in ring]
R = blockdiag([spd(rng, K, 0.5) for _ in ring]); Ri = np.linalg.inv(R)
Pbar = blockdiag([spd(rng, K) for _ in range(3)])
for name, th0 in cases.items():
    D, _ = twisted(K, 3, ring, codes, cs, th0)
    f = lambda eps: 0.5 * np.linalg.slogdet(eps * Pbar + D.T @ Ri @ D)[1]
    slope = (f(1e-8) - f(1e-6)) / (np.log(1e-8) - np.log(1e-6))
    print("   ", name, ": slope =", round(float(slope), 4), " predicted h0/2 =", (3 * K - np.linalg.matrix_rank(D, tol=1e-9)) / 2)
