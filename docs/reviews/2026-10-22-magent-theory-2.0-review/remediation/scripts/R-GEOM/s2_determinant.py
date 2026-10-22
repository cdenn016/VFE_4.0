r"""Determinant appendix checks.

(a) Ranks for proper complex Gaussian targets CN(0, M_i), GL(2,C) acting by M -> g M g^dagger,
    collective metric sum_i w_i tr(M_i^{-1} h_i M_i^{-1} k_i):
    rank J, dim H = dim of Fisher-orthogonal complement, rank of dM|_H with dM(h) = sum_i w_i h_i.
    Generic n = 1..4, a proportional pair (n = 2), and the explicit four-agent witness.
(b) gamma with kappa = det M equals (1/2)[tr((M^{-1}e)^2) - (tr M^{-1}e)^2].
(c) Scalar curvature of -det(dM)/det M on the positive Hermitian 2x2 cone (sympy).
"""
import numpy as np

rng = np.random.default_rng(0)

def herm_basis():
    B = [np.array([[1, 0], [0, 0]], complex), np.array([[0, 0], [0, 1]], complex),
         np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex)]
    return B

def gl_basis():
    out = []
    for r in range(2):
        for s in range(2):
            E = np.zeros((2, 2), complex); E[r, s] = 1; out.append(E)
            E = np.zeros((2, 2), complex); E[r, s] = 1j; out.append(E)
    return out

HB = herm_basis()

def coords(Hm):
    return np.array([Hm[0, 0].real, Hm[1, 1].real, Hm[0, 1].real, -Hm[0, 1].imag])

def ranks(Ms, w=None):
    n = len(Ms)
    w = np.ones(n) if w is None else np.asarray(w)
    # metric on R^{4n}
    G = np.zeros((4 * n, 4 * n))
    for i, M in enumerate(Ms):
        Mi = np.linalg.inv(M)
        for a in range(4):
            for b in range(4):
                G[4 * i + a, 4 * i + b] = w[i] * np.trace(Mi @ HB[a] @ Mi @ HB[b]).real
    Jm = np.zeros((4 * n, 8))
    for k, xi in enumerate(gl_basis()):
        for i, M in enumerate(Ms):
            Jm[4 * i:4 * i + 4, k] = coords(xi @ M + M @ xi.conj().T)
    rJ = np.linalg.matrix_rank(Jm, tol=1e-9)
    # H = G-orthogonal complement of im J: {v : J^T G v = 0}
    Cmat = Jm.T @ G
    u, s, vt = np.linalg.svd(Cmat)
    r = (s > 1e-9).sum()
    Hbasis = vt[r:].T
    dM = np.hstack([w[i] * np.eye(4) for i in range(n)])
    rH = np.linalg.matrix_rank(dM @ Hbasis, tol=1e-9) if Hbasis.shape[1] else 0
    return rJ, Hbasis.shape[1], rH

def rand_pd():
    X = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return X @ X.conj().T + 0.3 * np.eye(2)

for n in range(1, 5):
    print('generic n =', n, [ranks([rand_pd() for _ in range(n)], rng.uniform(0.5, 2, n)) for _ in range(4)])
M = rand_pd()
print('proportional pair (M, 2M):', ranks([M, 2 * M]))
print('three agents (I, 2I, generic):', ranks([np.eye(2), 2 * np.eye(2), rand_pd()]))
W4 = [np.eye(2), 2 * np.eye(2), np.diag([1.0, 2.0]).astype(complex), np.array([[2, 1], [1, 2]], complex)]
print('four-agent witness (I, 2I, diag(1,2), [[2,1],[1,2]]):', ranks(W4))

# n = 3 structural kernel vectors
M1, M2, M3 = rand_pd(), rand_pd(), rand_pd()
A = np.linalg.inv(M1) - np.linalg.inv(M3); B = np.linalg.inv(M2) - np.linalg.inv(M3)
Ai, Bi = np.linalg.inv(A), np.linalg.inv(B)
print('kernel residuals:', np.abs(A @ Ai - B @ Bi).max(), np.abs(A @ (Ai @ B @ Ai) - B @ Ai).max())

# (b) canonical scale identity
for _ in range(3):
    M = rand_pd(); E = rand_pd() - rand_pd(); E = (E + E.conj().T) / 2
    lhs = (0.5 * (np.trace(E @ E) - np.trace(E) ** 2)).real / np.linalg.det(M).real
    Q = np.linalg.solve(M, E)
    rhs = 0.5 * (np.trace(Q @ Q) - np.trace(Q) ** 2).real
    print('kappa = det M identity residual:', lhs - rhs, ' dilation gamma:', 0.5 * (np.trace(np.eye(2)) - 4).real)

# (c) curvature
import sympy as sp
t, xx, yy, zz = sp.symbols('t x y z', real=True)
Hs = sp.Matrix([[t + zz, xx - sp.I * yy], [xx + sp.I * yy, t - zz]])
co = [t, xx, yy, zz]
detH = sp.simplify(Hs.det())
# metric g_ab = -(polarized det of basis)/det H = eta_ab/det H with eta = diag(-1,1,1,1)
eta = sp.diag(-1, 1, 1, 1)
g = eta / detH
ginv = g.inv()
n4 = 4
Gam = [[[sp.simplify(sum(ginv[a, d] * (sp.diff(g[d, b], co[c]) + sp.diff(g[d, c], co[b]) - sp.diff(g[b, c], co[d])) for d in range(n4)) / 2)
         for c in range(n4)] for b in range(n4)] for a in range(n4)]
def Ric(b, c):
    return sum(sp.diff(Gam[a][b][c], co[a]) - sp.diff(Gam[a][b][a], co[c])
               + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a] for d in range(n4)) for a in range(n4))
Rs = sp.simplify(sum(ginv[b, c] * Ric(b, c) for b in range(n4) for c in range(n4)))
print('scalar curvature of -det(dM)/det M:', Rs)
