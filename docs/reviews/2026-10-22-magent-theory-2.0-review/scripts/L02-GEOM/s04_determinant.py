r"""L02-GEOM s04: determinant appendix (geo:determinant-metric, geo:circle-completion).

1. eta(H,H) = -det H, Pauli Gram matrix diag(-1,1,1,1), scaling |det g|^2, gamma invariance.
2. NEW: canonical scale kappa = det M. Identity
      -det(H)/det(M) = 1/2 tr((M^{-1}H)^2) - 1/2 (tr M^{-1}H)^2
   and the complex-Gaussian Fisher metric of CN(0,M) is tr(M^{-1}H M^{-1}K) (checked through the
   real 4x4 representation). Scalar curvature of the GL(2,C)-invariant metric -det(dM)/det(M) on
   the positive Hermitian cone (expected R x H^3, scalar curvature -6).
3. Circle connection: Im tr(dg g^{-1}) = d arg det g.
4. Nonvacuity: rank of dM|_H for collectives of n proper complex Gaussians CN(0,M_i) under GL(2,C)
   (soldering e = dM r needs rank 4 at dim C = 4).
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(3)
t, x, y, zz = sp.symbols('t x y z', real=True)
Hs = sp.Matrix([[t + zz, x - sp.I * y], [x + sp.I * y, t - zz]])
eta = lambda H, K: ((H * K).trace() - H.trace() * K.trace()) / 2
print('1. eta(H,H) + det H =', sp.simplify(sp.expand(eta(Hs, Hs) + Hs.det())))
sig = [sp.eye(2), sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]),
       sp.Matrix([[1, 0], [0, -1]])]
print('1. Pauli Gram:', sp.Matrix(4, 4, lambda i, j: sp.simplify(eta(sig[i], sig[j]))).tolist())
def etan(H, K):
    return 0.5 * (np.trace(H @ K) - np.trace(H) * np.trace(K)).real
def rherm():
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return A + A.conj().T
g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
H1, H2 = rherm(), rherm()
print('1. scaling residual:', etan(g @ H1 @ g.conj().T, g @ H2 @ g.conj().T) - abs(np.linalg.det(g)) ** 2 * etan(H1, H2))
kap = 1.7
gam = etan(H1, H2) / kap
gam2 = etan(g @ H1 @ g.conj().T, g @ H2 @ g.conj().T) / (abs(np.linalg.det(g)) ** 2 * kap)
print('1. gamma invariance residual:', gam - gam2)

# 2. canonical scale
L = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); M = L @ L.conj().T + np.eye(2)
H = rherm()
Mi = np.linalg.inv(M)
lhs = -np.linalg.det(H).real / np.linalg.det(M).real
rhs = 0.5 * np.trace(Mi @ H @ Mi @ H).real - 0.5 * np.trace(Mi @ H).real ** 2
print('2. det-normalized form vs (1/2)AIRM - (1/2)(dlogdet)^2 residual:', lhs - rhs)
def realrep(S):
    return 0.5 * np.block([[S.real, -S.imag], [S.imag, S.real]])
SR = realrep(M); dSR = realrep(H)
fisher_real = 0.5 * np.trace(np.linalg.inv(SR) @ dSR @ np.linalg.inv(SR) @ dSR)
print('2. complex-Gaussian Fisher tr(M^-1 H M^-1 H) vs real-rep Fisher residual:',
      fisher_real - np.trace(Mi @ H @ Mi @ H).real)
coords = [t, x, y, zz]
etaM = sp.diag(-1, 1, 1, 1)
gmet = etaM / (t ** 2 - x ** 2 - y ** 2 - zz ** 2)
gin = sp.simplify(gmet.inv())
Gam = [[[sp.simplify(sum(gin[a, d] * (sp.diff(gmet[d, b], coords[c]) + sp.diff(gmet[d, c], coords[b])
                                      - sp.diff(gmet[b, c], coords[d])) for d in range(4)) / 2)
         for c in range(4)] for b in range(4)] for a in range(4)]
def Riem(a, b, c, d):
    e = sp.diff(Gam[a][b][d], coords[c]) - sp.diff(Gam[a][b][c], coords[d])
    e += sum(Gam[a][c][k] * Gam[k][b][d] - Gam[a][d][k] * Gam[k][b][c] for k in range(4))
    return e
Ric = sp.Matrix(4, 4, lambda b, d: sum(Riem(a, b, a, d) for a in range(4)))
Rs = sp.simplify(sum(gin[b, d] * Ric[b, d] for b in range(4) for d in range(4)))
print('2. scalar curvature of -det(dM)/det(M) on the positive cone:', Rs)

# 3. circle connection
g0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) + 2 * np.eye(2)
dg = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
eps = 1e-6
darg = (np.angle(np.linalg.det(g0 + eps * dg)) - np.angle(np.linalg.det(g0 - eps * dg))) / (2 * eps)
print('3. Im tr(dg g^-1) - d arg det g:', np.trace(dg @ np.linalg.inv(g0)).imag - darg)

# 4. rank of dM|_H
basis_h = [np.array(s_.tolist(), dtype=complex) for s_ in sig]
gens = []
for i in range(2):
    for j in range(2):
        E = np.zeros((2, 2), complex); E[i, j] = 1
        gens += [E, 1j * E]
def hvec(H):
    return np.array([0.5 * np.trace(b_ @ H).real for b_ in basis_h])  # Pauli coordinates
for nlaw in (1, 2, 3, 4):
    ranks = []
    for trial in range(5):
        Ms = []
        for _ in range(nlaw):
            L = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); Ms.append(L @ L.conj().T + 0.3 * np.eye(2))
        wts = rng.random(nlaw) + 0.5
        dim = 4 * nlaw
        Gm = np.zeros((dim, dim))
        for i_, Mi_ in enumerate(Ms):
            Mi_inv = np.linalg.inv(Mi_)
            for a_ in range(4):
                for b_ in range(4):
                    Gm[4 * i_ + a_, 4 * i_ + b_] = wts[i_] * np.trace(Mi_inv @ basis_h[a_] @ Mi_inv @ basis_h[b_]).real
        Jm = np.column_stack([np.concatenate([hvec(xi @ Mi_ + Mi_ @ xi.conj().T) for Mi_ in Ms]) for xi in gens])
        # H = G-orthogonal complement of im J
        U, S, Vt = np.linalg.svd(Jm.T @ Gm)
        rk = int((S > 1e-9).sum())
        Hb = Vt[rk:].T                       # basis of {v : J^T G v = 0}
        dMm = np.hstack([wts[i_] * np.eye(4) for i_ in range(nlaw)])
        ranks.append((rk, Hb.shape[1], np.linalg.matrix_rank(dMm @ Hb, tol=1e-9) if Hb.shape[1] else 0))
    print('4. n=%d proper complex Gaussians: (rank J, dim H, rank dM|_H) over 5 draws:' % nlaw, ranks)
