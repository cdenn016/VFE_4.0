"""D03-GEOM s05: determinant appendix (geo:determinant-metric, geo:det-canonical,
geo:det-rank, geo:det-horizontal, geo:circle-completion premises).

(a) eta(H,H) = -det H; congruence scaling; gauge invariance of gamma with kappa = det M.
(b) canonical identity eta(H,K)/det M = (1/2)[Tr(M^-1 H M^-1 K) - Tr(M^-1 H) Tr(M^-1 K)];
    h^CN equals the Fisher metric of CN(0,M) computed through the real 4x4 Gaussian;
    theta = d log det M + 2 Re Tr A; dilation e = cM gives gamma = -c^2;
    differential entropy of CN(0,M) equals log det(pi e M).
(c) rank of dM restricted to H for proper complex Gaussian targets: n = 1, 2 (generic
    and proportional), 3 (generic), 4 (the manuscript's witness and random).
(d) scalar curvature of -det(dM)/det M on the positive Hermitian cone (sympy, exact).
"""
import numpy as np
import sympy as sp
rng = np.random.default_rng(7)

def herm(): X = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)); return (X + X.conj().T)/2
def pd(): X = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)); return X @ X.conj().T + 0.5*np.eye(2)
def eta(H, K): return 0.5*(np.trace(H @ K) - np.trace(H)*np.trace(K)).real

print("== (a)")
H, K, M = herm(), herm(), pd()
g = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
print("  eta(H,H)+det H:", f"{abs(eta(H, H) + np.linalg.det(H).real):.1e}")
print("  scaling residual:", f"{abs(eta(g@H@g.conj().T, g@K@g.conj().T) - abs(np.linalg.det(g))**2*eta(H, K)):.1e}")
gam = eta(H, K)/np.linalg.det(M).real
Mp = g @ M @ g.conj().T
gam_p = eta(g@H@g.conj().T, g@K@g.conj().T)/np.linalg.det(Mp).real
print("  gamma invariance with kappa = det M:", f"{abs(gam - gam_p):.1e}")

print("== (b)")
Mi = np.linalg.inv(M)
lhs = eta(H, K)/np.linalg.det(M).real
rhs = 0.5*(np.trace(Mi@H@Mi@K) - np.trace(Mi@H)*np.trace(Mi@K)).real
print("  canonical identity residual:", f"{abs(lhs - rhs):.1e}")
def realcov(Mc): return 0.5*np.block([[Mc.real, -Mc.imag], [Mc.imag, Mc.real]])
S = realcov(M); dS = realcov(H); Si = np.linalg.inv(S)
fisher_real = 0.5*np.trace(Si@dS@Si@dS)
print("  Fisher of CN(0,M) via real Gaussian vs Tr(M^-1 H M^-1 H):", f"{abs(fisher_real - np.trace(Mi@H@Mi@H).real):.1e}")
A = rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)); dM = herm()
e = dM + A@M + M@A.conj().T
th = np.trace(Mi@e).real
print("  theta - (dlogdet M + 2 Re Tr A):", f"{abs(th - (np.trace(Mi@dM).real + 2*np.trace(A).real)):.1e}")
cdil = 1.7
print("  dilation gamma + c^2:", f"{abs(eta(cdil*M, cdil*M)/np.linalg.det(M).real + cdil**2):.1e}")
ent_real = 0.5*np.log((2*np.pi*np.e)**4*np.linalg.det(S))
print("  entropy residual vs log det(pi e M):", f"{abs(ent_real - np.log(np.linalg.det(np.pi*np.e*M).real)):.1e}")

print("== (c) ranks")
basisH = [np.array([[1, 0], [0, 0]], complex), np.array([[0, 0], [0, 1]], complex),
          np.array([[0, 1], [1, 0]], complex), np.array([[0, 1j], [-1j, 0]])]  # matches hv: H01 = Re + i Im
def hv(Hm): return np.array([Hm[0, 0].real, Hm[1, 1].real, Hm[0, 1].real, Hm[0, 1].imag])
gl = []
for i in range(2):
    for j in range(2):
        Eij = np.zeros((2, 2), complex); Eij[i, j] = 1; gl += [Eij, 1j*Eij]
def ranks(Ms, w=None):
    n = len(Ms); w = np.ones(n) if w is None else w
    # orbit directions
    Jcols = [np.concatenate([hv(xi@Mi_ + Mi_@xi.conj().T) for Mi_ in Ms]) for xi in gl]
    Jm = np.column_stack(Jcols)
    # metric on Herm^n: sum w_i Tr(M_i^-1 h M_i^-1 k)
    Gm = np.zeros((4*n, 4*n))
    for i, Mi_ in enumerate(Ms):
        Mii = np.linalg.inv(Mi_)
        for a in range(4):
            for b in range(4):
                Gm[4*i + a, 4*i + b] = w[i]*np.trace(Mii@basisH[a]@Mii@basisH[b]).real
    rJ = np.linalg.matrix_rank(Jm, tol=1e-9)
    # horizontal space: G-orthogonal complement of im J
    U, s, Vt = np.linalg.svd(Jm.T @ Gm)
    Hb = Vt[np.sum(s > 1e-9):].T
    dMm = np.hstack([w[i]*np.eye(4) for i in range(n)])
    sv = np.linalg.svd(dMm @ Hb, compute_uv=False) if Hb.shape[1] else np.array([])
    return rJ, Hb.shape[1], int(np.sum(sv > 1e-9)), sv
for n in (1, 2, 3, 4):
    out = [ranks([pd() for _ in range(n)], rng.random(n) + 0.3)[:3] for _ in range(5)]
    print(f"  n={n} random (rank J, dim H, rank dM|H):", out)
rJ, dH, rk, sv = ranks([pd() for _ in range(3)])
print("  n=3 singular values of dM|H:", np.array2string(sv, precision=3))
# consistency check of coordinates: reconstruct H from hv
Ht = herm(); rec = sum(hv(Ht)[a]*basisH[a] for a in range(4)); print("  coordinate reconstruction residual:", f"{np.max(np.abs(rec - Ht)):.1e}")
# explicit kernel vectors of the proof for a random triple
T3 = [pd() for _ in range(3)]; Ai = np.linalg.inv(np.linalg.inv(T3[0]) - np.linalg.inv(T3[2])); Bm = np.linalg.inv(T3[1]) - np.linalg.inv(T3[2]); Am = np.linalg.inv(Ai)
for k1, k2 in [(Ai, -np.linalg.inv(Bm)), (Ai@Bm@Ai, -Ai)]:
    k3 = -k1 - k2; print("  proof kernel vector: |sum k| =", f"{np.max(np.abs(k1+k2+k3)):.1e}", " |sum M^-1 k| =", f"{np.max(np.abs(sum(np.linalg.inv(Mm)@kk for Mm, kk in zip(T3, (k1,k2,k3))))):.1e}")
M4 = [np.eye(2, dtype=complex), 2*np.eye(2, dtype=complex), np.diag([1, 2]).astype(complex), np.array([[2, 1], [1, 2]], complex)]
print("  n=4 manuscript witness:", ranks(M4)[:3])
Mx = pd(); print("  n=2 proportional (M, 2M):", ranks([Mx, 2*Mx])[:3])
print("  n=3 proportional (M, 2M, 3M):", ranks([Mx, 2*Mx, 3*Mx])[:3])

print("== (d) scalar curvature of -det(dM)/det M")
t, x, y, z = co = sp.symbols('t x y z', real=True)
Om = 1/(t**2 - x**2 - y**2 - z**2)
g4 = sp.diag(-Om, Om, Om, Om); gi = g4.inv()
Gam = [[[sum(gi[a, d]*(sp.diff(g4[d, b], co[cc]) + sp.diff(g4[d, cc], co[b]) - sp.diff(g4[b, cc], co[d])) for d in range(4))/2
         for cc in range(4)] for b in range(4)] for a in range(4)]
def Ric(b, cc):
    return sp.simplify(sum(sp.diff(Gam[a][b][cc], co[a]) - sp.diff(Gam[a][b][a], co[cc])
               + sum(Gam[a][a][d]*Gam[d][b][cc] - Gam[a][cc][d]*Gam[d][b][a] for d in range(4)) for a in range(4)))
Rs = sp.simplify(sum(gi[b, cc]*Ric(b, cc) for b in range(4) for cc in range(4) if gi[b, cc] != 0))
print("  scalar curvature:", Rs)
