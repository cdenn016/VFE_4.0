r"""L02-GEOM s03: collective minimizing connection (geo:collective, geo:minimal-pullback,
geo:retained-connection) and proposed extensions.

A. Aff(1) acting diagonally on n=3 normal laws (coords (mu, s=log sigma), Fisher diag(e^{-2s},2)):
   normal equation, orthogonality, h^A = h^min + k(A-A^F), affine covariance of A^F.
B. Translation group acting on n normal laws; fixed equivariant Markov retention adding independent
   N(0,tau_i^2) noise. Checks geo:retained-connection and the NEW identity
       h^min(X,X) - hbar^min(X,X) = Delta(r_X,r_X) + kbar(A^F(X)-Abar^F(X))  (>= 0),
   including the manuscript example (means (x,-x), noise on first sample): 2 = 4/3 + 1/2 + 1/6.
C. Labeled-mixture reading of the weighted metric G = sum_i w_i g_i: weight scores are orthogonal
   to orbit scores (Gauss-Hermite quadrature).
D. Rank budget: orbit codimension (= dim H, an upper bound for rank h^min) for GL(K), SO(K) and
   Aff(K) acting on n Gaussian laws N(mu, Sigma) on R^K.
"""
import numpy as np

rng = np.random.default_rng(11)

# ---------- A ----------
n = 3
mu = rng.normal(size=n); s = rng.normal(size=n) * 0.3; w = rng.random(n) + 0.5
def Gmat(mu, s):
    return np.diag(np.concatenate([[w[i] * np.exp(-2 * s[i]), 2 * w[i]] for i in range(n)]))
def Jmat(mu, s):
    return np.vstack([[[mu[i], 1.0], [1.0, 0.0]] for i in range(n)])
G = Gmat(mu, s); J = Jmat(mu, s)
dp = rng.normal(size=(2 * n, 2))
k = J.T @ G @ J
AF = -np.linalg.solve(k, J.T @ G @ dp)
r = dp + J @ AF
print('A. orthogonality |J^T G r|:', np.abs(J.T @ G @ r).max())
Arand = rng.normal(size=(2, 2))
hA = (dp + J @ Arand).T @ G @ (dp + J @ Arand)
hmin = r.T @ G @ r
print('A. h^A - h^min - k(A-A^F):', np.abs(hA - hmin - (Arand - AF).T @ k @ (Arand - AF)).max())
a, b = np.exp(rng.normal()), rng.normal()
da, db = rng.normal(size=2), rng.normal(size=2)
mu2, s2 = a * mu + b, s + np.log(a)
T = np.diag(np.concatenate([[a, 1.0] for _ in range(n)]))
gm = np.array([[a, b], [0, 1]]); gi = np.linalg.inv(gm)
dgginv = [np.array([[da[jx], db[jx]], [0, 0]]) @ gi for jx in range(2)]
J2 = Jmat(mu2, s2); G2 = Gmat(mu2, s2)
dp2 = T @ dp + np.column_stack([J2 @ np.array([M[0, 0], M[0, 1]]) for M in dgginv])
k2 = J2.T @ G2 @ J2
AF2 = -np.linalg.solve(k2, J2.T @ G2 @ dp2)
pred_cols = []
for jx in range(2):
    M = gm @ np.array([[AF[0, jx], AF[1, jx]], [0, 0]]) @ gi - dgginv[jx]
    pred_cols.append(np.array([M[0, 0], M[0, 1]]))
pred = np.column_stack(pred_cols)
print('A. affine-law residual |A^F(p2) - (Ad_g A^F - dg g^-1)|:', np.abs(AF2 - pred).max())
print('A. residual transforms by T(g):', np.abs((dp2 + J2 @ AF2) - T @ r).max(),
      '; h^min invariant:', np.abs((dp2 + J2 @ AF2).T @ G2 @ (dp2 + J2 @ AF2) - hmin).max())

# ---------- B ----------
def collective_min(gdiag, J, dp):
    Gm = np.diag(gdiag)
    km = J.T @ Gm @ J
    A = -np.linalg.solve(km, J.T @ Gm @ dp)
    rr = dp + J @ A
    return A, rr, rr.T @ Gm @ rr, km
for trial in range(3):
    m = 4
    sig = np.exp(rng.normal(size=m) * 0.3); tau = np.abs(rng.normal(size=m))
    wB = rng.random(m) + 0.5
    g_f = np.concatenate([[wB[i] / sig[i] ** 2, 2 * wB[i] / sig[i] ** 2] for i in range(m)])
    sbar = np.sqrt(sig ** 2 + tau ** 2)
    dR = np.diag(np.concatenate([[1.0, sig[i] / sbar[i]] for i in range(m)]))
    g_c = np.concatenate([[wB[i] / sbar[i] ** 2, 2 * wB[i] / sbar[i] ** 2] for i in range(m)])
    JB = np.concatenate([[1.0, 0.0] for _ in range(m)])[:, None]
    Delta = np.diag(g_f) - dR.T @ np.diag(g_c) @ dR
    dpB = rng.normal(size=(2 * m, 1))
    A1, r1, h1, k1 = collective_min(g_f, JB, dpB)
    A2, r2, h2, k2b = collective_min(g_c, dR @ JB, dR @ dpB)
    lhs = h1 - h2
    rhs = r1.T @ Delta @ r1 + (A1 - A2).T @ k2b @ (A1 - A2)
    print('B. trial %d: min eig Delta = %.3e, h^min - hbar^min = %.6f, identity residual = %.2e,'
          ' Delta(r,J) = %.3e, A^F shift = %.3e'
          % (trial, np.linalg.eigvalsh(Delta).min(), lhs[0, 0], abs(lhs - rhs).max(),
             (r1.T @ Delta @ JB)[0, 0], (A1 - A2)[0, 0]))
gf = np.array([1.0, 1.0]); gc = np.array([0.5, 1.0]); Jl = np.array([[1.0], [1.0]])
dpl = np.array([[1.0], [-1.0]])
A1, r1, h1, _ = collective_min(gf, Jl, dpl)
A2, r2, h2, k2b = collective_min(gc, Jl, dpl)
Dl = np.diag(gf - gc)
print('B. manuscript example: A^F=%.4f, Abar^F=%.4f, h^min=%.4f, hbar^min=%.4f, Delta(r,r)=%.4f,'
      ' kbar shift=%.4f, residual=%.1e' % (A1[0, 0], A2[0, 0], h1[0, 0], h2[0, 0],
      (r1.T @ Dl @ r1)[0, 0], ((A1 - A2).T @ k2b @ (A1 - A2))[0, 0],
      abs(h1 - h2 - r1.T @ Dl @ r1 - (A1 - A2).T @ k2b @ (A1 - A2))[0, 0]))
dm, dd = rng.normal(size=2)
A1, r1, h1, _ = collective_min(gf, Jl, np.array([[dm + dd], [dm - dd]]))
print('B. companion example: A^F + dm = %.2e, h^min - 2 dd^2 = %.2e' % (A1[0, 0] + dm, h1[0, 0] - 2 * dd ** 2))

# ---------- C ----------
xs, ws = np.polynomial.hermite_e.hermegauss(40)
ws = ws / ws.sum()
mC = 3
wC = rng.random(mC); wC /= wC.sum(); muC = rng.normal(size=mC)
dw = rng.normal(size=mC)
dw = dw - wC * dw.sum()          # keep total weight fixed: sum dw = 0
dmu = rng.normal(size=mC); xi = rng.normal()
# score of labeled law R(i,o) = w_i N(o; mu_i, 1) at o = mu_i + x:  dw_i/w_i + x dmu_i ; orbit: x xi
cross = sum(wC[i] * np.sum(ws * (dw[i] / wC[i]) * xs * xi) for i in range(mC))
fis = sum(wC[i] * np.sum(ws * ((dw[i] / wC[i]) + xs * dmu[i]) ** 2) for i in range(mC))
print('C. weight/orbit Fisher pairing:', cross, '; Fisher - [sum w dmu^2 + sum dw^2/w]:',
      fis - (np.sum(wC * dmu ** 2) + np.sum(dw ** 2 / wC)))

# ---------- D ----------
def orbit_codim(K, nlaw, group):
    laws = []
    for _ in range(nlaw):
        mu_ = rng.normal(size=K); L = rng.normal(size=(K, K)); Sig = L @ L.T + K * np.eye(K)
        laws.append((mu_, Sig))
    gens = []
    if group in ('GL', 'Aff'):
        for i in range(K):
            for jj in range(K):
                E = np.zeros((K, K)); E[i, jj] = 1; gens.append((E, np.zeros(K)))
    if group == 'SO':
        for i in range(K):
            for jj in range(i + 1, K):
                E = np.zeros((K, K)); E[i, jj] = 1; E[jj, i] = -1; gens.append((E, np.zeros(K)))
    if group == 'Aff':
        for i in range(K):
            v = np.zeros(K); v[i] = 1; gens.append((np.zeros((K, K)), v))
    iu = np.triu_indices(K)
    cols = []
    for E, v in gens:
        col = []
        for mu_, Sig in laws:
            col.append(E @ mu_ + v)
            col.append((E @ Sig + Sig @ E.T)[iu])
        cols.append(np.concatenate(col))
    Jm = np.array(cols).T
    dimN = nlaw * (K + K * (K + 1) // 2)
    rk = np.linalg.matrix_rank(Jm, tol=1e-9)
    return dimN, rk, dimN - rk
for group in ('Aff', 'GL', 'SO'):
    for K in (2, 3):
        for nl in (1, 2, 3):
            dimN, rk, cod = orbit_codim(K, nl, group)
            print('D. %-3s K=%d n=%d: dim N^n=%2d, orbit dim=%2d, dim H (rank bound for h^min)=%2d'
                  % (group, K, nl, dimN, rk, cod))
