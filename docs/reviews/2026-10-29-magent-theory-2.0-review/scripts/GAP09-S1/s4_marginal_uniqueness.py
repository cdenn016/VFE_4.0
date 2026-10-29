"""GAP09-S1 s4: marginal uniqueness of product tilts f0(d0) f1(d1) lambda0(dd) (focus question 3).

 (a) Automatic mutual absolute continuity, finite case with zeros: for random sparse lambda0 and
     random tilts with zeros, any second product tilt with the same marginals has the same
     support (checked by brute force over all support patterns of the second tilt on 3x3 grids:
     whenever the IPF/Sinkhorn scaling of f'g'lambda0 to the marginals of P succeeds exactly,
     its limit is P).
 (b) Uniqueness of positive diagonal scaling on the common support: Sinkhorn on P from 200
     random positive initial scalings, target = marginals of P, always returns P.
 (c) Nondegenerate Gaussian lambda0 (product-equivalent): two product tilts with quadratic
     potentials have the same off-diagonal precision; equal marginal variances then force equal
     laws (unique positive root of the determinant equation), checked with sympy.
"""
import itertools
import numpy as np
import sympy as sp

rng = np.random.default_rng(11)


def sinkhorn(R, r, c, iters=20000, tol=1e-15):
    A = R.copy()
    for _ in range(iters):
        rs = A.sum(1); A = A * np.where(rs > 0, r / np.where(rs > 0, rs, 1), 0)[:, None]
        cs = A.sum(0); A = A * np.where(cs > 0, c / np.where(cs > 0, cs, 1), 0)[None, :]
        if np.abs(A.sum(1) - r).max() < tol and np.abs(A.sum(0) - c).max() < tol:
            return A, True
    return A, (np.abs(A.sum(1) - r).max() < 1e-10 and np.abs(A.sum(0) - c).max() < 1e-10)


# (a) brute force over supports of the second tilt
def is_product_tilt(A, lam, tol=1e-9):
    """True if A = a(x) b(y) lam exactly: support equals (supp rows x supp cols) cap supp lam and
    log(A/lam) is additive on the support graph (least-squares residual)."""
    rs, cs = A.sum(1) > 0, A.sum(0) > 0
    supp = A > 1e-14
    if not np.array_equal(supp, np.outer(rs, cs) & (lam > 0)):
        return False
    idx = np.argwhere(supp)
    nr, nc = A.shape
    M = np.zeros((len(idx), nr + nc)); y = np.zeros(len(idx))
    for k, (i, j) in enumerate(idx):
        M[k, i] = 1.0; M[k, nr + j] = 1.0; y[k] = np.log(A[i, j] / lam[i, j])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    return np.abs(M @ coef - y).max() < tol


n = 3
trials, converged, tilts, maxdiff, nontilt = 0, 0, 0, 0.0, 0
for t in range(15):
    lam = rng.uniform(0.1, 1.0, (n, n)) * (rng.uniform(size=(n, n)) > 0.35)
    if lam.sum() == 0:
        continue
    lam /= lam.sum()
    f = rng.uniform(0.2, 2.0, n) * (rng.uniform(size=n) > 0.25)
    g = rng.uniform(0.2, 2.0, n) * (rng.uniform(size=n) > 0.25)
    P = np.outer(f, g) * lam
    if P.sum() == 0:
        continue
    P /= P.sum(); r, c = P.sum(1), P.sum(0)
    trials += 1
    for S0 in itertools.product([0, 1], repeat=n):
        for S1 in itertools.product([0, 1], repeat=n):
            fp = np.array(S0) * rng.uniform(0.2, 2.0, n)
            gp = np.array(S1) * rng.uniform(0.2, 2.0, n)
            R = np.outer(fp, gp) * lam
            if R.sum() == 0:
                continue
            A, ok = sinkhorn(R / R.sum(), r, c, iters=1500, tol=1e-13)
            if not ok:
                continue
            converged += 1
            if is_product_tilt(A, lam):
                tilts += 1
                maxdiff = max(maxdiff, np.abs(A - P).max())
            else:
                nontilt += 1
print("(a) %d random sparse preparations with zero patterns; %d scalings reached the marginals of P;"
      " %d of the limits are exact product tilts of lambda0, all equal to P (max |P' - P| = %.2e);"
      " %d limits are not exact product tilts" % (trials, converged, tilts, maxdiff, nontilt))

# (b) random positive initial scalings of P itself
md = 0.0
for t in range(100):
    lam = rng.uniform(0.0, 1.0, (4, 5)) * (rng.uniform(size=(4, 5)) > 0.4)
    lam[np.arange(4), np.arange(4)] += 0.5   # keep every row and column charged
    lam[:, 4] += 0.2
    P = lam / lam.sum(); r, c = P.sum(1), P.sum(0)
    R = np.outer(np.exp(rng.normal(0, 3, 4)), np.exp(rng.normal(0, 3, 5))) * P
    A, ok = sinkhorn(R / R.sum(), r, c)
    md = max(md, np.abs(A - P).max())
print('(b) Sinkhorn from 100 random positive scalings returns P: max |limit - P| = %.2e' % md)

# (c) Gaussian: lambda0 = N(0, Sigma0) on R^2; product tilt exp(alpha(x0) + beta(x1)) with
# quadratic potentials keeps the off-diagonal precision c = Lambda0[0,1].
a, b, cc, sx, sy = sp.symbols('a b c s_x s_y', real=True)
D = sp.symbols('D')
# precision [[a, c], [c, b]] with det D = ab - c^2 > 0; marginal variances b/D = sx^2, a/D = sy^2
sol = sp.solve(sp.Eq(D, (sy ** 2 * D) * (sx ** 2 * D) - cc ** 2), D)
print('(c) roots for det D given marginal variances and fixed cross-precision c:', sol)
print('    product of roots = %s (negative for c != 0, so exactly one positive root)' %
      sp.simplify(sol[0] * sol[1]) if len(sol) == 2 else '    single root %s' % sol)
