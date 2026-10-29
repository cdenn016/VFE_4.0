"""D03-GEOM s08: worked examples of chapters/02_geometry.tex and companion/c06_geometry.tex.
(a) SO(2) on unit-covariance planar normal laws, mean (eps, c): A^F, its integral, h^min bound.
(b) common nonzero mean e1 for n agents: locked inertia n > 0 (free action at that tuple).
(c) companion two-location example: A^F = -dm, h^min = 2 dd^2.
(d) companion Bernoulli pullback p'^2/(p(1-p)).
(e) transitive pullback: rank of k_{p0} for one Gaussian under Aff(1) and Aff(2).
(f) labeled-family reading of the weights: Fisher of sum_i w_i delta_i (x) p_i and the
    weight term sum dw^2/w, zero pairing with orbit directions (normal location, translations).
"""
import sympy as sp
import numpy as np
eps, c = sp.symbols('epsilon c', positive=True)
cr = sp.symbols('c', real=True)
mu = sp.Matrix([eps, cr]); Jxi = sp.Matrix([-mu[1], mu[0]]); dmu = sp.diff(mu, cr)
k = (Jxi.T*Jxi)[0]; AF = -(dmu.T*Jxi)[0]/k
hmin = sp.simplify((dmu + Jxi*AF).dot(dmu + Jxi*AF))
print("(a) k =", sp.simplify(k), "; A^F =", sp.simplify(AF), "; int A^F dc =", sp.simplify(sp.limit(-sp.atan(cr/eps), cr, sp.oo) - sp.limit(-sp.atan(cr/eps), cr, -sp.oo)), "; d/dc(-atan(c/eps)) - A^F =", sp.simplify(sp.diff(-sp.atan(cr/eps), cr) - AF),
      "; h^min =", hmin, "; 1 - h^min =", sp.simplify(1 - hmin))
n = 5; mus = np.tile([1.0, 0.0], (n, 1)); Jv = np.concatenate([[-m[1], m[0]] for m in mus])
print("(b) locked inertia for", n, "agents at common mean e1:", Jv @ Jv)
m, d, dm, dd, a = sp.symbols('m d dm dd a', real=True)
cost = (dm + dd + a)**2 + (dm - dd + a)**2
amin = sp.solve(sp.diff(cost, a), a)[0]
print("(c) A^F =", amin, "; h^min =", sp.expand(cost.subs(a, amin)))
p = sp.Function('p')(cr)
h = p*sp.diff(sp.log(p), cr)**2 + (1 - p)*sp.diff(sp.log(1 - p), cr)**2
print("(d) Bernoulli pullback residual:", sp.simplify(h - sp.diff(p, cr)**2/(p*(1 - p))))
# (e) Aff(K) on one Gaussian N(mu, Sigma): J(b, B) = (B mu + b, B Sigma + Sigma B^T)
rng = np.random.default_rng(1)
for K in (1, 2):
    muv = rng.normal(size=K); L = rng.normal(size=(K, K)); Sig = L @ L.T + np.eye(K)
    cols = []
    for idx in range(K + K*K):
        b = np.zeros(K); B = np.zeros((K, K))
        if idx < K: b[idx] = 1
        else: B.flat[idx - K] = 1
        dS = B @ Sig + Sig @ B.T
        cols.append(np.concatenate([B @ muv + b, dS[np.triu_indices(K)]]))
    Jm = np.column_stack(cols)
    print(f"(e) Aff({K}) on one Gaussian: rank J = {np.linalg.matrix_rank(Jm)} = dim N = {K + K*(K+1)//2}")
# (f) labeled family: weights w(c), location laws N(mu_i(c), 1), translations (Gauss-Hermite)
xg, wg = np.polynomial.hermite_e.hermegauss(40); wg = wg/np.sqrt(2*np.pi)
c0 = 0.4
w = lambda t: np.array([1/3 + t/7, 2/3 - t/7]); dw = np.array([1/7, -1/7])
muf = lambda t: np.array([t**2, -t]); dmu_ = np.array([2*c0, -1.0])
Fcc = 0.0; pair = 0.0
for i in range(2):
    ysamp = muf(c0)[i] + xg
    lc = dw[i]/w(c0)[i] + (ysamp - muf(c0)[i])*dmu_[i]   # contextual score of the labeled law
    lo = (ysamp - muf(c0)[i])                            # common-translation orbit score
    Fcc += w(c0)[i]*np.sum(wg*lc**2)
    pair += w(c0)[i]*np.sum(wg*(dw[i]/w(c0)[i])*lo)
pred = np.sum(w(c0)*dmu_**2) + np.sum(dw**2/w(c0))
print("(f) labeled-family contextual Fisher - (sum w dmu^2 + sum dw^2/w):", f"{Fcc - pred:.1e}",
      "; weight-score pairing with orbit score:", f"{pair:.1e}")
