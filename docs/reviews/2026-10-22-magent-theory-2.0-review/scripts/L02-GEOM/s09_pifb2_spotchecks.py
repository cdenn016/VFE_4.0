r"""L02-GEOM s09: spot checks of PIFB2 sections 18-20 formulas used in the comparison.

1. 18 eq:isotropic_base_fisher: Sigma = sigma^2 I_K gives sigma^{-2}(dmu^T dmu + 2K dsigma^2).
2. 18:46 'identifiability makes the vertical Fisher metric positive definite': N(theta^3, 1) is
   identifiable with Fisher 9 theta^4, zero at theta = 0.
3. 19 eq:lor-message-aggregation (Cauchy-Binet for 2x2), numerically.
4. 19:165 FRW: gamma = -dt^2 + a(t)^2 dx^2 has R = 6 (a''/a + (a'/a)^2).
5. 20 eq:collective_correlation_geometry: (1 + rho^2)/(1 - rho^2)^2.
"""
import numpy as np
import sympy as sp

K = sp.Symbol('K', positive=True, integer=True)
sig, dsig = sp.symbols('sigma dsigma', positive=True)
# 1: (1/2) tr(Sigma^-1 dSigma Sigma^-1 dSigma) with dSigma = 2 sigma dsigma I_K
cov_term = sp.Rational(1, 2) * K * (2 * sig * dsig / sig ** 2) ** 2
print('1. covariance block minus 2K dsigma^2/sigma^2:', sp.simplify(cov_term - 2 * K * dsig ** 2 / sig ** 2))
th = sp.Symbol('theta', real=True)
print('2. Fisher of N(theta^3,1):', sp.diff(th ** 3, th) ** 2, ' at 0:', (sp.diff(th ** 3, th) ** 2).subs(th, 0))
rng = np.random.default_rng(0)
zs = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)); ws = rng.random(4)
M = sum(ws[r] * np.outer(zs[r], zs[r].conj()) for r in range(4))
cb = sum(ws[r] * ws[s] * abs(zs[r, 0] * zs[s, 1] - zs[r, 1] * zs[s, 0]) ** 2 for r in range(4) for s in range(r + 1, 4))
print('3. det M - Cauchy-Binet sum:', abs(np.linalg.det(M) - cb))
t = sp.Symbol('t'); X = sp.symbols('x1 x2 x3'); a = sp.Function('a')(t)
co = [t, *X]
g = sp.diag(-1, a ** 2, a ** 2, a ** 2); gi = g.inv()
Gam = [[[sum(gi[i, l] * (sp.diff(g[l, j], co[k]) + sp.diff(g[l, k], co[j]) - sp.diff(g[j, k], co[l])) for l in range(4)) / 2
         for k in range(4)] for j in range(4)] for i in range(4)]
def Riem(r, s_, m, n):
    e = sp.diff(Gam[r][n][s_], co[m]) - sp.diff(Gam[r][m][s_], co[n])
    e += sum(Gam[r][m][l] * Gam[l][n][s_] - Gam[r][n][l] * Gam[l][m][s_] for l in range(4))
    return e
Ric = sp.Matrix(4, 4, lambda s_, n: sum(Riem(r, s_, r, n) for r in range(4)))
R = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(4) for j in range(4)))
print('4. FRW scalar curvature minus 6(a\'\'/a + (a\'/a)^2):', sp.simplify(R - 6 * (sp.diff(a, t, 2) / a + (sp.diff(a, t) / a) ** 2)))
rho = sp.Symbol('rho', real=True)
S = sp.Matrix([[1, rho], [rho, 1]]); A_ = S.inv() * sp.diff(S, rho)
print('5. correlation Fisher minus (1+rho^2)/(1-rho^2)^2:', sp.simplify((A_ * A_).trace() / 2 - (1 + rho ** 2) / (1 - rho ** 2) ** 2))
