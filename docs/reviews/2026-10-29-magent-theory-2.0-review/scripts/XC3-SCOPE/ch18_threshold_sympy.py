"""XC3-SCOPE sympy check of it2:coupled-threshold: stationarity of x_* and T_c = g^T H^{-1} g / 4,
T_mean with variances frozen.  Exact rational arithmetic."""
import sympy as sp
mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2 = sp.symbols('mu0 mu1 mu2 a1 a2 v0 v1 v2 V1 V2', positive=True)
x = [mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2]
def kl1(m, v, m0, v0_):   # KL(N(m,v)||N(m0,v0_)) in one dimension
    return sp.Rational(1, 2)*(v/v0_ + (m-m0)**2/v0_ - 1 + sp.log(v0_/v))
Pp = [1, sp.Rational(5, 4), sp.Rational(5, 4)]; mm = [0, -sp.Rational(23, 15), sp.Rational(23, 15)]
F0 = kl1(mu0, v0, mm[0], 1/sp.Integer(Pp[0])) + kl1(mu1, v1, mm[1], 1/Pp[1]) + kl1(mu2, v2, mm[2], 1/Pp[2]) \
     + kl1(a1, V1, 0, sp.Rational(4, 3)) + kl1(a2, V2, 0, sp.Rational(4, 3))
S0, S1, S2 = v0+1, v1+1, v2+1
d1, d2 = mu0-mu1-a1, mu0-mu2-a2
C1 = sp.Rational(1, 2)*((S0+d1**2+V1)/S1 - 1 + sp.log(S1) - sp.log(S0))
C2 = sp.Rational(1, 2)*((S0+d2**2+V2)/S2 - 1 + sp.log(S2) - sp.log(S0))
xs = {mu0: 0, mu1: -sp.Rational(4, 3), mu2: sp.Rational(4, 3), a1: sp.Rational(1, 3), a2: -sp.Rational(1, 3),
      v0: 1, v1: 1, v2: 1, V1: 1, V2: 1}
A = F0 + (C1+C2)/2; Dl = C1 - C2
grad = [sp.simplify(sp.diff(A, xi).subs(xs)) for xi in x]
print("matter gradient at x_* (should be zero):", grad, " C1-C2 =", sp.simplify(Dl.subs(xs)))
H = sp.Matrix(10, 10, lambda i, j: sp.nsimplify(sp.simplify(sp.diff(A, x[i], x[j]).subs(xs))))
g = sp.Matrix([sp.nsimplify(sp.simplify(sp.diff(Dl, xi).subs(xs))) for xi in x])
print("g =", list(g))
print("H positive definite:", all(ev > 0 for ev in H.eigenvals().keys()))
Hig = H.LUsolve(g); q = sp.nsimplify((g.T*Hig)[0])
print("g^T H^-1 g =", q, " T_c =", q/4, " (manuscript 3269/10118):", sp.simplify(q/4 - sp.Rational(3269, 10118)) == 0)
print("H^-1 g * 5059 =", list(sp.simplify(Hig*5059)))
Hm = H[:5, :5]; gm = g[:5, :]; qm = sp.nsimplify((gm.T*Hm.LUsolve(gm))[0])
print("frozen-variance T_mean =", qm/4, " (manuscript 31/122):", sp.simplify(qm/4 - sp.Rational(31, 122)) == 0)
