r"""GAP08-S1 script 2a: supplements/s03_variation.tex, exact checks, parts (a) to (d).

(a) prop:sup-residual-cost at a non-Gaussian seven-point joint law of (X_i, X_j) in R^2 x R^3,
    exact rational arithmetic (sympy Rational); the constant c_R is common and omitted.
(b) visible cross covariance: a dependent, uncorrelated pair; a dependent pair whose dependence
    is invisible through B; two couplings with equal marginals and different costs.
(c) prop:sup-row-equivalence, eq:sup-row-constant, eq:sup-covariance-difference symbolically
    in d = 2 (generic symmetric V, W; symbols for the log-determinants where they enter).
(d) T = 1 entropy identity; self-comparison; independent snapshot cost d/T and odds e^{d/T}.
Part (e) is in s2b_s03_jensen.py.
"""
import random
import sympy as sp
import mpmath as mp
from sympy import Rational as Q

random.seed(7)


def rq(lo=-5, hi=5, den=3):
    return Q(random.randint(lo, hi), den)


print("== (a) prop:sup-residual-cost, non-Gaussian 7-point joint, exact ==")
npts = 7
raw = [random.randint(1, 9) for _ in range(npts)]
w = [Q(r, sum(raw)) for r in raw]
Xi = [sp.Matrix([rq(), rq()]) for _ in range(npts)]
Xj = [sp.Matrix([rq(), rq(), rq()]) for _ in range(npts)]
A = sp.Matrix(2, 2, lambda i, j: rq())
B = sp.Matrix(2, 3, lambda i, j: rq())
M = sp.Matrix(2, 2, lambda i, j: rq())
R = M * M.T + sp.eye(2) * Q(1, 2)
Ri = R.inv()
o = sp.Matrix([rq(), rq()])
mu_i = sum((w[k] * Xi[k] for k in range(npts)), sp.zeros(2, 1))
mu_j = sum((w[k] * Xj[k] for k in range(npts)), sp.zeros(3, 1))
S_ii = sum((w[k] * (Xi[k] - mu_i) * (Xi[k] - mu_i).T for k in range(npts)), sp.zeros(2, 2))
S_jj = sum((w[k] * (Xj[k] - mu_j) * (Xj[k] - mu_j).T for k in range(npts)), sp.zeros(3, 3))
S_ij = sum((w[k] * (Xi[k] - mu_i) * (Xj[k] - mu_j).T for k in range(npts)), sp.zeros(2, 3))
lhs = sum(w[k] * Q(1, 2) * ((A * Xi[k] - B * Xj[k] - o).T * Ri * (A * Xi[k] - B * Xj[k] - o))[0, 0]
          for k in range(npts))
delta = A * mu_i - B * mu_j - o
V_ij = A * S_ii * A.T + B * S_jj * B.T - A * S_ij * B.T - B * S_ij.T * A.T
rhs = Q(1, 2) * (delta.T * Ri * delta)[0, 0] + Q(1, 2) * (Ri * V_ij).trace()
print("weights:", w)
print("exact residual E[quadratic] - (delta^T R^-1 delta + Tr R^-1 V_ij)/2 =", sp.simplify(lhs - rhs))
print("third central moment of first coordinate of A X_i - B X_j (nonzero => non-Gaussian):",
      sum(w[k] * ((A * Xi[k] - B * Xj[k])[0] - (A * mu_i - B * mu_j)[0]) ** 3 for k in range(npts)))

print("\n== (b) visible cross covariance ==")
# Y1 uniform on {-1,0,1}; Z1 = Y1^2 (dependent, uncorrelated); d = 1, A = [1], B = [1]
pts = [(-1, 1), (0, 0), (1, 1)]
pw = [Q(1, 3)] * 3


def cost_quadratic(points, weights, Am, Bm, Rm, om):
    """E[(A x_i - B x_j - o)^T R^{-1} (...)]/2 for a finite joint law."""
    Rinv = Rm.inv()
    tot = 0
    for (xi, xj), ww in zip(points, weights):
        r = Am * xi - Bm * xj - om
        tot += ww * Q(1, 2) * (r.T * Rinv * r)[0, 0]
    return sp.nsimplify(tot)


Am = sp.Matrix([[1]])
Bm = sp.Matrix([[1]])
Rm = sp.Matrix([[Q(2)]])
om = sp.Matrix([[Q(1, 4)]])
joint = [(sp.Matrix([z]), sp.Matrix([y])) for (y, z) in pts]
prod = [(sp.Matrix([z]), sp.Matrix([y])) for (y, _) in pts for (_, z) in pts]
prodw = [Q(1, 9)] * 9
cj = cost_quadratic(joint, pw, Am, Bm, Rm, om)
cp = cost_quadratic(prod, prodw, Am, Bm, Rm, om)
print("dependent, uncorrelated (Z1 = Y1^2): joint cost", cj, "; product-of-marginals cost", cp,
      "; equal:", sp.simplify(cj - cp) == 0)
# dependence through a coordinate that B discards: X_j = (Y1, Y2) with Y2 = Z1, B reads Y1 only
pts2 = [(sp.Matrix([z]), sp.Matrix([y, z])) for (y, z) in [(-1, 1), (1, 1), (0, -2)]]
w2 = [Q(1, 3)] * 3
B2 = sp.Matrix([[1, 0]])
c2j = cost_quadratic(pts2, w2, Am, B2, Rm, om)
marg_i = [(sp.Matrix([z]), Q(1, 3)) for z in [1, 1, -2]]
marg_j = [(sp.Matrix([y, z]), Q(1, 3)) for (y, z) in [(-1, 1), (1, 1), (0, -2)]]
prod2 = [(a[0], b[0]) for a in marg_i for b in marg_j]
prod2w = [a[1] * b[1] for a in marg_i for b in marg_j]
c2p = cost_quadratic(prod2, prod2w, Am, B2, Rm, om)
Sij2 = sum((ww * (xi - sp.Matrix([0])) * xj.T for (xi, xj), ww in zip(pts2, w2)), sp.zeros(1, 2))
print("Cov(Z1, (Y1, Y2)) =", Sij2, "(Y2 = Z1 is fully dependent); A Cov B^T =", (Am * Sij2 * B2.T)[0, 0],
      "; joint cost", c2j, "= product cost", c2p, ":", sp.simplify(c2j - c2p) == 0)
# same marginals, different coupling: Z1 = Y1 versus Z1 = -Y1 (both marginals uniform on {-1,0,1})
cA = cost_quadratic([(sp.Matrix([y]), sp.Matrix([y])) for y in (-1, 0, 1)], [Q(1, 3)] * 3, Am, Bm, Rm, om)
cB = cost_quadratic([(sp.Matrix([-y]), sp.Matrix([y])) for y in (-1, 0, 1)], [Q(1, 3)] * 3, Am, Bm, Rm, om)
print("equal marginals, couplings Z1 = Y1 and Z1 = -Y1: costs", cA, "and", cB)

print("\n== (c) prop:sup-row-equivalence, d = 2 symbolic ==")
v1, v2, v3, w1, w2_, w3, a1, a2, b1, b2 = sp.symbols('v1 v2 v3 w1 w2 w3 a1 a2 b1 b2', real=True)
T = sp.Symbol('T', positive=True)
logdetV, logdetW = sp.symbols('logdetV logdetW', real=True)
d = 2
V = sp.Matrix([[v1, v2], [v2, v3]])
W = sp.Matrix([[w1, w2_], [w2_, w3]])
a = sp.Matrix([a1, a2])
b = sp.Matrix([b1, b2])
Wi = W.inv()
c_TW = Q(1, 2) * (d * sp.log(2 * sp.pi * T) + logdetW)  # (1/2) log det(2 pi T W)
C = c_TW + Q(1, 2) * (((a - b).T * (Wi / T) * (a - b))[0, 0] + ((Wi / T) * (V + W)).trace())
D = Q(1, 2) * (((a - b).T * Wi * (a - b))[0, 0] + (Wi * V).trace() - d + (logdetW - logdetV))
rowconst = c_TW + Q(d) / T - (logdetW - logdetV) / (2 * T)
print("C - D/T - [c_TW + d/T - log(detW/detV)/(2T)] =", sp.simplify(C - D / T - rowconst))
print("contains a or b_j:", any(s in rowconst.free_symbols for s in (a1, a2, b1, b2)))
# covariance derivative: replace logdetV by log det V explicitly and differentiate in v1, v2, v3
detV = V.det()
Cv = C
Dv = D.subs(logdetV, sp.log(detV))
Vi = V.inv()
ok = True
for sym, dV in [(v1, sp.Matrix([[1, 0], [0, 0]])), (v2, sp.Matrix([[0, 1], [1, 0]])), (v3, sp.Matrix([[0, 0], [0, 1]]))]:
    diff = sp.simplify(sp.diff(Cv, sym) - sp.diff(Dv / T, sym) - (Vi * dV).trace() / (2 * T))
    print(f"d/d{sym}: d_V C - d_V(D/T) - Tr(V^-1 dV)/(2T) =", diff)
    ok = ok and diff == 0
# receiver-mean and source-mean derivatives agree
for sym in (a1, a2, b1, b2):
    print(f"d/d{sym} of (C - D/T):", sp.simplify(sp.diff(C - D / T, sym)))

print("\n== (d) T = 1 entropy identity, self-comparison, independent snapshots ==")
H_V = Q(1, 2) * (d * sp.log(2 * sp.pi * sp.E) + logdetV)  # entropy of N(a, V) in d = 2
print("at T = 1: (C - D) - (H(N(a,V)) + d/2) =",
      sp.simplify(sp.expand_log((C - D).subs(T, 1) - (H_V + Q(d, 2)), force=True)))
# self-comparison: A = B = I, same variable: V = S + S - S - S = 0, delta = -o
S = sp.Matrix([[v1, v2], [v2, v3]])
Vself = S + S - S - S.T
print("self comparison V_ii =", Vself.tolist())
# independent snapshots X, X~ iid N(0, I_d), R = T I_d: (1/2) Tr(R^-1 (I + I)) = d/T
dd = sp.Symbol('d', positive=True, integer=True)
print("independent snapshot excess cost (1/2) Tr((T I)^-1 2 I) =", sp.simplify(Q(1, 2) * 2 * dd / T),
      "; odds exp(cost_indep - cost_self) =", sp.exp(dd / T))

