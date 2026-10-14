r"""L9 check S3: projected (quotient) connections of def:rg-cover-minimum-norm-splitting and
thm:rg-cover-quotient-composition (b04).  A^quot = F (dF^dagger + A F^dagger), Curv = dA + A^A.
Recompute: (i) the projected-curvature example at 0 (= -J dc1^dc2), (ii) the 3x4 / 2x3 composition
example: curvature-coefficient determinants at (1/3,1/5) for propagated vs reset intermediate metric,
claimed 15625/61731 and 3645/103823; (iii) direct = staged connection identically.
"""
import sympy as sp

c1, c2 = sp.symbols('c1 c2', real=True)
X = [c1, c2]

def pinv_metric(F, G=None):
    # minimum-norm lift with respect to fiber metric G on the source (G=None means identity)
    if G is None:
        return F.T * (F * F.T).inv()
    Gi = G.inv()
    return Gi * F.T * (F * Gi * F.T).inv()

def quot_conn(F, A, Fd):
    # A: list of matrices (components along dc1, dc2); returns components of A^quot
    return [sp.simplify(F * (sp.diff(Fd, x) + A[k] * Fd)) for k, x in enumerate(X)]

def curv12(A):
    # coefficient of dc1^dc2 in dA + A^A
    return sp.simplify(sp.diff(A[1], c1) - sp.diff(A[0], c2) + A[0] * A[1] - A[1] * A[0])

# (i)
F = sp.Matrix([[1, 0, -c1], [0, 1, -c2]])
A0 = [sp.zeros(3, 3), sp.zeros(3, 3)]
Aq = quot_conn(F, A0, pinv_metric(F))
R = curv12(Aq).subs({c1: 0, c2: 0})
print("(i) Curv(A^quot)(0) coefficient of dc1^dc2 =", R.tolist(), " expected -J =", (-sp.Matrix([[0, -1], [1, 0]])).tolist())

# (ii)
F1 = sp.Matrix([[2, 0, 0, 1], [0, 1, 0, 0], [0, 0, 3, 0]])
F2 = sp.Matrix([[1, c1, 0], [0, 1, c2]])
Z4 = [sp.zeros(4, 4), sp.zeros(4, 4)]
F1d = pinv_metric(F1)
A1 = quot_conn(F1, Z4, F1d)               # F1 constant -> A1 = 0
g1 = (F1 * F1.T).inv()                     # induced quotient metric on E1
F2d_prop = pinv_metric(F2, g1)
A2_prop = quot_conn(F2, A1, F2d_prop)
F2d_reset = pinv_metric(F2)
A2_reset = quot_conn(F2, [sp.zeros(3, 3)] * 2, F2d_reset)
F21 = F2 * F1
A2_direct = quot_conn(F21, Z4, pinv_metric(F21))
pt = {c1: sp.Rational(1, 3), c2: sp.Rational(1, 5)}
dprop = sp.nsimplify(sp.simplify(curv12(A2_prop).subs(pt).det()))
dreset = sp.nsimplify(sp.simplify(curv12(A2_reset).subs(pt).det()))
ddirect = sp.nsimplify(sp.simplify(curv12(A2_direct).subs(pt).det()))
print("(ii) det curvature coeff: propagated =", dprop, " direct =", ddirect, " reset =", dreset)
print("     claimed: propagated 15625/61731, reset 3645/103823")
# (iii) staged-with-propagated-metric equals direct connection identically
diff = [sp.simplify(A2_prop[k] - A2_direct[k]) for k in range(2)]
print("(iii) staged(propagated) - direct connection components:", [d.tolist() for d in diff])
# lift composition identity
print("     F1^dagger F2^dagger_1 - (F2F1)^dagger =", sp.simplify(F1d * F2d_prop - pinv_metric(F21)).tolist())
