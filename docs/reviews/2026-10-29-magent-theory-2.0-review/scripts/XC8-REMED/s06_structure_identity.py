r"""Symbolic check of geo:structure-identity and geo:basic-criterion (02_geometry.tex:155-171).

Family: N(mu, e^{2s}) in chart p=(mu,s); group Aff(1) acting by mu -> a mu + b, s -> s + log a (a>0).
Lie algebra element xi=(alpha,beta) (matrix [[alpha,beta],[0,0]]), fundamental map J_p xi=(alpha mu+beta, alpha),
bracket [xi1,xi2]=(0, alpha1 beta2 - alpha2 beta1). Fisher metric g = diag(e^{-2s}, 2), invariant.
Connection A = sum_k A_k dc^k with A_k=(alpha_k(c), beta_k(c)), F = dA + [A^A].
Identity (geo:structure-identity), for vector fields X,Y on the base:
  V([X,Y]) = (X(VY) + d_pj(VY, A(X))) - (Y(VX) + d_pj(VX, A(Y))) - J_p F(X,Y),
with V = dp + J_p A and d_pj(v, xi) the p-derivative of J_p xi in direction v.
Criterion (geo:basic-criterion): for Z with VZ = 0,
  (L_Z h)(X,Y) = g(J_p F(Z,X), VY) + g(VX, J_p F(Z,Y)),  h(X,Y) = g(VX, VY).
"""
import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
c = (x, y, z)
mu = sp.Function("mu")(*c)
s = sp.Function("s")(*c)
p = sp.Matrix([mu, s])


def J(pv, xi):
    al, be = xi
    return sp.Matrix([al * pv[0] + be, al])


def dpj(v, xi):
    al, be = xi
    return sp.Matrix([al * v[0], 0])


def bracket(x1, x2):
    return (0, x1[0] * x2[1] - x2[0] * x1[1])


def vec_apply(Xc, f):
    """Apply vector field with components Xc (in coordinates c) to scalar or matrix f."""
    if isinstance(f, sp.MatrixBase):
        return f.applyfunc(lambda e: sum(Xc[k] * sp.diff(e, c[k]) for k in range(3)))
    return sum(Xc[k] * sp.diff(f, c[k]) for k in range(3))


def lie_bracket(Xc, Yc):
    return [vec_apply(Xc, Yc[k]) - vec_apply(Yc, Xc[k]) for k in range(3)]


# general connection components
al = [sp.Function("al%d" % k)(*c) for k in range(3)]
be = [sp.Function("be%d" % k)(*c) for k in range(3)]


def A_of(Xc, al=al, be=be):
    return (sum(Xc[k] * al[k] for k in range(3)), sum(Xc[k] * be[k] for k in range(3)))


def dp(Xc):
    return sp.Matrix([vec_apply(Xc, mu), vec_apply(Xc, s)])


def V(Xc, al=al, be=be):
    return dp(Xc) + J(p, A_of(Xc, al, be))


def F_of(Xc, Yc, al=al, be=be):
    # dA(X,Y) = X(A(Y)) - Y(A(X)) - A([X,Y]) ; F = dA + [A(X),A(Y)]
    AX, AY = A_of(Xc, al, be), A_of(Yc, al, be)
    AXY = A_of(lie_bracket(Xc, Yc), al, be)
    dA = tuple(vec_apply(Xc, AY[i]) - vec_apply(Yc, AX[i]) - AXY[i] for i in range(2))
    br = bracket(AX, AY)
    return (dA[0] + br[0], dA[1] + br[1])


f = sp.Function("f")(*c)
g2 = sp.Function("g")(*c)
X = [1, 0, f]          # non-coordinate fields with a nonzero bracket
Y = [g2, 1, 0]

lhs = V(lie_bracket(X, Y))
rhs = (vec_apply(X, V(Y)) + dpj(V(Y), A_of(X))) - (vec_apply(Y, V(X)) + dpj(V(X), A_of(Y))) - J(p, F_of(X, Y))
res1 = sp.simplify(lhs - rhs)
print("structure identity residual:", list(res1))

# basic criterion: make Z = d/dz radical by choosing A(d/dz) = -J_p^{-1} dp(d/dz)
Zc = [0, 0, 1]
dpz = dp(Zc)
alz = dpz[1] * (-1)                 # second comp of J: alpha = -ds/dz
bez = -(dpz[0]) - alz * mu          # first comp: alpha mu + beta = -dmu/dz
al_r = [al[0], al[1], alz]
be_r = [be[0], be[1], bez]
VZ = sp.simplify(V(Zc, al_r, be_r))
print("V(Z) =", list(VZ))
gmat = sp.diag(sp.exp(-2 * s), 2)


def gF(u, v):
    return (u.T * gmat * v)[0, 0]


Xc = [1, 0, 0]
Yc = [0, 1, 0]
h = gF(V(Xc, al_r, be_r), V(Yc, al_r, be_r))
LZh = vec_apply(Zc, h) - gF(V(lie_bracket(Zc, Xc), al_r, be_r), V(Yc, al_r, be_r)) - gF(V(Xc, al_r, be_r), V(lie_bracket(Zc, Yc), al_r, be_r))
crit = gF(J(p, F_of(Zc, Xc, al_r, be_r)), V(Yc, al_r, be_r)) + gF(V(Xc, al_r, be_r), J(p, F_of(Zc, Yc, al_r, be_r)))
res2 = sp.simplify(LZh - crit)
print("basic criterion residual:", res2)
# also the diagonal entry X=Y
hxx = gF(V(Xc, al_r, be_r), V(Xc, al_r, be_r))
crit_xx = 2 * gF(J(p, F_of(Zc, Xc, al_r, be_r)), V(Xc, al_r, be_r))
res3 = sp.simplify(vec_apply(Zc, hxx) - crit_xx)
print("basic criterion residual (X,X):", res3)
with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
    fh.write("structure identity residual: %s\nV(Z) = %s\nbasic criterion residual: %s\nbasic criterion residual (X,X): %s\n" % (
        list(res1), list(VZ), res2, res3))
