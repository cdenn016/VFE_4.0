r"""Symbolic check of the vertical-differential structure identities used in 02_geometry.

Aff(1) acts on N(mu, sigma^2) by (a, b) . (mu, sigma) = (e^a mu + b, e^a sigma) (left action).
Fisher metric g = (dmu^2 + 2 dsigma^2)/sigma^2.  Base R^3 with coordinates (x, y, z).
Checks, for D = dp + J_p A and F = dA + [A ^ A]:
  (1) D[X,Y] = X(DY) + dj(DY, A X) - Y(DX) - dj(DX, A Y) - J_p F(X,Y),
  (2) (L_Z h)(X,Y) = g(J F(Z,X), DY) + g(DX, J F(Z,Y)) + g(dX-terms) when DZ = 0 is *not* imposed:
      we check the general formula
      (L_Z h)(X,Y) = g(Z(DX)+dj(DX,AZ) - D[Z,X] , DY) + ... via the reduced identity
      (L_Z h)(X,Y) = g(X(DZ)+dj(DZ,AX) + J F(Z,X), DY) + g(DX, Y(DZ)+dj(DZ,AY) + J F(Z,Y)).
"""
import sympy as sp

x, y, z = sp.symbols('x y z', real=True)
c = (x, y, z)
mu_s, sg_s = sp.symbols('mu sigma', positive=True)

# Lie algebra basis e_a (scale), e_b (translation); [e_a, e_b] = e_b.
def J(p, xi):
    mu, sg = p
    a, b = xi
    return sp.Matrix([a * mu + b, a * sg])

def bracket(xi, eta):
    a1, b1 = xi
    a2, b2 = eta
    return (0, a1 * b2 - a2 * b1)

def g(p, u, v):
    mu, sg = p
    return (u[0] * v[0] + 2 * u[1] * v[1]) / sg**2

R = sp.Rational
p = (R(1, 3) + x * y - z**2 / 2 + R(2, 5) * x * z, 2 + x**2 / 3 + y * z / 4 + R(1, 7) * y)
A = [  # A_mu components: A = sum_k A[k] dc^k, each a pair (a, b)
    (R(1, 2) * y + x * z / 3, z - R(1, 4) * x**2),
    (x - R(2, 3) * z * y, R(1, 5) + y**2 / 2),
    (R(1, 3) * x * y, x - y + z / 2),
]
def Aof(X):
    return tuple(sum(A[k][m] * X[k] for k in range(3)) for m in range(2))

def vec(f):
    return [f[0], f[1], f[2]]

X = (1 + y, x * z, R(1, 2))
Y = (z, 1 - x / 2, x * y)
Z = (y * z, R(1, 3), 1 + x)

def apply(V, f):
    return sum(V[k] * sp.diff(f, c[k]) for k in range(3))

def lie(V, W):
    return tuple(apply(V, W[k]) - apply(W, V[k]) for k in range(3))

def dp(V):
    return sp.Matrix([apply(V, p[0]), apply(V, p[1])])

def D(V):
    return dp(V) + J(p, Aof(V))

def dj(u, xi):
    # derivative of j(p, xi) in p along u (J is linear in p here)
    a, b = xi
    return sp.Matrix([a * u[0], a * u[1]])

def F(V, W):
    # dA(V,W) + [A(V), A(W)]
    dA = tuple(apply(V, Aof(W)[m]) - apply(W, Aof(V)[m]) - Aof(lie(V, W))[m] for m in range(2))
    br = bracket(Aof(V), Aof(W))
    return (dA[0] + br[0], dA[1] + br[1])

def Vder(V, W):
    """Componentwise derivative of D(W) along V plus dj(D W, A V)."""
    DW = D(W)
    return sp.Matrix([apply(V, DW[0]), apply(V, DW[1])]) + dj(DW, Aof(V))

lhs = D(lie(X, Y))
rhs = Vder(X, Y) - Vder(Y, X) - J(p, F(X, Y))
res1 = sp.simplify(lhs - rhs)
print('structure identity residual:', list(res1))

def h(V, W):
    return g(p, D(V), D(W))

LZh = apply(Z, h(X, Y)) - h(lie(Z, X), Y) - h(X, lie(Z, Y))
DZ = D(Z)
def Xder_DZ(V):
    return sp.Matrix([apply(V, DZ[0]), apply(V, DZ[1])]) + dj(DZ, Aof(V))
u_, v_ = D(X), D(Y)
dgDZ = (sp.diff(g((mu_s, sg_s), u_, v_), mu_s) * DZ[0] + sp.diff(g((mu_s, sg_s), u_, v_), sg_s) * DZ[1]).subs({mu_s: p[0], sg_s: p[1]})
rhs2 = dgDZ + g(p, Xder_DZ(X) + J(p, F(Z, X)), D(Y)) + g(p, D(X), Xder_DZ(Y) + J(p, F(Z, Y)))
pt = {x: R(1, 3), y: R(-1, 2), z: R(2, 5)}
print('Lie-derivative identity residual at rational point:', sp.nsimplify(sp.simplify((LZh - rhs2).subs(pt))))
print('values: D[X,Y] =', [sp.N(v) for v in lhs.subs(pt)], ' L_Z h(X,Y) =', sp.N(LZh.subs(pt)))
