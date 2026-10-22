r"""L02-GEOM s02: covariant pullback identities (geo:local-derivative, geo:gauge-connection,
geo:connection-change, geo:transported-kl, geo:quotient) on a nonabelian example.

Model: N = {N(mu, sigma^2)} in coordinates p=(mu, s=log sigma), Fisher g = diag(e^{-2s}, 2).
G = Aff(1)^+ = {[[a,b],[0,1]], a>0}, acting by pushforward x -> a x + b:
   (mu, s) -> (a mu + b, s + log a).
Lie algebra basis delta=[[1,0],[0,0]], tau=[[0,1],[0,0]];  J_p(xi) = (xi_d mu + xi_t, xi_d).
Convention (chapter 3): D^A p(X) = dp(X) + J_p A(X),  A2 = g A g^{-1} - dg g^{-1},
curvature F = dA + [A ^ A],  F(X,Y) = X A(Y) - Y A(X) - A([X,Y]) + [A X, A Y].

Checks:
  (a) passive covariance D^{A2} p2 = T(g.) D^A p and h^{A2}_{p2} = h^A_p   (geo:passive)
  (b) connection change expansion (geo:connection-change)
  (c) NEW structure identity (proposed improvement for geo:quotient):
        V([X,Y]) = X(V Y) - Y(V X) - dj(V X, A Y) + dj(V Y, A X) - J_p F(X,Y),  V = D^A p
      so for X,Y in ker V:  V([X,Y]) = -J_p F(X,Y).
  (d) NEW Lie-derivative identity: with nabla^A_X W = X(W) + Gamma(V X, W) + dj(W, A X),
        (L_Z h)(X,Y) = g(nabla^A_X VZ + J F(Z,X), VY) + g(VX, nabla^A_Y VZ + J F(Z,Y)),
      so for Z in ker V: (L_Z h)(X,Y) = g(J F(Z,X), VY) + g(VX, J F(Z,Y)).
  (e) the contact counterexample (geo:quotient text) satisfies (c) with F = -dx^dy.
  (f) geo:transported-kl: KL(zeta(c) || transported zeta(c+tX)) / (t^2/2) -> h(X,X).
All random coefficients are rational; evaluation at a random rational point is exact.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
x, y, z = sp.symbols('x y z', real=True)
C = [x, y, z]

def rpoly(deg=2):
    terms = [sp.Integer(1), x, y, z, x * y, y * z, x * z, x ** 2, y ** 2, z ** 2]
    if deg == 1:
        terms = terms[:4]
    return sum(sp.Rational(int(rng.integers(-5, 6)), int(rng.integers(1, 4))) * t for t in terms)

mu, s = rpoly(), rpoly()
p = sp.Matrix([mu, s])
Ad = [rpoly() for _ in C]
At = [rpoly() for _ in C]

def g_metric(pv):
    return sp.diag(sp.exp(-2 * pv[1]), 2)

def j(pv, xi_d, xi_t):
    return sp.Matrix([xi_d * pv[0] + xi_t, xi_d])

def dj(pv, W, xi_d, xi_t):
    return sp.Matrix([xi_d * W[0], 0])

def apply(Xv, f):
    if isinstance(f, sp.MatrixBase):
        return f.applyfunc(lambda e: sum(Xv[k] * sp.diff(e, C[k]) for k in range(3)))
    return sum(Xv[k] * sp.diff(f, C[k]) for k in range(3))

def bracket(Xv, Yv):
    return sp.Matrix([apply(Xv, Yv[k]) - apply(Yv, Xv[k]) for k in range(3)])

def Aof(comp, Xv):
    return sum(comp[k] * Xv[k] for k in range(3))

def V(Xv, pv=None, ad=None, at=None):
    pv = p if pv is None else pv
    ad = Ad if ad is None else ad
    at = At if at is None else at
    dp = sp.Matrix([apply(Xv, pv[0]), apply(Xv, pv[1])])
    return dp + j(pv, Aof(ad, Xv), Aof(at, Xv))

def F(Xv, Yv):
    br = bracket(Xv, Yv)
    Fd = apply(Xv, Aof(Ad, Yv)) - apply(Yv, Aof(Ad, Xv)) - Aof(Ad, br)
    Ft = apply(Xv, Aof(At, Yv)) - apply(Yv, Aof(At, Xv)) - Aof(At, br)
    Ft += Aof(Ad, Xv) * Aof(At, Yv) - Aof(Ad, Yv) * Aof(At, Xv)
    return Fd, Ft

pt = {x: sp.Rational(1, 3), y: sp.Rational(-2, 5), z: sp.Rational(3, 7)}
def ev(e):
    return e.subs(pt)

X = sp.Matrix([rpoly(1), rpoly(1), rpoly(1)])
Y = sp.Matrix([rpoly(1), rpoly(1), rpoly(1)])
Z = sp.Matrix([rpoly(1), rpoly(1), rpoly(1)])

# (a) passive covariance
a = sp.exp(rpoly(1)); b = rpoly(1)
gmat = sp.Matrix([[a, b], [0, 1]])
ginv = gmat.inv()
def Amat(Xv):
    return sp.Matrix([[Aof(Ad, Xv), Aof(At, Xv)], [0, 0]])
pprime = sp.Matrix([a * mu + b, s + sp.log(a)])
ad2, at2 = [], []
for k in range(3):
    e = sp.Matrix([1 if i == k else 0 for i in range(3)])
    M = gmat * Amat(e) * ginv - apply(e, gmat) * ginv
    assert sp.simplify(M[1, 0]) == 0 and sp.simplify(M[1, 1]) == 0
    ad2.append(M[0, 0]); at2.append(M[0, 1])
lhs = V(X, pprime, ad2, at2)
rhs = sp.diag(a, 1) * V(X)
res_a = [sp.N(ev(sp.simplify(e)), 30) for e in (lhs - rhs)]
hA = (V(X).T * g_metric(p) * V(Y))[0]
hA2 = (V(X, pprime, ad2, at2).T * g_metric(pprime) * V(Y, pprime, ad2, at2))[0]
print('(a) passive covariance residual at point:', res_a, '; h residual:',
      sp.N(ev(hA - hA2), 30), '; h value:', sp.N(ev(hA), 8))

# (b) connection change
al_d = [rpoly() for _ in C]; al_t = [rpoly() for _ in C]
Ad_new = [Ad[k] + al_d[k] for k in range(3)]; At_new = [At[k] + al_t[k] for k in range(3)]
bX = j(p, Aof(al_d, X), Aof(al_t, X)); bY = j(p, Aof(al_d, Y), Aof(al_t, Y))
uX, uY = V(X), V(Y)
G = g_metric(p)
hnew = (V(X, p, Ad_new, At_new).T * G * V(Y, p, Ad_new, At_new))[0]
hexp = (uX.T * G * uY + uX.T * G * bY + bX.T * G * uY + bX.T * G * bY)[0]
print('(b) connection-change residual:', sp.simplify(ev(hnew - hexp)))

# (c) structure identity
VX, VY = V(X), V(Y)
Fd, Ft = F(X, Y)
lhs_c = V(bracket(X, Y))
rhs_c = (apply(X, VY) - apply(Y, VX) - dj(p, VX, Aof(Ad, Y), Aof(At, Y))
         + dj(p, VY, Aof(Ad, X), Aof(At, X)) - j(p, Fd, Ft))
print('(c) structure identity residual:', [sp.simplify(ev(e)) for e in (lhs_c - rhs_c)],
      '; V([X,Y]) at point =', [sp.N(ev(e), 6) for e in lhs_c])

# (d) Lie derivative identity
mus, ss = sp.symbols('mu_ s_')
gN = sp.diag(sp.exp(-2 * ss), 2); gNi = gN.inv(); qv = [mus, ss]
Gam = [[[sum(gNi[i, l] * (sp.diff(gN[l, jj], qv[k]) + sp.diff(gN[l, k], qv[jj])
                          - sp.diff(gN[jj, k], qv[l])) / 2 for l in range(2))
         for k in range(2)] for jj in range(2)] for i in range(2)]
def Gamma(U, W):
    out = sp.zeros(2, 1)
    for i in range(2):
        out[i] = sum(Gam[i][jj][k] * U[jj] * W[k] for jj in range(2) for k in range(2))
    return out.subs({mus: p[0], ss: p[1]})
def nablaA(Xv, W):
    return apply(Xv, W) + Gamma(V(Xv), W) + dj(p, W, Aof(Ad, Xv), Aof(At, Xv))
def h(Xv, Yv):
    return (V(Xv).T * G * V(Yv))[0]
LZh = apply(Z, h(X, Y)) - h(bracket(Z, X), Y) - h(X, bracket(Z, Y))
FZX = F(Z, X)
FZY = F(Z, Y)
term1 = ((nablaA(X, V(Z)) + j(p, FZX[0], FZX[1])).T * G * V(Y))[0]
term2 = (V(X).T * G * (nablaA(Y, V(Z)) + j(p, FZY[0], FZY[1])))[0]
rhs_d = term1 + term2
print('(d) Lie-derivative identity residual:', sp.N(ev(LZh - rhs_d), 30),
      '; L_Z h(X,Y) value:', sp.N(ev(LZh), 12))
W1 = sp.Matrix([rpoly(), rpoly()])
W2 = sp.Matrix([rpoly(), rpoly()])
mc = apply(X, (W1.T * G * W2)[0]) - (nablaA(X, W1).T * G * W2)[0] - (W1.T * G * nablaA(X, W2))[0]
print('    metric compatibility of nabla^A residual:', sp.N(ev(mc), 30))

# (e) contact counterexample: translation group, J = id, p = 0, A = dz - x dy
Xc = sp.Matrix([1, 0, 0])
Yc = sp.Matrix([0, 1, x])
def Acon(v):
    return v[2] - x * v[1]
brc = bracket(Xc, Yc)
Fc = apply(Xc, Acon(Yc)) - apply(Yc, Acon(Xc)) - Acon(brc)
print('(e) contact: V(X)=', Acon(Xc), ' V(Y)=', sp.simplify(Acon(Yc)), ' V([X,Y])=', Acon(brc),
      ' -F(X,Y)=', -Fc)

# (f) transported KL, numerically (base R^2, slice z=0)
names = ['mu', 's', 'Adx', 'Ady', 'Atx', 'Aty']
exprs = [mu, s, Ad[0], Ad[1], At[0], At[1]]
lam = {}
for nm, ex in zip(names, exprs):
    lam[nm] = sp.lambdify((x, y), ex.subs(z, 0))
c0 = np.array([0.2, -0.1])
Xd = np.array([0.7, -0.4])
def transport_back(t):
    def rhs(u, kv):
        aa, bb = kv
        cc = c0 + u * Xd
        Adv = lam['Adx'](*cc) * Xd[0] + lam['Ady'](*cc) * Xd[1]
        Atv = lam['Atx'](*cc) * Xd[0] + lam['Aty'](*cc) * Xd[1]
        return [-Adv * aa, -(Adv * bb + Atv)]
    sol = solve_ivp(rhs, (t, 0.0), [1.0, 0.0], rtol=1e-12, atol=1e-14)
    aa, bb = sol.y[:, -1]
    ct = c0 + t * Xd
    m_t, s_t = lam['mu'](*ct), lam['s'](*ct)
    return aa * m_t + bb, s_t + np.log(aa)
m0, s0 = lam['mu'](*c0), lam['s'](*c0)
def KL(m0, s0, m1, s1):
    return (s1 - s0) + (np.exp(2 * s0) + (m0 - m1) ** 2) / (2 * np.exp(2 * s1)) - 0.5
pnum = sp.Matrix([mu, s]).subs(z, 0)
dpX = sp.Matrix([sum(Xd[k] * sp.diff(pnum[i], C[k]) for k in range(2)) for i in range(2)])
AdX = Ad[0].subs(z, 0) * Xd[0] + Ad[1].subs(z, 0) * Xd[1]
AtX = At[0].subs(z, 0) * Xd[0] + At[1].subs(z, 0) * Xd[1]
Vnum = dpX + j(pnum, AdX, AtX)
Vn = np.array([float(e.subs({x: c0[0], y: c0[1]})) for e in Vnum])
hXX = Vn[0] ** 2 * np.exp(-2 * s0) + 2 * Vn[1] ** 2
for t in (1e-1, 1e-2, 1e-3):
    m1, s1 = transport_back(t)
    ratio = KL(m0, s0, m1, s1) / (t * t / 2)
    print('(f) t=%g: KL/(t^2/2) = %.9f   h(X,X) = %.9f   rel.err = %.2e' % (t, ratio, hXX, abs(ratio / hXX - 1)))
