"""D03-GEOM s02: symbolic checks of geo:passive, geo:connection-change,
geo:structure-identity, geo:basic-criterion, and the contact example.

Fiber: N(mu, sigma^2), sigma>0, under Aff^+(1) acting on samples y -> a y + b.
Lie algebra element xi = (alpha, beta) ~ [[alpha, beta],[0,0]];
J_p xi = (alpha mu + beta, alpha sigma); Fisher g = diag(1/sigma^2, 2/sigma^2).
Base R^3 with coordinates x1,x2,x3; random polynomial data; exact rational evaluation.
"""
import sympy as sp
import random
random.seed(20261029)
x1, x2, x3 = X = sp.symbols('x1 x2 x3', real=True)
pt = {x1: sp.Rational(1, 3), x2: sp.Rational(-2, 5), x3: sp.Rational(3, 7)}

def rpoly(deg=2, const=0):
    terms = [1, x1, x2, x3, x1*x2, x2*x3, x1*x3, x1**2, x2**2, x3**2]
    return const + sum(sp.Rational(random.randint(-5, 5), random.randint(1, 4))*t for t in terms[:1+3*deg])

mu = rpoly()
sigma = 3 + sp.Rational(1, 10)*rpoly()  # positive near the point
p = sp.Matrix([mu, sigma])

def J(pv, xi):
    return sp.Matrix([xi[0]*pv[0] + xi[1], xi[0]*pv[1]])

def dpj(w, xi):  # derivative of J_p xi in p along w
    return sp.Matrix([xi[0]*w[0], xi[0]*w[1]])

def bracket(a, b):  # [[a0,a1],[0,0]] commutator
    return sp.Matrix([0, a[0]*b[1] - b[0]*a[1]])

def gF(pv, u, w):
    return (u[0]*w[0] + 2*u[1]*w[1])/pv[1]**2

def vf_apply(Vf, fexpr):  # vector field (3-list) acting on scalar/matrix expr
    return sum((Vf[i]*sp.diff(fexpr, X[i]) for i in range(3)), sp.zeros(*fexpr.shape) if hasattr(fexpr, 'shape') else 0)

def lie_bracket(V1, V2):
    return [sum(V1[j]*sp.diff(V2[i], X[j]) - V2[j]*sp.diff(V1[i], X[j]) for j in range(3)) for i in range(3)]

# connection: A_i = (alpha_i, beta_i) components along dx_i
Acomp = [sp.Matrix([rpoly(1), rpoly(1)]) for _ in range(3)]
def A_of(Vf, Acomp=Acomp):
    return sum((Vf[i]*Acomp[i] for i in range(3)), sp.zeros(2, 1))
def dp_of(Vf, pv=p):
    return sp.Matrix([vf_apply(Vf, pv[0]), vf_apply(Vf, pv[1])])
def Vop(Vf, pv=p, Acomp=Acomp):
    return dp_of(Vf, pv) + J(pv, A_of(Vf, Acomp))
def F_of(V1, V2, Acomp=Acomp):
    a1, a2 = A_of(V1, Acomp), A_of(V2, Acomp)
    dA = vf_apply(V1, a2) - vf_apply(V2, a1) - A_of(lie_bracket(V1, V2), Acomp)
    return dA + bracket(a1, a2)

Xf = [rpoly(1), rpoly(1), rpoly(1)]
Yf = [rpoly(1), rpoly(1), rpoly(1)]

# (1) passive covariance: g = (a, b), a>0
a = 2 + sp.Rational(1, 10)*rpoly(1); b = rpoly(1)
gm = sp.Matrix([[a, b], [0, 1]]); gi = gm.inv()
def to_mat(xi): return sp.Matrix([[xi[0], xi[1]], [0, 0]])
def from_mat(M): return sp.Matrix([M[0, 0], M[0, 1]])
pprime = sp.Matrix([a*mu + b, a*sigma])
Aprime = []
for i in range(3):
    dg = sp.diff(gm, X[i])
    Aprime.append(from_mat(gm*to_mat(Acomp[i])*gi - dg*gi))
lhs = Vop(Xf, pprime, Aprime)
rhs = a*Vop(Xf)  # T(g_#) acts by a on (mu, sigma)
print("(1) passive covariance residual:", sp.simplify((lhs - rhs).subs(pt)).T)
h = gF(p, Vop(Xf), Vop(Yf)); hp = gF(pprime, Vop(Xf, pprime, Aprime), Vop(Yf, pprime, Aprime))
print("    h' - h:", sp.simplify((hp - h).subs(pt)))

# (2) connection change
alpha = [sp.Matrix([rpoly(1), rpoly(1)]) for _ in range(3)]
A2 = [Acomp[i] + alpha[i] for i in range(3)]
uX, uY = Vop(Xf), Vop(Yf)
bX, bY = J(p, A_of(Xf, alpha)), J(p, A_of(Yf, alpha))
lhs = gF(p, Vop(Xf, p, A2), Vop(Yf, p, A2))
rhs = gF(p, uX, uY) + gF(p, uX, bY) + gF(p, bX, uY) + gF(p, bX, bY)
print("(2) connection-change residual:", sp.simplify((lhs - rhs).subs(pt)))

# (3) structure identity for arbitrary X, Y
VX, VY = Vop(Xf), Vop(Yf)
lhs = Vop(lie_bracket(Xf, Yf))
rhs = (vf_apply(Xf, VY) + dpj(VY, A_of(Xf))) - (vf_apply(Yf, VX) + dpj(VX, A_of(Yf))) - J(p, F_of(Xf, Yf))
val = lhs.subs(pt)
print("(3) structure identity residual:", sp.simplify((lhs - rhs).subs(pt)).T, " |V([X,Y])| =", [sp.N(v, 6) for v in val])

# (4) basic criterion: choose A_3 so that V(d/dx3) = 0, Z = f d/dx3
d3p = sp.Matrix([sp.diff(mu, x3), sp.diff(sigma, x3)])
al3 = -d3p[1]/sigma; be3 = -d3p[0] - al3*mu
Ab = [Acomp[0], Acomp[1], sp.Matrix([al3, be3])]
fz = 1 + rpoly(1)
Zf = [0, 0, fz]
print("(4) V(Z) =", sp.simplify(Vop(Zf, p, Ab)).T)
def hfun(V1, V2): return gF(p, Vop(V1, p, Ab), Vop(V2, p, Ab))
LZh = vf_apply(Zf, sp.Matrix([hfun(Xf, Yf)]))[0] - hfun(lie_bracket(Zf, Xf), Yf) - hfun(Xf, lie_bracket(Zf, Yf))
rhs = gF(p, J(p, F_of(Zf, Xf, Ab)), Vop(Yf, p, Ab)) + gF(p, Vop(Xf, p, Ab), J(p, F_of(Zf, Yf, Ab)))
print("    basic-criterion residual:", sp.simplify((LZh - rhs).subs(pt)), " L_Z h(X,Y) =", sp.N(LZh.subs(pt), 8))

# (5) contact example: unit-variance location, translations, A = dz - x dy
xx, yy, zz = sp.symbols('x y z')
Acont = {xx: 0, yy: -xx, zz: 1}
def Vc(v): return sum(v[k]*Acont[k] for k in (xx, yy, zz))
N1 = {xx: 1, yy: 0, zz: 0}; N2 = {xx: 0, yy: 1, zz: xx}
br = {zz: sp.diff(N2[zz], xx)*N1[xx], xx: 0, yy: 0}
print("(5) contact: V(d_x) =", Vc(N1), " V(d_y + x d_z) =", sp.simplify(Vc(N2)), " V([.,.]) =", Vc(br),
      " -F(N1,N2) = -(-dx^dy)(N1,N2) =", -(-(N1[xx]*N2[yy] - N1[yy]*N2[xx])))
