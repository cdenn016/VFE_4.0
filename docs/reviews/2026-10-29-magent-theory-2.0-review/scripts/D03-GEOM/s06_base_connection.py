"""D03-GEOM s06: appendices/base_connection.tex (eq:conn-infinitesimal, prop:conn-curvature,
prop:conn-patches, curved example, circle holonomy) and the chapter 16 SO(3)
link-alignment curvature formula quoted in open:conn-derived.
"""
import sympy as sp
import random
random.seed(5)
x, y, d1, d2, s = sp.symbols('x y d1 d2 s', real=True)
pt = {x: sp.Rational(1, 3), y: sp.Rational(-1, 4)}
def rp(): return sum(sp.Rational(random.randint(-4, 4), random.randint(1, 3))*t for t in [1, x, y, x*y, x**2, y**2])
def rmat(inv=False):
    M = sp.Matrix(2, 2, lambda i, j: rp())
    return M + 3*sp.eye(2) if inv else M
U = rmat(True); a = [rmat(), rmat()]
Ui = U.inv()
dU = [sp.diff(U, x), sp.diff(U, y)]
B = [U*a[k]*Ui - dU[k]*Ui for k in range(2)]
# first-order expansion of Theta(c+dc, c) = U(c+dc) exp(-a dc) U(c)^{-1}
Th = U.subs({x: x + s*d1, y: y + s*d2})*(sp.eye(2) - s*(a[0]*d1 + a[1]*d2))*Ui
lin = sp.diff(Th, s).subs(s, 0)
print("first jet: Theta = I - B(dc) residual:", sp.simplify((lin + B[0]*d1 + B[1]*d2).subs(pt)).T)
def curv(Bf):
    return sp.diff(Bf[1], x) - sp.diff(Bf[0], y) + Bf[0]*Bf[1] - Bf[1]*Bf[0]
F = curv(B); Fa = curv(a)
print("F = U F^a U^{-1} residual:", sp.simplify((F - U*Fa*Ui).subs(pt)).T)
g = rmat(True); gi = g.inv(); Up = g*U; Upi = Up.inv()
Bp = [Up*a[k]*Upi - sp.diff(Up, v)*Upi for k, v in enumerate((x, y))]
print("U' = gU: B' - (g B g^-1 - dg g^-1):", [sp.simplify((Bp[k] - (g*B[k]*gi - sp.diff(g, v)*gi)).subs(pt)).T for k, v in enumerate((x, y))])
h = rmat(True); hi = h.inv(); Uq = U*h; Uqi = Uq.inv()
ap = [hi*a[k]*h + hi*sp.diff(h, v) for k, v in enumerate((x, y))]
Bq = [Uq*ap[k]*Uqi - sp.diff(Uq, v)*Uqi for k, v in enumerate((x, y))]
print("U' = Uh with a' = h^-1 a h + h^-1 dh: B' - B:", [sp.simplify((Bq[k] - B[k]).subs(pt)).T for k in range(2)])
E12 = sp.Matrix([[0, 1], [0, 0]]); E21 = sp.Matrix([[0, 0], [1, 0]])
print("curved example F_xy:", curv([E12, E21]))
b, th = sp.symbols('b theta', real=True)
T = sp.Function('T')
sol = sp.dsolve(sp.Eq(T(th).diff(th), -b*T(th)), T(th), ics={T(0): 1})
print("circle holonomy:", sp.simplify(sol.rhs.subs(th, 2*sp.pi)))

# chapter 16 SO(3) example (quoted in open:conn-derived)
xx, yy, be = sp.symbols('x y beta', positive=True)
def Rx(t): return sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])
def Rz(t): return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])
Q = [Rx(xx), Rz(yy)*Rx(be)*Rz(-yy)]
def vee(M): return sp.Matrix([M[2, 1], M[0, 2], M[1, 0]])
def Ad(Qm): return Qm  # Ad_Q v = Q v for so(3) ~ R^3
C = [sp.eye(3) - Ad(Qm) for Qm in Q]
V = [[sp.simplify(vee(sp.diff(Qm, v)*Qm.T)) for v in (xx, yy)] for Qm in Q]  # right-trivialized derivative, per base direction
Kmat = sp.simplify(C[0].T*C[0] + C[1].T*C[1])
avec = [sp.simplify(-Kmat.inv()*sum((C[r].T*V[r][k] for r in range(2)), sp.zeros(3, 1))) for k in range(2)]
print("SO(3) link alignment: a_x =", [sp.simplify(e) for e in avec[0]], "; a_y =", [sp.simplify(e) for e in avec[1]])
wx = 4*sp.sin(xx/2)**2; wb = 4*sp.sin(be/2)**2
print("  a_y + wb/(wx+wb) e3 residual:", sp.simplify(avec[1] + wb/(wx + wb)*sp.Matrix([0, 0, 1])).T)
Fxy = sp.diff(avec[1], xx) - sp.diff(avec[0], yy) + avec[0].cross(avec[1])
print("  F_xy - 2 wb sin x/(wx+wb)^2 e3:", sp.simplify(Fxy - 2*wb*sp.sin(xx)/(wx + wb)**2*sp.Matrix([0, 0, 1])).T)
import random as _r
_r.seed(11)
r1 = sp.simplify(avec[1] + wb/(wx + wb)*sp.Matrix([0, 0, 1]))
r2 = sp.simplify(Fxy - 2*wb*sp.sin(xx)/(wx + wb)**2*sp.Matrix([0, 0, 1]))
mx = 0.0
for _ in range(5):
    sub = {xx: _r.uniform(0.1, 3.0), yy: _r.uniform(0.1, 3.0), be: _r.uniform(0.1, 3.0)}
    mx = max(mx, max(abs(float(e.subs(sub))) for e in list(r1) + list(r2)))
print("  numeric max residual of a_y and F_xy formulas at 5 random points:", f"{mx:.1e}")
