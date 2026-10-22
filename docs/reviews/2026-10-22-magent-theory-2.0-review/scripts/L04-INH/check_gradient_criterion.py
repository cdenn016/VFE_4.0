r"""Gradient inheritance (chapters/05_inheritance.tex, inh:gradient-inheritance).

1. Reproduces the proposition on a horizontally conformal example.
2. Tests the sharper criterion (report L04-INH-04): for a Riemannian submersion R,
   DR grad_g E = (grad_{g_c} E_c) o R  iff  d(E - E_c o R) vanishes on horizontal vectors.
   Example: R(x1,x2)=(x1+x2)/sqrt2 on Euclidean R^2,
   E = V(x1)+V(x2)+W(x1-x2). With V quadratic and arbitrary W, the center of mass descends exactly
   although E is not of the form chi(E_c o R). With non-quadratic V it fails.
3. Counterexample of the manuscript, E=(u-v)^2/2+v^2/2, R=v: horizontal criterion fails.
"""
import sympy as sp

x1, x2, zc = sp.symbols('x1 x2 z', real=True)
W = sp.Function('W')

def pushed_grad(E, Rmap, metric_inv):
    X = sp.Matrix([x1, x2])
    grad = metric_inv * sp.Matrix([sp.diff(E, x1), sp.diff(E, x2)])
    DR = sp.Matrix([[sp.diff(Rmap, x1), sp.diff(Rmap, x2)]])
    return sp.simplify((DR * grad)[0])

Rm = (x1 + x2) / sp.sqrt(2)
I2 = sp.eye(2)
# (2) quadratic V, arbitrary pair interaction W
V = lambda t: t**2 / 2
E = V(x1) + V(x2) + W(x1 - x2)
pg = pushed_grad(E, Rm, I2)
print("quadratic V, arbitrary W: DR grad E =", pg, "; grad E_c at R =", sp.simplify(Rm))
print("  residual:", sp.simplify(pg - Rm))
print("  E constant on fibers? d/ds E(x1+s,x2-s) at s=0 =",
      sp.simplify(sp.diff(E.subs({x1: x1 + sp.Symbol('s'), x2: x2 - sp.Symbol('s')}, simultaneous=True),
                          sp.Symbol('s')).subs(sp.Symbol('s'), 0)))
# horizontal derivative of E - E_c o R, E_c(z)=z^2/2 + W(0)
Ec = lambda t: t**2 / 2 + W(0)
hvec = sp.Matrix([1, 1]) / sp.sqrt(2)
dE = sp.Matrix([[sp.diff(E - Ec(Rm), x1), sp.diff(E - Ec(Rm), x2)]])
print("  d(E - E_c o R)(horizontal) =", sp.simplify((dE * hvec)[0]))

# non-quadratic V fails
V4 = lambda t: t**4 / 4
E4 = V4(x1) + V4(x2) + W(x1 - x2)
pg4 = pushed_grad(E4, Rm, I2)
p1 = {x1: sp.Rational(1, 2), x2: sp.Rational(1, 2)}
p2 = {x1: 1, x2: 0}
print("quartic V: DR grad E at (1/2,1/2) and (1,0) (same fiber):",
      sp.simplify(pg4.subs(p1, simultaneous=True)), sp.simplify(pg4.subs(p2, simultaneous=True)))

# (3) manuscript counterexample R(u,v)=v
u, v = x1, x2
Ecx = 0.5 * u**2  # placeholder not used
E3 = (u - v)**2 / 2 + v**2 / 2
DR3 = sp.Matrix([[0, 1]])
g3 = DR3 * sp.Matrix([sp.diff(E3, u), sp.diff(E3, v)])
print("manuscript example: DR grad E =", sp.simplify(g3[0]), "(depends on u: no descent)")

# (1) horizontally conformal: R(x1,x2) = 2*(x1+x2)/sqrt2 on Euclidean (dilation d=2), E = chi(E_c o R)
Rm2 = 2 * (x1 + x2) / sp.sqrt(2)
Ec2 = lambda t: sp.cos(t) + t**2
chi = sp.exp
E5 = chi(Ec2(Rm2))
pg5 = pushed_grad(E5, Rm2, I2)
target = (sp.exp(Ec2(zc)) * 4 * sp.diff(Ec2(zc), zc)).subs(zc, Rm2)
print("conformal case: DR grad E - chi' d^2 grad E_c =", sp.simplify(pg5 - target))
