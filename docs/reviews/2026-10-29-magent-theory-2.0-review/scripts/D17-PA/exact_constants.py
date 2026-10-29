r"""D17-PA check 9: exact (SymPy) verification of the closed-form constants of chapter 13."""
import sympy as sp

J = sp.Rational(3, 2)
rq = sp.cosh(J) / (1 + sp.cosh(J))
# self rows at the record-free uniform table (pa:four-grouping)
Dii = sp.log(sp.cosh(J))     # sum q log q + log(4 cosh J) at uniform
bii = sp.simplify(rq * sp.exp(-Dii) / (rq * sp.exp(-Dii) + (1 - rq)))
print("beta_ii at uniform:", sp.nsimplify(sp.simplify(bii)))
# sign linearization pa:four-linear, coordinates (m1, m2, n1, n2)
b, g = sp.Rational(1, 2), sp.Rational(1, 2)
Mlin = sp.Matrix([[-1, 1 - b, b * J, 0], [1 - b, -1, 0, b * J], [b * J, 0, -1, 1 - g], [0, b * J, 1 - g, -1]])
print("sign linearization:", Mlin.tolist(), "eigenvalues:", sorted(Mlin.eigenvals().keys()))
t = sp.symbols("t", real=True)
eta = sp.symbols("eta", positive=True)
x0 = sp.Matrix([3, -1, 3, -1]) * eta
sol = (Mlin * t).exp() * x0
print("m(t) at log 4:", [sp.nsimplify(sp.simplify(sol[k].subs(t, sp.log(4)) / eta)) for k in range(4)], "(claim 3/sqrt2, 1/sqrt2)")
print("agent-2 zero crossing time:", sp.solve(sp.Eq(sp.exp(t / 4) - 2 * sp.exp(-3 * t / 4), 0), t))
# three-agent matrix pa:bit-limit
M = -sp.Rational(7, 5) * sp.eye(3) + sp.Rational(2, 5) * sp.ones(3, 3)
o = sp.Matrix([1, 1, -1])
print("eig M:", M.eigenvals(), " u_* =", (-M.inv() * o).T, " min ratio:", sp.Rational(5, 7) / sp.Rational(15, 7))
# null box arithmetic at eps = 1/100
e = sp.Rational(1, 100)
bound = 72 * e**2 + sp.Rational(1, 25) + 18 * e**6
print("72e^2+1/25+18e^6 =", sp.N(bound, 12), "< log2 =", sp.N(sp.log(2), 12), ":", bool(bound < sp.log(2)))
print("2|eta-eta_G|^2 upper bound expansion:", sp.expand(2 * ((12 * e / 2)**2 + sp.Rational(1, 100) + (3 * e**3)**2 + sp.Rational(1, 100))) == bound)
print("beta0 at D=log2:", sp.simplify(sp.Rational(1, 5) * sp.Rational(1, 2) / (sp.Rational(1, 5) * sp.Rational(1, 2) + sp.Rational(4, 5))), " beta0 at D=0:", sp.Rational(1, 5) / (sp.Rational(1, 5) + sp.Rational(4, 5)))
print("A-face derivative bound eps - 1/90 at eps=1/100:", e - sp.Rational(1, 90))
print("|L-l| bound 2*2e^3*(1/10+2e^3) =", sp.expand(2 * 2 * e**3 * (sp.Rational(1, 10) + 2 * e**3)), " vs eps/8 =", e / 8, ":", bool(2 * 2 * e**3 * (sp.Rational(1, 10) + 2 * e**3) < e / 8))
print("r-face: 1 - beta0/4 >= ", 1 - sp.Rational(1, 5) / 4, "; 1 - 12 beta0 <= ", 1 - 12 * sp.Rational(1, 9))
# curvature of A = -I/(I+B) d chi with I(s), s = e^{-t} a
a_, tt, B = sp.symbols("a t B", positive=True)
I = sp.Function("I")
A_chi = -I(sp.exp(-tt) * a_) / (I(sp.exp(-tt) * a_) + B)
F = sp.simplify(sp.diff(A_chi, a_))
target = -B * sp.exp(-tt) * sp.Subs(sp.Derivative(I(sp.Symbol("u")), sp.Symbol("u")), sp.Symbol("u"), sp.exp(-tt) * a_).doit() / (I(sp.exp(-tt) * a_) + B)**2
print("curvature formula residual:", sp.simplify(F - target))
