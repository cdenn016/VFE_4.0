r"""Symbolic checks for prop:boundary-observable (eq:boundary-binary-flow,
eq:boundary-response-derivatives, eq:boundary-observability-det) and for the
recorded smooth-angular-contrast failure F(a) = log(1-a^2)/2 + 3 a atanh a.

The binary flow is rederived from the potential: two agents, k, m in {-1,+1},
q_i = b_{x_i}, s_i = b_{y_i}, b_x(k) = (1 + k x)/2, model reference r = uniform
(an assumption the proposition does not state), evaluator exp(J k m)/(2 cosh J),
identity peer channels, one source per row (beta = 1).
"""
import sympy as sp

x1, x2, y1, y2, J = sp.symbols('x1 x2 y1 y2 J', real=True)
m, n, a, b, eps, c1, c2 = sp.symbols('m n a b epsilon c1 c2', real=True)

def negent(x):
    return sum((1 + k * x) / 2 * sp.log((1 + k * x) / 2) for k in (1, -1))
def kl(x, y):
    return sum((1 + k * x) / 2 * sp.log((1 + k * x) / (1 + k * y)) for k in (1, -1))

# local term per agent with uniform model reference: KL(s||U) + E_s KL(q || G(m,.))
def local(x, y):
    return (negent(y) + sp.log(2)) + negent(x) - J * x * y + sp.log(2 * sp.cosh(J))
E = local(x1, y1) + local(x2, y2) + kl(x1, x2) + kl(x2, x1) + kl(y1, y2) + kl(y2, y1)

# Fisher flow for a binary table in mean coordinate: xdot = -(1 - x^2) dE/dx
xd1 = sp.simplify(-(1 - x1**2) * sp.diff(E, x1))
claimed = -(1 - x1**2) * (2 * sp.atanh(x1) - sp.atanh(x2) - J * y1) - (x1 - x2)
diff = sp.simplify((xd1 - claimed).rewrite(sp.log))
print("eq:boundary-binary-flow residual (symbolic):", sp.simplify(sp.expand_log(diff, force=True)))
import random
vals = {x1: 0.31, x2: -0.17, y1: 0.22, y2: 0.4, J: 0.13}
print("numeric residual:", sp.N((xd1 - claimed).subs(vals)))

yd1 = -(1 - y1**2) * sp.diff(E, y1)
claimed_y = -(1 - y1**2) * (2 * sp.atanh(y1) - sp.atanh(y2) - J * x1) - (y1 - y2)
print("exchanged model equation numeric residual:", sp.N((yd1 - claimed_y).subs(vals)))

# response velocities
xd2 = claimed.subs({x1: x2, x2: x1, y1: y2}, simultaneous=True)
Fx = sp.Rational(1, 2) * (claimed + xd2)
Fx = Fx.subs({x1: m + a, x2: m - a, y1: n + b, y2: n - b}, simultaneous=True)
t = sp.symbols('t')
ser = sp.series(Fx.subs({m: t * m, n: t * n, a: t * a, b: t * b}), t, 0, 5).removeO()
ser = sp.expand(ser.subs(t, 1))
claimed_Fx = sp.expand(sp.series(-(1 - (t*m)**2) * (sp.atanh(t*m) - J * t*n), t, 0, 5).removeO().subs(t, 1)
                       + (6 * m - J * n) * a**2 - 2 * J * m * a * b)
print("F_x through total degree 4, residual vs eq:boundary-response-derivatives:", sp.simplify(ser - claimed_Fx))

# exact Jacobian determinant at m=n=eps c1, a=b=eps c2
yd2 = claimed_y.subs({y1: y2, y2: y1, x1: x2}, simultaneous=True)
Fy = sp.Rational(1, 2) * (claimed_y + yd2)
Fy = Fy.subs({x1: m + a, x2: m - a, y1: n + b, y2: n - b}, simultaneous=True)
Jac = sp.Matrix([[sp.diff(Fx, a), sp.diff(Fx, b)], [sp.diff(Fy, a), sp.diff(Fy, b)]])
det = Jac.det().subs({m: eps * c1, n: eps * c1, a: eps * c2, b: eps * c2})
lead = sp.series(det, eps, 0, 6).removeO()
print("det D H series to eps^5:", sp.factor(sp.simplify(lead)))
print("claimed leading term:", sp.factor(4 * eps**4 * c1**2 * c2**2 * (6 - 3 * J) * (6 - J)))
# where does the leading coefficient vanish?
print("roots in J of (6-3J)(6-J):", sp.solve((6 - 3 * J) * (6 - J), J))

# smooth angular contrast: q_pm = m (1 +- a); mixture velocity term
aa = sp.symbols('aa', real=True)
F = sp.log(1 - aa**2) / 2 + 3 * aa * sp.atanh(aa)
print("F''(a) - (5-a^2)/(1-a^2)^2 =", sp.simplify(sp.diff(F, aa, 2) - (5 - aa**2) / (1 - aa**2)**2))
# derive the bracket from the identity-peer force for two agents
q1 = 1 + aa; q2 = 1 - aa
brk = q1 * (2 * sp.log(q1) - sp.log(q2)) + q2 * (2 * sp.log(q2) - sp.log(q1))
print("bracket/2 - F =", sp.simplify(sp.expand_log(brk / 2 - F, force=True).rewrite(sp.log)),
      " numeric at a=0.37:", sp.N((brk / 2 - F).subs(aa, 0.37)))
