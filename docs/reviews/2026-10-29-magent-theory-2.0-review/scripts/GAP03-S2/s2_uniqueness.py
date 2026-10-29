r"""GAP03-S2 s2: the uniqueness clause of form:full-flow.

Checks, symbolically, the two steps of the proof given in the report:
 (i)  integrating factor: for any locally integrable C(t), c(t), the function
        l(t,X) = e^{-2t} l0(X) + (1 - e^{-2t}) c0 log det X - Tr(Chat(t) X) + k(t),
        Chat' = -2 Chat + I + C(t), Chat(0) = 0,  k' = -2k + c(t), k(0) = 0,
      solves  d/dt l = -2 l + 2 c0 log det X - Tr((I + C(t)) X) + c(t)   (pointwise in X),
      and with Wishart l0 it is the Wishart log density with
        alpha(t) = alpha0 + e^{-2t}(alpha(0) - alpha0),  P(t) = e^{-2t} P(0) + Chat(t),
        P' = -2P + I + C(t).
 (ii) normalizability forces P(t) > 0: for a symmetric Pi with an eigenvalue <= 0 the
      integral of det(X)^{a-3/2} exp(-Tr(Pi X)) over the cone diverges (rotation to diagonal
      form, triangular chart u, z, v; the v-integral diverges).
 (iii) a non-Wishart positive initial law: the same formula gives the explicit solution
      q(t) ∝ q0^{e^{-2t}} det(X)^{(1-e^{-2t}) c0} e^{-Tr(Chat X)}, so the whole density flow of
      form:energy reduces to a 3-dimensional ODE for Chat (common to all agents); we check the
      identity log(q_i/q_j)(t) = e^{-2t} log(q_i/q_j)(0) + const (form:mean-field).
"""
import sympy as sp

t, s = sp.symbols('t s', real=True)
c0 = sp.symbols('c0', real=True)
x11, x12, x22 = sp.symbols('x11 x12 x22', real=True)
X = sp.Matrix([[x11, x12], [x12, x22]])
C11, C12, C22, cfun = (sp.Function(n)(t) for n in ('C11', 'C12', 'C22', 'c'))
H11, H12, H22, kfun = (sp.Function(n)(t) for n in ('H11', 'H12', 'H22', 'k'))
l0 = sp.Function('l0')(x11, x12, x22)
Chat = sp.Matrix([[H11, H12], [H12, H22]])
Cm = sp.Matrix([[C11, C12], [C12, C22]])
ld = sp.log(X.det())
l = sp.exp(-2 * t) * l0 + (1 - sp.exp(-2 * t)) * c0 * ld - (Chat * X).trace() + kfun
rhs = -2 * l + 2 * c0 * ld - ((sp.eye(2) + Cm) * X).trace() + cfun
lhs = sp.diff(l, t)
# impose Chat' = -2 Chat + I + C and k' = -2k + c
sub = {sp.Derivative(H11, t): -2 * H11 + 1 + C11, sp.Derivative(H12, t): -2 * H12 + C12,
       sp.Derivative(H22, t): -2 * H22 + 1 + C22, sp.Derivative(kfun, t): -2 * kfun + cfun}
res = sp.simplify(lhs.subs(sub) - rhs)
print("(i) residual of the integrating-factor solution:", res)

al0, a0 = sp.symbols('alpha_0i alpha0', positive=True)
q11, q12, q22, K0 = sp.symbols('q11 q12 q22 K0', real=True)
P0 = sp.Matrix([[q11, q12], [q12, q22]])
lw = (al0 - sp.Rational(3, 2)) * ld - (P0 * X).trace() + K0
lt = l.subs(l0, lw)
# coefficient of log det X and of X entries
cand = (a0 + sp.exp(-2 * t) * (al0 - a0) - sp.Rational(3, 2)) * ld - ((sp.exp(-2 * t) * P0 + Chat) * X).trace()
print("(i) Wishart form at time t, residual (should be X-free):",
      sp.simplify(sp.expand(lt - cand).subs(c0, a0 - sp.Rational(3, 2))))
Pt = sp.exp(-2 * t) * P0 + Chat
dP = sp.diff(Pt, t).subs(sub)
print("(i) P(t) = e^{-2t}P(0) + Chat satisfies P' = -2P + I + C:",
      sp.simplify(dP - (-2 * Pt + sp.eye(2) + Cm)))

# (ii) divergence for Pi = diag(pi1, pi2) with pi2 <= 0 in the triangular chart
u, v, z = sp.symbols('u v z', positive=True)
a, pi1 = sp.symbols('a pi1', positive=True)
pi2 = sp.symbols('pi2', nonpositive=True)
integrand_v = v**(a - sp.Rational(3, 2)) * sp.exp(-pi2 * v)
# for pi2 <= 0 and a > 1/2, the integrand in v does not tend to zero: lower bound v^(a-3/2) on v >= 1
print("(ii) v-integrand at pi2 = 0:", integrand_v.subs(pi2, 0),
      "; its integral over v in (1, oo) for a = 1:", sp.integrate(integrand_v.subs({pi2: 0, a: 1}), (v, 1, sp.oo)))
print("(ii) and for a = 2, pi2 = -1/5:", sp.integrate(integrand_v.subs({pi2: -sp.Rational(1, 5), a: 2}), (v, 1, sp.oo)))

# (iii) common-clock identity for two agents with arbitrary initial log densities
l0j = sp.Function('l0j')(x11, x12, x22)
kj = sp.Function('kj')(t)
li = l
lj = sp.exp(-2 * t) * l0j + (1 - sp.exp(-2 * t)) * c0 * ld - (Chat * X).trace() + kj
print("(iii) log q_i - log q_j - e^{-2t}(l0 - l0j) =", sp.simplify(li - lj - sp.exp(-2 * t) * (l0 - l0j)),
      "(sample independent)")
