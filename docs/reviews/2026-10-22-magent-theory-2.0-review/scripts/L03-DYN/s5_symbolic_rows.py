"""L03-DYN script 5: symbolic checks of the row algebra and the companion Bernoulli flow.

(1) prop:dyn-gibbs / eq:comp-source-row: f(beta) = sum beta_j D_j + T KL(beta||pi)
    equals -T log Z + T KL(beta||beta*), for a symbolic three-source row.
(2) eq:dyn-envelope: -T d log Z / dD_j = beta*_j.
(3) Row force at frozen costs: h_j = D_j + T(log(beta_j/pi_j)+1) = T log(beta_j/beta*_j) + T(1 - log Z),
    so log-odds contrasts of the row relax at rate T under unit Fisher mobility.
(4) eq:comp-bernoulli-flow: Fisher flow of KL(Ber p||Ber p0) gives pdot = -p(1-p)(l-l0), ldot = -(l-l0).
(5) Bounds: -T log Z lies between min_j D_j and min_j (D_j - T log pi_j), and below sum pi_j D_j (numeric).
"""
import numpy as np
import sympy as sp

T = sp.symbols('T', positive=True)
D = sp.symbols('D1:4', real=True)
pi = sp.symbols('pi1:4', positive=True)
b = sp.symbols('b1:4', positive=True)
Z = sum(pi[j] * sp.exp(-D[j] / T) for j in range(3))
bstar = [pi[j] * sp.exp(-D[j] / T) / Z for j in range(3)]
cons = {b[2]: 1 - b[0] - b[1], pi[2]: 1 - pi[0] - pi[1]}
f = sum(b[j] * D[j] for j in range(3)) + T * sum(b[j] * sp.log(b[j] / pi[j]) for j in range(3))
rhs = -T * sp.log(Z) + T * sum(b[j] * sp.log(b[j] / bstar[j]) for j in range(3))
res = sp.simplify(sp.expand_log((f - rhs).subs(cons), force=True))
print("(1) row identity residual:", res)
env = [sp.simplify(sp.diff(-T * sp.log(Z), D[j]) - bstar[j]) for j in range(3)]
print("(2) envelope residuals:", env)
h = [D[j] + T * (sp.log(b[j] / pi[j]) + 1) for j in range(3)]
hh = [T * sp.log(b[j] / bstar[j]) + T * (1 - sp.log(Z)) for j in range(3)]
print("(3) row force identity residuals:", [sp.simplify(sp.expand_log(h[j] - hh[j], force=True)) for j in range(3)])

p, r, p0, r0 = sp.symbols('p r p0 r0', positive=True)   # r = 1-p, r0 = 1-p0
u1, u2 = sp.symbols('u1 u2', positive=True)
Eu = u1 * sp.log(u1 / p0) + u2 * sp.log(u2 / r0)
h1, h2 = [sp.diff(Eu, v).subs({u1: p, u2: r}) for v in (u1, u2)]
pdot = -p * (h1 - (p * h1 + r * h2))
pdot = sp.expand(pdot.subs(p * h1, p * h1))
l, l0 = sp.log(p / r), sp.log(p0 / r0)
res4 = sp.simplify(sp.expand_log((pdot + p * r * (l - l0)).subs(r, 1 - p).subs(p, 1 - r).subs(r, 1 - p), force=True))
res4 = sp.simplify(sp.expand_log((-p * (h1 - h2) * (1 - p) + p * r * (l - l0)).subs(r, 1 - p), force=True))
print("(4) with p + r = 1: pdot = -p(1-p)(h1-h2) and h1-h2 - (l-l0) =",
      sp.simplify(sp.expand_log(h1 - h2 - (l - l0), force=True)),
      "; centered form minus -p r (h1-h2) at r=1-p:",
      sp.simplify((-p * (h1 - (p * h1 + r * h2)) + p * r * (h1 - h2)).subs(r, 1 - p)))

rng = np.random.default_rng(3)
worst = [np.inf, np.inf, np.inf]
for _ in range(2000):
    n = rng.integers(2, 6)
    Dn = rng.exponential(2.0, size=n)
    pn = rng.dirichlet(np.ones(n))
    Tn = rng.exponential(1.0)
    a = np.log(pn) - Dn / Tn
    v = -Tn * (a.max() + np.log(np.sum(np.exp(a - a.max()))))
    worst[0] = min(worst[0], v - Dn.min())
    worst[1] = min(worst[1], (Dn - Tn * np.log(pn)).min() - v)
    worst[2] = min(worst[2], pn @ Dn - v)
print("(5) min over 2000 random rows of [ -TlogZ - min D , min(D - T log pi) + TlogZ , sum pi D + TlogZ ]:", worst)
