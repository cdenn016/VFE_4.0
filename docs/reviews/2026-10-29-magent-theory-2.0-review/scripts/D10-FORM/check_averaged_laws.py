r"""D10-FORM check 1: averaged laws under smooth peer dynamics (ch05 lines 823-1008).

Verifies, symbolically where cheap and numerically otherwise:
  (a) inh:smooth-local-energy  E_0 = psi(x)+psi(y)-Jxy+log cosh J;
  (b) inh:smooth-channel       q_x K_x = q_{f(x)}, min entry - 1/4 = (1-x^2)/8;
  (c) Fisher information of (1+kx)/2 equals 1/(1-x^2);
  (d) partial derivatives of d(x,z)=KL(q_x||q_z);
  (e) inh:smooth-mixture-defect at A=(0,c,0,c), B=(a,c,-a,c);
  (f) the model-sector formulas with h(c);
  (g) inh:smooth-boundary-defect 3/4 m a^2 and the means 20/256, 17/256;
  (h) exchange equivariance of the vector field (inh:smooth-no-finite-entry);
  (i) linear opinion pool minimizes sum_j w_j KL(p_j||q);
  (j) EXTENSION (not a manuscript claim): the Gaussian moment-projection parent of a
      two-agent restricted-Gaussian peer-KL flow fails the fiber test, while in linear
      location consensus the moment projection (centroid, spread) is exact.
Coordinates follow the manuscript order (x1, y1, x2, y2).
"""
from __future__ import annotations

import numpy as np
import sympy as sp

x, y, z, J, a, c, m = sp.symbols("x y z J a c m", real=True)
k_vals = (-1, 1)


def q(par, k):
    return (1 + k * par) / 2


def psi(t):
    return (1 + t) / 2 * sp.log(1 + t) + (1 - t) / 2 * sp.log(1 - t)


out = []

# (a) local energy
E0 = sum(q(x, k) * q(y, mm) * sp.log(q(x, k) * q(y, mm) / (sp.Rational(1, 2) * sp.exp(J * k * mm) / (2 * sp.cosh(J))))
         for k in k_vals for mm in k_vals)
E0_closed = psi(x) + psi(y) - J * x * y + sp.log(sp.cosh(J))
rng = np.random.default_rng(1)
res_a = 0.0
for _ in range(20):
    vals = {x: rng.uniform(-0.95, 0.95), y: rng.uniform(-0.95, 0.95), J: rng.uniform(-2, 2)}
    res_a = max(res_a, abs(float((E0 - E0_closed).evalf(30, subs=vals))))
out.append(f"(a) local energy closed form, max residual over 20 random points: {res_a:.2e}")

# (b) channel
theta = (1 + x**2) / 4
f = (x + x**3) / 4
Kx = {(l, k): (1 + theta * k * l) / 2 for l in k_vals for k in k_vals}
push = {l: sp.simplify(sum(q(x, k) * Kx[(l, k)] for k in k_vals)) for l in k_vals}
res_b = [sp.simplify(push[l] - q(f, l)) for l in k_vals]
min_entry_gap = sp.simplify((1 - theta) / 2 - sp.Rational(1, 4))
out.append(f"(b) q_x K_x - q_f(x) residuals: {res_b}; min entry - 1/4 = {sp.factor(min_entry_gap)}")

# (c) Fisher information
fisher = sp.simplify(sum(sp.diff(q(x, k), x) ** 2 / q(x, k) for k in k_vals))
out.append(f"(c) Fisher information of (1+kx)/2: {fisher}")

# (d) KL derivatives
d = sum(q(x, k) * sp.log(q(x, k) / q(z, k)) for k in k_vals)
dx = sp.simplify(sp.diff(d, x) - (sp.atanh(x) - sp.atanh(z)))
dz = sp.simplify(sp.diff(d, z) - (z - x) / (1 - z**2))
dxf = sp.lambdify((x, z), sp.diff(d, x) - (sp.atanh(x) - sp.atanh(z)), "mpmath")
res_d = max(abs(dxf(rng.uniform(-0.95, 0.95), rng.uniform(-0.95, 0.95))) for _ in range(20))
out.append(f"(d) max |d_x d - (atanh x - atanh z)| over 20 random points: {float(res_d):.2e}; d_z d - (z-x)/(1-z^2) simplifies to {dz}")

# full energy and flow
x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2", real=True)


def dKL(u, v):
    return sum(q(u, k) * sp.log(q(u, k) / q(v, k)) for k in k_vals)


def fmap(t):
    return (t + t**3) / 4


Pfun = lambda u, v: dKL(u, fmap(v)) + dKL(v, fmap(u))
E = (E0_closed.subs({x: x1, y: y1}) + E0_closed.subs({x: x2, y: y2}) + Pfun(x1, x2) + Pfun(y1, y2))
vars4 = (x1, y1, x2, y2)
grad = [sp.diff(E, v) for v in vars4]
field = [-(1 - v**2) * g for v, g in zip(vars4, grad)]
field_num = sp.lambdify((x1, y1, x2, y2, J), field, "mpmath")

import mpmath as mp
mp.mp.dps = 40
Jv = mp.mpf(1) / 2
res_e = 0
for _ in range(10):
    av = mp.mpf(rng.uniform(0.05, 0.9)) * (1 if rng.random() < 0.5 else -1)
    cv = mp.mpf(rng.uniform(0.05, 0.9)) * (1 if rng.random() < 0.5 else -1)
    FA = field_num(0, cv, 0, cv, Jv)
    FB = field_num(av, cv, -av, cv, Jv)
    xbarA = (FA[0] + FA[2]) / 2
    xbarB = (FB[0] + FB[2]) / 2
    res_e = max(res_e, abs(xbarA - Jv * cv), abs(xbarB - Jv * cv * (1 - av**2)))
    # (f) model sector
    cc = cv
    fc = (cc + cc**3) / 4
    fpc = (1 + 3 * cc**2) / 4
    hc = 2 * mp.atanh(cc) - mp.atanh(fc) + fpc * (fc - cc) / (1 - fc**2)
    res_e = max(res_e, abs(FA[1] + (1 - cc**2) * hc), abs(FA[3] + (1 - cc**2) * hc),
                abs(FB[1] + (1 - cc**2) * (hc - Jv * av)), abs(FB[3] + (1 - cc**2) * (hc + Jv * av)))
out.append(f"(e,f) mixture defect and model-sector formulas, max residual over 10 random (a,c): {mp.nstr(res_e, 5)}")
FA = field_num(0, mp.mpf(1) / 4, 0, mp.mpf(1) / 4, Jv)
FB = field_num(mp.mpf(1) / 2, mp.mpf(1) / 4, -mp.mpf(1) / 2, mp.mpf(1) / 4, Jv)
out.append(f"(e) at a=1/2,c=1/4: xbar_dot(A)={mp.nstr((FA[0]+FA[2])/2, 12)} (1/8), xbar_dot(B)={mp.nstr((FB[0]+FB[2])/2, 12)} (3/32); ybar_dot(A)-ybar_dot(B)={mp.nstr((FA[1]+FA[3]-FB[1]-FB[3])/2, 5)}")
# averaged joint laws
R0, R14, R12 = sp.Integer(0), sp.Rational(1, 4), sp.Rational(1, 2)
jointA = [[(q(R0, k) * q(R14, mm) + q(R0, k) * q(R14, mm)) / 2 for mm in k_vals] for k in k_vals]
jointB = [[(q(R12, k) * q(R14, mm) + q(-R12, k) * q(R14, mm)) / 2 for mm in k_vals] for k in k_vals]
out.append(f"(e) averaged joint laws A={jointA}, B={jointB}, equal: {all(sp.simplify(jointA[i][j]-jointB[i][j])==0 for i in range(2) for j in range(2))}")

# (g) boundary defect
bfun = lambda u, v: (fmap(u) + fmap(v)) / 2
diffb = sp.simplify(bfun(m + a, m - a) - bfun(m, m) - sp.Rational(3, 4) * m * a**2)
q4 = sp.Rational(1, 4)
out.append(f"(g) b(m+a,m-a)-b(m,m)-3/4 m a^2 = {diffb}; means at m=a=1/4: {bfun(2*q4, 0)*256}/256, {bfun(q4, q4)*256}/256")

# (h) exchange equivariance: F(sigma x) = sigma F(x)
res_h = 0
for _ in range(10):
    p = [mp.mpf(rng.uniform(-0.9, 0.9)) for _ in range(4)]
    F1 = field_num(*p, Jv)
    F2 = field_num(p[2], p[3], p[0], p[1], Jv)
    res_h = max(res_h, abs(F1[0] - F2[2]), abs(F1[1] - F2[3]), abs(F1[2] - F2[0]), abs(F1[3] - F2[1]))
out.append(f"(h) exchange equivariance residual: {mp.nstr(res_h, 5)}")

# (i) linear opinion pool
pj = rng.dirichlet(np.ones(4), size=3)
w = rng.dirichlet(np.ones(3))
pool = w @ pj
obj = lambda qq: float(np.sum(w[:, None] * pj * (np.log(pj) - np.log(qq))))
best = obj(pool)
kl = lambda p_, q_: float(np.sum(p_ * (np.log(p_) - np.log(q_))))
res_i = 0.0
min_gap = np.inf
for _ in range(200):
    qq = rng.dirichlet(np.ones(4))
    res_i = max(res_i, abs(obj(qq) - best - kl(pool, qq)))
    min_gap = min(min_gap, obj(qq) - best)
out.append(f"(i) identity sum_j w_j KL(p_j||q) = sum_j w_j KL(p_j||pool) + KL(pool||q): max residual {res_i:.2e} over 200 random q; min gap {min_gap:.3e} > 0")

# (j) EXTENSION: Gaussian moment-projection parent
mu1, mu2, v1, v2, cG = sp.symbols("mu1 mu2 v1 v2 cG", real=True)


def klg(ma, va, mb, vb):
    return sp.Rational(1, 2) * (va / vb + (ma - mb) ** 2 / vb - 1 + sp.log(vb / va))


# two agents, common evaluator N(0,1), identity channels, fixed unit rows, restricted Gaussian family flow
Eg = klg(mu1, v1, 0, 1) + klg(mu2, v2, 0, 1) + klg(mu1, v1, mu2, v2) + klg(mu2, v2, mu1, v1)
mu1d = -v1 * sp.diff(Eg, mu1)
mu2d = -v2 * sp.diff(Eg, mu2)
v1d = -2 * v1**2 * sp.diff(Eg, v1)
v2d = -2 * v2**2 * sp.diff(Eg, v2)
mbar = (mu1 + mu2) / 2
vbar = (v1 + v2) / 2 + (mu1 - mu2) ** 2 / 4
mbar_d = (mu1d + mu2d) / 2
vbar_d = (v1d + v2d) / 2 + (mu1 - mu2) * (mu1d - mu2d) / 2
ptA = {mu1: 0, mu2: 0, v1: 1, v2: 1}
aa = sp.Rational(1, 2)
ptB = {mu1: aa, mu2: -aa, v1: 1 - aa**2, v2: 1 - aa**2}
retA = (sp.nsimplify(mbar.subs(ptA)), sp.nsimplify(vbar.subs(ptA)))
retB = (sp.nsimplify(mbar.subs(ptB)), sp.nsimplify(vbar.subs(ptB)))
velA = (sp.nsimplify(sp.simplify(mbar_d.subs(ptA))), sp.nsimplify(sp.simplify(vbar_d.subs(ptA))))
velB = (sp.nsimplify(sp.simplify(mbar_d.subs(ptB))), sp.nsimplify(sp.simplify(vbar_d.subs(ptB))))
out.append(f"(j) restricted Gaussian peer-KL, moment projection (mean,var) of pool: retained A={retA}, B={retB}; velocities A={velA}, B={velB}")
# linear location consensus with fixed unit variances: moment projection (centroid, spread) exact?
xs = sp.symbols("s1:5", real=True)
Elin = sum(xi**2 / 2 for xi in xs) + sum((xs[i] - xs[j]) ** 2 / 2 for i in range(4) for j in range(4) if i != j) / 3
vel = [-sp.diff(Elin, xi) for xi in xs]
cen = (xs[0] + xs[1]) / 2
spr = (xs[0] - xs[1]) ** 2 / 4
cen_d = sp.expand((vel[0] + vel[1]) / 2)
spr_d = sp.expand((xs[0] - xs[1]) * (vel[0] - vel[1]) / 2)
out.append(f"(j) linear location consensus: d/dt centroid = {cen_d}; d/dt spread = {sp.factor(spr_d)} (spread = (s1-s2)^2/4)")

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
