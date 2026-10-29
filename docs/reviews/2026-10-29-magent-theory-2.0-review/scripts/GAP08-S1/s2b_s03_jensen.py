r"""GAP08-S1 script 2b: supplements/s03_variation.tex:80-101, part (e).

eq:sup-integrated-score and eq:sup-gaussian-jensen in d = 1 by direct symbolic integration over
standardized normal variables (sympy); the Gaussian convolution by mpmath quadrature; the eigenvalue
form of the first line of G, positive semidefiniteness of R^-1 - (W+R)^-1 and G > 0 at 200 random
d = 2 instances, one in four with rank-one W (mpmath, 30 digits).
"""
import random
import sympy as sp
import mpmath as mp
from sympy import Rational as Q

random.seed(7)


def rq(lo=-5, hi=5, den=3):
    return Q(random.randint(lo, hi), den)


print("\n== (e) eq:sup-integrated-score and eq:sup-gaussian-jensen ==")
x, y = sp.symbols('x y', real=True)
aa, bb = sp.symbols('a b', real=True)
vv, ww_, rr = sp.symbols('v w rho', positive=True)


def npdf(z, m, s2):
    return sp.exp(-(z - m) ** 2 / (2 * s2)) / sp.sqrt(2 * sp.pi * s2)


z1, z2 = sp.symbols('z1 z2', real=True)
phi = lambda z: sp.exp(-z ** 2 / 2) / sp.sqrt(2 * sp.pi)


def E2(expr):
    """Expectation over independent standard normals z1, z2 by direct integration."""
    e = sp.expand(expr)
    i1 = sp.integrate(e * phi(z1), (z1, -sp.oo, sp.oo))
    return sp.simplify(sp.integrate(sp.expand(i1) * phi(z2), (z2, -sp.oo, sp.oo)))


X = aa + sp.sqrt(vv) * z1
Y = bb + sp.sqrt(ww_) * z2
E_neglog_k = E2(sp.log(2 * sp.pi * rr) / 2 + (X - Y) ** 2 / (2 * rr))
E_neglog_r = E2(sp.log(2 * sp.pi * (ww_ + rr)) / 2 + (X - bb) ** 2 / (2 * (ww_ + rr)))
formula_r = sp.log(2 * sp.pi * (ww_ + rr)) / 2 + ((aa - bb) ** 2 / (ww_ + rr) + vv / (ww_ + rr)) / 2
print("integrated score: -E log r(X) - formula =", sp.simplify(E_neglog_r - formula_r))
mp.mp.dps = 30
bq, wq, rq_ = mp.mpf(1) / 3, mp.mpf(1) / 2, mp.mpf(2) / 3
npd = lambda z, m, s2: mp.exp(-(z - m) ** 2 / (2 * s2)) / mp.sqrt(2 * mp.pi * s2)
conv_res = [mp.quad(lambda yy: npd(xv, yy, rq_) * npd(yy, bq, wq), [-mp.inf, bq, mp.inf]) - npd(xv, bq, wq + rq_)
            for xv in (mp.mpf(-1), mp.mpf(0), mp.mpf(5) / 2)]
print("convolution of N(.; y, rho) with N(y; b, w) minus N(x; b, w + rho) at x = -1, 0, 5/2 (mpmath quad):",
      [mp.nstr(c, 3) for c in conv_res])
G_formula = (sp.log(rr / (ww_ + rr)) / 2 + ww_ / (2 * rr)
             + (aa - bb) ** 2 * (1 / rr - 1 / (ww_ + rr)) / 2 + vv * (1 / rr - 1 / (ww_ + rr)) / 2)
print("Jensen gap: (E[-log k] + E log r) - G_formula =",
      sp.simplify(sp.expand_log(E_neglog_k - E_neglog_r - G_formula, force=True)))

mp.mp.dps = 30
random.seed(11)
worst = mp.mpf(0)
minG = mp.inf
minPSD = mp.inf
for trial in range(200):
    Mr = mp.matrix([[random.uniform(-2, 2) for _ in range(2)] for _ in range(2)])
    Rr = Mr * Mr.T + mp.eye(2) * mp.mpf('0.1')
    Mw = mp.matrix([[random.uniform(-2, 2) for _ in range(2)] for _ in range(2)])
    if trial % 4 == 0:
        Mw[1, 0] = 0
        Mw[1, 1] = 0  # singular W (rank one)
    Wr = Mw * Mw.T
    Vr = mp.matrix([[random.uniform(0.5, 2), 0], [0, random.uniform(0.5, 2)]])
    am = mp.matrix([random.uniform(-1, 1), random.uniform(-1, 1)])
    Ri_ = Rr ** -1
    WRi = (Wr + Rr) ** -1
    first = mp.log(mp.det(Rr) / mp.det(Wr + Rr)) / 2 + (Ri_ * Wr).apply(lambda z: z)[0, 0] / 2 + (Ri_ * Wr)[1, 1] / 2
    # eigenvalues of R^{-1/2} W R^{-1/2} = eigenvalues of R^{-1} W
    ev, _ = mp.eig(Ri_ * Wr)
    ev = [mp.re(e) for e in ev]
    eig_form = sum(e - mp.log(1 + e) for e in ev) / 2
    worst = max(worst, abs(first - eig_form))
    Dm = Ri_ - WRi
    evD, _ = mp.eig(Dm)
    minPSD = min(minPSD, min(mp.re(e) for e in evD))
    G = first + (am.T * Dm * am)[0, 0] / 2 + (Dm * Vr)[0, 0] / 2 + (Dm * Vr)[1, 1] / 2
    minG = min(minG, G)
print("200 random d = 2 instances (one in four with rank-one W):")
print("  |first line - (1/2) sum (lambda - log(1 + lambda))| max =", mp.nstr(worst, 5))
print("  min eigenvalue of R^-1 - (W+R)^-1 =", mp.nstr(minPSD, 5), "; min G =", mp.nstr(minG, 5))
Wz = mp.zeros(2, 2)
print("  W = 0 gives first line", mp.nstr(mp.log(mp.det(Rr) / mp.det(Rr)) / 2, 5),
      "and R^-1 - (0+R)^-1 = 0, hence G = 0")
