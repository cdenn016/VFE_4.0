r"""D06-PRED s4: projection, reverse-divergence counterexample, later-readout counterexample, joint Fisher.

  cmem:projection          KL(p||q) - KL(p||q_G) = KL(q_G||q) for random Gaussian q (1D quadrature);
                           projection-loss identity KL(Pi_{A,Y}||w x q_G) = I(A;Y) + KL(p_Y||q_G) (1D and 2D);
                           diagonal loss 1/2 log(prod S_kk / det S) (2D)
  cmem:reverse-counterexample  exact KL(q_a||p_a) by quadrature vs the stated lower bound; KL(N(a,1)||p_a) <= log 2;
                           crossing points; stationarity of the reverse divergence at the moment carrier:
                           g(a) = (1+a^2) E_{N(0,1+a^2)} sech^2(aY) - 1 (zero iff sigma^2 = 1+a^2 is stationary
                           for KL(N(0,sigma^2)||p_a)); reverse-KL optimal Gaussian for several a
  cmem:readout-loss        (p_t K_c)(0) vs (q_G K_c)(0), the derivative -t^3/(1-t^2+c)^2 (sympy), interval
                           probabilities, and the characteristic-function argument (injectivity of Gaussian
                           convolution): phi_{p_t}(s) e^{s^2/2} = cos(ts) e^{t^2 s^2/2} = 1 - t^4 s^4/12 + O(s^6)
  cmem:joint-fisher        coefficient (1+t^2)/(1-t^2)^2 symbolically and from the score by quadrature;
                           fourth moment 3 - 2t^4; Fisher information of the result marginal p_t alone
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy import integrate, optimize, stats

rng = np.random.default_rng(7)

# 1D mixture
wv = np.array([0.5, 0.3, 0.2])
mv = np.array([-1.0, 0.7, 2.5])
Pv = np.array([0.6, 1.3, 0.4])


def pY(y):
    return np.sum(wv[:, None] * stats.norm.pdf(np.atleast_1d(y)[None, :], mv[:, None], np.sqrt(Pv)[:, None]), axis=0)


mbar = float(wv @ mv)
S = float(wv @ (Pv + (mv - mbar) ** 2))
lo, hi = -15, 15


def kl_p_gauss(m, s2):
    f = lambda yy: pY(yy)[0] * (np.log(pY(yy)[0]) - stats.norm.logpdf(yy, m, np.sqrt(s2)))
    return integrate.quad(f, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


def kl_gauss(m1, s1, m2, s2):
    return 0.5 * (s1 / s2 + (m1 - m2) ** 2 / s2 - 1 + np.log(s2 / s1))


base = kl_p_gauss(mbar, S)
res = max(abs(kl_p_gauss(m, s2) - base - kl_gauss(mbar, S, m, s2))
          for m, s2 in zip(rng.normal(0, 1.5, 6), rng.uniform(0.3, 6, 6)))
print("cmem:projection 1D: max |KL(p||q) - KL(p||q_G) - KL(q_G||q)| over 6 random q =", res)
# projection loss, 1D: Pi_{A,Y} = w_a N(m_a, P_a)
I_AY = sum(wv[a] * integrate.quad(lambda yy: stats.norm.pdf(yy, mv[a], np.sqrt(Pv[a])) *
                                   (stats.norm.logpdf(yy, mv[a], np.sqrt(Pv[a])) - np.log(pY(yy)[0])),
                                   lo, hi, limit=400, epsabs=1e-13)[0] for a in range(3))
lhs = sum(wv[a] * integrate.quad(lambda yy: stats.norm.pdf(yy, mv[a], np.sqrt(Pv[a])) *
                                  (stats.norm.logpdf(yy, mv[a], np.sqrt(Pv[a])) - stats.norm.logpdf(yy, mbar, np.sqrt(S))),
                                  lo, hi, limit=400, epsabs=1e-13)[0] for a in range(3))
print("cmem:projection-loss 1D residual =", abs(lhs - I_AY - base), " (I =", I_AY, ", KL(p||q_G) =", base, ")")

# 2D mixture with Gauss-Hermite quadrature per component
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(60)
GX, GY = np.meshgrid(gh_x, gh_x, indexing="ij")
GW = np.outer(gh_w, gh_w) / (2 * np.pi)
nodes = np.stack([GX.ravel(), GY.ravel()], axis=1)
w2 = np.array([0.6, 0.4])
m2 = [np.array([0.0, 1.0]), np.array([1.5, -1.0])]
P2 = [np.array([[1.0, 0.6], [0.6, 0.8]]), np.array([[0.5, -0.2], [-0.2, 1.2]])]
mb2 = sum(w2[a] * m2[a] for a in range(2))
S2 = sum(w2[a] * (P2[a] + np.outer(m2[a] - mb2, m2[a] - mb2)) for a in range(2))
lmv = stats.multivariate_normal


def logp2(yy):
    return np.log(sum(w2[a] * lmv.pdf(yy, m2[a], P2[a]) for a in range(2)))


Ia = 0.0
lhs2 = 0.0
klG = 0.0
klD = 0.0
for a in range(2):
    Lc = np.linalg.cholesky(P2[a])
    pts = m2[a] + nodes @ Lc.T
    wt = GW.ravel()
    lpa = lmv.logpdf(pts, m2[a], P2[a])
    Ia += w2[a] * np.sum(wt * (lpa - logp2(pts)))
    lhs2 += w2[a] * np.sum(wt * (lpa - lmv.logpdf(pts, mb2, S2)))
    klG += w2[a] * np.sum(wt * (logp2(pts) - lmv.logpdf(pts, mb2, S2)))
    klD += w2[a] * np.sum(wt * (logp2(pts) - lmv.logpdf(pts, mb2, np.diag(np.diag(S2)))))
print("cmem:projection-loss 2D residual =", abs(lhs2 - Ia - klG))
print("cmem:diagonal-loss 2D residual =", abs(klD - klG - 0.5 * np.log(np.prod(np.diag(S2)) / np.linalg.det(S2))))

# reverse-divergence counterexample
def pa_logpdf(yy, a):
    return np.logaddexp(stats.norm.logpdf(yy, -a, 1), stats.norm.logpdf(yy, a, 1)) + np.log(0.5)


GHZ, GHW = np.polynomial.hermite_e.hermegauss(120)
GHW = GHW / np.sqrt(2 * np.pi)


def kl_q_p(a, m, s2):
    """KL(N(m,s2)||p_a) by 120-node Gauss-Hermite quadrature under the Gaussian (log p_a is smooth)."""
    s = np.sqrt(s2)
    return float(-0.5 * np.log(2 * np.pi * np.e * s2) - np.sum(GHW * pa_logpdf(m + s * GHZ, a)))


def kl_q_p_quad(a, m, s2):
    s = np.sqrt(s2)
    f = lambda yy: stats.norm.pdf(yy, m, s) * (stats.norm.logpdf(yy, m, s) - pa_logpdf(yy, a))
    return integrate.quad(f, m - 40 * s, m + 40 * s, limit=400, epsabs=1e-12)[0]


def bound(a):
    return a ** 2 - a * np.sqrt(2 * (1 + a ** 2) / np.pi) - 0.5 * np.log(1 + a ** 2)


print(" a     KL(q_a||p_a)   bound      KL(N(a,1)||p_a)  min reverse KL over N(m,s2) [m*, s2*]")
viol = 0
for a in [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
    exact = kl_q_p_quad(a, 0.0, 1 + a ** 2)
    comp = kl_q_p_quad(a, a, 1.0)
    viol += exact < bound(a) - 1e-10
    best = optimize.minimize(lambda th: kl_q_p(a, th[0], np.exp(th[1])), x0=[a, 0.0], method="Nelder-Mead",
                             options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400})
    best0 = optimize.minimize(lambda th: kl_q_p(a, th[0], np.exp(th[1])), x0=[0.0, np.log(1 + a ** 2)], method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 400})
    bb = best if best.fun < best0.fun else best0
    print(f"{a:4.2f}  {exact:12.6f}  {bound(a):9.5f}  {comp:12.6f}     {bb.fun:10.6f} [{bb.x[0]:.4f}, {np.exp(bb.x[1]):.4f}]")
print("bound violations:", viol)
print("GH vs adaptive quadrature check at a=2:", abs(kl_q_p(2.0, 0.0, 5.0) - kl_q_p_quad(2.0, 0.0, 5.0)),
      abs(kl_q_p(2.0, 2.0, 1.0) - kl_q_p_quad(2.0, 2.0, 1.0)))
cross_bound = optimize.brentq(lambda a: bound(a) - np.log(2), 1.0, 10.0)
cross_exact = optimize.brentq(lambda a: kl_q_p_quad(a, 0.0, 1 + a ** 2) - kl_q_p_quad(a, a, 1.0), 0.3, 6.0, xtol=1e-6)
print("bound exceeds log 2 for a >", cross_bound, "; exact KL(q_a||p_a) exceeds KL(N(a,1)||p_a) for a >", cross_exact)


def g_stat(a):
    s = np.sqrt(1 + a ** 2)
    def sech2(z):
        e = np.exp(-2 * np.abs(z))
        return 4 * e / (1 + e) ** 2

    f = lambda yy: stats.norm.pdf(yy, 0, s) * sech2(a * yy)
    return (1 + a ** 2) * integrate.quad(f, -40 * s, 40 * s, limit=400, epsabs=1e-14)[0] - 1


ga = np.array([g_stat(a) for a in np.linspace(0.05, 8, 160)])
print("stationarity g(a) on a in [0.05, 8]: max =", ga.max(), " (negative means sigma^2 = 1+a^2 is not stationary)")
aa = sp.symbols("a", positive=True)
zz = sp.symbols("z", real=True)
sech2_series = sp.series(sp.sech(zz) ** 2, zz, 0, 12).removeO()
sig2 = 1 + aa ** 2
Eser = sum(sech2_series.coeff(zz, 2 * k) * aa ** (2 * k) * sp.factorial2(2 * k - 1) * sig2 ** k for k in range(0, 6))
g_series = sp.series(sp.expand(sig2 * Eser - 1), aa, 0, 11).removeO()
print("g(a) series (sympy, Gaussian moments) =", g_series)
for a_ in [0.05, 0.1, 0.2, 0.3]:
    print(f"  g({a_}) = {g_stat(a_):.6e}, series = {float(g_series.subs(aa, a_)):.6e}")

# readout loss
t, c, s = sp.symbols("t c s", positive=True)
Dn = 1 - t ** 2 + c
logf = -t ** 2 / (2 * Dn) - sp.log(2 * sp.pi * Dn) / 2
print("d/dt log (p_t K_c)(0) + t^3/(1-t^2+c)^2 =", sp.simplify(sp.diff(logf, t) + t ** 3 / Dn ** 2))
ratio_cf = sp.series(sp.cos(t * s) * sp.exp(t ** 2 * s ** 2 / 2), s, 0, 8).removeO()
print("phi_{p_t}(s)/phi_{N(0,1)}(s) series =", sp.expand(ratio_cf))
for tt, cc in [(0.5, 1.0), (0.9, 0.1), (0.2, 3.0)]:
    f0 = np.exp(-tt ** 2 / (2 * (1 - tt ** 2 + cc))) / np.sqrt(2 * np.pi * (1 - tt ** 2 + cc))
    g0 = 1 / np.sqrt(2 * np.pi * (1 + cc))
    d = 0.1
    Pp = 0.5 * sum(stats.norm.cdf(d, sgn * tt, np.sqrt(1 - tt ** 2 + cc)) - stats.norm.cdf(-d, sgn * tt, np.sqrt(1 - tt ** 2 + cc)) for sgn in (1, -1))
    Pq = stats.norm.cdf(d, 0, np.sqrt(1 + cc)) - stats.norm.cdf(-d, 0, np.sqrt(1 + cc))
    print(f"t={tt}, c={cc}: densities at 0: {f0:.6f} vs {g0:.6f};  P(|O|<0.1): {Pp:.6f} vs {Pq:.6f}")

# joint Fisher of the label/result family
P_t = 1 - t ** 2
coef = sp.Rational(1, 2) * 2 * (1 / P_t + sp.Rational(1, 2) * (sp.diff(P_t, t) / P_t) ** 2)
print("joint Fisher coefficient - (1+t^2)/(1-t^2)^2 =", sp.simplify(coef - (1 + t ** 2) / (1 - t ** 2) ** 2))
yy = sp.symbols("y", real=True)
m4 = sp.simplify(sp.Rational(1, 2) * sum((sgn * t) ** 4 + 6 * (sgn * t) ** 2 * P_t + 3 * P_t ** 2 for sgn in (1, -1)))
print("fourth moment =", sp.expand(m4))
for tt in [0.3, 0.6, 0.9]:
    sc = lambda y_, sgn: (stats.norm.logpdf(y_, sgn * tt + 1e-6 * sgn, np.sqrt(1 - (tt + 1e-6) ** 2)) -
                          stats.norm.logpdf(y_, sgn * tt - 1e-6 * sgn, np.sqrt(1 - (tt - 1e-6) ** 2))) / 2e-6
    fj = sum(0.5 * integrate.quad(lambda y_: stats.norm.pdf(y_, sgn * tt, np.sqrt(1 - tt ** 2)) * sc(y_, sgn) ** 2,
                                  -20, 20, limit=400)[0] for sgn in (1, -1))
    lpt = lambda y_, t_: np.logaddexp(stats.norm.logpdf(y_, t_, np.sqrt(1 - t_ ** 2)),
                                      stats.norm.logpdf(y_, -t_, np.sqrt(1 - t_ ** 2))) + np.log(0.5)
    scm = lambda y_: (lpt(y_, tt + 1e-6) - lpt(y_, tt - 1e-6)) / 2e-6
    fm = integrate.quad(lambda y_: np.exp(lpt(y_, tt)) * scm(y_) ** 2, -12, 12, limit=400)[0]
    print(f"t={tt}: joint Fisher by quadrature {fj:.6f} vs formula {(1 + tt ** 2) / (1 - tt ** 2) ** 2:.6f}; "
          f"result-marginal Fisher {fm:.6f}; Gaussian carrier Fisher 0")
