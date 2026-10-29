r"""GAP06-S2 focus question 4: explicit remainder for eq:public-phase-binary-parent
(chapters/09_public_compression.tex:142-168).

System (eq:public-phase-binary-fine), x_{1,2} = m +- a:
  x_i' = -(1-x_i^2)[2 atanh x_i - atanh x_j - J x_i] - (x_i - x_j).
Phase parent Pi(m,a): the unique Lipschitz intertwining retraction (cor:public-phase-unique); in the
invariant binary subfamily it solves the invariance equation  Pi_m m' + Pi_a a' = f_b(Pi),
Pi(m,0) = m, Pi odd in m and even in a.
Part 1 (sympy, exact): series of m', a' to total degree 9; coefficients c_pq of
  Pi = m + sum c_pq m^p a^q  (p odd, q even >= 2, p+q <= 9)
from the invariance equation; check c_12 = 3(2-J)/(2(5-J)); closed forms of c_32, c_14.
Part 2 (numerics): Pi computed independently by the Koenigs formula
  h(Pi) = lim_t e^{(1-J)t} h(m(t)),  h' f_b = -(1-J) h, h(0)=0, h'(0)=1,
with DOP853 forward integration (rtol 1e-14) and mpmath quadrature for h; compare
  R(m,a) = Pi - m - c_12 m a^2  with  c_32 m^3 a^2 + c_14 m a^4  and report
  K_meas = max |R| / (|m| a^2 (m^2+a^2))  on the box |m|,|a| <= rho.
Part 3: the constants of the analytic bound in the report (K_Phi, K_Gamma, K_mu, K_h) as grid
suprema on the box (not interval certified).  Single-threaded, < 30 s.
"""
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import solve_ivp

from sympy import ring, QQ
Jsym = sp.Symbol('J')


def truncate(P, order):
    return P.ring.from_dict({k: v for k, v in P.items() if sum(k) <= order})


def atanh_ser(x, order):
    out = x * 0; xp = x; x2 = truncate(x * x, order)
    for k in range((order + 1) // 2 + 1):
        out += xp * QQ(1, 2 * k + 1)
        xp = truncate(xp * x2, order)
    return truncate(out, order)


def phase_series(Jval, ORDER):
    """Return dict {(p,q): c_pq} solving the invariance equation to total degree ORDER."""
    if Jval is None:
        K = QQ.frac_field(Jsym)
        Jv = K.convert(Jsym)
    else:
        K = QQ
        Jv = K.convert(Jval)
    R, mm, aa = ring("m,a", K)
    x1_, x2_ = mm + aa, mm - aa
    A1_, A2_ = atanh_ser(x1_, ORDER), atanh_ser(x2_, ORDER)
    xd1_ = truncate(-(1 - x1_**2) * (2 * A1_ - A2_ - Jv * x1_) - (x1_ - x2_), ORDER)
    xd2_ = truncate(-(1 - x2_**2) * (2 * A2_ - A1_ - Jv * x2_) - (x2_ - x1_), ORDER)
    md = (xd1_ + xd2_) * QQ(1, 2); ad = (xd1_ - xd2_) * QQ(1, 2)
    fb_m = truncate(-(1 - mm**2) * (atanh_ser(mm, ORDER) - Jv * mm), ORDER)
    lam1 = -(1 - Jv); lam2 = -(5 - Jv)
    Pi_ = mm * 1
    cs = {}
    for tot in range(3, ORDER + 1, 2):
        Pm = Pi_.diff(mm); Pa = Pi_.diff(aa)
        fbPi = truncate(-(1 - truncate(Pi_**2, ORDER)) * (atanh_ser(Pi_, ORDER) - Jv * Pi_), ORDER)
        inv_ = truncate(Pm * md + Pa * ad - fbPi, ORDER)
        for p in range(1, tot, 2):
            q = tot - p
            if q < 2 or q % 2:
                continue
            r = inv_.coeff(mm**p * aa**q)
            # linear part of the equation for c_pq: (p lam1 + q lam2 - lam1) c_pq + r = 0
            cpq = -r / (p * lam1 + q * lam2 - lam1)
            cs[(p, q)] = cpq
            Pi_ = Pi_ + cpq * mm**p * aa**q
    Pm = Pi_.diff(mm); Pa = Pi_.diff(aa)
    fbPi = truncate(-(1 - truncate(Pi_**2, ORDER)) * (atanh_ser(Pi_, ORDER) - Jv * Pi_), ORDER)
    inv_ = truncate(Pm * md + Pa * ad - fbPi, ORDER)
    gb_ = md - fb_m
    return cs, inv_, gb_, ad, K


print("Part 1: invariance equation, symbolic J, total degree 5")
cs5, inv5, gb5, ad5, K5 = phase_series(None, 5)
print("  degree <= 5 part of m' - f_b(m):", sp.factor(gb5.as_expr()))
print("  a' to degree 3:", sp.factor(truncate(ad5, 3).as_expr()))
print("  invariance residual after substitution, degree <= 5 (should be 0):", inv5.as_expr())
c12 = sp.factor(K5.to_sympy(cs5[(1, 2)]))
c32 = sp.factor(K5.to_sympy(cs5[(3, 2)]))
c14 = sp.factor(K5.to_sympy(cs5[(1, 4)]))
J = Jsym
print("  c_12 =", c12, "; equals 3(2-J)/(2(5-J)):", sp.simplify(c12 - 3 * (2 - J) / (2 * (5 - J))) == 0)
print("  c_32 =", c32)
print("  c_14 =", c14)
print("\n  degree-7 coefficients at rational J (exact rationals, printed as floats):")
for Jv in (sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(9, 10), sp.Rational(99, 100)):
    cs7, inv7, _, _, _ = phase_series(Jv, 7)
    print("  J = %-6s c_12 = %.6f c_32 = %.6f c_14 = %.6f | c_52 = %.5f c_34 = %.5f c_16 = %.5f | residual %s"
          % (Jv, float(cs7[(1, 2)]), float(cs7[(3, 2)]), float(cs7[(1, 4)]), float(cs7[(5, 2)]),
             float(cs7[(3, 4)]), float(cs7[(1, 6)]), "0" if inv7 == 0 else "nonzero"))

print("\nPart 2: independent numerical phase via Koenigs function and forward integration")
mp.mp.dps = 20


def make_h(Jv):
    lam = 1 - Jv
    fbm = lambda s: -(1 - s**2) * (mp.atanh(s) - Jv * s)

    def integrand(s):
        if abs(s) < mp.mpf('1e-6'):
            # series: -lam/f_b(s) - 1/s = -beta s + O(s^3), beta = (1/3 - lam... ) computed numerically
            s1 = mp.mpf('1e-6') * (1 if s >= 0 else -1)
            return (-lam / fbm(s1) - 1 / s1) * (s / s1)
        return -lam / fbm(s) - 1 / s

    def h(x):
        x = mp.mpf(x)
        return x * mp.exp(mp.quad(integrand, [0, x]))
    return h


def phase(m0, a0, Jv, h, t0=9.0):
    lam = 1 - Jv

    def rhs(t, y):
        x1_, x2_ = y[0] + y[1], y[0] - y[1]
        d1 = -(1 - x1_**2) * (2 * np.arctanh(x1_) - np.arctanh(x2_) - Jv * x1_) - (x1_ - x2_)
        d2 = -(1 - x2_**2) * (2 * np.arctanh(x2_) - np.arctanh(x1_) - Jv * x2_) - (x2_ - x1_)
        return [(d1 + d2) / 2, (d1 - d2) / 2]
    sol_ = solve_ivp(rhs, (0, t0), [m0, a0], method='DOP853', rtol=1e-14, atol=1e-20)
    mt = sol_.y[0, -1]
    target = mp.exp(lam * t0) * h(mt)
    return float(mp.findroot(lambda x: h(x) - target, mp.mpf(m0)))


for Jv in (0.2, 0.5, 0.9):
    h = make_h(Jv)
    Jr = sp.Rational(Jv).limit_denominator(100)
    c12v = float(c12.subs(J, Jr)); c32v = float(c32.subs(J, Jr)); c14v = float(c14.subs(J, Jr))
    print("  J = %.2f: c_12 = %.6f, c_32 = %.6f, c_14 = %.6f" % (Jv, c12v, c32v, c14v))
    Kmax = 0.0; worst = None; err5 = 0.0
    for rho in (0.1,):
        for mm in (0.02, 0.05, 0.1, -0.07):
            for aa in (0.02, 0.05, 0.1):
                Pi_num = phase(mm, aa, Jv, h)
                R = Pi_num - mm - c12v * mm * aa**2
                lead = c32v * mm**3 * aa**2 + c14v * mm * aa**4
                K = abs(R) / (abs(mm) * aa**2 * (mm**2 + aa**2))
                if K > Kmax:
                    Kmax, worst = K, (mm, aa)
                rel = abs(R - lead) / (abs(mm) * aa**2 * (mm**2 + aa**2)**2)
                err5 = max(err5, rel)
                print("    m = %+.3f a = %.3f : R = %+.6e, c32 m^3a^2 + c14 m a^4 = %+.6e, R/(|m|a^2 r2^2) = %+.4f"
                      % (mm, aa, R, lead, R / (abs(mm) * aa**2 * (mm**2 + aa**2))))
    print("    K_meas = max |R|/(|m| a^2 (m^2+a^2)) on the grid = %.4f at %s; max |R - lead|/(|m|a^2(m^2+a^2)^2) = %.3f"
          % (Kmax, worst, err5))

print("\nPart 3: constants of the Koenigs-integral bound (grid suprema on the box Q_rho, not certified)")
# exact functions: m' = m(-lam + mu), a' = a Gamma, m' - f_b(m) = m a^2 Phi, h'(m) = -lam h(m)/f_b(m)
Jv = 0.5; lam = 1 - Jv; nu = 5 - Jv; Phi0 = 6 - 3 * Jv; rho = 0.2
c12v = 3 * (2 - Jv) / (2 * (5 - Jv))


def fields(mm, aa):
    x1_, x2_ = mm + aa, mm - aa
    d1 = -(1 - x1_**2) * (2 * np.arctanh(x1_) - np.arctanh(x2_) - Jv * x1_) - (x1_ - x2_)
    d2 = -(1 - x2_**2) * (2 * np.arctanh(x2_) - np.arctanh(x1_) - Jv * x2_) - (x2_ - x1_)
    return (d1 + d2) / 2, (d1 - d2) / 2


def fbn(x):
    return -(1 - x**2) * (np.arctanh(x) - Jv * x)


g = np.linspace(-rho, rho, 161); g = g[np.abs(g) > 1e-3]
Mg, Ag = np.meshgrid(g, g)
md_, ad_ = fields(Mg, Ag)
s2 = Mg**2 + Ag**2
PhiG = (md_ - fbn(Mg)) / (Mg * Ag**2)
GamG = ad_ / Ag
muG = md_ / Mg + lam
K_Phi = float(np.max(np.abs(PhiG - Phi0) / s2))
K_Gam = float(np.max(np.abs(GamG + nu) / s2))
K_mu = float(np.max(np.abs(muG) / s2))
h = make_h(Jv)
hs = [(x, float(-lam * h(x) / fbn(x))) for x in np.linspace(1e-3, rho, 60)]
K_h = max(abs(hp - 1) / x**2 for x, hp in hs)
print("  J = %.2f, rho = %.2f: K_Phi = %.4f, K_Gamma = %.4f, K_mu = %.4f, K_h = %.4f" % (Jv, rho, K_Phi, K_Gam, K_mu, K_h))
for (m0, a0) in ((0.02, 0.02), (0.05, 0.05), (0.1, 0.05), (0.05, 0.1)):
    nup = nu - 2 * K_Gam * rho**2
    mhat = abs(m0) * np.exp(K_mu * a0**2 / (2 * nup))
    assert lam > K_mu * rho**2 and mhat <= rho
    al = (K_mu + 2 * K_Gam) * mhat**2; be = (K_mu + 2 * K_Gam) * a0**2 / (2 * nup)
    fac = K_Phi * (mhat**2 + a0**2) / Phi0
    Ip = (1 + K_h * mhat**2) * (1 + fac) * np.exp(be) / (2 * nu - al)
    Im = (1 - K_h * mhat**2) * (1 - fac) * np.exp(-be) / (2 * nu + al)
    Pi_num = phase(m0, a0, Jv, h)
    rs = max(abs(m0), abs(Pi_num))
    up = 2 * nu * Ip / (1 - K_h * rs**2) - 1; lo = 2 * nu * Im / (1 + K_h * rs**2) - 1
    R = Pi_num - m0 - c12v * m0 * a0**2
    print("  (m0,a0) = (%.2f,%.2f): R/(c m a^2) = %+.3e, rigorous-structure bracket [%+.3e, %+.3e]; "
          "bracket/(m^2+a^2) = [%+.3f, %+.3f]" % (m0, a0, R / (c12v * m0 * a0**2), lo, up,
                                                  lo / (m0**2 + a0**2), up / (m0**2 + a0**2)))
