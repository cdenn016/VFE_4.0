r"""D13-ERAPUB s05: the binary two-agent example of sec:public-phase-response and the null-flow
control of sec:public-status (chapters/09_public_compression.tex).

(a) sympy: expansion eq:public-phase-binary-expansion of the fine binary equation
    eq:public-phase-binary-fine, including the absence of order-4 terms in mdot - f_b(m).
(b) coupled flow: phase by forward-backward truncation Pi_T = phi_{-T}(rho(Phi_T x)) for
    J in {0.2, 0.5, 0.9, 0.99}, with rho the arithmetic mean of mean parameters and with the
    Lipschitz, nonsmooth symmetric retraction rho_c = mean + c |x_1 - x_2|; compare
    (pbar - m)/(m a^2) with c(J) = 3(2-J)/(2(5-J)) along m = a.
(c) null flow (peer costs deleted; each agent follows f_b): the Koenigs function
    h(x) = x exp(int_0^x [-lam/f_b(s) - 1/s] ds), lam = 1-J, linearizes f_b; then
    Pi0_c(x) = h^{-1}( (h(x_1)+h(x_2))/2 + c |h(x_1)-h(x_2)| )
    is, for every c, an exact Lipschitz intertwining retraction onto the diagonal flow with
    nontrivial fibers (c = 0 is analytic). Checks: intertwining residuals; the truncated limit
    phi_{-T}(rho(Phi0_T x)) converges to Pi0_0 for the smooth mean and to Pi0_c for rho_c.
Single-threaded, short.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


def part_a():
    m, a, J = sp.symbols('m a J', real=True)
    x1, x2 = m + a, m - a

    def fine(xi, xj):
        return -(1 - xi**2) * (2 * sp.atanh(xi) - sp.atanh(xj) - J * xi) - (xi - xj)

    fb = lambda x: -(1 - x**2) * (sp.atanh(x) - J * x)
    mdot = (fine(x1, x2) + fine(x2, x1)) / 2
    adot = (fine(x1, x2) - fine(x2, x1)) / 2
    t = sp.symbols('t')
    g = sp.series((mdot - fb(m)).subs({m: t * m, a: t * a}), t, 0, 6).removeO()
    print('(a) mdot - f_b(m) through order 5:', sp.factor(sp.expand(g.subs(t, 1))))
    ad = sp.series(adot.subs({m: t * m, a: t * a}), t, 0, 3).removeO()
    print('(a) adot through order 2:', sp.expand(ad.subs(t, 1)))


def fb(x, J):
    return -(1 - x**2) * (np.arctanh(x) - J * x)


def fine_rhs(t, x, J):
    x1, x2 = x
    return [-(1 - x1**2) * (2 * np.arctanh(x1) - np.arctanh(x2) - J * x1) - (x1 - x2),
            -(1 - x2**2) * (2 * np.arctanh(x2) - np.arctanh(x1) - J * x2) - (x2 - x1)]


def null_rhs(t, x, J):
    return [fb(x[0], J), fb(x[1], J)]


def back(xT, J, T):
    s = solve_ivp(lambda t, y: [-fb(y[0], J)], (0, T), [xT], method='DOP853', rtol=1e-13, atol=1e-30)
    return s.y[0, -1]


def phase_T(x0, J, T, rhs, c=0.0):
    s = solve_ivp(rhs, (0, T), list(x0), args=(J,), method='DOP853', rtol=1e-13, atol=1e-30)
    x1, x2 = s.y[:, -1]
    rho = 0.5 * (x1 + x2) + c * abs(x1 - x2)
    return back(rho, J, T)


def part_b():
    for J in (0.2, 0.5, 0.9, 0.99):
        cJ = 3 * (2 - J) / (2 * (5 - J))
        ratios = []
        for s in (0.02, 0.01, 0.005):
            p = phase_T((2 * s, 0.0), J, 4.0, fine_rhs)
            ratios.append((p - s) / s**3)
        p6 = phase_T((0.02, 0.0), J, 6.0, fine_rhs)
        p4 = phase_T((0.02, 0.0), J, 4.0, fine_rhs)
        pc = phase_T((0.02, 0.0), J, 6.0, fine_rhs, c=0.3)
        print(f'(b) coupled J={J}: c(J)={cJ:.6f}; (pbar-m)/(m a^2) at s=.02,.01,.005: {[f"{r:.6f}" for r in ratios]}; '
              f'|Pi_6-Pi_4| = {abs(p6-p4):.1e}; |Pi_6(rho_c=0.3) - Pi_6(mean)| = {abs(pc-p6):.1e}')


def koenigs(J):
    lam = 1 - J

    def integrand(s):
        if abs(s) < 1e-8:
            return 0.0
        return -lam / fb(s, J) - 1 / s

    def h(x):
        return x * np.exp(quad(integrand, 0, x, epsabs=1e-14, epsrel=1e-13, limit=200)[0])

    def hinv(y):
        return brentq(lambda x: h(x) - y, -0.9, 0.9, xtol=1e-15) if y != 0 else 0.0
    return h, hinv, lam


def flow_b(x, J, t):
    s = solve_ivp(lambda tt, y: [fb(y[0], J)], (0, t), [x], method='DOP853', rtol=1e-13, atol=1e-30)
    return s.y[0, -1]


def part_c():
    J = 0.2
    h, hinv, lam = koenigs(J)
    x0 = np.array([0.04, 0.0])
    # Koenigs linearization check
    for t in (0.5, 2.0):
        print(f'(c) h(phi_t x) - e^(-lam t) h(x) at x=0.04, t={t}: {h(flow_b(0.04, J, t)) - np.exp(-lam*t)*h(0.04):.2e}')
    for c in (0.0, 0.3):
        P = lambda x: hinv(0.5 * (h(x[0]) + h(x[1])) + c * abs(h(x[0]) - h(x[1])))
        for t in (1.0, 3.0):
            xt = np.array([flow_b(x0[0], J, t), flow_b(x0[1], J, t)])
            res = P(xt) - flow_b(P(x0), J, t)
            print(f'(c) null flow, c={c}: intertwining residual Pi0_c(Phi0_t x) - phi_t(Pi0_c x) at t={t}: {res:.2e}; Pi0_c(x) = {P(x0):.10f}')
        # a second configuration on the same fiber of Pi0_c (distinct from x0 and from the diagonal)
    P0 = lambda x: hinv(0.5 * (h(x[0]) + h(x[1])))
    target = P0(x0)
    x1b = 0.03
    x2b = hinv(2 * h(target) - h(x1b))
    print(f'(c) null flow fiber: Pi0_0(0.04, 0) = {target:.10f} = Pi0_0({x1b}, {x2b:.10f}) = {P0(np.array([x1b, x2b])):.10f}')
    for T in (8.0, 16.0, 24.0, 32.0):
        p_mean = phase_T(tuple(x0), J, T, null_rhs)
        p_lip = phase_T(tuple(x0), J, T, null_rhs, c=0.3)
        print(f'(c) null flow truncation T={T}: phi_-T(mean) - Pi0_0 = {p_mean - target:+.3e}; '
              f'phi_-T(rho_0.3) - Pi0_0.3 = {p_lip - hinv(0.5*(h(x0[0])+h(x0[1])) + 0.3*abs(h(x0[0])-h(x0[1]))):+.3e}')
    # contrast: the coupled flow's phase at the same point, by the smooth and the Lipschitz retraction
    pc0 = phase_T(tuple(x0), J, 6.0, fine_rhs)
    pc3 = phase_T(tuple(x0), J, 6.0, fine_rhs, c=0.3)
    print(f'(c) coupled flow at the same x: Pi (mean) = {pc0:.10f}, Pi (rho_0.3) = {pc3:.10f}, difference {abs(pc3-pc0):.1e}')
    print(f'(c) null flow:                 Pi0 (mean) = {target:.10f}, Pi0 (rho_0.3) = '
          f'{hinv(0.5*(h(x0[0])+h(x0[1])) + 0.3*abs(h(x0[0])-h(x0[1]))):.10f}')


if __name__ == '__main__':
    part_a(); part_b(); part_c()
