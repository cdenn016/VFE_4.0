r"""Numerical checks of imem:gaussian-receiver and imem:projection-loss.

Lens D08-IMEM2.  Scope: appendices/interacting_memory.tex lines 484-506 and
655-661.  Numerical quadrature; agreement supports the identities, it does
not prove them.

(a) Receiver likelihood l_0(h|a,x) of imem:gaussian-local-likelihood versus a
    direct computation in which P(b|a,x) and the conditional law of X_1 are
    obtained from the bivariate cue density itself (no use of u_ab, U_ab) and
    the sender record D_1 is integrated explicitly (2-D quadrature).
(b) KL(Pi_{A,Y} || w x q_G) = I(A;Y) + KL(Pi_Y || q_G) for a two-label
    one-dimensional Gaussian mixture, with q_G the moment match and also with
    an arbitrary Gaussian q (the identity does not need the moment match).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal, norm

# two receiver labels a, two sender labels b, scalar cues/results
p = np.array([[0.3, 0.2], [0.1, 0.4]])                  # p_ab
pi0 = p.sum(axis=1)
nu0 = np.array([-0.5, 0.7])
nu1 = np.array([1.0, -1.0])
L0 = np.array([1.0, 0.6])
L1 = np.array([0.8, 1.3])
Sab = np.array([[0.4, -0.3], [0.2, 0.5]])
assert np.all(Sab**2 < np.outer(L0, L1))
pi1 = p.sum(axis=0)
om1, F1, Q1 = np.array([0.5, -0.2]), np.array([1.5, -0.7]), np.array([0.4, 0.9])
N1, Bc, Rn = 0.6, 1.0, 0.25
T = L1 + N1
G = L1 / T
Pb = L1 - L1**2 / T
SigY = Q1 + F1**2 * Pb


def w(d: float) -> np.ndarray:
    v = pi1 * norm.pdf(d, nu1, np.sqrt(T))
    return v / v.sum()


def k(h: float, d: float) -> float:
    mb = om1 + F1 * G * (d - nu1)
    return float(np.sum(w(d) * norm.pdf(h, Bc * mb, np.sqrt(Rn + Bc**2 * SigY))))


def ell_formula(h: float, a: int, x: float) -> float:
    tot = 0.0
    for b in (0, 1):
        u = nu1[b] + Sab[a, b] / L0[a] * (x - nu0[a])
        U = N1 + L1[b] - Sab[a, b] ** 2 / L0[a]
        val, _ = integrate.quad(lambda d: norm.pdf(d, u, np.sqrt(U)) * k(h, d), u - 12 * np.sqrt(U), u + 12 * np.sqrt(U),
                                epsabs=1e-12, limit=200)
        tot += p[a, b] / pi0[a] * val
    return tot


def ell_direct(h: float, a: int, x: float) -> float:
    # P(b | a, x) from the joint label/cue density
    dens = np.array([p[a, b] * norm.pdf(x, nu0[a], np.sqrt(L0[a])) for b in (0, 1)])
    pb = dens / dens.sum()
    tot = 0.0
    for b in (0, 1):
        mv = multivariate_normal([nu0[a], nu1[b]], [[L0[a], Sab[a, b]], [Sab[a, b], L1[b]]])
        cx = lambda x1: mv.pdf([x, x1]) / norm.pdf(x, nu0[a], np.sqrt(L0[a]))      # p(x1 | a, b, x)
        f = lambda d, x1: cx(x1) * norm.pdf(d, x1, np.sqrt(N1)) * k(h, d)
        val, _ = integrate.dblquad(f, -9, 9, lambda _: -12, lambda _: 12, epsabs=1e-9)
        tot += pb[b] * val
    return tot


print("(a) receiver likelihood, formula vs direct 2-D integration:")
for (h, a, x) in [(0.3, 0, -1.0), (1.7, 1, 0.4), (-0.9, 0, 2.0)]:
    lf, ld = ell_formula(h, a, x), ell_direct(h, a, x)
    print(f"    h={h:5.2f} a={a} x={x:5.2f}: formula={lf:.10f} direct={ld:.10f} |diff|={abs(lf - ld):.2e}")

# (b) projection-loss identity
wA = np.array([0.35, 0.65])
mA, sA = np.array([-1.0, 1.5]), np.array([0.6, 1.1])
mbar = float(np.sum(wA * mA))
Sbar = float(np.sum(wA * (sA**2 + (mA - mbar) ** 2)))


def piY(y: float) -> float:
    return float(np.sum(wA * norm.pdf(y, mA, sA)))


def kl_1d(f, g, lo=-25, hi=25) -> float:
    val, _ = integrate.quad(lambda y: f(y) * (np.log(f(y)) - np.log(g(y))) if f(y) > 0 else 0.0, lo, hi,
                            epsabs=1e-13, limit=400)
    return val


for label, (mq, vq) in (("moment match q_G", (mbar, Sbar)), ("arbitrary q", (0.3, 2.0))):
    q = lambda y, mq=mq, vq=vq: norm.pdf(y, mq, np.sqrt(vq))
    lhs = sum(wA[a] * kl_1d(lambda y, a=a: norm.pdf(y, mA[a], sA[a]), q) for a in (0, 1))
    mi = sum(wA[a] * kl_1d(lambda y, a=a: norm.pdf(y, mA[a], sA[a]), piY) for a in (0, 1))
    rhs = mi + kl_1d(piY, q)
    print(f"(b) {label:17s}: KL(Pi||w x q)={lhs:.12f}, I(A;Y)+KL(Pi_Y||q)={rhs:.12f}, |diff|={abs(lhs - rhs):.2e}")
