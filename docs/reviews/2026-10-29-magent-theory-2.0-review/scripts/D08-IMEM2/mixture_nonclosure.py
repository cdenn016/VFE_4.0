r"""Numerical checks of imem:mixture-nonclosure.

Lens D08-IMEM2.  Scope: appendices/interacting_memory.tex lines 582-615 and
companion/c04_interacting_memory.tex lines 194-211.

Checks (numerical, not proofs):
 1. w_+(d) = 1/(1+exp(-2 alpha d/(L1+N1))) against the direct label posterior.
 2. k(h|d) = c_- + (c_+ - c_-) w_+(d) against the direct mixture density.
 3. tail limits of the receiver likelihood l_0(h|x) -> c_+ (x->+inf), c_- (x->-inf).
 4. the excluded message h = 0: c_+ = c_-, l_0 constant, posterior = g_0 (Gaussian).
 5. common-state check: the D1-marginal of lambda_1 is EXACTLY a two-component
    Gaussian mixture, while the D0-marginal has density ratio to the Gaussian
    lambda_0^{D0} with distinct tail limits, so lambda_1 itself is not a finite
    Gaussian mixture (by the manuscript's own tail lemma applied to D0).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import expit
from scipy.stats import norm

alpha, L0, L1, S, m, Q, R, N0, N1 = 1.0, 1.0, 1.5, 0.6, 1.2, 0.5, 0.3, 0.4, 0.7
assert S**2 < L0 * L1
T = L1 + N1
U = L1 - S**2 / L0 + N1


def w_plus_direct(d: float) -> float:
    a = 0.5 * norm.pdf(d, alpha, np.sqrt(T))
    b = 0.5 * norm.pdf(d, -alpha, np.sqrt(T))
    return a / (a + b)


def w_plus(d: float) -> float:
    return float(expit(2 * alpha * d / T))


ds = np.linspace(-6, 6, 25)
print("1. max |w_+ formula - direct|              :", max(abs(w_plus(d) - w_plus_direct(d)) for d in ds))


def k_direct(h: float, d: float) -> float:
    wp = w_plus_direct(d)
    return wp * norm.pdf(h, m, np.sqrt(Q + R)) + (1 - wp) * norm.pdf(h, -m, np.sqrt(Q + R))


h = 0.8
cp, cm = norm.pdf(h, m, np.sqrt(Q + R)), norm.pdf(h, -m, np.sqrt(Q + R))
print("2. c_+ > c_- > 0 at h=0.8                  :", cp > cm > 0, f"(c_+={cp:.6f}, c_-={cm:.6f})")
print("   max |k formula - direct|                :",
      max(abs(cm + (cp - cm) * w_plus(d) - k_direct(h, d)) for d in ds))


def ell0(hh: float, x: float) -> float:
    c_p, c_m = norm.pdf(hh, m, np.sqrt(Q + R)), norm.pdf(hh, -m, np.sqrt(Q + R))
    tot = 0.0
    for bb in (-1.0, 1.0):
        mu = bb * alpha + (S / L0) * x
        f = lambda z: (c_m + (c_p - c_m) * w_plus(mu + np.sqrt(U) * z)) * norm.pdf(z)
        val, _ = integrate.quad(f, -12, 12, epsabs=1e-13, limit=200)
        tot += 0.5 * val
    return tot


print("3. receiver likelihood tails (h=0.8):")
for x in (-60, -30, -10, 0, 10, 30, 60):
    print(f"   x={x:4d}: l_0 = {ell0(h, x):.10f}")
print(f"   targets  c_- = {cm:.10f}, c_+ = {cp:.10f}")

# 4. h = 0
print("4. h=0: c_+ - c_- =", norm.pdf(0, m, np.sqrt(Q + R)) - norm.pdf(0, -m, np.sqrt(Q + R)),
      "; l_0(0|x) over x in {-20,0,20}:", [round(ell0(0.0, x), 12) for x in (-20, 0, 20)])

# 5. common state lambda_1 marginals
#   D1 marginal: lambda_0^{D1}(d) k(h|d) = (c_-/2) N(d;-alpha,T) + (c_+/2) N(d;alpha,T)
lhs = lambda d: (0.5 * norm.pdf(d, alpha, np.sqrt(T)) + 0.5 * norm.pdf(d, -alpha, np.sqrt(T))) * (cm + (cp - cm) * w_plus(d))
rhs = lambda d: 0.5 * cm * norm.pdf(d, -alpha, np.sqrt(T)) + 0.5 * cp * norm.pdf(d, alpha, np.sqrt(T))
print("5. D1-marginal of lambda_1 is exactly a 2-component Gaussian mixture: max residual",
      max(abs(lhs(d) - rhs(d)) for d in ds))
#   D0 marginal: lambda_1^{D0}(d0)/lambda_0^{D0}(d0) = P(H=h|D0=d0)/P(H=h)
#   P(H=h|D0=d0) = E[l_0(h|X0) | D0=d0],  X0|D0=d0 ~ N(L0 d0/(L0+N0), L0 N0/(L0+N0))


def pH_given_d0(d0: float) -> float:
    mu, sd = L0 * d0 / (L0 + N0), np.sqrt(L0 * N0 / (L0 + N0))
    f = lambda z: ell0(h, mu + sd * z) * norm.pdf(z)
    val, _ = integrate.quad(f, -10, 10, epsabs=1e-11, limit=100)
    return val


for d0 in (-80, -40, 40, 80):
    print(f"   P(H=h|D0={d0:4d}) = {pH_given_d0(d0):.10f}")
print("   (tends to c_- and c_+: distinct tail limits of the D0-marginal ratio; the D0 prior")
print("    marginal is the single Gaussian N(0, L0+N0), so the tail lemma of the proof applies)")
