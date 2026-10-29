r"""GAP10-S3 s06: regularity of the own-block remark (chapters/15_contextual_coupling.tex:257).

The remark claims a locally well-posed own-block field "under the same local regularity
assumptions", i.e. C^3 local potentials E_v (15:72, ctx:f:thm:ode).  Counterexamples in which
the reciprocal potential is C^3 (Part A) or C^infinity (Part C) while an own-block gradient is only
Hoelder-3/4, so the own-block field is not locally Lipschitz and its solutions are not unique.

Part A (chapter 15 receiver split, singleton rows): q_i = N(m_i, 1), R_1 = N(1,1), R_2 = N(-2,1),
  law-dependent translation channels with shift sigma(m_1) = |m_1|^{7/4}:
  D_12 = (m_1 - m_2 - sigma)^2/2, D_21 = (m_2 - m_1 - sigma)^2/2,
  E_v = KL_1 + KL_2 + D_12 + D_21 = (m_1-1)^2/2 + (m_2+2)^2/2 + (m_1-m_2)^2 + |m_1|^{7/2}  (C^3),
  F_1 = KL_1 + D_12, F_2 = KL_2 + D_21; own-block field m_i' = -d_{m_i} F_i; equilibrium (0,-1).
Part C (chapter 3 per-agent free energies, self sources, unit-temperature evolving rows,
  law-dependent evaluators and channels): the shared term phi(m_1) = -k sgn(m_1)|m_1|^{7/4}
  enters agent 1's self and peer costs and, with opposite sign, agent 2's; F_tot is C^infinity.
"""
from __future__ import annotations

import time
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

t0 = time.time()
print("=== Part A: symbolic structure ===", flush=True)
m2 = sp.symbols('m2', real=True)
xp = sp.symbols('x', positive=True)          # x = |m1|
for side, sgn in (("m1>0", 1), ("m1<0", -1)):
    m1v = sgn * xp
    s_ = xp ** sp.Rational(7, 4)
    D12 = (m1v - m2 - s_) ** 2 / 2
    D21 = (m2 - m1v - s_) ** 2 / 2
    E = (m1v - 1) ** 2 / 2 + (m2 + 2) ** 2 / 2 + D12 + D21
    target = (m1v - 1) ** 2 / 2 + (m2 + 2) ** 2 / 2 + (m1v - m2) ** 2 + xp ** sp.Rational(7, 2)
    print(f"  [{side}] E_v - [(m1-1)^2/2 + (m2+2)^2/2 + (m1-m2)^2 + |m1|^(7/2)] =", sp.expand(E - target), flush=True)
f72 = xp ** sp.Rational(7, 2); f74 = xp ** sp.Rational(7, 4)
d72 = [sp.limit(sp.diff(f72, xp, k), xp, 0, '+') for k in range(1, 4)]
d74 = [sp.limit(sp.diff(f74, xp, k), xp, 0, '+') for k in range(1, 3)]
print("  right limits at 0 of d^k |m|^(7/2), k=1..3:", d72, "(even function: odd derivatives change sign, so all are continuous at 0 iff these limits vanish)")
print("  right limits at 0 of d^k |m|^(7/4), k=1..2:", d74, " -> sigma is C^1 but not C^2", flush=True)


def sigma(x):
    return np.abs(x) ** 1.75


def dsigma(x):
    return 1.75 * np.sign(x) * np.abs(x) ** 0.75


def ownblock(t, y):
    a, b = y
    Dl = a - b - sigma(a)
    A1 = (a - 1) + Dl * (1 - dsigma(a))           # d F_1 / d m_1
    A2 = (b + 2) + (b - a - sigma(a))              # d F_2 / d m_2
    return [-A1, -A2]


def reciprocal(t, y):
    a, b = y
    g1 = (a - 1) + 2 * (a - b) + 3.5 * np.sign(a) * np.abs(a) ** 2.5
    g2 = (b + 2) - 2 * (a - b)
    return [-g1, -g2]


print("\n=== Part A: numerics ===")
print("  own-block field at the equilibrium (0,-1):", ownblock(0, [0.0, -1.0]))
for x in [1e-2, 1e-4, 1e-6, 1e-8]:
    q = (ownblock(0, [x, -1.0])[0] - ownblock(0, [0.0, -1.0])[0]) / x
    print(f"  difference quotient of the m1-component at x={x:.0e}: {q:.4e}  (~ (7/4) x^(-1/4) = {1.75*x**-0.25:.4e})")
print("  solutions from (x0, -1) as x0 -> 0 (DOP853, rtol 1e-12):")
ts = [0.5, 1.0, 2.0]
for x0 in [1e-4, 1e-8, 1e-12]:
    sol = solve_ivp(ownblock, (0, 2.0), [x0, -1.0], method='DOP853', rtol=1e-12, atol=1e-30, t_eval=ts, dense_output=False)
    print(f"    x0 = {x0:.0e}: m1(0.5, 1, 2) = {sol.y[0]}")
print("  the x0 -> 0+ limit is a solution from (0,-1) with m1(2) ~ 0.137 > 0 (Arzela-Ascoli limit of the sequence above);")
print("  the constant m(t) = (0,-1) is also a solution (field = 0 there): two distinct solutions -> non-uniqueness.")
sol = solve_ivp(reciprocal, (0, 2.0), [0.0, -1.0], method='DOP853', rtol=1e-12, atol=1e-14, t_eval=[2.0])
print(f"  reciprocal gradient flow of E_v from (0,-1) (C^2 field, unique): m(2) = {sol.y[:, 0]}")

print("\n=== Part C: chapter 3 per-agent free energies with self sources and evolving rows ===")
k = sp.Rational(1, 1)
b11, b22, p11, p12, p21, p22 = sp.symbols('b11 b22 pi11 pi12 pi21 pi22', positive=True)
for side, sgn in (("m1>0", 1), ("m1<0", -1)):
    m1v = sgn * xp
    phi = -k * sgn * xp ** sp.Rational(7, 4)
    s11 = s12 = 1 + xp ** 2
    s22 = s21 = sp.Integer(1)
    g1 = m1v - sp.sqrt(2 * (phi + s11))       # evaluator mean of agent 1 (law dependent)
    t12 = m1v - sp.sqrt(2 * (phi + s12))      # channel output mean for receiver 1 (q_2 = N(0,1) shifted)
    g2 = sp.sqrt(2 * (s22 - phi))             # evaluator mean of agent 2 (depends on m1)
    t21 = sp.sqrt(2 * (s21 - phi))            # channel output mean for receiver 2 (q_1 shifted)
    c11 = (m1v - g1) ** 2 / 2                 # KL(N(m1,1) || N(g1,1))
    D12 = (m1v - t12) ** 2 / 2                # KL(N(m1,1) || N(t12,1))
    c22 = (0 - g2) ** 2 / 2                   # KL(N(0,1) || N(g2,1))
    D21 = (0 - t21) ** 2 / 2                  # KL(N(0,1) || N(t21,1))
    rowKL = b11 * sp.log(b11 / p11) + (1 - b11) * sp.log((1 - b11) / p12) + b22 * sp.log(b22 / p22) + (1 - b22) * sp.log((1 - b22) / p21)
    F1 = b11 * c11 + (1 - b11) * D12
    F2 = b22 * c22 + (1 - b22) * D21
    Ftot = sp.expand(F1 + F2)
    print(f"  [{side}] costs: c11 - (phi+s11) = {sp.expand(c11 - phi - s11)}, D12 - (phi+s12) = {sp.expand(D12 - phi - s12)}, "
          f"c22 - (s22-phi) = {sp.expand(c22 - s22 + phi)}, D21 - (s21-phi) = {sp.expand(D21 - s21 + phi)}")
    print(f"  [{side}] F_tot - [2 + |m1|^2] (row entropies aside) = {sp.expand(Ftot - (2 + xp**2))}")
    dF1 = sp.expand(sp.diff(F1, xp) * sgn)       # d/dm1 = sgn d/dx
    print(f"  [{side}] own-block d F_1/d m1 = {dF1}", flush=True)
print("  F_tot is C^infinity (phi cancels); agent 1's own-block force is -phi'(m1) - 2 m1 with -phi' = (7/4)|m1|^(3/4).")


def perag(t, y):
    a = y[0]
    phip = -1.75 * np.abs(a) ** 0.75           # phi'(m1) for phi = -sgn|m1|^{7/4}
    return [-(phip + 2 * a)]


for x0 in [1e-8, 1e-12]:
    sol = solve_ivp(perag, (0, 2.0), [x0], method='DOP853', rtol=1e-10, atol=1e-24, t_eval=[1.0, 2.0])
    print(f"  per-agent m1 from {x0:.0e}: m1(1), m1(2) = {sol.y[0]}")
print(f"\nruntime {time.time()-t0:.1f} s")
