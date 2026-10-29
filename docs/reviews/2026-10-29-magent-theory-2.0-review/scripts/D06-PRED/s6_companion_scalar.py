r"""D06-PRED s6: every number of the companion scalar example (companion/c04_communication.tex:27-44).

X ~ N(0,1), eps ~ N(0,1/4), eta ~ N(0,1) independent; Y = -2X + eps, O = X + eta; observe O = xi.
Three independent routes: (1) completing the square as in the companion, (2) Gaussian conditioning of
the joint (Y, O) covariance, (3) the appendix formulas with F = -2, L = 1, C = 1, Q = 1/4, nu = omega = 0.
"""
from __future__ import annotations

import sympy as sp

xi, x = sp.symbols("xi x", real=True)
R = sp.Rational
# (1) completing the square: x^2 + (xi - x)^2 = 2 (x - xi/2)^2 + xi^2/2
print("x^2 + (xi-x)^2 - [2(x - xi/2)^2 + xi^2/2] =", sp.expand(x ** 2 + (xi - x) ** 2 - 2 * (x - xi / 2) ** 2 - xi ** 2 / 2))
print("so p(x|xi) propto exp[-(x - xi/2)^2] = exp[-(x-m)^2/(2*v)] with v =", R(1, 2))
mX, PX = xi / 2, R(1, 2)
mY = -2 * mX
PY = 4 * PX + R(1, 4)
print("route 1: E[Y|xi] =", mY, ", Var(Y|xi) =", PY)
# (2) joint covariance of (Y, O)
VarY = 4 * 1 + R(1, 4)
VarO = sp.Integer(2)
CovYO = sp.Integer(-2)
print("route 2: E[Y|xi] =", sp.simplify(CovYO / VarO * xi), ", Var(Y|xi) =", VarY - CovYO ** 2 / VarO)
# (3) appendix formulas
F, L, C, Q = -2, 1, 1, R(1, 4)
mX3 = L / sp.Integer(L + C) * xi
PX3 = 1 / (R(1, L) + R(1, C))
print("route 3: m_X =", mX3, ", P_X =", PX3, ", m_Y =", F * mX3, ", P_Y =", Q + F ** 2 * PX3,
      ", response F L/(L+C) =", R(F * L, L + C), ", cue slope L/(L+C) =", R(L, L + C),
      ", Fisher I_xi = P_X/C^2 =", PX3 / C ** 2)
print("variance arithmetic 4(1/2) + 1/4 =", 4 * R(1, 2) + R(1, 4))
