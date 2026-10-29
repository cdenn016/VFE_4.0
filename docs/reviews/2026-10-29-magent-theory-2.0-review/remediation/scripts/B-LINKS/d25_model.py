r"""Shared model for the D25-CRIT checks of chapters/18_link_dynamics.tex, it2:coupled-threshold
and it2:critical-pitchfork.

The model is rebuilt from the general Gaussian formulas of it2:coupled-costs (eq. it2:coupled-data
and it2:coupled-costs), specialized to the data of it2:coupled-threshold.  It is written
independently of the research-record scripts.

Coordinates: x = (mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2), row imbalance s = b - 1/2.

Edge cost (scalar code Tbar = 1, kappa_0 = 1, noise R = 1):
    C_e = 1/2 [ Tr(J_e Q_e) - n_e + log det S_{j,e} - log det S_{i,e} ],
    S_{k,e} = Tbar v_k Tbar + R,  J_e = S_{j,e}^{-1},  K_e = Tbar kappa_i,
    d_e = Tbar (mu_i - mu_j) - K_e a_e,  Q_e = S_{i,e} + d_e^2 + K_e V_e K_e.
Self terms: KL(N(mu, v) || N(m, P^{-1})) = 1/2 [P v + P (mu - m)^2 - 1 - log(P v)].
"""
from __future__ import annotations

import sympy as sp

R = sp.Rational

mu0, mu1, mu2, a1, a2 = sp.symbols("mu0 mu1 mu2 a1 a2", real=True)
v0, v1, v2, V1, V2 = sp.symbols("v0 v1 v2 V1 V2", positive=True)
XS = [mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2]
X = sp.Matrix(XS)
s = sp.Symbol("s", real=True)
T = sp.Symbol("T", positive=True)

P_REF = [R(1), R(5, 4), R(5, 4)]
M_REF = [R(0), R(-23, 15), R(23, 15)]
A_OMEGA = R(3, 4)  # precision of the dictionary reference N(0, 4/3)

XSTAR = [R(0), R(-4, 3), R(4, 3), R(1, 3), R(-1, 3), R(1), R(1), R(1), R(1), R(1)]
POINT = dict(zip(XS, XSTAR))

T_C_CLAIM = R(3269, 10118)
T_MEAN_CLAIM = R(31, 122)


def kl_gauss(mu: sp.Expr, v: sp.Expr, mean: sp.Expr, prec: sp.Expr) -> sp.Expr:
    r"""KL(N(mu, v) || N(mean, 1/prec)) = 1/2 [prec v + prec (mu - mean)^2 - 1 - log(prec v)]."""
    return (prec * v + prec * (mu - mean) ** 2 - 1 - sp.log(prec * v)) / 2


def edge_cost(mu_i, v_i, mu_j, v_j, a, V, code=1, kappa=1, noise=1) -> sp.Expr:
    r"""Expected conditional message KL, C_e = 1/2[Tr(J Q) - n + log S_j - log S_i] (n = 1)."""
    S_i = code * v_i * code + noise
    S_j = code * v_j * code + noise
    K = code * kappa
    d = code * (mu_i - mu_j) - K * a
    Q = S_i + d ** 2 + K * V * K
    return (Q / S_j - 1 + sp.log(S_j) - sp.log(S_i)) / 2


F0 = (kl_gauss(mu0, v0, M_REF[0], P_REF[0])
      + kl_gauss(mu1, v1, M_REF[1], P_REF[1])
      + kl_gauss(mu2, v2, M_REF[2], P_REF[2])
      + kl_gauss(a1, V1, 0, A_OMEGA)
      + kl_gauss(a2, V2, 0, A_OMEGA))
C1 = edge_cost(mu0, v0, mu1, v1, a1, V1)
C2 = edge_cost(mu0, v0, mu2, v2, a2, V2)

A_FUN = F0 + (C1 + C2) / 2          # A(x) of it2:critical-functions
DELTA = C1 - C2                      # Delta(x)
H_ENT = (R(1, 2) + s) * sp.log(1 + 2 * s) + (R(1, 2) - s) * sp.log(1 - 2 * s)  # h(s)

# Full potential in the row imbalance s = b - 1/2 (prior (1/2,1/2)):
#   E_T(x,b) = F0 + b C1 + (1-b) C2 + T [b log(2b) + (1-b) log(2(1-b))].
ECAL = F0 + (R(1, 2) + s) * C1 + (R(1, 2) - s) * C2 + T * H_ENT

# Gibbs-reduced potential, log-sum-exp form and an exact algebraic rewriting:
#   E_T(x) = F0 - T log((e^{-C1/T} + e^{-C2/T})/2) = A(x) - T log cosh(Delta/(2T)).
E_RED_LSE = F0 - T * sp.log((sp.exp(-C1 / T) + sp.exp(-C2 / T)) / 2)
E_RED = A_FUN - T * sp.log(sp.cosh(DELTA / (2 * T)))

MOB = sp.diag(v0, v1, v2, V1, V2, 2 * v0 ** 2, 2 * v1 ** 2, 2 * v2 ** 2, 2 * V1 ** 2, 2 * V2 ** 2)


def grad(expr: sp.Expr, vars_: list) -> sp.Matrix:
    return sp.Matrix([sp.diff(expr, z) for z in vars_])


def at_point(expr, extra=None):
    sub = dict(POINT)
    if extra:
        sub.update(extra)
    return expr.subs(sub)
