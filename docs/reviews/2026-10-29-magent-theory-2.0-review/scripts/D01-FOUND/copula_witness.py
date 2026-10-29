r"""Witness: on an infinite content space the reciprocal potential equals F(Q(x)) of a fixed model
at a recognition law Q(x) whose agent marginals are the agent laws.

Lens D01-FOUND.  Claim under test (chapters/01_foundations.tex:226): "prop:dyn-no-global-vfe shows
that it [the reciprocal potential] is not F(Q;o) for any fixed normalized model with these
marginals as recognition laws."

Setting: two agents, trivial model sector, K_1 = K_2 = R, references R_i = N(0,1), one edge 1 <- 2
through the identity channel with fixed row weight beta, so up to fixed row constants
    E(x) = KL(q_1||N(0,1)) + KL(q_2||N(0,1)) + beta KL(q_1||q_2).
Content model: Z = R^2, f_i(z) = z_i, P_0 = N(0,1) x N(0,1), no record (Z_o = 1).
Rule: Q(x) = Gaussian-copula coupling of (q_1, q_2) with correlation rho(x),
    -1/2 log(1 - rho^2) = beta KL(q_1||q_2).
Identity used:  KL(Q||P_0) = I_Q(Z_1;Z_2) + sum_i KL(q_i||P_{0,i})  for product P_0, and the mutual
information of a Gaussian copula is -1/2 log(1-rho^2) for atomless marginals.
Part 1 is closed form (Gaussian marginals); part 2 uses Laplace marginals and a 2D Gauss-Hermite
rule in copula (normal-score) coordinates.  Agreement at the tested instances supports the identity.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr

rng = np.random.default_rng(7)
beta = 0.8

print("== 1  Gaussian marginals q_i = N(mu_i, 1)")
worst = 0.0
for _ in range(5):
    mu1, mu2 = rng.normal(size=2) * 1.5
    E = 0.5 * mu1**2 + 0.5 * mu2**2 + beta * 0.5 * (mu1 - mu2) ** 2
    rho = math.sqrt(1.0 - math.exp(-beta * (mu1 - mu2) ** 2))
    F = 0.5 * (mu1**2 + mu2**2) - 0.5 * math.log(1.0 - rho**2)      # KL(N(mu, Sigma_rho)||N(0, I))
    worst = max(worst, abs(E - F))
    print(f"  mu=({mu1:+.4f},{mu2:+.4f})  rho={rho:.6f}  E={E:.12f}  F(Q(x))={F:.12f}")
print("  max |E - F| =", worst)

print("== 2  Laplace marginals q_i = Laplace(m_i, b), Gaussian copula, quadrature")
b = 1.0 / math.sqrt(2.0)          # variance 1


def lap_quantile_from_score(z: np.ndarray, m: float) -> np.ndarray:
    r"""Laplace quantile at u = Phi(z), computed through log Phi to avoid u = 0 or 1 in floating point."""
    lo = m + b * (math.log(2.0) + log_ndtr(np.minimum(z, 0.0)))
    hi = m - b * (math.log(2.0) + log_ndtr(-np.maximum(z, 0.0)))
    return np.where(z < 0.0, lo, hi)


def lap_logpdf(x: np.ndarray, m: float) -> np.ndarray:
    return -np.log(2 * b) - np.abs(x - m) / b


def kl_lap_std_normal(m: float) -> float:
    # E[log f - log phi] with E X^2 = m^2 + 2 b^2 and entropy log(2 b e)
    return -math.log(2 * b) - 1.0 + 0.5 * math.log(2 * math.pi) + 0.5 * (m * m + 2 * b * b)


def kl_lap_lap(m1: float, m2: float) -> float:
    d = abs(m1 - m2)
    return d / b + math.exp(-d / b) - 1.0


xg, wg = np.polynomial.hermite_e.hermegauss(120)   # probabilists' Hermite: weight exp(-x^2/2)
wg = wg / math.sqrt(2 * math.pi)
for m1, m2 in [(0.3, -0.4), (1.2, 0.1), (-0.5, 0.9)]:
    E = kl_lap_std_normal(m1) + kl_lap_std_normal(m2) + beta * kl_lap_lap(m1, m2)
    I_target = beta * kl_lap_lap(m1, m2)
    rho = math.sqrt(1.0 - math.exp(-2.0 * I_target))
    # (Z1, Z2) ~ N(0, [[1, rho], [rho, 1]]):  Z1 = X1, Z2 = rho X1 + sqrt(1-rho^2) X2
    X1, X2 = np.meshgrid(xg, xg, indexing="ij")
    W = np.outer(wg, wg)
    Z1 = X1
    Z2 = rho * X1 + math.sqrt(1 - rho**2) * X2
    # log copula density at (Phi(z1), Phi(z2)) = log phi_rho(z1,z2) - log phi(z1) - log phi(z2)
    logc = (-0.5 * math.log(1 - rho**2)
            - (rho**2 * (Z1**2 + Z2**2) - 2 * rho * Z1 * Z2) / (2 * (1 - rho**2)))
    Y1 = lap_quantile_from_score(Z1, m1)
    Y2 = lap_quantile_from_score(Z2, m2)
    log_q = logc + lap_logpdf(Y1, m1) + lap_logpdf(Y2, m2)
    log_p0 = -math.log(2 * math.pi) - 0.5 * (Y1**2 + Y2**2)
    F = float(np.sum(W * (log_q - log_p0)))
    MI = float(np.sum(W * logc))
    print(f"  m=({m1:+.2f},{m2:+.2f})  rho={rho:.6f}  E={E:.8f}  F(Q(x))={F:.8f}  |E-F|={abs(E-F):.2e}"
          f"  copula MI={MI:.8f} target={I_target:.8f}")
print("  The 2D Gauss-Hermite residual comes from the kink of the Laplace log density at the median.")
print("  Convergence in the node count, and the decomposed evaluation MI(2D GH) + marginal KL (1D quad):")
from scipy.integrate import quad
m1, m2 = 0.3, -0.4
E = kl_lap_std_normal(m1) + kl_lap_std_normal(m2) + beta * kl_lap_lap(m1, m2)
I_target = beta * kl_lap_lap(m1, m2)
rho = math.sqrt(1.0 - math.exp(-2.0 * I_target))
for n in [60, 120, 240, 480]:
    xg_, wg_ = np.polynomial.hermite_e.hermegauss(n); wg_ = wg_ / math.sqrt(2 * math.pi)
    X1, X2 = np.meshgrid(xg_, xg_, indexing="ij"); W = np.outer(wg_, wg_)
    Z1 = X1; Z2 = rho * X1 + math.sqrt(1 - rho**2) * X2
    logc = (-0.5 * math.log(1 - rho**2) - (rho**2 * (Z1**2 + Z2**2) - 2 * rho * Z1 * Z2) / (2 * (1 - rho**2)))
    Y1 = lap_quantile_from_score(Z1, m1); Y2 = lap_quantile_from_score(Z2, m2)
    F = float(np.sum(W * (logc + lap_logpdf(Y1, m1) + lap_logpdf(Y2, m2) + math.log(2 * math.pi) + 0.5 * (Y1**2 + Y2**2))))
    print(f"    n={n:4d}  |E - F_GH| = {abs(E - F):.3e}")
def marg_kl(m: float) -> float:
    f = lambda x: math.exp(-abs(x - m) / b) / (2 * b) * (-math.log(2 * b) - abs(x - m) / b + 0.5 * math.log(2 * math.pi) + 0.5 * x * x)
    return quad(f, -np.inf, m, limit=200)[0] + quad(f, m, np.inf, limit=200)[0]
xg_, wg_ = np.polynomial.hermite_e.hermegauss(120); wg_ = wg_ / math.sqrt(2 * math.pi)
X1, X2 = np.meshgrid(xg_, xg_, indexing="ij"); W = np.outer(wg_, wg_)
Z1 = X1; Z2 = rho * X1 + math.sqrt(1 - rho**2) * X2
MI = float(np.sum(W * (-0.5 * math.log(1 - rho**2) - (rho**2 * (Z1**2 + Z2**2) - 2 * rho * Z1 * Z2) / (2 * (1 - rho**2)))))
F_dec = MI + marg_kl(m1) + marg_kl(m2)
print(f"    decomposed: F = {F_dec:.12f}, E = {E:.12f}, |E - F| = {abs(E - F_dec):.3e}")
print("  On finite alphabets the mutual information of a coupling is at most log|K|, which is why")
print("  part (a) of prop:dyn-no-global-vfe (finite variables) is not contradicted.")
