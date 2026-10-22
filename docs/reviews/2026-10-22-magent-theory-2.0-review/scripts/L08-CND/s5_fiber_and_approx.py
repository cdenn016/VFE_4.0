r"""L08-CND check 5: energy non-factorization, relative heterogeneity, near-common-precision defect.

(i)  Energy split at P=I: replacing shapes (alpha0, alpha0) by (alpha0+e, alpha0-e) raises the fine energy by
     2 KL(W(alpha0+e,I)||W(alpha0,I)) + 2 KL(W(alpha0-e,I)||W(alpha0,I)) (claimed, main text 266-271).
(ii) Mean-field relative state: with unequal initial precisions, (P_i - P_j)(t) = e^{-2t}(P_i - P_j)(0) and
     (alpha_i - alpha_j)(t) = e^{-2t}(alpha_i-alpha_j)(0), so their ratio is time invariant; at a pair's join
     time the precision mismatch equals delta * |dP(0)| / |d alpha(0)|.
(iii) Near-common precision: log Laplace transform of X_1 + X_2 (X_i ~ W(alpha_i, P_i)) minus that of the
     moment-matched Wishart W(A, Pbar), Pbar^{-1} = sum alpha_i P_i^{-1}/A, scales as eps^2 when
     P_i^{-1} = Pbar^{-1} + eps F_i with sum alpha_i F_i = 0 (first-order term cancels).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import digamma, gammaln

N = 4


def lG2(x):
    return 0.5 * np.log(np.pi) + gammaln(x) + gammaln(x - 0.5)


def psi2(x):
    return digamma(x) + digamma(x - 0.5)


def KL(a, P, b, Q):
    return ((a - b) * psi2(a) - lG2(a) + lG2(b) + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2))


def fine_energy(al, Ps, a0, kappa):
    M = sum(a * np.linalg.inv(P) for a, P in zip(al, Ps))
    Lam = np.eye(2) + kappa / N * M
    return sum(KL(a, P, a0, np.eye(2)) + KL(a, P, a0, Lam) for a, P in zip(al, Ps))


# (i)
a0, kappa, e = 1.3, 0.8, 0.2
I2 = np.eye(2)
base = fine_energy([a0, a0, 1.9, 2.4], [I2] * 4, a0, kappa)
split = fine_energy([a0 + e, a0 - e, 1.9, 2.4], [I2] * 4, a0, kappa)
claim = 2 * KL(a0 + e, I2, a0, I2) + 2 * KL(a0 - e, I2, a0, I2)
print("(i) energy increase:", split - base, " claimed:", claim, " diff:", split - base - claim)

# (ii)
al0 = np.array([1.0, 1.2, 1.8, 2.1])
P0s = [np.array([[1.3, 0.2], [0.2, 0.9]]), np.array([[1.1, -0.1], [-0.1, 1.0]]),
       np.array([[0.9, 0.0], [0.0, 1.4]]), np.array([[1.2, 0.3], [0.3, 1.1]])]


def rhs(t, y):
    al = y[:N]
    Ps = [y[N + 4 * i:N + 4 * i + 4].reshape(2, 2) for i in range(N)]
    M = sum(a * np.linalg.inv(P) for a, P in zip(al, Ps))
    Lam = np.eye(2) + kappa / N * M
    C = Lam + kappa / N * (M - N * 1.0 * np.linalg.inv(Lam))
    out = [-2 * (al - 1.0)]
    for P in Ps:
        out.append((-2 * P + np.eye(2) + C).ravel())
    return np.concatenate(out)


y0 = np.concatenate([al0] + [P.ravel() for P in P0s])
sol = solve_ivp(rhs, (0, 2.0), y0, rtol=1e-11, atol=1e-13, dense_output=True)
ratios = []
for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
    y = sol.sol(t)
    dP = y[N:N + 4] - y[N + 4:N + 8]
    da = y[0] - y[1]
    ratios.append(np.linalg.norm(dP) / abs(da))
    if t > 0:
        print(f"(ii) t={t}: |dP(t)|/(e^-2t|dP(0)|) = {np.linalg.norm(dP) / (np.exp(-2 * t) * np.linalg.norm(P0s[0] - P0s[1])):.12f}")
print("    ratio |P_1-P_2|/|alpha_1-alpha_2| over time:", np.round(ratios, 12))
print("    at the first join (delta=0.1): precision mismatch =", 0.1 * ratios[0])


# (iii)
def logL_sum(T, als, Pinvs):
    return -sum(a * np.log(np.linalg.det(np.eye(2) + Pi @ T)) for a, Pi in zip(als, Pinvs))


als = [0.9, 1.7]
A = sum(als)
Pbar_inv = np.array([[0.8, 0.1], [0.1, 1.2]])
F1 = np.array([[0.3, -0.2], [-0.2, 0.1]])
F = [F1, -als[0] / als[1] * F1]
T = np.array([[0.4, 0.1], [0.1, 0.7]])
prev = None
for eps in [0.2, 0.1, 0.05, 0.025]:
    Pinvs = [Pbar_inv + eps * Fi for Fi in F]
    d = logL_sum(T, als, Pinvs) - (-A * np.log(np.linalg.det(np.eye(2) + Pbar_inv @ T)))
    print(f"(iii) eps={eps}: log-Laplace defect = {d:.4e}" + ("" if prev is None else f"   ratio to previous = {prev / d:.4f}"))
    prev = d
