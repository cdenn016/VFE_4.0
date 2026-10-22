r"""X3-PIFB check 4: determinant metric as Fisher minus entropy-rate squared; proper Fisher-Rao volume;
Fisher flow of learned source priors.

(a) For 2x2 Hermitian H and M > 0, with kappa_det = det M,
    -det(H)/det(M) = (1/2)[ tr(M^{-1} H M^{-1} H) - (tr M^{-1} H)^2 ],
    i.e. gamma = (1/2)(g_CN - theta (x) theta), g_CN the Fisher metric of CN(0,M),
    theta = d log det M = d(entropy of CN(0,M)).  Signature (1,3) because ||theta||^2_{g^{-1}} = 2 > 1.
(b) Fisher-Rao volume of the open (n-1)-simplex equals pi^{n/2}/Gamma(n/2) (Dirichlet(1/2) normalizer).
(c) Unit-mobility Fisher flow of T*KL(beta||pi) in pi is d pi/dt = T(beta - pi).
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

rng = np.random.default_rng(7)
res = 0.0
sig_ok = True
for _ in range(200):
    Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    M = Z @ Z.conj().T + 0.1 * np.eye(2)
    Hr = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    H = 0.5 * (Hr + Hr.conj().T)
    Mi = np.linalg.inv(M)
    lhs = -np.linalg.det(H).real / np.linalg.det(M).real
    rhs = 0.5 * (np.trace(Mi @ H @ Mi @ H).real - np.trace(Mi @ H).real ** 2)
    res = max(res, abs(lhs - rhs))
    # signature of the quadratic form H -> rhs on the real 4-dim space of Hermitian matrices
    basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[0, 1], [1, 0]]),
             np.array([[0, -1j], [1j, 0]])]
    G = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            Hp, Hq = basis[p], basis[q]
            G[p, q] = 0.5 * (np.trace(Mi @ Hp @ Mi @ Hq).real - np.trace(Mi @ Hp).real * np.trace(Mi @ Hq).real)
    ev = np.linalg.eigvalsh(G)
    if not (np.sum(ev < 0) == 1 and np.sum(ev > 0) == 3):
        sig_ok = False
print("(a) max residual of determinant/Fisher/entropy identity:", res, " signature (1,3) in all trials:", sig_ok)

# (b) Fisher-Rao volume of simplices by Monte Carlo-free formula check: sphere-orthant area vs Dirichlet(1/2)
for n in (2, 3, 4, 6):
    orthant = 2 ** (n - 1) * (2 * math.pi ** (n / 2) / math.gamma(n / 2)) / 2 ** n  # radius-2 sphere, one orthant
    dirichlet = math.gamma(0.5) ** n / math.gamma(n / 2)
    print(f"(b) n={n}: radius-2 sphere orthant area = {orthant:.6f}, Dirichlet(1/2) normalizer = {dirichlet:.6f}")

# (c) symbolic Fisher flow of T*KL(beta||pi) in pi on a 3-simplex
T = sp.symbols("T", positive=True)
pi = sp.symbols("pi1:4", positive=True)
be = sp.symbols("b1:4", positive=True)
F = T * sum(be[k] * sp.log(be[k] / pi[k]) for k in range(3))
h = [sp.diff(F, pi[k]) for k in range(3)]
hbar = sum(pi[k] * h[k] for k in range(3))
flow = [sp.simplify(-pi[k] * (h[k] - hbar)) for k in range(3)]
target = [T * (be[k] - pi[k] * sum(be)) for k in range(3)]
print("(c) Fisher flow minus T(beta - pi*sum beta):", [sp.simplify(flow[k] - target[k]) for k in range(3)],
      " (sum beta = 1 gives T(beta - pi))")
