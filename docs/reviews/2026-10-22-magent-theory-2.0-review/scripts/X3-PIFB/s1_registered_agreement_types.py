r"""X3-PIFB check 1: registered-agreement (method of types) semantics of the 2.0 peer potential.

Claims checked on finite alphabets, exact enumeration (no sampling):
 (i)   P_nu(type(X^n) = P) = |T_P| exp(-n[H(P) + KL(P||nu)])          (Cover-Thomas Thm 11.1.2)
 (ii)  with a source label J ~ pi and messages from nu_J, the posterior of J given
       {type = q_i} is the 2.0 Gibbs row beta*_j \propto pi_j exp(-KL(q_i||nu_j)/T) with T = 1/n
 (iii) -T log P(type = q_i) = -T log Z_i + c_n(q_i),  c_n(q) = H(q) - (1/n) log|T_q|,
       0 <= c_n(q) <= |A| log(n+1)/n                                     (Cover-Thomas Thm 11.1.3)
 (iv)  T * [sum_j beta_j KL(q_i||nu_j) + T KL(beta||pi)]/T equals T*(label VFE) up to c_n:
       sum_j beta_j D_j + T KL(beta||pi) = T*KL(beta||post) - T log P(o=1) - c_n(q_i)
Run: OMP_NUM_THREADS=1 timeout 20 python s1_registered_agreement_types.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def kl(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL(p||q) = \sum_a p_a log(p_a/q_a), with 0 log 0 = 0."""
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def entropy(p: np.ndarray) -> float:
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def type_prob_enumerated(P_counts: tuple[int, ...], nu: np.ndarray, n: int) -> float:
    """Probability that n iid draws from nu have exactly these counts, by brute-force enumeration."""
    A = len(nu)
    tot = 0.0
    for seq in itertools.product(range(A), repeat=n):
        c = tuple(seq.count(a) for a in range(A))
        if c == P_counts:
            tot += math.prod(nu[s] for s in seq)
    return tot


def type_class_size(counts: tuple[int, ...]) -> int:
    n = sum(counts)
    out = math.factorial(n)
    for c in counts:
        out //= math.factorial(c)
    return out


rng = np.random.default_rng(0)
max_err_i = 0.0
max_err_ii = 0.0
max_err_iii = 0.0
max_err_iv = 0.0
cn_ok = True
for trial in range(6):
    A = 3
    n = [4, 5, 6, 7, 6, 5][trial]
    # receiver law q_i: an n-type
    counts = tuple(int(c) for c in rng.multinomial(n, [1 / A] * A))
    if min(counts) == 0:
        counts = (n - 2, 1, 1)
    q = np.array(counts, dtype=float) / n
    # three candidate sources with positive message laws nu_j (e.g. q_j K_j)
    nus = [rng.dirichlet(np.ones(A) * 2.0) for _ in range(3)]
    pi = rng.dirichlet(np.ones(3))
    T = 1.0 / n
    # (i)
    for nu in nus:
        lhs = type_prob_enumerated(counts, nu, n)
        rhs = type_class_size(counts) * math.exp(-n * (entropy(q) + kl(q, nu)))
        max_err_i = max(max_err_i, abs(lhs - rhs) / rhs)
    # (ii) label posterior given registered agreement {type = q}
    lik = np.array([type_prob_enumerated(counts, nu, n) for nu in nus])
    post = pi * lik / np.sum(pi * lik)
    D = np.array([kl(q, nu) for nu in nus])
    Z = float(np.sum(pi * np.exp(-D / T)))
    gibbs = pi * np.exp(-D / T) / Z
    max_err_ii = max(max_err_ii, float(np.max(np.abs(post - gibbs))))
    # (iii)
    P_o = float(np.sum(pi * lik))
    cn = entropy(q) - math.log(type_class_size(counts)) / n
    max_err_iii = max(max_err_iii, abs((-T * math.log(P_o)) - (-T * math.log(Z) + cn)))
    if not (-1e-12 <= cn <= A * math.log(n + 1) / n + 1e-12):
        cn_ok = False
    # (iv) unreduced row objective at an arbitrary row beta
    beta = rng.dirichlet(np.ones(3))
    row_obj = float(np.sum(beta * D)) + T * kl(beta, pi)
    label_vfe_scaled = T * kl(beta, post) - T * math.log(P_o) - cn
    max_err_iv = max(max_err_iv, abs(row_obj - label_vfe_scaled))

print(f"(i)   max relative error of type-class identity      : {max_err_i:.3e}")
print(f"(ii)  max |posterior(J|type=q) - Gibbs row(T=1/n)|   : {max_err_ii:.3e}")
print(f"(iii) max |-T log P(o=1) - (-T log Z + c_n)|         : {max_err_iii:.3e}")
print(f"      c_n within [0, |A| log(n+1)/n] in all trials   : {cn_ok}")
print(f"(iv)  max |row objective - T*labelVFE + c_n|         : {max_err_iv:.3e}")
