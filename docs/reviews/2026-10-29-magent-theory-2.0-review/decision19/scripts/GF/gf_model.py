r"""Two-level conjugate linear-Gaussian partition model under the global free energy.

Shared helpers for the GF scripts.  Scalar contents y_i, child noise s_i^2,
first-level block latents z_B with spread tau^2 around second-level latents
w_C ~ N(0, sigma^2), single-agent records o_i ~ N(y_i, rho_i^2).

For a nested structure Pi = (Pi1, Pi2) the prior covariance of the contents is
    C_Pi = diag(s^2) + tau^2 M1 + sigma^2 M2,
with M1, M2 the co-membership matrices of the first-level and the induced
second-level partitions of agents.

Declared dynamics (hard structural family): one sweep is
    Pi_{n+1} = argmax_Pi [log pi(Pi) + log N(m_n; 0, C_Pi)],
    m_{n+1}  = a o + b (I - S C_{Pi_{n+1}}^{-1}) m_n,
with S = diag(s^2), a_i = s_i^2/(s_i^2+rho_i^2), b_i = 1 - a_i.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

Partition = Tuple[Tuple[int, ...], ...]
Structure = Tuple[Partition, Partition]  # (Pi1 over agents, Pi2 over agents, induced)


def set_partitions(items: Sequence[int]) -> List[List[List[int]]]:
    """All set partitions of a finite list (Bell-number many)."""
    items = list(items)
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    out: List[List[List[int]]] = []
    for p in set_partitions(rest):
        out.append([[first]] + [list(b) for b in p])
        for k in range(len(p)):
            q = [list(b) for b in p]
            q[k] = [first] + q[k]
            out.append(q)
    return out


def canon(p: Sequence[Sequence[int]]) -> Partition:
    return tuple(sorted(tuple(sorted(b)) for b in p))


def nested_catalog(n_agents: int) -> List[Structure]:
    """All pairs (Pi1, Pi2) with Pi2 a partition of the blocks of Pi1, Pi2 stored on agents."""
    cat: List[Structure] = []
    for p1 in set_partitions(list(range(n_agents))):
        blocks = [tuple(sorted(b)) for b in p1]
        for p2 in set_partitions(list(range(len(blocks)))):
            induced = [sum((list(blocks[k]) for k in grp), []) for grp in p2]
            cat.append((canon(blocks), canon(induced)))
    return cat


def comembership(p: Partition, n: int) -> np.ndarray:
    M = np.zeros((n, n))
    for b in p:
        for i in b:
            for j in b:
                M[i, j] = 1.0
    return M


class Model:
    def __init__(self, s2: np.ndarray, rho2: np.ndarray, tau2: float, sig2: float,
                 catalog: List[Structure] | None = None,
                 log_prior: Dict[Structure, float] | None = None) -> None:
        self.n = len(s2)
        self.s2 = np.asarray(s2, float)
        self.rho2 = np.asarray(rho2, float)
        self.tau2 = float(tau2)
        self.sig2 = float(sig2)
        self.cat = catalog if catalog is not None else nested_catalog(self.n)
        if log_prior is None:
            log_prior = {S: -np.log(len(self.cat)) for S in self.cat}
        self.lp = log_prior
        self.a = self.s2 / (self.s2 + self.rho2)
        self.b = 1.0 - self.a
        self.C: Dict[Structure, np.ndarray] = {}
        self.Ci: Dict[Structure, np.ndarray] = {}
        self.ld: Dict[Structure, float] = {}
        for S in self.cat:
            C = np.diag(self.s2) + self.tau2 * comembership(S[0], self.n) + self.sig2 * comembership(S[1], self.n)
            self.C[S] = C
            self.Ci[S] = np.linalg.inv(C)
            self.ld[S] = float(np.linalg.slogdet(C)[1])

    # structure score: log pi + log N(m; 0, C) up to a Pi-independent constant
    def score(self, S: Structure, m: np.ndarray) -> float:
        return self.lp[S] - 0.5 * float(m @ self.Ci[S] @ m) - 0.5 * self.ld[S]

    def scores(self, m: np.ndarray, cat: List[Structure] | None = None) -> List[Tuple[float, Structure]]:
        cat = cat if cat is not None else self.cat
        return sorted(((self.score(S, m), S) for S in cat), reverse=True)

    def select(self, m: np.ndarray, cat: List[Structure] | None = None) -> Tuple[Structure, float]:
        sc = self.scores(m, cat)
        return sc[0][1], sc[0][0] - sc[1][0]

    def K(self, S: Structure) -> np.ndarray:
        return np.eye(self.n) - np.diag(self.s2) @ self.Ci[S]

    def sweep(self, m: np.ndarray, o: np.ndarray, cat: List[Structure] | None = None) -> Tuple[np.ndarray, Structure, float]:
        S, gap = self.select(m, cat)
        m_new = self.a * o + self.b * (self.K(S) @ m)
        return m_new, S, gap

    def fixed_point(self, S: Structure, o: np.ndarray) -> np.ndarray:
        """m* = a o + b K m*  <=>  m* = C (C + R)^{-1} o (checked in gf_symbolic)."""
        A = np.eye(self.n) - np.diag(self.b) @ self.K(S)
        return np.linalg.solve(A, self.a * o)

    def exact_log_post(self, o: np.ndarray) -> List[Tuple[float, Structure]]:
        out = []
        for S in self.cat:
            Co = self.C[S] + np.diag(self.rho2)
            v = self.lp[S] - 0.5 * float(o @ np.linalg.solve(Co, o)) - 0.5 * float(np.linalg.slogdet(Co)[1])
            out.append((v, S))
        return sorted(out, reverse=True)


FINEST4: Structure = (((0,), (1,), (2,), (3,)), ((0,), (1,), (2,), (3,)))
A_SEP: Structure = (((0, 1), (2, 3)), ((0, 1), (2, 3)))
A_JOIN: Structure = (((0, 1), (2, 3)), ((0, 1, 2, 3),))


def name(S: Structure) -> str:
    f = lambda p: "|".join("".join(str(i + 1) for i in b) for b in p)
    return f"[{f(S[0])} ; {f(S[1])}]"


def history(model: Model, m0: np.ndarray, o: np.ndarray, n_sweeps: int,
            cat: List[Structure] | None = None):
    """Selector history Pihat(m_n), n = 0..n_sweeps, with gaps and the means."""
    m = m0.copy()
    sel, gaps, ms = [], [], [m.copy()]
    for _ in range(n_sweeps + 1):
        S, gap = model.select(m, cat)
        sel.append(S)
        gaps.append(gap)
        m = model.a * o + model.b * (model.K(S) @ m)
        ms.append(m.copy())
    return sel, gaps, ms


def compress(seq: List[Structure]) -> List[Structure]:
    out: List[Structure] = []
    for S in seq:
        if not out or out[-1] != S:
            out.append(S)
    return out
