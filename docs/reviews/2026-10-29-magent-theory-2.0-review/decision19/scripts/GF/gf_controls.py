r"""Control flows for the GF construction under the declared global conventions.

full  : declared sweep (shared structural block, hard family).
lp    : label-preserving global null = private copies of the non-agent blocks.
        A single agent's private copy is uninformative about the structure, and
        its latent pathway gives  m_i' = a_i o_i + b_i k_i m_i,
        k_i = (tau^2+sigma^2)/(s_i^2+tau^2+sigma^2)  (the finest-structure sweep).
rn    : renormalized global null = non-agent blocks integrated out; the agent
        prior is the single-agent marginal N(0, v_i), v_i = s_i^2+tau^2+sigma^2,
        and one sweep gives m_i = kappa_i o_i, kappa_i = v_i/(v_i+rho_i^2).
frozen: frozen-environment control; agent i carries a private copy of the
        non-agent blocks updated from (Q_i(t), Q_{-i}(0)).
The selector is the original-objective structure minimizer at the control's
agent laws (readout convention).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from gf_model import Model, Structure, compress


def run_full(M: Model, m0: np.ndarray, o: np.ndarray, n: int):
    m = m0.copy()
    sel, ms, gaps = [], [m.copy()], []
    for _ in range(n + 1):
        S, g = M.select(m)
        sel.append(S)
        gaps.append(g)
        m = M.a * o + M.b * (M.K(S) @ m)
        ms.append(m.copy())
    return sel, ms, gaps


def run_lp(M: Model, m0: np.ndarray, o: np.ndarray, n: int):
    k = (M.tau2 + M.sig2) / (M.s2 + M.tau2 + M.sig2)
    m = m0.copy()
    sel, ms, gaps = [], [m.copy()], []
    for _ in range(n + 1):
        S, g = M.select(m)
        sel.append(S)
        gaps.append(g)
        m = M.a * o + M.b * k * m
        ms.append(m.copy())
    return sel, ms, gaps


def run_rn(M: Model, m0: np.ndarray, o: np.ndarray, n: int):
    v = M.s2 + M.tau2 + M.sig2
    kap = v / (v + M.rho2)
    sel, ms, gaps = [], [m0.copy()], []
    m = m0.copy()
    for _ in range(n + 1):
        S, g = M.select(m)
        sel.append(S)
        gaps.append(g)
        m = kap * o
        ms.append(m.copy())
    return sel, ms, gaps


def run_frozen(M: Model, m0: np.ndarray, o: np.ndarray, n: int):
    m = m0.copy()
    sel, ms, gaps = [], [m.copy()], []
    for _ in range(n + 1):
        S, g = M.select(m)
        sel.append(S)
        gaps.append(g)
        new = np.empty_like(m)
        for i in range(M.n):
            mi = m0.copy()
            mi[i] = m[i]
            Si, _ = M.select(mi)
            new[i] = M.a[i] * o[i] + M.b[i] * (M.K(Si) @ mi)[i]
        m = new
        ms.append(m.copy())
    return sel, ms, gaps


def seqs(M: Model, m0: np.ndarray, o: np.ndarray, n: int = 150) -> Tuple[List[Structure], ...]:
    return tuple(compress(f(M, m0, o, n)[0]) for f in (run_full, run_lp, run_rn, run_frozen))
