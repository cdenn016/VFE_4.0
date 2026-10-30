r"""Scan the between-cluster record gap: full-flow partition sequence vs null selection.

For each gap, report P (full), Pihat(kappa o) (null after one sweep), the exact MAP
structure, and whether [12|34 ; 12|34] and [12|34 ; 1234] are self-consistent
(Pi = Pihat(m*_Pi)).
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name, history, compress  # noqa: E402

s2 = np.ones(4)
for r2, tau2, sig2, c1 in itertools.product([0.25, 0.5], [0.05, 0.1], [4.0, 16.0], [0.0, 3.0]):
    rho2 = np.full(4, r2)
    M = Model(s2, rho2, tau2, sig2)
    v0 = 1.0 + tau2 + sig2
    kappa = v0 / (v0 + r2)
    print(f"--- rho2={r2} tau2={tau2} sig2={sig2} c1={c1}")
    for gap12 in np.arange(1.0, 6.01, 0.25):
        e = 0.1
        o = np.array([c1 - e, c1 + e, c1 + gap12 - e, c1 + gap12 + e])
        m0 = 3.0 * np.array([-1.5, -0.5, 0.5, 1.5])
        if M.select(m0)[0] != FINEST4:
            print("   m0 not finest")
            continue
        sel, gaps, ms = history(M, m0, o, 80)
        P = [name(S) for S in compress(sel)]
        null = name(M.select(kappa * o)[0])
        ex = name(M.exact_log_post(o)[0][1])
        sc = []
        for S in (A_SEP, A_JOIN):
            ms_ = M.fixed_point(S, o)
            sc.append(M.select(ms_)[0] == S)
        print(f"   gap={gap12:.2f} P={P} null={null} exact={ex} selfcons(sep,join)={sc}")
