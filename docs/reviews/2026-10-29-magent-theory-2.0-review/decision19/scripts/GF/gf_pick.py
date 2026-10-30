r"""List candidate instances with P = finest -> A_sep -> A_join and their margins.

Uses simple decimal records and initial means.  Reports the smallest selector gap
along the full and the label-preserving null histories (first 80 sweeps).
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name, compress  # noqa: E402
from gf_controls import run_full, run_lp  # noqa: E402

s2 = np.ones(4)
target = [FINEST4, A_SEP, A_JOIN]
rows = []
for r2, tau2, sig2 in itertools.product([0.25], [0.05, 0.1], [4.0]):
    M = Model(s2, np.full(4, r2), tau2, sig2)
    for c1, gap12, e in itertools.product([0.0, 1.5, 3.0], [1.5, 2.0, 2.5], [0.1]):
        o = np.array([c1 - e, c1 + e, c1 + gap12 - e, c1 + gap12 + e])
        kappa = (1.0 + tau2 + sig2) / (1.0 + tau2 + sig2 + r2)
        for D, p, q, lam0 in itertools.product([1.0, 2.0], [-3.0, -1.0, 0.0, 1.0, 3.0], [-3.0, -1.0, 0.0, 1.0, 3.0], [3.0, 6.0]):
            m0 = np.round(kappa * o + lam0 * (D * np.array([-1.0, 1.0, -1.0, 1.0]) + np.array([p, p, q, q])), 1)
            if M.select(m0)[0] != FINEST4:
                continue
            sel, ms, g = run_full(M, m0, o, 80)
            if compress(sel) != target:
                continue
            sl, ml, gl = run_lp(M, m0, o, 80)
            rows.append((min(min(g), min(gl)), r2, tau2, sig2, tuple(o), tuple(m0), compress(sl) == target))
rows.sort(reverse=True)
for r in rows[:15]:
    print(f"mingap={r[0]:.4f} rho2={r[1]} tau2={r[2]} sig2={r[3]} o={r[4]} m0={r[5]} lp_same={r[6]}")
print(len(rows))
