r"""Coarse search for a two-stage instance of the declared sweep dynamics.

Looks for (rho^2, tau^2, sigma^2, records, initial means) with
  full history   finest -> [12|34 ; 12|34] -> [12|34 ; 1234],
and reports the null history (selector at kappa o), the frozen-environment
history, and the exact MAP structure.  Single-threaded, small.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name, history, compress  # noqa: E402

s2 = np.ones(4)
hits = 0
patterns = {
    "1324": np.array([-1.5, 0.5, -0.5, 1.5]),
    "1234": np.array([-1.5, -0.5, 0.5, 1.5]),
    "3142": np.array([-0.5, 1.5, -1.5, 0.5]),
}
for r2, tau2, sig2 in itertools.product([0.1, 0.25, 0.5], [0.02, 0.05, 0.1], [1.0, 4.0, 16.0]):
    rho2 = np.full(4, r2)
    M = Model(s2, rho2, tau2, sig2)
    v0 = 1.0 + tau2 + sig2
    kappa = v0 / (v0 + r2)
    for c1, gap12, e in itertools.product([0.0, 2.0, 4.0], [1.5, 2.0, 2.5, 3.0, 4.0], [0.05, 0.2]):
        c2 = c1 + gap12
        o = np.array([c1 - e, c1 + e, c2 - e, c2 + e])
        for (pn, pat), spread in itertools.product(patterns.items(), [3.0, 6.0, 12.0]):
            m0 = spread * pat
            if M.select(m0)[0] != FINEST4:
                continue
            sel, gaps, ms = history(M, m0, o, 80)
            P = compress(sel)
            if P != [FINEST4, A_SEP, A_JOIN]:
                continue
            S0, _ = M.select(kappa * o)
            m1 = M.a * o + M.b * (M.K(FINEST4) @ m0)
            Pfr = compress([FINEST4, M.select(m1)[0]])
            ex = M.exact_log_post(o)
            hits += 1
            print(f"rho2={r2} tau2={tau2} sig2={sig2} c1={c1} gap={gap12} e={e} m0={pn}x{spread} min_gap={min(gaps):.3g}")
            print("   null sel(kappa o):", name(S0), "  Pfr:", [name(S) for S in Pfr], "  exact MAP:", name(ex[0][1]), f"{ex[0][0]-ex[1][0]:.3g}")
print("hits", hits)
