r"""Search for a two-stage instance in the parent-bistable regime.

Parent catalog: {A_sep, A_join} (first level held at 12|34).  A record vector is
parent-bistable when both structures are self-consistent for the restricted
catalog.  For such records, vary the initial block means (u, v) and the initial
within-block spreads (d1, d2) and look for
   P = finest -> A_sep -> A_join   with   Pihat(kappa o) = A_sep.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name, history, compress  # noqa: E402

s2 = np.ones(4)
PAR = [A_SEP, A_JOIN]
found = 0
for r2, tau2, sig2 in itertools.product([0.25], [0.05, 0.1], [4.0, 16.0]):
    rho2 = np.full(4, r2)
    M = Model(s2, rho2, tau2, sig2)
    v0 = 1.0 + tau2 + sig2
    kappa = v0 / (v0 + r2)
    for c1, gap12 in itertools.product([0.0, 1.5, 3.0], np.arange(1.5, 3.6, 0.25)):
        e = 0.1
        o = np.array([c1 - e, c1 + e, c1 + gap12 - e, c1 + gap12 + e])
        sc = [M.select(M.fixed_point(S, o), PAR)[0] == S for S in PAR]
        if not all(sc):
            continue
        null = M.select(kappa * o)[0]
        outcomes = {}
        for u, v, d1, d2 in itertools.product(np.linspace(-6, 9, 16), np.linspace(-6, 9, 16), [3.0, 5.0], [4.0]):
            m0 = np.array([u - d1, u + d1, v - d2, v + d2])
            if M.select(m0)[0] != FINEST4:
                continue
            sel, gaps, ms = history(M, m0, o, 120)
            P = tuple(compress(sel))
            outcomes.setdefault(P, []).append((u, v, d1, d2, min(gaps)))
        target = (FINEST4, A_SEP, A_JOIN)
        if True:
            found += 1
            print(f"rho2={r2} tau2={tau2} sig2={sig2} c1={c1} gap={gap12:.2f} null={name(null)}")
            for P, lst in outcomes.items():
                print("    ", [name(S) for S in P], len(lst), "example", tuple(round(x, 3) for x in lst[0]))
print("found", found)
