r"""Search for full P = finest -> A_sep -> A_join with both nulls different.

Initial means on lines through kappa o:  m0 = kappa o + lam0 * d,
d = D(-1, 1, -1, 1) + (p, p, q, q).  Reports the frozen control too.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name  # noqa: E402
from gf_controls import run_full, run_lp, run_rn, run_frozen  # noqa: E402
from gf_model import compress  # noqa: E402

s2 = np.ones(4)
PAR = [A_SEP, A_JOIN]
target = [FINEST4, A_SEP, A_JOIN]
count = 0
stats = {}
for r2, tau2, sig2 in itertools.product([0.25], [0.05, 0.1], [4.0]):
    M = Model(s2, np.full(4, r2), tau2, sig2)
    v0 = 1.0 + tau2 + sig2
    kappa = v0 / (v0 + r2)
    for c1, gap12 in itertools.product([0.0, 1.5, 3.0], np.arange(1.5, 4.01, 0.25)):
        e = 0.1
        o = np.array([c1 - e, c1 + e, c1 + gap12 - e, c1 + gap12 + e])
        bis = [M.select(M.fixed_point(S, o), PAR)[0] == S for S in PAR]
        for D, p, q in itertools.product([1.0, 2.0], [-3.0, -1.0, 0.0, 1.0, 3.0], [-3.0, -1.0, 0.0, 1.0, 3.0]):
            d = D * np.array([-1.0, 1.0, -1.0, 1.0]) + np.array([p, p, q, q])
            for lam0 in [3.0, 6.0]:
                m0 = kappa * o + lam0 * d
                if M.select(m0)[0] != FINEST4:
                    continue
                P = compress(run_full(M, m0, o, 80)[0])
                if P != target:
                    continue
                Plp = compress(run_lp(M, m0, o, 80)[0])
                Prn = compress(run_rn(M, m0, o, 3)[0])
                stats[(Plp == P, Prn == P, tuple(bis))] = stats.get((Plp == P, Prn == P, tuple(bis)), 0) + 1
                if Plp == P or Prn == P:
                    continue
                Pfr = compress(run_frozen(M, m0, o, 60)[0])
                count += 1
                if count <= 30:
                    print(f"r2={r2} t2={tau2} g2={sig2} c1={c1} gap={gap12:.2f} D={D} p={p} q={q} lam0={lam0} bistable={bis}")
                    print("    lp:", [name(S) for S in Plp])
                    print("    rn:", [name(S) for S in Prn])
                    print("    fr:", [name(S) for S in Pfr])
print("count", count)
print(stats)
