r"""Diagnostic: tabulate which partition sequences the sweep dynamics produces."""
from __future__ import annotations

import collections
import itertools
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, name, history, compress  # noqa: E402

s2 = np.ones(4)
rho2 = np.ones(4)
counter: collections.Counter = collections.Counter()
for tau2, sig2 in itertools.product([0.05, 0.2], [1.0, 4.0]):
    M = Model(s2, rho2, tau2, sig2)
    v0 = 1.0 + tau2 + sig2
    kappa = v0 / (v0 + 1.0)
    for c1, gap12, e in itertools.product([1.0, 3.0], [1.0, 2.0], [0.05]):
        c2 = c1 + gap12
        o = np.array([c1 - e, c1 + e, c2 - e, c2 + e])
        for spread in [0.0, 4.0]:
            m0 = np.array([-spread, spread, -spread * 0.5, spread * 1.5])
            sel, gaps, ms = history(M, m0, o, 60)
            P = tuple(name(S) for S in compress(sel))
            print(f"tau2={tau2} sig2={sig2} c1={c1} gap={gap12} spread={spread}: P={P}  null={name(M.select(kappa*o)[0])}  exact={name(M.exact_log_post(o)[0][1])}")
