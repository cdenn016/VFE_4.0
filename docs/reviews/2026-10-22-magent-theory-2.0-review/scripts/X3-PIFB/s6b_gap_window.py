r"""X3-PIFB check 6b (numerical observation): window of initial asymmetry eps for which the ch.12 limit
field produces both formation stages while the no-peer-force control produces none (scale 3)."""
from __future__ import annotations
import importlib.util, sys, io, contextlib
import numpy as np
spec = importlib.util.spec_from_file_location("s6", "s6_gap_amplification.py")
m = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
for eps in (-0.03, -0.02, -0.01, 0.01, 0.03, 0.05, 0.1):
    x1 = 3.0 * (1.0 + eps); x3 = 9.0
    u0 = np.array([-x3, -x1, x1, x3])
    print(f"eps={eps:+.2f}: full {m.events(u0, 'full', tmax=5.0)} | control {m.events(u0, 'control', tmax=5.0)}")
