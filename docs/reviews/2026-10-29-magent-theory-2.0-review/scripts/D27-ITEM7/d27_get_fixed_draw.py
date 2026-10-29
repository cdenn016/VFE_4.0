r"""D27-ITEM7: recover the fixed draw y0 and the records o of item7_checks/checks.py without running its scans.

Executes only lines 1-231 of checks.py (imports, checks (1)-(5), class definitions and the draw), which use
the random generator before y0 is drawn; everything printed by them is discarded. Saves o and y0.
"""
import os, io, contextlib, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

src = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"
with open(src, "r", encoding="utf-8") as fh:
    lines = fh.readlines()
prefix = "".join(lines[:231])
assert "y0 = o + 0.005" in lines[230], lines[230]
ns = {"__name__": "item7_prefix"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(prefix, "checks_prefix", "exec"), ns)
o, y0 = ns["o"], ns["y0"]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d27_fixed_draw.txt")
np.savetxt(out, np.vstack([o, y0]), fmt="%.17g", header="row 0: records o; row 1: fixed draw y0 (checks.py line 231)")
print("o  =", np.round(o, 4))
print("y0 =", np.array2string(y0, precision=6))
print("max |y0 - o| =", np.abs(y0 - o).max())
