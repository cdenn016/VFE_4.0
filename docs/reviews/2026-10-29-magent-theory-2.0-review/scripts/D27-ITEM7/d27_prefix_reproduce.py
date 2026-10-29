r"""D27-ITEM7: run only lines 1-231 of item7_checks/checks.py (checks (1)-(5) and definitions, under one second)
and compare their printed lines with item7_checks/checks_output.txt lines 1-11."""
import os, io, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
base = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/"
lines = open(base + "checks.py", encoding="utf-8").readlines()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile("".join(lines[:231]), "checks_prefix", "exec"), {"__name__": "prefix"})
got = buf.getvalue().strip().splitlines()
ref = open(base + "checks_output.txt", encoding="utf-8").read().strip().splitlines()[:len(got)]
for g, r in zip(got, ref):
    print("SAME" if g == r else "DIFF", "|", g[:110])
