r"""D27-ITEM7: derived-selector histories for windows tau_p = 1.70 and 1.72, using the authors' own DerivedPop
(executes only the definition blocks of item7_checks/checks.py: lines 1-231 and 413-525; no scans).

Purpose: the minimizer along the flow with individual rows contains R1 u R2 but not L1 u L2 on [1.75, 1.77)
(check (22) output and d27_derived.py). A handoff at t = 1.75 (tau_p = 1.70 on the 0.05 evaluation grid) should
then select L1, L2 and later R1 u R2, a regime the text's three-regime account does not list.
"""
import os, io, contextlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
base = r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks.py"
L = open(base, encoding="utf-8").readlines()
assert L[412].startswith("# Derived selector") and L[523].startswith("    return DerivedPop"), (L[412], L[523])
ns = {"__name__": "item7_defs"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile("".join(L[:231]) + "".join(L[412:525]), "checks_defs", "exec"), ns)
dpop, o, y0 = ns["dpop"], ns["o"], ns["y0"]
for Tp_ in (1.70, 1.72):
    for mode in ("full", "null"):
        P = dpop(o, Tp_, mode); yf = P.run(y0, 120.0 if mode == "full" else 20.0)
        print(f"tau_p = {Tp_}, {mode}: history {[(h[0], h[1], h[2]) for h in P.hist]}")
