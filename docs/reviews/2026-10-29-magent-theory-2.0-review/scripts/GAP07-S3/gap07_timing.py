r"""GAP07-S3: alignment and timing probe (no claims). Confirms that the replayed draws match checks.py by
recomputing the null-box smallest interval-minimizer margin of check (22) (stored value 1.8249), and times one
SharedPop run and one DerivedPop run of the instance so that the reduced reruns can be sized to the 30 s cap."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gap07_load import load, draws

t0 = time.time(); ns = load(); D = draws(ns); print("load + draws", round(time.time() - t0, 2), "s")
sing = [frozenset([i]) for i in range(12)]
mgs = []; ok = True
for yb in D["c22"]:
    bi, mg = ns["interval_minimizer"](yb, sing); mgs.append(mg)
    ok &= sorted(tuple(sorted(u)) for u in bi) == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
print("check (22) replay: four groups at all 200:", ok, "; smallest margin", round(min(mgs), 4), "(stored 1.8249)")
SP = ns["SharedPop"]; o = ns["o"]; y0 = ns["y0"]
args = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"], ns["Tp"])
t0 = time.time(); P = SP(o, *args, mode="full"); P.run(y0, 20.0); t1 = time.time()
print("SharedPop full to t=20 (dt 0.05):", round(t1 - t0, 2), "s; events", [(h[0], h[1], len(h[2])) for h in P.hist])
t0 = time.time(); Pd = ns["dpop"](o, 1.25, "full"); Pd.run(y0, 20.0); t1 = time.time()
print("DerivedPop full to t=20 (Tp 1.25):", round(t1 - t0, 2), "s; events", [(h[0], h[1], len(h[2])) for h in Pd.hist])
