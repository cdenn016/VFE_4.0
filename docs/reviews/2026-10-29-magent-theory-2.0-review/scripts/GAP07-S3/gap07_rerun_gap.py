r"""GAP07-S3, focus question 3: reduced-size reruns of checks (11) and (14) (record-gap variants), with the null control
the text omits and a continuous-time cross-check.

Uses the authors' own SharedPop (checks.py lines 1-231, loaded without running the scans) and the random draws of
checks (11) and (14), replayed exactly (gap07_load.draws). Horizon reduced from t = 1500 / 3000 to t = 30: every event
of the stored output occurs before t = 10.4, and after t = 30 no event is possible: after the unions only two walled
units remain (unit means about 5 apart, KL about 75 > delta_w, never coherent), and where no union occurred the parent
gaps of both sides at t = 2.6 lie above d_u(s) + 0.005, a forward-invariant band of the gap equation certified in
gap07_certify_output.txt (requires the reduction |r| <= 1e-10 of it7:reduced, whose bounds hold verbatim for s = 1.2, 1.3).
The cross-check exec's the source of scripts/D27-ITEM7/d27_hybrid.py (read only; no bytecode is written there), an
independent implementation with continuous-time candidate onsets, for the full and null modes.
"""
import os, sys, time, re
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_load import load, draws

T0 = time.time()
ns = load(); D = draws(ns)
SP = ns["SharedPop"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"], ns["Tp"])
STORED = open(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks_output.txt",
              encoding="utf-8").read().splitlines()
H27 = {}
exec(compile(open(os.path.join(os.path.dirname(HERE), "D27-ITEM7", "d27_hybrid.py"), encoding="utf-8").read(),
             "d27_hybrid", "exec"), H27)
Hybrid = H27["Hybrid"]


def short(P):
    return [(h[0], h[1], len(h[2])) for h in P.hist]


def gaps(P, y):
    m = {u: y[list(u)].mean() for u in P.top}
    L1, L2, R1, R2 = (frozenset(range(3 * b, 3 * b + 3)) for b in range(4))
    if all(u in m for u in (L1, L2, R1, R2)):
        return round(m[L2] - m[L1], 4), round(m[R2] - m[R1], 4)
    return None


DU = {1.1: 0.963036, 1.2: 0.917321, 1.3: 0.893362}      # certified middle roots (gap07_certify_output.txt)
print("== check (11): record gap window, draws of check (11), horizon t = 30 instead of 1500 ==")
for gp in (1.0, 1.1, 1.2, 1.3):
    oo, yy = D["c11"][gp]
    row = []
    for mode in ("off", "full"):
        P = SP(oo, *ARGS, mode=mode); y = P.run(yy, 2.6); g26 = gaps(P, y); y = P.run(y, 30.0)
        row.append((mode, short(P)))
        if mode == "full":
            gfull = (g26, gaps(P, y))
    Pn = SP(oo, *ARGS, mode="null"); yn = Pn.run(yy, 30.0)
    line = [l for l in STORED if l.startswith(f"(11) gap {gp}:")][0]
    same = line == f"(11) gap {gp}: {row}"
    print(f"(11) gap {gp}: {row}")
    print(f"     identical to stored line: {same}; full-flow (left, right) parent gaps at t=2.6 and t=30: {gfull};"
          f" null events: {Pn.hist}; null max|y-o| at t=30: {np.abs(yn - oo).max():.5f}")
    Hf = Hybrid(oo, yy); Hf.run(30.0, h=0.01)
    Hn = Hybrid(oo, yy, mode="null"); Hn.run(30.0, h=0.01)
    print(f"     continuous-time simulator, full: {[(round(t, 4), lev, len(u)) for t, lev, u in Hf.hist]}; null: {Hn.hist}")

print("== check (14): depth one at record gaps 1.2 and 1.3, draws of check (14), dt 0.01 to t = 2.6, then t = 30 ==")
for gp in (1.1, 1.2, 1.3):
    oo, yy = D["c14"][gp]
    P = SP(oo, *ARGS, mode="full"); y = P.run(yy, 2.6, dt=0.01)
    m = P.A @ y; iA, iB = P.top.index(frozenset([0, 1, 2])), P.top.index(frozenset([3, 4, 5]))
    g26 = m[iB] - m[iA]; g26R = gaps(P, y)[1]
    y = P.run(y, 30.0, dt=0.05)
    line = [l for l in STORED if l.startswith(f"(14) gap {gp}:")][0]
    stored_ev = re.search(r"events (\[.*?\]), gap", line).group(1)
    stored_hg = re.search(r"handoff gap d\(2\.6\) = ([0-9.]+)", line).group(1)
    in_band = (g26 > DU[gp] + 0.005, g26R > DU[gp] + 0.005)
    print(f"(14) gap {gp}: handoff gap d(2.6) = {g26:.4f} (stored {stored_hg}), right-side gap {g26R:.4f};"
          f" events {short(P)} identical to stored: {str(short(P)) == stored_ev}; gaps at t=30: {gaps(P, y)};"
          f" both sides above d_u+0.005 = {DU[gp] + 0.005:.4f}: {in_band}")
    Pn = SP(oo, *ARGS, mode="null"); Pn.run(yy, 30.0)
    Hf = Hybrid(oo, yy); Hf.run(30.0, h=0.01)
    print(f"     null events: {Pn.hist}; continuous-time full: {[(round(t, 4), lev, len(u)) for t, lev, u in Hf.hist]}")
print("elapsed", round(time.time() - T0, 1), "s")
