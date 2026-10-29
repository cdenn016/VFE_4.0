r"""GAP07-S3, focus questions 2 and 3: checks (22) and (23) of the derived selector rerun with the minimizer over ALL
coarsenings (side reduction of gap07_parts) checked against the authors' interval-coarsening minimizer at every
evaluation step.

(22) Flow with individual rows from the fixed draw, dt = 0.01 on [0, 2.2]: switch times and margins of both minimizers,
     the time the left and right gaps fall below d_u - 0.005 = 0.958.
(23) DerivedPop histories on the fixed draw for tau_p = 0.5, 1.25, 2.0 (full, null; off at 1.25), stored horizons
     (300 for full/off, 100 for null). At every evaluation the all-coarsening minimizer is compared with the interval one.
     For tau_p = 0.5 the left gap at the handoff is compared with d_u + 0.005 = 0.968 (trapping band).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_load import load, draws
import gap07_parts as GP

T0 = time.time()
ns = load(); D = draws(ns)
y0 = ns["y0"]; o = ns["o"]; FB, BS = ns["F_batch"], ns["block_stats"]
SP, DP, IM = ns["SharedPop"], ns["DerivedPop"], ns["interval_minimizer"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"])
key = lambda bl: sorted(tuple(sorted(u)) for u in bl if len(u) > 1)

# (22)
Pind = SP(o, *ARGS, ns["Tp"], mode="off"); rk = ns["rk"]
yv = y0.copy(); tv = 0.0; prev_i = prev_a = None; sw_i, sw_a = [], []; mg_i = mg_a = np.inf; tL = tR = None
agree = True; minmix = np.inf
while tv < 2.2 - 1e-9:
    sing = [frozenset([i]) for i in range(12)]
    bi, mi = IM(yv, sing); ba, ma, va, ea, mx = GP.coarsening_min(yv, sing, FB, BS)
    agree &= key(bi) == key(ba); minmix = min(minmix, mx - va)
    if key(bi) != prev_i: sw_i.append((round(tv, 2), key(bi), round(float(mi), 4))); prev_i = key(bi)
    if key(ba) != prev_a: sw_a.append((round(tv, 2), key(ba), round(float(ma), 4))); prev_a = key(ba)
    if tv <= 1.3 + 1e-9: mg_i = min(mg_i, mi); mg_a = min(mg_a, ma)
    if tL is None and yv[3:6].mean() - yv[:3].mean() < 0.958: tL = round(tv, 2)
    if tR is None and yv[9:12].mean() - yv[6:9].mean() < 0.958: tR = round(tv, 2)
    yv = rk(Pind.vel, yv, 0.01, 1); tv += 0.01
print("(22) interval minimizer switches:", sw_i)
print("     all-coarsening minimizer switches:", sw_a)
print(f"     agree at every step on [0, 2.2]: {agree}; smallest margin on [0, 1.3]: interval {mg_i:.4f} (stored 0.8618),"
      f" all coarsenings {mg_a:.4f}; mixing bound exceeds the minimum by >= {minmix:.2f}")
print(f"     left gap below 0.958 from t = {tL} (stored 0.96); right gap from t = {tR}")


class Checked(DP):
    r"""DerivedPop whose selector also computes the all-coarsening minimizer at every evaluation."""

    def select(s, y, t):
        if not hasattr(s, "chk"): s.chk = []
        bi, mi = IM(y, s.top); ba, ma, va, ea, mx = GP.coarsening_min(y, s.top, FB, BS)
        s.chk.append((t, key(bi) == key(ba), mi, ma, mx > va, ea))
        DP.select(s, y, t)


def summarize(P):
    c = P.chk
    ok = all(x[1] for x in c); mixok = all(x[4] for x in c)
    pre = [x for x in c if x[0] <= (P.hist[0][0] if P.hist else 1e9) + 1e-9]
    return ok, mixok, (min(x[2] for x in pre), min(x[3] for x in pre)) if pre else None, max(x[5] for x in c)


STORED = open(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks_output.txt",
              encoding="utf-8").read().splitlines()
for Tp_ in (0.5, 1.25, 2.0):
    row = []
    for mode in ("full", "null") + (("off",) if Tp_ == 1.25 else ()):
        P = Checked(o, *ARGS, Tp_, mode=mode)
        if mode == "full" and Tp_ == 0.5:
            yh = P.run(y0, 0.55); ghand = yh[3:6].mean() - yh[:3].mean(); ghR = yh[9:12].mean() - yh[6:9].mean()
            yf = P.run(yh, 300.0)
        else:
            yf = P.run(y0, 300.0 if mode != "null" else 100.0)
        row.append((mode, [(h[0], h[1], len(h[2])) for h in P.hist], round(float(yf[3:6].mean() - yf[:3].mean()), 4)))
        ok, mixok, pre, emax = summarize(P)
        print(f"     Tp={Tp_} {mode}: all-coarsening == interval minimizer at all {len(P.chk)} evaluations: {ok};"
              f" mixing excluded: {mixok}; margins (interval, all) up to the first handoff: "
              f"({pre[0]:.4f}, {pre[1]:.4f}); eps <= {emax:.1e}")
    line = [l for l in STORED if l.startswith(f"(23) Tp = {Tp_}:")][0]
    print(f"(23) Tp = {Tp_}: {row}")
    print("     identical to stored line:", line == f"(23) Tp = {Tp_}: {row}")
print(f"     Tp = 0.5: left gap at the handoff t = 0.55: {ghand:.4f}, right {ghR:.4f}; above d_u + 0.005 = 0.9680:"
      f" {ghand > 0.968 and ghR > 0.968}")
print("elapsed", round(time.time() - T0, 1), "s")
