r"""GAP07-S3, focus question 3: reduced-size rerun of check (24), the twelve draws of the admitted class under the derived
selector at tau_p = 1.25, with the all-coarsening minimizer checked at every evaluation.

Authors' DerivedPop with the replayed draws of check (24). Horizons reduced from 300 (full) and 60 (null) to 70 and 5:
the stored union times end at 60.90, after both unions no coarsening other than the side-mixing one exists (F >= 19), and
the null history after its first stage is fixed by Gamma_3 < -2.5 on the invariant box (gap07_certify) together with the
box certificate of gap07_partitions. The left and right gaps at the handoff t = 1.30 are compared with d_u - 0.005 = 0.958.
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
o = ns["o"]; FB, BS = ns["F_batch"], ns["block_stats"]
DP, IM = ns["DerivedPop"], ns["interval_minimizer"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"])
key = lambda bl: sorted(tuple(sorted(u)) for u in bl if len(u) > 1)


class Checked(DP):
    def select(s, y, t):
        if not hasattr(s, "chk"): s.chk = []
        bi, mi = IM(y, s.top); ba, ma, va, ea, mx = GP.coarsening_min(y, s.top, FB, BS)
        s.chk.append((t, key(bi) == key(ba), ma, mx > va))
        DP.select(s, y, t)


seqs = set(); t2s = []; allok = True; pre_m = np.inf; gaps = []
for k, yy in enumerate(D["c24"]):
    Pf = Checked(o, *ARGS, 1.25, mode="full"); yh = Pf.run(yy, 1.30)
    gaps.append((yh[3:6].mean() - yh[:3].mean(), yh[9:12].mean() - yh[6:9].mean()))
    Pf.run(yh, 70.0)
    Pz = Checked(o, *ARGS, 1.25, mode="null"); Pz.run(yy, 5.0)
    seqs.add((tuple(sorted((h[1], h[2]) for h in Pf.hist)), tuple(sorted((h[1], h[2]) for h in Pz.hist))))
    t2s += [h[0] for h in Pf.hist if h[1] == 2]
    ok = all(c[1] and c[3] for c in Pf.chk + Pz.chk); allok &= ok
    pre_m = min(pre_m, min(c[2] for c in Pf.chk if c[0] <= 1.3 + 1e-9))
    print(f"draw {k:2d}: full {[(h[0], h[1], len(h[2])) for h in Pf.hist]}; null {[(h[0], h[1], len(h[2])) for h in Pz.hist]};"
          f" gaps at t=1.30 L {gaps[-1][0]:.4f} R {gaps[-1][1]:.4f}; all-coarsening == interval at every evaluation: {ok}")
print("(24) distinct (shared-label, null) partition sequences over 12 draws:", len(seqs))
for s_ in seqs: print("     ", s_)
print("     union times in [%.2f, %.2f]  (stored [36.50, 60.90])" % (min(t2s), max(t2s)))
print(f"     all-coarsening minimizer equals the interval minimizer at every evaluation of all 24 runs: {allok};"
      f" smallest all-coarsening margin on [0, 1.30] over the 12 full runs: {pre_m:.4f}")
gl = [g[0] for g in gaps] + [g[1] for g in gaps]
print(f"     gaps at the handoff t = 1.30 in [{min(gl):.4f}, {max(gl):.4f}]; all below 0.958: {max(gl) < 0.958};"
      f" all above the radius d_*(3) = 0.7685: {min(gl) > 0.7685}")
print("elapsed", round(time.time() - T0, 1), "s")
