r"""GAP07-S3, focus question 3: reduced-size rerun of check (12), the twelve uniform draws of the admitted class under
the declared selector, plus the gaps that the counterfactual corollary needs and a continuous-time cross-check.

Authors' SharedPop with the replayed draws of check (12). Horizon reduced from t = 400 to t = 12: the null flow never
selects from any state of A (it7:null, proved), and after both unions only two walled units remain. For each draw the
left and right parent gaps at t = 2.6 (after the four first-stage handoffs) are compared with d_u - 0.005 = 0.958, below
which it7:counterfactual (with the certified signs) gives the union. Cross-check: the source of
scripts/D27-ITEM7/d27_hybrid.py exec'd read-only (continuous-time candidate onsets), full mode, h = 0.01.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_load import load, draws

T0 = time.time()
ns = load(); D = draws(ns)
SP = ns["SharedPop"]; o = ns["o"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"], ns["Tp"])
H27 = {}
exec(compile(open(os.path.join(os.path.dirname(HERE), "D27-ITEM7", "d27_hybrid.py"), encoding="utf-8").read(),
             "d27_hybrid", "exec"), H27)
Hybrid = H27["Hybrid"]
L1, L2, R1, R2 = (frozenset(range(3 * b, 3 * b + 3)) for b in range(4))

seqs = set(); t1s, t2s = [], []; gl, gr = [], []; c1, c2 = [], []; inA = []
for k, yy in enumerate(D["c12"]):
    inA.append(np.abs(yy - o).max() < 0.005)
    P = SP(o, *ARGS, mode="full"); y = P.run(yy, 2.6)
    m = {u: y[list(u)].mean() for u in P.top}
    gl.append(m[L2] - m[L1]); gr.append(m[R2] - m[R1])
    P.run(y, 12.0)
    Pz = SP(o, *ARGS, mode="null"); Pz.run(yy, 12.0)
    t1s += [h[0] for h in P.hist if h[1] == 1]; t2s += [h[0] for h in P.hist if h[1] == 2]
    seqs.add((tuple(sorted((h[1], h[2]) for h in P.hist)), len(Pz.hist)))
    Hf = Hybrid(o, yy); Hf.run(12.0, h=0.01)
    c1 += [t for t, lev, u in Hf.hist if lev == 1]; c2 += [t for t, lev, u in Hf.hist if lev == 2]
    print(f"draw {k:2d}: events {[(h[0], h[1], len(h[2])) for h in P.hist]}; null events {len(Pz.hist)};"
          f" gaps at t=2.6 L {gl[-1]:.4f} R {gr[-1]:.4f}; continuous-time {[(round(t, 3), lev) for t, lev, u in Hf.hist]}")
print("all draws in A:", all(inA))
print("(12) distinct (selected blocks by level, number of null events) over 12 draws:", len(seqs))
for s_ in seqs: print("     ", s_)
print("     first-stage times in [%.2f, %.2f], union times in [%.2f, %.2f]  (stored: [2.15, 2.30], [7.05, 8.05])"
      % (min(t1s), max(t1s), min(t2s), max(t2s)))
print("     continuous-time: first-stage in [%.4f, %.4f], unions in [%.4f, %.4f]" % (min(c1), max(c1), min(c2), max(c2)))
print("     parent gaps at t = 2.6: left in [%.4f, %.4f], right in [%.4f, %.4f]; all below d_u - 0.005 = 0.958: %s"
      % (min(gl), max(gl), min(gr), max(gr), max(gl + gr) < 0.958))
print("elapsed", round(time.time() - T0, 1), "s")
