r"""D27-ITEM7: event history of the instance from the fixed draw with the independent simulator d27_hybrid.py,
at two step sizes, the post-first-stage gap, and the first-stage margins over the persistence windows.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from d27_hybrid import Hybrid, LAM, DC, DW

o, y0 = np.loadtxt(os.path.join(here, "d27_fixed_draw.txt"))
t0 = time.time()
for h in (0.01, 0.005):
    P = Hybrid(o, y0)
    gap = {}
    def rec(t, y):
        if abs(t - 2.6) < h / 2: gap["2.6"] = y[3:6].mean() - y[:3].mean()
    P.run(10.0, h=h, record=rec)
    print(f"h={h}: events", [(round(t, 4), lev, len(u)) for t, lev, u in P.hist], "; gap d(2.6) =", round(gap["2.6"], 5))
print("time", round(time.time() - t0, 2), "s")

# margins over the persistence windows (unit-level walls as in the selector; agent-level walls as in check (17))
P = Hybrid(o, y0)
traj = []
P.run(2.5, h=0.005, record=lambda t, y: traj.append((t, y.copy(), [tuple(u) for u in P.units])))
groups = [tuple(range(3 * b, 3 * b + 3)) for b in range(4)]
onset = {}
for g in groups:
    for t, y, units in traj:
        gaps = np.diff(np.sort(y[list(g)]))
        if LAM / 2 * gaps.max() ** 2 < DC:
            onset[g] = t; break
hand = {tuple(u): t for t, lev, u in P.hist}
for g in groups:
    cohm, wallu, walla = np.inf, np.inf, np.inf
    for t, y, units in traj:
        if not (onset[g] + 0.05 <= t <= hand[g] - 1e-9): continue
        cohm = min(cohm, DC - LAM / 2 * np.diff(np.sort(y[list(g)])).max() ** 2)
        means = [y[list(u)].mean() for u in units if not set(u) & set(g)]
        wallu = min(wallu, min(LAM / 2 * (y[i] - mm) ** 2 for i in g for mm in means) - DW)
        walla = min(walla, min(LAM / 2 * (y[i] - y[j]) ** 2 for i in g for j in range(12) if j not in g) - DW)
    print(f"group {g}: onset {onset[g]:.3f}, handoff {hand[g]:.3f}; min coherence margin {cohm:.5f}, "
          f"min unit-level wall margin {wallu:.4f}, min agent-level wall margin {walla:.4f}")
