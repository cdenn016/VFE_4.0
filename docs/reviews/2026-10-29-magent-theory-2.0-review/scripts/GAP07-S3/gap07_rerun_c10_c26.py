r"""GAP07-S3, focus question 3: reduced-size reruns of checks (10) and (26), counterfactual first-stage handoffs.

Code of checks.py lines 284-293 and 673-680 with the horizons reduced from 4000 and 2000 to 100 and 200: every stored
union occurs before t = 155.6, and a handoff gap above d_u + 0.005 = 0.968 never unites (certified trapping band of the
gap equation, gap07_certify_output.txt, plus the reduction of it7:reduced). Union times and handoff gaps are compared with
the stored lines; final gaps at the reduced horizon are reported instead of the converged ones.
"""
import os, sys, time, copy, re
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_load import load

T0 = time.time()
ns = load()
o, y0 = ns["o"], ns["y0"]
SP, dpop = ns["SharedPop"], ns["dpop"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"], ns["Tp"])
STORED = open(r"C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/item7_checks/checks_output.txt",
              encoding="utf-8").read().splitlines()

for mode in ("full", "off"):
    Pb = SP(o, *ARGS, mode=mode); yb = Pb.run(y0, 2.6, dt=0.05)
    out = []
    for s_ in (0.0, 0.1, 0.15, 0.25, 0.4):
        Qc = copy.deepcopy(Pb); yc = yb.copy(); yc[[0, 1, 2]] -= s_; yc[[3, 4, 5]] += s_
        yf = Qc.run(yc, 100.0)
        un = [h[0] for h in Qc.hist if h[1] == 2 and min(h[2]) == 0]
        out.append((s_, round(float(yc[3:6].mean() - yc[:3].mean()), 3), un[0] if un else None,
                    round(float(yf[3:6].mean() - yf[:3].mean()), 4)))
    st = [l for l in STORED if l.startswith(f"(10) {mode}:")][0]
    st_t = re.findall(r"\(([0-9.]+), ([0-9.]+), ([0-9.]+|None), ([0-9.]+)\)", st)
    same = [(float(a), float(b), None if c == "None" else float(c)) == (x[0], x[1], x[2]) for (a, b, c, d), x in zip(st_t, out)]
    print(f"(10) {mode} (horizon 100): (shift, gap at handoff, union time, gap at t=100) {out}")
    print(f"     shift, handoff gap and union time identical to stored for every shift: {all(same)};"
          f" non-uniting handoff gaps above 0.968: {[x[1] > 0.968 for x in out if x[2] is None]}")

for mode, shifts in (("full", (-0.15, -0.05, 0.0, 0.025, 0.05, 0.1, 0.25)), ("off", (0.0, 0.1, 0.25, 0.4))):
    Pb = dpop(o, 1.25, mode); yb = Pb.run(y0, 1.35); out = []
    for s_ in shifts:
        Qc = copy.deepcopy(Pb); yc = yb.copy(); yc[[0, 1, 2]] -= s_; yc[[3, 4, 5]] += s_
        yf = Qc.run(yc, 200.0); un = [h[0] for h in Qc.hist if h[1] == 2 and min(h[2]) == 0]
        out.append((round(float(yc[3:6].mean() - yc[:3].mean()), 3), un[0] if un else None,
                    round(float(yf[3:6].mean() - yf[:3].mean()), 4), float(yc[3:6].mean() - yc[:3].mean())))
    st = [l for l in STORED if l.startswith(f"(26) {mode}:")][0]
    st_t = re.findall(r"\(([0-9.]+), ([0-9.]+|None), ([0-9.]+)\)", st.split("final gap)")[1])
    same = [(float(a), None if b == "None" else float(b)) == (x[0], x[1]) for (a, b, c), x in zip(st_t, out)]
    print(f"(26) {mode} (horizon 200): first stage {[(h[0], h[1], len(h[2])) for h in Pb.hist]};"
          f" (gap at handoff, union time, gap at t=200) {[x[:3] for x in out]}")
    print(f"     handoff gap and union time identical to stored for every shift: {all(same)};"
          f" unrounded handoff gaps {[round(x[3], 5) for x in out]};"
          f" non-uniting handoff gaps above 0.968: {[x[3] > 0.968 for x in out if x[1] is None]};"
          f" uniting gaps below 0.958: {[x[3] < 0.958 for x in out if x[1] is not None]}")
print("elapsed", round(time.time() - T0, 1), "s")
