r"""GAP07-S3: the record-gap window of the declared-selector instance (line 279, "the parameter window is narrow").

For s in a grid on [1.00, 1.24], records(s) with the fixed-draw offsets y0 - o of check (7), authors' SharedPop (full
mode, dt 0.05, horizon 300; a gap above d_u(s) + 0.005 after the first stage never unites, by the trapping band). Reports the partition sequence, the left/right parent gaps at t = 2.6, and the middle root d_u(s) of the gap
equation (float brentq; the interval-certified values for s = 1.1, 1.2, 1.3 are in gap07_certify_output.txt).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
from gap07_load import load

T0 = time.time()
ns = load(); SP = ns["SharedPop"]; records = ns["records"]; o, y0 = ns["o"], ns["y0"]
ARGS = (ns["lam"], ns["rho"], ns["ps"], ns["g"], ns["delta"], ns["Delta"], ns["Tp"])
off = y0 - o


def Fd(d, s):
    e = 3 * np.exp(-9 * d * d); return 0.005 * (s - d) - 2 * e / (2 + e) * d


L1, L2, R1, R2 = (frozenset(range(3 * b, 3 * b + 3)) for b in range(4))
S_LIST = sorted(set(np.round(np.arange(1.00, 1.245, 0.02), 2).tolist() + [1.05, 1.07, 1.15, 1.17, 1.19]))
for s in S_LIST:
    oo = records(s); P = SP(oo, *ARGS, mode="full"); y = P.run(oo + off, 2.6)
    m = {u: y[list(u)].mean() for u in P.top}
    gp = (round(m[L2] - m[L1], 4), round(m[R2] - m[R1], 4)) if all(u in m for u in (L1, L2, R1, R2)) else None
    P.run(y, 300.0)
    ds = np.linspace(1e-4, s, 20001); v = Fd(ds, s)
    rts = [brentq(Fd, ds[j], ds[j + 1], args=(s,)) for j in range(len(ds) - 1) if v[j] * v[j + 1] < 0]
    du = rts[1] if len(rts) == 3 else None
    print(f"s = {s:.2f}: events {[(h[0], h[1], len(h[2])) for h in P.hist]}; gaps at t=2.6 {gp};"
          f" d_u(s) = {None if du is None else round(du, 4)}")
# saddle-node value s_c at which F (n = 3) acquires its middle and separated zeros: count zeros, bisect in s
def nzeros(s):
    ds = np.linspace(1e-4, s, 200001); v = Fd(ds, s); return int(np.sum(v[:-1] * v[1:] < 0))
lo_, hi_ = 1.0, 1.1
assert nzeros(lo_) == 1 and nzeros(hi_) == 3
for _ in range(40):
    mid = (lo_ + hi_) / 2
    if nzeros(mid) == 3: hi_ = mid
    else: lo_ = mid
print(f"saddle-node of the gap equation (n = 3): three zeros for s > s_c, one for s < s_c, s_c in [{lo_:.6f}, {hi_:.6f}] (float grid count)")
print("elapsed", round(time.time() - T0, 1), "s")
