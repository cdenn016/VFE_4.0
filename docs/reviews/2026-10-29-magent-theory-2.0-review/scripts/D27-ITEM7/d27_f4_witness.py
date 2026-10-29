r"""D27-ITEM7: numerical (F4) witnesses for the declared-selector instance, at both stages.

For admitted initial states y0 in A (|y0 - o| < 0.005), find y0' in A whose complete post-handoff retained state
coincides with that of y0 while the configurations differ:
  stage 1, at t* = 3 (all four first-stage handoffs done, no running candidate clock): the four unit means and
           the four first-stage handoff times;
  stage 2, at t** = 8 (both unions done, no running clock): the two union means, the four first-stage and the
           two union handoff times.
Newton (chord) iteration with a finite-difference Jacobian; integration as in d27_hybrid.py (not validated).
Also reports the singular values of D(y0 -> x(t*)) and D(y0 -> x(t**)).
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from d27_hybrid import Hybrid

o, y0 = np.loadtxt(os.path.join(here, "d27_fixed_draw.txt"))
yb = o + 0.5 * (y0 - o)                                     # admitted base point, box margin >= 0.0025
TS1, TS2, H = 3.0, 8.0, 0.01
groups = [tuple(range(3 * b, 3 * b + 3)) for b in range(4)]
unions = [tuple(range(6)), tuple(range(6, 12))]


def simulate(y):
    P = Hybrid(o, y); snap = {}
    for ts in (TS1, TS2):
        P.run(ts, h=H)
        snap[ts] = (P.y.copy(), dict(P.cand), [tuple(u) for u in P.units])
    hand = {tuple(u): t for t, lev, u in P.hist}
    return P, snap, hand


def psi(y):
    P, snap, hand = simulate(y)
    x1, c1, u1 = snap[TS1]; x2, c2, u2 = snap[TS2]
    v1 = np.array([x1[list(g)].mean() for g in groups] + [hand[g] for g in groups])
    v2 = np.array([x2[list(u)].mean() for u in unions] + [hand[g] for g in groups] + [hand[u] for u in unions])
    return v1, v2, x1, x2, (c1, u1), (c2, u2), [(round(t, 4), len(u)) for t, lev, u in P.hist]


b1, b2, xb1, xb2, s1, s2, hb = psi(yb)
print("base history:", hb, "; running candidate clocks at t*:", s1[0], "at t**:", s2[0])
eps_fd = 1e-6
J1 = np.zeros((8, 12)); J2 = np.zeros((8, 12)); X1 = np.zeros((12, 12)); X2 = np.zeros((12, 12))
for k in range(12):
    e = np.zeros(12); e[k] = eps_fd
    v1, v2, x1, x2, _, _, _ = psi(yb + e)
    J1[:, k] = (v1 - b1) / eps_fd; J2[:, k] = (v2 - b2) / eps_fd
    X1[:, k] = (x1 - xb1) / eps_fd; X2[:, k] = (x2 - xb2) / eps_fd
print("singular values of D x(t*):", np.array2string(np.linalg.svd(X1, compute_uv=False), precision=3))
print("singular values of D x(t**):", np.array2string(np.linalg.svd(X2, compute_uv=False), precision=3))

for stage, J, base, idx in (("stage 1", J1, b1, 0), ("stage 2", J2, b2, 1)):
    U, S, Vt = np.linalg.svd(J)
    Nul = Vt[8:]                                           # null space of the retained-state map (4 dims)
    X = (X1, X2)[idx]
    if idx == 0:
        M = X @ Nul.T                                      # configuration change per null coordinate
    else:
        lrow = np.zeros(12); lrow[3:6] = 1 / 3; lrow[:3] = -1 / 3     # relative coordinate ybar_L2 - ybar_L1
        M = (lrow @ X @ Nul.T)[None, :]
    w_ = np.linalg.svd(M)[2][0]
    v = Nul.T @ w_                                         # null direction that moves the forgotten data most
    yw = yb + 1.5e-3 * v / np.abs(v).max()
    Jp = np.linalg.pinv(J)
    for it in range(8):
        vals = psi(yw)
        r = vals[idx] - base
        if np.abs(r).max() < 1e-12: break
        yw = yw - Jp @ r
    vals = psi(yw)
    xw = vals[2 + idx]; xbb = (xb1, xb2)[idx]; st = vals[4 + idx]
    print(f"{stage}: Newton residual {np.abs(vals[idx] - base).max():.2e}; max|yw - o| = {np.abs(yw - o).max():.5f} "
          f"(< 0.005: {np.abs(yw - o).max() < 0.005}); history {vals[6]}; running clocks {st[0]}; units equal "
          f"{sorted(st[1]) == sorted((s1, s2)[idx][1])}")
    print(f"   max |x'(t) - x(t)| over agents = {np.abs(xw - xbb).max():.3e}; "
          f"agent-level probe y_0: {xbb[0]:.9f} vs {xw[0]:.9f}")
    if idx == 1:
        print(f"   first-stage parent means (L1, L2): base ({xbb[:3].mean():.9f}, {xbb[3:6].mean():.9f}), "
              f"witness ({xw[:3].mean():.9f}, {xw[3:6].mean():.9f}); union mean {xbb[:6].mean():.12f} vs {xw[:6].mean():.12f}")
        print(f"   relative coordinate ybar_L2 - ybar_L1: base {xbb[3:6].mean() - xbb[:3].mean():.9f}, "
              f"witness {xw[3:6].mean() - xw[:3].mean():.9f}")

# Stage 2 again: does the union event time carry the forgotten relative coordinate?
lrow = np.zeros(12); lrow[3:6] = 1 / 3; lrow[:3] = -1 / 3
sens_full = lrow @ X2                                      # gradient of ybar_L2 - ybar_L1 at t** w.r.t. y0
for name, rows in (("union means + all event times (8 constraints)", list(range(8))),
                   ("union means + first-stage times (6 constraints)", list(range(6))),
                   ("current union means only (2 constraints)", [0, 1])):
    Nul = np.linalg.svd(J2[rows])[2][len(rows):]
    print(f"stage 2, retained = {name}: largest sensitivity of the relative coordinate inside the fiber "
          f"= {np.linalg.norm(sens_full @ Nul.T):.3e} (unrestricted: {np.linalg.norm(sens_full):.3e})")
# witness with the current union means only: same union means at t**, different relative coordinate
rows = [0, 1]
Nul = np.linalg.svd(J2[rows])[2][2:]
w_ = (sens_full @ Nul.T); v = Nul.T @ (w_ / np.linalg.norm(w_))
yw = yb + 3e-4 * v / np.abs(v).max(); Jp = np.linalg.pinv(J2[rows])
for it in range(8):
    vals = psi(yw); r = vals[1][rows] - b2[rows]
    if np.abs(r).max() < 1e-12: break
    yw = yw - Jp @ r
vals = psi(yw); xw = vals[3]
print(f"stage 2 witness (current union means only): residual {np.abs(vals[1][rows] - b2[rows]).max():.2e}; "
      f"max|yw - o| = {np.abs(yw - o).max():.5f}; history {vals[6]}; running clocks {vals[5][0]}")
print(f"   union means base {xb2[:6].mean():.12f}, {xb2[6:].mean():.12f}; witness {xw[:6].mean():.12f}, {xw[6:].mean():.12f}")
print(f"   relative coordinate ybar_L2 - ybar_L1: base {xb2[3:6].mean() - xb2[:3].mean():.9f}, "
      f"witness {xw[3:6].mean() - xw[:3].mean():.9f}  (a receiver attending to A_L1 reads ybar_L1)")
