r"""GAP07-S3, focus question 2: the all-partition minimizer of check (21) without the 4,213,597-partition enumeration.

(1) Fixed draw y0: float minimizer and margin over all partitions via the exact side reduction of gap07_parts, compared
    with the stored check (21) line (F = 7.5630, margin 1.7149), and the second-best partition named.
(2) Cross-check of the per-side formula against the authors' F_batch on the 2 x 203 side partitions.
(3) Interval certificate at y0 (y0 taken as exact binary64 values): per-side minimizer and margins in mpmath iv.
(4) The 200 sampled null-box states of check (22): all-partition margins against the stated "margin at least 1.82".
(5) Interval certificate on the whole null box prod_i [o_i - 0.005, o_i + 0.005] (decimal records), with subdivision.
(6) Timing of one 20,000-partition batch of the authors' F_batch, to size a literal rerun of check (21).
"""
import os, sys, time, itertools
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from mpmath import iv, mpf, nstr
from gap07_load import load, draws
import gap07_parts as GP

T0 = time.time()
ns = load(); D = draws(ns)
y0 = ns["y0"]; o = ns["o"]
GRP = sorted(GP.GROUPS_L + GP.GROUPS_R)

# (1) fixed draw, float
best, Fb, marg, (mL, mR), eps, mix = GP.allpart_min(y0)
fl = np.array([GP.f_own(y0, p) for p in GP.SIDE_L]); fr = np.array([GP.f_own(y0, p) for p in GP.SIDE_R])
oL, oR = np.argsort(fl), np.argsort(fr)
second = (GP.SIDE_L[oL[1]] + GP.SIDE_R[oR[0]]) if mL <= mR else (GP.SIDE_L[oL[0]] + GP.SIDE_R[oR[1]])
print("(1) fixed draw: minimizer over all partitions", best, "== four groups:", best == GRP)
print(f"    F = {Fb:.4f} (stored 7.5630); margin to the next = {marg:.4f} (stored 1.7149); per-side margins"
      f" L {mL:.4f}, R {mR:.4f}; cross-side error eps <= {eps:.2e}; mixing partitions have F >= {mix:.2f}")
print("    second-best partition:", sorted(second))
print("    third/fourth per side (L):", [GP.SIDE_L[k] for k in oL[1:4]], [round(fl[k] - fl[oL[0]], 4) for k in oL[1:4]])
print("    third/fourth per side (R):", [GP.SIDE_R[k] for k in oR[1:4]], [round(fr[k] - fr[oR[0]], 4) for k in oR[1:4]])
# interval-coarsening margin at the same state, for comparison (the authors' interval_minimizer)
sing = [frozenset([i]) for i in range(12)]
bi, mg_int = ns["interval_minimizer"](y0, sing)
print(f"    interval-coarsening minimizer {sorted(tuple(sorted(u)) for u in bi)} with margin {mg_int:.4f} (check (22): 1.8305)")

# (2) cross-check against F_batch on side-respecting partitions
def rg_of(part):
    rg = np.zeros(12, int)
    for b, blk in enumerate(part): rg[list(blk)] = b
    return rg
RG = np.array([rg_of(p + list(GP.GROUPS_R)) for p in GP.SIDE_L] + [rg_of(list(GP.GROUPS_L) + p) for p in GP.SIDE_R])
Fv = ns["F_batch"](*ns["block_stats"](y0, RG))
pred = np.concatenate([fl + fr[GP.SIDE_R.index(sorted(GP.GROUPS_R))], fl[GP.SIDE_L.index(sorted(GP.GROUPS_L))] + fr])
print(f"(2) max |F_batch - (F_L^own + F_R^own)| over 406 side-respecting partitions = {np.abs(Fv - pred).max():.2e}"
      f" (bound eps = {eps:.2e} plus rounding)")

# (3) interval certificate at y0
Y = {i: iv.mpf(float(y0[i])) for i in range(12)}
def side_cert(Y, parts, grp):
    fG = GP.f_own_iv(Y, grp)
    lows, highs = [], []
    for p in parts:
        if p == grp: continue
        d = GP.f_own_iv(Y, p) - fG
        lows.append(GP.lo(d)); highs.append(GP.hi(d))
    return min(lows), min(highs), fG
cL = side_cert(Y, GP.SIDE_L, sorted(GP.GROUPS_L)); cR = side_cert(Y, GP.SIDE_R, sorted(GP.GROUPS_R))
# rigorous eps at y0
d_iv = Y[6] - Y[5]
A_lo = GP.lo(5 * GP.IPP * iv.exp(-GP.ILAM / 2 * (Y[5] - Y[0]) ** 2))
A_lo = min(A_lo, GP.lo(5 * GP.IPP * iv.exp(-GP.ILAM / 2 * (Y[11] - Y[6]) ** 2)), GP.lo(GP.IPP))
eps_iv = GP.hi(12 * 6 * GP.IPP * iv.exp(-GP.ILAM / 2 * d_iv ** 2)) / A_lo
mixlb = GP.lo(GP.ILAM * d_iv ** 2 / 4)
FG_hi = GP.hi(cL[2] + cR[2])
print(f"(3) interval certificate at y0: per-side margins L in [{nstr(cL[0], 8)}, {nstr(cL[1], 8)}],"
      f" R in [{nstr(cR[0], 8)}, {nstr(cR[1], 8)}] (all > 0: {cL[0] > 0 and cR[0] > 0}); eps <= {nstr(eps_iv, 3)};"
      f" F(four groups) <= {nstr(FG_hi, 8)}; mixing partitions F >= {nstr(mixlb, 6)}")
print(f"    hence over all partitions the minimizer is the four groups and the margin lies in "
      f"[{nstr(min(cL[0], cR[0]) - eps_iv, 8)}, {nstr(min(cL[1], cR[1]) + eps_iv, 8)}]")

# (4) the 200 sampled null-box states of check (22)
am, im = [], []
for yb in D["c22"]:
    b_, F_, m_, _, e_, mx_ = GP.allpart_min(yb)
    assert b_ == GRP and mx_ > F_ + m_ and e_ < 1e-9
    am.append(m_); im.append(ns["interval_minimizer"](yb, sing)[1])
print(f"(4) 200 null-box states: all-partition minimizer = four groups at all; smallest all-partition margin {min(am):.4f}"
      f" (interval-coarsening margin {min(im):.4f}; the text says 'margin at least 1.82')")
print("    elapsed so far", round(time.time() - T0, 1), "s")

# (5) whole null box, interval arithmetic with subdivision of failing partitions
rec = ["-3.2", "-3.05", "-2.9", "-2.1", "-1.95", "-1.8", "1.8", "1.95", "2.1", "2.9", "3.05", "3.2"]
BOX = {i: iv.mpf([mpf(rec[i]) - mpf("0.005"), mpf(rec[i]) + mpf("0.005")]) for i in range(12)}
def box_cert(side, parts, grp, depth_max=3):
    worst = None; nfail = 0; nevals = 0
    for p in parts:
        if p == grp: continue
        stack = [(dict(BOX), 0)]
        while stack:
            Yb, dep = stack.pop(); nevals += 1
            d = GP.f_own_iv(Yb, p) - GP.f_own_iv(Yb, grp)
            if GP.lo(d) > 0:
                worst = GP.lo(d) if worst is None else min(worst, GP.lo(d)); continue
            if dep >= depth_max:
                nfail += 1; break
            k = max(side, key=lambda i: GP.hi(Yb[i]) - GP.lo(Yb[i]))
            a_, b_ = GP.lo(Yb[k]), GP.hi(Yb[k]); m_ = (a_ + b_) / 2
            for lo_, hi_ in ((a_, m_), (m_, b_)):
                Yc = dict(Yb); Yc[k] = iv.mpf([lo_, hi_]); stack.append((Yc, dep + 1))
    return worst, nfail, nevals
wL = box_cert(GP.LEFT, GP.SIDE_L, sorted(GP.GROUPS_L)); print("(5) left side:", nstr(wL[0], 6), "failures", wL[1], "evals", wL[2],
                                                               round(time.time() - T0, 1), "s")
wR = box_cert(GP.RIGHT, GP.SIDE_R, sorted(GP.GROUPS_R)); print("    right side:", nstr(wR[0], 6), "failures", wR[1], "evals", wR[2],
                                                                round(time.time() - T0, 1), "s")
dB = BOX[6] - BOX[5]
A_lo = min(GP.lo(GP.IPP), GP.lo(5 * GP.IPP * iv.exp(-GP.ILAM / 2 * (BOX[5] - BOX[0]) ** 2)),
           GP.lo(5 * GP.IPP * iv.exp(-GP.ILAM / 2 * (BOX[11] - BOX[6]) ** 2)))
eps_box = GP.hi(12 * 6 * GP.IPP * iv.exp(-GP.ILAM / 2 * dB ** 2)) / A_lo
FG_box = GP.hi(GP.f_own_iv(BOX, sorted(GP.GROUPS_L)) + GP.f_own_iv(BOX, sorted(GP.GROUPS_R)))
print(f"    box: eps <= {nstr(eps_box, 3)}; F(four groups) <= {nstr(FG_box, 6)}; mixing partitions F >= {nstr(GP.lo(GP.ILAM * dB ** 2 / 4), 6)}")

# (6) timing of one F_batch batch of 20,000 twelve-agent partitions (valid restricted growth strings)
R8 = ns["all_rgs"](8); mxs = R8.max(1)
ext = np.stack([mxs + 1, mxs + 1, mxs + 2, mxs + 2], 1)
RGb = np.concatenate([R8, ext], 1)
RGb = np.concatenate([RGb] * 5)[:20000]
t1 = time.time(); ns["F_batch"](*ns["block_stats"](y0, RGb)); t2 = time.time()
t3 = time.time(); ns["all_rgs"](10); t4 = time.time()
print(f"(6) one F_batch batch of 20,000 partitions: {t2 - t1:.2f} s; all_rgs(10) (115,975 rows): {t4 - t3:.2f} s")
print("elapsed", round(time.time() - T0, 1), "s")
