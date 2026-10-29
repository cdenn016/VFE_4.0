r"""D27-ITEM7: independent check of the derived-selector statements of it7:derived-events along the flow with
individual rows from the fixed draw (no handoffs), using structure-gauss written from the text.

Prints: the interval-coarsening minimizer of the structure free energy with its margin at t = 0 and at every
change, the first time the left gap falls below d_u - 0.005 = 0.958, the smallest margin on [0, 1.3], and a
comparison with the minimizer over all 41,209 side-respecting partitions at a few times.
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from d27_hybrid import Hybrid

LAM, PS, G, T = 6.0, 0.05, 0.0, 1.0
PP = (1 - PS) / 11
o, y0 = np.loadtxt(os.path.join(here, "d27_fixed_draw.txt"))


def F_blocks(y, labels_batch):
    """Structure free energy minus records for a batch of partitions given as label arrays (B, 12)."""
    B = labels_batch.shape[0]; K = 12
    idx = (np.arange(B)[:, None] * K + labels_batch).ravel()
    n = np.bincount(idx, minlength=B * K).reshape(B, K).astype(float)
    S1 = np.bincount(idx, weights=np.tile(y, B), minlength=B * K).reshape(B, K)
    S2 = np.bincount(idx, weights=np.tile(y * y, B), minlength=B * K).reshape(B, K)
    m = np.where(n > 0, S1 / np.maximum(n, 1), 0.0)
    W = np.maximum(S2 - n * m * m, 0.0)
    expo = -(n[:, :, None] * LAM / (2 * T)) * (m[:, :, None] - m[:, None, :]) ** 2   # (B, U, V)
    pri = np.broadcast_to((n * PP)[:, None, :], (B, K, K)).copy()
    ii = np.arange(K); pri[:, ii, ii] = (n - 1).clip(min=0) * PP                      # own parent
    pri = np.where(n[:, None, :] > 0, pri, 0.0)
    terms = pri * np.exp(expo)
    terms[:, ii, ii] = (n - 1).clip(min=0) * PP                                      # own parent at cost zero
    Z = terms.sum(2) + PS * np.exp(-(n * LAM / (2 * T)) * (m - G) ** 2)
    val = np.where(n > 0, LAM / 2 * W - T * np.log(np.where(n > 0, Z, 1.0)), 0.0)
    return val.sum(1)


cuts = np.array(list(itertools.product([0, 1], repeat=11)))
def interval_min(y):
    order = np.argsort(y)
    lab_sorted = np.concatenate([np.zeros((len(cuts), 1), int), np.cumsum(cuts, 1)], 1)
    lab = np.empty_like(lab_sorted); lab[:, order] = lab_sorted
    Fv = F_blocks(y, lab); o2 = np.argsort(Fv)
    best = lab[o2[0]]
    blocks = sorted(tuple(np.nonzero(best == b)[0].tolist()) for b in np.unique(best))
    return [b for b in blocks if len(b) > 1], Fv[o2[1]] - Fv[o2[0]], Fv[o2[0]]


def rgs(n):
    out = [[0]]
    for k in range(1, n):
        new = []
        for r in out:
            for v in range(max(r) + 2):
                new.append(r + [v])
        out = new
    return np.array(out)
R6 = rgs(6)
side = np.array([np.concatenate([a, b + 6]) for a in R6 for b in R6])

P = Hybrid(o, y0); P.cand = {}
y = y0.copy(); t = 0.0; h = 0.01
prev = None; t_cross = None; mg_early = np.inf; checks = {}
while t < 2.0 - 1e-12:
    bl, mg, Fmin = interval_min(y)
    if bl != prev:
        print(f"t = {t:.2f}: interval minimizer blocks {bl}, margin {mg:.4f}"); prev = bl
    if t <= 1.3 + 1e-9: mg_early = min(mg_early, mg)
    if t_cross is None and y[3:6].mean() - y[:3].mean() < 0.958: t_cross = round(t, 2)
    for tt in (0.0, 1.0, 1.74, 1.76, 1.78):
        if abs(t - tt) < 5e-3: checks[tt] = y.copy()
    y = P.rk4(y, h); t += h                     # flow with individual rows (no selection)
print("smallest interval-minimizer margin on [0, 1.3]:", round(float(mg_early), 4))
print("left gap below 0.958 from t =", t_cross)
for tt, ys in sorted(checks.items()):
    Fv = F_blocks(ys, side); best = side[np.argmin(Fv)]
    blocks = sorted(tuple(np.nonzero(best == b)[0].tolist()) for b in np.unique(best))
    print(f"t = {tt}: side-respecting minimizer {[b for b in blocks if len(b) > 1]}; interval minimizer "
          f"{interval_min(ys)[0]}")

# right-side gap: first time below d_u - 0.005 along the same flow
y = y0.copy(); t = 0.0; tR = None
while t < 2.0 - 1e-12 and tR is None:
    if y[9:12].mean() - y[6:9].mean() < 0.958: tR = round(t, 2)
    y = P.rk4(y, h); t += h
print("right gap below 0.958 from t =", tR)
