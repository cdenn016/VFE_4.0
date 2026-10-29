r"""GAP07-S3 module: minimizer of the structure free energy it7:structure-gauss over ALL partitions of the twelve agents,
reduced exactly to 2 x 203 per-side partitions.

Reduction (proved in the report). Write F(Q) = sum_U [ (lam/2) W_U - log A_U ], with
A_U = (n_U - 1) pi_p + pi_s e^{-n_U lam/2 (ybar_U - g)^2} + sum_{V != U} n_V pi_p e^{-n_U lam/2 (ybar_U - ybar_V)^2}.
(a) A_U <= (n_U - 1) pi_p + pi_s + (N - n_U) pi_p = pi_s + (N - 1) pi_p = 1, so -log A_U >= 0 and F(Q) >= (lam/2) sum_U W_U.
    A block containing a left agent and a right agent at distance >= delta has W_U >= delta^2 / 2, so every partition that
    mixes the sides has F >= lam delta^2 / 4.
(b) For a partition Q = Q_L u Q_R that respects the sides, A_U = A_U^own + c_U, where A_U^own uses the evaluator and the
    blocks of U's own side and 0 <= c_U <= (sum of the other side's sizes) pi_p e^{-n_U lam/2 delta^2}. Hence
    F(Q) = F_L^own(Q_L) + F_R^own(Q_R) - e(Q), 0 <= e(Q) <= sum_U c_U / A_U^own =: eps, and the minimizer over all partitions
    is (argmin F_L^own, argmin F_R^own) whenever the per-side margins exceed eps and lam delta^2/4 exceeds the value.
All functions below evaluate F^own exactly as the displayed formula; the float versions use numpy, the interval versions
mpmath iv (outward rounding), with W_U = (1/n_U) sum_{i<j in U} (y_i - y_j)^2 to limit dependency.
"""
import itertools
import numpy as np
from mpmath import iv, mp, mpf

LAM, PS, G0, N = 6.0, 0.05, 0.0, 12
PP = (1 - PS) / (N - 1)
LEFT, RIGHT = tuple(range(6)), tuple(range(6, 12))
GROUPS_L = ((0, 1, 2), (3, 4, 5))
GROUPS_R = ((6, 7, 8), (9, 10, 11))


def set_partitions(items):
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for k in range(len(p)):
            yield p[:k] + [(first,) + p[k]] + p[k + 1:]
        yield [(first,)] + p


SIDE_L = [sorted(tuple(sorted(b)) for b in p) for p in set_partitions(LEFT)]
SIDE_R = [sorted(tuple(sorted(b)) for b in p) for p in set_partitions(RIGHT)]
assert len(SIDE_L) == 203 and len(SIDE_R) == 203


def f_own(y, part):
    r"""Float F^own of one side's partition (evaluator and own-side blocks only)."""
    means = [np.mean(y[list(b)]) for b in part]
    tot = 0.0
    for a, b in enumerate(part):
        n = len(b); yb = y[list(b)]
        W = np.sum((yb - means[a]) ** 2)
        A = (n - 1) * PP + PS * np.exp(-n * LAM / 2 * (means[a] - G0) ** 2)
        for c, b2 in enumerate(part):
            if c != a:
                A += len(b2) * PP * np.exp(-n * LAM / 2 * (means[a] - means[c]) ** 2)
        tot += LAM / 2 * W - np.log(A)
    return tot


def cross_eps(y):
    r"""Upper bound eps on e(Q), uniform over all side-respecting partitions Q (float).

    A_U^own >= pi_p for |U| >= 2; for a singleton the other five agents of its side lie within the side's span, so
    A_U^own >= 5 pi_p e^{-lam/2 span^2}; c_U <= 6 pi_p e^{-lam/2 delta^2}; at most 12 blocks."""
    delta = np.min(y[list(RIGHT)]) - np.max(y[list(LEFT)])
    span = max(np.ptp(y[list(LEFT)]), np.ptp(y[list(RIGHT)]))
    amin = min(PP, 5 * PP * np.exp(-LAM / 2 * span ** 2))
    cmax = 6 * PP * np.exp(-LAM / 2 * delta ** 2)
    return 12 * cmax / amin, delta


def allpart_min(y):
    r"""Float minimizer over all 4,213,597 partitions via the reduction; returns
    (best partition, F(best), margin to the next, per-side margins, eps, mixing lower bound)."""
    fl = np.array([f_own(y, p) for p in SIDE_L]); fr = np.array([f_own(y, p) for p in SIDE_R])
    oL, oR = np.argsort(fl), np.argsort(fr)
    mL, mR = fl[oL[1]] - fl[oL[0]], fr[oR[1]] - fr[oR[0]]
    eps, delta = cross_eps(y)
    best = sorted(SIDE_L[oL[0]] + SIDE_R[oR[0]])
    Fbest = fl[oL[0]] + fr[oR[0]]
    mix = LAM * delta ** 2 / 4
    return best, Fbest, min(mL, mR), (mL, mR), eps, mix


# ---------------- interval versions -----------------------------------------------------------------------------
iv.dps = 30
mp.dps = 30
IPP = (1 - iv.mpf("0.05")) / 11
IPS = iv.mpf("0.05")
ILAM = iv.mpf(6)


def f_own_iv(Y, part):
    r"""Interval F^own: Y is a dict agent -> iv interval."""
    means = [sum(Y[i] for i in b) / len(b) for b in part]
    tot = iv.mpf(0)
    for a, b in enumerate(part):
        n = len(b)
        W = iv.mpf(0)
        for i, j in itertools.combinations(b, 2):
            W += (Y[i] - Y[j]) ** 2
        W = W / n
        A = (n - 1) * IPP + IPS * iv.exp(-n * ILAM / 2 * means[a] ** 2)
        for c, b2 in enumerate(part):
            if c != a:
                A += len(b2) * IPP * iv.exp(-n * ILAM / 2 * (means[a] - means[c]) ** 2)
        tot += ILAM / 2 * W - iv.log(A)
    return tot


def lo(x):
    return mpf(x._mpi_[0])


def hi(x):
    return mpf(x._mpi_[1])


# ---------------- coarsenings of arbitrary side-respecting unit structures (float, authors' F_batch) ------------
def _rg(blocks):
    rg = np.zeros(N, int)
    for b, blk in enumerate(blocks):
        rg[list(blk)] = b
    return rg


def coarsening_min(y, units, F_batch, block_stats):
    r"""Minimizer of F over ALL coarsenings of the current units, by the side reduction.

    Every unit must lie on one side. A coarsening that joins a left and a right unit has F >= lam delta^2 / 4
    (delta = smallest left-right agent distance). Side-respecting coarsenings: F(P_L u P_R) equals
    F(P_L u U_R) + F(U_L u P_R) - F(U_L u U_R) up to 2 eps (U_side = the side's units unmerged), so each side is
    minimized separately over its set partitions. Returns (blocks of the minimizer, margin to the next coarsening,
    value, eps, mixing lower bound)."""
    left = [u for u in units if max(u) < 6]; right = [u for u in units if min(u) >= 6]
    assert len(left) + len(right) == len(units), "a unit straddles the sides"
    res = []
    for mine, other in ((left, right), (right, left)):
        parts = list(set_partitions(list(range(len(mine)))))
        rows = []
        for p in parts:
            blocks = [frozenset().union(*[mine[k] for k in grp]) for grp in p] + list(other)
            rows.append(_rg(blocks))
        Fv = F_batch(*block_stats(y, np.array(rows)))
        o_ = np.argsort(Fv)
        best = [frozenset().union(*[mine[k] for k in grp]) for grp in parts[o_[0]]]
        marg = Fv[o_[1]] - Fv[o_[0]] if len(Fv) > 1 else np.inf
        res.append((best, marg, Fv[o_[0]]))
    eps, delta = cross_eps(y)
    blocks = res[0][0] + res[1][0]
    val = F_batch(*block_stats(y, _rg(blocks)[None, :]))[0]
    mix = LAM * delta ** 2 / 4
    # margin to the next coarsening of any kind: the next side-respecting one, or a mixing one (F >= mix)
    return blocks, min(res[0][1], res[1][1], mix - val), val, eps, mix
