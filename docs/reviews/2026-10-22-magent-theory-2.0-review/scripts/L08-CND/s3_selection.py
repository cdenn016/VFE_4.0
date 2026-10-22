r"""L08-CND check 3: common-precision evolution, eigenvalue barriers, threshold selection.

(a) Matrix ODE P' = -2P + I + C(P), M = N abar(t) P^{-1}, vs scalar eigenvalue ODE
    p' = -2p + 2 + 2 kappa abar/p - kappa alpha0 p/(p + kappa abar); eigenvectors fixed.
(b) Barrier claims: f>0 on (0, eps], eps = min(1/2, m_-/alpha0); f <= -2 on [L, inf), L = 2 + kappa m_+.
(c) Selection rule simulated on the full coupled ODE (alpha_i and P) by time stepping with crossing
    detection, for kappa in {0, 1, 5}: event times and tree vs closed form 1/2 log 2, 1/2 log 3, 1/2 log(17/2).
(d) Random search inside the stated gap class (theorem hypotheses) for violations.
(e) Exact region for the ((12)(34)) tree: compare the theorem's sufficient class with
    {min(d1,d3) < d2, max(d1,d3) < d2 + min(d1,d3)/2}.
(f) General N: the event-driven rule equals centroid (UPGMC) agglomerative clustering of the initial
    belief shapes, with join times 1/2 log(height/delta); no cascades in one dimension.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
N = 4


def C_of(P: np.ndarray, M: np.ndarray, kappa: float, a0: float) -> np.ndarray:
    Lam = np.eye(2) + kappa / N * M
    return Lam + kappa / N * (M - N * a0 * np.linalg.inv(Lam))


def f_scalar(p: float, abar: float, kappa: float, a0: float) -> float:
    return -2 * p + 2 + 2 * kappa * abar / p - kappa * a0 * p / (p + kappa * abar)


# ---------- (a) ----------
a0, kappa = 1.0, 1.0
alphas0 = np.array([1.0, 1.2, 1.8, 2.1])
a_ctx, chi = 0.3, 0.7
R = np.array([[np.cos(chi), -np.sin(chi)], [np.sin(chi), np.cos(chi)]])
P0 = R @ np.diag([np.exp(a_ctx), np.exp(-a_ctx)]) @ R.T


def rhs_matrix(t, y):
    al = y[:N]
    P = y[N:].reshape(2, 2)
    M = al.sum() * np.linalg.inv(P)
    Pd = -2 * P + np.eye(2) + C_of(P, M, kappa, a0)
    return np.r_[-2 * (al - a0), Pd.ravel()]


def rhs_eig(t, y):
    abar = a0 + np.exp(-2 * t) * (alphas0.mean() - a0)
    return [f_scalar(y[0], abar, kappa, a0), f_scalar(y[1], abar, kappa, a0)]


T = 3.0
sm = solve_ivp(rhs_matrix, (0, T), np.r_[alphas0, P0.ravel()], rtol=1e-11, atol=1e-13, dense_output=True)
se = solve_ivp(rhs_eig, (0, T), [np.exp(a_ctx), np.exp(-a_ctx)], rtol=1e-11, atol=1e-13, dense_output=True)
errs = []
for t in np.linspace(0, T, 31):
    P = sm.sol(t)[N:].reshape(2, 2)
    pe = se.sol(t)
    Pfrom = R @ np.diag(pe) @ R.T
    errs.append(np.max(np.abs(P - Pfrom)))
print("(a) max |P_matrixODE - R diag(p_eigODE) R^T| over t in [0,3]:", max(errs))

# ---------- (b) barriers ----------
worst_low, worst_high = np.inf, -np.inf
for _ in range(20000):
    k = rng.uniform(0.01, 10)
    a0r = rng.uniform(0.51, 5)
    ab0 = rng.uniform(0.51, 5)
    mm, mp_ = min(ab0, a0r), max(ab0, a0r)
    abar = rng.uniform(mm, mp_)
    eps = min(0.5, mm / a0r)
    L = 2 + k * mp_
    p_low = rng.uniform(1e-6, eps)
    p_high = L * (1 + rng.exponential(1.0))
    worst_low = min(worst_low, f_scalar(p_low, abar, k, a0r))
    worst_high = max(worst_high, f_scalar(p_high, abar, k, a0r) + 2)
print("(b) min f on (0,eps]:", worst_low, " (>0 claimed);  max f+2 on [L,inf):", worst_high, " (<=0 claimed)")


# ---------- event-driven exact rule ----------
def run_rule(b0: np.ndarray, delta: float):
    r"""Exact event-driven threshold rule on scores b(t) = e^{-2t} b(0); joins average with counts.
    Returns list of (time, merged-member-sets)."""
    blocks = [((i,), 1, b0[i]) for i in range(len(b0))]  # (members, count, initial-score centroid)
    events = []
    t_now = 0.0
    while len(blocks) > 1:
        best = None
        for (I, J) in itertools.combinations(range(len(blocks)), 2):
            D = abs(blocks[I][2] - blocks[J][2])
            if D * np.exp(-2 * t_now) <= delta + 1e-15:
                raise RuntimeError("distance already at/below threshold after recompute: rule undefined")
            tc = 0.5 * np.log(D / delta)
            if best is None or tc < best[0] - 1e-14:
                best = (tc, [(I, J)])
            elif abs(tc - best[0]) <= 1e-14:
                best[1].append((I, J))
        t_now = best[0]
        # connected components of simultaneous edges
        parent = list(range(len(blocks)))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for (I, J) in best[1]:
            parent[find(I)] = find(J)
        groups = {}
        for k in range(len(blocks)):
            groups.setdefault(find(k), []).append(k)
        new = []
        for g in groups.values():
            mem = tuple(sorted(sum((blocks[k][0] for k in g), ())))
            cnt = sum(blocks[k][1] for k in g)
            cen = sum(blocks[k][1] * blocks[k][2] for k in g) / cnt
            new.append((mem, cnt, cen))
            if len(g) > 1:
                events.append((t_now, mem))
        blocks = new
    return events


# ---------- (c) time-stepped full coupled ODE with crossing detection ----------
def simulate_full(kappa_: float, alphas: np.ndarray, delta: float, dt: float = 2e-4, T_end: float = 1.5):
    r"""Time-step alpha_i and P with RK4; detect first crossings of |b_I - b_J| below delta by linear
    interpolation; joins replace blocks by count-weighted means (state kept at fine level)."""
    y = np.r_[alphas, np.eye(2).ravel() * 1.3]
    blocks = [[i] for i in range(N)]
    events = []

    def f(yv):
        al = yv[:N]
        P = yv[N:].reshape(2, 2)
        M = al.sum() * np.linalg.inv(P)
        Lam = np.eye(2) + kappa_ / N * M
        Cm = Lam + kappa_ / N * (M - N * a0 * np.linalg.inv(Lam))
        return np.r_[-2 * (al - a0), (-2 * P + np.eye(2) + Cm).ravel()]

    def dists(yv):
        sc = [np.mean(yv[B]) - a0 for B in blocks]
        return {(i, j): abs(sc[i] - sc[j]) for i, j in itertools.combinations(range(len(blocks)), 2)}

    t = 0.0
    d_prev = dists(y)
    Ptraj = []
    while t < T_end and len(blocks) > 1:
        k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y_new = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        d_new = dists(y_new)
        crossed = [(e, d_prev[e], d_new[e]) for e in d_new if d_prev[e] > delta >= d_new[e]]
        if crossed:
            # earliest crossing by linear interpolation
            tc = [(t + dt * (dp - delta) / (dp - dn), e) for e, dp, dn in crossed]
            tc.sort()
            te, (i, j) = tc[0]
            merged = blocks[i] + blocks[j]
            blocks = [B for k, B in enumerate(blocks) if k not in (i, j)] + [sorted(merged)]
            events.append((te, tuple(sorted(merged))))
            d_new = dists(y_new)
        y, t, d_prev = y_new, t + dt, d_new
        Ptraj.append(y[N:].reshape(2, 2)[0, 0])
    return events, y


delta = 0.1
print("(c) closed-form times:", 0.5 * np.log(2), 0.5 * np.log(3), 0.5 * np.log(17 / 2))
print("    event-driven rule:", [(round(t, 12), m) for t, m in run_rule(alphas0 - a0, delta)])
for kap_ in [0.0, 1.0, 5.0]:
    ev, yend = simulate_full(kap_, alphas0, delta)
    print(f"    full ODE kappa={kap_}: events", [(round(t, 6), m) for t, m in ev], " P11(end)=", round(yend[N], 6))

# ---------- (d) random search inside the theorem class ----------
viol = 0
trials = 20000
for _ in range(trials):
    d1, d3 = rng.uniform(0.01, 1, 2)
    d2 = max(d1, d3) + rng.uniform(1e-6, 2)
    dl = rng.uniform(1e-4, min(d1, d3))
    a1 = 0.5 + rng.uniform(0.0, 2)
    a = np.array([a1, a1 + d1, a1 + d1 + d2, a1 + d1 + d2 + d3])
    perm = rng.permutation(4)
    ev = run_rule(a[perm] - 1.0, dl)
    inv = np.argsort(perm)  # sorted index k sits at position inv[k]
    low = tuple(sorted((inv[0], inv[1])))
    up = tuple(sorted((inv[2], inv[3])))
    times = {m: t for t, m in ev}
    ok = (low in times and up in times and (0, 1, 2, 3) in times and len(ev) == 3
          and abs(times[low] - 0.5 * np.log(d1 / dl)) < 1e-9 and abs(times[up] - 0.5 * np.log(d3 / dl)) < 1e-9
          and abs(times[(0, 1, 2, 3)] - 0.5 * np.log((d2 + (d1 + d3) / 2) / dl)) < 1e-9)
    viol += (not ok)
print(f"(d) random configurations in the stated class: {trials}, violations: {viol}")

# ---------- (e) exact region ----------
in_thm, in_exact, tree_ok, mismatch = 0, 0, 0, 0
for _ in range(trials):
    d1, d2, d3 = rng.uniform(0.01, 1, 3)
    dl = 0.5 * min(d1, d2, d3)
    a = np.cumsum([1.0, d1, d2, d3])
    ev = run_rule(a - 1.0, dl)
    got = [m for t, m in ev]
    tree = (len(ev) == 3 and set(got[:2]) == {(0, 1), (2, 3)})
    exact = (min(d1, d3) < d2) and (max(d1, d3) < d2 + min(d1, d3) / 2)
    thm = d2 > max(d1, d3)
    in_thm += thm; in_exact += exact; tree_ok += tree
    mismatch += (tree != exact)
print(f"(e) trials {trials}: tree ((12)(34)) occurs {tree_ok}; exact-region predicate {in_exact};"
      f" theorem class {in_thm}; mismatches tree vs exact predicate: {mismatch}")

# ---------- (f) general N vs centroid linkage ----------
bad = 0
for _ in range(500):
    n = rng.integers(3, 9)
    x = np.sort(rng.uniform(0.6, 4.0, n))
    dl = 0.5 * np.min(np.diff(x))
    ev = run_rule(x - 1.0, dl)
    Z = linkage(x.reshape(-1, 1), method="centroid")
    heights = np.sort(Z[:, 2])
    times = np.sort([t for t, m in ev])
    if len(times) != len(heights) or np.max(np.abs(times - 0.5 * np.log(heights / dl))) > 1e-9:
        bad += 1
    if np.any(np.diff(Z[:, 2]) < -1e-12):
        bad += 1  # an inversion would show as decreasing heights
print("(f) general N (3..8), 500 random 1-D configurations: mismatches with centroid linkage or inversions:", bad)
