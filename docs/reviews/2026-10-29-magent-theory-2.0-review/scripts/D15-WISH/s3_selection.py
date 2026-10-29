r"""D15-WISH s3: form:threshold-rule, form:kappa-independence (ii), form:selection.

(a) Exact event rule (every block score is e^{-2t} times its initial count-weighted
    centroid) against scipy.cluster.hierarchy.linkage(method="centroid") on random
    one-dimensional configurations, N = 3..9: merged sets, heights, join times
    t_k = (1/2) log(D_k/delta), strict increase of heights, no cascades.
(b) delta independence of the tree.
(c) Single-linkage contrast example 0, 1, 2.05, 3.5 (heights claimed 1, 1.45, 2.275).
(d) N = 4 exact region of form:gaps (sufficiency and necessity off ties).
(e) Direct numerical integration of the full coupled (alpha, P) system for
    kappa in {0, 1, 5}, with event detection on the integrated shapes (not on the
    closed form), from the chapter example.
"""
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.integrate import solve_ivp

rng = np.random.default_rng(11)


def event_rule(b0, delta):
    blocks = [(frozenset([i]), 1, float(x)) for i, x in enumerate(b0)]
    for (_, _, x), (_, _, y) in itertools.combinations(blocks, 2):
        if abs(x - y) <= delta:
            return None
    out, t, ok_inc, ok_cas, prevD = [], 0.0, True, True, 0.0
    while len(blocks) > 1:
        cands = sorted(((abs(blocks[a][2] - blocks[b][2]), a, b)
                        for a, b in itertools.combinations(range(len(blocks)), 2)))
        D, a, b = cands[0]
        if len(cands) > 1 and abs(cands[1][0] - D) < 1e-12:
            return "tie"
        tj = 0.5 * np.log(D / delta)
        ok_inc &= D > prevD
        prevD, t = D, tj
        (Ia, na, xa), (Ib, nb, xb) = blocks[a], blocks[b]
        new = (Ia | Ib, na + nb, (na * xa + nb * xb) / (na + nb))
        blocks = [blk for k, blk in enumerate(blocks) if k not in (a, b)] + [new]
        for (_, _, x), (_, _, y) in itertools.combinations(blocks, 2):
            ok_cas &= abs(x - y) * np.exp(-2 * t) > delta
        out.append((tj, new[0], D))
    return out, ok_inc, ok_cas


def scipy_sets(b0, method):
    Z = linkage(np.asarray(b0).reshape(-1, 1), method=method)
    cl = {i: frozenset([i]) for i in range(len(b0))}
    out = []
    for k, (i, j, h, _) in enumerate(Z):
        c = cl[int(i)] | cl[int(j)]
        cl[len(b0) + k] = c
        out.append((h, c))
    return out


trials = mism = bad_inc = bad_cas = ties = 0
max_dt = 0.0
for _ in range(3000):
    n = int(rng.integers(3, 10))
    b0 = rng.uniform(-3, 3, n)
    delta = 0.9 * np.min(np.abs(np.subtract.outer(b0, b0))[~np.eye(n, dtype=bool)])
    r = event_rule(b0, delta)
    if r is None:
        continue
    if r == "tie":
        ties += 1
        continue
    ev, ok_inc, ok_cas = r
    trials += 1
    bad_inc += not ok_inc
    bad_cas += not ok_cas
    cen = scipy_sets(b0, "centroid")
    same = all(s1 == s2 for (_, s1, _), (_, s2) in zip(ev, cen))
    mism += not same
    if same:
        max_dt = max(max_dt, max(abs(t - 0.5 * np.log(h / delta)) for (t, _, _), (h, _) in zip(ev, cen)))
print(f"(a) {trials} random configurations (N=3..9): centroid-linkage set mismatches {mism}, "
      f"max |t_k - (1/2)log(h_k/delta)| = {max_dt:.1e}, height inversions {bad_inc}, cascades {bad_cas}, ties skipped {ties}")

b0 = np.sort(rng.uniform(0, 5, 8))
trees = set()
for f in [0.95, 0.5, 0.1, 0.001]:
    ev, _, _ = event_rule(b0, f * np.min(np.diff(b0)))
    trees.add(tuple(tuple(sorted(s)) for _, s, _ in ev))
print(f"(b) distinct trees over four thresholds: {len(trees)}")

b0 = np.array([0.0, 1.0, 2.05, 3.5])
ev, _, _ = event_rule(b0, 0.1)
print("(c) rule:", [(round(D, 4), sorted(s)) for _, s, D in ev])
print("    single:", [(round(h, 4), sorted(s)) for h, s in scipy_sets(b0, "single")])

agree = cnt = suff = 0
for _ in range(40000):
    d = rng.uniform(0.11, 3, 3)
    b0 = np.concatenate([[0], np.cumsum(d)])
    r = event_rule(b0, 0.1)
    if r in (None, "tie"):
        continue
    ev, _, _ = r
    tree = [sorted(s) for _, s, _ in ev]
    balanced = sorted(map(tuple, tree[:2])) == [(0, 1), (2, 3)]
    pred = (0.1 < min(d[0], d[2]) < d[1]) and (max(d[0], d[2]) < d[1] + 0.5 * min(d[0], d[2]))
    agree += balanced == pred
    cnt += 1
    suff += balanced
print(f"(d) N=4: predicate of form:gaps agrees with the rule in {agree}/{cnt} gap triples ({suff} balanced)")

N = 4


def full_rhs(t, y, a0, kap):
    al = y[:N]
    P = np.array([[y[N], y[N + 1]], [y[N + 1], y[N + 2]]])
    M = al.sum() * np.linalg.inv(P)
    Lam = np.eye(2) + kap / N * M
    C = np.eye(2) + 2 * kap / N * M - kap * a0 * np.linalg.inv(Lam)
    dP = -2 * P + np.eye(2) + C
    return np.concatenate([-2 * (al - a0), [dP[0, 0], dP[0, 1], dP[1, 1]]])


al0 = np.array([1.0, 1.2, 1.8, 2.1])
P0 = np.array([[np.exp(0.3), 0.0], [0.0, np.exp(-0.3)]])
for kap in [0.0, 1.0, 5.0]:
    y0 = np.concatenate([al0, [P0[0, 0], P0[0, 1], P0[1, 1]]])
    pairs = [((0,), (1,)), ((2,), (3,)), ((0, 1), (2, 3))]
    events = []
    for I, J in pairs:
        def ev(t, y, a0_, kap_, I=I, J=J):
            return abs(np.mean(y[list(I)]) - np.mean(y[list(J)])) - 0.1
        ev.terminal = False
        ev.direction = -1
        sol = solve_ivp(full_rhs, (0, 3), y0, args=(1.0, kap), events=ev, rtol=1e-12, atol=1e-14, method="DOP853")
        events.append(sol.t_events[0][0])
    sol = solve_ivp(full_rhs, (0, 1.5), y0, args=(1.0, kap), rtol=1e-12, atol=1e-14, method="DOP853")
    print(f"(e) kappa={kap}: integrated event times {np.round(events, 7)}; P11(1.5) = {sol.y[N, -1]:.4f}")
print("    closed form:", np.round([0.5 * np.log(2), 0.5 * np.log(3), 0.5 * np.log(8.5)], 7),
      "coexistence (1/2)log(17/6) =", round(0.5 * np.log(17 / 6), 7))
