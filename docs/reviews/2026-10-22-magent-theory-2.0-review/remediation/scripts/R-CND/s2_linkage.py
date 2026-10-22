r"""R-CND check 2: linkage type of the chapter-11 threshold rule.

Every block score is b_I(t) = e^{-2t} b_I(0) with b_I(0) the count-weighted
initial centroid, so the event rule is simulated exactly on initial-data
distances D: a join happens at t = (1/2) log(D/delta). Compare the merge
sequence with scipy centroid (UPGMC) linkage and with single linkage.
"""
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage

rng = np.random.default_rng(7)


def event_rule(b0: np.ndarray, delta: float):
    """Exact event-driven rule; returns list of (time, frozenset merged) or None if ill-posed."""
    blocks = [(frozenset([i]), 1, float(x)) for i, x in enumerate(b0)]
    # require every initial distance above delta
    for (_, _, x), (_, _, y) in itertools.combinations(blocks, 2):
        if abs(x - y) <= delta:
            return None
    t, out = 0.0, []
    while len(blocks) > 1:
        best = None
        for a, b in itertools.combinations(range(len(blocks)), 2):
            D = abs(blocks[a][2] - blocks[b][2])
            tj = 0.5 * np.log(D / delta)
            if best is None or tj < best[0]:
                best = (tj, a, b)
        tj, a, b = best
        assert tj >= t - 1e-12, "inversion"
        t = tj
        (Ia, na, xa), (Ib, nb, xb) = blocks[a], blocks[b]
        new = (Ia | Ib, na + nb, (na * xa + nb * xb) / (na + nb))
        blocks = [blk for k, blk in enumerate(blocks) if k not in (a, b)] + [new]
        # no-cascade check: all current distances still strictly above threshold
        for (_, _, x), (_, _, y) in itertools.combinations(blocks, 2):
            assert abs(x - y) * np.exp(-2 * t) > delta - 1e-12, "cascade"
        out.append((t, new[0]))
    return out


def scipy_sets(b0: np.ndarray, method: str):
    Z = linkage(b0.reshape(-1, 1), method=method)
    clusters = {i: frozenset([i]) for i in range(len(b0))}
    out = []
    for k, (i, j, h, _) in enumerate(Z):
        c = clusters[int(i)] | clusters[int(j)]
        clusters[len(b0) + k] = c
        out.append((h, c))
    return out


mism_c = mism_s = trials = 0
for _ in range(1000):
    n = int(rng.integers(3, 9))
    b0 = np.sort(rng.uniform(0, 10, n))
    delta = 0.5 * np.min(np.diff(b0))
    ev = event_rule(b0, delta)
    if ev is None:
        continue
    trials += 1
    cen = scipy_sets(b0, "centroid")
    sin = scipy_sets(b0, "single")
    ok_c = all(s1 == s2 and abs(t - 0.5 * np.log(h / delta)) < 1e-9
               for (t, s1), (h, s2) in zip(ev, cen))
    ok_s = [s for _, s in ev] == [s for _, s in sin]
    mism_c += not ok_c
    mism_s += not ok_s
print(f"{trials} configurations: centroid mismatches {mism_c}, single-linkage set-sequence mismatches {mism_s}")

# explicit four-point example where trees differ
b0 = np.array([0.0, 1.0, 2.05, 3.5])
ev = event_rule(b0, 0.1)
print("example", b0, "event rule:", [(round(t, 4), sorted(s)) for t, s in ev])
print("  centroid:", [(round(h, 4), sorted(s)) for h, s in scipy_sets(b0, "centroid")])
print("  single  :", [(round(h, 4), sorted(s)) for h, s in scipy_sets(b0, "single")])

# chapter example
b0 = np.array([1.0, 1.2, 1.8, 2.1]) - 1.0
ev = event_rule(b0, 0.1)
print("chapter example times", [round(t, 6) for t, _ in ev],
      "expected", [round(0.5 * np.log(x), 6) for x in (2, 3, 8.5)])

# tree independence of delta
b0 = np.sort(rng.uniform(0, 10, 7))
trees = {tuple(tuple(sorted(s)) for _, s in event_rule(b0, d)) for d in
         np.min(np.diff(b0)) * np.array([0.9, 0.5, 0.1, 0.01])}
print("distinct trees over four deltas:", len(trees))

# exact N=4 region for ((12)(34)) vs the sufficient class d2 > max(d1,d3)
cnt = inreg = suff = bad = 0
for _ in range(20000):
    d = rng.uniform(0.2, 3, 3)
    b0 = np.concatenate([[0], np.cumsum(d)])
    ev = event_rule(b0, 0.1)
    tree = [sorted(s) for _, s in ev]
    balanced = tree[:2] in ([[0, 1], [2, 3]], [[2, 3], [0, 1]])
    pred = (d[1] > min(d[0], d[2])) and (max(d[0], d[2]) < d[1] + min(d[0], d[2]) / 2)
    cnt += balanced
    inreg += pred
    suff += d[1] > max(d[0], d[2])
    bad += balanced != pred
print(f"N=4: balanced {cnt}, exact predicate {inreg}, mismatches {bad}, sufficient class {suff}")

# X2 example: an initial gap below delta is never joined
b0 = np.array([0.0, 0.05, 0.25])
print("gap below threshold -> rule requires initial distances above delta:", event_rule(b0, 0.1))
