r"""GAP07-S3: does K = 1 force the minimizer of the structure free energy it7:structure-gauss to be an interval partition?

it7:derived-events justifies the interval-coarsening minimizer by "Since K=1". Brute force over all set partitions of
N = 4, 5, 6 agents on a line (lam = 6, T = 1, pi_s = 0.05, pi_p = 0.95/(N-1), evaluator mean g = 0), random positions:
count configurations whose global minimizer is not an interval partition of the agents sorted by position, and print one.
Also evaluates the four-agent example of the report (positions 0, 0.1, 0.2, 0.3, evaluator far away).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

LAM, PS = 6.0, 0.05


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


def F(y, part, g=0.0):
    N = len(y); pp = (1 - PS) / (N - 1); means = [y[list(b)].mean() for b in part]; tot = 0.0
    for a, b in enumerate(part):
        n = len(b); W = np.sum((y[list(b)] - means[a]) ** 2)
        A = (n - 1) * pp + PS * np.exp(-n * LAM / 2 * (means[a] - g) ** 2)
        for c, b2 in enumerate(part):
            if c != a: A += len(b2) * pp * np.exp(-n * LAM / 2 * (means[a] - means[c]) ** 2)
        tot += LAM / 2 * W - np.log(A)
    return tot


def is_interval(y, part):
    order = np.argsort(y); rank = np.empty(len(y), int); rank[order] = np.arange(len(y))
    return all(max(rank[list(b)]) - min(rank[list(b)]) == len(b) - 1 for b in part)


y4 = np.array([0.0, 0.1, 0.2, 0.3])
for part in ([(0, 1, 2, 3)], [(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]):
    print("four equally spaced agents, evaluator at 10:", part, round(F(y4, part, g=10.0), 4))
rng = np.random.default_rng(7)
T0 = time.time()
for N in (4, 5, 6):
    parts = list(set_partitions(range(N))); cnt = tot = 0; ex = None
    while time.time() - T0 < 8 * (N - 3) and tot < 3000:
        tot += 1
        y = np.sort(rng.uniform(-1, 1, N) * rng.choice([0.3, 1.0, 2.0])) + rng.choice([0.0, 1.5])
        vals = [F(y, p) for p in parts]; k = int(np.argmin(vals)); s = sorted(vals)
        if not is_interval(y, parts[k]):
            cnt += 1
            if ex is None or s[1] - s[0] > ex[3]: ex = (y.round(4).tolist(), parts[k], round(s[0], 4), round(s[1] - s[0], 4))
    print(f"N = {N}: {cnt} of {tot} random configurations have a non-interval global minimizer; example (positions,"
          f" minimizer, F, margin): {ex}")
print("elapsed", round(time.time() - T0, 1), "s")
