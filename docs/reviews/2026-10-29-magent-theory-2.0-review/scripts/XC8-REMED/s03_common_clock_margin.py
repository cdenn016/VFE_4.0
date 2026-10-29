r"""Robust version of s02: a non-uniform shared evaluator G_0 so that the null and full flows relax
toward different common laws, and an explicit margin for the KL-threshold selector.

Same architecture as s02 (hypotheses of inh:common-clock): shared model-independent evaluator G_0,
self excluded, constant-output channel C[M]=softmax(kappa*M) with M=sum_i q_i.
Reports crossing times of the within-pair Jeffreys divergences under the full and the null flow,
and the fraction of random perturbations (all 12 log coordinates) on which the first selected pair
differs between the two flows.
"""
import itertools

import numpy as np
from scipy.integrate import solve_ivp

N, K = 4, 3
PAIRS = list(itertools.combinations(range(N), 2))


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def laws(ell):
    q = np.exp(ell - ell.max(axis=1, keepdims=True))
    return q / q.sum(axis=1, keepdims=True)


def make_fields(G0, kappa):
    lg = np.log(G0)

    def h_full(q):
        M = q.sum(axis=0)
        C = softmax(kappa * M)
        return 2.0 * np.log(q) - lg[None, :] - np.log(C)[None, :] - kappa * (M - N * C)[None, :]

    def h_null(q):
        return np.log(q) - lg[None, :]
    return h_full, h_null


def rhs(field):
    def f(t, y):
        q = laws(y.reshape(N, K))
        h = field(q)
        return (-(h - (q * h).sum(axis=1, keepdims=True))).ravel()
    return f


def jeff(q, i, j):
    return float(np.sum((q[i] - q[j]) * (np.log(q[i]) - np.log(q[j]))))


def crossing_times(field, ell0, delta, tmax=10.0, n=1201):
    sol = solve_ivp(rhs(field), (0.0, tmax), ell0.ravel(), rtol=1e-9, atol=1e-11, dense_output=True)
    ts = np.linspace(0.0, tmax, n)
    times = {p: np.inf for p in PAIRS}
    prev = None
    for t in ts:
        q = laws(sol.sol(t).reshape(N, K))
        d = np.array([jeff(q, i, j) for i, j in PAIRS])
        if prev is None and np.any(d <= delta):
            return None
        if prev is not None:
            for k, p in enumerate(PAIRS):
                if times[p] == np.inf and prev[k] > delta >= d[k]:
                    times[p] = float(t)
        prev = d
    return times


def first_pair(times):
    if times is None:
        return None
    p = min(times, key=times.get)
    return p if np.isfinite(times[p]) else "none"


def config(a, c, s=0.9, D=1.6):
    ell = np.array([[a, s, D], [-a, s, D], [0.0, s + c, -D], [0.0, s - c, -D]])
    return ell - ell.mean(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(7)
    delta = 0.05
    best = None
    for G0 in [np.array([0.6, 0.3, 0.1]), np.array([0.7, 0.2, 0.1])]:
        for kappa in [2.0, 4.0]:
            hf, hn = make_fields(G0, kappa)
            for a in [0.6, 0.8, 1.0]:
                for c in [0.6, 0.8, 1.0]:
                    e = config(a, c)
                    tf, tn = crossing_times(hf, e, delta), crossing_times(hn, e, delta)
                    if tf is None or tn is None:
                        continue
                    m1 = tf[(0, 1)] - tf[(2, 3)]
                    m2 = tn[(2, 3)] - tn[(0, 1)]
                    for sgn in (+1, -1):
                        m = min(sgn * m1, sgn * m2)
                        if best is None or m > best[0]:
                            best = (m, G0, kappa, a, c, tf, tn, sgn)
    m, G0, kappa, a, c, tf, tn, sgn = best
    hf, hn = make_fields(G0, kappa)
    lines = []
    lines.append("best margin %.4f at G0=%s kappa=%.1f a=%.2f c=%.2f (sign %+d)" % (m, G0, kappa, a, c, sgn))
    lines.append("full crossing times: %s" % {k: round(v, 3) for k, v in tf.items()})
    lines.append("null crossing times: %s" % {k: round(v, 3) for k, v in tn.items()})
    lines.append("first pair: full %s, null %s" % (first_pair(tf), first_pair(tn)))
    e0 = config(a, c)
    for eps in (0.02, 0.05):
        diff = tot = 0
        for _ in range(25):
            e = e0 + eps * rng.standard_normal(e0.shape)
            e = e - e.mean(axis=1, keepdims=True)
            pf, pn = first_pair(crossing_times(hf, e, delta)), first_pair(crossing_times(hn, e, delta))
            if pf is None or pn is None:
                continue
            tot += 1
            diff += int(pf != pn)
        lines.append("perturbation size %.2f: first pair differs in %d of %d admissible trials" % (eps, diff, tot))
    text = "\n".join(lines)
    print(text)
    with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
