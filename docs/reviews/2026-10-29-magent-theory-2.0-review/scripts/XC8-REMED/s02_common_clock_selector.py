r"""Common-clock architecture of inh:common-clock (ch5:1329) with a KL-threshold selector.

Architecture (all hypotheses of inh:common-clock): finite alphabet K={A,B,C}, N=4 agents,
shared model-independent evaluator G_0, self excluded from belief sources, every peer channel
equal to the constant-output law C[M] = softmax(kappa*M), M = sum_i q_i (additive moment of the
identity statistic). All peer costs in a row are equal, so Gibbs rows equal the priors and
    E_red = sum_l [ KL(q_l || G_0) + KL(q_l || C[M]) ].
Full Fisher flow:  d log q_i/dt = -(h_i - E_{q_i} h_i),
    h_i(k) = 2 log q_i(k) - log G_0(k) - log C_k - kappa (M_k - N C_k).
Null flow (inh:null-flow, peer costs deleted): h_i^0(k) = log q_i(k) - log G_0(k).

Checks:
 (1) the identity d/dt log(q_i/q_j) = -2 log(q_i/q_j) + c_ij(t) along the full flow (residual of its
     k-dependence), and -1 along the null flow;
 (2) a relabeling-equivariant selector reading symmetrized KL (Jeffreys) between agents, joining a
     pair when the divergence first reaches delta from above, selects a different first pair under
     the full and the null flow on an open set of initial data.
This tests the sentence "Interaction-sensitive selection under the reciprocal potential therefore
requires receiver-dependent targets" (ch5:1372-1373, ch11:208, outlook:16, companion c05:251).
"""
import itertools

import numpy as np
from scipy.integrate import solve_ivp

np.set_printoptions(precision=6, suppress=True)
N, K = 4, 3
KAPPA = 3.0
G0 = np.ones(K) / K


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def laws(ell):
    q = np.exp(ell - ell.max(axis=1, keepdims=True))
    return q / q.sum(axis=1, keepdims=True)


def h_full(q):
    M = q.sum(axis=0)
    C = softmax(KAPPA * M)
    common = -np.log(G0) - np.log(C) - KAPPA * (M - N * C)
    return 2.0 * np.log(q) + common[None, :]


def h_null(q):
    return np.log(q) - np.log(G0)[None, :]


def rhs(field):
    def f(t, y):
        ell = y.reshape(N, K)
        q = laws(ell)
        h = field(q)
        return (-(h - (q * h).sum(axis=1, keepdims=True))).ravel()
    return f


def jeffreys(q, i, j):
    return float(np.sum((q[i] - q[j]) * (np.log(q[i]) - np.log(q[j]))))


def first_join(field, ell0, delta, tmax=12.0):
    sol = solve_ivp(rhs(field), (0.0, tmax), ell0.ravel(), method="RK45", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0.0, tmax, 6001)
    pairs = list(itertools.combinations(range(N), 2))
    prev = None
    for t in ts:
        q = laws(sol.sol(t).reshape(N, K))
        d = np.array([jeffreys(q, i, j) for i, j in pairs])
        if prev is None:
            if np.any(d <= delta):
                return None, 0.0, d
        else:
            hit = np.where((prev > delta) & (d <= delta))[0]
            if hit.size:
                k = hit[np.argmin(d[hit])]
                return pairs[k], float(t), d
        prev = d
    return "none", tmax, prev


def crossing_times(field, ell0, delta, tmax=12.0):
    """First time each pair's Jeffreys divergence reaches delta from above (inf if never)."""
    sol = solve_ivp(rhs(field), (0.0, tmax), ell0.ravel(), method="RK45", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    ts = np.linspace(0.0, tmax, 6001)
    pairs = list(itertools.combinations(range(N), 2))
    times = {p: np.inf for p in pairs}
    prev = None
    for t in ts:
        q = laws(sol.sol(t).reshape(N, K))
        d = np.array([jeffreys(q, i, j) for i, j in pairs])
        if prev is not None:
            for k, p in enumerate(pairs):
                if times[p] == np.inf and prev[k] > delta >= d[k]:
                    times[p] = float(t)
        prev = d
    return times


def clock_residual(field, ell0, rate):
    q = laws(ell0)
    h = field(q)
    dlog = -(h - (q * h).sum(axis=1, keepdims=True))
    worst = 0.0
    for i, j in itertools.combinations(range(N), 2):
        r = dlog[i] - dlog[j] + rate * (np.log(q[i]) - np.log(q[j]))
        worst = max(worst, float(r.max() - r.min()))
    return worst


def base_config():
    # letters A, B, C. Pair {0,1} differs on A, pair {2,3} differs on B; the pairs are separated on C;
    # a common offset s on B makes B the heaviest letter, so C[M] concentrates on B.
    a, c, D, s = 0.80, 0.72, 1.6, 0.9
    ell = np.array([[a, s, D], [-a, s, D], [0.0, s + c, -D], [0.0, s - c, -D]])
    return ell - ell.mean(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(20261029)
    ell0 = base_config()
    out = []
    out.append("(1) common-clock identity, spread over k of d/dt log(q_i/q_j) + r log(q_i/q_j):")
    out.append("    full flow, r=2: %.3e" % clock_residual(h_full, ell0, 2.0))
    out.append("    null flow, r=1: %.3e" % clock_residual(h_null, ell0, 1.0))
    q0 = laws(ell0)
    pairs = list(itertools.combinations(range(N), 2))
    out.append("initial laws:\n" + str(q0))
    out.append("initial Jeffreys divergences by pair %s:\n%s" % (pairs, np.array([jeffreys(q0, i, j) for i, j in pairs])))
    delta = 0.05
    pf, tf, df = first_join(h_full, ell0, delta)
    pn, tn, dn = first_join(h_null, ell0, delta)
    out.append("(2) delta=%.3f  full flow: first pair %s at t=%.4f ; null flow: first pair %s at t=%.4f" % (delta, pf, tf, pn, tn))
    # late-time base laws
    solf = solve_ivp(rhs(h_full), (0, 12), ell0.ravel(), rtol=1e-10, atol=1e-12)
    soln = solve_ivp(rhs(h_null), (0, 12), ell0.ravel(), rtol=1e-10, atol=1e-12)
    out.append("    late common law, full: %s ; null: %s" % (laws(solf.y[:, -1].reshape(N, K))[0], laws(soln.y[:, -1].reshape(N, K))[0]))
    # open-set check: random perturbations of all 12 log-coordinates
    differ = 0
    trials = 40
    eps = 0.03
    for _ in range(trials):
        e = ell0 + eps * rng.standard_normal(ell0.shape)
        e = e - e.mean(axis=1, keepdims=True)
        a1, _, _ = first_join(h_full, e, delta)
        b1, _, _ = first_join(h_null, e, delta)
        if a1 is not None and b1 is not None and a1 != b1:
            differ += 1
    out.append("    perturbations of size %.2f in all 12 log coordinates: first pair differs in %d of %d trials" % (eps, differ, trials))
    # margins at a tuned configuration: crossing times of the two within-pair divergences
    best = None
    for a in [0.70, 0.75, 0.80, 0.85, 0.90]:
        for c in [0.55, 0.60, 0.65, 0.70]:
            ell = np.array([[a, 0.9, 1.6], [-a, 0.9, 1.6], [0.0, 0.9 + c, -1.6], [0.0, 0.9 - c, -1.6]])
            ell = ell - ell.mean(axis=1, keepdims=True)
            tf_ = crossing_times(h_full, ell, delta)
            tn_ = crossing_times(h_null, ell, delta)
            mf = tf_[(0, 1)] - tf_[(2, 3)]   # >0: full joins {2,3} first
            mn = tn_[(2, 3)] - tn_[(0, 1)]   # >0: null joins {0,1} first
            m = min(mf, mn)
            if best is None or m > best[0]:
                best = (m, a, c, tf_, tn_)
    m, a, c, tf_, tn_ = best
    out.append("    tuned a=%.2f c=%.2f: full crossing times %s" % (a, c, {k: round(v, 4) for k, v in tf_.items()}))
    out.append("                          null crossing times %s" % ({k: round(v, 4) for k, v in tn_.items()}))
    out.append("    margin min(full t_{01}-t_{23}, null t_{23}-t_{01}) = %.4f" % m)
    ell = np.array([[a, 0.9, 1.6], [-a, 0.9, 1.6], [0.0, 0.9 + c, -1.6], [0.0, 0.9 - c, -1.6]])
    ell = ell - ell.mean(axis=1, keepdims=True)
    differ = 0
    for _ in range(trials):
        e = ell + eps * rng.standard_normal(ell.shape)
        e = e - e.mean(axis=1, keepdims=True)
        a1, _, _ = first_join(h_full, e, delta)
        b1, _, _ = first_join(h_null, e, delta)
        if a1 is not None and b1 is not None and a1 != b1:
            differ += 1
    out.append("    tuned configuration, perturbations of size %.2f: first pair differs in %d of %d trials" % (eps, differ, trials))
    text = "\n".join(out)
    print(text)
    with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
