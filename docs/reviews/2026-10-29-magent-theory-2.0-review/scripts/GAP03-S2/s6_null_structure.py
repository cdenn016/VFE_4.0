r"""GAP03-S2 s6: does the declared null (convention (b)) change WHICH blocks form on the
form:gaps class, or only their order and times?  Also compares the two null-parent conventions
(N1 fine-level, N2 parent-level) at kappa = 0, P(0) = I, where N2 is exact.

Random gap triples from the form:gaps class (delta = 0.1, u1 in [0.6, 2.0]).
Classification of the null history against the full history:
  same     : identical partition sequence (times may differ)
  order    : same set of blocks, different order
  struct   : a different block is formed (e.g. {1,2,3})
Single-threaded, short.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln, digamma

N, A0, DELTA = 4, 1.0, 0.1
rng = np.random.default_rng(29102026)


def lG2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl_iso(a, x, b, l):
    """KL(W(a, xI) || W(b, lI)) on 2x2"""
    return (a - b) * psi2(a) - lG2(a) + lG2(b) + 2 * b * np.log(x / l) + a * (2 * l / x - 2)


def history(S_fun, tmax, grid=1500):
    """threshold rule on a vectorized shape trajectory S_fun(ts) -> (N, len(ts))"""
    ts = np.linspace(0, tmax, grid)
    S = S_fun(ts)
    blocks = [(i,) for i in range(N)]
    tcur, hist = 0.0, []
    while len(blocks) > 1:
        best = None
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                I, J = list(blocks[a]), list(blocks[b])
                d = np.abs(S[I].mean(0) - S[J].mean(0)) - DELTA
                ok = np.where((ts > tcur) & (d <= 0))[0]
                if len(ok) == 0:
                    continue
                k = ok[0]
                lo = max(ts[k - 1], tcur + 1e-12)
                g = lambda tt: abs(S_fun(np.array([tt]))[I, 0].mean() - S_fun(np.array([tt]))[J, 0].mean()) - DELTA
                if g(lo) <= 0:
                    continue
                te = brentq(g, lo, ts[k], xtol=1e-12)
                if best is None or te < best[0]:
                    best = (te, a, b)
        if best is None:
            break
        te, a, b = best
        hist.append((te, tuple(sorted(blocks[a] + blocks[b]))))
        blocks = [blocks[k] for k in range(len(blocks)) if k not in (a, b)] + [tuple(sorted(blocks[a] + blocks[b]))]
        tcur = te
    return hist


def null_b_shapes(u, T, kap, p0):
    x0 = np.full(N, p0)
    mo = np.sum(u / x0) - u / x0

    def rhs(t, s):
        al, x = s[:N], s[N:]
        l = 1 + (kap / N) * (al / x + mo)
        C = l + (kap / N) * (al / x - A0 / l)
        D = kl_iso(al, x, A0, l)
        w = 0.25 * np.exp(-D / T)
        beta = w / (w + 0.75)
        return np.concatenate([-(1 + beta) * (al - A0), -(1 + beta) * x + 1 + beta * C])
    sol = solve_ivp(rhs, (0, 9.0), np.concatenate([u, x0]), method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    return lambda ts: sol.sol(ts)[:N]


def null_b_N2_k0_I(u, T):
    """parent-level null at kappa=0, P(0)=I (exact Wishart parents W(A_U, I))"""
    units = [((i,), float(u[i])) for i in range(N)]
    tcur, hist = 0.0, []
    while len(units) > 1:
        nn = np.array([len(x[0]) for x in units], float)
        A = np.array([x[1] for x in units])

        def rhs(t, s):
            D = kl_iso(s, 1.0, nn * A0, 1.0)
            pi = nn / N
            w = pi * np.exp(-D / T)
            beta = np.where(pi < 1, w / (w + 1 - pi), 1.0)
            return -(1 + beta) * (s - nn * A0)
        sol = solve_ivp(rhs, (tcur, tcur + 9.0), A, method='DOP853', rtol=1e-11, atol=1e-13, dense_output=True)
        ts = np.linspace(tcur, tcur + 9.0, 1500)
        Sv = sol.sol(ts) / nn[:, None]
        best = None
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                d = np.abs(Sv[a] - Sv[b]) - DELTA
                ok = np.where(d <= 0)[0]
                if len(ok) == 0 or ok[0] == 0:
                    continue
                g = lambda tt: abs(sol.sol(tt)[a] / nn[a] - sol.sol(tt)[b] / nn[b]) - DELTA
                te = brentq(g, ts[ok[0] - 1], ts[ok[0]], xtol=1e-12)
                if best is None or te < best[0]:
                    best = (te, a, b)
        if best is None:
            break
        te, a, b = best
        st = sol.sol(te)
        blk = tuple(sorted(units[a][0] + units[b][0]))
        hist.append((te, blk))
        units = [(units[k][0], st[k]) for k in range(len(units)) if k not in (a, b)] + [(blk, st[a] + st[b])]
        tcur = te
    return hist


def classify(h, h0):
    p, p0 = [b for _, b in h], [b for _, b in h0]
    if p == p0:
        return 'same'
    if sorted(p) == sorted(p0):
        return 'order'
    return 'struct'


def sample_gaps():
    while True:
        d1, d3 = rng.uniform(0.12, 0.6, 2)
        lo = min(d1, d3)
        d2 = rng.uniform(lo, lo + 0.8)
        if DELTA < lo < d2 and max(d1, d3) < d2 + 0.5 * lo:
            u1 = rng.uniform(0.6, 2.0)
            return np.array([u1, u1 + d1, u1 + d1 + d2, u1 + d1 + d2 + d3])


samples = [sample_gaps() for _ in range(36)]
out = []
configs = [(1.0, 0.0, 1.0), (0.3, 0.0, 1.0), (3.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 2.0)]
for (T, kap, p0) in configs:
    counts = {'same': 0, 'order': 0, 'struct': 0}
    ex = None
    for u in samples:
        hf = history(lambda ts: A0 + np.exp(-2 * ts)[None, :] * (u - A0)[:, None], 6.0)
        hb = history(null_b_shapes(u, T, kap, p0), 9.0)
        c = classify(hf, hb)
        counts[c] += 1
        if c == 'struct' and ex is None:
            ex = (u, hf, hb)
    out.append("null (b), N1, T=%g kappa=%g P(0)=%gI: %s of %d samples" % (T, kap, p0, counts, len(samples)))
    if ex is not None:
        u, hf, hb = ex
        out.append("    structural example u=%s: full %s ; null %s" % (np.round(u, 4), [(round(t, 4), b) for t, b in hf], [(round(t, 4), b) for t, b in hb]))
# N1 vs N2 at kappa = 0, P(0) = I, T = 1
cnt = {'same': 0, 'order': 0, 'struct': 0}
tdiff = []
for u in samples:
    h1 = history(null_b_shapes(u, 1.0, 0.0, 1.0), 9.0)
    h2 = null_b_N2_k0_I(u, 1.0)
    cnt[classify(h1, h2)] += 1
    if [b for _, b in h1] == [b for _, b in h2]:
        tdiff.append(max(abs(a[0] - b[0]) for a, b in zip(h1, h2)))
out.append("N1 vs N2 (null (b), kappa=0, P(0)=I, T=1): %s; max event-time difference when sequences agree: %.4f (min %.4f)"
           % (cnt, max(tdiff), min(tdiff)))
for line in out:
    print(line)
