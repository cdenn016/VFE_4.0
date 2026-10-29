r"""D11-FUS: the tau-frame sticky solution of fus:sticky-flow equals the l1 convex-clustering path.

Claim checked numerically: for weighted atoms z^0 with multiplicities n,
    z(tau) = argmin_y  (1/2) sum_i n_i (y_i - z_i^0)^2 + (lam*tau) sum_{i<j} n_i n_j |y_i - y_j|,
i.e. the sticky-particle solution at auxiliary time tau is the convex-clustering (Clusterpath,
Hocking et al. 2011) minimizer at fusion parameter s = lam*tau.  The minimizer is computed exactly
by the order-preserving isotonic reduction (PAVA); the sticky solution by exact event stepping.
Single-threaded, about one second.
"""
import numpy as np


def pava(y, w):
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi*wi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2]/wts[-2] >= vals[-1]/wts[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] += v2; wts[-1] += w2; cnt[-1] += c2
    out = []
    for v_, w_, c_ in zip(vals, wts, cnt):
        out += [v_/w_]*c_
    return np.array(out)


def clusterpath(z0, n, s):
    order = np.argsort(z0, kind='mergesort')
    zs, ns = z0[order], n[order]
    L = np.concatenate([[0.0], np.cumsum(ns)[:-1]]); R = ns.sum() - L - ns
    y = pava(zs - s*(L - R), ns)
    out = np.empty_like(z0); out[order] = y
    return out


def sticky_tau(z0, n, lam, tau_end):
    order = np.argsort(z0)
    blocks = [[float(z0[i]), float(n[i]), [int(i)]] for i in order]
    tau = 0.0
    while len(blocks) > 1:
        ntot = sum(b[1] for b in blocks)
        left = np.cumsum([0.0] + [b[1] for b in blocks[:-1]])
        vel = [lam*((ntot - left[k] - blocks[k][1]) - left[k]) for k in range(len(blocks))]
        dts = [(blocks[k + 1][0] - blocks[k][0])/(vel[k] - vel[k + 1]) for k in range(len(blocks) - 1)]
        dmin = min(dts)
        if tau + dmin > tau_end:
            for k, b in enumerate(blocks):
                b[0] += vel[k]*(tau_end - tau)
            tau = tau_end
            break
        for k, b in enumerate(blocks):
            b[0] += vel[k]*dmin
        tau += dmin
        merged = []; k = 0
        while k < len(blocks):
            cur = blocks[k]
            while k + 1 < len(blocks) and dts[k] <= dmin*(1 + 1e-12) + 1e-14:
                nxt = blocks[k + 1]
                cur = [(cur[0]*cur[1] + nxt[0]*nxt[1])/(cur[1] + nxt[1]), cur[1] + nxt[1], cur[2] + nxt[2]]
                k += 1
            merged.append(cur); k += 1
        blocks = merged
    out = np.empty(len(z0))
    for b in blocks:
        for i in b[2]:
            out[i] = b[0]
    return out


rng = np.random.default_rng(3)
worst = 0.0
for _ in range(2000):
    r = int(rng.integers(2, 10))
    z0 = rng.normal(0, 3, size=r)
    n = rng.integers(1, 5, size=r).astype(float)
    lam = float(rng.uniform(0.1, 3.0))
    tau = float(rng.uniform(0.0, 3.0))
    worst = max(worst, np.abs(sticky_tau(z0, n, lam, tau) - clusterpath(z0, n, lam*tau)).max())
print(f"max |z(tau) - clusterpath(lam*tau)| over 2000 random weighted configurations: {worst:.2e}")
