"""D04-DYN shared model: the three-agent binary law-dependent-channel example of s1 (same seed)."""
import numpy as np

rng = np.random.default_rng(20261029)
NA = 3
Tq, Ts = 0.37, 0.61


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# references r_i(m) and evaluators G_i(m,k) (rows m, columns k), strictly positive
r = [rng.dirichlet([2, 2]) for _ in range(NA)]
G = [np.array([rng.dirichlet([2, 2]) for _m in range(2)]) for _ in range(NA)]

# source sets: receiver i -> list of sources j ; self with identity channel where j == i
Nq = {0: [0, 1, 2], 1: [0, 2], 2: [1, 2]}
Ns = {0: [1, 2], 1: [0, 1, 2], 2: [0, 1]}
piq = {i: rng.dirichlet(np.ones(len(Nq[i])) * 2) for i in range(NA)}
pis = {i: rng.dirichlet(np.ones(len(Ns[i])) * 2) for i in range(NA)}

# law-dependent channel parameters: K(0,1)=eps, K(1,0)=dlt, both depending on
# X = (P(state 1) of every table of the sector)
chan_par = {}
for sec, N in (("q", Nq), ("s", Ns)):
    for i in range(NA):
        for j in N[i]:
            if j == i:
                continue
            chan_par[(sec, i, j)] = (rng.normal(size=NA + 1), rng.normal(size=NA + 1))


def channel(sec, i, j, X):
    a, b = chan_par[(sec, i, j)]
    eps = 0.05 + 0.4 * sig(a[0] + a[1:] @ X)
    dlt = 0.05 + 0.4 * sig(b[0] + b[1:] @ X)
    K = np.empty((2, 2), dtype=X.dtype)
    K[0, 0] = 1 - eps
    K[0, 1] = eps
    K[1, 0] = dlt
    K[1, 1] = 1 - dlt
    return K


def dchannel(sec, i, j, X):
    """Derivative of K[v,u] with respect to X_k = P_k(1), shape (NA,2,2)."""
    a, b = chan_par[(sec, i, j)]
    se = sig(a[0] + a[1:] @ X)
    sd = sig(b[0] + b[1:] @ X)
    deps = 0.4 * se * (1 - se) * a[1:]
    ddlt = 0.4 * sd * (1 - sd) * b[1:]
    dK = np.zeros((NA, 2, 2), dtype=X.dtype)
    dK[:, 0, 0] = -deps
    dK[:, 0, 1] = deps
    dK[:, 1, 0] = ddlt
    dK[:, 1, 1] = -ddlt
    return dK


def kl(a, b):
    return np.sum(a * (np.log(a) - np.log(b)))


def costs(q, s):
    Xq = q[:, 1]
    Xs = s[:, 1]
    Dq, Ds, nuq, nus = {}, {}, {}, {}
    for i in range(NA):
        for j in Nq[i]:
            K = np.eye(2, dtype=q.dtype) if j == i else channel("q", i, j, Xq)
            nuq[(i, j)] = q[j] @ K
            Dq[(i, j)] = kl(q[i], nuq[(i, j)])
        for j in Ns[i]:
            K = np.eye(2, dtype=s.dtype) if j == i else channel("s", i, j, Xs)
            nus[(i, j)] = s[j] @ K
            Ds[(i, j)] = kl(s[i], nus[(i, j)])
    return Dq, Ds, nuq, nus


def E0(q, s):
    tot = 0.0
    for i in range(NA):
        zeta = np.outer(q[i], s[i])            # zeta[k,m]
        R = (G[i] * r[i][:, None]).T           # R[k,m] = r(m) G(m,k)
        tot = tot + np.sum(zeta * (np.log(zeta) - np.log(R)))
    return tot


def E_full(u):
    q, s, be, ga = u["q"], u["s"], u["beta"], u["gamma"]
    Dq, Ds, _, _ = costs(q, s)
    tot = E0(q, s)
    for i in range(NA):
        tot = tot + sum(be[i][n] * Dq[(i, j)] for n, j in enumerate(Nq[i])) + Tq * kl(be[i], piq[i])
        tot = tot + sum(ga[i][n] * Ds[(i, j)] for n, j in enumerate(Ns[i])) + Ts * kl(ga[i], pis[i])
    return tot


def gibbs(q, s):
    Dq, Ds, _, _ = costs(q, s)
    bs, gs, Zq, Zs = {}, {}, {}, {}
    for i in range(NA):
        w = piq[i] * np.exp(-np.array([Dq[(i, j)] for j in Nq[i]]) / Tq)
        Zq[i] = w.sum()
        bs[i] = w / Zq[i]
        w = pis[i] * np.exp(-np.array([Ds[(i, j)] for j in Ns[i]]) / Ts)
        Zs[i] = w.sum()
        gs[i] = w / Zs[i]
    return bs, gs, Zq, Zs


def E_red(q, s):
    _, _, Zq, Zs = gibbs(q, s)
    return E0(q, s) - Tq * sum(np.log(Zq[i]) for i in range(NA)) - Ts * sum(np.log(Zs[i]) for i in range(NA))


def analytic_force(u, include_channel_derivative=True):
    """h = dE/du for every table entry, from the manuscript's role formulas."""
    q, s, be, ga = u["q"], u["s"], u["beta"], u["gamma"]
    Xq = q[:, 1]
    Xs = s[:, 1]
    hq = np.zeros((NA, 2))
    hs = np.zeros((NA, 2))
    for i in range(NA):                        # E_0 roles
        logG = np.log(G[i])                    # [m,k]
        hq[i] += np.log(q[i]) - s[i] @ logG + (np.sum(s[i] * (np.log(s[i]) - np.log(r[i]))) + 1.0)
        hs[i] += np.log(s[i]) - np.log(r[i]) - logG @ q[i] + (np.sum(q[i] * np.log(q[i])) + 1.0)
    for sec, tabs, h, N, rows, X in (("q", q, hq, Nq, be, Xq), ("s", s, hs, Ns, ga, Xs)):
        for i in range(NA):
            for n, j in enumerate(N[i]):
                w = rows[i][n]
                K = np.eye(2) if j == i else channel(sec, i, j, X)
                nu = tabs[j] @ K
                ratio = tabs[i] / nu
                h[i] += w * (np.log(ratio) + 1.0)            # receiver role
                h[j] += -w * (K @ ratio)                     # source role, eq:dyn-incoming
                if include_channel_derivative and j != i:
                    dK = dchannel(sec, i, j, X)              # derivative w.r.t. P_k(1)
                    for k in range(NA):
                        dnu = tabs[j] @ dK[k]
                        h[k, 1] += -w * np.sum(ratio * dnu)  # channel role, eq:dyn-channel-differential
    hb, hg = {}, {}
    Dq, Ds, _, _ = costs(q, s)
    for i in range(NA):
        hb[i] = np.array([Dq[(i, j)] for j in Nq[i]]) + Tq * (np.log(be[i] / piq[i]) + 1.0)
        hg[i] = np.array([Ds[(i, j)] for j in Ns[i]]) + Ts * (np.log(ga[i] / pis[i]) + 1.0)
    return hq, hs, hb, hg


def random_point():
    q = np.array([rng.dirichlet([1.5, 1.5]) for _ in range(NA)])
    s = np.array([rng.dirichlet([1.5, 1.5]) for _ in range(NA)])
    be = {i: rng.dirichlet(np.ones(len(Nq[i]))) for i in range(NA)}
    ga = {i: rng.dirichlet(np.ones(len(Ns[i]))) for i in range(NA)}
    return {"q": q, "s": s, "beta": be, "gamma": ga}


def centered(z):
    return z - z.mean()


def random_tangent():
    vq = rng.normal(size=(NA, 2))
    vq -= vq.mean(axis=1, keepdims=True)
    vs = rng.normal(size=(NA, 2))
    vs -= vs.mean(axis=1, keepdims=True)
    vb = {i: centered(rng.normal(size=len(Nq[i]))) for i in range(NA)}
    vg = {i: centered(rng.normal(size=len(Ns[i]))) for i in range(NA)}
    return {"q": vq, "s": vs, "beta": vb, "gamma": vg}


def shift(u, v, t):
    return {"q": u["q"] + t * v["q"], "s": u["s"] + t * v["s"],
            "beta": {i: u["beta"][i] + t * v["beta"][i] for i in range(NA)},
            "gamma": {i: u["gamma"][i] + t * v["gamma"][i] for i in range(NA)}}


def pair(h, v):
    hq, hs, hb, hg = h
    return (np.sum(hq * v["q"]) + np.sum(hs * v["s"])
            + sum(np.sum(hb[i] * v["beta"][i]) for i in range(NA))
            + sum(np.sum(hg[i] * v["gamma"][i]) for i in range(NA)))


def to_complex(u):
    return {"q": u["q"].astype(complex), "s": u["s"].astype(complex),
            "beta": {i: u["beta"][i].astype(complex) for i in range(NA)},
            "gamma": {i: u["gamma"][i].astype(complex) for i in range(NA)}}
