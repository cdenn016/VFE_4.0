r"""D04-DYN s1: reciprocal potential, Gibbs rows, envelope, and the full first variation.

Three agents, binary state and model alphabets, law-dependent channels in both sectors
(each channel entry depends on the tables of every agent of the sector, including the receiver
and the source), identity self channels for some agents.  Checks (chapters/03_dynamics.tex):
  def:dyn-potential: E >= E_red >= E_0 >= 0 at random interior points
  prop:dyn-gibbs, eq:dyn-row-gap: E - E_red - T KL(beta||beta*) = 0; uniqueness vs random and boundary rows
  eq:dyn-envelope with law-dependent channels: d E_red = d E_0 + sum beta* dD (complex step)
  eq:dyn-kl-differential + eq:dyn-channel-differential + eq:dyn-incoming: the analytic force h
    (receiver + source + channel-derivative roles) against complex-step and central differences
    of E along random mass-preserving directions; error made by freezing channel values
  line 54 pooling rule (receiver stationarity with sources frozen)
  rem:dyn-self-sources: row bounds and the prior-learning Fisher flow eq:dyn-prior-learning
Single-threaded, a few seconds.
"""
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


print("== def:dyn-potential ordering E >= E_red >= E_0 >= 0, and eq:dyn-row-gap ==")
worst_gap = 0.0
worst_order = np.inf
for trial in range(200):
    u = random_point()
    q, s, be, ga = u["q"], u["s"], u["beta"], u["gamma"]
    bs, gs, Zq, Zs = gibbs(q, s)
    Ef, Er, E00 = E_full(u), E_red(q, s), E0(q, s)
    rowgap = Tq * sum(kl(be[i], bs[i]) for i in range(NA)) + Ts * sum(kl(ga[i], gs[i]) for i in range(NA))
    worst_gap = max(worst_gap, abs(Ef - Er - rowgap))
    worst_order = min(worst_order, Ef - Er, Er - E00, E00)
print(f"  max abs(E - E_red - row KL gap) over 200 points = {worst_gap:.2e}")
print(f"  min of (E-E_red, E_red-E_0, E_0) over 200 points = {worst_order:.3e} (must be nonnegative)")

print("== prop:dyn-gibbs uniqueness: Gibbs row vs random interior and boundary competitors ==")
u = random_point()
q, s = u["q"], u["s"]
Dq, Ds, _, _ = costs(q, s)
bs, gs, Zq, Zs = gibbs(q, s)
i0 = 0
Dv = np.array([Dq[(i0, j)] for j in Nq[i0]])


def rowobj(b):
    pos = b > 0
    ent = np.sum(b[pos] * np.log(b[pos] / piq[i0][pos]))
    return np.sum(b * Dv) + Tq * ent


fstar = rowobj(bs[i0])
margins = []
for _ in range(5000):
    b = rng.dirichlet(np.ones(len(Dv)) * 0.3)
    if rng.random() < 0.3:
        b[rng.integers(len(b))] = 0.0
        b /= b.sum()
    margins.append(rowobj(b) - fstar)
print(f"  min over 5000 competitors of f(beta)-f(beta*) = {min(margins):.3e}; f(beta*) + T log Z = {fstar + Tq*np.log(Zq[i0]):.2e}")

print("== eq:dyn-envelope with law-dependent channels (complex step, step 1e-30) ==")
worst = 0.0
hcs = 1e-30
for trial in range(20):
    u = random_point()
    v = random_tangent()
    q, s = u["q"], u["s"]
    qc = q.astype(complex) + 1j * hcs * v["q"]
    sc = s.astype(complex) + 1j * hcs * v["s"]
    dEred = np.imag(E_red(qc, sc)) / hcs
    dE0 = np.imag(E0(qc, sc)) / hcs
    bs, gs, _, _ = gibbs(q, s)
    Dqc, Dsc, _, _ = costs(qc, sc)
    rhs = dE0
    rhs += sum(bs[i][n] * np.imag(Dqc[(i, j)]) / hcs for i in range(NA) for n, j in enumerate(Nq[i]))
    rhs += sum(gs[i][n] * np.imag(Dsc[(i, j)]) / hcs for i in range(NA) for n, j in enumerate(Ns[i]))
    worst = max(worst, abs(dEred - rhs) / max(1.0, abs(dEred)))
print(f"  max relative error of dE_red - (dE_0 + sum beta* dD) over 20 directions = {worst:.2e}")

print("== first variation: all roles (receiver, source, channel derivative) ==")
worst_cs = 0.0
worst_fd = 0.0
frozen_err = []
for trial in range(50):
    u = random_point()
    v = random_tangent()
    h = analytic_force(u, True)
    lhs = pair(h, v)
    dcs = np.imag(E_full(shift(to_complex(u), v, 1j * hcs))) / hcs
    t = 1e-5
    dfd = (E_full(shift(u, v, t)) - E_full(shift(u, v, -t))) / (2 * t)
    worst_cs = max(worst_cs, abs(lhs - dcs) / max(1.0, abs(dcs)))
    worst_fd = max(worst_fd, abs(lhs - dfd) / max(1.0, abs(dfd)))
    hfro = analytic_force(u, False)
    frozen_err.append(abs(pair(hfro, v) - dcs) / max(1.0, abs(dcs)))
print(f"  max relative error, analytic vs complex step (50 directions) = {worst_cs:.2e}")
print(f"  max relative error, analytic vs central difference, t=1e-5  = {worst_fd:.2e}")
print(f"  channel derivative omitted: median relative error = {np.median(frozen_err):.3f}, max = {max(frozen_err):.3f}")

print("== self channel: receiver and source roles cancel exactly ==")
u = random_point()
q = u["q"]
w = u["beta"][0][0]
rec = w * (np.log(q[0] / q[0]) + 1.0)
src = -w * (np.eye(2) @ (q[0] / q[0]))
print(f"  self receiver+source contribution = {np.max(np.abs(rec + src)):.1e}")

print("== line 54 pooling rule: receiver stationarity with sources, rows, channels frozen ==")
L = 4
g = np.array([rng.dirichlet(np.ones(L)) for _m in range(3)])   # g[m, k]
sm = rng.dirichlet(np.ones(3))
nus = [rng.dirichlet(np.ones(L)) for _ in range(3)]
bet = rng.dirichlet(np.ones(3))
logq = (sm @ np.log(g) + sum(bet[n] * np.log(nus[n]) for n in range(3))) / (1 + bet.sum())
qp = np.exp(logq - logq.max())
qp /= qp.sum()


def recobj(qq):
    return sum(sm[m] * kl(qq, g[m]) for m in range(3)) + sum(bet[n] * kl(qq, nus[n]) for n in range(3))


grad = np.log(qp) - sm @ np.log(g) + 1 + sum(bet[n] * (np.log(qp / nus[n]) + 1) for n in range(3))
print(f"  centered receiver gradient at the pooled law: {np.max(np.abs(grad - grad.mean())):.2e}")
best_rand = min(recobj(rng.dirichlet(np.ones(L))) for _ in range(20000))
print(f"  objective at pooled law {recobj(qp):.6f}; best of 20000 random laws {best_rand:.6f}")

print("== rem:dyn-self-sources: bounds and eq:dyn-prior-learning ==")
viol = 0
for trial in range(2000):
    n = rng.integers(2, 6)
    T = rng.uniform(0.05, 3)
    pi = rng.dirichlet(np.ones(n))
    D = np.concatenate([[0.0], rng.exponential(2.0, size=n - 1)])
    Z = np.sum(pi * np.exp(-D / T))
    b = pi * np.exp(-D / T) / Z
    ok = (pi[0] <= Z + 1e-15) and (Z <= 1 + 1e-15) and (b[0] >= pi[0] - 1e-15)
    ok = ok and bool(np.all(b[1:] <= pi[1:] / pi[0] * np.exp(-D[1:] / T) + 1e-15))
    viol += (not ok)
print(f"  bound violations over 2000 random rows: {viol}")
n = 4
T = 0.3
pi = rng.dirichlet(np.ones(n))
beta = rng.dirichlet(np.ones(n))
hpi = -T * beta / pi
pidot = -pi * (hpi - np.sum(pi * hpi))
print(f"  max abs(pidot - T(beta - pi)) = {np.max(np.abs(pidot - T * (beta - pi))):.2e}")
D = np.concatenate([[0.0], rng.exponential(1.0, size=n - 1)])
Z = np.sum(pi * np.exp(-D / T))
bstar = pi * np.exp(-D / T) / Z
rep = (T / Z) * pi * (np.exp(-D / T) - Z)
print(f"  max abs(T(beta* - pi) - (T/Z) pi (exp(-D/T) - Z)) = {np.max(np.abs(T * (bstar - pi) - rep)):.2e}; self rate {rep[0]:.4f} (nonnegative)")

print("== central differences: step refinement and Richardson extrapolation ==")
worst_r = 0.0
ratios = []
for trial in range(50):
    u = random_point()
    v = random_tangent()
    lhs = pair(analytic_force(u, True), v)
    cd = lambda t: (E_full(shift(u, v, t)) - E_full(shift(u, v, -t))) / (2 * t)
    e1 = abs(cd(1e-4) - lhs)
    e2 = abs(cd(1e-5) - lhs)
    rich = (4 * cd(5e-5) - cd(1e-4)) / 3
    worst_r = max(worst_r, abs(rich - lhs) / max(1.0, abs(lhs)))
    if e2 > 1e-12:
        ratios.append(e1 / e2)
print(f"  error ratio (t=1e-4)/(t=1e-5): median {np.median(ratios):.1f} (second order gives 100)")
print(f"  Richardson-extrapolated central difference, max relative error = {worst_r:.2e}")
