r"""GAP04-S4 s2: chapter 10 identity-channel fine flow (lem:boundary-fine-wellposed).

(a) Continuity of the time-1 flow map for convergence in measure (L^1) on L^infty-bounded
    sets.  Two initial states differ by eta on a set S of reference mass eps in one agent's
    log density.  The flow is L^1-Lipschitz on bounded invariant sets (proved in the report);
    here we check that ||x(1)-y(1)||_1 / ||x(0)-y(0)||_1 stays bounded as eps -> 0 and as the
    grid is refined, while the sup-norm distance does not shrink with eps.  This is the
    mechanism that makes the flow Borel for the evaluation sigma-algebra although L^infty is
    nonseparable.
(b) prop:boundary-joint-spectrum beyond the symmetry group: a reference with unequal atoms in
    each base fiber, and two configurations with the same joint law of
    (r, q_1(xi), ..., q_N(xi)) that no reference-preserving bijection relates.  The proposition
    predicts equal joint spectra and rows at every time.
Fine force (eq:boundary-fine-force):
    h_i = log q_i + V_i(r) + sum_j beta_ij (log q_i - log q_j) - sum_l beta_li q_l / q_i,
    V_i^q(r) = - int s_{i,0}(w) log rho_i(r,w) dB(w),
chart  dl_i/dt = -(h_i - E_{q_i} h_i), rows by the Fisher row equation.
Single-threaded numpy, RK4.
"""
import numpy as np

N = 3
Tq, Ts = 0.7, 0.9
Jc = np.array([0.8, -0.5, 1.1])           # couplings of the three local references
pri = 0.5                                  # uniform priors on the two nonself sources
EDGES = [[j for j in range(N) if j != i] for i in range(N)]


def make_space(fiber_w):
    """Base {0,1} with A uniform times a fiber with weights fiber_w (sum 1)."""
    lam = np.concatenate([0.5 * fiber_w, 0.5 * fiber_w])
    base = np.concatenate([np.zeros(len(fiber_w), int), np.ones(len(fiber_w), int)])
    return lam, base


def rho(i):
    s = np.array([1.0, -1.0])
    return np.exp(Jc[i] * np.outer(s, s)) / np.cosh(Jc[i])      # rho_i(r,w), int dA dB = 1


def field(state, X, Y):
    lamX, baseX = X
    lamY, baseY = Y
    l, m, b, g = state
    q = np.exp(l) / (lamX * np.exp(l)).sum(axis=1, keepdims=True)
    s = np.exp(m) / (lamY * np.exp(m)).sum(axis=1, keepdims=True)
    lq, ls = np.log(q), np.log(s)
    # base-marginal densities (A, B uniform on two points, weight 1/2 each)
    q0 = np.stack([np.array([(lamX * q[i])[baseX == r].sum() / 0.5 for r in (0, 1)]) for i in range(N)])
    s0 = np.stack([np.array([(lamY * s[i])[baseY == w].sum() / 0.5 for w in (0, 1)]) for i in range(N)])
    Vq = np.stack([-(0.5 * s0[i][None, :] * np.log(rho(i))).sum(axis=1) for i in range(N)])   # V_i^q(r)
    Vs = np.stack([-(0.5 * q0[i][:, None] * np.log(rho(i))).sum(axis=0) for i in range(N)])   # V_i^s(w)
    Dq = np.array([[(lamX * q[i] * (lq[i] - lq[j])).sum() if j != i else 0.0 for j in range(N)] for i in range(N)])
    Ds = np.array([[(lamY * s[i] * (ls[i] - ls[j])).sum() if j != i else 0.0 for j in range(N)] for i in range(N)])
    hq = np.zeros_like(l)
    hs = np.zeros_like(m)
    for i in range(N):
        hq[i] = lq[i] + Vq[i][baseX]
        hs[i] = ls[i] + Vs[i][baseY]
        for k, j in enumerate(EDGES[i]):
            hq[i] += b[i, k] * (lq[i] - lq[j])
            hs[i] += g[i, k] * (ls[i] - ls[j])
        for ll in range(N):
            if i in EDGES[ll]:
                k = EDGES[ll].index(i)
                hq[i] -= b[ll, k] * q[ll] / q[i]
                hs[i] -= g[ll, k] * s[ll] / s[i]
    dl = -(hq - (lamX * q * hq).sum(axis=1, keepdims=True))
    dm = -(hs - (lamY * s * hs).sum(axis=1, keepdims=True))
    db = np.zeros_like(b)
    dg = np.zeros_like(g)
    for i in range(N):
        Hb = np.array([Dq[i, j] for j in EDGES[i]]) + Tq * (np.log(b[i] / pri) + 1)
        Hg = np.array([Ds[i, j] for j in EDGES[i]]) + Ts * (np.log(g[i] / pri) + 1)
        db[i] = -b[i] * (Hb - (b[i] * Hb).sum())
        dg[i] = -g[i] * (Hg - (g[i] * Hg).sum())
    return [dl, dm, db, dg]


def rk4(state, X, Y, T=1.0, dt=0.01):
    st = [x.copy() for x in state]
    for _ in range(int(round(T / dt))):
        k1 = field(st, X, Y)
        k2 = field([x + 0.5 * dt * k for x, k in zip(st, k1)], X, Y)
        k3 = field([x + 0.5 * dt * k for x, k in zip(st, k2)], X, Y)
        k4 = field([x + dt * k for x, k in zip(st, k3)], X, Y)
        st = [x + dt / 6 * (a + 2 * bb + 2 * c + d) for x, a, bb, c, d in zip(st, k1, k2, k3, k4)]
    return st


def normalize(l, lam):
    return l - np.log((lam * np.exp(l)).sum(axis=1, keepdims=True))


rng = np.random.default_rng(7)
print("(a) L^1 versus L^infty continuity of the time-1 flow map")
for M in (64, 256, 1024):
    X = make_space(np.full(M, 1.0 / M))
    Y = make_space(np.full(M, 1.0 / M))
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    l0 = np.stack([0.6 * np.cos(th + i) for i in range(N)])
    l0 = np.concatenate([l0, 0.4 * l0[:, ::-1]], axis=1) + 0.2 * rng.uniform(-1, 1, size=(N, 2 * M))
    m0 = np.concatenate([np.stack([0.5 * np.sin(2 * th + i) for i in range(N)])] * 2, axis=1)
    l0, m0 = normalize(l0, X[0]), normalize(m0, Y[0])
    b0 = np.full((N, 2), 0.5); b0[:, 0] = [0.3, 0.6, 0.45]; b0[:, 1] = 1 - b0[:, 0]
    g0 = np.full((N, 2), 0.5)
    xs = rk4([l0, m0, b0, g0], X, Y)
    for k in (2, 4, 6, 8):
        eps = 2.0 ** (-k)
        if int(eps * 2 * M) < 1:
            continue                                  # perturbation set would be empty on this grid
        S = rng.permutation(2 * M)[: int(eps * 2 * M)]
        l1 = l0.copy(); l1[0, S] += 1.0
        l1 = normalize(l1, X[0])
        ys = rk4([l1, m0, b0, g0], X, Y)
        d1_0 = (X[0] * np.abs(l0 - l1)).sum()
        d1_t = (X[0] * np.abs(xs[0] - ys[0])).sum() + (Y[0] * np.abs(xs[1] - ys[1])).sum() \
            + np.abs(xs[2] - ys[2]).sum() + np.abs(xs[3] - ys[3]).sum()
        dinf_0 = np.abs(l0 - l1).max()
        dinf_t = max(np.abs(xs[0] - ys[0]).max(), np.abs(xs[1] - ys[1]).max())
        print(f"  M={M:5d} eps=2^-{k}: L1 {d1_0:.3e} -> {d1_t:.3e} (ratio {d1_t / d1_0:6.3f});"
              f"  sup {dinf_0:.3f} -> {dinf_t:.3f}")

print("(b) equal joint spectra without a reference-preserving bijection")
fw = np.array([0.5, 0.25, 0.25])                 # unequal atoms in each base fiber
X = make_space(fw)
Y = make_space(np.array([0.5, 0.5]))
P = rng.uniform(0.3, 1.7, size=(N, 2))          # value vector carried by mass 1/2 of fiber r
Q = rng.uniform(0.3, 1.7, size=(N, 2))          # value vector carried by the other half
# normalize each agent: sum_r (1/2)[(1/2)P + (1/2)Q] = 1
c = 0.25 * (P + Q).sum(axis=1, keepdims=True)
P, Q = P / c, Q / c
qx = np.concatenate([np.stack([P[:, r], Q[:, r], Q[:, r]], axis=1) for r in (0, 1)], axis=1)   # atoms (1/2,1/4,1/4)
qy = np.concatenate([np.stack([Q[:, r], P[:, r], P[:, r]], axis=1) for r in (0, 1)], axis=1)
mz = normalize(rng.uniform(-0.5, 0.5, size=(N, 4)), Y[0])
b0 = np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]); g0 = np.array([[0.4, 0.6], [0.5, 0.5], [0.7, 0.3]])
for name, val in (("x", qx), ("y", qy)):
    print(f"  {name}: normalizations", np.round((X[0] * val).sum(axis=1), 15))
xs = rk4([np.log(qx), mz, b0, g0], X, Y, T=2.0)
ys = rk4([np.log(qy), mz, b0, g0], X, Y, T=2.0)
# joint spectrum: atom 1 of y must carry x's atom-2 vector and atoms 2,3 of y x's atom-1 vector
res = 0.0
for r in (0, 1):
    o = 3 * r
    res = max(res, np.abs(ys[0][:, o] - xs[0][:, o + 1]).max(), np.abs(ys[0][:, o + 1] - xs[0][:, o]).max(),
              np.abs(ys[0][:, o + 2] - xs[0][:, o]).max(), np.abs(xs[0][:, o + 1] - xs[0][:, o + 2]).max())
print(f"  max deviation from the predicted value-vector exchange at t=2: {res:.2e}")
print(f"  rows: max |beta_x - beta_y| = {np.abs(xs[2] - ys[2]).max():.2e}, "
      f"max |gamma_x - gamma_y| = {np.abs(xs[3] - ys[3]).max():.2e}; model laws {np.abs(xs[1] - ys[1]).max():.2e}")
print(f"  the configurations differ: sup |log q_x - log q_y| at t=2 = {np.abs(xs[0] - ys[0]).max():.3f}")
