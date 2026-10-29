r"""D04-DYN s2: Fisher dynamics with equal mobilities on finite positive tables.

(a) thm:dyn-finite-flow: integrate eq:dyn-fisher-flow for the law-dependent three-agent model of s1
    and compare E(t)-E(0) with minus the integrated variance (eq:dyn-dissipation); mass and positivity.
(b) prop:dyn-row-relaxation: frozen costs, log-odds contrasts relax at rate T.
(c) prop:dyn-global-flow: fixed sparse channels with no null column, initial entries 1e-9 and 1e-12;
    check the floor m(t) >= min(m(0), -C) with the constant C of the proof, the model-sector floor,
    the row floor, the dissipation bound and convergence (field norm).
Single-threaded; integrates in log coordinates with DOP853.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, ".")
import d04_model as M

NA = M.NA


def flatten(u):
    parts = [u["q"].ravel(), u["s"].ravel()]
    parts += [u["beta"][i] for i in range(NA)] + [u["gamma"][i] for i in range(NA)]
    return np.concatenate(parts)


SZ = [2] * NA + [2] * NA + [len(M.Nq[i]) for i in range(NA)] + [len(M.Ns[i]) for i in range(NA)]
OFF = np.concatenate([[0], np.cumsum(SZ)])


def unflatten(x):
    blocks = [x[OFF[a]:OFF[a + 1]] for a in range(len(SZ))]
    q = np.array(blocks[:NA])
    s = np.array(blocks[NA:2 * NA])
    be = {i: blocks[2 * NA + i] for i in range(NA)}
    ga = {i: blocks[3 * NA + i] for i in range(NA)}
    return {"q": q, "s": s, "beta": be, "gamma": ga}


def rhs_log(t, y):
    x = np.exp(y[:-1])
    for a in range(len(SZ)):
        x[OFF[a]:OFF[a + 1]] /= x[OFF[a]:OFF[a + 1]].sum()
    u = unflatten(x)
    hq, hs, hb, hg = M.analytic_force(u, True)
    h = flatten({"q": hq, "s": hs, "beta": hb, "gamma": hg})
    dy = np.empty_like(y)
    var = 0.0
    for a in range(len(SZ)):
        sl = slice(OFF[a], OFF[a + 1])
        mean = np.sum(x[sl] * h[sl])
        dy[sl] = -(h[sl] - mean)
        var += np.sum(x[sl] * (h[sl] - mean) ** 2)
    dy[-1] = var
    return dy


print("== (a) thm:dyn-finite-flow, law-dependent channels: dissipation identity ==")
M.rng = np.random.default_rng(7)
u0 = M.random_point()
x0 = flatten(u0)
y0 = np.concatenate([np.log(x0), [0.0]])
ts = np.linspace(0, 8, 41)
sol = solve_ivp(rhs_log, (0, 8), y0, method="DOP853", t_eval=ts, rtol=1e-11, atol=1e-12)
Es, res, mass = [], [], []
for k in range(len(ts)):
    x = np.exp(sol.y[:-1, k])
    mass.append(max(abs(x[OFF[a]:OFF[a + 1]].sum() - 1) for a in range(len(SZ))))
    for a in range(len(SZ)):
        x[OFF[a]:OFF[a + 1]] /= x[OFF[a]:OFF[a + 1]].sum()
    Es.append(M.E_full(unflatten(x)))
    res.append(Es[-1] - Es[0] + sol.y[-1, k])
print(f"  E(0) = {Es[0]:.6f}, E(8) = {Es[-1]:.6f}; nonincreasing: {bool(np.all(np.diff(Es) <= 1e-12))}")
print(f"  max abs(E(t) - E(0) + int_0^t sum Var) = {max(abs(np.array(res))):.2e}; max mass drift {max(mass):.1e}")

print("== (b) prop:dyn-row-relaxation at frozen costs ==")
rng = np.random.default_rng(3)
for T in (0.05, 0.25, 1.7):
    n = 5
    D = rng.exponential(1.0, n)
    pi = rng.dirichlet(np.ones(n))
    bstar = pi * np.exp(-D / T)
    bstar /= bstar.sum()

    def rowrhs(t, lb):
        b = np.exp(lb)
        b /= b.sum()
        h = D + T * (np.log(b / pi) + 1)
        return -(h - np.sum(b * h))

    b0 = rng.dirichlet(np.ones(n))
    sl = solve_ivp(rowrhs, (0, 3), np.log(b0), method="DOP853", rtol=1e-12, atol=1e-13, t_eval=[0, 1, 2, 3])
    l = sl.y - np.log(bstar)[:, None]
    c = l[0] - l[1]
    rates = -np.diff(np.log(np.abs(c)))
    print(f"  T = {T}: measured rates of l_0 - l_1 per unit time = {np.round(rates, 10)}")

print("== (c) prop:dyn-global-flow: fixed channels, no null column, tiny initial entries ==")
rng = np.random.default_rng(11)
NA2, Lk, Lm = 3, 3, 2
T_q, T_s = 0.3, 0.5
r2 = [rng.dirichlet(np.ones(Lm) * 2) for _ in range(NA2)]
G2 = [np.array([rng.dirichlet(np.ones(Lk) * 2) for _m in range(Lm)]) for _ in range(NA2)]
Nq2 = {0: [0, 1, 2], 1: [0, 1], 2: [0, 1]}
Ns2 = {0: [1, 2], 1: [0, 1, 2], 2: [1]}
pq2 = {i: rng.dirichlet(np.ones(len(Nq2[i]))) for i in range(NA2)}
ps2 = {i: rng.dirichlet(np.ones(len(Ns2[i]))) for i in range(NA2)}
sparse_q = [np.array([[1, 0, 0], [1, 0, 0], [0, .5, .5]]), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            np.array([[.2, .8, 0], [0, 0, 1], [0, 0, 1]])]
sparse_s = [np.array([[1, 0], [.3, .7]]), np.array([[0, 1], [1, 0]])]
Kq2, Ks2 = {}, {}
for i in range(NA2):
    for j in Nq2[i]:
        Kq2[(i, j)] = np.eye(Lk) if j == i else sparse_q[(i + j) % 3]
    for j in Ns2[i]:
        Ks2[(i, j)] = np.eye(Lm) if j == i else sparse_s[(i + j) % 2]
cq = min(Kq2[e].sum(axis=0).min() for e in Kq2)
cs = min(Ks2[e].sum(axis=0).min() for e in Ks2)
SZ2 = [Lk] * NA2 + [Lm] * NA2 + [len(Nq2[i]) for i in range(NA2)] + [len(Ns2[i]) for i in range(NA2)]
OFF2 = np.concatenate([[0], np.cumsum(SZ2)])


def split2(x):
    b = [x[OFF2[a]:OFF2[a + 1]] for a in range(len(SZ2))]
    return b[:NA2], b[NA2:2 * NA2], b[2 * NA2:3 * NA2], b[3 * NA2:]


def kl(a, b):
    return float(np.sum(a * (np.log(a) - np.log(b))))


def energy2(x):
    q, s, be, ga = split2(x)
    tot = 0.0
    for i in range(NA2):
        R = (G2[i] * r2[i][:, None]).T
        z = np.outer(q[i], s[i])
        tot += np.sum(z * (np.log(z) - np.log(R)))
        tot += sum(be[i][n] * kl(q[i], q[j] @ Kq2[(i, j)]) for n, j in enumerate(Nq2[i])) + T_q * kl(be[i], pq2[i])
        tot += sum(ga[i][n] * kl(s[i], s[j] @ Ks2[(i, j)]) for n, j in enumerate(Ns2[i])) + T_s * kl(ga[i], ps2[i])
    return tot


def force2(x):
    q, s, be, ga = split2(x)
    hq = [np.log(q[i]) - s[i] @ np.log(G2[i]) for i in range(NA2)]
    hs = [np.log(s[i]) - np.log(r2[i]) - np.log(G2[i]) @ q[i] for i in range(NA2)]
    hb, hg = [], []
    for i in range(NA2):
        Dv = []
        for n, j in enumerate(Nq2[i]):
            nu = q[j] @ Kq2[(i, j)]
            hq[i] = hq[i] + be[i][n] * np.log(q[i] / nu)
            hq[j] = hq[j] - be[i][n] * (Kq2[(i, j)] @ (q[i] / nu))
            Dv.append(kl(q[i], nu))
        hb.append(np.array(Dv) + T_q * (np.log(be[i] / pq2[i]) + 1))
        Dv = []
        for n, j in enumerate(Ns2[i]):
            nu = s[j] @ Ks2[(i, j)]
            hs[i] = hs[i] + ga[i][n] * np.log(s[i] / nu)
            hs[j] = hs[j] - ga[i][n] * (Ks2[(i, j)] @ (s[i] / nu))
            Dv.append(kl(s[i], nu))
        hg.append(np.array(Dv) + T_s * (np.log(ga[i] / ps2[i]) + 1))
    return np.concatenate(hq + hs + hb + hg)


def rhs2(t, y):
    x = np.exp(y[:-1])
    for a in range(len(SZ2)):
        x[OFF2[a]:OFF2[a + 1]] /= x[OFF2[a]:OFF2[a + 1]].sum()
    h = force2(x)
    dy = np.empty_like(y)
    var = 0.0
    for a in range(len(SZ2)):
        sl = slice(OFF2[a], OFF2[a + 1])
        mean = np.sum(x[sl] * h[sl])
        dy[sl] = -(h[sl] - mean)
        var += np.sum(x[sl] * (h[sl] - mean) ** 2)
    dy[-1] = var
    return dy


C_G = max(float(np.max(-np.log(G2[i]))) for i in range(NA2))
C_r = max(float(np.max(-np.log(r2[i]))) for i in range(NA2))
Cq = C_G + max(0.0, -np.log(cq)) + np.log(Lk) + NA2
Cs = C_r + C_G + max(0.0, -np.log(cs)) + NA2
print(f"  c_min (belief) = {cq}, c_min (model) = {cs}; proof constants C_q = {Cq:.3f}, C_s = {Cs:.3f}")
for tiny in (1e-9, 1e-12):
    x0 = []
    for i in range(NA2):
        v = np.full(Lk, tiny)
        v[i % Lk] = 1 - (Lk - 1) * tiny
        x0.append(v)
    for i in range(NA2):
        v = np.full(Lm, tiny)
        v[(i + 1) % Lm] = 1 - (Lm - 1) * tiny
        x0.append(v)
    for i in range(NA2):
        v = np.full(len(Nq2[i]), tiny)
        v[-1] = 1 - (len(v) - 1) * tiny
        x0.append(v)
    for i in range(NA2):
        v = np.full(len(Ns2[i]), tiny)
        v[0] = 1 - (len(v) - 1) * tiny
        x0.append(v)
    x0 = np.concatenate(x0)
    y0 = np.concatenate([np.log(x0), [0.0]])
    tgrid = np.concatenate([np.linspace(0, 2, 201), np.linspace(2.5, 150, 296)])
    sol = solve_ivp(rhs2, (0, 150), y0, method="DOP853", t_eval=tgrid, rtol=1e-10, atol=1e-11)
    ly = sol.y[:-1]
    mq = ly[:NA2 * Lk].min(axis=0)
    ms = ly[NA2 * Lk:NA2 * (Lk + Lm)].min(axis=0)
    mb = ly[NA2 * (Lk + Lm):].min(axis=0)
    E0v = energy2(np.exp(y0[:-1]))
    xend = np.exp(ly[:, -1])
    Eend = energy2(xend)
    hend = force2(xend)
    vel = 0.0
    for a in range(len(SZ2)):
        sl = slice(OFF2[a], OFF2[a + 1])
        vel += np.sum(xend[sl] * (hend[sl] - np.sum(xend[sl] * hend[sl])) ** 2)
    print(f"  tiny = {tiny:.0e}: m_q(0) = {mq[0]:.2f}, min_t m_q = {mq.min():.2f}, bound min(m(0),-C_q) = {min(mq[0], -Cq):.2f}")
    print(f"     model: m_s(0) = {ms[0]:.2f}, min_t m_s = {ms.min():.2f}, bound {min(ms[0], -Cs):.2f}; rows: l(0) = {mb[0]:.2f}, min_t l = {mb.min():.2f}")
    print(f"     E(0) = {E0v:.4f}, E(150) = {Eend:.6f}, int Var = {sol.y[-1, -1]:.4f}; squared Fisher speed at t=150: {vel:.2e}; status {sol.status}")
    print(f"     final minimum log entry {ly[:, -1].min():.3f}")
    mstar = min(mq[0], -Cq)
    Dbar = -np.log(cq) - mstar
    rowb = min(mb[0], -Dbar / T_q - np.log(1 / min(min(pq2[i]) for i in range(NA2))))
    print(f"     proof row bound with Dbar = -log c_min - m_* = {Dbar:.2f}: min(l(0), -Dbar/T - log(1/min pi)) = {rowb:.2f}; satisfied: {mb.min() >= rowb}")

print("== (d) prop:dyn-global-flow when the energy pulls entries down (peaked evaluators) ==")
G2 = [np.array([[1 - 2e-6, 1e-6, 1e-6], [1e-6, 1 - 2e-6, 1e-6]]) for _ in range(NA2)]
r2 = [np.array([1 - 1e-5, 1e-5]) for _ in range(NA2)]
C_G = max(float(np.max(-np.log(G2[i]))) for i in range(NA2))
C_r = max(float(np.max(-np.log(r2[i]))) for i in range(NA2))
Cq = C_G + max(0.0, -np.log(cq)) + np.log(Lk) + NA2
Cs = C_r + C_G + max(0.0, -np.log(cs)) + NA2
x0 = np.concatenate([np.full(Lk, 1 / Lk)] * NA2 + [np.full(Lm, 1 / Lm)] * NA2
                    + [np.full(len(Nq2[i]), 1 / len(Nq2[i])) for i in range(NA2)]
                    + [np.full(len(Ns2[i]), 1 / len(Ns2[i])) for i in range(NA2)])
y0 = np.concatenate([np.log(x0), [0.0]])
sol = solve_ivp(rhs2, (0, 150), y0, method="DOP853", rtol=1e-10, atol=1e-11, dense_output=False,
                t_eval=np.linspace(0, 150, 1501))
ly = sol.y[:-1]
mq = ly[:NA2 * Lk].min(axis=0)
ms = ly[NA2 * Lk:NA2 * (Lk + Lm)].min(axis=0)
print(f"  C_q = {Cq:.2f}, C_s = {Cs:.2f}; min_t m_q = {mq.min():.3f} (bound {min(mq[0], -Cq):.2f}), min_t m_s = {ms.min():.3f} (bound {min(ms[0], -Cs):.2f}); status {sol.status}")

print("== (e) law-dependent channel leaving the interior in finite time (hypotheses of thm:dyn-finite-flow) ==")
# Two binary agents, trivial model sector. Agent 1 attends only to agent 2 through a binary symmetric channel
# whose flip probability eps = 0.05 + 0.3 (B - 1)/(sqrt2 - 1) depends on the Bhattacharyya coefficient
# B(q_1) = sqrt(q_1(0)) + sqrt(q_1(1)) of the receiver: entries are C-infinity on the interior and the compared law
# is positive there.  Agent 2 attends only to itself (identity self channel).
g1 = np.array([0.5, 0.5])
g2 = np.array([0.01, 0.99])
kap = 0.3 / (np.sqrt(2) - 1)


def parts(p1, p2):
    q1 = np.array([p1, 1 - p1])
    q2 = np.array([p2, 1 - p2])
    B = np.sqrt(q1).sum()
    eps = 0.05 + kap * (B - 1)
    K = np.array([[1 - eps, eps], [eps, 1 - eps]])
    nu = q2 @ K
    return q1, q2, B, eps, K, nu


def energy_e(p1, p2):
    q1, q2, B, eps, K, nu = parts(p1, p2)
    return kl(q1, g1) + kl(q2, g2) + kl(q1, nu)


def force_e(p1, p2):
    q1, q2, B, eps, K, nu = parts(p1, p2)
    ratio = q1 / nu
    h1 = np.log(q1 / g1) + np.log(ratio)                     # local + receiver roles
    dK = np.array([[-1.0, 1.0], [1.0, -1.0]])
    dnu = q2 @ dK                                            # d nu / d eps
    dD_deps = -np.sum(ratio * dnu)
    h1 = h1 + dD_deps * kap * 0.5 / np.sqrt(q1)              # channel role: d eps/d q1(w) = kap/(2 sqrt q1(w))
    h2 = np.log(q2 / g2) - (K @ ratio)                       # local + source roles
    return h1, h2, dD_deps


def rhs_e(t, y):
    p1, p2 = np.exp(y[0]), np.exp(y[1])
    q1 = np.array([p1, 1 - p1])
    q2 = np.array([p2, 1 - p2])
    h1, h2, _ = force_e(p1, p2)
    return [-(h1[0] - q1 @ h1), -(h2[0] - q2 @ h2)]


# check the analytic force against complex-free central differences in the tangent direction
p1, p2 = 3e-3, 0.02
h1, h2, dDe = force_e(p1, p2)
t = 1e-8
fd1 = (energy_e(p1 + t, p2) - energy_e(p1 - t, p2)) / (2 * t)
fd2 = (energy_e(p1, p2 + t) - energy_e(p1, p2 - t)) / (2 * t)
print(f"  force check: (h1_0 - h1_1) = {h1[0]-h1[1]:.6f} vs FD {fd1:.6f}; (h2_0 - h2_1) = {h2[0]-h2[1]:.6f} vs FD {fd2:.6f}; dD/deps = {dDe:.4f}")


levels = [-30.0, -60.0, -120.0, -240.0, -480.0]


def make_event(level):
    def ev(t, y):
        return y[0] - level
    ev.terminal = (level == levels[-1])
    ev.direction = -1
    return ev


evs = [make_event(L) for L in levels]
for p10 in (1e-3, 1e-5):
    sol = solve_ivp(rhs_e, (0, 5), [np.log(p10), np.log(0.02)], method="Radau", rtol=1e-10, atol=1e-12,
                    events=evs, max_step=1e-3)
    times = [sol.t_events[k][0] if len(sol.t_events[k]) else np.nan for k in range(len(levels))]
    print(f"  q_1(0) initial {p10:.0e}: first times log q_1(0) reaches {levels}: {np.round(times, 9)}")
    print(f"     successive differences {np.round(np.diff(times), 12)} (geometric decay means a finite exit time); q_2(0) at exit {np.exp(sol.y[1, -1]):.4f}")
    print(f"     solver status {sol.status}: {sol.message}; last t = {sol.t[-1]:.9f}, last log q_1(0) = {sol.y[0, -1]:.2f}")
    # asymptotic rate check: d log q / dt * sqrt(q) should approach a negative constant near the exit
    p1e, p2e = np.exp(sol.y[0, -1]), np.exp(sol.y[1, -1])
    h1, h2, dDe = force_e(p1e, p2e)
    rate = -(h1[0] - np.array([p1e, 1 - p1e]) @ h1)
    print(f"     at the last point d(log q)/dt * sqrt(q) = {rate*np.sqrt(p1e):.4f}; predicted -kap dD/deps / 2 = {-kap*dDe/2:.4f}")
