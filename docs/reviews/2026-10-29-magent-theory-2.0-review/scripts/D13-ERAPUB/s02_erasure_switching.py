r"""D13-ERAPUB s02: preference-switch histories of the erasure construction under the
reciprocal flow and under its null flow (chapters/08_smooth_erasure.tex, sec:erase-status).

Model (written independently of the 2026-10-22 scripts): 24 Bernoulli means
theta = (x, z, h, y, w, hh), four agents each, unit Fisher gradient flow of the exact
Gibbs-reduced potential,
    \dot\theta = -(1-\theta^2) \partial_\theta E_red,
    d E_red = d E_0 + \sum_{ij} \beta^*_{ij} d C^q_{ij} + \sum_{ij} \gamma^*_{ij} d C^s_{ij},
with C^q_{ij} = D_eps(x_i, x_j) + d(z_i, alpha(H_q) rho z_j) + d(h_i, h_*).
The analytic gradient is checked against central finite differences of E_red.
Null flow (inh:null-flow): every peer cost deleted, \dot\theta = -(1-\theta^2) \partial E_0.
Selector (sec:erase-status): each agent's source of smallest row cost at the current state.

Outputs: (1) gradient check; (2) for an asymmetric crossed preparation, the exact per-agent
switch times along both flows and whether the preference map is an involution in between;
(3) a one-parameter scan of the initial public state of agent 0, comparing the ORDER of the
per-agent switches along the full and the null flow, and the resulting partition sequences
under two completions of the selector (mutual pairs; connected components).
Single-threaded; well under 30 s.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

J = np.log(3.0)
HSTAR, H0 = -0.5, 0.25
DELTA, RHO, EPS, SIG, T = 0.04, 0.01, 0.015, 0.5, 1.0
N = 4


def chi(r):
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = (r > 0) & (r < 1)
    rm = r[m]
    out[m] = 1.0 / (1.0 + np.exp(1 / rm - 1 / (1 - rm)))
    out[r >= 1] = 1.0
    return out


def dchi(r):
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = (r > 0) & (r < 1)
    rm = r[m]
    s = 1.0 / (1.0 + np.exp(1 / rm - 1 / (1 - rm)))
    out[m] = s * (1 - s) * (1 / rm**2 + 1 / (1 - rm) ** 2)
    return out


def alpha(H):
    return float(chi(np.array([(H - H0) / DELTA]))[0])


def dalpha(H):
    return float(dchi(np.array([(H - H0) / DELTA]))[0]) / DELTA


def d(x, c):
    return 0.5 * (1 + x) * np.log((1 + x) / (1 + c)) + 0.5 * (1 - x) * np.log((1 - x) / (1 - c))


def d1(x, c):  # derivative in first argument
    return np.arctanh(x) - np.arctanh(c)


def d2(x, c):  # derivative in second argument
    return (c - x) / (1 - c**2)


def costs(p, zz, g):
    """Row cost matrix C[i, j] of one sector (inf on the diagonal)."""
    a = alpha(g.mean())
    P = p[:, None]
    Q = p[None, :]
    c = (1 - EPS) * P + EPS * Q
    C = d(P, c) + d(zz[:, None], a * RHO * zz[None, :]) + d(g, HSTAR)[:, None]
    np.fill_diagonal(C, np.inf)
    return C


def rows(C):
    m = np.min(C, axis=1, keepdims=True)
    W = np.exp(-(C - m) / T)
    W[~np.isfinite(C)] = 0.0
    return W / W.sum(axis=1, keepdims=True)


def energy(th):
    x, z, h, y, w, hh = th.reshape(6, N)
    E0 = (d(x, 0 * x) + d(y, 0 * y) - J * x * y + np.log(np.cosh(J)) + d(z, 0 * z) + d(w, 0 * w)
          + d(h, HSTAR + 0 * h) + d(hh, HSTAR + 0 * hh)).sum()
    red = 0.0
    for p, zz, g in ((x, z, h), (y, w, hh)):
        C = costs(p, zz, g)
        m = np.min(C, axis=1)
        red += np.sum(m - T * np.log(np.sum(np.exp(-(C - m[:, None]) / T), axis=1) / (N - 1)))
    return E0 + red


def sector_grad(p, zz, g):
    a, da = alpha(g.mean()), dalpha(g.mean())
    B = rows(costs(p, zz, g))
    P, Q = p[:, None], p[None, :]
    c = (1 - EPS) * P + EPS * Q
    # receiver role on public: d/dP D_eps(P,Q) = d1(P,c) + (1-EPS) d2(P,c)
    recv_p = d1(P, c) + (1 - EPS) * d2(P, c)
    src_p = EPS * d2(P, c)  # d/dQ D_eps(P,Q): force on the source Q (column index)
    np.fill_diagonal(recv_p, 0.0)
    np.fill_diagonal(src_p, 0.0)
    gp = np.sum(B * recv_p, axis=1) + np.sum(B * src_p, axis=0)
    cz = a * RHO * zz[None, :]
    recv_z = d1(zz[:, None], cz)
    src_z = d2(zz[:, None], cz) * a * RHO
    np.fill_diagonal(recv_z, 0.0)
    np.fill_diagonal(src_z, 0.0)
    gz = np.sum(B * recv_z, axis=1) + np.sum(B * src_z, axis=0)
    gh = np.sum(B, axis=1) * d1(g, HSTAR)
    dad = d2(zz[:, None], cz) * RHO * zz[None, :]  # d/d alpha of d(z_i, alpha rho z_j)
    np.fill_diagonal(dad, 0.0)
    gh = gh + da / N * np.sum(B * dad)
    return gp, gz, gh


def grad(th):
    x, z, h, y, w, hh = th.reshape(6, N)
    gx, gz, gh = sector_grad(x, z, h)
    gy, gw, ghh = sector_grad(y, w, hh)
    gx = gx + np.arctanh(x) - J * y
    gy = gy + np.arctanh(y) - J * x
    gz = gz + np.arctanh(z)
    gw = gw + np.arctanh(w)
    gh = gh + d1(h, HSTAR)
    ghh = ghh + d1(hh, HSTAR)
    return np.concatenate([gx, gz, gh, gy, gw, ghh])


def grad0(th):
    x, z, h, y, w, hh = th.reshape(6, N)
    return np.concatenate([np.arctanh(x) - J * y, np.arctanh(z), d1(h, HSTAR),
                           np.arctanh(y) - J * x, np.arctanh(w), d1(hh, HSTAR)])


def rhs_full(t, th):
    return -(1 - th**2) * grad(th)


def rhs_null(t, th):
    return -(1 - th**2) * grad0(th)


def fd_grad(th, hstep=1e-6):
    g = np.zeros_like(th)
    for k in range(th.size):
        e = np.zeros_like(th)
        e[k] = hstep
        g[k] = (energy(th + e) - energy(th - e)) / (2 * hstep)
    return g


def prep(eta=0.0, x=None):
    # agents: 0 (P,+), 1 (N,+), 2 (P,-), 3 (N,-): crossed (0,1),(2,3); box (0,2),(1,3)
    if x is None:
        x = np.array([0.4990 + eta, -0.4950, 0.5060, -0.5040])
    z = SIG * np.array([1.0, 1.0, -1.0, -1.0])
    h = np.array([0.2730, 0.2745, 0.2760, 0.2775])
    y, w, hh = x.copy(), z.copy(), h.copy()  # model sector prepared like the belief sector
    return np.concatenate([x, z, h, y, w, hh])


def pref_map(th, sector=0):
    x, z, h, y, w, hh = th.reshape(6, N)
    p, zz, g = ((x, z, h), (y, w, hh))[sector]
    return tuple(int(k) for k in np.argmin(costs(p, zz, g), axis=1))


def switch_times(sol, t_end, ngrid=4000):
    """Per-agent times at which the belief-sector argmin changes (refined by brentq)."""
    ts = np.linspace(0, t_end, ngrid)
    maps = [pref_map(sol.sol(t)) for t in ts]
    events = []
    for k in range(1, ngrid):
        if maps[k] != maps[k - 1]:
            for i in range(N):
                if maps[k][i] != maps[k - 1][i]:
                    a_, b_ = maps[k - 1][i], maps[k][i]

                    def f(t, i=i, a_=a_, b_=b_):
                        th = sol.sol(t)
                        C = costs(th[0:4], th[4:8], th[8:12])
                        return C[i, a_] - C[i, b_]
                    tt = brentq(f, ts[k - 1], ts[k], xtol=1e-14)
                    events.append((tt, i, a_, b_))
    events.sort()
    return events, maps


def mutual_partition(m):
    blocks, used = [], set()
    for i in range(N):
        j = m[i]
        if m[j] == i and i < j:
            blocks.append((i, j)); used |= {i, j}
    blocks += [(i,) for i in range(N) if i not in used]
    return tuple(sorted(blocks))


def component_partition(m):
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i in range(N):
        ra, rb = find(i), find(m[i])
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for i in range(N):
        comps.setdefault(find(i), []).append(i)
    return tuple(sorted(tuple(sorted(v)) for v in comps.values()))


def history_from_events(th0, events, rule):
    m = list(pref_map(th0))
    seq = [rule(tuple(m))]
    for (_, i, a_, b_) in events:
        m[i] = b_
        p = rule(tuple(m))
        if p != seq[-1]:
            seq.append(p)
    return seq


def integrate(th0, rhs, t_end):
    return solve_ivp(rhs, (0, t_end), th0, method='LSODA', rtol=1e-11, atol=1e-13, dense_output=True)


def main():
    th0 = prep()
    rng = np.random.default_rng(3)
    thp = th0 + 0.003 * rng.standard_normal(th0.size)
    for pt in (th0, thp):
        print('(1) analytic vs finite-difference gradient of E_red, max |diff| =',
              f'{np.abs(grad(pt) - fd_grad(pt)).max():.2e}', '(gate alpha =', f'{alpha(pt[8:12].mean()):.4f})')
    t_end = 0.06
    full = integrate(th0, rhs_full, t_end)
    null = integrate(th0, rhs_null, t_end)
    for name, sol in (('full', full), ('null', null)):
        ev, maps = switch_times(sol, t_end)
        hq = np.array([sol.sol(t)[8:12].mean() for t in np.linspace(0, t_end, 6001)])
        tclose = np.linspace(0, t_end, 6001)[np.argmax(hq <= H0)]
        print(f'(2) {name}: initial map {pref_map(th0)}; gate closes at t ~ {tclose:.5f}')
        for (tt, i, a_, b_) in ev:
            th = sol.sol(tt + 1e-12)
            m = pref_map(th)
            inv = all(m[m[k]] == k for k in range(N))
            print(f'    t = {tt:.8f}: agent {i} switches {a_} -> {b_}; map {m}; involution: {inv}; alpha = {alpha(th[8:12].mean()):.3e}')
        third = [k for k in range(len(maps)) if any(maps[k][i] not in ((1, 0, 3, 2)[i], (2, 3, 0, 1)[i]) for i in range(N))]
        print(f'    grid times with a third-source argmin: {len(third)}')
        print('    mutual-pair partition sequence:', history_from_events(th0, ev, mutual_partition))
        print('    component partition sequence:  ', history_from_events(th0, ev, component_partition))
    # (3) scan of the initial public state of agent 0 (both sectors shifted together)
    print('(3) scan eta (shift of x_0 = y_0): order of first two switching agents, full vs null')
    rows_out = []
    for eta in np.linspace(-0.0080, 0.0080, 33):
        th = prep(eta)
        ef, _ = switch_times(integrate(th, rhs_full, t_end), t_end, ngrid=1500)
        en, _ = switch_times(integrate(th, rhs_null, t_end), t_end, ngrid=1500)
        of = tuple(e[1] for e in ef)
        on = tuple(e[1] for e in en)
        pf = history_from_events(th, ef, mutual_partition)
        pn = history_from_events(th, en, mutual_partition)
        rows_out.append((eta, of, on, pf == pn))
        print(f'    eta={eta:+.4f}: full order {of}, null order {on}; mutual-pair sequences equal: {pf == pn}')
    diff = [r for r in rows_out if r[1] != r[2]]
    print(f'(3) {len(diff)} of {len(rows_out)} scanned preparations have different switch orders along the two flows')


if __name__ == '__main__':
    main()
