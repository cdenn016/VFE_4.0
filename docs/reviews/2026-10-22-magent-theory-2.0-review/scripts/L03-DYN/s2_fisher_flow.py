"""L03-DYN script 2: chapter 3 potential on finite tables.

Checks prop:dyn-gibbs (eq:dyn-row-gap), eq:dyn-envelope, thm:dyn-finite-flow (eq:dyn-dissipation),
lower bounds, self-source bounds, the row relaxation rate at frozen costs, and global interior
existence for fixed channels (identity, and sparse channels without a null column) from
near-boundary initial data. Gradients are analytic and checked by complex-step differentiation.
Single-threaded, CPU, a few seconds.
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
N, nK, nM = 3, 3, 2
Tq, Ts = 0.6, 1.3


def rand_simplex(n, conc=1.0):
    x = rng.gamma(conc, size=n)
    return x / x.sum()


def make_problem(channel_kind):
    r = [rand_simplex(nM) for _ in range(N)]
    G = [np.array([rand_simplex(nK) for _ in range(nM)]) for _ in range(N)]
    pi = [np.full(N, 1.0 / N) for _ in range(N)]
    Kq = [[None] * N for _ in range(N)]
    Ks = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j or channel_kind == "identity":
                Kq[i][j] = np.eye(nK)
                Ks[i][j] = np.eye(nM)
            else:
                Pm = np.eye(nK)[rng.permutation(nK)]
                M = Pm.copy()
                M[0] = 0.6 * Pm[0] + 0.4 * rand_simplex(nK)
                Kq[i][j] = M
                Ks[i][j] = np.eye(nM)[rng.permutation(nM)]
    return dict(r=r, G=G, pi=pi, Kq=Kq, Ks=Ks)


sizes = [nK] * N + [nM] * N + [N] * N + [N] * N


def split(u):
    out, p = [], 0
    for n in sizes:
        out.append(u[p:p + n])
        p += n
    return out[:N], out[N:2 * N], out[2 * N:3 * N], out[3 * N:]


def kl(a, b):
    return np.sum(a * np.log(a / b))


def energy(u, P, parts=False):
    q, s, b, g = split(u)
    E0 = sum(np.sum(q[i] * np.log(q[i])) + np.sum(s[i] * np.log(s[i] / P['r'][i]))
             - s[i] @ np.log(P['G'][i]) @ q[i] for i in range(N))
    Eq = sum(b[i][j] * kl(q[i], q[j] @ P['Kq'][i][j]) for i in range(N) for j in range(N)) \
        + Tq * sum(kl(b[i], P['pi'][i]) for i in range(N))
    Es = sum(g[i][j] * kl(s[i], s[j] @ P['Ks'][i][j]) for i in range(N) for j in range(N)) \
        + Ts * sum(kl(g[i], P['pi'][i]) for i in range(N))
    if parts:
        return E0, Eq, Es
    return E0 + Eq + Es


def grad(u, P):
    q, s, b, g = split(u)
    gq = [np.zeros(nK) for _ in range(N)]
    gs = [np.zeros(nM) for _ in range(N)]
    gb = [np.zeros(N) for _ in range(N)]
    gg = [np.zeros(N) for _ in range(N)]
    for i in range(N):
        LG = np.log(P['G'][i])
        gq[i] += np.log(q[i]) + 1 - s[i] @ LG
        gs[i] += np.log(s[i] / P['r'][i]) + 1 - LG @ q[i]
        for j in range(N):
            K = P['Kq'][i][j]
            nu = q[j] @ K
            gq[i] += b[i][j] * (np.log(q[i] / nu) + 1)
            gq[j] += -b[i][j] * (K @ (q[i] / nu))
            gb[i][j] = kl(q[i], nu) + Tq * (np.log(b[i][j] / P['pi'][i][j]) + 1)
            Ksm = P['Ks'][i][j]
            nus = s[j] @ Ksm
            gs[i] += g[i][j] * (np.log(s[i] / nus) + 1)
            gs[j] += -g[i][j] * (Ksm @ (s[i] / nus))
            gg[i][j] = kl(s[i], nus) + Ts * (np.log(g[i][j] / P['pi'][i][j]) + 1)
    return np.concatenate(gq + gs + gb + gg)


def field(u, P):
    h = grad(u, P)
    v, p, var = np.empty_like(u), 0, 0.0
    for n in sizes:
        uu, hh = u[p:p + n], h[p:p + n]
        m = uu @ hh
        v[p:p + n] = -uu * (hh - m)
        var += uu @ (hh - m) ** 2
        p += n
    return v, var


def rand_state():
    return np.concatenate([rand_simplex(n) for n in sizes])


def reduced(u, P):
    q, s, b, g = split(u)
    E0, _, _ = energy(u, P, parts=True)
    val, bstar, gstar = E0, [], []
    for i in range(N):
        Dq = np.array([kl(q[i], q[j] @ P['Kq'][i][j]) for j in range(N)])
        Ds = np.array([kl(s[i], s[j] @ P['Ks'][i][j]) for j in range(N)])
        wq = P['pi'][i] * np.exp(-Dq / Tq)
        ws = P['pi'][i] * np.exp(-Ds / Ts)
        val = val - Tq * np.log(wq.sum()) - Ts * np.log(ws.sum())
        bstar.append(wq / wq.sum())
        gstar.append(ws / ws.sum())
    return val, bstar, gstar


def cstep_grad(f, u, h=1e-30):
    gr = np.empty(len(u))
    for k in range(len(u)):
        e = np.zeros(len(u), dtype=complex)
        e[k] = 1j * h
        gr[k] = np.imag(f(u.astype(complex) + e)) / h
    return gr



def to_tables(x):
    """log-coordinates -> normalized tables (softmax per table)."""
    u, p = np.empty_like(x), 0
    for n in sizes:
        z = x[p:p + n] - x[p:p + n].max()
        e = np.exp(z)
        u[p:p + n] = e / e.sum()
        p += n
    return u


def log_field(x, P):
    """d/dt log u_z = -(h_z - E_u h), the same Fisher flow in log coordinates."""
    u = to_tables(x)
    h = grad(u, P)
    v, p = np.empty_like(x), 0
    for n in sizes:
        uu, hh = u[p:p + n], h[p:p + n]
        v[p:p + n] = -(hh - uu @ hh)
        p += n
    return v


for kind in ("identity", "sparse"):
    P = make_problem(kind)
    print(f"=== channels: {kind} ===")
    u = rand_state()
    gc = cstep_grad(lambda x: energy(x, P), u)
    print("gradient: max |analytic - complex-step|      ", np.abs(gc - grad(u, P)).max())
    Ered, bs, gs_ = reduced(u, P)
    q, s, b, g = split(u)
    gapterm = Tq * sum(kl(b[i], bs[i]) for i in range(N)) + Ts * sum(kl(g[i], gs_[i]) for i in range(N))
    print("eq:dyn-row-gap residual                       ", abs(energy(u, P) - (Ered + gapterm)))
    E0, _, _ = energy(u, P, parts=True)
    print("ordering E, E_red, E0, 0 holds:", energy(u, P) + 1e-14 > Ered, Ered + 1e-14 > E0, E0 > 0)
    print("self rows beta*_ii at least pi_ii:", all(bs[i][i] + 1e-15 > P['pi'][i][i] for i in range(N)))
    ustar = np.concatenate(list(q) + list(s) + bs + gs_)
    nmat = N * (nK + nM)

    def ered_c(xm):
        x = np.concatenate([xm, u[nmat:].astype(complex)])
        qq, ss, _, _ = split(x)
        val = sum(np.sum(qq[i] * np.log(qq[i])) + np.sum(ss[i] * np.log(ss[i] / P['r'][i]))
                  - ss[i] @ np.log(P['G'][i]) @ qq[i] for i in range(N))
        for i in range(N):
            Dq = np.array([np.sum(qq[i] * np.log(qq[i] / (qq[j] @ P['Kq'][i][j]))) for j in range(N)])
            Ds = np.array([np.sum(ss[i] * np.log(ss[i] / (ss[j] @ P['Ks'][i][j]))) for j in range(N)])
            val = val - Tq * np.log(np.sum(P['pi'][i] * np.exp(-Dq / Tq)))
            val = val - Ts * np.log(np.sum(P['pi'][i] * np.exp(-Ds / Ts)))
        return val

    gred = cstep_grad(ered_c, u[:nmat])
    print("eq:dyn-envelope residual (max over partials)  ", np.abs(gred - grad(ustar, P)[:nmat]).max())
    X, var = field(u, P)
    print("eq:dyn-dissipation residual                   ", abs(grad(u, P) @ X + var))
    # near-boundary initial data, integrated in log coordinates (stiff solver)
    u0 = rand_state()
    q0, s0, b0, g0 = split(u0)
    for i in range(N):
        q0[i][:] = np.array([1 - 2e-9, 1e-9, 1e-9])[rng.permutation(nK)]
        s0[i][:] = np.array([1 - 1e-9, 1e-9])[rng.permutation(nM)]
        b0[i][:] = np.array([1 - 2e-9, 1e-9, 1e-9])[rng.permutation(N)]
    u0 = np.concatenate(list(q0) + list(s0) + list(b0) + list(g0))
    sol = solve_ivp(lambda t, x: log_field(x, P), (0, 60), np.log(u0), method="Radau", rtol=1e-9, atol=1e-11,
                    t_eval=np.linspace(0, 60, 601))
    U = np.array([to_tables(x) for x in sol.y.T])
    Es = np.array([energy(x, P) for x in U])
    mins = U.min(axis=1)
    print("flow success:", sol.success, " E(0)=%.4f E(60)=%.6f" % (Es[0], Es[-1]),
          " largest increase of E between samples: %.2e" % np.max(np.diff(Es)))
    print("min table entry: t=0 %.1e, min over t %.2e (at t=%.2f), t=60 %.3e"
          % (mins[0], mins.min(), sol.t[mins.argmin()], mins[-1]))
    print("final field norm %.2e" % np.linalg.norm(field(U[-1], P)[0]))
    nq, ns, nb = N * nK, N * nM, N * N
    blocks = {"q": U[:, :nq], "s": U[:, nq:nq + ns], "beta": U[:, nq + ns:nq + ns + nb], "gamma": U[:, nq + ns + nb:]}
    for name, blk in blocks.items():
        print("   block %-5s min at t=0 %.2e, min over t %.2e" % (name, blk[0].min(), blk.min()))
    CG = max(-np.log(P['G'][i]).max() for i in range(N))
    cmin = min(P['Kq'][i][j].sum(axis=0).min() for i in range(N) for j in range(N))
    Cbound = np.log(nK) + N + 1 + CG + np.log(1 / cmin)
    print("   q-block floor predicted min(m0, exp(-C)) = %.2e with C = %.3f; observed min %.2e"
          % (min(blocks["q"][0].min(), np.exp(-Cbound)), Cbound, blocks["q"].min()))
    t1 = np.searchsorted(sol.t, 1.0)
    print("min entry by t=1: %.3e" % mins[t1])

T = 0.25
Dfix = rng.uniform(0, 2, size=4)
pi4 = rand_simplex(4)
bstar = pi4 * np.exp(-Dfix / T)
bstar /= bstar.sum()
b0 = rand_simplex(4)
hb = lambda b: Dfix + T * (np.log(b / pi4) + 1)
f = lambda t, b: -b * (hb(b) - b @ hb(b))
sol = solve_ivp(f, (0, 8), b0, rtol=1e-12, atol=1e-14, t_eval=[0, 4, 8])
l = np.log(sol.y / bstar[:, None])
contrast = l[0] - l[1]
print("row log-odds contrast decay at frozen costs: measured rate %.8f, predicted T = %.8f"
      % (-np.log(contrast[2] / contrast[1]) / 4, T))
