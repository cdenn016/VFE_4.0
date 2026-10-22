r"""G2: global bounded-log propagation for identity public channels (max/min principle with
energy-controlled centering), illustrated on a discretized public interval.

Agents i = 0..N-1, all others as sources, identity public channels (nu_ij = mu_j), Fisher rows,
self reference G_i = e^{g_i} rho, private forcing H_i(z,t) from explicit spectra.
First variation (normalized log densities m_i, mean(e^{m_i}) = 1):
  h_i = (m_i - log G_i) + sum_j beta_ij (m_i - m_j) - sum_k beta_ki e^{m_k - m_i} + H_i.
Claimed a priori bounds (u = max_i ess sup m_i, l = min_i ess inf m_i):
  u(t) <= e^{-t} u(0) + (1 - e^{-t}) C_up,  C_up = max_i sup log G_i + (N-1) + E(0),
  l(t) >= e^{-t} l(0) - (1 - e^{-t}) C_lo,  C_lo = (N-1) + M_H - min_i inf log G_i,
where M_H bounds H_i and E(0) is the total initial energy (public + private + rows).
"""
import numpy as np

N, T_ROW = 4, 0.4
SRC = {i: [j for j in range(N) if j != i] for i in range(N)}


def setup(n, mgrid, scale, seed):
    rng = np.random.default_rng(seed)
    z = (np.arange(n) + 0.5) / n
    v = (np.arange(mgrid) + 0.5) / mgrid
    logG, m0, ell0 = [], [], []
    for i in range(N):
        g = 1.2 * np.cos(2 * np.pi * (z + 0.25 * i))
        logG.append(g - np.log(np.mean(np.exp(g))))
        mi = scale * (np.sin(2 * np.pi * (i + 1) * z) + (z < 0.1 * (i + 1)))
        m0.append(mi - np.log(np.mean(np.exp(mi))))
        cz = 1.5 + np.sin(2 * np.pi * z + i)
        e = cz[:, None] * np.sin(2 * np.pi * (v[None, :] * (i + 1) + z[:, None]))
        ell0.append(e - np.log(np.mean(np.exp(e), axis=1, keepdims=True)))
    lb0 = [rng.normal(0, 1.5, N - 1) for _ in range(N)]
    return z, logG, m0, ell0, lb0


def H_of(ell0, t):
    a = np.exp(-t)
    lt = a * ell0
    lt = lt - np.log(np.mean(np.exp(lt), axis=1, keepdims=True))
    return np.mean(np.exp(lt) * lt, axis=1)


def h_all(m, beta, logG, Hs):
    mu = [np.exp(x) for x in m]
    h = []
    for i in range(N):
        hi = m[i] - logG[i]
        for a, j in enumerate(SRC[i]):
            hi = hi + beta[i][a] * (m[i] - m[j])
        for k in range(N):
            if i in SRC[k]:
                hi = hi - beta[k][SRC[k].index(i)] * mu[k] / mu[i]
        h.append(hi + Hs[i])
    return mu, h


def costs(m, beta):
    mu = [np.exp(x) for x in m]
    return [np.array([np.mean(mu[i] * (m[i] - m[j])) + T_ROW * (np.log(beta[i][a] * (N - 1)) + 1)
                      for a, j in enumerate(SRC[i])]) for i in range(N)]


def energy(m, beta, logG, Hs):
    mu = [np.exp(x) for x in m]
    E = 0.0
    for i in range(N):
        E += np.mean(mu[i] * (m[i] - logG[i])) + np.mean(mu[i] * Hs[i])
        for a, j in enumerate(SRC[i]):
            E += beta[i][a] * np.mean(mu[i] * (m[i] - m[j]))
        E += T_ROW * np.sum(beta[i] * np.log(beta[i] * (N - 1)))
    return E


def rhs(m, lb, logG, ell0, t):
    beta = [np.exp(l) / np.sum(np.exp(l)) for l in lb]
    Hs = [H_of(ell0[i], t) for i in range(N)]
    mu, h = h_all(m, beta, logG, Hs)
    cs = costs(m, beta)
    return ([-(h[i] - np.mean(mu[i] * h[i])) for i in range(N)],
            [-(cs[i] - np.sum(beta[i] * cs[i])) for i in range(N)])


# finite-difference check of h_i
z, logG, m0, ell0, lb0 = setup(20, 8, 1.0, 0)
beta0 = [np.exp(l) / np.sum(np.exp(l)) for l in lb0]
Hs0 = [H_of(ell0[i], 0.0) for i in range(N)]
mu, h = h_all(m0, beta0, logG, Hs0)
rng = np.random.default_rng(1)
worst = 0.0
for i in range(N):
    u = rng.normal(0, 1, z.size)
    eps = 1e-6
    mp = [x.copy() for x in m0]; mp[i] = m0[i] + eps * u; mp[i] -= np.log(np.mean(np.exp(mp[i])))
    mm = [x.copy() for x in m0]; mm[i] = m0[i] - eps * u; mm[i] -= np.log(np.mean(np.exp(mm[i])))
    fd = (energy(mp, beta0, logG, Hs0) - energy(mm, beta0, logG, Hs0)) / (2 * eps)
    an = np.mean(h[i] * mu[i] * (u - np.mean(mu[i] * u)))
    worst = max(worst, abs(fd - an))
print(f"FD check of identity-channel first variation: max residual {worst:.2e}")

for n, scale in [(32, 3.0), (128, 3.0), (512, 3.0)]:
    z, logG, m0, ell0, lb0 = setup(n, 12, scale, 2)
    beta0 = [np.exp(l) / np.sum(np.exp(l)) for l in lb0]
    E0 = energy(m0, beta0, logG, [H_of(ell0[i], 0.0) for i in range(N)])
    M_H = max(float(np.max(e)) for e in ell0)
    C_up = max(float(np.max(g)) for g in logG) + (N - 1) + E0
    C_lo = (N - 1) + M_H - min(float(np.min(g)) for g in logG)
    u0 = max(float(np.max(x)) for x in m0); l0 = min(float(np.min(x)) for x in m0)
    m, lb, t, dt = [x.copy() for x in m0], [x.copy() for x in lb0], 0.0, 0.002
    worst_u, worst_l, Emax_increase, Eprev = -np.inf, -np.inf, -np.inf, E0
    for step in range(1500):
        k1 = rhs(m, lb, logG, ell0, t)
        k2 = rhs([m[i] + 0.5 * dt * k1[0][i] for i in range(N)], [lb[i] + 0.5 * dt * k1[1][i] for i in range(N)], logG, ell0, t + 0.5 * dt)
        k3 = rhs([m[i] + 0.5 * dt * k2[0][i] for i in range(N)], [lb[i] + 0.5 * dt * k2[1][i] for i in range(N)], logG, ell0, t + 0.5 * dt)
        k4 = rhs([m[i] + dt * k3[0][i] for i in range(N)], [lb[i] + dt * k3[1][i] for i in range(N)], logG, ell0, t + dt)
        m = [m[i] + dt / 6 * (k1[0][i] + 2 * k2[0][i] + 2 * k3[0][i] + k4[0][i]) for i in range(N)]
        m = [x - np.log(np.mean(np.exp(x))) for x in m]
        lb = [lb[i] + dt / 6 * (k1[1][i] + 2 * k2[1][i] + 2 * k3[1][i] + k4[1][i]) for i in range(N)]
        t += dt
        u = max(float(np.max(x)) for x in m); l = min(float(np.min(x)) for x in m)
        worst_u = max(worst_u, u - (np.exp(-t) * u0 + (1 - np.exp(-t)) * C_up))
        worst_l = max(worst_l, (np.exp(-t) * l0 - (1 - np.exp(-t)) * C_lo) - l)
        beta = [np.exp(x) / np.sum(np.exp(x)) for x in lb]
        Et = energy(m, beta, logG, [H_of(ell0[i], t) for i in range(N)])
        Emax_increase = max(Emax_increase, Et - Eprev); Eprev = Et
    print(f"n={n}: u0={u0:.2f}, l0={l0:.2f}, u(3)={u:.2f}, l(3)={l:.2f}, C_up={C_up:.2f}, C_lo={C_lo:.2f}; "
          f"max excess over upper bound {worst_u:.3f}, over lower bound {worst_l:.3f} (<= 0 expected); "
          f"max one-step energy increase {Emax_increase:.1e}")
