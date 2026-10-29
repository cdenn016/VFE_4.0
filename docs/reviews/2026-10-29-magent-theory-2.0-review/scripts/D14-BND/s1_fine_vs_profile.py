r"""D14-BND s1: thm:boundary-tree-flow, lem:boundary-fine-wellposed (finite instances).

(a) The analytic fine first variation (eq:boundary-fine-force) against central finite
    differences of the fine potential at a random NON-family configuration.
(b) E = N F on the family (eq:boundary-tree-energy).
(c) RK4 integration of the fine Fisher flow (all tables, rows, both sectors, coupled rho)
    against the profile flow (eq:boundary-tree-flow, eq:boundary-tree-rows), t in [0, 1].
(d) Remove the incoming term from the profile flow: change in f(1).
(e) Null flow (peer costs deleted): the family is still invariant and its lift agrees.
"""
import numpy as np

from bnd_common import (setup, lift, class_rows, fine_energy, fine_forces, fine_velocity, rk4,
                        profile_energy, profile_velocity, random_profile, normalize_profile,
                        ProfileData)

rng = np.random.default_rng(20261029)


def fd_check(d, realization):
    N, M, paths, FD, PD = setup(d, 2, 2, rng, realization=realization)
    lam = FD.spq.A[:, None] * FD.spq.Uc[None, :]
    mu = FD.sps.A[:, None] * FD.sps.Uc[None, :]
    q = np.exp(0.5 * rng.normal(size=(N,) + lam.shape))
    s = np.exp(0.5 * rng.normal(size=(N,) + mu.shape))
    for i in range(N):
        q[i] /= np.sum(lam * q[i])
        s[i] /= np.sum(mu * s[i])
    beta = rng.dirichlet(np.ones(N - 1), size=N)
    gamma = rng.dirichlet(np.ones(N - 1), size=N)
    B = np.zeros((N, N)); G = np.zeros((N, N))
    for i in range(N):
        B[i, FD.edges[i]] = beta[i]
        G[i, FD.edges[i]] = gamma[i]
    hq, hs, hb, hg, _, _ = fine_forces(q, s, B, G, FD)
    worst = 0.0
    for trial in range(3):
        dq = rng.normal(size=q.shape)
        ds = rng.normal(size=s.shape)
        for i in range(N):  # mass-preserving directions
            dq[i] -= np.sum(lam * dq[i])
            ds[i] -= np.sum(mu * ds[i])
        dB = np.zeros_like(B); dG = np.zeros_like(G)
        for i in range(N):
            m = FD.edges[i]
            v = rng.normal(size=N - 1); v -= v.mean(); dB[i, m] = v
            v = rng.normal(size=N - 1); v -= v.mean(); dG[i, m] = v
        eps = 1e-6
        Ep = fine_energy(q + eps * dq, s + eps * ds, B + eps * dB, G + eps * dG, FD)
        Em = fine_energy(q - eps * dq, s - eps * ds, B - eps * dB, G - eps * dG, FD)
        fd = (Ep - Em) / (2 * eps)
        an = sum(np.sum(lam * hq[i] * dq[i]) + np.sum(mu * hs[i] * ds[i]) for i in range(N))
        an += np.sum(hb * dB) + np.sum(hg * dG)
        worst = max(worst, abs(fd - an) / max(1.0, abs(an)))
    return worst


def flow_check(d, realization, peer=True, t1=1.0, nsteps=100):
    N, M, paths, FD, PD = setup(d, 2, 3, rng, realization=realization, peer=peer)
    A, B = FD.spq.A, FD.sps.A
    f = normalize_profile(random_profile(rng, 2, d), A)
    g = normalize_profile(random_profile(rng, 3, d), B)
    bq = rng.dirichlet(np.ones(d) * 2)
    bs = rng.dirichlet(np.ones(d) * 2)
    q, s = lift(f, FD.spq), lift(g, FD.sps)
    beta, gamma = class_rows(bq, d, paths), class_rows(bs, d, paths)
    nd = len({tuple(np.round(q[i].ravel(), 12)) for i in range(N)})
    E = fine_energy(q, s, beta, gamma, FD)
    F = profile_energy(f, g, bq, bs, PD)
    dt = t1 / nsteps
    xf = rk4(fine_velocity, (q, s, beta, gamma), dt, nsteps, FD)
    xp = rk4(profile_velocity, (f, g, bq, bs), dt, nsteps, PD)
    qf, sf, bf, gf = xf
    fp, gp, bqp, bsp = xp
    eq = np.max(np.abs(qf - lift(fp, FD.spq)))
    es = np.max(np.abs(sf - lift(gp, FD.sps)))
    eb = max(np.max(np.abs(bf - class_rows(bqp, d, paths))), np.max(np.abs(gf - class_rows(bsp, d, paths))))
    # energy decrease
    E1 = fine_energy(qf, sf, bf, gf, FD)
    return dict(N=N, distinct_leaf_laws=nd, E_minus_NF=E - N * F, belief=eq, model=es, rows=eb,
                E0=E, E1=E1)


def incoming_check(d):
    N, M, paths, FD, PD = setup(d, 2, 3, rng)
    A, B = FD.spq.A, FD.sps.A
    f = normalize_profile(random_profile(rng, 2, d), A)
    g = normalize_profile(random_profile(rng, 3, d), B)
    bq = rng.dirichlet(np.ones(d) * 2); bs = rng.dirichlet(np.ones(d) * 2)
    xp = rk4(profile_velocity, (f, g, bq, bs), 0.01, 100, PD)

    def vel_no_incoming(state, PD):
        f, g, bq, bs = state
        f0, g0 = f.mean(axis=1), g.mean(axis=1)
        Vq = -(PD.logrho * (PD.B * g0)[None, :]).sum(axis=1)
        Vs = -(PD.logrho * (PD.A * f0)[:, None]).sum(axis=0)

        def force(p, V, b):
            lp = np.log(p)
            h = 2 * lp + V[:, None]
            for bk, Pk in zip(b, PD.P):
                h -= bk * (lp @ Pk.T)          # incoming ratio omitted
            return h
        hq, hs = force(f, Vq, bq), force(g, Vs, bs)
        wq = PD.A[:, None] / f.shape[1]; ws = PD.B[:, None] / g.shape[1]
        out = profile_velocity(state, PD)
        return (-f * (hq - np.sum(wq * f * hq)), -g * (hs - np.sum(ws * g * hs)), out[2], out[3])
    xn = rk4(vel_no_incoming, (f, g, bq, bs), 0.01, 100, PD)
    return np.max(np.abs(xp[0] - xn[0]))


print("(a) analytic fine first variation vs central FD of the fine potential (random non-family states)")
for d, rl in [(2, "coin"), (2, "circle"), (3, "coin")]:
    print(f"   d={d} {rl:6s}: max relative directional-derivative residual {fd_check(d, rl):.2e}")

print("(b,c) E - N F at t=0, and fine flow vs lifted profile flow at t=1 (RK4, dt=0.01)")
for d, rl in [(2, "coin"), (2, "circle"), (3, "coin")]:
    out = flow_check(d, rl)
    print(f"   d={d} {rl:6s}: N={out['N']} distinct leaf laws={out['distinct_leaf_laws']} "
          f"E-NF={out['E_minus_NF']:.2e}  sup|q - lift f|={out['belief']:.2e} "
          f"sup|s - lift g|={out['model']:.2e} sup|rows|={out['rows']:.2e}  E(0)={out['E0']:.6f} E(1)={out['E1']:.6f}")

print("(d) incoming term removed from the profile flow: sup |f(1) - f_noincoming(1)|")
for d in (2, 3):
    print(f"   d={d}: {incoming_check(d):.4f}")

print("(e) null flow (all peer costs deleted): fine vs lifted profile null flow at t=1")
for d, rl in [(2, "circle"), (3, "coin")]:
    out = flow_check(d, rl, peer=False)
    print(f"   d={d} {rl:6s}: sup|q - lift f|={out['belief']:.2e} sup|s - lift g|={out['model']:.2e} rows={out['rows']:.2e}")
