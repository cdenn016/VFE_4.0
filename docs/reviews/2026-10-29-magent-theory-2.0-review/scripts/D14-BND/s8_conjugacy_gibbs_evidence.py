r"""D14-BND s8.

(a) rem:boundary-heterogeneous: exact time derivative at t = 0 of the root-sign contrast of the fresh
    root response for a two-block state that satisfies the conjugacy condition at t = 0
    (f^L_1 = f^R_1, deeper structure different). A nonzero value proves the conjugacy set is not
    invariant (finite tables; the derivative is the exact fine velocity, no integration).
(b) Gibbs-reduced variant (paragraph after lem:boundary-global): fine flow of E_red (rows replaced by
    beta* computed from the current costs, envelope identity) against the profile flow with b*.
(c) rem:boundary-evidence: persistent base-only evidence term in every agent's potential; fine flow
    against the profile flow with V^q replaced by V^q - log L(o|r).
"""
import numpy as np

from bnd_common import (setup, lift, class_rows, fine_forces, fine_velocity, rk4, normalize_profile,
                        profile_velocity, profile_D, lca_class)

rng = np.random.default_rng(4242)

# ----------------------------------------------------------------------------- (a)
for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    fL = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    fR = fL.copy(); half = 2 ** (d - 1)
    for r in range(2):
        for blk in (slice(0, half), slice(half, 2 ** d)):
            w = np.exp(0.8 * rng.normal(size=half)); w *= fR[r, blk].mean() / w.mean(); fR[r, blk] = w
    g = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    left = np.array([paths[i][0][1] == 1 for i in range(N)])
    q = np.where(left[:, None, None], lift(fL, FD.spq), lift(fR, FD.spq))
    s = lift(g, FD.sps)
    b = rng.dirichlet(np.ones(d) * 2); c = rng.dirichlet(np.ones(d) * 2)
    beta, gamma = class_rows(b, d, paths), class_rows(c, d, paths)
    dq, ds, db, dg = fine_velocity((q, s, beta, gamma), FD)
    sig0 = FD.spq.sig[:, :, 0]
    resp, dresp = q.mean(axis=0), dq.mean(axis=0)
    for r in range(2):
        c0 = resp[r, sig0[r] > 0].mean() - resp[r, sig0[r] < 0].mean()
        c1 = dresp[r, sig0[r] > 0].mean() - dresp[r, sig0[r] < 0].mean()
        print(f"(a) d={d}, base point r={r}: root-sign contrast of the root response at t=0: {c0:.2e}; exact d/dt at t=0: {c1:.4e}")

# ----------------------------------------------------------------------------- (b)
def gibbs_rows(Dmat, pi_row, T, edges):
    beta = np.zeros_like(Dmat)
    for i in range(Dmat.shape[0]):
        m = edges[i]
        w = pi_row[i, m] * np.exp(-Dmat[i, m] / T)
        beta[i, m] = w / w.sum()
    return beta

for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    f = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    g = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    lam = A[:, None] * FD.spq.Uc[None, :]; mu = B[:, None] * FD.sps.Uc[None, :]

    def fine_red_vel(state, FD):
        q, s = state
        N = q.shape[0]
        # costs at the current laws
        Dq = np.zeros((N, N)); Ds = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if FD.edges[i, j]:
                    Dq[i, j] = np.sum(lam * q[i] * (np.log(q[i]) - np.log(q[j])))
                    Ds[i, j] = np.sum(mu * s[i] * (np.log(s[i]) - np.log(s[j])))
        bst, gst = gibbs_rows(Dq, FD.pi_row, FD.Tq, FD.edges), gibbs_rows(Ds, FD.pi_row, FD.Ts, FD.edges)
        hq, hs, _, _, _, _ = fine_forces(q, s, bst, gst, FD)
        dq = np.stack([-q[i] * (hq[i] - np.sum(lam * q[i] * hq[i])) for i in range(N)])
        ds = np.stack([-s[i] * (hs[i] - np.sum(mu * s[i] * hs[i])) for i in range(N)])
        return (dq, ds)

    def prof_red_vel(state, PD):
        f, g = state
        Dq, Ds = profile_D(f, PD), profile_D(g, PD, PD.B)
        bq = PD.pik * np.exp(-Dq / PD.Tq); bq /= bq.sum()
        bs = PD.pik * np.exp(-Ds / PD.Ts); bs /= bs.sum()
        out = profile_velocity((f, g, bq, bs), PD)
        return (out[0], out[1])

    xf = rk4(fine_red_vel, (lift(f, FD.spq), lift(g, FD.sps)), 0.01, 100, FD)
    xp = rk4(prof_red_vel, (f, g), 0.01, 100, PD)
    print(f"(b) Gibbs-reduced flow d={d}: sup|q - lift f| = {np.max(np.abs(xf[0] - lift(xp[0], FD.spq))):.2e}, "
          f"sup|s - lift g| = {np.max(np.abs(xf[1] - lift(xp[1], FD.sps))):.2e}")

# ----------------------------------------------------------------------------- (c)
for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    logL = np.log(np.array([0.25, 0.9]))           # log L(o | r) for the recorded o, per base point
    f = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    g = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    bq = rng.dirichlet(np.ones(d) * 2); bs = rng.dirichlet(np.ones(d) * 2)
    lam = A[:, None] * FD.spq.Uc[None, :]

    def fine_ev_vel(state, FD):
        q, s, beta, gamma = state
        out = list(fine_velocity(state, FD))
        # add the evidence force -log L(o|r) to every belief law: extra velocity -q (phi - E_q phi)
        phi = -logL[:, None] * np.ones_like(q[0])
        out[0] = out[0] + np.stack([-q[i] * (phi - np.sum(lam * q[i] * phi)) for i in range(N)])
        return tuple(out)

    def prof_ev_vel(state, PD):
        f, g, bq, bs = state
        out = list(profile_velocity(state, PD))
        phi = -logL[:, None] * np.ones_like(f)
        wq = PD.A[:, None] / f.shape[1]
        out[0] = out[0] - f * (phi - np.sum(wq * f * phi))
        return tuple(out)

    x0f = (lift(f, FD.spq), lift(g, FD.sps), class_rows(bq, d, paths), class_rows(bs, d, paths))
    xf = rk4(fine_ev_vel, x0f, 0.01, 100, FD)
    xp = rk4(prof_ev_vel, (f, g, bq, bs), 0.01, 100, PD)
    print(f"(c) evidence-augmented potential d={d}: sup|q - lift f| = {np.max(np.abs(xf[0] - lift(xp[0], FD.spq))):.2e}; "
          f"rows {np.max(np.abs(xf[2] - class_rows(xp[2], d, paths))):.2e}")
