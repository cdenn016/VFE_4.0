r"""L05-FUS: numerical checks of chapters/07_smooth_closure.tex on positive finite tables.

Architecture (smooth:reference, smooth:shield): three agents; belief space Z x V, model space W x U,
Z = W = {-1,+1}; reference R_i = R_i^c (x) A^q (x) A^s with R_i^c(z,w) = r_i(w) G_i(w,z);
peer channels K_ij^q((z_j,v_j),(z_i,v_i)) = B_ij^q[mu](z_j,z_i) kappa_{i,z_i}(v_i) with a
law-dependent public channel (depends on source AND receiver public means). Gibbs-reduced rows.
The energy is evaluated from FULL tables (the cancellation identity is NOT used), and the
Fisher flow  u' = -u (h - E_u h) uses finite-difference gradients, so every channel derivative
(including dependence on the receiver conditional) is included.

Tests
 T1  full derivative w.r.t. q_i equals h_i(z) + ell_{i,z}(v) up to constants (split-dynamics proof)
 T2  rate d/dt log kappa_z = H(z) - ell_z (smooth:split-flow) and explicit solution (smooth:conditional-solution)
 T3  swapped conditionals with equal spectra -> identical public trajectories (smooth:spectrum-flow)
 T4  different spectra, same public marginals -> different public trajectories (force is nontrivial)
 T5  equal spectra across z -> public flow = coarse E_c Fisher flow (smooth:equal-spectra)
 T6  NON-MINIMALITY: private space V = V1 x V2 (4 points, uniform A); kappa_z = kappa1_z (x) psi vs
     kappa1_z (x) psi' with psi, psi' of different spectra -> spectra differ at every z but public
     trajectories coincide; control with psi at one z and psi' at the other -> they differ.
Single-threaded; runtime a few seconds.
"""
import numpy as np

ZV = np.array([-1.0, 1.0])
T_q, T_s = 0.7, 0.9
rng = np.random.default_rng(3)
NA = 3
r_ref = [np.array([0.4, 0.6]), np.array([0.55, 0.45]), np.array([0.3, 0.7])]
g_ref = [0.5, -0.3, 0.2]
Rc = [np.array([[r_ref[i][w] * (1 + g_ref[i] * ZV[w] * ZV[z]) / 2 for w in range(2)] for z in range(2)])
      for i in range(NA)]  # Rc[i][z, w]


def channel(m_src, m_rcv, sector):
    rho = (1 + m_src**2 + 0.5 * m_rcv**2) / 4 if sector == "q" else (1 + 0.5 * m_src**2 + m_rcv**2) / 4
    return (1 + rho * np.outer(ZV, ZV)) / 2   # B[z_src, z_rcv], rows sum to one


def energy(tabs, dV, dU):
    """tabs: list of 2*NA positive tables (q_0, s_0, q_1, s_1, ...); normalized inside."""
    q = [tabs[2 * i] / tabs[2 * i].sum() for i in range(NA)]
    s = [tabs[2 * i + 1] / tabs[2 * i + 1].sum() for i in range(NA)]
    Aq = np.full(dV, 1.0 / dV)
    As = np.full(dU, 1.0 / dU)
    E = 0.0
    for i in range(NA):
        joint = np.einsum('zv,wu->zvwu', q[i], s[i])
        ref = np.einsum('zw,v,u->zvwu', Rc[i], Aq, As)
        E += np.sum(joint * np.log(joint / ref))
    for sec, lst in (("q", q), ("s", s)):
        T = T_q if sec == "q" else T_s
        mus = [x.sum(1) for x in lst]
        ms = [mu @ ZV for mu in mus]
        conds = [lst[i] / mus[i][:, None] for i in range(NA)]
        for i in range(NA):
            Ds = []
            for j in range(NA):
                if j == i:
                    continue
                B = channel(ms[j], ms[i], sec)
                # full channel K((z_j,v_j),(z_i,v_i)) = B[z_j,z_i] * cond_i[z_i,v_i]
                nu = np.einsum('ab,bv->bv', lst[j].sum(1)[:, None] * B, conds[i]) if False else \
                    (lst[j].sum(1) @ B)[:, None] * conds[i]
                # explicit full-table computation of q_j K (sums over v_j through lst[j]):
                nu_full = np.einsum('ac,ab,bv->bv', lst[j], B, conds[i]) / 1.0
                Ds.append(np.sum(lst[i] * np.log(lst[i] / nu_full)))
            Ds = np.array(Ds)
            E += -T * np.log(np.mean(np.exp(-Ds / T)))
    return E


def coarse_energy(pub):
    """E_c on public tables mu_0, sigma_0, mu_1, ... (smooth:split-dynamics notation)."""
    mu = [pub[2 * i] / pub[2 * i].sum() for i in range(NA)]
    sg = [pub[2 * i + 1] / pub[2 * i + 1].sum() for i in range(NA)]
    E = 0.0
    for i in range(NA):
        joint = np.outer(mu[i], sg[i])
        E += np.sum(joint * np.log(joint / Rc[i]))
    for sec, lst in (("q", mu), ("s", sg)):
        T = T_q if sec == "q" else T_s
        ms = [x @ ZV for x in lst]
        for i in range(NA):
            Ds = np.array([np.sum(lst[i] * np.log(lst[i] / (lst[j] @ channel(ms[j], ms[i], sec))))
                           for j in range(NA) if j != i])
            E += -T * np.log(np.mean(np.exp(-Ds / T)))
    return E


def fd_grad(f, tabs, h=1e-6):
    g = []
    for k, t in enumerate(tabs):
        gk = np.zeros_like(t)
        for idx in np.ndindex(t.shape):
            tp = [x.copy() for x in tabs]
            tm = [x.copy() for x in tabs]
            tp[k][idx] += h
            tm[k][idx] -= h
            gk[idx] = (f(tp) - f(tm)) / (2 * h)
        g.append(gk)
    return g


def fisher_rhs(f, tabs):
    g = fd_grad(f, tabs)
    return [-t * (gk - np.sum(t * gk)) for t, gk in zip(tabs, g)]


def rk4(f, tabs, dt, nsteps, record=None):
    traj = [[x.copy() for x in tabs]]
    for _ in range(nsteps):
        k1 = fisher_rhs(f, tabs)
        k2 = fisher_rhs(f, [x + 0.5 * dt * k for x, k in zip(tabs, k1)])
        k3 = fisher_rhs(f, [x + 0.5 * dt * k for x, k in zip(tabs, k2)])
        k4 = fisher_rhs(f, [x + dt * k for x, k in zip(tabs, k3)])
        tabs = [x + dt / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(tabs, k1, k2, k3, k4)]
        traj.append([x.copy() for x in tabs])
    return traj


def make_state(mus, conds_q, sigs, conds_s):
    tabs = []
    for i in range(NA):
        tabs.append(mus[i][:, None] * conds_q[i])
        tabs.append(sigs[i][:, None] * conds_s[i])
    return tabs


def public(traj):
    return np.array([[t.sum(1) for t in state] for state in traj])  # (time, 2*NA, 2)


import sys
PART = sys.argv[1] if len(sys.argv) > 1 else "1"
dV = dU = 2
mus = [np.array([0.3, 0.7]), np.array([0.6, 0.4]), np.array([0.45, 0.55])]
sigs = [np.array([0.5, 0.5]), np.array([0.35, 0.65]), np.array([0.7, 0.3])]
cq = [np.array([[0.75, 0.25], [0.6, 0.4]]), np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([[0.9, 0.1], [0.3, 0.7]])]
cs = [np.array([[0.55, 0.45], [0.8, 0.2]]), np.array([[0.4, 0.6], [0.35, 0.65]]), np.array([[0.65, 0.35], [0.1, 0.9]])]
X0 = make_state(mus, cq, sigs, cs)
f = lambda t: energy(t, dV, dU)


def part1():
    global q0
    # T1: structure of the full derivative for q_0
    g = fd_grad(f, X0)
    q0 = X0[0]
    ell = np.log((q0 / q0.sum(1, keepdims=True)) / 0.5)
    resid = g[0] - ell
    print("T1: (dE/dq_0 - ell) column spread within each z (should be ~0):", np.ptp(resid, axis=1))
    g1 = g[1]; s0 = X0[1]
    ell_s = np.log((s0 / s0.sum(1, keepdims=True)) / 0.5)
    print("    same for s_0:", np.ptp(g1 - ell_s, axis=1))

    # T2: conditional rate and explicit solution
    rhs = fisher_rhs(f, X0)
    kap = q0 / q0.sum(1, keepdims=True)
    mu0 = q0.sum(1)
    dlogq = rhs[0] / q0
    dlogmu = rhs[0].sum(1) / mu0
    dlogkap = dlogq - dlogmu[:, None]
    H = np.sum(kap * ell, axis=1)
    print("T2: max |d/dt log kappa - (H - ell)| at t=0:", np.abs(dlogkap - (H[:, None] - ell)).max())
    dt, nst = 0.01, 100
    trajX = rk4(f, X0, dt, nst)
    qT = trajX[-1][0]
    kT = qT / qT.sum(1, keepdims=True)
    aT = np.exp(-dt * nst)
    kexp = kap**aT * 0.5**(1 - aT)
    kexp /= kexp.sum(1, keepdims=True)
    print("    explicit solution kappa_t ~ kappa_0^a A^(1-a), max residual at t=1:", np.abs(kT - kexp).max())

    # T3: swap conditional rows' entries for agent 0 at both public values (equal spectra under uniform A)
    cq_sw = [cq[0][:, ::-1].copy(), cq[1], cq[2]]
    cs_sw = [cs[0], cs[1][:, ::-1].copy(), cs[2]]
    X1 = make_state(mus, cq_sw, sigs, cs_sw)
    trajX1 = rk4(f, X1, dt, nst)
    print("T3: equal spectra (swapped private points), max public trajectory difference:",
          np.abs(public(trajX) - public(trajX1)).max())

    # T4: different spectra with the same public marginals
    cq_d = [np.array([[0.95, 0.05], [0.6, 0.4]]), cq[1], cq[2]]
    X2 = make_state(mus, cq_d, sigs, cs)
    trajX2 = rk4(f, X2, dt, nst)
    print("T4: different spectrum at one public value, max public trajectory difference:",
          np.abs(public(trajX) - public(trajX2)).max())

    # T5: equal spectra across z for every agent and sector -> coarse flow
    cq_e = [np.array([[0.75, 0.25], [0.25, 0.75]]), np.array([[0.2, 0.8], [0.8, 0.2]]), np.array([[0.6, 0.4], [0.6, 0.4]])]
    cs_e = [np.array([[0.55, 0.45], [0.45, 0.55]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.3, 0.7], [0.7, 0.3]])]
    X3 = make_state(mus, cq_e, sigs, cs_e)
    trajX3 = rk4(f, X3, dt, nst)
    pub0 = []
    for i in range(NA):
        pub0 += [mus[i].copy(), sigs[i].copy()]
    trajC = rk4(coarse_energy, pub0, dt, nst)
    pubC = np.array([[t for t in state] for state in trajC])
    print("T5: equal spectra across z: max |public(full) - coarse E_c flow|:", np.abs(public(trajX3) - pubC).max())


def part2():
    dt = 0.01
    # T6: non-minimality of the spectrum (private V = V1 x V2, uniform A on 4 points)
    dV = 4
    f4 = lambda t: energy(t, dV, dU)
    k1 = [np.array([[0.75, 0.25], [0.6, 0.4]]), np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([[0.9, 0.1], [0.3, 0.7]])]
    psi, psi2 = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    def prod_cond(k, p_minus, p_plus):
        return np.array([np.kron(k[0], p_minus), np.kron(k[1], p_plus)])
    def spectrum(row):
        return np.sort(np.log(row / 0.25))
    XA = make_state(mus, [prod_cond(k1[i], psi, psi) for i in range(NA)], sigs, cs)
    XB = make_state(mus, [prod_cond(k1[i], psi2, psi2) for i in range(NA)], sigs, cs)
    XC = make_state(mus, [prod_cond(k1[i], psi, psi2) for i in range(NA)], sigs, cs)
    print("T6: spectra of agent 0 at z=-1 under psi / psi':", spectrum(XA[0][0] / XA[0][0].sum()),
          spectrum(XB[0][0] / XB[0][0].sum()))
    nst6 = 50
    tA = rk4(f4, XA, dt, nst6)
    tB = rk4(f4, XB, dt, nst6)
    tC = rk4(f4, XC, dt, nst6)
    print("    max public difference, common psi vs common psi' (spectra differ everywhere):",
          np.abs(public(tA) - public(tB)).max())
    print("    control: psi at z=-1, psi' at z=+1 vs common psi:", np.abs(public(tA) - public(tC)).max())


if PART == "1":
    part1()
else:
    part2()
