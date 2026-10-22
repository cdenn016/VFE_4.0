r"""G2 checks for the public part of the closed system of chapter 7 (smooth:split-dynamics,
smooth:spectrum-flow) with a law-dependent public channel and Fisher rows.

Model (belief sector only; the model sector is the same computation with sectors exchanged):
  three agents, each attending to the other two, public space Z = [0,1] discretized by n
  midpoints with uniform reference rho, private space V = [0,1] discretized by m midpoints
  with uniform reference A.
  Self term: KL(mu_i || G_i), G_i = e^{g_i} rho / norm, g_i(z) = 1.5 sin(2 pi z + i)
  (this is the z-dependent part of KL(mu_i (x) sigma_i || R_i^c) at frozen sigma_i).
  Channel j -> i: B_ij[mu](z', dz) = (1 + c(x_j) phi(z') phi(z)) rho(dz),
     phi = cos(2 pi z), x_j = E_{mu_j} phi, c(x) = 0.3 + 0.15 tanh(x)   (law dependent).
  Hence nu_ij = mu_j B_ij has density 1 + g(x_j) phi with g(x) = c(x) x.
  Rows: Fisher rows with temperature T and uniform prior.

Checks:
  (1) finite-difference check of the first variation h_i (all interaction roles);
  (2) the fine Fisher flow on (z,v) tables equals the lift of the retained flow
      (public flow forced by H plus the explicit conditional solution);
  (3) dissipation identity along the retained flow;
  (4) the a priori oscillation bound
      osc m_i(t) <= e^{-2t} osc m_i(0) + (1 - e^{-2t}) C_i / 2
      with C_i computed from the channel and reference bounds only, for n = 16, 64, 256.
Numerical agreement illustrates the analytic estimate; it does not prove it.
"""
import numpy as np

T_ROW = 0.5
NAG = 3
SRC = {i: [j for j in range(NAG) if j != i] for i in range(NAG)}


def cfun(x):
    return 0.3 + 0.15 * np.tanh(x)


def gfun(x):
    return cfun(x) * x


def gprime(x):
    return cfun(x) + 0.15 * x / np.cosh(x) ** 2


class Model:
    def __init__(self, n: int):
        self.n = n
        self.z = (np.arange(n) + 0.5) / n
        self.phi = np.cos(2 * np.pi * self.z)
        self.logG = []
        for i in range(NAG):
            g = 1.5 * np.sin(2 * np.pi * self.z + i)
            self.logG.append(g - np.log(np.mean(np.exp(g))))

    def parts(self, m, beta):
        r"""Return mu, nu, D, h (first variations w.r.t. mu_i as functions of z)."""
        mu = [np.exp(mi) for mi in m]  # densities w.r.t. rho, mean 1
        x = [np.mean(mu[j] * self.phi) for j in range(NAG)]
        nu = {(i, j): 1.0 + gfun(x[j]) * self.phi for i in range(NAG) for j in SRC[i]}
        D = {(i, j): np.mean(mu[i] * (m[i] - np.log(nu[(i, j)]))) for i in range(NAG) for j in SRC[i]}
        h = []
        for i in range(NAG):
            hi = (m[i] - self.logG[i]).copy()
            for a, j in enumerate(SRC[i]):
                hi += beta[i][a] * (m[i] - np.log(nu[(i, j)]))
            # i as a source for every receiver k (channel depends on mu_i through x_i)
            for k in range(NAG):
                if i in SRC[k]:
                    a = SRC[k].index(i)
                    Iki = np.mean(mu[k] / nu[(k, i)] * self.phi)
                    hi += -beta[k][a] * gprime(x[i]) * Iki * self.phi
            h.append(hi)
        return mu, nu, D, h

    def energy_c(self, m, beta):
        mu, nu, D, h = self.parts(m, beta)
        E = 0.0
        for i in range(NAG):
            E += np.mean(mu[i] * (m[i] - self.logG[i]))
            for a, j in enumerate(SRC[i]):
                E += beta[i][a] * D[(i, j)]
            E += T_ROW * np.sum(beta[i] * np.log(beta[i] / 0.5))
        return E

    def row_costs(self, m, beta):
        mu, nu, D, h = self.parts(m, beta)
        return [np.array([D[(i, j)] + T_ROW * (np.log(beta[i][a] / 0.5) + 1.0)
                          for a, j in enumerate(SRC[i])]) for i in range(NAG)]


def normalize(mi):
    return mi - np.log(np.mean(np.exp(mi)))


def spectra_H(ell0, t):
    r"""Explicit conditional solution: ell_t = a ell_0 - log Z(a), H = E_kappa ell_t (per z)."""
    a = np.exp(-t)
    lt = a * ell0
    lt = lt - np.log(np.mean(np.exp(lt), axis=1, keepdims=True))
    H = np.mean(np.exp(lt) * lt, axis=1)
    return lt, H


def retained_rhs(model, m, lb, ell0s, t):
    beta = [np.exp(l) / np.sum(np.exp(l)) for l in lb]
    mu, nu, D, h = model.parts(m, beta)
    cs = model.row_costs(m, beta)
    dm, dlb = [], []
    for i in range(NAG):
        _, H = spectra_H(ell0s[i], t)
        f = h[i] + H
        dm.append(-(f - np.mean(mu[i] * f)))
        dlb.append(-(cs[i] - np.sum(beta[i] * cs[i])))
    return dm, dlb


def fine_rhs(model, lq, lb):
    r"""Fine Fisher flow on the (z,v) tables: d log q = -(h(z) + ell_z(v) - E_q[...])."""
    q = [np.exp(l) for l in lq]
    mu = [np.mean(qi, axis=1) for qi in q]
    m = [np.log(mi) for mi in mu]
    beta = [np.exp(l) / np.sum(np.exp(l)) for l in lb]
    _, _, D, h = model.parts(m, beta)
    cs = model.row_costs(m, beta)
    dl, dlb = [], []
    for i in range(NAG):
        ell = lq[i] - m[i][:, None]
        phi_full = h[i][:, None] + ell
        dl.append(-(phi_full - np.mean(q[i] * phi_full)))
        dlb.append(-(cs[i] - np.sum(beta[i] * cs[i])))
    return dl, dlb


def rk4_retained(model, m0, lb0, ell0s, T, dt):
    m, lb = [x.copy() for x in m0], [x.copy() for x in lb0]
    traj = [(0.0, [x.copy() for x in m], [x.copy() for x in lb])]
    t = 0.0
    nsteps = int(round(T / dt))
    for _ in range(nsteps):
        k1 = retained_rhs(model, m, lb, ell0s, t)
        m2 = [m[i] + 0.5 * dt * k1[0][i] for i in range(NAG)]; l2 = [lb[i] + 0.5 * dt * k1[1][i] for i in range(NAG)]
        k2 = retained_rhs(model, m2, l2, ell0s, t + 0.5 * dt)
        m3 = [m[i] + 0.5 * dt * k2[0][i] for i in range(NAG)]; l3 = [lb[i] + 0.5 * dt * k2[1][i] for i in range(NAG)]
        k3 = retained_rhs(model, m3, l3, ell0s, t + 0.5 * dt)
        m4 = [m[i] + dt * k3[0][i] for i in range(NAG)]; l4 = [lb[i] + dt * k3[1][i] for i in range(NAG)]
        k4 = retained_rhs(model, m4, l4, ell0s, t + dt)
        m = [normalize(m[i] + dt / 6 * (k1[0][i] + 2 * k2[0][i] + 2 * k3[0][i] + k4[0][i])) for i in range(NAG)]
        lb = [lb[i] + dt / 6 * (k1[1][i] + 2 * k2[1][i] + 2 * k3[1][i] + k4[1][i]) for i in range(NAG)]
        t += dt
        traj.append((t, [x.copy() for x in m], [x.copy() for x in lb]))
    return traj


def rk4_fine(model, lq0, lb0, T, dt):
    lq, lb = [x.copy() for x in lq0], [x.copy() for x in lb0]
    nsteps = int(round(T / dt))
    for _ in range(nsteps):
        k1 = fine_rhs(model, lq, lb)
        q2 = [lq[i] + 0.5 * dt * k1[0][i] for i in range(NAG)]; l2 = [lb[i] + 0.5 * dt * k1[1][i] for i in range(NAG)]
        k2 = fine_rhs(model, q2, l2)
        q3 = [lq[i] + 0.5 * dt * k2[0][i] for i in range(NAG)]; l3 = [lb[i] + 0.5 * dt * k2[1][i] for i in range(NAG)]
        k3 = fine_rhs(model, q3, l3)
        q4 = [lq[i] + dt * k3[0][i] for i in range(NAG)]; l4 = [lb[i] + dt * k3[1][i] for i in range(NAG)]
        k4 = fine_rhs(model, q4, l4)
        lq = [lq[i] + dt / 6 * (k1[0][i] + 2 * k2[0][i] + 2 * k3[0][i] + k4[0][i]) for i in range(NAG)]
        lq = [l - np.log(np.mean(np.exp(l))) for l in lq]
        lb = [lb[i] + dt / 6 * (k1[1][i] + 2 * k2[1][i] + 2 * k3[1][i] + k4[1][i]) for i in range(NAG)]
    return lq, lb


def initial_data(model, mgrid, osc_scale, seed):
    rng = np.random.default_rng(seed)
    z = model.z
    v = (np.arange(mgrid) + 0.5) / mgrid
    m0, ell0s = [], []
    for i in range(NAG):
        mi = osc_scale * (0.5 * np.sin(2 * np.pi * (z + 0.2 * i)) + 0.5 * (z > 0.6 + 0.1 * i))
        m0.append(normalize(mi))
        cz = 2.0 + 1.5 * np.sin(2 * np.pi * z * (i + 1))           # z-dependent private shape
        ell = cz[:, None] * np.cos(2 * np.pi * (v[None, :] + 0.3 * z[:, None] + 0.1 * i))
        ell = ell - np.log(np.mean(np.exp(ell), axis=1, keepdims=True))
        ell0s.append(ell)
    lb0 = [rng.normal(0, 1.0, 2) for _ in range(NAG)]
    return m0, lb0, ell0s


def osc(x):
    return float(np.max(x) - np.min(x))


print("(1) finite-difference check of the first variation h_i")
model = Model(24)
m0, lb0, ell0s = initial_data(model, 8, 2.0, 1)
beta0 = [np.exp(l) / np.sum(np.exp(l)) for l in lb0]
rng = np.random.default_rng(5)
mu, nu, D, h = model.parts(m0, beta0)
worst = 0.0
for i in range(NAG):
    u = rng.normal(0, 1, model.n)
    eps = 1e-6
    mp = [x.copy() for x in m0]; mp[i] = normalize(m0[i] + eps * u)
    mm = [x.copy() for x in m0]; mm[i] = normalize(m0[i] - eps * u)
    fd = (model.energy_c(mp, beta0) - model.energy_c(mm, beta0)) / (2 * eps)
    an = np.mean(h[i] * mu[i] * (u - np.mean(mu[i] * u)))
    worst = max(worst, abs(fd - an))
    print(f"  agent {i}: FD {fd:+.10f}  formula {an:+.10f}")
print(f"  max |FD - formula| = {worst:.2e}")

print("(1b) independent check: full fine energy with receiver-reconstruction channels")
print("     K_ij((z',v'),(z,v)) = b_ij[mu](z',z) kappa_{i,z}(v); claim dE/dq_i = h_i(z) + ell_{i,z}(v) + const")
model = Model(6)
mg = 4
m0, lb0, ell0s = initial_data(model, mg, 1.5, 7)
beta0 = [np.exp(l) / np.sum(np.exp(l)) for l in lb0]
rng = np.random.default_rng(11)
lq0 = [m0[i][:, None] + ell0s[i] + 0.3 * rng.normal(0, 1, (model.n, mg)) for i in range(NAG)]
lq0 = [l - np.log(np.mean(np.exp(l))) for l in lq0]


def full_energy(lq):
    Q = [np.exp(l) for l in lq]                       # densities w.r.t. rho (x) A, shape (n, mg)
    mu = [np.mean(Qi, axis=1) for Qi in Q]            # (n,)
    kap = [Q[i] / mu[i][:, None] for i in range(NAG)] # densities w.r.t. A, (n, mg)
    x = [np.mean(mu[j] * model.phi) for j in range(NAG)]
    E = 0.0
    for i in range(NAG):
        E += np.mean(Q[i] * (lq[i] - model.logG[i][:, None]))
        for a, j in enumerate(SRC[i]):
            b = 1.0 + cfun(x[j]) * np.outer(model.phi, model.phi)          # b(z', z), (n, n)
            # full 4-index kernel K[(z',v'),(z,v)] = b(z',z) kappa_i(z,v)
            K = b[:, None, :, None] * kap[i][None, None, :, :]           # (n, mg, n, mg)
            Nn = np.einsum('ab,abcd->cd', Q[j], K) / (model.n * mg)       # density of q_j K_ij
            E += beta0[i][a] * np.mean(Q[i] * (lq[i] - np.log(Nn)))
        E += T_ROW * np.sum(beta0[i] * np.log(beta0[i] / 0.5))
    return E


Q0 = [np.exp(l) for l in lq0]
mu0 = [np.mean(Qi, axis=1) for Qi in Q0]
_, _, _, h0 = model.parts([np.log(x) for x in mu0], beta0)
worst = 0.0
for i in range(NAG):
    U = rng.normal(0, 1, (model.n, mg))
    eps = 1e-6
    lp = [l.copy() for l in lq0]; lp[i] = lq0[i] + eps * U; lp[i] -= np.log(np.mean(np.exp(lp[i])))
    lm = [l.copy() for l in lq0]; lm[i] = lq0[i] - eps * U; lm[i] -= np.log(np.mean(np.exp(lm[i])))
    fd = (full_energy(lp) - full_energy(lm)) / (2 * eps)
    ell = lq0[i] - np.log(mu0[i])[:, None]
    Phi = h0[i][:, None] + ell
    an = np.mean(Phi * Q0[i] * (U - np.mean(Q0[i] * U)))
    worst = max(worst, abs(fd - an))
    print(f"  agent {i}: FD {fd:+.10f}  h(z)+ell(z,v) {an:+.10f}")
print(f"  max |FD - formula| = {worst:.2e}")

print("(2) fine flow on (z,v) tables vs lift of the retained flow, n=32, m=24, T=1.5")
model = Model(32)
mgrid = 24
m0, lb0, ell0s = initial_data(model, mgrid, 2.0, 2)
lq0 = [m0[i][:, None] + ell0s[i] for i in range(NAG)]
T, dt = 1.5, 0.005
traj = rk4_retained(model, m0, lb0, ell0s, T, dt)
lqT, lbT = rk4_fine(model, lq0, lb0, T, dt)
err_mu, err_kappa, err_rows = 0.0, 0.0, 0.0
for i in range(NAG):
    muF = np.mean(np.exp(lqT[i]), axis=1)
    muR = np.exp(traj[-1][1][i])
    err_mu = max(err_mu, np.max(np.abs(muF - muR)))
    ellF = lqT[i] - np.log(muF)[:, None]
    ellR, _ = spectra_H(ell0s[i], T)
    err_kappa = max(err_kappa, np.max(np.abs(ellF - ellR)))
    bF = np.exp(lbT[i]) / np.sum(np.exp(lbT[i])); bR = np.exp(traj[-1][2][i]) / np.sum(np.exp(traj[-1][2][i]))
    err_rows = max(err_rows, np.max(np.abs(bF - bR)))
print(f"  max |mu_fine - mu_retained| = {err_mu:.2e}; max |ell_fine - ell_explicit| = {err_kappa:.2e}; rows {err_rows:.2e}")

print("(3) dissipation identity along the retained flow (central differences)")
def total_energy(model, m, lb, ell0s, t):
    beta = [np.exp(l) / np.sum(np.exp(l)) for l in lb]
    E = model.energy_c(m, beta)
    for i in range(NAG):
        _, H = spectra_H(ell0s[i], t)
        E += np.mean(np.exp(m[i]) * H)
    return E

def dissipation(model, m, lb, ell0s, t):
    beta = [np.exp(l) / np.sum(np.exp(l)) for l in lb]
    mu, nu, D, h = model.parts(m, beta)
    cs = model.row_costs(m, beta)
    tot = 0.0
    for i in range(NAG):
        lt, H = spectra_H(ell0s[i], t)
        f = h[i] + H
        tot += np.mean(mu[i] * f ** 2) - np.mean(mu[i] * f) ** 2
        varl = np.mean(np.exp(lt) * lt ** 2, axis=1) - H ** 2
        tot += np.mean(mu[i] * varl)
        tot += np.sum(beta[i] * cs[i] ** 2) - np.sum(beta[i] * cs[i]) ** 2
    return -tot

for dt_e in [0.005, 0.0025]:
    traj = rk4_retained(model, m0, lb0, ell0s, 1.01, dt_e)
    worst = 0.0
    for tt in [0.1, 0.5, 1.0]:
        k = int(round(tt / dt_e))
        t = traj[k][0]
        Ep = total_energy(model, traj[k + 1][1], traj[k + 1][2], ell0s, traj[k + 1][0])
        Em = total_energy(model, traj[k - 1][1], traj[k - 1][2], ell0s, traj[k - 1][0])
        fd = (Ep - Em) / (2 * dt_e)
        an = dissipation(model, traj[k][1], traj[k][2], ell0s, t)
        worst = max(worst, abs(fd - an))
        print(f"  dt={dt_e}: t={t:.3f}: dE/dt (FD) {fd:+.8f}, -sum Var {an:+.8f}")
    print(f"  dt={dt_e}: max residual {worst:.2e} (central-difference error should scale like dt^2)")

print("(4) a priori oscillation bound, osc m_i(0) about 10")
nu_min, nu_max = 1 - 0.45, 1 + 0.45
xs = np.linspace(-1, 1, 20001)
gp_max = float(np.max(np.abs(gprime(xs))))
for n in [16, 64, 256]:
    model = Model(n)
    m0, lb0, ell0s = initial_data(model, 16, 10.0, 3)
    T, dt = 3.0, 0.002
    traj = rk4_retained(model, m0, lb0, ell0s, T, dt)
    worst_excess = -np.inf
    for i in range(NAG):
        Mplus = float(np.max(ell0s[i]))
        C_i = osc(model.logG[i]) + np.log(nu_max / nu_min) + 2 * 2 * gp_max / nu_min + Mplus
        o0 = osc(m0[i])
        for (t, m, lb) in traj[::25]:
            if t < 0.04:
                continue
            bound = np.exp(-2 * t) * o0 + (1 - np.exp(-2 * t)) * C_i / 2
            worst_excess = max(worst_excess, osc(m[i]) - bound)
        if i == 0:
            print(f"  n={n}: agent 0 C_0 = {C_i:.3f}, osc m_0(0) = {o0:.3f}, osc m_0(T) = {osc(traj[-1][1][0]):.3f}, "
                  f"min beta over agents at T = {min(float(np.min(np.exp(l)/np.sum(np.exp(l)))) for l in traj[-1][2]):.3f}")
    print(f"  n={n}: max over t in [0.05,3], i of [osc m_i(t) - bound_i(t)] = {worst_excess:.3f} (< 0 expected)")
