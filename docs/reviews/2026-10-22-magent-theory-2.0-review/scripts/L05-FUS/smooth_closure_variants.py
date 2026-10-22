r"""L05-FUS: two variants of the private part of the peer channel in chapters/07_smooth_closure.tex.

Three agents, belief space Z x V (Z={-1,+1}, V two points, uniform A), trivial model private factor
(dU = 1), law-dependent public channel B[mu], Gibbs-reduced rows; full-table energy and
finite-difference Fisher flow as in smooth_closure_flow.py.

 V1 FIXED reconstruction: K((z_j,v_j),(z_i,v_i)) = B[z_j,z_i] * A(v_i)  (state independent, same for
    all sources). Claim checked: private peer terms are equal across sources, cancel in the Gibbs
    rows up to the additive term int H dmu, so d/dt log kappa = 2 (H - ell) and swapped (equal-spectrum)
    conditionals give identical public trajectories.
 V2 PRIVATE-TRANSMITTING channel: K((z_j,v_j),(z_i,v_i)) = B[z_j,z_i] * delta_{v_j}(v_i).
    Claim checked: swapping the receiver's private points (spectrum preserved) changes D_ij and the
    public velocity, so the spectrum completion fails for channels that carry sender private data.
"""
import numpy as np

ZV = np.array([-1.0, 1.0])
T_q, T_s = 0.7, 0.9
NA = 3
r_ref = [np.array([0.4, 0.6]), np.array([0.55, 0.45]), np.array([0.3, 0.7])]
g_ref = [0.5, -0.3, 0.2]
Rc = [np.array([[r_ref[i][w] * (1 + g_ref[i] * ZV[w] * ZV[z]) / 2 for w in range(2)] for z in range(2)])
      for i in range(NA)]
dV = 2
Aq = np.full(dV, 1.0 / dV)


def channel(m_src, m_rcv, sector):
    rho = (1 + m_src**2 + 0.5 * m_rcv**2) / 4 if sector == "q" else (1 + 0.5 * m_src**2 + m_rcv**2) / 4
    return (1 + rho * np.outer(ZV, ZV)) / 2


def energy(tabs, mode):
    q = [tabs[2 * i] / tabs[2 * i].sum() for i in range(NA)]
    s = [tabs[2 * i + 1] / tabs[2 * i + 1].sum() for i in range(NA)]   # public model laws only
    E = 0.0
    for i in range(NA):
        joint = np.einsum('zv,w->zvw', q[i], s[i])
        ref = np.einsum('zw,v->zvw', Rc[i], Aq)
        E += np.sum(joint * np.log(joint / ref))
    mus = [x.sum(1) for x in q]
    mq = [mu @ ZV for mu in mus]
    for i in range(NA):
        Ds = []
        for j in range(NA):
            if j == i:
                continue
            B = channel(mq[j], mq[i], "q")
            if mode == "fixed":
                nu = (mus[j] @ B)[:, None] * Aq[None, :]
            else:  # private-transmitting: v_i = v_j
                nu = np.einsum('ac,ab->bc', q[j], B)
            Ds.append(np.sum(q[i] * np.log(q[i] / nu)))
        E += -T_q * np.log(np.mean(np.exp(-np.array(Ds) / T_q)))
    ms = [x @ ZV for x in s]
    for i in range(NA):
        Ds = np.array([np.sum(s[i] * np.log(s[i] / (s[j] @ channel(ms[j], ms[i], "s"))))
                       for j in range(NA) if j != i])
        E += -T_s * np.log(np.mean(np.exp(-Ds / T_s)))
    return E


def fd_grad(f, tabs, h=1e-6):
    g = []
    for k, t in enumerate(tabs):
        gk = np.zeros_like(t)
        for idx in np.ndindex(t.shape):
            tp = [x.copy() for x in tabs]; tm = [x.copy() for x in tabs]
            tp[k][idx] += h; tm[k][idx] -= h
            gk[idx] = (f(tp) - f(tm)) / (2 * h)
        g.append(gk)
    return g


def fisher_rhs(f, tabs):
    g = fd_grad(f, tabs)
    return [-t * (gk - np.sum(t * gk)) for t, gk in zip(tabs, g)]


def rk4(f, tabs, dt, nsteps):
    traj = [[x.copy() for x in tabs]]
    for _ in range(nsteps):
        k1 = fisher_rhs(f, tabs)
        k2 = fisher_rhs(f, [x + 0.5 * dt * k for x, k in zip(tabs, k1)])
        k3 = fisher_rhs(f, [x + 0.5 * dt * k for x, k in zip(tabs, k2)])
        k4 = fisher_rhs(f, [x + dt * k for x, k in zip(tabs, k3)])
        tabs = [x + dt / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(tabs, k1, k2, k3, k4)]
        traj.append([x.copy() for x in tabs])
    return traj


def public(traj):
    return np.array([[t.sum(1) if t.ndim == 2 else t for t in st] for st in traj], dtype=object)


def pub_arr(traj):
    return np.array([np.concatenate([st[2 * i].sum(1) for i in range(NA)] + [st[2 * i + 1] for i in range(NA)])
                     for st in traj])


mus = [np.array([0.3, 0.7]), np.array([0.6, 0.4]), np.array([0.45, 0.55])]
sigs = [np.array([0.5, 0.5]), np.array([0.35, 0.65]), np.array([0.7, 0.3])]
cq = [np.array([[0.75, 0.25], [0.6, 0.4]]), np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([[0.9, 0.1], [0.3, 0.7]])]
def state(conds):
    out = []
    for i in range(NA):
        out += [mus[i][:, None] * conds[i], sigs[i].copy()]
    return out
X = state(cq)
Xsw = state([cq[0][:, ::-1].copy(), cq[1], cq[2]])   # agent 0 private points swapped (same spectra)

# V1
f1 = lambda t: energy(t, "fixed")
rhs = fisher_rhs(f1, X)
q0 = X[0]
kap = q0 / q0.sum(1, keepdims=True)
ell = np.log(kap / 0.5)
H = np.sum(kap * ell, 1)
dlogkap = rhs[0] / q0 - (rhs[0].sum(1) / q0.sum(1))[:, None]
print("V1 fixed reconstruction: max |dlog kappa/dt - 2(H - ell)| =", np.abs(dlogkap - 2 * (H[:, None] - ell)).max())
tr1 = rk4(f1, X, 0.01, 40)
tr2 = rk4(f1, Xsw, 0.01, 40)
print("   swapped (equal spectra) max public trajectory difference:", np.abs(pub_arr(tr1) - pub_arr(tr2)).max())

# V2
f2 = lambda t: energy(t, "transmit")
r1 = fisher_rhs(f2, X)
r2 = fisher_rhs(f2, Xsw)
pub_vel = lambda r: np.concatenate([r[2 * i].sum(1) for i in range(NA)])
print("V2 private-transmitting channel: public belief velocity, original vs swapped:")
print("   ", pub_vel(r1))
print("   ", pub_vel(r2))
print("   max difference:", np.abs(pub_vel(r1) - pub_vel(r2)).max())
