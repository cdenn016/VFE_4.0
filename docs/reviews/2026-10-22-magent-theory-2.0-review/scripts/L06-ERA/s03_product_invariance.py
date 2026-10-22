r"""L06-ERA script 3: full-table Fisher flow preserves the product family (erase:family).

On full positive tables u^a over {-1,1}^3 (four agents, two sectors), define E_red
with channel parameters taken from marginal means, as the chapter specifies.
At a product point compute the table Fisher velocity
    \dot u_z = -u_z (h_z - \sum_w u_w h_w),   h = \partial E / \partial u
(complex-step derivative), then check (i) \dot u / u is additive in the three
coordinates, and (ii) the induced mean velocities equal the Bernoulli-mean flow
of s02 (gate open, alpha in the transition band so alpha' != 0).
"""
import itertools
import numpy as np
import s02_erasure_simulation as s2

K = np.array(list(itertools.product([-1, 1], repeat=3)))  # 8 x 3


def prod_table(m):
    return np.prod((1 + K * m[None, :]) / 2, axis=1)


def marg(u):
    return (K * u[:, None]).sum(axis=0)  # marginal means (k1,k2,k3)


def bern3(m):
    return prod_table(np.asarray(m))


def energy_tables(U, T):
    """U: shape (2, 4, 8) complex/real tables (sector q=0, s=1)."""
    q, s = U[0], U[1]
    Jv = s2.J
    # reference r(m) = b_0 b_0 b_h*, evaluator G(k|m) = G_J(k1|m1) b_0(k2) b_h*(k3)
    rm = bern3([0.0, 0.0, s2.HSTAR])
    E = 0.0
    for i in range(4):
        for a_ in range(8):
            for b_ in range(8):
                G = np.exp(Jv * K[a_, 0] * K[b_, 0]) / (2 * np.cosh(Jv)) * 0.5 * (1 + K[a_, 2] * s2.HSTAR)
                G = G * 0.5  # b_0(k2)
                E += q[i, a_] * s[i, b_] * np.log(q[i, a_] * s[i, b_] / (rm[b_] * G))
    for sec in (0, 1):
        tabs = U[sec]
        means = np.array([marg(t) for t in tabs])  # 4 x 3
        H = means[:, 2].mean()
        al = s2.alpha(H)
        for i in range(4):
            terms = []
            for j in range(4):
                if j == i:
                    continue
                cpub = (1 - s2.EPS) * means[i, 0] + s2.EPS * means[j, 0]
                nu = np.prod(np.stack([(1 + K[:, 0] * cpub) / 2,
                                       (1 + K[:, 1] * al * s2.RHO * means[j, 1]) / 2,
                                       (1 + K[:, 2] * s2.HSTAR) / 2 + 0 * cpub]), axis=0)
                Dij = (tabs[i] * np.log(tabs[i] / nu)).sum()
                terms.append(np.exp(-Dij / T))
            E += -T * np.log(sum(terms) / 3)
    return E


def main():
    T = 1.0
    th = s2.initial()
    th[0:4] += np.array([0.003, -0.002, 0.001, 0.004])
    x, z, h, y, w, hh = s2.unpack(th)
    U = np.zeros((2, 4, 8))
    for i in range(4):
        U[0, i] = prod_table(np.array([x[i], z[i], h[i]]))
        U[1, i] = prod_table(np.array([y[i], w[i], hh[i]]))
    step = 1e-30
    Hgrad = np.zeros_like(U)
    for idx in np.ndindex(U.shape):
        Uc = U.astype(complex)
        Uc[idx] += 1j * step
        Hgrad[idx] = np.imag(energy_tables(Uc, T)) / step
    Udot = -U * (Hgrad - (U * Hgrad).sum(axis=2, keepdims=True))
    # additivity of Udot/U
    design = np.column_stack([np.ones(8), K])
    worst = 0.0
    for sec in range(2):
        for i in range(4):
            g = Udot[sec, i] / U[sec, i]
            coef, *_ = np.linalg.lstsq(design, g, rcond=None)
            worst = max(worst, np.abs(design @ coef - g).max())
    print(f'max non-additive residual of udot/u at a product point: {worst:.2e}')
    # induced mean velocities vs Bernoulli-mean flow
    mean_vel = np.zeros((2, 4, 3))
    for sec in range(2):
        for i in range(4):
            mean_vel[sec, i] = (K * Udot[sec, i][:, None]).sum(axis=0)
    flow = s2.rhs(0.0, th, T).reshape(6, 4)
    ref = np.zeros((2, 4, 3))
    ref[0, :, 0], ref[0, :, 1], ref[0, :, 2] = flow[0], flow[1], flow[2]
    ref[1, :, 0], ref[1, :, 1], ref[1, :, 2] = flow[3], flow[4], flow[5]
    print(f'alpha_q at test point = {s2.alpha(h.mean()):.4f}, alpha_q\' = {s2.dalpha(h.mean()):.4f}')
    print(f'max |table-induced mean velocity - mean-coordinate flow| = {np.abs(mean_vel-ref).max():.2e}')
    print(f'scale of velocities: {np.abs(ref).max():.3e}')


if __name__ == '__main__':
    main()
