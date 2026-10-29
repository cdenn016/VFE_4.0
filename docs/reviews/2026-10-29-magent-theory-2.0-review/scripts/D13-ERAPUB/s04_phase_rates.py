r"""D13-ERAPUB s04: linearization rates of thm:public-phase (chapters/09_public_compression.tex)
and of companion eq comp:eq-phase-rates, recomputed on finite alphabets.

Data: K = M = {0,1,2,3}, nonuniform references A, B with balanced features phi, psi in {-1,1};
N identical agents, identity peer channels, uniform nonself priors, Gibbs-reduced potential
with unequal temperatures T_q, T_s. Chart u_i with E_A u_i = 0, q_i = e^{u_i} A / E_A e^{u_i}
(likewise v_i, s_i). First variation of E_red with respect to q_i:
  h_i^q = log(q_i/A) - J phi E_{s_i} psi + sum_j beta_ij log(q_i/q_j) - sum_a beta_ai q_a/q_i,
chart flow u_i' = -(h_i^q - E_A h_i^q) (eq:public-phase-chart-flow).

Checks: (a) the chart field vanishes on the diagonal with the diagonal field f of
eq:public-phase-diagonal; (b) DV(0) by central differences, decomposed into common and
difference blocks; expected spectra {-1+J, -1-J, -1 (x4)} and the same shifted by
-kappa_N = -2N/(N-1); (c) the exact operator inf-norm of Df(u,v) equals 1 + |J| (1 - m^2)
for the relevant feature mean m, hence L = 1+|J| and it is attained at the reference;
(d) the gap condition 1+|J| < 1+kappa_N-|J| holds iff |J| < N/(N-1).
"""
from __future__ import annotations

import numpy as np

A = np.array([0.1, 0.4, 0.3, 0.2]); PHI = np.array([1.0, 1.0, -1.0, -1.0])   # A(phi=+1) = 0.5
B = np.array([0.3, 0.2, 0.15, 0.35]); PSI = np.array([1.0, 1.0, -1.0, -1.0])  # B(psi=+1) = 0.5
K = 4


def law(u, ref):
    w = ref * np.exp(u)
    return w / w.sum()


def field(Z, N, J, Tq, Ts):
    """Z has shape (N, 2K): rows (u_i, v_i). Returns the chart velocity, same shape."""
    U, Vv = Z[:, :K], Z[:, K:]
    q = np.array([law(U[i], A) for i in range(N)])
    s = np.array([law(Vv[i], B) for i in range(N)])
    out = np.zeros_like(Z)
    for sector, (P, ref, T, other, feat, ofeat) in enumerate(((q, A, Tq, s, PHI, PSI), (s, B, Ts, q, PSI, PHI))):
        KLm = np.array([[np.sum(P[i] * np.log(P[i] / P[j])) for j in range(N)] for i in range(N)])
        W = np.exp(-KLm / T)
        np.fill_diagonal(W, 0.0)
        W /= W.sum(axis=1, keepdims=True)
        for i in range(N):
            h = np.log(P[i] / ref) - J * feat * np.dot(other[i], ofeat)
            for j in range(N):
                if j != i:
                    h = h + W[i, j] * np.log(P[i] / P[j]) - W[j, i] * P[j] / P[i]
            vel = -(h - np.dot(ref, h))
            out[i, sector * K:(sector + 1) * K] = vel
    return out


def diag_field(u, v, J):
    q, s = law(u, A), law(v, B)
    return np.concatenate([-u + J * PHI * np.dot(s, PSI), -v + J * PSI * np.dot(q, PHI)])


def basis_mean_zero(ref):
    # K x (K-1) basis of {u : E_ref u = 0}: columns e_k - ref_k * 1
    E = np.eye(K)[:, :K - 1]
    return E - np.outer(np.ones(K), ref[:K - 1])


def main():
    rng = np.random.default_rng(11)
    PA, PB = basis_mean_zero(A), basis_mean_zero(B)
    # (a) diagonal check
    J, Tq, Ts = 0.7, 0.6, 1.9
    u0 = PA @ rng.normal(0, 0.5, K - 1)
    v0 = PB @ rng.normal(0, 0.5, K - 1)
    for N in (2, 3):
        Z = np.tile(np.concatenate([u0, v0]), (N, 1))
        F = field(Z, N, J, Tq, Ts)
        print(f'(a) N={N}: max |V(diag) - f| = {np.abs(F - diag_field(u0, v0, J)[None, :]).max():.2e}')
    # (b) linearization at the reference
    Pfull = np.block([[PA, np.zeros((K, K - 1))], [np.zeros((K, K - 1)), PB]])  # (2K) x (2K-2)
    for (N, J, Tq, Ts) in ((2, 0.3, 1.0, 1.0), (3, 0.9, 0.6, 1.9), (4, -0.6, 0.3, 2.5), (3, 0.99, 5.0, 0.05)):
        dim = 2 * K - 2
        n = N * dim
        Jac = np.zeros((n, n))
        hstep = 1e-6
        for k in range(n):
            e = np.zeros(n)
            e[k] = hstep
            Zp = np.array([Pfull @ e[i * dim:(i + 1) * dim] for i in range(N)])
            Fp = field(Zp, N, J, Tq, Ts)
            Fm = field(-Zp, N, J, Tq, Ts)
            col = np.concatenate([np.linalg.lstsq(Pfull, (Fp[i] - Fm[i]) / (2 * hstep), rcond=None)[0] for i in range(N)])
            Jac[:, k] = col
        # common and difference blocks
        Cmean = np.kron(np.ones((1, N)) / N, np.eye(dim))
        Emb = np.kron(np.ones((N, 1)), np.eye(dim))
        common = Cmean @ Jac @ Emb
        # zero-sum (difference) subspace basis: columns (e_i - e_N) (x) I, i < N
        S = np.zeros((n, (N - 1) * dim))
        for i in range(N - 1):
            S[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = np.eye(dim)
            S[(N - 1) * dim:, i * dim:(i + 1) * dim] = -np.eye(dim)
        diffb = np.linalg.lstsq(S, Jac @ S, rcond=None)[0]
        inv_res = np.abs(S @ diffb - Jac @ S).max()  # invariance of the difference subspace
        ev_all = np.sort(np.linalg.eigvals(Jac).real)
        ev_c = np.sort(np.linalg.eigvals(common).real)
        ev_d = np.sort(np.linalg.eigvals(diffb).real)
        kap = 2 * N / (N - 1)
        exp_c = np.sort(np.array([-1 + J, -1 - J] + [-1.0] * (dim - 2)))
        exp_d = np.sort(np.concatenate([np.array([-1 - kap + J, -1 - kap - J] + [-1.0 - kap] * (dim - 2))] * (N - 1)))
        exp_all = np.sort(np.concatenate([exp_c, exp_d]))
        print(f'(b) N={N}, J={J}, Tq={Tq}, Ts={Ts}: kappa_N={kap:.4f}; difference subspace invariant to {inv_res:.1e}')
        print(f'    common-block eigenvalues {np.round(ev_c, 7)}; max err vs {{-1+J,-1-J,-1}}: {np.abs(ev_c-exp_c).max():.1e}')
        print(f'    difference-block max err vs shifted spectrum: {np.abs(ev_d-exp_d).max():.1e}; full spectrum max err: {np.abs(ev_all-exp_all).max():.1e}')
        print(f'    slowest common rate {-ev_c.max():.7f} (1-|J| = {1-abs(J):.7f}); slowest difference rate {-ev_d.max():.7f} '
              f'(1+kappa_N-|J| = {1+kap-abs(J):.7f})')
    # (c) exact inf-norm of Df
    worst = 0.0
    for _ in range(2000):
        J = rng.uniform(-0.999, 0.999)
        u = PA @ rng.normal(0, 2, K - 1)
        v = PB @ rng.normal(0, 2, K - 1)
        q, s = law(u, A), law(v, B)
        mq, ms = np.dot(q, PHI), np.dot(s, PSI)
        # Jacobian of f on R^{2K} (Cov is shift invariant, so the formula holds on the full space)
        top = np.hstack([-np.eye(K), J * np.outer(PHI, s * (PSI - ms))])
        bot = np.hstack([J * np.outer(PSI, q * (PHI - mq)), -np.eye(K)])
        Mx = np.vstack([top, bot])
        nrm = np.abs(Mx).sum(axis=1).max()
        pred = 1 + abs(J) * max(1 - ms**2, 1 - mq**2)
        worst = max(worst, abs(nrm - pred))
    print(f'(c) max |inf-norm(Df) - (1 + |J| max(1-m_s^2, 1-m_q^2))| over 2000 samples: {worst:.2e}; at the reference this is 1+|J|')
    # (d) gap condition
    for N in (2, 3, 5, 50):
        print(f'(d) N={N}: gap 1+|J| < 1+kappa_N-|J| iff |J| < N/(N-1) = {N/(N-1):.4f}; stability needs |J| < 1')


if __name__ == '__main__':
    main()
