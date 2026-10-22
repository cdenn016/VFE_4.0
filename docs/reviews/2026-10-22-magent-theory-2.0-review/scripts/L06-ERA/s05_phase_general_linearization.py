r"""L06-ERA script 5: finite-alphabet instance of thm:public-phase.

K = M = 4 points, A = B uniform, phi = (1,1,-1,-1), psi = (1,-1,1,-1), N = 3 agents,
identity channels, uniform nonself priors, T_q != T_s.  Chart: u_i = log(dq_i/dA)
centered under A, likewise v_i.  Fisher flow in the chart: \dot u_i = -(h_i - E_A h_i),
h_i = \partial E_red / \partial q_i (complex step).

Checks:
 (1) diagonal field equals eq:public-phase-diagonal;
 (2) DV(0) equals eq:public-phase-linearization with kappa_N = 2N/(N-1);
 (3) at a nonzero diagonal point the averaged derivative kills zero-sum directions;
 (4) the average residual g(zbar, s delta) scales like s^2 (eq:public-phase-residual);
 (5) Pi computed numerically intertwines: Pi(Phi_t x) = phi_t(Pi x); Pi(x) differs from
     the log-average (geometric pool) at second order.
"""
import numpy as np
from scipy.integrate import solve_ivp

nK = 4
N = 3
J = 0.2
Tq, Ts = 0.7, 1.9
phi = np.array([1, 1, -1, -1.0])
psi = np.array([1, -1, 1, -1.0])
Aw = np.full(nK, 1 / nK)


def laws(u):
    w = np.exp(u)
    return w / (Aw * w).sum() * Aw  # probability vector


def kl(p, q):
    return (p * np.log(p / q)).sum()


def energy_tables(Q, S):
    E = 0.0
    for i in range(N):
        E += kl(Q[i], Aw) + kl(S[i], Aw) - J * (Q[i] * phi).sum() * (S[i] * psi).sum() + np.log(np.cosh(J))
        for P, T in ((Q, Tq), (S, Ts)):
            Z = sum(np.exp(-kl(P[i], P[j]) / T) for j in range(N) if j != i) / (N - 1)
            E += -T * np.log(Z)
    return E


def gibbs(P, T):
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j != i:
                B[i, j] = np.exp(-kl(P[i], P[j]) / T)
        B[i] /= B[i].sum()
    return B


def field(U):
    """Analytic envelope gradient; chart velocity -(h - E_A h)."""
    Q = np.array([laws(U[0, i]) for i in range(N)])
    S = np.array([laws(U[1, i]) for i in range(N)])
    H = np.zeros((2, N, nK))
    for sec, (P, O, T, f1, f2) in enumerate(((Q, S, Tq, phi, psi), (S, Q, Ts, psi, phi))):
        B = gibbs(P, T)
        for i in range(N):
            h = np.log(P[i] / Aw) - J * f1 * (O[i] * f2).sum()
            for j in range(N):
                if j != i:
                    h = h + B[i, j] * np.log(P[i] / P[j]) - B[j, i] * P[j] / P[i]
            H[sec, i] = h
    return -(H - (Aw * H).sum(axis=2, keepdims=True))


def field_cs(U):
    """U shape (2, N, nK): chart coordinates; returns chart velocity."""
    Q = np.array([laws(U[0, i]) for i in range(N)])
    S = np.array([laws(U[1, i]) for i in range(N)])
    H = np.zeros((2, N, nK))
    st = 1e-30
    for sec in range(2):
        for i in range(N):
            for k in range(nK):
                Qc, Sc = Q.astype(complex), S.astype(complex)
                if sec == 0:
                    Qc[i, k] += 1j * st
                else:
                    Sc[i, k] += 1j * st
                H[sec, i, k] = np.imag(energy_tables(Qc, Sc)) / st
    return -(H - (Aw * H).sum(axis=2, keepdims=True))


def center(u):
    return u - (Aw * u).sum(axis=-1, keepdims=True)


def fdiag(z):
    u, v = z
    q, s = laws(u), laws(v)
    return np.array([-u + J * phi * (s * psi).sum(), -v + J * psi * (q * phi).sum()])


def Aop(h):
    u, v = h
    return np.array([-u + J * phi * (Aw * psi * v).sum(), -v + J * psi * (Aw * phi * u).sum()])


def main():
    rng = np.random.default_rng(3)
    Ut = center(0.2 * rng.standard_normal((2, N, nK)))
    print('(0) analytic vs complex-step field:', f'{np.abs(field(Ut) - field_cs(Ut)).max():.2e}')
    # (1) diagonal field
    z = center(0.3 * rng.standard_normal((2, nK)))
    U = np.repeat(z[:, None, :], N, axis=1)
    V = field(U)
    print('(1) diagonal: max |V_i - f(z)| =', f'{np.abs(V - fdiag(z)[:, None, :]).max():.2e}')
    # (2) linearization at 0
    kap = 2 * N / (N - 1)
    worst = 0.0
    for _ in range(3):
        Hd = center(rng.standard_normal((2, N, nK)))
        s = 1e-5
        DV = (field(s * Hd) - field(-s * Hd)) / (2 * s)
        hbar = Hd.mean(axis=1, keepdims=True)
        pred = np.stack([Aop(Hd[:, i, :]) for i in range(N)], axis=1) - kap * (Hd - hbar)
        worst = max(worst, np.abs(DV - pred).max())
    print(f'(2) DV(0) vs A h_i - kappa_N (h_i - hbar), kappa_N={kap}: max residual {worst:.2e}')
    # (3) averaged derivative at a nonzero diagonal point on zero-sum directions
    Hd = center(rng.standard_normal((2, N, nK)))
    Hd -= Hd.mean(axis=1, keepdims=True)
    s = 1e-5
    DVbar = ((field(U + s * Hd) - field(U - s * Hd)) / (2 * s)).mean(axis=1)
    print(f'(3) |D Vbar(diag) . zero-sum| = {np.abs(DVbar).max():.2e} (scale of D V: '
          f'{np.abs((field(U + s*Hd) - field(U - s*Hd))/(2*s)).max():.2e})')
    # (4) residual scaling
    for sc in (2e-2, 1e-2, 5e-3):
        g = field(U + sc * Hd).mean(axis=1) - fdiag(z)
        print(f'(4) s={sc:.0e}: |g|={np.abs(g).max():.3e}, |g|/s^2={np.abs(g).max()/sc**2:.4f}')
    # (5) phase numerics
    def fine_rhs(t, y):
        return field(y.reshape(2, N, nK)).ravel()

    def diag_rhs(t, y):
        return fdiag(y.reshape(2, nK)).ravel()

    def Pi(X, tf=7.0):
        sol = solve_ivp(fine_rhs, (0, tf), X.ravel(), method='DOP853', rtol=1e-11, atol=1e-14)
        zb = sol.y[:, -1].reshape(2, N, nK).mean(axis=1)
        back = solve_ivp(lambda t, y: -diag_rhs(t, y), (0, tf), zb.ravel(), method='DOP853', rtol=1e-12, atol=1e-16)
        return back.y[:, -1].reshape(2, nK)

    X0 = center(0.15 * rng.standard_normal((2, N, nK)))
    P0 = Pi(X0)
    t1 = 0.5
    sol = solve_ivp(fine_rhs, (0, t1), X0.ravel(), method='DOP853', rtol=1e-11, atol=1e-14)
    X1 = sol.y[:, -1].reshape(2, N, nK)
    P1 = Pi(X1)
    fwd = solve_ivp(diag_rhs, (0, t1), P0.ravel(), method='DOP853', rtol=1e-12, atol=1e-16).y[:, -1].reshape(2, nK)
    delta0 = np.abs(X0 - X0.mean(axis=1, keepdims=True)).max()
    print(f'(5) intertwining |Pi(Phi_t x) - phi_t(Pi x)| = {np.abs(P1 - fwd).max():.2e}; '
          f'|Pi(x) - zbar| = {np.abs(P0 - X0.mean(axis=1)).max():.3e}; ||delta||^2 = {delta0**2:.3e}')
    print(f'(5) Pi(x) - Pi(x, tf=9) = {np.abs(P0 - Pi(X0, 9.0)).max():.2e}')


if __name__ == '__main__':
    main()
