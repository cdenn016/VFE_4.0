r"""D13-ERAPUB s03: independent check of erase:pair-convergence (chapters/08_smooth_erasure.tex).

Post-closure public system: 8 Bernoulli means v = (x_1..x_4, y_1..y_4), unit Fisher gradient
flow of E_pub = sum_i [d_0(x_i) + d_0(y_i) - J x_i y_i] - T sum_i log Z^q_i - T sum_i log Z^s_i,
Z^q_i = (1/3) sum_{j != i} exp(-D_eps(x_i, x_j)/T), and the same in the model sector.

Checks, for eps in {0.015, 0.05} and T from 1 down to 1e-6 (so including T << eps^2):
  (a) multi-start root finding of the public field in Q = ([l,u]^2)^2 x ([-u,-l]^2)^2:
      number of distinct zeros, box-mates equal, boxes opposite, x_eps - 1/2;
  (b) spectrum of the Jacobian at the zero (all eigenvalues negative real);
  (c) sampled sup over Q of ||D V_peer||_max / eps^2, where V_peer is the peer part of the field,
      to test whether the C^1 size of the peer field is O(eps^2) uniformly in T;
  (d) one long trajectory at T = 1e-6 from a heterogeneous admitted point.
Single-threaded, short.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

J = np.log(3.0)
L_BOX, U_BOX = 0.49, 0.51
N = 4


def d(x, c):
    return 0.5 * (1 + x) * np.log((1 + x) / (1 + c)) + 0.5 * (1 - x) * np.log((1 - x) / (1 - c))


def peer_grad(p, eps, T):
    P, Q = p[:, None], p[None, :]
    c = (1 - eps) * P + eps * Q
    C = d(P, c)
    np.fill_diagonal(C, np.inf)
    m = C.min(axis=1, keepdims=True)
    W = np.exp(-(C - m) / T)
    W[~np.isfinite(C)] = 0
    W /= W.sum(axis=1, keepdims=True)
    recv = np.arctanh(P) - np.arctanh(c) + (1 - eps) * (c - P) / (1 - c**2)
    src = eps * (c - P) / (1 - c**2)
    np.fill_diagonal(recv, 0)
    np.fill_diagonal(src, 0)
    return np.sum(W * recv, axis=1) + np.sum(W * src, axis=0)


def field(v, eps, T, peer=True, local=True):
    x, y = v[:4], v[4:]
    gx = np.zeros(4)
    gy = np.zeros(4)
    if local:
        gx += np.arctanh(x) - J * y
        gy += np.arctanh(y) - J * x
    if peer:
        gx += peer_grad(x, eps, T)
        gy += peer_grad(y, eps, T)
    return -(1 - v**2) * np.concatenate([gx, gy])


def jac(fun, v, h=1e-7):
    n = v.size
    Jm = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        Jm[:, k] = (fun(v + e) - fun(v - e)) / (2 * h)
    return Jm


def sample_Q(rng, n):
    s = np.array([1, 1, -1, -1.0])
    x = s * rng.uniform(L_BOX, U_BOX, (n, 4))
    y = s * rng.uniform(L_BOX, U_BOX, (n, 4))
    return np.hstack([x, y])


def in_Q(v):
    s = np.array([1, 1, -1, -1.0] * 2)
    w = s * v
    return np.all((w >= L_BOX - 1e-12) & (w <= U_BOX + 1e-12))


def main():
    rng = np.random.default_rng(7)
    starts = sample_Q(rng, 40)
    for eps in (0.015, 0.05):
        for T in (1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            zeros = []
            for v0 in starts:
                sol = root(lambda v: field(v, eps, T), v0, method='hybr', tol=1e-14)
                if sol.success and in_Q(sol.x) and np.abs(field(sol.x, eps, T)).max() < 1e-11:
                    if not any(np.abs(sol.x - z).max() < 1e-8 for z in zeros):
                        zeros.append(sol.x)
            z = zeros[0] if zeros else None
            line = f'eps={eps:<6} T={T:<7g}: zeros in Q found {len(zeros)}'
            if z is not None:
                ev = np.linalg.eigvals(jac(lambda v: field(v, eps, T), z))
                mates = max(abs(z[0] - z[1]), abs(z[2] - z[3]), abs(z[4] - z[5]), abs(z[6] - z[7]))
                opp = max(abs(z[0] + z[2]), abs(z[4] + z[6]))
                line += (f'; box-mate diff {mates:.1e}, box opposition {opp:.1e}, x_eps-1/2 = {z[0]-0.5:+.3e}, '
                         f'y_eps-x_eps = {z[4]-z[0]:+.1e}; max Re eig {ev.real.max():+.4f}, min Re eig {ev.real.min():+.4f}, '
                         f'max |Im| {np.abs(ev.imag).max():.1e}')
            print(line)
        # (c) C^1 size of the peer field, uniformly in T
        pts = sample_Q(rng, 60)
        for T in (1.0, 1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-8):
            worst0 = max(np.abs(field(v, eps, T, local=False)).max() for v in pts)
            worst1 = max(np.abs(jac(lambda w: field(w, eps, T, local=False), v)).sum(axis=1).max() for v in pts)
            print(f'   (c) eps={eps}, T={T:<7g} eps^2/T={eps**2/T:9.3g}: sup|V_peer|/eps^2 = {worst0/eps**2:.3f}, '
                  f'sup||D V_peer||_inf/eps^2 = {worst1/eps**2:.3f}')
    # (d) long trajectory at small T
    eps, T = 0.015, 1e-6
    v0 = np.array([0.4905, 0.5095, -0.4920, -0.5080, 0.5090, 0.4910, -0.5085, -0.4915])
    sol = solve_ivp(lambda t, v: field(v, eps, T), (0, 150), v0, method='LSODA', rtol=1e-11, atol=1e-13,
                    t_eval=[0, 10, 25, 50, 100, 150])
    for k, t in enumerate(sol.t):
        v = sol.y[:, k]
        sep = np.hypot(v[0] - v[1], v[4] - v[5])
        print(f'   (d) eps={eps}, T={T:g}: t={t:6.1f}  in-box separation (+ box) {sep:.3e}; in Q: {in_Q(v)}')


if __name__ == '__main__':
    main()
