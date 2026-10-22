r"""L09-OPEN: full bit-circle density model open:full-field on a discretized circle.

Angles on an M-point grid; nu = (1/2) sum_z delta_z x (1/M) sum_m delta_{theta_m}. The discretized
model is itself a finite-alphabet instance of the same architecture (bit kept, angle replaced by
the uniform law on the grid), so open:full-field is its exact Fisher flow.

  h_i(z,th) = 2 log q_i - sum_j beta_ij log mu_j(z) - sum_k beta_ki mu_k(z)/mu_i(z),
  dq_i/dt   = -q_i (h_i - E_{q_i} h_i),
  beta_ij   = softmax_j(-KL(q_i || mu_j x Haar)/T)   (pi_ij = 1/N, self included).

Tests (numerical evidence only, no proof):
  T1 rows from the full D_ij equal rows from KL(mu_i||mu_j) (cancellation of H_i^q);
  T2 product reference: angular factor follows f0^{e^{-2t}} / norm; bit marginals follow the binary run;
  T3 correlated initial laws of size eps_c: event sequence of the fixed selector;
  T4 H_I equivariance: rotate agents 0,1 by (+k,-k) grid steps -> identical rows/events, rotated laws;
  T5 moving representative open:moving-flow at a time with nonzero phase velocities;
  T6 normalization drift.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location("s3", pathlib.Path(__file__).with_name("s3_exact_binary_events.py"))
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)

N, M = 6, 64
TH = 2 * np.pi * np.arange(M) / M
ZS = np.array([1.0, -1.0])  # index 0: z=+1, index 1: z=-1


def mu_of(q: np.ndarray) -> np.ndarray:
    return q.mean(axis=2)  # (N,2): density of bit marginal w.r.t. uniform bit measure


def rows_full(q: np.ndarray, T: float) -> np.ndarray:
    mu = mu_of(q)  # (N,2)
    lq = np.log(q)
    # D_ij = E_nu[ q_i (log q_i - log mu_j(z)) ]
    term1 = (q * lq).mean(axis=(1, 2))  # (N,)
    term2 = np.einsum("izm,jz->ij", q, np.log(mu)) / (2 * M)
    D = term1[:, None] - term2
    D = D - D.min(axis=1, keepdims=True)
    W = np.exp(-D / T)
    return W / W.sum(axis=1, keepdims=True)


def rows_binary(q: np.ndarray, T: float) -> np.ndarray:
    mu = mu_of(q)
    D = 0.5 * np.einsum("iz,ijz->ij", mu, np.log(mu[:, None, :] / mu[None, :, :]))
    D = D - D.min(axis=1, keepdims=True)
    W = np.exp(-D / T)
    return W / W.sum(axis=1, keepdims=True)


def sector_V(q: np.ndarray, T: float) -> np.ndarray:
    mu = mu_of(q)
    beta = rows_full(q, T)
    h = 2 * np.log(q) - np.einsum("ij,jz->iz", beta, np.log(mu))[:, :, None] \
        - (np.einsum("ki,kz->iz", beta, mu) / mu)[:, :, None]
    Eh = (q * h).mean(axis=(1, 2))
    return -q * (h - Eh[:, None, None])


def full_field(T: float):
    def f(t: float, y: np.ndarray) -> np.ndarray:
        q = y[: N * 2 * M].reshape(N, 2, M); s = y[N * 2 * M:].reshape(N, 2, M)
        return np.concatenate([sector_V(q, T).ravel(), sector_V(s, T).ravel()])
    return f


def vonmises(k: float, a: float) -> np.ndarray:
    v = np.exp(k * np.cos(TH - a))
    return v / v.mean()


def make_laws(x: np.ndarray, kap: float, phases: np.ndarray, eps_c: float, cph: np.ndarray) -> np.ndarray:
    q = np.empty((N, 2, M))
    for i in range(N):
        for zi, z in enumerate(ZS):
            q[i, zi] = (1 + z * x[i]) * vonmises(kap, phases[i]) * (1 + eps_c * z * np.cos(TH - cph[i]))
        # renormalize each bit-conditional so bit marginal stays 1 + z x and total mass is 1
        for zi, z in enumerate(ZS):
            q[i, zi] *= (1 + z * x[i]) / q[i, zi].mean()
    return q


def rotate(f: np.ndarray, a: float) -> np.ndarray:
    r"""(tau_a f)(theta) = f(theta - a), spectral, along last axis."""
    F = np.fft.rfft(f, axis=-1)
    n = np.arange(F.shape[-1])
    return np.fft.irfft(F * np.exp(-1j * n * a), n=M, axis=-1)


def dtheta(f: np.ndarray) -> np.ndarray:
    F = np.fft.rfft(f, axis=-1)
    n = np.arange(F.shape[-1]).astype(float)
    n[-1] = 0.0  # drop Nyquist derivative
    return np.fft.irfft(1j * n * F, n=M, axis=-1)


def first_moment(q: np.ndarray) -> np.ndarray:
    return (q * np.exp(1j * TH)).mean(axis=(-2, -1))


def selector_from_traj(Yq, Ys, ts, T):
    P = [frozenset([i]) for i in range(N)]
    ev = []
    for k, t in enumerate(ts):
        bq = rows_full(Yq[k], T); bs = rows_full(Ys[k], T)
        Bs = [s3.block_rows(bq, P), s3.block_rows(bs, P)]
        n = len(P); used = set(); newP = []
        for I in range(n):
            for J in range(I + 1, n):
                ok, m = s3.qualifies(Bs, I, J)
                if ok:
                    used |= {I, J}; newP.append(P[I] | P[J]); ev.append((float(t), sorted(P[I] | P[J])))
        if used:
            P = newP + [P[K] for K in range(n) if K not in used]
    return ev


def integrate(q0, s0, T, t_end, nts):
    y0 = np.concatenate([q0.ravel(), s0.ravel()])
    sol = solve_ivp(full_field(T), (0, t_end), y0, rtol=1e-9, atol=1e-11, dense_output=True, method="RK45")
    ts = np.linspace(0, t_end, nts)
    Y = sol.sol(ts).T
    Yq = Y[:, : N * 2 * M].reshape(-1, N, 2, M); Ys = Y[:, N * 2 * M:].reshape(-1, N, 2, M)
    return sol, ts, Yq, Ys


def main() -> None:
    A, R, h = 6.0, 60.0, 5e-3
    T = h * h
    ref = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    rng = np.random.default_rng(7)
    xq = h * (ref + 0.05 * rng.standard_normal(N)); xs = h * (ref + 0.05 * rng.standard_normal(N))
    kap, lam = 1.5, 1.0
    ph_q = rng.uniform(0, 2 * np.pi, N); ph_s = rng.uniform(0, 2 * np.pi, N)
    t_end, nts = 4.0, 801

    # T1 + T2: product reference
    q0 = make_laws(xq, kap, ph_q, 0.0, ph_q); s0 = make_laws(xs, lam, ph_s, 0.0, ph_s)
    print("[T1] max |rows_full - rows_binary| at t=0:", np.max(np.abs(rows_full(q0, T) - rows_binary(q0, T))))
    sol, ts, Yq, Ys = integrate(q0, s0, T, t_end, nts)
    t = 2.0; k = np.searchsorted(ts, t)
    f_num = Yq[k, 0, 0] / Yq[k, 0, 0].mean()
    f_th = vonmises(kap * np.exp(-2 * ts[k]), ph_q[0])
    print(f"[T2] angular factor vs f0^(e^-2t)/norm at t={ts[k]:.2f}: max rel err =", np.max(np.abs(f_num / f_th - 1)))
    # binary comparison: s3 field in u-coordinates
    zb0 = np.concatenate([xq, xs]) / h
    solb = solve_ivp(s3.make_field(h, N), (0, t_end), zb0, rtol=1e-10, atol=1e-11, dense_output=True, method="LSODA")
    ub = solb.sol(ts[k])[:N]
    x_full = mu_of(Yq[k])[:, 0] - 1.0
    print(f"[T2] bit means full vs binary at t={ts[k]:.2f}: max|diff|/h =", np.max(np.abs(x_full - h * ub)) / h)
    ev0 = selector_from_traj(Yq, Ys, ts, T)
    print("[T3] product reference events:", [(round(e[0], 3), e[1]) for e in ev0])
    print("[T6] max normalization drift:", np.max(np.abs(Yq.mean(axis=(2, 3)) - 1)))

    # T3: correlated perturbations of different sizes
    for eps_c in [1e-4, 1e-3, 1e-2]:
        cph = rng.uniform(0, 2 * np.pi, N)
        q1 = make_laws(xq, kap, ph_q, eps_c, cph); s1 = make_laws(xs, lam, ph_s, eps_c, cph + 1.0)
        _, ts1, Yq1, Ys1 = integrate(q1, s1, T, t_end, nts)
        ev = selector_from_traj(Yq1, Ys1, ts1, T)
        print(f"[T3] correlated eps_c={eps_c:.0e} (eps_c/h={eps_c/h:.2f}) events:", [(round(e[0], 3), e[1]) for e in ev])

    # T4: H_I equivariance with correlated laws eps_c = 1e-4
    eps_c = 1e-4; cph = rng.uniform(0, 2 * np.pi, N)
    q1 = make_laws(xq, kap, ph_q, eps_c, cph); s1 = make_laws(xs, lam, ph_s, eps_c, cph + 1.0)
    kshift = 11
    q2 = q1.copy(); s2 = s1.copy()
    q2[0] = np.roll(q1[0], kshift, axis=-1); s2[0] = np.roll(s1[0], kshift, axis=-1)
    q2[1] = np.roll(q1[1], -kshift, axis=-1); s2[1] = np.roll(s1[1], -kshift, axis=-1)
    _, tsa, Yqa, Ysa = integrate(q1, s1, T, t_end, 201)
    _, tsb, Yqb, Ysb = integrate(q2, s2, T, t_end, 201)
    drow = max(np.max(np.abs(rows_full(Yqa[j], T) - rows_full(Yqb[j], T))) for j in range(len(tsa)))
    dlaw = max(np.max(np.abs(np.roll(Yqa[j, 0], kshift, axis=-1) - Yqb[j, 0])) for j in range(len(tsa)))
    resp = max(abs(np.prod(first_moment(Yqa[j, :2])) - np.prod(first_moment(Yqb[j, :2]))) for j in range(len(tsa)))
    indiv = max(abs(first_moment(Yqa[j, 0]) - first_moment(Yqb[j, 0])) for j in range(len(tsa)))
    print(f"[T4] H_I rotation (+{kshift},-{kshift}) grid steps: max row diff {drow:.2e}; max |rotated law diff| {dlaw:.2e};"
          f" aggregate-angle first-moment diff {resp:.2e}; individual first-moment diff {indiv:.2e}")

    # T5: moving representative for I = {0,1}, correlated laws eps_c = 1e-2 to get phase motion
    eps_c = 1e-2
    q1 = make_laws(xq, kap, ph_q, eps_c, ph_q + 1.3); s1 = make_laws(xs, lam, ph_s, eps_c, ph_s - 0.7)
    sol5, _, _, _ = integrate(q1, s1, T, 2.0, 11)

    def state(t):
        y = sol5.sol(t)
        return y[: N * 2 * M].reshape(N, 2, M), y[N * 2 * M:].reshape(N, 2, M)

    def canon(q, s, I=(0, 1)):
        th = np.angle(first_moment(q[list(I)]))
        th = np.unwrap(th)
        phi = th.mean()
        qt, st = q.copy(), s.copy()
        for n_, i in enumerate(I):
            qt[i] = rotate(q[i], phi - th[n_]); st[i] = rotate(s[i], phi - th[n_])
        return qt, st

    t0, dt = 1.0, 1e-4
    qa, sa = canon(*state(t0 + dt)); qb, sb = canon(*state(t0 - dt)); qc, sc = canon(*state(t0))
    lhs = (qa - qb) / (2 * dt)
    Vq = sector_V(qc, T)
    om = np.imag((Vq * np.exp(1j * TH)).mean(axis=(1, 2)) / first_moment(qc))
    I = [0, 1]; ombar = om[I].mean()
    rhs = Vq.copy()
    for i in I:
        rhs[i] = Vq[i] - (ombar - om[i]) * dtheta(qc[i])
    print(f"[T5] omega_0={om[0]:.3e}, omega_1={om[1]:.3e}; max|lhs - rhs| over block = {np.max(np.abs(lhs[I] - rhs[I])):.2e};"
          f" size of correction term = {np.max(np.abs([(ombar-om[i])*dtheta(qc[i]) for i in I])):.2e};"
          f" error if correction omitted = {np.max(np.abs(lhs[I] - Vq[I])):.2e}")


if __name__ == "__main__":
    main()
