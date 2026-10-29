r"""GAP02-S2 s4: independent numerical checks for open:parent-flow, open:quotient-gradient, D16-OPEN-06 and
the covariance estimate of open:open-laws (chapters/12_open_formation.tex:295-303, 362-402).

Discretization: z in {-1, 1}, K equispaced angles; densities relative to nu = (1/2) sum_z delta_z x dtheta/2pi,
so int f dnu = mean over the (2, K) grid. Belief sector only for the fine field (the model sector is a
separate copy and is rotated by the same belief-anchored angles); N = 3 agents, block I = {0, 1}, singleton {2}.
Fine field (open:full-field): h_i = 2 log q_i - sum_j beta_ij log mu_j(z) - sum_k beta_ki mu_k(z)/mu_i(z),
V_i = -q_i (h_i - E_{q_i} h_i), beta from KL(mu_i || mu_j)/T, self included.
Checks
 [1] open:moving-flow: q~_i(t) = tau_{a_i(t)} q_i(t), a_i = phi_I - theta_i (anchor phases of the fine
     trajectory), against V_i(x~) - (omegabar_I - omega_i) d_theta q~_i, at several times (central differences).
 [2] slice invariance: d/dt arg u~_i = omegabar_I for i in I (derived in the report).
 [3] Fisher orthogonality g(V, J xi) for single-agent rotations; dissipation dE/dt = -g(V, V); the slice
     velocity norm g(V + J eta, V + J eta) - g(V, V) = g(J eta, J eta) > 0 when phases move.
 [4] D16-OPEN-06: axially symmetric profile log f0 = -0.3 cos + 2.5 cos 2 + 1.2 cos 3; real anchor moment u(rho)
     of f0^rho; its root rho* and t* = -log(rho*)/2.
 [5] open:open-laws: |R_rho(l1) - R_rho(l2)| against the stated bound rho^2 M ||l1 - l2||_inf (M = max osc of
     the normalized log-conditionals) and the sharper rho^2 (M/2) osc(l1 - l2), on random profiles.
Run: OMP_NUM_THREADS=1 ... timeout 30 python s4_parent_flow.py
"""
from __future__ import annotations

import numpy as np

K = 128
TH = 2 * np.pi * np.arange(K) / K
KF = np.fft.fftfreq(K, d=1.0 / K)  # integer wave numbers
T = 0.5
rng = np.random.default_rng(7)


def mean(f):
    return f.mean(axis=(-2, -1))


def rot(f, a):
    """(tau_a f)(z, theta) = f(z, theta - a), spectral."""
    F = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(F * np.exp(-1j * KF * a), axis=-1))


def dth(f):
    F = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(1j * KF * F, axis=-1))


def marg(q):  # mu_i(z), density relative to uniform on {-1,1}
    return q.mean(axis=-1)


def kl_bits(m1, m2):
    return 0.5 * np.sum(m1 * np.log(m1 / m2))


def betas(q):
    N = q.shape[0]
    mu = marg(q)
    D = np.array([[kl_bits(mu[i], mu[j]) for j in range(N)] for i in range(N)])
    W = np.exp(-D / T)
    return W / W.sum(axis=1, keepdims=True), mu


def V(q):
    N = q.shape[0]
    b, mu = betas(q)
    out = np.empty_like(q)
    for i in range(N):
        h = 2 * np.log(q[i])
        h = h - sum(b[i, j] * np.log(mu[j])[:, None] for j in range(N))
        h = h - sum(b[k, i] * (mu[k] / mu[i])[:, None] for k in range(N))
        out[i] = -q[i] * (h - mean(q[i] * h))
    return out


def energy(q):
    N = q.shape[0]
    mu = marg(q)
    E = 0.0
    for i in range(N):
        E += mean(q[i] * np.log(q[i]))
        Hi = mean(q[i] * np.log(q[i] / mu[i][:, None]))
        Dij = np.array([Hi + kl_bits(mu[i], mu[j]) for j in range(N)])
        E += -T * np.log(np.mean(np.exp(-Dij / T)))
    return E


def rk4_step(q, dt):
    k1 = V(q); k2 = V(q + dt / 2 * k1); k3 = V(q + dt / 2 * k2); k4 = V(q + dt * k3)
    return q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def law(mu1, pars):
    """Correlated bit-angle law: q(z, th) = mu(z) f(th | z), f(.|z) = exp(k1 cos(th-c1) + k2 cos(2th-c2))/norm."""
    q = np.empty((2, K))
    for zi, (k1, c1, k2, c2) in enumerate(pars):
        f = np.exp(k1 * np.cos(TH - c1) + k2 * np.cos(2 * TH - c2))
        q[zi] = f / f.mean()
    mu = np.array([1 + mu1, 1 - mu1])  # z = +1 row first, z = -1 row second
    return q * mu[:, None]


q0 = np.stack([
    law(0.30, [(1.2, 0.3, 0.9, 1.1), (0.7, -0.4, 1.3, -0.6)]),
    law(0.10, [(0.9, 1.7, 1.1, 0.2), (1.4, 2.2, 0.5, 2.9)]),
    law(-0.25, [(1.1, -1.0, 0.8, 0.7), (0.6, -2.0, 1.2, -1.5)]),
])


def anchor(q):
    return (q * np.exp(1j * TH)).mean(axis=(-2, -1))  # u_i = int e^{i theta} q_i dnu


def omega(q):
    return np.imag(anchor(V(q)) / anchor(q))


I = [0, 1]


def slice_rep(q):
    """q~ = tau_a q with a_i = phi_I - theta_i on I, 0 on the singleton; phases lifted near 0."""
    th = np.angle(anchor(q))
    a = np.zeros(3)
    phiI = th[I].mean()
    for i in I:
        a[i] = phiI - th[i]
    return np.stack([rot(q[i], a[i]) for i in range(3)]), a


dt = 1e-3
qs = [q0.copy()]
for _ in range(1200):
    qs.append(rk4_step(qs[-1], dt))
qs = np.array(qs)

print("[1]-[2] moving representative (block {0,1}, singleton {2}):")
for n in (100, 400, 800, 1100):
    qt_mm, _ = slice_rep(qs[n - 2]); qt_m, _ = slice_rep(qs[n - 1]); qt_0, a0 = slice_rep(qs[n])
    qt_p, _ = slice_rep(qs[n + 1]); qt_pp, _ = slice_rep(qs[n + 2])
    lhs = (-qt_pp + 8 * qt_p - 8 * qt_m + qt_mm) / (12 * dt)  # fourth-order central difference
    om = omega(qt_0)
    ombar = om[I].mean()
    eta = np.array([ombar - om[0], ombar - om[1], 0.0])
    rhs = V(qt_0) - eta[:, None, None] * dth(qt_0)
    rhs0 = V(qt_0)
    res = np.abs(lhs - rhs).max(); res0 = np.abs(lhs - rhs0).max()
    ph_m = np.angle(anchor(qt_m)); ph_p = np.angle(anchor(qt_p))
    dph = (ph_p - ph_m) / (2 * dt)
    om_fine = omega(qs[n])
    print(f"  t={n*dt:.2f}: residual {res:.2e} (without moving term {res0:.2e}); omega(x~) - omega(x) = {np.abs(om - om_fine).max():.1e};"
          f" omega = {np.round(om, 4)}; d/dt arg u~_I = {np.round(dph[I], 6)} vs omegabar = {ombar:.6f}")

print("[3] orthogonality and dissipation:")
for n in (0, 600):
    q = qs[n]
    v = V(q)
    J = [-dth(q[i]) for i in range(3)]
    gVJ = [mean(v[i] * J[i] / q[i]) for i in range(3)]
    gVV = sum(mean(v[i] ** 2 / q[i]) for i in range(3))
    dE = (energy(qs[n + 1]) - energy(qs[n - 1])) / (2 * dt) if n > 0 else (energy(qs[1]) - energy(qs[0])) / dt
    om = omega(q); ombar = om[I].mean(); eta = [ombar - om[0], ombar - om[1], 0.0]
    gJJ = sum(eta[i] ** 2 * mean(J[i] ** 2 / q[i]) for i in range(3))
    print(f"  t={n*dt:.2f}: g(V, J_i) = {np.round(gVJ, 15)}; |V|_g^2 = {gVV:.6f}; dE/dt = {dE:.6f} (difference "
          f"{abs(dE + gVV):.1e}); g(J eta, J eta) = {gJJ:.3e}")

print("[4] D16-OPEN-06 anchor moment along the angular flow of an axially symmetric profile:")
l0 = -0.3 * np.cos(TH) + 2.5 * np.cos(2 * TH) + 1.2 * np.cos(3 * TH)


def u_of(rho):
    f = np.exp(rho * l0)
    return float((np.cos(TH) * f).mean() / f.mean())


lo, hi = 0.05, 1.0
assert u_of(lo) * u_of(hi) < 0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if u_of(mid) * u_of(lo) < 0:
        hi = mid
    else:
        lo = mid
rho_s = 0.5 * (lo + hi)
print(f"  u(1) = {u_of(1.0):.5f}, u(0.05) = {u_of(0.05):.5f}; root rho* = {rho_s:.6f}, t* = {-np.log(rho_s) / 2:.6f};"
      f" imag part identically 0 (even profile): {abs(float((np.sin(TH) * np.exp(0.7 * l0)).mean())):.1e}")

print("[5] open:open-laws covariance estimate:")


def Rrho(l, rho):
    f = np.exp(rho * l)
    p = f / f.mean()
    return float((p * np.log(p)).mean())


worst_txt, worst_sharp = 0.0, 0.0
for _ in range(300):
    c = rng.normal(0, 1, (2, 6))
    l1 = sum(c[0, k] * np.cos((k + 1) * TH + c[1, k]) for k in range(6))
    l2 = l1 + rng.normal(0, 0.3) * np.cos(TH + rng.uniform(0, 6.3)) + rng.normal(0, 0.3) * np.cos(3 * TH)
    l1n = l1 - np.log(np.exp(l1).mean()); l2n = l2 - np.log(np.exp(l2).mean())  # normalized log-conditionals
    Mosc = max(np.ptp(l1n), np.ptp(l2n))
    for rho in (1.0, 0.5, 0.1):
        d = abs(Rrho(l1n, rho) - Rrho(l2n, rho))
        worst_txt = max(worst_txt, d / (rho ** 2 * Mosc * np.abs(l1n - l2n).max()))
        worst_sharp = max(worst_sharp, d / (rho ** 2 * 0.5 * Mosc * np.ptp(l1n - l2n)))
print(f"  300 random profiles x rho in {{1, 0.5, 0.1}}: max ratio to stated bound rho^2 M ||dl||_inf = {worst_txt:.3f};"
      f" to sharper rho^2 (M/2) osc(dl) = {worst_sharp:.3f}")
