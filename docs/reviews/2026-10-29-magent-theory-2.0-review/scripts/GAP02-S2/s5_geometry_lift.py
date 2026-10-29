r"""GAP02-S2 s5: the contextual geometry claim at chapters/12_open_formation.tex:477-489 (open:equivariant-lift
and the C^r-neighborhood sentence), checked at one reference and one perturbed correlated law (N = 2, one block).

Reduction used (derived in the report): for the tied common S^1 and a lifted section that is a pure belief
rotation in chi, A_chi = -a/(a+b), A_kappa is chi-independent, so F = d_kappa A_chi = -(b d_kappa a)/(a+b)^2,
where a(kappa, t) = sum_i int (d_theta q_i)^2/q_i dnu (belief rotation inertia at context kappa) and
b(t) = sum_i int (d_psi s_i)^2/s_i dnu (model sections are context independent at t = 0 and the model sector
evolves autonomously, so b does not depend on kappa). Reference: q_i = mu_i(z) p_kappa*(theta),
s_i = eta_i(w) p_lambda(psi); closed form F_t = -I(rho lam) rho I'(rho kap) / (I(rho kap) + I(rho lam))^2.
Checks: [1] reference F_t and h_chichi against the closed form at t = 0, 0.5, 1 (fine flow, RK4);
[2] perturbed correlated law (bit-angle correlation and second harmonics of size 0.05): F_t, h_chichi, anchor
moment, against the reference values; [3] the second-order jet: d_kappa d_theta q(t) from the flow vs the
finite-difference second variational structure (difference quotient convergence under delta -> delta/2).
Run: OMP_NUM_THREADS=1 ... timeout 30 python s5_geometry_lift.py
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0, i1, ive

K = 128
TH = 2 * np.pi * np.arange(K) / K
KF = np.fft.fftfreq(K, d=1.0 / K)
T = 0.5
KAP, LAM = 1.5, 1.0


def dth(f):
    return np.real(np.fft.ifft(1j * KF * np.fft.fft(f, axis=-1), axis=-1))


def mean(f):
    return f.mean(axis=(-2, -1))


def field(q):
    """open:full-field for one sector, N agents, q shape (N, 2, K)."""
    N = q.shape[0]
    mu = q.mean(axis=-1)
    D = np.array([[0.5 * np.sum(mu[i] * np.log(mu[i] / mu[j])) for j in range(N)] for i in range(N)])
    W = np.exp(-D / T); b = W / W.sum(axis=1, keepdims=True)
    out = np.empty_like(q)
    for i in range(N):
        h = 2 * np.log(q[i]) - sum(b[i, j] * np.log(mu[j])[:, None] for j in range(N)) \
            - sum(b[k, i] * (mu[k] / mu[i])[:, None] for k in range(N))
        out[i] = -q[i] * (h - mean(q[i] * h))
    return out


def evolve(q, t1, dt=2e-3):
    n = int(round(t1 / dt))
    for _ in range(n):
        k1 = field(q); k2 = field(q + dt / 2 * k1); k3 = field(q + dt / 2 * k2); k4 = field(q + dt * k3)
        q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return q


def inertia(q):
    return sum(mean(dth(q[i]) ** 2 / q[i]) for i in range(q.shape[0]))


def lift(q, kap):
    """open:equivariant-lift at chi = 0: tilt by exp((kap - kap*) cos(theta - vartheta_i)) and normalize."""
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        vt = np.angle((q[i] * np.exp(1j * TH)).mean())
        f = np.exp((kap - KAP) * np.cos(TH - vt)) * q[i]
        out[i] = f / mean(f)
    return out


def vm(k):
    return np.exp(k * np.cos(TH)) / i0(k)


def Ifun(v):
    return v * i1(v) / i0(v)


def Iprime(v, h=1e-6):
    return (Ifun(v + h) - Ifun(v - h)) / (2 * h)


def product(mu1, ang):
    return np.stack([np.array([1 + m, 1 - m])[:, None] * ang[None, :] for m in mu1])


q_ref = product([0.3, -0.2], vm(KAP))
s_ref = product([0.1, 0.25], vm(LAM))
rng = np.random.default_rng(3)


def perturb(q, size):
    out = q.copy()
    for i in range(q.shape[0]):
        for zi in range(2):
            pert = size * (rng.normal() * np.cos(TH + rng.uniform(0, 6.3)) + rng.normal() * np.cos(2 * TH + rng.uniform(0, 6.3)))
            out[i, zi] = q[i, zi] * np.exp(pert)
        out[i] = out[i] / mean(out[i])
    return out


q_pert = perturb(q_ref, 0.05)
s_pert = perturb(s_ref, 0.05)


def geometry(q0, s0, t, delta=1e-3):
    a_vals = []
    for kap in (KAP - delta, KAP, KAP + delta):
        a_vals.append(inertia(evolve(lift(q0, kap), t)))
    b = inertia(evolve(s0, t))
    a = a_vals[1]
    da = (a_vals[2] - a_vals[0]) / (2 * delta)
    F = -b * da / (a + b) ** 2
    hcc = a * b / (a + b)
    u = np.abs((evolve(lift(q0, KAP), t) * np.exp(1j * TH)).mean(axis=(-2, -1)))
    return F, hcc, u.min(), da


print("[1] reference product family (n = 2 block), F and h_chichi vs closed form:")
for t in (0.0, 0.5, 1.0):
    F, hcc, umin, _ = geometry(q_ref, s_ref, t)
    rho = np.exp(-2 * t)
    a1, b1 = Ifun(rho * KAP), Ifun(rho * LAM)
    Fc = -b1 * rho * Iprime(rho * KAP) / (a1 + b1) ** 2
    print(f"  t={t}: F = {F:.8f}, closed form {Fc:.8f} (diff {abs(F - Fc):.1e}); h_chichi = {hcc:.8f}, closed form 2ab/(a+b) = {2 * a1 * b1 / (a1 + b1):.8f}")
print("[2] perturbed correlated law (size 0.05) vs reference:")
for t in (0.0, 0.5, 1.0):
    Fp, hp, up, _ = geometry(q_pert, s_pert, t)
    Fr, hr, ur, _ = geometry(q_ref, s_ref, t)
    print(f"  t={t}: F {Fp:.6f} vs {Fr:.6f} (rel diff {abs(Fp - Fr) / abs(Fr):.3f}); h_chichi {hp:.6f} vs {hr:.6f};"
          f" min anchor |u| {up:.4f} vs {ur:.4f}")
print("[3] mixed jet d_kappa a at t = 1 under delta -> delta/2 (central-difference convergence):")
for d in (2e-2, 1e-2, 5e-3):
    print(f"  delta={d:.0e}: d_kappa a = {geometry(q_pert, s_pert, 1.0, delta=d)[3]:.10f}")
