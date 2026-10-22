r"""X3-PIFB check 3: friction without inertia, Nash versus potential fixed points, learned priors, memory.

(a) Two unit-variance Gaussian agents, mean sector, references a,b, mutual unit peer weights.
    2.0 potential: E = (mu1-a)^2/2 + (mu2-b)^2/2 + (mu1-mu2)^2  -> gap (a-b)/5.
    Receiver-only (each agent descends its own VFE with the peer law as prior) -> gap (a-b)/3.
(b) Three 1-D Gaussian agents (mean and variance) on a directed 3-cycle, fixed unit rows.
    Linearization of the 2.0 Fisher gradient flow at its equilibrium: real spectrum (gradient flow).
    Receiver-only field (no incoming force): complex spectrum (damped oscillation).
(c) Learned source priors at unit Fisher mobility: d pi/dt = T (beta* - pi) (replicator form).
    With an identity self channel (D_ii = 0) the self prior is nondecreasing.
(d) Linear symmetric (gradient) flow, orthogonal retained block: Mori-Zwanzig kernel
    K(t) = S_ab exp(-t S_bb) S_ba is a positive-semidefinite exponential mixture.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=5, suppress=True)

# (a)
a, b = 1.3, -0.4
H_pot = np.array([[1 + 2, -2], [-2, 1 + 2]], dtype=float)
mu_pot = np.linalg.solve(H_pot, np.array([a, b]))
# receiver-only fixed point: mu1 = (a + mu2)/2, mu2 = (b + mu1)/2
M = np.array([[2, -1], [-1, 2]], dtype=float)
mu_nash = np.linalg.solve(M, np.array([a, b]))
print("(a) potential gap/(a-b) =", (mu_pot[0] - mu_pot[1]) / (a - b), " receiver-only gap/(a-b) =",
      (mu_nash[0] - mu_nash[1]) / (a - b))

# (b) three 1-D Gaussians on a directed 3-cycle
refs = np.array([1.0, -0.5, 0.2])


def kl_g(m1: float, v1: float, m2: float, v2: float) -> float:
    return 0.5 * (v1 / v2 + (m1 - m2) ** 2 / v2 - 1.0 + np.log(v2 / v1))


def energy(th: np.ndarray) -> float:
    mu, lv = th[:3], th[3:]
    v = np.exp(lv)
    e = 0.0
    for i in range(3):
        e += kl_g(mu[i], v[i], refs[i], 1.0)
        j = (i + 1) % 3
        e += kl_g(mu[i], v[i], mu[j], v[j])
    return e


def own_energy(th: np.ndarray, i: int) -> float:
    mu, lv = th[:3], th[3:]
    v = np.exp(lv)
    j = (i + 1) % 3
    return kl_g(mu[i], v[i], refs[i], 1.0) + kl_g(mu[i], v[i], mu[j], v[j])


def grad(f, th: np.ndarray, h: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(th)
    for k in range(th.size):
        e = np.zeros_like(th)
        e[k] = h
        g[k] = (f(th + e) - f(th - e)) / (2 * h)
    return g


def fisher_inv(th: np.ndarray) -> np.ndarray:
    # coordinates (mu, log v): Fisher metric diag(1/v, 1/2)
    v = np.exp(th[3:])
    return np.concatenate([v, 2.0 * np.ones(3)])


def field_pot(th: np.ndarray) -> np.ndarray:
    return -fisher_inv(th) * grad(energy, th)


def field_recv(th: np.ndarray) -> np.ndarray:
    g = np.zeros_like(th)
    for i in range(3):
        gi = grad(lambda t: own_energy(t, i), th)
        g[i] = gi[i]
        g[3 + i] = gi[3 + i]
    return -fisher_inv(th) * g


def equilibrate(field, th0: np.ndarray, steps: int = 6000, dt: float = 0.02) -> np.ndarray:
    th = th0.copy()
    for _ in range(steps):
        th = th + dt * field(th)
    return th


def jac(field, th: np.ndarray, h: float = 1e-5) -> np.ndarray:
    J = np.zeros((th.size, th.size))
    for k in range(th.size):
        e = np.zeros_like(th)
        e[k] = h
        J[:, k] = (field(th + e) - field(th - e)) / (2 * h)
    return J


th0 = np.array([0.5, 0.0, -0.3, 0.0, 0.0, 0.0])
thp = equilibrate(field_pot, th0)
thr = equilibrate(field_recv, th0)
print("(b) |field| at potential equilibrium:", np.linalg.norm(field_pot(thp)),
      " at receiver-only equilibrium:", np.linalg.norm(field_recv(thr)))
ev_p = np.linalg.eigvals(jac(field_pot, thp))
ev_r = np.linalg.eigvals(jac(field_recv, thr))
print("    potential-flow eigenvalues:", np.sort_complex(ev_p))
print("    max |imag| (potential):", np.max(np.abs(ev_p.imag)))
print("    receiver-only eigenvalues:", np.sort_complex(ev_r))
print("    max |imag| (receiver-only):", np.max(np.abs(ev_r.imag)))

# (c) learned priors, two binary agents, identity peer and self channels, Gibbs-reduced rows
T = 0.5
aref = np.array([0.6, -0.6])


def d(x: float, z: float) -> float:
    return sum(((1 + k * x) / 2) * np.log((1 + k * x) / (1 + k * z)) for k in (-1, 1))


def d1(x: float, z: float) -> float:
    return np.arctanh(x) - np.arctanh(z)


def d2(x: float, z: float) -> float:
    return (z - x) / (1 - z * z)


x = np.array([0.3, -0.2])
pi_self = np.array([0.5, 0.5])  # prior on self; other source gets 1 - pi_self
hist = []
dt = 0.01
for step in range(20000):
    gx = np.zeros(2)
    bstar_self = np.zeros(2)
    for i in range(2):
        j = 1 - i
        Dij = d(x[i], x[j])
        Z = pi_self[i] + (1 - pi_self[i]) * np.exp(-Dij / T)
        bo = (1 - pi_self[i]) * np.exp(-Dij / T) / Z
        bstar_self[i] = pi_self[i] / Z
        gx[i] += d1(x[i], aref[i]) + bo * d1(x[i], x[j])
        gx[j] += bo * d2(x[i], x[j])
    dpi = T * (bstar_self - pi_self)
    hist.append(dpi.min())
    x = x - dt * (1 - x * x) * gx
    pi_self = pi_self + dt * dpi
print("(c) min over time of d(pi_self)/dt:", min(hist), " final pi_self:", pi_self, " final x:", x,
      " references:", aref)

# (d) Mori-Zwanzig kernel for a symmetric linear flow
rng = np.random.default_rng(3)
Bm = rng.normal(size=(6, 6))
S = Bm @ Bm.T + 0.5 * np.eye(6)
Saa, Sab, Sba, Sbb = S[:2, :2], S[:2, 2:], S[2:, :2], S[2:, 2:]
mins = []
for t in (0.0, 0.3, 1.0, 3.0):
    Kt = Sab @ expm(-t * Sbb) @ Sba
    mins.append(np.linalg.eigvalsh(0.5 * (Kt + Kt.T)).min())
print("(d) min eigenvalue of K(t) at t=0,0.3,1,3:", np.array(mins))
# verify the exact memory equation along one trajectory: integrate full system and compare a-dot
y0 = rng.normal(size=6)
ts = np.linspace(0, 2.0, 2001)
Y = np.array([expm(-t * S) @ y0 for t in ts])
A_ = Y[:, :2]
k = 1500
t = ts[k]
adot = -(S @ Y[k])[:2]
integrand = np.array([Sab @ expm(-(t - u) * Sbb) @ Sba @ A_[m] for m, u in enumerate(ts[: k + 1])])
mem = np.trapezoid(integrand, ts[: k + 1], axis=0)
rhs = -Saa @ A_[k] - Sab @ expm(-t * Sbb) @ y0[2:] + mem
print("    Mori-Zwanzig residual |adot - rhs| (trapezoid, h=1e-3):", np.linalg.norm(adot - rhs))
