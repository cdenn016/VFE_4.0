r"""G1 s2: constants of open:distant-bound and the moving-coordinate comparison (12:209-231).

Notation: x_ij = u_j - u_i, n_ij = exp(-x_ij^2/2), Z_i = sum_k n_ik (self included),
w_ij = n_ij / Z_i, F_i = -u_i + sum_j (w_ij + w_ji) x_ij  (open:limit).
Decoupled field: weights restricted to the cluster of i (core {0,1,2,3}, pair {4,5}).
phi(D) = exp(-D^2/2), psi(D) = (1 + D^2) phi(D), D = min core-pair separation.

Analytic bounds being tested (derived in the G1 report):
  C0:  |F - F_dec|_inf             <= 8 (s1 + D) phi(D)            (s1 = e^{-1/2})
  C1:  ||D(F - F_dec)||_inf (rows) <= max(120.3 + 38.8 D + 8 D^2, 79.1 + 58.2 D + 16 D^2) phi(D)
       hence <= 40 psi(D) for D >= 3
  OBS: |B6 - B4| <= 2 phi(D),  ||grad(B6 - B4)||_1 <= (56 s1 + 8 D + 16 D phi) phi(D)
  LIP: ||DF||_inf <= 1 + 27.75 + 35.75 = 64.5 for every configuration (absolute)
  MOV: in moving coordinates Y = X - R e^{-t} e_P, dG/dt = -R e^{-t} (DF e_P + e_P), and
       DF e_P + e_P involves only cross-cluster derivatives.

mpmath at 50 digits with central differences (step 1e-16) for Jacobians of the small differences.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python s2_distant_bound.py
"""
from __future__ import annotations

import math
import random

import mpmath as mp
import numpy as np

mp.mp.dps = 50
S1 = math.exp(-0.5)
CORE = (0, 1, 2, 3)
PAIR = (4, 5)
CL = [0, 0, 0, 0, 1, 1]


def field(u: list, decoupled: bool) -> list:
    n6 = len(u)
    W = [[mp.mpf(0)] * n6 for _ in range(n6)]
    for i in range(n6):
        row = []
        for k in range(n6):
            if decoupled and CL[k] != CL[i]:
                row.append(mp.mpf(0))
            else:
                row.append(mp.e ** (-(u[k] - u[i]) ** 2 / 2))
        Z = mp.fsum(row)
        for k in range(n6):
            W[i][k] = row[k] / Z
    return [-u[i] + mp.fsum((W[i][j] + W[j][i]) * (u[j] - u[i]) for j in range(n6)) for i in range(n6)]


def block_row(u: list, decoupled: bool) -> mp.mpf:
    r"""B_{P2,P1} = (1/2) sum_{i in {2,3}} sum_{j in {0,1}} w_ij (core-parent inherited mass)."""
    tot = mp.mpf(0)
    for i in (2, 3):
        ks = CORE if decoupled else tuple(range(6))
        Z = mp.fsum(mp.e ** (-(u[k] - u[i]) ** 2 / 2) for k in ks)
        tot += mp.fsum(mp.e ** (-(u[j] - u[i]) ** 2 / 2) for j in (0, 1)) / Z
    return tot / 2


def jac(fun, u: list, step: mp.mpf) -> list:
    cols = []
    for m in range(len(u)):
        up = list(u)
        um = list(u)
        up[m] += step
        um[m] -= step
        fp, fm = fun(up), fun(um)
        if isinstance(fp, list):
            cols.append([(a - b) / (2 * step) for a, b in zip(fp, fm)])
        else:
            cols.append((fp - fm) / (2 * step))
    return cols  # cols[m] = d fun / d u_m


random.seed(7)
step = mp.mpf("1e-16")
max_r0 = max_r1 = max_rB0 = max_rB1 = max_rmov = 0.0
worst = {}
ncfg = 0
for A in (8.0, 100.0, 4100.0):
    for D_target in (3.0, 4.0, 6.0, 8.0, 10.0):
        for rep in range(8):
            if rep % 2 == 0:  # generic core inside [-(A+3), A+3]
                core = sorted(random.uniform(-(A + 3), A + 3) for _ in range(4))
            else:  # reflection-type core with small within-pair gaps, as in the proof
                a, b = random.uniform(1, A), random.uniform(0, 2)
                core = [-a - b, -a + b, a - b, a + b]
            g = random.uniform(0, 6)
            p0 = max(core) + D_target + random.uniform(0, 0.5)
            u = [mp.mpf(c) for c in core] + [mp.mpf(p0), mp.mpf(p0 + g)]
            D = min(abs(float(u[j] - u[i])) for i in CORE for j in PAIR)
            phi = math.exp(-D * D / 2)
            psi = (1 + D * D) * phi
            dF = [a_ - b_ for a_, b_ in zip(field(u, False), field(u, True))]
            r0 = float(max(abs(x) for x in dF)) / (8 * (S1 + D) * phi)
            Jf = jac(lambda v: field(v, False), u, step)
            Jd = jac(lambda v: field(v, True), u, step)
            rows = [sum(abs(float(Jf[m][i] - Jd[m][i])) for m in range(6)) for i in range(6)]
            bound_core = (120.3 + 38.8 * D + 8 * D * D) * phi
            bound_pair = (79.1 + 58.2 * D + 16 * D * D) * phi
            r1 = max(max(rows[i] for i in CORE) / bound_core, max(rows[i] for i in PAIR) / bound_pair)
            dB = float(abs(block_row(u, False) - block_row(u, True)))
            rB0 = dB / (2 * phi)
            gB = jac(lambda v: block_row(v, False) - block_row(v, True), u, step)
            rB1 = sum(abs(float(x)) for x in gB) / ((56 * S1 + 8 * D + 16 * D * phi) * phi)
            # moving-coordinate time derivative structure: DF e_P + e_P
            mov = [Jf[4][i] + Jf[5][i] + (1 if i in PAIR else 0) for i in range(6)]
            rmov = float(max(abs(x) for x in mov)) / (40 * psi)
            for key, val in (("C0", r0), ("C1", r1), ("B0", rB0), ("B1", rB1), ("MOV", rmov)):
                if val > worst.get(key, (0, None))[0]:
                    worst[key] = (val, (A, round(D, 3)))
            ncfg += 1

print(f"configurations: {ncfg} (A in 8,100,4100; D in [3, 10.5])")
for key in ("C0", "C1", "B0", "B1", "MOV"):
    v, where = worst[key]
    print(f"  max ratio observed / analytic bound for {key}: {v:.4f} at (A, D) = {where}  (<= 1 required)")

# 40 psi(D) dominates the C1 bound for D >= 3
Ds = np.linspace(3, 60, 5000)
ratio = np.maximum(120.3 + 38.8 * Ds + 8 * Ds**2, 79.1 + 58.2 * Ds + 16 * Ds**2) / (1 + Ds**2)
print(f"  sup_(D>=3) C1-polynomial / (1 + D^2) = {ratio.max():.3f} (so C1 constant 40 suffices for D >= 3)")
Ds1 = np.linspace(1, 60, 5000)
ratio1 = np.maximum(120.3 + 38.8 * Ds1 + 8 * Ds1**2, 79.1 + 58.2 * Ds1 + 16 * Ds1**2) / (1 + Ds1**2)
print(f"  sup_(D>=1) C1-polynomial / (1 + D^2) = {ratio1.max():.3f}")


# ---------------------------------------------------------------- global Lipschitz of open:limit
def field_np(u: np.ndarray) -> np.ndarray:
    x = u[None, :] - u[:, None]
    nn = np.exp(-0.5 * x**2)
    w = nn / nn.sum(axis=1, keepdims=True)
    return -u + ((w + w.T) * x).sum(axis=1)


rng = np.random.default_rng(3)
lipmax = 0.0
for _ in range(3000):
    scale = rng.choice([0.3, 1.0, 2.0, 5.0, 50.0])
    u = rng.normal(0, scale, 6)
    J = np.zeros((6, 6))
    for m in range(6):
        e = np.zeros(6)
        e[m] = 1e-6
        J[:, m] = (field_np(u + e) - field_np(u - e)) / 2e-6
    lipmax = max(lipmax, np.abs(J).sum(axis=1).max())
print(f"LIP: max ||DF||_inf over 3000 random 6-agent configurations = {lipmax:.3f} (analytic bound 64.5)")

# ---------------------------------------------------------------- D_R and the moving-frame inequality
A, L = 4096.0, 8.358
for D_req in (5.0, 12.0, 33.0):
    R = (D_req + A + 6) * math.exp(L)
    print(f"D_R = R e^-L - A - 6 = {D_req}: R = {R:.4e}")
# check R e^{-t} <= rho + A + 6 on the radius-one tube for random admissible points
viol = 0
for _ in range(20000):
    R = 1.7546e7
    t = rng.uniform(0, L)
    ui = rng.uniform(-(A + 3), A + 3)  # core within tube
    rj = rng.uniform(-3, 3)  # pair moving coordinate within tube
    rho = R * math.exp(-t) + rj - ui
    if not (R * math.exp(-t) * rho <= (rho + A + 6) * rho + 1e-9 * rho * rho) or rho < R * math.exp(-L) - A - 6 - 1e-9:
        viol += 1
print(f"moving-frame inequality R e^-t rho <= (rho+A+6) rho and rho >= D_R: violations {viol}/20000")

# ---------------------------------------------------------------- h-step: common-mode field error of a far agent
print("h-step: isolated far agent, exact scaled decay vs limit  (1/h)(1-h^2u^2)atanh(hu) - u")
for h in (1e-9, 1e-10, 1e-11):
    u0 = mp.mpf("1.7546e7")
    hh = mp.mpf(h)
    err = (1 - hh**2 * u0**2) * mp.atanh(hh * u0) / hh - u0
    k = 2 * h * h / 3
    # leading-order closed form of u' = -u + (2/3) h^2 u^3: deviation from R e^{-t}
    Rv = 1.7546e7
    tt = np.linspace(0, L, 20001)
    dev = Rv * np.exp(-tt) * ((1 - k * Rv * Rv * (1 - np.exp(-2 * tt))) ** -0.5 - 1)
    print(f"  h={h:.0e}: field error at u=R {float(err):+.4e} (~ -(2/3)h^2R^3 = {-(2/3)*h*h*Rv**3:+.4e});"
          f" max_t |u(t) - R e^-t| = {dev.max():.4e} at t={tt[dev.argmax()]:.3f}; at t=L {dev[-1]:.3e}")
h_tube = math.sqrt(0.5 / (0.1283 * 1.7546e7**3))
print(f"  radius-1/2 tube for the far pair alone needs h <~ {h_tube:.2e}, i.e. T <~ {h_tube**2:.2e}")
