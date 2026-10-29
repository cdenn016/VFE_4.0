r"""GAP02-S2 s3: constants of the R-step of open:routing (chapters/12_open_formation.tex:237-243) and of
open:persistence (12:129-146), and the chain A -> eps_b -> R -> h.

Limit field (open:limit): F_i(u) = -u_i + M_i + N_i,  M_i = sum_j w_ij x_ij,  N_i = sum_j w_ji x_ij,
x_ij = u_j - u_i, n_ij = exp(-x_ij^2/2), w_ij = n_ij / Z_i, Z_i = sum_k n_ik (self included).

Analytic constants derived in the report (n agents in a block, S = (n-1) e^{-1/2} bounds sum_j w_ij|x_ij|):
  Jacobian row sums  ||DF||_inf <= C_n = 1 + 2[min(1+2(n-1)/e, n-1) + (n-1)^2/e] + (n-1)[2 + 2(n-1)/e]
  block-row gradient |grad B_IJ|_1 <= G_n = 4 (n-1) e^{-1/2}   (also for contrasts)
  block-row Hessian  sum_{k,m}|d_k d_m B| <= L1_n = 8(n-1)/e + 16 S^2 + 4 + 4 min(1+2(n-1)/e, n-1)
  six-agent minus decoupled, cluster K (n), complement K' (n'), separation D >= 1, phi = e^{-D^2/2}:
     field          |Psi - Phi|_inf <= 8 (e^{-1/2} + D) phi                         (text, open:distant-bound)
     rows           |B^(6) - B^(dec)| <= n' phi (in-cluster rows), contrasts <= 2 n' phi
     row gradients  |grad(B^(6) - B^(dec))|_1 <= n' phi [8 S_K + 4 D + 3 n' D phi]
     far part only  <= n' D phi (2 + n' phi)
Checks: [1] Jacobian formula vs central differences; [2] sampled ratios to C_n, G_n, L1_n on random and
clustered configurations; [3] sampled ratios of the six-vs-decoupled differences to the bounds on tube-like
configurations with D in [3, 12]; [4] decoupled reference at A = 34 (b(t_A) certified <= 1 by s2):
actual ||DF||_inf and logarithmic norm mu_inf along it; [5] the written chain with the rigorous
constants: required D_R, R, and the h needed by the h-step (leading common-mode error 2/3 h^2 u^3).
Run: OMP_NUM_THREADS=1 ... timeout 30 python s3_rstep_constants.py
"""
from __future__ import annotations

import math

import numpy as np

rng = np.random.default_rng(20261029)
E = math.e
EH = math.exp(-0.5)


def rows(u: np.ndarray, blocks: list[list[int]] | None = None) -> np.ndarray:
    """Row matrix w[i, j]; with blocks, rows are normalized within the block of i (decoupled field)."""
    x = u[None, :] - u[:, None]
    n = np.exp(-0.5 * x ** 2)
    if blocks is not None:
        mask = np.zeros_like(n)
        for blk in blocks:
            for i in blk:
                mask[i, blk] = 1.0
        n = n * mask
    return n / n.sum(axis=1, keepdims=True)


def field(u: np.ndarray, blocks=None) -> np.ndarray:
    x = u[None, :] - u[:, None]
    w = rows(u, blocks)
    return -u + ((w + w.T) * x).sum(axis=1)


def jac(u: np.ndarray, blocks=None) -> np.ndarray:
    r"""Exact Jacobian of field(): d_m F_i = -delta + d_m M_i + d_m N_i (formulas in the report)."""
    N = len(u)
    x = u[None, :] - u[:, None]  # x[i, j] = u_j - u_i
    w = rows(u, blocks)
    Mv = (w * x).sum(axis=1)
    J = -np.eye(N)
    for i in range(N):
        for m in range(N):
            if m != i:
                dM = w[i, m] * ((1 - x[i, m] ** 2) + x[i, m] * Mv[i])
                J[i, m] += dM
                J[i, i] -= dM
    # N_i = - sum_j w_ji x_ji ; d_m(w_ji x_ji) = w_ji(1-x_ji^2)(d_im - d_jm) + w_ji x_ji (w_jm x_jm - d_jm M_j)
    for i in range(N):
        for j in range(N):
            if j == i or w[j, i] == 0.0:
                continue
            xji = x[j, i]
            for m in range(N):
                t = w[j, i] * (1 - xji ** 2) * ((m == i) - (m == j))
                t += w[j, i] * xji * (w[j, m] * x[j, m] - (m == j) * Mv[j])
                J[i, m] -= t
    return J


def grad_w(u: np.ndarray, blocks=None) -> np.ndarray:
    r"""g[i, j, m] = d w_ij / d u_m = w_ij [x_ij(d_im - d_jm) + w_im x_im - d_im M_i]."""
    N = len(u)
    x = u[None, :] - u[:, None]
    w = rows(u, blocks)
    Mv = (w * x).sum(axis=1)
    g = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            if w[i, j] == 0.0:
                continue
            for m in range(N):
                g[i, j, m] = w[i, j] * (x[i, j] * ((m == i) - (m == j)) + w[i, m] * x[i, m] - (m == i) * Mv[i])
    return g


def block_coef(I, J, N):
    c = np.zeros((N, N))
    for i in I:
        for j in J:
            c[i, j] = 1.0 / len(I)
    return c


def contrast_coef(I, J, others, N):
    c = block_coef(I, J, N)
    for K in others:
        c -= block_coef(I, K, N)
    return c


def obs(u, c, blocks=None):
    return float((c * rows(u, blocks)).sum())


def obs_grad(u, c, blocks=None):
    return np.einsum("ij,ijm->m", c, grad_w(u, blocks))


def C_n(n):
    return 1 + 2 * (min(1 + 2 * (n - 1) / E, n - 1) + (n - 1) ** 2 / E) + (n - 1) * (2 + 2 * (n - 1) / E)


def G_n(n):
    return 4 * (n - 1) * EH


def L1_n(n):
    S = (n - 1) * EH
    return 8 * (n - 1) / E + 16 * S * S + 4 + 4 * min(1 + 2 * (n - 1) / E, n - 1)


print("[0] analytic constants: C_2, C_4, C_6 =", [round(C_n(n), 3) for n in (2, 4, 6)],
      "; G_2, G_4, G_6 =", [round(G_n(n), 3) for n in (2, 4, 6)], "; L1_2, L1_4, L1_6 =", [round(L1_n(n), 2) for n in (2, 4, 6)])

# [1] Jacobian formula check
err = 0.0
for _ in range(30):
    u = rng.normal(0, 1.5, 6)
    J = jac(u)
    h = 1e-6
    Jfd = np.zeros((6, 6))
    for m in range(6):
        e = np.zeros(6); e[m] = h
        Jfd[:, m] = (field(u + e) - field(u - e)) / (2 * h)
    err = max(err, np.abs(J - Jfd).max())
gerr = 0.0
for _ in range(30):
    u = rng.normal(0, 1.5, 6)
    c = contrast_coef([0, 1], [2, 3], [[4, 5]], 6)
    gr = obs_grad(u, c)
    fd = np.array([(obs(u + h * np.eye(6)[m], c) - obs(u - h * np.eye(6)[m], c)) / (2 * h) for m in range(6)])
    gerr = max(gerr, np.abs(gr - fd).max())
print(f"[1] Jacobian formula vs central differences, 30 configs: max |err| = {err:.2e}; block-contrast gradient: {gerr:.2e}")


# [2] sampled ratios to the global bounds
def sample_configs(N, k):
    out = []
    for t in range(k):
        mode = t % 3
        if mode == 0:
            out.append(rng.normal(0, rng.uniform(0.3, 3), N))
        elif mode == 1:
            out.append(np.sort(rng.uniform(-4, 4, N)))
        else:  # adversarial spacing near |x| ~ 1.4 where |1-x^2| e^{-x^2/2} terms peak
            out.append(np.cumsum(rng.uniform(0.8, 2.0, N)))
    return out


for N in (2, 4, 6):
    rj, rg, rh = 0.0, 0.0, 0.0
    for u in sample_configs(N, 600):
        rj = max(rj, np.abs(jac(u)).sum(axis=1).max() / C_n(N))
        I = [0]; Jb = [1]; others = [[k] for k in range(2, N)]
        c = contrast_coef(I, Jb, others, N)
        rg = max(rg, np.abs(obs_grad(u, c)).sum() / G_n(N))
        hh = 1e-5
        H = np.zeros((N, N))
        for m in range(N):
            em = np.zeros(N); em[m] = hh
            H[:, m] = (obs_grad(u + em, c) - obs_grad(u - em, c)) / (2 * hh)
        rh = max(rh, np.abs(H).sum() / L1_n(N))
    print(f"[2] N={N}: sampled max ||DF||_inf / C_N = {rj:.3f}; |grad contrast|_1 / G_N = {rg:.3f}; sum|Hess| / L1_N = {rh:.3f}")


# [3] six-agent vs decoupled on tube-like configurations
def tube_config(D):
    a = rng.uniform(1.0, 6.0); b = rng.uniform(0.0, 2.0)
    core = np.array([-a - b, -a + b, a - b, a + b]) + rng.uniform(-1, 1, 4)
    bc = rng.uniform(0.0, 2.0)
    U = core.max() + D + bc
    far = np.array([U - bc, U + bc]) + rng.uniform(-0.2, 0.2, 2)
    u = np.concatenate([core, far])
    dmin = np.abs(u[4:, None] - u[None, :4]).min()
    return u, dmin


blocks = [[0, 1, 2, 3], [4, 5]]
worst = dict(field=0, row=0, con=0, g_core=0, g_far_only=0, g_pair=0)
for _ in range(1500):
    u, D = tube_config(rng.uniform(2.5, 7.0))  # float64: keep bounds far above the roundoff of differences
    if D < 1:
        continue
    ph = math.exp(-D * D / 2)
    worst["field"] = max(worst["field"], np.abs(field(u) - field(u, blocks)).max() / (8 * (EH + D) * ph))
    cB = block_coef([0, 1], [2, 3], 6)
    worst["row"] = max(worst["row"], abs(obs(u, cB) - obs(u, cB, blocks)) / (2 * ph))
    cC6 = contrast_coef([0, 1], [2, 3], [[4, 5]], 6)
    cC4 = contrast_coef([0, 1], [2, 3], [], 6)
    worst["con"] = max(worst["con"], abs(obs(u, cC6) - obs(u, cC4, blocks)) / (4 * ph))
    dg = obs_grad(u, cC6) - obs_grad(u, cC4, blocks)
    Sk = 3 * EH
    worst["g_core"] = max(worst["g_core"], np.abs(dg).sum() / (2 * ph * (8 * Sk + 4 * D + 6 * D * ph)))
    worst["g_far_only"] = max(worst["g_far_only"], np.abs(dg[4:]).sum() / (2 * D * ph * (2 + 2 * ph)))
    cP6 = contrast_coef([4], [5], [[0], [1], [2], [3]], 6)
    cP2 = contrast_coef([4], [5], [], 6)
    dgp = obs_grad(u, cP6) - obs_grad(u, cP2, blocks)
    worst["g_pair"] = max(worst["g_pair"], np.abs(dgp).sum() / (4 * ph * (8 * EH + 4 * D + 12 * D * ph)))
print("[3] float64, 1500 tube-like configurations, D in [2.5,7]: max sampled ratio to bound",
      {k: round(float(v), 4) for k, v in worst.items()})

# [3b] the same ratios at 40 digits for D in [3, 12] (no cancellation)
import mpmath as mpm
mpm.mp.dps = 40


def mp_rows(u, blocks=None):
    N = len(u)
    W = [[mpm.mpf(0)] * N for _ in range(N)]
    for i in range(N):
        allowed = range(N) if blocks is None else next(b for b in blocks if i in b)
        n = {j: mpm.exp(-(u[j] - u[i]) ** 2 / 2) for j in allowed}
        Z = mpm.fsum(n.values())
        for j in allowed:
            W[i][j] = n[j] / Z
    return W


def mp_field(u, blocks=None):
    W = mp_rows(u, blocks)
    N = len(u)
    return [-u[i] + mpm.fsum((W[i][j] + W[j][i]) * (u[j] - u[i]) for j in range(N)) for i in range(N)]


def mp_grad_obs(u, c, blocks=None):
    W = mp_rows(u, blocks)
    N = len(u)
    Mv = [mpm.fsum(W[i][k] * (u[k] - u[i]) for k in range(N)) for i in range(N)]
    g = [mpm.mpf(0)] * N
    for i in range(N):
        for j in range(N):
            if c[i][j] == 0 or W[i][j] == 0:
                continue
            x_ij = u[j] - u[i]
            for m in range(N):
                g[m] += c[i][j] * W[i][j] * (x_ij * ((m == i) - (m == j)) + W[i][m] * (u[m] - u[i]) - (m == i) * Mv[i])
    return g


def mp_obs(u, c, blocks=None):
    W = mp_rows(u, blocks)
    return mpm.fsum(c[i][j] * W[i][j] for i in range(len(u)) for j in range(len(u)))


def to_list(cmat):
    return [[mpm.mpf(float(v)) for v in row] for row in cmat]


cB = to_list(block_coef([0, 1], [2, 3], 6))
cC6 = to_list(contrast_coef([0, 1], [2, 3], [[4, 5]], 6)); cC4 = to_list(contrast_coef([0, 1], [2, 3], [], 6))
cP6 = to_list(contrast_coef([4], [5], [[0], [1], [2], [3]], 6)); cP2 = to_list(contrast_coef([4], [5], [], 6))
w2 = dict(field=0, row=0, con=0, g_core=0, g_far_only=0, g_pair=0)
for _ in range(150):
    uf, Df = tube_config(rng.uniform(3, 12))
    u = [mpm.mpf(float(v)) for v in uf]
    D = mpm.mpf(float(Df)); ph = mpm.exp(-D * D / 2)
    f6, fd = mp_field(u), mp_field(u, blocks)
    w2["field"] = max(w2["field"], float(max(abs(f6[i] - fd[i]) for i in range(6)) / (8 * (mpm.exp(-0.5) + D) * ph)))
    w2["row"] = max(w2["row"], float(abs(mp_obs(u, cB) - mp_obs(u, cB, blocks)) / (2 * ph)))
    w2["con"] = max(w2["con"], float(abs(mp_obs(u, cC6) - mp_obs(u, cC4, blocks)) / (4 * ph)))
    g6, g4 = mp_grad_obs(u, cC6), mp_grad_obs(u, cC4, blocks)
    dg = [g6[m] - g4[m] for m in range(6)]
    Sk = 3 * mpm.exp(-0.5)
    w2["g_core"] = max(w2["g_core"], float(mpm.fsum(abs(v) for v in dg) / (2 * ph * (8 * Sk + 4 * D + 6 * D * ph))))
    w2["g_far_only"] = max(w2["g_far_only"], float(mpm.fsum(abs(v) for v in dg[4:]) / (2 * D * ph * (2 + 2 * ph))))
    gp6, gp2 = mp_grad_obs(u, cP6), mp_grad_obs(u, cP2, blocks)
    w2["g_pair"] = max(w2["g_pair"], float(mpm.fsum(abs(gp6[m] - gp2[m]) for m in range(6)) / (4 * ph * (8 * mpm.exp(-0.5) + 4 * D + 12 * D * ph))))
print("[3b] 40 digits, 150 tube-like configurations, D in [3,12]: max sampled ratio to bound",
      {k: round(v, 4) for k, v in w2.items()})


# [4] decoupled reference at A = 34 and the logarithmic norm along it
def rk4(f, y, t1, dt):
    n = int(math.ceil(t1 / dt)); dt = t1 / n
    traj = [y.copy()]
    for _ in range(n):
        k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(y.copy())
    return np.array(traj), dt


A = 34.0
core0 = np.array([-A - 2, -A + 2, A - 2, A + 2])
pair0 = np.array([-2.0, 2.0])  # pair relative to its centre (centre is R e^{-t}, decoupled)
Lh = math.log(A / 4) + math.log(4 / math.sqrt(math.log(2)))  # t_A upper + exit-time upper bound
trc, dtc = rk4(lambda y: field(y), core0, Lh, 2e-3)
trp, _ = rk4(lambda y: field(y), pair0, Lh, 2e-3)
jn, mu = [], []
for yc, yp in zip(trc[::5], trp[::5]):
    Jc = jac(yc); Jp = jac(yp)
    jn.append(max(np.abs(Jc).sum(axis=1).max(), np.abs(Jp).sum(axis=1).max()))
    mu_c = max(Jc[i, i] + np.abs(np.delete(Jc[i], i)).sum() for i in range(4))
    mu_p = max(Jp[i, i] + np.abs(np.delete(Jp[i], i)).sum() for i in range(2))
    mu.append(max(mu_c, mu_p))
mu = np.array(mu)
print(f"[4] A=34 decoupled reference on [0, {Lh:.3f}]: max ||DPhi||_inf along it = {max(jn):.3f} (bound C_4 = {C_n(4):.2f});"
      f" max mu_inf = {mu.max():.3f}; int mu_inf dt = {np.trapezoid(mu, dx=5 * dtc):.3f}")
bt = (trc[:, 3] - trc[:, 2]) / 2
at = (trc[:, 3] + trc[:, 2]) / 2
iA = np.argmax(at <= 4.0)
print(f"    t_A = {iA * dtc:.4f}, b(t_A) = {bt[iA]:.4e} (s2 certificate needs b(t_A) <= 1)")


# [5] the written chain with rigorous constants
def chain(A, L, C, G, L1, m, s, rho=0.5):
    """Smallest D_R (grid 0.01) with Delta < rho, eta0 + G Delta < m and
    eta1 M + L1 Delta M + G (e + C Delta) < s, M = R + 3 + 10 e^{-1/2}, R = (D + A + 6) e^L."""
    for D in np.arange(3.0, 80.0, 0.01):
        ph = math.exp(-D * D / 2)
        e = 8 * (EH + D) * ph
        Delta = e * (math.exp(C * L) - 1) / C
        R = (D + A + 6) * math.exp(L)
        M = R + 3 + 10 * EH
        eta0 = 4 * ph
        eta1 = 2 * ph * (8 * 3 * EH + 4 * D + 6 * D * ph)
        if Delta < rho and eta0 + G * Delta < m and eta1 * M + L1 * Delta * M + G * (e + C * Delta) < s and 4 * ph < 1e-3:
            return D, R, Delta
    return None


for A in (34.0, 4096.0):
    L = math.log(A / 4) + math.log(4 / math.sqrt(math.log(2)))
    for C in (C_n(4), max(jn)):
        res = chain(A, L, C, G_n(4), L1_n(4), m=0.01, s=0.01)
        D, R, Delta = res
        # h-step: Gronwall with C_6 on the six-agent limit field; leading error (2/3) h^2 (R+3)^3
        C6 = C_n(6) if C == C_n(4) else C
        need = 0.5 * C6 / (math.expm1(C6 * L))  # field error budget for a radius-1/2 tube
        h2 = need / ((2.0 / 3.0) * (R + 3) ** 3)
        print(f"[5] A={A:g}, L={L:.4f}, C={C:.3f}: D_R={D:.2f}, R={R:.4e}, Delta={Delta:.2e};"
              f" h-step (C_6={C6:.2f}) needs T = h^2 < {h2:.2e} (leading order, before observable terms)")


# [6] last links of the chain: initial-data radius delta_0 (scaled units), its size in bit-mean units,
# and the open-law correlation budget M*eps < h*e_0 (open:open-laws), with the same rigorous constants.
for A in (34.0, 4096.0):
    L = math.log(A / 4) + math.log(4 / math.sqrt(math.log(2)))
    D, R, _ = chain(A, L, C_n(4), G_n(4), L1_n(4), m=0.01, s=0.01)
    C6, G6, L16 = C_n(6), G_n(6), L1_n(6)
    M = R + 3 + 10 * EH
    lnF = C6 * L  # log of the Gronwall factor e^{C L}
    log_h = 0.5 * (math.log(0.5 * C6) - lnF - math.log((2.0 / 3.0) * (R + 3) ** 3))
    # half of each margin to initial data, half to the forcing: Delta <= min(m/G, s/(L1 M + G C))/2
    budget = 0.5 * min(0.01 / G6, 0.01 / (L16 * M + G6 * C6))
    log_d0 = math.log(budget) - lnF
    log_e0 = math.log(budget * C6) - lnF
    print(f"[6] A={A:g}: log10 h = {log_h / math.log(10):.1f}; delta_0 (u units) ~ 10^{log_d0 / math.log(10):.1f};"
          f" bit-mean radius h*delta_0 ~ 10^{(log_h + log_d0) / math.log(10):.1f}; correlation budget M*eps < h*e_0 ~ 10^{(log_h + log_e0) / math.log(10):.1f}")
