r"""GAP04-S4 s1: chapter 9 chart field (09:23-47).

Checks, on a finite uniform reference A (a stand-in for a general probability
space; every bound used below is independent of the grid size):

(a) the first variation of the Gibbs-reduced reciprocal potential with identity
    channels, all-other sources, uniform nonself priors,
        h_i^q = l_i - J phi m^s_i + sum_j beta*_ij (l_i - l_j) - sum_l beta*_li q_l/q_i + const,
    against complex-step directional derivatives of E_red in the chart
    u_i in L^infty_0(A), via  dE/de = sum_i Cov_{q_i}(h_i^q, du_i);
(b) the linearization DV(0)h = A h_i - kappa_N (h_i - hbar), kappa_N = 2N/(N-1);
(c) the diagonal field f(u,v) = (-u + J phi E_{s_v} psi, -v + J psi E_{q_u} phi);
(d) the explicit Lipschitz bound on the sup-norm ball B_R,
        L(R) = 2[6 + |J| + 16 R (2R+1)/T + 4 (N-1) e^{4R} (1 + (2R+1)/T)],
    against sampled difference quotients and sampled directional derivatives
    (sampling gives lower estimates of the true constant; the bound is proved in
    the report, not here).
Single-threaded numpy; runs in well under 30 s.
"""
import numpy as np

rng = np.random.default_rng(20261029)
M, N = 64, 3
J, Tq, Ts = 0.6, 0.7, 1.3
a = np.full(M, 1.0 / M)                     # reference A (and B), uniform weights
phi = np.where(np.arange(M) < M // 2, 1.0, -1.0)          # balanced sign on K
psi = np.where(np.arange(M) % 2 == 0, 1.0, -1.0)          # balanced sign on M-space
pi_ = 1.0 / (N - 1)


def laws(u):
    """u: (N, M) chart coordinates -> densities dq/dA, log densities."""
    Z = (a * np.exp(u)).sum(axis=1, keepdims=True)
    dens = np.exp(u) / Z
    return dens, u - np.log(Z)


def E_red(u, v):
    dq, lq = laws(u)
    ds, ls = laws(v)
    mq = (a * dq * phi).sum(axis=1)
    ms = (a * ds * psi).sum(axis=1)
    E = 0.0
    for i in range(N):
        E += (a * dq[i] * lq[i]).sum() + (a * ds[i] * ls[i]).sum() - J * mq[i] * ms[i] + np.log(np.cosh(J))
        Dq = np.array([(a * dq[i] * (lq[i] - lq[j])).sum() for j in range(N) if j != i])
        Ds = np.array([(a * ds[i] * (ls[i] - ls[j])).sum() for j in range(N) if j != i])
        E += -Tq * np.log((pi_ * np.exp(-Dq / Tq)).sum()) - Ts * np.log((pi_ * np.exp(-Ds / Ts)).sum())
    return E


def rows(dens, logd, T):
    D = np.zeros((N, N), dtype=np.result_type(dens, logd))   # complex-safe for complex-step
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (a * dens[i] * (logd[i] - logd[j])).sum()
    W = np.where(np.eye(N) > 0, 0.0, pi_ * np.exp(-D / T))
    return W / W.sum(axis=1, keepdims=True)


def forces(u, v):
    dq, lq = laws(u)
    ds, ls = laws(v)
    mq = (a * dq * phi).sum(axis=1)
    ms = (a * ds * psi).sum(axis=1)
    bq, bs = rows(dq, lq, Tq), rows(ds, ls, Ts)
    hq = np.zeros_like(u)
    hs = np.zeros_like(v)
    for i in range(N):
        hq[i] = lq[i] - J * phi * ms[i]
        hs[i] = ls[i] - J * psi * mq[i]
        for j in range(N):
            if j != i:
                hq[i] += bq[i, j] * (lq[i] - lq[j]) - bq[j, i] * dq[j] / dq[i]
                hs[i] += bs[i, j] * (ls[i] - ls[j]) - bs[j, i] * ds[j] / ds[i]
    return hq, hs


def field(u, v):
    hq, hs = forces(u, v)
    return -(hq - (a * hq).sum(axis=1, keepdims=True)), -(hs - (a * hs).sum(axis=1, keepdims=True))


def centered(x):
    return x - (a * x).sum(axis=-1, keepdims=True)


def rand_ball(R):
    x = centered(rng.normal(size=(N, M)))
    return R * x / np.abs(x).max()


def cov(dens, f, g):
    Ef = (a * dens * f).sum()
    return (a * dens * (f - Ef) * g).sum()


# (a) first variation by complex step
print("(a) first variation vs complex-step directional derivative of E_red")
worst = 0.0
for trial in range(5):
    R = [0.3, 0.8, 1.5, 0.5, 1.0][trial]
    u, v = rand_ball(R), rand_ball(R)
    du, dv = centered(rng.normal(size=(N, M))), centered(rng.normal(size=(N, M)))
    hstep = 1e-30
    cs = E_red(u + 1j * hstep * du, v + 1j * hstep * dv).imag / hstep
    hq, hs = forces(u, v)
    dq, _ = laws(u)
    ds, _ = laws(v)
    pred = sum(cov(dq[i], hq[i], du[i]) + cov(ds[i], hs[i], dv[i]) for i in range(N))
    rel = abs(cs - pred) / max(1.0, abs(cs))
    worst = max(worst, rel)
    print(f"  R={R:4.2f}  complex-step={cs:+.15f}  pairing={pred:+.15f}  rel.diff={rel:.2e}")
print(f"  worst relative residual: {worst:.2e}")

# (b) linearization at the origin
print("(b) DV(0) against A h_i - kappa_N (h_i - hbar)")
kappa = 2 * N / (N - 1)
u0 = np.zeros((N, M))
h, k = centered(rng.normal(size=(N, M))), centered(rng.normal(size=(N, M)))
hstep = 1e-30
Vq_c, Vs_c = field(u0 + 1j * hstep * h, u0 + 1j * hstep * k)
DVq, DVs = Vq_c.imag / hstep, Vs_c.imag / hstep
hbar, kbar = h.mean(axis=0), k.mean(axis=0)
predq = np.array([-h[i] + J * phi * (a * psi * k[i]).sum() - kappa * (h[i] - hbar) for i in range(N)])
preds = np.array([-k[i] + J * psi * (a * phi * h[i]).sum() - kappa * (k[i] - kbar) for i in range(N)])
print(f"  max |DV(0)h - prediction| = {max(np.abs(DVq - predq).max(), np.abs(DVs - preds).max()):.2e}")

# (c) diagonal field
print("(c) field on the diagonal against f(u,v)")
z_u, z_v = rand_ball(1.2)[0], rand_ball(1.2)[0]
Vq, Vs = field(np.tile(z_u, (N, 1)), np.tile(z_v, (N, 1)))
dq1, _ = laws(z_u[None, :])
ds1, _ = laws(z_v[None, :])
fu = -z_u + J * phi * (a * ds1[0] * psi).sum()
fv = -z_v + J * psi * (a * dq1[0] * phi).sum()
print(f"  max |V - f| on the diagonal = {max(np.abs(Vq - fu).max(), np.abs(Vs - fv).max()):.2e}")


# (d) Lipschitz bound on B_R
def L_bound(R):
    T = min(Tq, Ts)
    return 2 * (6 + abs(J) + 16 * R * (2 * R + 1) / T + 4 * (N - 1) * np.exp(4 * R) * (1 + (2 * R + 1) / T))


print("(d) sampled Lipschitz quotients on B_R (sup norm) against L(R)")
for R in [0.25, 0.5, 1.0, 2.0]:
    best_q = 0.0
    for _ in range(300):
        x_u, x_v = rand_ball(R), rand_ball(R)
        if rng.random() < 0.5:
            y_u, y_v = rand_ball(R), rand_ball(R)
        else:
            eps = 1e-3 * R
            y_u = centered(x_u + eps * rng.normal(size=(N, M)))
            y_v = centered(x_v + eps * rng.normal(size=(N, M)))
            sc = max(1.0, max(np.abs(y_u).max(), np.abs(y_v).max()) / R)
            y_u, y_v = y_u / sc, y_v / sc
        Vx = field(x_u, x_v)
        Vy = field(y_u, y_v)
        num = max(np.abs(Vx[0] - Vy[0]).max(), np.abs(Vx[1] - Vy[1]).max())
        den = max(np.abs(x_u - y_u).max(), np.abs(x_v - y_v).max())
        best_q = max(best_q, num / den)
    # directional derivatives by complex step, directions = signed indicator patterns
    best_d = 0.0
    for _ in range(200):
        x_u, x_v = rand_ball(R), rand_ball(R)
        d_u = centered(np.sign(rng.normal(size=(N, M))))
        d_v = centered(np.sign(rng.normal(size=(N, M))))
        nrm = max(np.abs(d_u).max(), np.abs(d_v).max())
        d_u, d_v = d_u / nrm, d_v / nrm
        Vq_c, Vs_c = field(x_u + 1j * 1e-30 * d_u, x_v + 1j * 1e-30 * d_v)
        best_d = max(best_d, max(np.abs(Vq_c.imag / 1e-30).max(), np.abs(Vs_c.imag / 1e-30).max()))
    print(f"  R={R:4.2f}  max sampled quotient={best_q:9.4f}  max sampled |DV d|={best_d:9.4f}  bound L(R)={L_bound(R):12.4f}"
          f"  ok={max(best_q, best_d) <= L_bound(R)}")
