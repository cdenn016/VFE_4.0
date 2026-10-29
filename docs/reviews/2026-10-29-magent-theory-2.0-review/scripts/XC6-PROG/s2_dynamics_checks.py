r"""XC6-PROG s2: checks of chapters/03_dynamics.tex results on small finite instances.

 (a) prop:dyn-global-flow: fixed-channel Fisher flow (rows dynamic, identity self channel
     included) on three agents with |K|=3, |M|=2; finite-difference check of the force,
     the uniform floor min(m(0), -C) on log entries, monotone E, decay of the dissipated
     variance.
 (b) prop:dyn-nonreciprocal: spectrum of the three-cycle linear part.
 (c) prop:dyn-no-global-vfe (b): mixed derivative d_w d_v^2 of beta*KL(q_i||q_j) on a
     two-point alphabet versus beta v1^3 (q_j(1)^-2 - q_j(2)^-2); the same derivative of a
     mean-field free energy vanishes.
 (d) thm:dyn-registered (i)-(iii) by exact multinomial enumeration at n = 6.
 (e) prop:dyn-label-model identity on random finite alphabets.
 (f) prop:dyn-compensation, and the source-role force as maximum-likelihood deconvolution:
     the Fisher velocity of W KL(qbar || q_j K) is W q_j (K(qbar/nu) - 1), and the EM
     (Richardson-Lucy) step q_j <- q_j K(qbar/nu) decreases KL(qbar || q_j K).
 (g) Schroedinger identity: S(q_j,q_i) = min over couplings Q of (q_j,q_i) of
     KL(Q || q_j(dv)K(v,dw)) satisfies S >= KL(q_i||q_jK), with equality exactly when
     K(q_i/nu) = 1 on supp q_j (source-stationary point).  Sinkhorn computation.
"""
import itertools
import math

import numpy as np

rng = np.random.default_rng(2026)


def simplex(n: int, scale: float = 1.0) -> np.ndarray:
    v = np.exp(scale * rng.normal(size=n))
    return v / v.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


# (a) ---------------------------------------------------------------------------------
N, K, M = 3, 3, 2
Tq, Ts = 0.3, 0.5
r = [simplex(M) for _ in range(N)]
G = [np.array([simplex(K) for _ in range(M)]) for _ in range(N)]   # G[i][m, k]
Kq = {}
Ks = {}
for i in range(N):
    for j in range(N):
        if i == j:
            Kq[i, j] = np.eye(K)
            Ks[i, j] = np.eye(M)
        else:
            A = np.array([simplex(K, 1.5) for _ in range(K)])
            A[:, 0] = 1e-3 * A[:, 0]          # nearly null column: small c_min
            A /= A.sum(axis=1, keepdims=True)
            Kq[i, j] = A
            Ks[i, j] = np.array([simplex(M, 1.5) for _ in range(M)])
piq = [simplex(N) for _ in range(N)]
pis = [simplex(N) for _ in range(N)]


def unpack(u: np.ndarray):
    q = u[:N * K].reshape(N, K)
    s = u[N * K:N * K + N * M].reshape(N, M)
    b = u[N * K + N * M:N * K + N * M + N * N].reshape(N, N)
    g = u[N * K + N * M + N * N:].reshape(N, N)
    return q, s, b, g


def energy(u: np.ndarray) -> float:
    q, s, b, g = unpack(u)
    E = 0.0
    for i in range(N):
        joint = np.outer(s[i], q[i])                      # [m, k]
        ref = r[i][:, None] * G[i]
        E += np.sum(joint * np.log(joint / ref))
        for j in range(N):
            E += b[i, j] * kl(q[i], q[j] @ Kq[i, j]) + g[i, j] * kl(s[i], s[j] @ Ks[i, j])
        E += Tq * kl(b[i], piq[i]) + Ts * kl(g[i], pis[i])
    return E


def grad(u: np.ndarray) -> np.ndarray:
    q, s, b, g = unpack(u)
    hq, hs = np.zeros((N, K)), np.zeros((N, M))
    hb, hg = np.zeros((N, N)), np.zeros((N, N))
    for i in range(N):
        L = np.log(np.outer(s[i], q[i]) / (r[i][:, None] * G[i]))   # [m, k]
        hq[i] += s[i] @ L + 1.0
        hs[i] += L @ q[i] + 1.0
        for j in range(N):
            nu = q[j] @ Kq[i, j]
            hq[i] += b[i, j] * (np.log(q[i] / nu) + 1.0)
            hq[j] += -b[i, j] * (Kq[i, j] @ (q[i] / nu))
            nus = s[j] @ Ks[i, j]
            hs[i] += g[i, j] * (np.log(s[i] / nus) + 1.0)
            hs[j] += -g[i, j] * (Ks[i, j] @ (s[i] / nus))
            hb[i, j] = kl(q[i], nu) + Tq * (np.log(b[i, j] / piq[i][j]) + 1.0)
            hg[i, j] = kl(s[i], nus) + Ts * (np.log(g[i, j] / pis[i][j]) + 1.0)
    return np.concatenate([hq.ravel(), hs.ravel(), hb.ravel(), hg.ravel()])


blocks = [(a * K, (a + 1) * K) for a in range(N)]
blocks += [(N * K + a * M, N * K + (a + 1) * M) for a in range(N)]
off = N * K + N * M
blocks += [(off + a * N, off + (a + 1) * N) for a in range(N)]
off += N * N
blocks += [(off + a * N, off + (a + 1) * N) for a in range(N)]


def field(u: np.ndarray) -> np.ndarray:
    h = grad(u)
    v = np.zeros_like(u)
    for lo, hi in blocks:
        uu, hh = u[lo:hi], h[lo:hi]
        v[lo:hi] = -uu * (hh - np.dot(uu, hh))
    return v


def variance(u: np.ndarray) -> float:
    h = grad(u)
    tot = 0.0
    for lo, hi in blocks:
        uu, hh = u[lo:hi], h[lo:hi]
        tot += np.dot(uu, hh**2) - np.dot(uu, hh)**2
    return tot


u0 = np.concatenate([np.concatenate([simplex(K, 2.0) for _ in range(N)]),
                     np.concatenate([simplex(M, 2.0) for _ in range(N)]),
                     np.concatenate([simplex(N, 2.0) for _ in range(N)]),
                     np.concatenate([simplex(N, 2.0) for _ in range(N)])])
# finite-difference check of the force along tangent directions
fd_err = 0.0
for lo, hi in blocks[:3] + blocks[N:N + 1] + blocks[2 * N:2 * N + 1]:
    d = np.zeros_like(u0)
    w = rng.normal(size=hi - lo)
    d[lo:hi] = w - w.mean()
    eps = 1e-6
    num = (energy(u0 + eps * d) - energy(u0 - eps * d)) / (2 * eps)
    fd_err = max(fd_err, abs(num - grad(u0) @ d))
print(f"(a) force finite-difference error along tangent directions: {fd_err:.2e}")

cmin = min(Kq[i, j].sum(axis=0).min() for i in range(N) for j in range(N))
CG = max(-np.log(G[i]).max() for i in range(N))
C_bound = CG + max(0.0, np.log(1 / cmin)) + np.log(K) + N
m0 = np.log(u0[:N * K]).min()
floor = min(m0, -C_bound)


def normalize_log(th: np.ndarray) -> np.ndarray:
    out = th.copy()
    for lo, hi in blocks:
        mx = out[lo:hi].max()
        out[lo:hi] -= mx + np.log(np.exp(out[lo:hi] - mx).sum())
    return out


def log_field(th: np.ndarray) -> np.ndarray:
    uu = np.exp(th)
    h = grad(uu)
    v = np.zeros_like(th)
    for lo, hi in blocks:
        v[lo:hi] = -(h[lo:hi] - np.dot(uu[lo:hi], h[lo:hi]))
    return v


# Fisher flow integrated in log coordinates (d/dt log u = -(h - E_u h)), RK4, renormalized per table
dt, Tend = 1e-2, 40.0
th = normalize_log(np.log(u0))
Es, mins = [energy(np.exp(th))], [th[:N * K].min()]
for step in range(int(Tend / dt)):
    k1 = log_field(th)
    k2 = log_field(normalize_log(th + 0.5 * dt * k1))
    k3 = log_field(normalize_log(th + 0.5 * dt * k2))
    k4 = log_field(normalize_log(th + dt * k3))
    th = normalize_log(th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    if step % 50 == 0:
        Es.append(energy(np.exp(th)))
        mins.append(th[:N * K].min())
u = np.exp(th)
Es.append(energy(u))
mins.append(th[:N * K].min())
print(f"    c_min = {cmin:.3e}; C = {C_bound:.3f}; m(0) = {m0:.3f}; floor min(m(0),-C) = {floor:.3f}")
print(f"    min over sampled times of min log q = {min(mins):.3f} (>= floor: {min(mins) >= floor})")
print(f"    E monotone along samples: {bool(np.all(np.diff(Es) <= 1e-12))}; E(0) = {Es[0]:.5f}, E(T) = {Es[-1]:.5f}")
print(f"    dissipated variance at T = {variance(u):.3e}; |field| at T = {np.abs(field(u)).max():.3e}")

# (b) ---------------------------------------------------------------------------------
w = 0.7
P = np.roll(np.eye(3), 1, axis=1)          # (P l)_i = l_{i+1}
Amat = (-(1 + w) * np.eye(3) + w * P) / (1 + w)
ev = np.sort_complex(np.linalg.eigvals(Amat))
pred = np.sort_complex(np.array([-1 / (1 + w), (-1 - 1.5 * w + 1j * np.sqrt(3) * w / 2) / (1 + w),
                                 (-1 - 1.5 * w - 1j * np.sqrt(3) * w / 2) / (1 + w)]))
print(f"(b) three-cycle spectrum error: {np.abs(ev - pred).max():.2e}; eigenvalues {ev}")

# (c) ---------------------------------------------------------------------------------
beta, qi, qj, v1 = 0.37, np.array([0.3, 0.7]), np.array([0.2, 0.8]), 1.0
vv = np.array([v1, -v1])


def f(sa: float, ta: float) -> float:
    return beta * kl(qi + sa * vv, qj + ta * vv)


h = 1e-3
num = 0.0
for sa, cs in [(h, 1), (-h, -1)]:
    for ta, ct in [(h, 1), (0.0, -2), (-h, 1)]:
        num += cs * ct * f(sa, ta)
num /= (2 * h) * h**2
pred = beta * v1**3 * (qj[0]**-2 - qj[1]**-2)
# mean-field free energy: sum entropies + integral of a fixed function against qi x qj
Pfix = simplex(4).reshape(2, 2)


def fmf(sa: float, ta: float) -> float:
    a, b = qi + sa * vv, qj + ta * vv
    Q = np.outer(a, b)
    return float(np.sum(Q * np.log(Q / Pfix)))


nmf = 0.0
for sa, cs in [(h, 1), (-h, -1)]:
    for ta, ct in [(h, 1), (0.0, -2), (-h, 1)]:
        nmf += cs * ct * fmf(sa, ta)
nmf /= (2 * h) * h**2
print(f"(c) mixed derivative of beta KL: numeric {num:.5f}, formula {pred:.5f}; mean-field ELBO: {nmf:.2e}")

# (d) ---------------------------------------------------------------------------------
n = 6


def type_prob(p: np.ndarray, nu: np.ndarray) -> float:
    counts = np.rint(n * p).astype(int)
    multin = math.factorial(n) / np.prod([math.factorial(c) for c in counts])
    return float(multin * np.prod(nu ** counts))


def type_class(p: np.ndarray) -> float:
    counts = np.rint(n * p).astype(int)
    return math.factorial(n) / np.prod([math.factorial(c) for c in counts])


def H(p: np.ndarray) -> float:
    pp = p[p > 0]
    return float(-np.sum(pp * np.log(pp)))


qtype = np.array([3, 3]) / n
stype = np.array([2, 4]) / n
joint = np.outer(qtype, stype).ravel()           # (1/6, 2/6, 1/6, 2/6): a 6-type
nus = [simplex(2) for _ in range(3)]
err_i = max(abs(type_prob(qtype, nu) - type_class(qtype) * np.exp(-n * (H(qtype) + kl(qtype, nu))))
            for nu in nus)
pi3 = simplex(3)
T = 1.0 / n
post = np.array([pi3[j] * type_prob(qtype, nus[j]) for j in range(3)])
post /= post.sum()
gibbs = np.array([pi3[j] * np.exp(-kl(qtype, nus[j]) / T) for j in range(3)])
gibbs /= gibbs.sum()
# (iii) one receiver, one belief row, reference batch; recognition delta_x (x) beta
Rref = simplex(4)
beta_r = simplex(3)
# free energy contributions of the labels and the batches (configuration prior constant dropped)
negloglik = -np.log(type_prob(joint, Rref)) - sum(beta_r[j] * np.log(type_prob(qtype, nus[j]))
                                                  for j in range(3))
TF = T * (kl(beta_r, pi3) + negloglik)
E_here = kl(joint, Rref) + sum(beta_r[j] * kl(qtype, nus[j]) for j in range(3)) + T * kl(beta_r, pi3)
Cn = (H(joint) - np.log(type_class(joint)) / n) + (H(qtype) - np.log(type_class(qtype)) / n)
print(f"(d) registered agreement n=6: (i) error {err_i:.2e}; (ii) |posterior - Gibbs| {np.abs(post - gibbs).max():.2e}; "
      f"(iii) |T F - E - C_n| {abs(TF - E_here - Cn):.2e}")

# (e) ---------------------------------------------------------------------------------
NA, KA, Tl = 3, 3, 0.7
rr = [simplex(KA) for _ in range(NA)]
qq = [simplex(KA) for _ in range(NA)]
kern = {}
for i in range(NA):
    for j in range(NA):
        if i != j:
            kern[i, j] = np.array([simplex(KA) for _ in range(KA)])   # kern[i,j][v, w] = k(w|v)
pil = [None] * NA
bet = [None] * NA
for i in range(NA):
    p = np.array([simplex(1)[0] if j != i else 0.0 for j in range(NA)])
    p = np.where(np.arange(NA) != i, rng.uniform(0.2, 1.0, NA), 0.0)
    pil[i] = p / p.sum()
    b = np.where(np.arange(NA) != i, rng.uniform(0.2, 1.0, NA), 0.0)
    bet[i] = b / b.sum()
# free energy of the label model: T F = T sum KL(q_i||r_i) + T sum KL(beta_i||pi_i) + sum beta_ij E[-log k_ij]
TF_l = 0.0
for i in range(NA):
    TF_l += Tl * kl(qq[i], rr[i])
    m = bet[i] > 0
    TF_l += Tl * np.sum(bet[i][m] * np.log(bet[i][m] / pil[i][m]))
    for j in range(NA):
        if j != i:
            TF_l += bet[i][j] * np.sum(np.outer(qq[j], qq[i]) * -np.log(kern[i, j]))
Epot = 0.0
Gsum = 0.0
extra = 0.0
for i in range(NA):
    Epot += kl(qq[i], rr[i])
    m = bet[i] > 0
    Epot += Tl * np.sum(bet[i][m] * np.log(bet[i][m] / pil[i][m]))
    for j in range(NA):
        if j != i:
            nu = qq[j] @ kern[i, j]
            Epot += bet[i][j] * kl(qq[i], nu)
            # G_ij = int KL(q_j || q_j^w) q_i(dw), q_j^w(v) = k(w|v) q_j(v)/nu(w)
            Gij = 0.0
            for wv in range(KA):
                qjw = kern[i, j][:, wv] * qq[j] / nu[wv]
                Gij += qq[i][wv] * kl(qq[j], qjw)
            Gsum += bet[i][j] * Gij
    extra += H(qq[i]) + (Tl - 1) * kl(qq[i], rr[i])
print(f"(e) label-model identity residual: {TF_l - (Epot + extra + Gsum):.2e}")

# (f) ---------------------------------------------------------------------------------
KK = 4
Kj = np.array([simplex(KK) for _ in range(KK)])
recv = [simplex(KK) for _ in range(3)]
wts = rng.uniform(0.2, 1.0, 3)
W = wts.sum()
qbar = sum(wts[a] * recv[a] for a in range(3)) / W
qj = simplex(KK)
nu = qj @ Kj
lhs = sum(wts[a] * kl(recv[a], nu) for a in range(3))
rhs = sum(wts[a] * kl(recv[a], qbar) for a in range(3)) + W * kl(qbar, nu)
grad_src = -sum(wts[a] * (Kj @ (recv[a] / nu)) for a in range(3))
vel = -qj * (grad_src - qj @ grad_src)
vel_em = W * qj * (Kj @ (qbar / nu) - 1.0)
print(f"(f) compensation residual {lhs - rhs:.2e}; |Fisher velocity - W q_j (K(qbar/nu)-1)| {np.abs(vel - vel_em).max():.2e}; "
      f"E_q_j K(qbar/nu) - 1 = {qj @ (Kj @ (qbar / nu)) - 1:.2e}")
q_em = qj.copy()
vals = [kl(qbar, q_em @ Kj)]
for _ in range(200):
    q_em = q_em * (Kj @ (qbar / (q_em @ Kj)))
    vals.append(kl(qbar, q_em @ Kj))
print(f"    EM/Richardson-Lucy: KL(qbar||q_j K) {vals[0]:.6f} -> {vals[-1]:.6f}, monotone {bool(np.all(np.diff(vals) <= 1e-14))}")

# (g) ---------------------------------------------------------------------------------


def schrodinger(mu: np.ndarray, nu_target: np.ndarray, Pref: np.ndarray, iters: int = 20000) -> float:
    r"""min over couplings Q of (mu, nu_target) of KL(Q || Pref), by Sinkhorn scaling."""
    a = np.ones_like(mu)
    b = np.ones_like(nu_target)
    for _ in range(iters):
        a = mu / (Pref @ b)
        b = nu_target / (Pref.T @ a)
    Q = a[:, None] * Pref * b[None, :]
    return float(np.sum(Q * np.log(Q / Pref)))


qi_t = simplex(KK)
# generic source law
Pref = qj[:, None] * Kj
S_gen = schrodinger(qj, qi_t, Pref)
KL_gen = kl(qi_t, qj @ Kj)
# source-stationary point with KL > 0: a rank-two channel K = A B (non-injective), so q_i is off
# the image {q K} and the maximum-likelihood source law can be interior
Aleft = np.array([simplex(2) for _ in range(KK)])           # 4 x 2 stochastic
Bright = np.array([simplex(KK) for _ in range(2)])          # 2 x 4 stochastic
K2 = Aleft @ Bright
Kj_save = Kj
Kj = K2
# explicit interior source-stationary configuration: q_s interior, q_i = nu * (1 + y) with K y = 0
qs = simplex(KK)
nu_s = qs @ Kj
_, svals, Vt = np.linalg.svd(Kj)
y = Vt[-1]                                  # right null vector: K y = 0 (rank 2)
y = 0.5 * y / np.abs(y).max()
qi_s = nu_s * (1.0 + y)
qi_s = qi_s / qi_s.sum()                    # sum is already 1 up to rounding
Pref_s = qs[:, None] * Kj
S_fix = schrodinger(qs, qi_s, Pref_s)
KL_fix = kl(qi_s, nu_s)
print(f"(g) generic q_j: S = {S_gen:.6f} > KL = {KL_gen:.6f};  source-stationary interior point "
      f"(rank-2 channel, min q_j = {qs.min():.3f}, |K y| = {np.abs(Kj @ y).max():.1e}): "
      f"S = {S_fix:.10f}, KL = {KL_fix:.10f}, max|K(q_i/nu)-1| = {np.abs(Kj @ (qi_s / nu_s) - 1).max():.2e}")
Kj = Kj_save

# (a') floor test from a nearly degenerate initial table: m(0) = -20 < -C
u1 = u0.copy()
u1[0:K] = np.array([np.exp(-20.0), 0.5, 0.5 - np.exp(-20.0)])
th = normalize_log(np.log(u1))
ms = [th[:N * K].min()]
for step in range(int(10.0 / dt)):
    k1 = log_field(th)
    k2 = log_field(normalize_log(th + 0.5 * dt * k1))
    k3 = log_field(normalize_log(th + 0.5 * dt * k2))
    k4 = log_field(normalize_log(th + dt * k3))
    th = normalize_log(th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    ms.append(th[:N * K].min())
ms = np.array(ms)
print(f"(a') start m(0) = {ms[0]:.3f} below -C = {-C_bound:.3f}: min_t m(t) = {ms.min():.3f}, "
      f"m nondecreasing while below -C: {bool(np.all(np.diff(ms)[ms[:-1] < -C_bound] >= -1e-12))}, "
      f"m(10) = {ms[-1]:.3f}")
