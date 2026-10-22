r"""Check of the associative selective-channel construction recorded in
Research/docs/research/2026-10-21-boundary-closure.md (lines 60-126).

Finite X (6 points), reference A, bounded T: X -> R, N = 3 agents, belief sector
only (product evaluator), lambda(m, e) = e + kappa m, channel
  K_ij(x, dy) = eps delta_x(dy) + nu_{m,e}(dy) - eps q_j(dy),
all nonself sources, uniform priors.  The fine energy is coded generically
(channel built from the current laws), and its gradient is taken by central
finite differences, so the channel and source derivatives are included without
using the claimed cancellation.
"""
import itertools
import numpy as np

rng = np.random.default_rng(5)
K = 6; N = 3; T_row = 0.8; kappa = 0.7; e = 0.3; epsc = 0.02
A = rng.dirichlet(np.ones(K) * 3)
Tx = rng.uniform(-1, 1, size=K)          # bounded observable, R = 1

def nu_of(m):
    lam = e + kappa * m
    w = A * np.exp(lam * Tx)
    return w / w.sum(), lam

def energy(P, beta):
    m = np.mean([p @ Tx for p in P])
    nu, _ = nu_of(m)
    E = 0.0
    for i in range(N):
        E += (P[i] * np.log(P[i] / A)).sum()
        for j in range(N):
            if j == i: continue
            Kmat = epsc * np.eye(K) + (nu - epsc * P[j])[None, :]   # rows x, cols y
            assert np.all(Kmat >= 0)
            out = P[j] @ Kmat
            E += beta[i, j] * (P[i] * np.log(P[i] / out)).sum()
        off = [j for j in range(N) if j != i]
        E += T_row * sum(beta[i, j] * np.log(beta[i, j] * (N - 1)) for j in off)
    return E

def W_of(m):
    lam = e + kappa * m
    calA = np.log((A * np.exp(lam * Tx)).sum())
    return calA - lam * m

def c_of(m):
    lam = e + kappa * m
    nu, _ = nu_of(m)
    return kappa * (nu @ Tx - m) - lam     # (D_m lambda)^T (grad calA - m) - lambda

P = [rng.dirichlet(np.ones(K) * 4) for _ in range(N)]
beta = np.full((N, N), 1 / (N - 1)); np.fill_diagonal(beta, 0)
m = np.mean([p @ Tx for p in P])
print("== energy identity E = 2 sum KL(q_i||A) + N W(m,e) (rows at prior) ==")
E1 = energy(P, beta)
E2 = 2 * sum((p * np.log(p / A)).sum() for p in P) + N * W_of(m)
print("E =", E1, " formula =", E2, " residual =", abs(E1 - E2))

print("== q_j K_ij = nu exactly ==")
nu, _ = nu_of(m)
Kmat = epsc * np.eye(K) + (nu - epsc * P[1])[None, :]
print("max |q_j K - nu| =", np.abs(P[1] @ Kmat - nu).max(), " row sums:", np.abs(Kmat.sum(1) - 1).max())

print("== finite-difference gradient vs 2 log f_i + c T (centred) ==")
h = 1e-6; worst = 0
for i in range(N):
    g = np.zeros(K)
    for z in range(K):
        Pp = [p.copy() for p in P]; Pm = [p.copy() for p in P]
        Pp[i][z] += h; Pm[i][z] -= h
        g[z] = (energy(Pp, beta) - energy(Pm, beta)) / (2 * h)
    gc = g - P[i] @ g
    an = 2 * np.log(P[i] / A) + c_of(m) * Tx
    anc = an - P[i] @ an
    worst = max(worst, np.abs(gc - anc).max())
print("max centred residual =", worst)

print("== row gradient independent of matter laws ==")
gb = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j: continue
        bp = beta.copy(); bm = beta.copy(); bp[i, j] += h; bm[i, j] -= h
        gb[i, j] = (energy(P, bp) - energy(P, bm)) / (2 * h)
costs = np.array([[ (P[i] * np.log(P[i] / nu)).sum() if i != j else 0 for j in range(N)] for i in range(N)])
print("row costs equal within each row:", np.ptp(costs[~np.eye(N, dtype=bool)].reshape(N, N - 1), axis=1))

print("== explicit solution ell_i(t) = a ell_i^0 + b T - log Z_i, a = e^{-2t}, bdot = -2b - c ==")
def fine_rhs(P):
    m = np.mean([p @ Tx for p in P]); c = c_of(m)
    out = []
    for p in P:
        hh = 2 * np.log(p / A) + c * Tx
        out.append(-p * (hh - p @ hh))
    return out
def rk4(P, dt, steps):
    for _ in range(steps):
        k1 = fine_rhs(P); k2 = fine_rhs([p + dt / 2 * k for p, k in zip(P, k1)])
        k3 = fine_rhs([p + dt / 2 * k for p, k in zip(P, k2)]); k4 = fine_rhs([p + dt * k for p, k in zip(P, k3)])
        P = [p + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4) for p, a1, a2, a3, a4 in zip(P, k1, k2, k3, k4)]
    return P
dt, steps = 0.001, 1000
PT = rk4(P, dt, steps)
ell0 = [np.log(p / A) for p in P]
def densities(a, b):
    out = []
    for l in ell0:
        w = A * np.exp(a * l + b * Tx); out.append(w / w.sum())
    return out
b = 0.0; t = 0.0
for _ in range(steps):
    def bd(t, b):
        Pab = densities(np.exp(-2 * t), b); mm = np.mean([p @ Tx for p in Pab]); return -2 * b - c_of(mm)
    k1 = bd(t, b); k2 = bd(t + dt / 2, b + dt / 2 * k1); k3 = bd(t + dt / 2, b + dt / 2 * k2); k4 = bd(t + dt, b + dt * k3)
    b += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
Pex = densities(np.exp(-2 * t), b)
print("max |fine(t=1) - explicit| =", max(np.abs(p - q).max() for p, q in zip(PT, Pex)))

print("== block statistic P_I and convolution join ==")
def block_law(Ps):
    """law of (L, S) = (sum log f_i(X_i), sum T(X_i)), X_i ~ q_i independent."""
    law = {}
    for xs in itertools.product(range(K), repeat=len(Ps)):
        pr = np.prod([Ps[k][x] for k, x in enumerate(xs)])
        L = round(sum(np.log(Ps[k][x] / A[x]) for k, x in enumerate(xs)), 9)
        S = round(sum(Tx[x] for x in xs), 9)
        law[(L, S)] = law.get((L, S), 0) + pr
    return law
def convolve(l1, l2):
    out = {}
    for (L1, S1), p1 in l1.items():
        for (L2, S2), p2 in l2.items():
            key = (round(L1 + L2, 9), round(S1 + S2, 9)); out[key] = out.get(key, 0) + p1 * p2
    return out
lI = block_law([P[0]]); lJ = block_law([P[1], P[2]]); lIJ = block_law(P)
conv = convolve(lI, lJ)
# compare joint characteristic functions on a grid (avoids float-key rounding)
def cf(law, s, t):
    return sum(p * np.exp(1j * (s * L + t * S)) for (L, S), p in law.items())
grid = [(s, t) for s in (-1.3, 0.4, 2.1) for t in (-0.7, 0.9, 1.6)]
print("max |cf(P_{I u J}) - cf(P_I) cf(P_J)| =", max(abs(cf(lIJ, s, t) - cf(lI, s, t) * cf(lJ, s, t)) for s, t in grid))
print("max |cf(P_I * P_J) - cf(P_I) cf(P_J)| (convolution routine) =", max(abs(cf(conv, s, t) - cf(lI, s, t) * cf(lJ, s, t)) for s, t in grid))

print("== noninjectivity example: (ab, 1) vs (a, b) on A = A_U x A_V ==")
AU = np.array([0.3, 0.7]); AV = np.array([0.6, 0.4])
Agrid = np.outer(AU, AV).ravel()
tU = np.array([0.2, -0.5]); tV = np.array([0.9, 0.1])
Tgrid = (tU[:, None] + 0 * tV[None, :]).ravel(), (0 * tU[:, None] + tV[None, :]).ravel()
aU = np.array([1.4, 1.0]); aU = aU / (AU @ aU); bV = np.array([0.5, 1.75]); bV = bV / (AV @ bV)
f_ab = np.outer(aU, bV).ravel(); f_a = np.outer(aU, np.ones(2)).ravel(); f_b = np.outer(np.ones(2), bV).ravel()
one = np.ones(4)
def stat_law(fs):
    law = {}
    for xs in itertools.product(range(4), repeat=2):
        pr = np.prod([Agrid[x] * fs[k][x] for k, x in enumerate(xs)])
        L = round(sum(np.log(fs[k][x]) for k, x in enumerate(xs)), 9)
        S = (round(sum(Tgrid[0][x] for x in xs), 9), round(sum(Tgrid[1][x] for x in xs), 9))
        law[(L, S)] = law.get((L, S), 0) + pr
    return law
l1 = stat_law([f_ab, one]); l2 = stat_law([f_a, f_b])
keys = set(l1) | set(l2)
print("max |P_I(ab,1) - P_I(a,b)| =", max(abs(l1.get(k, 0) - l2.get(k, 0)) for k in keys))
print("child laws differ:", np.abs(Agrid * f_ab - Agrid * f_a).max() > 1e-6)
