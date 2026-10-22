"""Ch.11 form:full-flow, form:selection example, form:geometry inertias.

(1) Natural (Fisher) gradient of the closed-form Gibbs-reduced Wishart energy form:energy,
    with separate (alpha_i, P_i) per agent and the moment-dependent channel differentiated,
    compared with form:ode.  (2) Event times of the stated example.  (3) Orbit inertia k1.
"""
import numpy as np
from scipy.special import gammaln, digamma, polygamma

N, a0, kap = 4, 1.0, 1.0
def lgam2(v): return 0.5 * np.log(np.pi) + gammaln(v) + gammaln(v - 0.5)
def psi2(v): return digamma(v) + digamma(v - 0.5)
def tri2(v): return polygamma(1, v) + polygamma(1, v - 0.5)
def sym(p): return np.array([[p[0], p[1]], [p[1], p[2]]])
def neg_entropy(al, P):   # int q log q dX
    ld = np.linalg.slogdet(P)[1]
    return al * ld - lgam2(al) + (al - 1.5) * (psi2(al) - ld) - 2 * al
def cross(al, P, nu, L):  # -int q log W(nu, L)
    ld = np.linalg.slogdet(P)[1]
    EX = al * np.linalg.inv(P)
    return -(nu * np.linalg.slogdet(L)[1] - lgam2(nu) + (nu - 1.5) * (psi2(al) - ld) - np.trace(L @ EX))
def energy(theta):
    al = theta[:, 0]; Ps = [sym(t[1:]) for t in theta]
    M = sum(a * np.linalg.inv(P) for a, P in zip(al, Ps))
    Lam = np.eye(2) + kap / N * M
    return sum(2 * neg_entropy(a, P) + cross(a, P, a0, np.eye(2)) + cross(a, P, a0, Lam) for a, P in zip(al, Ps))
def fisher(al, P):  # metric in (alpha, p11, p12, p22) with U basis E11, E12+E21, E22
    Pi = np.linalg.inv(P)
    Us = [np.array([[1., 0], [0, 0]]), np.array([[0., 1], [1, 0]]), np.array([[0., 0], [0, 1]])]
    g = np.zeros((4, 4)); g[0, 0] = tri2(al)
    for i, U in enumerate(Us):
        g[0, i + 1] = g[i + 1, 0] = -np.trace(Pi @ U)
        for j, V in enumerate(Us):
            g[i + 1, j + 1] = al * np.trace(Pi @ U @ Pi @ V)
    return g
rng = np.random.default_rng(3)
theta = np.zeros((N, 4))
theta[:, 0] = [1.3, 1.7, 2.2, 0.9]
for i in range(N):
    B = rng.standard_normal((2, 2)); P = B @ B.T + 0.5 * np.eye(2)
    theta[i, 1:] = [P[0, 0], P[0, 1], P[1, 1]]
grad = np.zeros_like(theta); h = 1e-6
for i in range(N):
    for c in range(4):
        tp, tm = theta.copy(), theta.copy(); tp[i, c] += h; tm[i, c] -= h
        grad[i, c] = (energy(tp) - energy(tm)) / (2 * h)
al = theta[:, 0]; Ps = [sym(t[1:]) for t in theta]
M = sum(a * np.linalg.inv(P) for a, P in zip(al, Ps)); Lam = np.eye(2) + kap / N * M
C = np.eye(2) + 2 * kap / N * M - kap * a0 * np.linalg.inv(Lam)
worst = 0.0
for i in range(N):
    vel = -np.linalg.solve(fisher(al[i], Ps[i]), grad[i])
    claim_P = -2 * Ps[i] + np.eye(2) + C
    claim = np.array([-2 * (al[i] - a0), claim_P[0, 0], claim_P[0, 1], claim_P[1, 1]])
    worst = max(worst, np.abs(vel - claim).max())
print('max |natural-gradient velocity - form:ode| (separate precisions, FD grad):', worst)
# same with the channel derivative frozen (Lambda held at its value): show it matters
def energy_frozen(theta, L):
    al = theta[:, 0]; Ps = [sym(t[1:]) for t in theta]
    return sum(2 * neg_entropy(a, P) + cross(a, P, a0, np.eye(2)) + cross(a, P, a0, L) for a, P in zip(al, Ps))
gf = np.zeros_like(theta)
for i in range(N):
    for c in range(4):
        tp, tm = theta.copy(), theta.copy(); tp[i, c] += h; tm[i, c] -= h
        gf[i, c] = (energy_frozen(tp, Lam) - energy_frozen(tm, Lam)) / (2 * h)
diff = max(np.abs(np.linalg.solve(fisher(al[i], Ps[i]), grad[i] - gf[i])).max() for i in range(N))
print('velocity change if the moment-channel derivative is frozen:', diff)
# (2) example event times
a = np.array([1, 6 / 5, 9 / 5, 21 / 10]); d = np.diff(a); delta = 0.1
print('t_L, t_R, t2 =', 0.5 * np.log(d[0] / delta), 0.5 * np.log(d[2] / delta),
      0.5 * np.log((d[1] + (d[0] + d[2]) / 2) / delta), ' claimed', 0.5 * np.log(2), 0.5 * np.log(3), 0.5 * np.log(8.5))
# (3) orbit inertia of so(2) at P = diag(p+, p-) with shape A
pp, pm, AI = 3.0, 0.7, 2.5
Pi = np.diag([1 / pp, 1 / pm]); xi = np.array([[0., -1], [1, 0]]); U = xi @ np.diag([pp, pm]) - np.diag([pp, pm]) @ xi
print('orbit inertia numeric', AI * np.trace(Pi @ U @ Pi @ U), ' claimed k1', 2 * AI * (pp - pm)**2 / (pp * pm))
