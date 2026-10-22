"""R-FOUND checks for statements added to chapters 1, 3, 4 and the record appendix.

Single-threaded, small finite examples only.
"""
import itertools
import math

import numpy as np
import sympy as sp
from scipy.special import gammaln
from scipy.integrate import solve_ivp

rng = np.random.default_rng(0)


def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def H(p):
    p = np.asarray(p, float)
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


# 1. Per-agent Bernoulli 3-cycle: complex spectrum; summed potential: real.
print("1. directed 3-cycle")
w = sp.symbols("w", positive=True)
J = sp.zeros(3, 3)
for i in range(3):
    J[i, i] = -1 - w
    J[i, (i + 1) % 3] = w
ev = J.subs(w, 2).eigenvals()
print("   agent-wise eigenvalues at w=2:", [sp.nsimplify(e) for e in ev])
lam = sp.symbols("lam")
print("   char poly:", sp.factor(J.charpoly(lam).as_expr()))

# 2. Row relaxation at frozen costs: d/dt (l_j - l_k) = -T (l_j - l_k).
print("2. row relaxation")
T = 0.25
D = rng.random(4) * 2
pi = rng.random(4); pi /= pi.sum()
bstar = pi * np.exp(-D / T); bstar /= bstar.sum()
def rowfield(t, b):
    h = D + T * (np.log(b / pi) + 1)
    return -b * (h - b @ h)
b0 = rng.random(4); b0 /= b0.sum()
sol = solve_ivp(rowfield, (0, 3), b0, rtol=1e-11, atol=1e-13)
l = np.log(sol.y / bstar[:, None])
c0 = l[0, 0] - l[1, 0]; c1 = l[0, -1] - l[1, -1]
print("   measured rate:", -math.log(c1 / c0) / sol.t[-1], "expected", T)

# 3. Learned prior: Fisher flow of T KL(beta||pi) in pi is T(beta - pi).
print("3. learned prior")
p = sp.symbols("p0:3", positive=True)
bb = sp.symbols("b0:3", positive=True)
Ts = sp.symbols("T", positive=True)
E = Ts * sum(bb[j] * sp.log(bb[j] / p[j]) for j in range(3))
h = [sp.diff(E, p[j]) for j in range(3)]
mean = sum(p[j] * h[j] for j in range(3))
res = [sp.simplify(-p[j] * (h[j] - mean) - Ts * (bb[j] - p[j])).subs(bb[2], 1 - bb[0] - bb[1]).subs(p[2], 1 - p[0] - p[1]) for j in range(3)]
print("   residuals:", [sp.simplify(r) for r in res])

# 4. Registered agreement: T F = E + C_n + T log|X_n| at point recognition, rows free.
print("4. registered agreement")
from math import comb, factorial
def type_class_size(counts):
    n = sum(counts)
    r = factorial(n)
    for c in counts:
        r //= factorial(c)
    return r
n = 6
T = 1.0 / n
Kq = 3
# two agents, one sector only for q and a trivial model sector of size 1 to keep it small
q1 = np.array([2, 3, 1]) / n
q2 = np.array([1, 1, 4]) / n
R1 = rng.random(Kq) + 0.2; R1 /= R1.sum()
R2 = rng.random(Kq) + 0.2; R2 /= R2.sum()
Kch = rng.random((Kq, Kq)) + 0.1; Kch /= Kch.sum(1, keepdims=True)
# receiver 1: sources {2 (channel Kch), 1 (identity self)}; receiver 2: sources {1 (identity), 2 self}
nu = {(1, 2): q2 @ Kch, (1, 1): q1, (2, 1): q1, (2, 2): q2}
pis = {1: np.array([0.3, 0.7]), 2: np.array([0.6, 0.4])}
srcs = {1: [2, 1], 2: [1, 2]}
qs = {1: q1, 2: q2}
Rs = {1: R1, 2: R2}
def logp_type(p, Q):
    counts = np.rint(p * n).astype(int)
    return math.log(type_class_size(counts)) - n * (H(p) + kl(p, Q))
def cn(p):
    counts = np.rint(p * n).astype(int)
    return H(p) - math.log(type_class_size(counts)) / n
beta = {1: np.array([0.45, 0.55]), 2: np.array([0.2, 0.8])}
Epot = 0.0
TF = 0.0
logXn = math.log(17.0)  # arbitrary lattice size
TF += T * logXn
for i in (1, 2):
    Epot += kl(qs[i], Rs[i])
    TF += -T * logp_type(qs[i], Rs[i])
    for jj, j in enumerate(srcs[i]):
        Dij = kl(qs[i], nu[(i, j)])
        Epot += beta[i][jj] * Dij
        TF += -T * beta[i][jj] * logp_type(qs[i], nu[(i, j)])
    Epot += T * kl(beta[i], pis[i])
    TF += T * kl(beta[i], pis[i])
Cn = sum(cn(qs[i]) + cn(qs[i]) for i in (1, 2))
print("   TF - (E + C_n + T log|X|):", TF - (Epot + Cn + T * logXn))
# posterior of label equals Gibbs row
for i in (1, 2):
    lik = np.array([math.exp(logp_type(qs[i], nu[(i, j)])) for j in srcs[i]])
    post = pis[i] * lik; post /= post.sum()
    D = np.array([kl(qs[i], nu[(i, j)]) for j in srcs[i]])
    g = pis[i] * np.exp(-D / T); g /= g.sum()
    print("   row", i, "posterior - Gibbs:", np.max(np.abs(post - g)))

# 5. Frozen-source tied-copy identity: T KL(beta x q^N || pi nu_J^N) = sum beta D + T KL(beta||pi).
print("5. frozen source, tied copies")
N = 3
T = 1.0 / N
q = rng.random(3); q /= q.sum()
nus = [rng.random(3) + 0.1 for _ in range(3)]
nus = [v / v.sum() for v in nus]
pi = rng.random(3); pi /= pi.sum()
beta = rng.random(3); beta /= beta.sum()
lhs = 0.0
for j in range(3):
    for xs in itertools.product(range(3), repeat=N):
        a = beta[j] * np.prod([q[x] for x in xs])
        b = pi[j] * np.prod([nus[j][x] for x in xs])
        lhs += a * math.log(a / b)
lhs *= T
rhs = sum(beta[j] * kl(q, nus[j]) for j in range(3)) + T * kl(beta, pi)
print("   residual:", lhs - rhs)

# 6. Label-model identity (X4-PRIOR-07): T F = E + sum_i [H(q_i) + (T-1) KL(q_i||r_i)] + sum beta G.
print("6. label model")
T = 0.7
A = 3
Nag = 3
qs = [rng.random(A) + 0.05 for _ in range(Nag)]; qs = [v / v.sum() for v in qs]
rs = [rng.random(A) + 0.05 for _ in range(Nag)]; rs = [v / v.sum() for v in rs]
Ks = {(i, j): (lambda M: M / M.sum(1, keepdims=True))(rng.random((A, A)) + 0.05) for i in range(Nag) for j in range(Nag) if i != j}
pis = [None] * Nag; betas = [None] * Nag
for i in range(Nag):
    v = rng.random(Nag - 1); pis[i] = v / v.sum()
    v = rng.random(Nag - 1); betas[i] = v / v.sum()
F = 0.0
Epot = 0.0
extra = 0.0
for i in range(Nag):
    F += kl(qs[i], rs[i]) + kl(betas[i], pis[i])
    Epot += kl(qs[i], rs[i]) + T * kl(betas[i], pis[i])
    extra += H(qs[i]) + (T - 1) * kl(qs[i], rs[i])
    for jj, j in enumerate([j for j in range(Nag) if j != i]):
        K = Ks[(i, j)]  # K[v, w] = k(w|v) from source value v to receiver value w
        # record success prob k(y_i|y_j)^{1/T}
        F += betas[i][jj] * (1 / T) * sum(qs[i][w] * qs[j][v] * (-math.log(K[v, w])) for v in range(A) for w in range(A))
        b = qs[j] @ K
        Dij = kl(qs[i], b)
        Epot += betas[i][jj] * Dij
        G = 0.0
        for w_ in range(A):
            pw = qs[j] * K[:, w_] / b[w_]
            G += qs[i][w_] * kl(qs[j], pw)
        extra += betas[i][jj] * G
print("   residual:", T * F - (Epot + extra))

# 7. Compensation identity (E6).
print("7. compensation identity")
qsrc = rng.random(4) + 0.1; qsrc /= qsrc.sum()
Kc = rng.random((4, 4)) + 0.1; Kc /= Kc.sum(1, keepdims=True)
pK = qsrc @ Kc
recv = [rng.random(4) + 0.1 for _ in range(3)]; recv = [v / v.sum() for v in recv]
bw = rng.random(3)
W = bw.sum(); qbar = sum(bw[i] * recv[i] for i in range(3)) / W
lhs = sum(bw[i] * kl(recv[i], pK) for i in range(3))
rhs = sum(bw[i] * kl(recv[i], qbar) for i in range(3)) + W * kl(qbar, pK)
print("   residual:", lhs - rhs)

# 8. Bayesian versus damped record process: chain rule identity and bound.
print("8. record processes")
M, O, steps, alpha = 3, 3, 4, 0.6
k = rng.random((M, O)) + 0.1; k /= k.sum(1, keepdims=True)
s0 = rng.random(M); s0 /= s0.sum()
def bayes(s, o):
    p = s * k[:, o]; return p / p.sum()
def damped(s, o):
    return (1 - alpha) * s + alpha * bayes(s, o)
KLtot = 0.0; sumpred = 0.0; sumpost = 0.0
for hist in itertools.product(range(O), repeat=steps):
    s = s0.copy(); st = s0.copy(); P = 1.0; Pt = 1.0
    for t, o in enumerate(hist):
        P *= (s @ k)[o]; Pt *= (st @ k)[o]
        s = bayes(s, o); st = damped(st, o)
    KLtot += P * math.log(P / Pt)
for t in range(steps):
    for hist in itertools.product(range(O), repeat=t):
        s = s0.copy(); st = s0.copy(); P = 1.0
        for o in hist:
            P *= (s @ k)[o]
            s = bayes(s, o); st = damped(st, o)
        sumpred += P * kl(s @ k, st @ k)
        sumpost += P * kl(s, st)
print("   KL:", KLtot, "sum predictive:", sumpred, "bound:", sumpost, "residual:", KLtot - sumpred)

# 9. Global floor for fixed channels (L03-DYN-05): min entry never below min(m0, -C).
print("9. global floor")
A = 3; Nag = 3; Tq = 0.3
G = rng.random((A, A)) + 0.2; G /= G.sum(1, keepdims=True)  # G[m, k]
r = np.ones(A) / A
Kfix = np.eye(A)[[1, 2, 0]]  # permutation channels
def unpack(u):
    return u[:Nag * A].reshape(Nag, A), u[Nag * A:].reshape(Nag, Nag - 1)
s_fixed = np.ones(A) / A
def energy_grad(u):
    q, b = unpack(u)
    hq = np.zeros_like(q); hb = np.zeros_like(b)
    E = 0.0
    lg = s_fixed @ np.log(G)  # expected log evaluator over model
    for i in range(Nag):
        E += np.sum(q[i] * (np.log(q[i]) - lg))
        hq[i] += np.log(q[i]) + 1 - lg
        others = [j for j in range(Nag) if j != i]
        for jj, j in enumerate(others):
            nu = q[j] @ Kfix
            Dij = kl(q[i], nu)
            E += b[i, jj] * Dij
            hq[i] += b[i, jj] * (np.log(q[i] / nu) + 1)
            hq[j] += -b[i, jj] * (Kfix @ (q[i] / nu))
            hb[i, jj] += Dij
        pi = np.ones(Nag - 1) / (Nag - 1)
        E += Tq * kl(b[i], pi)
        hb[i] += Tq * (np.log(b[i] / pi) + 1)
    return E, hq, hb
def field_log(t, x):
    u = np.exp(x)
    q, b = unpack(u)
    q = q / q.sum(1, keepdims=True); b = b / b.sum(1, keepdims=True)
    uu = np.concatenate([q.ravel(), b.ravel()])
    _, hq, hb = energy_grad(uu)
    dq = -(hq - np.sum(q * hq, 1, keepdims=True))
    db = -(hb - np.sum(b * hb, 1, keepdims=True))
    return np.concatenate([dq.ravel(), db.ravel()])
q0 = np.array([[1 - 2e-9, 1e-9, 1e-9], [1e-9, 1 - 2e-9, 1e-9], [0.3, 0.3, 0.4]])
b0 = np.array([[0.5, 0.5]] * Nag)
x0 = np.log(np.concatenate([q0.ravel(), b0.ravel()]))
sol = solve_ivp(field_log, (0, 60), x0, method="LSODA", rtol=1e-10, atol=1e-12)
def norm_state(x):
    q, b = unpack(np.exp(x)); return np.concatenate([(q / q.sum(1, keepdims=True)).ravel(), (b / b.sum(1, keepdims=True)).ravel()])
qmins = [unpack(norm_state(sol.y[:, k]))[0].min() for k in range(sol.y.shape[1])]
Es = [energy_grad(norm_state(sol.y[:, k]))[0] for k in range(sol.y.shape[1])]
print("   initial min", q0.min(), "trajectory min", min(qmins), "max E increase:", max(np.diff(Es)), "final |field|:", np.linalg.norm(field_log(0, sol.y[:, -1])))

# 10. Row bounds with identity self-channel.
print("10. self-row bounds")
worst = 0.0
for _ in range(2000):
    m = 5
    D = rng.random(m) * 3; D[0] = 0.0
    pi = rng.random(m); pi /= pi.sum()
    T = rng.random() + 0.05
    g = pi * np.exp(-D / T); Z = g.sum(); g /= Z
    worst = min(worst, g[0] - pi[0], np.min(pi[1:] * np.exp(-D[1:] / T) / pi[0] - g[1:]))
print("   worst margin:", worst)

# 11. Joint-law flow generates dependence: d/dt LOR = -(LOR - LOR(R)).
print("11. log odds ratio")
Jc = 0.7
R = np.array([[math.exp(Jc * a * b) for b in (-1, 1)] for a in (-1, 1)]); R /= R.sum()
zeta = np.outer([0.3, 0.7], [0.6, 0.4])
hq = rng.random(2); hs = rng.random(2)
h = np.log(zeta / R) + hq[:, None] + hs[None, :]
dlog = -(h - np.sum(zeta * h))
sgn = np.array([[1, -1], [-1, 1]])
lor = lambda z: np.sum(sgn * np.log(z))
print("   residual:", np.sum(sgn * dlog) + (lor(zeta) - lor(R)), "LOR(R) =", lor(R), "4J =", 4 * Jc)

# 12. Dirichlet predictive increments bounded by 1/(A+n).
print("12. Dirichlet predictive")
al = np.array([0.5, 1.2, 2.0]); Atot = al.sum(); c = np.zeros(3); worst = 0
for nstep in range(1, 200):
    prev = (al + c) / (Atot + nstep - 1)
    o = rng.integers(3); c[o] += 1
    new = (al + c) / (Atot + nstep)
    worst = max(worst, 0.5 * np.abs(new - prev).sum() * (Atot + nstep))
print("   max TV*(A+n):", worst)
