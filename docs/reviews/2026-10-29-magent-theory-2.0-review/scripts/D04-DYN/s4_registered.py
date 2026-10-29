r"""D04-DYN s4: thm:dyn-registered (registered agreement, 2026-10-22 C1) and cor:dyn-frozen-source by brute force.

Two agents, binary state and model alphabets, n = 9 (so T = 1/9), configurations with q_i, s_i in {1/3, 2/3}
(every q_i, s_i and q_i x s_i is a 9-type), law-dependent belief channels and one law-dependent model
channel, identity self channels in the source sets.  Every registration probability is computed by summing
sequence probabilities over all 2^9 or 4^9 batches; the method of types is not used to compute them.
"""
import itertools
import numpy as np

n = 9
T = 1.0 / n
rng = np.random.default_rng(5)


def H(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))


SEQ = {L: np.array(list(itertools.product(range(L), repeat=n)), dtype=np.int8) for L in (2, 4)}
CNT = {L: np.stack([(SEQ[L] == c).sum(axis=1) for c in range(L)], axis=1) for L in (2, 4)}


def p_type(target, nu):
    """P(empirical type of a batch of n draws from nu equals target), summing over all sequences."""
    L = len(nu)
    tc = np.rint(np.asarray(target) * n).astype(int)
    mask = np.all(CNT[L] == tc, axis=1)
    logp = (CNT[L][mask] * np.log(np.asarray(nu))[None, :]).sum(axis=1)
    return float(np.exp(logp).sum()), int(mask.sum())


r = [np.array([0.35, 0.65]), np.array([0.6, 0.4])]
G = [np.array([[0.7, 0.3], [0.2, 0.8]]), np.array([[0.55, 0.45], [0.1, 0.9]])]   # G[m, k]
Rj = [(G[i] * r[i][:, None]).T.ravel() for i in range(2)]                      # R[k, m], k major
Nq = {0: [0, 1], 1: [0, 1]}
Ns = {0: [1], 1: [0, 1]}
piq = {0: np.array([0.3, 0.7]), 1: np.array([0.55, 0.45])}
pis = {0: np.array([1.0]), 1: np.array([0.4, 0.6])}


def chan(sec, i, j, x):
    q0, s0, q1, s1 = x
    if j == i:
        return np.eye(2)
    if sec == "q" and (i, j) == (0, 1):
        e = 0.1 + 0.2 * q0[1] * q1[0]
        return np.array([[1 - e, e], [e, 1 - e]])
    if sec == "q" and (i, j) == (1, 0):
        return np.array([[0.8, 0.2], [0.3 + 0.2 * q1[1], 0.7 - 0.2 * q1[1]]])
    if sec == "s" and (i, j) == (0, 1):
        return np.array([[0.7, 0.3], [0.25, 0.75]])
    if sec == "s" and (i, j) == (1, 0):
        e = 0.15 + 0.1 * s0[1]
        return np.array([[1 - e, e], [e, 1 - e]])
    raise KeyError


def laws(x):
    q0, s0, q1, s1 = x
    return [q0, q1], [s0, s1]


def compared(x):
    q, s = laws(x)
    nq = {(i, j): q[j] @ chan("q", i, j, x) for i in range(2) for j in Nq[i]}
    ns = {(i, j): s[j] @ chan("s", i, j, x) for i in range(2) for j in Ns[i]}
    return nq, ns


def potential(x, be, ga):
    q, s = laws(x)
    nq, ns = compared(x)
    E = sum(KL(np.outer(q[i], s[i]).ravel(), Rj[i]) for i in range(2))
    for i in range(2):
        E += sum(be[i][k] * KL(q[i], nq[(i, j)]) for k, j in enumerate(Nq[i])) + T * KL(be[i], piq[i])
        E += sum(ga[i][k] * KL(s[i], ns[(i, j)]) for k, j in enumerate(Ns[i])) + T * KL(ga[i], pis[i])
    return E


def c_n(p, size):
    return H(p) - np.log(size) / n


grid = [np.array([1 - a / 3, a / 3]) for a in (1, 2)]
X = [(q0, s0, q1, s1) for q0 in grid for s0 in grid for q1 in grid for s1 in grid]
print(f"lattice size |X_n| = {len(X)}, n = {n}")
worst_i = worst_ii = worst_iii = worst_iv = 0.0
cn_ok = True
Pox = []
for x in X:
    q, s = laws(x)
    nq, ns = compared(x)
    P0 = [p_type(np.outer(q[i], s[i]).ravel(), Rj[i]) for i in range(2)]
    Pq = {e: p_type(q[e[0]], nq[e]) for e in nq}
    Ps = {e: p_type(s[e[0]], ns[e]) for e in ns}
    for e, (pr, size) in Pq.items():
        pred = size * np.exp(-n * (H(q[e[0]]) + KL(q[e[0]], nq[e])))
        worst_i = max(worst_i, abs(pr - pred) / pred)
    for i in range(2):
        pr, size = P0[i]
        z = np.outer(q[i], s[i]).ravel()
        pred = size * np.exp(-n * (H(z) + KL(z, Rj[i])))
        worst_i = max(worst_i, abs(pr - pred) / pred)
    Cn = 0.0
    for i in range(2):
        for p, size in ((np.outer(q[i], s[i]).ravel(), P0[i][1]), (q[i], Pq[(i, Nq[i][0])][1]), (s[i], Ps[(i, Ns[i][0])][1])):
            c = c_n(p, size)
            cn_ok = cn_ok and (-1e-12 <= c <= np.count_nonzero(p) * np.log(n + 1) / n + 1e-12)
            Cn += c
    logZ = 0.0
    for i in range(2):
        w = np.array([piq[i][k] * Pq[(i, j)][0] for k, j in enumerate(Nq[i])])
        Dv = np.array([KL(q[i], nq[(i, j)]) for j in Nq[i]])
        gb = piq[i] * np.exp(-Dv / T)
        logZ += np.log(gb.sum())
        worst_ii = max(worst_ii, np.max(np.abs(w / w.sum() - gb / gb.sum())))
        w = np.array([pis[i][k] * Ps[(i, j)][0] for k, j in enumerate(Ns[i])])
        Dv = np.array([KL(s[i], ns[(i, j)]) for j in Ns[i]])
        gb = pis[i] * np.exp(-Dv / T)
        logZ += np.log(gb.sum())
        worst_ii = max(worst_ii, np.max(np.abs(w / w.sum() - gb / gb.sum())))
    for trial in range(5):
        be = {i: rng.dirichlet(np.ones(len(Nq[i]))) for i in range(2)}
        ga = {i: rng.dirichlet(np.ones(len(Ns[i]))) for i in range(2)}
        F = np.log(len(X))
        for i in range(2):
            F += KL(be[i], piq[i]) + KL(ga[i], pis[i]) - np.log(P0[i][0])
            F += -sum(be[i][k] * np.log(Pq[(i, j)][0]) for k, j in enumerate(Nq[i]))
            F += -sum(ga[i][k] * np.log(Ps[(i, j)][0]) for k, j in enumerate(Ns[i]))
        rhs = potential(x, be, ga) + Cn + T * np.log(len(X))
        worst_iii = max(worst_iii, abs(T * F - rhs))
    Po = np.prod([P0[i][0] for i in range(2)])
    for i in range(2):
        Po *= sum(piq[i][k] * Pq[(i, j)][0] for k, j in enumerate(Nq[i]))
        Po *= sum(pis[i][k] * Ps[(i, j)][0] for k, j in enumerate(Ns[i]))
    Ered = sum(KL(np.outer(q[i], s[i]).ravel(), Rj[i]) for i in range(2)) - T * logZ
    worst_iv = max(worst_iv, abs(Ered - (-T * np.log(Po) - Cn)))
    Pox.append((Po, Ered, Cn))
Pevid = np.mean([p[0] for p in Pox])
bound_gap = min(p[1] + p[2] + T * np.log(len(X)) + T * np.log(Pevid) for p in Pox)
print(f"(i)   max relative error of |T_p| exp(-n[H + D]) vs brute force = {worst_i:.2e}")
print(f"      0 <= c_n(p) <= |supp p| log(n+1)/n on every registered target: {cn_ok}")
print(f"(ii)  max abs(label posterior - Gibbs row at T = 1/n) = {worst_ii:.2e}")
print(f"(iii) max abs(T F - [E + C_n + T log|X_n|]) over 16 configurations x 5 random rows = {worst_iii:.2e}")
print(f"(iv)  max abs(E_red - [-T log P(o|x) - C_n]) = {worst_iv:.2e}; min over x of E_red + C_n + T log|X_n| + T log P(o) = {bound_gap:.4f} (nonnegative)")

print("== cor:dyn-frozen-source by brute force (N = 3, ternary alphabet) ==")
N = 3
TN = 1.0 / N
nus = [rng.dirichlet(np.ones(3)) for _ in range(3)]
pi = rng.dirichlet(np.ones(3))
beta = rng.dirichlet(np.ones(3))
qq = np.array([2 / 3, 1 / 3, 0.0])
qq2 = rng.dirichlet(np.ones(3))
lhs = 0.0
for j in range(3):
    for sq in itertools.product(range(3), repeat=N):
        pq = np.prod([qq2[c] for c in sq])
        pn = np.prod([nus[j][c] for c in sq])
        lhs += beta[j] * pq * (np.log(beta[j] * pq) - np.log(pi[j] * pn))
rhs = sum(beta[j] * KL(qq2, nus[j]) for j in range(3)) + TN * KL(beta, pi)
print(f"  abs(T KL(beta x q^N || pi nu_J^N) - [sum beta D + T KL(beta||pi)]) = {abs(TN * lhs - rhs):.2e}")
sq = (0, 0, 1)
post = np.array([pi[j] * np.prod([nus[j][c] for c in sq]) for j in range(3)])
post /= post.sum()
gib = pi * np.exp(-np.array([KL(qq, nus[j]) for j in range(3)]) / TN)
gib /= gib.sum()
print(f"  posterior of J for a sequence of type (2/3,1/3,0) vs Gibbs row at T = 1/N: {np.max(np.abs(post - gib)):.2e}")

print("== remark after thm:dyn-registered: multinomial covariance equals T times the inverse Fisher metric ==")
p = rng.dirichlet(np.ones(3))
nn = 4
seq = np.array(list(itertools.product(range(3), repeat=nn)))
prob = np.prod(p[seq], axis=1)
freq = np.stack([(seq == c).mean(axis=1) for c in range(3)], axis=1)
mean = prob @ freq
cov = (freq - mean).T @ ((freq - mean) * prob[:, None])
print(f"  max abs(Cov(empirical freq) - (diag p - p p^T)/n) = {np.max(np.abs(cov - (np.diag(p) - np.outer(p, p)) / nn)):.2e}")
h = rng.normal(size=3)
print(f"  (diag p - p p^T) h equals the Fisher gradient p (h - E_p h): {np.max(np.abs((np.diag(p) - np.outer(p, p)) @ h - p * (h - p @ h))):.2e}")
