r"""Depth-three fine-versus-profile check with ATOMIC circles: every internal node
carries a fair coin (M = 2 points) instead of a Haar circle, with base-dependent
orientation sigma_v(r, c) = c * s_v(r), s_v(r) in {-1, +1}.

Purpose: (1) test thm:boundary-tree-flow at depth 3 (N = 8 leaves, both sectors,
row dynamics, incoming terms); (2) show that atomlessness is not used by the
invariance, response, or record statements; (3) measure the decay rate of the
public contrast and compare with the linearized rate 3 + 2/(N-1) obtained from
h = 3 phi - 2 sum_k b_k P_k phi on mean-zero Walsh modes with b = pi.
"""
import itertools
import numpy as np

rng = np.random.default_rng(11)
d = 3; N = 2 ** d; nint = N - 1; M = 2
Tq, Ts = 0.9, 0.6
nr, nw = 3, 2
A = np.array([0.25, 0.45, 0.30]); B = np.array([0.55, 0.45])
rho = np.exp(0.9 * rng.normal(size=(nr, nw))); rho /= (A[:, None] * B[None, :] * rho).sum()
logrho = np.log(rho)

cells = np.array(list(itertools.product(range(M), repeat=nint)))   # (C, nint)
C = len(cells)
Ubits = np.array(list(itertools.product([1, -1], repeat=d)))       # profile index order
def uidx(u):
    # u: (..., d) in {+1,-1} -> row index in Ubits (+1 -> 0, -1 -> 1, big-endian)
    b = (u < 0).astype(int)
    return (b * (2 ** np.arange(d - 1, -1, -1))).sum(axis=-1)

def leaf_path(i):
    """heap indices of internal nodes on leaf i's path and edge signs."""
    digs = [(i >> (d - 1 - j)) & 1 for j in range(d)]
    nodes, eps, v = [], [], 0
    for j in range(d):
        nodes.append(v); eps.append(+1 if digs[j] == 0 else -1)
        v = 2 * v + 1 + digs[j]
    return nodes, eps

def leaf_bits(orient, nb):
    """orient: (nint, nb) in {+1,-1}. returns idx array (N, nb, C) of profile rows."""
    coin = 2 * cells - 1                       # (C, nint) in {-1,+1}
    out = np.zeros((N, nb, C), dtype=int)
    for i in range(N):
        nodes, eps = leaf_path(i)
        u = np.stack([eps[j] * orient[nodes[j]][:, None] * coin[None, :, nodes[j]] for j in range(d)], axis=-1)
        out[i] = uidx(u)
    return out

def lca_class(i, j):
    k = 0
    while k < d and ((i >> (d - 1 - k)) & 1) == ((j >> (d - 1 - k)) & 1):
        k += 1
    return k
cls = np.array([[lca_class(i, j) if i != j else -1 for j in range(N)] for i in range(N)])
n_k = np.array([2 ** (d - k - 1) for k in range(d)]); pi_k = n_k / (N - 1)

def Pk_matrix(k):
    n = len(Ubits); P = np.zeros((n, n))
    for a, u in enumerate(Ubits):
        for b, v in enumerate(Ubits):
            if np.all(u[:k] == v[:k]) and v[k] == -u[k]:
                P[a, b] = 2.0 ** (-(d - k - 1))
    return P
PK = [Pk_matrix(k) for k in range(d)]
W = 1.0 / len(Ubits)

# ----- fine system: P has shape (N, nb, C) probabilities -----
def fine_rhs(state):
    P, S, beta, gamma = state
    out = []
    refP = A[:, None] / C; refS = B[:, None] / C
    pb = P.sum(axis=2); sb = S.sum(axis=2)          # base marginals (N, nb)
    Vq = -(sb @ logrho.T)                            # (N, nr)
    Vs = -(pb @ logrho)                              # (N, nw)
    lP = np.log(P); lS = np.log(S)
    KLq = (P[:, None] * (lP[:, None] - lP[None, :])).sum(axis=(2, 3))
    KLs = (S[:, None] * (lS[:, None] - lS[None, :])).sum(axis=(2, 3))
    hP = np.log(P / refP) + 1 + Vq[:, :, None] + np.einsum('ij,iac->iac', beta, lP + 1) \
         - np.einsum('ij,jac->iac', beta, lP) - np.einsum('ji,jac->iac', beta, P) / P
    hS = np.log(S / refS) + 1 + Vs[:, :, None] + np.einsum('ij,iac->iac', gamma, lS + 1) \
         - np.einsum('ij,jac->iac', gamma, lS) - np.einsum('ji,jac->iac', gamma, S) / S
    dP = -P * (hP - (P * hP).sum(axis=(1, 2), keepdims=True))
    dS = -S * (hS - (S * hS).sum(axis=(1, 2), keepdims=True))
    off = ~np.eye(N, dtype=bool)
    hb = np.where(off, KLq + Tq * (np.log(np.where(off, beta, 1) * (N - 1)) + 1), 0)
    hg = np.where(off, KLs + Ts * (np.log(np.where(off, gamma, 1) * (N - 1)) + 1), 0)
    db = -beta * (hb - (beta * hb).sum(axis=1, keepdims=True))
    dg = -gamma * (hg - (gamma * hg).sum(axis=1, keepdims=True))
    return [dP, dS, db, dg]

def rk4(rhs, s, dt, steps):
    for _ in range(steps):
        k1 = rhs(s); k2 = rhs([a + dt / 2 * b for a, b in zip(s, k1)])
        k3 = rhs([a + dt / 2 * b for a, b in zip(s, k2)]); k4 = rhs([a + dt * b for a, b in zip(s, k3)])
        s = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
    return s

# ----- profile system: f shape (nb, 2^d) density against A x U_d -----
def prof_rhs(s):
    f, g, bq, bs = s
    f0 = f.mean(axis=1); g0 = g.mean(axis=1)
    Vq = -(logrho @ (B * g0)); Vs = -(logrho.T @ (A * f0))
    def Dk(x, base):
        lx = np.log(x)
        return np.array([(base[:, None] * W * x * (lx - lx @ P.T)).sum() for P in PK])
    def force(x, V, b):
        lx = np.log(x)
        h = 2 * lx + V[:, None]
        for k, P in enumerate(PK):
            h = h - b[k] * (lx @ P.T + (x @ P.T) / x)
        return h
    hq = force(f, Vq, bq); hs = force(g, Vs, bs)
    df = -f * (hq - (A[:, None] * W * f * hq).sum()); dg = -g * (hs - (B[:, None] * W * g * hs).sum())
    Dq, Ds = Dk(f, A), Dk(g, B)
    aq = Dq + Tq * np.log(bq / pi_k); as_ = Ds + Ts * np.log(bs / pi_k)
    return [df, dg, -bq * (aq - (bq * aq).sum()), -bs * (as_ - (bs * as_).sum())]

def lift(f, idx, base):
    nb = f.shape[0]
    return np.stack([f[np.arange(nb)[:, None], idx[i]] * base[:, None] / C for i in range(N)])

f = np.exp(0.7 * rng.normal(size=(nr, N))); f /= (A[:, None] * W * f).sum()
g = np.exp(0.7 * rng.normal(size=(nw, N))); g /= (B[:, None] * W * g).sum()
bq = rng.dirichlet(np.ones(d)); bs = pi_k.copy()
orient_q = rng.choice([-1, 1], size=(nint, nr)); orient_s = rng.choice([-1, 1], size=(nint, nw))
iq = leaf_bits(orient_q, nr); is_ = leaf_bits(orient_s, nw)
P0 = lift(f, iq, A); S0 = lift(g, is_, B)
beta0 = np.zeros((N, N)); gamma0 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            beta0[i, j] = bq[cls[i, j]] / n_k[cls[i, j]]; gamma0[i, j] = bs[cls[i, j]] / n_k[cls[i, j]]

dt, steps = 0.002, 400
fin = rk4(fine_rhs, [P0, S0, beta0, gamma0], dt, steps)
pro = rk4(prof_rhs, [f, g, bq, bs], dt, steps)
rq = np.abs(fin[0] - lift(pro[0], iq, A)).max() * C / A.min()
rs = np.abs(fin[1] - lift(pro[1], is_, B)).max() * C / B.min()
rb = max(abs(fin[2][i, j] - pro[2][cls[i, j]] / n_k[cls[i, j]]) + abs(fin[3][i, j] - pro[3][cls[i, j]] / n_k[cls[i, j]])
         for i in range(N) for j in range(N) if i != j)
print(f"depth 3, coins: max density residual belief {rq:.2e} model {rs:.2e} rows {rb:.2e}")
print("distinct leaf laws at t=0:", len({tuple(np.round(P0[i].ravel(), 10)) for i in range(N)}),
      " profile moved:", np.abs(pro[0] - f).max())

# ----- contrast decay rate near the u-independent manifold -----
f0 = np.exp(0.3 * rng.normal(size=nr)); g0 = np.exp(0.3 * rng.normal(size=nw))
phi = rng.normal(size=(nr, N)); phi -= phi.mean(axis=1, keepdims=True)
fs = f0[:, None] * (1 + 1e-4 * phi); fs /= (A[:, None] * W * fs).sum()
gs = np.repeat(g0[:, None], N, axis=1); gs /= (B[:, None] * W * gs).sum()
s = [fs, gs, pi_k.copy(), pi_k.copy()]
def contrast(x):
    return np.abs(x / x.mean(axis=1, keepdims=True) - 1).max()
c0 = contrast(s[0]); s1 = rk4(prof_rhs, s, 0.001, 500); c1 = contrast(s1[0])
print(f"measured contrast decay rate over t in [0,0.5]: {np.log(c0/c1)/0.5:.4f}; predicted 3 + 2/(N-1) = {3 + 2/(N-1):.4f}")
