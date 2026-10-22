"""Ch.10 thm:boundary-tree-flow at depth two as an exact finite test.

The fine laws are measurable for (r, sigma_root, sigma_L, sigma_R), with fair signs, so the
original four-agent identity-channel Fisher flow (dynamic rows, unit mobility, coupled
evaluator) reduces to finite tables. We compare its velocity at a profile-lifted state with
the profile equations eq:boundary-tree-flow and eq:boundary-tree-rows, check D_k = KL(q_i||q_j),
and the Jensen inequality eq:boundary-tree-jensen.
"""
import itertools
import numpy as np

rng = np.random.default_rng(7)
Jc, Tq, Ts = 0.2, 0.7, 1.3
phi = np.array([1.0, -1.0])                     # balanced base signs
rho = np.exp(Jc * np.outer(phi, phi)) / np.cosh(Jc)   # rho(r,w) against uniform A x B
signs = list(itertools.product([1, -1], repeat=3))    # (s_root, s_L, s_R)
paths = {0: lambda s: (s[0], s[1]), 1: lambda s: (s[0], -s[1]),
         2: lambda s: (-s[0], s[2]), 3: lambda s: (-s[0], -s[2])}
def cls(i, j):                       # number of shared initial edges
    return 1 if i // 2 == j // 2 else 0
n_k = {0: 2, 1: 1}                   # n_k = 2^(d-k-1)
pi_k = np.array([2 / 3, 1 / 3])

def profile(seed):
    f = np.exp(0.5 * seed.standard_normal((2, 2, 2)))   # f[r, u1idx, u2idx], idx 0 <-> +1
    return f / f.mean()
idx = lambda u: 0 if u == 1 else 1

def lift(f):                          # q[i, r, s] over 2 x 8 fine points
    q = np.zeros((4, 2, 8))
    for i in range(4):
        for r in range(2):
            for t, s in enumerate(signs):
                u = paths[i](s)
                q[i, r, t] = f[r, idx(u[0]), idx(u[1])]
    return q

def fine_velocity(q, s_, beta_q, beta_s):
    out = []
    for laws, other, beta in ((q, s_, beta_q), (s_, q, beta_s)):
        vel = np.zeros_like(laws)
        for i in range(4):
            other0 = other[i].mean(axis=1)              # base marginal density of other sector
            V = -np.log(rho) @ other0 / 2 if laws is q else -np.log(rho).T @ other0 / 2
            h = 2 * np.log(laws[i]) + V[:, None]
            for j in range(4):
                if j != i:
                    h -= beta[i, j] * np.log(laws[j]) + beta[j, i] * laws[j] / laws[i]
            Eh = (laws[i] * h).mean()
            vel[i] = -laws[i] * (h - Eh)
        out.append(vel)
    rows = []
    for laws, beta, T in ((q, beta_q, Tq), (s_, beta_s, Ts)):
        db = np.zeros_like(beta)
        for i in range(4):
            c = np.zeros(4)
            for j in range(4):
                if j != i:
                    D = (laws[i] * (np.log(laws[i]) - np.log(laws[j]))).mean()
                    c[j] = D + T * np.log(beta[i, j] / (n_k[cls(i, j)] ** -1 * pi_k[cls(i, j)]))
            mask = np.arange(4) != i
            db[i, mask] = -beta[i, mask] * (c[mask] - (beta[i, mask] * c[mask]).sum())
        rows.append(db)
    return out[0], out[1], rows[0], rows[1]

def P(k, F):                          # bit-profile operator
    G = np.empty_like(F)
    for u1, u2 in itertools.product(range(2), repeat=2):
        if k == 0:
            G[:, u1, u2] = F[:, 1 - u1, :].mean(axis=1)
        else:
            G[:, u1, u2] = F[:, u1, 1 - u2]
    return G

def Dk(k, F):
    return (F * (np.log(F) - P(k, np.log(F)))).mean()

def profile_velocity(f, g, bq, bs):
    vels = []
    for F, G, b, transpose in ((f, g, bq, False), (g, f, bs, True)):
        G0 = G.mean(axis=(1, 2))
        V = -(np.log(rho).T if transpose else np.log(rho)) @ G0 / 2
        h = 2 * np.log(F) + V[:, None, None]
        for k in range(2):
            h -= b[k] * (P(k, np.log(F)) + P(k, F) / F)
        vels.append(-F * (h - (F * h).mean()))
    rows = []
    for F, b, T in ((f, bq, Tq), (g, bs, Ts)):
        c = np.array([Dk(k, F) + T * np.log(b[k] / pi_k[k]) for k in range(2)])
        rows.append(-b * (c - (b * c).sum()))
    return vels[0], vels[1], rows[0], rows[1]

f, g = profile(rng), profile(rng)
bq, bs = np.array([0.55, 0.45]), np.array([0.3, 0.7])
q, s_ = lift(f), lift(g)
beta_q = np.array([[0 if i == j else bq[cls(i, j)] / n_k[cls(i, j)] for j in range(4)] for i in range(4)])
beta_s = np.array([[0 if i == j else bs[cls(i, j)] / n_k[cls(i, j)] for j in range(4)] for i in range(4)])
vq, vs, dbq, dbs = fine_velocity(q, s_, beta_q, beta_s)
pf, pg, pbq, pbs = profile_velocity(f, g, bq, bs)
print('max |fine belief velocity - lifted profile velocity| =', np.abs(vq - lift(pf)).max())
print('max |fine model velocity - lifted profile velocity|  =', np.abs(vs - lift(pg)).max())
err = max(abs(dbq[i, j] - pbq[cls(i, j)] / n_k[cls(i, j)]) for i in range(4) for j in range(4) if i != j)
err2 = max(abs(dbs[i, j] - pbs[cls(i, j)] / n_k[cls(i, j)]) for i in range(4) for j in range(4) if i != j)
print('max row-velocity residual (belief, model) =', err, err2)
for k in range(2):
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j and cls(i, j) == k]
    kl = [(q[i] * (np.log(q[i]) - np.log(q[j]))).mean() for i, j in pairs]
    print(f'class {k}: D_k(f)={Dk(k, f):.12f}, fine KL range=({min(kl):.12f},{max(kl):.12f})')
    jensen = (f * (np.log(P(k, f)) - P(k, np.log(f)))).mean()
    print(f'  Jensen gap D_k - KL(f||P_k f) = {Dk(k, f) - (f * (np.log(f) - np.log(P(k, f)))).mean():.3e}, direct = {jensen:.3e}')
# load-bearing incoming term: drop P_k f / f
def profile_velocity_no_incoming(F, G, b):
    G0 = G.mean(axis=(1, 2)); V = -np.log(rho) @ G0 / 2
    h = 2 * np.log(F) + V[:, None, None]
    for k in range(2):
        h -= b[k] * P(k, np.log(F))
    return -F * (h - (F * h).mean())
print('velocity change if incoming term is dropped:', np.abs(profile_velocity_no_incoming(f, g, bq) - pf).max())
