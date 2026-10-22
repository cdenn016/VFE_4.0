r"""X4-PRIOR bridge between the 2.0 declared potential and a normalized sample-level label model.

Model (finite alphabets, counting reference, one sector): prior P0(y) = prod_i r_i(y_i);
labels J_i ~ pi_i over the nonself sources of i; for each receiver one binary record o_i = 1 with
P(o_i = 1 | y, J_i = j) = k_ij(y_i | y_j)^{1/T}  (k_ij is a normalized channel kernel, entries <= 1).
Recognition Q(y, J) = prod_i q_i(y_i) prod_i beta_i(J_i)  (constant rows, product matter).

Claimed exact identity (derived from 2.0 prop:dyn-jensen and the earlier prop:obs-attention-elbo(b)):
  T * F_sample(q, beta) = E_20(q, beta) + sum_i [H(q_i) + (T-1) KL(q_i || r_i)] + sum_ij beta_ij G_ij,
with E_20 = sum_i KL(q_i||r_i) + sum_i [sum_j beta_ij D_ij + T KL(beta_i||pi_i)], D_ij = KL(q_i || q_j K_ij),
G_ij = int KL(q_j || q_j^w) q_i(dw) (the Jensen gap). Hence the label-model recognition row is
beta_ij ∝ pi_ij exp(-(D_ij + G_ij)/T), the 2.0 Gibbs row only when G_ij is independent of j.
"""
import itertools
import numpy as np

rng = np.random.default_rng(3)
N, n = 3, 3
T = 0.7
r = [rng.dirichlet(np.ones(n)) for _ in range(N)]
q = [rng.dirichlet(np.ones(n)) for _ in range(N)]
src = {i: [j for j in range(N) if j != i] for i in range(N)}
pi = {i: rng.dirichlet(np.ones(len(src[i]))) for i in range(N)}
beta = {i: rng.dirichlet(np.ones(len(src[i]))) for i in range(N)}
K = {(i, j): rng.dirichlet(np.ones(n), size=n) for i in range(N) for j in src[i]}  # K[(i,j)][v,w]


def KL(a, b):
    return float((a * np.log(a / b)).sum())

# direct enumeration of F_sample = E_Q[log Q - log P(y, J, o=1)]
F = 0.0
for y in itertools.product(range(n), repeat=N):
    qy = np.prod([q[i][y[i]] for i in range(N)])
    for J in itertools.product(*[range(len(src[i])) for i in range(N)]):
        bJ = np.prod([beta[i][J[i]] for i in range(N)])
        logQ = np.log(qy) + np.log(bJ)
        logP = sum(np.log(r[i][y[i]]) for i in range(N))
        for i in range(N):
            j = src[i][J[i]]
            logP += np.log(pi[i][J[i]]) + np.log(K[(i, j)][y[j], y[i]]) / T
        F += qy * bJ * (logQ - logP)

E20 = sum(KL(q[i], r[i]) for i in range(N))
extra = 0.0
for i in range(N):
    H = -float((q[i] * np.log(q[i])).sum())
    extra += H + (T - 1) * KL(q[i], r[i])
    E20 += T * KL(beta[i], pi[i])
    for a, j in enumerate(src[i]):
        b = q[j] @ K[(i, j)]
        D = KL(q[i], b)
        pw = (K[(i, j)] * q[j][:, None]) / b[None, :]  # posterior of source sample given w
        G = sum(q[i][w] * KL(q[j], pw[:, w]) for w in range(n))
        E20 += beta[i][a] * D
        extra += beta[i][a] * G
print("T*F_sample =", T * F)
print("E_20 + entropy/local terms + Jensen gaps =", E20 + extra)
print("residual:", T * F - (E20 + extra))
