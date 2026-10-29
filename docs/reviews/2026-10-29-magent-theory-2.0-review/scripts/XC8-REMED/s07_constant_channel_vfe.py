r"""Scope of prop:dyn-no-global-vfe (03_dynamics.tex:375): a reciprocal potential that IS an affine image
of a fixed-model free energy with the agent laws as product recognition laws.

Take N=3 agents with binary state and model alphabets, fixed positive references R_i(k,m), fixed rows
beta_ij (rows of each agent sum to one over j != i, likewise gamma), and FIXED constant-output channels
K^q_ij(v,.) = C^q, K^s_ij(w,.) = C^s for every edge (normalized kernels that ignore the source sample;
admitted by def:int-channel and eq:dyn-peer-costs). Then
  E = sum_i [ KL(q_i x s_i || R_i) + KL(q_i || C^q) + KL(s_i || C^s) ] + row terms (constant),
and with the normalized joint  P(y, o=1) = U(y) * l(1|y),  U uniform on ({0,1}^2)^N,
  l(1|y) = prod_i sqrt(R_i(k_i,m_i) C^q(k_i) C^s(m_i)) / c,  c = max of the product (so l <= 1),
the free energy of the product recognition Q = (x)_i (q_i x s_i) satisfies E = 2 F(Q; o) + b with b constant.
prop:dyn-no-global-vfe is not contradicted (it assumes an identity-channel edge); the unqualified
statements in the abstract (frontmatter.tex:5) and elsewhere are.
"""
import itertools

import numpy as np

rng = np.random.default_rng(3)
N = 3
R = [rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(N)]   # R_i(k, m)
Cq = rng.dirichlet(np.ones(2))
Cs = rng.dirichlet(np.ones(2))
beta = rng.dirichlet(np.ones(N - 1), size=N)
gamma = rng.dirichlet(np.ones(N - 1), size=N)
T = 0.7
piq = np.full((N, N - 1), 1.0 / (N - 1))


def kl(p, q):
    p = np.asarray(p, float).ravel()
    q = np.asarray(q, float).ravel()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def potential(q, s):
    E = 0.0
    for i in range(N):
        E += kl(np.outer(q[i], s[i]), R[i])
        E += sum(beta[i, j] * kl(q[i], Cq) for j in range(N - 1)) + T * kl(beta[i], piq[i])
        E += sum(gamma[i, j] * kl(s[i], Cs) for j in range(N - 1)) + T * kl(gamma[i], piq[i])
    return E


# normalized joint over y=(k_i,m_i)_i with a registered binary record
ys = list(itertools.product(range(2), repeat=2 * N))
w = []
for y in ys:
    v = 1.0
    for i in range(N):
        k, m = y[2 * i], y[2 * i + 1]
        v *= np.sqrt(R[i][k, m] * Cq[k] * Cs[m])
    w.append(v)
w = np.array(w)
c = w.max()
lik = w / c                       # l(1|y) in (0,1]
P0 = np.full(len(ys), 1.0 / len(ys))


def free_energy(q, s):
    Q = np.array([np.prod([q[i][y[2 * i]] * s[i][y[2 * i + 1]] for i in range(N)]) for y in ys])
    return float(np.sum(Q * (np.log(Q) - np.log(P0))) - np.sum(Q * np.log(lik)))


vals = []
for _ in range(200):
    q = rng.dirichlet(np.ones(2), size=N)
    s = rng.dirichlet(np.ones(2), size=N)
    vals.append(potential(q, s) - 2.0 * free_energy(q, s))
vals = np.array(vals)
txt = ("E - 2F over 200 random interior configurations: mean %.12f, spread (max-min) %.3e\n"
       "record probability check: sum_y P0(y) l(1|y) = %.6f (in (0,1], so the joint is normalized with o in {0,1})"
       % (vals.mean(), vals.max() - vals.min(), float(np.sum(P0 * lik))))
print(txt)
with open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
