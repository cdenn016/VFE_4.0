r"""Scope of the unqualified claim 'the reciprocal law potential is not the free energy of any fixed
normalized model whose recognition laws are the agent laws' (frontmatter.tex:5, chapters/06_outlook.tex:6,
companion/c03_dynamics.tex:12, companion/glossary.tex:20) versus prop:dyn-no-global-vfe
(chapters/03_dynamics.tex:375), which assumes fixed positive rows and an identity channel on an edge j->i.

Counterexample to the unqualified claim in a degenerate case: fixed rows, channels whose compared laws are
fixed positive laws C (constant in the input and independent of the configuration), source sets excluding
self.  Then, with rows summing to one,
    E = sum_i KL(q_i x s_i || R_i) + sum_{i,j} b_ij KL(q_i || C_ij) + sum_{i,l} g_il KL(s_i || C^s_il) + const
      = 2 sum_i [H(q_i) + H(s_i)]_neg - E_Q[Phi] + const,
and with the fixed normalized joint Pi propto exp(Phi/2) on the finite product alphabet,
    2 F(Q; o) = 2 KL(Q || Pi) + const = E + const   for the product recognition Q = (x)_k (q_k x s_k).
The script checks E - 2F is constant over random configurations (finite alphabets, three agents).
It also confirms that the identity-channel case of the proposition is not of this form (E - 2F varies).
"""
import itertools

import numpy as np

rng = np.random.default_rng(1029)
nK, nM, N = 3, 2, 3


def simplex(n):
    p = rng.random(n) + 0.05
    return p / p.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


R = [np.outer(simplex(nK), np.ones(nM)) * simplex(nK * nM).reshape(nK, nM) for _ in range(N)]
R = [r / r.sum() for r in R]
C = {(i, j): simplex(nK) for i in range(N) for j in range(N) if i != j}
Cs = {(i, j): simplex(nM) for i in range(N) for j in range(N) if i != j}
beta = {}
gamma = {}
for i in range(N):
    w = simplex(N - 1)
    v = simplex(N - 1)
    js = [j for j in range(N) if j != i]
    for a, j in enumerate(js):
        beta[(i, j)] = w[a]
        gamma[(i, j)] = v[a]


def potential(q, s, identity=False):
    E = sum(kl(np.outer(q[i], s[i]).ravel(), R[i].ravel()) for i in range(N))
    for (i, j), b in beta.items():
        nu = q[j] if identity else C[(i, j)]
        E += b * kl(q[i], nu)
    for (i, j), g in gamma.items():
        E += g * kl(s[i], Cs[(i, j)])
    return E


# Phi on the product alphabet ((k_i, m_i)_i): sum_i log R_i + sum b_ij log C_ij(k_i) + sum g_ij log Cs_ij(m_i)
states = list(itertools.product(range(nK), range(nM), repeat=N))
def phi(st):
    ks = st[0::2]
    ms = st[1::2]
    val = 0.0
    for i in range(N):
        val += np.log(R[i][ks[i], ms[i]])
        for j in range(N):
            if j != i:
                val += beta[(i, j)] * np.log(C[(i, j)][ks[i]]) + gamma[(i, j)] * np.log(Cs[(i, j)][ms[i]])
    return val
logPi = np.array([phi(st) / 2 for st in states])
logPi -= np.log(np.sum(np.exp(logPi)))


def free_energy(q, s):
    # F(Q;o) = KL(Q||Pi) for the product recognition with a trivial record (Z_o = 1)
    F = 0.0
    for idx, st in enumerate(states):
        ks = st[0::2]
        ms = st[1::2]
        Qst = np.prod([q[i][ks[i]] * s[i][ms[i]] for i in range(N)])
        F += Qst * (np.log(Qst) - logPi[idx])
    return F


diffs_const, diffs_id = [], []
for _ in range(25):
    q = [simplex(nK) for _ in range(N)]
    s = [simplex(nM) for _ in range(N)]
    Fv = free_energy(q, s)
    diffs_const.append(potential(q, s) - 2 * Fv)
    diffs_id.append(potential(q, s, identity=True) - 2 * Fv)
out = [
    f"constant channels: spread of E - 2F over 25 configurations = {np.ptp(diffs_const):.2e} (constant => E = 2F + b)",
    f"identity channels: spread of E - 2F over the same configurations = {np.ptp(diffs_id):.3f} (not constant)",
]
print("\n".join(out))
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
