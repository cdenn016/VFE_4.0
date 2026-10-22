r"""Partition and separator checks for chapters/05_inheritance.tex.

(1) inh:separator-gluing, three-block extension fails under pairwise conditional independence:
    U,V iid Ber(1/2), W = U xor V, Z constant. Pairwise independent, but
    E[L_U L_V L_W] != F_U F_V F_W for indicator factors.
(2) Four-cycle: pairwise marginals on edges (A,B),(B,C),(C,D),(D,A) that agree on shared single
    variables but admit no global joint law (LP infeasible). Shows that gluing along a cyclic
    separator structure needs a running-intersection (junction-tree) hypothesis.
(3) inh:finest-independent and inh:independence-discontinuity for N=4 Ising pairs: brute force over
    all 15 partitions at eps=0 and eps=0.01.
(4) inh:causal-memory identity on a random law of three ternary records with a lossy statistic.
(5) supplements: filter cycle example.
"""
from fractions import Fraction as Fr
from itertools import product
import math, random
import numpy as np
from scipy.optimize import linprog

# (1)
law = {(u, v, u ^ v): Fr(1, 4) for u in (0, 1) for v in (0, 1)}
def marg(idx):
    out = {}
    for k, p in law.items():
        kk = tuple(k[i] for i in idx)
        out[kk] = out.get(kk, 0) + p
    return out
pair_indep = all(marg((i, j))[(a, b)] == marg((i,))[(a,)] * marg((j,))[(b,)]
                 for i, j in [(0, 1), (0, 2), (1, 2)] for a in (0, 1) for b in (0, 1))
lhs = sum(p for k, p in law.items() if k == (1, 1, 1))
rhs = marg((0,))[(1,)] * marg((1,))[(1,)] * marg((2,))[(1,)]
print("(1) pairwise independent:", pair_indep, "; E[1_{U=1}1_{V=1}1_{W=1}] =", lhs, " vs product", rhs)

# (2) four-cycle: A=B, B=C, C=D, D != A on uniform bits
edges = {(0, 1): 'eq', (1, 2): 'eq', (2, 3): 'eq', (3, 0): 'neq'}
states = list(product((0, 1), repeat=4))
A_eq, b_eq = [], []
for (i, j), rel in edges.items():
    for a in (0, 1):
        for b in (0, 1):
            target = 0.5 if ((a == b) == (rel == 'eq')) else 0.0
            A_eq.append([1.0 if (s[i] == a and s[j] == b) else 0.0 for s in states]); b_eq.append(target)
A_eq.append([1.0] * 16); b_eq.append(1.0)
res = linprog(np.zeros(16), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, None)] * 16, method="highs")
print("(2) edge marginals locally consistent (all single marginals uniform); global joint exists:", res.status == 0,
      "|", res.message)

# (3) Ising pairs N=4
def ising(J, eps):
    w = {y: math.exp(J * (y[0] * y[1] + y[2] * y[3]) + eps * y[1] * y[2]) for y in product((-1, 1), repeat=4)}
    Z = sum(w.values())
    return {y: v / Z for y, v in w.items()}
def partitions(s):
    if not s:
        yield []
        return
    first, rest = s[0], s[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p
def is_indep(P, part, tol=1e-12):
    margs = []
    for B in part:
        m = {}
        for y, p in P.items():
            k = tuple(y[i] for i in B); m[k] = m.get(k, 0) + p
        margs.append(m)
    return all(abs(P[y] - math.prod(margs[b][tuple(y[i] for i in B)] for b, B in enumerate(part))) < tol for y in P)
for eps in (0.0, 0.01):
    P = ising(0.7, eps)
    ind = [p for p in partitions([0, 1, 2, 3]) if is_indep(P, p)]
    finest = max(ind, key=len)
    print(f"(3) eps={eps}: independent partitions = {len(ind)}; finest = {sorted(sorted(B) for B in finest)}")

# (4) causal memory identity
random.seed(7)
Zs = (0, 1, 2)
w = {h: random.random() + 0.05 for h in product(Zs, repeat=3)}
Zt = sum(w.values()); Pm = {h: v / Zt for h, v in w.items()}
f_stat = [lambda h: (), lambda h: (h[0] % 2,), lambda h: (h[0] + h[1]) % 2]  # lossy statistics
def prob(prefix):
    return sum(p for h, p in Pm.items() if h[:len(prefix)] == prefix)
Tt = {}
for t in range(3):
    for m in set(f_stat[t](h[:t]) for h in Pm):
        ww = [random.random() + 0.1 for _ in Zs]
        Tt[(t, m)] = [v / sum(ww) for v in ww]
def Ptilde(h):
    return math.prod(Tt[(t, f_stat[t](h[:t]))][h[t]] for t in range(3))
KL = sum(p * math.log(p / Ptilde(h)) for h, p in Pm.items())
I_sum, E_sum = 0.0, 0.0
for t in range(3):
    # joint law of (H_t, Z_t)
    for h in set(h[:t] for h in Pm):
        ph = prob(h)
        m = f_stat[t](h)
        pm_ = sum(prob(hh) for hh in set(g[:t] for g in Pm) if f_stat[t](hh) == m)
        for zv in Zs:
            pz_h = prob(h + (zv,)) / ph
            pz_m = sum(prob(hh + (zv,)) for hh in set(g[:t] for g in Pm) if f_stat[t](hh) == m) / pm_
            I_sum += ph * pz_h * math.log(pz_h / pz_m)
            E_sum += ph * pz_h * math.log(pz_m / Tt[(t, m)][zv])
print(f"(4) KL(P||P~) = {KL:.15f}; sum I + sum E KL = {I_sum + E_sum:.15f}; residual = {KL - I_sum - E_sum:.2e}")

# (5) filter cycle 1->2->3->1, R(1)=R(2)=0, R(3)=1, uniform initial
Tm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
Rv = np.array([0, 0, 1])
p0 = np.ones(3) / 3
Pz = {z: np.diag((Rv == z).astype(float)) for z in (0, 1)}
for hist in [(0, 0), (1, 0)]:
    f_ = p0 @ Pz[hist[0]]
    for zz in hist[1:]:
        f_ = f_ @ Tm @ Pz[zz]
    ph = f_ / f_.sum()
    print("(5) history", hist, "filter", ph.round(3).tolist(), "P(next retained = 1) =", float(ph @ Tm @ Pz[1] @ np.ones(3)))
