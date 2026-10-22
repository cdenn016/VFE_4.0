r"""Lumpability variants relevant to chapters/05_inheritance.tex (inh:finite-lumpability, inh:boundary-square,
channel square) and supplements/s04_inheritance.tex.

(A) Rogers-Pitman intertwining without strong lumpability: S={1,2,3}, R(1)=R(2)=a, R(3)=b,
    Lambda(a)=(1/2,1/2,0), Lambda(b)=delta_3, Lambda T = T^c Lambda. Retained process is Markov(T^c)
    for every initial law nu Lambda, but not for initial delta_1.
(B) Coarsest strongly lumpable refinement of a given partition (probabilistic bisimulation):
    partition refinement versus brute force over all partitions on a 6-state chain.
(C) Response and transition each descend separately, but their joint kernel does not.
(D) Stochastic retention: channel square K C_i = C_j K^c has no stochastic solution K^c for
    K = identity, C_i = identity, C_j = BSC(1/4), although C_j is an invertible matrix.
All exact rationals except (D), which uses an LP.
"""
from fractions import Fraction as Fr
from itertools import product
import random

# (A)
T = {1: {1: Fr(3, 10), 2: Fr(5, 10), 3: Fr(2, 10)},
     2: {1: Fr(4, 10), 2: Fr(2, 10), 3: Fr(4, 10)},
     3: {1: Fr(1, 4), 2: Fr(1, 4), 3: Fr(1, 2)}}
R = {1: 'a', 2: 'a', 3: 'b'}
Lam = {'a': {1: Fr(1, 2), 2: Fr(1, 2), 3: 0}, 'b': {1: 0, 2: 0, 3: 1}}
strong = all(sum(T[x][y] for y in T if R[y] == 'b') == sum(T[xp][y] for y in T if R[y] == 'b')
             for x in T for xp in T if R[x] == R[xp])
print("(A) strongly lumpable:", strong)
LT = {z: {y: sum(Lam[z][x] * T[x][y] for x in T) for y in T} for z in Lam}
Tc = {z: {w: sum(LT[z][y] for y in T if R[y] == w) for w in Lam} for z in Lam}
TcL = {z: {y: sum(Tc[z][w] * Lam[w][y] for w in Lam) for y in T} for z in Lam}
print("(A) Lambda T == T^c Lambda:", LT == TcL, "; T^c =", {z: dict(Tc[z]) for z in Tc})

def retained_path_law(p0, n):
    out = {}
    for path in product(T, repeat=n):
        pr = p0[path[0]]
        for u, v in zip(path, path[1:]):
            pr *= T[u][v]
        key = tuple(R[x] for x in path)
        out[key] = out.get(key, 0) + pr
    return out

def markov_path_law(p0, n):
    pc = {}
    for x in T:
        pc[R[x]] = pc.get(R[x], 0) + p0[x]
    out = {}
    for path in product(Lam, repeat=n):
        pr = pc[path[0]]
        for u, v in zip(path, path[1:]):
            pr *= Tc[u][v]
        out[path] = pr
    return out

for name, p0 in [("nu Lambda, nu=(1/3,2/3)", {1: Fr(1, 6), 2: Fr(1, 6), 3: Fr(2, 3)}),
                 ("delta_1", {1: Fr(1), 2: Fr(0), 3: Fr(0)})]:
    A_, B_ = retained_path_law(p0, 5), markov_path_law(p0, 5)
    print(f"(A) initial {name}: max path-law difference (length 5) =",
          max(abs(A_.get(k, 0) - B_.get(k, 0)) for k in set(A_) | set(B_)))

# (B) coarsest strongly lumpable refinement
def partitions(s):
    if not s:
        yield []
        return
    first, rest = s[0], s[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p

def is_lumpable(P, part):
    blk = {x: i for i, B in enumerate(part) for x in B}
    for B in part:
        rows = {tuple(sum(P[x][y] for y in C) for C in part) for x in B}
        if len(rows) > 1:
            return False
    return True

def refines(p, q):  # p finer than or equal to q
    blk = {x: i for i, B in enumerate(q) for x in B}
    return all(len({blk[x] for x in B}) == 1 for B in p)

def coarsest_refinement(P, init):
    part = [list(B) for B in init]
    while True:
        new = []
        for B in part:
            groups = {}
            for x in B:
                key = tuple(sum(P[x][y] for y in C) for C in part)
                groups.setdefault(key, []).append(x)
            new.extend(groups.values())
        if len(new) == len(part):
            return sorted(sorted(B) for B in new)
        part = new

random.seed(3)
n = 6
S6 = list(range(n))
# planted lumpable structure on {0,1},{2,3},{4,5} perturbed so only part survives
P = {x: {} for x in S6}
for x in S6:
    w = [Fr(random.randint(1, 5)) for _ in S6]
    P[x] = {y: w[y] / sum(w) for y in S6}
# make 0 and 1 have identical rows, 2 and 3 identical rows
P[1] = dict(P[0]); P[3] = dict(P[2])
init = [[0, 1, 2, 3], [4, 5]]
cr = coarsest_refinement(P, init)
brute = [p for p in partitions(S6) if refines(p, init) and is_lumpable(P, p)]
coarsest = [p for p in brute if all(refines(q, p) for q in brute)]
print("(B) refinement result:", cr, "; brute-force lumpable refinements:", len(brute),
      "; unique coarsest:", [sorted(sorted(B) for B in p) for p in coarsest])

# (C) joint response/next-state kernel
# fine x=(u,v) bits, R(x)=u; fresh xi~Ber(1/2): u'=u xor xi, v' fresh Ber(1/2), O = xi xor v
Sx = list(product((0, 1), repeat=2))
def joint(x):
    u, v = x
    out = {}
    for xi in (0, 1):
        for v2 in (0, 1):
            key = (xi ^ v, u ^ xi)  # (O, R(next))
            out[key] = out.get(key, 0) + Fr(1, 4)
    return out
marg_O = {x: {o: sum(p for (oo, _), p in joint(x).items() if oo == o) for o in (0, 1)} for x in Sx}
marg_R = {x: {r: sum(p for (_, rr), p in joint(x).items() if rr == r) for r in (0, 1)} for x in Sx}
print("(C) response marginal constant on fibers:",
      all(marg_O[x] == marg_O[xp] for x in Sx for xp in Sx if x[0] == xp[0]))
print("(C) next retained marginal constant on fibers:",
      all(marg_R[x] == marg_R[xp] for x in Sx for xp in Sx if x[0] == xp[0]))
print("(C) joint kernel constant on fibers:",
      all(joint(x) == joint(xp) for x in Sx for xp in Sx if x[0] == xp[0]),
      "; joint at (0,0):", joint((0, 0)), " joint at (0,1):", joint((0, 1)))

# (D) stochastic channel square
import numpy as np
from scipy.optimize import linprog
Cj = np.array([[0.75, 0.25], [0.25, 0.75]])
target = np.eye(2)  # K_ij C_i with K_ij = C_i = identity
# unknown K^c (2x2 row stochastic): Cj @ Kc = target
A_eq, b_eq = [], []
for i in range(2):
    for jj in range(2):
        row = np.zeros(4)
        for k in range(2):
            row[2 * k + jj] = Cj[i, k]
        A_eq.append(row); b_eq.append(target[i, jj])
for k in range(2):
    row = np.zeros(4); row[2 * k] = 1; row[2 * k + 1] = 1
    A_eq.append(row); b_eq.append(1.0)
res = linprog(np.zeros(4), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, None)] * 4, method="highs")
print("(D) stochastic K^c exists:", res.status == 0, "; status:", res.message,
      "; unique linear solution Cj^{-1} =", np.linalg.inv(Cj).round(3).tolist())
