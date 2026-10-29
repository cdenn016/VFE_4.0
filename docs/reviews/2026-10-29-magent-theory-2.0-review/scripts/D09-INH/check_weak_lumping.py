r"""D09-INH check: does a coarsest exact parent exist for a single-preparation class?

inh:inherited-parent (D) admits a preparation class P_I smaller than all laws.  This script
ATTEMPTED to build a 4-state chain T, interface partition B0 = {012|3} and one initial law p0
with two incomparable maximal exact refinements P1 = {01|2|3}, P2 = {02|1|3} whose join B0 is
not exact for p0.  RESULT (see .out): the attempt FAILED as a counterexample.  With both
intertwinings Lambda_i proportional to p0 on fibers, the join B0 is also intertwined
(posterior ratios p(x)/p(x') = p0(x)/p0(x') are preserved and multiply along chains of
merged pairs), so p0-proportional intertwinings are closed under joins.  The output is kept
as a recorded negative attempt; it does not show that coarsest exact parents exist for
general restricted preparation classes, which remains unproved in the manuscript.
Construction: the intertwining constraints are linear in T; solve them by LP with random
objectives, then verify with exact enumeration of retained histories (float arithmetic).
"""
import itertools
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(5)
p0 = np.array([0.4, 0.3, 0.2, 0.1])
al = p0[0] / (p0[0] + p0[1])     # Lambda_1(A,.) = (al, 1-al, 0, 0)
be = p0[0] / (p0[0] + p0[2])     # Lambda_2(A',.) = (be, 0, 1-be, 0)

# variables T[x,y], x,y in 0..3 -> index 4x+y
def idx(x, y):
    return 4 * x + y

A_eq, b_eq = [], []
for x in range(4):
    row = np.zeros(16); row[[idx(x, y) for y in range(4)]] = 1; A_eq.append(row); b_eq.append(1.0)
# Lambda_1 intertwining: every row of Lambda_1 T restricted to {0,1} is proportional to (al, 1-al)
#   (1-al) * m[0] - al * m[1] = 0 for m = Lambda_1(w,.) T, w in {A, 2, 3}
L1_rows = [np.array([al, 1 - al, 0, 0]), np.eye(4)[2], np.eye(4)[3]]
for lam in L1_rows:
    row = np.zeros(16)
    for x in range(4):
        row[idx(x, 0)] += (1 - al) * lam[x]
        row[idx(x, 1)] -= al * lam[x]
    A_eq.append(row); b_eq.append(0.0)
L2_rows = [np.array([be, 0, 1 - be, 0]), np.eye(4)[1], np.eye(4)[3]]
for lam in L2_rows:
    row = np.zeros(16)
    for x in range(4):
        row[idx(x, 0)] += (1 - be) * lam[x]
        row[idx(x, 2)] -= be * lam[x]
    A_eq.append(row); b_eq.append(0.0)

best = None
for trial in range(50):
    c = rng.normal(size=16)
    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0.02, 1)] * 16, method='highs')
    if res.status != 0:
        continue
    T = res.x.reshape(4, 4)
    # randomize within the feasible polytope by averaging a few vertices
    if best is None:
        best = T
    else:
        best = 0.5 * best + 0.5 * T
T = best / best.sum(axis=1, keepdims=True)
print("T =\n", np.round(T, 6))

def lam_matrix(part, p0):
    L = np.zeros((len(part), 4))
    for i, b in enumerate(part):
        L[i, b] = p0[b] / p0[b].sum()
    return L

def phi(part):
    F = np.zeros((4, len(part)))
    for i, b in enumerate(part):
        F[b, i] = 1
    return F

def strongly_lumpable(T, part):
    F = phi(part)
    M = T @ F
    return all(np.abs(M[b] - M[b[0]]).max() < 1e-9 for b in part)

def markov_deviation(T, part, p0, depth=5):
    S = 4
    P = [np.diag(phi(part)[:, i]) for i in range(len(part))]
    seen = {}
    dev = 0.0
    for L in range(1, depth + 1):
        for hist in itertools.product(range(len(part)), repeat=L):
            f = p0 @ P[hist[0]]
            for k in range(1, L):
                f = f @ T @ P[hist[k]]
            m = f.sum()
            if m < 1e-14:
                continue
            nxt = np.array([(f @ T @ P[z]).sum() for z in range(len(part))]) / m
            if hist[-1] in seen:
                dev = max(dev, np.abs(seen[hist[-1]] - nxt).max())
            else:
                seen[hist[-1]] = nxt
    return dev

P1 = [[0, 1], [2], [3]]
P2 = [[0, 2], [1], [3]]
P3 = [[1, 2], [0], [3]]
B0 = [[0, 1, 2], [3]]
for name, part in (("P1", P1), ("P2", P2), ("P3", P3), ("B0 = P1 v P2", B0)):
    L = lam_matrix(part, p0)
    F = phi(part)
    Tc = L @ T @ F
    inter = np.abs(L @ T - Tc @ L).max()
    print(f"{name}: |Lambda T - (Lambda T Phi) Lambda| = {inter:.2e}, strongly lumpable = {strongly_lumpable(T, part)},"
          f" max deviation from homogeneous Markov (histories up to length 5) from p0 = {markov_deviation(T, part, p0):.3e}")
