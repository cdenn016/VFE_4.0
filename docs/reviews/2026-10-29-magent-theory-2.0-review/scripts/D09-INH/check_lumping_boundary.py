r"""D09-INH check: inh:canonical-lumping, boundary-response examples, separator examples,
channel-square example, and the s04 filter example.

  1. Partition refinement (inh:canonical-lumping) versus brute force over all set partitions
     of 3-, 4- and 5-state chains: the refinement output equals the unique coarsest strongly
     lumpable partition refining the interface partition.
  2. Weak lumpability (closure for one preparation): exhibit a 4-state chain and initial law
     with two different maximal (incomparable) partitions refining the interface partition
     under which the retained process is Markov, so no unique coarsest exact parent exists
     for a single-preparation class.
  3. Joint response/transition example (lines 557-563).
  4. XOR example and four-cycle infeasibility (lines 611-622) by LP.
  5. BSC(1/4) garbling (lines 640-644): unique linear solution and LP infeasibility.
  6. s04 deterministic-cycle filter example (s04 lines 114-124).
"""
import itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(11)

def set_partitions(elements):
    if not elements:
        yield []
        return
    first, rest = elements[0], elements[1:]
    for smaller in set_partitions(rest):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]
        yield [[first]] + smaller

def is_lumpable(T, part, tol=1e-12):
    for blk in part:
        rows = np.array([[T[x, b].sum() for b in part] for x in blk])
        if np.abs(rows - rows[0]).max() > tol:
            return False
    return True

def refines(p, q):
    """p refines q."""
    return all(any(set(b) <= set(c) for c in q) for b in p)

def canon(part):
    return tuple(sorted(tuple(sorted(b)) for b in part))

def refinement(T, part0):
    part = [list(b) for b in part0]
    while True:
        new = []
        for blk in part:
            groups = {}
            for x in blk:
                key = tuple(np.round([T[x, b].sum() for b in part], 12))
                groups.setdefault(key, []).append(x)
            new.extend(groups.values())
        if len(new) == len(part):
            return new
        part = new

def make_partially_lumpable(n, rng):
    """Random chain that is lumpable for a random partition (so nontrivial answers occur)."""
    elems = list(range(n))
    parts = list(set_partitions(elems))
    target = parts[rng.integers(len(parts))]
    Tc = rng.dirichlet(np.ones(len(target)), size=len(target))
    T = np.zeros((n, n))
    for i, blk in enumerate(target):
        for x in blk:
            for j, b2 in enumerate(target):
                T[x, b2] = Tc[i, j] * rng.dirichlet(np.ones(len(b2)))
    return T

agree = 0; total = 0
for n in (3, 4, 5):
    for trial in range(60):
        T = make_partially_lumpable(n, rng)
        # interface: random labeling
        labels = rng.integers(0, 2, size=n)
        part0 = [[x for x in range(n) if labels[x] == l] for l in set(labels)]
        cands = [p for p in set_partitions(list(range(n))) if refines(p, part0) and is_lumpable(T, p)]
        coarsest = [p for p in cands if all(refines(q, p) for q in cands)]
        ref = refinement(T, part0)
        total += 1
        if len(coarsest) == 1 and canon(coarsest[0]) == canon(ref):
            agree += 1
print(f"1. partition refinement equals the unique coarsest lumpable refinement in {agree}/{total} random cases (n = 3,4,5)")

# ---------- 2. weak lumpability: non-unique maximal partitions ----------
def retained_markov_for_law(T, part, p0, n_steps=4, tol=1e-10):
    S = T.shape[0]
    idx = {x: i for i, b in enumerate(part) for x in b}
    P = [np.diag([1.0 if idx[x] == i else 0.0 for x in range(S)]) for i in range(len(part))]
    seen = {}
    for L in range(1, n_steps + 1):
        for hist in itertools.product(range(len(part)), repeat=L):
            f = p0 @ P[hist[0]]
            for k in range(1, L):
                f = f @ T @ P[hist[k]]
            m = f.sum()
            if m < 1e-13:
                continue
            nxt = np.array([(f @ T @ P[z]).sum() for z in range(len(part))]) / m
            # Markov with time-homogeneous kernel: depends only on last block
            if hist[-1] in seen:
                if np.abs(seen[hist[-1]] - nxt).max() > tol:
                    return False
            else:
                seen[hist[-1]] = nxt
    return True

found = None
for trial in range(4000):
    n = 4
    T = rng.dirichlet(np.ones(n), size=n)
    # make states 0,1 and 2,3 each "weakly mergeable" from a doubly-structured chain:
    # use a chain with a symmetry swapping (0,1) and a different symmetry swapping (2,3)
    a, b, c, d = rng.dirichlet(np.ones(4)), None, None, None
    p0 = np.full(n, 0.25)
    part0 = [[0, 1, 2, 3]]
    cands = []
    for p in set_partitions(list(range(n))):
        if len(p) in (2, 3) and retained_markov_for_law(T, p, p0):
            cands.append(p)
    maxl = [p for p in cands if not any(refines(p, q) and canon(p) != canon(q) for q in cands)]
    if len(maxl) >= 2:
        found = (T, maxl)
        break
if found is None:
    # deterministic construction: T invariant under the permutation (0 1) and under (2 3)
    # separately is lumpable; instead use a chain with two different 2-cycle symmetries
    # fixing p0 that do not commute: sigma=(0 1)(2 3) and tau=(0 2)(1 3).
    base = rng.dirichlet(np.ones(4), size=4)
    perms = [np.eye(4)[[1, 0, 3, 2]], np.eye(4)[[2, 3, 0, 1]], np.eye(4)[[3, 2, 1, 0]]]
    T = base.copy()
    for Pm in perms:
        T = T + Pm @ base @ Pm.T
    T = T / T.sum(axis=1, keepdims=True)
    p0 = np.full(4, 0.25)
    cands = [p for p in set_partitions([0, 1, 2, 3]) if 1 < len(p) < 4 and retained_markov_for_law(T, p, p0)]
    maxl = [p for p in cands if not any(refines(p, q) and canon(p) != canon(q) for q in cands)]
    strongly = [canon(p) for p in cands if is_lumpable(T, p)]
    found = (T, maxl)
    print("2. Klein-four-symmetric 4-state chain, uniform initial law (fixed by the symmetry):")
    print("   partitions (2 or 3 blocks) with Markov retained process for this law:", [canon(p) for p in cands])
    print("   maximal ones:", [canon(p) for p in maxl], " strongly lumpable among them:", strongly)
    joins = set()
    for p, q in itertools.combinations(maxl, 2):
        # join of two partitions = finest common coarsening
        blocks = [set(b) for b in p] + [set(b) for b in q]
        merged = True
        while merged:
            merged = False
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    if blocks[i] & blocks[j]:
                        blocks[i] |= blocks.pop(j); merged = True; break
                if merged:
                    break
        joins.add(tuple(sorted(tuple(sorted(b)) for b in blocks)))
    print("   joins of pairs of maximal partitions:", joins, "(the trivial one-block partition carries no interface)")
else:
    T, maxl = found
    print("2. random 4-state chain with uniform initial law; maximal weakly-Markov partitions:", [canon(p) for p in maxl])

# ---------- 3. joint response/transition ----------
def joint(u, v):
    out = {}
    for xi in (0, 1):
        for v2 in (0, 1):
            O = xi ^ v; u2 = u ^ xi
            out[(O, u2)] = out.get((O, u2), 0) + Fr(1, 4)
    # marginalize v2 already (it is summed); return law of (O, u')
    return out
for x in ((0, 0), (0, 1), (1, 0), (1, 1)):
    j = joint(*x)
    mO = {o: sum(p for (oo, uu), p in j.items() if oo == o) for o in (0, 1)}
    mU = {u2: sum(p for (oo, uu), p in j.items() if uu == u2) for u2 in (0, 1)}
    print(f"3. x={x}: joint law of (O,u') = { {k: v for k, v in j.items() if v} }, marginals O {mO}, u' {mU}")

# ---------- 4. XOR and four-cycle ----------
tot = 0
for U, V in itertools.product((0, 1), repeat=2):
    Wb = U ^ V
    tot += Fr(1, 4) * (U == 1) * (V == 1) * (Wb == 1)
print("4. E[1{U=1}1{V=1}1{W=1}] =", tot, " vs F_U F_V F_W =", Fr(1, 8))
# four-cycle LP: variables p(A,B,C,D) >= 0, constraints on edge marginals
cfg = list(itertools.product((0, 1), repeat=4))
Aeq, beq = [], []
def edge(i, j, law):
    for vi, vj in itertools.product((0, 1), repeat=2):
        Aeq.append([1.0 if (c[i] == vi and c[j] == vj) else 0.0 for c in cfg]); beq.append(law(vi, vj))
eq = lambda a, b: 0.5 if a == b else 0.0
ne = lambda a, b: 0.5 if a != b else 0.0
edge(0, 1, eq); edge(1, 2, eq); edge(2, 3, eq); edge(3, 0, ne)
res = linprog(np.zeros(len(cfg)), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * len(cfg), method='highs')
print("   four-cycle joint-law LP status:", res.status, "(2 = infeasible)", res.message)
# same LP on a path A-B-C-D (decomposable) with the same three 'equal' edges and a fourth edge removed: feasible
Aeq, beq = [], []
edge(0, 1, eq); edge(1, 2, eq); edge(2, 3, eq)
res2 = linprog(np.zeros(len(cfg)), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * len(cfg), method='highs')
print("   path A-B-C-D (running intersection) LP status:", res2.status, "(0 = feasible)")

# ---------- 5. BSC(1/4) ----------
Cj = np.array([[0.75, 0.25], [0.25, 0.75]])
print("5. unique linear solution C_j^{-1} =", np.linalg.inv(Cj).tolist())
# LP: find Markov M with Cj M = I
Aeq, beq = [], []
for i in range(2):
    for k in range(2):
        row = np.zeros(4)
        for j in range(2):
            row[j * 2 + k] = Cj[i, j]
        Aeq.append(row); beq.append(1.0 if i == k else 0.0)
for j in range(2):
    row = np.zeros(4); row[j * 2:(j + 1) * 2] = 1; Aeq.append(row); beq.append(1.0)
res3 = linprog(np.zeros(4), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=[(0, None)] * 4, method='highs')
print("   Markov garbling LP status:", res3.status, "(2 = infeasible)")

# ---------- 6. s04 cycle filter ----------
Tcyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
Rc = [0, 0, 1]
P = [np.diag([1.0 if Rc[x] == z else 0.0 for x in range(3)]) for z in (0, 1)]
p0 = np.full(3, 1 / 3)
for hist in ((0, 0), (1, 0)):
    f = p0 @ P[hist[0]] @ Tcyc @ P[hist[1]]
    ph = f / f.sum()
    nxt = [(ph @ Tcyc @ P[z]).sum() for z in (0, 1)]
    print(f"6. history {hist}: prob {f.sum():.4f}, filter {ph.round(3).tolist()} (states 1,2,3), next retained law {nxt}")
