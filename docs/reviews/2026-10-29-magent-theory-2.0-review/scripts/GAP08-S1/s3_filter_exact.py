r"""GAP08-S1 script 3: supplements/s04_inheritance.tex:61-147 (sinh:filter and the tower identity),
exact rational arithmetic (fractions.Fraction), by path enumeration rather than by the matrix formula.

(a) random 5-state chain with zeros in T, retained alphabet of size 3, horizon N = 5 (5^6 paths):
    for every history h_n (n <= 3) of positive probability compare
      - the conditional law of X_n given h_n from path enumeration with f_h/(f_h 1) (sinh:filter-law)
      - P(Z_{n+1} = z | h_n) with p_hat T P_z 1 (sinh:next-record)
      - the recursion (sinh:filter-update)
      - every future word of length 2 with p_hat T P_w1 T P_w2 1
    and check that zero-probability histories have zero-mass extensions.
(b) Markov property of the filter process: a chain built so that distinct histories share a filter;
    the conditional law of the next filter value given the history depends on the history only
    through the current filter value (checked by enumeration, not by the formula).
(c) the deterministic three-cycle example of s04:115-124.
(d) the tower identity sinh:conditional-tower on a random finite joint law.
(e) a non-Markov hidden process (X_{n+1} = X_{n-1}): the terminal conditional law does not
    determine the next prediction, so the general sentence at chapters/05_inheritance.tex:1013-1015
    needs the Markov hypothesis that the supplement states.
"""
from fractions import Fraction as F
from itertools import product
import random

random.seed(3)


def rand_stoch_row(n, zero_prob=0.3):
    while True:
        raw = [0 if random.random() < zero_prob else random.randint(1, 6) for _ in range(n)]
        if sum(raw) > 0:
            s = sum(raw)
            return [F(r, s) for r in raw]


def vecmat(v, M):
    n = len(M[0])
    return [sum(v[k] * M[k][j] for k in range(len(v))) for j in range(n)]


def select(v, Rmap, z):
    return [v[x] if Rmap[x] == z else F(0) for x in range(len(v))]


def run_chain(S, Zs, T, p0, Rmap, N, label):
    paths = {}
    for xs in product(range(S), repeat=N + 1):
        pr = p0[xs[0]]
        for k in range(N):
            pr *= T[xs[k]][xs[k + 1]]
            if pr == 0:
                break
        if pr != 0:
            paths[xs] = pr
    worst = {"law": 0, "next": 0, "update": 0, "word2": 0}
    nhist = 0
    zero_hist_bad = 0
    for n in range(0, N - 1):
        for h in product(range(Zs), repeat=n + 1):
            # enumeration: joint mass of (h, X_n = x)
            joint = [F(0)] * S
            for xs, pr in paths.items():
                if all(Rmap[xs[k]] == h[k] for k in range(n + 1)):
                    joint[xs[n]] += pr
            mass = sum(joint)
            # matrix formula
            f = select(p0, Rmap, h[0])
            for k in range(1, n + 1):
                f = select(vecmat(f, T), Rmap, h[k])
            if mass == 0:
                # every extension must have zero mass
                for z in range(Zs):
                    fz = select(vecmat(f, T), Rmap, z)
                    if sum(fz) != 0 or any(c != 0 for c in f):
                        zero_hist_bad += 1
                continue
            nhist += 1
            phat = [c / sum(f) for c in f]
            law = [c / mass for c in joint]
            worst["law"] = max(worst["law"], max(abs(a - b) for a, b in zip(phat, law)))
            for z in range(Zs):
                # enumeration of P(Z_{n+1} = z | h)
                num = sum(pr for xs, pr in paths.items()
                          if all(Rmap[xs[k]] == h[k] for k in range(n + 1)) and Rmap[xs[n + 1]] == z)
                pz = sum(select(vecmat(phat, T), Rmap, z))
                worst["next"] = max(worst["next"], abs(num / mass - pz))
                if num > 0:
                    joint2 = [F(0)] * S
                    for xs, pr in paths.items():
                        if all(Rmap[xs[k]] == h[k] for k in range(n + 1)) and Rmap[xs[n + 1]] == z:
                            joint2[xs[n + 1]] += pr
                    upd = select(vecmat(phat, T), Rmap, z)
                    upd = [c / sum(upd) for c in upd]
                    worst["update"] = max(worst["update"], max(abs(a - b / num) for a, b in zip(upd, joint2)))
                for z2 in range(Zs):
                    num2 = sum(pr for xs, pr in paths.items()
                               if all(Rmap[xs[k]] == h[k] for k in range(n + 1))
                               and Rmap[xs[n + 1]] == z and Rmap[xs[n + 2]] == z2)
                    v = select(vecmat(select(vecmat(phat, T), Rmap, z), T), Rmap, z2)
                    worst["word2"] = max(worst["word2"], abs(num2 / mass - sum(v)))
    print(f"[{label}] positive-probability histories checked: {nhist}; max exact discrepancies: "
          f"law {worst['law']}, next-record {worst['next']}, update {worst['update']}, "
          f"length-2 words {worst['word2']}; zero-history violations {zero_hist_bad}")
    return paths


print("== (a) random chain ==")
S, Zs, N = 5, 3, 5
T = [rand_stoch_row(S) for _ in range(S)]
p0 = rand_stoch_row(S, zero_prob=0.0)
Rmap = [0, 1, 2, 0, 1]
print("T =", [[str(c) for c in row] for row in T])
run_chain(S, Zs, T, p0, Rmap, N, "random")

print("\n== (b) Markov property of the filter process ==")
# states 0,1 in fiber 0; 2,3 in fiber 1. Rows of T equal within each fiber, so the filter after
# a history depends only on the last retained value: many histories share a filter.
rowA = [F(1, 6), F(1, 3), F(1, 4), F(1, 4)]
rowB = [F(1, 2), F(1, 8), F(1, 8), F(1, 4)]
T2 = [rowA, rowA, rowB, rowB]
p02 = [F(1, 4)] * 4
R2 = [0, 0, 1, 1]
paths2 = run_chain(4, 2, T2, p02, R2, 5, "fiber-constant rows")


def filt(h):
    f = select(p02, R2, h[0])
    for k in range(1, len(h)):
        f = select(vecmat(f, T2), R2, h[k])
    s = sum(f)
    return tuple(c / s for c in f) if s != 0 else None


groups = {}
n = 3
for h in product(range(2), repeat=n + 1):
    ph = filt(h)
    if ph is None:
        continue
    # enumeration: law of the next filter value given h
    mass = sum(pr for xs, pr in paths2.items() if all(R2[xs[k]] == h[k] for k in range(n + 1)))
    nxt = {}
    for z in range(2):
        num = sum(pr for xs, pr in paths2.items()
                  if all(R2[xs[k]] == h[k] for k in range(n + 1)) and R2[xs[n + 1]] == z)
        if num > 0:
            nxt[filt(h + (z,))] = nxt.get(filt(h + (z,)), F(0)) + num / mass
    groups.setdefault(ph, []).append((h, tuple(sorted(nxt.items()))))
viol = 0
sizes = []
for ph, lst in groups.items():
    sizes.append(len(lst))
    if len(set(x[1] for x in lst)) != 1:
        viol += 1
print(f"histories of length {n + 1} grouped by filter value: group sizes {sorted(sizes)}; "
      f"groups whose next-filter law differs across histories: {viol}")

print("\n== (c) deterministic three-cycle, uniform start ==")
Tc = [[F(0), F(1), F(0)], [F(0), F(0), F(1)], [F(1), F(0), F(0)]]  # 1->2->3->1 as 0->1->2->0
p0c = [F(1, 3)] * 3
Rc = [0, 0, 1]
run_chain(3, 2, Tc, p0c, Rc, 5, "three-cycle (has zero-probability histories)")
for h in [(0, 0), (1, 0)]:
    f = select(p0c, Rc, h[0])
    f = select(vecmat(f, Tc), Rc, h[1])
    ph = [c / sum(f) for c in f]
    pz = [sum(select(vecmat(ph, Tc), Rc, z)) for z in range(2)]
    print(f"history {h}: probability {sum(f)}, terminal law {[str(c) for c in ph]} (states 1,2,3), "
          f"next-record law {[str(c) for c in pz]}")

print("\n== (d) sinh:conditional-tower on a random finite joint law ==")
Xs, Os = 6, 3
pj = [[F(random.randint(1, 9)) for _ in range(Os)] for _ in range(Xs)]
tot = sum(sum(r) for r in pj)
pj = [[c / tot for c in r] for r in pj]
R10 = [0, 0, 1, 1, 2, 3]
R21 = [0, 1, 1, 0]
g = [F(2), F(-1), F(5, 3)]
EgZ = {}
for zval in set(R10):
    num = sum(pj[x][o] * g[o] for x in range(Xs) for o in range(Os) if R10[x] == zval)
    den = sum(pj[x][o] for x in range(Xs) for o in range(Os) if R10[x] == zval)
    EgZ[zval] = (num / den, den)
maxdiff = 0
for wval in set(R21):
    lhs_num = sum(pj[x][o] * g[o] for x in range(Xs) for o in range(Os) if R21[R10[x]] == wval)
    lhs_den = sum(pj[x][o] for x in range(Xs) for o in range(Os) if R21[R10[x]] == wval)
    rhs = sum(EgZ[z][0] * EgZ[z][1] for z in EgZ if R21[z] == wval) / lhs_den
    maxdiff = max(maxdiff, abs(lhs_num / lhs_den - rhs))
print("max |E[g(O)|W] - E[E[g(O)|Z]|W]| =", maxdiff)

print("\n== (e) non-Markov hidden process: X_{n+1} = X_{n-1}, retention R = identity ==")
# X_0, X_1 iid fair bits, X_{n+1} = X_{n-1}; enumerate the joint law of (X_0, X_1, X_2)
law3 = {(x0, x1, x0): F(1, 4) for x0 in (0, 1) for x1 in (0, 1)}
for h in [(0, 0), (1, 0)]:
    mass = sum(pr for xs, pr in law3.items() if xs[:2] == h)
    term = {}
    nxt = {}
    for xs, pr in law3.items():
        if xs[:2] == h:
            term[xs[1]] = term.get(xs[1], F(0)) + pr / mass
            nxt[xs[2]] = nxt.get(xs[2], F(0)) + pr / mass
    print(f"history {h}: probability {mass}, terminal conditional law of X_1 = "
          f"{ {k: str(v) for k, v in term.items()} }, law of the next record X_2 = "
          f"{ {k: str(v) for k, v in nxt.items()} }")
print("same terminal conditional law, different next-record laws: the current conditional law "
      "is not sufficient for prediction without the Markov property")
