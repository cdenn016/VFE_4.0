r"""D10-FORM check 2: memory and formation; statistical partitions (ch05 lines 1010-1279).

  (a) inh:causal-memory: KL(P||Ptilde) = sum_t I(Z_t;H_t|M_t) + sum_t E KL(P^M_t || Ttilde_t)
      on a random law of three ternary records with lossy statistics (exact enumeration).
  (b) reverse-channel tautology: zeta_I K_{iI} = zeta_i for the exact conditional.
  (c) filter support for deterministic dynamics: consistent-initial-set shrinkage on a
      finite example; the filter becomes a point mass only once the consistent set is a point.
  (d) inh:finest-independent / inh:independence-discontinuity: brute force over all
      partitions for the Ising chain, N = 4 and N = 6; conditional odds ratio e^{4J_e}.
  (e) inh:conditional-partition: example with C = Y_V and with C constant.
  (f) inh:selector-symmetry: invariant partitions of D_4 and S_4 on four points.
  (g) covariance formula for x' = A x, x(0) ~ N(0, I).
  (h) doubling map F(x) = 2x mod 1 observed through R(x) = 1{x >= 1/2} with a uniform prior:
      the consistent initial set shrinks to a point (observability class of a non-dyadic x0 is a
      point), but the filter on the current state X_n is uniform on a half interval at every n.
"""
from __future__ import annotations

import itertools as it
import math

import numpy as np

rng = np.random.default_rng(7)
out: list[str] = []

# (a) causal memory identity on three ternary records
nZ = 3
P = rng.dirichlet(np.ones(nZ**3)).reshape(nZ, nZ, nZ)
f1 = lambda h: 0  # statistic at t=1: constant (forgets Z0)
f2 = lambda h: (h[0] + h[1]) % 2  # lossy statistic at t=2
stats = [lambda h: 0, lambda h: f1(h), lambda h: f2(h)]
Ttil = [None, None, None]
for t in range(3):
    nm = 2
    Ttil[t] = rng.dirichlet(np.ones(nZ), size=nm)


def path_prob_tilde(path):
    p = 1.0
    for t in range(3):
        mval = stats[t](path[:t])
        p *= Ttil[t][mval][path[t]]
    return p


KL = 0.0
for path in it.product(range(nZ), repeat=3):
    pp = P[path]
    KL += pp * math.log(pp / path_prob_tilde(path))


def marg(prefix_len):
    return P.sum(axis=tuple(range(prefix_len, 3))) if prefix_len < 3 else P


rhs = 0.0
for t in range(3):
    # joint law of (H_t, Z_t)
    J = P.sum(axis=tuple(range(t + 1, 3))) if t + 1 < 3 else P  # shape nZ^(t+1)
    # conditional mutual information I(Z_t; H_t | M_t)
    pm_z = {}
    pm = {}
    for hz in it.product(range(nZ), repeat=t + 1):
        h, zt = hz[:t], hz[t]
        mval = stats[t](h)
        pm_z[(mval, zt)] = pm_z.get((mval, zt), 0.0) + J[hz]
        pm[mval] = pm.get(mval, 0.0) + J[hz]
    ph = {}
    for hz in it.product(range(nZ), repeat=t + 1):
        h = hz[:t]
        ph[h] = ph.get(h, 0.0) + J[hz]
    cmi = 0.0
    for hz in it.product(range(nZ), repeat=t + 1):
        h, zt = hz[:t], hz[t]
        mval = stats[t](h)
        p_hz = J[hz]
        cmi += p_hz * math.log(p_hz * pm[mval] / (ph[h] * pm_z[(mval, zt)]))
    pred = 0.0
    for (mval, zt), pmz in pm_z.items():
        pred += pmz * math.log((pmz / pm[mval]) / Ttil[t][mval][zt])
    rhs += cmi + pred
out.append(f"(a) causal-memory identity: KL={KL:.15f}, rhs={rhs:.15f}, residual={KL-rhs:.2e}")

# (b) reverse-channel tautology: zeta_I = law of statistic S=(Y1+Y2) of independent Y1,Y2; K_{1I} = law(Y1|S)
z1 = rng.dirichlet(np.ones(3))
z2 = rng.dirichlet(np.ones(3))
joint = np.outer(z1, z2)
S = {}
for y1, y2 in it.product(range(3), repeat=2):
    S.setdefault(y1 + y2, []).append((y1, y2))
zetaI = {s: sum(joint[p] for p in S[s]) for s in S}
K1 = {s: np.array([sum(joint[(y1, y2)] for (y1, y2) in S[s] if y1 == v) for v in range(3)]) / zetaI[s] for s in S}
back = sum(zetaI[s] * K1[s] for s in S)
out.append(f"(b) zeta_I K_1I - zeta_1 max residual: {np.max(np.abs(back - z1)):.2e}")

# (c) deterministic filter on a finite state space: F a permutation, R a coarse readout
Sn = 6
Fperm = np.array([1, 2, 0, 4, 5, 3])  # two 3-cycles
Rmap = np.array([0, 0, 1, 0, 1, 1])
x0 = 0
mu0 = np.full(Sn, 1 / Sn)
cons = [x for x in range(Sn)]
hist = []
xs = x0
supports = []
for n in range(5):
    hist.append(Rmap[xs])

    def orbit(x, k):
        for _ in range(k):
            x = Fperm[x]
        return x
    cons = [x for x in range(Sn) if all(Rmap[orbit(x, k)] == hist[k] for k in range(n + 1))]
    supports.append(sorted({orbit(x, n) for x in cons}))
    xs = Fperm[xs]
out.append(f"(c) deterministic 6-state example: consistent current-state supports by step: {supports}")

# (d) Ising chain finest independent partition, brute force
def set_partitions(seq):
    if not seq:
        yield []
        return
    first, rest = seq[0], seq[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def ising(N, Jc, eps):
    states = list(it.product((-1, 1), repeat=N))
    w = []
    for s in states:
        e = Jc * sum(s[2 * b] * s[2 * b + 1] for b in range(N // 2)) + eps * sum(s[2 * b + 1] * s[2 * b + 2] for b in range(N // 2 - 1))
        w.append(math.exp(e))
    Z = sum(w)
    return {s: wi / Z for s, wi in zip(states, w)}


def is_independent(Pd, N, part, tol=1e-12):
    margs = []
    for B in part:
        mB = {}
        for s, p in Pd.items():
            key = tuple(s[i] for i in B)
            mB[key] = mB.get(key, 0.0) + p
        margs.append((B, mB))
    for s, p in Pd.items():
        prod = 1.0
        for B, mB in margs:
            prod *= mB[tuple(s[i] for i in B)]
        if abs(prod - p) > tol:
            return False
    return True


def finest(Pd, N):
    indep = [sorted(sorted(B) for B in part) for part in set_partitions(list(range(N))) if is_independent(Pd, N, part)]
    # common refinement of all independent partitions
    blocks = {i: frozenset(range(N)) for i in range(N)}
    for part in indep:
        for B in part:
            for i in B:
                blocks[i] = blocks[i] & frozenset(B)
    return sorted({tuple(sorted(b)) for b in blocks.values()}), len(indep)


for N in (4, 6):
    for eps in (0.0, 1e-3, 0.3):
        fin, nind = finest(ising(N, 0.7, eps), N)
        out.append(f"(d) N={N}, J=0.7, eps={eps}: finest independent partition {fin} ({nind} independent partitions)")
# conditional odds ratio across the cut edge (1,2) for N=4, eps=0.3
Pd = ising(4, 0.7, 0.3)
rest = (1, -1)
ratio = (Pd[(rest[0], 1, 1, rest[1])] * Pd[(rest[0], -1, -1, rest[1])]) / (Pd[(rest[0], 1, -1, rest[1])] * Pd[(rest[0], -1, 1, rest[1])])
out.append(f"(d) conditional odds ratio across edge (2,3) given others: {ratio:.12f} vs e^(4*0.3)={math.exp(1.2):.12f}")

# (e) conditional partition: C = Y_V makes every partition conditionally independent (conditional law is a point mass)
out.append("(e) C = Y_V: conditional laws are point masses, which are products for every partition (direct)")

# (f) invariant partitions of D_4 and S_4
def act(perm, part):
    return sorted(sorted(perm[i] for i in B) for B in part)


r = (1, 2, 3, 0)
sref = (0, 3, 2, 1)
def compose(p, q_):
    return tuple(p[q_[i]] for i in range(4))
D4 = {(0, 1, 2, 3)}
frontier = [r, sref]
while frontier:
    g = frontier.pop()
    if g in D4:
        continue
    D4.add(g)
    for h in list(D4):
        frontier += [compose(g, h), compose(h, g)]
S4 = set(it.permutations(range(4)))
allparts = [sorted(sorted(B) for B in p) for p in set_partitions([0, 1, 2, 3])]
inv_D4 = [p for p in allparts if all(act(g, p) == p for g in D4)]
inv_S4 = [p for p in allparts if all(act(g, p) == p for g in S4)]
out.append(f"(f) |D4|={len(D4)}; D4-invariant partitions of the 4-cycle 0-1-2-3: {inv_D4}")
out.append(f"(f) S4-invariant partitions: {inv_S4}")

# (g) covariance for x' = A x
lam, t = 0.37, 0.8
A = np.array([[-1 - lam, lam], [lam, -1 - lam]])
w_, V_ = np.linalg.eigh(A)
eAt = V_ @ np.diag(np.exp(w_ * t)) @ V_.T
cov = eAt @ eAt.T
formula = 0.5 * (math.exp(-2 * t) - math.exp(-2 * (1 + 2 * lam) * t))
out.append(f"(g) Cov(x1,x2)(t) = {cov[0,1]:.15f}; formula {formula:.15f}; residual {cov[0,1]-formula:.2e}")

# (h) doubling map, exact dyadic interval arithmetic
from fractions import Fraction as Fr

x0 = Fr(1, 3)  # non-dyadic; binary 0.010101...
lo, hi = Fr(0), Fr(1)  # consistent initial set C_n = [lo, hi)
xs_ = x0
rows = []
for n in range(8):
    z = 1 if xs_ >= Fr(1, 2) else 0  # record z_n = R(F^n x0)
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if z == 1 else (lo, mid)  # C_n: initial points with the same n+1 records
    img_len = (hi - lo) * 2**n  # F^n is affine with slope 2^n on the dyadic interval C_n
    rows.append((n, str(hi - lo), str(img_len)))
    xs_ = (2 * xs_) % 1
out.append("(h) doubling map: (n, |C_n|, |F^n(C_n)|) = " + str(rows)
           + "; the filter on X_n is uniform on F^n(C_n), an interval of length 1/2, at every n")

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
