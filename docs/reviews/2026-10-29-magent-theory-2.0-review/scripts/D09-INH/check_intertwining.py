r"""D09-INH check: inh:intertwining (Rogers-Pitman) and inh:finite-lumpability.

Checks, with exact rational arithmetic where possible:
  1. The worked example of chapters/05_inheritance.tex:460-469:
     Lambda T = T^c Lambda, lumpability failure, and the claimed total-variation
     gap "1/10" between retained path laws from delta_1 and the T^c chain
     ("path laws of length five").
  2. The Rogers-Pitman conclusion (conditional law of X_n given the retained
     history equals Lambda(Z_n, .)) by brute-force path enumeration on the
     example and on random 3- and 4-state intertwined chains.
  3. Strong lumpability <=> Markov retained process for every initial law,
     on random 3- and 4-state chains (necessity tested via point masses).
  4. Incompatibility of the intertwining branch of (D) with clause (B) of
     inh:inherited-parent: the next-retained-state kernel is not constant on
     fibers in the worked example.
"""
from fractions import Fraction as Fr
import itertools
import numpy as np

def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

# ---------- 1. worked example ----------
T = [[Fr(3, 10), Fr(5, 10), Fr(2, 10)],
     [Fr(4, 10), Fr(2, 10), Fr(4, 10)],
     [Fr(1, 4), Fr(1, 4), Fr(1, 2)]]
R = {0: 'a', 1: 'a', 2: 'b'}          # states 1,2,3 -> indices 0,1,2
Zs = ['a', 'b']
Lam = [[Fr(1, 2), Fr(1, 2), Fr(0)],
       [Fr(0), Fr(0), Fr(1)]]
Tc = [[Fr(7, 10), Fr(3, 10)], [Fr(1, 2), Fr(1, 2)]]
Phi = [[Fr(1) if R[x] == z else Fr(0) for z in Zs] for x in range(3)]

LT = mat_mul(Lam, T)
TcL = mat_mul(Tc, Lam)
print("1. Lambda T == T^c Lambda:", LT == TcL)
print("   Lambda T Phi_R =", mat_mul(mat_mul(Lam, T), Phi), " (should equal T^c)")
print("   fiber masses into {3}: row1 =", T[0][2], " row2 =", T[1][2], " -> lumpable:", T[0][2] == T[1][2])

def retained_path_law(p0, T, R, Zs, n_steps):
    """Exact law of (Z_0..Z_n) for fine chain with initial law p0 (n_steps transitions)."""
    S = len(T)
    law = {}
    for path in itertools.product(range(S), repeat=n_steps + 1):
        pr = p0[path[0]]
        for k in range(n_steps):
            pr *= T[path[k]][path[k + 1]]
        if pr == 0:
            continue
        key = tuple(R[x] for x in path)
        law[key] = law.get(key, 0) + pr
    return law

def markov_path_law(nu, Tc, Zs, n_steps):
    law = {}
    idx = {z: i for i, z in enumerate(Zs)}
    for path in itertools.product(Zs, repeat=n_steps + 1):
        pr = nu[idx[path[0]]]
        for k in range(n_steps):
            pr *= Tc[idx[path[k]]][idx[path[k + 1]]]
        if pr != 0:
            law[path] = pr
    return law

def tv(l1, l2):
    keys = set(l1) | set(l2)
    return sum(abs(l1.get(k, 0) - l2.get(k, 0)) for k in keys) / 2

p_delta1 = [Fr(1), Fr(0), Fr(0)]
for n in range(1, 7):
    l1 = retained_path_law(p_delta1, T, R, Zs, n)
    l2 = markov_path_law([Fr(1), Fr(0)], Tc, Zs, n)
    print(f"   TV(path law of Z_0..Z_{n} from delta_1, T^c chain) = {tv(l1, l2)} = {float(tv(l1, l2)):.6f}"
          f"   [{n + 1} retained states, {n} transitions]")

# Rogers-Pitman check under nu Lambda on the example
def check_rp(T, R, Zs, Lam, Tc, nu, n_steps, exact=True):
    S = len(T)
    idx = {z: i for i, z in enumerate(Zs)}
    p0 = [sum(nu[k] * Lam[k][x] for k in range(len(Zs))) for x in range(S)]
    # joint mass of history and terminal state
    joint = {}
    for path in itertools.product(range(S), repeat=n_steps + 1):
        pr = p0[path[0]]
        for k in range(n_steps):
            pr *= T[path[k]][path[k + 1]]
        if pr == 0:
            continue
        key = tuple(R[x] for x in path)
        joint.setdefault(key, [0] * S)
        joint[key][path[-1]] += pr
    worst = 0
    for key, f in joint.items():
        tot = sum(f)
        cond = [fx / tot for fx in f]
        target = Lam[idx[key[-1]]]
        err = max(abs(cond[x] - target[x]) for x in range(S))
        worst = max(worst, err)
        # path probability vs Markov chain
        pm = nu[idx[key[0]]]
        for k in range(n_steps):
            pm *= Tc[idx[key[k]]][idx[key[k + 1]]]
        worst = max(worst, abs(pm - tot))
    return worst

for nu in ([Fr(1), Fr(0)], [Fr(0), Fr(1)], [Fr(2, 7), Fr(5, 7)]):
    print("   Rogers-Pitman residual (exact) under nu Lambda, nu =", nu, ":", check_rp(T, R, Zs, Lam, Tc, nu, 5))

# ---------- 4. (B) versus intertwining ----------
next_ret = [[sum(T[x][y] for y in range(3) if R[y] == z) for z in Zs] for x in range(3)]
print("4. next-retained-state kernel on fiber a: x=1 ->", next_ret[0], " x=2 ->", next_ret[1],
      " constant on fiber:", next_ret[0] == next_ret[1])

# ---------- 2. random intertwined 3- and 4-state chains ----------
rng = np.random.default_rng(20261029)

def random_intertwined(S_sizes, rng):
    """Build T with Lambda T = T^c Lambda by construction: choose Lambda, T^c, then
    T(x,.) for x in fiber z must satisfy sum_x Lambda(z,x) T(x,.) = (T^c Lambda)(z,.).
    Choose all but one row of each fiber freely (nonnegative), solve the last row,
    and retry until it is nonnegative."""
    fibers = []
    start = 0
    for s in S_sizes:
        fibers.append(list(range(start, start + s)))
        start += s
    S = start
    Zn = len(S_sizes)
    for _ in range(10000):
        Tc = rng.dirichlet(np.ones(Zn), size=Zn)
        Lam = np.zeros((Zn, S))
        for z, fib in enumerate(fibers):
            Lam[z, fib] = rng.dirichlet(np.ones(len(fib)))
        target = Tc @ Lam
        T = np.zeros((S, S))
        ok = True
        for z, fib in enumerate(fibers):
            *free, last = fib
            acc = np.zeros(S)
            for x in free:
                # perturb the target row
                row = target[z] + 0.25 * rng.normal(size=S) * target[z]
                row = np.clip(row, 0, None)
                row /= row.sum()
                T[x] = row
                acc += Lam[z, x] * row
            row_last = (target[z] - acc) / Lam[z, last]
            if row_last.min() < -1e-12:
                ok = False
                break
            T[last] = np.clip(row_last, 0, None)
            T[last] /= T[last].sum()
        if ok:
            return T, Lam, Tc, fibers
    raise RuntimeError("no sample")

def lumpable(T, fibers):
    for fib in fibers:
        M = np.array([[T[x, f].sum() for f in fibers] for x in fib])
        if np.abs(M - M[0]).max() > 1e-10:
            return False
    return True

def rp_float(T, Lam, Tc, fibers, nu, n_steps):
    S = T.shape[0]
    Rmap = {x: z for z, fib in enumerate(fibers) for x in fib}
    P = [np.diag([1.0 if Rmap[x] == z else 0.0 for x in range(S)]) for z in range(len(fibers))]
    worst = 0.0
    for hist in itertools.product(range(len(fibers)), repeat=n_steps + 1):
        f = (nu @ Lam) @ P[hist[0]]
        for k in range(1, n_steps + 1):
            f = f @ T @ P[hist[k]]
        mass = f.sum()
        pm = nu[hist[0]] * np.prod([Tc[hist[k], hist[k + 1]] for k in range(n_steps)])
        worst = max(worst, abs(mass - pm))
        if mass > 1e-14:
            worst = max(worst, np.abs(f / mass - Lam[hist[-1]]).max())
    return worst

print("2. random intertwined chains (3 and 4 states):")
for sizes in ([2, 1], [2, 2], [3, 1], [1, 2, 1]):
    T_, L_, Tc_, fib = random_intertwined(sizes, rng)
    res = np.abs(L_ @ T_ - Tc_ @ L_).max()
    nu = rng.dirichlet(np.ones(len(sizes)))
    w = rp_float(T_, L_, Tc_, fib, nu, 5)
    print(f"   fibers {sizes}: |Lambda T - T^c Lambda| = {res:.2e}, strongly lumpable = {lumpable(T_, fib)},"
          f" RP residual over all histories of 6 states = {w:.2e}")

# ---------- 3. strong lumpability <=> Markov for every initial law ----------
def markov_test(T, fibers, p0, n_steps=3):
    """Max deviation of P(Z_n+1 | Z_0..Z_n) from P(Z_n+1 | Z_n) over histories (float)."""
    S = T.shape[0]
    Rmap = {x: z for z, fib in enumerate(fibers) for x in fib}
    P = [np.diag([1.0 if Rmap[x] == z else 0.0 for x in range(S)]) for z in range(len(fibers))]
    cond = {}
    for hist in itertools.product(range(len(fibers)), repeat=n_steps):
        f = p0 @ P[hist[0]]
        for k in range(1, n_steps):
            f = f @ T @ P[hist[k]]
        m = f.sum()
        if m < 1e-12:
            continue
        nxt = np.array([(f @ T @ P[z]).sum() for z in range(len(fibers))]) / m
        cond.setdefault(hist[-1], []).append(nxt)
    dev = 0.0
    for z, rows in cond.items():
        rows = np.array(rows)
        dev = max(dev, np.abs(rows - rows[0]).max())
    return dev

print("3. strong lumpability vs Markov retained process (point-mass initial laws):")
for sizes in ([2, 1], [2, 2], [3, 1]):
    fibers = []
    s0 = 0
    for s in sizes:
        fibers.append(list(range(s0, s0 + s)))
        s0 += s
    S = s0
    # lumpable chain: choose block masses per fiber then distribute
    Tc = rng.dirichlet(np.ones(len(sizes)), size=len(sizes))
    TL = np.zeros((S, S))
    for z, fib in enumerate(fibers):
        for x in fib:
            for w, fw in enumerate(fibers):
                TL[x, fw] = Tc[z, w] * rng.dirichlet(np.ones(len(fw)))
    TN = rng.dirichlet(np.ones(S), size=S)  # generic, not lumpable
    devL = max(markov_test(TL, fibers, np.eye(S)[x]) for x in range(S))
    devN = max(markov_test(TN, fibers, np.eye(S)[x]) for x in range(S))
    # one-step necessity: point masses in the same fiber give different next-block laws
    print(f"   fibers {sizes}: lumpable chain max Markov deviation {devL:.2e} (lumpable={lumpable(TL, fibers)});"
          f" generic chain one-step fiber discrepancy "
          f"{max(np.abs(np.array([[TN[x, f].sum() for f in fibers] for x in fib]) - np.array([TN[fib[0], f].sum() for f in fibers])).max() for fib in fibers):.3f}"
          f" (lumpable={lumpable(TN, fibers)})")
