r"""D12-SMOOTH check 3: equimeasurability versus symmetry of the private reference.

Chapter 7 intro (07:7) says the spectrum "identifies distinct conditionals only when the
reference has nontrivial symmetry"; companion c08:42 says "The compression is the symmetry of
the reference". Section 3 (07:153) states the correct finite criterion (equal-mass atom sets).
(a) A = (0.3, 0.2, 0.1, 0.4): no nontrivial A-preserving permutation, yet two distinct
    conditionals have equal spectra (atoms {1,3} and {4} both have mass 0.4).
(b) Dynamical content: receiver-reconstruction fine flow (model from check 1) with these two
    conditionals at the two public values gives the same public trajectory as the product
    configuration (smooth:equal-spectra), although no symmetry relates them.
(c) A = (1/2, 1/4, 1/4): equal spectra not related by the only nontrivial automorphism.
(d) Finite criterion: equal-mass disjoint atom sets give a constructive noninjective pair.
"""
import itertools
import numpy as np

src = open(__file__.replace("d12_spectrum_symmetry.py", "d12_closure_flow.py")).read()
exec(src.split('print("=" * 70)\nprint("Part A')[0])   # Model, fine_traj, ... (rng reseeded)


def spectrum(ell, A, nd=12):
    d = {}
    for r, m in zip(np.round(ell, nd), A):
        d[r] = d.get(r, 0.0) + m
    return sorted((k, round(v, nd)) for k, v in d.items())


def automorphisms(A, tol=1e-12):
    n = len(A)
    return [p for p in itertools.permutations(range(n)) if np.allclose(A[list(p)], A, atol=tol)]


print("(a) A = (0.3, 0.2, 0.1, 0.4)")
A = np.array([0.3, 0.2, 0.1, 0.4])
aut = automorphisms(A)
print("    A-preserving permutations:", aut)
ex, ey = 1.5, 0.5
ew = (1 - 0.4 * ex - 0.4 * ey) / 0.2          # normalization 0.4 e^x + 0.4 e^y + 0.2 e^w = 1
ell = np.log(np.array([ex, ew, ex, ey]))
ellp = np.log(np.array([ey, ew, ey, ex]))
k, kp = A * np.exp(ell), A * np.exp(ellp)
print("    kappa  =", np.round(k, 6), " sum", round(k.sum(), 12))
print("    kappa2 =", np.round(kp, 6), " sum", round(kp.sum(), 12))
print("    spectrum(kappa)  =", spectrum(ell, A))
print("    spectrum(kappa2) =", spectrum(ellp, A))
print("    equal spectra:", spectrum(ell, A) == spectrum(ellp, A), "; distinct conditionals:", not np.allclose(k, kp))
orbit_hit = any(np.allclose(ellp, ell[list(p)]) for p in aut)
print("    kappa2 in the Aut(V,A)-orbit of kappa:", orbit_hit)

print("(b) fine Fisher flow, receiver reconstruction, 2 agents, public bit")
MB = Model(N=2, nZ=2, nV=4, nW=2, nU=2, Tq=0.8, Ts=0.6, Aq=[A, A])
mu0 = [rprob(2), rprob(2)]; sg0 = [rprob(2), rprob(2)]
one = [np.array([1.0]), np.array([1.0])]
ks0 = [rprob(2, 2), rprob(2, 2)]
other = rprob(2, 4)
xX = build_state(MB, [np.array([k, kp]), other], ks0, mu0, sg0, one, one)    # kappa at z=0, kappa2 at z=1
xY = build_state(MB, [np.array([k, k]), other], ks0, mu0, sg0, one, one)     # product (kappa at both)
k3 = A * np.exp(np.log(np.array([1.2, 1.0, 0.7, 0.95])) - np.log(np.sum(A * np.array([1.2, 1.0, 0.7, 0.95]))))
xC = build_state(MB, [np.array([k, k3]), other], ks0, mu0, sg0, one, one)    # control: different spectrum at z=1
fX, fY, fC = (fine_traj(MB, x0, 0.5, 0.01) for x0 in (xX, xY, xC))
cX, cY, cC = coarse_from_fine(MB, fX), coarse_from_fine(MB, fY), coarse_from_fine(MB, fC)
print(f"    public/rows max |X - Y| at T=0.5: {np.max(np.abs(cX - cY)):.2e}   control max |X - C|: {np.max(np.abs(cX - cC)):.2e}")
qX, _, _, _ = split_fine(MB, fX)
kX = qX[0] / qX[0].sum(1)[:, None]
lX = np.log(kX / A[None, :])
print("    spectra of agent 0 at z=0 and z=1 still equal at T:", spectrum(lX[0], A, 9) == spectrum(lX[1], A, 9))

print("(c) A = (1/2, 1/4, 1/4)")
A3 = np.array([0.5, 0.25, 0.25])
aut3 = automorphisms(A3)
c1 = np.log(1.4); c2 = np.log((1 - 0.5 * 1.4) / 0.5)
l = np.array([c1, c2, c2]); lp = np.array([c2, c1, c1])
print("    automorphisms:", aut3)
print("    equal spectra:", spectrum(l, A3) == spectrum(lp, A3), "; in orbit:", any(np.allclose(lp, l[list(p)]) for p in aut3))

print("(d) finite criterion on random references")
rng2 = np.random.default_rng(7)
count_coinc = count_ok = 0
for trial in range(400):
    n = rng2.integers(3, 6)
    if trial % 2 == 0:
        A_ = rng2.dirichlet(np.ones(n))
    else:  # plant a coincidence: two atoms summing to a third
        w = rng2.dirichlet(np.ones(n - 1)); a, b = w[0] * 0.5, w[0] * 0.5
        A_ = np.concatenate([[a, b], w[1:]]); A_ = np.concatenate([A_[:2], [w[0]], w[1:-1]]); A_ /= A_.sum()
    subsets = [s for r in range(1, n) for s in itertools.combinations(range(n), r)]
    pair = None
    for s1, s2 in itertools.combinations(subsets, 2):
        if set(s1).isdisjoint(s2) and abs(A_[list(s1)].sum() - A_[list(s2)].sum()) < 1e-12:
            pair = (s1, s2); break
    if pair is None:
        continue
    count_coinc += 1
    m = A_[list(pair[0])].sum()
    rest = [v for v in range(n) if v not in pair[0] and v not in pair[1]]
    ex, ey = 1.3, 0.6
    ew = (1 - m * ex - m * ey) / (1 - 2 * m) if rest else None
    if rest and ew <= 0:
        ex, ey = 1.1, 0.9; ew = (1 - m * ex - m * ey) / (1 - 2 * m)
    if not rest:
        ex = 1.4; ey = 2 - ex   # m = 1/2
    L = np.zeros(n); Lp = np.zeros(n)
    L[list(pair[0])] = np.log(ex); L[list(pair[1])] = np.log(ey)
    Lp[list(pair[0])] = np.log(ey); Lp[list(pair[1])] = np.log(ex)
    if rest:
        L[rest] = np.log(ew); Lp[rest] = np.log(ew)
    ok = (spectrum(L, A_) == spectrum(Lp, A_)) and not np.allclose(L, Lp) and abs(np.sum(A_ * np.exp(L)) - 1) < 1e-12
    count_ok += ok
print(f"    references with an equal-mass disjoint pair: {count_coinc}; constructive noninjective pairs verified: {count_ok}")
