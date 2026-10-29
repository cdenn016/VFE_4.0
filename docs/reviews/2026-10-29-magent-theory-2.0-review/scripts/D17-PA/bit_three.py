r"""D17-PA check 2: three-agent Bernoulli construction pa:persistent-grouping.

  (a) pa:bit-flow from finite differences of the Gibbs-reduced per-agent free energy (Bernoulli log odds);
  (b) M, its spectrum, u_* = -M^{-1} o;
  (c) exact full and null flows from u_in=(4,1/2,-1) for several eps: partition history under pa:selector;
  (d) random open perturbations at eps=0.05.
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(7)
a, p = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])
PI = np.full((3, 3), p); np.fill_diagonal(PI, a)

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def klb(l1, l2):
    p1, p2 = sig(l1), sig(l2)
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))

def rows(l):
    D = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D[i, j] = klb(l[i], 0.0) if i == j else klb(l[i], l[j])
    w = PI * np.exp(-D)
    return w / w.sum(axis=1, keepdims=True), D

def F_red(li, i, l, eps):
    # reduced per-agent free energy of agent i as a function of its own log odds li (peers fixed)
    D = np.array([klb(li, 0.0) if j == i else klb(li, l[j]) for j in range(3)])
    pz = sig(li)
    rec = -(eps * o[i] / 2) * (2 * pz - 1)
    return -np.log(np.sum(PI[i] * np.exp(-D))) + rec

def field(l, eps):
    B, _ = rows(l)
    return np.array([-l[i] + sum(B[i, j] * l[j] for j in range(3) if j != i) + eps * o[i] for i in range(3)])

def null_field(l, eps):
    out = np.zeros(3)
    for i in range(3):
        D = klb(l[i], 0.0)
        b0 = a * np.exp(-D) / (a * np.exp(-D) + 2 * p)
        out[i] = -b0 * l[i] + eps * o[i]
    return out

print("(a) pa:bit-flow vs Fisher gradient of the reduced per-agent free energy")
worst = 0.0
for _ in range(20):
    l = rng.normal(size=3) * 1.5
    eps = abs(rng.normal())
    f = field(l, eps)
    for i in range(3):
        e = 1e-6
        # Bernoulli Fisher flow in log odds: ldot = -dF/dp, with p = sigma(l), dp/dl = p(1-p)
        dFdl = (F_red(l[i] + e, i, l, eps) - F_red(l[i] - e, i, l, eps)) / (2 * e)
        pz = sig(l[i])
        worst = max(worst, abs(f[i] + dFdl / (pz * (1 - pz))))
print("  max residual %.2e" % worst)

M = -1.4 * np.eye(3) + 0.4 * np.ones((3, 3))
print("(b) eig M =", np.round(np.sort(np.linalg.eigvalsh(M)), 6), "; u_* = -M^{-1}o =", np.round(-np.linalg.solve(M, o), 6), " (15/7 =", round(15 / 7, 6), ", 5/7 =", round(5 / 7, 6), ")")

def partition(l):
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            if l[i] * l[j] > 0 and min(abs(l[i]), abs(l[j])) / max(abs(l[i]), abs(l[j])) > 0.25:
                edges.append((i, j))
    # connected components on 3 vertices
    comp = list(range(3))
    for (i, j) in edges:
        ci, cj = comp[i], comp[j]
        comp = [ci if c == cj else c for c in comp]
    blocks = {}
    for k, c in enumerate(comp):
        blocks.setdefault(c, []).append(k + 1)
    return tuple(sorted(tuple(b) for b in blocks.values()))

def history(l0, eps, T, null=False, n=6001):
    f = null_field if null else field
    sol = solve_ivp(lambda t, y: f(y, eps), (0, T), l0, t_eval=np.linspace(0, T, n), rtol=1e-10, atol=1e-13)
    seq = []
    for k in range(sol.y.shape[1]):
        P = partition(sol.y[:, k])
        if not seq or seq[-1][1] != P:
            seq.append((sol.t[k], P))
    return seq, sol.y[:, -1]

u_in = np.array([4.0, 0.5, -1.0])
print("(c) partition histories (event time, partition)")
for eps in (0.02, 0.05, 0.1, 0.3, 1.0, 2.0):
    seq, lf = history(eps * u_in, eps, 80.0)
    seq0, lf0 = history(eps * u_in, eps, 400.0, null=True)
    print("  eps=%.2f full: %s | final l/eps=%s" % (eps, [(round(t, 3), P) for t, P in seq], np.round(lf / eps, 4)))
    print("           null: %s | final l/eps=%s" % ([(round(t, 3), P) for t, P in seq0], np.round(lf0 / eps, 4)))

print("(d) random open perturbations at eps=0.05")
eps = 0.05
nf = nn = 0
NS = 40
for _ in range(NS):
    l0 = eps * (u_in + 0.3 * rng.normal(size=3))
    seq, _ = history(l0, eps, 60.0, n=1201)
    seq0, _ = history(l0, eps, 200.0, null=True, n=2001)
    full_ok = seq[0][1] == ((1,), (2,), (3,)) and seq[-1][1] == ((1, 2, 3),)
    null_full = any(P == ((1, 2, 3),) for _, P in seq0)
    nf += int(full_ok); nn += int(null_full)
print("  full: starts singletons and ends in the full block %d/%d; null ever has the full block %d/%d" % (nf, NS, nn, NS))
