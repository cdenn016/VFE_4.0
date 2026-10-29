r"""D09-INH check of supplements/s04_inheritance.tex.

  1. sinh:filter (filter-law, next-record, filter-update) against brute-force path enumeration on a
     random 4-state chain with a 2-valued retention: conditional law of X_n given the retained history,
     next-record probability, recursive update.
  2. sinh:linear-descent: random linear maps with ker C_j in ker(C_i M_ij) descend (least-squares
     residual 0, uniqueness), and generic maps do not.
  3. additive noise: retained law equals the law of M^c C_j v + C_i eps (Monte Carlo-free: exact for
     a discrete noise law), and fails to factor when the kernel condition fails.
  4. sinh:conditional-tower on a random finite joint law.
"""
import itertools
import numpy as np
rng = np.random.default_rng(29)

# 1
S = 4; T = rng.dirichlet(np.ones(S), size=S); p0 = rng.dirichlet(np.ones(S)); R = np.array([0, 0, 1, 1])
P = [np.diag((R == z).astype(float)) for z in (0, 1)]
worst = 0.0
for n in range(1, 5):
    for hist in itertools.product((0, 1), repeat=n + 1):
        # brute force joint of history and X_n, and of history and Z_{n+1}
        joint = np.zeros(S); nxt = np.zeros(2)
        for path in itertools.product(range(S), repeat=n + 2):
            if any(R[path[k]] != hist[k] for k in range(n + 1)):
                continue
            pr = p0[path[0]] * np.prod([T[path[k], path[k + 1]] for k in range(n + 1)])
            joint[path[n]] += pr / 1.0 if False else 0.0
            nxt[R[path[n + 1]]] += pr
        for path in itertools.product(range(S), repeat=n + 1):
            if any(R[path[k]] != hist[k] for k in range(n + 1)):
                continue
            joint[path[n]] += p0[path[0]] * np.prod([T[path[k], path[k + 1]] for k in range(n)])
        if joint.sum() < 1e-14:
            continue
        f = p0 @ P[hist[0]]
        for k in range(1, n + 1):
            f = f @ T @ P[hist[k]]
        phat = f / f.sum()
        worst = max(worst, np.abs(phat - joint / joint.sum()).max())
        pred = np.array([(phat @ T @ P[z]).sum() for z in (0, 1)])
        worst = max(worst, np.abs(pred - nxt / nxt.sum()).max())
        # recursive update consistency
        for z in (0, 1):
            m = (phat @ T @ P[z]).sum()
            if m > 1e-14:
                f2 = f @ T @ P[z]
                worst = max(worst, np.abs(f2 / f2.sum() - (phat @ T @ P[z]) / m).max())
print(f"1. sinh:filter: max residual over all histories of length 2..5 = {worst:.2e}")

# 2
Ej, Ei, Zj, Zi = 4, 3, 2, 2
Cj = rng.normal(size=(Zj, Ej)); Ci = rng.normal(size=(Zi, Ei))
K = np.linalg.svd(Cj)[2][Zj:].T                  # basis of ker C_j
# build M with ker C_j in ker(C_i M): M = M0 (I - K K^T) + N K^T with C_i N = 0
Nci = np.linalg.svd(Ci)[2][Zi:].T                # ker C_i (Ei - Zi = 1 dim)
M = rng.normal(size=(Ei, Ej)) @ (np.eye(Ej) - K @ K.T) + Nci @ rng.normal(size=(1, Ej - Zj)) @ K.T
X, *_ = np.linalg.lstsq(Cj.T, (Ci @ M).T, rcond=None)
Mc = X.T
print(f"2. descending case: |C_i M - M^c C_j| = {np.abs(Ci @ M - Mc @ Cj).max():.2e}")
Mbad = rng.normal(size=(Ei, Ej))
X, *_ = np.linalg.lstsq(Cj.T, (Ci @ Mbad).T, rcond=None)
print(f"   generic case: best |C_i M - M^c C_j| = {np.abs(Ci @ Mbad - X.T @ Cj).max():.2e} (> 0: no factorization)")

# 3 discrete noise law: eps uniform on a few points
eps = rng.normal(size=(5, Ei)); w = np.full(5, 0.2)
v = rng.normal(size=Ej); v2 = v + K @ rng.normal(size=Ej - Zj)   # same C_j value
law = lambda vv, MM: sorted(tuple(np.round(Ci @ (MM @ vv + e), 10)) for e in eps)
print("3. additive noise, descending M: retained laws equal on a fiber:", law(v, M) == law(v2, M),
      "; non-descending M:", law(v, Mbad) == law(v2, Mbad))
pred = sorted(tuple(np.round(Mc @ (Cj @ v) + Ci @ e, 10)) for e in eps)
print("   retained law equals law of M^c C_j v + C_i eps:", pred == law(v, M))

# 4 tower property on a finite joint law of (X, O) with Z = R10(X), W = R21(Z)
nx, no = 6, 3
J = rng.dirichlet(np.ones(nx * no)).reshape(nx, no)
R10 = np.array([0, 0, 1, 1, 2, 2]); R21 = np.array([0, 0, 1])
g = rng.normal(size=no)
def cond_exp_given(labels):
    out = {}
    for l in set(labels):
        mask = labels == l
        out[l] = (J[mask] @ g).sum() / J[mask].sum()
    return out
EgZ = cond_exp_given(R10); EgW = cond_exp_given(R21[R10])
# E[E[g|Z] | W]
res = 0.0
for wv in set(R21[R10]):
    mask = R21[R10] == wv
    pz = {z: J[R10 == z].sum() for z in set(R10[mask])}
    tower = sum(pz[z] * EgZ[z] for z in pz) / sum(pz.values())
    res = max(res, abs(tower - EgW[wv]))
print(f"4. sinh:conditional-tower residual = {res:.2e}")
