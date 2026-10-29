r"""D14-BND s4: nested unordered tree vs flat multiset; pair-law form of D_k; Jensen gap.

(a) Flat multiset counterexample (two-point base, d = 2): profiles with the same multiset of
    values at every base point but different unordered trees have different D_k, different
    profile energy, and different future ROOT responses f_0(t) (the coarsest admitted output).
(b) Sufficiency control: a base-dependent tree automorphism leaves f_0(t) unchanged exactly.
(c) D_k = KL(J_k || J_k^rev) (eq. after eq:boundary-tree-jensen); Jensen identity
    D_k - KL(f || P_k f) = int f [log P_k f - P_k log f] >= 0 with equality iff no dependence on
    bits k+2..d; exact rational instance.
(d) The gradients of D_k and of the averaged-source KL(f || P_k f) differ, and the flows differ.
"""
import itertools

import numpy as np

from bnd_common import (P_matrix, all_bits, bits_to_idx, ProfileData, profile_velocity, profile_energy,
                        profile_D, rk4, normalize_profile)

rng = np.random.default_rng(2718)

# ----------------------------------------------------------------------------- (a), (b)
d = 2
A = np.array([0.5, 0.5]); B = np.array([0.4, 0.6])
logrho = np.array([[0.5, -0.4], [-0.6, 0.3]])
logrho -= np.log(np.sum(A[:, None] * B[None, :] * np.exp(logrho)))
pik = np.array([2 ** (d - k - 1) / (2 ** d - 1) for k in range(d)])
PD = ProfileData(d, A, B, logrho, 0.7, 1.1, pik, [P_matrix(k, d) for k in range(d)])
# bit order: idx 0=(+,+), 1=(+,-), 2=(-,+), 3=(-,-)
fA = np.array([[1.0, 2.0, 3.0, 4.0],      # tree ((1,2),(3,4)) at r=0
               [1.5, 2.5, 3.0, 1.0]])
fB = np.array([[1.0, 3.0, 2.0, 4.0],      # tree ((1,3),(2,4)) at r=0: same multiset {1,2,3,4}
               [1.5, 2.5, 3.0, 1.0]])
fA = normalize_profile(fA, A); fB = normalize_profile(fB, A)
assert all(sorted(fA[r]) == sorted(fB[r]) for r in range(2))
g = normalize_profile(np.array([[1.0, 1.4, 0.7, 1.2], [0.9, 1.1, 1.3, 0.8]]), B)
bq = np.array([0.6, 0.4]); bs = np.array([0.3, 0.7])
print("(a) same multiset of profile values at every base point, different unordered trees")
print(f"   D_k(fA) = {profile_D(fA, PD)},  D_k(fB) = {profile_D(fB, PD)}")
print(f"   F(fA) - F(fB) = {profile_energy(fA, g, bq, bs, PD) - profile_energy(fB, g, bq, bs, PD):.6f}")
vA = profile_velocity((fA, g, bq, bs), PD); vB = profile_velocity((fB, g, bq, bs), PD)
print(f"   root-response velocity d f_0/dt at t=0: A {vA[0].mean(axis=1)}, B {vB[0].mean(axis=1)}")
xA = rk4(profile_velocity, (fA, g, bq, bs), 0.01, 100, PD)
xB = rk4(profile_velocity, (fB, g, bq, bs), 0.01, 100, PD)
print(f"   root response f_0(1): A {xA[0].mean(axis=1)}, B {xB[0].mean(axis=1)}; sup diff {np.max(np.abs(xA[0].mean(axis=1) - xB[0].mean(axis=1))):.3e}")
print(f"   multisets at t=1 equal? {all(np.allclose(sorted(xA[0][r]), sorted(xB[0][r])) for r in range(2))}")
def canon(v):
    """Canonical form of a depth-2 unordered profile tree (sort pairs, then sort pairs lexicographically)."""
    pairs = sorted(tuple(sorted(v[2 * a:2 * a + 2])) for a in range(2))
    return np.array(pairs).ravel()
# (b) base-dependent automorphism of fA: swap at the root for r=0 and swap below the left child for r=1
U = all_bits(d)
tau0 = [bits_to_idx((-u[0], u[1])) for u in U]
tau1 = [bits_to_idx((u[0], -u[1] if u[0] == 1 else u[1])) for u in U]
fC = np.stack([fA[0, tau0], fA[1, tau1]])
xC = rk4(profile_velocity, (fC, g, bq, bs), 0.01, 100, PD)
print(f"(b) base-dependent automorphism image: sup|f_0^A(1) - f_0^C(1)| = {np.max(np.abs(xA[0].mean(axis=1) - xC[0].mean(axis=1))):.2e};"
      f" rows {np.max(np.abs(xA[2] - xC[2])):.2e}; unordered trees equal at t=1: "
      f"{all(np.allclose(canon(xA[0][r]), canon(xC[0][r])) for r in range(2))}")

# ----------------------------------------------------------------------------- (c)
def pair_law_D(f, k, d, Abase):
    P = P_matrix(k, d)
    n = 2 ** d
    J = Abase[:, None, None] * f[:, :, None] * (1.0 / n) * P[None, :, :]          # J(r,u,v)
    Jrev = Abase[:, None, None] * f[:, None, :] * (1.0 / n) * P.T[None, :, :]      # exchange u,v
    m = J > 0
    return np.sum(J[m] * np.log(J[m] / Jrev[m]))

print("(c) pair-law form and Jensen gap")
for d in (2, 3, 4):
    Ab = rng.dirichlet(np.ones(3))
    f = normalize_profile(np.exp(0.7 * rng.normal(size=(3, 2 ** d))), Ab)
    PDd = ProfileData(d, Ab, Ab, np.zeros((3, 3)), 1.0, 1.0, np.ones(d) / d, [P_matrix(k, d) for k in range(d)])
    Dk = profile_D(f, PDd)
    res, gaps = [], []
    for k in range(d):
        P = P_matrix(k, d)
        res.append(abs(pair_law_D(f, k, d, Ab) - Dk[k]))
        w = Ab[:, None] / 2 ** d
        kl_avg = np.sum(w * f * (np.log(f) - np.log(f @ P.T)))
        jensen = np.sum(w * f * (np.log(f @ P.T) - np.log(f) @ P.T))
        gaps.append((Dk[k] - kl_avg, jensen))
    print(f"   d={d}: max |KL(J_k||J_k^rev) - D_k| = {max(res):.2e}; (D_k - KL(f||P_k f), Jensen term) = "
          + ", ".join(f"({a:.4f},{b:.4f})" for a, b in gaps))
    # equality condition: profile depending on bits 1..k+1 only has zero gap for k
    for k in range(d):
        U = all_bits(d)
        h = np.exp(0.7 * rng.normal(size=(3, 2 ** (k + 1))))
        fk = np.stack([[h[r, bits_to_idx(u[:k + 1])] for u in U] for r in range(3)])
        fk = normalize_profile(fk, Ab)
        P = P_matrix(k, d); w = Ab[:, None] / 2 ** d
        jg = np.sum(w * fk * (np.log(fk @ P.T) - np.log(fk) @ P.T))
        # depends additionally on bit k+2 (if it exists): gap positive
        if k + 2 <= d:
            fk2 = fk * np.array([[1 + 0.3 * u[k + 1] for u in U]] * 3)
            fk2 = normalize_profile(fk2, Ab)
            jg2 = np.sum(w * fk2 * (np.log(fk2 @ P.T) - np.log(fk2) @ P.T))
        else:
            jg2 = float("nan")
        print(f"      k={k}: gap for profile in bits 1..{k+1}: {jg:.1e}; after adding bit-{k+2} dependence: {jg2:.4f}")

# exact rational instance: trivial base, d = 2, f = 1 + u_2/2, k = 0
import sympy as sp
f_ex = {u: sp.Rational(1) + sp.Rational(1, 2) * u[1] for u in itertools.product([1, -1], repeat=2)}
D0 = sp.Rational(1, 4) * sum(f_ex[u] * (sp.log(f_ex[u]) - sp.Rational(1, 2) * sum(sp.log(f_ex[(-u[0], t)]) for t in (1, -1)))
                            for u in f_ex)
KLavg = sp.Rational(1, 4) * sum(f_ex[u] * (sp.log(f_ex[u]) - sp.log(sp.Rational(1, 2) * sum(f_ex[(-u[0], t)] for t in (1, -1))))
                               for u in f_ex)
print(f"   exact instance f = 1 + u_2/2, k = 0: D_0 = {sp.nsimplify(sp.simplify(D0))} = {float(D0):.6f}, "
      f"KL(f||P_0 f) = {float(KLavg):.6f}, gap = {sp.simplify(D0 - KLavg)} = {float(D0 - KLavg):.6f}")

# ----------------------------------------------------------------------------- (d)
d = 3
Ab = np.array([1.0])
f = normalize_profile(np.exp(0.7 * rng.normal(size=(1, 8))), Ab)
P0 = P_matrix(0, d)
lf = np.log(f)
grad_D = lf + 1 - lf @ P0.T - (f @ P0.T) / f
grad_KL = (lf + 1 - np.log(f @ P0.T)) - ((f / (f @ P0.T)) @ P0)   # first variation of int f log(f/P0 f)
cD = grad_D - np.sum(f * grad_D) / 8
cK = grad_KL - np.sum(f * grad_KL) / 8
print(f"(d) d=3, k=0: centered first variations differ: sup |grad D_0 - grad KL(f||P_0 f)| (centered) = {np.max(np.abs(cD - cK)):.4f}")
# FD check of grad_KL
dv = rng.normal(size=(1, 8)); dv -= dv.mean()
eps = 1e-6
def KLf(f):
    return np.sum(f * (np.log(f) - np.log(f @ P0.T))) / 8
fd = (KLf(f + eps * dv) - KLf(f - eps * dv)) / (2 * eps)
print(f"    FD check of the averaged-source first variation: {abs(fd - np.sum(grad_KL * dv) / 8):.2e}")
