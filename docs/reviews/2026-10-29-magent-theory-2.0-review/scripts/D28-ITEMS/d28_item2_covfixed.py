r"""D28-ITEMS: uniqueness of the per-agent LAW-LEVEL covariance fixed point (Item 2 / ch. 18 tangent model).

Claim tested (proved in the D28-ITEMS report by a Loewner-order argument): the covariance update
    Phi_i(S) = A_i(S)^{-1},  A_i(S) = F_i + G_i(S),
where F_i (self precision, anchor, grounded term) is fixed and positive definite and G_i(S) collects the record
terms L^T (R_e + T_e Sigma_j T_e^T)^{-1} L and loop terms B^T ((lam W)^{-1} + sum_f V_f)^{-1} B, is order preserving
and strictly subhomogeneous, Phi(cS) < c Phi(S) for c > 1. Hence it has exactly one fixed point, and the
iterations from S = 0 (increasing) and from S = F^{-1} (decreasing) converge to it.

The script iterates from both ends on random complete exchanges and reports the distance between the limits,
monotonicity violations, and a subhomogeneity margin. Single thread, tiny sizes.
"""
import numpy as np

rng = np.random.default_rng(29104)
K = 2; mm = K * K
edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]   # (sender j, receiver i)
loops = [[0, 1], [2, 3], [4, 5], [1, 3, 5], [4, 2, 0]]
nA = 3
owned = [[e for e, (j, i) in enumerate(edges) if i == a] for a in range(nA)]


def build(rank, lam, s2, g):
    codes = [rng.standard_normal((rank, K)) for _ in edges]
    R = [np.eye(rank) * rng.uniform(0.5, 2.0) for _ in edges]
    c = [rng.standard_normal(K) for _ in range(nA)]
    P = [np.eye(K) * rng.uniform(0.3, 2.0) for _ in range(nA)]
    G = [g * np.kron(T.T @ T, np.eye(K)) for T in codes]
    return codes, R, c, P, G, lam, s2


def own_hessian(a, S, data):
    codes, R, c, P, G, lam, s2 = data
    my = owned[a]; n = K + len(my) * mm
    F = np.zeros((n, n)); F[:K, :K] = P[a]
    for k, e in enumerate(my):
        o = K + k * mm
        F[o:o + mm, o:o + mm] = np.eye(mm) / s2 + G[e]
    Gm = np.zeros((n, n))
    kap = np.kron(np.eye(K), c[a][None, :])
    for k, e in enumerate(my):
        j = edges[e][0]; T = codes[e]; o = K + k * mm
        Sig_j = S[j][:K, :K]
        L = np.zeros((T.shape[0], n)); L[:, :K] = T; L[:, o:o + mm] = -T @ kap
        Gm += L.T @ np.linalg.solve(R[e] + T @ Sig_j @ T.T, L)
    for gam in loops:
        mine = [e for e in gam if e in my]
        if not mine:
            continue
        Vs = np.zeros((mm, mm))
        for f in gam:
            if f in my:
                continue
            b = edges[f][1]; kk = owned[b].index(f); o = K + kk * mm
            Vs += S[b][o:o + mm, o:o + mm]
        Om = np.linalg.inv(np.eye(mm) / lam + Vs)
        Bm = np.zeros((mm, n))
        for e in mine:
            k = my.index(e); o = K + k * mm
            Bm[:, o:o + mm] += np.eye(mm)
        Gm += Bm.T @ Om @ Bm
    return F, Gm


def Phi(S, data):
    out = []
    for a in range(nA):
        F, Gm = own_hessian(a, S, data)
        out.append(np.linalg.inv(F + Gm))
    return out


def loewner_min(X, Y):
    r"""Smallest eigenvalue of Y - X over all blocks (>= 0 means X <= Y)."""
    return min(np.linalg.eigvalsh(y - x).min() for x, y in zip(X, Y))


worst_gap, worst_mono, min_margin = 0.0, np.inf, np.inf
for trial in range(12):
    data = build(rank=1 + trial % 2, lam=float(rng.uniform(0.5, 5.0)), s2=float(rng.uniform(0.3, 2.0)),
                 g=float(rng.choice([0.0, 1.0])))
    zero = [np.zeros((K + len(owned[a]) * mm,) * 2) for a in range(nA)]
    upper = []
    for a in range(nA):
        F, _ = own_hessian(a, zero, data)
        upper.append(np.linalg.inv(F))
    lo, hi = zero, upper
    for it in range(4000):
        lo_n, hi_n = Phi(lo, data), Phi(hi, data)
        worst_mono = min(worst_mono, loewner_min(lo, lo_n), loewner_min(hi_n, hi))
        done = max(np.abs(x - y).max() for x, y in zip(lo_n + hi_n, lo + hi)) < 1e-14
        lo, hi = lo_n, hi_n
        if done:
            break
    gap = max(np.abs(x - y).max() for x, y in zip(lo, hi))
    worst_gap = max(worst_gap, gap)
    # strict subhomogeneity margin at the fixed point: c Phi(S) - Phi(c S) for c = 2
    cS = [2.0 * x for x in lo]
    margin = loewner_min(Phi(cS, data), [2.0 * x for x in Phi(lo, data)])
    min_margin = min(min_margin, margin)
    print("trial", trial, ": iterations", it, " |lower limit - upper limit| =", gap,
          " min eig(2 Phi(S) - Phi(2S)) =", margin)
print("worst gap between the two limits =", worst_gap)
print("worst monotonicity value (>= -1e-12 means the lower sequence increased and the upper decreased) =", worst_mono)
print("smallest strict-subhomogeneity margin =", min_margin)
