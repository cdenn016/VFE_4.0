"""E4 checks: adaptive-tower corollary, dropped-datum second reading, Dobrushin-product multiplier bound."""
import itertools

import numpy as np

rng = np.random.default_rng(20261012)


def kl(q: np.ndarray, p: np.ndarray) -> float:
    q, p = q.ravel(), p.ravel()
    m = q > 0
    if np.any(p[m] == 0):
        return np.inf
    return float(np.sum(q[m] * np.log(q[m] / p[m])))


def stoch(*shape: int) -> np.ndarray:
    a = rng.random(shape) + 0.05
    return a / a.sum(axis=tuple(range(1, len(shape))), keepdims=True)


# ---- A. corollary: F_tower = F_0 + E KL(Q(.|y0) || C(y0,.)); singular branch ----
n0, no, nR0, n1, nR1, n2 = 6, 3, 2, 4, 2, 3
P0 = rng.random((no, n0))
P0 /= P0.sum()
C1 = stoch(n0, nR0, n1)  # stochastic step y0 -> (r0, y1)
f = rng.integers(0, n2, size=(n1, nR1))  # deterministic part: y2 = f(y1, r1)
CR1 = stoch(n1, nR1)
C2 = np.zeros((n1, nR1, n2))
for a in range(n1):
    for r in range(nR1):
        C2[a, r, f[a, r]] = CR1[a, r]
C = np.einsum("aqb,brc->aqbrc", C1, C2)  # composite kernel y0 -> (r0, y1, r1, y2)
o = 1
p_o = P0[o].sum()
Pi0 = P0[o] / p_o
Pi_tow = Pi0[:, None, None, None, None] * C


def F0(Q0: np.ndarray) -> float:
    return -np.log(p_o) + kl(Q0, Pi0)


def Ft(Q: np.ndarray) -> float:
    return -np.log(p_o) + kl(Q, Pi_tow)


worst_push, min_gap, worst_id = 0.0, np.inf, 0.0
for _ in range(200):
    Q0 = rng.random(n0)
    Q0 /= Q0.sum()
    Qp = Q0[:, None, None, None, None] * C
    worst_push = max(worst_push, abs(Ft(Qp) - F0(Q0)))
    C1q = stoch(n0, nR0, n1)
    CR1q = stoch(n1, nR1)
    C2q = np.zeros_like(C2)
    for a in range(n1):
        for r in range(nR1):
            C2q[a, r, f[a, r]] = CR1q[a, r]
    Cq = np.einsum("aqb,brc->aqbrc", C1q, C2q)
    Q = Q0[:, None, None, None, None] * Cq
    cond = sum(Q0[a] * kl(Cq[a], C[a]) for a in range(n0))
    worst_id = max(worst_id, abs(Ft(Q) - (F0(Q0) + cond)))
    min_gap = min(min_gap, Ft(Q) - F0(Q0))
Q0 = rng.random(n0)
Q0 /= Q0.sum()
mu2 = rng.random(n2)
mu2 /= mu2.sum()
Qfree = np.einsum("a,aqb,br,c->aqbrc", Q0, C1, CR1, mu2)
print("A. |F_tower - F_0| pushed (max over 200):", worst_push)
print("A. identity residual F_tower = F_0 + E KL(cond) (max):", worst_id)
print("A. min F_tower - F_0 for unpushed recognition:", min_gap)
print("A. F_tower with free coarse coordinate on deterministic part:", Ft(Qfree))

# ---- B. dropped datum, R-factor reading Y_{l+1} or only Z_{l+1} (L = 1) ----
nZ0, nN0, nR, nZ1, nN1 = 3, 2, 2, 3, 2


def cmi(J: np.ndarray, a: tuple, b: tuple, c: tuple) -> float:
    """I(A;B|C) for a joint array J with axis tuples a, b, c (other axes summed)."""
    ax = tuple(range(J.ndim))

    def marg(keep: tuple) -> np.ndarray:
        return J.sum(axis=tuple(i for i in ax if i not in keep), keepdims=True)

    pabc, pac, pbc, pc = marg(a + b + c), marg(a + c), marg(b + c), marg(c)
    ratio = pabc * pc / (pac * pbc)
    return float(np.sum(pabc * np.log(ratio)))


worst1 = worst2 = 0.0
for _ in range(100):
    M0 = rng.random((nZ0, nN0))
    M0 /= M0.sum()
    Cs = stoch(nZ0, nN0, nR, nZ1, nN1)
    J = M0[:, :, None, None, None] * Cs  # axes: Z0 N0 R Z1 N1
    PY1 = J.sum(axis=(0, 1, 2))  # (Z1, N1)
    PR_Y1 = J.sum(axis=(0, 1)) / PY1  # (R, Z1, N1)
    PRZ = J.sum(axis=(0, 1, 4))  # (R, Z1)
    PZ1 = PRZ.sum(axis=0)
    PY0_RZ1 = J.sum(axis=4) / PRZ  # (Z0, N0, R, Z1)
    PR_Z1 = PRZ / PZ1
    red1 = PY1[None, None, None] * PR_Y1[None, None] * PY0_RZ1[..., None]
    red2 = PY1[None, None, None] * PR_Z1[None, None, :, :, None] * PY0_RZ1[..., None]
    c1 = cmi(J, (0, 1), (4,), (2, 3))
    c2 = cmi(J, (2,), (4,), (3,))
    worst1 = max(worst1, abs(kl(J, red1) - c1))
    worst2 = max(worst2, abs(kl(J, red2) - (c1 + c2)))
print("B. dropped datum, R reads Y1: |KL - I(Y0;N1|Z1,R)|:", worst1)
print("B. dropped datum, R reads Z1 only: |KL - [I(Y0;N1|Z1,R)+I(R;N1|Z1)]|:", worst2)


# ---- C. ||D_{n<-l}|| <= (3^{|V_n|}-1) prod delta(Cdag_psi_k), and the tilt bound ----
def states(n: int) -> np.ndarray:
    return np.array(list(itertools.product([0, 1], repeat=n)))


def gnorm(fvals: np.ndarray, n: int) -> float:
    """Sum over nonempty B of sup |P_B f|, binary coordinates, uniform product reference."""
    S = states(n)
    F = fvals.reshape([2] * n)
    total = 0.0
    for r in range(1, n + 1):
        for B in itertools.combinations(range(n), r):
            g = np.zeros(len(S))
            for rr in range(r + 1):
                for Bp in itertools.combinations(B, rr):
                    out_axes = tuple(i for i in range(n) if i not in Bp)
                    E = F.mean(axis=out_axes, keepdims=True) if out_axes else F
                    g += (-1) ** (r - rr) * np.broadcast_to(E, F.shape).ravel()
            total += np.abs(g).max()
    return total


sizes = [4, 2, 1]
worst_excess, worst_tilt = -np.inf, -np.inf
for _ in range(30):
    mbar = [rng.random(2 ** sizes[0]) + 0.1]
    mbar[0] /= mbar[0].sum()
    Cks = [stoch(2 ** sizes[k], 2 ** sizes[k + 1]) for k in range(len(sizes) - 1)]
    for k in range(len(Cks)):
        mbar.append(mbar[k] @ Cks[k])
    psi = [0.8 * rng.standard_normal(2 ** sizes[0])]
    Us, deltas = [], []
    for k, Ck in enumerate(Cks):
        joint = mbar[k][:, None] * Ck
        Cdag = (joint / joint.sum(0)).T  # (z, y)
        tilt = Cdag * np.exp(-psi[k])[None]
        Ct = tilt / tilt.sum(1, keepdims=True)
        d = max(0.5 * np.abs(Ct[a] - Ct[b]).sum() for a in range(len(Ct)) for b in range(len(Ct)))
        d0 = max(0.5 * np.abs(Cdag[a] - Cdag[b]).sum() for a in range(len(Cdag)) for b in range(len(Cdag)))
        w = psi[k].max() - psi[k].min()
        worst_tilt = max(worst_tilt, d - (1 - np.exp(-w) * (1 - d0)))
        Us.append(Ct)
        deltas.append(d)
        nxt = -np.log(Cdag @ np.exp(-psi[k]))
        psi.append(nxt - nxt.mean())
    S = states(sizes[0])
    norm = 0.0
    for r in range(1, sizes[0] + 1):
        for B in itertools.combinations(range(sizes[0]), r):
            h = np.prod([S[:, i] - 0.5 for i in B], axis=0)
            h = h / np.abs(h).max()
            g = h
            for U in Us:
                g = U @ g
            norm = max(norm, gnorm(g, sizes[-1]))
    bound = (3 ** sizes[-1] - 1) * np.prod(deltas)
    worst_excess = max(worst_excess, norm - bound)
print("C. max(||D_{2<-0}|| - (3^|V_2|-1) prod delta_k), 30 towers 4->2->1 (<= 0 expected):", worst_excess)
print("C. max(delta(Cdag_psi) - [1 - e^{-w}(1 - delta(Cdag))]) (<= 0 expected):", worst_tilt)
