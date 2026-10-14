r"""L5 checks for c09 and c12: the common-channel VFE chain rule (thm:rg-exact-coarse-vfe),
the stochastic-infimum and hard-identification witnesses, the adaptive-tower reversal and the
exact hierarchical decomposition (lem:full-graph-adaptive-reversal,
thm:full-graph-vfe-decomposition), the dropped-datum cost (prop:full-graph-dropped-datum), and
the reduction of the adaptive-tower VFE to the depth-0 VFE.

All laws are finite; float64; logs natural.
"""
from __future__ import annotations

import itertools
import numpy as np

rng = np.random.default_rng(7)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p.ravel(); q = q.ravel()
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


# --------------------------------------------------------------- thm:rg-exact-coarse-vfe
def check_common_channel() -> dict:
    out = {}
    worst = 0.0
    for _ in range(200):
        nY, nZ, nO = 7, 3, 4
        prior = rng.dirichlet(np.ones(nY))
        lik = rng.dirichlet(np.ones(nO), size=nY)  # p(o|y)
        C = rng.dirichlet(np.ones(nZ), size=nY)
        o = rng.integers(nO)
        joint_o = prior * lik[:, o]
        evid = joint_o.sum()
        Pi = joint_o / evid
        Q = rng.dirichlet(np.ones(nY))
        F = -np.log(evid) + kl(Q, Pi)
        Qc, Pic = Q @ C, Pi @ C
        # coarse evidence from the pushed joint
        Pc_joint = (prior[:, None] * lik)[:, :, None] * C[:, None, :]  # (y,o,z)
        evid_c = Pc_joint[:, o, :].sum()
        Pic_from_joint = Pc_joint[:, o, :].sum(0) / evid_c
        Fc = -np.log(evid_c) + kl(Qc, Pic)
        Qhat = Q[:, None] * C
        Pihat = Pi[:, None] * C
        Delta = sum(Qc[z] * kl(Qhat[:, z] / Qc[z], Pihat[:, z] / Pic[z]) for z in range(nZ))
        worst = max(worst, abs(F - (Fc + Delta)), abs(evid - evid_c), np.abs(Pic - Pic_from_joint).max())
    out["max |F - (F^c + Delta)|, |p - p^c|, |Pi C - Pi^c|"] = worst
    # stochastic infimum witness
    Pi = np.array([0.5, 0.5]); C = np.array([[0.75, 0.25], [0.25, 0.75]]); Q = np.array([1.0, 0.0])
    gap = kl(Q, Pi) - kl(Q @ C, Pi @ C)
    out["stochastic-infimum gap vs 2log2-(3/4)log3"] = (gap, 2 * np.log(2) - 0.75 * np.log(3))
    # hard identification witness
    P = np.array([1, 1, 98]) / 100; Qh = np.array([1, 2, 97]) / 100
    out["hard identification KL before/after"] = (kl(P, Qh), kl(P[:2] / P[:2].sum(), Qh[:2] / Qh[:2].sum()))
    return out


# --------------------------------------------------------------- adaptive tower
def make_tower():
    nZ0, nN0 = 3, 2
    nR0, nZ1, nN1 = 2, 2, 2
    nR1, nZ2, nN2 = 2, 2, 2
    nO = 3
    shapes = dict(Y0=(nZ0, nN0), Y1=(nZ1, nN1), Y2=(nZ2, nN2))
    C0 = rng.dirichlet(0.6 * np.ones(nR0 * nZ1 * nN1), size=nZ0 * nN0).reshape(nZ0, nN0, nR0, nZ1, nN1)
    C1 = rng.dirichlet(0.6 * np.ones(nR1 * nZ2 * nN2), size=nZ1 * nN1).reshape(nZ1, nN1, nR1, nZ2, nN2)
    lik = rng.dirichlet(np.ones(nO), size=nZ0)  # p(o | Z0)
    return C0, C1, lik


def tower_joint(M0: np.ndarray, C0: np.ndarray, C1: np.ndarray) -> np.ndarray:
    """Law on (Z0,N0,R0,Z1,N1,R1,Z2,N2) = M0(y0) C0(y0; r0,y1) C1(y1; r1,y2)."""
    return np.einsum("ab,abcde,defgh->abcdefgh", M0, C0, C1)


def conditional(J: np.ndarray, target_axes: tuple, given_axes: tuple) -> np.ndarray:
    """Conditional law of target given 'given' from a joint over 8 axes, as an array over
    (given..., target...) with other axes summed; nan where the given has zero mass."""
    keep = tuple(sorted(set(target_axes) | set(given_axes)))
    other = tuple(a for a in range(J.ndim) if a not in keep)
    M = J.sum(axis=other)
    # reorder to given + target
    order = [keep.index(a) for a in given_axes] + [keep.index(a) for a in target_axes]
    M = np.transpose(M, order)
    G = M.reshape(M.shape[:len(given_axes)] + (-1,)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return M / G.reshape(G.shape + (1,) * len(target_axes))


AX = dict(Z0=0, N0=1, R0=2, Z1=3, N1=4, R1=5, Z2=6, N2=7)


def expected_conditional_kl(JQ, JP, target, given) -> float:
    """E_Q KL(Q(target|given) || P(target|given))."""
    t = tuple(AX[a] for a in target); g = tuple(AX[a] for a in given)
    cq = conditional(JQ, t, g); cp = conditional(JP, t, g)
    keep = tuple(sorted(set(t) | set(g)))
    other = tuple(a for a in range(JQ.ndim) if a not in keep)
    Mq = np.transpose(JQ.sum(axis=other), [keep.index(a) for a in g] + [keep.index(a) for a in t])
    m = Mq > 0
    if np.any(cp[m] == 0):
        return np.inf
    return float(np.sum(Mq[m] * np.log(cq[m] / cp[m])))


def check_tower() -> dict:
    out = {}
    worst_dec, worst_red, worst_markov = 0.0, 0.0, 0.0
    gaps = []
    for trial in range(100):
        C0, C1, lik = make_tower()
        P0 = rng.dirichlet(np.ones(6)).reshape(3, 2)
        Q0 = rng.dirichlet(np.ones(6)).reshape(3, 2)
        o = rng.integers(3)
        JP = tower_joint(P0, C0, C1)  # latent prior tower
        JQ = tower_joint(Q0, C0, C1)  # pushed recognition tower
        # depth-Markov reversal: P(R0, Y0 | Y1, R1, Y2) depends only on Y1
        c_full = conditional(JP, (AX["R0"], AX["Z0"], AX["N0"]), (AX["Z1"], AX["N1"], AX["R1"], AX["Z2"], AX["N2"]))
        c_red = conditional(JP, (AX["R0"], AX["Z0"], AX["N0"]), (AX["Z1"], AX["N1"]))
        worst_markov = max(worst_markov, np.nanmax(np.abs(c_full - c_red[:, :, None, None, None])))
        # tower VFE in density form
        likz = lik[:, o]
        pjoint = JP * likz[:, None, None, None, None, None, None, None]
        evid = pjoint.sum()
        m = JQ > 0
        F_tower = float(np.sum(JQ[m] * (np.log(JQ[m]) - np.log(pjoint[m]))))
        # depth-0 VFE
        Pi0 = (P0 * likz[:, None]) / evid
        F0 = -np.log(evid) + kl(Q0, Pi0)
        # decomposition
        obs = -float(np.sum(JQ.sum(axis=(1, 2, 3, 4, 5, 6, 7)) * np.log(likz)))
        top = kl(JQ.sum(axis=(0, 1, 2, 3, 4, 5)), JP.sum(axis=(0, 1, 2, 3, 4, 5)))
        terms = obs + top
        for ell, (R, N, Z, Zn, Nn) in enumerate([("R0", "N0", "Z0", "Z1", "N1"), ("R1", "N1", "Z1", "Z2", "N2")]):
            terms += expected_conditional_kl(JQ, JP, (R,), (Zn, Nn))
            terms += expected_conditional_kl(JQ, JP, (N,), (Zn, Nn, R))
            terms += expected_conditional_kl(JQ, JP, (Z,), (Zn, Nn, R, N))
        worst_dec = max(worst_dec, abs(F_tower - terms))
        worst_red = max(worst_red, abs(F_tower - F0))
        # a non-pushed recognition tower (different channels, still bottom-up Markov)
        C0q, C1q = make_tower()[:2]
        JQ2 = tower_joint(Q0, C0q, C1q)
        m2 = JQ2 > 0
        F2 = float(np.sum(JQ2[m2] * (np.log(JQ2[m2]) - np.log(pjoint[m2]))))
        gaps.append(F2 - F0)
    out["decomposition residual |F_tower - sum of terms| (pushed recognition)"] = worst_dec
    out["depth-Markov reversal residual (version reads only Y_{l+1})"] = worst_markov
    out["|F_tower - F_0| for pushed recognition (unstated reduction)"] = worst_red
    out["min (F_tower - F_0) for a recognition tower with other channels (should be >= 0)"] = min(gaps)
    return out


def check_dropped_datum() -> dict:
    out = {}
    res_stated, extra = 0.0, 0.0
    for _ in range(100):
        C0, C1, _ = make_tower()
        P0 = rng.dirichlet(np.ones(6)).reshape(3, 2)
        JP = tower_joint(P0, C0, C1)
        # reduced product with R factor reading Y_{l+1} and Y factors reading (R_l, Z_{l+1})
        PY2 = JP.sum(axis=(0, 1, 2, 3, 4, 5))
        cR1 = conditional(JP, (AX["R1"],), (AX["Z2"], AX["N2"]))
        cY1 = conditional(JP, (AX["Z1"], AX["N1"]), (AX["R1"], AX["Z2"]))
        cR0 = conditional(JP, (AX["R0"],), (AX["Z1"], AX["N1"]))
        cY0 = conditional(JP, (AX["Z0"], AX["N0"]), (AX["R0"], AX["Z1"]))
        red = np.einsum("gh,ghf,fgde,dec,cdab->abcdefgh", PY2, cR1, cY1, cR0, cY0)
        # R factors reading only Z_{l+1}
        cR1z = conditional(JP, (AX["R1"],), (AX["Z2"],))
        cR0z = conditional(JP, (AX["R0"],), (AX["Z1"],))
        red2 = np.einsum("gh,gf,fgde,dc,cdab->abcdefgh", PY2, cR1z, cY1, cR0z, cY0)

        def cmi(J, a, b, cond):
            ax = lambda names: tuple(AX[n] for n in names)
            def H(names):
                keep = ax(names)
                M = J.sum(axis=tuple(i for i in range(J.ndim) if i not in keep)).ravel()
                M = M[M > 0]
                return -float(np.sum(M * np.log(M)))
            return H(a + cond) + H(b + cond) - H(a + b + cond) - H(cond)
        stated = cmi(JP, ["Z0", "N0"], ["N1"], ["Z1", "R0"]) + cmi(JP, ["Z1", "N1"], ["N2"], ["Z2", "R1"])
        res_stated = max(res_stated, abs(kl(JP, red) - stated))
        Ir = cmi(JP, ["R0"], ["N1"], ["Z1"]) + cmi(JP, ["R1"], ["N2"], ["Z2"])
        extra = max(extra, abs(kl(JP, red2) - (stated + Ir)))
        out.setdefault("example extra term I(R;N|Z)", Ir)
    out["|KL(P^tow||P^red) - stated CMI sum| (R factor reads Y_{l+1})"] = res_stated
    out["|KL(P^tow||P^red') - (stated + sum I(R_l;N_{l+1}|Z_{l+1}))| (R factor reads Z_{l+1})"] = extra
    # four-vertex witness: two atoms of mass 1/2, N1 a bijective image of N0
    J = np.array([[0.5, 0.0], [0.0, 0.5]])
    out["witness KL vs log 2"] = (kl(J, np.outer(J.sum(1), J.sum(0))), np.log(2))
    return out


def check_declared_hierarchy() -> dict:
    """Declared top-down hierarchy with arbitrary kernels: decomposition identity."""
    worst = 0.0
    for _ in range(100):
        nY2, nR, nN, nZ = 4, 2, 2, 3
        P2 = rng.dirichlet(np.ones(nY2))
        PR = rng.dirichlet(np.ones(nR), size=nY2)
        PN = rng.dirichlet(np.ones(nN), size=(nY2, nR))
        Kd = rng.dirichlet(np.ones(nZ), size=(nY2, nR, nN))
        lik = rng.dirichlet(np.ones(3), size=nZ)
        o = 1
        JP = np.einsum("a,ab,abc,abcd->abcd", P2, PR, PN, Kd)
        Q2 = rng.dirichlet(np.ones(nY2))
        QR = rng.dirichlet(np.ones(nR), size=nY2)
        QN = rng.dirichlet(np.ones(nN), size=(nY2, nR))
        QZ = rng.dirichlet(np.ones(nZ), size=(nY2, nR, nN))
        JQ = np.einsum("a,ab,abc,abcd->abcd", Q2, QR, QN, QZ)
        pj = JP * lik[None, None, None, :, o]
        F = float(np.sum(JQ * (np.log(JQ) - np.log(pj))))
        terms = -float(np.sum(JQ.sum((0, 1, 2)) * np.log(lik[:, o]))) + kl(Q2, P2)
        terms += sum(Q2[a] * kl(QR[a], PR[a]) for a in range(nY2))
        terms += sum(Q2[a] * QR[a, b] * kl(QN[a, b], PN[a, b]) for a in range(nY2) for b in range(nR))
        terms += sum(JQ[a, b, c].sum() * kl(QZ[a, b, c], Kd[a, b, c]) for a in range(nY2) for b in range(nR) for c in range(nN))
        worst = max(worst, abs(F - terms))
    return {"declared hierarchy decomposition residual": worst}


if __name__ == "__main__":
    for name, fn in [("common channel (c09)", check_common_channel), ("adaptive tower (c12)", check_tower),
                     ("dropped datum (c12)", check_dropped_datum), ("declared hierarchy (c12)", check_declared_hierarchy)]:
        print(f"== {name}")
        for k, v in fn().items():
            print(f"  {k}: {v}")
