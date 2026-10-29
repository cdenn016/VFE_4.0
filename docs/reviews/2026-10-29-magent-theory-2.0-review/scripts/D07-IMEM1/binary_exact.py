r"""Independent exact checks of the finite constructions in appendices/interacting_memory.tex
and companion/c04_interacting_memory.tex (lens D07-IMEM1).

Written from scratch for the 2026-10-29 review; it does not import or copy the 2026-10-28
record script.  All probabilities are fractions.Fraction unless a logarithm is needed
(information and free-energy identities), where floats are used and residuals reported.

State order of the represented-plus-record variable E: (x0, x1, a0, a1, y0, y1, d0, d1).
Messages use identity encoders: H_{t+1} is a fresh draw from P(Y_j = . | D_j, H_{1:t}).

Checks:
  (A) every number quoted in imem:binary-* and comp:interaction-example;
  (B) the history law imem:history-law, recomputed from an AUGMENTED table that contains the
      fresh samples explicitly, against the recursion imem:bayes (direct conditioning, not the
      recursion checking itself);
  (C) sender invariance and the receiver likelihood imem:receiver-update via R_{ij,t};
  (D) reduction imem:reduction and the reduced recursion imem:reduced-update;
  (E) product-tilt (rectangle) structure lambda_t = F_0(d_0) F_1(d_1) lambda_0 / Z_t,
      invariance of the cross-ratio, and exact recovery of lambda_t from its two marginals
      and lambda_0 (iterative proportional fitting);
  (F) own past samples carry no information about E given (D_j, H_{1:t});
  (G) the information identity imem:information-bound (float residual);
  (H) the free-energy identities imem:full-free-energy-identity, imem:private-free-energy,
      imem:local-free-energy-split, imem:conditional-free-energy (float residuals);
  (I) the parity counterexample imem:local-retention-failure for several rho;
  (J) the product-preparation remark (appendix line 215);
  (K) a three-valued private-record model for the product-tilt structure in general.
"""
from __future__ import annotations

import json
import math
import random
import time
from fractions import Fraction as F
from itertools import product
from pathlib import Path

T0 = time.perf_counter()
OUT: dict = {}
X0, X1, A0, A1, Y0, Y1, D0, D1 = range(8)
DIDX = (D0, D1)
YIDX = (Y0, Y1)
XIDX = (X0, X1)


def ch(v: int, t: int, r: F = F(3, 4)) -> F:
    """Binary channel P(v | t) with reliability r."""
    return r if v == t else 1 - r


def preparation(agree: F = F(3, 4)) -> dict:
    """Lambda_0 of imem:binary-preparation; agree = P(X1 = X0)."""
    law = {}
    for s in product((0, 1), repeat=8):
        x0, x1, a0, a1, y0, y1, d0, d1 = s
        law[s] = (F(1, 2) * ch(x1, x0, agree) * ch(a0, x0) * ch(a1, x1)
                  * ch(y0, 1 - x0) * ch(y1, 1 - x1) * ch(d0, x0) * ch(d1, x1))
    assert sum(law.values()) == 1
    return law


def prob(law: dict, pred) -> F:
    return sum((p for s, p in law.items() if pred(s)), F(0))


def cond(law: dict, pred) -> dict:
    m = prob(law, pred)
    assert m > 0
    return {s: p / m for s, p in law.items() if pred(s)}


def speaker_kernel(gamma: dict, j: int) -> dict:
    """k_{j,t}(h | d_j) = P_Gamma(Y_j = h | D_j = d_j) for identity encoder."""
    out = {}
    for d in (0, 1):
        loc = cond(gamma, lambda s, d=d: s[DIDX[j]] == d)
        out[d] = {h: prob(loc, lambda s, h=h: s[YIDX[j]] == h) for h in (0, 1)}
    return out


def bayes(gamma: dict, j: int, h: int) -> tuple[dict, F, dict]:
    """imem:bayes for one message of speaker j."""
    k = speaker_kernel(gamma, j)
    un = {s: p * k[s[DIDX[j]]][h] for s, p in gamma.items()}
    z = sum(un.values())
    return {s: p / z for s, p in un.items()}, z, k


def run(law0: dict, schedule: tuple, hist: tuple) -> tuple[list, list, F]:
    gammas, kernels, ev = [law0], [], F(1)
    for j, h in zip(schedule, hist):
        g, z, k = bayes(gammas[-1], j, h)
        gammas.append(g)
        kernels.append(k)
        ev *= z
    return gammas, kernels, ev


def augmented_history_law(law0: dict, schedule: tuple) -> dict:
    """Joint law of (E, Ytilde_1..n, H_1..n) built step by step from the protocol definition:
    at step t the speaker's sampling law is P(Y_j | D_j, H_{1:t-1}) computed by conditioning the
    augmented joint constructed so far.  Identity encoder: H_t = Ytilde_t."""
    joint = {(s, ()): p for s, p in law0.items()}  # key: (e, ((ytilde, h), ...))
    for j in schedule:
        # marginal of (Y_j, D_j, H_{1:t-1}) under the current joint
        num: dict = {}
        den: dict = {}
        for (s, rec), p in joint.items():
            hs = tuple(h for _, h in rec)
            key = (s[DIDX[j]], hs)
            den[key] = den.get(key, F(0)) + p
            k2 = (s[YIDX[j]], s[DIDX[j]], hs)
            num[k2] = num.get(k2, F(0)) + p
        new = {}
        for (s, rec), p in joint.items():
            hs = tuple(h for _, h in rec)
            key = (s[DIDX[j]], hs)
            for yt in (0, 1):
                samp = num.get((yt, s[DIDX[j]], hs), F(0)) / den[key]
                new[(s, rec + ((yt, yt),))] = p * samp
        joint = new
    assert sum(joint.values()) == 1
    return joint


def marg_E_given_H(aug: dict, hist: tuple) -> dict:
    sub = {}
    for (s, rec), p in aug.items():
        if tuple(h for _, h in rec) == hist:
            sub[s] = sub.get(s, F(0)) + p
    z = sum(sub.values())
    return {s: p / z for s, p in sub.items()}


def lam(gamma: dict) -> dict:
    return {d: prob(gamma, lambda s, d=d: (s[D0], s[D1]) == d) for d in product((0, 1), repeat=2)}


# (A) quoted numbers ---------------------------------------------------------------------
law0 = preparation()
g0 = law0
k10 = speaker_kernel(g0, 1)
assert (k10[0][1], k10[1][1]) == (F(5, 8), F(3, 8))
# likelihood of H1=1 given X1 and given X0, integrating the unknown D1 under the PRIOR
pH_X1 = {x: sum(ch(d, x) * k10[d][1] for d in (0, 1)) for x in (0, 1)}
assert (pH_X1[0], pH_X1[1]) == (F(9, 16), F(7, 16))
pH_X0 = {x: sum(ch(x1, x) * pH_X1[x1] for x1 in (0, 1)) for x in (0, 1)}
assert (pH_X0[0], pH_X0[1]) == (F(17, 32), F(15, 32))
g1, z1, _ = bayes(g0, 1, 1)
assert z1 == F(1, 2)
post0 = {d: prob(cond(g1, lambda s, d=d: s[D0] == d), lambda s: s[X0] == 1) for d in (0, 1)}
assert post0 == {0: F(5, 22), 1: F(45, 62)}
k01 = speaker_kernel(g1, 0)
assert (k01[0][1], k01[1][1]) == (F(7, 11), F(12, 31))
g2, z2, _ = bayes(g1, 0, 1)
assert z2 == F(33, 64)
L0, L1, L2 = lam(g0), lam(g1), lam(g2)
order = [(0, 0), (0, 1), (1, 0), (1, 1)]
assert [L0[d] for d in order] == [F(9, 32), F(7, 32), F(7, 32), F(9, 32)]
assert [L1[d] for d in order] == [F(45, 128), F(21, 128), F(35, 128), F(27, 128)]
assert [L2[d] for d in order] == [F(105, 242), F(49, 242), F(70, 341), F(54, 341)]
W_x0 = [prob(cond(law0, lambda s, d=d: (s[D0], s[D1]) == d), lambda s: s[X0] == 1) for d in order]
assert W_x0 == [F(1, 6), F(5, 14), F(9, 14), F(5, 6)]
V0 = [prob(cond(law0, lambda s, d=d: (s[D0], s[D1]) == d), lambda s: s[Y0] == 1) for d in order]
V1 = [prob(cond(law0, lambda s, d=d: (s[D0], s[D1]) == d), lambda s: s[Y1] == 1) for d in order]
assert V0 == [F(2, 3), F(4, 7), F(3, 7), F(1, 3)] and V1 == [F(2, 3), F(3, 7), F(4, 7), F(1, 3)]
pD0_given = L2[(1, 0)] / (L2[(1, 0)] + L2[(0, 0)])
assert pD0_given == F(308, 959)
a1final = prob(cond(g2, lambda s: s[D1] == 0), lambda s: s[X1] == 1)
assert a1final == F(437, 1918)
assert F(651, 959) * F(1, 6) + F(308, 959) * F(5, 14) == F(437, 1918)
table = {
    "agent0_D0=1": [prob(cond(g, lambda s: s[D0] == 1), lambda s: s[X0] == 1) for g in (g0, g1, g2)],
    "agent1_D1=0": [prob(cond(g, lambda s: s[D1] == 0), lambda s: s[X1] == 1) for g in (g0, g1, g2)],
}
assert table["agent0_D0=1"] == [F(3, 4), F(45, 62), F(45, 62)]
assert table["agent1_D1=0"] == [F(1, 4), F(1, 4), F(437, 1918)]
# direct sensor of represented Y1 = 1
assert prob(cond(cond(law0, lambda s: s[D0] == 1), lambda s: s[Y1] == 1), lambda s: s[X0] == 1) == F(9, 14)
# two fresh samples from agent 1 before a reply (non-alternating schedule)
gg, _, _ = run(law0, (1, 1), (1, 1))
assert prob(cond(gg[-1], lambda s: s[D0] == 1), lambda s: s[X0] == 1) == F(45, 64)
two = {x: sum(ch(d, x) * k10[d][1] ** 2 for d in (0, 1)) for x in (0, 1)}
assert (two[0], two[1]) == (F(21, 64), F(13, 64))
naive_lik = {x0: sum(ch(x1, x0) * pH_X1[x1] ** 2 for x1 in (0, 1)) for x0 in (0, 1)}
naive = F(3, 4) * naive_lik[1] / (F(3, 4) * naive_lik[1] + F(1, 4) * naive_lik[0])
assert naive == F(171, 244)
OUT["A_quoted_numbers"] = "all quoted fractions reproduced"

# (B) history law vs augmented direct conditioning; (C) sender/receiver; (D) reduction --------
checked = 0
for schedule in [(1, 0, 1), (1, 1, 0), (0, 1, 0), (1, 0, 1, 0)]:
    n = len(schedule)
    aug = augmented_history_law(law0, schedule)
    total = F(0)
    for hist in product((0, 1), repeat=n):
        gs, ks, ev = run(law0, schedule, hist)
        direct = marg_E_given_H(aug, hist)
        assert direct == gs[-1], (schedule, hist)
        pdir = sum(p for (s, rec), p in aug.items() if tuple(h for _, h in rec) == hist)
        assert pdir == ev
        total += ev
        # history likelihood L_n(d; h) as product of speaker factors; Gamma_n = L Lambda_0 / Z
        Ln = {}
        for s in law0:
            v = F(1)
            for t, (j, h) in enumerate(zip(schedule, hist)):
                v *= ks[t][s[DIDX[j]]][h]
            Ln[s] = v
        Zn = sum(Ln[s] * law0[s] for s in law0)
        assert Zn == ev and all(gs[-1][s] == Ln[s] * law0[s] / Zn for s in law0)
        for t, (j, h) in enumerate(zip(schedule, hist)):
            gb, ga = gs[t], gs[t + 1]
            i = 1 - j
            for dj in (0, 1):  # sender invariance of the whole private posterior
                assert cond(gb, lambda s: s[DIDX[j]] == dj) == cond(ga, lambda s: s[DIDX[j]] == dj)
            for di in (0, 1):  # receiver: ell via R_{ij,t}, update of the local Z_i marginal
                pb = cond(gb, lambda s: s[DIDX[i]] == di)
                pa = cond(ga, lambda s: s[DIDX[i]] == di)
                zi_idx = (XIDX[i], (A0, A1)[i], YIDX[i])
                zvals = set(tuple(s[k] for k in zi_idx) for s in pb)
                num = {}
                for zv in zvals:
                    sub = cond(pb, lambda s, zv=zv: tuple(s[k] for k in zi_idx) == zv)
                    R = {dj: prob(sub, lambda s, dj=dj: s[DIDX[j]] == dj) for dj in (0, 1)}
                    ell = sum(ks[t][dj][h] * R[dj] for dj in (0, 1))
                    num[zv] = ell * prob(pb, lambda s, zv=zv: tuple(s[k] for k in zi_idx) == zv)
                zz = sum(num.values())
                for zv in zvals:
                    assert num[zv] / zz == prob(pa, lambda s, zv=zv: tuple(s[k] for k in zi_idx) == zv)
                # imem:binary-correlations at this information state
                p = prob(pa, lambda s: s[XIDX[i]] == 1)
                assert prob(pa, lambda s: s[(A0, A1)[i]] == 1) == F(1, 4) + p / 2
                assert prob(pa, lambda s: s[YIDX[i]] == 1) == F(3, 4) - p / 2
                assert prob(pa, lambda s: s[(A0, A1)[i]] == 1 and s[XIDX[i]] == 1) - (F(1, 4) + p / 2) * p == p * (1 - p) / 2
            checked += 1
        # (D) reduced recursion with fixed W and V_j, then reconstruction W lambda
        lam_t = lam(law0)
        Wcond = {d: cond(law0, lambda s, d=d: (s[D0], s[D1]) == d) for d in order}
        for j, h in zip(schedule, hist):
            Vj = {d: prob(Wcond[d], lambda s: s[YIDX[j]] == h) for d in order}
            kred = {}
            for dj in (0, 1):
                m = sum(lam_t[d] for d in order if d[j] == dj)
                kred[dj] = sum(Vj[d] * lam_t[d] for d in order if d[j] == dj) / m
            un = {d: kred[d[j]] * lam_t[d] for d in order}
            zz = sum(un.values())
            lam_t = {d: un[d] / zz for d in order}
        recon = {s: Wcond[(s[D0], s[D1])][s] * lam_t[(s[D0], s[D1])] for s in law0}
        assert recon == gs[-1]
        # (E) product tilt, cross-ratio invariance
        F0 = {d0: F(1) for d0 in (0, 1)}
        F1 = {d1: F(1) for d1 in (0, 1)}
        for t, (j, h) in enumerate(zip(schedule, hist)):
            for dj in (0, 1):
                (F0 if j == 0 else F1)[dj] *= ks[t][dj][h]
        tilt = {d: F0[d[0]] * F1[d[1]] * L0[d] for d in order}
        zt = sum(tilt.values())
        assert all(tilt[d] / zt == lam_t[d] for d in order)
        cr = lam_t[(0, 0)] * lam_t[(1, 1)] / (lam_t[(0, 1)] * lam_t[(1, 0)])
        assert cr == F(81, 49)
    assert total == 1
OUT["B_history_law_vs_augmented_conditioning"] = "exact equality on all histories of schedules (1,0,1),(1,1,0),(0,1,0),(1,0,1,0)"
OUT["C_sender_receiver_steps_checked"] = checked
OUT["D_reduction"] = "W lambda_t reconstructs Gamma_t exactly on all histories above"
OUT["E_cross_ratio_invariant"] = "lambda_t(00)lambda_t(11)/(lambda_t(01)lambda_t(10)) = 81/49 on every history"


def ipf(ref: dict, m0: dict, m1: dict, iters: int = 200) -> dict:
    """Iterative proportional fitting of ref to marginals (float)."""
    q = {d: float(v) for d, v in ref.items()}
    for _ in range(iters):
        for a in (0, 1):
            s = sum(q[d] for d in q if d[0] == a)
            for d in q:
                if d[0] == a:
                    q[d] *= float(m0[a]) / s
        for b in (0, 1):
            s = sum(q[d] for d in q if d[1] == b)
            for d in q:
                if d[1] == b:
                    q[d] *= float(m1[b]) / s
    return q


# exact recovery of lambda_2 from its marginals and lambda_0: solve the 2-parameter tilt exactly.
m0 = {a: sum(L2[d] for d in order if d[0] == a) for a in (0, 1)}
m1 = {b: sum(L2[d] for d in order if d[1] == b) for b in (0, 1)}
q = ipf(L0, m0, m1)
OUT["E_ipf_max_abs_error_lambda2_from_marginals"] = max(abs(q[d] - float(L2[d])) for d in order)
# product of marginals (the replacement the manuscript warns against) instead:
prodm = {d: m0[d[0]] * m1[d[1]] for d in order}
OUT["E_product_of_marginals_differs"] = [str(prodm[d]) for d in order]
OUT["E_lambda2"] = [str(L2[d]) for d in order]

# (F) own past samples are uninformative about E given (D_j, H_{1:t}) --------------------------
aug = augmented_history_law(law0, (1, 0, 1))
for hist in product((0, 1), repeat=3):
    for d1 in (0, 1):
        sub = {k: p for k, p in aug.items() if tuple(h for _, h in k[1]) == hist and k[0][D1] == d1}
        z = sum(sub.values())
        base = {}
        for (s, rec), p in sub.items():
            base[s] = base.get(s, F(0)) + p / z
        # condition additionally on the speaker's own first and third samples (equal to H here,
        # identity encoder); with a noisy encoder they would differ, so also test a noisy one below
        for yt in (0, 1):
            sub2 = {k: p for k, p in sub.items() if k[1][0][0] == yt}
            if sum(sub2.values()) == 0:
                continue
            z2 = sum(sub2.values())
            b2 = {}
            for (s, rec), p in sub2.items():
                b2[s] = b2.get(s, F(0)) + p / z2
            assert b2 == base
OUT["F_own_samples_identity_encoder"] = "trivially equal (sample = message)"


def augmented_noisy(law0: dict, schedule: tuple, flip: F) -> dict:
    """Same as augmented_history_law but H_t = Ytilde_t passed through a binary symmetric
    encoder with flip probability `flip`, so the sample is not recoverable from the message."""
    joint = {(s, ()): p for s, p in law0.items()}
    for j in schedule:
        num: dict = {}
        den: dict = {}
        for (s, rec), p in joint.items():
            hs = tuple(h for _, h in rec)
            key = (s[DIDX[j]], hs)
            den[key] = den.get(key, F(0)) + p
            k2 = (s[YIDX[j]], s[DIDX[j]], hs)
            num[k2] = num.get(k2, F(0)) + p
        new = {}
        for (s, rec), p in joint.items():
            hs = tuple(h for _, h in rec)
            key = (s[DIDX[j]], hs)
            for yt in (0, 1):
                samp = num.get((yt, s[DIDX[j]], hs), F(0)) / den[key]
                for h in (0, 1):
                    enc = 1 - flip if h == yt else flip
                    new[(s, rec + ((yt, h),))] = p * samp * enc
        joint = new
    assert sum(joint.values()) == 1
    return joint


augn = augmented_noisy(law0, (1, 0, 1), F(1, 5))
worst = 0
for hist in product((0, 1), repeat=3):
    for d1 in (0, 1):
        sub = {k: p for k, p in augn.items() if tuple(h for _, h in k[1]) == hist and k[0][D1] == d1}
        z = sum(sub.values())
        base = {}
        for (s, rec), p in sub.items():
            base[s] = base.get(s, F(0)) + p / z
        for y1 in (0, 1):
            for y3 in (0, 1):
                sub2 = {k: p for k, p in sub.items() if k[1][0][0] == y1 and k[1][2][0] == y3}
                z2 = sum(sub2.values())
                b2 = {}
                for (s, rec), p in sub2.items():
                    b2[s] = b2.get(s, F(0)) + p / z2
                assert b2 == base
                worst += 1
OUT["F_own_past_samples_noisy_encoder_cases"] = worst


# (G) information identity, floats --------------------------------------------------------------
def H_ent(pmf: dict) -> float:
    return -sum(float(p) * math.log(float(p)) for p in pmf.values() if p > 0)


def marg(joint: dict, f) -> dict:
    out = {}
    for k, p in joint.items():
        key = f(k)
        out[key] = out.get(key, F(0)) + p
    return out


def cmi(joint: dict, fa, fb, fc) -> float:
    """I(A;B|C) = H(A,C)+H(B,C)-H(A,B,C)-H(C)."""
    return (H_ent(marg(joint, lambda k: (fa(k), fc(k)))) + H_ent(marg(joint, lambda k: (fb(k), fc(k))))
            - H_ent(marg(joint, lambda k: (fa(k), fb(k), fc(k)))) - H_ent(marg(joint, fc)))


jointEH = {}
for (s, rec), p in augmented_history_law(law0, (1, 0, 1)).items():
    key = (s, tuple(h for _, h in rec))
    jointEH[key] = jointEH.get(key, F(0)) + p
Zf = lambda k: k[0][:6]
res = []
for i in (0, 1):
    Di = lambda k, i=i: k[0][DIDX[i]]
    Do = lambda k, i=i: k[0][DIDX[1 - i]]
    Hf = lambda k: k[1]
    lhs = cmi(jointEH, Zf, Hf, Di)
    rhs = cmi(jointEH, Zf, Do, Di) - cmi(jointEH, Zf, Do, lambda k, i=i: (k[0][DIDX[i]], k[1]))
    zero = cmi(jointEH, Zf, Hf, lambda k: (k[0][D0], k[0][D1]))
    res.append({"agent": i, "I(Z;H|D_i)": lhs, "rhs": rhs, "residual": lhs - rhs, "I(Z;H|D)": zero})
OUT["G_information_identity"] = res

# (H) free-energy identities, floats ------------------------------------------------------------
rng = random.Random(20261029)
schedule, hist = (1, 0, 1), (1, 1, 0)
gs, ks, ev = run(law0, schedule, hist)
Gn = gs[-1]
Ln = {}
for s in law0:
    v = F(1)
    for t, (j, h) in enumerate(zip(schedule, hist)):
        v *= ks[t][s[DIDX[j]]][h]
    Ln[s] = v
logZ = math.log(float(ev))
states = list(law0)


def kl(q: dict, p: dict) -> float:
    return sum(qv * math.log(qv / float(p[s])) for s, qv in q.items() if qv > 0)


fe_res = []
for trial in range(5):
    w = [rng.random() + 1e-3 for _ in states]
    zw = sum(w)
    Q = {s: wi / zw for s, wi in zip(states, w)}
    Fn = kl(Q, law0) - sum(Q[s] * math.log(float(Ln[s])) for s in states)
    fe_res.append(Fn - (kl(Q, Gn) - logZ))
    # reduced: Q = Q^Z(d,.) nu(d)
    nu = {d: sum(Q[s] for s in states if (s[D0], s[D1]) == d) for d in order}
    Wd = {d: cond(law0, lambda s, d=d: (s[D0], s[D1]) == d) for d in order}
    lam_n = lam(Gn)
    condKL = 0.0
    for d in order:
        qz = {s: Q[s] / nu[d] for s in states if (s[D0], s[D1]) == d}
        condKL += nu[d] * kl(qz, Wd[d])
    red = kl(nu, lam_n) + condKL - logZ
    fe_res.append(Fn - red)
    # F^D(nu) first line vs second line
    FD1 = kl(nu, lam(law0)) - sum(nu[d] * math.log(float(Ln[next(s for s in states if (s[D0], s[D1]) == d)])) for d in order)
    fe_res.append(FD1 - (kl(nu, lam_n) - logZ))
# private objective of agent 0 at d0 = 1 and the local split over A_0
d0 = 1
base = cond(law0, lambda s: s[D0] == d0)
Zi = sum(Ln[s] * base[s] for s in base)
Pin = cond(Gn, lambda s: s[D0] == d0)
sts = list(base)
w = [rng.random() + 1e-3 for _ in sts]
Q = {s: wi / sum(w) for s, wi in zip(sts, w)}
Fi = kl(Q, base) - sum(Q[s] * math.log(float(Ln[s])) for s in sts)
fe_res.append(Fi - (kl(Q, Pin) - math.log(float(Zi))))
# local objective on Z_0 = (A0, X0, Y0)
zi = lambda s: (s[A0], s[X0], s[Y0])
zeta0 = marg(base, zi)
bnum = marg({s: Ln[s] * base[s] for s in base}, zi)
b = {z: bnum[z] / zeta0[z] for z in zeta0}
zetan = marg(Pin, zi)
assert all(zetan[z] == b[z] * zeta0[z] / Zi for z in zeta0)
zl = list(zeta0)
w = [rng.random() + 1e-3 for _ in zl]
Qi = {z: wi / sum(w) for z, wi in zip(zl, w)}
Floc = sum(Qi[z] * (math.log(Qi[z] / float(zeta0[z])) - math.log(float(b[z]))) for z in zl)
s_i = {a: sum(Qi[z] for z in zl if z[0] == a) for a in (0, 1)}
s_n = {a: sum(float(zetan[z]) for z in zl if z[0] == a) for a in (0, 1)}
split = sum(s_i[a] * math.log(s_i[a] / s_n[a]) for a in (0, 1))
for a in (0, 1):
    split += s_i[a] * sum((Qi[z] / s_i[a]) * math.log((Qi[z] / s_i[a]) / (float(zetan[z]) / s_n[a]))
                          for z in zl if z[0] == a)
fe_res.append(Floc - (split - math.log(float(Zi))))
OUT["H_free_energy_max_abs_residual"] = max(abs(r) for r in fe_res)


# (I) parity counterexample --------------------------------------------------------------------
def parity_law(rho: F) -> dict:
    """State (x0,x1,a0,a1,y0,y1,c,d1); B=c with P(B=0)=rho, X1 = X0 xor B; D0 uninformative."""
    law = {}
    for x0, bb, a0, a1, y0, y1, d1 in product((0, 1), repeat=7):
        x1 = x0 ^ bb
        p = (F(1, 2) * (rho if bb == 0 else 1 - rho) * ch(a0, x0) * ch(a1, x1)
             * ch(y0, 1 - x0) * ch(y1, 1 - x1) * ch(d1, x1))
        law[(x0, x1, a0, a1, y0, y1, bb, d1)] = p
    assert sum(law.values()) == 1
    return law


par = []
for rho in (F(1, 3), F(1, 2), F(9, 10)):
    law = parity_law(rho)
    rows = {}
    for c in (0, 1):
        lc = cond(law, lambda s, c=c: s[6] == c)
        # local laws of (A0,X0,Y0) and of (A1,X1,Y1,D1) given C
        loc0 = marg(lc, lambda s: (s[2], s[0], s[4]))
        loc1 = marg(lc, lambda s: (s[3], s[1], s[5], s[7]))
        k = {d: prob(cond(lc, lambda s, d=d: s[7] == d), lambda s: s[5] == 1) for d in (0, 1)}
        un = {s: p * (k[s[7]] if True else 0) for s, p in lc.items()}  # message H=1
        z = sum(un.values())
        post = {s: p / z for s, p in un.items()}
        rows[c] = {
            "loc0": loc0, "loc1": loc1, "k": k, "PH1": z,
            "PX0": prob(post, lambda s: s[0] == 1), "PA0": prob(post, lambda s: s[2] == 1),
            "PY0": prob(post, lambda s: s[4] == 1),
            "sender_after": marg(cond(post, lambda s: s[7] == 0), lambda s: (s[3], s[1], s[5])),
            "sender_before": marg(cond(lc, lambda s: s[7] == 0), lambda s: (s[3], s[1], s[5])),
        }
    assert rows[0]["loc0"] == rows[1]["loc0"] and rows[0]["loc1"] == rows[1]["loc1"]
    assert rows[0]["k"] == rows[1]["k"] == {0: F(5, 8), 1: F(3, 8)}
    assert rows[0]["PH1"] == rows[1]["PH1"] == F(1, 2)
    assert (rows[0]["PX0"], rows[1]["PX0"]) == (F(7, 16), F(9, 16))
    assert (rows[0]["PA0"], rows[1]["PA0"]) == (F(15, 32), F(17, 32))
    assert (rows[0]["PY0"], rows[1]["PY0"]) == (F(17, 32), F(15, 32))
    assert rows[0]["sender_after"] == rows[0]["sender_before"] == rows[1]["sender_after"]
    par.append(str(rho))
OUT["I_parity_counterexample_rhos"] = par

# (J) product preparation: own represented-state posterior never changes ---------------------
lawp = preparation(agree=F(1, 2))
for schedule in [(1, 0, 1, 0)]:
    for hist in product((0, 1), repeat=4):
        gs, _, _ = run(lawp, schedule, hist)
        for i in (0, 1):
            for di in (0, 1):
                ref = marg(cond(gs[0], lambda s: s[DIDX[i]] == di), lambda s: (s[XIDX[i]], (A0, A1)[i], s[YIDX[i]]))
                for g in gs:
                    assert marg(cond(g, lambda s: s[DIDX[i]] == di), lambda s: (s[XIDX[i]], (A0, A1)[i], s[YIDX[i]])) == ref
OUT["J_product_preparation_own_posterior_constant"] = True


# (K) three-valued private records, random rational model --------------------------------------
def rand_frac_vec(n: int, r: random.Random) -> list:
    w = [r.randint(1, 9) for _ in range(n)]
    s = sum(w)
    return [F(x, s) for x in w]


r = random.Random(7)
pX = rand_frac_vec(4, r)  # joint of (X0, X1) bits
Lk = [[rand_frac_vec(3, r) for _ in range(2)] for _ in range(2)]  # L_i(x, d), d in {0,1,2}
Kk = [[rand_frac_vec(2, r) for _ in range(2)] for _ in range(2)]  # K_i(x, y)
law3 = {}
for (x0, x1), px in zip(product((0, 1), repeat=2), pX):
    for y0, y1, d0, d1 in product((0, 1), (0, 1), (0, 1, 2), (0, 1, 2)):
        law3[(x0, x1, y0, y1, d0, d1)] = px * Kk[0][x0][y0] * Kk[1][x1][y1] * Lk[0][x0][d0] * Lk[1][x1][d1]
assert sum(law3.values()) == 1
DI3, YI3 = (4, 5), (2, 3)


def spk3(g: dict, j: int) -> dict:
    out = {}
    for d in (0, 1, 2):
        loc = cond(g, lambda s, d=d: s[DI3[j]] == d)
        out[d] = {h: prob(loc, lambda s, h=h: s[YI3[j]] == h) for h in (0, 1)}
    return out


lam30 = {d: prob(law3, lambda s, d=d: (s[4], s[5]) == d) for d in product((0, 1, 2), repeat=2)}
g = law3
Fk = [{d: F(1) for d in (0, 1, 2)} for _ in range(2)]
for j, h in zip((1, 0, 1, 0, 0), (1, 0, 0, 1, 1)):
    k = spk3(g, j)
    for d in (0, 1, 2):
        Fk[j][d] *= k[d][h]
    un = {s: p * k[s[DI3[j]]][h] for s, p in g.items()}
    z = sum(un.values())
    g = {s: p / z for s, p in un.items()}
lam3 = {d: prob(g, lambda s, d=d: (s[4], s[5]) == d) for d in lam30}
tilt = {d: Fk[0][d[0]] * Fk[1][d[1]] * lam30[d] for d in lam30}
zt = sum(tilt.values())
assert all(tilt[d] / zt == lam3[d] for d in lam30)


def cross(l: dict, a: int, a2: int, b: int, b2: int) -> F:
    return l[(a, b)] * l[(a2, b2)] / (l[(a, b2)] * l[(a2, b)])


assert all(cross(lam3, a, a2, b, b2) == cross(lam30, a, a2, b, b2)
           for a, a2, b, b2 in product((0, 1, 2), repeat=4) if a != a2 and b != b2)
OUT["K_three_valued_product_tilt_and_cross_ratios"] = "exact; free parameters of reachable lambda: (3-1)+(3-1)=4 versus 3*3-1=8"

OUT["wall_seconds"] = round(time.perf_counter() - T0, 3)
OUT["scope"] = ("Exact enumeration of the stated finite models only; supports the displayed identities on "
                "these instances and does not prove the general theorems.")
Path(__file__).with_name("binary_exact_output.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n",
                                                               encoding="utf-8")
print(json.dumps(OUT, indent=2, default=str))
