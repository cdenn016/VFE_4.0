r"""Exact rational recomputation of the two-way binary exchange.

Lens D08-IMEM2, review of MAgent Theory 2.0 (2026-10-29).
Scope: appendices/interacting_memory.tex lines 325-424 and
companion/c04_interacting_memory.tex lines 106-169.

The protocol is implemented directly from its definition
(imem:message-kernel, imem:history-law): at each step the speaker j samples
H_{t+1} from P(Y_j | D_j, H_{1:t}) computed from the current joint law of
(E, H_{1:t}); identity encoder.  Nothing from imem:reduction is used to
generate the law; the reduction Gamma_t = W lambda_t is then CHECKED.
All arithmetic is exact (fractions.Fraction) except the mutual-information
check, which uses floats on exact probabilities.
"""
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as Fr
from typing import Dict, List, Sequence, Tuple

Q3 = Fr(3, 4)
Q1 = Fr(1, 4)
# E = (X0, X1, A0, A1, Y0, Y1, D0, D1)
IX0, IX1, IA0, IA1, IY0, IY1, ID0, ID1 = range(8)
IX = (IX0, IX1)
IA = (IA0, IA1)
IY = (IY0, IY1)
ID = (ID0, ID1)

checks: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def same(p: Fr, a: int, b: int) -> Fr:
    """P(var = a | parent = b) for a 3/4-faithful binary channel."""
    return Q3 if a == b else Q1


def preparation() -> Dict[Tuple[int, ...], Fr]:
    r"""\Lambda_0 for imem:binary-preparation."""
    lam: Dict[Tuple[int, ...], Fr] = {}
    for e in itertools.product((0, 1), repeat=8):
        x0, x1, a0, a1, y0, y1, d0, d1 = e
        p = Fr(1, 2) * same(None, x1, x0)
        for xi, ai, yi, di in ((x0, a0, y0, d0), (x1, a1, y1, d1)):
            p *= same(None, ai, xi)               # P(A_i = X_i | X_i) = 3/4
            p *= Q3 if yi == 1 - xi else Q1       # P(Y_i = 1 - X_i | X_i) = 3/4
            p *= same(None, di, xi)               # P(D_i = X_i | X_i) = 3/4
        lam[e] = p
    return lam


def run_protocol(lam0: Dict[Tuple[int, ...], Fr], schedule: Sequence[int]
                 ) -> List[Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Fr]]:
    """Joint laws of (E, H_{1:t}) for t = 0..len(schedule), by definition."""
    joints = [{(e, ()): p for e, p in lam0.items()}]
    for j in schedule:
        J = joints[-1]
        # speaker sampling law S_j(d_j, h) = P(Y_j = 1 | D_j = d_j, H_{1:t} = h)
        num: Dict[Tuple[int, Tuple[int, ...]], Fr] = {}
        den: Dict[Tuple[int, Tuple[int, ...]], Fr] = {}
        for (e, h), p in J.items():
            key = (e[ID[j]], h)
            den[key] = den.get(key, Fr(0)) + p
            if e[IY[j]] == 1:
                num[key] = num.get(key, Fr(0)) + p
        Jn: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Fr] = {}
        for (e, h), p in J.items():
            s = num.get((e[ID[j]], h), Fr(0)) / den[(e[ID[j]], h)]
            Jn[(e, h + (1,))] = p * s
            Jn[(e, h + (0,))] = p * (1 - s)
        joints.append(Jn)
    return joints


def cond(J, pred_num, pred_den) -> Fr:
    n = sum((p for (e, h), p in J.items() if pred_num(e, h)), Fr(0))
    d = sum((p for (e, h), p in J.items() if pred_den(e, h)), Fr(0))
    return n / d


lam0 = preparation()
check("preparation normalized", sum(lam0.values()) == 1)
check("all preparation atoms positive", all(p > 0 for p in lam0.values()))

# Local bank form: A_i fair and P(X_i = A_i | A_i) = 3/4
for i in (0, 1):
    pa1 = sum(p for e, p in lam0.items() if e[IA[i]] == 1)
    pxa = sum(p for e, p in lam0.items() if e[IA[i]] == 1 and e[IX[i]] == 1) / pa1
    check(f"A_{i} fair, P(X_{i}=A_{i}|A_{i})=3/4", pa1 == Fr(1, 2) and pxa == Q3,
          f"P(A=1)={pa1}, P(X=1|A=1)={pxa}")

schedule = [1, 0, 1, 0]
joints = run_protocol(lam0, schedule)
for t, J in enumerate(joints):
    check(f"history law normalized at t={t}", sum(J.values()) == 1)

J0, J1, J2 = joints[0], joints[1], joints[2]

# Initial sampling probabilities k_{1,0}(1|d)
k10 = {d: cond(J0, lambda e, h, d=d: e[ID1] == d and e[IY1] == 1,
               lambda e, h, d=d: e[ID1] == d) for d in (0, 1)}
check("k_{1,0}(1|0)=5/8, k_{1,0}(1|1)=3/8", k10[0] == Fr(5, 8) and k10[1] == Fr(3, 8), str(k10))

# imem:binary-first-likelihood
pH1_X1 = {x: cond(J1, lambda e, h, x=x: e[IX1] == x and h == (1,),
                  lambda e, h, x=x: e[IX1] == x) for x in (0, 1)}
pH1_X0 = {x: cond(J1, lambda e, h, x=x: e[IX0] == x and h == (1,),
                  lambda e, h, x=x: e[IX0] == x) for x in (0, 1)}
check("P(H1=1|X1=0,1) = 9/16, 7/16", pH1_X1 == {0: Fr(9, 16), 1: Fr(7, 16)}, str(pH1_X1))
check("P(H1=1|X0=0,1) = 17/32, 15/32", pH1_X0 == {0: Fr(17, 32), 1: Fr(15, 32)}, str(pH1_X0))
check("P(H1=1|X0,D0) = P(H1=1|X0) (receiver record does not enter)",
      all(cond(J1, lambda e, h, x=x, d=d: e[IX0] == x and e[ID0] == d and h == (1,),
               lambda e, h, x=x, d=d: e[IX0] == x and e[ID0] == d) == pH1_X0[x]
          for x in (0, 1) for d in (0, 1)))

# imem:binary-first-posterior and the D0=0 value
p0_after1 = {d: cond(J1, lambda e, h, d=d: e[ID0] == d and h == (1,) and e[IX0] == 1,
                     lambda e, h, d=d: e[ID0] == d and h == (1,)) for d in (0, 1)}
check("P(X0=1|D0=1,H1=1)=45/62 and P(X0=1|D0=0,H1=1)=5/22",
      p0_after1 == {1: Fr(45, 62), 0: Fr(5, 22)}, str(p0_after1))

# imem:binary-reply: k_{0,1}(1 | D0, H1=1) = P(Y0=1 | D0, H1=1)
k01 = {d: cond(J1, lambda e, h, d=d: e[ID0] == d and h == (1,) and e[IY0] == 1,
               lambda e, h, d=d: e[ID0] == d and h == (1,)) for d in (0, 1)}
check("k_{0,1}(1|D0=0,H1=1)=7/11, k_{0,1}(1|D0=1,H1=1)=12/31",
      k01 == {0: Fr(7, 11), 1: Fr(12, 31)}, str(k01))
k00 = {d: cond(J0, lambda e, h, d=d: e[ID0] == d and e[IY0] == 1,
               lambda e, h, d=d: e[ID0] == d) for d in (0, 1)}
check("reply differs from agent 0's initial sampling law", k00[0] != k01[0] and k00[1] != k01[1],
      f"initial {k00}, reply {k01}")

# imem:binary-private-weights
order = [(0, 0), (0, 1), (1, 0), (1, 1)]


def lam_t(J, h):
    ph = sum((p for (e, hh), p in J.items() if hh == h), Fr(0))
    return [sum((p for (e, hh), p in J.items() if hh == h and (e[ID0], e[ID1]) == dd), Fr(0)) / ph
            for dd in order], ph


l0, _ = lam_t(J0, ())
l1, ph1 = lam_t(J1, (1,))
l2, ph12 = lam_t(J2, (1, 1))
check("lambda_0 = (9,7,7,9)/32", l0 == [Fr(9, 32), Fr(7, 32), Fr(7, 32), Fr(9, 32)], str(l0))
check("lambda_1|H1=1 = (45,21,35,27)/128", l1 == [Fr(45, 128), Fr(21, 128), Fr(35, 128), Fr(27, 128)], str(l1))
check("lambda_2|H1=H2=1 = (105/242, 49/242, 70/341, 54/341)",
      l2 == [Fr(105, 242), Fr(49, 242), Fr(70, 341), Fr(54, 341)], str(l2))
check("P(H1=1)=1/2", ph1 == Fr(1, 2), str(ph1))
check("P(H2=1|H1=1)=33/64", ph12 / ph1 == Fr(33, 64), str(ph12 / ph1))

# imem:binary-reconstruction
W_X0 = [cond(J0, lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd and e[IX0] == 1,
             lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd) for dd in order]
W_X1 = [cond(J0, lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd and e[IX1] == 1,
             lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd) for dd in order]
V0 = [cond(J0, lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd and e[IY0] == 1,
           lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd) for dd in order]
V1 = [cond(J0, lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd and e[IY1] == 1,
           lambda e, h, dd=dd: (e[ID0], e[ID1]) == dd) for dd in order]
check("P(X0=1|D) = (1/6, 5/14, 9/14, 5/6)", W_X0 == [Fr(1, 6), Fr(5, 14), Fr(9, 14), Fr(5, 6)], str(W_X0))
check("V_0(D;{1}) = (2/3, 4/7, 3/7, 1/3)", V0 == [Fr(2, 3), Fr(4, 7), Fr(3, 7), Fr(1, 3)], str(V0))
check("V_1(D;{1}) = (2/3, 3/7, 4/7, 1/3)", V1 == [Fr(2, 3), Fr(3, 7), Fr(4, 7), Fr(1, 3)], str(V1))
check("P(X1=1|D) = (1/6, 9/14, 5/14, 5/6) (exchange of D0,D1)",
      W_X1 == [Fr(1, 6), Fr(9, 14), Fr(5, 14), Fr(5, 6)], str(W_X1))

# Reply kernel from the reduced readout kappa_0^{lambda_1}
for d0 in (0, 1):
    idx = [k for k, dd in enumerate(order) if dd[0] == d0]
    red = sum(V0[k] * l1[k] for k in idx) / sum(l1[k] for k in idx)
    check(f"reduced readout reproduces k_(0,1)(1|D0={d0})", red == k01[d0], str(red))

# Final posterior of agent 1 and the table imem:binary-exchange
pD0_1 = l2[2] / (l2[0] + l2[2])
check("P(D0=1|D1=0,H=11) = 308/959", pD0_1 == Fr(308, 959), f"{pD0_1} (lowest terms)")
check("308/959 and 651/959 are not in lowest terms (=44/137, 93/137)",
      Fr(308, 959) == Fr(44, 137) and Fr(651, 959) == Fr(93, 137) and math.gcd(308, 959) == 7)
p1_final = cond(J2, lambda e, h: e[ID1] == 0 and h == (1, 1) and e[IX1] == 1,
                lambda e, h: e[ID1] == 0 and h == (1, 1))
check("P(X1=1|D1=0,H1=H2=1) = 437/1918 (brute force)", p1_final == Fr(437, 1918), str(p1_final))
check("437/1918 equals the displayed reconstruction sum",
      Fr(651, 959) * Fr(1, 6) + Fr(308, 959) * Fr(5, 14) == Fr(437, 1918))

table = {}
for t, (J, h) in enumerate(((J0, ()), (J1, (1,)), (J2, (1, 1)))):
    a0 = cond(J, lambda e, hh, h=h: e[ID0] == 1 and hh == h and e[IX0] == 1,
              lambda e, hh, h=h: e[ID0] == 1 and hh == h)
    a1 = cond(J, lambda e, hh, h=h: e[ID1] == 0 and hh == h and e[IX1] == 1,
              lambda e, hh, h=h: e[ID1] == 0 and hh == h)
    table[t] = (a0, a1)
check("table imem:binary-exchange",
      table == {0: (Q3, Q1), 1: (Fr(45, 62), Q1), 2: (Fr(45, 62), Fr(437, 1918))}, str(table))

# Exact reduction Gamma_t = W lambda_t, sender invariance, binary-correlations,
# for every history of length <= 4 (schedule 1,0,1,0)
W_full: Dict[Tuple[Tuple[int, int], Tuple[int, ...]], Fr] = {}
pD = {dd: sum(p for e, p in lam0.items() if (e[ID0], e[ID1]) == dd) for dd in order}
for e, p in lam0.items():
    W_full[((e[ID0], e[ID1]), e[:6])] = p / pD[(e[ID0], e[ID1])]

red_ok = True
sender_ok = True
corr_ok = True
interior_ok = True
n_hist = 0
for t, J in enumerate(joints):
    hists = sorted({h for (e, h) in J})
    for h in hists:
        n_hist += 1
        ph = sum((p for (e, hh), p in J.items() if hh == h), Fr(0))
        lam = {dd: sum((p for (e, hh), p in J.items() if hh == h and (e[ID0], e[ID1]) == dd), Fr(0)) / ph
               for dd in order}
        for (e, hh), p in J.items():
            if hh != h:
                continue
            dd = (e[ID0], e[ID1])
            if p / ph != W_full[(dd, e[:6])] * lam[dd]:
                red_ok = False
        for i in (0, 1):
            for di in (0, 1):
                sub = {e: p for (e, hh), p in J.items() if hh == h and e[ID[i]] == di}
                tot = sum(sub.values())
                p_x = sum(p for e, p in sub.items() if e[IX[i]] == 1) / tot
                p_a = sum(p for e, p in sub.items() if e[IA[i]] == 1) / tot
                p_y = sum(p for e, p in sub.items() if e[IY[i]] == 1) / tot
                p_ax = sum(p for e, p in sub.items() if e[IA[i]] == 1 and e[IX[i]] == 1) / tot
                p_yx = sum(p for e, p in sub.items() if e[IY[i]] == 1 and e[IX[i]] == 1) / tot
                ok = (p_a == Q1 + p_x / 2 and p_y == Q3 - p_x / 2
                      and p_ax - p_a * p_x == p_x * (1 - p_x) / 2
                      and p_yx - p_y * p_x == -p_x * (1 - p_x) / 2)
                corr_ok &= ok
                interior_ok &= (0 < p_x < 1)
        # sender invariance: the speaker of step t+1 keeps its whole private posterior
        if t < len(schedule):
            j = schedule[t]
            Jn = joints[t + 1]
            for hn in (0, 1):
                for dj in (0, 1):
                    old = {e: p for (e, hh), p in J.items() if hh == h and e[ID[j]] == dj}
                    new = {e: p for (e, hh), p in Jn.items() if hh == h + (hn,) and e[ID[j]] == dj}
                    so, sn = sum(old.values()), sum(new.values())
                    if any(old[e] / so != new[e] / sn for e in old):
                        sender_ok = False
check(f"Gamma_t = W lambda_t exactly on all {n_hist} histories (t<=4)", red_ok)
check("sender invariance of the entire private posterior, all histories", sender_ok)
check("imem:binary-correlations at every information state (t<=4)", corr_ok)
check("cue probabilities strictly inside (0,1) at every history (nonzero covariance)", interior_ok)

# Direct observation of represented Y1 = 1
pY1_X0 = {x: sum(p for e, p in lam0.items() if e[IX0] == x and e[IY1] == 1) /
          sum(p for e, p in lam0.items() if e[IX0] == x) for x in (0, 1)}
post_dir = (sum(p for e, p in lam0.items() if e[ID0] == 1 and e[IY1] == 1 and e[IX0] == 1) /
            sum(p for e, p in lam0.items() if e[ID0] == 1 and e[IY1] == 1))
check("direct Y1=1: cue likelihoods 5/8, 3/8 and posterior 9/14",
      pY1_X0 == {0: Fr(5, 8), 1: Fr(3, 8)} and post_dir == Fr(9, 14), f"{pY1_X0}, {post_dir}")

# Two fresh samples from agent 1 before a reply (schedule 1,1)
jr = run_protocol(lam0, [1, 1])[2]
pHH_X1 = {x: cond(jr, lambda e, h, x=x: e[IX1] == x and h == (1, 1),
                  lambda e, h, x=x: e[IX1] == x) for x in (0, 1)}
check("P(H1=H2=1|X1=0,1) = 21/64, 13/64", pHH_X1 == {0: Fr(21, 64), 1: Fr(13, 64)}, str(pHH_X1))
post_rep = cond(jr, lambda e, h: e[ID0] == 1 and h == (1, 1) and e[IX0] == 1,
                lambda e, h: e[ID0] == 1 and h == (1, 1))
check("receiver posterior after two fresh samples = 45/64", post_rep == Fr(45, 64), str(post_rep))
sq = {0: Fr(9, 16) ** 2, 1: Fr(7, 16) ** 2}
lik_sq = {x0: sum(same(None, x1, x0) * sq[x1] for x1 in (0, 1)) for x0 in (0, 1)}
post_sq = Q3 * lik_sq[1] / (Q3 * lik_sq[1] + Q1 * lik_sq[0])
check("squared once-marginalized likelihoods give 171/244", post_sq == Fr(171, 244), str(post_sq))

# Information identity imem:information-bound (companion lines 227-232), n = 2
J = J2


def H_ent(margs: Sequence[int], cond_on: Sequence[int]) -> float:
    """Conditional entropy H(margs | cond_on) with indices into (Z, D, H) layout."""
    def key(e, h, idx):
        full = e + h
        return tuple(full[k] for k in idx)
    pj: Dict[tuple, Fr] = {}
    pc: Dict[tuple, Fr] = {}
    for (e, h), p in J.items():
        kj = key(e, h, list(margs) + list(cond_on))
        kc = key(e, h, cond_on)
        pj[kj] = pj.get(kj, Fr(0)) + p
        pc[kc] = pc.get(kc, Fr(0)) + p
    s = 0.0
    for kj, p in pj.items():
        if p == 0:
            continue
        kc = kj[len(margs):]
        s -= float(p) * math.log(float(p / pc[kc]))
    return s


Zidx = list(range(6))
Hidx = [8, 9]
for i in (0, 1):
    Di, Do = [ID[i]], [ID[1 - i]]
    I_ZH = H_ent(Zidx, Di) - H_ent(Zidx, Di + Hidx)
    I_ZD = H_ent(Zidx, Di) - H_ent(Zidx, Di + Do)
    I_ZD_H = H_ent(Zidx, Di + Hidx) - H_ent(Zidx, Di + Hidx + Do)
    res = I_ZH - (I_ZD - I_ZD_H)
    check(f"information identity, i={i}, n=2", abs(res) < 1e-12,
          f"I(Z;H|D_i)={I_ZH:.6f}, rhs={I_ZD - I_ZD_H:.6f}, residual={res:.2e}")

# Reduced free-energy identity comp:eq-interaction-free-energy / imem:private-record-free-energy
rng = random.Random(20261029)
h = (1, 1)
Lh = {dd: sum((p for (e, hh), p in J2.items() if hh == h and (e[ID0], e[ID1]) == dd), Fr(0)) / pD[dd]
      for dd in order}
Zt = sum(Lh[dd] * pD[dd] for dd in order)
lamh = {dd: Lh[dd] * pD[dd] / Zt for dd in order}
worst = 0.0
for _ in range(5):
    w = [rng.random() + 0.05 for _ in order]
    nu = {dd: w[k] / sum(w) for k, dd in enumerate(order)}
    lhs = sum(nu[dd] * (math.log(nu[dd] / float(pD[dd])) - math.log(float(Lh[dd]))) for dd in order)
    rhs = sum(nu[dd] * math.log(nu[dd] / float(lamh[dd])) for dd in order) - math.log(float(Zt))
    worst = max(worst, abs(lhs - rhs))
check("reduced free-energy identity at h=(1,1), 5 random nu", worst < 1e-12, f"max residual {worst:.2e}")

# History likelihood imem:history-likelihood: L_2(d;h) = k_{1,0}(h1|d1) k_{0,1}(h2|d0;h1),
# evidence Z_2(h) = sum_d lambda_0(d) L_2(d;h), for all four two-message histories.
Zs = {}
lik_ok = True
for h in itertools.product((0, 1), repeat=2):
    k1 = {d: (k10[d] if h[0] == 1 else 1 - k10[d]) for d in (0, 1)}
    s01 = {d: cond(J1, lambda e, hh, d=d: e[ID0] == d and hh == (h[0],) and e[IY0] == 1,
                   lambda e, hh, d=d: e[ID0] == d and hh == (h[0],)) for d in (0, 1)}
    k2 = {d: (s01[d] if h[1] == 1 else 1 - s01[d]) for d in (0, 1)}
    L2 = {dd: k1[dd[1]] * k2[dd[0]] for dd in order}
    Zs[h] = sum(pD[dd] * L2[dd] for dd in order)
    ph = sum((p for (e, hh), p in J2.items() if hh == h), Fr(0))
    lik_ok &= (Zs[h] == ph)
    # Bayes: lambda_2 = lambda_0 L_2 / Z_2
    lam2 = {dd: sum((p for (e, hh), p in J2.items() if hh == h and (e[ID0], e[ID1]) == dd), Fr(0)) / ph
            for dd in order}
    lik_ok &= all(lam2[dd] == pD[dd] * L2[dd] / Zs[h] for dd in order)
check("history likelihood and evidence Z_2 for all four histories; lambda_2 = lambda_0 L_2/Z_2",
      lik_ok, "Z_2: " + ", ".join(f"{''.join(map(str, h))}={Zs[h]}" for h in Zs))
check("Z_2(11) = 33/128, Z_2(01) = Z_2(10) = 31/128", Zs[(1, 1)] == Fr(33, 128) and Zs[(0, 1)] == Fr(31, 128)
      and Zs[(1, 0)] == Fr(31, 128) and Zs[(0, 0)] == Fr(33, 128))

# Full free-energy identity imem:full-free-energy-identity on the 256-atom space, h = (1,1)
L2full = {e: Lh[(e[ID0], e[ID1])] for e in lam0}
Gam = {e: lam0[e] * L2full[e] / Zt for e in lam0}
worst = 0.0
for _ in range(3):
    wts = {e: rng.random() + 0.01 for e in lam0}
    tot = sum(wts.values())
    Qd = {e: v / tot for e, v in wts.items()}
    lhs = sum(Qd[e] * (math.log(Qd[e] / float(lam0[e])) - math.log(float(L2full[e]))) for e in lam0)
    rhs = sum(Qd[e] * math.log(Qd[e] / float(Gam[e])) for e in lam0) - math.log(float(Zt))
    worst = max(worst, abs(lhs - rhs))
check("full free-energy identity F = KL(Q||Gamma_2) - log Z_2, 3 random Q", worst < 1e-12, f"max residual {worst:.2e}")

# Extension check (not a manuscript claim): lambda_1 is recoverable from the next two
# speakers' message kernels at ALL private values (agent 0 at t=1, agent 1 at t=2 after
# H2=1), by inverting the two-point V tables and applying Brook's lemma.
def invert_two_point(kval: Fr, v_a: Fr, v_b: Fr) -> Fr:
    """Solve kval = x v_a + (1-x) v_b for x (weight on the first point)."""
    return (kval - v_b) / (v_a - v_b)


Vt0 = dict(zip(order, V0))
Vt1 = dict(zip(order, V1))
c10 = {d0: invert_two_point(k01[d0], Vt0[(d0, 0)], Vt0[(d0, 1)]) for d0 in (0, 1)}   # lambda_1(D1=0 | d0)
k12 = {d1: cond(J2, lambda e, hh, d=d1: e[ID1] == d and hh == (1, 1) and e[IY1] == 1,
                lambda e, hh, d=d1: e[ID1] == d and hh == (1, 1)) for d1 in (0, 1)}
c2 = {d1: invert_two_point(k12[d1], Vt1[(0, d1)], Vt1[(1, d1)]) for d1 in (0, 1)}      # lambda_2(D0=0 | d1)
# undo the known reply reweighting k_{0,1}(1|d0) to get lambda_1(D0=0 | d1)
c01 = {d1: (c2[d1] / k01[0]) / (c2[d1] / k01[0] + (1 - c2[d1]) / k01[1]) for d1 in (0, 1)}
cond_d1_given_d0 = {(d0, d1): (c10[d0] if d1 == 0 else 1 - c10[d0]) for d0 in (0, 1) for d1 in (0, 1)}
cond_d0_given_d1 = {(d0, d1): (c01[d1] if d0 == 0 else 1 - c01[d1]) for d0 in (0, 1) for d1 in (0, 1)}
# Brook: lambda(d0,d1)/lambda(0,0) = [l(d0|d1=0)/l(0|d1=0)] [l(d1|d0)/l(0|d0)]
raw = {(d0, d1): (cond_d0_given_d1[(d0, 0)] / cond_d0_given_d1[(0, 0)]) *
       (cond_d1_given_d0[(d0, d1)] / cond_d1_given_d0[(d0, 0)]) for d0 in (0, 1) for d1 in (0, 1)}
tot = sum(raw.values())
rec_l1 = [raw[dd] / tot for dd in order]
check("extension: lambda_1 recovered exactly from next two speakers' kernels (Brook)", rec_l1 == l1, str(rec_l1))

width = max(len(c[0]) for c in checks)
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")
