r"""Checks for supplements/s01_probability.tex and appendices/record_process.tex (lens D01-FOUND).

Finite enumerations and one-dimensional quadrature.  Agreement at the tested instances supports
the identities; it does not prove them.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln

rng = np.random.default_rng(1029)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float).ravel(); q = np.asarray(q, float).ravel()
    if np.any((q == 0) & (p > 0)):
        return math.inf
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def simplex(shape: tuple[int, ...]) -> np.ndarray:
    x = rng.random(shape) + 0.05
    return x / x.sum()


print("== s01 eq:s-chain-rule")
mu = simplex((3, 4)); nu = simplex((3, 4))       # rows x, columns y
muY, nuY = mu.sum(0), nu.sum(0)
rhs = kl(muY, nuY) + sum(muY[y] * kl(mu[:, y] / muY[y], nu[:, y] / nuY[y]) for y in range(4))
print("  residual:", abs(kl(mu, nu) - rhs))
nu2 = nu.copy(); nu2[1, 2] = 0.0; nu2 /= nu2.sum(); nu2Y = nu2.sum(0)
rhs2 = kl(muY, nu2Y) + sum(muY[y] * kl(mu[:, y] / muY[y], nu2[:, y] / nu2Y[y]) for y in range(4))
print("  infinite case: lhs", kl(mu, nu2), " rhs", rhs2)
# negative part bound used in the proof: sup_x>0 of (x log x)^- = 1/e
xs = np.linspace(1e-9, 1, 100001)
print("  max (x log x)^- on (0,1]:", float(np.max(-xs * np.log(xs))), " 1/e =", 1 / math.e)

print("== s01 eq:s-different-evaluators")
for trial in range(2):
    s = simplex((3,)); t = simplex((3,))
    K = rng.random((3, 4)); K /= K.sum(1, keepdims=True)
    Lk = rng.random((3, 4)); Lk /= Lk.sum(1, keepdims=True)
    if trial == 1:   # presentation change Theta (a permutation): t -> Theta# t, L -> L(Theta^-1 m, .)
        Th = [2, 0, 1]
        t = np.array([t[Th.index(m)] for m in range(3)])
        Lk = np.array([Lk[Th.index(m)] for m in range(3)])
    mu_j = s[:, None] * K; nu_j = t[:, None] * Lk
    lhs = kl(s, t) + sum(s[m] * kl(K[m], Lk[m]) for m in range(3))
    sK, tL = mu_j.sum(0), nu_j.sum(0)
    rhs = kl(sK, tL) + sum(sK[o] * kl(mu_j[:, o] / sK[o], nu_j[:, o] / tL[o]) for o in range(4))
    print(f"  trial {trial}: lhs={lhs:.15f} rhs={rhs:.15f} residual={abs(lhs - rhs):.2e}")
s = simplex((3,)); K = rng.random((3, 4)); K /= K.sum(1, keepdims=True); Lk = rng.random((3, 4)); Lk /= Lk.sum(1, keepdims=True)
print("  equal model laws keep the evaluator term:", sum(s[m] * kl(K[m], Lk[m]) for m in range(3)))

print("== s01 four-point reconstruction")
q = np.array([[0.5, 0.0], [0.0, 0.5]])          # q[h, v]
p = np.full((2, 2), 0.25)
qH = q.sum(1)
qhat = qH[:, None] * (p / p.sum(1, keepdims=True))
print("  KL(q||qhat) =", kl(q, qhat), " log 2 =", math.log(2))
print("  conditional KL at H=0,1:", [kl(q[h] / qH[h], qhat[h] / qhat[h].sum()) for h in range(2)])
p_dep = np.array([[0.3, 0.0], [0.0, 0.7]])       # prior with V = H
qhat2 = qH[:, None] * (p_dep / p_dep.sum(1, keepdims=True))
print("  prior with V=H reconstructs q exactly:", float(np.max(np.abs(qhat2 - q))))
# Jeffrey's rule is the I-projection: minimize KL(r||p) over r with r_H = q_H
p_gen = simplex((2, 2))
best = math.inf; rb = None
for a in np.linspace(1e-4, 1 - 1e-4, 2001):
    for c in np.linspace(1e-4, 1 - 1e-4, 2001)[::4]:
        r = np.array([[qH[0] * (1 - a), qH[0] * a], [qH[1] * (1 - c), qH[1] * c]])
        v = kl(r, p_gen)
        if v < best:
            best, rb = v, r
jeff = qH[:, None] * (p_gen / p_gen.sum(1, keepdims=True))
print("  I-projection (grid) vs Jeffrey:", float(np.max(np.abs(rb - jeff))), " values", best, kl(jeff, p_gen))

print("== s01 shared source and two routes")
for bmark in [0, 1]:
    pO_shared = np.zeros(2); pO_indep = np.zeros(2)
    for x in range(2):
        pO_shared[(x ^ (x ^ bmark))] += 0.5
        for xp in range(2):
            pO_indep[(x ^ (xp ^ bmark))] += 0.25
    print(f"  mark b={bmark}: shared-source law of O {pO_shared}, independent redraw {pO_indep}")

print("== s01 predictive sufficiency")
# X=(A,B) uniform, U=A.  I(Y;X|U) for Y=A and Y=B.
def cmi_target(target: str) -> float:
    tot = 0.0
    for a_ in range(2):
        # given U=a, X=(a,B) with B uniform
        pXY = {}
        for b_ in range(2):
            y = a_ if target == "A" else b_
            pXY[(b_, y)] = pXY.get((b_, y), 0.0) + 0.5
        pX = {b_: 0.5 for b_ in range(2)}
        pY: dict[int, float] = {}
        for (b_, y), v in pXY.items():
            pY[y] = pY.get(y, 0.0) + v
        tot += 0.5 * sum(v * math.log(v / (pX[b_] * pY[y])) for (b_, y), v in pXY.items() if v > 0)
    return tot
print("  I(A;X|U) =", cmi_target("A"), " I(B;X|U) =", cmi_target("B"))

print("== record_process prop:registered-history")
nS, nO = 6, 2
rho = [x % 2 for x in range(nS)]                  # readout: parity
Rk = rng.random((nS, nO)); Rk /= Rk.sum(1, keepdims=True)
Aup = rng.integers(0, nS // 2, size=(nS, nO))
Amap = [[2 * int(Aup[x, o]) + o for o in range(nO)] for x in range(nS)]  # rho(A(x,o)) = o
mu0 = simplex((nS,))
n = 4
mass = 0.0; agree_fail = 0.0
marg_prev = None
for hist in itertools.product(range(nO), repeat=n):
    for x0 in range(nS):
        pr = mu0[x0]; x = x0
        for o in hist:
            pr *= Rk[x, o]; x = Amap[x][o]
            if rho[x] != o:
                agree_fail += pr
        mass += pr
print("  total mass:", mass, " mass on histories with a record/readout mismatch:", agree_fail)
# consistency: marginalizing the last pair of P_n gives P_{n-1}
def hist_law(nsteps: int) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for hist in itertools.product(range(nO), repeat=nsteps):
        tot = 0.0
        for x0 in range(nS):
            pr = mu0[x0]; x = x0
            for o in hist:
                pr *= Rk[x, o]; x = Amap[x][o]
            tot += pr
        out[hist] = tot
    return out
P4, P3 = hist_law(4), hist_law(3)
cons = max(abs(sum(P4[h + (o,)] for o in range(nO)) - P3[h]) for h in P3)
print("  record-marginal consistency residual:", cons)

print("== record_process prop:bayesian-history")
nM = 3
kd = rng.random((nM, nO)); kd /= kd.sum(1, keepdims=True)    # k(o|m)
s0 = simplex((nM,))
worst = 0.0
for hist in itertools.product(range(nO), repeat=4):
    s = s0.copy(); pseq = 1.0
    for o in hist:
        pred = float(s @ kd[:, o]); pseq *= pred
        s = s * kd[:, o] / pred
    direct = float(s0 @ np.prod(kd[:, list(hist)], axis=1))
    worst = max(worst, abs(pseq - direct))
print("  sequential product vs fixed-latent integral, max residual:", worst)

print("== record_process Dirichlet (Beta) predictive step")
a1, a0 = 1.3, 0.7; Atot = a1 + a0
obs = [1, 0, 0, 1, 1, 1, 0, 1]
c1 = 0; worst_pred = 0.0; worst_step = 0.0; ok_tv = True
mbar_prev = a1 / Atot
for nn, o in enumerate(obs, start=1):
    c1 += o
    mbar = (a1 + c1) / (Atot + nn)
    # exact Bayes predictive by quadrature of m * Beta(m; a1+c1, a0+nn-c1)
    lb = betaln(a1 + c1, a0 + nn - c1)
    val = quad(lambda m: m * math.exp((a1 + c1 - 1) * math.log(m) + (a0 + nn - c1 - 1) * math.log(1 - m) - lb), 0, 1)[0]
    worst_pred = max(worst_pred, abs(val - mbar))
    step = mbar - mbar_prev
    worst_step = max(worst_step, abs(step - ((1.0 if o == 1 else 0.0) - mbar_prev) / (Atot + nn)))
    tv = abs(step)   # for a binary alphabet TV = |difference in P(o=1)|
    ok_tv = ok_tv and (tv <= 1.0 / (Atot + nn) + 1e-15)
    mbar_prev = mbar
print("  predictive (alpha+c)/(A+n) vs quadrature, max residual:", worst_pred)
print("  step identity residual:", worst_step, " TV step <= 1/(A+n) at every step:", ok_tv)

print("== record_process prop:variational-history")
alpha = 0.6
def bayes(s: np.ndarray, o: int) -> np.ndarray:
    w = s * kd[:, o]; return w / w.sum()
def damped(s: np.ndarray, o: int) -> np.ndarray:
    return (1 - alpha) * s + alpha * bayes(s, o)
nsteps = 4
P = {}; Pt = {}; chain = 0.0; bound = 0.0
for hist in itertools.product(range(nO), repeat=nsteps):
    s = s0.copy(); st = s0.copy(); pr = 1.0; prt = 1.0
    for o in hist:
        pr_o = float(s @ kd[:, o]); prt_o = float(st @ kd[:, o])
        pr *= pr_o; prt *= prt_o
        s = bayes(s, o); st = damped(st, o)
    P[hist] = pr; Pt[hist] = prt
lhs = sum(P[h] * math.log(P[h] / Pt[h]) for h in P)
# sum_t E_P KL(s_{t-1}k || s~_{t-1}k) and the posterior bound
for t in range(nsteps):
    for prefix in itertools.product(range(nO), repeat=t):
        s = s0.copy(); st = s0.copy(); pr = 1.0
        for o in prefix:
            pr *= float(s @ kd[:, o]); s = bayes(s, o); st = damped(st, o)
        chain += pr * kl(s @ kd, st @ kd)
        bound += pr * kl(s, st)
print(f"  KL(P_n||P~_n) = {lhs:.15f}, chain sum = {chain:.15f}, residual = {abs(lhs - chain):.2e}, bound = {bound:.6f}")
print("  binary registration budget n log 2 for n=4:", 4 * math.log(2))
