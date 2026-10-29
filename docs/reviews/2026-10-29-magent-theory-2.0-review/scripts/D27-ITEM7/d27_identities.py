r"""D27-ITEM7: independent checks of the algebraic claims of Item_7_formation_feedback.tex.

Written from the statements, not from item7_checks/checks.py. Single thread, float64 and sympy.
Each block prints a residual or a counterexample value. Numerical agreement supports an identity at the
tested points; it is not a proof (the proofs are reconstructed in the D27-ITEM7 report).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import sympy as sp

rng = np.random.default_rng(271029)


def simplex(k, floor=0.02):
    p = rng.random(k) + floor
    return p / p.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def lse(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


# ---------------------------------------------------------------------------------------------------------
# A. it7:pool. Compensation identity, pooled row at temperature T/n, Pythagoras for a moment projection.
#    Exponential family on {0,..,6} with t(x) = (x, x^2, x^3) (d = 3), sources in the family.
k, n, T = 7, 5, 0.8
Q = [simplex(k) for _ in range(n)]
qbar = sum(Q) / n
srcs = [simplex(k) for _ in range(6)]
res_comp = max(abs(sum(kl(q, nu) for q in Q) - sum(kl(q, qbar) for q in Q) - n * kl(qbar, nu)) for nu in srcs)
pri = simplex(6)
c_sum = np.array([sum(kl(q, nu) for q in Q) for nu in srcs])
c_pool = np.array([n * kl(qbar, nu) for nu in srcs])
row_sum = np.exp(np.log(pri) - c_sum / T - lse(np.log(pri) - c_sum / T))
row_pool = np.exp(np.log(pri) - c_pool / T - lse(np.log(pri) - c_pool / T))
print("A1 compensation residual (general laws, n=5, 6 sources):", res_comp)
print("A2 |shared row - Gibbs row of n KL(qbar||nu) at temperature T|:", np.abs(row_sum - row_pool).max())

xs = np.arange(k, dtype=float)
Tst = np.stack([xs, xs ** 2 / 7.0, xs ** 3 / 49.0], 1)


def ef(th):
    a = Tst @ th
    a -= a.max()
    p = np.exp(a)
    return p / p.sum()


def mproj(mu):
    th = np.zeros(3)
    for _ in range(100):
        p = ef(th)
        m = Tst.T @ p
        C = (Tst * p[:, None]).T @ Tst - np.outer(m, m)
        th = th + np.linalg.solve(C, mu - m)
    return th


Pst = ef(mproj(Tst.T @ qbar))
fam = [ef(rng.standard_normal(3)) for _ in range(6)]
res_py = max(abs(kl(qbar, p) - kl(qbar, Pst) - kl(Pst, p)) for p in fam)
print("A3 Pythagoras residual KL(qbar||p) - KL(qbar||P*) - KL(P*||p), d=3 family:", res_py,
      "; moment match", np.abs(Tst.T @ Pst - Tst.T @ qbar).max())
# shared row with family sources depends only on the pooled mean parameter: change members at fixed mean param.
eps = np.zeros(k); eps[1], eps[2], eps[4], eps[5] = 0.004, -0.007, 0.006, -0.003   # sum 0; check moments below
# project eps onto the null space of the moment map so that sum_i E t is unchanged
Mmat = np.vstack([np.ones(k), Tst.T])
eps = eps - Mmat.T @ np.linalg.lstsq(Mmat.T, eps, rcond=None)[0]
Q2 = [Q[0] + eps, Q[1] - 0.5 * eps, Q[2] - 0.5 * eps] + Q[3:]
assert min(q.min() for q in Q2) > 0
c1 = np.array([sum(kl(q, p) for q in Q) for p in fam]); c2 = np.array([sum(kl(q, p) for q in Q2) for p in fam])
print("A4 family sources: change of summed-cost differences at fixed pooled mean parameter (should be 0):",
      np.abs((c1 - c1[0]) - (c2 - c2[0])).max(), "; pools differ by", np.abs(sum(Q2) / n - qbar).max())

# ---------------------------------------------------------------------------------------------------------
# B. it7:capture: beta'_nu = (1 - beta'_A) beta_nu.
cost = rng.random(5) * 2; pr = simplex(5); a, DA = 0.37, 0.4
b = pr * np.exp(-cost / T); b /= b.sum()
w = np.concatenate([(1 - a) * pr * np.exp(-cost / T), [a * np.exp(-DA / T)]]); w /= w.sum()
print("B1 capture rows residual:", np.abs(w[:-1] - (1 - w[-1]) * b).max(),
      "; relative weights unchanged:", np.abs(w[:-1] / w[0] - b / b[0]).max())

# ---------------------------------------------------------------------------------------------------------
# C. it7:exact (i)-(iv): symbolic, 1-D, block of n = 3 members, sources {G, own parent A, two outside units}.
y1, y2, y3, mG, mV, mW, lam, R, o1, o2, o3, Tt = sp.symbols("y1 y2 y3 mG mV mW lam R o1 o2 o3 T", real=True)
pG, pA, pV, pW = sp.symbols("piG piA piV piW", positive=True)
Y = [y1, y2, y3]; O = [o1, o2, o3]; nb = 3
ybar = sum(Y) / nb
means = {"G": mG, "A": ybar, "V": mV, "W": mW}
pris = {"G": pG, "A": pA, "V": pV, "W": pW}
Csum = {s: sum(lam / 2 * (yi - m) ** 2 for yi in Y) for s, m in means.items()}     # summed KL costs
spread = sum(lam / 2 * (yi - ybar) ** 2 for yi in Y)
okC = all(sp.simplify(Csum[s] - (spread + nb * lam / 2 * (ybar - means[s]) ** 2)) == 0 for s in means)
Z = sum(pris[s] * sp.exp(-Csum[s] / Tt) for s in means)
beta = {s: pris[s] * sp.exp(-Csum[s] / Tt) / Z for s in means}
# derivative of the block free energy in y_i with the row at its minimizer and the parent law held as data
def dF(yi, oi):
    return sum(beta[s] * lam * (yi - means[s]) for s in means) + R * (yi - oi)
vel = [-dF(yi, oi) / lam for yi, oi in zip(Y, O)]
rho = R / lam
pooled = sp.simplify(sum(vel) / nb - (sum(beta[s] * (means[s] - ybar) for s in means) + rho * (sum(O) / nb - ybar)))
dev = sp.simplify((vel[0] - sum(vel) / nb) - (-(1 + rho) * (y1 - ybar) + rho * (o1 - sum(O) / nb)))
# rows depend on the block only through ybar: d(beta_s)/d(y1 - y2) at fixed ybar vanishes
tvar = sp.symbols("t", real=True)
subs_line = {y1: ybar + tvar, y2: ybar - tvar}
row_dep = sp.simplify(sp.diff(beta["V"].subs({y1: sp.Symbol("u") + tvar, y2: sp.Symbol("u") - tvar}), tvar))
print("C1 summed cost = spread + n lam/2 (ybar-m)^2 for every source:", okC)
print("C2 pooled-velocity identity (iii) residual:", pooled)
print("C3 deviation identity (iv) residual:", dev)
print("C4 d beta_V / d t along (y1,y2) -> (u+t, u-t) at fixed y3 (should be 0):", row_dep)

# ---------------------------------------------------------------------------------------------------------
# D. it7:type composition. Blocks I (n=2, mean 0) and J (n=4, mean 2), common precision 1, T = 1.
#    Primitive union rule vs an unweighted two-parent handoff vs the evidence-weighted parent rule
#    (exponent sum_U D_U/T_U with T_U = T/n_U, priors averaged with masses n_U).
yI = np.array([-0.3, 0.3]); yJ = np.array([1.5, 1.9, 2.2, 2.4])
assert abs(yI.mean()) < 1e-12 and abs(yJ.mean() - 2) < 1e-12
srcm = np.array([-1.0, 0.0, 1.0, 2.0, 3.5]); prs = np.array([0.1, 0.25, 0.2, 0.3, 0.15])
yall = np.concatenate([yI, yJ])
prim = np.array([np.sum(0.5 * (yall - m) ** 2) for m in srcm])                  # primitive summed costs
unw = np.array([0.5 * (0.0 - m) ** 2 + 0.5 * (2.0 - m) ** 2 for m in srcm])      # one cost per parent
evw = np.array([2 * 0.5 * (0.0 - m) ** 2 + 4 * 0.5 * (2.0 - m) ** 2 for m in srcm])  # D_U / T_U, T_U = 1/n_U
gib = lambda c: (lambda w_: w_ / w_.sum())(prs * np.exp(-(c - c.min())))
print("D1 union mean: primitive", yall.mean(), "; unweighted parent average", (0 + 2) / 2)
print("D2 |primitive row - unweighted-parent row|:", np.abs(gib(prim) - gib(unw)).max(),
      "; |primitive row - evidence-weighted parent row|:", np.abs(gib(prim) - gib(evw)).max())
# heterogeneous priors: primitive average over agents vs unweighted and mass-weighted parent averages
rows_agents = [simplex(5) for _ in range(6)]
p_prim = sum(rows_agents) / 6
p_I = sum(rows_agents[:2]) / 2; p_J = sum(rows_agents[2:]) / 4
print("D3 prior: |primitive - unweighted parent average|:", np.abs(p_prim - (p_I + p_J) / 2).max(),
      "; |primitive - mass-weighted|:", np.abs(p_prim - (2 * p_I + 4 * p_J) / 6).max())

# ---------------------------------------------------------------------------------------------------------
# E. it7:characterization. Bernoulli counterexample and the lognormal (t = log y) exact closure.
H = lambda p: p * (1 - p); lo = lambda p: np.log(p / (1 - p))
v1 = np.mean([H(p) * (1.0 - lo(p)) for p in (0.5, 0.5)]); v2 = np.mean([H(p) * (1.0 - lo(p)) for p in (0.2, 0.8)])
print("E1 Bernoulli pooled velocities at equal pools (c = 1):", v1, v2)
# Lognormal family p_theta(y) = exp(theta log y - sigma^2 theta^2/2) * h(y), t = log y, psi = sigma^2 theta^2/2.
sig2 = 0.7
theta = lambda mu: mu / sig2                       # mean parameter mu = E log y = sigma^2 theta
vmax = 0.0
for _ in range(200):
    m1, m2, c = rng.standard_normal(3) * 2
    # natural-coordinate Fisher flow theta_i' = c - theta_i (no records), so mu_i' = H (c - theta_i), H = sig2
    vbar = 0.5 * sum(sig2 * (c - theta(m)) for m in (m1, m2))
    vmax = max(vmax, abs(vbar - (sig2 * c - 0.5 * (m1 + m2))))
print("E2 lognormal (t = log y): |pooled velocity - (sigma^2 c - mubar)| max over 200 draws:", vmax)

# ---------------------------------------------------------------------------------------------------------
# F. it7:null constants and it7:reduced neglected-weight bounds (instance: lam=6, T=1, ps=0.05, pp=0.95/11).
lam_, ps, pp, rho_ = 6.0, 0.05, 0.95 / 11, 0.005
bnd = ps / (1 - ps) * np.exp(-3 * 1.795 ** 2)
yy = np.linspace(1.795, 3.205, 100001)
evt = (ps * np.exp(-3 * yy ** 2) / (ps * np.exp(-3 * yy ** 2) + 1 - ps) * yy).max()
print("F1 null-row bound at |y|=1.795:", bnd, "; max evaluator term on [1.795, 3.205]:", evt,
      "; record term at the box edge:", rho_ * 0.005)
print("F2 coherence and wall radii:", np.sqrt(2 * 0.05 / lam_), np.sqrt(2 * 0.5 / lam_),
      "; KL at separation 0.14:", lam_ / 2 * 0.14 ** 2)
print("F3 evaluator weight bound pi_s/(2 pi_p) e^{-9*1.7^2}:", ps / (2 * pp) * np.exp(-9 * 1.7 ** 2),
      "; cross-side weight 1.5 e^{-9*3.4^2} / e^{-100}:", 1.5 * np.exp(-9 * 3.4 ** 2) / np.exp(-100))
print("F4 deviation limits rho/(1+rho)*0.3 and *1.4:", rho_ / (1 + rho_) * 0.3, rho_ / (1 + rho_) * 1.4)

# ---------------------------------------------------------------------------------------------------------
# G. it7:structure-fe, it7:radius. Gamma_n from structure-gauss (symbolic), zeros and Gamma_n(0).
d, nn, Ts, ls, ppS = sp.symbols("d n T lambda p_p", positive=True)
Fsep = 2 * (-Ts * sp.log((nn - 1) * ppS + nn * ppS * sp.exp(-nn * ls * d ** 2 / (2 * Ts))))
Fun = ls / 2 * (nn * nn / (2 * nn)) * d ** 2 - Ts * sp.log((2 * nn - 1) * ppS)
Gam_text = sp.log((2 * nn - 1) / ppS) - 2 * sp.log(nn - 1 + nn * sp.exp(-nn * ls * d ** 2 / (2 * Ts))) \
    - nn * ls / (4 * Ts) * d ** 2
print("G1 (F(sep)-F(union))/T - Gamma_n (symbolic, should be 0):", sp.simplify(sp.expand_log((Fsep - Fun) / Ts - Gam_text, force=True)))
from scipy.optimize import brentq
def Gam(dd, n_):
    return np.log((2 * n_ - 1) / pp) - 2 * np.log(n_ - 1 + n_ * np.exp(-n_ * 6 * dd * dd / 2)) - n_ * 6 / 4 * dd * dd
for n_ in (2, 3, 4, 6):
    print(f"G2 n={n_}: d* = {brentq(Gam, 1e-3, 3.0, args=(n_,)):.5f}, Gamma_n(0) = {Gam(0.0, n_):.4f}, "
          f"(2n-1) pp = {(2 * n_ - 1) * pp:.4f}")
# description check: the second term of structure-split, -T log(...), versus the free energy at temperature T/n
# of a single receiver holding the pool, -(T/n) log(...): they differ by the factor n.
cst = rng.random(4) * 0.5; prr = simplex(4); piA = 0.2; nU = 3; T1 = 1.0
term2 = -T1 * np.log(piA + np.sum(prr * np.exp(-nU * cst / T1)))
fe_Tn = -(T1 / nU) * np.log(piA + np.sum(prr * np.exp(-cst / (T1 / nU))))
print("G3 second term of structure-split / free energy at T/n of the pooled receiver:", term2 / fe_Tn)

# ---------------------------------------------------------------------------------------------------------
# H. it7:mutual bounds on fresh random draws with a non-identity Markov channel for the outsider.
sm, so = np.inf, np.inf
for _ in range(3000):
    kk = 6; T_ = 0.2 + 1.5 * rng.random()
    qi, qj, qv = simplex(kk), simplex(kk), simplex(kk); oth = [simplex(kk) for _ in range(3)]
    pi_i, pi_j = simplex(4), simplex(4)
    Di = np.array([kl(qi, qj)] + [kl(qi, x) for x in oth]); Dj = np.array([kl(qj, qi)] + [kl(qj, x) for x in oth])
    Zi = np.sum(pi_i * np.exp(-Di / T_)); Zj = np.sum(pi_j * np.exp(-Dj / T_))
    bij = pi_i[0] * np.exp(-Di[0] / T_) / Zi; bji = pi_j[0] * np.exp(-Dj[0] / T_) / Zj
    qb = (qi + qj) / 2; piA_ = (pi_i[0] + pi_j[0]) / 2
    Zij = piA_ * np.exp(-(kl(qi, qb) + kl(qj, qb)) / T_) + np.sum((pi_i[1:] + pi_j[1:]) / 2 * np.exp(-(Di[1:] + Dj[1:]) / T_))
    gain = -T_ * np.log(Zi) - T_ * np.log(Zj) + T_ * np.log(Zij)
    sm = min(sm, gain - (T_ * np.log(bij * bji * piA_ / (pi_i[0] * pi_j[0])) + max(Di[0], Dj[0])))
    K = rng.random((kk, kk)) + 0.1; K /= K.sum(1, keepdims=True)              # Markov channel, rows sum to 1
    pi_v = simplex(5)
    Dv = np.array([kl(qv, qi @ K), kl(qv, qj @ K)] + [kl(qv, x) for x in oth])
    Zv = np.sum(pi_v * np.exp(-Dv / T_)); bv = np.sum(pi_v[:2] * np.exp(-Dv[:2] / T_)) / Zv
    Zv2 = (pi_v[0] + pi_v[1]) * np.exp(-kl(qv, qb @ K) / T_) + np.sum(pi_v[2:] * np.exp(-Dv[2:] / T_))
    so = min(so, -T_ * np.log(1 - bv * (1 - np.exp(-abs(Dv[0] - Dv[1]) / (2 * T_)))) - T_ * np.log(Zv / Zv2))
print("H1 min slack of mutual-bound over 3000 draws:", sm, "; of outsider-bound with Markov channel:", so)

# ---------------------------------------------------------------------------------------------------------
# I. Derived selector, consequence of the double use: two isolated mutually attending agents always merge.
#    N = 2 agents far from the evaluator (mean 0), separation dd; Gaussian location, lam = 6, T = 1,
#    pi_s = 0.05, pi_p = 0.95. Structure free energy gain F(singletons) - F(pair) as a function of dd.
def gain_pair(dd, c=20.0, lam1=6.0, ps1=0.05):
    pp1 = 1 - ps1
    y = np.array([c - dd / 2, c + dd / 2])
    Fs = 0.0
    for i in range(2):
        j = 1 - i
        Fs += -np.log(ps1 * np.exp(-lam1 / 2 * y[i] ** 2) + pp1 * np.exp(-lam1 / 2 * (y[i] - y[j]) ** 2))
    yb = y.mean(); W = np.sum((y - yb) ** 2)
    Fp = lam1 / 2 * W - np.log(1 * pp1 + ps1 * np.exp(-2 * lam1 / 2 * yb ** 2))
    return Fs - Fp
print("I1 structure-free-energy gain of merging two isolated agents at separation 0.5, 2, 5, 10:",
      [round(gain_pair(x), 3) for x in (0.5, 2.0, 5.0, 10.0)])
