r"""GAP08-S1 script 5: appendices/conditional_memory.tex:40-86.

(a) cmem:free-energy-split on finite alphabets at 50 digits: F defined as in eq:agent-free-energy,
    F = KL(Q || P(.|o)) - log P(o), compared with the split and with the log density ratio
    E_Q log(dQ / d(r G l(o|.))). Candidates: full support, with zeros, and not absolutely continuous.
    Also a presentation with Z_m(o) = 0 (violating the first line of cmem:evidence).
(b) cmem:gibbs: f_a in KL form and separated form; Gibbs minimizer; f_a = -log Z_a at the conditionals.
(c) cmem:product-label: coordinate updates checked against brute-force perturbation; the normalizer
    of the q_X update is bounded by sum_a s_a Z_a(o) (weighted AM-GM) and can vanish.
(d) the D06-PRED-04 example: Z(0) = e and divergence of the integral of l log l (closed form and quadrature).
(e) cmem:joint-covariance and its Schur complement with a singular regression F (exact).
"""
import random
import mpmath as mp
import sympy as sp
from sympy import Rational as Q

mp.mp.dps = 50
random.seed(17)


def rprob(n, zeros=0):
    raw = [random.randint(1, 7) for _ in range(n)]
    for k in random.sample(range(n), zeros):
        raw[k] = 0
    s = sum(raw)
    return [mp.mpf(r) / s for r in raw]


def xlogy_ratio(p, q):
    """p log(p/q) with conventions 0 log 0/q = 0 and p log p/0 = +inf."""
    if p == 0:
        return mp.mpf(0)
    if q == 0:
        return mp.inf
    return p * mp.log(p / q)


nM, nU, nV, nO = 3, 4, 2, 3
r = rprob(nM)
mu = [rprob(nU) for _ in range(nM)]
K = [[rprob(nV) for _ in range(nU)] for _ in range(nM)]
lik = [rprob(nO) for _ in range(nU)]
o = 1
lik[2][o] = mp.mpf(0)  # cue u = 2 cannot produce record o: a zeta^o-null region
Zm = [sum(mu[m][u] * lik[u][o] for u in range(nU)) for m in range(nM)]
Z = sum(r[m] * Zm[m] for m in range(nM))
w = [r[m] * Zm[m] / Z for m in range(nM)]
Pi = [[[lik[u][o] * mu[m][u] * K[m][u][v] / Zm[m] for v in range(nV)] for u in range(nU)] for m in range(nM)]
zeta = [[[w[m] * Pi[m][u][v] for v in range(nV)] for u in range(nU)] for m in range(nM)]
unnorm = [[[r[m] * mu[m][u] * K[m][u][v] * lik[u][o] for v in range(nV)] for u in range(nU)] for m in range(nM)]


def check(Qjoint, label):
    s = [sum(Qjoint[m][u][v] for u in range(nU) for v in range(nV)) for m in range(nM)]
    F_def = sum(xlogy_ratio(Qjoint[m][u][v], zeta[m][u][v])
                for m in range(nM) for u in range(nU) for v in range(nV)) - mp.log(Z)
    F_ratio = sum(xlogy_ratio(Qjoint[m][u][v], unnorm[m][u][v])
                  for m in range(nM) for u in range(nU) for v in range(nV))
    kl_s = sum(xlogy_ratio(s[m], w[m]) for m in range(nM))
    cond = mp.mpf(0)
    for m in range(nM):
        if s[m] == 0:
            continue
        qm = [[Qjoint[m][u][v] / s[m] for v in range(nV)] for u in range(nU)]
        cond += s[m] * sum(xlogy_ratio(qm[u][v], Pi[m][u][v]) for u in range(nU) for v in range(nV))
    split = kl_s + cond - mp.log(Z)
    if mp.isinf(F_def):
        print(f"{label}: F(def) = {F_def}, F(log ratio) = {F_ratio}, split = {split}")
    else:
        print(f"{label}: F(def) = {mp.nstr(F_def, 12)}; |F(def) - split| = {mp.nstr(abs(F_def - split), 3)}; "
              f"|F(def) - F(log ratio)| = {mp.nstr(abs(F_def - F_ratio), 3)}; F + log Z = {mp.nstr(F_def + mp.log(Z), 8)}")


print("== (a) cmem:free-energy-split ==")
print("Z_m(o) =", [mp.nstr(z, 8) for z in Zm], " Z(o) =", mp.nstr(Z, 8))
for trial in range(3):
    raw = [[[mp.mpf(random.randint(1, 9)) for _ in range(nV)] for _ in range(nU)] for _ in range(nM)]
    for m in range(nM):
        for v in range(nV):
            raw[m][2][v] = mp.mpf(0)  # keep Q << zeta^o
    tot = sum(raw[m][u][v] for m in range(nM) for u in range(nU) for v in range(nV))
    check([[[raw[m][u][v] / tot for v in range(nV)] for u in range(nU)] for m in range(nM)], f"correlated Q #{trial}")
check(zeta, "posterior")
raw = [[[mp.mpf(1) for _ in range(nV)] for _ in range(nU)] for _ in range(nM)]
tot = sum(raw[m][u][v] for m in range(nM) for u in range(nU) for v in range(nV))
check([[[raw[m][u][v] / tot for v in range(nV)] for u in range(nU)] for m in range(nM)], "uniform Q (mass on the null cue)")

# presentation with Z_m(o) = 0 while r(m) > 0 (violates the r-a.e. positivity in cmem:evidence)
mu_b = [list(x) for x in mu]
mu_b[0] = [mp.mpf(0), mp.mpf(0), mp.mpf(1), mp.mpf(0)]  # all cue mass on the null cue
Zm_b = [sum(mu_b[m][u] * lik[u][o] for u in range(nU)) for m in range(nM)]
Z_b = sum(r[m] * Zm_b[m] for m in range(nM))
w_b = [r[m] * Zm_b[m] / Z_b for m in range(nM)]
print("presentation 0 with Z_0(o) = 0:", mp.nstr(Zm_b[0], 3), "; posterior weights", [mp.nstr(x, 6) for x in w_b],
      "(well defined, w_0 = 0; the r-a.e. positivity in the first line of cmem:evidence is not needed)")

print("\n== (b) cmem:gibbs ==")
G = [[[mu[m][u] * K[m][u][v] for v in range(nV)] for u in range(nU)] for m in range(nM)]
fa_sep, fa_kl = [], []
qs = []
for m in range(nM):
    raw = [[mp.mpf(random.randint(1, 9)) for _ in range(nV)] for _ in range(nU)]
    for v in range(nV):
        raw[2][v] = mp.mpf(0)
    tot = sum(raw[u][v] for u in range(nU) for v in range(nV))
    q = [[raw[u][v] / tot for v in range(nV)] for u in range(nU)]
    qs.append(q)
    kl_G = sum(xlogy_ratio(q[u][v], G[m][u][v]) for u in range(nU) for v in range(nV))
    El = sum(q[u][v] * mp.log(lik[u][o]) for u in range(nU) for v in range(nV) if q[u][v] > 0)
    fa_sep.append(kl_G - El)
    fa_kl.append(sum(xlogy_ratio(q[u][v], Pi[m][u][v]) for u in range(nU) for v in range(nV)) - mp.log(Zm[m]))
print("max |f_a(separated) - f_a(KL form)| =", mp.nstr(max(abs(x - y) for x, y in zip(fa_sep, fa_kl)), 3))
sg = [r[m] * mp.e ** (-fa_kl[m]) for m in range(nM)]
sg = [x / sum(sg) for x in sg]


def F_of(s_vec):
    return sum(xlogy_ratio(s_vec[m], r[m]) for m in range(nM)) + sum(s_vec[m] * fa_kl[m] for m in range(nM))


best = F_of(sg)
worse = min(F_of([x / sum(y) for x in y]) - best
            for y in [[mp.mpf(random.random()) for _ in range(nM)] for _ in range(300)])
print("Gibbs s minimizes KL(s||r) + sum s_a f_a: smallest excess over 300 random s =", mp.nstr(worse, 5))
f_exact = [sum(xlogy_ratio(Pi[m][u][v], Pi[m][u][v]) for u in range(nU) for v in range(nV)) - mp.log(Zm[m]) for m in range(nM)]
s_exact = [r[m] * mp.e ** (-f_exact[m]) for m in range(nM)]
s_exact = [x / sum(s_exact) for x in s_exact]
print("at exact conditionals: max |f_a + log Z_a| =", mp.nstr(max(abs(f_exact[m] + mp.log(Zm[m])) for m in range(nM)), 3),
      "; max |s - w| =", mp.nstr(max(abs(s_exact[m] - w[m]) for m in range(nM)), 3))

print("\n== (c) cmem:product-label ==")
# structured family s_a q_X(u) K_a(u, v); counting measure on cues
lo = [lik[u][o] for u in range(nU)]


def F_struct(s_vec, qX):
    val = sum(xlogy_ratio(s_vec[m], r[m]) for m in range(nM))
    for m in range(nM):
        if s_vec[m] > 0:
            val += s_vec[m] * sum(xlogy_ratio(qX[u], mu[m][u]) for u in range(nU))
    for u in range(nU):
        if qX[u] > 0:
            val += -qX[u] * mp.log(lo[u]) if lo[u] > 0 else mp.inf
    return val


qX = [mp.mpf(1) / 4, mp.mpf(1) / 4, mp.mpf(0), mp.mpf(1) / 2]
s_upd = [r[m] * mp.e ** (-sum(xlogy_ratio(qX[u], mu[m][u]) for u in range(nU))) for m in range(nM)]
s_upd = [x / sum(s_upd) for x in s_upd]
exc_s = min(F_struct([x / sum(y) for x in y], qX) - F_struct(s_upd, qX)
            for y in [[mp.mpf(random.random()) for _ in range(nM)] for _ in range(300)])
unq = [lo[u] * mp.e ** (sum(s_upd[m] * mp.log(mu[m][u]) for m in range(nM))) for u in range(nU)]
Nq = sum(unq)
q_upd = [x / Nq for x in unq]
cands = []
for _ in range(300):
    y = [mp.mpf(random.random()) for _ in range(nU)]
    y[2] = mp.mpf(0)  # candidates with finite F must avoid the cue that cannot produce o
    cands.append([x / sum(y) for x in y])
exc_q = min(F_struct(s_upd, c) - F_struct(s_upd, q_upd) for c in cands)
print("label update minimizes over s at fixed q_X: smallest excess =", mp.nstr(exc_s, 5))
print("cue update minimizes over q_X at fixed s: smallest excess =", mp.nstr(exc_q, 5))
print("normalizer of the cue update", mp.nstr(Nq, 10), "<= sum_a s_a Z_a(o) =",
      mp.nstr(sum(s_upd[m] * Zm[m] for m in range(nM)), 10))
# disjoint supports: mu_1 = U{0,1}, mu_2 = U{2,3} (as densities on 4 points), s = (1/2, 1/2)
m1 = [mp.mpf(1) / 2, mp.mpf(1) / 2, 0, 0]
m2 = [0, 0, mp.mpf(1) / 2, mp.mpf(1) / 2]
prodd = [(m1[u] ** mp.mpf(0.5)) * (m2[u] ** mp.mpf(0.5)) for u in range(4)]
print("disjoint cue supports, s = (1/2, 1/2): product of weighted densities =", [mp.nstr(x, 3) for x in prodd],
      "so the normalizer is 0 and every q_X has KL(q_X||mu_1) or KL(q_X||mu_2) infinite")

print("\n== (d) the D06-PRED-04 example ==")
mp.mp.dps = 30
# U ~ Uniform(0,1); l(0|u) = 1/(u log(u)^2) on (0, 1/e], = e on (1/e, 1)
Z0 = mp.quad(lambda s: 1 / s ** 2, [1, mp.inf]) + mp.e * (1 - 1 / mp.e)  # s = -log u on the first piece
print("Z(0) =", mp.nstr(Z0, 20), " e =", mp.nstr(mp.e, 20))
for S in [10, 10 ** 3, 10 ** 6, 10 ** 12]:
    closed = mp.log(S) - 2 + 2 * (mp.log(S) + 1) / S
    quad = mp.quad(lambda s: (s - 2 * mp.log(s)) / s ** 2, [1, 10, 100, 1000, S] if S > 1000 else [1, S])
    print(f"S = {S:>14}: int over (e^-S, 1/e) of l log l du = {mp.nstr(quad, 8)} (closed form {mp.nstr(closed, 8)})")
print("the integral of l log l diverges like log S, so KL(posterior || prior) = +inf and the separated form "
      "is +inf - (+inf) at the posterior, while KL(posterior || posterior) - log Z(0) = -1")

print("\n== (e) cmem:joint-covariance with a singular regression ==")
Lm = sp.Matrix([[2, Q(1, 2)], [Q(1, 2), 1]])
Qm = sp.Matrix([[1, Q(1, 3)], [Q(1, 3), Q(1, 2)]])
Fm = sp.Matrix([[1, 2], [2, 4]])  # rank one
Cj = sp.BlockMatrix([[Lm, Lm * Fm.T], [Fm * Lm, Fm * Lm * Fm.T + Qm]]).as_explicit()
schur = (Fm * Lm * Fm.T + Qm) - Fm * Lm * Lm.inv() * Lm * Fm.T
print("rank F =", Fm.rank(), "; Schur complement - Q =", (schur - Qm).tolist(),
      "; joint covariance positive definite (Sylvester minors):", Cj.is_positive_definite,
      [Cj[:k, :k].det() for k in range(1, 5)])
# covariance from the model: X = nu + L^{1/2} e1, Y = omega + F (X - nu) + Q^{1/2} e2
e1 = sp.Matrix(sp.symbols('e1_1 e1_2'))
e2 = sp.Matrix(sp.symbols('e2_1 e2_2'))
Lh = Lm.cholesky()
Qh = Qm.cholesky()
Xc = Lh * e1
Yc = Fm * Xc + Qh * e2
vecs = list(Xc) + list(Yc)
cov = sp.zeros(4, 4)
for i in range(4):
    for j in range(4):
        prodij = sp.expand(vecs[i] * vecs[j])
        val = 0
        for sym in list(e1) + list(e2):
            val += prodij.coeff(sym, 2)
        cov[i, j] = sp.simplify(val)  # E[e e^T] = I picks the squared terms
print("model covariance minus cmem:joint-covariance:", sp.simplify(cov - Cj).tolist())
