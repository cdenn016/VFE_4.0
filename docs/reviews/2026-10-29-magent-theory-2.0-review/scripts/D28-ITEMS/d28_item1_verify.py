r"""D28-ITEMS independent checks of Item 1 (communication links) claims.

Single-threaded, tiny sizes. Each block prints a residual or a counterexample value.
Numerical agreement supports an identity at the tested instance; it is not a proof.

Blocks
  A  it1:loss on a finite alphabet with a non-injective code (general-law identity).
  B  it1:noisy on a finite alphabet with a random Markov kernel (general-law bound).
  C  it1:stack-compatibility: rank-deficient compatible stacks and the stated counterexample.
  D  it1:complementary(ii) with a NONTRIVIAL detectable space (reciprocal pair, rank-2 codes, K=3):
     B omega_inf -> Q B omega* with Q = P_cap, and it1:learning(iii) covariance formula.
  E  Item 1 line 476 ("everything invisible is repaired ... complete repair costs complete trust"):
     pair with complementary row spaces and a DISSONANT truth; the lambda -> infinity limit is the flat
     representative, not the truth.
  F  Item 1 it1:owned ("in the per-agent picture ... ownership leaves the estimates unchanged"):
     exact posterior means vs receiver-factorized reverse-KL means vs per-agent LAW-LEVEL fixed point
     (Item 2 convolution of the loop factor against peer dictionary laws) on the complete exchange.
  G  it1:bandwidth: strong-law limit of normalized posterior log-odds for Gaussian message laws.
"""
import numpy as np
from scipy.linalg import null_space

rng = np.random.default_rng(2910)


def kl_disc(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))


# ---------------- A: noiseless loss identity on a finite alphabet ----------------
n_y, n_h = 7, 3
T = rng.integers(0, n_h, size=n_y)            # measurable non-injective code y -> T(y)
T[:n_h] = np.arange(n_h)                      # make it onto
q = rng.dirichlet(np.ones(n_y)); p = rng.dirichlet(np.ones(n_y))
Tq = np.array([q[T == h].sum() for h in range(n_h)])
Tp = np.array([p[T == h].sum() for h in range(n_h)])
qhat = np.array([p[y] / Tp[T[y]] * Tq[T[y]] for y in range(n_y)])   # int p(.|h) Tq(dh)
lhs = kl_disc(q, qhat); rhs = kl_disc(q, p) - kl_disc(Tq, Tp)
print("A  it1:loss finite alphabet: KL(q||qhat) =", lhs, " KL(q||p)-KL(Tq||Tp) =", rhs, " residual", lhs - rhs)

# ---------------- B: noisy-code bound on a finite alphabet ----------------
Kmat = rng.dirichlet(np.ones(4), size=n_y)     # kernel y -> h, rows sum to one
qK, pK = q @ Kmat, p @ Kmat
post_p = (p[:, None] * Kmat) / pK[None, :]     # p^h(y) columns indexed by h
qhat2 = post_p @ qK
gap = kl_disc(q, p) - kl_disc(qK, pK) - kl_disc(q, qhat2)
print("B  it1:noisy finite alphabet: slack KL(q||p)-KL(qK||pK)-KL(q||qhat) =", gap, "(must be >= 0)")
worst = np.inf
for _ in range(2000):
    qq = rng.dirichlet(0.3 * np.ones(n_y)); pp = rng.dirichlet(0.3 * np.ones(n_y))
    KK = rng.dirichlet(0.5 * np.ones(4), size=n_y)
    a, b = qq @ KK, pp @ KK
    qh = ((pp[:, None] * KK) / b[None, :]) @ a
    worst = min(worst, kl_disc(qq, pp) - kl_disc(a, b) - kl_disc(qq, qh))
print("   minimum slack over 2000 random instances =", worst)

# ---------------- C: stacked readings ----------------
K = 4
Tst = rng.standard_normal((3, K)); Tst[2] = Tst[0] + 2 * Tst[1]      # rank 2, one row relation
A_true = rng.standard_normal((K, K)) + 3 * np.eye(K)
Sst = Tst @ A_true                                                    # compatible: same left kernel
same = np.allclose(null_space(Tst.T).ravel() / np.linalg.norm(null_space(Tst.T)),
                   null_space(Sst.T).ravel() / np.linalg.norm(null_space(Sst.T))) or \
       np.allclose(null_space(Tst.T).ravel() / np.linalg.norm(null_space(Tst.T)),
                   -null_space(Sst.T).ravel() / np.linalg.norm(null_space(Sst.T)))
# construct theta from the proof: extend row bases
rowsT = Tst[:2]; rowsS = Sst[:2]
compT = null_space(rowsT).T; compS = null_space(rowsS).T
Pm = np.vstack([rowsT, compT]); Qm = np.vstack([rowsS, compS])
A = np.linalg.solve(Pm, Qm)                                           # rowsT A = rowsS
print("C  compatible rank-deficient stack: equal left kernels", same,
      " |T_* A - S_*| =", np.abs(Tst @ A - Sst).max(), " det A =", np.linalg.det(A))
T1 = np.array([[1.0, 0.0], [1.0, 0.0]]); S1 = np.array([[1.0, 0.0], [2.0, 0.0]])
print("   counterexample: ker(T^T) =", null_space(T1.T).ravel(), " ker(S^T) =", null_space(S1.T).ravel())

# ---------------- helpers for the tangent learning model ----------------
Kd = 3; mm = Kd * Kd


def gl_fisher(code, Rc, Mc):
    return np.kron(code.T @ np.linalg.inv(Rc) @ code, Mc)


def blockdiag(blocks):
    n = sum(b.shape[0] for b in blocks); out = np.zeros((n, n)); k = 0
    for b in blocks:
        s = b.shape[0]; out[k:k + s, k:k + s] = b; k += s
    return out


def loop_matrix(loops, nE):
    B = np.zeros((len(loops) * mm, nE * mm))
    for g, gam in enumerate(loops):
        for e in gam:
            B[g * mm:(g + 1) * mm, e * mm:(e + 1) * mm] += np.eye(mm)
    return B


def limit_mean(Z, Lam0, w):
    S = Z.T @ Lam0 @ Z
    return w - Z @ np.linalg.solve(S, Z.T @ Lam0 @ w)


# ---------------- D: complementary(ii) with Q != 0 ----------------
s2 = 0.7
T0, T1c = rng.standard_normal((2, Kd)), rng.standard_normal((2, Kd))   # codes on 0->1 and 1->0
edges = [(1, 0), (0, 1)]                                                # (sender, receiver); e0 = omega_01, e1 = omega_10
codes = [T1c, T0]                                                       # edge 1->0 carries sender 1's code
F = blockdiag([gl_fisher(c, np.eye(2), np.eye(Kd)) for c in codes])
Z = null_space(F)
B = loop_matrix([[0, 1]], 2)
k0, k1 = null_space(T0), null_space(T1c)
U = null_space(np.hstack([k0, k1]).T)                                   # basis of row(T0) cap row(T1)
Pcap = U @ U.T
w_star = rng.standard_normal(2 * mm)                                    # dissonant truth
Hstar = (w_star[:mm] + w_star[mm:]).reshape(Kd, Kd)
QBw = (Pcap @ Hstar).ravel()
for lam in (1e2, 1e4, 1e6):
    L0 = np.eye(2 * mm) / s2 + lam * B.T @ B
    w_inf = limit_mean(Z, L0, w_star)
    print("D  lam =", lam, ": |B w_inf - Q B w*| =", np.abs(B @ w_inf - QBw).max(),
          " loop energy", float((B @ w_inf) @ (B @ w_inf)), " D(w*) =", float(QBw @ QBw))
L_N = Z.T @ B.T @ B @ Z
lam = 3.0
S = Z.T @ (np.eye(2 * mm) / s2 + lam * B.T @ B) @ Z
print("   it1:learning(iii): |S^-1 - s2 (I + lam s2 L_N)^-1| =",
      np.abs(np.linalg.inv(S) - s2 * np.linalg.inv(np.eye(Z.shape[1]) + lam * s2 * L_N)).max(),
      " rank L_N =", np.linalg.matrix_rank(L_N, tol=1e-9), " predicted K dim(K0+K1) =",
      Kd * np.linalg.matrix_rank(np.hstack([k0, k1]), tol=1e-9))

# ---------------- E: complementary row spaces, dissonant truth ----------------
Ta = np.array([[1.0, 0.0, 0.0]]); Tb = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])   # row spaces span R^3, trivial intersection
codesE = [Ta, Tb]
FE = blockdiag([gl_fisher(c, np.eye(c.shape[0]), np.eye(Kd)) for c in codesE])
ZE = null_space(FE)
w_dis = rng.standard_normal(2 * mm)
w_flat = np.concatenate([np.zeros(mm), np.zeros(mm)]) + 0.0
a = rng.standard_normal(mm); w_flat = np.concatenate([a, -a])        # flat truth: omega_01 + omega_10 = 0
for name, wt in (("flat truth", w_flat), ("dissonant truth", w_dis)):
    for lam in (1e3, 1e7):
        L0 = np.eye(2 * mm) / s2 + lam * B.T @ B
        w_inf = limit_mean(ZE, L0, wt)
        print("E ", name, " lam =", lam, ": |w_inf - w*| =", np.abs(w_inf - wt).max(),
              " |B w*| =", np.abs(B @ wt).max(), " |B w_inf| =", np.abs(B @ w_inf).max(),
              " invisible-part error |Z^T(w_inf - w*)| =", np.abs(ZE.T @ (w_inf - wt)).max())

# ---------------- F: ownership in the complete exchange ----------------
cx_edges = [(1, 0), (0, 1), (2, 1), (1, 2), (0, 2), (2, 0)]          # (sender, receiver)
cx_loops = [[0, 1], [2, 3], [4, 5], [1, 3, 5], [4, 2, 0]]
senders = [rng.standard_normal((1, Kd)) for _ in range(3)]
cx_codes = [senders[t] for (t, h) in cx_edges]
Mref = np.eye(Kd) + 0.3 * np.ones((Kd, Kd))
nmsg, lamF = 40.0, 2.0
Fx = blockdiag([gl_fisher(c, np.eye(1), Mref) for c in cx_codes])
Bc = loop_matrix(cx_loops, 6)
Lam = np.eye(6 * mm) / s2 + lamF * Bc.T @ Bc + nmsg * Fx             # exact tangent posterior precision
w_true = 0.3 * rng.standard_normal(6 * mm)
eta = nmsg * Fx @ w_true + 0.2 * rng.standard_normal(6 * mm) * 0.0 + 0.05 * (Fx @ rng.standard_normal(6 * mm))
exact = np.linalg.solve(Lam, eta)
owner = [h for (t, h) in cx_edges]
idx = [np.concatenate([np.arange(e * mm, (e + 1) * mm) for e in range(6) if owner[e] == i]) for i in range(3)]
# receiver-factorized reverse-KL (block coordinate ascent on the exact posterior)
mf = np.zeros(6 * mm)
for _ in range(500):
    for ix in idx:
        rest = np.setdiff1d(np.arange(6 * mm), ix)
        mf[ix] = np.linalg.solve(Lam[np.ix_(ix, ix)], eta[ix] - Lam[np.ix_(ix, rest)] @ mf[rest])
# per-agent law-level model: own anchor + own messages + each loop factor convolved with the
# product of the peers' current dictionary laws N(a_f, V_f) (Item 2, eq. it2:widening)
own_prec = np.eye(6 * mm) / s2 + nmsg * Fx
means = np.zeros(6 * mm); V = [np.zeros((mm, mm)) for _ in range(6)]
for it in range(3000):
    new_means = means.copy(); newV = list(V)
    for i in range(3):
        ix = idx[i]; my_edges = [e for e in range(6) if owner[e] == i]
        H = own_prec[np.ix_(ix, ix)].copy(); b = eta[ix].copy()
        for gam in cx_loops:
            mine = [e for e in gam if e in my_edges]
            if not mine:
                continue
            others = [f for f in gam if f not in my_edges]
            Vs = sum((V[f] for f in others), np.zeros((mm, mm)))
            Om = np.linalg.inv(np.eye(mm) / lamF + Vs)
            # loop factor 1/2 || sum_{mine} w_e + sum_{others} a_f ||_Om^2
            sel = np.zeros((mm, len(ix)))
            for e in mine:
                pos = my_edges.index(e)
                sel[:, pos * mm:(pos + 1) * mm] += np.eye(mm)
            abar = sum((means[f * mm:(f + 1) * mm] for f in others), np.zeros(mm))
            H += sel.T @ Om @ sel; b -= sel.T @ Om @ abar
        C = np.linalg.inv(H)
        new_means[ix] = C @ b
        for pos, e in enumerate(my_edges):
            newV[e] = C[pos * mm:(pos + 1) * mm, pos * mm:(pos + 1) * mm]
    delta = np.abs(new_means - means).max() + max(np.abs(a_ - b_).max() for a_, b_ in zip(newV, V))
    means, V = new_means, newV
    if delta < 1e-13:
        break
# plug-in per-agent model: loop factors with peers at their means, precision lam W
pl = np.zeros(6 * mm)
for _ in range(3000):
    old = pl.copy()
    for i in range(3):
        ix = idx[i]; rest = np.setdiff1d(np.arange(6 * mm), ix)
        pl[ix] = np.linalg.solve(Lam[np.ix_(ix, ix)], eta[ix] - Lam[np.ix_(ix, rest)] @ pl[rest])
    if np.abs(pl - old).max() < 1e-14:
        break
print("F  complete exchange, tangent learning model:")
print("   |receiver-factorized reverse-KL means - exact| =", np.abs(mf - exact).max())
print("   |plug-in per-agent fixed point - exact|        =", np.abs(pl - exact).max())
print("   |law-level per-agent fixed point - exact|      =", np.abs(means - exact).max(),
      " (iterations", it, ", |exact| =", np.abs(exact).max(), ")")
ex_cov = np.linalg.inv(Lam)
tr_ex = sum(np.trace(ex_cov[np.ix_(ix, ix)]) for ix in idx)
tr_mf = sum(np.trace(np.linalg.inv(Lam[np.ix_(ix, ix)])) for ix in idx)
tr_law = sum(np.trace(v) for v in V)
print("   total variance: exact", tr_ex, " receiver-factorized", tr_mf, " law-level per-agent", tr_law)

# ---------------- G: bandwidth strong-law limit ----------------
mu = np.array([0.0, 0.4, -0.3]); sd = np.array([1.0, 1.0, 1.5]); true_mu, true_sd = 0.2, 1.2
def logpdf(x, m, s):
    return -0.5 * ((x - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
def kl_g(m1, s1, m2, s2_):
    return np.log(s2_ / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2_ ** 2) - 0.5
h = rng.normal(true_mu, true_sd, 200000)
for n in (2000, 200000):
    lr = (logpdf(h[:n], mu[0], sd[0]) - logpdf(h[:n], mu[2], sd[2])).mean()
    print("G  n =", n, ": (1/n) log odds(0 vs 2) =", lr, " KL(nu||nu_2) - KL(nu||nu_0) =",
          kl_g(true_mu, true_sd, mu[2], sd[2]) - kl_g(true_mu, true_sd, mu[0], sd[0]))

# ---------------- E2: confidence of the wrong limit (complementary row spaces, dissonant truth) ----------------
for lam in (1e3, 1e7):
    L0 = np.eye(2 * mm) / s2 + lam * B.T @ B
    cov_inv_coords = np.linalg.inv(ZE.T @ L0 @ ZE)            # limiting covariance of invisible coordinates
    w_inf = limit_mean(ZE, L0, w_dis)
    err = ZE.T @ (w_inf - w_dis)
    print("E2 lam =", lam, ": largest limiting invisible variance =", np.linalg.eigvalsh(cov_inv_coords).max(),
          " squared Mahalanobis error of the truth under the limit law =", float(err @ ZE.T @ L0 @ ZE @ err))

# ---------------- H: it1:frustration needs strong connectivity (base-point dependence) ----------------
def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
# edges as (receiver i, sender j, theta_ij): mu_i = theta_ij mu_j
fr_edges = [(0, 1, rot(0.3)), (1, 0, rot(0.4)),        # component A, two-cycle holonomy rotation 0.7
            (2, 3, np.eye(2)), (3, 2, np.eye(2)),      # component B, identity holonomy
            (0, 2, rot(1.1))]                          # 0 receives from 2; no edge from A back to B
rows = []
for (i, j, Th) in fr_edges:
    Bm = np.zeros((2, 8)); Bm[:, 2 * i:2 * i + 2] = np.eye(2); Bm[:, 2 * j:2 * j + 2] = -Th
    rows.append(Bm)
Bm = np.vstack(rows)
print("H  weakly connected graph: least eigenvalue of the consensus energy =", np.linalg.eigvalsh(Bm.T @ Bm).min(),
      "; holonomies based at vertex 2 are {I} (common fixed space R^2), based at vertex 0 rotation 0.7 (none)")
