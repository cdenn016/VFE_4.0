r"""L01-FOUND checks of s01_probability.tex, record_process.tex, the agreement-bit example of
01_foundations.tex, the binary mixture of c01_objects.tex, and two extensions (stabilizer
blindness of law-level loop comparisons; variational record-process gap).

All computations are exact enumerations on finite alphabets except where stated.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python check_probability_records.py
"""
from __future__ import annotations

import itertools

import numpy as np

rng = np.random.default_rng(7)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL(p||q) = \sum p log(p/q) on a finite alphabet, +inf if p is not << q."""
    p, q = np.asarray(p, float).ravel(), np.asarray(q, float).ravel()
    m = p > 0
    if np.any(q[m] == 0):
        return np.inf
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def rand_simplex(*shape: int) -> np.ndarray:
    x = rng.gamma(1.0, size=shape)
    return x / x.sum(axis=-1, keepdims=True)


def report(name: str, val: float) -> None:
    print(f"{name:72s} residual = {val:.3e}")


# eq:s-chain-rule on a random joint law on X x Y (|X| = 3, |Y| = 4)
mu = rand_simplex(12).reshape(3, 4)
nu = rand_simplex(12).reshape(3, 4)
muY, nuY = mu.sum(0), nu.sum(0)
cond = sum(muY[y] * kl(mu[:, y] / muY[y], nu[:, y] / nuY[y]) for y in range(4))
report("eq:s-chain-rule", abs(kl(mu, nu) - (kl(muY, nuY) + cond)))
# chain rule without the finiteness hypothesis: both sides infinite together
nu0 = nu.copy(); nu0[0, 0] = 0.0; nu0 /= nu0.sum()
nu0Y = nu0.sum(0)
cond0 = sum(muY[y] * kl(mu[:, y] / muY[y], nu0[:, y] / nu0Y[y]) for y in range(4))
print(f"   chain rule with KL = inf: lhs = {kl(mu, nu0)}, rhs = {kl(muY, nu0Y) + cond0}")

# eq:s-different-evaluators: s,t on |M| = 3, K,L kernels into |O| = 4
s, t = rand_simplex(3), rand_simplex(3)
K, L = rand_simplex(3, 4), rand_simplex(3, 4)
lhs = kl(s, t) + sum(s[m] * kl(K[m], L[m]) for m in range(3))
J1, J2 = s[:, None] * K, t[:, None] * L
sK, tL = J1.sum(0), J2.sum(0)
rhs = kl(sK, tL) + sum(sK[o] * kl(J1[:, o] / sK[o], J2[:, o] / tL[o]) for o in range(4))
report("eq:s-different-evaluators", abs(lhs - rhs))
# presentation change: relabel model points by a permutation Theta; compare s with Theta# t, L(Theta^-1 m)
perm = np.array([2, 0, 1])  # Theta(m) = perm[m]
t_push = np.zeros(3); t_push[perm] = t
L_push = np.zeros_like(L); L_push[perm] = L
lhs2 = kl(s, t_push) + sum(s[m] * kl(K[m], L_push[m]) for m in range(3))
J2p = t_push[:, None] * L_push
rhs2 = kl(sK, J2p.sum(0)) + sum(sK[o] * kl(J1[:, o] / sK[o], J2p[:, o] / J2p.sum(0)[o]) for o in range(4))
report("eq:s-different-evaluators after presentation change", abs(lhs2 - rhs2))

# four-point conditional reconstruction
q = np.array([[0.5, 0.0], [0.0, 0.5]])  # q[H, V], V = H
qhat = q.sum(1)[:, None] * np.array([[0.5, 0.5], [0.5, 0.5]])
report("four-point reconstruction KL(q||qhat) = log 2", abs(kl(q, qhat) - np.log(2)))
report("four-point conditional divergence at each H = log 2",
       max(abs(kl(q[h] / q[h].sum(), qhat[h] / qhat[h].sum()) - np.log(2)) for h in range(2)))

# shared source and two routes
for bmark in (0, 1):
    pO_shared = np.zeros(2)
    for x in (0, 1):
        pO_shared[(x ^ (x ^ bmark))] += 0.5
    pO_indep = np.zeros(2)
    for x, xp in itertools.product((0, 1), repeat=2):
        pO_indep[x ^ (xp ^ bmark)] += 0.25
    print(f"   route mark b={bmark}: shared-source O law {pO_shared.tolist()}, independent-redraw O law {pO_indep.tolist()}")

# predictive sufficiency
pXAB = np.full((2, 2), 0.25)  # (A,B) uniform independent, U = A
HB_given_A = -np.sum(pXAB * np.log(pXAB / pXAB.sum(1, keepdims=True)))
print(f"   I(B;X|U=A) = H(B|A) - 0 = {HB_given_A:.6f} (log 2 = {np.log(2):.6f}); I(A;X|U=A) = 0")

# agreement-bit example (01_foundations, c01)
P = np.full((2, 2), 0.25)
Pc = P * np.eye(2); Pc /= Pc.sum()
print(f"   agreement: marginals {Pc.sum(1).tolist()} {Pc.sum(0).tolist()}, E[XY] {P[1,1]:.3f} -> {Pc[1,1]:.3f},"
      f" mutual information {kl(Pc, np.outer(Pc.sum(1), Pc.sum(0))):.6f}")
# product (mean-field) variational description of the same record: hard registration
best = None
for a, b in itertools.product(np.linspace(0, 1, 101), repeat=2):
    Q = np.outer([1 - a, a], [1 - b, b])
    F = kl(Q, Pc)
    if best is None or F < best[0]:
        best = (F, a, b)
print(f"   hard agreement record: best product KL(Q||posterior) = {best[0]:.6f} at (a,b) = ({best[1]:.2f},{best[2]:.2f})")
# soft registration: posterior proportional to exp(J s1 s2), s = +-1; mean-field fixed point m = tanh(J m)
for J in (0.5, 0.9, 1.1, 2.0):
    Pj = np.array([[np.exp(J), np.exp(-J)], [np.exp(-J), np.exp(J)]]); Pj /= Pj.sum()
    grid = np.linspace(-0.999, 0.999, 1999)
    vals = []
    for m in grid:
        p1 = np.array([(1 - m) / 2, (1 + m) / 2])
        vals.append(kl(np.outer(p1, p1), Pj))
    m_star = grid[int(np.argmin(vals))]
    print(f"   soft record J={J}: symmetric product optimum magnetization |m*| = {abs(m_star):.3f}"
          f" (mean-field theory: 0 iff J <= 1), KL at m=0: {kl(np.full((2,2),0.25), Pj):.4f}, at m*: {min(vals):.4f}")

# c01 binary mixture
s1 = 0.6
print(f"   c01 mixture (s G)(k=1) = {s1 * 0.9 + (1 - s1) * 0.2:.4f}")

# prop:registered-history: finite S = (core c in {0,1,2}) x (register o in {0,1}); O = {0,1}
cores, obs = [0, 1, 2], [0, 1]
states = list(itertools.product(cores, obs))
Rk = {x: rand_simplex(2) for x in states}           # R(x, do)
Aup = {(x, o): ((x[0] + o + 1) % 3, o) for x in states for o in obs}  # A(x, o) = (new core, o)
rho = lambda x: x[1]
mu0 = rand_simplex(len(states))
n = 4
Pn = {}
for path in itertools.product(range(len(states)), *([obs] * n)):
    x = states[path[0]]
    p = mu0[path[0]]
    hist = [x]
    for o in path[1:]:
        p *= Rk[x][o]
        x = Aup[(x, o)]
        hist.append(x)
    key = (tuple(hist), tuple(path[1:]))
    Pn[key] = Pn.get(key, 0.0) + p
total = sum(Pn.values())
agree = sum(p for (h, os), p in Pn.items() if all(rho(h[t + 1]) == os[t] for t in range(n)))
report("prop:registered-history total mass", abs(total - 1))
report("prop:registered-history records agree with readouts a.s.", abs(agree - 1))
# marginal consistency: sum over last pair gives P_{n-1}
Pn1 = {}
for (h, os), p in Pn.items():
    key = (h[:-1], os[:-1])
    Pn1[key] = Pn1.get(key, 0.0) + p
Pdirect = {}
for path in itertools.product(range(len(states)), *([obs] * (n - 1))):
    x = states[path[0]]; p = mu0[path[0]]; hist = [x]
    for o in path[1:]:
        p *= Rk[x][o]; x = Aup[(x, o)]; hist.append(x)
    key = (tuple(hist), tuple(path[1:]))
    Pdirect[key] = Pdirect.get(key, 0.0) + p
report("prop:registered-history marginal consistency", max(abs(Pn1.get(k, 0) - v) for k, v in Pdirect.items()))

# prop:bayesian-history and the variational record-process gap
nm, no, n = 3, 3, 4
k = rand_simplex(nm, no)          # k(o | m)
s0 = rand_simplex(nm)
alpha = 0.6                        # damped (non-Bayesian) assimilation: s <- (1-alpha) s + alpha Bayes(s, o)


def bayes(sv: np.ndarray, o: int) -> np.ndarray:
    w = sv * k[:, o]
    return w / w.sum()


worst_bayes = 0.0
KL_total = 0.0
sum_pred_kl = 0.0
sum_post_kl = 0.0
for os_ in itertools.product(range(no), repeat=n):
    sb, sv = s0.copy(), s0.copy()
    pb, pv = 1.0, 1.0
    # weights of the prefix under the Bayesian process for the chain-rule decomposition
    for tt, o in enumerate(os_):
        predb, predv = sb @ k, sv @ k
        pb_prefix = pb
        pb *= predb[o]
        pv *= predv[o]
        sb_next = bayes(sb, o)
        sv_next = (1 - alpha) * sv + alpha * bayes(sv, o)
        sb, sv = sb_next, sv_next
    direct = float(s0 @ np.prod(k[:, list(os_)], axis=1))
    worst_bayes = max(worst_bayes, abs(pb - direct))
    KL_total += pb * np.log(pb / pv)
# chain-rule terms computed by recursion over prefixes
def prefix_terms(prefix: tuple, sb: np.ndarray, sv: np.ndarray, w: float) -> tuple[float, float]:
    if len(prefix) == n:
        return 0.0, 0.0
    predb, predv = sb @ k, sv @ k
    a = w * kl(predb, predv)
    c = w * kl(sb, sv)
    for o in range(no):
        a2, c2 = prefix_terms(prefix + (o,), bayes(sb, o), (1 - alpha) * sv + alpha * bayes(sv, o), w * predb[o])
        a += a2; c += c2
    return a, c
sum_pred_kl, sum_post_kl = prefix_terms((), s0.copy(), s0.copy(), 1.0)
report("prop:bayesian-history sequential product = mixture", worst_bayes)
report("extension: KL(P_Bayes||P_var) = sum_t E KL(one-step predictives)", abs(KL_total - sum_pred_kl))
print(f"   extension: KL(P_Bayes||P_var) = {KL_total:.6f} <= sum_t E KL(posteriors) = {sum_post_kl:.6f}")

# Extension: stabilizer blindness of law-level loop comparisons
# Z2 route mark (s01 two-route example): nu uniform on {0,1}, H = flip
nu_bit = np.array([0.5, 0.5])
H_push = nu_bit[::-1]
pair_law = np.zeros((2, 2))
for y in (0, 1):
    pair_law[y, 1 - y] += 0.5
print(f"   Z2 loop: KL(nu||H#nu) = {kl(nu_bit, H_push):.3e}; shared-sample pair law (Y, HY) = {pair_law.tolist()}"
      f"; XOR record = {sum(pair_law[y, z] for y in (0,1) for z in (0,1) if y ^ z == 1):.1f}")
# Gaussian rotation: isotropic law cannot see a rotation holonomy, anisotropic law can
def gauss_kl(m0, S0, m1, S1):
    S1i = np.linalg.inv(S1); dm = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + dm @ S1i @ dm - len(m0) + np.log(np.linalg.det(S1) / np.linalg.det(S0)))
th = 0.7
Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
S_iso, S_an = np.eye(2), np.diag([1.0, 3.0])
print(f"   rotation holonomy theta={th}: KL isotropic = {gauss_kl(np.zeros(2), S_iso, np.zeros(2), Rot @ S_iso @ Rot.T):.3e},"
      f" KL anisotropic = {gauss_kl(np.zeros(2), S_an, np.zeros(2), Rot @ S_an @ Rot.T):.4f}")
