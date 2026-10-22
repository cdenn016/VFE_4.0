r"""L06-ERA script 6: scope of prop:public-gibbs-identifies.

Block normalizer for a fixed comparison channel K (rows: input letters, columns: outputs):
    Z(r) = sum_j w_j exp(-KL(r || q_j K)/T).
(a) Identity channel: recover the weighted empirical measure from Z along one probe line
    by fitting exponentials (Vandermonde), illustrating the proof.
(b) Deterministic merge channel pi: {0,1,2} -> {0,1}: two different child collections with
    the same pushforwards give identical Z for random receivers; a single law with the
    same image reproduces a block of children with a common image.
(c) Incoming Fisher force under a receiver r: \dot q_j = -q_j(h - E_q h), h = -beta K(r/nu_j).
    For the merge channel the image obeys \dot nu_j = beta (r - nu_j) (closed); for a
    non-deterministic, non-injective K two inputs with the same image have different
    image velocities (not closed).
(d) The three-atom restricted-descriptor counterexample recorded in the research notes.
"""
import numpy as np

rng = np.random.default_rng(7)
T = 0.8


def kl(p, q):
    return float((p * np.log(p / q)).sum())


def Z(r, Qs, ws, K):
    return sum(w * np.exp(-kl(r, q @ K) / T) for q, w in zip(Qs, ws))


# (a) identity channel recovery along a probe line
lam = np.full(3, 1 / 3)
F = [np.array([1.2, 0.9, 0.9]), np.array([0.6, 1.5, 0.9]), np.array([1.0, 1.0, 1.0])]
ws = [0.5, 0.3, 0.2]
h = np.array([1.0, -0.4, -0.6])  # mean zero under lam
v = np.array([(lam * h * np.log(f)).sum() for f in F])
ts = np.linspace(-0.05, 0.05, 7)
rows = []
rhs = []
for t in ts:
    r = 1 + t * h
    ent = (lam * r * np.log(r)).sum()
    val = np.exp(ent / T) * sum(w * np.exp(-((lam * r * np.log(r / f)).sum()) / T) for f, w in zip(F, ws))
    rows.append(np.exp(t * v / T))
    rhs.append(val)
coef, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
base = np.array([np.exp((lam * np.log(f)).sum() / T) for f in F])
print('(a) recovered weights:', np.round(coef / base, 10), ' true:', ws)

# (b) merge channel
Km = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
C1 = [np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.4, 0.4])]
C2 = [np.array([0.5, 0.1, 0.4]), np.array([0.2, 0.7, 0.1])]
w2 = [1.0, 1.0]
worst = 0.0
for _ in range(200):
    r = rng.dirichlet([2, 2])
    worst = max(worst, abs(Z(r, C1, w2, Km) - Z(r, C2, w2, Km)))
print(f'(b) merge channel: max |Z_C1(r) - Z_C2(r)| over 200 receivers = {worst:.2e} (distinct child laws)')
C3 = [np.array([0.5, 0.3, 0.2]), np.array([0.5, 0.1, 0.4])]
single = [np.array([0.5, 0.25, 0.25])]
worst = 0.0
for _ in range(200):
    r = rng.dirichlet([2, 2])
    worst = max(worst, abs(Z(r, C3, [1.0, 1.0], Km) - Z(r, single, [2.0], Km)))
print(f'(b) merge channel: block of two distinct children vs one ordinary law, max |dZ| = {worst:.2e}')
# injective non-identity channel (binary symmetric, crossover 0.2): images determine inputs
Kb = np.array([[0.8, 0.2], [0.2, 0.8]])
print('(b) BSC(0.2) is injective on laws: det =', np.linalg.det(Kb))


def image_velocity(q, K, r, beta=1.0):
    nu = q @ K
    hfun = -beta * (K @ (r / nu))
    qdot = -q * (hfun - (q * hfun).sum())
    return qdot @ K, nu


r = np.array([0.35, 0.65])
for q in C3:
    nd, nu = image_velocity(q, Km, r)
    print(f'(c) merge: q={q}, nu={nu}, nu_dot={nd}, beta(r-nu)={r-nu}, residual={np.abs(nd-(r-nu)).max():.1e}')
Kn = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45]])
qa = np.array([0.4, 0.3, 0.3])
nu_a = qa @ Kn
# find qb != qa with the same image: move along the kernel of K^T restricted to sum zero
null = np.array([1.0, 1.0, -2.0])  # sum zero; check image change
dimg = null @ Kn
# adjust so the image is unchanged: solve for direction d with sum(d)=0 and d@Kn = 0
M = np.vstack([np.ones(3), Kn.T])
_, _, Vt = np.linalg.svd(M)
dvec = Vt[-1]
qb = qa + 0.15 * dvec / np.abs(dvec).max()
print('(c) non-deterministic K: images', qa @ Kn, qb @ Kn)
na, _ = image_velocity(qa, Kn, r)
nb, _ = image_velocity(qb, Kn, r)
print(f'(c) non-deterministic K: image velocities {na} vs {nb}; difference {np.abs(na-nb).max():.3e}')

# (d) restricted descriptor counterexample
c = (1 + np.sqrt(5)) / 8
p1 = np.array([0.25, 0.25, 0.5])
p2 = np.array([c, c, 1 - 2 * c])
lam = np.full(3, 1 / 3)
b1 = (lam * np.log(p1 / lam)).sum()
b2 = (lam * np.log(p2 / lam)).sum()
db1 = (lam * lam / p1).sum() - 1   # alpha=1, r0 = r_theta = 1
db2 = (lam * lam / p2).sum() - 1
print(f'(d) intercepts {b1:.12f} {b2:.12f}; intercept velocities {db1:.6f} {db2:.6f} '
      f'(claimed 1/9={1/9:.6f}, (5sqrt5-10)/9={(5*np.sqrt(5)-10)/9:.6f})')
