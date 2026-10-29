r"""XC6-PROG s4: recomputation of worked examples in chapters/05_inheritance.tex.

 (a) inh:intertwining example: Lambda T = T^c Lambda, failure of strong lumpability, and the
     total-variation gap of retained path laws from delta_1.
 (b) inh:role-loss numbers.
 (c) inh:smooth-mixture-failure: retained belief derivatives at A=(0,c,0,c), B=(a,c,-a,c),
     a=1/2, c=1/4, from the full flow by automatic finite differences of E.
 (d) inh:smooth-boundary-failure numbers.
 (e) inh:common-clock: d/dt log(q_i/q_j) = -2 log(q_i/q_j) + c_ij(t) (full, dynamic rows) and
     -1 under the null, for an additive-moment channel C[M] on finite tables.
 (f) finite two-level construction: lumpability of R10 and R21 by enumeration.
 (g) anisotropic example and the covariance formula for dx/dt = A x.
"""
import itertools

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(11)

# (a) ---------------------------------------------------------------------------------
Tm = np.array([[0.3, 0.5, 0.2], [0.4, 0.2, 0.4], [0.25, 0.25, 0.5]])
R = np.array([0, 0, 1])                        # states 1,2 -> a ; 3 -> b
Lam = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
Tc = np.array([[0.7, 0.3], [0.5, 0.5]])
print(f"(a) |Lambda T - T^c Lambda| = {np.abs(Lam @ Tm - Tc @ Lam).max():.2e}; "
      f"lumpability sums into b from states 1,2: {Tm[0, 2]}, {Tm[1, 2]}")


def fine_path_law(init: np.ndarray, length: int) -> dict:
    """Law of (Z_0,...,Z_{length-1}) for the fine chain observed through R."""
    out = {}
    for path in itertools.product([0, 1], repeat=length):
        f = init * (R == path[0])
        for z in path[1:]:
            f = (f @ Tm) * (R == z)
        out[path] = f.sum()
    return out


def coarse_path_law(nu: np.ndarray, length: int) -> dict:
    out = {}
    for path in itertools.product([0, 1], repeat=length):
        pr = nu[path[0]]
        for a, b in zip(path[:-1], path[1:]):
            pr *= Tc[a, b]
        out[path] = pr
    return out


for L in [5, 6]:
    fp = fine_path_law(np.array([1.0, 0.0, 0.0]), L)
    cp = coarse_path_law(np.array([1.0, 0.0]), L)
    tv = 0.5 * sum(abs(fp[k] - cp[k]) for k in fp)
    print(f"    TV of retained path laws of {L} states from delta_1: {tv:.6f}")
fp = fine_path_law(np.array([0.5, 0.5, 0.0]), 6)
cp = coarse_path_law(np.array([1.0, 0.0]), 6)
print(f"    from nu Lambda with nu = delta_a (6 states): TV = {0.5 * sum(abs(fp[k] - cp[k]) for k in fp):.2e}")

# (b) ---------------------------------------------------------------------------------
for p1, p2 in [(0.8, 0.2), (0.2, 0.8)]:
    law = [(1 - p1) * (1 - p2), p1 * (1 - p2) + (1 - p1) * p2, p1 * p2]
    print(f"(b) (p1,p2)=({p1},{p2}): law of K1+K2 = {np.round(law, 4)}, d/dt E = {-(p1 + 2 * p2):.2f}")

# (c) ---------------------------------------------------------------------------------
J = 0.5


def psi(x: float) -> float:
    return 0.5 * (1 + x) * np.log(1 + x) + 0.5 * (1 - x) * np.log(1 - x)


def fmap(x: float) -> float:
    return (x + x**3) / 4.0


def dkl(x: float, z: float) -> float:
    p, q = (1 + x) / 2, (1 + z) / 2
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def Pp(x1: float, x2: float) -> float:
    return dkl(x1, fmap(x2)) + dkl(x2, fmap(x1))


def Etot(v: np.ndarray) -> float:
    x1, y1, x2, y2 = v
    E0 = sum(psi(x) + psi(y) - J * x * y + np.log(np.cosh(J)) for x, y in [(x1, y1), (x2, y2)])
    return E0 + Pp(x1, x2) + Pp(y1, y2)


def flow(v: np.ndarray) -> np.ndarray:
    g = np.zeros(4)
    h = 1e-6
    for k in range(4):
        e = np.zeros(4)
        e[k] = h
        g[k] = (Etot(v + e) - Etot(v - e)) / (2 * h)
    return -(1 - v**2) * g


a, c = 0.5, 0.25
A = np.array([0.0, c, 0.0, c])
B = np.array([a, c, -a, c])
fa, fb = flow(A), flow(B)
print(f"(c) xbar' at A = {0.5 * (fa[0] + fa[2]):.6f} (claim J c = {J * c}); at B = {0.5 * (fb[0] + fb[2]):.6f} "
      f"(claim J c (1-a^2) = {J * c * (1 - a**2)}); ybar' equal: {abs(0.5*(fa[1]+fa[3]) - 0.5*(fb[1]+fb[3])):.2e}")

# (d) ---------------------------------------------------------------------------------
m, aa = 0.25, 0.25
b0 = (fmap(m) + fmap(m)) / 2
b1 = (fmap(m + aa) + fmap(m - aa)) / 2
print(f"(d) response means {b1 * 256:.4f}/256 and {b0 * 256:.4f}/256; difference {b1 - b0:.6f} vs 3/4 m a^2 = {0.75 * m * aa**2:.6f}")

# (e) ---------------------------------------------------------------------------------
Nn, Kk, Mm = 3, 3, 2
Tq = 0.4
Tstat = rng.normal(size=(Kk, 2))               # statistic T: K -> R^2
G0 = np.exp(rng.normal(size=Kk))
G0 /= G0.sum()


def Cof(Mv: np.ndarray) -> np.ndarray:
    w = np.exp(Tstat @ np.tanh(Mv))
    return w / w.sum()


piq = np.full((Nn, Nn), 0.5)
np.fill_diagonal(piq, 0.0)


def belief_force(Q: np.ndarray, Bm: np.ndarray, null: bool = False) -> np.ndarray:
    """h_i = dE/dq_i with model-independent evaluator G0 and channel C[M] (self excluded)."""
    Mv = sum(Q[i] @ Tstat for i in range(Nn))
    Cv = Cof(Mv)
    h = np.zeros((Nn, Kk))
    for i in range(Nn):
        h[i] = np.log(Q[i] / G0) + 1.0
        if null:
            continue
        for j in range(Nn):
            if j != i:
                h[i] += Bm[i, j] * (np.log(Q[i] / Cv) + 1.0)
    if not null:
        # source role through M: numerical derivative of sum_l sum_j beta_lj KL(q_l || C[M])
        def S_of(Mv2: np.ndarray) -> float:
            C2 = Cof(Mv2)
            return sum(Bm[l, j] * np.sum(Q[l] * np.log(Q[l] / C2)) for l in range(Nn) for j in range(Nn) if j != l)
        gradM = np.zeros(2)
        for d in range(2):
            e = np.zeros(2)
            e[d] = 1e-6
            gradM[d] = (S_of(Mv + e) - S_of(Mv - e)) / 2e-6
        for i in range(Nn):
            h[i] += Tstat @ gradM
    return h


Q = np.array([np.exp(rng.normal(size=Kk)) for _ in range(Nn)])
Q /= Q.sum(axis=1, keepdims=True)
Bm = np.array([[0.0, 0.3, 0.7], [0.6, 0.0, 0.4], [0.2, 0.8, 0.0]])
for null in [False, True]:
    h = belief_force(Q, Bm, null)
    dlog = np.array([-(h[i] - Q[i] @ h[i]) for i in range(Nn)])
    coef = -2.0 if not null else -1.0
    resid = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        lr = np.log(Q[i] / Q[j])
        d = dlog[i] - dlog[j] - coef * lr
        resid.append(np.ptp(d))                 # must be constant in k
    print(f"(e) common clock ({'null' if null else 'full'} flow, coefficient {coef}): "
          f"max spread over k of residual = {max(resid):.2e}")

# (f) ---------------------------------------------------------------------------------


def fine_T(x: tuple, xp: tuple) -> float:
    aa_, bb, cc, rr = x
    ap, bp, cp_, rp = xp
    fq = lambda p, z: p if z == 1 else 1 - p
    xi = (ap ^ aa_)
    eta = (bp ^ bb ^ aa_)
    ups = (cp_ ^ cc ^ bb)
    nu1 = 0.5 + aa_ / 8.0
    return fq(0.25, xi) * fq(1 / 3, eta) * fq(0.2, ups) * fq(nu1, rp)


states = list(itertools.product([0, 1], repeat=4))
row_sums = max(abs(sum(fine_T(x, y) for y in states) - 1) for x in states)
err1 = 0.0
for x in states:
    for yp in itertools.product([0, 1], repeat=3):        # (a',b',r')
        s = sum(fine_T(x, (yp[0], yp[1], cpp, yp[2])) for cpp in [0, 1])
        a_, b_, c_, r_ = x
        fq = lambda p, z: p if z == 1 else 1 - p
        T1 = fq(0.25, yp[0] ^ a_) * fq(0.5 + a_ / 8, yp[2]) * fq(1 / 3, yp[1] ^ b_ ^ a_)
        err1 = max(err1, abs(s - T1))
err2 = 0.0
for x in states:
    for yp in itertools.product([0, 1], repeat=2):        # (a',r')
        s = sum(fine_T(x, (yp[0], bpp, cpp, yp[1])) for bpp in [0, 1] for cpp in [0, 1])
        a_ = x[0]
        fq = lambda p, z: p if z == 1 else 1 - p
        err2 = max(err2, abs(s - fq(0.25, yp[0] ^ a_) * fq(0.5 + a_ / 8, yp[1])))
print(f"(f) fine rows normalized ({row_sums:.1e}); first-parent kernel error {err1:.1e}; second-parent kernel error {err2:.1e}")

# (g) ---------------------------------------------------------------------------------
Ad = np.diag([-1.0, -3.0])
d1 = expm(Ad) @ np.array([1.0, 0.0])
d2 = expm(Ad) @ np.array([0.0, 1.5])
print(f"(g) anisotropic differences at t=1: {np.round(d1, 4)}, {np.round(d2, 4)}")
lam, t = 0.7, 0.9
Am = np.array([[-1 - lam, lam], [lam, -1 - lam]])
Cov = expm(Am * t) @ expm(Am * t).T
print(f"    Cov(x1,x2) = {Cov[0, 1]:.8f} vs formula {0.5 * (np.exp(-2 * t) - np.exp(-2 * (1 + 2 * lam) * t)):.8f}")
