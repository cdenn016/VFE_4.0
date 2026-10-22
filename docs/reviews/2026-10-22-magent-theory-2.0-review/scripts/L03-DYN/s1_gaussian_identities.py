"""L03-DYN script 1: recompute every Gaussian identity of supplements/s03_variation.tex.

Independent routes: sympy symbolic algebra (d=1) and exact Gauss-Hermite quadrature
(exact for the polynomial/log-quadratic integrands here) for d=1,2,3 with random SPD data.
Single-threaded, CPU, a few seconds.
"""
import itertools
import numpy as np
import sympy as sp

rng = np.random.default_rng(20261022)


def spd(n, s=1.0):
    A = rng.normal(size=(n, n))
    return A @ A.T + s * np.eye(n)


def gh_nodes(dim, npts=5):
    x, w = np.polynomial.hermite_e.hermegauss(npts)
    w = w / np.sqrt(2 * np.pi)
    pts = np.array(list(itertools.product(x, repeat=dim)))
    wts = np.array([np.prod(c) for c in itertools.product(w, repeat=dim)])
    return pts, wts


def gauss_expect(f, mean, cov, npts=5):
    L = np.linalg.cholesky(cov)
    z, w = gh_nodes(len(mean), npts)
    xs = mean[None, :] + z @ L.T
    return sum(wi * f(xi) for xi, wi in zip(xs, w))


def neglog_normal(x, m, S):
    r = x - m
    return 0.5 * np.log(np.linalg.det(2 * np.pi * S)) + 0.5 * r @ np.linalg.solve(S, r)


def c_R(R):
    return 0.5 * np.log(np.linalg.det(2 * np.pi * R))


def kl_gauss(a, V, b, W):
    d = len(a)
    Wi = np.linalg.inv(W)
    return 0.5 * ((a - b) @ Wi @ (a - b) + np.trace(Wi @ V) - d
                  + np.log(np.linalg.det(W) / np.linalg.det(V)))


out = []

# (1) prop:sup-residual-cost with a correlated joint law
ni, nj, d = 2, 1, 2
S = spd(ni + nj)
mu = rng.normal(size=ni + nj)
A = rng.normal(size=(d, ni))
B = rng.normal(size=(d, nj))
R = spd(d, 0.5)
o = rng.normal(size=d)
lhs = gauss_expect(lambda x: neglog_normal(o, A @ x[:ni] - B @ x[ni:], R), mu, S, npts=5)
Sii, Sjj, Sij = S[:ni, :ni], S[ni:, ni:], S[:ni, ni:]
delta = A @ mu[:ni] - B @ mu[ni:] - o
Vij = A @ Sii @ A.T + B @ Sjj @ B.T - A @ Sij @ B.T - B @ Sij.T @ A.T
rhs = c_R(R) + 0.5 * delta @ np.linalg.solve(R, delta) + 0.5 * np.trace(np.linalg.solve(R, Vij))
out.append(("eq:sup-residual-cost (correlated joint, d=2)", abs(lhs - rhs)))

# (2) eq:sup-peer-score is KL between Gaussians
for dd in (1, 2, 3):
    a = rng.normal(size=dd)
    b = rng.normal(size=dd)
    V = spd(dd)
    W = spd(dd)
    kl_q = gauss_expect(lambda x: neglog_normal(x, b, W) - neglog_normal(x, a, V), a, V, npts=4)
    out.append((f"eq:sup-peer-score = KL quadrature, d={dd}", abs(kl_q - kl_gauss(a, V, b, W))))

# (3) eq:sup-nll-score
dd = 2
a = rng.normal(size=dd)
V = spd(dd)
bj = rng.normal(size=dd)
Wj = spd(dd)
R = spd(dd)
Cj_formula = c_R(R) + 0.5 * ((a - bj) @ np.linalg.solve(R, a - bj) + np.trace(np.linalg.solve(R, V + Wj)))
jm = np.concatenate([a, bj])
jc = np.block([[V, np.zeros((dd, dd))], [np.zeros((dd, dd)), Wj]])
Cj_quad = gauss_expect(lambda x: neglog_normal(np.zeros(dd), x[:dd] - x[dd:], R), jm, jc, npts=4)
out.append(("eq:sup-nll-score quadrature, d=2", abs(Cj_quad - Cj_formula)))

# (4) prop:sup-row-equivalence, symbolic d=1
aS, bS, VS, WS, TS = sp.symbols('a b V W T', positive=True)
half = sp.Rational(1, 2)
cR = lambda Rv: half * sp.log(2 * sp.pi * Rv)
C1 = cR(TS * WS) + half * ((aS - bS) ** 2 / (TS * WS) + (VS + WS) / (TS * WS))
D1 = half * ((aS - bS) ** 2 / WS + VS / WS - 1 + sp.log(WS / VS))
claim = cR(TS * WS) + 1 / TS - sp.log(WS / VS) / (2 * TS)
out.append(("eq:sup-row-constant symbolic d=1 residual",
            sp.simplify(sp.expand_log(C1 - D1 / TS - claim, force=True))))
out.append(("eq:sup-covariance-difference symbolic d=1 residual",
            sp.simplify(sp.diff(C1 - D1 / TS, VS) - 1 / (2 * TS * VS))))
out.append(("d/da (C - D/T) with V free (mean derivatives agree)", sp.simplify(sp.diff(C1 - D1 / TS, aS))))
out.append(("d/db (C - D/T) (source-mean derivatives agree)", sp.simplify(sp.diff(C1 - D1 / TS, bS))))
H1 = half * sp.log(2 * sp.pi * sp.E * VS)
out.append(("T=1: C - D - H(N(a,V)) - d/2, d=1",
            sp.simplify(sp.expand_log((C1 - D1).subs(TS, 1) - H1 - half, force=True))))

# numeric d=2
dd = 2
T = 0.7
a = rng.normal(size=dd)
V = spd(dd)
W = spd(dd)


def gap(Vm, b_):
    Cj = c_R(T * W) + 0.5 * ((a - b_) @ np.linalg.solve(T * W, a - b_) + np.trace(np.linalg.solve(T * W, Vm + W)))
    return Cj - kl_gauss(a, Vm, b_, W) / T


claimed = c_R(T * W) + dd / T - np.log(np.linalg.det(W) / np.linalg.det(V)) / (2 * T)
bs = [rng.normal(size=dd) for _ in range(4)]
out.append(("eq:sup-row-constant numeric d=2, 4 sources, max residual", max(abs(gap(V, b_) - claimed) for b_ in bs)))
Edir = spd(dd) * 1e-3
h = 1e-5
dirder = (gap(V + h * Edir, bs[0]) - gap(V - h * Edir, bs[0])) / (2 * h)
out.append(("eq:sup-covariance-difference numeric d=2 (directional, FD)",
            abs(dirder - np.trace(np.linalg.solve(V, Edir)) / (2 * T))))

# (5) self cost, independent snapshot, odds
dd = 3
T = 0.4
R = T * np.eye(dd)
o = np.zeros(dd)
self_cost = c_R(R)
indep = gauss_expect(lambda x: neglog_normal(o, x[:dd] - x[dd:], R), np.zeros(2 * dd), np.eye(2 * dd), npts=3)
out.append(("independent snapshot cost - (c_R + d/T)", abs(indep - (c_R(R) + dd / T))))
out.append(("relative error of odds e^{d/T}", abs(np.exp(indep - self_cost) - np.exp(dd / T)) / np.exp(dd / T)))

# (6) eq:sup-integrated-score and eq:sup-gaussian-jensen
for dd, singular in ((1, False), (2, False), (2, True)):
    a = rng.normal(size=dd)
    b = rng.normal(size=dd)
    V = spd(dd)
    R = spd(dd, 0.3)
    if singular:
        u = rng.normal(size=(dd, 1))
        W = u @ u.T
    else:
        W = spd(dd)
    Ek = c_R(R) + 0.5 * ((a - b) @ np.linalg.solve(R, a - b) + np.trace(np.linalg.solve(R, V + W)))
    Er = c_R(W + R) + 0.5 * ((a - b) @ np.linalg.solve(W + R, a - b) + np.trace(np.linalg.solve(W + R, V)))
    Er_q = gauss_expect(lambda x: neglog_normal(x, b, W + R), a, V, npts=4)
    M = np.linalg.inv(R) - np.linalg.inv(W + R)
    G = (0.5 * np.log(np.linalg.det(R) / np.linalg.det(W + R)) + 0.5 * np.trace(np.linalg.solve(R, W))
         + 0.5 * (a - b) @ M @ (a - b) + 0.5 * np.trace(M @ V))
    Lr = np.linalg.cholesky(np.linalg.inv(R))
    lam = np.linalg.eigvalsh(Lr.T @ W @ Lr)
    fl = 0.5 * np.sum(lam - np.log1p(lam))
    out.append((f"eq:sup-integrated-score quadrature d={dd} singularW={singular}", abs(Er_q - Er)))
    out.append((f"eq:sup-gaussian-jensen residual d={dd} singularW={singular}", abs((Ek - Er) - G)))
    out.append((f"first line = 1/2 sum(lam-log(1+lam)) d={dd} singularW={singular}",
                abs(fl - (0.5 * np.log(np.linalg.det(R) / np.linalg.det(W + R)) + 0.5 * np.trace(np.linalg.solve(R, W))))))
    out.append((f"min eig R^-1-(W+R)^-1 (should be >=0) d={dd} singularW={singular}", np.linalg.eigvalsh(M).min()))
    out.append((f"G value (should be >0) d={dd} singularW={singular}", G))
    if not singular:
        Sp = np.linalg.inv(np.linalg.inv(W) + np.linalg.inv(R))

        def klpost(x):
            mpost = Sp @ (np.linalg.solve(W, b) + np.linalg.solve(R, x))
            return kl_gauss(b, W, mpost, Sp)

        Gpost = gauss_expect(klpost, a, V, npts=4)
        out.append((f"G - E_X KL(p||p^X) [main text prop:dyn-jensen] d={dd}", abs(Gpost - G)))

for name, val in out:
    print(f"{name:72s} {val}")
