r"""D04-DYN s6: every formula of supplements/s03_variation.tex.

Gaussian expectations by tensor Gauss-Hermite quadrature (exact for the quadratic integrands), plus a
non-Gaussian discrete joint for prop:sup-residual-cost (which needs only second moments).
"""
import itertools
import numpy as np

rng = np.random.default_rng(9)


def gh_expect(f, mean, cov, npts=6):
    x, w = np.polynomial.hermite_e.hermegauss(npts)
    w = w / w.sum()
    n = len(mean)
    L = np.linalg.cholesky(cov + 1e-300 * np.eye(n))
    tot = 0.0
    for idx in itertools.product(range(npts), repeat=n):
        z = mean + L @ x[list(idx)]
        tot += np.prod(w[list(idx)]) * f(z)
    return tot


def spd(n, s=0.5):
    A = rng.normal(size=(n, n))
    return A @ A.T + s * np.eye(n)


def cR(R):
    return 0.5 * np.log(np.linalg.det(2 * np.pi * R))


def nll(o, mean_res, R):
    d = o - mean_res
    return cR(R) + 0.5 * d @ np.linalg.solve(R, d)


ni, nj, d = 2, 2, 3
A = rng.normal(size=(d, ni))
B = rng.normal(size=(d, nj))
R = spd(d)
o = rng.normal(size=d)
C = spd(ni + nj)
mu = rng.normal(size=ni + nj)
Sii, Sjj, Sij = C[:ni, :ni], C[ni:, ni:], C[:ni, ni:]
delta = A @ mu[:ni] - B @ mu[ni:] - o
Vij = A @ Sii @ A.T + B @ Sjj @ B.T - A @ Sij @ B.T - B @ Sij.T @ A.T
formula = cR(R) + 0.5 * delta @ np.linalg.solve(R, delta) + 0.5 * np.trace(np.linalg.solve(R, Vij))
quadv = gh_expect(lambda z: nll(o, A @ z[:ni] - B @ z[ni:], R), mu, C)
print(f"prop:sup-residual-cost (Gaussian joint, d=3): formula {formula:.10f}, quadrature {quadv:.10f}, diff {abs(formula - quadv):.1e}")
pts = rng.normal(size=(7, ni + nj)) * rng.uniform(0.5, 2, size=(7, 1))
wts = rng.dirichlet(np.ones(7))
m2 = wts @ pts
C2 = (pts - m2).T @ ((pts - m2) * wts[:, None])
delta2 = A @ m2[:ni] - B @ m2[ni:] - o
V2 = A @ C2[:ni, :ni] @ A.T + B @ C2[ni:, ni:] @ B.T - A @ C2[:ni, ni:] @ B.T - B @ C2[ni:, :ni] @ A.T
f2 = cR(R) + 0.5 * delta2 @ np.linalg.solve(R, delta2) + 0.5 * np.trace(np.linalg.solve(R, V2))
e2 = sum(wts[k] * nll(o, A @ pts[k, :ni] - B @ pts[k, ni:], R) for k in range(7))
print(f"  non-Gaussian 7-point joint with the same formula: diff {abs(f2 - e2):.1e}")

dd = 2
a = rng.normal(size=dd)
V = spd(dd)
W = spd(dd)
b = rng.normal(size=dd)


def logN(x, m, S):
    r = x - m
    return -0.5 * (dd * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + r @ np.linalg.solve(S, r))


Dj = 0.5 * ((a - b) @ np.linalg.solve(W, a - b) + np.trace(np.linalg.solve(W, V)) - dd + np.log(np.linalg.det(W) / np.linalg.det(V)))
Dq = gh_expect(lambda x: logN(x, a, V) - logN(x, b, W), a, V)
print(f"eq:sup-peer-score: closed form {Dj:.10f} vs quadrature of the log ratio {Dq:.10f}, diff {abs(Dj - Dq):.1e}")
recq = gh_expect(lambda x: (x - a) @ np.linalg.solve(V, x - a), a, V)
srcq = gh_expect(lambda x: (x - b) @ np.linalg.solve(W, x - b), a, V)
print(f"  receiver quadratic expectation {recq:.10f} (d = {dd}); source quadratic {srcq:.10f} vs tr(W^-1V)+(a-b)'W^-1(a-b) = {np.trace(np.linalg.solve(W, V)) + (a - b) @ np.linalg.solve(W, a - b):.10f}")

T = 0.7
Rt = T * W


def Cj(a, bj, V):
    return cR(Rt) + 0.5 * ((a - bj) @ np.linalg.solve(Rt, a - bj) + np.trace(np.linalg.solve(Rt, V + W)))


def Djf(a, bj, V):
    return 0.5 * ((a - bj) @ np.linalg.solve(W, a - bj) + np.trace(np.linalg.solve(W, V)) - dd + np.log(np.linalg.det(W) / np.linalg.det(V)))


const = cR(Rt) + dd / T - 1 / (2 * T) * np.log(np.linalg.det(W) / np.linalg.det(V))
diffs = [Cj(a, bj, V) - Djf(a, bj, V) / T - const for bj in rng.normal(size=(5, dd)) * 3]
print(f"prop:sup-row-equivalence: max abs(C_j - D_j/T - eq:sup-row-constant) over 5 sources = {max(abs(np.array(diffs))):.1e}")
dV = rng.normal(size=(dd, dd))
dV = dV + dV.T
h = 1e-6
bj = rng.normal(size=dd)
lhs = ((Cj(a, bj, V + h * dV) - Djf(a, bj, V + h * dV) / T) - (Cj(a, bj, V - h * dV) - Djf(a, bj, V - h * dV) / T)) / (2 * h)
print(f"  eq:sup-covariance-difference: FD {lhs:.10f} vs tr(V^-1 dV)/(2T) = {np.trace(np.linalg.solve(V, dV)) / (2 * T):.10f}")
da = rng.normal(size=dd)
g_a = ((Cj(a + h * da, bj, V) - Djf(a + h * da, bj, V) / T) - (Cj(a - h * da, bj, V) - Djf(a - h * da, bj, V) / T)) / (2 * h)
print(f"  receiver-mean derivative of C_j - D_j/T (any V): {g_a:.1e}")
Rt = 1.0 * W
H = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * V))
print(f"  T = 1: C_j - D_j = {Cj(a, bj, V) - Djf(a, bj, V):.10f} vs H(N(a,V)) + d/2 = {H + dd / 2:.10f}")

Rs = 0.4 * np.eye(dd)
oo = rng.normal(size=dd)
self_cost = gh_expect(lambda x: nll(oo, x - x, Rs), a, V)
print(f"eq:sup-self-cost: {self_cost:.10f} vs c_R + o'R^-1 o/2 = {cR(Rs) + 0.5 * oo @ np.linalg.solve(Rs, oo):.10f}")
Ts_ = 0.4
z0 = np.zeros(dd)
snap = gh_expect(lambda z: nll(z0, z[:dd] - z[dd:], Ts_ * np.eye(dd)), np.zeros(2 * dd), np.eye(2 * dd))
print(f"independent snapshots: cost {snap:.10f} vs c_R + d/T = {cR(Ts_ * np.eye(dd)) + dd / Ts_:.10f}; odds exp(d/T) = {np.exp(dd / Ts_):.6f} vs exp(cost difference) = {np.exp(snap - cR(Ts_ * np.eye(dd))):.6f}")

for Wc, label in ((spd(dd), "W positive definite"), (np.outer(rng.normal(size=dd), rng.normal(size=dd)) * 0 + np.outer(*(2 * [rng.normal(size=dd)])), "W rank one"), (np.zeros((dd, dd)), "W = 0")):
    Rr = spd(dd)
    integ = cR(Wc + Rr) + 0.5 * ((a - b) @ np.linalg.solve(Wc + Rr, a - b) + np.trace(np.linalg.solve(Wc + Rr, V)))
    integ_q = gh_expect(lambda x: -logN(x, b, Wc + Rr), a, V)
    Mexp = cR(Rr) + 0.5 * ((a - b) @ np.linalg.solve(Rr, a - b) + np.trace(np.linalg.solve(Rr, V + Wc)))
    Gcl = (0.5 * np.log(np.linalg.det(Rr) / np.linalg.det(Wc + Rr)) + 0.5 * np.trace(np.linalg.solve(Rr, Wc))
           + 0.5 * (a - b) @ (np.linalg.inv(Rr) - np.linalg.inv(Wc + Rr)) @ (a - b)
           + 0.5 * np.trace((np.linalg.inv(Rr) - np.linalg.inv(Wc + Rr)) @ V))
    ev, U = np.linalg.eigh(Rr)
    Rmh = U @ np.diag(ev ** -0.5) @ U.T
    lam = np.linalg.eigvalsh(Rmh @ Wc @ Rmh)
    first = 0.5 * np.sum(lam - np.log1p(lam))
    psd = np.linalg.eigvalsh(np.linalg.inv(Rr) - np.linalg.inv(Wc + Rr)).min()
    print(f"{label}: eq:sup-integrated-score diff {abs(integ - integ_q):.1e}; Jensen gap closed form {Gcl:.10f} vs Mexp - integrated {Mexp - integ:.10f}; first line vs eigen form {abs(0.5*np.log(np.linalg.det(Rr)/np.linalg.det(Wc+Rr)) + 0.5*np.trace(np.linalg.solve(Rr, Wc)) - first):.1e}; min eig of R^-1-(W+R)^-1 = {psd:.2e}")
