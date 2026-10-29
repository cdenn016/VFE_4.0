r"""D15-WISH s5: remaining identities and one counterexample search.

(a) Covariance of same-batch block sums: Cov(Tr(U X_C), Tr(V X_D)) = A_{C cap D} Tr(P^{-1} U P^{-1} V),
    exactly from the cumulant generating function K(T) = nu[log det P - log det(P+T)] (sympy),
    and by Monte Carlo with Bartlett sampling.
(b) form:coupling: DC_q[H] = (2 kappa/N) H + (kappa^2 alpha0/N) Lambda^{-1} H Lambda^{-1}.
(c) Eigenvalue barriers in the proof of form:full-flow on random parameters.
(d) Energy non-factorization example after form:parent (P = I, kappa > 0).
(e) form:approximate-open: second-order cancellation of the log-Laplace defect.
(f) Counterexample search for the sentence after form:mean-field: a relabeling-equivariant
    selector reading the agent costs D_i = KL(q_i || nu[M_q]) along the FULL flow, kappa 0 vs 5.
"""
import itertools
import numpy as np
import sympy as sp
from scipy.special import gammaln, digamma
from scipy.integrate import solve_ivp

rng = np.random.default_rng(5)

nu, t1, t2 = sp.symbols("nu t1 t2", positive=True)
Pm = sp.Matrix([[sp.Rational(13, 10), sp.Rational(1, 5)], [sp.Rational(1, 5), sp.Rational(9, 10)]])
Um = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), -1]])
Vm = sp.Matrix([[2, 0], [0, 1]])
K = nu * (sp.log(Pm.det()) - sp.log((Pm + t1 * Um + t2 * Vm).det()))
cov = sp.diff(K, t1, t2).subs({t1: 0, t2: 0})
print("(a) CGF covariance - nu Tr(P^-1 U P^-1 V) =", sp.simplify(cov - nu * (Pm.inv() * Um * Pm.inv() * Vm).trace()))


def sample_wishart(a, P, n):
    L = np.linalg.cholesky(np.linalg.inv(2 * P))
    c1 = np.sqrt(rng.chisquare(2 * a, n))
    c2 = np.sqrt(rng.chisquare(2 * a - 1, n))
    g = rng.standard_normal(n)
    Aa = np.zeros((n, 2, 2))
    Aa[:, 0, 0], Aa[:, 1, 0], Aa[:, 1, 1] = c1, g, c2
    LA = L @ Aa
    return LA @ np.transpose(LA, (0, 2, 1))


P = np.array([[1.3, 0.2], [0.2, 0.9]])
U = np.array([[1, 0.5], [0.5, -1.0]])
V = np.array([[2.0, 0], [0, 1.0]])
shapes = [1.0, 1.2, 1.8, 2.1]
n = 300000
Xs = [sample_wishart(a, P, n) for a in shapes]
XC, XD = sum(Xs[i] for i in [0, 1, 2]), sum(Xs[i] for i in [1, 2, 3])
lhs = np.cov(np.einsum("ij,nji->n", U, XC), np.einsum("ij,nji->n", V, XD))[0, 1]
Pi = np.linalg.inv(P)
print(f"    Monte Carlo cov = {lhs:.4f}, formula = {(shapes[1] + shapes[2]) * np.trace(Pi @ U @ Pi @ V):.4f} (n = {n})")

N, kap, a0 = 4, 1.3, 1.1
M = np.array([[2.0, 0.4], [0.4, 1.5]])
H = np.array([[0.3, -0.2], [-0.2, 0.7]])


def Cq(M):
    Lam = np.eye(2) + kap / N * M
    return np.eye(2) + 2 * kap / N * M - kap * a0 * np.linalg.inv(Lam)


e = 1e-6
num = (Cq(M + e * H) - Cq(M - e * H)) / (2 * e)
Li = np.linalg.inv(np.eye(2) + kap / N * M)
form = 2 * kap / N * H + kap**2 * a0 / N * Li @ H @ Li
print(f"(b) DC_q[H] residual {np.max(np.abs(num - form)):.1e}; Tr(H DC_q[H]) = {np.trace(H @ form):.4f}")


def f(p, ab, kap, a0):
    return -2 * p + 2 + 2 * kap * ab / p - kap * a0 * p / (p + kap * ab)


worst_lo, worst_hi = np.inf, -np.inf
for _ in range(20000):
    kp, al0, ab0 = rng.uniform(0, 10), rng.uniform(0.51, 5), rng.uniform(0.51, 5)
    ab = rng.uniform(min(ab0, al0), max(ab0, al0))
    eps = min(0.5, min(ab0, al0) / al0)
    L = 2 + kp * max(ab0, al0)
    worst_lo = min(worst_lo, f(eps * rng.uniform(1e-6, 1), ab, kp, al0))
    worst_hi = max(worst_hi, f(L * rng.uniform(1, 50), ab, kp, al0))
print(f"(c) min f on (0, eps] = {worst_lo:.4f} (claimed > 0); max f on [L, inf) = {worst_hi:.4f} (claimed <= -2)")


def lg2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl(a, P, b, Q):
    return ((a - b) * psi2(a) - lg2(a) + lg2(b) + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2.0))


def fine_energy(al, P, a0, kap):
    M = sum(a * np.linalg.inv(P) for a in al)
    Lam = np.eye(2) + kap / len(al) * M
    return sum(kl(a, P, a0, np.eye(2)) + kl(a, P, a0, Lam) for a in al)


a0, kap, epsl = 1.0, 1.0, 0.15
dE = fine_energy([a0 + epsl, a0 - epsl, 1.7, 2.0], np.eye(2), a0, kap) - fine_energy([a0, a0, 1.7, 2.0], np.eye(2), a0, kap)
claim = 2 * kl(a0 + epsl, np.eye(2), a0, np.eye(2)) + 2 * kl(a0 - epsl, np.eye(2), a0, np.eye(2))
print(f"(d) fine energy increase {dE:.12f}; claimed {claim:.12f}; difference {dE - claim:.1e}")

alphas = np.array([1.0, 1.2, 1.8, 2.1])
Pbar_inv = np.array([[0.8, 0.1], [0.1, 1.2]])
Fr = rng.normal(size=(4, 2, 2))
Fr = 0.5 * (Fr + np.transpose(Fr, (0, 2, 1)))
Fr -= (alphas[:, None, None] * Fr).sum(0) / alphas.sum()
T = np.array([[0.7, -0.2], [-0.2, 0.5]])
prev = None
for epsv in [0.2, 0.1, 0.05, 0.025]:
    Pinv = Pbar_inv + epsv * Fr
    sum_law = -sum(a * np.log(np.linalg.det(np.eye(2) + Pi_ @ T)) for a, Pi_ in zip(alphas, Pinv))
    wish = -alphas.sum() * np.log(np.linalg.det(np.eye(2) + Pbar_inv @ T))
    d = abs(sum_law - wish)
    print(f"(e) eps={epsv:.3f}: log-Laplace defect {d:.3e}" + ("" if prev is None else f", ratio {prev/d:.3f}"))
    prev = d


def full_P(al0, a0, kap, ts):
    Nn = len(al0)

    def rhs(t, y):
        P = np.array([[y[0], y[1]], [y[1], y[2]]])
        A = np.sum(a0 + np.exp(-2 * t) * (al0 - a0))
        M = A * np.linalg.inv(P)
        Lam = np.eye(2) + kap / Nn * M
        Cc = np.eye(2) + 2 * kap / Nn * M - kap * a0 * np.linalg.inv(Lam)
        dP = -2 * P + np.eye(2) + Cc
        return [dP[0, 0], dP[0, 1], dP[1, 1]]
    return solve_ivp(rhs, (0, ts[-1]), [1.0, 0.0, 1.0], t_eval=ts, rtol=1e-10, atol=1e-12).y


def costs(al0, a0, kap, ts):
    Nn = len(al0)
    Ps = full_P(al0, a0, kap, ts)
    out = []
    for k, t in enumerate(ts):
        al = a0 + np.exp(-2 * t) * (al0 - a0)
        P = np.array([[Ps[0, k], Ps[1, k]], [Ps[1, k], Ps[2, k]]])
        Lam = np.eye(2) + kap / Nn * al.sum() * np.linalg.inv(P)
        out.append([kl(a, P, a0, Lam) for a in al])
    return np.array(out)


def first_pair(Dt, delta):
    for k in range(1, len(Dt)):
        hits = [(i, j) for i, j in itertools.combinations(range(Dt.shape[1]), 2)
                if abs(Dt[k - 1, i] - Dt[k - 1, j]) > delta >= abs(Dt[k, i] - Dt[k, j])]
        if hits:
            return hits[0] if len(hits) == 1 else "tie"
    return None


ts = np.linspace(0, 4, 2001)
found = 0
for trial in range(150):
    al0 = np.sort(rng.uniform(0.6, 3.0, 3))
    D0, D5 = costs(al0, 1.0, 0.0, ts), costs(al0, 1.0, 5.0, ts)
    dmin = min(min(abs(D0[0, i] - D0[0, j]), abs(D5[0, i] - D5[0, j])) for i, j in itertools.combinations(range(3), 2))
    delta = 0.5 * dmin
    f0, f5 = first_pair(D0, delta), first_pair(D5, delta)
    if f0 not in (None, "tie") and f5 not in (None, "tie") and f0 != f5:
        found += 1
        if found <= 3:
            print(f"(f) shapes {np.round(al0, 4)}, delta {delta:.4f}: first cost-difference join kappa=0 -> {f0}, kappa=5 -> {f5}")
print(f"    configurations (of 150, N = 3) whose first cost-difference join depends on kappa: {found}")
