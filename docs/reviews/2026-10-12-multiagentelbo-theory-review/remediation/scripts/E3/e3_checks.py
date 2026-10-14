"""E3 checks: L1-1 marked sweep, L1-3 slow-rate bound, L11-3 simultaneous-update remark."""
import numpy as np
from math import log, exp, sqrt
from scipy.linalg import expm
rng = np.random.default_rng(0)

def kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))

# ---------- L1-1 ----------
print("[L1-1] marked sweep on q (x) psi'")
worst_gap = 0.0; viol = 0
for trial in range(2000):
    n = rng.integers(2, 6); nr = rng.integers(2, 4)
    p0 = rng.random(n) + 0.1; L = rng.random(n) + 0.1
    p = p0 * L; p /= p.sum()
    par = rng.integers(0, max(1, n // 2), size=n)
    a = rng.random((n, n)) * (rng.random((n, n)) < 0.6); a = (a + a.T) / 2; np.fill_diagonal(a, 0)
    rs = a.sum(1).max(); a = a / max(2 * rs, 1e-12) if rs > 0 else a
    M = a * np.minimum(1, p[None, :] / p[:, None]); np.fill_diagonal(M, 0); np.fill_diagonal(M, 1 - M.sum(1))
    zs = np.unique(par); C = np.array([[1.0 if par[x] == z else 0 for z in zs] for x in range(n)])
    pZ = p @ C; J = np.array([[p[x] * (par[x] == z) / pZ[k] for x in range(n)] for k, z in enumerate(zs)])
    T = M @ C @ J
    psi = rng.dirichlet(np.ones(nr), size=n); psip = rng.dirichlet(np.ones(nr), size=n)
    q = rng.dirichlet(np.ones(n))
    if trial % 3 == 0: psip = psi.copy()
    F0 = kl(q, p) + sum(q[x] * kl(psip[x], psi[x]) for x in range(n))
    # marked kernel acting on q (x) psi'
    joint = (q[:, None] * psip).reshape(-1)
    Tm = np.kron(T, np.ones((nr, 1))).reshape(n * nr, n)  # rows (x,r) -> x'
    Tm = np.einsum('ab,bc->abc', Tm, psi).reshape(n * nr, n * nr)
    out = joint @ Tm
    qT = q @ T
    assert np.allclose(out.reshape(n, nr), qT[:, None] * psi)
    F1 = kl(qT, p)
    dec = (kl(q, p) - kl(qT, p)) + sum(q[x] * kl(psip[x], psi[x]) for x in range(n))
    worst_gap = max(worst_gap, abs((F0 - F1) - dec))
    if F1 > F0 + 1e-12: viol += 1
print("  identity residual", worst_gap, " increases:", viol)
# witness of reviewer [1a]: unmarked T on q with psi' held
p = np.array([.5, .5]); a01 = .5
M = np.array([[1 - a01, a01], [a01, 1 - a01]]); T = M  # singleton fibers, D = I
psi1 = np.array([.999, .001]); psip1 = np.array([.001, .999])
q = np.array([1., 0.]); qT = q @ T
Fb = kl(q, p) + q[1] * kl(psip1, psi1); Fa = kl(qT, p) + qT[1] * kl(psip1, psi1)
print("  unmarked witness F before/after", round(Fb, 4), round(Fa, 4))

# ---------- L1-3 ----------
print("[L1-3] slow-rate bracket and |r^m| <= (1+4(em+ew)) e^{-(4em+ew)t}")
worst = -1e9; worstb = -1e9
s2 = sqrt(2)
for em in np.linspace(0, .5, 26):
    for ew in np.linspace(0, .5, 26):
        R = sqrt((1 - em) ** 2 + ew ** 2); sm = 2 * (1 + em + ew) - 2 * R
        lo, hi = 4 * em + 2 * ew - 2 * ew ** 2, 4 * em + 2 * ew
        worstb = max(worstb, lo - sm, sm - hi)
        H = np.array([[4, 0, -2 * s2], [0, 4, -2 * s2], [-2 * s2, -2 * s2, 4]])
        A = -np.diag([1, em, ew]) @ H
        # start with residuals -1: rb = s2 k - w, rm = s2 m - w ; choose k=m=-1/s2, w=0
        x0 = np.array([-1 / s2, -1 / s2, 0.])
        for t in np.linspace(0, 200, 401):
            x = expm(A * t) @ x0; rm = s2 * x[1] - x[2]
            worst = max(worst, abs(rm) - (1 + 4 * (em + ew)) * exp(-(4 * em + ew) * t))
print("  bracket violation (<=0 ok):", worstb, " bound violation (<=0 ok):", worst)
em, ew = 1e-4, 1e-2
R = sqrt((1 - em) ** 2 + ew ** 2); print("  sigma_- at (1e-4,1e-2):", 2 * (1 + em + ew) - 2 * R)

# ---------- L11-3 ----------
print("[L11-3] Gaussian three-agent witness")
Lam = 0.4 * np.eye(3) + 0.6 * np.ones((3, 3))
print("  Lam eigenvalues", np.linalg.eigvalsh(Lam), " det", np.linalg.det(Lam), "44/125=", 44 / 125)
def KLprod(e, s2v):
    return 0.5 * (np.sum(np.diag(Lam) * s2v) + e @ Lam @ e - 3 - np.log(np.linalg.det(Lam)) - np.sum(np.log(s2v)))
e = np.ones(3); s = np.ones(3)
def jacobi(e, w=1.0):
    br = -(Lam - np.diag(np.diag(Lam))) @ e / np.diag(Lam)
    return (1 - w) * e + w * br
def gs(e):
    e = e.copy()
    for i in range(3):
        e[i] = -(Lam[i] @ e - Lam[i, i] * e[i]) / Lam[i, i]
    return e
print("  KL before", KLprod(e, s), "=", 3.3 + 0.5 * log(125 / 44))
e1 = jacobi(e); print("  after simultaneous best responses", e1, KLprod(e1, s), "=", 594 / 125 + 0.5 * log(125 / 44))
ej, eg = e.copy(), e.copy(); seqJ, seqG = [], []
for k in range(6):
    ej = jacobi(ej); eg = gs(eg); seqJ.append(KLprod(ej, s)); seqG.append(KLprod(eg, s))
print("  Jacobi", np.round(seqJ, 4)); print("  Gauss-Seidel", np.round(seqG, 4))
# Euler step equivalence: grad wrt m of KL = Lam e ; Fisher of N(m,1) in m is 1
for w in [0.3, 0.9, 10 / 11, 0.95]:
    print("  w", round(w, 4), "damped simultaneous factor on 1:", (jacobi(e, w) / e)[0], " Euler:", (e - w * Lam @ e)[0], " KL", KLprod(jacobi(e, w), s))

print("[L11-3] Poisson: natural damping monotone, chord fails; mean damping chord holds")
def klpois(lam, lam0):  # KL(Poi(lam)||Poi(lam0))
    return lam * log(lam / lam0) - lam + lam0
L4 = 4.0
f = lambda t: klpois(exp(-t * L4), 1.0)
print("  f(1)=", f(1), "=1-5e^-4", 1 - 5 * exp(-4), " f(1/2)=", f(.5), "=1-3e^-2", 1 - 3 * exp(-2), " (1/2)f(1)=", f(1) / 2)
ts = np.linspace(0, 1, 1001); fv = [f(t) for t in ts]
print("  natural segment monotone:", bool(np.all(np.diff(fv) >= -1e-15)))
worst = -1e9
for _ in range(20000):
    mu, mus, w = rng.random() * 10 + 1e-3, rng.random() * 10 + 1e-3, rng.random()
    worst = max(worst, klpois((1 - w) * mu + w * mus, mus) - (1 - w) * klpois(mu, mus))
print("  Poisson mean-damping chord violation (<=0 ok):", worst)
# Gaussian full family N(m, v): mean damping in (m, m^2+v); natural damping (m/v, -1/(2v))
def klg(m1, v1, m0, v0):
    return 0.5 * (v1 / v0 + (m1 - m0) ** 2 / v0 - 1 - log(v1 / v0))
worst_mean = -1e9; worst_nat_mono = -1e9; chord_nat_fail = 0
for _ in range(20000):
    m, v, ms, vs, w = rng.normal() * 3, rng.random() * 5 + .01, rng.normal() * 3, rng.random() * 5 + .01, rng.random()
    M2, M2s = m * m + v, ms * ms + vs
    mw = (1 - w) * m + w * ms; M2w = (1 - w) * M2 + w * M2s; vw = M2w - mw * mw
    worst_mean = max(worst_mean, klg(mw, vw, ms, vs) - (1 - w) * klg(m, v, ms, vs))
    e1, e2 = m / v, -1 / (2 * v); s1, s2_ = ms / vs, -1 / (2 * vs)
    n1, n2 = (1 - w) * e1 + w * s1, (1 - w) * e2 + w * s2_
    vn = -1 / (2 * n2); mn = n1 * vn
    k_n = klg(mn, vn, ms, vs)
    worst_nat_mono = max(worst_nat_mono, k_n - klg(m, v, ms, vs))
    if k_n > (1 - w) * klg(m, v, ms, vs) + 1e-12: chord_nat_fail += 1
print("  Gaussian mean-damping chord violation (<=0 ok):", worst_mean)
print("  Gaussian natural-damping increase (<=0 ok):", worst_nat_mono, " natural chord failures:", chord_nat_fail)
