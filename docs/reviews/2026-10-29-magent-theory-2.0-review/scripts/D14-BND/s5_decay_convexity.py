r"""D14-BND s5: prop:boundary-contrast-decay, convexity of D_k, lem:boundary-global, long-time behavior.

(a) Walsh rates lambda_m = 3 + 2 b_{m-1} - 2 sum_{k>=m} b_k for random rows versus the spectrum of
    3 - 2 sum_k b_k P_k on mean-zero functions; the common rate 3 + 2/(N-1) at the class priors.
(b) Nonlinear decay rate of a small contrast under the full coupled profile flow (d = 3).
(c) Convexity of D_k in f: second central differences along random directions (min over trials).
(d) Long-time behavior on a finite base from large-contrast initial data (finite-base LaSalle claim):
    contrast sup|log f - log f_0| at t = 0, 5, 10, 20 and monotone energy; the bounds of
    eq:boundary-global-bound along the way.
"""
import numpy as np

from bnd_common import (P_matrix, ProfileData, profile_velocity, profile_energy, profile_D, rk4,
                        normalize_profile, all_bits)

rng = np.random.default_rng(1618)

# ----------------------------------------------------------------------------- (a)
print("(a) Walsh rates vs spectrum of 3 - 2 sum_k b_k P_k on mean-zero profiles")
for d in (2, 3, 4):
    N = 2 ** d
    P = [P_matrix(k, d) for k in range(d)]
    U = np.array(all_bits(d), dtype=float)
    for trial, b in enumerate([np.array([2 ** (d - k - 1) / (N - 1) for k in range(d)]), rng.dirichlet(np.ones(d))]):
        L = 3 * np.eye(N) - 2 * sum(bk * Pk for bk, Pk in zip(b, P))
        worst = 0.0
        for S in range(1, N):
            Sset = [j for j in range(d) if (S >> (d - 1 - j)) & 1]
            chi = np.prod(U[:, Sset], axis=1)
            m = max(Sset) + 1
            lam = 3 + 2 * b[m - 1] - 2 * sum(b[m:])
            worst = max(worst, np.max(np.abs(L @ chi - lam * chi)))
        ev = np.sort(np.linalg.eigvals(L).real)
        tag = "class priors" if trial == 0 else "random rows "
        print(f"   d={d} {tag}: Walsh eigen-residual {worst:.1e}; eigenvalues {np.round(ev, 6)}"
              + (f"; 3+2/(N-1) = {3 + 2/(N-1):.6f}" if trial == 0 else f"; min >= 1: {ev.min() >= 1 - 1e-12}"))

# ----------------------------------------------------------------------------- (b)
d = 3; N = 8
A = np.array([0.3, 0.7]); B = np.array([0.5, 0.5])
logrho = np.array([[0.4, -0.3], [-0.2, 0.25]])
logrho -= np.log(np.sum(A[:, None] * B[None, :] * np.exp(logrho)))
pik = np.array([2 ** (d - k - 1) / (N - 1) for k in range(d)])
PD = ProfileData(d, A, B, logrho, 2.0, 2.0, pik, [P_matrix(k, d) for k in range(d)])
f0 = normalize_profile(np.ones((2, N)) * np.array([[1.2], [0.9]]), A)
g0 = normalize_profile(np.ones((2, N)), B)
phi = 1e-4 * rng.normal(size=(2, N)); phi -= phi.mean(axis=1, keepdims=True)
f = f0 * (1 + phi)
g = g0 * (1 + 1e-4 * (lambda z: z - z.mean(axis=1, keepdims=True))(rng.normal(size=(2, N))))
x = (f, g, pik.copy(), pik.copy())
def contrast(f):
    return np.max(np.abs(np.log(f) - np.log(f.mean(axis=1, keepdims=True))))
c_prev = contrast(x[0]); rates = []
for step in range(4):
    x = rk4(profile_velocity, x, 0.005, 100, PD)
    c = contrast(x[0]); rates.append(np.log(c_prev / c) / 0.5); c_prev = c
print(f"(b) nonlinear contrast decay rate (d=3, rows at class priors): {np.round(rates, 5)}; predicted 3+2/7 = {3+2/7:.5f}")

# ----------------------------------------------------------------------------- (c)
print("(c) convexity of D_k along random directions (min second central difference / eps^2)")
for d in (2, 3):
    Ab = rng.dirichlet(np.ones(3))
    PDd = ProfileData(d, Ab, Ab, np.zeros((3, 3)), 1.0, 1.0, np.ones(d) / d, [P_matrix(k, d) for k in range(d)])
    mins = np.full(d, np.inf)
    for trial in range(300):
        f = np.exp(1.0 * rng.normal(size=(3, 2 ** d)))
        v = rng.normal(size=f.shape) * f * 0.5
        eps = 1e-3
        D0 = profile_D(f, PDd); Dp = profile_D(f + eps * v, PDd); Dm = profile_D(f - eps * v, PDd)
        mins = np.minimum(mins, (Dp - 2 * D0 + Dm) / eps ** 2)
    print(f"   d={d}: min second differences per k = {mins}")

# ----------------------------------------------------------------------------- (d)
print("(d) long-time behavior on a finite base, large-contrast initial data (full coupled profile flow)")
for d, seed in [(2, 1), (3, 2), (3, 3)]:
    r2 = np.random.default_rng(seed)
    N = 2 ** d; R = 3; W = 2
    A = r2.dirichlet(np.ones(R) * 2); B = r2.dirichlet(np.ones(W) * 2)
    logrho = 1.5 * r2.normal(size=(R, W)); logrho -= np.log(np.sum(A[:, None] * B[None, :] * np.exp(logrho)))
    pik = np.array([2 ** (d - k - 1) / (N - 1) for k in range(d)])
    PD = ProfileData(d, A, B, logrho, 0.3, 0.5, pik, [P_matrix(k, d) for k in range(d)])
    f = normalize_profile(np.exp(2.0 * r2.normal(size=(R, N))), A)
    g = normalize_profile(np.exp(2.0 * r2.normal(size=(W, N))), B)
    bq = r2.dirichlet(np.ones(d) * 0.5); bs = r2.dirichlet(np.ones(d) * 0.5)
    x = (f, g, bq, bs)
    F0 = profile_energy(*x, PD)
    K = F0 + 3 * np.max(np.abs(logrho)) + 2
    lo = min(np.log(f).min(), -K); hi = max(np.log(f).max(), K)
    report = []; Fs = [F0]; viol = 0
    for seg in range(20):
        x = rk4(profile_velocity, x, 0.005, 200, PD)
        Fs.append(profile_energy(*x, PD))
        lf = np.log(x[0])
        viol += int(lf.min() < lo - 1e-9 or lf.max() > hi + 1e-9)
        if seg + 1 in (5, 10, 20):
            report.append((seg + 1, contrast(x[0]), contrast(x[1]), np.max(np.abs(x[2] - pik))))
    mono = np.all(np.diff(Fs) <= 1e-12)
    print(f"   d={d} seed={seed}: contrast(0)={contrast(f):.3f}; " +
          "; ".join(f"t={t}: belief {cq:.2e}, model {cs:.2e}, |b^q - pi| {db:.2e}" for t, cq, cs, db in report) +
          f"; F monotone: {mono}; global-bound violations: {viol}")
