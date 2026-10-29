r"""D14-BND s10.

(a) D_k(f) equals the Dirichlet form of the reversible kernel P_k at (f, log f):
    E_{P_k}(f, log f) = (1/2) int int (f(u) - f(v)) (log f(u) - log f(v)) U(du) P_k(u, dv),
    the entropy-production functional of modified log-Sobolev inequalities.
(b) Null flow (peer costs deleted): linearized contrast rate. The u-dependent part of the
    null-flow force is phi, so every Walsh mode decays at rate 1; compare 3 + 2/(N-1) with peers.
"""
import numpy as np

from bnd_common import P_matrix, ProfileData, profile_velocity, rk4, normalize_profile, profile_D

rng = np.random.default_rng(5)
print("(a) D_k versus the symmetric Dirichlet form")
for d in (2, 3, 4):
    A = rng.dirichlet(np.ones(3))
    f = normalize_profile(np.exp(0.8 * rng.normal(size=(3, 2 ** d))), A)
    PD = ProfileData(d, A, A, np.zeros((3, 3)), 1.0, 1.0, np.ones(d) / d, [P_matrix(k, d) for k in range(d)])
    Dk = profile_D(f, PD)
    res = []
    for k in range(d):
        P = P_matrix(k, d)
        lf = np.log(f)
        dir_form = 0.5 * sum(A[r] * np.sum((1.0 / 2 ** d) * P * (f[r][:, None] - f[r][None, :]) * (lf[r][:, None] - lf[r][None, :]))
                             for r in range(3))
        res.append(abs(dir_form - Dk[k]))
    print(f"   d={d}: max |D_k - E_Pk(f, log f)| = {max(res):.2e}")

print("(b) null-flow contrast decay rate vs full flow (d = 3, rows at class priors)")
d = 3; N = 8
A = np.array([0.3, 0.7]); B = np.array([0.5, 0.5])
logrho = np.array([[0.4, -0.3], [-0.2, 0.25]])
logrho -= np.log(np.sum(A[:, None] * B[None, :] * np.exp(logrho)))
pik = np.array([2 ** (d - k - 1) / (N - 1) for k in range(d)])
for peer in (False, True):
    PD = ProfileData(d, A, B, logrho, 2.0, 2.0, pik, [P_matrix(k, d) for k in range(d)], peer)
    f0 = normalize_profile(np.ones((2, N)) * np.array([[1.2], [0.9]]), A)
    phi = 1e-4 * rng.normal(size=(2, N)); phi -= phi.mean(axis=1, keepdims=True)
    x = (f0 * (1 + phi), normalize_profile(np.ones((2, N)), B), pik.copy(), pik.copy())
    def contrast(f):
        return np.max(np.abs(np.log(f) - np.log(f.mean(axis=1, keepdims=True))))
    c_prev = contrast(x[0]); rates = []
    for step in range(3):
        x = rk4(profile_velocity, x, 0.005, 100, PD)
        c = contrast(x[0]); rates.append(np.log(c_prev / c) / 0.5); c_prev = c
    print(f"   peer terms {'on ' if peer else 'off'}: measured rates {np.round(rates, 5)}")
