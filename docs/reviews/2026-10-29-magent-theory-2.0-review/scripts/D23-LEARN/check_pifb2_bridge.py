r"""D23-LEARN check 6: the tangent loop prior as the second-order form of the PIFB2 generator prior.

(a) Q(H;S) = 1/2 ||S^{-1/2}(I - H)S^{1/2}||_F^2 with H = e^{eps X3} e^{eps X2} e^{eps X1}:
    Q/eps^2 -> 1/2 ||S^{-1/2}(X1+X2+X3)S^{1/2}||_F^2 = 1/2 vec(sum X)^T (S^{-1} kron S) vec(sum X),
    i.e. the dressed tangent loop form of chapter 17 line 307 with W_gamma = S^{-1} kron S.
(b) Gaussian loop normalizer ratio: for nu = N(0, s2 I) on a scalar two-cycle,
    E_nu exp(-lam/2 (w1 + w2)^2) = (1 + 2 lam s2)^{-1/2} = det(Pi0)^{1/2} det(Pi0 + lam B^T B)^{-1/2}.
(c) Nonlinear dressed two-cycle prior in K = 2: Monte Carlo of Z = E_nu exp(-lam Q(e^{w1} e^{w2}; S))
    against the Jensen bounds exp(-lam E_nu Q) <= Z <= 1 (pifb2/05_section.tex:51).
Monte Carlo numbers are estimates with the stated sample size, not exact values.
"""
import numpy as np
from scipy.linalg import expm, sqrtm

rng = np.random.default_rng(5)
K = 3
X = [rng.standard_normal((K, K)) for _ in range(3)]
A = rng.standard_normal((K, K)); S = A @ A.T + K * np.eye(K)
Sh = np.real(sqrtm(S)); Shi = np.linalg.inv(Sh)
target = 0.5 * np.sum((Shi @ sum(X) @ Sh) ** 2)
v = sum(X).reshape(-1)
kron_form = 0.5 * v @ np.kron(np.linalg.inv(S), S) @ v
for eps in [1e-2, 1e-3]:
    H = expm(eps * X[2]) @ expm(eps * X[1]) @ expm(eps * X[0])
    Q = 0.5 * np.sum((Shi @ (np.eye(K) - H) @ Sh) ** 2)
    print(f"(a) eps={eps:.0e}: Q/eps^2 = {Q / eps ** 2:.6f}; tangent form = {target:.6f}; kron form = {kron_form:.6f}")

s2, lam, n = 0.7, 1.5, 400000
w = rng.standard_normal((n, 2)) * np.sqrt(s2)
mc = np.exp(-0.5 * lam * (w.sum(1)) ** 2).mean()
Pi0 = np.eye(2) / s2; Bm = np.array([[1.0, 1.0]])
exact = np.sqrt(np.linalg.det(Pi0) / np.linalg.det(Pi0 + lam * Bm.T @ Bm))
print(f"(b) MC {mc:.5f} (n={n}) vs (1+2 lam s2)^(-1/2) = {(1 + 2 * lam * s2) ** -0.5:.5f} vs det ratio = {exact:.5f}")

K = 2; m = 20000; s2 = 0.25; lam = 2.0
S = np.array([[2.0, 0.3], [0.3, 0.5]]); Sh = np.real(sqrtm(S)); Shi = np.linalg.inv(Sh)
Qs = np.empty(m)
for t in range(m):
    w1 = rng.standard_normal((K, K)) * np.sqrt(s2); w2 = rng.standard_normal((K, K)) * np.sqrt(s2)
    H = expm(w1) @ expm(w2)
    Qs[t] = 0.5 * np.sum((Shi @ (np.eye(K) - H) @ Sh) ** 2)
Z = np.exp(-lam * Qs).mean()
print(f"(c) MC Z = {Z:.4f} (n={m}); Jensen lower bound exp(-lam E Q) = {np.exp(-lam * Qs.mean()):.4f}; upper bound 1")
