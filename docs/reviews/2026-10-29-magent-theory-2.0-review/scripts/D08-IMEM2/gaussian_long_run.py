r"""Long-run behavior of the reduced Gaussian filter (extension probe, not a manuscript claim).

Lens D08-IMEM2.  Iterates imem:gaussian-reduced-recursion's covariance part
S_{t+1}^{-1} = S_t^{-1} + A_t^T R_t^{-1} A_t (history independent) for a scalar
two-agent affine Gaussian preparation with alternating speakers, in float64,
and reports t * S_t to test the conjecture S_t = O(1/t) (asymptotic pooling of
the private records).  Floating point; a numerical observation only.
"""
from __future__ import annotations

import numpy as np

L0, L1, S, N0, N1 = 1.0, 2.0, 2.0 / 3.0, 1.0, 1.0 / 3.0
F = (-1.0, 2.0)
Q = (1.0, 0.5)
B = (1.0, 2.0)
R = (1.0, 0.5)
SigX = np.array([[L0, S], [S, L1]])
SigD = SigX + np.diag([N0, N1])
# reconstruction W: E[X|D] = C D (centered), Cov(X|D) = Om
C = SigX @ np.linalg.inv(SigD)
Om = SigX - C @ SigX
St = SigD.copy()
rows = []
for t in range(400):
    j = 1 if t % 2 == 0 else 0
    o = 1 - j
    reg = St[o, j] / St[j, j]
    Vo = St[o, o] - reg * St[j, o]
    # E[Y_j | D_j, H] slope on D_j and Var(Y_j | D_j, H)
    coef = F[j] * (C[j, j] + C[j, o] * reg)
    varY = Q[j] + F[j] ** 2 * Om[j, j] + (F[j] * C[j, o]) ** 2 * Vo
    A = np.zeros((1, 2))
    A[0, j] = B[j] * coef
    Rt = B[j] ** 2 * varY + R[j]
    St = np.linalg.inv(np.linalg.inv(St) + A.T @ A / Rt)
    if t + 1 in (1, 2, 10, 50, 100, 200, 400):
        rows.append((t + 1, St.copy(), coef))
for t, St_, coef in rows:
    print(f"t={t:4d}  diag(S_t)={np.diag(St_)}  t*diag(S_t)={t * np.diag(St_)}  corr={St_[0,1]/np.sqrt(St_[0,0]*St_[1,1]):+.4f}  last slope={coef:+.4f}")
