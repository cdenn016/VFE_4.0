r"""D21-BT s02d: T(c',c) is an isometry (E_c, H_X(c)^{-1}) -> (E_c', H_X(c')^{-1}), and N(c',c) has
positive real spectrum wherever C(c',c) is invertible (not only near the diagonal).
Generic 2x4 message matrix, far-apart endpoint pairs.
"""
from __future__ import annotations
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
from scipy.linalg import sqrtm


def X(c):
    x, y = c
    return np.array([[1 + x, y, x * y, 2.0], [x * x, 1 - y, 3.0, x + y * y]])


def Qx(c):
    Xc = X(c); return Xc.T @ np.linalg.inv(Xc @ Xc.T)


Minv = lambda c: np.linalg.inv(X(c) @ X(c).T)
rng = np.random.default_rng(1)
worst_iso, min_eig, max_imag = 0.0, np.inf, 0.0
for _ in range(200):
    c, cp = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
    C = X(cp) @ Qx(c)
    if abs(np.linalg.det(C)) < 1e-3:
        continue
    N = X(c) @ Qx(cp) @ C
    ev = np.linalg.eigvals(N)
    min_eig, max_imag = min(min_eig, ev.real.min()), max(max_imag, abs(ev.imag).max())
    T = np.real(C @ np.linalg.inv(sqrtm(N)))
    worst_iso = max(worst_iso, np.linalg.norm(T.T @ Minv(cp) @ T - Minv(c)) / np.linalg.norm(Minv(c)))
print(f"200 random far-apart pairs: min real eigenvalue of N = {min_eig:.3e}, max |imag| = {max_imag:.1e}, "
      f"max relative isometry defect of T = {worst_iso:.1e}")
