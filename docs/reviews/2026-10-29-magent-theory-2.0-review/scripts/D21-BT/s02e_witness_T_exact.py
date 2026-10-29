r"""D21-BT s02e: for the ch16 witness X = [I, v], compare T(c',c) with exact transport along the
straight segment from c to c'. Prediction: equal iff the principal angle is attained along the
gnomonic image of the segment, i.e. iff n(c).n(c') > 0, i.e. 1 + v.v' > 0 (the image arc is then
the RP^2 geodesic). For 1 + v.v' < 0 the RP^2 geodesic runs through -n(c') and T differs."""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
X = lambda c: np.column_stack((np.eye(2), c))
Qx = lambda c: X(c).T @ np.linalg.inv(X(c) @ X(c).T)
C = lambda cp, c: X(cp) @ Qx(c)
T = lambda cp, c: np.real(C(cp, c) @ np.linalg.inv(sqrtm(C(c, cp) @ C(cp, c))))
rng = np.random.default_rng(2)
pos, neg = [], []
for _ in range(40):
    c, cp = rng.uniform(-1.5, 1.5, 2), rng.uniform(-1.5, 1.5, 2)
    if abs(1 + c @ cp) < 0.05:
        continue
    rhs = lambda t, y: (np.outer(cp - c, c + t * (cp - c)) / (1 + (c + t * (cp - c)) @ (c + t * (cp - c))) @ y.reshape(2, 2)).ravel()
    E = solve_ivp(rhs, (0, 1), np.eye(2).ravel(), rtol=1e-12, atol=1e-14, method="DOP853").y[:, -1].reshape(2, 2)
    (pos if 1 + c @ cp > 0 else neg).append(np.linalg.norm(T(cp, c) - E))
print(f"1 + v.v' > 0: {len(pos)} pairs, max |T - exact straight transport| = {max(pos):.2e}")
print(f"1 + v.v' < 0: {len(neg)} pairs, min |T - exact straight transport| = {min(neg):.2e}")
