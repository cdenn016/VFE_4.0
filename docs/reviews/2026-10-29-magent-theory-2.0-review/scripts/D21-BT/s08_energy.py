r"""D21-BT s08: minimized residual of ibt:group at M = H_X^{-1} versus the Dirichlet energy
density of the Grassmannian map c -> W(c) = im X(c)^T:
  1/2 Tr(H^{-1} dX (I - P) dX^T)  =?  1/4 ||dP||_F^2   (per base direction)."""
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
rng = np.random.default_rng(4)
A, B, C = rng.normal(size=(3, 6)), rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
X = lambda t: A + t * B + t * t * C
P = lambda t: X(t).T @ np.linalg.solve(X(t) @ X(t).T, X(t))
t0, h = 0.3, 1e-6
dX = (X(t0 + h) - X(t0 - h)) / (2 * h)
dP = (P(t0 + h) - P(t0 - h)) / (2 * h)
H = X(t0) @ X(t0).T
lhs = 0.5 * np.trace(np.linalg.solve(H, dX @ (np.eye(6) - P(t0)) @ dX.T))
rhs = 0.25 * np.linalg.norm(dP) ** 2
print(f"1/2 Tr(H^-1 dX (I-P) dX^T) = {lhs:.10f};  1/4 |dP|_F^2 = {rhs:.10f};  difference {abs(lhs - rhs):.2e}")
