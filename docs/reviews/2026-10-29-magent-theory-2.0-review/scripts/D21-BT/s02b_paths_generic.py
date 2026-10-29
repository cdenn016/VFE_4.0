r"""D21-BT s02b: endpoint maps versus transport for a generic 2x4 message matrix.

The witness of ch16 has a special property (its reciprocal map T equals exact
transport along straight segments, see s02). This script uses a generic
polynomial X (r = 4, K = 2) to separate that coincidence from the general rates:
  (1) |C - exact straight transport| ~ eps^2,  |T - exact| ~ eps^3 (not zero);
  (2) observed convergence orders of ordered products of C and T along a smooth path.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm


def X(c: np.ndarray) -> np.ndarray:
    x, y = c
    return np.array([[1 + x, y, x * y, 2.0], [x * x, 1 - y, 3.0, x + y * y]])


def dX(c: np.ndarray, vel: np.ndarray) -> np.ndarray:
    x, y = c
    Dx = np.array([[1.0, 0.0, y, 0.0], [2 * x, 0.0, 0.0, 1.0]])
    Dy = np.array([[0.0, 1.0, x, 0.0], [0.0, -1.0, 0.0, 2 * y]])
    return vel[0] * Dx + vel[1] * Dy


def Qx(c: np.ndarray) -> np.ndarray:
    Xc = X(c)
    return Xc.T @ np.linalg.inv(Xc @ Xc.T)


def C(cp, c):
    return X(cp) @ Qx(c)


def T(cp, c):
    N = C(c, cp) @ C(cp, c)
    return np.real(C(cp, c) @ np.linalg.inv(sqrtm(N)))


def exact(path, dpath, t0, t1):
    def rhs(t, yv):
        a = -dX(path(t), dpath(t)) @ Qx(path(t))
        return (-a @ yv.reshape(2, 2)).ravel()
    sol = solve_ivp(rhs, (t0, t1), np.eye(2).ravel(), rtol=1e-12, atol=1e-14, method="DOP853")
    assert sol.success
    return sol.y[:, -1].reshape(2, 2)


nrm = lambda A: float(np.linalg.norm(A))
c0 = np.array([0.3, -0.4])
d = np.array([0.6, 0.8])
for eps in (0.08, 0.04, 0.02, 0.01):
    cp = c0 + eps * d
    E = exact(lambda t, cp=cp: c0 + t * (cp - c0), lambda t, cp=cp: cp - c0, 0.0, 1.0)
    print(f"(1) eps={eps:5.3f}  |C-E|/eps^2={nrm(C(cp, c0) - E) / eps**2:.5f}  |T-E|/eps^3={nrm(T(cp, c0) - E) / eps**3:.5f}")

path = lambda t: np.array([0.1 + 0.6 * t, -0.3 + 0.5 * np.sin(1.3 * t)])
dpath = lambda t: np.array([0.6, 0.65 * np.cos(1.3 * t)])
E1 = exact(path, dpath, 0.0, 1.0)
for name, rule in (("C", C), ("T", T)):
    errs = []
    for n in (8, 16, 32, 64, 128):
        prod = np.eye(2)
        for k in range(n):
            prod = rule(path((k + 1) / n), path(k / n)) @ prod
        errs.append(nrm(prod - E1))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"(2) {name}: errors {['%.2e' % e for e in errs]}  orders {['%.2f' % o for o in orders]}")
