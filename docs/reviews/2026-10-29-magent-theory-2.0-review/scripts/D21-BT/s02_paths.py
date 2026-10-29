r"""D21-BT s02: finite endpoint maps versus path transport for the witness (ch16).

Witness: X(c) = [I_2, v], v = c in R^2, a = -(dv) v^T / s, s = 1 + |v|^2.
Checks:
  (1) C(c',c) = X(c') Q_X(c) fails composition and reversal at finite separation;
  (2) C and T differ from exact straight-segment transport at second order;
  (3) ordered products of C and of T converge to the ODE transport; observed orders;
  (4) a path with a corner, partitions that do NOT contain the corner;
  (5) holonomy of the exact transport around a loop lies in SO(H_X^{-1}(c_0)).
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm


def X(c: np.ndarray) -> np.ndarray:
    return np.column_stack((np.eye(2), c))


def Qx(c: np.ndarray) -> np.ndarray:
    Xc = X(c)
    return Xc.T @ np.linalg.inv(Xc @ Xc.T)


def C(cp: np.ndarray, c: np.ndarray) -> np.ndarray:
    return X(cp) @ Qx(c)


def T(cp: np.ndarray, c: np.ndarray) -> np.ndarray:
    N = C(c, cp) @ C(cp, c)
    return np.real(C(cp, c) @ np.linalg.inv(sqrtm(N)))


def a_of(c: np.ndarray, vel: np.ndarray) -> np.ndarray:
    r"""a_c(vel) = -vel v^T / s with v = c."""
    return -np.outer(vel, c) / (1.0 + c @ c)


def exact(path, dpath, t0: float, t1: float) -> np.ndarray:
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return (-a_of(path(t), dpath(t)) @ y.reshape(2, 2)).ravel()
    sol = solve_ivp(rhs, (t0, t1), np.eye(2).ravel(), rtol=1e-12, atol=1e-14, method="DOP853")
    assert sol.success
    return sol.y[:, -1].reshape(2, 2)


def nrm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A))


out: list[str] = []
c0 = np.array([0.4, -0.3])
c1 = c0 + np.array([0.15, 0.05])
c2 = c1 + np.array([-0.05, 0.12])
out.append(f"(1) composition defect |C(c2,c1)C(c1,c0)-C(c2,c0)| = {nrm(C(c2, c1) @ C(c1, c0) - C(c2, c0)):.3e}")
out.append(f"(1) reversal defect |C(c0,c1)C(c1,c0)-I| = {nrm(C(c0, c1) @ C(c1, c0) - np.eye(2)):.3e}")
out.append(f"(1) T reversal defect |T(c0,c1)T(c1,c0)-I| = {nrm(T(c0, c1) @ T(c1, c0) - np.eye(2)):.3e}")
out.append(f"(1) T composition defect |T(c2,c1)T(c1,c0)-T(c2,c0)| = {nrm(T(c2, c1) @ T(c1, c0) - T(c2, c0)):.3e}")

# (2) second-order deviation from straight-segment transport
d = np.array([0.6, 0.8])
for eps in (0.1, 0.05, 0.025):
    cp = c0 + eps * d
    seg = lambda t, cp=cp: c0 + t * (cp - c0)
    dseg = lambda t, cp=cp: cp - c0
    E = exact(seg, dseg, 0.0, 1.0)
    out.append(f"(2) eps={eps:6.3f}  |C-Tex|/eps^2 = {nrm(C(cp, c0) - E) / eps**2:.4f}   |T-Tex|/eps^2 = {nrm(T(cp, c0) - E) / eps**2:.4f}  |T-Tex|/eps^3 = {nrm(T(cp, c0) - E) / eps**3:.4f}")

# (3) smooth path products
path = lambda t: np.array([0.3 + 0.5 * t, -0.2 + 0.4 * np.sin(1.1 * t)])
dpath = lambda t: np.array([0.5, 0.44 * np.cos(1.1 * t)])
E1 = exact(path, dpath, 0.0, 1.0)
for name, rule in (("C", C), ("T", T)):
    errs = []
    for n in (8, 16, 32, 64, 128):
        prod = np.eye(2)
        for k in range(n):
            prod = rule(path((k + 1) / n), path(k / n)) @ prod
        errs.append(nrm(prod - E1))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    out.append(f"(3) {name}: errors {['%.2e' % e for e in errs]}  observed orders {['%.2f' % o for o in orders]}")

# (4) path with a corner at t = 0.5; uniform partitions with odd n never contain t = 0.5
def cpath(t: float) -> np.ndarray:
    return np.array([0.2 + t, -0.1]) if t <= 0.5 else np.array([0.7, -0.1 + 1.2 * (t - 0.5)])


def dcpath(t: float) -> np.ndarray:
    return np.array([1.0, 0.0]) if t <= 0.5 else np.array([0.0, 1.2])


Ecorner = exact(cpath, dcpath, 0.5, 1.0) @ exact(cpath, dcpath, 0.0, 0.5)
for name, rule in (("C", C), ("T", T)):
    errs = []
    for n in (9, 17, 33, 65, 129):
        prod = np.eye(2)
        for k in range(n):
            prod = rule(cpath((k + 1) / n), cpath(k / n)) @ prod
        errs.append(nrm(prod - Ecorner))
    out.append(f"(4) corner path, odd n (corner not a node), {name}: errors {['%.2e' % e for e in errs]}")

# (5) holonomy around a loop based at c0
loop = lambda t: c0 + 0.5 * np.array([np.cos(2 * np.pi * t) - 1.0, np.sin(2 * np.pi * t)])
dloop = lambda t: 0.5 * 2 * np.pi * np.array([-np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
Hol = exact(loop, dloop, 0.0, 1.0)
M0 = np.linalg.inv(X(c0) @ X(c0).T)
out.append(f"(5) |Hol - I| = {nrm(Hol - np.eye(2)):.3e} (nontrivial holonomy)")
out.append(f"(5) |Hol^T M0 Hol - M0| = {nrm(Hol.T @ M0 @ Hol - M0):.3e}  (M0 = H_X(c0)^-1)")
out.append(f"(5) det Hol = {np.linalg.det(Hol):.15f}")
out.append(f"(5) eigenvalue moduli of Hol = {np.abs(np.linalg.eigvals(Hol))}")

print("\n".join(out))
