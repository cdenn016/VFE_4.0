r"""D21-BT s05b: can the augmented (law+link) GL(2) alignment connection have hyperbolic curvature?

For random smooth link fields K(c) = expm(A0 + x A1 + y A2) and the augmented metric of
ch16 with link weight lam, compute F_xy at c0 by finite differences of the normal-equation
connection and record det F_xy (F is traceless; det F < 0 means real eigenvalues, so small
loops have hyperbolic holonomy, which lies in no compact subgroup). The rectangular
selector (lam = 0) always gives det F >= 0 because F is H_X^{-1}-skew.
"""
from __future__ import annotations
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

B4 = []
for i in range(2):
    for j in range(2):
        E = np.zeros((2, 2)); E[i, j] = 1.0
        B4.append(E)
Xf = lambda c: np.array([[1.0, 0.0, c[0]], [0.0, 1.0, c[1]]])


def make_K(A0, A1, A2):
    return lambda c: expm(A0 + c[0] * A1 + c[1] * A2)


def d(fun, c, Y, h=1e-5):
    return (fun(c + h * Y) - fun(c - h * Y)) / (2 * h)


def metric(X, dX1, dK1, dX2, dK2, lam):
    H = X @ X.T; Hi = np.linalg.inv(H)
    return np.trace(Hi @ dX1 @ dX2.T) + lam * np.trace(Hi @ dK1 @ H @ dK2.T)


def conn(c, Y, lam, Kf):
    X, K = Xf(c), Kf(c)
    dX, dK = d(Xf, c, Y), d(Kf, c, Y)
    J = [(E @ X, E @ K - K @ E) for E in B4]
    G = np.array([[metric(X, *Ji, *Jj, lam) for Jj in J] for Ji in J])
    rhs = -np.array([metric(X, dX, dK, *Ji, lam) for Ji in J])
    return sum(ci * Ei for ci, Ei in zip(np.linalg.solve(G, rhs), B4))


def curvature(c, lam, Kf, h=1e-4):
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ax = lambda p: conn(p, ex, lam, Kf)
    ay = lambda p: conn(p, ey, lam, Kf)
    return (ay(c + h * ex) - ay(c - h * ex)) / (2 * h) - (ax(c + h * ey) - ax(c - h * ey)) / (2 * h) + ax(c) @ ay(c) - ay(c) @ ax(c)


rng = np.random.default_rng(5)
c0 = np.array([0.2, -0.1])
best = None
for trial in range(40):
    A0, A1, A2 = 0.5 * rng.normal(size=(2, 2)), rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
    Kf = make_K(A0, A1, A2)
    for lam in (1.0, 10.0):
        F = curvature(c0, lam, Kf)
        dF = np.linalg.det(F)
        if best is None or dF < best[0]:
            best = (dF, trial, lam, A0, A1, A2, F)
F0 = curvature(c0, 0.0, make_K(*best[3:6]))
print(f"lam=0 (rectangular): det F = {np.linalg.det(F0):.4f}, tr F = {np.trace(F0):.1e}, eig F = {np.round(np.linalg.eigvals(F0), 4)}")
dF, trial, lam, A0, A1, A2, F = best
print(f"most negative det F over scan: {dF:.4f} (trial {trial}, lam {lam}); tr F = {np.trace(F):.1e}; eig F = {np.round(np.linalg.eigvals(F), 4)}")
# holonomy of a small square for that case
Kf = make_K(A0, A1, A2)
side = 0.05
corners = [c0 + side * np.array(v) for v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5), (-.5, -.5))]
T = np.eye(2)
for p, q in zip(corners[:-1], corners[1:]):
    sol = solve_ivp(lambda t, yv, p=p, q=q: (-conn(p + t * (q - p), q - p, lam, Kf) @ yv.reshape(2, 2)).ravel(),
                    (0, 1), T.ravel(), rtol=1e-10, atol=1e-12)
    T = sol.y[:, -1].reshape(2, 2)
ev = np.linalg.eigvals(T)
print(f"small-square holonomy (side {side}): eigenvalues {np.round(ev, 8)}, moduli {np.round(np.abs(ev), 8)}, det {np.linalg.det(T):.8f}, |tr| = {abs(np.trace(T)):.8f}")
np.save("s05b_best_links.npy", np.stack([A0, A1, A2]))
