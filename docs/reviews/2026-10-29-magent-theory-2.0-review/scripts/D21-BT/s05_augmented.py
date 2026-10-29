r"""D21-BT s05: the augmented alignment metric of ch16 (Section ibt:sec:alignment), G = GL(2).

Target Z = (X, K) with X a full-rank 2x3 matrix and K in GL(2); action
g.(X, K) = (g X, g K g^{-1}); metric
  H((dX,dK),(dX,dK)) = Tr(H_X^{-1} dX dX^T) + lam Tr(H_X^{-1} dK H_X dK^T).
Checks:
  (1) invariance of the metric under X -> h^{-1}X, K -> h^{-1} K h (nonorthogonal h);
  (2) affine law of the normal-equation connection (finite differences);
  (3) Tr a = -1/2 d log det H_X still holds (links are blind to scalars);
  (4) metric compatibility with H_X^{-1} fails once lam > 0;
  (5) small-loop holonomy: eigenvalue moduli (off the unit circle means the holonomy
      lies in no compact subgroup); lam = 0 reproduces the rectangular selector.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

B4 = []
for i in range(2):
    for j in range(2):
        E = np.zeros((2, 2)); E[i, j] = 1.0
        B4.append(E)


def Xf(c):
    x, y = c
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y]])


def Kf(c):
    x, y = c
    return expm(np.array([[0.3 * x, 0.8 * y], [0.5 * x * y, -0.2 + 0.4 * y]]))


def d(fun, c, Y, h=1e-6):
    return (fun(c + h * Y) - fun(c - h * Y)) / (2 * h)


def metric(X, dX1, dK1, dX2, dK2, lam):
    Hinv = np.linalg.inv(X @ X.T); H = X @ X.T
    return np.trace(Hinv @ dX1 @ dX2.T) + lam * np.trace(Hinv @ dK1 @ H @ dK2.T)


def conn(c, Y, lam, Xfun=Xf, Kfun=Kf):
    X, K = Xfun(c), Kfun(c)
    dX, dK = d(Xfun, c, Y), d(Kfun, c, Y)
    J = [(E @ X, E @ K - K @ E) for E in B4]
    G = np.array([[metric(X, *Ji, *Jj, lam) for Jj in J] for Ji in J])
    rhs = -np.array([metric(X, dX, dK, *Ji, lam) for Ji in J])
    coef = np.linalg.solve(G, rhs)
    return sum(ci * Ei for ci, Ei in zip(coef, B4))


rng = np.random.default_rng(3)
c0 = np.array([0.35, -0.25]); ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
h0 = np.array([[1.3, 0.4], [-0.2, 0.7]])
X0, K0 = Xf(c0), Kf(c0)
dX1, dK1 = rng.normal(size=(2, 3)), rng.normal(size=(2, 2))
v1 = metric(X0, dX1, dK1, dX1, dK1, 1.0)
X0p, dX1p = np.linalg.solve(h0, X0), np.linalg.solve(h0, dX1)
dK1p = np.linalg.solve(h0, dK1 @ h0)
v2 = metric(X0p, dX1p, dK1p, dX1p, dK1p, 1.0)
print(f"(1) metric invariance under nonorthogonal h: |H' - H| = {abs(v1 - v2):.2e}")


def hfield(c):
    x, y = c
    return np.array([[1.0 + 0.3 * x, 0.2 * y], [0.1 * x * y, 0.9 + 0.2 * y * y]])


lam = 1.0
for Y in (ex, ey):
    a = conn(c0, Y, lam)
    Xp = lambda c: np.linalg.solve(hfield(c), Xf(c))
    Kp = lambda c: np.linalg.solve(hfield(c), Kf(c) @ hfield(c))
    ap = conn(c0, Y, lam, Xp, Kp)
    tgt = np.linalg.solve(hfield(c0), a @ hfield(c0) + d(hfield, c0, Y))
    Hm = X0 @ X0.T
    trace_res = np.trace(a) + 0.5 * np.trace(np.linalg.solve(Hm, d(lambda c: Xf(c) @ Xf(c).T, c0, Y)))
    comp = d(lambda c: Xf(c) @ Xf(c).T, c0, Y) + a @ Hm + Hm @ a.T
    print(f"(2-4) Y={Y}: affine-law residual {np.linalg.norm(ap - tgt):.2e}; trace-identity residual {trace_res:.2e}; "
          f"|dH + aH + Ha^T| = {np.linalg.norm(comp):.3e}")


def holonomy(lam_v, center, side):
    corners = [center + side * np.array(v) for v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5), (-.5, -.5))]
    T = np.eye(2)
    for p, q in zip(corners[:-1], corners[1:]):
        def rhs(t, yv, p=p, q=q):
            return (-conn(p + t * (q - p), q - p, lam_v) @ yv.reshape(2, 2)).ravel()
        sol = solve_ivp(rhs, (0, 1), T.ravel(), rtol=1e-11, atol=1e-13)
        T = sol.y[:, -1].reshape(2, 2)
    return T


for lam_v in (0.0, 1.0):
    Hl = holonomy(lam_v, c0, 0.2)
    ev = np.linalg.eigvals(Hl)
    Fapprox = -(Hl - np.eye(2)) / 0.04
    print(f"(5) lam={lam_v}: holonomy eigenvalues {np.round(ev, 6)}  moduli {np.round(np.abs(ev), 6)}  det {np.linalg.det(Hl):.6f}; "
          f"eigenvalues of -(Hol-I)/area {np.round(np.linalg.eigvals(Fapprox), 4)}")
a_rect = -d(Xf, c0, ex) @ X0.T @ np.linalg.inv(X0 @ X0.T)
print(f"(5) lam=0 reproduces -dX Q_X: |a - a_rect| = {np.linalg.norm(conn(c0, ex, 0.0) - a_rect):.2e}")
