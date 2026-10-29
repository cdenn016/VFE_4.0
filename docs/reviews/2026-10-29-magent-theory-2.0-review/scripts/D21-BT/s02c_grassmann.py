r"""D21-BT s02c: geometric meaning of N(c',c) and T(c',c) (ch16, ibt:reciprocal).

Claim tested: with W(c) = im X(c)^T in the message space (an r-dim Euclidean space),
  (a) N(c',c) has eigenvalues cos^2 of the principal angles between W(c) and W(c');
  (b) T(c',c), carried to the message space by the isometries Q_X, equals the direct
      rotation W(c) -> W(c') (orthogonal polar factor of the projection), and
  (c) the direct rotation equals parallel transport of the tautological (projected)
      connection along the Grassmannian geodesic from W(c) to W(c').
Numerical evidence at generic points; the proof sketch is in the D21-BT report.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm, subspace_angles, polar, logm, expm, null_space

rng = np.random.default_rng(7)


def X(c: np.ndarray) -> np.ndarray:
    x, y = c
    return np.array([[1 + x, y, x * y, 2.0], [x * x, 1 - y, 3.0, x + y * y]])


def Qx(c):
    Xc = X(c)
    return Xc.T @ np.linalg.inv(Xc @ Xc.T)


def C(cp, c):
    return X(cp) @ Qx(c)


def T(cp, c):
    return np.real(C(cp, c) @ np.linalg.inv(sqrtm(C(c, cp) @ C(cp, c))))


def onb(c):
    """Orthonormal basis of W(c) = im X(c)^T (r x K)."""
    q, _ = np.linalg.qr(X(c).T)
    return q


for trial in range(3):
    c = rng.uniform(-0.6, 0.6, 2)
    cp = c + rng.uniform(-0.25, 0.25, 2)
    N = C(c, cp) @ C(cp, c)
    eigN = np.sort(np.real(np.linalg.eigvals(N)))
    ang = subspace_angles(X(c).T, X(cp).T)
    cos2 = np.sort(np.cos(ang) ** 2)
    # (b) direct rotation via polar factor of Y'^T Y
    Y, Yp = onb(c), onb(cp)
    Up, _ = polar(Yp.T @ Y)
    D = Yp @ Up @ Y.T                      # r x r operator, acts on W(c)
    PW = Y @ Y.T
    lhs = Qx(cp) @ T(cp, c) @ X(c) @ PW    # T carried to message space
    # (c) transport along the Grassmann geodesic W(t) = exp(t Xi) W(c),
    # with Xi in so(r) mapping W -> W^perp and W^perp -> W (horizontal generator)
    # build Xi from the direct rotation: R = D on W extended; use principal vectors
    Wp = null_space(Y.T)                   # basis of W(c)^perp
    # tangent B (dim(W^perp) x K) of the geodesic: Y' Up^T = Y cos(S) + Wp U sin(S) form
    Mfull = np.hstack([Y, Wp])
    # find B via the log of the rotation that takes Y to Yp Up (in the plane of principal vectors)
    target = Yp @ Up                       # = Y V cos(S) V^T + Wp U sin(S) V^T
    A = Y.T @ target                       # V cos(S) V^T (symmetric PD)
    Bm = Wp.T @ target                     # U sin(S) V^T
    # recover S, V: A = V cos S V^T
    w, V = np.linalg.eigh(A)
    S = np.arccos(np.clip(w, -1, 1))
    U = Bm @ V @ np.diag(1 / np.sin(S))
    Btan = U @ np.diag(S) @ V.T            # (r-K) x K tangent
    Xi = Mfull @ np.block([[np.zeros((2, 2)), -Btan.T], [Btan, np.zeros((Btan.shape[0], Btan.shape[0]))]]) @ Mfull.T

    def proj(t):
        Yt = expm(t * Xi) @ Y
        return Yt @ Yt.T

    def rhs(t, wv):
        h = 1e-6
        dP = (proj(t + h) - proj(t - h)) / (2 * h)
        return (dP @ wv.reshape(4, 2)).ravel()

    sol = solve_ivp(rhs, (0.0, 1.0), Y.ravel(), rtol=1e-11, atol=1e-13)
    Ytrans = sol.y[:, -1].reshape(4, 2)
    geo_end = expm(Xi) @ Y
    print(f"trial {trial}: |eig N - cos^2(angles)| = {np.max(np.abs(eigN - cos2)):.2e}; "
          f"|T_msg - D| = {np.linalg.norm(lhs - D):.2e}; "
          f"|geodesic endpoint span - W(c')| = {np.linalg.norm(geo_end @ geo_end.T - Yp @ Yp.T):.2e}; "
          f"|transported frame - D Y| = {np.linalg.norm(Ytrans - D @ Y):.2e}")
