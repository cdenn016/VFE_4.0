r"""Shared closed-form model of chapters/14_collective_local_parent.tex (lens D18-LP).

State x has shape (3, 8): per agent [l, A_x, A_y, C_x, C_y, B_x, B_y, n].
Law (lp:law): q_i \propto exp{l z/2 + A.e_phi + z C.e_phi + B.e_psi} relative to
uniform sign and normalized Haar circles; presentation s_i(w)=e^{n w/2}/(2 cosh(n/2)).
Rows (lp:weights): w_ii = e^{J t_i}/(5 I0(J)), w_ij = 2 e^{L_j m_i/2 + B_j.M(B_i)}/(5 cosh(L_j/2) I0(|B_j|)).
Field (lp:field): dl=-l+sum_{j!=i} beta_ij L_j + H_i, dA=-A+Lam_i, dC=-C+beta_ii J e_1,
dB=-B+sum_{j!=i} beta_ij B_j, dn=-n+sum_{j!=i} gamma_ij n_j.
Vectorized for speed (single-threaded numpy).
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e, i1e

PI_SELF = 0.2
PI_PEER = 0.4
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
OFF = 1.0 - np.eye(3)


def logI0(r):
    r = np.abs(r)
    return np.log(i0e(r)) + r


def bessel_ratio(r):
    """I_1(r)/I_0(r)."""
    return i1e(r) / i0e(r)


def Mrows(B: np.ndarray) -> np.ndarray:
    r"""Row-wise M(B) = grad log I0(|B|) for B of shape (k,2)."""
    r = np.hypot(B[:, 0], B[:, 1])
    safe = np.where(r > 1e-300, r, 1.0)
    f = np.where(r > 1e-300, bessel_ratio(r) / safe, 0.5)
    return B * f[:, None]


def Mvec(B: np.ndarray) -> np.ndarray:
    return Mrows(np.asarray(B, dtype=float).reshape(1, 2))[0]


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def Umat(S: np.ndarray) -> np.ndarray:
    """U(S) = |S|^{-1} [[S_x, S_y], [-S_y, S_x]] (rotates S to the positive first axis)."""
    r = float(np.hypot(S[0], S[1]))
    return np.array([[S[0], S[1]], [-S[1], S[0]]]) / r


def rotB(x: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply the physical rotation U to every B block (R_U in the manuscript)."""
    x = np.asarray(x, dtype=float).reshape(3, 8).copy()
    x[:, 5:7] = x[:, 5:7] @ U.T
    return x


def calJ(x: np.ndarray) -> np.ndarray:
    """The generator calJ: J2 on each B block, zero elsewhere."""
    x = np.asarray(x).reshape(3, 8)
    out = np.zeros_like(x)
    out[:, 5:7] = x[:, 5:7] @ J2.T
    return out


def public_odds(x: np.ndarray) -> np.ndarray:
    x = x.reshape(3, 8)
    A, C = x[:, 1:3], x[:, 3:5]
    P, Q = A + C, A - C
    return x[:, 0] + logI0(np.hypot(P[:, 0], P[:, 1])) - logI0(np.hypot(Q[:, 0], Q[:, 1]))


def t_stat(x: np.ndarray, e1=None) -> np.ndarray:
    r"""t_i = E_{q_i}[Z (e1 . e_Phi)] in closed form."""
    if e1 is None:
        e1 = np.array([1.0, 0.0])
    x = x.reshape(3, 8)
    l, A, C = x[:, 0], x[:, 1:3], x[:, 3:5]
    P, Q = A + C, A - C
    wp = l / 2 + logI0(np.hypot(P[:, 0], P[:, 1]))
    wm = -l / 2 + logI0(np.hypot(Q[:, 0], Q[:, 1]))
    pp = 1.0 / (1.0 + np.exp(wm - wp))
    return pp * (Mrows(P) @ e1) - (1 - pp) * (Mrows(Q) @ e1)


def content_rows(x: np.ndarray, J: float, e1=None) -> np.ndarray:
    x = x.reshape(3, 8)
    L = public_odds(x)
    m = np.tanh(L / 2)
    t = t_stat(x, e1)
    B = x[:, 5:7]
    Mi = Mrows(B)
    logw = (np.log(2.0 / 5.0) + L[None, :] * m[:, None] / 2 + Mi @ B.T
            - np.log(np.cosh(L / 2))[None, :] - logI0(np.hypot(B[:, 0], B[:, 1]))[None, :])
    np.fill_diagonal(logw, J * t - np.log(5.0) - logI0(J))
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)


def pres_rows(x: np.ndarray) -> np.ndarray:
    x = x.reshape(3, 8)
    n = x[:, 7]
    pp = 1.0 / (1.0 + np.exp(-n))  # s(+1)
    p = np.stack([pp, 1 - pp], axis=1)
    D = (p[:, None, :] * (np.log(p[:, None, :]) - np.log(p[None, :, :]))).sum(axis=2)
    Dself = (p * np.log(2 * p)).sum(axis=1)
    np.fill_diagonal(D, Dself)
    pri = np.where(np.eye(3) > 0, PI_SELF, PI_PEER)
    w = pri * np.exp(-(D - D.min(axis=1, keepdims=True)))
    return w / w.sum(axis=1, keepdims=True)


def V(x: np.ndarray, H: np.ndarray, Lam: np.ndarray, J: float, e1=None) -> np.ndarray:
    """Native own-block field lp:field. Lam has shape (3,2) (embedded marked record sums)."""
    if e1 is None:
        e1 = np.array([1.0, 0.0])
    x = np.asarray(x, dtype=float).reshape(3, 8)
    beta = content_rows(x, J, e1)
    gam = pres_rows(x)
    L = public_odds(x)
    bo, go = beta * OFF, gam * OFF
    dx = np.empty_like(x)
    dx[:, 0] = -x[:, 0] + bo @ L + H
    dx[:, 1:3] = -x[:, 1:3] + Lam
    dx[:, 3:5] = -x[:, 3:5] + np.outer(np.diag(beta) * J, e1)
    dx[:, 5:7] = -x[:, 5:7] + bo @ x[:, 5:7]
    dx[:, 7] = -x[:, 7] + go @ x[:, 7]
    return dx


def S_of(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(3, 8)[:, 5:7].sum(axis=0)


def omega_of(x: np.ndarray, H, Lam, J, e1=None) -> float:
    r"""Angular velocity det(S, sum_i V_{B_i})/|S|^2 of the B-sum (lp:omega / lp:local-angular-velocity)."""
    S = S_of(x)
    VS = V(x, H, Lam, J, e1)[:, 5:7].sum(axis=0)
    return float((S[0] * VS[1] - S[1] * VS[0]) / (S @ S))
