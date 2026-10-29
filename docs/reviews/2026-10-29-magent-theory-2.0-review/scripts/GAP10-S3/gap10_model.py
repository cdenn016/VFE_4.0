r"""Shared model for GAP10-S3: the three-agent transmitted-angle field of
chapters/14_collective_local_parent.tex (lp:law, lp:weights, lp:field).

State x in R^24 (or C^24 for complex-step differentiation), per agent i=0,1,2:
  l_i (1), A_i (2), C_i (2), B_i (2), n_i (1), stored as
  x = [l(3), A(3x2), C(3x2), B(3x2), n(3)].

Field (lp:field):
  dl_i = -l_i + sum_{j!=i} beta_ij L_j + H_i
  dA_i = -A_i + Lambda_i(c)
  dC_i = -C_i + beta_ii J e_ev(c)
  dB_i = -B_i + sum_{j!=i} beta_ij B_j
  dn_i = -n_i + sum_{j!=i} gamma_ij n_j
with the Gibbs rows of lp:weights,
  w_ii = e^{J t_i}/(5 I_0(J)),  w_ij = 2 e^{L_j m_i/2 + B_j . M(B_i)}/(5 cosh(L_j/2) I_0(|B_j|)),
  t_i = E_{q_i}[Z e_ev . e_Phi],  m_i = tanh(L_i/2),  M(B) = (I_1/I_0)(|B|) B/|B|.
The marked frame varies with context: e_ev(c) = e_{theta(c)}, Lambda_i(c) = R(theta(c)) Lambda_i^0.
All operations are analytic so that complex-step derivatives are exact to rounding.
Norms use sqrt(v1^2+v2^2) (no conjugation) for that reason.
"""
from __future__ import annotations

import numpy as np
from scipy.special import iv

NAG = 3


def cnorm(v: np.ndarray) -> np.ndarray:
    r"""Analytic Euclidean norm sqrt(v_1^2+v_2^2) along the last axis."""
    return np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)


def rot(th) -> np.ndarray:
    r"""Rotation matrix R(th) = [[cos, -sin],[sin, cos]] (complex-safe)."""
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def unpack(x: np.ndarray):
    l = x[0:3]
    A = x[3:9].reshape(3, 2)
    C = x[9:15].reshape(3, 2)
    B = x[15:21].reshape(3, 2)
    n = x[21:24]
    return l, A, C, B, n


def pack(l, A, C, B, n) -> np.ndarray:
    return np.concatenate([l, A.reshape(-1), C.reshape(-1), B.reshape(-1), n])


class Model:
    r"""Container for constants: J, H, Lambda0 and the evaluator-angle polynomial theta(c)."""

    def __init__(self, J: float = 0.4,
                 H=(0.4, -0.3, 0.2),
                 Lam0=((0.3, 0.1), (-0.2, 0.25), (0.1, -0.15)),
                 th_coef=(0.2, 0.4, -0.3, 0.2)) -> None:
        self.J = J
        self.H = np.array(H, dtype=float)
        self.Lam0 = np.array(Lam0, dtype=float)
        self.th_coef = th_coef
        self.I0J = iv(0, J)

    def theta(self, c):
        a0, a1, a2, a12 = self.th_coef
        return a0 + a1 * c[0] + a2 * c[1] + a12 * c[0] * c[1]

    def rows(self, x: np.ndarray, c):
        r"""Return (beta, gamma, L, m, t) of lp:weights at state x and context c."""
        l, A, C, B, n = unpack(x)
        th = self.theta(c)
        eev = np.array([np.cos(th), np.sin(th)])
        ApC, AmC = A + C, A - C
        rp, rm = cnorm(ApC), cnorm(AmC)
        I0p, I0m, I1p, I1m = iv(0, rp), iv(0, rm), iv(1, rp), iv(1, rm)
        L = l + np.log(I0p) - np.log(I0m)
        m = np.tanh(L / 2)
        ep, em = np.exp(l / 2), np.exp(-l / 2)
        EZe = (ep[:, None] * (I1p / rp)[:, None] * ApC - em[:, None] * (I1m / rm)[:, None] * AmC) \
            / (ep * I0p + em * I0m)[:, None]
        t = EZe @ eev
        rB = cnorm(B)
        MB = (iv(1, rB) / iv(0, rB) / rB)[:, None] * B
        I0B = iv(0, rB)
        dt = np.result_type(x.dtype, np.asarray(c).dtype, float)
        w = np.zeros((3, 3), dtype=dt)
        for i in range(3):
            w[i, i] = np.exp(self.J * t[i]) / (5 * self.I0J)
            for j in range(3):
                if j != i:
                    w[i, j] = 2 * np.exp(L[j] * m[i] / 2 + B[j] @ MB[i]) / (5 * np.cosh(L[j] / 2) * I0B[j])
        beta = w / w.sum(axis=1, keepdims=True)
        mp = np.tanh(n / 2)
        wp = np.zeros((3, 3), dtype=dt)
        for i in range(3):
            wp[i, i] = 0.2 * 0.5
            for j in range(3):
                if j != i:
                    wp[i, j] = 0.4 * np.exp(n[j] * mp[i] / 2 - np.log(2 * np.cosh(n[j] / 2)))
        gam = wp / wp.sum(axis=1, keepdims=True)
        return beta, gam, L, m, t, eev

    def field(self, x: np.ndarray, c, H=None) -> np.ndarray:
        r"""Native field V(x, c, h) of lp:field; H overrides the persistent sign coefficients."""
        Hh = self.H if H is None else H
        l, A, C, B, n = unpack(x)
        beta, gam, L, m, t, eev = self.rows(x, c)
        th = self.theta(c)
        Lam = self.Lam0 @ rot(th).T
        offb = beta - np.diag(np.diag(beta))
        offg = gam - np.diag(np.diag(gam))
        dl = -l + offb @ L + Hh
        dA = -A + Lam
        dC = -C + np.diag(beta)[:, None] * self.J * eev[None, :]
        dB = -B + offb @ B
        dn = -n + offg @ n
        return pack(dl, dA, dC, dB, dn)


def Ssum(x: np.ndarray) -> np.ndarray:
    return unpack(x)[3].sum(axis=0)


def omega_loc(model: Model, x: np.ndarray, c, H=None):
    r"""lp:local-angular-velocity: det(S, sum_i V_{B_i}) / |S|^2 (analytic)."""
    S = Ssum(x)
    VB = unpack(model.field(x, c, H))[3].sum(axis=0)
    return (S[0] * VB[1] - S[1] * VB[0]) / (S[0] ** 2 + S[1] ** 2)


def Umat(S) -> np.ndarray:
    r"""lp:retention: U(S) = |S|^{-1} [[S_x, S_y], [-S_y, S_x]] (rotation by -arg S)."""
    r = np.sqrt(S[0] ** 2 + S[1] ** 2)
    return np.array([[S[0], S[1]], [-S[1], S[0]]]) / r


def rotB(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    r"""Physical action R_U: rotate only the B blocks by the 2x2 matrix R."""
    l, A, C, B, n = unpack(x)
    return pack(l, A, C, B @ R.T, n)


def argS(S):
    r"""Analytic argument of S for S_x > 0: atan(S_y/S_x)."""
    return np.arctan(S[1] / S[0])


def rk4(f, x0: np.ndarray, t0: float, t1: float, nsteps: int) -> np.ndarray:
    r"""Fixed-step classical RK4; returns the array of states at nsteps+1 grid points.

    A fixed step makes the discrete solution a smooth function of the context, so
    finite and complex-step differences in c are not polluted by adaptive-step noise.
    """
    h = (t1 - t0) / nsteps
    xs = np.empty((nsteps + 1, x0.size), dtype=x0.dtype)
    xs[0] = x0
    x = x0
    t = t0
    for k in range(nsteps):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        xs[k + 1] = x
    return xs


def initial_section(c) -> np.ndarray:
    r"""A generic smooth prepared section x(c,0) on a 2-d context patch (not coherent)."""
    c1, c2 = c[0], c[1]
    l = np.array([0.3 + 0.2 * c1 - 0.1 * c2 ** 2, -0.2 + 0.1 * c1 * c2, 0.5 - 0.3 * c2])
    A = np.array([[0.6 + 0.2 * c1, -0.1 + 0.3 * c2],
                  [0.2 - 0.1 * c2, 0.4 + 0.1 * c1 ** 2],
                  [-0.3 + 0.1 * c1, 0.2 + 0 * c1]])
    C = np.array([[0.1 + 0 * c1, 0.05 * c1],
                  [-0.05 + 0.02 * c2, 0.1 + 0 * c1],
                  [0.0 * c1, -0.08 + 0.03 * c1]])
    b = np.array([0.8 + 0.1 * c1, 0.5 + 0.1 * c2, 0.6 - 0.05 * c1 * c2])
    chi = np.array([0.3 + 0.5 * c1 - 0.2 * c2, -0.2 + 0.3 * c2 + 0.1 * c1 ** 2, 0.1 - 0.4 * c1 + 0.3 * c2])
    B = np.stack([b * np.cos(chi), b * np.sin(chi)], axis=1)
    n = np.array([0.2 + 0.1 * c1, -0.3 + 0 * c1, 0.1 * c2])
    return pack(l, A, C, B, n)
