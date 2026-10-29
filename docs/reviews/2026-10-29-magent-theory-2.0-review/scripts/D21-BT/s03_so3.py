r"""D21-BT s03: independent recomputation of the SO(3) reciprocal-link example (ch16).

Q1 = R_x(x), Q2 = R_z(y) R_x(beta) R_z(-y); product bi-invariant metric on SO(3)^2
normalized by <v^, w^> = v.w; action by simultaneous conjugation.
Part A (sympy, exact): normal equation from scratch, connection a, curvature F.
Part B (numpy/scipy): finite alignment branch, its first jet, reversal, endpoint law,
small-loop holonomy versus curvature, and the x -> 0 continuation.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

# ---------------- Part A: exact ----------------
# Weierstrass parametrization t = tan(angle/2) makes every quantity rational, so the
# normal equation K a = rhs can be checked exactly by cancellation.
tx, ty, tb = sp.symbols("t_x t_y t_b", real=True)


def cs(t):
    return (1 - t**2) / (1 + t**2), 2 * t / (1 + t**2)


def Rx(t):
    c, s = cs(t)
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(t):
    c, s = cs(t)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def vee(M):
    return sp.Matrix([M[2, 1], M[0, 2], M[1, 0]])


def hat(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Rz_inv(t):
    return Rz(t).T


Q = [Rx(tx), Rz(ty) * Rx(tb) * Rz_inv(ty)]
# d/dx = (1 + t_x^2)/2 d/dt_x, and similarly for y
dQ = {"x": [sp.diff(Qr, tx) * (1 + tx**2) / 2 for Qr in Q],
      "y": [sp.diff(Qr, ty) * (1 + ty**2) / 2 for Qr in Q]}
E = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
Cr = []
for r in range(2):
    cols = [vee((hat(e) * Q[r] - Q[r] * hat(e)) * Q[r].T) for e in E]
    Cm = sp.Matrix.hstack(*cols).applyfunc(sp.cancel)
    assert (Cm - (sp.eye(3) - Q[r])).applyfunc(sp.cancel) == sp.zeros(3)   # C_r = I - Ad_{Q_r}
    Cr.append(Cm)
K = sum((Cm.T * Cm for Cm in Cr), sp.zeros(3)).applyfunc(sp.cancel)
wx = 4 * tx**2 / (1 + tx**2)          # 4 sin^2(x/2)
wb = 4 * tb**2 / (1 + tb**2)          # 4 sin^2(beta/2)
claim = {"x": sp.zeros(3, 1), "y": sp.Matrix([0, 0, -wb / (wx + wb)])}
for c in ("x", "y"):
    V = [vee(dQ[c][r] * Q[r].T).applyfunc(sp.cancel) for r in range(2)]
    rhs = -sum((Cr[r].T * V[r] for r in range(2)), sp.zeros(3, 1))
    res = (K * claim[c] - rhs).applyfunc(sp.cancel)
    print(f"A: normal-equation residual K a_{c} - rhs_{c} =", list(res))
    if c == "y":
        # also the two identities quoted in the text
        print("A: C_1^T V_1 (x-direction) =", list((Cr[0].T * vee(dQ['x'][0] * Q[0].T)).applyfunc(sp.cancel)))
        print("A: C_2^T V_2 (y-direction) - w_b e3 =", list((Cr[1].T * V[1] - sp.Matrix([0, 0, wb])).applyfunc(sp.cancel)))
detK = sp.factor(sp.cancel(K.det()))
print("A: det K =", detK)
# curvature from the (now verified) connection, in the original angle x
xs, bs = sp.symbols("x beta", real=True)
wxs, wbs = 4 * sp.sin(xs / 2) ** 2, 4 * sp.sin(bs / 2) ** 2
ay = -wbs / (wxs + wbs)
Fz = sp.diff(ay, xs)                  # a has only a dy component, so F = d a
print("A: F_xy e3-coefficient minus claim:", sp.simplify(Fz - 2 * wbs * sp.sin(xs) / (wxs + wbs) ** 2))

# ---------------- Part B: numerical ----------------
beta = 1.2


def links(c, h=None):
    Q1 = Rot.from_rotvec([c[0], 0, 0]).as_matrix()
    Q2 = Rot.from_rotvec([0, 0, c[1]]).as_matrix() @ Rot.from_rotvec([beta, 0, 0]).as_matrix() @ Rot.from_rotvec([0, 0, -c[1]]).as_matrix()
    Ls = [Q1, Q2]
    if h is not None:
        H = h(c)
        Ls = [H.T @ L @ H for L in Ls]               # reference change: h^{-1} Q h
    return Ls


def align(cp, c, h=None, g0=None):
    """Local branch near e of argmin_g sum_r dist(Q_r(c'), g Q_r(c) g^{-1})^2."""
    Lt, Ls = links(cp, h), links(c, h)

    def res(p):
        g = Rot.from_rotvec(p).as_matrix()
        return np.concatenate([Rot.from_matrix(A.T @ g @ B @ g.T).as_rotvec() for A, B in zip(Lt, Ls)])

    fit = least_squares(res, np.zeros(3) if g0 is None else g0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    return Rot.from_rotvec(fit.x).as_matrix()


def a_num(c, vel):
    w_x, w_b = 4 * np.sin(c[0] / 2) ** 2, 4 * np.sin(beta / 2) ** 2
    return -w_b / (w_x + w_b) * vel[1] * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]])


c0 = np.array([0.9, 1.1])
for k, Y in enumerate((np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, -0.8]))):
    eps = 1e-4
    jet = (align(c0 + eps * Y, c0) - align(c0 - eps * Y, c0)) / (2 * eps)
    print(f"B: first jet direction {Y}: |jet + a(Y)| = {np.linalg.norm(jet + a_num(c0, Y)):.2e}")
cp = c0 + np.array([0.12, -0.07])
t1, t2 = align(cp, c0), align(c0, cp)
print(f"B: reversal |t(c,c') t(c',c) - I| = {np.linalg.norm(t2 @ t1 - np.eye(3)):.2e}")


def hfield(c):
    return Rot.from_rotvec([0.3 * np.sin(c[0]), 0.2 * c[1] ** 2, -0.4 * c[0] * c[1]]).as_matrix()


tp = align(cp, c0, h=hfield)
print(f"B: endpoint law |t' - h(c')^-1 t h(c)| = {np.linalg.norm(tp - hfield(cp).T @ t1 @ hfield(c0)):.2e}")
print(f"B: |t(c',c) - I| at this separation = {np.linalg.norm(t1 - np.eye(3)):.3e}")


def holonomy(center, side):
    """Counterclockwise square loop, exact ODE transport dT/dt = -a(gamma') T."""
    corners = [center + side * np.array(v) for v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5), (-.5, -.5))]
    Tm = np.eye(3)
    for p, q in zip(corners[:-1], corners[1:]):
        def rhs(t, yv, p=p, q=q):
            return (-a_num(p + t * (q - p), q - p) @ yv.reshape(3, 3)).ravel()
        sol = solve_ivp(rhs, (0, 1), Tm.ravel(), rtol=1e-12, atol=1e-14)
        Tm = sol.y[:, -1].reshape(3, 3)
    return Tm


w_x0, w_b0 = 4 * np.sin(c0[0] / 2) ** 2, 4 * np.sin(beta / 2) ** 2
Fcoef = 2 * w_b0 * np.sin(c0[0]) / (w_x0 + w_b0) ** 2
for side in (0.04, 0.02, 0.01):
    Hl = holonomy(c0, side)
    gen = (Hl - np.eye(3)) / side ** 2
    print(f"B: side {side}: (Hol - I)/area e3-component = {gen[1, 0]:.6f};  -F_xy e3-coefficient = {-Fcoef:.6f}")

# discrete holonomy: product of finite alignments around the same square
side = 0.04
corners = [c0 + side * np.array(v) for v in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5), (-.5, -.5))]
for n in (4, 8, 16):
    Tm = np.eye(3)
    for p, q in zip(corners[:-1], corners[1:]):
        for j in range(n):
            Tm = align(p + (j + 1) / n * (q - p), p + j / n * (q - p)) @ Tm
    print(f"B: discrete holonomy (n={n} per side) vs exact: |diff| = {np.linalg.norm(Tm - holonomy(c0, side)):.2e}")

# continuation through x = 0: at x = 0 the locked form is singular (stabilizer of Q1 = I)
for xv in (0.1, 0.01, 0.0, -0.01):
    Kn = sum((np.eye(3) - L).T @ (np.eye(3) - L) for L in links(np.array([xv, 1.1])))
    print(f"B: x={xv:+.2f}: min eig of locked form = {np.linalg.eigvalsh(Kn).min():.3e}; "
          f"a_y coefficient = {-4*np.sin(beta/2)**2/(4*np.sin(xv/2)**2+4*np.sin(beta/2)**2):.6f}")
