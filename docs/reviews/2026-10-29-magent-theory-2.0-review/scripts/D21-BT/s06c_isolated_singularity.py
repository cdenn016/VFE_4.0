r"""D21-BT s06c: an isolated isotropy jump in a generic two-link SO(3) field.

Q1 = R_{e1}(alpha) constant, Q2 = R_{u(c)}(beta) with u(c) = (1, x, y)/|(1, x, y)|.
At c* = (0, 0) the axes are parallel and the common stabilizer is SO(2)_{e1}; elsewhere
the infinitesimal action is injective. We measure the alignment connection a near c*
(size versus radius r), the e1-component of a around circles, and the holonomy of
small circles about c* (expected: |a| ~ 1/r, holonomy -> rotation by 2 pi = I).
"""
from __future__ import annotations
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

alpha, beta = 0.8, 1.1
hat = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])
vee = lambda M: np.array([M[2, 1], M[0, 2], M[1, 0]])


def links(c):
    u = np.array([1.0, c[0], c[1]]); u /= np.linalg.norm(u)
    return [Rot.from_rotvec(alpha * np.array([1.0, 0, 0])).as_matrix(), Rot.from_rotvec(beta * u).as_matrix()]


def d(fun, c, Y, h=1e-7):
    return (np.array(fun(c + h * Y)) - np.array(fun(c - h * Y))) / (2 * h)


def avec(c, Y):
    Ls = links(c); dLs = d(links, c, Y)
    Cs = [np.eye(3) - L for L in Ls]
    K = sum(Cm.T @ Cm for Cm in Cs)
    rhs = -sum(Cm.T @ vee(dL @ L.T) for Cm, dL, L in zip(Cs, dLs, Ls))
    return np.linalg.solve(K, rhs)


for r in (0.1, 0.01, 0.001):
    th = np.linspace(0, 2 * np.pi, 9)[:-1]
    norms = []
    for t in th:
        c = r * np.array([np.cos(t), np.sin(t)])
        tang = np.array([-np.sin(t), np.cos(t)])
        norms.append(np.linalg.norm(avec(c, tang)))
    print(f"r={r:7.3f}: max |a(unit tangent)| on circle = {max(norms):.4e};  r * max = {r * max(norms):.4f}")


def circle_holonomy(r, n_steps=1):
    def rhs(t, yv):
        c = r * np.array([np.cos(t), np.sin(t)])
        vel = r * np.array([-np.sin(t), np.cos(t)])
        return (-hat(avec(c, vel)) @ yv.reshape(3, 3)).ravel()
    sol = solve_ivp(rhs, (0, 2 * np.pi), np.eye(3).ravel(), rtol=1e-10, atol=1e-12)
    return sol.y[:, -1].reshape(3, 3)


for r in (0.1, 0.03, 0.01):
    Hl = circle_holonomy(r)
    ang = np.linalg.norm(Rot.from_matrix(Hl).as_rotvec())
    print(f"r={r:5.2f}: holonomy rotation angle = {ang:.4e} rad (mod 2 pi);  |Hol - I| = {np.linalg.norm(Hl - np.eye(3)):.3e}")
# line integral of the e1-component, which a continuous connection would send to 0
for r in (0.1, 0.01):
    th = np.linspace(0, 2 * np.pi, 2001)
    vals = [avec(r * np.array([np.cos(t), np.sin(t)]), r * np.array([-np.sin(t), np.cos(t)]))[0] for t in th]
    print(f"r={r:5.2f}: circulation of the e1-component of a = {np.trapezoid(vals, th):.4f}  (2 pi = {2*np.pi:.4f})")
