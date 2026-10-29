r"""D21-BT s06b: Abelian reduction of two-link SO(3) alignment and a three-link non-Abelian witness.

For two SO(3) links with nonparallel axes u1, u2, the covariant unit normal
nu = u1 x u2 / |u1 x u2| is built from the links alone. Test: nu is parallel for the
alignment connection (D nu = d nu + a nu = 0), so holonomy lies in the SO(2) fixing nu.
Then three links with non-coplanar axes: holonomies of two loops at one base point
fail to commute, so the restricted holonomy group is non-Abelian (hence SO(3)).
"""
from __future__ import annotations
import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

hat = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])
vee = lambda M: np.array([M[2, 1], M[0, 2], M[1, 0]])
rot = lambda axis, ang: Rot.from_rotvec(ang * np.asarray(axis) / np.linalg.norm(axis)).as_matrix()
axis_of = lambda Q: Rot.from_matrix(Q).as_rotvec() / np.linalg.norm(Rot.from_rotvec(Rot.from_matrix(Q).as_rotvec()).as_rotvec())


def d(fun, c, Y, h=1e-6):
    return (np.array(fun(c + h * Y)) - np.array(fun(c - h * Y))) / (2 * h)


def conn(links, c, Y):
    Ls = links(c); dLs = d(links, c, Y)
    Cs = [np.eye(3) - L for L in Ls]
    K = sum(Cm.T @ Cm for Cm in Cs)
    rhs = -sum(Cm.T @ vee(dL @ L.T) for Cm, dL, L in zip(Cs, dLs, Ls))
    return hat(np.linalg.solve(K, rhs))


# two links with general moving axes (not confined to a fixed plane)
def two_links(c):
    x, y = c
    u1 = np.array([np.cos(0.4 * y), np.sin(0.4 * y) * np.cos(x), np.sin(0.4 * y) * np.sin(x)])
    u2 = np.array([np.sin(y) * np.cos(0.7 * x), np.cos(y), np.sin(y) * np.sin(0.7 * x)])
    return [rot(u1, 0.8 + 0.3 * x), rot(u2, 1.1 + 0.2 * y)]


def normal(c):
    Ls = two_links(c)
    u1, u2 = (Rot.from_matrix(L).as_rotvec() for L in Ls)
    nu = np.cross(u1, u2)
    return nu / np.linalg.norm(nu)


c0 = np.array([0.6, 0.9])
for Y in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
    Dnu = d(normal, c0, Y) + conn(two_links, c0, Y) @ normal(c0)
    print(f"two links: |D_Y nu| = {np.linalg.norm(Dnu):.2e}  (|d_Y nu| = {np.linalg.norm(d(normal, c0, Y)):.3f})")


def three_links(c):
    x, y = c
    return [rot([1, 0, 0], 0.7 + 0.4 * x),
            rot([np.cos(y), np.sin(y), 0.0], 1.1),
            rot([0.0, np.sin(0.5 + 0.6 * x), np.cos(0.5 + 0.6 * x)], 0.9 + 0.3 * y)]


def transport(links, pts):
    T = np.eye(3)
    for p, q in zip(pts[:-1], pts[1:]):
        sol = solve_ivp(lambda t, yv, p=p, q=q: (-conn(links, p + t * (q - p), q - p) @ yv.reshape(3, 3)).ravel(),
                        (0, 1), T.ravel(), rtol=1e-11, atol=1e-13)
        T = sol.y[:, -1].reshape(3, 3)
    return T


base, s = np.array([0.6, 0.9]), 0.3
loop1 = [base, base + [s, 0], base + [s, s], base + [0, s], base]
loop2 = [base, base + [0, -s], base + [-s, -s], base + [-s, 0], base]
for name, lk in (("two links", two_links), ("three links", three_links)):
    H1, H2 = transport(lk, loop1), transport(lk, loop2)
    print(f"{name}: |H1-I| = {np.linalg.norm(H1 - np.eye(3)):.3e}, |H2-I| = {np.linalg.norm(H2 - np.eye(3)):.3e}, |[H1,H2]| = {np.linalg.norm(H1 @ H2 - H2 @ H1):.3e}")
