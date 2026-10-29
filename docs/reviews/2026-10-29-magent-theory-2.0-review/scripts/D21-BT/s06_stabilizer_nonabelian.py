r"""D21-BT s06: pointwise stabilizers and a non-Abelian variant of the SO(3) example (ch16).

(a) Pointwise obstruction. In the ch16 SO(3) example at x = 0 the link Q1 = I and the
    common stabilizer of (Q1, Q2) is span(n), n = (cos y, sin y, 0). For h(c) = exp(f(c) n^)
    with f(c0) = 0, df(c0) != 0, the transformed link fields have the same value AND the
    same first derivatives at c0, while the affine law requires a shift n df(c0).
    So no selector depending only on the 1-jet of the links at c0 is covariant there.
(b) The ch16 connection takes values in span(e3^) everywhere (Abelian holonomy).
    Variant with both axes moving: Q1 = R_y(x) R_x(alpha) R_y(-x), Q2 = R_z(y) R_x(beta) R_z(-y).
    Holonomies of two loops based at c0 fail to commute: non-Abelian link-only holonomy.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

beta, alpha = 1.2, 0.9


def R(axis, ang):
    v = np.zeros(3); v["xyz".index(axis)] = ang
    return Rot.from_rotvec(v).as_matrix()


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def links_orig(c):
    return [R("x", c[0]), R("z", c[1]) @ R("x", beta) @ R("z", -c[1])]


def links_var(c):
    return [R("y", c[0]) @ R("x", alpha) @ R("y", -c[0]), R("z", c[1]) @ R("x", beta) @ R("z", -c[1])]


def d(fun, c, Y, h=1e-6):
    return (np.array(fun(c + h * Y)) - np.array(fun(c - h * Y))) / (2 * h)


# (a) pointwise stabilizer at x = 0
c0 = np.array([0.0, 1.1])
n = np.array([np.cos(c0[1]), np.sin(c0[1]), 0.0])
f = lambda c: 0.7 * (c[0] - c0[0]) + 0.4 * (c[1] - c0[1]) ** 2 + 0.3 * (c[1] - c0[1])
hgauge = lambda c: Rot.from_rotvec(f(c) * n).as_matrix()
links_h = lambda c: [hgauge(c).T @ Q @ hgauge(c) for Q in links_orig(c)]
val = np.linalg.norm(np.array(links_h(c0)) - np.array(links_orig(c0)))
jet = max(np.linalg.norm(d(links_h, c0, Y) - d(links_orig, c0, Y)) for Y in (np.array([1.0, 0]), np.array([0, 1.0])))
second = np.linalg.norm((np.array(links_h(c0 + 1e-3 * np.array([1.0, 0]))) - 2 * np.array(links_h(c0)) + np.array(links_h(c0 - 1e-3 * np.array([1.0, 0]))))
                        - (np.array(links_orig(c0 + 1e-3 * np.array([1.0, 0]))) - 2 * np.array(links_orig(c0)) + np.array(links_orig(c0 - 1e-3 * np.array([1.0, 0]))))) / 1e-6
print(f"(a) at x=0: |value change| = {val:.1e}; max |first-derivative change| = {jet:.1e}; |second x-derivative change| = {second:.3f}")
print(f"(a) required connection shift |n df(c0)| = {np.linalg.norm(n) * np.hypot(0.7, 0.3):.3f}")


# (b) connection from the normal equation, generic links
def conn(links, c, Y):
    Ls = links(c)
    dLs = d(links, c, Y)
    Cs = [np.eye(3) - L for L in Ls]
    Kmat = sum(Cm.T @ Cm for Cm in Cs)
    rhs = -sum(Cm.T @ vee(dL @ L.T) for Cm, dL, L in zip(Cs, dLs, Ls))
    return hat(np.linalg.solve(Kmat, rhs))


ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
for name, lk in (("original", links_orig), ("variant", links_var)):
    pts = [np.array([0.9, 1.1]), np.array([0.5, 0.7]), np.array([1.3, 1.6])]
    comps = [vee(conn(lk, p, Y)) for p in pts for Y in (ex, ey)]
    span_rank = np.linalg.matrix_rank(np.array(comps), tol=1e-8)
    print(f"(b) {name}: rank of span of a-values at 3 points = {span_rank}")


def transport(links, path_pts, steps=1):
    T = np.eye(3)
    for p, q in zip(path_pts[:-1], path_pts[1:]):
        sol = solve_ivp(lambda t, yv, p=p, q=q: (-conn(links, p + t * (q - p), q - p) @ yv.reshape(3, 3)).ravel(),
                        (0, 1), T.ravel(), rtol=1e-11, atol=1e-13)
        T = sol.y[:, -1].reshape(3, 3)
    return T


base = np.array([0.9, 1.1]); s = 0.3
loop1 = [base, base + [s, 0], base + [s, s], base + [0, s], base]          # square to the upper right
loop2 = [base, base + [0, -s], base + [-s, -s], base + [-s, 0], base]      # square to the lower left
for name, lk in (("original", links_orig), ("variant", links_var)):
    H1, H2 = transport(lk, loop1), transport(lk, loop2)
    print(f"(b) {name}: |H1 - I| = {np.linalg.norm(H1 - np.eye(3)):.3e}, |H2 - I| = {np.linalg.norm(H2 - np.eye(3)):.3e}, "
          f"|[H1, H2]| = {np.linalg.norm(H1 @ H2 - H2 @ H1):.3e}")
