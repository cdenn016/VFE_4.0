r"""GAP10-S3 s05: SU(2) monodromy of the link-alignment connection (ibt:alignment,
ibt:eq:SO3-normal) at an isolated isotropy jump.

Two links Q_1 = R_{u_1}(alpha), Q_2 = R_{u_2(c)}(beta(c)) with the product bi-invariant metric.
Normal equation (ibt:eq:SO3-normal): K a = -sum_r C_r^T V_r, C_r = I - Ad_{Q_r}, V_r = vee(dQ_r Q_r^T).
At c = 0 the axes are parallel (common stabilizer SO(2)): an isolated isotropy jump.
The same Lie-algebra form a is transported in SO(3) (dT/dt = -a(gamma') T, ibt:eq:path-ode) and in
SU(2) via rho(v) = -(i/2) v.sigma (so Ad rho = id).  Circles of radius r about the jump.
Cases: (a) u_2 ~ (1, x, y), constant angles (flat, as in D21-BT s06c);
       (b) nonflat: u_2 ~ (1, x + 0.4 y^2, y + 0.3 x^2), alpha, beta varying;
       (c) winding 2: u_2 ~ (1, x^2 - y^2, 2xy);  (d) winding -1: u_2 ~ (1, x, -y).
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import expm

t0 = time.time()
sig = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex)]


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def Rax(u, ang):
    u = u / np.linalg.norm(u)
    return expm(ang * hat(u))


def rho(v):
    return -0.5j * (v[0] * sig[0] + v[1] * sig[1] + v[2] * sig[2])


def Ad_su2(U):
    r"""Rotation matrix of Ad_U on su(2) ~ R^3."""
    Rm = np.zeros((3, 3))
    for j in range(3):
        X = U @ rho(np.eye(3)[j]) @ U.conj().T
        # recover vector v with rho(v) = X:  v_k = i Tr(X sigma_k)
        Rm[:, j] = np.real([1j * np.trace(X @ s) for s in sig])
    return Rm


def make_case(name):
    if name == 'a':
        return (lambda c: np.array([1.0, c[0], c[1]]), lambda c: 0.9, lambda c: 1.3)
    if name == 'b':
        return (lambda c: np.array([1.0, c[0] + 0.4 * c[1] ** 2, c[1] + 0.3 * c[0] ** 2]),
                lambda c: 0.9 + 0.3 * c[1], lambda c: 1.3 + 0.5 * c[0])
    if name == 'c':
        return (lambda c: np.array([1.0, c[0] ** 2 - c[1] ** 2, 2 * c[0] * c[1]]), lambda c: 0.9, lambda c: 1.3)
    if name == 'd':
        return (lambda c: np.array([1.0, c[0], -c[1]]), lambda c: 0.9, lambda c: 1.3)


def links(case, c):
    u2, al, be = case
    return [Rax(np.array([1.0, 0, 0]), al(c)), Rax(u2(c), be(c))]


def conn(case, c, hd=1e-7):
    Q = links(case, c)
    Cr = [np.eye(3) - Qr for Qr in Q]          # Ad_Q on R^3 is Q itself
    Kf = sum(C.T @ C for C in Cr)
    a = []
    for k in range(2):
        dc = np.zeros(2); dc[k] = hd
        Qp, Qm = links(case, c + dc), links(case, c - dc)
        rhs = np.zeros(3)
        for Cm, qp, qm, q in zip(Cr, Qp, Qm, Q):
            dQ = (qp - qm) / (2 * hd)
            rhs -= Cm.T @ vee(dQ @ q.T)
        a.append(np.linalg.solve(Kf, rhs))
    return a, np.linalg.eigvalsh(Kf).min()


def holonomy(case, r, nst=400):
    T3 = np.eye(3); T2 = np.eye(2, dtype=complex)

    def gen(tt):
        p = r * np.array([np.cos(tt), np.sin(tt)])
        v = r * np.array([-np.sin(tt), np.cos(tt)])
        a, _ = conn(case, p)
        av = a[0] * v[0] + a[1] * v[1]
        return -hat(av), -rho(av), av
    h = 2 * np.pi / nst
    circ = 0.0
    for k in range(nst):
        tt = k * h
        g1, s1, a1 = gen(tt); g2, s2, a2 = gen(tt + h / 2); g4, s4, a4 = gen(tt + h)
        k1 = g1 @ T3; k2 = g2 @ (T3 + h / 2 * k1); k3 = g2 @ (T3 + h / 2 * k2); k4 = g4 @ (T3 + h * k3)
        T3 = T3 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        m1 = s1 @ T2; m2 = s2 @ (T2 + h / 2 * m1); m3 = s2 @ (T2 + h / 2 * m2); m4 = s4 @ (T2 + h * m3)
        T2 = T2 + h / 6 * (m1 + 2 * m2 + 2 * m3 + m4)
        circ += h / 6 * (a1[0] + 4 * a2[0] + a4[0])
    return T3, T2, circ


for name, label in [('a', 'flat, winding 1'), ('b', 'nonflat, winding 1'), ('c', 'winding 2'), ('d', 'winding -1')]:
    case = make_case(name)
    print(f"case ({name}) {label}:")
    for r in [0.4, 0.1, 0.02, 0.004]:
        a, kmin = conn(case, np.array([r, 0.0]))
        T3, T2, circ = holonomy(case, r)
        dev3 = np.linalg.norm(T3 - np.eye(3))
        devp = np.linalg.norm(T2 - np.eye(2)); devm = np.linalg.norm(T2 + np.eye(2))
        cons = np.linalg.norm(Ad_su2(T2) - T3)
        print(f"   r={r:<6}: r|a| = {r*(np.linalg.norm(a[0])+np.linalg.norm(a[1])):.4f}, min eig K = {kmin:.2e}, "
              f"e1-circulation = {circ: .5f} (/2pi = {circ/(2*np.pi): .4f}); |Hol_SO3 - I| = {dev3:.2e}; "
              f"|Hol_SU2 - I| = {devp:.2e}, |Hol_SU2 + I| = {devm:.2e}; |Ad(Hol_SU2) - Hol_SO3| = {cons:.1e}")
print(f"runtime {time.time()-t0:.1f} s")
