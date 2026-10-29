r"""D21-BT s04: the declared-group selector and the joint two-sector selector (ch16, ibt:group).

(a) g = so(3): affine law for SO(3)-valued reference changes; failure for a GL(3) change.
(b) g = gl(K): minimizer independent of the fiber metric M; g = so(K): it depends on M.
(c) two sectors, G = GL(2), rho_q = defining, rho_s = dual (d rho_s(xi) = -xi^T):
    joint minimizer obeys the affine law, differs from each one-sector minimizer,
    depends on the relative sector weight, and the one-sector gl connections are not
    the two images of one generator.
Finite differences (step 1e-6) for contextual derivatives; residuals are reported.
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(11)
A3 = rng.normal(size=(3, 5)); B3 = rng.normal(size=(3, 5)); C3 = rng.normal(size=(3, 5))


def X3(c):
    return A3 + c[0] * B3 + np.sin(c[1]) * C3 + 0.3 * c[0] * c[1] * np.ones((3, 5))


Lm = rng.normal(size=(3, 3))


def M3(c):
    L = Lm + c[0] * np.eye(3) + 0.2 * c[1] * Lm.T
    return L @ L.T + np.eye(3)


def basis_so3():
    out = []
    for (i, j) in ((1, 2), (2, 0), (0, 1)):
        E = np.zeros((3, 3)); E[i, j] = -1; E[j, i] = 1
        out.append(E)
    return out


def basis_gl(k):
    out = []
    for i in range(k):
        for j in range(k):
            E = np.zeros((k, k)); E[i, j] = 1
            out.append(E)
    return out


def d(fun, c, Y, h=1e-6):
    return (fun(c + h * Y) - fun(c - h * Y)) / (2 * h)


def selector(Xf, Mf, basis, c, Y):
    r"""argmin_{xi in span(basis)} 1/2 Tr[M (dX(Y) + xi X)(dX(Y) + xi X)^T]."""
    Xc, Mc, dXc = Xf(c), Mf(c), d(Xf, c, Y)
    G = np.array([[np.trace(Mc @ (Ei @ Xc) @ (Ej @ Xc).T) for Ej in basis] for Ei in basis])
    rhs = -np.array([np.trace(Mc @ dXc @ (Ei @ Xc).T) for Ei in basis])
    coef = np.linalg.solve(G, rhs)
    return sum(ci * Ei for ci, Ei in zip(coef, basis))


c0 = np.array([0.3, -0.5]); Y = np.array([0.6, 0.8])


def h_so3(c):
    return expm(np.array([[0, -0.7 * c[0], 0.4 * c[1]], [0.7 * c[0], 0, -0.5 * c[0] * c[1]], [-0.4 * c[1], 0.5 * c[0] * c[1], 0]]))


def h_gl3(c):
    return np.array([[1 + 0.4 * c[0], 0.3 * c[1], 0], [0.1, 1.2 + c[1] ** 2, 0.2 * c[0]], [0, 0.5 * c[0], 0.9]])


for name, hf in (("SO(3)-valued h", h_so3), ("GL(3)-valued h", h_gl3)):
    a = selector(X3, M3, basis_so3(), c0, Y)
    Xp = lambda c, hf=hf: np.linalg.solve(hf(c), X3(c))
    Mp = lambda c, hf=hf: hf(c).T @ M3(c) @ hf(c)
    ap = selector(Xp, Mp, basis_so3(), c0, Y)
    target = np.linalg.solve(hf(c0), a @ hf(c0) + d(hf, c0, Y))
    print(f"(a) so(3) selector, {name}: |a' - (h^-1 a h + h^-1 dh)| = {np.linalg.norm(ap - target):.2e}")

aM1 = selector(X3, M3, basis_gl(3), c0, Y)
aM2 = selector(X3, lambda c: np.eye(3), basis_gl(3), c0, Y)
Xc = X3(c0); aExpl = -d(X3, c0, Y) @ Xc.T @ np.linalg.inv(Xc @ Xc.T)
print(f"(b) gl(3): |a(M) - a(I)| = {np.linalg.norm(aM1 - aM2):.2e}; |a(M) - (-dX Q_X)| = {np.linalg.norm(aM1 - aExpl):.2e}")
sM1 = selector(X3, M3, basis_so3(), c0, Y)
sM2 = selector(X3, lambda c: np.eye(3), basis_so3(), c0, Y)
print(f"(b) so(3): |a(M) - a(I)| = {np.linalg.norm(sM1 - sM2):.3e} (M-dependence of the constrained selector)")

# (c) two sectors with rho_s = dual representation of GL(2)
Aq, Bq = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
As_, Bs_ = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
Xq = lambda c: Aq + c[0] * Bq + 0.2 * np.sin(c[1]) * np.ones((2, 3))
Xs = lambda c: As_ + np.cos(c[0]) * Bs_ + 0.3 * c[1] * Aq
Mq = lambda c: np.array([[2.0 + c[0] ** 2, 0.3], [0.3, 1.0]])
Ms = lambda c: np.array([[1.5, -0.2 * c[1]], [-0.2 * c[1], 1.0 + 0.1 * c[0]]])
B4 = basis_gl(2)


def joint(Xqf, Xsf, Mqf, Msf, c, Yv, lam=1.0):
    Xqc, Xsc, Mqc, Msc = Xqf(c), Xsf(c), Mqf(c), Msf(c)
    dq, ds = d(Xqf, c, Yv), d(Xsf, c, Yv)
    act_q = lambda E: E @ Xqc
    act_s = lambda E: -E.T @ Xsc
    G = np.array([[np.trace(Mqc @ act_q(Ei) @ act_q(Ej).T) + lam * np.trace(Msc @ act_s(Ei) @ act_s(Ej).T)
                   for Ej in B4] for Ei in B4])
    rhs = -np.array([np.trace(Mqc @ dq @ act_q(Ei).T) + lam * np.trace(Msc @ ds @ act_s(Ei).T) for Ei in B4])
    coef = np.linalg.solve(G, rhs)
    return sum(ci * Ei for ci, Ei in zip(coef, B4))


def h2(c):
    return np.array([[1.0 + 0.5 * c[0], 0.4 * c[1]], [-0.2 * c[0] * c[1], 0.8 + 0.3 * c[1] ** 2]])


a = joint(Xq, Xs, Mq, Ms, c0, Y)
Xqp = lambda c: np.linalg.solve(h2(c), Xq(c))
Xsp = lambda c: h2(c).T @ Xs(c)                         # rho_s(h)^{-1} = h^T
Mqp = lambda c: h2(c).T @ Mq(c) @ h2(c)
Msp = lambda c: np.linalg.inv(h2(c)) @ Ms(c) @ np.linalg.inv(h2(c)).T   # rho_s(h)^T M rho_s(h), rho_s(h) = h^{-T}
ap = joint(Xqp, Xsp, Mqp, Msp, c0, Y)
target = np.linalg.solve(h2(c0), a @ h2(c0) + d(h2, c0, Y))
print(f"(c) joint selector affine law: |a' - (h^-1 a h + h^-1 dh)| = {np.linalg.norm(ap - target):.2e}")
aq_only = -d(Xq, c0, Y) @ Xq(c0).T @ np.linalg.inv(Xq(c0) @ Xq(c0).T)
as_gl = -d(Xs, c0, Y) @ Xs(c0).T @ np.linalg.inv(Xs(c0) @ Xs(c0).T)     # one-sector gl(V_s) connection
print(f"(c) |joint - q-only| = {np.linalg.norm(a - aq_only):.3e}; |d rho_s(joint) - s-only gl connection| = {np.linalg.norm(-a.T - as_gl):.3e}")
print(f"(c) one generator for both one-sector gl connections? |(-aq_only^T) - as_gl| = {np.linalg.norm(-aq_only.T - as_gl):.3e}")
for lam in (0.1, 1.0, 10.0):
    print(f"(c) relative sector weight lam={lam:5.1f}: joint a(Y) = {np.round(joint(Xq, Xs, Mq, Ms, c0, Y, lam), 4).tolist()}")
