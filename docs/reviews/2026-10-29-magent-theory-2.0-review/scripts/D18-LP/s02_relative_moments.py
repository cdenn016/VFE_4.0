r"""D18-LP check 2: lp:relative-identification and the Bessel facts used in chapter 14.

(a) c_ij = E exp(i(Psi_i - Psi_j)) = zeta_i conj(zeta_j), zeta_i = (I1/I0)(|B_i|) e^{i arg B_i},
    by direct quadrature over (psi_i, psi_j) of the product von Mises law.
(b) |zeta_i|^2 = |c_ij||c_ik|/|c_jk|.
(c) d/db (I1/I0)(b) = Var_b(cos Psi) > 0, and b^2 E_b sin^2 Psi = b I1(b)/I0(b) (lp:phase-information).
(d) A proper-block rotation changes a cross moment; the common rotation changes none.
"""
import numpy as np
from scipy.special import i0e, i1e

N = 256
ps = 2 * np.pi * np.arange(N) / N
B = np.array([[0.7, 0.2], [-0.3, 0.9], [1.1, -0.6]])


def vm(Bi):
    d = np.exp(Bi[0] * np.cos(ps) + Bi[1] * np.sin(ps))
    return d / d.sum()


def zeta(Bi):
    r = np.hypot(*Bi)
    return i1e(r) / i0e(r) * np.exp(1j * np.arctan2(Bi[1], Bi[0]))


def cmom(Bi, Bj):
    pi, pj = vm(Bi), vm(Bj)
    return np.sum(pi[:, None] * pj[None, :] * np.exp(1j * (ps[:, None] - ps[None, :])))


err = 0.0
c = {}
for i in range(3):
    for j in range(3):
        if i != j:
            c[i, j] = cmom(B[i], B[j])
            err = max(err, abs(c[i, j] - zeta(B[i]) * np.conj(zeta(B[j]))))
print("(a) max |c_ij - zeta_i conj zeta_j| =", err)
err = 0.0
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    err = max(err, abs(np.sqrt(abs(c[i, j]) * abs(c[i, k]) / abs(c[j, k])) - abs(zeta(B[i]))))
print("(b) max | sqrt(|c_ij||c_ik|/|c_jk|) - |zeta_i| | =", err)
bs = np.linspace(0.0, 5.0, 11)
h = 1e-5
worst = 0.0; worst2 = 0.0; minder = 1e9
for b in bs:
    der = (i1e(b + h) / i0e(b + h) - i1e(abs(b - h)) / i0e(abs(b - h)) * (1 if b - h >= 0 else -1)) / (2 * h) if b > 0 else 0.5
    p = np.exp(b * np.cos(ps)); p /= p.sum()
    var = np.sum(p * np.cos(ps) ** 2) - np.sum(p * np.cos(ps)) ** 2
    if b > 0:
        worst = max(worst, abs(der - var))
    minder = min(minder, var)
    worst2 = max(worst2, abs(b * b * np.sum(p * np.sin(ps) ** 2) - b * i1e(b) / i0e(b)))
print("(c) max |d(I1/I0)/db - Var(cos)| =", worst, "; min Var over grid =", minder)
print("(c) max |b^2 E sin^2 - b I1/I0| =", worst2)
R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
a = 0.8
Bblock = B.copy(); Bblock[0] = R(a) @ B[0]
Ball = (R(a) @ B.T).T
print("(d) proper block {0} rotated: |c_01 change| =", abs(cmom(Bblock[0], Bblock[1]) - c[0, 1]),
      "; common rotation: max change =", max(abs(cmom(Ball[i], Ball[j]) - c[i, j]) for (i, j) in c))
