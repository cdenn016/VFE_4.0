r"""Exact (rational / symbolic) checks of the GF closed forms.

Four agents, heterogeneous child noises s_i^2 and record noises rho_i^2,
nested structure Pi = (12|34 ; 1234) and (12|34 ; 12|34), finest.

(1) Collapsed free energy: min_{Q_nu} F = -log N(m; 0, C_Pi) + sum_i S_i/(2 s_i^2) + const,
    and the sweep m' = a o + b K m equals m - eta * grad_m Fbar (unit natural-gradient
    step at the stationary variance eta_i = (s_i^-2 + rho_i^-2)^-1).
(2) Tweedie: K m = (I - S C^-1) m = m + S grad log N(m; 0, C).
(3) Fixed point of the sweep for fixed Pi is C (C + R)^{-1} o (exact posterior mean),
    and K m* = (C - S)(C + R)^{-1} o (exact posterior mean of the block latents).
(4) Block decomposition: for Pi with first level A = 12|34,
    log N(m; 0, C_Pi) = W_A(m) + log N(mhat; 0, C^par_Pi2) with W_A independent of Pi2,
    mhat_B = sum_{i in B} s_i^-2 m_i / lambda_B, parent child noise 1/lambda_B.
(5) Type closure: the update of mhat_B equals a_B ohat_B + b_B zbar_B with
    b_B = sum s_i^-4 eta_i / lambda_B, ohat_B = sum alpha_i o_i / sum alpha_i,
    alpha_i = s_i^-2 eta_i rho_i^-2, and the law of yhat_B has variance eta_B = b_B/lambda_B.
(6) Finest structure: the full sweep equals the private-copy (label-preserving) null sweep.
"""
from __future__ import annotations

import sympy as sp

R = sp.Rational
n = 4
s2 = [R(1), R(3, 2), R(1, 2), R(2)]
r2 = [R(1, 4), R(1, 3), R(1, 2), R(1, 5)]
tau2, sig2 = R(1, 10), R(4)
S = sp.diag(*s2)
Rm = sp.diag(*r2)
eta = [1 / (1 / s2[i] + 1 / r2[i]) for i in range(n)]
a = [s2[i] / (s2[i] + r2[i]) for i in range(n)]
b = [1 - a[i] for i in range(n)]
Adiag, Bdiag = sp.diag(*a), sp.diag(*b)


def comem(blocks):
    M = sp.zeros(n, n)
    for B in blocks:
        for i in B:
            for j in B:
                M[i, j] = 1
    return M


A1 = [(0, 1), (2, 3)]
structs = {
    "sep": (A1, A1),
    "join": (A1, [(0, 1, 2, 3)]),
    "finest": ([(0,), (1,), (2,), (3,)], [(0,), (1,), (2,), (3,)]),
}
C = {k: S + tau2 * comem(v[0]) + sig2 * comem(v[1]) for k, v in structs.items()}

m = sp.Matrix(sp.symbols("m1:5"))
o = sp.Matrix(sp.symbols("o1:5"))
Sv = sp.symbols("S1:5", positive=True)
ok = True

for k, Ck in C.items():
    Ci = Ck.inv()
    K = sp.eye(n) - S * Ci
    # (2) Tweedie
    logN = -R(1, 2) * (m.T * Ci * m)[0]
    grad = sp.Matrix([sp.diff(logN, mi) for mi in m])
    ok &= sp.simplify(K * m - (m + S * grad)) == sp.zeros(n, 1)
    # (1) collapsed free energy gradient step
    Fbar = sum(((o[i] - m[i]) ** 2 + Sv[i]) / (2 * r2[i]) for i in range(n)) - logN \
        + sum(Sv[i] / (2 * s2[i]) for i in range(n))
    gF = sp.Matrix([sp.diff(Fbar, mi) for mi in m])
    sweep = Adiag * o + Bdiag * K * m
    ok &= sp.simplify(sweep - (m - sp.diag(*eta) * gF)) == sp.zeros(n, 1)
    # (3) fixed point
    mstar = (sp.eye(n) - Bdiag * K).inv() * Adiag * o
    ok &= sp.simplify(mstar - Ck * (Ck + Rm).inv() * o) == sp.zeros(n, 1)
    ok &= sp.simplify(K * mstar - (Ck - S) * (Ck + Rm).inv() * o) == sp.zeros(n, 1)
print("(1)-(3) collapsed step, Tweedie, fixed point:", ok)

# (1') the Q_nu minimization: E_{Q_y}[-log p(y|u)] = -log N(m; u, S) + sum S_i/(2 s_i^2)
u = sp.Matrix(sp.symbols("u1:5"))
y = sp.symbols("y")
ok1 = True
for i in range(n):
    expr = sp.integrate((y - u[i]) ** 2 / (2 * s2[i]) * sp.exp(-(y - m[i]) ** 2 / (2 * Sv[i])) / sp.sqrt(2 * sp.pi * Sv[i]),
                        (y, -sp.oo, sp.oo))
    ok1 &= sp.simplify(expr - ((m[i] - u[i]) ** 2 / (2 * s2[i]) + Sv[i] / (2 * s2[i]))) == 0
print("(1') expected child log-likelihood depends on Q_i through (m_i, S_i) only, S-term Pi-free:", ok1)

# (4) block decomposition with heterogeneous noise
lam = [s2[0] ** -1 + s2[1] ** -1, s2[2] ** -1 + s2[3] ** -1]
mhat = sp.Matrix([(m[0] / s2[0] + m[1] / s2[1]) / lam[0], (m[2] / s2[2] + m[3] / s2[3]) / lam[1]])
Cpar = {"sep": sp.diag(1 / lam[0], 1 / lam[1]) + (tau2 + sig2) * sp.eye(2),
        "join": sp.diag(1 / lam[0], 1 / lam[1]) + tau2 * sp.eye(2) + sig2 * sp.ones(2, 2)}


def logN_full(Ck, x):
    return -R(1, 2) * (x.T * Ck.inv() * x)[0] - R(1, 2) * sp.log(Ck.det())


diffs = []
for k in ("sep", "join"):
    W = sp.simplify(logN_full(C[k], m) - logN_full(Cpar[k], mhat))
    diffs.append(W)
ok4 = sp.simplify(diffs[0] - diffs[1]) == 0
print("(4) within-block term independent of Pi2:", ok4)

# (5) type closure for block B1 = {1,2} under any Pi with first level A
zb = sp.symbols("zbar")
upd = [eta[i] * (o[i] / r2[i] + zb / s2[i]) for i in (0, 1)]
mhat_new = (upd[0] / s2[0] + upd[1] / s2[1]) / lam[0]
alpha = [eta[i] / (s2[i] * r2[i]) for i in (0, 1)]
bB = (eta[0] / s2[0] ** 2 + eta[1] / s2[1] ** 2) / lam[0]
aB = 1 - bB
ohat = (alpha[0] * o[0] + alpha[1] * o[1]) / (alpha[0] + alpha[1])
ok5 = sp.simplify(mhat_new - (aB * ohat + bB * zb)) == 0
sB2 = 1 / lam[0]
rhoB2 = sB2 * bB / (1 - bB)
etaB = 1 / (1 / sB2 + 1 / rhoB2)
var_yhat = (eta[0] / s2[0] ** 2 + eta[1] / s2[1] ** 2) / lam[0] ** 2
ok5 &= sp.simplify(etaB - var_yhat) == 0
ok5 &= sp.simplify(aB - sB2 / (sB2 + rhoB2)) == 0
ok5 &= sp.simplify(sum(alpha) - lam[0] * aB) == 0
print("(5) parent update a_B ohat_B + b_B zbar_B, parent variance eta_B:", ok5)

# (6) finest structure = private-copy null
Kf = sp.eye(n) - S * C["finest"].inv()
kpriv = sp.diag(*[(tau2 + sig2) / (s2[i] + tau2 + sig2) for i in range(n)])
ok6 = sp.simplify(Kf - kpriv) == sp.zeros(n, n)
v = [s2[i] + tau2 + sig2 for i in range(n)]
kap = [v[i] / (v[i] + r2[i]) for i in range(n)]
mfix = (sp.eye(n) - Bdiag * kpriv).inv() * Adiag * o
ok6 &= sp.simplify(mfix - sp.diag(*kap) * o) == sp.zeros(n, 1)
print("(6) finest sweep = label-preserving null sweep; both nulls fix kappa o:", ok6)

# (7) mean-field variance vs exact posterior variance at fixed Pi (join)
Cj = C["join"]
post = Cj - Cj * (Cj + Rm).inv() * Cj
print("(7) exact posterior variances (join):", [sp.nsimplify(post[i, i]) for i in range(n)])
print("    mean-field variances eta_i      :", eta)
print("    exact - eta >= 0:", all(post[i, i] - eta[i] >= 0 for i in range(n)))
