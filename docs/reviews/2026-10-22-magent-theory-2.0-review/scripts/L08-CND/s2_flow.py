r"""L08-CND check 2: complete moment-channel derivative, Fisher flow, parent flow.

Fine belief energy (N agents, p=2, separate precisions P_i allowed):
  E_q = sum_i [ KL(W(alpha_i,P_i) || W(alpha0,I)) + KL(W(alpha_i,P_i) || W(alpha0,Lambda)) ],
  Lambda = I + (kappa/N) M,  M = sum_j alpha_j P_j^{-1}.
Claimed flow: alpha_i' = -2(alpha_i-alpha0),  P_i' = -2 P_i + I + C,
  C = Lambda + (kappa/N)(M - N alpha0 Lambda^{-1}).
Check 1: natural gradient xi' = -G^{-1} grad E in coords (alpha_i, p11_i, p12_i, p22_i), with G the
         blockdiag Hessian of Psi_i = log Gamma_2(alpha_i) - alpha_i log det P_i (closed form), and
         grad E by mpmath central differences (30 digits).
Check 2: tangency of the unrestricted density field (symbolic in X).
Check 3: parent energy with blocks (n_I alpha0 references) gives the same C and A_I' = -2(A_I-n_I alpha0).
Check 4: p=1 (Gamma) fully symbolic natural gradient.
"""
from __future__ import annotations

import mpmath as mp
import sympy as sp

mp.mp.dps = 30
N = 4
kap = mp.mpf(3) / 2
al0 = mp.mpf(6) / 5


def lG2(x):
    return mp.log(mp.sqrt(mp.pi)) + mp.loggamma(x) + mp.loggamma(x - mp.mpf(1) / 2)


def psi2(x):
    return mp.digamma(x) + mp.digamma(x - mp.mpf(1) / 2)


def tr(A):
    return A[0, 0] + A[1, 1]


def KL(a, P, b, Q):
    r"""KL(W(a,P)||W(b,Q)) = (a-b)psi_2(a) - lG2(a) + lG2(b) + b log det(P Q^{-1}) + a(Tr(Q P^{-1})-2)."""
    return ((a - b) * psi2(a) - lG2(a) + lG2(b) + b * (mp.log(mp.det(P)) - mp.log(mp.det(Q)))
            + a * (tr(Q * mp.inverse(P)) - 2))


def unpack(xi, n):
    al = [xi[4 * i] for i in range(n)]
    Ps = [mp.matrix([[xi[4 * i + 1], xi[4 * i + 2]], [xi[4 * i + 2], xi[4 * i + 3]]]) for i in range(n)]
    return al, Ps


def moment(al, Ps):
    M = mp.zeros(2, 2)
    for a, P in zip(al, Ps):
        M += a * mp.inverse(P)
    return M


def energy(xi, refs, n, frozen_M=None):
    al, Ps = unpack(xi, n)
    M = moment(al, Ps) if frozen_M is None else frozen_M
    Lam = mp.eye(2) + kap / N * M
    return sum(KL(al[i], Ps[i], refs[i], mp.eye(2)) + KL(al[i], Ps[i], refs[i], Lam) for i in range(n))


def grad(f, xi, h=mp.mpf("1e-10")):
    g = []
    for k in range(len(xi)):
        xp = list(xi)
        xm = list(xi)
        xp[k] += h
        xm[k] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


def fisher_block(a, P):
    r"""Hessian of Psi = lG2(a) - a log det P in coords (a, p11, p12, p22), closed form."""
    Pi = mp.inverse(P)
    E = [mp.matrix([[1, 0], [0, 0]]), mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, 0], [0, 1]])]
    G = mp.zeros(4, 4)
    G[0, 0] = mp.psi(1, a) + mp.psi(1, a - mp.mpf(1) / 2)
    for r in range(3):
        G[0, r + 1] = -tr(Pi * E[r])
        G[r + 1, 0] = G[0, r + 1]
        for s in range(3):
            G[r + 1, s + 1] = a * tr(Pi * E[r] * Pi * E[s])
    return G


def fisher(xi, n):
    al, Ps = unpack(xi, n)
    G = mp.zeros(4 * n, 4 * n)
    for i in range(n):
        B = fisher_block(al[i], Ps[i])
        for r in range(4):
            for s in range(4):
                G[4 * i + r, 4 * i + s] = B[r, s]
    return G


def natural_velocity(xi, refs, n, frozen_M=None):
    g = grad(lambda z: energy(z, refs, n, frozen_M), xi)
    return -(mp.lu_solve(fisher(xi, n), mp.matrix(g)))


def claimed_velocity(xi, refs, n):
    al, Ps = unpack(xi, n)
    M = moment(al, Ps)
    Lam = mp.eye(2) + kap / N * M
    C = Lam + kap / N * (M - N * al0 * mp.inverse(Lam))
    C2 = mp.eye(2) + 2 * kap / N * M - kap * al0 * mp.inverse(Lam)
    assert mp.norm(C - C2) < mp.mpf("1e-25")
    out = []
    for i in range(n):
        Pd = -2 * Ps[i] + mp.eye(2) + C
        out += [-2 * (al[i] - refs[i]), Pd[0, 0], Pd[0, 1], Pd[1, 1]]
    return mp.matrix(out)


def maxdiff(u, v):
    return max(abs(u[k] - v[k]) for k in range(len(u)))


vals_alpha = [mp.mpf("0.9"), mp.mpf("1.3"), mp.mpf("2.2"), mp.mpf("3.1")]
vals_P = [["1.4", "0.3", "0.8"], ["0.9", "-0.2", "1.2"], ["2.0", "0.5", "1.0"], ["0.7", "0.1", "0.6"]]
xi = []
for i in range(N):
    xi += [vals_alpha[i]] + [mp.mpf(v) for v in vals_P[i]]
refs = [al0] * N
v_nat = natural_velocity(xi, refs, N)
v_cl = claimed_velocity(xi, refs, N)
print("Check 1 (fine, unequal P_i, N=4, p=2): max |natural - claimed| =", mp.nstr(maxdiff(v_nat, v_cl), 5),
      " scale", mp.nstr(max(abs(v_cl[k]) for k in range(4 * N)), 5))
al_, Ps_ = unpack(xi, N)
v_fr = natural_velocity(xi, refs, N, frozen_M=moment(al_, Ps_))
print("    frozen-target flow minus claimed, max:", mp.nstr(maxdiff(v_fr, v_cl), 5))

# ---------- Check 3: parents on the common slice ----------
Pc = [mp.mpf("1.4"), mp.mpf("0.3"), mp.mpf("0.8")]
xi_fine_c = []
for i in range(N):
    xi_fine_c += [vals_alpha[i]] + Pc
v_fine = natural_velocity(xi_fine_c, refs, N)
xi_par = [vals_alpha[0] + vals_alpha[1]] + Pc + [vals_alpha[2] + vals_alpha[3]] + Pc
v_par = natural_velocity(xi_par, [2 * al0, 2 * al0], 2)
print("Check 3 parents {12},{34}: A velocities", mp.nstr(v_par[0], 12), mp.nstr(v_par[4], 12),
      "| summed fine", mp.nstr(v_fine[0] + v_fine[4], 12), mp.nstr(v_fine[8] + v_fine[12], 12),
      "| claimed", mp.nstr(-2 * (xi_par[0] - 2 * al0), 12), mp.nstr(-2 * (xi_par[4] - 2 * al0), 12))
print("    parent P velocity minus fine P velocity, max:",
      mp.nstr(max(abs(v_par[1 + r] - v_fine[1 + r]) for r in range(3)), 5),
      "; spread of fine P velocities across agents:",
      mp.nstr(max(abs(v_fine[4 * i + 1 + r] - v_fine[1 + r]) for i in range(N) for r in range(3)), 5))
xi_root = [sum(vals_alpha)] + Pc
v_root = natural_velocity(xi_root, [4 * al0], 1)
print("    root parent: A' =", mp.nstr(v_root[0], 12), " claimed", mp.nstr(-2 * (xi_root[0] - 4 * al0), 12),
      "; P' diff vs fine:", mp.nstr(max(abs(v_root[1 + r] - v_fine[1 + r]) for r in range(3)), 5))
# Mis-specified parent: channel built from the parent count |Pi| instead of primitive N.
# (Illustrates that the primitive count is load-bearing.)

# ---------- Check 2: symbolic tangency ----------
x11, x12, x22 = sp.symbols("x11 x12 x22")
X = sp.Matrix([[x11, x12], [x12, x22]])
a, a0s = sp.symbols("alpha alpha0")
q11, q12, q22, c11, c12, c22 = sp.symbols("q11 q12 q22 c11 c12 c22")
Pg = sp.Matrix([[q11, q12], [q12, q22]])
Cg = sp.Matrix([[c11, c12], [c12, c22]])
I2 = sp.eye(2)
logq_stat = (a - sp.Rational(3, 2)) * sp.log(X.det()) - (Pg * X).trace()
h = 2 * logq_stat - 2 * (a0s - sp.Rational(3, 2)) * sp.log(X.det()) + ((I2 + Cg) * X).trace()
dlogq_stat = -2 * (a - a0s) * sp.log(X.det()) - ((-2 * Pg + I2 + Cg) * X).trace()
print("Check 2 (X-dependent part of -h minus d/dt log q):", sp.simplify(sp.expand(-h - dlogq_stat)))

# ---------- Check 4: p = 1 symbolic ----------
al_s = sp.symbols("b1:5", positive=True)
p_s = sp.symbols("r1:5", positive=True)
k_s, a0_s = sp.symbols("kappa alpha0", positive=True)


def KL1(a_, p_, b_, q_):
    return (a_ - b_) * sp.polygamma(0, a_) - sp.loggamma(a_) + sp.loggamma(b_) + b_ * sp.log(p_ / q_) + a_ * (q_ / p_ - 1)


M1 = sum(al_s[i] / p_s[i] for i in range(N))
L1 = 1 + k_s / N * M1
E1 = sum(KL1(al_s[i], p_s[i], a0_s, 1) + KL1(al_s[i], p_s[i], a0_s, L1) for i in range(N))
C1 = L1 + k_s / N * (M1 - N * a0_s / L1)
ok = True
for i in range(N):
    Psi = sp.loggamma(al_s[i]) - al_s[i] * sp.log(p_s[i])
    Gi = sp.hessian(Psi, [al_s[i], p_s[i]])
    gi = sp.Matrix([sp.diff(E1, al_s[i]), sp.diff(E1, p_s[i])])
    vel = -(Gi.inv() * gi)
    r1 = sp.simplify(vel[0] + 2 * (al_s[i] - a0_s))
    r2 = sp.simplify(vel[1] - (-2 * p_s[i] + 1 + C1))
    ok = ok and (r1 == 0) and (r2 == 0)
print("Check 4 (p=1 Gamma, symbolic, N=4, unequal rates): all residuals zero:", ok)
