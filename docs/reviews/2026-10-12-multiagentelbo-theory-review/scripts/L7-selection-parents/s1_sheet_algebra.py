r"""L7 check 1: sheet algebra of the four-cycle selection proof.

Builds the 12x12 drift matrices A_sigma = H_sigma + T(I + R_ev) of the
homogeneous four-cycle (c13 def:rg-formation-cycle-experiment), and checks
  * the determinant formulas d_M, d_E, d_N, d_C (ca_selection_proofs.tex:560-582),
  * the leading asymptotics of r_E, r_N, r_C and the threshold T0 (r <= 1/8),
  * the saddle Jacobian of the reduced field V(p) = beta_bar(p) - p and its
    eigenvalues alpha, beta, eta (eq:phil-selection-saddle-spectrum),
  * vertex Jacobian (r_E - 1) I,
  * active determinant bounds and constants of prop:phil-temperature-tracking,
  * r_E, r_N <= 5T/2, r_C <= 1.5 T^{3/2}, departure-rate bound 31T
    (ca_selected_coherence.tex:129-154),
  * normal rates 8+T, 4+5T of A_M, physical 8+T, 4+T; center path residuals.
"""
from __future__ import annotations

import itertools
import numpy as np
import sympy as sp

np.set_printoptions(precision=6, suppress=True)

# coordinates x = (kbar_0..3, mbar_0..3, omega_0..3); edge e_i joins i -> i+1
def residual_covectors() -> tuple[np.ndarray, np.ndarray]:
    ab = np.zeros((4, 12)); am = np.zeros((4, 12))
    for i in range(4):
        j = (i + 1) % 4
        ab[i, j] += 1; ab[i, i] -= 1; ab[i, 8 + i] -= 1
        am[i, 4 + j] += 1; am[i, 4 + i] -= 1; am[i, 8 + i] -= 1
    return ab, am

AB, AM = residual_covectors()
Dfr = np.zeros((4, 12)); Dfr[:, :4] = np.eye(4); Dfr[:, 4:8] = -np.eye(4)
R_ev = 2 * Dfr.T @ Dfr  # = D^T D /(1 - sigma_ev^2), sigma_ev^2 = 1/2
Omega = np.eye(12) + R_ev

def mult(sig: tuple[int, ...]) -> np.ndarray:
    """m_e = 1{sigma_e=+1} + 1{sigma_{e+1}=-1}."""
    return np.array([(sig[e] == 1) + (sig[(e + 1) % 4] == -1) for e in range(4)], float)

def H_of(m: np.ndarray) -> np.ndarray:
    # V = sum_e m_e * 1/2 [(r^b_e)^2 + (r^m_e)^2]
    return sum(m[e] * (np.outer(AB[e], AB[e]) + np.outer(AM[e], AM[e])) for e in range(4))

def A_of(sig, T):
    return H_of(mult(sig)) + T * Omega

sigmas = list(itertools.product([1, -1], repeat=4))

def dets_formula(T):
    dM = 25 * T**8 * (T + 8)**2 * (5*T + 4)**2
    dE = 5 * T**6 * (T**3 + 16*T**2 + 77*T + 112) * (125*T**3 + 200*T**2 + 85*T + 8)
    dN = 5 * T**6 * (T + 4) * (5*T + 2) * (T**2 + 12*T + 28) * (25*T**2 + 30*T + 4)
    dC = 5 * T**5 * (T + 2) * (T + 4)**2 * (T + 6) * (5*T + 2)**2 * (5*T + 4)
    return dM, dE, dN, dC

def classify(sig):
    # matching coordinates x_i = 1 if receiver i picks the even edge
    x = tuple(int((sig[i] == 1) if i % 2 == 0 else (sig[i] == -1)) for i in range(4))
    k = sum(x)
    if k in (0, 4):
        return "M"
    if k in (1, 3):
        return "E"
    zeros = [i for i in range(4) if x[i] == 1]
    return "C" if (zeros[1] - zeros[0]) == 2 else "N"

print("== determinant formulas (relative residual, max over 16 sheets, 3 temperatures)")
for T in [0.003, 0.05, 0.7]:
    f = dict(zip("MENC", dets_formula(T)))
    worst = 0.0
    for s in sigmas:
        d = np.linalg.det(A_of(s, T))
        worst = max(worst, abs(d / f[classify(s)] - 1))
    print(f"T={T}: max rel residual {worst:.2e}")

# symbolic determinant check on representatives
Ts = sp.symbols("T", positive=True)
def A_sym(m):
    H = sp.Matrix(H_of(np.array(m, float)).astype(int))
    return H + Ts * sp.Matrix(Omega.astype(int))
dM, dE, dN, dC = dets_formula(Ts)
# det A is a polynomial of degree <= 12 in T; exact agreement at 13 rational
# points proves the polynomial identity.
for name, m, d in [("M", (2,0,2,0), dM), ("E", (2,1,1,0), dE), ("N", (1,2,1,0), dN), ("C", (1,1,1,1), dC)]:
    Am = A_sym(m)
    bad = 0
    for k in range(1, 14):
        Tk = sp.Rational(k, 7)
        if sp.simplify(Am.subs(Ts, Tk).det(method="bareiss") - d.subs(Ts, Tk)) != 0:
            bad += 1
    print(f"exact det identity {name} {m}: mismatches at 13 rational T = {bad}")
counts = {c: sum(classify(s) == c for s in sigmas) for c in "MENC"}
print("class counts", counts)

def rs(T):
    dM, dE, dN, dC = dets_formula(T)
    return np.sqrt(dM / dE), np.sqrt(dM / dN), np.sqrt(dM / dC)

print("== asymptotics r/T and r_C/T^1.5 at T=1e-5:", np.array(rs(1e-5)) / np.array([1e-5, 1e-5, 1e-5**1.5]),
      "targets", np.sqrt(40/7), np.sqrt(40/7), np.sqrt(5/3))
from scipy.optimize import brentq
T0 = min(brentq(lambda T, k=k: rs(T)[k] - 0.125, 1e-6, 5) for k in range(3))
print(f"T0 (all r <= 1/8): {T0:.5f}; r at T0: {rs(T0)}")

# reduced field V(p) = beta_bar(p) - p with weights w_sigma = det(A/T)^-1/2
def wvec(T):
    return np.array([np.linalg.det(A_of(s, T) / T) ** -0.5 for s in sigmas])

def V(p, w):
    pis = np.array([np.prod([p[i] if s[i] == 1 else 1 - p[i] for i in range(4)]) for s in sigmas])
    Z = pis @ w
    bb = np.array([sum(pis[k] * w[k] for k, s in enumerate(sigmas) if s[i] == 1) / Z for i in range(4)])
    return bb - p

print("== saddle Jacobian and spectrum")
for T in [0.01, 0.03, T0]:
    w = wvec(T); h = 1e-6
    J = np.array([(V(0.5 + h*np.eye(4)[j], w) - V(0.5 - h*np.eye(4)[j], w)) / (2*h) for j in range(4)]).T
    rE, rN, rC = rs(T); D0 = 1 + 4*rE + 2*rN + rC
    al = (3 - 2*rN - rC)/D0; be = (-1 - 2*rN + 3*rC)/D0; et = (-1 + 2*rN - rC)/D0
    ev = np.sort(np.linalg.eigvalsh((J + J.T)/2))
    print(f"T={T:.4f} eig(J)={ev}, formula={np.sort([al,be,et,et])}, |J-J^T|={np.abs(J-J.T).max():.1e}, alpha>=7/5:{al>=1.4}, beta<=-1/3:{be<=-1/3}, eta<=-2/5:{et<=-0.4}")
    v = np.array([1, -1, 1, -1.]); print("   J v / v =", (J @ v) / v)
    # vertex Jacobian at S^(0): p = (1,0,1,0)
    pv = np.array([1, 0, 1, 0.])
    Jv = np.array([(V(pv + h*np.eye(4)[j]*np.sign(0.5 - pv[j]), w) - V(pv, w)) / (h*np.sign(0.5 - pv[j])) for j in range(4)]).T
    print("   vertex Jacobian diag", np.diag(Jv), "expected r_E-1 =", rE - 1, "offdiag max", np.abs(Jv - np.diag(np.diag(Jv))).max())

print("== temperature-tracking constants (T <= 1/100)")
Tt = 0.01
dM, dE, dN, dC = dets_formula(Tt)
act = lambda d: d / (5 * Tt**5)
print("active dets / scale:", act(dM)/Tt**3, act(dE)/Tt, act(dN)/Tt, act(dC), "claims [5120,5262], [896,1017], [896,2175], [3072,5120)")
for T in [1e-6, 1e-4, 1e-3, 0.01]:
    dM, dE, dN, dC = dets_formula(T)
    a = [d/(5*T**5) for d in (dM, dE, dN, dC)]
    print(f"  T={T}: dM/T^3={a[0]/T**3:.1f} dE/T={a[1]/T:.1f} dN/T={a[2]/T:.1f} dC={a[3]:.1f}")
K0 = (2*np.pi)**3.5 / ((np.pi**3/840) * np.sqrt(5120))
print("K0", K0, "210 sqrt(2pi/5) =", 210*np.sqrt(2*np.pi/5))
from math import gamma, pi
print("vol 7-ball r=1/2:", pi**3.5 * 0.5**7 / gamma(4.5), "pi^3/840 =", pi**3/840)
Tg = np.geomspace(1e-8, 0.01, 200)
r_all = np.array([rs(T) for T in Tg])
print("max r_E/T, r_N/T, r_C/T^1.5 on (1e-8, 0.01]:", (r_all[:,0]/Tg).max(), (r_all[:,1]/Tg).max(), (r_all[:,2]/Tg**1.5).max(), "claims 2.5, 2.5, 1.5")
dep = 8*r_all[:,0] + 4*r_all[:,1] + 2.5*r_all[:,2]
print("max (8rE+4rN+2.5rC)/T:", (dep/Tg).max(), "claim <= 31")
kap = (8*r_all[:,0] + 4*r_all[:,1] + 2*r_all[:,2]) / (2 + 8*r_all[:,0] + 4*r_all[:,1] + 2*r_all[:,2])
print("max nonmatching mass / T:", (kap/Tg).max(), "claim <= 16")
# adjacent weight ratios <= 1/T (one-receiver flips)
mx = 0
for T in Tg:
    dM, dE, dN, dC = dets_formula(T)
    wM, wE, wN, wC = [d**-0.5 for d in (dM, dE, dN, dC)]
    mx = max(mx, T*max(wM/wE, wE/wM, wE/wN, wN/wE, wE/wC, wC/wE))
print("max T * adjacent weight ratio:", mx, "claim <= 1")
# g bound |d log Z/dp_i| <= 1/T directly
mg = 0
rng = np.random.default_rng(0)
for T in [1e-4, 1e-3, 0.01]:
    w = wvec(T)
    for _ in range(300):
        p = rng.uniform(1e-6, 1 - 1e-6, 4)
        pis = np.array([np.prod([p[i] if s[i] == 1 else 1 - p[i] for i in range(4)]) for s in sigmas])
        Z = pis @ w
        g = [(sum(pis[k]*w[k]/p[i] for k, s in enumerate(sigmas) if s[i] == 1) - sum(pis[k]*w[k]/(1-p[i]) for k, s in enumerate(sigmas) if s[i] == -1)) / Z for i in range(4)]
        mg = max(mg, T*np.max(np.abs(g)))
print("max T*|g_i| over random rows:", mg, "claim <= 1")

print("== normal rates of the matching sheet, physical rates, ||H||")
T = 0.01
s0 = (1, -1, 1, -1)
AMm = A_of(s0, T)
print("eig A_M (nonpassive part):", np.sort(np.linalg.eigvalsh(AMm))[-4:])
Hm = H_of(mult(s0))
print("eig H_M + T I (top 4):", np.sort(np.linalg.eigvalsh(Hm + T*np.eye(12)))[-4:])
print("max ||H_sigma|| =", max(np.linalg.norm(H_of(mult(s)), 2) for s in sigmas), "claim <= 12")
# center path from x0 = (0,0,1_4) under S^(0)
from scipy.linalg import expm
x0 = np.r_[np.zeros(8), np.ones(4)]
for t in [0.1, 0.5, 2.0]:
    x = expm(-AMm*t) @ x0
    r = AB @ x
    print(f"t={t}: residuals {r}, int pred {-np.exp(-(8+T)*t):.6f}, ext pred {np.exp(-T*t)*(-1.5+np.exp(-8*t)/2):.6f}")
