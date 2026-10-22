"""X1-GAUGE: geo:collective normal equation and decomposition, geo:retained-connection
counterexample, geo:quotient counterexample, and the curvature claims of open:curvature
(ch12) and form:curvature (ch11) under TIED versus UNTIED belief/model structure groups."""
import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import quad

rng = np.random.default_rng(3)

# geo:collective on random linear data: tangent dim n, algebra dim k with a kernel
n, k = 6, 3
J = rng.normal(size=(n, k)); J[:, 2] = J[:, 0] + J[:, 1]            # ker J nontrivial
Gm = rng.normal(size=(n, n)); Gm = Gm @ Gm.T + np.eye(n)              # positive metric
dp = rng.normal(size=n)
kfrak = J.T @ Gm @ J
AF = np.linalg.lstsq(kfrak, -J.T @ Gm @ dp, rcond=None)[0]
r = dp + J @ AF
print("normal equation G(r, J xi) =", np.abs(J.T @ Gm @ r).max())
A = rng.normal(size=k)
lhs = (dp + J @ A) @ Gm @ (dp + J @ A)
rhs = r @ Gm @ r + (A - AF) @ kfrak @ (A - AF)
print("h^A = h^min + k(A-A^F,A-A^F) residual =", abs(lhs - rhs))
# projection characterization r = P_H dp
P = J @ np.linalg.pinv(J.T @ Gm @ J) @ J.T @ Gm
print("r = P_H dp residual =", np.abs(r - (dp - P @ dp)).max())

# geo:retained-connection counterexample: a = 1/3 after noise on first sample
a_grid = np.linspace(-1, 1, 200001)
cost = 0.5 * (1 + a_grid) ** 2 + (-1 + a_grid) ** 2
print("retained-connection counterexample argmin a =", a_grid[np.argmin(cost)])

# geo:quotient: A = dz - x dy, null fields d_x and d_y + x d_z, bracket d_z, A(d_z) = 1
print("quotient counterexample: A([d_x, d_y + x d_z]) = A(d_z) = 1 (non-involutive radical)")

# ---------- ch12 von Mises inertia and curvature ----------
def Iv(kap):          # rotation inertia E[(d_theta log p)^2] = kappa I1/I0
    return kap * i1e(kap) / i0e(kap)
def Iv_quad(kap):
    f = lambda th: (kap * np.sin(th)) ** 2 * np.exp(kap * np.cos(th)) / (2 * np.pi * np.i0(kap))
    return quad(f, -np.pi, np.pi)[0]
print("von Mises inertia closed form vs quadrature:", Iv(1.7), Iv_quad(1.7))

def tied(kap, lam, rho, nblk=2):
    a, b = Iv(rho * kap), Iv(rho * lam)
    AF = -a / (a + b)                      # coefficient of d chi
    hcc = nblk * a * b / (a + b)
    return AF, hcc
kap, lam, rho = 1.3, 0.8, np.exp(-2 * 0.4)
eps = 1e-5
AF_p, _ = tied(kap + eps, lam, rho); AF_m, _ = tied(kap - eps, lam, rho)
F_num = (AF_p - AF_m) / (2 * eps)          # F = d_kappa A_chi dkappa ^ dchi
dI = (Iv(rho * kap + 1e-6) - Iv(rho * kap - 1e-6)) / 2e-6
a, b = Iv(rho * kap), Iv(rho * lam)
F_formula = -b * rho * dI / (a + b) ** 2
print("ch12 tied: A_chi, h_chichi =", tied(kap, lam, rho), " F numeric", F_num, " F formula", F_formula)
# untied: separate S^1_q and S^1_s; minimize a (1 + xq)^2 + b xs^2 independently
xq, xs = -1.0, 0.0
print("ch12 UNTIED: A_q,chi = -1, A_s,chi = 0 (kappa-independent) -> F = 0; h_chichi =",
      2 * (a * (1 + xq) ** 2 + b * xs ** 2))

# ---------- ch11 Wishart SO(2) inertia ----------
Om = np.array([[0., -1.], [1., 0.]])
def inertia_prec(P, alpha):
    U = Om @ P - P @ Om
    Pi = np.linalg.inv(P)
    return alpha * np.trace(Pi @ U @ Pi @ U)
pp, pm, alpha = 2.5, 0.7, 3.0
P = np.diag([pp, pm])
print("ch11 SO(2) inertia: Fisher", inertia_prec(P, alpha), " formula 2A(p+-p-)^2/(p+p-)", 2 * alpha * (pp - pm) ** 2 / (pp * pm))
# rotation-invariance of radial/orbit orthogonality: diag variation vs orbit variation
Ud = np.diag([0.3, -0.2]); Uo = Om @ P - P @ Om
print("radial-orbit Fisher pairing:", alpha * np.trace(np.linalg.inv(P) @ Ud @ np.linalg.inv(P) @ Uo))
# tied: A = -k1/(k1+k2) dchi ; untied: A_q = -dchi, A_s = 0 -> flat, h_chichi = 0
k1 = inertia_prec(P, alpha); k2 = inertia_prec(np.diag([1.8, 0.9]), 2.0)
print("ch11 tied A_chi =", -k1 / (k1 + k2), " h_chichi =", k1 * k2 / (k1 + k2), "; untied h_chichi = 0, F = 0")
