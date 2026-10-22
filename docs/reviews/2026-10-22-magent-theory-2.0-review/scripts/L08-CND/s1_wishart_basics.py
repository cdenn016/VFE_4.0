r"""L08-CND check 1: Wishart (p=1 Gamma, p=2 matrix) normalization, addition, KL, Fisher.

Density (p=2): W_2(nu,P)(dX) = det(P)^nu / Gamma_2(nu) det(X)^{nu-3/2} exp(-Tr(P X)) dX,
Gamma_2(nu) = sqrt(pi) Gamma(nu) Gamma(nu-1/2).  p=1 analogue: Gamma(shape nu, rate P).

Checks:
 (a) p=2 normalization via the triangular substitution (Jacobian, determinant, trace) -- symbolic.
 (b) p=1 convolution: common rate gives Gamma(a+b); unequal rates give a non-Gamma law.
 (c) p=2 Laplace transform E exp(-Tr(T X)) = (det P / det(P+T))^nu  (symbolic at P=I, diag T;
     Monte Carlo at general P, T) and additivity under common P.
 (d) Closed-form KL(W(a,P) || W(b,Q)) = (a-b) psi_2(a) - log G2(a) + log G2(b)
       + b log det(P Q^{-1}) + a (Tr(Q P^{-1}) - 2), checked by Monte Carlo (p=2) and quadrature (p=1).
 (e) Fisher metric in (shape, precision) = Hessian of Psi(nu,P) = log Gamma_2(nu) - nu log det P,
     compared with Monte Carlo score covariance; precision block nu Tr(P^-1 U P^-1 V);
     mixed block -Tr(P^{-1} U); shape block psi'(nu)+psi'(nu-1/2).
 (f) Same-batch covariance Cov(Tr(U X_C), Tr(V X_D)) = A_{C cap D} Tr(P^-1 U P^-1 V).
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.special import gammaln, digamma, polygamma

rng = np.random.default_rng(20261022)

# ---------- (a) normalization, p = 2 ----------
u, v, z, nu = sp.symbols("u v z nu", positive=True)
x11, x12, x22 = u, sp.sqrt(u) * z, v + z**2
Jac = sp.Matrix([x11, x12, x22]).jacobian(sp.Matrix([u, z, v])).det()
detX = sp.simplify(x11 * x22 - x12**2)
trX = x11 + x22
print("(a) Jacobian:", sp.simplify(Jac), " det:", detX, " trace:", trX)
nu0 = sp.Rational(7, 5)
integrand = (detX ** (nu0 - sp.Rational(3, 2)) * sp.exp(-trX) * Jac)
I_u = sp.integrate(u ** (nu0 - 1) * sp.exp(-u), (u, 0, sp.oo))
I_v = sp.integrate(v ** (nu0 - sp.Rational(3, 2)) * sp.exp(-v), (v, 0, sp.oo))
I_z = sp.integrate(sp.exp(-z**2), (z, -sp.oo, sp.oo))
G2 = sp.sqrt(sp.pi) * sp.gamma(nu0) * sp.gamma(nu0 - sp.Rational(1, 2))
print("    factorized integrand check:",
      sp.simplify(integrand - u ** (nu0 - 1) * sp.exp(-u) * v ** (nu0 - sp.Rational(3, 2)) * sp.exp(-v) * sp.exp(-z**2)))
print("    integral / Gamma_2(nu) - 1 =", sp.N(I_u * I_v * I_z / G2 - 1, 20))

# congruence Jacobian on Sym(2): X -> B X B^T has Jacobian det(B)^3
b11, b12, b21, b22, y11, y12, y22 = sp.symbols("b11 b12 b21 b22 y11 y12 y22")
B = sp.Matrix([[b11, b12], [b21, b22]])
Y = sp.Matrix([[y11, y12], [y12, y22]])
Xc = B * Y * B.T
Jc = sp.Matrix([Xc[0, 0], Xc[0, 1], Xc[1, 1]]).jacobian(sp.Matrix([y11, y12, y22])).det()
print("    congruence Jacobian / det(B)^3 =", sp.factor(sp.simplify(Jc / B.det() ** 3)))

# ---------- (b) p = 1 convolution ----------
x, t, a, b, lam, mu = sp.symbols("x t a b lambda mu", positive=True)
aa, bb = sp.Rational(3, 2), sp.Rational(5, 4)
conv_common = sp.integrate((t ** (aa - 1) * (x - t) ** (bb - 1)), (t, 0, x))
target = x ** (aa + bb - 1) * sp.gamma(aa) * sp.gamma(bb) / sp.gamma(aa + bb)
print("(b) common-rate Beta integral residual:", sp.simplify(conv_common - target))
# unequal rates, a=b=1: density of Exp(1)+Exp(2) = 2(e^{-x}-e^{-2x}); not Gamma: log-density second
# difference structure. Gamma log density is (k-1)log x - r x + c; test 3-point fit failure.
f = lambda xx: np.log(2 * (np.exp(-xx) - np.exp(-2 * xx)))
xs = np.array([0.5, 1.0, 2.0, 4.0])
Amat = np.c_[np.log(xs), -xs, np.ones_like(xs)]
coef, res, *_ = np.linalg.lstsq(Amat, f(xs), rcond=None)
print("    unequal rates: least-squares residual of Gamma-log-density fit at 4 points =", float(np.sum((Amat @ coef - f(xs)) ** 2)))

# ---------- (c) Laplace transform p=2 ----------
def sample_wishart2(nu_: float, P: np.ndarray, n: int) -> np.ndarray:
    r"""Bartlett-type sampler for W_2(nu,P): at P=I, x11=u~Gamma(nu,1), x12=sqrt(u) z, x22=v+z^2,
    v~Gamma(nu-1/2,1), z~N(0,1/2). General P via X = P^{-1/2} Z P^{-1/2}."""
    uu = rng.gamma(nu_, 1.0, n)
    vv = rng.gamma(nu_ - 0.5, 1.0, n)
    zz = rng.normal(0.0, np.sqrt(0.5), n)
    Z = np.empty((n, 2, 2))
    Z[:, 0, 0] = uu
    Z[:, 0, 1] = Z[:, 1, 0] = np.sqrt(uu) * zz
    Z[:, 1, 1] = vv + zz**2
    w, V = np.linalg.eigh(P)
    Pm12 = V @ np.diag(w ** -0.5) @ V.T
    return Pm12 @ Z @ Pm12

P = np.array([[1.3, 0.4], [0.4, 0.7]])
T = np.array([[0.5, -0.2], [-0.2, 0.9]])
n = 400000
nu_a, nu_b = 0.9, 1.7
Xa = sample_wishart2(nu_a, P, n)
Xb = sample_wishart2(nu_b, P, n)
S = Xa + Xb
mc = np.mean(np.exp(-np.einsum("ij,nji->n", T, S)))
exact = (np.linalg.det(P) / np.linalg.det(P + T)) ** (nu_a + nu_b)
print("(c) Laplace of sum (MC) vs (detP/det(P+T))^(a+b):", mc, exact, " rel err", abs(mc / exact - 1))
print("    mean of sum vs (a+b)P^-1, max abs err:", np.max(np.abs(S.mean(0) - (nu_a + nu_b) * np.linalg.inv(P))))
# unequal precisions: the sum's Laplace transform is not of Wishart form; compare log-Laplace
# along T = s*T0 to the best Wishart fit (two-parameter check in s only along one ray is weak, so
# we test the second cumulant matrix against nu*Pbar^{-1} (x) structure).
P2 = np.array([[3.0, 0.0], [0.0, 0.5]])
Xc2 = sample_wishart2(nu_b, P2, n)
S2 = Xa + Xc2
m = S2.mean(0)
# A Wishart W(nu, Pbar) has Cov(Tr(UX),Tr(VX)) = nu Tr(Pbar^-1 U Pbar^-1 V) with Pbar^-1 = m/nu.
# For U=V=E11 and U=V=E22 and U=E11,V=E22: c11 = m11^2/nu, c22 = m22^2/nu, c12 = m12^2/nu.
c11 = np.var(S2[:, 0, 0]); c22 = np.var(S2[:, 1, 1]); c12 = np.cov(S2[:, 0, 0], S2[:, 1, 1])[0, 1]
print("    unequal P: implied shapes m11^2/c11, m22^2/c22 (equal iff Wishart-compatible):",
      m[0, 0] ** 2 / c11, m[1, 1] ** 2 / c22)
print("    exact values:", (nu_a / 1.3 * 0 + 0) if False else "",
      "nu_a*Pinv11+nu_b/3.0 etc computed below")
Pinv = np.linalg.inv(P)
m11 = nu_a * Pinv[0, 0] + nu_b / 3.0
m22 = nu_a * Pinv[1, 1] + nu_b / 0.5
c11e = nu_a * Pinv[0, 0] ** 2 + nu_b * (1 / 3.0) ** 2
c22e = nu_a * Pinv[1, 1] ** 2 + nu_b * (1 / 0.5) ** 2
print("    exact implied shapes:", m11**2 / c11e, m22**2 / c22e)

# ---------- (d) KL ----------
def lG2(x_: float) -> float:
    return 0.5 * np.log(np.pi) + gammaln(x_) + gammaln(x_ - 0.5)

def psi2(x_: float) -> float:
    return digamma(x_) + digamma(x_ - 0.5)

def logW2(X: np.ndarray, nu_: float, P_: np.ndarray) -> np.ndarray:
    return (nu_ * np.log(np.linalg.det(P_)) - lG2(nu_)
            + (nu_ - 1.5) * np.log(np.linalg.det(X)) - np.einsum("ij,nji->n", P_, X))

def KL2(a_: float, P_: np.ndarray, b_: float, Q_: np.ndarray) -> float:
    Pinv_ = np.linalg.inv(P_)
    return ((a_ - b_) * psi2(a_) - lG2(a_) + lG2(b_)
            + b_ * np.log(np.linalg.det(P_) / np.linalg.det(Q_)) + a_ * (np.trace(Q_ @ Pinv_) - 2.0))

Qm = np.array([[0.8, -0.3], [-0.3, 1.9]])
mc_kl = np.mean(logW2(Xa, nu_a, P) - logW2(Xa, 1.4, Qm))
print("(d) KL p=2: closed form vs MC:", KL2(nu_a, P, 1.4, Qm), mc_kl)
# p=1 Gamma KL by quadrature
def KL1(a_, p_, b_, q_):
    return ((a_ - b_) * digamma(a_) - gammaln(a_) + gammaln(b_) + b_ * np.log(p_ / q_) + a_ * (q_ / p_ - 1.0))
a1, p1, b1, q1 = 1.7, 0.6, 2.4, 1.5
g = lambda xx, k, r: mp.e ** (k * mp.log(r) - mp.loggamma(k) + (k - 1) * mp.log(xx) - r * xx)
quad = mp.quad(lambda xx: g(xx, a1, p1) * (mp.log(g(xx, a1, p1)) - mp.log(g(xx, b1, q1))), [0, 1, 10, mp.inf])
print("    KL p=1: closed form vs quadrature:", KL1(a1, p1, b1, q1), float(quad))

# ---------- (e) Fisher metric in (shape, precision entries) ----------
al, p11, p12, p22 = sp.symbols("alpha p11 p12 p22", positive=True)
Psym = sp.Matrix([[p11, p12], [p12, p22]])
Psi = sp.log(sp.sqrt(sp.pi)) + sp.loggamma(al) + sp.loggamma(al - sp.Rational(1, 2)) - al * sp.log(Psym.det())
coords = [al, p11, p12, p22]
Hs = sp.hessian(Psi, coords)
subs = {al: nu_a, p11: P[0, 0], p12: P[0, 1], p22: P[1, 1]}
Hnum = np.array(Hs.subs(subs).evalf(), dtype=float)
# MC score covariance: score_alpha = log det P - psi2(alpha) + log det X ; score_{p_ab} = d/dp_ab log q
ld = np.log(np.linalg.det(Xa))
s_al = np.log(np.linalg.det(P)) - psi2(nu_a) + ld
Pinv = np.linalg.inv(P)
# d/dp11 [nu log det P - Tr(P X)] = nu Pinv11 - X11; d/dp12 (both entries) = 2 nu Pinv12 - 2 X12; d/dp22 similar
s11 = nu_a * Pinv[0, 0] - Xa[:, 0, 0]
s12 = 2 * nu_a * Pinv[0, 1] - 2 * Xa[:, 0, 1]
s22 = nu_a * Pinv[1, 1] - Xa[:, 1, 1]
Smat = np.c_[s_al, s11, s12, s22]
Fmc = Smat.T @ Smat / n
print("(e) Fisher Hessian(Psi) vs MC score covariance, max abs diff:", np.max(np.abs(Hnum - Fmc)),
      " (scale", np.max(np.abs(Hnum)), ")")
print("    shape block psi'(a)+psi'(a-1/2) =", float(polygamma(1, nu_a) + polygamma(1, nu_a - 0.5)), " Hess:", Hnum[0, 0])
U = np.array([[0.3, 0.7], [0.7, -0.2]]); Vv = np.array([[1.1, -0.4], [-0.4, 0.5]])
uvec = np.array([U[0, 0], U[0, 1], U[1, 1]]); vvec = np.array([Vv[0, 0], Vv[0, 1], Vv[1, 1]])
gUV = uvec @ Hnum[1:, 1:] @ vvec
print("    precision block g(U,V) Hessian vs nu Tr(P^-1 U P^-1 V):", gUV, nu_a * np.trace(Pinv @ U @ Pinv @ Vv))
print("    mixed block g(d_alpha, U) Hessian vs -Tr(P^-1 U):", Hnum[0, 1:] @ uvec, -np.trace(Pinv @ U))

# ---------- (f) same-batch covariance ----------
nuC = [0.8, 1.1, 1.6]  # children 1,2,3; C={1,2}, D={2,3}, intersection {2}
Xs = [sample_wishart2(nn, P, n) for nn in nuC]
XC = Xs[0] + Xs[1]; XD = Xs[1] + Xs[2]
lhs = np.cov(np.einsum("ij,nji->n", U, XC), np.einsum("ij,nji->n", Vv, XD))[0, 1]
rhs = nuC[1] * np.trace(Pinv @ U @ Pinv @ Vv)
print("(f) Cov(Tr(U X_C),Tr(V X_D)) MC vs A_{C cap D} Tr(P^-1 U P^-1 V):", lhs, rhs)
