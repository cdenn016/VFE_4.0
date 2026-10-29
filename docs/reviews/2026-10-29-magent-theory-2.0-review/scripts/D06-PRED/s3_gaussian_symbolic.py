r"""D06-PRED s3: symbolic checks of the Gaussian realization (appendices/conditional_memory.tex).

All Gaussian product identities are checked as exact rational identities of the exponents and of
the normalizing prefactors (no symbolic integration), so every residual below is an exact sympy
cancel() result.

1D, fully symbolic parameters:
  cmem:gaussian-posterior        N(x;nu,L) N(xi;x,C) = N(xi;nu,L+C) N(x;m_X,P_X)
                                 joint of (X,Y) given (xi,a) is Gaussian with mean (m_X,m_Y) and
                                 covariance [[P_X, P_X F],[F P_X, F^2 P_X + Q]]  (gives m_Y, P_Y)
  cmem:result-only-gaussian      N(v;om,V) N(xi; nu + B(v-om), C+S) = N(xi;nu,L+C) N(v;m_Y,P_Y)
  cmem:joint-covariance          Schur complement equals Q
  cmem:response, two components  d/dxi E[Y|xi] - Cov(Y,X|xi)/C == 0, and
                                 Cov(X|xi)/C^2 == 1/C + d^2/dxi^2 log Z(xi)  (= -E d^2 log posterior)
                                 The exponentials E_a = exp(e_a(xi)) are carried as symbols with the
                                 chain rule dE_a/dxi = e_a'(xi) E_a, which makes both identities rational.
  one-component response         F L/(L+C), and F = H (L+C)/L gives H
2D, generic symbolic 2x2 matrices: P_X C^{-1} = L W^{-1};  C^{-1}(m_X - xi) = -W^{-1}(xi - nu)
2D, rational data, symbolic xi evaluated at points to 40 digits: response Jacobian, Fisher,
  frame laws cmem:frame-laws for one non-monomial h, singular F_a allowed.
"""
from __future__ import annotations

import time

import sympy as sp

t0 = time.time()
xi, x, v, y = sp.symbols("xi x v y", real=True)
nu1, om1, F1 = sp.symbols("nu1 om1 F1", real=True)
L1, Q1, C = sp.symbols("L1 Q1 C", positive=True)


def expo(z, m, var):
    """Exponent of N(z; m, var)."""
    return -(z - m) ** 2 / (2 * var)


mX = nu1 + L1 / (L1 + C) * (xi - nu1)
PX = 1 / (1 / L1 + 1 / C)
e_res = sp.cancel(expo(x, nu1, L1) + expo(xi, x, C) - expo(xi, nu1, L1 + C) - expo(x, mX, PX))
pre_res = sp.cancel(L1 * C - (L1 + C) * PX)
print("cmem:gaussian-posterior cue product: exponent residual =", e_res, ", prefactor residual =", pre_res)

mY = om1 + F1 * (mX - nu1)
PY = Q1 + F1 ** 2 * PX
Sig = sp.Matrix([[PX, PX * F1], [F1 * PX, F1 ** 2 * PX + Q1]])
dvec = sp.Matrix([x - mX, y - mY])
e_joint = sp.cancel(expo(x, mX, PX) + expo(y, om1 + F1 * (x - nu1), Q1) + (dvec.T * Sig.inv() * dvec)[0] / 2)
print("joint (X,Y)|xi,a Gaussian: exponent residual =", e_joint, ", det residual =", sp.cancel(Sig.det() - PX * Q1))

V = F1 ** 2 * L1 + Q1
B = L1 * F1 / V
S = L1 - L1 ** 2 * F1 ** 2 / V
e_ro = sp.cancel(expo(v, om1, V) + expo(xi, nu1 + B * (v - om1), C + S) - expo(xi, nu1, L1 + C) - expo(v, mY, PY))
pre_ro = sp.cancel(V * (C + S) - (L1 + C) * PY)
print("cmem:result-only-gaussian: exponent residual =", e_ro, ", prefactor residual =", pre_ro,
      ", S*V =", sp.factor(sp.cancel(S * V)))
Sigma = sp.Matrix([[L1, L1 * F1], [F1 * L1, F1 ** 2 * L1 + Q1]])
print("cmem:joint-covariance Schur complement - Q =", sp.cancel(Sigma[1, 1] - Sigma[1, 0] * Sigma[0, 1] / Sigma[0, 0] - Q1))

# two-component mixture: U_a = pi_a c_a exp(e_a(xi)) carried as positive symbols with the chain rule
# dU_a/dxi = e_a'(xi) U_a (pi_a and the normalizing constants c_a do not depend on xi).
nu = sp.symbols("nu_1 nu_2", real=True)
om = sp.symbols("om_1 om_2", real=True)
Fs = sp.symbols("F_1 F_2", real=True)
Ls = sp.symbols("L_1 L_2", positive=True)
U = sp.symbols("U_1 U_2", positive=True)
dea = [-(xi - nu[a]) / (Ls[a] + C) for a in range(2)]


def D(expr):
    """d/dxi with U_a treated by the chain rule."""
    return sp.diff(expr, xi) + sum(sp.diff(expr, U[a]) * dea[a] * U[a] for a in range(2))


def is_zero(expr):
    num, _ = sp.fraction(sp.together(expr))
    return sp.expand(num) == 0


Zt = U[0] + U[1]
w = [U[a] / Zt for a in range(2)]
mXa = [nu[a] + Ls[a] / (Ls[a] + C) * (xi - nu[a]) for a in range(2)]
PXa = [Ls[a] * C / (Ls[a] + C) for a in range(2)]
mYa = [om[a] + Fs[a] * (mXa[a] - nu[a]) for a in range(2)]
mbarY = w[0] * mYa[0] + w[1] * mYa[1]
mbarX = w[0] * mXa[0] + w[1] * mXa[1]
CovYX = sum(w[a] * (Fs[a] * PXa[a] + (mYa[a] - mbarY) * (mXa[a] - mbarX)) for a in range(2))
CovX = sum(w[a] * (PXa[a] + (mXa[a] - mbarX) ** 2) for a in range(2))
print("1D two-component response identity exact (numerator expands to 0):", is_zero(D(mbarY) - CovYX / C))
dlogZ = D(Zt) / Zt
print("1D two-component Fisher identity exact (numerator expands to 0):", is_zero(1 / C + D(dlogZ) - CovX / C ** 2))
print("1D same-state response d/dxi E[X|xi] = Cov(X|xi)/C exact:", is_zero(D(mbarX) - CovX / C))
print("  elapsed", round(time.time() - t0, 1), "s")

Hs = sp.symbols("H", real=True)
one = sp.diff(mY, xi)
print("one-component response - F L/(L+C) =", sp.cancel(one - F1 * L1 / (L1 + C)),
      "; with F = H(L+C)/L the response is", sp.cancel(one.subs(F1, Hs * (L1 + C) / L1)))

# 2D generic symbolic sub-identities
a11, a12, a22, c11, c12, c22 = sp.symbols("a11 a12 a22 c11 c12 c22", real=True)
Lm = sp.Matrix([[a11, a12], [a12, a22]])
Cm = sp.Matrix([[c11, c12], [c12, c22]])
Wm = Lm + Cm
PXm = (Lm.inv() + Cm.inv()).inv()
print("2D generic: P_X C^-1 - L W^-1 =", (PXm * Cm.inv() - Lm * Wm.inv()).applyfunc(sp.cancel).T)
nuv = sp.Matrix(sp.symbols("n1 n2", real=True))
xiv = sp.Matrix(sp.symbols("x1 x2", real=True))
mXv = nuv + Lm * Wm.inv() * (xiv - nuv)
print("2D generic: C^-1 (m_X - xi) + W^-1 (xi - nu) =", (Cm.inv() * (mXv - xiv) + Wm.inv() * (xiv - nuv)).applyfunc(sp.cancel).T)
print("  elapsed", round(time.time() - t0, 1), "s")

# 2D rational mixture, symbolic xi, evaluated at points
R = sp.Rational
Lr = [sp.Matrix([[2, R(1, 2)], [R(1, 2), 1]]), sp.Matrix([[1, -R(1, 3)], [-R(1, 3), R(3, 2)]])]
Qr = [sp.Matrix([[R(1, 2), R(1, 5)], [R(1, 5), 1]]), sp.Matrix([[1, 0], [0, R(1, 4)]])]
Fr = [sp.Matrix([[-2, 1], [R(1, 2), 3]]), sp.Matrix([[1, 2], [R(1, 2), 1]])]   # F_2 singular
Cr = sp.Matrix([[1, R(1, 4)], [R(1, 4), R(1, 2)]])
nur = [sp.Matrix([0, 1]), sp.Matrix([R(3, 2), -1])]
omr = [sp.Matrix([1, -1]), sp.Matrix([0, 2])]
pir = [R(1, 3), R(2, 3)]
print("det F_2 =", Fr[1].det())
X1, X2 = sp.symbols("X1 X2", real=True)
xv = sp.Matrix([X1, X2])


def mvn(z, m, Sm):
    d = z - m
    return sp.exp(-(d.T * Sm.inv() * d)[0] / 2) / (2 * sp.pi * sp.sqrt(Sm.det()))


def mixture(pis_, nus, Ls_, Fs_, oms, Qs_, Cmat, xvec):
    Zs = [mvn(xvec, nus[a], Ls_[a] + Cmat) for a in range(2)]
    Ztot = pis_[0] * Zs[0] + pis_[1] * Zs[1]
    ws = [pis_[a] * Zs[a] / Ztot for a in range(2)]
    mXs = [nus[a] + Ls_[a] * (Ls_[a] + Cmat).inv() * (xvec - nus[a]) for a in range(2)]
    PXs = [(Ls_[a].inv() + Cmat.inv()).inv() for a in range(2)]
    mYs = [oms[a] + Fs_[a] * (mXs[a] - nus[a]) for a in range(2)]
    mbY = ws[0] * mYs[0] + ws[1] * mYs[1]
    mbX = ws[0] * mXs[0] + ws[1] * mXs[1]
    CYX = sum((ws[a] * (Fs_[a] * PXs[a] + (mYs[a] - mbY) * (mXs[a] - mbX).T) for a in range(2)), sp.zeros(2, 2))
    CXX = sum((ws[a] * (PXs[a] + (mXs[a] - mbX) * (mXs[a] - mbX).T) for a in range(2)), sp.zeros(2, 2))
    SY = sum((ws[a] * (Qs_[a] + Fs_[a] * PXs[a] * Fs_[a].T + (mYs[a] - mbY) * (mYs[a] - mbY).T) for a in range(2)), sp.zeros(2, 2))
    return dict(Z=Ztot, w=ws, mbY=mbY, CYX=CYX, CXX=CXX, SY=SY)


M0 = mixture(pir, nur, Lr, Fr, omr, Qr, Cr, xv)
Jac = M0["mbY"].jacobian(xv)
Hess = sp.hessian(sp.log(M0["Z"]), (X1, X2))
Ci = Cr.inv()
pts = [(R(0), R(0)), (R(1, 2), -R(3, 4)), (R(5, 2), R(1, 3)), (-R(2), R(3))]
mr = mf = 0
for pt in pts:
    sub = {X1: pt[0], X2: pt[1]}
    mr = max(mr, max(abs(e) for e in (Jac - M0["CYX"] * Ci).subs(sub).evalf(40)))
    mf = max(mf, max(abs(e) for e in (Ci + Hess - Ci * M0["CXX"] * Ci).subs(sub).evalf(40)))
print("2D response Jacobian residual (4 points, 40 digits) =", sp.N(mr, 5))
print("2D Fisher residual C^-1 + Hess log Z - C^-1 Cov(X|xi) C^-1 =", sp.N(mf, 5))
at = {X1: R(1, 2), X2: -R(3, 4)}
fis = (Ci * M0["CXX"] * Ci).subs(at).evalf(30)
resp = (M0["CYX"] * Ci).subs(at).evalf(30)
print("2D Fisher eigenvalues at (1/2,-3/4) =", [sp.N(e, 6) for e in fis.eigenvals()])
print("2D response matrix at (1/2,-3/4) =", [[sp.N(resp[i, j], 6) for j in range(2)] for i in range(2)])
print("  elapsed", round(time.time() - t0, 1), "s")

h = sp.Matrix([[2, 1], [R(-1, 2), 1]])
Mh = mixture(pir, [h * n for n in nur], [h * La * h.T for La in Lr], [h * Fa * h.inv() for Fa in Fr],
             [h * o for o in omr], [h * Qa * h.T for Qa in Qr], h * Cr * h.T, h * xv)
fr = 0
for pt in pts:
    sub = {X1: pt[0], X2: pt[1]}
    fr = max(fr, abs(sp.N((Mh["Z"] * abs(h.det()) - M0["Z"]).subs(sub), 40)))
    fr = max(fr, abs(sp.N((Mh["w"][0] - M0["w"][0]).subs(sub), 40)))
    fr = max(fr, max(abs(e) for e in (Mh["mbY"] - h * M0["mbY"]).subs(sub).evalf(40)))
    fr = max(fr, max(abs(e) for e in (Mh["SY"] - h * M0["SY"] * h.T).subs(sub).evalf(40)))
print("frame laws residual (|det h| Z' - Z, weights, result mean, result covariance) =", sp.N(fr, 5))
print("total elapsed", round(time.time() - t0, 1), "s")
