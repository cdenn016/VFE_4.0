r"""Symbolic check of the scalar one-presentation formulas.

Lens D08-IMEM2.  Scope: appendices/interacting_memory.tex lines 508-558
(imem:scalar-gain, imem:scalar-message, imem:scalar-calibration,
imem:scalar-receiver) and companion/c04_interacting_memory.tex lines 176-192.

Method: every variable is written as an affine function of independent
sources; the sender's posterior predictive P(Y1 | D1) is obtained by generic
Gaussian conditioning (not by the manuscript's formula), a fresh sample is
drawn from it, and the receiver's posterior is again generic Gaussian
conditioning on (D0, H).  The results are compared with the displayed
formulas by sympy simplification.
"""
from __future__ import annotations

import sympy as sp

L0, L1, N0, N1, Q0, Q1, R = sp.symbols("L0 L1 N0 N1 Q0 Q1 R", positive=True)
S, f0, f1, b, w0, w1, nu0, nu1, d0, h = sp.symbols("S f0 f1 b omega0 omega1 nu0 nu1 d0 h", real=True)

# sources: X0c, X1c (correlated pair), eps0, eps1, zeta0, zeta1, xi, eta
src_cov_base = sp.diag(sp.Matrix([[L0, S], [S, L1]]), N0, N1, Q0, Q1)
nsrc = 8


def vec(**coef):
    v = sp.zeros(1, nsrc)
    names = ["X0", "X1", "e0", "e1", "z0", "z1", "xi", "eta"]
    for k, c in coef.items():
        v[0, names.index(k)] = c
    return v


X0 = (nu0, vec(X0=1))
X1 = (nu1, vec(X1=1))
D0 = (nu0, vec(X0=1, e0=1))
D1 = (nu1, vec(X1=1, e1=1))
Y0 = (w0, vec(X0=f0, z0=1))
Y1 = (w1, vec(X1=f1, z1=1))


def cov(u, v, Sig):
    return sp.simplify((u[1] * Sig * v[1].T)[0, 0])


# sender predictive P(Y1 | D1) by generic conditioning (xi variance not needed yet)
Sig0 = sp.diag(src_cov_base, 0, 0)
K = sp.simplify(cov(Y1, D1, Sig0) / cov(D1, D1, Sig0))
sig2 = sp.simplify(cov(Y1, Y1, Sig0) - K * cov(D1, Y1, Sig0))
g = L1 / (L1 + N1)
P = L1 * N1 / (L1 + N1)
a = b * f1 * g
print("predictive gain K - f1*g        :", sp.simplify(K - f1 * g))
print("predictive var - (Q1 + f1^2 P)  :", sp.simplify(sig2 - (Q1 + f1**2 * P)))

Sig = sp.diag(src_cov_base, sig2, R)
# fresh sample Ytilde = w1 + K (D1 - nu1) + xi ; H = b Ytilde + eta
Yt = (w1, K * vec(X1=1, e1=1) + vec(xi=1))
H = (b * w1, b * Yt[1] + vec(eta=1))
Hdir = (b * w1, b * Y1[1] + vec(eta=1))

Neff = a**2 * N1 + b**2 * (Q1 + f1**2 * P) + R
print("Cov(H,X1) - a L1                :", sp.simplify(cov(H, X1, Sig) - a * L1))
print("Var(H|X1) - Neff                :", sp.simplify(cov(H, H, Sig) - cov(H, X1, Sig)**2 / L1 - Neff))
print("Var(H) - (a^2 L1 + Neff)        :", sp.simplify(cov(H, H, Sig) - (a**2 * L1 + Neff)))
print("Var(H) - Var(Hdir)              :", sp.simplify(cov(H, H, Sig) - cov(Hdir, Hdir, Sig)))
print("Var(Hdir) - (b^2(f1^2L1+Q1)+R)  :", sp.simplify(cov(Hdir, Hdir, Sig) - (b**2 * (f1**2 * L1 + Q1) + R)))
print("Cov(X0,H) - a S                 :", sp.simplify(cov(X0, H, Sig) - a * S))
print("Cov(X0,Hdir) - b f1 S           :", sp.simplify(cov(X0, Hdir, Sig) - b * f1 * S))
print("Var(Hdir|X1) - (b^2 Q1 + R)     :", sp.simplify(cov(Hdir, Hdir, Sig) - cov(Hdir, X1, Sig)**2 / L1 - (b**2 * Q1 + R)))
print("g^2(L1+N1)+P - L1               :", sp.simplify(g**2 * (L1 + N1) + P - L1))

# receiver: condition (X0, Y0) on (D0, H) generically
obs = [D0, H]
Cobs = sp.Matrix(2, 2, lambda i, j: cov(obs[i], obs[j], Sig))
mobs = sp.Matrix([nu0, b * w1])
y = sp.Matrix([d0, h])
CX = sp.Matrix([[cov(X0, o, Sig) for o in obs]])
CY = sp.Matrix([[cov(Y0, o, Sig) for o in obs]])
Cinv = Cobs.inv()
mX = sp.simplify(nu0 + (CX * Cinv * (y - mobs))[0, 0])
vX = sp.simplify(L0 - (CX * Cinv * CX.T)[0, 0])
mY = sp.simplify(w0 + (CY * Cinv * (y - mobs))[0, 0])
vY = sp.simplify(cov(Y0, Y0, Sig) - (CY * Cinv * CY.T)[0, 0])

m0 = nu0 + L0 / (L0 + N0) * (d0 - nu0)
v0 = L0 * N0 / (L0 + N0)
m10 = nu1 + S / (L0 + N0) * (d0 - nu0)
v10 = L1 - S**2 / (L0 + N0)
c01 = S * N0 / (L0 + N0)
mH = b * w1 + a * (m10 - nu1)
VH = a**2 * v10 + Neff
m0p = m0 + a * c01 / VH * (h - mH)
v0p = v0 - a**2 * c01**2 / VH
print("m0+ (generic) - m0+ (displayed) :", sp.simplify(mX - m0p))
print("v0+ (generic) - v0+ (displayed) :", sp.simplify(vX - v0p))
print("E[Y0|d0,h] - displayed          :", sp.simplify(mY - (w0 + f0 * (m0p - nu0))))
print("Var(Y0|d0,h) - displayed        :", sp.simplify(vY - (Q0 + f0**2 * v0p)))
print("Var(H|D0) - V_H                 :", sp.simplify(cov(H, H, Sig) - cov(H, D0, Sig)**2 / (L0 + N0) - VH))
print("Cov(X0,H|D0) - a c01            :", sp.simplify(cov(X0, H, Sig) - cov(X0, D0, Sig) * cov(D0, H, Sig) / (L0 + N0) - a * c01))

# Direct-sensor receiver gain vs sampled-message receiver gain on X0 given D0
gain_samp = sp.simplify(a * c01 / VH)
VHdir = (b * f1)**2 * v10 + b**2 * Q1 + R
gain_dir = sp.simplify(b * f1 * c01 / VHdir)
ratio = sp.simplify(gain_samp / gain_dir)
print("receiver gain ratio sampled/direct:", sp.factor(ratio))
num = {L0: 1, L1: 2, N0: sp.Rational(1, 2), N1: 1, Q1: 1, R: 1, S: 1, f1: 1, b: 1}
print("  at L0=1,L1=2,N0=1/2,N1=1,Q1=1,R=1,S=1,f1=1,b=1 :", ratio.subs(num))
