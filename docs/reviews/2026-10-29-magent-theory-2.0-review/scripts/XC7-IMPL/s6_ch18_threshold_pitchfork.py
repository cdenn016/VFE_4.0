r"""XC7-IMPL s6: conformance checks for chapter 18 (it2:coupled-threshold, it2:critical-pitchfork).

Exact rational recomputation (sympy) of H, g, H^{-1}g, g^T H^{-1} g, T_c, T_mean, D and c,
then an independent numerical check of the nonlinear consequences: the asymmetric
stationary branches s_+-(delta) and the unstable eigenvalues 8 delta/D (Gibbs-reduced
flow) and 8 delta/(D+4) (full row relaxation), with the inverse Fisher metric
M(x) = diag(v, V, 2v^2, 2V^2) and b(1-b) for the row.
"""
import sympy as sp
import numpy as np
from scipy.optimize import fsolve

X = sp.symbols('mu0 mu1 mu2 a1 a2 v0 v1 v2 V1 V2', real=True)
mu0, mu1, mu2, a1, a2, v0, v1, v2, V1, V2 = X
P = [1, sp.Rational(5, 4), sp.Rational(5, 4)]; m = [0, -sp.Rational(23, 15), sp.Rational(23, 15)]
mus, vs = [mu0, mu1, mu2], [v0, v1, v2]
def kl1(mu, v, mref, prec):
    return sp.Rational(1, 2) * (prec * v + prec * (mu - mref) ** 2 - 1 - sp.log(prec * v))
F0 = sum(kl1(mus[i], vs[i], m[i], P[i]) for i in range(3)) + kl1(a1, V1, 0, sp.Rational(3, 4)) + kl1(a2, V2, 0, sp.Rational(3, 4))
S = [v + 1 for v in vs]
def Ccost(j, aj, Vj):
    d = mu0 - mus[j] - aj
    return sp.Rational(1, 2) * ((S[0] + d ** 2 + Vj) / S[j] - 1 + sp.log(S[j]) - sp.log(S[0]))
C1, C2 = Ccost(1, a1, V1), Ccost(2, a2, V2)
A = F0 + (C1 + C2) / 2; Dlt = C1 - C2
xs = dict(zip(X, [0, -sp.Rational(4, 3), sp.Rational(4, 3), sp.Rational(1, 3), -sp.Rational(1, 3), 1, 1, 1, 1, 1]))
grad = sp.Matrix([sp.diff(A, xi) for xi in X]).subs(xs)
gradD = sp.Matrix([sp.diff(Dlt, xi) for xi in X]).subs(xs)
print("stationarity of A at x*:", list(grad), " C1, C2 at x*:", C1.subs(xs), C2.subs(xs))
H = sp.hessian(A, X).subs(xs)
g = gradD
Hig = H.LUsolve(g)
gHg = sp.nsimplify((g.T * Hig)[0])
print("g =", list(g)); print("H^-1 g * 5059 =", [sp.nsimplify(e * 5059) for e in Hig])
print("g^T H^-1 g =", gHg, " T_c =", gHg / 4, " claimed 3269/10118:", gHg / 4 == sp.Rational(3269, 10118))
print("H positive definite:", all(ev > 0 for ev in H.eigenvals(multiple=True) if True) if False else
      all(float(e) > 0 for e in np.linalg.eigvalsh(np.array(H, dtype=float))))
Hm = H[:5, :5]; gm = g[:5, :]
Tmean = (gm.T * Hm.LUsolve(gm))[0] / 4
print("T_mean =", sp.nsimplify(Tmean), " claimed 31/122:", sp.nsimplify(Tmean) == sp.Rational(31, 122))
p = -Hig
Minv = sp.diag(*([1] * 5 + [sp.Rational(1, 2)] * 5))
Dval = sp.nsimplify((p.T * Minv * p)[0])
print("D =", Dval, " claimed 19189456/25593481:", Dval == sp.Rational(19189456, 25593481))
# quartic coefficient Phi_ssss = A_xxxx[p^4] + 4 Delta_xxx[p^3] - 3 k^T H^-1 k + 32 T_c,   k = A_xxx[p,p] + 2 Delta_xx p
s_, t_ = sp.symbols('s t')
def along(expr, extra=None):
    sub = {xi: xs[xi] + s_ * p[k] + (t_ if extra == k else 0) for k, xi in enumerate(X)}
    return expr.subs(sub, simultaneous=True)
A_s = along(A); D_s = along(Dlt)
A4 = sp.nsimplify(sp.diff(A_s, s_, 4).subs(s_, 0))
D3 = sp.nsimplify(sp.diff(D_s, s_, 3).subs(s_, 0))
kvec = []
for k in range(10):
    Ak = along(A, k); Dk = along(Dlt, k)
    kvec.append(sp.nsimplify(sp.diff(Ak, s_, 2, t_, 1).subs({s_: 0, t_: 0}) + 2 * sp.diff(Dk, s_, 1, t_, 1).subs({s_: 0, t_: 0})))
kvec = sp.Matrix(kvec)
kHk = sp.nsimplify((kvec.T * H.LUsolve(kvec))[0])
Tc = gHg / 4
Phi4 = A4 + 4 * D3 - 3 * kHk + 32 * Tc
cval = sp.nsimplify(Phi4 / 6)
print("A_xxxx[p^4] =", A4, " (claimed 236914859731152/655026269697361:", A4 == sp.Rational(236914859731152, 655026269697361), ")")
print("Delta_xxx[p^3] =", D3, " (claimed -31633266640/129477420379:", D3 == -sp.Rational(31633266640, 129477420379), ")")
print("3 k^T H^-1 k =", 3 * kHk, " (claimed:", 3 * kHk == sp.Rational(21485600170644766720, 2074468196131542287), ")")
print("c =", cval, " claimed:", cval == -sp.Rational(657534393590529472, 6223404588394626861), " float", float(cval))

# ---------------- numerical nonlinear check of branches and eigenvalues
fA = sp.lambdify(X, A, 'numpy'); fD = sp.lambdify(X, Dlt, 'numpy')
gA = sp.lambdify(X, [sp.diff(A, xi) for xi in X], 'numpy'); gD = sp.lambdify(X, [sp.diff(Dlt, xi) for xi in X], 'numpy')
xstar = np.array([float(xs[xi]) for xi in X]); pf = np.array([float(e) for e in p]); Df, cf, Tcf = float(Dval), float(cval), float(Tc)
def hprime(s): return np.log(1 + 2 * s) - np.log(1 - 2 * s)
def full_eqs(z, T):       # z = (x, s); stationarity of G(x,s) + T h(s)
    x, s = z[:10], z[10]
    gx = np.array(gA(*x)) + s * np.array(gD(*x))
    return np.concatenate([gx, [fD(*x) + T * hprime(s)]])
def Mx(x): return np.concatenate([x[5:10] * 0 + x[[5, 6, 7, 8, 9]] * 0 + np.array([x[5], x[6], x[7], x[8], x[9]]), 2 * x[5:10] ** 2])
def reduced_field(x, T):
    C1v = 0.5 * ((x[5] + 1 + (x[0] - x[1] - x[3]) ** 2 + x[8]) / (x[6] + 1) - 1 + np.log(x[6] + 1) - np.log(x[5] + 1))
    C2v = 0.5 * ((x[5] + 1 + (x[0] - x[2] - x[4]) ** 2 + x[9]) / (x[7] + 1) - 1 + np.log(x[7] + 1) - np.log(x[5] + 1))
    b = 1 / (1 + np.exp(-(C2v - C1v) / T))          # Gibbs row for prior (1/2,1/2)
    s = b - 0.5
    return -Mx(x) * (np.array(gA(*x)) + s * np.array(gD(*x)))
def full_field(z, T):
    x, b = z[:10], z[10]; s = b - 0.5
    fx = -Mx(x) * (np.array(gA(*x)) + s * np.array(gD(*x)))
    fb = -b * (1 - b) * (fD(*x) + T * hprime(s))        # row Fisher metric b(1-b) on the one free coordinate
    return np.concatenate([fx, [fb]])
def jac(f, z, T, h=1e-7):
    n = len(z); Jm = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = h
        Jm[:, k] = (f(z + e, T) - f(z - e, T)) / (2 * h)
    return Jm
for delta in (2e-4, 5e-5):
    T = Tcf + delta; s0 = np.sqrt(-4 * delta / cf)
    z = fsolve(full_eqs, np.concatenate([xstar + pf * s0, [s0]]), args=(T,), xtol=1e-14)
    lam_red = np.max(np.linalg.eigvals(jac(reduced_field, z[:10], T)).real)
    lam_full = np.max(np.linalg.eigvals(jac(full_field, np.concatenate([z[:10], [0.5 + z[10]]]), T)).real)
    print(f"delta={delta:.0e}: branch s={z[10]:.6e} vs sqrt(-4delta/c)={s0:.6e} (ratio {z[10]/s0:.5f}); "
          f"unstable eig reduced {lam_red:.6e} vs 8delta/D {8*delta/Df:.6e}; full {lam_full:.6e} vs 8delta/(D+4) {8*delta/(Df+4):.6e}")
