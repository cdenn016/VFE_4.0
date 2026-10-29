r"""D15-WISH s1: Wishart basics and the complete Fisher flow of chapter 11.

(a) sympy: triangular chart of SPD(2) (det, trace, Jacobian), congruence Jacobian
    det(B)^3, and the matrix-beta substitution used in form:addition.
(b) Monte Carlo (Bartlett sampling, real shape) of the closed form
    KL(W_2(a,P) || W_2(b,Q)) = (a-b) psi_2(a) - log Gamma_2(a) + log Gamma_2(b)
                               + b log det(P Q^{-1}) + a (Tr(Q P^{-1}) - 2).
(c) mpmath (40 digits): natural gradient -G^{-1} grad E of the Gibbs-reduced
    potential form:energy on four agents with UNEQUAL precisions, G the exact
    Wishart Fisher matrix (Hessian of log Gamma_2(alpha) - alpha log det P in the
    affine coordinates (alpha, p11, p12, p22)). Compared with form:ode,
    alpha' = -2(alpha - alpha_0), P_i' = -2 P_i + I + C_q,
    C_q = I + (2 kappa/N) M - kappa alpha_0 Lambda^{-1}, M = sum_i alpha_i P_i^{-1}.
    Also the frozen-target field (channel derivative dropped) for contrast.
"""
import numpy as np
import sympy as sp
import mpmath as mp

# ---------------- (a) symbolic checks ----------------
u, v, z = sp.symbols('u v z', positive=True)
x11, x12, x22 = u, sp.sqrt(u) * z, v + z**2
X = sp.Matrix([[x11, x12], [x12, x22]])
Jac = sp.Matrix([x11, x12, x22]).jacobian([u, z, v]).det()
print("(a) det =", sp.simplify(X.det()), "; trace =", sp.simplify(X.trace()), "; Jacobian =", sp.simplify(Jac))
nu = sp.Rational(7, 5)
I_u = sp.integrate(u**(nu - 1) * sp.exp(-u), (u, 0, sp.oo))
I_v = sp.integrate(v**(nu - sp.Rational(3, 2)) * sp.exp(-v), (v, 0, sp.oo))
I_z = sp.integrate(sp.exp(-z**2), (z, -sp.oo, sp.oo))
G2 = sp.sqrt(sp.pi) * sp.gamma(nu) * sp.gamma(nu - sp.Rational(1, 2))
print("    normalizer / Gamma_2(7/5) - 1 =", sp.simplify(I_u * I_v * I_z / G2 - 1))
b11, b12, b21, b22, s11, s12, s22 = sp.symbols('b11 b12 b21 b22 s11 s12 s22')
B = sp.Matrix([[b11, b12], [b21, b22]])
S = sp.Matrix([[s11, s12], [s12, s22]])
Y = B * S * B.T
JB = sp.Matrix([Y[0, 0], Y[0, 1], Y[1, 1]]).jacobian([s11, s12, s22]).det()
print("    congruence Jacobian / det(B)^3 =", sp.simplify(JB / B.det()**3))

# ---------------- (b) KL closed form by Monte Carlo ----------------
rng = np.random.default_rng(20261029)
from scipy.special import gammaln, digamma


def lg2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl_closed(a, P, b, Q):
    return ((a - b) * psi2(a) - lg2(a) + lg2(b)
            + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2.0))


def sample_wishart(a, P, n):
    """Density det(P)^a/Gamma_2(a) det(X)^{a-3/2} exp(-Tr(P X)): Wishart with 2a dof, scale (2P)^{-1}."""
    L = np.linalg.cholesky(np.linalg.inv(2 * P))
    c1 = np.sqrt(rng.chisquare(2 * a, n))
    c2 = np.sqrt(rng.chisquare(2 * a - 1, n))
    g = rng.standard_normal(n)
    A = np.zeros((n, 2, 2))
    A[:, 0, 0], A[:, 1, 0], A[:, 1, 1] = c1, g, c2
    LA = L @ A
    return LA @ np.transpose(LA, (0, 2, 1))


def logw(Xs, a, P):
    d = np.linalg.det(Xs)
    tr = np.einsum('ij,nji->n', P, Xs)
    return a * np.log(np.linalg.det(P)) - lg2(a) + (a - 1.5) * np.log(d) - tr


a, P = 1.7, np.array([[1.3, 0.2], [0.2, 0.9]])
b, Q = 1.1, np.array([[0.8, -0.3], [-0.3, 1.6]])
Xs = sample_wishart(a, P, 400000)
lr = logw(Xs, a, P) - logw(Xs, b, Q)
print(f"(b) KL MC = {lr.mean():.5f} +- {lr.std() / np.sqrt(len(lr)):.5f}; closed form = {kl_closed(a, P, b, Q):.5f}")
print(f"    mean MC - a P^-1 (max abs) = {np.max(np.abs(Xs.mean(0) - a * np.linalg.inv(P))):.2e}")

# ---------------- (c) natural gradient in 40-digit arithmetic ----------------
mp.mp.dps = 40
N = 4


def mlg2(a):
    return mp.log(mp.sqrt(mp.pi)) + mp.loggamma(a) + mp.loggamma(a - mp.mpf(1) / 2)


def mpsi2(a):
    return mp.digamma(a) + mp.digamma(a - mp.mpf(1) / 2)


def mkl(a, P, b, Q):
    Pi = P ** -1
    return ((a - b) * mpsi2(a) - mlg2(a) + mlg2(b) + b * mp.log(mp.det(P) / mp.det(Q))
            + a * ((Q * Pi)[0, 0] + (Q * Pi)[1, 1] - 2))


def unpack(th):
    return th[0], mp.matrix([[th[1], th[2]], [th[2], th[3]]])


def energy(TH, a0, kap, frozen_Lam=None):
    M = mp.zeros(2, 2)
    for th in TH:
        al, P = unpack(th)
        M += al * P ** -1
    Lam = mp.eye(2) + (kap / N) * M if frozen_Lam is None else frozen_Lam
    return sum(mkl(*unpack(th), a0, mp.eye(2)) + mkl(*unpack(th), a0, Lam) for th in TH)


def fisher(th):
    def psi(*x):
        al, P = unpack(x)
        return mlg2(al) - al * mp.log(mp.det(P))
    G = mp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            order = [0] * 4
            order[i] += 1
            order[j] += 1
            G[i, j] = mp.diff(psi, tuple(th), tuple(order))
    return G


def natgrad(TH, Efun):
    V = []
    for i in range(N):
        g = mp.matrix(4, 1)
        for k in range(4):
            def f(x, i=i, k=k):
                T2 = [list(t) for t in TH]
                T2[i][k] = x
                return Efun(T2)
            g[k] = mp.diff(f, TH[i][k])
        V.append(-(fisher(TH[i]) ** -1) * g)
    return V


TH = [[mp.mpf('1.0'), mp.mpf('1.3'), mp.mpf('0.2'), mp.mpf('0.9')],
      [mp.mpf('1.2'), mp.mpf('0.8'), mp.mpf('-0.1'), mp.mpf('1.1')],
      [mp.mpf('1.8'), mp.mpf('1.0'), mp.mpf('0.3'), mp.mpf('1.4')],
      [mp.mpf('2.1'), mp.mpf('1.6'), mp.mpf('0.0'), mp.mpf('0.7')]]
a0 = mp.mpf('1.0')
for kap in [mp.mpf(0), mp.mpf(1), mp.mpf(5)]:
    V = natgrad(TH, lambda T: energy(T, a0, kap))
    M = mp.zeros(2, 2)
    for th in TH:
        al, P = unpack(th)
        M += al * P ** -1
    Lam = mp.eye(2) + (kap / N) * M
    C = mp.eye(2) + (2 * kap / N) * M - kap * a0 * Lam ** -1
    ra = max(abs(V[i][0] + 2 * (TH[i][0] - a0)) for i in range(N))
    rP = 0
    scale = 0
    for i in range(N):
        _, P = unpack(TH[i])
        R = -2 * P + mp.eye(2) + C
        rP = max(rP, abs(V[i][1] - R[0, 0]), abs(V[i][2] - R[0, 1]), abs(V[i][3] - R[1, 1]))
        scale = max(scale, abs(R[0, 0]), abs(R[0, 1]), abs(R[1, 1]))
    Vf = natgrad(TH, lambda T: energy(T, a0, kap, frozen_Lam=Lam))
    dfro = max(abs(Vf[i][k] - V[i][k]) for i in range(N) for k in range(1, 4))
    print(f"(c) kappa={float(kap)}: shape residual {float(ra):.1e}, precision residual {float(rP):.1e} "
          f"(velocity scale {float(scale):.3f}); frozen-target field differs by {float(dfro):.4f}")
