r"""L08-CND check 6 (extension sketch): context-dependent reference precision keeps the geometry alive.

Modification: belief reference r = W(alpha0, Gamma(c)), Gamma(a,chi) = R_chi diag(e^{s a}, e^{-s a}) R_chi^T,
channel target W(alpha0, Lambda), Lambda = Gamma + (kappa/N) M.  Model sector keeps identity reference.
Then C = Lambda + (kappa/N)(M - N alpha0 Lambda^{-1}) and P' = -2P + Gamma + C (same derivation as the
chapter with I replaced by Gamma in the reference and in Lambda).  In the common eigenframe of Gamma,
  p' = -2p + g + l + kappa abar/p - kappa alpha0/l,  l = g + kappa abar/p.
The section is P(0;a,chi) = Gamma(a,chi) (so all frames agree).  We report k1, k2, A_chi, F, g_aa at large t.
Also verifies the modified flow by natural gradient at one point (p=2, N=4, mpmath finite differences).
"""
from __future__ import annotations

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

N = 4
alpha0 = eta0 = kappa = 1.0
s = 1.0
al0s = np.array([1.0, 1.2, 1.8, 2.1])
et0s = np.array([0.8, 1.1, 1.4, 1.7])
b = 0.3


def f_mod(p, ab, g, kap, ref):
    l = g + kap * ab / p
    return -2 * p + g + l + kap * ab / p - kap * ref / l


def f_orig(p, ab, kap, ref):
    return -2 * p + 2 + 2 * kap * ab / p - kap * ref * p / (p + kap * ab)


def evolve(fun, p0, t, shapes0, ref):
    def rhs(u, y):
        ab = ref + np.exp(-2 * u) * (shapes0.mean() - ref)
        return [fun(y[0], ab)]
    if t == 0:
        return p0
    return solve_ivp(rhs, (0, t), [p0], rtol=1e-12, atol=1e-14).y[0, -1]


def geometry(t, a, h=1e-5):
    A_I = (alpha0 + np.exp(-2 * t) * (al0s - alpha0))[:2].sum()
    B_I = (eta0 + np.exp(-2 * t) * (et0s - eta0))[:2].sum()

    def peig(aa):
        gp, gm = np.exp(s * aa), np.exp(-s * aa)
        pp = evolve(lambda p, ab: f_mod(p, ab, gp, kappa, alpha0), gp, t, al0s, alpha0)
        pm = evolve(lambda p, ab: f_mod(p, ab, gm, kappa, alpha0), gm, t, al0s, alpha0)
        return pp, pm
    pp, pm = peig(a)
    ppp, pmp = peig(a + h)
    ppm, pmm = peig(a - h)
    dpp, dpm = (ppp - ppm) / (2 * h), (pmp - pmm) / (2 * h)
    if MODEL_ANISO:
        qp = evolve(lambda p, ab: f_mod(p, ab, np.exp(b), kappa, eta0), np.exp(b), t, et0s, eta0)
        qm = evolve(lambda p, ab: f_mod(p, ab, np.exp(-b), kappa, eta0), np.exp(-b), t, et0s, eta0)
    else:
        qp = evolve(lambda p, ab: f_orig(p, ab, kappa, eta0), np.exp(b), t, et0s, eta0)
        qm = evolve(lambda p, ab: f_orig(p, ab, kappa, eta0), np.exp(-b), t, et0s, eta0)
    k1f = lambda x, y: 2 * A_I * (x - y) ** 2 / (x * y)
    k2 = 2 * B_I * (qp - qm) ** 2 / (qp * qm)
    k1 = k1f(pp, pm)
    dk1 = (k1f(ppp, pmp) - k1f(ppm, pmm)) / (2 * h)
    return dict(k1=k1, k2=k2, Achi=-k1 / (k1 + k2), F=-dk1 * k2 / (k1 + k2) ** 2,
                gaa=A_I * ((dpp / pp) ** 2 + (dpm / pm) ** 2), hchi=k1 * k2 / (k1 + k2))


for MODEL_ANISO in (False, True):
    print("context-dependent belief reference precision; model reference",
          "fixed anisotropic diag(e^b,e^-b)" if MODEL_ANISO else "identity", "; block {1,2}, a = 0.3")
    for t in [0.0, 1.0, 2.0, 4.0, 8.0]:
        gm = geometry(t, 0.3)
        print(f"  t={t}: F={gm['F']:.5f}  A_chi={gm['Achi']:.5f}  g_aa={gm['gaa']:.5e}  h_chichi={gm['hchi']:.5e}"
              f"  k1={gm['k1']:.4e} k2={gm['k2']:.4e}")

# natural-gradient verification of P' = -2P + Gamma + C for the modified potential
mp.mp.dps = 30
kap = mp.mpf(1)
a0 = mp.mpf(1)
Gam = mp.matrix([[mp.mpf("1.5"), mp.mpf("0.2")], [mp.mpf("0.2"), mp.mpf("0.7")]])


def tr(A):
    return A[0, 0] + A[1, 1]


def lG2(x):
    return mp.log(mp.sqrt(mp.pi)) + mp.loggamma(x) + mp.loggamma(x - mp.mpf(1) / 2)


def KL(a, P, bb, Q):
    psi2 = mp.digamma(a) + mp.digamma(a - mp.mpf(1) / 2)
    return ((a - bb) * psi2 - lG2(a) + lG2(bb) + bb * (mp.log(mp.det(P)) - mp.log(mp.det(Q)))
            + a * (tr(Q * mp.inverse(P)) - 2))


def unpack(xi):
    al = [xi[4 * i] for i in range(N)]
    Ps = [mp.matrix([[xi[4 * i + 1], xi[4 * i + 2]], [xi[4 * i + 2], xi[4 * i + 3]]]) for i in range(N)]
    return al, Ps


def energy(xi):
    al, Ps = unpack(xi)
    M = mp.zeros(2, 2)
    for a, P in zip(al, Ps):
        M += a * mp.inverse(P)
    Lam = Gam + kap / N * M
    return sum(KL(al[i], Ps[i], a0, Gam) + KL(al[i], Ps[i], a0, Lam) for i in range(N))


xi = []
vals = [("0.9", "1.4", "0.3", "0.8"), ("1.3", "0.9", "-0.2", "1.2"), ("2.2", "2.0", "0.5", "1.0"), ("3.1", "0.7", "0.1", "0.6")]
for v in vals:
    xi += [mp.mpf(x) for x in v]
hh = mp.mpf("1e-10")
g = []
for k in range(len(xi)):
    xp = list(xi); xm = list(xi)
    xp[k] += hh; xm[k] -= hh
    g.append((energy(xp) - energy(xm)) / (2 * hh))
al, Ps = unpack(xi)
G = mp.zeros(4 * N, 4 * N)
E = [mp.matrix([[1, 0], [0, 0]]), mp.matrix([[0, 1], [1, 0]]), mp.matrix([[0, 0], [0, 1]])]
for i in range(N):
    Pi = mp.inverse(Ps[i])
    G[4 * i, 4 * i] = mp.psi(1, al[i]) + mp.psi(1, al[i] - mp.mpf(1) / 2)
    for r in range(3):
        G[4 * i, 4 * i + r + 1] = G[4 * i + r + 1, 4 * i] = -tr(Pi * E[r])
        for c in range(3):
            G[4 * i + r + 1, 4 * i + c + 1] = al[i] * tr(Pi * E[r] * Pi * E[c])
v = -(mp.lu_solve(G, mp.matrix(g)))
M = mp.zeros(2, 2)
for a, P in zip(al, Ps):
    M += a * mp.inverse(P)
Lam = Gam + kap / N * M
C = Lam + kap / N * (M - N * a0 * mp.inverse(Lam))
err = mp.mpf(0)
for i in range(N):
    Pd = -2 * Ps[i] + Gam + C
    cl = [-2 * (al[i] - a0), Pd[0, 0], Pd[0, 1], Pd[1, 1]]
    err = max(err, max(abs(v[4 * i + r] - cl[r]) for r in range(4)))
print("natural gradient vs P' = -2P + Gamma + C (modified potential), max abs diff:", mp.nstr(err, 5))
