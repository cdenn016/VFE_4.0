r"""D15-WISH s4: form:geometry, form:geometry-decay, form:anisotropic-reference.

Isotropic references (form:context): both eigenvalues of P solve dp/dt = f(p, abar(t)),
f = -2p + 2 + 2 kappa abar/p - kappa alpha0 p/(p + kappa abar). The splitting
p_+ - p_- = int_{e^{-a}}^{e^{a}} exp(L_t(p0)) dp0, with L_t = int_0^t f_p ds, is computed
by Gauss-Legendre quadrature of the log-sensitivity (no cancellation), so the late-time
curvature limit is not polluted by round-off.
I_1 = 2 A (p_+-p_-)^2/(p_+ p_-), I_2 likewise for the model sector, A_chi = -I_1/(I_1+I_2),
h_chichi = I_1 I_2/(I_1+I_2), g_aa = A[(d_a p_+/p_+)^2 + (d_a p_-/p_-)^2],
F = -(d_a I_1) I_2/(I_1+I_2)^2.
Anisotropic variant: dp/dt = -2p + 2 gamma + 2 kappa abar/p - kappa alpha0 p/(gamma p + kappa abar).
Also a brute-force minimization of the collective Fisher cost on the full matrix ODE
(fine children versus parent) at one time.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

XG, WG = np.polynomial.legendre.leggauss(24)


def f(p, ab, kap, a0, gam=1.0):
    return -2 * p + 2 * gam + 2 * kap * ab / p - kap * a0 * p / (gam * p + kap * ab)


def fp(p, ab, kap, a0, gam=1.0):
    return -2 - 2 * kap * ab / p**2 - kap**2 * a0 * ab / (gam * p + kap * ab) ** 2


def fg(p, ab, kap, a0, gam=1.0):
    return 2 + kap * a0 * p**2 / (gam * p + kap * ab) ** 2


def traj(p0, abar0, a0, kap, ts, gam=1.0, s0=None, dgam=0.0):
    """Returns p(t), L_t = int f_p, and the sensitivity s(t) driven by s0 and dgam."""
    abar = lambda t: a0 + np.exp(-2 * t) * (abar0 - a0)

    def rhs(t, y):
        p, L, s = y
        return [f(p, abar(t), kap, a0, gam), fp(p, abar(t), kap, a0, gam),
                fp(p, abar(t), kap, a0, gam) * s + fg(p, abar(t), kap, a0, gam) * dgam]
    sol = solve_ivp(rhs, (0, ts[-1]), [p0, 0.0, 1.0 if s0 is None else s0], t_eval=ts,
                    rtol=1e-12, atol=1e-14, method="DOP853")
    return sol.y


def split(lo, hi, ab0, al0, kap, ts):
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    tot = np.zeros_like(ts)
    for x, w in zip(XG, WG):
        _, L, _ = traj(mid + half * x, ab0, al0, kap, ts)
        tot += w * half * np.exp(L)
    return tot


def iso_block(a, at, A0, B0, abar0, ebar0, a0, e0, kap, ts, n=2):
    pp, Lp, _ = traj(np.exp(a), abar0, a0, kap, ts)
    pm, Lm, _ = traj(np.exp(-a), abar0, a0, kap, ts)
    qp, _, _ = traj(np.exp(at), ebar0, e0, kap, ts)
    qm, _, _ = traj(np.exp(-at), ebar0, e0, kap, ts)
    dP = split(np.exp(-a), np.exp(a), abar0, a0, kap, ts)
    dQ = split(np.exp(-at), np.exp(at), ebar0, e0, kap, ts)
    A = n * a0 + np.exp(-2 * ts) * (A0 - n * a0)
    B = n * e0 + np.exp(-2 * ts) * (B0 - n * e0)
    dap, dam = np.exp(a) * np.exp(Lp), -np.exp(-a) * np.exp(Lm)
    I1 = 2 * A * dP**2 / (pp * pm)
    I2 = 2 * B * dQ**2 / (qp * qm)
    dI1 = 2 * A * (2 * dP * (dap - dam) / (pp * pm) - dP**2 * (dap * pm + pp * dam) / (pp * pm) ** 2)
    gaa = A * ((dap / pp) ** 2 + (dam / pm) ** 2)
    Achi = -I1 / (I1 + I2)
    hcc = I1 * I2 / (I1 + I2)
    F = -dI1 * I2 / (I1 + I2) ** 2
    return gaa, hcc, Achi, F, I1, I2


alphas, etas = np.array([1.0, 1.2, 1.8, 2.1]), np.array([0.8, 1.1, 1.4, 1.7])
ts = np.array([0.0, 1.0, 0.5 * np.log(8.5), 2.0, 4.0, 6.0, 8.0, 12.0])
for (a0, e0, kap) in [(1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 0.0)]:
    gaa, hcc, Achi, F, I1, I2 = iso_block(0.3, 0.3, alphas[:2].sum(), etas[:2].sum(), alphas.mean(),
                                          etas.mean(), a0, e0, kap, ts)
    ps = brentq(lambda p: f(p, a0, kap, a0), 1e-6, 50)
    lam = -fp(ps, a0, kap, a0)
    qs = brentq(lambda p: f(p, e0, kap, e0), 1e-6, 50)
    lamt = -fp(qs, e0, kap, e0)
    rate = -np.log(gaa[7] / gaa[6]) / (ts[7] - ts[6])
    print(f"alpha0={a0} eta0={e0} kappa={kap}: p*={ps:.6f} 2*lambda={2*lam:.6f} model 2*lambda_tilde={2*lamt:.6f} "
          f"fitted g_aa rate (t=8..12) {rate:.6f}")
    for k, t in enumerate(ts):
        print(f"   t={t:7.4f}  g_aa={gaa[k]:.4e}  h_chichi={hcc[k]:.4e}  A_chi={Achi[k]:+.5f}  F={F[k]:+.5f}  "
              f"g_aa/g_aa(0)={gaa[k]/gaa[0]:.3e}")
print("12 - 4 sqrt 2 =", 12 - 4 * np.sqrt(2), "; f(sqrt2,1) =", f(np.sqrt(2), 1, 1, 1), "; f_p(sqrt2,1)+6-2sqrt2 =",
      fp(np.sqrt(2), 1, 1, 1) + 6 - 2 * np.sqrt(2))

# anisotropic references (form:anisotropic-reference), block {1,2}, alpha0 = eta0 = kappa = 1, a = at = 0.3
print()
a, at, a0, e0, kap = 0.3, 0.3, 1.0, 1.0, 1.0
ts2 = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
res_p, res_q = {}, {}
for sg in (+1, -1):
    g = np.exp(sg * a)
    p, _, s = traj(g, alphas.mean(), a0, kap, ts2, gam=g, s0=sg * g, dgam=sg * g)
    res_p[sg] = (p, s)
    gq = np.exp(sg * at)
    q, _, _ = traj(gq, etas.mean(), e0, kap, ts2, gam=gq, s0=0.0, dgam=0.0)
    res_q[sg] = q
A = 2 * a0 + np.exp(-2 * ts2) * (alphas[:2].sum() - 2 * a0)
B = 2 * e0 + np.exp(-2 * ts2) * (etas[:2].sum() - 2 * e0)
pp, spp = res_p[1]
pm, smm = res_p[-1]
qp, qm = res_q[1], res_q[-1]
I1 = 2 * A * (pp - pm) ** 2 / (pp * pm)
I2 = 2 * B * (qp - qm) ** 2 / (qp * qm)
dI1 = 2 * A * (2 * (pp - pm) * (spp - smm) / (pp * pm) - (pp - pm) ** 2 * (spp * pm + pp * smm) / (pp * pm) ** 2)
gaa = A * ((spp / pp) ** 2 + (smm / pm) ** 2)
F = -dI1 * I2 / (I1 + I2) ** 2
for k, t in enumerate(ts2):
    print(f"anisotropic t={t:5.1f}: g_aa={gaa[k]:.4f} h_chichi={I1[k]*I2[k]/(I1[k]+I2[k]):.4f} "
          f"A_chi={-I1[k]/(I1[k]+I2[k]):+.4f} F={F[k]:+.4f}")
for sg in (+1, -1):
    g = np.exp(sg * a)
    ps = brentq(lambda p: f(p, a0, kap, a0, g), 1e-6, 50)
    print(f"   fixed point gamma=exp({sg}a): p*={ps:.6f}")

# brute-force collective minimization on the matrix ODE (isotropic references)
print()
N = 4


def mat_rhs(t, y, a0, kap, n):
    al = y[:n]
    P = np.array([[y[n], y[n + 1]], [y[n + 1], y[n + 2]]])
    M = al.sum() * np.linalg.inv(P)
    Lam = np.eye(2) + kap / N * M
    C = np.eye(2) + 2 * kap / N * M - kap * a0 * np.linalg.inv(Lam)
    dP = -2 * P + np.eye(2) + C
    return np.concatenate([-2 * (al - a0), [dP[0, 0], dP[0, 1], dP[1, 1]]])


def evolve(sh, P0, t, a0, kap):
    y0 = np.concatenate([sh, [P0[0, 0], P0[0, 1], P0[1, 1]]])
    y = solve_ivp(mat_rhs, (0, t), y0, args=(a0, kap, len(sh)), rtol=1e-12, atol=1e-14, method="DOP853").y[:, -1]
    n = len(sh)
    return y[:n], np.array([[y[n], y[n + 1]], [y[n + 1], y[n + 2]]])


def rot(c):
    return np.array([[np.cos(c), -np.sin(c)], [np.sin(c), np.cos(c)]])


def P_section(a, chi):
    return rot(chi) @ np.diag([np.exp(a), np.exp(-a)]) @ rot(chi).T


def fisherP(shape, P, U, V):
    Pi = np.linalg.inv(P)
    return shape * np.trace(Pi @ U @ Pi @ V)


Jg = np.array([[0.0, -1.0], [1.0, 0.0]])
t, a, chi, h = 0.6, 0.3, 0.2, 1e-5
a0, kap = 1.0, 1.0
al, P = evolve(alphas, P_section(a, chi), t, a0, kap)
eq, Q = evolve(etas, np.diag([np.exp(0.3), np.exp(-0.3)]), t, a0, kap)
dPa = (evolve(alphas, P_section(a + h, chi), t, a0, kap)[1] - evolve(alphas, P_section(a - h, chi), t, a0, kap)[1]) / (2 * h)
dPc = (evolve(alphas, P_section(a, chi + h), t, a0, kap)[1] - evolve(alphas, P_section(a, chi - h), t, a0, kap)[1]) / (2 * h)
JP, JQ = Jg @ P - P @ Jg, Jg @ Q - Q @ Jg
Z = 0 * Q
blk = [0, 1]
for desc, wP, wQ in [("fine children", list(al[blk]), list(eq[blk])), ("parent", [al[blk].sum()], [eq[blk].sum()])]:
    def G(U1, V1, U2, V2):
        return sum(fisherP(w, P, U1, U2) for w in wP) + sum(fisherP(w, Q, V1, V2) for w in wQ)
    kJ = G(JP, JQ, JP, JQ)
    Aa = -G(dPa, Z, JP, JQ) / kJ
    Ac = -G(dPc, Z, JP, JQ) / kJ
    ra = (dPa + Aa * JP, Aa * JQ)
    rc = (dPc + Ac * JP, Ac * JQ)
    print(f"brute force t={t} block {blk} {desc}: A_a={Aa:+.2e} A_chi={Ac:+.10f} h_aa={G(*ra, *ra):.10f} "
          f"h_chichi={G(*rc, *rc):.10f} h_achi={G(*ra, *rc):+.1e}")
ts3 = np.array([0.0, t])
gaa, hcc, Achi, F, I1, I2 = iso_block(a, 0.3, alphas[:2].sum(), etas[:2].sum(), alphas.mean(), etas.mean(), a0, 1.0, kap, ts3)
print(f"formulas   t={t}: A_chi={Achi[-1]:+.10f} g_aa={gaa[-1]:.10f} h_chichi={hcc[-1]:.10f}")
