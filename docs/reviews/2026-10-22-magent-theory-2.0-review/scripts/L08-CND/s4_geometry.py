r"""L08-CND check 4: collective minimizing connection, contextual tensor and curvature.

Section: P(t;a,chi) = R_chi diag(p_+(t,a), p_-(t,a)) R_chi^T, Q(t) = diag(q_+(t), q_-(t)).
Fisher metric at fixed shape: g(U,V) = nu Tr(P^{-1} U P^{-1} V); collective = sum over children (unit weights).
Orbit: J xi = xi([J,P], [J,Q]) with J the so(2) generator.
Claims: k1 = 2A (p+-p-)^2/(p+ p-), k2 = 2B (q+-q-)^2/(q+ q-),
        A^F = -k1/(k1+k2) dchi,  h_min = g_aa da^2 + k1 k2/(k1+k2) dchi^2,
        g_aa = A[(d_a p+/p+)^2 + (d_a p-/p-)^2],  F = -(d_a k1) k2/(k1+k2)^2 da^dchi.
Checks:
 (a) direct numerical minimization of the collective cost from matrix-ODE sections (finite differences
     in a and chi) vs the formulas, for the fine block {1,2} (children separately) and its parent.
 (b) curvature by finite differences of A^F_chi(a) vs the formula; A^F_a = 0; chi-independence.
 (c) long-time behavior of F and h_min for alpha0 = eta0 (the manuscript's example) and alpha0 != eta0.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

N = 4
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(c: float) -> np.ndarray:
    return np.array([[np.cos(c), -np.sin(c)], [np.sin(c), np.cos(c)]])


def make_sector(shapes0: np.ndarray, ref: float, kappa: float):
    def mean_shape(t):
        return ref + np.exp(-2 * t) * (shapes0.mean() - ref)

    def rhs(t, y):
        P = y.reshape(2, 2)
        M = N * mean_shape(t) * np.linalg.inv(P)
        Lam = np.eye(2) + kappa / N * M
        C = Lam + kappa / N * (M - N * ref * np.linalg.inv(Lam))
        return (-2 * P + np.eye(2) + C).ravel()

    def shapes(t):
        return ref + np.exp(-2 * t) * (shapes0 - ref)
    return rhs, shapes


def solve_P(rhs, P0: np.ndarray, t: float) -> np.ndarray:
    if t == 0:
        return P0
    s = solve_ivp(rhs, (0, t), P0.ravel(), rtol=1e-12, atol=1e-14)
    return s.y[:, -1].reshape(2, 2)


def g(nu: float, P: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    Pi = np.linalg.inv(P)
    return nu * np.trace(Pi @ U @ Pi @ V)


def collective(t, a, chi, al_shapes, et_shapes, rhsP, rhsQ, b, h=1e-5):
    r"""Return (A_a, A_chi, h_min 2x2) from direct minimization with children shapes lists."""
    P0 = lambda aa, cc: rot(cc) @ np.diag([np.exp(aa), np.exp(-aa)]) @ rot(cc).T
    Q0 = np.diag([np.exp(b), np.exp(-b)])
    P = solve_P(rhsP, P0(a, chi), t)
    Q = solve_P(rhsQ, Q0, t)
    dPa = (solve_P(rhsP, P0(a + h, chi), t) - solve_P(rhsP, P0(a - h, chi), t)) / (2 * h)
    dPc = (solve_P(rhsP, P0(a, chi + h), t) - solve_P(rhsP, P0(a, chi - h), t)) / (2 * h)
    dQ = np.zeros((2, 2))  # Q independent of context
    OP, OQ = J @ P - P @ J, J @ Q - Q @ J

    def G(u, v):  # u, v = (belief tangent, model tangent), summed over children
        return sum(g(al, P, u[0], v[0]) for al in al_shapes) + sum(g(et, Q, u[1], v[1]) for et in et_shapes)
    orbit = (OP, OQ)
    kk = G(orbit, orbit)
    A = []
    res = []
    for d in [(dPa, dQ), (dPc, dQ)]:
        coef = -G(d, orbit) / kk
        A.append(coef)
        res.append((d[0] + coef * OP, d[1] + coef * OQ))
    hmin = np.array([[G(res[i], res[j]) for j in range(2)] for i in range(2)])
    return A[0], A[1], hmin, P, Q


def eig_with_variation(f, p0, t, ref, shapes0, kappa):
    r"""Scalar eigenvalue ODE and its variational equation dPhi/dp0."""
    def rhs(s, y):
        ab = ref + np.exp(-2 * s) * (shapes0.mean() - ref)
        p, w = y
        fp = -2 - 2 * kappa * ab / p**2 - kappa**2 * ref * ab / (p + kappa * ab) ** 2
        return [f(p, ab, kappa, ref), fp * w]
    if t == 0:
        return p0, 1.0
    s = solve_ivp(rhs, (0, t), [p0, 1.0], rtol=1e-12, atol=1e-16)
    return s.y[0, -1], s.y[1, -1]


def f_scalar(p, ab, kappa, ref):
    return -2 * p + 2 + 2 * kappa * ab / p - kappa * ref * p / (p + kappa * ab)


def formulas(t, a, A_I, B_I, ref_a, ref_e, al0s, et0s, kappa, b):
    pp, wp = eig_with_variation(f_scalar, np.exp(a), t, ref_a, al0s, kappa)
    pm, wm = eig_with_variation(f_scalar, np.exp(-a), t, ref_a, al0s, kappa)
    qp, _ = eig_with_variation(f_scalar, np.exp(b), t, ref_e, et0s, kappa)
    qm, _ = eig_with_variation(f_scalar, np.exp(-b), t, ref_e, et0s, kappa)
    dpp, dpm = wp * np.exp(a), -wm * np.exp(-a)
    k1 = 2 * A_I * (pp - pm) ** 2 / (pp * pm)
    k2 = 2 * B_I * (qp - qm) ** 2 / (qp * qm)
    r = pp / pm
    dr = (dpp * pm - pp * dpm) / pm**2
    dk1 = 2 * A_I * (1 - 1 / r**2) * dr
    gaa = A_I * ((dpp / pp) ** 2 + (dpm / pm) ** 2)
    Achi = -k1 / (k1 + k2)
    F = -dk1 * k2 / (k1 + k2) ** 2
    return dict(k1=k1, k2=k2, gaa=gaa, Achi=Achi, F=F, hchi=k1 * k2 / (k1 + k2), dpp=dpp, dpm=dpm)


# ---------- manuscript example ----------
alpha0 = eta0 = kappa = 1.0
al0s = np.array([1.0, 1.2, 1.8, 2.1])
et0s = np.array([0.8, 1.1, 1.4, 1.7])
b = 0.3
rhsP, al_t = make_sector(al0s, alpha0, kappa)
rhsQ, et_t = make_sector(et0s, eta0, kappa)
print("(a) direct minimization vs formulas, block {1,2} fine and parent, root parent")
for t in [0.0, 0.4, 0.8, 1.2]:
    for a in [0.25, 0.35]:
        chi = 0.4
        al = al_t(t)
        et = et_t(t)
        fineA = collective(t, a, chi, al[:2], et[:2], rhsP, rhsQ, b)
        parA = collective(t, a, chi, [al[:2].sum()], [et[:2].sum()], rhsP, rhsQ, b)
        rootA = collective(t, a, chi, [al.sum()], [et.sum()], rhsP, rhsQ, b)
        fm = formulas(t, a, al[:2].sum(), et[:2].sum(), alpha0, eta0, al0s, et0s, kappa, b)
        fr = formulas(t, a, al.sum(), et.sum(), alpha0, eta0, al0s, et0s, kappa, b)
        print(f"  t={t} a={a}: A_a fine={fineA[0]:.2e}; A_chi fine/parent/formula="
              f"{fineA[1]:.10f}/{parA[1]:.10f}/{fm['Achi']:.10f}; h_aa fine/formula={fineA[2][0,0]:.8f}/{fm['gaa']:.8f};"
              f" h_chichi fine/formula={fineA[2][1,1]:.8f}/{fm['hchi']:.8f}; h_achi={fineA[2][0,1]:.1e};"
              f" root A_chi direct/formula={rootA[1]:.10f}/{fr['Achi']:.10f}")

print("(b) curvature: finite difference of A_chi in a vs formula; chi-independence")
for t in [0.0, 0.5, 1.0, 1.0700330817481354, 2.0]:
    a, hh = 0.3, 1e-4
    al, et = al_t(t), et_t(t)
    Ap = formulas(t, a + hh, al[:2].sum(), et[:2].sum(), alpha0, eta0, al0s, et0s, kappa, b)["Achi"]
    Am = formulas(t, a - hh, al[:2].sum(), et[:2].sum(), alpha0, eta0, al0s, et0s, kappa, b)["Achi"]
    F = formulas(t, a, al[:2].sum(), et[:2].sum(), alpha0, eta0, al0s, et0s, kappa, b)["F"]
    c1 = collective(t, a, 0.1, al[:2], et[:2], rhsP, rhsQ, b)[1]
    c2 = collective(t, a, -0.6, al[:2], et[:2], rhsP, rhsQ, b)[1]
    print(f"  t={t:.4f}: dA_chi/da (FD)={(Ap - Am) / (2 * hh):.10f}  F formula={F:.10f}  A_chi(chi=0.1)-A_chi(chi=-0.6)={c1 - c2:.1e}")

print("(c) long-time behavior (block {1,2}, a=0.3)")
for label, eta_ref, kap_ in [("alpha0=eta0=1, kappa=1 (manuscript example)", 1.0, 1.0),
                             ("alpha0=1, eta0=2, kappa=1", 2.0, 1.0),
                             ("alpha0=eta0=1, kappa=0 (no interaction)", 1.0, 0.0)]:
    rhsP2, al_t2 = make_sector(al0s, alpha0, kap_)
    rhsQ2, et_t2 = make_sector(et0s + (eta_ref - 1.0), eta_ref, kap_)
    out = []
    for t in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]:
        al, et = al_t2(t), et_t2(t)
        fm = formulas(t, 0.3, al[:2].sum(), et[:2].sum(), alpha0, eta_ref, al0s, et0s + (eta_ref - 1.0), kap_, b)
        out.append(f"t={t}: F={fm['F']:.4e}, A_chi={fm['Achi']:.4f}, g_aa={fm['gaa']:.3e}, h_chichi={fm['hchi']:.3e}")
    print("  " + label)
    for o in out:
        print("    " + o)
