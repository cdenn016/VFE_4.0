r"""GAP03-S2 s3: form:geometry-decay and form:anisotropic-reference.

(A) exact (sympy): p_* and lambda = -f_p(p_*, alpha0); 12 - 4 sqrt 2; kappa = 0 rate 4 and
    closed-form limits (exact solution p(t) = 1 + e^{-2t}(p0 - 1)).
(B) numerical (scipy DOP853, rtol 1e-12): integrate p_+, p_-, log(p_+ - p_-) and log dp/dp0
    with analytic divided differences (no cancellation), and show that
    e^{lam t}(p_+ - p_-), e^{lam t} d_a p_+-, e^{2 lam t} g_aa, e^{2 lam t} I_1 converge;
    compare the finite-time curvature with the limit formula; recompute the example numbers.
(C) the anisotropic remark: fixed points, implicit derivatives, limit tensor and curvature,
    and convergence of the finite-time quantities to them.
Single-threaded, short.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

out = []
# ---------------- (A) exact -----------------
p, k, a0, ab, gam = sp.symbols('p kappa alpha0 alphabar gamma', positive=True)
f = -2 * p + 2 + 2 * k * ab / p - k * a0 * p / (p + k * ab)
fp = sp.diff(f, p)
out.append("(A) f_p = %s" % sp.simplify(fp))
out.append("(A) f_p + 2 + 2 k ab/p^2 + k^2 a0 ab/(p + k ab)^2 = %s"
           % sp.simplify(fp + 2 + 2 * k * ab / p**2 + k**2 * a0 * ab / (p + k * ab)**2))
f11 = sp.factor(sp.together(f.subs({k: 1, a0: 1, ab: 1})))
out.append("(A) f(p, 1) at alpha0 = kappa = 1: %s" % f11)
ps = sp.sqrt(2)
lam = sp.simplify(-fp.subs({k: 1, a0: 1, ab: 1, p: ps}))
out.append("(A) f(sqrt2) = %s, lambda = %s, 2 lambda = %s = %.12f"
           % (sp.simplify(f.subs({k: 1, a0: 1, ab: 1, p: ps})), lam, sp.expand(2 * lam), float(2 * lam)))
out.append("(A) kappa = 0: f = %s, p_* = 1, lambda = %s, 2 lambda = 4" % (f.subs(k, 0), -fp.subs(k, 0)))
# kappa = 0 closed forms
t, aa, at, n, e0 = sp.symbols('t a atilde n eta0', positive=True)
pp = 1 + sp.exp(-2 * t) * (sp.exp(aa) - 1)
pm = 1 + sp.exp(-2 * t) * (sp.exp(-aa) - 1)
D = sp.simplify(pp - pm)
out.append("(A) kappa=0: p_+ - p_- = %s" % D)
gaa_scaled = sp.limit(sp.exp(4 * t) * n * a0 * ((sp.diff(pp, aa) / pp)**2 + (sp.diff(pm, aa) / pm)**2), t, sp.oo)
I1_scaled = sp.limit(sp.exp(4 * t) * 2 * n * a0 * D**2 / (pp * pm), t, sp.oo)
out.append("(A) kappa=0 (A_I -> n alpha0): lim e^{4t} g_aa = %s ; lim e^{4t} I_1 = %s"
           % (sp.simplify(gaa_scaled), sp.simplify(I1_scaled)))
L1 = 8 * n * a0 * sp.sinh(aa)**2
L2 = 8 * n * e0 * sp.sinh(at)**2
Finf0 = sp.simplify(-sp.diff(L1, aa) * L2 / (L1 + L2)**2)
out.append("(A) kappa=0 limit curvature coefficient = %s ; at a = atilde, alpha0 = eta0: %s ; value at 0.3: %.6f"
           % (Finf0, sp.simplify(Finf0.subs({at: aa, e0: a0})), float(Finf0.subs({at: aa, e0: a0}).subs(aa, sp.Rational(3, 10)))))

# ---------------- (B) numerical -----------------


def fnum(pv, abar, kap, al0, g=1.0):
    return -2 * pv + 2 * g + 2 * kap * abar / pv - kap * al0 * pv / (g * pv + kap * abar)


def fp_num(pv, abar, kap, al0, g=1.0):
    return -2 - 2 * kap * abar / pv**2 - kap**2 * al0 * abar / (g * pv + kap * abar)**2


def fdd(pv, qv, abar, kap, al0, g=1.0):
    """divided difference (f(p) - f(q))/(p - q), analytic."""
    return -2 - 2 * kap * abar / (pv * qv) - kap**2 * al0 * abar / ((g * pv + kap * abar) * (g * qv + kap * abar))


def fgam(pv, abar, kap, al0, g):
    """d f / d gamma"""
    return 2 + kap * al0 * pv**2 / (g * pv + kap * abar)**2


def sector(a_ctx, shape_mean0, al0, kap, Tend, aniso=False, gamma_ctx=None):
    """Integrate the two eigenvalues p_+ (init e^a), p_- (init e^-a), log D (D = p_+ - p_-),
    and the a-derivatives d_a p_+, d_a p_-.  If aniso, gamma = e^{+-a} enter the field
    (form:anisotropic-reference) and d_a p includes the gamma-derivative."""
    def abar(tt):
        return al0 + np.exp(-2 * tt) * (shape_mean0 - al0)

    gp, gm = (np.exp(a_ctx), np.exp(-a_ctx)) if aniso else (1.0, 1.0)

    def rhs(tt, y):
        pP, pM, logD, sP, sM = y
        A = abar(tt)
        if not aniso:
            # isotropic references: log-sensitivities d log S/dt = f_p, no decay of the unknowns
            return [fnum(pP, A, kap, al0), fnum(pM, A, kap, al0), fdd(pP, pM, A, kap, al0),
                    fp_num(pP, A, kap, al0), fp_num(pM, A, kap, al0)]
        # anisotropic references: D' = f(pP; gp) - f(pM; gm); sensitivities forced by d gamma/da
        Dv = np.exp(logD)
        dlogD = (fnum(pP, A, kap, al0, gp) - fnum(pM, A, kap, al0, gm)) / Dv
        ddP = fp_num(pP, A, kap, al0, gp) * sP + fgam(pP, A, kap, al0, gp) * gp
        ddM = fp_num(pM, A, kap, al0, gm) * sM - fgam(pM, A, kap, al0, gm) * gm
        return [fnum(pP, A, kap, al0, gp), fnum(pM, A, kap, al0, gm), dlogD, ddP, ddM]

    if aniso:
        y0 = [np.exp(a_ctx), np.exp(-a_ctx), np.log(np.exp(a_ctx) - np.exp(-a_ctx)), np.exp(a_ctx), -np.exp(-a_ctx)]
    else:
        y0 = [np.exp(a_ctx), np.exp(-a_ctx), np.log(np.exp(a_ctx) - np.exp(-a_ctx)), 0.0, 0.0]
    sol = solve_ivp(rhs, (0, Tend), y0, method='DOP853', rtol=1e-12, atol=1e-13, dense_output=True)
    sol.aniso, sol.a_ctx = aniso, a_ctx
    return sol


def pstar(al0, kap, g=1.0):
    from scipy.optimize import brentq
    return brentq(lambda pv: fnum(pv, al0, kap, al0, g), 1e-6, 50)


def block_quantities(tt, solq, sols, AI0, BI0, nI, al0, et0):
    pP, pM, logD, sP, sM = solq.sol(tt)
    qP, qM, logDt, _, _ = sols.sol(tt)
    if solq.aniso:
        dP, dM = sP, sM
    else:
        dP, dM = np.exp(solq.a_ctx) * np.exp(sP), -np.exp(-solq.a_ctx) * np.exp(sM)
    A_I = nI * al0 + np.exp(-2 * tt) * (AI0 - nI * al0)
    B_I = nI * et0 + np.exp(-2 * tt) * (BI0 - nI * et0)
    Dv, Dt = np.exp(logD), np.exp(logDt)
    gaa = A_I * ((dP / pP)**2 + (dM / pM)**2)
    I1 = 2 * A_I * Dv**2 / (pP * pM)
    I2 = 2 * B_I * Dt**2 / (qP * qM)
    daI1 = 2 * A_I * (2 * Dv * (dP - dM) / (pP * pM) - Dv**2 * (dP * pM + pP * dM) / (pP * pM)**2)
    F = -daI1 * I2 / (I1 + I2)**2
    hchi = I1 * I2 / (I1 + I2)
    Achi = -I1 / (I1 + I2)
    return dict(gaa=gaa, I1=I1, I2=I2, F=F, hchi=hchi, Achi=Achi, D=Dv, dP=dP, dM=dM, pP=pP, pM=pM)


al0, et0, kap = 1.0, 1.0, 1.0
u = np.array([1.0, 1.2, 1.8, 2.1])
eta = np.array([0.8, 1.1, 1.4, 1.7])
a_ctx, at_ctx = 0.3, 0.3
Tend = 12.0
solq = sector(a_ctx, u.mean(), al0, kap, Tend)
sols = sector(at_ctx, eta.mean(), et0, kap, Tend)
lam_b = -fp_num(pstar(al0, kap), al0, kap, al0)
lam_s = -fp_num(pstar(et0, kap), et0, kap, et0)
out.append("(B) p_* = %.12f (sqrt2 = %.12f); lambda = %.12f (6 - 2 sqrt2 = %.12f); model lambda = %.12f"
           % (pstar(al0, kap), np.sqrt(2), lam_b, 6 - 2 * np.sqrt(2), lam_s))
out.append("(B) convergence of scaled quantities, block {1,2} (A_I(0)=2.2, B_I(0)=1.9), a = atilde = 0.3:")
out.append("     t    e^{lam t}D    e^{lam t}d_a p+   e^{lam t}d_a p-   e^{2lam t}g_aa   e^{2lam t}I_1    F(t)        A_chi(t)")
for tt in [1, 2, 4, 6, 8, 10, 12]:
    q = block_quantities(tt, solq, sols, 2.2, 1.9, 2, al0, et0)
    out.append("   %4.1f  %.10f  %.10f   %.10f   %.10f    %.10f   %.8f  %.8f"
               % (tt, np.exp(lam_b * tt) * q['D'], np.exp(lam_b * tt) * q['dP'], np.exp(lam_b * tt) * q['dM'],
                  np.exp(2 * lam_b * tt) * q['gaa'], np.exp(2 * lam_b * tt) * q['I1'], q['F'], q['Achi']))
# limit curvature from limit constants at t = 12
q = block_quantities(12.0, solq, sols, 2.2, 1.9, 2, al0, et0)
cD = np.exp(lam_b * 12) * q['D']
cP, cM = np.exp(lam_b * 12) * q['dP'], np.exp(lam_b * 12) * q['dM']
qs = sols.sol(12.0)
cDt = np.exp(lam_s * 12) * np.exp(qs[2])
ps_b, ps_s = pstar(al0, kap), pstar(et0, kap)
L1 = 2 * 2 * al0 * cD**2 / ps_b**2
L2 = 2 * 2 * et0 * cDt**2 / ps_s**2
daL1 = 2 * 2 * al0 * 2 * cD * (cP - cM) / ps_b**2
out.append("(B) limit-constant formula F_inf = -(d_a L1) L2/(L1+L2)^2 = %.8f ; finite-time F(12) = %.8f"
           % (-daL1 * L2 / (L1 + L2)**2, q['F']))

# example numbers (paragraph after form:geometry-decay)
t2 = 0.5 * np.log(8.5)
g0 = block_quantities(1e-12, solq, sols, 2.2, 1.9, 2, al0, et0)
vals = [block_quantities(tt, solq, sols, 2.2, 1.9, 2, al0, et0) for tt in (1.0, 2.0, t2)]
out.append("(B) g_aa at t = 0, 1, 2: %.4f, %.4e, %.4e ; g_aa(t_2)/g_aa(0) = %.4e ; h_chichi(t_2)/h_chichi(0) = %.4e (t_2 = %.6f)"
           % (g0['gaa'], vals[0]['gaa'], vals[1]['gaa'], vals[2]['gaa'] / g0['gaa'], vals[2]['hchi'] / g0['hchi'], t2))
# companion: "by the root join it has lost more than three orders of magnitude" on a in (1/5, 2/5)
rat = []
for ac in [0.2, 0.25, 0.3, 0.35, 0.4]:
    sq = sector(ac, u.mean(), al0, kap, 2.0)
    b0 = block_quantities(1e-12, sq, sector(0.3, eta.mean(), et0, kap, 2.0), 2.2, 1.9, 2, al0, et0)
    bt = block_quantities(t2, sq, sector(0.3, eta.mean(), et0, kap, 2.0), 2.2, 1.9, 2, al0, et0)
    rat.append((ac, bt['gaa'] / b0['gaa'], bt['hchi'] / b0['hchi']))
out.append("(B) companion check, block {1,2}: (a, g_aa(t2)/g_aa(0), h_chichi(t2)/h_chichi(0)) = "
           + "; ".join("(%.2f, %.3e, %.3e)" % r for r in rat))
# other blocks at a = 0.3: block {3,4} and the root
for (lab, AI0, BI0, nI) in [("{3,4}", 3.9, 3.1, 2), ("root", 6.1, 5.0, 4)]:
    b0 = block_quantities(1e-12, solq, sols, AI0, BI0, nI, al0, et0)
    bt = block_quantities(t2, solq, sols, AI0, BI0, nI, al0, et0)
    out.append("(B) block %s: g_aa(t2)/g_aa(0) = %.3e, h_chichi(t2)/h_chichi(0) = %.3e, F(12) = %.6f"
               % (lab, bt['gaa'] / b0['gaa'], bt['hchi'] / b0['hchi'],
                  block_quantities(12.0, solq, sols, AI0, BI0, nI, al0, et0)['F']))
# sensitivity of the curvature limit to the unstated atilde and to model shapes in the neighborhood
sens = []
for atv in [0.2, 0.3, 0.5]:
    for em in [1.20, 1.25, 1.30]:
        ss = sector(atv, em, et0, kap, Tend)
        sens.append((atv, em, block_quantities(12.0, solq, ss, 2.2, 1.9, 2, al0, et0)['F']))
out.append("(B) F(12) for (atilde, model mean shape): " + "; ".join("(%.2f, %.2f) -> %.4f" % s for s in sens))
# eta0 = 2: rates differ -> A_chi -> -1, F -> 0
sols2 = sector(at_ctx, eta.mean() + 1.0, 2.0, kap, Tend)  # model shapes shifted so that they relax to 2
for tt in [2, 6, 12]:
    q2 = block_quantities(tt, solq, sols2, 2.2, 1.9 + 2.0, 2, al0, 2.0)
    out.append("(B) eta0 = 2: t = %4.1f  A_chi = %.6f  F = %.3e" % (tt, q2['Achi'], q2['F']))
out.append("(B) model lambda at eta0 = 2: %.6f > belief lambda %.6f"
           % (-fp_num(pstar(2.0, kap), 2.0, kap, 2.0), lam_b))
# kappa = 0 numerics against the closed form
solq0 = sector(a_ctx, u.mean(), al0, 0.0, Tend)
sols0 = sector(at_ctx, eta.mean(), et0, 0.0, Tend)
q0 = block_quantities(12.0, solq0, sols0, 2.2, 1.9, 2, al0, et0)
out.append("(B) kappa = 0: e^{4t} g_aa(12) = %.10f vs 2 n alpha0 cosh(2a) = %.10f ; F(12) = %.8f vs -coth(0.3)/2 = %.8f"
           % (np.exp(4 * 12) * q0['gaa'], 2 * 2 * np.cosh(0.6), q0['F'], -1 / np.tanh(0.3) / 2))

# ---------------- (C) anisotropic remark -----------------


def aniso_limits(a_c, at_c, al0, et0, kap, nI):
    pPs, pMs = pstar(al0, kap, np.exp(a_c)), pstar(al0, kap, np.exp(-a_c))
    qPs, qMs = pstar(et0, kap, np.exp(at_c)), pstar(et0, kap, np.exp(-at_c))
    # implicit derivative d p_*/d a = -(f_gamma dgamma/da)/f_p
    dP = -fgam(pPs, al0, kap, al0, np.exp(a_c)) * np.exp(a_c) / fp_num(pPs, al0, kap, al0, np.exp(a_c))
    dM = fgam(pMs, al0, kap, al0, np.exp(-a_c)) * np.exp(-a_c) / fp_num(pMs, al0, kap, al0, np.exp(-a_c))
    gaa = nI * al0 * ((dP / pPs)**2 + (dM / pMs)**2)
    I1 = 2 * nI * al0 * (pPs - pMs)**2 / (pPs * pMs)
    I2 = 2 * nI * et0 * (qPs - qMs)**2 / (qPs * qMs)
    daI1 = 2 * nI * al0 * (2 * (pPs - pMs) * (dP - dM) / (pPs * pMs) - (pPs - pMs)**2 * (dP * pMs + pPs * dM) / (pPs * pMs)**2)
    return dict(pP=pPs, pM=pMs, qP=qPs, qM=qMs, gaa=gaa, hchi=I1 * I2 / (I1 + I2), F=-daI1 * I2 / (I1 + I2)**2)


Lm = aniso_limits(0.3, 0.3, 1.0, 1.0, 1.0, 2)
out.append("(C) anisotropic: fixed points p_*(e^{+-a}) = %.6f, %.6f; model %.6f, %.6f" % (Lm['pP'], Lm['pM'], Lm['qP'], Lm['qM']))
out.append("(C) limit tensor (n_I = 2): g_aa = %.6f, h_chichi = %.6f; limit curvature = %.6f" % (Lm['gaa'], Lm['hchi'], Lm['F']))
# finite-time convergence (integrate the anisotropic belief and model sectors)
sqa = sector(0.3, u.mean(), 1.0, 1.0, Tend, aniso=True)
ssa = sector(0.3, eta.mean(), 1.0, 1.0, Tend, aniso=True)
for tt in [1, 4, 12]:
    qa = block_quantities(tt, sqa, ssa, 2.2, 1.9, 2, 1.0, 1.0)
    out.append("(C) t = %4.1f: g_aa = %.6f, h_chichi = %.6f, F = %.6f" % (tt, qa['gaa'], qa['hchi'], qa['F']))
# uniqueness of the anisotropic fixed point: f_p < -2 symbolic
fa = -2 * p + 2 * gam + 2 * k * ab / p - k * a0 * p / (gam * p + k * ab)
out.append("(C) anisotropic f_p = %s" % sp.simplify(sp.diff(fa, p)))
for line in out:
    print(line)
