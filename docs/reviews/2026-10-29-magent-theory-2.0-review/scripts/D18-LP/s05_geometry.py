r"""D18-LP check 5: collective Fisher geometry identities of chapter 14.

(a) lp:nonflat at eps=delta=0 by quadrature: connection A_chi, curvature coefficient (FD in a),
    h_aa, h_chichi, h_achi.
(b) Generic positive-parameter state, conditional experiment with recognition, presentation and
    Gibbs-label factors: (i) rows are invariant under the passive marked action (with evaluator
    direction rotated) and under common physical rotation; (ii) the physical tangent is
    orthogonal to the marked orbit; (iii) a phase-field parent  xbar = R_{-Theta(c)} x  has the
    same minimizing connection and  hbar - h = -2 (X Theta) b(X) + (X Theta)^2 I_phys  with
    b(X) = sum_i (X arg B_i) I(|B_i|), I_phys = sum_i I(|B_i|), I(r) = r I1(r)/I0(r)
    (lp:initial-phase-score-defect, lp:path-memory-polynomial, lp:physical-phase-inertia).
"""
import numpy as np
from scipy.special import i0e, i1e
import lp_model as M

Ifun = lambda r: r * i1e(r) / i0e(r)
N = 128
ph = 2 * np.pi * np.arange(N) / N

# (a) zero-parameter limit
def vm(A):
    d = np.exp(A[0] * np.cos(ph) + A[1] * np.sin(ph)); return d / d.sum()


def marked_geometry(a, chi, t, d=0.45, A3=np.array([0.2, 0.3])):
    v = np.exp(-t) * a
    A = [v * np.array([np.cos(chi), np.sin(chi)]), np.exp(-t) * d * np.array([1.0, 0.0]), np.exp(-t) * A3]
    p = [vm(Ai) for Ai in A]
    orb = [-Ai[1] * np.cos(ph) + Ai[0] * np.sin(ph) for Ai in A]  # (J2 A).e_phi : score of passive frame rotation A -> R_g A
    orb = [o - np.sum(pi * o) for o, pi in zip(orb, p)]
    s_a = np.exp(-t) * (np.cos(ph - chi)); s_a = s_a - np.sum(p[0] * s_a)
    s_c = v * np.sin(ph - chi); s_c = s_c - np.sum(p[0] * s_c)
    K = sum(np.sum(pi * o * o) for pi, o in zip(p, orb))
    j = {X: np.sum(p[0] * sX * orb[0]) for X, sX in (("a", s_a), ("chi", s_c))}
    Bm = {(X, Y): np.sum(p[0] * sX * sY) for X, sX in (("a", s_a), ("chi", s_c)) for Y, sY in (("a", s_a), ("chi", s_c))}
    Acon = {X: -j[X] / K for X in j}
    h = {k: Bm[k] - j[k[0]] * j[k[1]] / K for k in Bm}
    return Acon, h, K


a, chi, t, d = 0.7, 0.3, 0.6, 0.45
A3 = np.array([0.2, 0.3])
Acon, h, K = marked_geometry(a, chi, t, d, A3)
v = np.exp(-t) * a
Dt = Ifun(np.exp(-t) * d) + Ifun(np.exp(-t) * np.hypot(*A3))
p1 = vm(v * np.array([1.0, 0.0]))
var = np.sum(p1 * np.cos(ph) ** 2) - np.sum(p1 * np.cos(ph)) ** 2
print("(a) A_chi:", Acon["chi"], " closed:", -Ifun(v) / (Ifun(v) + Dt), " A_a:", Acon["a"])
print("    h_aa:", h["a", "a"], " closed:", np.exp(-2 * t) * var)
print("    h_chichi:", h["chi", "chi"], " closed:", Ifun(v) * Dt / (Ifun(v) + Dt), " h_achi:", h["a", "chi"])
hh = 1e-5
F_fd = (marked_geometry(a + hh, chi, t, d, A3)[0]["chi"] - marked_geometry(a - hh, chi, t, d, A3)[0]["chi"]) / (2 * hh) \
     - (marked_geometry(a, chi + hh, t, d, A3)[0]["a"] - marked_geometry(a, chi - hh, t, d, A3)[0]["a"]) / (2 * hh)
Ip = (Ifun(v + 1e-6) - Ifun(v - 1e-6)) / 2e-6
print("    F_{a chi} (FD of dA):", F_fd, " closed:", -Dt * np.exp(-t) * Ip / (Ifun(v) + Dt) ** 2)
print("    I'(v) =", Ip, " = I1/I0 + v Var:", i1e(v) / i0e(v) + v * var)

# (b) generic state
rng = np.random.default_rng(7)
J = 0.3
Z_, PHI, PSI = np.meshgrid(np.array([-1.0, 1.0]), 2 * np.pi * np.arange(48) / 48, 2 * np.pi * np.arange(48) / 48, indexing="ij")
Wq = np.full(Z_.shape, 0.5 / 48 / 48)
Tstat = np.stack([Z_ / 2, np.cos(PHI), np.sin(PHI), Z_ * np.cos(PHI), Z_ * np.sin(PHI), np.cos(PSI), np.sin(PSI)])


def cov_T(p):
    l, A, C, B = p[0], p[1:3], p[3:5], p[5:7]
    lg = l * Z_ / 2 + (A[0] + Z_ * C[0]) * np.cos(PHI) + (A[1] + Z_ * C[1]) * np.sin(PHI) + B[0] * np.cos(PSI) + B[1] * np.sin(PSI)
    q = np.exp(lg); q /= np.sum(Wq * q)
    m = np.array([np.sum(Wq * q * T) for T in Tstat])
    return np.array([[np.sum(Wq * q * (Tstat[i] - m[i]) * (Tstat[k] - m[k])) for k in range(7)] for i in range(7)])


def section(c):
    x = np.zeros((3, 8))
    x[:, 0] = [0.5 + 0.2 * c, 0.2, -0.3 + 0.1 * c ** 2]
    x[0, 1:3] = (0.6 + 0.2 * c) * np.array([np.cos(0.5 * c), np.sin(0.5 * c)])
    x[1, 1:3] = [0.4, 0.1 * c]; x[2, 1:3] = [0.3, -0.2]
    x[:, 3:5] = 0.1 * np.array([[1.0, 0.2 * c], [0.3, -0.1], [-0.2, 0.1 * c]])
    th = np.array([0.1, 0.6, 1.0]) + np.array([0.3, -0.2, 0.5]) * c
    amp = np.array([0.8, 0.6, 0.9]) + np.array([0.1, -0.1, 0.05]) * c
    x[:, 5:7] = amp[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
    x[:, 7] = [0.2 + 0.1 * c, -0.3, 0.1]
    return x


def tangent_norm_parts(x, dx, dbeta, dgam):
    """Inner products of tangents (dx, dbeta, dgam) at x in the conditional experiment metric."""
    x = x.reshape(3, 8)
    beta, gam = M.content_rows(x, J), M.pres_rows(x)
    def ip(u, w):
        (dxu, dbu, dgu), (dxw, dbw, dgw) = u, w
        tot = 0.0
        for i in range(3):
            Cv = cov_T(x[i, :7])
            tot += dxu[i, :7] @ Cv @ dxw[i, :7]
            s = 1 / (1 + np.exp(-x[i, 7])); tot += dxu[i, 7] * dxw[i, 7] * s * (1 - s) / 4 * 4 / 4
            tot += np.sum(dbu[i] * dbw[i] / beta[i]) + np.sum(dgu[i] * dgw[i] / gam[i])
        return tot
    return ip


x0 = section(0.0)
hfd = 1e-5
dx = (section(hfd) - section(-hfd)) / (2 * hfd)
db = (M.content_rows(section(hfd), J) - M.content_rows(section(-hfd), J)) / (2 * hfd)
dg = (M.pres_rows(section(hfd)) - M.pres_rows(section(-hfd))) / (2 * hfd)
# marked orbit tangent (passive: rotate A, C and evaluator direction e1)
def marked(x, g):
    y = x.copy(); R = M.rot(g)
    y[:, 1:3] = x[:, 1:3] @ R.T; y[:, 3:5] = x[:, 3:5] @ R.T
    return y
o_x = (marked(x0, hfd) - marked(x0, -hfd)) / (2 * hfd)
o_b = (M.content_rows(marked(x0, hfd), J, M.rot(hfd) @ [1, 0]) - M.content_rows(marked(x0, -hfd), J, M.rot(-hfd) @ [1, 0])) / (2 * hfd)
p_x = M.calJ(x0)
p_b = (M.content_rows(M.rotB(x0, M.rot(hfd)), J) - M.content_rows(M.rotB(x0, M.rot(-hfd)), J)) / (2 * hfd)
print("(b)(i) row tangent along passive marked action:", np.abs(o_b).max(), "; along physical rotation:", np.abs(p_b).max())
ip = tangent_norm_parts(x0, None, None, None)
Z3 = np.zeros((3, 3))
uX = (dx, db, dg); oT = (o_x, Z3, Z3); pT = (p_x, Z3, Z3)
print("(b)(ii) <physical, marked orbit> =", ip(pT, oT))
Iphys = ip(pT, pT)
Bs = x0[:, 5:7]
print("     I_phys =", Iphys, " closed sum I(|B_i|):", sum(Ifun(np.hypot(*Bi)) for Bi in Bs))
uo, oo = ip(uX, oT), ip(oT, oT)
h = ip(uX, uX) - uo ** 2 / oo
Acon = -uo / oo
th = lambda c: np.arctan2(section(c)[:, 6], section(c)[:, 5])
dth = (th(hfd) - th(-hfd)) / (2 * hfd)
bX_closed = sum(dth[i] * Ifun(np.hypot(*Bs[i])) for i in range(3))
print("     b(X) = <u_X, physical> =", ip(uX, pT), " closed sum (X arg B_i) I(|B_i|):", bX_closed)
worst = 0.0
for lam in [-0.7, -0.2, 0.3, 1.1]:
    ub = (dx - lam * p_x, db, dg)
    hb = ip(ub, ub) - ip(ub, oT) ** 2 / oo
    Ab = -ip(ub, oT) / oo
    pred = h - 2 * lam * ip(uX, pT) + lam ** 2 * Iphys
    worst = max(worst, abs(hb - pred), abs(Ab - Acon))
print("(b)(iii) phase-field parent: max |hbar - (h - 2 lam b + lam^2 I)|, |Abar - A| over lam =", worst)
lam_reflect = 2 * ip(uX, pT) / Iphys
ub = (dx - lam_reflect * p_x, db, dg)
print("     reflection value X Theta = 2 b/I =", lam_reflect, " gives hbar - h =", ip(ub, ub) - ip(ub, oT) ** 2 / oo - h)
