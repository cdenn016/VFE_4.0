r"""GAP03-S2 s4: form:geometry beyond its gauge structure.

Independent implementation of the collective minimizing connection (geo:collective) with the
FULL Wishart Fisher metric in coordinates (nu, p11, p12, p22) (shape-precision cross terms
included), unit weights, and the common SO(2) acting on belief and model samples.

(a) prepared section form:context at t = 0 and along the flow (eigen solution): fine block
    (children) vs retained parent: A^F, h^min, curvature (finite differences) vs the closed
    forms form:connection, form:curvature.
(b) a section of the common-precision family whose SHAPES vary with the context: the
    connection class is still inherited (geo:retained-connection holds for every first jet,
    because Delta = g - R^* gbar vanishes on all precision directions), while the tensor
    defect equals the shape-sector Fisher loss
        Delta(r_a, r_a) = sum_i psi2'(alpha_i) (d_a alpha_i)^2 - psi2'(A) (d_a A)^2 > 0.
(c) Delta on random tangent vectors: zero whenever the shape part vanishes; mixed pairing
    with orbit directions identically zero.
Single-threaded, short.
"""
import numpy as np
from scipy.special import polygamma

J0 = np.array([[0.0, -1.0], [1.0, 0.0]])


def psi2p(a):
    return polygamma(1, a) + polygamma(1, a - 0.5)


def to_coords(U):
    return np.array([U[0, 0], U[0, 1], U[1, 1]])


def fisher(nu, P):
    """Fisher matrix of W_2(nu,P) in coordinates (nu, p11, p12, p22): Hessian of
    F = log Gamma_2(nu) - nu log det P."""
    Pi = np.linalg.inv(P)
    # d log det P / d(p11,p12,p22) = (Pi11, 2 Pi12, Pi22)
    g = np.array([Pi[0, 0], 2 * Pi[0, 1], Pi[1, 1]])
    # Hessian of log det P in these coordinates: -Tr(Pi E_a Pi E_b), E_a basis matrices
    E = [np.array([[1.0, 0], [0, 0]]), np.array([[0, 1.0], [1.0, 0]]), np.array([[0, 0], [0, 1.0]])]
    Hld = np.array([[-np.trace(Pi @ Ea @ Pi @ Eb) for Eb in E] for Ea in E])
    G = np.zeros((4, 4))
    G[0, 0] = psi2p(nu)
    G[0, 1:] = G[1:, 0] = -g
    G[1:, 1:] = -nu * Hld
    return G


def collective(tangents, orbits, laws):
    """tangents: list of 4-vectors d p_k(X) for each law k; orbits: list of 4-vectors J_k (generator);
    laws: list of (nu, P).  Returns (xi*, minimized cost) for cost sum_k |d p_k + xi J_k|^2."""
    A = sum(J @ fisher(*L) @ J for J, L in zip(orbits, laws))
    B = sum(t @ fisher(*L) @ J for t, J, L in zip(tangents, orbits, laws))
    C = sum(t @ fisher(*L) @ t for t, L in zip(tangents, laws))
    xi = -B / A
    return xi, C - B**2 / A


def section_P(a, chi, t, kap=1.0, al0=1.0, abar0=1.525, T=None):
    """common belief precision along the full flow from P(0) = R diag(e^a, e^-a) R^T (eigen solution)."""
    from scipy.integrate import solve_ivp

    def f(tt, y):
        A = al0 + np.exp(-2 * tt) * (abar0 - al0)
        return [-2 * p + 2 + 2 * kap * A / p - kap * al0 * p / (p + kap * A) for p in y]
    if t == 0:
        lam = [np.exp(a), np.exp(-a)]
    else:
        lam = solve_ivp(f, (0, t), [np.exp(a), np.exp(-a)], method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    R = np.array([[np.cos(chi), -np.sin(chi)], [np.sin(chi), np.cos(chi)]])
    return R @ np.diag(lam) @ R.T


def section_Q(t, at=0.3, kap=1.0, et0=1.0, ebar0=1.25):
    from scipy.integrate import solve_ivp

    def f(tt, y):
        A = et0 + np.exp(-2 * tt) * (ebar0 - et0)
        return [-2 * p + 2 + 2 * kap * A / p - kap * et0 * p / (p + kap * A) for p in y]
    lam = [np.exp(at), np.exp(-at)] if t == 0 else solve_ivp(f, (0, t), [np.exp(at), np.exp(-at)], method='DOP853', rtol=1e-12, atol=1e-14).y[:, -1]
    return np.diag(lam)


def block_geometry(alphas, etas, a, chi, t, dalpha=None, h=1e-5):
    """Fine (children) and parent minimizing connection and minimized tensor at context (a,chi), time t.
    alphas, etas: shapes at this time (context-constant unless dalpha given: d alpha_i/da)."""
    n = len(alphas)
    if dalpha is None:
        dalpha = np.zeros(n)
    P = section_P(a, chi, t)
    Q = section_Q(t)
    dPa = (section_P(a + h, chi, t) - section_P(a - h, chi, t)) / (2 * h)
    dPc = (section_P(a, chi + h, t) - section_P(a, chi - h, t)) / (2 * h)
    JP = J0 @ P - P @ J0
    JQ = J0 @ Q - Q @ J0
    res = {}
    for label, laws_b, laws_m, dsh in [("fine", [(al, P) for al in alphas], [(et, Q) for et in etas], dalpha),
                                       ("parent", [(sum(alphas), P)], [(sum(etas), Q)], np.array([dalpha.sum()]))]:
        laws = laws_b + laws_m
        orbits = [np.concatenate([[0.0], to_coords(JP)])] * len(laws_b) + [np.concatenate([[0.0], to_coords(JQ)])] * len(laws_m)
        out = {}
        for X, dP in (("a", dPa), ("chi", dPc)):
            tb = [np.concatenate([[dsh[i] if X == "a" else 0.0], to_coords(dP)]) for i in range(len(laws_b))]
            tm = [np.zeros(4) for _ in laws_m]
            out[X] = collective(tb + tm, orbits, laws)
        # mixed component via polarization: X = a + chi
        tb = [np.concatenate([[dsh[i]], to_coords(dPa + dPc)]) for i in range(len(laws_b))]
        tm = [np.zeros(4) for _ in laws_m]
        _, hsum = collective(tb + tm, orbits, laws)
        out["h_ac"] = 0.5 * (hsum - out["a"][1] - out["chi"][1])
        res[label] = out
    return res, P, Q


out = []
alphas0 = [1.0, 1.2]   # block {1,2}, example belief shapes
etas0 = [0.8, 1.1]
for t in [0.0, 0.6]:
    # shapes at time t along the flow (alpha0 = eta0 = 1)
    al = [1 + np.exp(-2 * t) * (x - 1) for x in alphas0]
    et = [1 + np.exp(-2 * t) * (x - 1) for x in etas0]
    res, P, Q = block_geometry(al, et, 0.3, 0.1, t)
    ev = np.linalg.eigvalsh(P)[::-1]
    evq = np.linalg.eigvalsh(Q)[::-1]
    A_I, B_I = sum(al), sum(et)
    I1 = 2 * A_I * (ev[0] - ev[1])**2 / (ev[0] * ev[1])
    I2 = 2 * B_I * (evq[0] - evq[1])**2 / (evq[0] * evq[1])
    out.append("(a) t=%.1f: A_chi fine %.12f parent %.12f formula %.12f | A_a fine %.2e parent %.2e"
               % (t, res['fine']['chi'][0], res['parent']['chi'][0], -I1 / (I1 + I2), res['fine']['a'][0], res['parent']['a'][0]))
    out.append("        h_chichi fine %.12f parent %.12f formula %.12f | h_aa fine %.12f parent %.12f | h_achi %.2e %.2e"
               % (res['fine']['chi'][1], res['parent']['chi'][1], I1 * I2 / (I1 + I2), res['fine']['a'][1], res['parent']['a'][1],
                  res['fine']['h_ac'], res['parent']['h_ac']))
    # curvature: F = d_a A_chi - d_chi A_a, finite differences
    h = 1e-4
    rp, _, _ = block_geometry(al, et, 0.3 + h, 0.1, t)
    rm, _, _ = block_geometry(al, et, 0.3 - h, 0.1, t)
    cp, _, _ = block_geometry(al, et, 0.3, 0.1 + h, t)
    cm, _, _ = block_geometry(al, et, 0.3, 0.1 - h, t)
    for lab in ("fine", "parent"):
        Fv = (rp[lab]['chi'][0] - rm[lab]['chi'][0]) / (2 * h) - (cp[lab]['a'][0] - cm[lab]['a'][0]) / (2 * h)
        out.append("        curvature %s (finite differences) = %.8f" % (lab, Fv))
    # closed form curvature: -(d_a I1) I2/(I1+I2)^2 with d_a I1 by differences of the eigen solution
    Pp = section_P(0.3 + h, 0.1, t); Pm = section_P(0.3 - h, 0.1, t)
    def I1f(PP):
        e = np.linalg.eigvalsh(PP)[::-1]
        return 2 * A_I * (e[0] - e[1])**2 / (e[0] * e[1])
    daI1 = (I1f(Pp) - I1f(Pm)) / (2 * h)
    out.append("        curvature closed form -(d_a I1) I2/(I1+I2)^2 = %.8f" % (-daI1 * I2 / (I1 + I2)**2))

# (b) context-varying shapes: connection inherited, tensor defect = shape Fisher loss
al = [1.0, 1.2]
et = [0.8, 1.1]
for dal in [np.array([0.3, -0.1]), np.array([0.2, 0.2]), np.array([1.0, -1.0])]:
    res, P, Q = block_geometry(al, et, 0.3, 0.1, 0.0, dalpha=dal)
    defect = res['fine']['a'][1] - res['parent']['a'][1]
    pred = sum(psi2p(x) * d**2 for x, d in zip(al, dal)) - psi2p(sum(al)) * dal.sum()**2
    out.append("(b) d alpha/da = %s: A_a fine %.3e parent %.3e; A_chi fine %.12f parent %.12f; "
               "h_aa defect %.10f, predicted shape Fisher loss %.10f"
               % (dal, res['fine']['a'][0], res['parent']['a'][0], res['fine']['chi'][0], res['parent']['chi'][0], defect, pred))

# (c) Delta = g_fine - R^* g_parent on the common-precision family, random tangents
rng = np.random.default_rng(7)
P = section_P(0.3, 0.1, 0.0)
alv = np.array([0.9, 1.4, 2.2])
maxmix, maxprec = 0.0, 0.0
minshape = np.inf
for _ in range(200):
    U = rng.normal(size=(2, 2)); U = U + U.T
    V = rng.normal(size=(2, 2)); V = V + V.T
    da, db = rng.normal(size=3), rng.normal(size=3)
    def gfine(d1, U1, d2, U2):
        return sum(np.concatenate([[d1[i]], to_coords(U1)]) @ fisher(alv[i], P) @ np.concatenate([[d2[i]], to_coords(U2)]) for i in range(3))
    def gpar(d1, U1, d2, U2):
        return np.concatenate([[d1.sum()], to_coords(U1)]) @ fisher(alv.sum(), P) @ np.concatenate([[d2.sum()], to_coords(U2)])
    Jt = J0 @ P - P @ J0
    mix = gfine(da, U, np.zeros(3), Jt) - gpar(da, U, np.zeros(3), Jt)
    prec = gfine(np.zeros(3), U, np.zeros(3), V) - gpar(np.zeros(3), U, np.zeros(3), V)
    sh = gfine(da, np.zeros((2, 2)), da, np.zeros((2, 2))) - gpar(da, np.zeros((2, 2)), da, np.zeros((2, 2)))
    maxmix, maxprec = max(maxmix, abs(mix)), max(maxprec, abs(prec))
    minshape = min(minshape, sh / (da @ da))
out.append("(c) Delta(v, J xi) max over 200 random v (with shape parts): %.2e ; Delta on precision directions: %.2e ; "
           "min Delta(v,v)/|d alpha|^2 on shape directions: %.6f (> 0)" % (maxmix, maxprec, minshape))
for line in out:
    print(line)
