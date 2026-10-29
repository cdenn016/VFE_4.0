r"""GAP01-S3 s3: positive-eps structure of the nonflat window (pa:nonflat-window, 13:352-392; lp:nonflat, 14:166-178).

Exact facts used (derived in the report):
  * full flow: A_i(t) = alpha_i + eps_t e(omega_i), alpha_i = e^{-t}A_i(0), eps_t = eps(1-e^{-t}); context enters only
    through A_1(0) = a e(chi), so d_c A_1(t) = e^{-t} d_c A_1(0) exactly;  C_i(t) = eps^3 cbar_i(t) e_1.
  * joint orbit score (rotate phi and the record angle omega together): since e'(w).e(f) + e(w).e'(f) = 0, the
    record-driven part of A_i drops out and the marked orbit score of agent i is the centered -(alpha_i+zC_i).e'(phi),
    taken under the law vM(A_i + zC_i) that does contain the record part.
Reduced model (leading order): l_1 = eps*ubar_1(t) (limit public trajectory), cbar_i = (1-e^{-t})/5, Haar history,
label and history scores and the O(eps^3) context dependence of l, C, rows neglected.  Computes K, j, B on a (z,phi)
grid, averages over omega, minimizes, differentiates in context; compares with D17-PA's brute-force values
(scripts/D17-PA/geometry_eps_t*.out) and with the variant whose orbit score ignores the history rotation.
Floating point; supports, does not prove.
"""
import numpy as np
from scipy.special import i0, i1

NPH, NOM = 96, 48
ph = 2 * np.pi * np.arange(NPH) / NPH
om = 2 * np.pi * np.arange(NOM) / NOM
ustar = np.array([15 / 7, 15 / 7, 5 / 7]); uin = np.array([4.0, 0.5, -1.0])
e0 = uin - ustar; c1 = np.full(3, e0.mean()); cp = e0 - c1

def ubar(t):
    return ustar + np.exp(-t / 5) * c1 + np.exp(-7 * t / 5) * cp

def agent_terms(alpha, dalpha, l, C, eps_t, joint=True):
    """Return per-omega arrays: E[s_orb^2], E[s_X s_orb] (2,), E[s_X s_Y] (2,2) for one agent.
    alpha (2,), dalpha (2,2) columns = d alpha/d c_k, C (2,), l scalar.  Law of (z,phi): p ∝ exp(l z/2 + (A+zC).e(phi))."""
    E = np.stack([np.cos(ph), np.sin(ph)], 1)      # (NPH,2) e(phi)
    Ep = np.stack([-np.sin(ph), np.cos(ph)], 1)    # e'(phi)
    out_K = np.empty(NOM); out_j = np.empty((NOM, 2)); out_B = np.empty((NOM, 2, 2))
    for k, w in enumerate(om):
        A = alpha + eps_t * np.array([np.cos(w), np.sin(w)])
        logp = []
        for z in (-1.0, 1.0):
            logp.append(l * z / 2 + E @ (A + z * C))
        logp = np.array(logp)                        # (2,NPH)
        p = np.exp(logp - logp.max()); p /= p.sum()
        # contextual scores: d log p / d c_k = dalpha_k.e(phi) - E[.] (C, l context-independent here)
        sX = np.einsum("pd,dk->pk", E, dalpha)       # (NPH,2), same for both z
        sX = np.stack([sX, sX])                      # (2,NPH,2)
        mX = np.einsum("zp,zpk->k", p, sX); sX = sX - mX
        # orbit score
        rot = np.stack([Ep @ (alpha + z * C) for z in (-1.0, 1.0)]) if joint else np.stack([Ep @ (A + z * C) for z in (-1.0, 1.0)])
        so = -rot
        so = so - np.sum(p * so)
        out_K[k] = np.sum(p * so ** 2)
        out_j[k] = np.einsum("zp,zpk,zp->k", p, sX, so)
        out_B[k] = np.einsum("zp,zpk,zpl->kl", p, sX, sX)
    return out_K.mean(), out_j.mean(0), out_B.mean(0)

def geometry(c, t, eps, amps=(0.03, 0.025), dirs=(0.0, 1.0), joint=True):
    a, chi = c
    eps_t = eps * (1 - np.exp(-t))
    cbar = (1 - np.exp(-t)) / 5
    C = np.array([eps ** 3 * cbar, 0.0])
    ub = ubar(t)
    K = 0.0; j = np.zeros(2); B = np.zeros((2, 2))
    # agent 1 (context dependent)
    al1 = np.exp(-t) * a * np.array([np.cos(chi), np.sin(chi)])
    dal1 = np.exp(-t) * np.array([[np.cos(chi), -a * np.sin(chi)], [np.sin(chi), a * np.cos(chi)]])
    k1, j1, B1 = agent_terms(al1, dal1, eps * ub[0], C, eps_t, joint)
    K += k1; j += j1; B += B1
    for idx, (amp, d0) in enumerate(zip(amps, dirs)):
        al = np.exp(-t) * amp * np.array([np.cos(d0), np.sin(d0)])
        ki, _, _ = agent_terms(al, np.zeros((2, 2)), eps * ub[idx + 1], C, eps_t, joint)
        K += ki
    conn = -j / K
    h = B - np.outer(j, j) / K
    return conn, h, K

def curvature(c, t, eps, joint=True, dd=1e-4):
    ca = (geometry(c + np.array([dd, 0]), t, eps, joint=joint)[0] - geometry(c - np.array([dd, 0]), t, eps, joint=joint)[0]) / (2 * dd)
    cc = (geometry(c + np.array([0, dd]), t, eps, joint=joint)[0] - geometry(c - np.array([0, dd]), t, eps, joint=joint)[0]) / (2 * dd)
    return ca[1] - cc[0]  # F = d_a A_chi - d_chi A_a

def I_(x):
    return x * i1(x) / i0(x)

def closed_F(a, t, amps=(0.03, 0.025)):
    s = np.exp(-t) * a
    Bt = sum(I_(np.exp(-t) * x) for x in amps)
    ph2 = 2 * np.pi * np.arange(1024) / 1024
    w = np.exp(s * np.cos(ph2)); w /= w.mean()
    V = np.mean(w * np.cos(ph2) ** 2) - (i1(s) / i0(s)) ** 2
    return -Bt * np.exp(-t) * (i1(s) / i0(s) + s * V) / (I_(s) + Bt) ** 2

c0 = np.array([0.04, 0.3])
ref = {(1.0, 0.2): -12.1292, (1.0, 0.1): -12.4687, (1.0, 0.05): -12.4910, (1.0, 0.02): -12.4924,
       (5.0, 0.2): -0.4257, (5.0, 0.05): -12.3914}
print("(a) reduced model vs D17-PA brute force at c0=(0.04,0.3), amplitudes d=0.03, d3=0.025:")
for (t, eps), Fref in ref.items():
    F = curvature(c0, t, eps)
    Fnj = curvature(c0, t, eps, joint=False)
    print("   t=%.0f eps=%.2f: F_reduced=%.4f  F_D17PA=%.4f  F_eps0=%.4f  | orbit without history rotation: %.4f"
          % (t, eps, F, Fref, closed_F(c0[0], t), Fnj))

# (b) small-amplitude closed form with kappa = sum |C_i|^2 (three agents)
def F_small(a, t, eps, amps=(0.03, 0.025)):
    D = sum(x ** 2 for x in amps)
    kap = 3 * (eps ** 3 * (1 - np.exp(-t)) / 5) ** 2
    q = np.exp(-2 * t)
    return -2 * a * q * (q * D + kap) / (q * (a ** 2 + D) + kap) ** 2
print("(b) small-amplitude law F ~ -2a e^{-2t}(e^{-2t}D+kappa)/(e^{-2t}(a^2+D)+kappa)^2, kappa = 3 eps^6 (1-e^{-t})^2/25:")
for (t, eps), Fref in ref.items():
    print("   t=%.0f eps=%.2f: F_small=%.4f  F_D17PA=%.4f" % (t, eps, F_small(c0[0], t, eps), Fref))

# (c) leading-order window threshold on a patch: largest eps with |F_eps| >= |F_0|/2 and h_chichi>0 on the patch
print("(c) leading-order threshold eps*(T+) with |F_eps|>=|F_0|/2 on the patch a in [0.02,0.045], chi in {0,1.5,3}, t in [0,T+]:")
for Tp in (4.0, 5.0, 6.0, 8.0):
    def ok(eps):
        for t in np.linspace(0, Tp, 9):
            for a in (0.02, 0.03, 0.045):
                F0 = F_small(a, t, 0.0); Fe = F_small(a, t, eps)
                if abs(Fe) < abs(F0) / 2:
                    return False
        return True
    lo, hi = 1e-3, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    print("   T+=%.0f: eps* = %.4f   (scaling check eps*/e^{-T+/3} = %.4f)" % (Tp, lo, lo / np.exp(-Tp / 3)))
# reduced-model check of the threshold at one patch point
print("(d) reduced-model |F_eps|/|F_0| at a=0.02 (smallest marked amplitude), chi=0.3, t=5:")
for eps in (0.02, 0.05, 0.08, 0.12):
    F = curvature(np.array([0.02, 0.3]), 5.0, eps)
    print("   eps=%.2f: ratio %.4f (small-amplitude law %.4f)" % (eps, F / closed_F(0.02, 5.0), F_small(0.02, 5.0, eps) / F_small(0.02, 5.0, 0.0)))

# (e) fixed eps > 0, growing t: the locked inertia stays >= kappa/2 > 0 while contextual scores decay like e^{-t},
#     so the reduced-model curvature should tend to 0 (it tends to -2aD/(a^2+D)^2 only at eps = 0).
print("(e) reduced-model curvature at fixed eps as t grows (c0=(0.04,0.3)); eps=0 closed form for comparison:")
for eps in (0.05, 0.02):
    row = []
    for t in (5.0, 8.0, 11.0, 14.0):
        F = curvature(c0, t, eps)
        row.append("t=%.0f: %.4g (eps0 %.4f)" % (t, F, closed_F(c0[0], t)))
    print("   eps=%.2f: " % eps + "; ".join(row))
