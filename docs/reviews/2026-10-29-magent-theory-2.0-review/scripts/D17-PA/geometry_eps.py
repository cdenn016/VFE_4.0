r"""D17-PA check 6: record-inclusive joint experiment of pa:joint-record-geometry at small positive eps.

Three agents of pa:angular-family, contextual section A_1(0)=a e(chi), A_2(0)=d e_1, A_3(0)=d3 e(al3),
C(0)=0, l(0)=eps*(4,1/2,-1), records o=(1,1,-1) with angular records omega_i drawn from the generative
predictive kernels of the agents (conditioned on o). Joint law Pi_c(dh) Q^h_ct(dy) with Gibbs-label laws.
Computes the locked inertia K, momentum j, unminimized tensor B, the minimizing connection, its curvature
and the minimized tensor, and compares them with the eps=0 closed forms of pa:nonflat-limit.
Also reports the size of the history orbit score (the contextual history score vanishes identically here).
Usage: python geometry_eps.py T EPS1,EPS2,...
"""
import sys
import time
import numpy as np
from scipy.special import i0e, i1e, i0, i1

T0 = time.time()
a_pr, p_pr = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])
d, d3, al3 = 0.03, 0.025, 1.0
u0 = np.array([4.0, 0.5, -1.0])
NW = 6
wg = 2 * np.pi * np.arange(NW) / NW
W1, W2, W3 = np.meshgrid(wg, wg, wg, indexing="ij")
OM = np.stack([W1.ravel(), W2.ravel(), W3.ravel()], axis=1)
NPH = 40
ph = 2 * np.pi * np.arange(NPH) / NPH

def logI0(r):
    return np.log(i0e(r)) + np.abs(r)

def ratio(r):
    return np.where(r > 1e-300, i1e(r) / i0e(r), 0.0)

def rows_and_L(S, eps):
    J = eps ** 3
    l = S[..., 0]; A = S[..., 1:3]; C = S[..., 3:5]
    rp = np.linalg.norm(A + C, axis=-1); rm = np.linalg.norm(A - C, axis=-1)
    L = l + logI0(rp) - logI0(rm)
    lp = l / 2 + logI0(rp); lmn = -l / 2 + logI0(rm); mx = np.maximum(lp, lmn)
    Pp = np.exp(lp - mx); Pm = np.exp(lmn - mx); s = Pp + Pm; Pp /= s; Pm /= s
    gp = ratio(rp)[..., None] * (A + C) / np.maximum(rp, 1e-300)[..., None]
    gm = ratio(rm)[..., None] * (A - C) / np.maximum(rm, 1e-300)[..., None]
    t_i = Pp * gp[..., 0] - Pm * gm[..., 0]
    m = np.tanh(L / 2)
    w = np.empty(S.shape[:-1] + (3,))
    for i in range(3):
        for j in range(3):
            if i == j:
                w[..., i, j] = a_pr * np.exp(J * t_i[..., i] - logI0(J))
            else:
                w[..., i, j] = p_pr * np.exp(L[..., j] * m[..., i] / 2) / np.cosh(L[..., j] / 2)
    beta = w / w.sum(axis=-1, keepdims=True)
    return beta, L

def field(S, eps, om):
    beta, L = rows_and_L(S, eps)
    out = np.empty_like(S)
    J = eps ** 3
    for i in range(3):
        out[..., i, 0] = -S[..., i, 0] + sum(beta[..., i, j] * L[..., j] for j in range(3) if j != i) + eps * o[i]
        out[..., i, 1] = -S[..., i, 1] + eps * np.cos(om[:, i])
        out[..., i, 2] = -S[..., i, 2] + eps * np.sin(om[:, i])
        out[..., i, 3] = -S[..., i, 3] + beta[..., i, i] * J
        out[..., i, 4] = -S[..., i, 4]
    return out

def integrate(c, eps, om, tt, dt=0.025):
    a, chi = c
    N = om.shape[0]
    S = np.zeros((N, 3, 5))
    S[:, :, 0] = eps * u0
    S[:, 0, 1:3] = a * np.array([np.cos(chi), np.sin(chi)])
    S[:, 1, 1:3] = d * np.array([1.0, 0.0])
    S[:, 2, 1:3] = d3 * np.array([np.cos(al3), np.sin(al3)])
    nst = int(round(tt / dt))
    for _ in range(nst):
        k1 = field(S, eps, om); k2 = field(S + dt / 2 * k1, eps, om)
        k3 = field(S + dt / 2 * k2, eps, om); k4 = field(S + dt * k3, eps, om)
        S = S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    beta, L = rows_and_L(S, eps)
    return S, beta

def history_density(eps):
    J = eps ** 3
    L0 = eps * u0
    dens = np.zeros((NW, 3)); dlog = np.zeros((NW, 3))
    for i in range(3):
        def P(om_val):
            tot = 0.0
            for z in (-1.0, 1.0):
                lik_o = np.exp(eps * o[i] * z / 2) / (2 * np.cosh(eps / 2))
                rG = np.hypot(eps * np.cos(om_val) + J * z, eps * np.sin(om_val))
                tot += a_pr * 0.5 * lik_o * np.exp(logI0(rG) - logI0(eps) - logI0(J))
                for j in range(3):
                    if j != i:
                        muj = np.exp(L0[j] * z / 2) / np.cosh(L0[j] / 2)
                        tot += p_pr * (muj / 2) * lik_o
            return tot
        vals = np.array([P(w) for w in wg]); Z = vals.mean()
        dens[:, i] = vals / Z
        h = 1e-6
        dlog[:, i] = np.array([(np.log(P(w + h)) - np.log(P(w - h))) / (2 * h) for w in wg])
    return dens, dlog

Tst = np.zeros((2, NPH, 5))
for zi, z in enumerate((-1.0, 1.0)):
    Tst[zi, :, 0] = z; Tst[zi, :, 1] = np.cos(ph); Tst[zi, :, 2] = np.sin(ph)
    Tst[zi, :, 3] = z * np.cos(ph); Tst[zi, :, 4] = z * np.sin(ph)

def nat(S):
    return np.concatenate([S[..., :1] / 2, S[..., 1:5]], axis=-1)

def covs(S):
    eta = nat(S)
    E = np.einsum("nik,zpk->nizp", eta, Tst)
    E -= E.max(axis=(-2, -1), keepdims=True)
    w = np.exp(E); w /= w.sum(axis=(-2, -1), keepdims=True)
    m = np.einsum("nizp,zpk->nik", w, Tst)
    M2 = np.einsum("nizp,zpk,zpl->nikl", w, Tst, Tst)
    return M2 - np.einsum("nik,nil->nikl", m, m), eta

RP = np.array([[0.0, -1.0], [1.0, 0.0]])

def geometry_at(c, eps, tt, dc=1e-5, dw=1e-5):
    dens, dlog = history_density(eps)
    idx = np.stack([np.arange(NW).repeat(NW * NW), np.tile(np.arange(NW).repeat(NW), NW), np.tile(np.arange(NW), NW * NW)], axis=1)
    wts = dens[idx[:, 0], 0] * dens[idx[:, 1], 1] * dens[idx[:, 2], 2]
    wts /= wts.sum()
    I_hist = sum(np.mean(dens[:, k] * dlog[:, k] ** 2) for k in range(3))
    S, beta = integrate(c, eps, OM, tt)
    Cov, eta = covs(S)
    dEta = []; dLb = []
    for k in range(2):
        e = np.zeros(2); e[k] = dc
        Sp, bp = integrate(c + e, eps, OM, tt); Sm, bm = integrate(c - e, eps, OM, tt)
        dEta.append((nat(Sp) - nat(Sm)) / (2 * dc)); dLb.append((np.log(bp) - np.log(bm)) / (2 * dc))
    sumdw = np.zeros_like(eta); sumdlb = np.zeros_like(beta)
    for k in range(3):
        omp = OM.copy(); omp[:, k] += dw; omm = OM.copy(); omm[:, k] -= dw
        Sp, bp = integrate(c, eps, omp, tt); Sm, bm = integrate(c, eps, omm, tt)
        sumdw += (nat(Sp) - nat(Sm)) / (2 * dw); sumdlb += (np.log(bp) - np.log(bm)) / (2 * dw)
    orb = np.zeros_like(eta)
    orb[..., 1:3] = np.einsum("nik,lk->nil", eta[..., 1:3], RP)
    orb[..., 3:5] = np.einsum("nik,lk->nil", eta[..., 3:5], RP)
    orb -= sumdw
    lab_orb = -sumdlb
    def pair(u, v):
        return np.einsum("nik,nikl,nil->n", u, Cov, v)
    def lpair(u, v):
        return np.einsum("nij,nij,nij->n", beta, u, v)
    K = I_hist + np.sum(wts * (pair(orb, orb) + lpair(lab_orb, lab_orb)))
    jv = np.array([np.sum(wts * (pair(dEta[k], orb) + lpair(dLb[k], lab_orb))) for k in range(2)])
    Bm = np.array([[np.sum(wts * (pair(dEta[k], dEta[l]) + lpair(dLb[k], dLb[l]))) for l in range(2)] for k in range(2)])
    Blab = np.array([[np.sum(wts * lpair(dLb[k], dLb[l])) for l in range(2)] for k in range(2)])
    return -jv / K, Bm - np.outer(jv, jv) / K, K, I_hist, Blab

def I_(x):
    return x * i1(x) / i0(x)

def V_(x):
    ph2 = 2 * np.pi * np.arange(512) / 512
    w = np.exp(x * np.cos(ph2)); w /= w.mean()
    return np.mean(w * np.cos(ph2) ** 2) - (i1(x) / i0(x)) ** 2

c0 = np.array([0.04, 0.3])
tt = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
eps_list = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.2]
s = np.exp(-tt) * c0[0]; Bt = I_(np.exp(-tt) * d) + I_(np.exp(-tt) * d3)
Ach = -I_(s) / (I_(s) + Bt)
Fcl = -Bt * np.exp(-tt) * (i1(s) / i0(s) + s * V_(s)) / (I_(s) + Bt) ** 2
print("t=%.1f  eps=0 closed form: A_chi=%.6f F=%.4f h_aa=%.4e h_cc=%.4e K=%.4e" % (tt, Ach, Fcl, np.exp(-2 * tt) * V_(s), I_(s) * Bt / (I_(s) + Bt), I_(s) + Bt))
for eps in eps_list:
    Acon, h, K, Ih, Blab = geometry_at(c0, eps, tt)
    dd = 1e-4
    Achi_p = geometry_at(c0 + np.array([dd, 0]), eps, tt)[0][1]; Achi_m = geometry_at(c0 - np.array([dd, 0]), eps, tt)[0][1]
    Aa_p = geometry_at(c0 + np.array([0, dd]), eps, tt)[0][0]; Aa_m = geometry_at(c0 - np.array([0, dd]), eps, tt)[0][0]
    F = (Achi_p - Achi_m) / (2 * dd) - (Aa_p - Aa_m) / (2 * dd)
    print("   eps=%.2f: A=(%.2e, %.6f) F=%.4f h_aa=%.4e h_cc=%.4e h_ac=%.1e K=%.4e | history orbit Fisher %.1e | label tensor max %.1e | %.1fs"
          % (eps, Acon[0], Acon[1], F, h[0, 0], h[1, 1], h[0, 1], K, Ih, np.abs(Blab).max(), time.time() - T0))
