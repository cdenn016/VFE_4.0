r"""Gaussian checks for appendices/interacting_memory.tex sections 6-7 and the companion
comp:interaction-gaussian (lens D07-IMEM1).

(S) sympy: scalar identities imem:scalar-gain, imem:scalar-message, imem:scalar-calibration,
    imem:scalar-receiver (exact symbolic residuals).
(G) numpy: the repeated affine Gaussian closure imem:gaussian-closure.  The protocol is built
    independently as a linear system in base noises: at each step the speaker's conditional
    mean E[Y_j | D_j, H_{1:t}] is computed by joint-Gaussian conditioning of the system built
    so far, a fresh sample is drawn from that conditional, and encoded with noise.  The exact
    posterior of D given the realized messages is compared with the reduced information
    filter imem:gaussian-reduced-recursion, and the off-diagonal block of the posterior
    precision of D is checked to be invariant (product-tilt structure).
(M) the mixture nonclosure example imem:mixture-nonclosure: logistic weight, the message
    density form, and the two tail limits of the receiver likelihood by quadrature.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

T0 = time.perf_counter()
OUT: dict = {}

# (S) scalar symbolic identities ----------------------------------------------------------------
L0, L1, N0, N1, Q0, Q1, R, S, f0, f1, b, w0, w1, nu0, nu1, d0, h = sp.symbols(
    "L0 L1 N0 N1 Q0 Q1 R S f0 f1 b omega0 omega1 nu0 nu1 d0 h", real=True)
g = L1 / (L1 + N1)
P = L1 * N1 / (L1 + N1)
a = b * f1 * g
Neff = a ** 2 * N1 + b ** 2 * (Q1 + f1 ** 2 * P) + R
varH = a ** 2 * L1 + Neff
varHdir = b ** 2 * (f1 ** 2 * L1 + Q1) + R
OUT["S_var_equality_residual"] = str(sp.simplify(varH - varHdir))
OUT["S_identity_g2T_plus_P"] = str(sp.simplify(g ** 2 * (L1 + N1) + P - L1))
# posterior-sample representation: Y1 | D1=d ~ N(omega1 + f1 g (d - nu1), Q1 + f1^2 P)
# check the cue posterior gain/variance by completing the square
x, d = sp.symbols("x d", real=True)
logpost = -(x - nu1) ** 2 / (2 * L1) - (d - x) ** 2 / (2 * N1)
coef2 = sp.Poly(sp.expand(logpost), x).coeff_monomial(x ** 2)
coef1 = sp.Poly(sp.expand(logpost), x).coeff_monomial(x)
post_var = sp.simplify(-1 / (2 * coef2))
post_mean = sp.simplify(coef1 * post_var)
OUT["S_cue_posterior_var_residual"] = str(sp.simplify(post_var - P))
OUT["S_cue_posterior_mean_residual"] = str(sp.simplify(post_mean - (nu1 + g * (d - nu1))))
# receiver conditioning, compare with full joint Gaussian of (X0, H | D0)
# joint of (X0, X1, D0, H): X ~ N(nu, [[L0,S],[S,L1]]); D0 = X0 + e0; H = b w1 + a (X1 - nu1) + noise(Neff)
Sig = sp.Matrix([[L0, S], [S, L1]])
# covariances
cov_X0_D0 = L0
var_D0 = L0 + N0
cov_X0_H = a * S
cov_D0_H = a * S
var_H = a ** 2 * L1 + Neff
J = sp.Matrix([[var_D0, cov_D0_H], [cov_D0_H, var_H]])
c = sp.Matrix([[cov_X0_D0, cov_X0_H]])
mu_obs = sp.Matrix([nu0, b * w1])
obs = sp.Matrix([d0, h])
m_full = nu0 + (c * J.inv() * (obs - mu_obs))[0]
v_full = L0 - (c * J.inv() * c.T)[0]
m0 = nu0 + L0 / (L0 + N0) * (d0 - nu0)
v0 = L0 * N0 / (L0 + N0)
m10 = nu1 + S / (L0 + N0) * (d0 - nu0)
v10 = L1 - S ** 2 / (L0 + N0)
c010 = S * N0 / (L0 + N0)
mH = b * w1 + a * (m10 - nu1)
VH = a ** 2 * v10 + Neff
m0p = m0 + a * c010 / VH * (h - mH)
v0p = v0 - a ** 2 * c010 ** 2 / VH
OUT["S_receiver_mean_residual"] = str(sp.simplify(m_full - m0p))
OUT["S_receiver_var_residual"] = str(sp.simplify(v_full - v0p))
OUT["S_cov_X0_H_vs_direct"] = [str(sp.simplify(cov_X0_H)), str(sp.simplify(b * f1 * S))]


# (G) repeated Gaussian protocol, brute force vs reduced filter -----------------------------------
def gaussian_protocol(seed: int, n_msgs: int) -> dict:
    rng = np.random.default_rng(seed)
    dx, dy, dh = 2, 2, 1  # per-agent cue and result dims, message dim
    # joint cue covariance (positive definite) for X = (X0, X1) in R^{2 dx}
    M = rng.normal(size=(2 * dx, 2 * dx))
    SigX = M @ M.T + 0.5 * np.eye(2 * dx)
    nuX = rng.normal(size=2 * dx)
    Fm = [rng.normal(size=(dy, dx)) for _ in range(2)]
    om = [rng.normal(size=dy) for _ in range(2)]
    Qm = [np.eye(dy) * (0.3 + rng.random()) for _ in range(2)]
    Nm = [np.eye(dx) * (0.2 + rng.random()) for _ in range(2)]
    Bm = [rng.normal(size=(dh, dy)) for _ in range(2)]
    Rm = [np.eye(dh) * (0.1 + rng.random()) for _ in range(2)]
    # base noises: X (via Cholesky), result noises, sensor noises, then per message xi_t, eta_t
    # represent each variable as const + G @ base, with base ~ N(0, I) growing over time
    Lx = np.linalg.cholesky(SigX)
    nb = 2 * dx + 2 * dy + 2 * dx
    def pad(G, n):
        out = np.zeros((G.shape[0], n)); out[:, :G.shape[1]] = G; return out
    Xc, XG = nuX.copy(), pad(Lx, nb)
    Yc, YG, Dc, DG = [], [], [], []
    off = 2 * dx
    for i in range(2):
        sl = slice(i * dx, (i + 1) * dx)
        Yc.append(om[i] + Fm[i] @ (Xc[sl] - nuX[sl]))
        G = Fm[i] @ XG[sl]
        G[:, off + i * dy: off + (i + 1) * dy] += np.linalg.cholesky(Qm[i])
        YG.append(G)
    off2 = 2 * dx + 2 * dy
    for i in range(2):
        sl = slice(i * dx, (i + 1) * dx)
        Dc.append(Xc[sl].copy())
        G = XG[sl].copy()
        G[:, off2 + i * dx: off2 + (i + 1) * dx] += np.linalg.cholesky(Nm[i])
        DG.append(G)
    Hc, HG = [], []
    sched = [1 if t % 2 == 0 else 0 for t in range(n_msgs)]
    hs = []
    def cond_mean_affine(tc, tG, oc, oG):
        """E[target | obs] = tc + K (obs - oc) and Cov; all in base coordinates."""
        n = max(tG.shape[1], oG.shape[1])
        tG2, oG2 = pad(tG, n), pad(oG, n)
        Soo = oG2 @ oG2.T
        Sto = tG2 @ oG2.T
        K = Sto @ np.linalg.inv(Soo)
        C = tG2 @ tG2.T - K @ Sto.T
        return K, C
    realized_h = []
    for t, j in enumerate(sched):
        # observation vector for the speaker: (D_j, H_1..H_t)
        oc = np.concatenate([Dc[j]] + Hc) if Hc else Dc[j].copy()
        nmax = max([YG[j].shape[1], DG[j].shape[1]] + [g.shape[1] for g in HG])
        oG = np.vstack([pad(DG[j], nmax)] + [pad(g, nmax) for g in HG])
        K, C = cond_mean_affine(Yc[j], YG[j], oc, oG)
        # fresh sample: Ytilde = Yc + K (obs - oc) + C^{1/2} xi ;  H = B Ytilde + R^{1/2} eta
        nnew = nmax + dy + dh
        obs_G = pad(oG, nnew)
        samp_c = Yc[j] - K @ oc  # constant part (obs enter linearly)
        samp_G = K @ obs_G
        samp_G[:, nmax: nmax + dy] += np.linalg.cholesky(C + 1e-14 * np.eye(dy))
        hc = Bm[j] @ (samp_c + K @ oc) - Bm[j] @ (K @ oc)  # constant offset of B Ytilde
        hc = Bm[j] @ samp_c + Bm[j] @ (K @ oc)
        hG = Bm[j] @ samp_G
        hG[:, nmax + dy: nmax + dy + dh] += np.linalg.cholesky(Rm[j])
        Hc.append(hc)
        HG.append(hG)
        realized_h.append(rng.normal(size=dh))  # arbitrary realized message values
    # exact posterior of D = (D0, D1) given realized H via joint-Gaussian conditioning
    nmax = max(g.shape[1] for g in HG)
    Dall_c = np.concatenate(Dc); Dall_G = np.vstack([pad(g, nmax) for g in DG])
    Hall_c = np.concatenate(Hc); Hall_G = np.vstack([pad(g, nmax) for g in HG])
    Shh = Hall_G @ Hall_G.T; Sdh = Dall_G @ Hall_G.T; Sdd = Dall_G @ Dall_G.T
    Kd = Sdh @ np.linalg.inv(Shh)
    hvec = np.concatenate(realized_h)
    mD = Dall_c + Kd @ (hvec - Hall_c)
    CD = Sdd - Kd @ Sdh.T
    # NOTE: the speaker means were built with obs entering linearly through K applied to the
    # realized-variable representation, so H_t is an affine function of (D_j, H_{1:t-1}) plus
    # fresh noise, exactly as in the protocol.
    # reduced filter: lambda_0 = N(m, S); W fixed; message H = b_t + A_t D + eta_t
    m = Dall_c.copy(); Sd = Sdd0 = None
    # prior of D and joint with Y
    Dp_G = np.vstack([pad(g, nb) for g in DG]); Dp_c = np.concatenate(Dc)
    S0 = Dp_G @ Dp_G.T
    Sd = S0.copy(); m = Dp_c.copy()
    YDcov = [pad(YG[j], nb) @ Dp_G.T for j in range(2)]  # Cov(Y_j, D)
    YYcov = [pad(YG[j], nb) @ pad(YG[j], nb).T for j in range(2)]
    # W(d, dy_j): E[Y_j | D=d] = Yc_j + YDcov_j S0^{-1}(d - Dp_c), Cov = YY - YD S0^{-1} DY
    S0i = np.linalg.inv(S0)
    WK = [YDcov[j] @ S0i for j in range(2)]
    WC = [YYcov[j] - YDcov[j] @ S0i @ YDcov[j].T for j in range(2)]
    precs = [np.linalg.inv(Sd)]
    for t, j in enumerate(sched):
        idx = np.arange(j * dx, (j + 1) * dx)
        oth = np.array([k for k in range(2 * dx) if k not in idx])
        # lambda_t conditional of D_{other} given D_j
        Sjj = Sd[np.ix_(idx, idx)]; Soj = Sd[np.ix_(oth, idx)]
        Kc = Soj @ np.linalg.inv(Sjj)
        Coo = Sd[np.ix_(oth, oth)] - Kc @ Soj.T
        # Y_j | D_j = dj under W lambda_t: mean = Yc + WK (E[D|dj] - Dp_c), affine in dj
        # E[D | D_j=dj] = P_sel dj + (const): build matrix E_D = Mj dj + cj
        Mj = np.zeros((2 * dx, dx)); Mj[idx, :] = np.eye(dx); Mj[oth, :] = Kc
        cj = np.zeros(2 * dx); cj[oth] = m[oth] - Kc @ m[idx]
        beta = WK[j] @ Mj
        alpha = Yc[j] + WK[j] @ (cj - Dp_c)
        CovD_given = np.zeros((2 * dx, 2 * dx)); CovD_given[np.ix_(oth, oth)] = Coo
        Sigma_y = WC[j] + WK[j] @ CovD_given @ WK[j].T
        At = np.zeros((dh, 2 * dx)); At[:, idx] = Bm[j] @ beta
        bt = Bm[j] @ alpha
        Rt = Bm[j] @ Sigma_y @ Bm[j].T + Rm[j]
        Sn = np.linalg.inv(np.linalg.inv(Sd) + At.T @ np.linalg.inv(Rt) @ At)
        m = Sn @ (np.linalg.inv(Sd) @ m + At.T @ np.linalg.inv(Rt) @ (realized_h[t] - bt))
        Sd = Sn
        precs.append(np.linalg.inv(Sd))
    off_block = [p[np.ix_(range(dx), range(dx, 2 * dx))] for p in precs]
    return {
        "max_abs_mean_diff": float(np.max(np.abs(m - mD))),
        "max_abs_cov_diff": float(np.max(np.abs(Sd - CD))),
        "max_change_offdiag_precision": float(max(np.max(np.abs(o - off_block[0])) for o in off_block)),
        "max_change_diag_precision": float(max(np.max(np.abs(p - precs[0])) for p in precs)),
    }


OUT["G_gaussian_protocol"] = [gaussian_protocol(s, 5) for s in (1, 2, 3)]

# (M) mixture nonclosure example -----------------------------------------------------------------
alpha, Lm0, Lm1, Sx, N0v, N1v, mval, Qv, Rv, hval = 1.3, 1.0, 1.5, 0.8, 0.4, 0.6, 1.1, 0.5, 0.3, 0.7
T = Lm1 + N1v
def npdf(z, mu, var):
    return np.exp(-(z - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
dgrid = np.linspace(-12, 12, 4001)
wplus_direct = npdf(dgrid, alpha, T) / (npdf(dgrid, alpha, T) + npdf(dgrid, -alpha, T))
wplus_formula = 1 / (1 + np.exp(-2 * alpha * dgrid / T))
OUT["M_logistic_weight_max_err"] = float(np.max(np.abs(wplus_direct - wplus_formula)))
cp, cm = npdf(hval, mval, Qv + Rv), npdf(hval, -mval, Qv + Rv)
def ell0(xv):
    tot = 0.0
    for bsign in (-1, 1):
        mean = bsign * alpha + (Sx / Lm0) * xv
        var = Lm1 - Sx ** 2 / Lm0 + N1v
        dd = np.linspace(mean - 12 * np.sqrt(var), mean + 12 * np.sqrt(var), 6001)
        dens = npdf(dd, mean, var)
        k = cm + (cp - cm) / (1 + np.exp(-2 * alpha * dd / T))
        tot += 0.5 * np.trapezoid(dens * k, dd)
    return tot
OUT["M_c_plus_c_minus"] = [float(cp), float(cm)]
OUT["M_ell0_at_x"] = {str(xv): ell0(xv) for xv in (-40.0, -10.0, 0.0, 10.0, 40.0)}
OUT["wall_seconds"] = round(time.perf_counter() - T0, 3)
OUT["scope"] = "Symbolic identities are exact; Gaussian and quadrature checks are numerical instances only."
Path(__file__).with_name("gaussian_check_output.json").write_text(json.dumps(OUT, indent=2) + "\n", encoding="utf-8")
print(json.dumps(OUT, indent=2))
