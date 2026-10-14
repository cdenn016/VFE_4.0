r"""L5 checks for c10 (cut completed square, anchored and no-new-anchor presentations, balanced
presentation, law-level excess, PSD-weight holonomy witness, triangle holonomy score) and for
c11 cor:rg-contracted-perturbation-tower (width recursion, depth-uniform bounds, offsets)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm, null_space
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(11)


def spd(K):
    A = rng.normal(size=(K, K))
    return A @ A.T + 0.3 * np.eye(K)


def psd_min_eig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


def gmean(A, B):
    """Matrix geometric mean A # B = A^{1/2}(A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}."""
    Ah = np.real(sqrtm(A)); Aih = np.linalg.inv(Ah)
    return Ah @ np.real(sqrtm(Aih @ B @ Aih)) @ Ah


def check_cuts() -> dict:
    out = {}
    w = {k: 0.0 for k in ["energy", "Delta_formula", "identity", "aniso_admissible", "balance", "balance_alt",
                          "interval", "end_symmetry", "gauge_law", "orth_polar"]}
    for _ in range(300):
        K, k = 3, rng.integers(2, 5)
        Th = [rng.normal(size=(K, K)) + 2 * np.eye(K) for _ in range(k)]
        W = [spd(K) for _ in range(k)]
        LII = sum(W); LIJ = sum(Wi @ Ti for Wi, Ti in zip(W, Th)); LJJ = sum(Ti.T @ Wi @ Ti for Wi, Ti in zip(W, Th))
        zI, zJ = rng.normal(size=K), rng.normal(size=K)
        E = sum((zI - Ti @ zJ) @ Wi @ (zI - Ti @ zJ) for Wi, Ti in zip(W, Th))
        Tb = np.linalg.solve(LII, LIJ)
        Delta = LJJ - LIJ.T @ np.linalg.solve(LII, LIJ)
        E2 = (zI - Tb @ zJ) @ LII @ (zI - Tb @ zJ) + zJ @ Delta @ zJ
        w["energy"] = max(w["energy"], abs(E - E2))
        w["Delta_formula"] = max(w["Delta_formula"], np.abs(Delta - sum((Ti - Tb).T @ Wi @ (Ti - Tb) for Wi, Ti in zip(W, Th))).max())
        lhs = LII - LIJ @ np.linalg.solve(LJJ, LIJ.T)
        rhs = LII @ np.linalg.inv(LIJ).T @ Delta @ np.linalg.solve(LJJ, LIJ.T)
        w["identity"] = max(w["identity"], np.abs(lhs - rhs).max())
        Wlo = LIJ @ np.linalg.solve(LJJ, LIJ.T); Whi = LII
        # random point of the segment: self terms PSD and energy reproduced
        t = rng.uniform()
        Wt = (1 - t) * Wlo + t * Whi
        Th_t = np.linalg.solve(Wt, LIJ)
        SI, SJ = LII - Wt, LJJ - LIJ.T @ np.linalg.solve(Wt, LIJ)
        E3 = (zI - Th_t @ zJ) @ Wt @ (zI - Th_t @ zJ) + zI @ SI @ zI + zJ @ SJ @ zJ
        w["aniso_admissible"] = max(w["aniso_admissible"], abs(E - E3), -min(psd_min_eig(SI), psd_min_eig(SJ)))
        # balanced weight
        Wb = gmean(Wlo, Whi)
        w["balance"] = max(w["balance"], np.abs(Wb @ np.linalg.solve(Wlo, Wb) - Whi).max())
        Tbal = np.linalg.solve(Wb, LIJ)
        SIb, SJb = LII - Wb, LJJ - LIJ.T @ np.linalg.solve(Wb, LIJ)
        w["balance_alt"] = max(w["balance_alt"], np.abs(Tbal.T @ LII @ Tbal - LJJ).max(),
                               np.abs(np.linalg.inv(Tbal).T @ SJb @ np.linalg.inv(Tbal) - SIb).max())
        w["interval"] = max(w["interval"], -psd_min_eig(Wb - Wlo), -psd_min_eig(Whi - Wb))
        # end symmetry: Th^T W Th is the geometric mean of LIJ^T LII^{-1} LIJ and LJJ
        Wrev = Tbal.T @ Wb @ Tbal
        w["end_symmetry"] = max(w["end_symmetry"], np.abs(Wrev - gmean(LIJ.T @ np.linalg.solve(LII, LIJ), LJJ)).max())
        # gauge law of the balanced weight
        gI, gJ = rng.normal(size=(K, K)) + 2 * np.eye(K), rng.normal(size=(K, K)) + 2 * np.eye(K)
        giI = np.linalg.inv(gI); giJ = np.linalg.inv(gJ)
        W2 = [giI.T @ Wi @ giI for Wi in W]; Th2 = [gI @ Ti @ giJ for Ti in Th]
        LII2 = sum(W2); LIJ2 = sum(Wi @ Ti for Wi, Ti in zip(W2, Th2)); LJJ2 = sum(Ti.T @ Wi @ Ti for Wi, Ti in zip(W2, Th2))
        Wb2 = gmean(LIJ2 @ np.linalg.solve(LJJ2, LIJ2.T), LII2)
        w["gauge_law"] = max(w["gauge_law"], np.abs(Wb2 - giI.T @ Wb @ giI).max() / np.abs(giI.T @ Wb @ giI).max())
        # orthogonal twists, scalar weights
        Os = [np.linalg.qr(rng.normal(size=(K, K)))[0] for _ in range(k)]
        ws = rng.uniform(0.2, 2, size=k)
        LIJo = sum(wi * Oi for wi, Oi in zip(ws, Os)); LIIo = sum(ws) * np.eye(K)
        Wbo = gmean(LIJo @ LIJo.T / sum(ws), LIIo)
        U, s, Vt = np.linalg.svd(LIJo)
        w["orth_polar"] = max(w["orth_polar"], np.abs(Wbo - np.real(sqrtm(LIJo @ LIJo.T))).max(),
                              np.abs(np.linalg.solve(Wbo, LIJo) - U @ Vt).max())
    out.update(w)
    # law-level excess for twists I and 2I
    w1, w2 = 0.7, 1.9
    S = spd(2)
    W1, W2 = 0.5 * w1 * np.linalg.inv(S), 0.5 * w2 * np.linalg.inv(4 * S)
    LII = W1 + W2; LIJ = W1 + 2 * W2; LJJ = W1 + 4 * W2
    Delta = LJJ - LIJ.T @ np.linalg.solve(LII, LIJ)
    out["law-level Delta for I,2I vs w1w2/(2(4w1+w2)) Sigma^{-1}"] = np.abs(Delta - w1 * w2 / (2 * (4 * w1 + w2)) * np.linalg.inv(S)).max()
    # singular cut witness: SO(2) twists, Lambda_IJ = 2 t I, balanced link sign(t) I
    for t in (0.3, -0.3, 1e-4, -1e-4):
        c, s_ = t, np.sqrt(1 - t * t)
        R = np.array([[c, -s_], [s_, c]])
        LII = 2 * np.eye(2); LIJ = R + R.T; LJJ = 2 * np.eye(2)
        Wb = gmean(LIJ @ np.linalg.solve(LJJ, LIJ.T), LII)
        out[f"balanced link at t={t}"] = np.round(np.linalg.solve(Wb, LIJ), 6).tolist()
    # PSD-weight witness: kernel of connection Laplacian on the triangle
    K = 2
    edges = [((0, 1), np.diag([1.0, 0.0]), np.diag([1.0, 2.0])),
             ((1, 2), np.diag([0.0, 1.0]), np.diag([2.0, 1.0])),
             ((2, 0), np.eye(2), np.eye(2))]
    L = np.zeros((6, 6))
    for (i, j), We, Te in edges:  # term (z_i - Te z_j)^T We (z_i - Te z_j)
        Bm = np.zeros((2, 6)); Bm[:, 2 * i:2 * i + 2] = np.eye(2); Bm[:, 2 * j:2 * j + 2] = -Te
        L += Bm.T @ We @ Bm
    ns = null_space(L)
    out["PSD witness: dim ker L, loop product"] = (ns.shape[1], (np.diag([1, 2]) @ np.diag([2, 1])).tolist())
    # triangle holonomy score closed form
    sigma2 = 0.7; m = rng.normal(size=(3, 2)); wts = rng.dirichlet(np.ones(3))
    A = float(np.sum(wts * np.sum(m ** 2, 1)))
    f = lambda s: float(np.sum(wts * (sigma2 / s + np.sum(m ** 2, 1) / (2 * s) - 1 + np.log(s / sigma2))))
    r = minimize_scalar(f, bounds=(1e-3, 50), method="bounded", options={"xatol": 1e-12})
    out["triangle score numeric vs log(1+A/(2 sigma^2)), argmin vs sigma^2+A/2"] = (r.fun, np.log(1 + A / (2 * sigma2)), r.x, sigma2 + A / 2)
    return out


# ------------------------------------------------------------- contracted perturbation tower
def check_tower(n_towers=300) -> dict:
    worst = {"recursion_violation": -np.inf, "offset_decomp": 0.0,
             "uniform_bound_violation": -np.inf}
    for _ in range(n_towers):
        L = 6
        sizes = [rng.integers(2, 6) for _ in range(L + 1)]
        m = rng.dirichlet(np.ones(sizes[0])) * rng.uniform(0.5, 2)
        psis = [rng.uniform(-1, 1, size=s) * rng.uniform(0, 1.5) for s in sizes]
        Cs = [rng.dirichlet(0.8 * np.ones(sizes[l + 1]), size=sizes[l]) for l in range(L)]
        mt = np.exp(-psis[0]) * m
        mm = m.copy()
        r = psis[0].copy()
        logratio_terms = [np.log((np.exp(-psis[0]) * m / m.sum()).sum())]
        dmax, smax = 0.0, 0.0
        omega0 = r.max() - r.min()
        widths = [omega0]
        for l in range(L):
            mbar = mm / mm.sum()
            joint = mbar[:, None] * Cs[l]
            mc = joint.sum(0)
            Cd = (joint / mc).T
            delta = max(0.5 * np.abs(Cd[a] - Cd[b]).sum() for a in range(len(mc)) for b in range(len(mc)))
            if np.all(Cd > 0):
                ratios = Cd[:, None, :] / Cd[None, :, :]  # (z, z', y)
                Dl = float(np.log((ratios.max(-1) / ratios.min(-1)).max()))  # per pair (z, z')
            else:
                Dl = np.inf
            # exact propagation
            mt_next_raw = mt @ Cs[l]
            nhat = mt_next_raw / mt_next_raw.sum()
            mm_next = mm @ Cs[l]
            A = -np.log(Cd @ np.exp(-r))
            r_next = A + psis[l + 1]
            mt = np.exp(-psis[l + 1]) * mt_next_raw
            logratio_terms.append(np.log((np.exp(-psis[l + 1]) * nhat).sum()))
            # recursion check
            w = widths[-1]; ws = psis[l + 1].max() - psis[l + 1].min()
            bound = min(np.log1p(delta * np.expm1(w)), np.tanh(Dl / 4) * w if np.isfinite(Dl) else w) + ws
            wn = r_next.max() - r_next.min()
            worst["recursion_violation"] = max(worst["recursion_violation"], wn - bound)
            # consistency of residual with direct densities
            worst["offset_decomp"] = max(worst["offset_decomp"], np.abs((np.exp(-r_next) * mm_next) - mt).max())
            widths.append(wn); mm = mm_next; r = r_next
            dmax = max(dmax, delta); smax = max(smax, ws)
        lhs = np.log(mt.sum() / mm.sum())
        worst["offset_decomp"] = max(worst["offset_decomp"], abs(lhs - sum(logratio_terms)))
        if smax < np.log(1 / dmax):
            k = lambda x: x - np.log1p(dmax * np.expm1(x))
            lo, hi = 0.0, 100.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if k(mid) < smax else (lo, mid)
            B = max(omega0, hi)
            worst["uniform_bound_violation"] = max(worst["uniform_bound_violation"], max(widths) - B)
    return worst


if __name__ == "__main__":
    print("== c10 cuts and holonomy witnesses")
    for k, v in check_cuts().items():
        print(f"  {k}: {v}")
    print("== c11 contracted perturbation tower (random finite towers, depth 6)")
    for k, v in check_tower().items():
        print(f"  {k}: {v:.3e}")
