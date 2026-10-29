r"""XC7-IMPL s7: tangent-model conformance checks for chapters 18 and 19.

ch18: it2:ring spectrum; it2:echo variances; it2:split (global vs per-agent shares);
      it2:coupled-derivatives against finite differences of it2:coupled-costs on a random
      3-agent graph with K=2 and rank-1/rank-2 codes.
ch19: it3:affine-action / it3:marginal (Gaussian elimination of contents), the weak-anchor
      limit it3:projection and it3:quotient, the rank K(|E|-|V|+1), it3:dag-example,
      it3:rank-one-eq, it3:cap, it3:rigid (P_A', m_A' and the O(1/tau) minimizer limit).
"""
import numpy as np
from scipy.linalg import null_space, block_diag
rng = np.random.default_rng(5)

# ---------------- it2:ring
for ell, qt, p, s, c in [(5, 2.0, 1.0, 3.0, 0.7), (7, 0.5, 2.0, 0.2, 1.5)]:
    A_i = np.array([[p + qt, -qt * c], [-qt * c, s + qt * c * c]])
    Cp = np.array([[-qt, 0], [qt * c, 0]])
    n = 2 * ell; Amat = np.zeros((n, n)); Cmat = np.zeros((n, n))
    for i in range(ell):
        Amat[2 * i:2 * i + 2, 2 * i:2 * i + 2] = A_i
        j = (i + 1) % ell
        Cmat[2 * i:2 * i + 2, 2 * j:2 * j + 2] = -Cp          # H = A - C with coupling C' in A x_i + C' x_{i+1}
    flow = -np.linalg.solve(Amat, Amat - Cmat)
    ev = np.linalg.eigvals(flow)
    w = qt * s / (p * s + p * qt * c * c + qt * s)
    pred = np.concatenate([-np.ones(ell), -(1 - np.exp(2j * np.pi * np.arange(ell) / ell) * w)])
    err = max(min(abs(e - pp) for pp in pred) for e in ev)
    print(f"it2:ring ell={ell}: max distance of spectrum to claimed set {err:.1e}; w={w:.6f}")

# ---------------- it2:echo
for ell, sig2, lam in [(2, 1.0, 50.0), (3, 2.0, 1e3), (4, 0.5, 7.0)]:
    Prec = np.eye(ell) / sig2 + lam * np.ones((ell, ell))
    exact = np.linalg.inv(Prec)[0, 0]
    fact = 1 / (1 / sig2 + lam)
    vv = sig2
    for _ in range(20000):
        vv = 1 / (1 / sig2 + 1 / (1 / lam + (ell - 1) * vv))
    cav = 1 / (1 / sig2 + 1 / (1 / lam + (ell - 1) * sig2))
    print(f"it2:echo ell={ell}: exact {exact:.6f} (claimed {sig2*(1+(ell-1)*lam*sig2)/(1+ell*lam*sig2):.6f}), "
          f"factorized {fact:.6f}, per-agent fixed point {vv:.6f}, cavity {cav:.6f}")

# ---------------- it2:split
K, nE = 3, 2
T = rng.normal(size=(nE, K)); R = np.eye(nE) * 0.4; c_i = rng.normal(size=K)
Si, Sj = np.eye(K) * 0.8, np.eye(K) * 1.3; Ve = np.eye(K * K) * 0.25
kap = np.kron(np.eye(K), c_i[None, :])           # kappa_i vec_row(omega) = omega c_i
Crho = kap @ Ve @ kap.T
# joint Gaussian conditioning on o = T(y_i - y_j - kappa omega) + eps
Sig = block_diag(Si, Sj, Ve)
Lmap = np.hstack([T, -T, -T @ kap])
Sobs = Lmap @ Sig @ Lmap.T + R
gain = Sig @ Lmap.T @ np.linalg.inv(Sobs)
rres = rng.normal(size=nE)
upd = gain @ rres
S_claim = T @ (Si + Sj + Crho) @ T.T + R
print("it2:split(i): |dy_i - claimed|", np.abs(upd[:K] - Si @ T.T @ np.linalg.solve(S_claim, rres)).max(),
      " |kappa domega - claimed|", np.abs(kap @ upd[2 * K:] + Crho @ T.T @ np.linalg.solve(S_claim, rres)).max())
Sprime = T @ (Si + Crho) @ T.T + R + T @ Sj @ T.T
print("it2:split(ii): S' - S =", np.abs(Sprime - S_claim).max(), "(the per-agent and global innovation covariances coincide)")

# ---------------- it2:coupled-derivatives by finite differences
K = 2; agents = 3; edges = [(0, 1), (1, 2), (2, 0), (0, 2)]          # (receiver i, sender j)
codes = [rng.normal(size=(1, K)), rng.normal(size=(2, K)), rng.normal(size=(1, K)), rng.normal(size=(2, K))]
Rn = [np.eye(Tc.shape[0]) * 0.5 for Tc in codes]
cops = [rng.normal(size=K) for _ in range(agents)]
Pref = [np.eye(K) * 1.2, np.eye(K) * 0.7, np.eye(K)]; mref = [rng.normal(size=K) for _ in range(agents)]
dw = K * K; Aw = np.eye(len(edges) * dw) / 1.5 + 0.3 * np.kron(np.ones((len(edges), len(edges))), np.eye(dw)) / len(edges)
bw = rng.normal(size=len(edges) * dw) * 0.1; a0 = np.linalg.solve(Aw, bw)
beta = np.array([0.7, 0.4, 0.9, 0.3])
def spd(d): M = rng.normal(size=(d, d)); return M @ M.T / d + 0.5 * np.eye(d)
state = dict(mu=[rng.normal(size=K) for _ in range(agents)], Sig=[spd(K) for _ in range(agents)],
             a=[rng.normal(size=dw) * 0.3 for _ in edges], V=[spd(dw) * 0.3 for _ in edges])
def kappa(i): return np.kron(np.eye(K), cops[i][None, :])
def energy(st):
    E = 0.0
    for i in range(agents):
        Pm, mm, mu, Sg = Pref[i], mref[i], st['mu'][i], st['Sig'][i]
        E += 0.5 * (np.trace(Pm @ Sg) + (mu - mm) @ Pm @ (mu - mm) - K - np.linalg.slogdet(Pm @ Sg)[1])
    a = np.concatenate(st['a']); V = block_diag(*st['V'])
    E += 0.5 * ((a - a0) @ Aw @ (a - a0) + np.trace(Aw @ V) - np.linalg.slogdet(Aw @ V)[1] - len(a))
    for e, (i, j) in enumerate(edges):
        Tm, Ke = codes[e], codes[e] @ kappa(i)
        Sie, Sje = Tm @ st['Sig'][i] @ Tm.T + Rn[e], Tm @ st['Sig'][j] @ Tm.T + Rn[e]
        d = Tm @ (st['mu'][i] - st['mu'][j]) - Ke @ st['a'][e]
        Q = Sie + np.outer(d, d) + Ke @ st['V'][e] @ Ke.T
        E += beta[e] * 0.5 * (np.trace(np.linalg.solve(Sje, Q)) - len(d) + np.linalg.slogdet(Sje)[1] - np.linalg.slogdet(Sie)[1])
    return E
def closed_derivs(st):
    g_mu = [Pref[i] @ (st['mu'][i] - mref[i]) for i in range(agents)]
    G_S = [0.5 * (Pref[i] - np.linalg.inv(st['Sig'][i])) for i in range(agents)]
    a = np.concatenate(st['a']); ga_all = Aw @ (a - a0)
    g_a = [ga_all[e * dw:(e + 1) * dw].copy() for e in range(len(edges))]
    G_V = [0.5 * (Aw[e * dw:(e + 1) * dw, e * dw:(e + 1) * dw] - np.linalg.inv(st['V'][e])) for e in range(len(edges))]
    for e, (i, j) in enumerate(edges):
        Tm, Ke = codes[e], codes[e] @ kappa(i)
        Sie, Sje = Tm @ st['Sig'][i] @ Tm.T + Rn[e], Tm @ st['Sig'][j] @ Tm.T + Rn[e]
        Je = np.linalg.inv(Sje)
        d = Tm @ (st['mu'][i] - st['mu'][j]) - Ke @ st['a'][e]
        Q = Sie + np.outer(d, d) + Ke @ st['V'][e] @ Ke.T
        g_mu[i] = g_mu[i] + beta[e] * Tm.T @ Je @ d
        g_mu[j] = g_mu[j] - beta[e] * Tm.T @ Je @ d
        g_a[e] = g_a[e] - beta[e] * Ke.T @ Je @ d
        G_S[i] = G_S[i] + 0.5 * beta[e] * Tm.T @ (Je - np.linalg.inv(Sie)) @ Tm
        G_S[j] = G_S[j] + 0.5 * beta[e] * Tm.T @ (Je - Je @ Q @ Je) @ Tm
        G_V[e] = G_V[e] + 0.5 * beta[e] * Ke.T @ Je @ Ke
    return g_mu, G_S, g_a, G_V
g_mu, G_S, g_a, G_V = closed_derivs(state)
h = 1e-6; errs = []
import copy
for key, idx_list in (('mu', range(agents)), ('a', range(len(edges)))):
    for k in idx_list:
        for comp in range(len(state[key][k])):
            sp_, sm_ = copy.deepcopy(state), copy.deepcopy(state)
            sp_[key][k][comp] += h; sm_[key][k][comp] -= h
            fd = (energy(sp_) - energy(sm_)) / (2 * h)
            errs.append(abs(fd - (g_mu[k][comp] if key == 'mu' else g_a[k][comp])))
for key, idx_list, Gs in (('Sig', range(agents), G_S), ('V', range(len(edges)), G_V)):
    for k in idx_list:
        n = state[key][k].shape[0]
        Dir = rng.normal(size=(n, n)); Dir = Dir + Dir.T
        sp_, sm_ = copy.deepcopy(state), copy.deepcopy(state)
        sp_[key][k] = sp_[key][k] + h * Dir; sm_[key][k] = sm_[key][k] - h * Dir
        fd = (energy(sp_) - energy(sm_)) / (2 * h)
        errs.append(abs(fd - np.sum(Gs[k] * Dir)))
print(f"it2:coupled-derivatives vs central differences: max error {max(errs):.2e} over {len(errs)} directional derivatives")

# ---------------- chapter 19
def build(Vn, edges, K, codes, Rs, cvec):
    """delta: Y -> H and calV: X -> H, row-major vec(omega_e)."""
    nH = sum(T_.shape[0] for T_ in codes); delta = np.zeros((nH, Vn * K)); calV = np.zeros((nH, len(edges) * K * K)); r0 = 0
    for e, (i, j) in enumerate(edges):
        T_ = codes[e]; ne = T_.shape[0]
        delta[r0:r0 + ne, i * K:(i + 1) * K] += T_; delta[r0:r0 + ne, j * K:(j + 1) * K] -= T_
        calV[r0:r0 + ne, e * K * K:(e + 1) * K * K] = T_ @ np.kron(np.eye(K), cvec[i][None, :])
        r0 += ne
    return delta, calV, block_diag(*Rs)
Vn, K = 3, 2
edges = [(1, 0), (2, 1), (0, 2), (0, 1)]           # (receiver, sender): 0->1, 1->2, 2->0, 1->0 strongly connected
codes = [rng.normal(size=(K, K)) + 2 * np.eye(K) for _ in edges]          # injective (square, invertible)
Rs = [np.eye(K) * 0.3 for _ in edges]; cc = rng.normal(size=K); cvec = [cc] * Vn
delta, calV, Rm = build(Vn, edges, K, codes, Rs, cvec)
sig2 = 0.8; Pm = block_diag(*[np.eye(K) * pk for pk in (1.0, 2.0, 0.5)]); mm = rng.normal(size=Vn * K); o = rng.normal(size=delta.shape[0])
Ri = np.linalg.inv(Rm)
# joint energy E(y,w) = 1/2 (y-m)^T P (y-m) + |w|^2/(2 sig2) + 1/2 ||o - delta y + calV w||^2_{R^-1}
Jyy = Pm + delta.T @ Ri @ delta; Jyw = -delta.T @ Ri @ calV; Jww = np.eye(calV.shape[1]) / sig2 + calV.T @ Ri @ calV
hy = Pm @ mm + delta.T @ Ri @ o; hw = -calV.T @ Ri @ o
Lam_marg = Jww - Jyw.T @ np.linalg.solve(Jyy, Jyw)
CP = Rm + delta @ np.linalg.solve(Pm, delta.T)
print("it3:marginal |Lambda_w - (I/sig2 + V^T C_P^-1 V)| =", np.abs(Lam_marg - (np.eye(calV.shape[1]) / sig2 + calV.T @ np.linalg.solve(CP, calV))).max())
lin_marg = hw - Jyw.T @ np.linalg.solve(Jyy, hy)
lin_claim = -calV.T @ np.linalg.solve(CP, o - delta @ mm)          # from 1/2||o + V w - delta m||^2_{C_P^-1}
print("it3:affine-action linear term residual:", np.abs(lin_marg - lin_claim).max())
PiH = Ri - Ri @ delta @ np.linalg.pinv(delta.T @ Ri @ delta) @ delta.T @ Ri
errs_eps = []
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    CPe = Rm + delta @ np.linalg.solve(eps * Pm, delta.T)
    errs_eps.append(np.abs(np.linalg.inv(CPe) - PiH).max())
print("it3:projection |C_P^-1(eps) - Pi_H| for eps=1e-2..1e-5:", ["%.2e" % e for e in errs_eps], "(O(eps))")
wv = rng.normal(size=calV.shape[1]); u = calV @ wv
amin = np.linalg.lstsq(np.linalg.cholesky(Ri).T @ delta, np.linalg.cholesky(Ri).T @ u, rcond=None)[0]
print("it3:quotient:", u @ PiH @ u, "vs min_a:", (u - delta @ amin) @ Ri @ (u - delta @ amin))
print("rank of V^T Pi_H V:", np.linalg.matrix_rank(calV.T @ PiH @ calV, tol=1e-9), " claimed K(|E|-|V|+1) =", K * (len(edges) - Vn + 1))
d1 = build(3, [(1, 0), (2, 1), (2, 0)], 1, [np.eye(1)] * 3, [np.eye(1)] * 3, [np.ones(1)] * 3)[0]
uu = np.array([0., 0., 1.]); aa = np.linalg.lstsq(d1, uu, rcond=None)[0]
print("it3:dag-example min_a ||(0,0,1) - delta a||^2 =", np.sum((uu - d1 @ aa) ** 2))
# it3:rank-one-eq at the joint stationary point
Jfull = np.block([[Jyy, Jyw], [Jyw.T, Jww]]); sol = np.linalg.solve(Jfull, np.concatenate([hy, hw]))
yst, wst = sol[:Vn * K].reshape(Vn, K), sol[Vn * K:].reshape(len(edges), K, K)
r0 = 0; errs = []
for e, (i, j) in enumerate(edges):
    T_ = codes[e]; ne = T_.shape[0]; ci = cvec[i]
    rhs = sig2 * T_.T @ np.linalg.solve(Rs[e] + sig2 * (ci @ ci) * T_ @ T_.T, T_ @ (yst[i] - yst[j]) - o[r0:r0 + ne])
    errs.append(np.abs(wst[e] - np.outer(rhs, ci)).max()); r0 += ne
print("it3:rank-one-eq max residual:", max(errs), " ranks:", [np.linalg.matrix_rank(wst[e], tol=1e-10) for e in range(len(edges))])
# it3:rigid: two clusters {0,1}->A, {2,3}->B; internal edge 1->0 (identity dictionary), bridges 2->0, 1->3 (free)
K = 2; ch = 4; cl = {0: 0, 1: 0, 2: 1, 3: 1}
Pc = [np.eye(K) * v for v in (1.0, 0.6, 1.4, 0.9)]; mc = [rng.normal(size=K) for _ in range(ch)]
PA = [np.eye(K) * 0.5, np.eye(K) * 0.8]; mA = [rng.normal(size=K) for _ in range(2)]
bridges = [(0, 2), (3, 1)]; bcodes = [rng.normal(size=(1, K)), rng.normal(size=(2, K))]; bo = [rng.normal(size=1), rng.normal(size=2)]
internal = [(0, 1)]; icode = [rng.normal(size=(2, K))]; io = [rng.normal(size=2)]
cop = [rng.normal(size=K) for _ in range(ch)]; s2b = 0.7; Rb = 0.5
def solve_global(tau):
    n = (ch + 2) * K + len(bridges) * K * K; Jg = np.zeros((n, n)); hg = np.zeros(n)
    yi = lambda i: slice(i * K, (i + 1) * K); yA = lambda a: slice((ch + a) * K, (ch + a + 1) * K)
    wi = lambda e: slice((ch + 2) * K + e * K * K, (ch + 2) * K + (e + 1) * K * K)
    for i in range(ch): Jg[yi(i), yi(i)] += Pc[i]; hg[yi(i)] += Pc[i] @ mc[i]
    for a in range(2): Jg[yA(a), yA(a)] += PA[a]; hg[yA(a)] += PA[a] @ mA[a]
    for i in range(ch):
        a = cl[i]; Jg[yi(i), yi(i)] += tau * np.eye(K); Jg[yA(a), yA(a)] += tau * np.eye(K)
        Jg[yi(i), yA(a)] -= tau * np.eye(K); Jg[yA(a), yi(i)] -= tau * np.eye(K)
    def add_record(i, j, T_, ob, wslice=None, ci=None):
        L = np.zeros((T_.shape[0], n)); L[:, yi(i)] += T_; L[:, yi(j)] -= T_
        if wslice is not None: L[:, wslice] -= T_ @ np.kron(np.eye(K), ci[None, :])
        Jg[:, :] += L.T @ L / Rb; hg[:] += L.T @ ob / Rb
    for (i, j), T_, ob in zip(internal, icode, io): add_record(i, j, T_, ob)
    for e, ((i, j), T_, ob) in enumerate(zip(bridges, bcodes, bo)):
        Jg[wi(e), wi(e)] += np.eye(K * K) / s2b; add_record(i, j, T_, ob, wi(e), cop[i])
    return np.linalg.solve(Jg, hg)
def solve_rigid():
    PAp = [PA[a] + sum(Pc[i] for i in range(ch) if cl[i] == a) for a in range(2)]
    mAp = [np.linalg.solve(PAp[a], PA[a] @ mA[a] + sum(Pc[i] @ mc[i] for i in range(ch) if cl[i] == a)) for a in range(2)]
    n = 2 * K + len(bridges) * K * K; Jr = np.zeros((n, n)); hr = np.zeros(n)
    for a in range(2): Jr[a * K:(a + 1) * K, a * K:(a + 1) * K] += PAp[a]; hr[a * K:(a + 1) * K] += PAp[a] @ mAp[a]
    for e, ((i, j), T_, ob) in enumerate(zip(bridges, bcodes, bo)):
        ws = slice(2 * K + e * K * K, 2 * K + (e + 1) * K * K); Jr[ws, ws] += np.eye(K * K) / s2b
        L = np.zeros((T_.shape[0], n)); L[:, cl[i] * K:(cl[i] + 1) * K] += T_; L[:, cl[j] * K:(cl[j] + 1) * K] -= T_
        L[:, ws] -= T_ @ np.kron(np.eye(K), cop[i][None, :]); Jr += L.T @ L / Rb; hr += L.T @ ob / Rb
    return np.linalg.solve(Jr, hr)
xr = solve_rigid(); out = []
for tau in (1e2, 1e3, 1e4):
    xg = solve_global(tau)
    par = xg[ch * K:(ch + 2) * K]; wb = xg[(ch + 2) * K:]
    err = max(np.abs(par - xr[:2 * K]).max(), np.abs(wb - xr[2 * K:]).max()); out.append((tau, err, tau * err))
print("it3:rigid minimizer limit (tau, error, tau*error):", [(t, f"{e:.2e}", f"{te:.4f}") for t, e, te in out])
