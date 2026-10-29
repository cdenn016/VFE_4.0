r"""D25-CRIT check 8: finite-difference check of it2:coupled-derivatives, it2:coupled-velocities,
it2:coupled-row-flow, it2:coupled-dissipation and the envelope gradient of E_T, on a random
noncommuting example: K = 2, three agents, edges 1->0, 2->0, 0->1, 2->1, 1->2 (a two-cycle 0<->1),
codes of rank 1 or 2, kappa_i(omega) = omega c_i on gl(2) (d_omega = 4, column-major vec),
a correlated joint dictionary prior N(a_0, A_omega^{-1}), product recognition.  Float64.
Also: Monte Carlo check of the expected conditional KL formula C_e = 1/2[Tr(J Q) - n + logdet S_j - logdet S_i].
Output: d25_random_flow_check.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
t0 = time.perf_counter()
rng = np.random.default_rng(20261029)
K, dw = 2, 4
edges = [(1, 0), (2, 0), (0, 1), (2, 1), (1, 2)]  # (source j, receiver i)
E = len(edges)
nA = 3
inc = {i: [e for e, (j, r) in enumerate(edges) if r == i] for i in range(nA)}


def spd(n, scale=1.0):
    Q = rng.normal(size=(n, n))
    return scale * (Q @ Q.T / n + 0.5 * np.eye(n))


codes = [rng.normal(size=(1 if e % 2 else 2, K)) for e in range(E)]
cop = [rng.normal(size=K) for _ in range(nA)]
kap = [np.kron(cop[i].reshape(1, -1), np.eye(K)) for i in range(nA)]  # omega c = (c^T kron I) vec(omega)
Rn = [spd(codes[e].shape[0]) for e in range(E)]
P = [spd(K) for _ in range(nA)]
m = [rng.normal(size=K) for _ in range(nA)]
Aw = spd(E * dw, 1.5)
a0 = rng.normal(size=E * dw) * 0.3
bw = Aw @ a0
Ti = {i: 0.4 + rng.random() for i in range(nA)}
pri = {i: (lambda w: w / w.sum())(0.5 + rng.random(len(inc[i]))) for i in range(nA)}

# state
mu = [rng.normal(size=K) for _ in range(nA)]
Sig = [spd(K, 0.7) for _ in range(nA)]
a = rng.normal(size=E * dw) * 0.4
V = [spd(dw, 0.5) for _ in range(E)]
beta = {i: (lambda w: w / w.sum())(0.5 + rng.random(len(inc[i]))) for i in range(nA)}


def logdet(M):
    return np.linalg.slogdet(M)[1]


def edge_parts(e, mu, Sig, a, V):
    j, i = edges[e]
    Tb = codes[e]
    Ke = Tb @ kap[i]
    Si = Tb @ Sig[i] @ Tb.T + Rn[e]
    Sj = Tb @ Sig[j] @ Tb.T + Rn[e]
    Je = np.linalg.inv(Sj)
    ae = a[e * dw:(e + 1) * dw]
    d = Tb @ (mu[i] - mu[j]) - Ke @ ae
    Q = Si + np.outer(d, d) + Ke @ V[e] @ Ke.T
    C = 0.5 * (np.trace(Je @ Q) - Tb.shape[0] + logdet(Sj) - logdet(Si))
    return C, dict(Tb=Tb, Ke=Ke, Si=Si, Sj=Sj, Je=Je, d=d, Q=Q, i=i, j=j)


def costs(mu, Sig, a, V):
    return [edge_parts(e, mu, Sig, a, V)[0] for e in range(E)]


def F0(mu, Sig, a, V):
    val = 0.0
    for i in range(nA):
        r = mu[i] - m[i]
        val += 0.5 * (r @ P[i] @ r + np.trace(P[i] @ Sig[i]) - logdet(P[i] @ Sig[i]) - K)
    Vb = np.zeros((E * dw, E * dw))
    for e in range(E):
        Vb[e * dw:(e + 1) * dw, e * dw:(e + 1) * dw] = V[e]
    r = a - a0
    val += 0.5 * (r @ Aw @ r + np.trace(Aw @ Vb) - logdet(Aw) - sum(logdet(V[e]) for e in range(E)) - E * dw)
    return val


def Ecal(mu, Sig, a, V, beta):
    C = costs(mu, Sig, a, V)
    val = F0(mu, Sig, a, V)
    for i in range(nA):
        b = beta[i]
        val += sum(b[k] * C[e] for k, e in enumerate(inc[i])) + Ti[i] * np.sum(b * np.log(b / pri[i]))
    return val


def Ered(mu, Sig, a, V):
    C = costs(mu, Sig, a, V)
    val = F0(mu, Sig, a, V)
    for i in range(nA):
        Z = sum(pri[i][k] * np.exp(-C[e] / Ti[i]) for k, e in enumerate(inc[i]))
        val -= Ti[i] * np.log(Z)
    return val


def gibbs(mu, Sig, a, V):
    C = costs(mu, Sig, a, V)
    out = {}
    for i in range(nA):
        w = np.array([pri[i][k] * np.exp(-C[e] / Ti[i]) for k, e in enumerate(inc[i])])
        out[i] = w / w.sum()
    return out


def formulas(mu, Sig, a, V, beta):
    """Manuscript derivatives it2:coupled-derivatives and row forces h_e."""
    bet = np.zeros(E)
    for i in range(nA):
        for k, e in enumerate(inc[i]):
            bet[e] = beta[i][k]
    gmu = [P[i] @ (mu[i] - m[i]) for i in range(nA)]
    G2S = [P[i] - np.linalg.inv(Sig[i]) for i in range(nA)]
    ga = Aw @ a - bw
    G2V = [Aw[e * dw:(e + 1) * dw, e * dw:(e + 1) * dw] - np.linalg.inv(V[e]) for e in range(E)]
    for e in range(E):
        C, q = edge_parts(e, mu, Sig, a, V)
        Tb, Ke, Je, d, i, j = q["Tb"], q["Ke"], q["Je"], q["d"], q["i"], q["j"]
        gmu[i] = gmu[i] + bet[e] * Tb.T @ Je @ d
        gmu[j] = gmu[j] - bet[e] * Tb.T @ Je @ d
        ga[e * dw:(e + 1) * dw] += -bet[e] * Ke.T @ Je @ d
        G2S[i] = G2S[i] + bet[e] * Tb.T @ (Je - np.linalg.inv(q["Si"])) @ Tb
        G2S[j] = G2S[j] + bet[e] * Tb.T @ (Je - Je @ q["Q"] @ Je) @ Tb
        G2V[e] = G2V[e] + bet[e] * Ke.T @ Je @ Ke
    Cs = costs(mu, Sig, a, V)
    h = {i: np.array([Cs[e] + Ti[i] * (np.log(beta[i][k] / pri[i][k]) + 1) for k, e in enumerate(inc[i])]) for i in range(nA)}
    return gmu, [G / 2 for G in G2S], ga, [G / 2 for G in G2V], h


def sym(n):
    Q = rng.normal(size=(n, n))
    return (Q + Q.T) / 2


res = {}
eps = 1e-6
gmu, GS, ga, GV, h = formulas(mu, Sig, a, V, beta)
errs = []
for i in range(nA):
    for k in range(K):
        dm = [np.zeros(K) for _ in range(nA)]
        dm[i][k] = eps
        fd = (Ecal([mu[l] + dm[l] for l in range(nA)], Sig, a, V, beta)
              - Ecal([mu[l] - dm[l] for l in range(nA)], Sig, a, V, beta)) / (2 * eps)
        errs.append(abs(fd - gmu[i][k]))
res["max_err_g_mu"] = max(errs)
errs = []
for i in range(nA):
    U = sym(K)
    Sp = [Sig[l] + (eps * U if l == i else 0) for l in range(nA)]
    Sm = [Sig[l] - (eps * U if l == i else 0) for l in range(nA)]
    fd = (Ecal(mu, Sp, a, V, beta) - Ecal(mu, Sm, a, V, beta)) / (2 * eps)
    errs.append(abs(fd - np.trace(GS[i] @ U)))
res["max_err_G_Sigma"] = max(errs)
errs = []
for idx in range(E * dw):
    da = np.zeros(E * dw)
    da[idx] = eps
    fd = (Ecal(mu, Sig, a + da, V, beta) - Ecal(mu, Sig, a - da, V, beta)) / (2 * eps)
    errs.append(abs(fd - ga[idx]))
res["max_err_g_a"] = max(errs)
errs = []
for e in range(E):
    U = sym(dw)
    Vp = [V[f] + (eps * U if f == e else 0) for f in range(E)]
    Vm = [V[f] - (eps * U if f == e else 0) for f in range(E)]
    fd = (Ecal(mu, Sig, a, Vp, beta) - Ecal(mu, Sig, a, Vm, beta)) / (2 * eps)
    errs.append(abs(fd - np.trace(GV[e] @ U)))
res["max_err_G_V"] = max(errs)
# row forces: d E / d beta along zero-sum tangent equals sum h_e v_e
errs = []
for i in range(nA):
    if len(inc[i]) < 2:
        continue
    v = rng.normal(size=len(inc[i]))
    v -= v.mean()
    bp = {l: beta[l] + (eps * v if l == i else 0) for l in range(nA)}
    bm = {l: beta[l] - (eps * v if l == i else 0) for l in range(nA)}
    fd = (Ecal(mu, Sig, a, V, bp) - Ecal(mu, Sig, a, V, bm)) / (2 * eps)
    errs.append(abs(fd - h[i] @ v))
res["max_err_row_force"] = max(errs)

# velocities and dissipation
mud = [-Sig[i] @ gmu[i] for i in range(nA)]
Sd = [-2 * Sig[i] @ GS[i] @ Sig[i] for i in range(nA)]
ad = np.concatenate([-V[e] @ ga[e * dw:(e + 1) * dw] for e in range(E)])
Vd = [-2 * V[e] @ GV[e] @ V[e] for e in range(E)]
bd = {i: -beta[i] * (h[i] - beta[i] @ h[i]) for i in range(nA)}
diss = (sum(gmu[i] @ Sig[i] @ gmu[i] + 2 * np.trace(Sig[i] @ GS[i] @ Sig[i] @ GS[i]) for i in range(nA))
        + sum(ga[e * dw:(e + 1) * dw] @ V[e] @ ga[e * dw:(e + 1) * dw] + 2 * np.trace(V[e] @ GV[e] @ V[e] @ GV[e]) for e in range(E))
        + sum(beta[i] @ (h[i] - beta[i] @ h[i]) ** 2 for i in range(nA)))
st = 1e-6
fd = (Ecal([mu[i] + st * mud[i] for i in range(nA)], [Sig[i] + st * Sd[i] for i in range(nA)], a + st * ad,
           [V[e] + st * Vd[e] for e in range(E)], {i: beta[i] + st * bd[i] for i in range(nA)})
      - Ecal([mu[i] - st * mud[i] for i in range(nA)], [Sig[i] - st * Sd[i] for i in range(nA)], a - st * ad,
             [V[e] - st * Vd[e] for e in range(E)], {i: beta[i] - st * bd[i] for i in range(nA)})) / (2 * st)
res["dissipation_formula"] = float(diss)
res["dE/dt_finite_difference"] = float(fd)
res["dissipation_rel_err"] = float(abs(fd + diss) / diss)
# reduced: gradient of E_red equals the formula with Gibbs rows (envelope), checked on means and one covariance
bg = gibbs(mu, Sig, a, V)
gmu_g, GS_g, ga_g, GV_g, _ = formulas(mu, Sig, a, V, bg)
errs = []
for i in range(nA):
    for k in range(K):
        dm = [np.zeros(K) for _ in range(nA)]
        dm[i][k] = eps
        fdr = (Ered([mu[l] + dm[l] for l in range(nA)], Sig, a, V) - Ered([mu[l] - dm[l] for l in range(nA)], Sig, a, V)) / (2 * eps)
        errs.append(abs(fdr - gmu_g[i][k]))
U = sym(dw)
Vp = [V[f] + (eps * U if f == 0 else 0) for f in range(E)]
Vm = [V[f] - (eps * U if f == 0 else 0) for f in range(E)]
errs.append(abs((Ered(mu, Sig, a, Vp) - Ered(mu, Sig, a, Vm)) / (2 * eps) - np.trace(GV_g[0] @ U)))
res["max_err_reduced_envelope"] = max(errs)
res["row_gap_identity_err"] = float(abs(Ecal(mu, Sig, a, V, beta) - Ered(mu, Sig, a, V)
                                       - sum(Ti[i] * np.sum(beta[i] * np.log(beta[i] / bg[i])) for i in range(nA))))
# Monte Carlo check of the expected conditional KL for edge 0
e = 0
C0, q0 = edge_parts(e, mu, Sig, a, V)
j, i = edges[e]
L = np.linalg.cholesky(V[e])
ns = 200000
om = a[e * dw:(e + 1) * dw][None, :] + rng.normal(size=(ns, dw)) @ L.T
Tb = q0["Tb"]
mrec = (Tb @ mu[i])[None, :] - om @ (q0["Ke"]).T
msrc = Tb @ mu[j]
diff = mrec - msrc[None, :]
kl = 0.5 * (np.trace(q0["Je"] @ q0["Si"]) - Tb.shape[0] + np.einsum("ni,ij,nj->n", diff, q0["Je"], diff)
            + logdet(q0["Sj"]) - logdet(q0["Si"]))
res["edge0_cost_formula"] = float(C0)
res["edge0_cost_monte_carlo"] = float(kl.mean())
res["edge0_cost_mc_stderr"] = float(kl.std() / np.sqrt(ns))
res["wall_seconds"] = round(time.perf_counter() - t0, 2)
(OUT / "d25_random_flow_check.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
