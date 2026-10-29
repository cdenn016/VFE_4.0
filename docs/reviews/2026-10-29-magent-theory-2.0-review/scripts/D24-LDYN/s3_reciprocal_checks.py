"""D24-LDYN script 3: it2:potential, it2:potential-forces (i)-(iv), it2:certainty (ii)-(iii),
the V_e stationarity remark, the Gibbs reduction it2:row-gap, and it2:echo (numerical)."""
import numpy as np
from tangent_lib import Model, kappa, code_kron, null_basis, spd

rng = np.random.default_rng(7)
edges = [(1, 0), (0, 1), (2, 1), (0, 2), (2, 0)]
M = Model(3, 2, edges, rng, sigma2=0.9, lam=0.6, ranks=[1, 2, 1, 2, 1])
K, d = M.K, M.d
wcols = np.concatenate(M.iw)
ycols = np.concatenate(M.iy)
Aom = np.eye(len(wcols)) / M.sigma2 + M.lam * sum((B.T @ W @ B)[np.ix_(wcols, wcols)] for B, W in zip(M.Bg, M.W))
for k in range(M.E):
    sl = slice(k * d, (k + 1) * d)
    Aom[sl, sl] += M.G(k)
bom = np.concatenate([M.G(k) @ M.what[k] for k in range(M.E)])
a0 = np.linalg.solve(Aom, bom)

def kl_gauss(ma, A, mb, B):
    Bi = np.linalg.inv(B)
    dm = ma - mb
    return 0.5 * (np.trace(Bi @ A) - len(ma) + dm @ Bi @ dm + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def edge_cost(k, mu, Sig, a, V):
    j, i = M.edges[k]
    T = M.T[k]
    kap = kappa(M.c[i])
    Si = T @ Sig[i] @ T.T + M.R[k]
    Sj = T @ Sig[j] @ T.T + M.R[k]
    dvec = T @ (mu[i] - mu[j] - kap @ a[k])
    Ji = np.linalg.inv(Sj)
    U = T @ kap @ V[k] @ kap.T @ T.T
    n = T.shape[0]
    return 0.5 * (np.trace(Ji @ Si) - n + dvec @ Ji @ dvec + np.trace(Ji @ U) + np.linalg.slogdet(Sj)[1] - np.linalg.slogdet(Si)[1])

def potential(mu, Sig, a, V, beta):
    tot = 0.0
    for i in range(M.N):
        tot += kl_gauss(mu[i], Sig[i], M.m[i], np.linalg.inv(M.P[i]))
    for k in range(M.E):
        tot += beta[k] * edge_cost(k, mu, Sig, a, V)
    avec = np.concatenate(a)
    Vb = np.zeros((len(wcols),) * 2)
    for k in range(M.E):
        Vb[k * d:(k + 1) * d, k * d:(k + 1) * d] = V[k]
    tot += kl_gauss(avec, Vb, a0, np.linalg.inv(Aom))
    return tot

# Monte Carlo check of the averaged edge cost (proof of it2:potential-forces (i))
mu = [rng.normal(size=K) for _ in range(M.N)]
Sig = [spd(K, rng, 0.3, 2) for _ in range(M.N)]
a = [rng.normal(size=d) for _ in range(M.E)]
V = [spd(d, rng, 0.05, 0.5) for _ in range(M.E)]
k = 2
j, i = M.edges[k]
T = M.T[k]
kap = kappa(M.c[i])
Si = T @ Sig[i] @ T.T + M.R[k]
Sj = T @ Sig[j] @ T.T + M.R[k]
Lc = np.linalg.cholesky(V[k])
ws = a[k][None, :] + rng.normal(size=(100000, d)) @ Lc.T
vals = [kl_gauss(T @ (mu[i] - kap @ w), Si, T @ mu[j], Sj) for w in ws[:20000]]
print("[it2:potential-forces (i)] MC average of KL over r_e:", np.mean(vals), "+-", np.std(vals) / np.sqrt(len(vals)), " closed form:", edge_cost(k, mu, Sig, a, V))

beta = rng.uniform(0.3, 1.5, size=M.E)
Srep = [(M.T[k] @ Sig[M.edges[k][0]] @ M.T[k].T + M.R[k]) / beta[k] for k in range(M.E)]
JS, etaS = M.global_J_eta(Rrep=Srep, ozero=True)
def ES(x):
    return 0.5 * x @ JS @ x - etaS @ x
diffs = []
for t in range(6):
    mu_t = [rng.normal(size=K) for _ in range(M.N)]
    a_t = [rng.normal(size=d) for _ in range(M.E)]
    x = np.concatenate(mu_t + a_t)
    diffs.append(potential(mu_t, Sig, a_t, V, beta) - ES(x))
print("  E - E^S at 6 random mean configurations (fixed covariances): spread =", np.ptp(diffs))
Mmob = np.zeros_like(JS)
for ii in range(M.N):
    Mmob[np.ix_(M.iy[ii], M.iy[ii])] = Sig[ii]
for kk in range(M.E):
    Mmob[np.ix_(M.iw[kk], M.iw[kk])] = V[kk]
ev = np.linalg.eigvals(-Mmob @ JS)
print("[it2:potential-forces (iii)] -M J^S: max|Im| =", np.abs(ev.imag).max(), " max Re =", ev.real.max())

# (iv) beta = 1 comparison with the per-agent law-level model at the same covariances, o = 0
beta1 = np.ones(M.E)
S1 = [M.T[k] @ Sig[M.edges[k][0]] @ M.T[k].T + M.R[k] for k in range(M.E)]
J1, eta1 = M.global_J_eta(Rrep=S1, ozero=True)
Hl, bl, Al = M.per_agent_H_b('law', Sig, V, ozero=True)
send = np.zeros_like(J1)
for ii in range(M.N):
    for kk in M.out[ii]:
        L = M.Lmat[kk]
        send[M.iy[ii]] += (L.T @ np.linalg.inv(S1[kk]) @ L)[M.iy[ii]]
print("[it2:potential-forces (iv)] |content rows of J^S - sender - per-agent law-level| =", np.abs((J1 - send)[ycols] - Hl[ycols]).max(),
      " |b rows| =", np.abs(eta1[ycols] - bl[ycols]).max())
lam_save = M.lam
# dictionary rows: replace per-agent widened loop precisions by lambda W, i.e. plug-in loops with law-level records
Hmix = np.zeros_like(J1)
for ii in range(M.N):
    Q, q = M.per_agent_Q(ii, 'law', Sig, V, ozero=True)
    # rebuild loop part with lambda W instead of Omega
    for g, cyc in enumerate(M.loops):
        if not any(kk in M.inn[ii] for kk in cyc):
            continue
        Vsum = sum(V[f] for f in cyc if f not in M.inn[ii])
        Om = np.linalg.inv(np.linalg.inv(M.lam * M.W[g]) + Vsum)
        B = M.Bg[g]
        Q = Q - B.T @ Om @ B + M.lam * B.T @ M.W[g] @ B
    Hmix[M.blocks[ii], :] = Q[M.blocks[ii], :]
print("  |dictionary rows of J^S - per-agent law-level with loops un-widened| =", np.abs(J1[wcols] - Hmix[wcols]).max(),
      "; with widening:", np.abs(J1[wcols] - Hl[wcols]).max())

# it2:certainty (ii): sender-covariance derivative of the unweighted edge cost
k = 3
j, i = M.edges[k]
T = M.T[k]
kap = kappa(M.c[i])
Sk = T @ Sig[i] @ T.T + M.R[k]
Sj = T @ Sig[j] @ T.T + M.R[k]
dvec = T @ (mu[i] - mu[j] - kap @ a[k])
Ji = np.linalg.inv(Sj)
Dform = 0.5 * T.T @ Ji @ (Sj - Sk - np.outer(dvec, dvec) - T @ kap @ V[k] @ kap.T @ T.T) @ Ji @ T
errs = []
for t in range(5):
    U = rng.normal(size=(K, K))
    U = U + U.T
    h = 1e-6
    Sp = [s.copy() for s in Sig]
    Sm = [s.copy() for s in Sig]
    Sp[j] = Sig[j] + h * U
    Sm[j] = Sig[j] - h * U
    num = (edge_cost(k, mu, Sp, a, V) - edge_cost(k, mu, Sm, a, V)) / (2 * h)
    errs.append(abs(num - np.trace(Dform @ U)))
print("[it2:certainty (ii)] max |finite difference - tr(formula U)| over 5 symmetric U:", max(errs))

# V_e stationarity remark (line 302), fixed rows
k = 1
j, i = M.edges[k]
T = M.T[k]
kap = kappa(M.c[i])
Sj = T @ Sig[j] @ T.T + M.R[k]
Vstar = np.linalg.inv(Aom[k * d:(k + 1) * d, k * d:(k + 1) * d] + beta[k] * kap.T @ T.T @ np.linalg.inv(Sj) @ T @ kap)
Vt = [v.copy() for v in V]
Vt[k] = Vstar
gmax = 0.0
for t in range(5):
    U = rng.normal(size=(d, d))
    U = U + U.T
    h = 1e-6
    Vp = [v.copy() for v in Vt]
    Vm = [v.copy() for v in Vt]
    Vp[k] = Vstar + h * U
    Vm[k] = Vstar - h * U
    gmax = max(gmax, abs((potential(mu, Sig, a, Vp, beta) - potential(mu, Sig, a, Vm, beta)) / (2 * h)))
print("[V_e stationarity] max |directional derivative of E in V_e| at the displayed V_e:", gmax)
k2 = 2  # rank-one code, nontrivial N_e
j2, i2 = M.edges[k2]
T2 = M.T[k2]
kap2 = kappa(M.c[i2])
Sj2 = T2 @ Sig[j2] @ T2.T + M.R[k2]
TK = code_kron(T2)
Nk, _ = null_basis(TK)
Gk = M.G(k2)
rec = beta[k2] * kap2.T @ T2.T @ np.linalg.inv(Sj2) @ T2 @ kap2
print("  dim N_e on edge 2:", Nk.shape[1])
print("  grounded and record information on N_e: |G_e Z|, |record Z| =", np.abs(Gk @ Nk).max(), np.abs(rec @ Nk).max())

# it2:certainty (iii): scalar sender, unique root increasing in d^2
def f(v, d2, pj=0.8, vk=0.6, R=0.5):
    return pj - 1 / v + 1 / (v + R) - (vk + R + d2) / (v + R) ** 2
from scipy.optimize import brentq
vgrid = np.logspace(-4, 3, 4000)
roots = []
for d2 in (0.0, 0.5, 1.0, 4.0, 16.0):
    fv = f(vgrid, d2)
    sign_changes = int(np.sum(np.diff(np.sign(fv)) > 0)) + int(np.sum(np.diff(np.sign(fv)) < 0))
    mono = bool(np.all(np.diff(fv) > 0))
    roots.append(brentq(lambda v: f(v, d2), 1e-6, 1e4))
    print(f"[it2:certainty (iii)] d2={d2}: f strictly increasing on grid {mono}, sign changes {sign_changes}, root {roots[-1]:.6f}")
print("  roots increase with d2:", bool(np.all(np.diff(roots) > 0)))

# it2:row-gap: Gibbs reduction, gap identity and the entropy-free surrogate
Tt = 0.37
pi = np.array([0.2, 0.5, 0.3])
def costs(x):
    return np.array([np.sin(x) + 1.5, x ** 2, 0.3 * np.exp(x)])
def dcosts(x):
    return np.array([np.cos(x), 2 * x, 0.3 * np.exp(x)])
def Ered(x):
    C = costs(x)
    return -Tt * np.log(np.sum(pi * np.exp(-C / Tt)))
def bstar(x):
    C = costs(x)
    w = pi * np.exp(-C / Tt)
    return w / w.sum()
x0 = 0.4
h = 1e-6
num = (Ered(x0 + h) - Ered(x0 - h)) / (2 * h)
print("[it2:row-gap] dE_T/dx (finite diff) - sum beta* dC =", num - bstar(x0) @ dcosts(x0))
bt = rng.dirichlet(np.ones(3))
full = bt @ costs(x0) + Tt * np.sum(bt * np.log(bt / pi))
print("  E_T(beta) - [E_T,red + T KL(beta||beta*)] =", full - (Ered(x0) + Tt * np.sum(bt * np.log(bt / bstar(x0)))))
def surrogate(x):
    return bstar(x) @ costs(x)
nums = (surrogate(x0 + h) - surrogate(x0 - h)) / (2 * h)
b = bstar(x0)
C, dC = costs(x0), dcosts(x0)
cov = b @ (C * dC) - (b @ C) * (b @ dC)
print("  surrogate derivative - [sum beta* dC - Cov_beta*(C,dC)/T] =", nums - (b @ dC - cov / Tt), "; surrogate gap size:", nums - b @ dC)

# it2:echo numerics
print("\n[it2:echo]")
sig2 = 1.3
for ell in (2, 3, 4, 6):
    for lam in (0.5, 5.0, 1e6):
        Jm = np.eye(ell) / sig2 + lam * np.ones((ell, ell))
        exact = np.linalg.inv(Jm)[0, 0]
        fact = 1 / Jm[0, 0]
        def phi(vv):
            return 1 / (1 / sig2 + 1 / (1 / lam + (ell - 1) * vv))
        vlo, vhi = 0.0, sig2
        for it in range(200000):
            vlo2, vhi2 = phi(vlo), phi(vhi)
            if abs(vlo2 - vlo) < 1e-15 and abs(vhi2 - vhi) < 1e-15:
                break
            vlo, vhi = vlo2, vhi2
        cav = 1 / (1 / sig2 + 1 / (1 / lam + (ell - 1) * sig2))
        print(f"  ell={ell} lam={lam:g}: exact {exact:.6g} cavity {cav:.6g} factorized {fact:.4g} per-agent(low start) {vlo:.6g} (high start) {vhi:.6g}; "
              f"order fact<pa<exact {fact < vlo < exact}")
    print(f"    lam->oo limits: exact {sig2*(ell-1)/ell:.6g}, per-agent {sig2*(ell-2)/(ell-1):.6g}" + ("" if ell > 2 else f" (ell=2: sigma lam^-1/2 = {np.sqrt(sig2/1e6):.4g})"))
# asymmetric fixed points of the echo map: v_e = phi(sum_{f ne e} v_f)
ell, lam = 4, 3.0
vs = rng.uniform(0, sig2, size=ell)
for it in range(5000):
    vs = np.array([1 / (1 / sig2 + 1 / (1 / lam + vs.sum() - vs[e])) for e in range(ell)])
print("  random asymmetric start converges to symmetric point:", np.ptp(vs) < 1e-14, vs[0])
