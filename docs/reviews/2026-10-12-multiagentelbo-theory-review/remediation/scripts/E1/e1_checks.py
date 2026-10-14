"""E1 checks: self copy in the exact receiver update (general A_e and tied case),
dot-product form of w_ij, H-head receiver update, self clause of the constant-row gap."""
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.stats import multivariate_normal as mvn
rng = np.random.default_rng(11)
def spd(k): A = rng.normal(size=(k, k)); return A @ A.T + k*np.eye(k)

# ---------- (1) self copy, general A_e = S^Q - S^K_e ----------
K, d = 3, 2
Sig = spd(K); mu = rng.normal(size=K); SQ = rng.normal(size=(d, K)); W = spd(d); tau = 0.7; Wt = W/tau
SKself_list = {'general': rng.normal(size=(d, K)), 'tied': SQ.copy()}
SK = [rng.normal(size=(d, K)) for _ in range(2)]; kj = [rng.normal(size=K) for _ in range(2)]
pi = np.array([0.3, 0.45, 0.25])
M = 2_000_000; ks = rng.multivariate_normal(mu, Sig, size=M)
Sp = np.linalg.inv(np.linalg.inv(Sig) + SQ.T @ Wt @ SQ); Spred = np.linalg.inv(Wt) + SQ @ Sig @ SQ.T
noise = mvn(mean=np.zeros(d), cov=np.linalg.inv(Wt))
for name, SKs in SKself_list.items():
    A = SQ - SKs
    L = np.vstack([noise.pdf(ks @ A.T)] + [noise.pdf(ks @ SQ.T - Sk @ k) for Sk, k in zip(SK, kj)])
    jw = pi[:, None]*L
    w_mc = jw.sum(1)/jw.sum(); mean_mc = (jw.sum(0)[:, None]*ks).sum(0)/jw.sum()
    # closed form of part (e)
    lw_self = np.log(pi[0]) + mvn(mean=A @ mu, cov=np.linalg.inv(Wt) + A @ Sig @ A.T).logpdf(np.zeros(d))
    lw = [np.log(p) + mvn(mean=SQ @ mu - Sk @ k, cov=Spred).logpdf(np.zeros(d)) for p, Sk, k in zip(pi[1:], SK, kj)]
    w = softmax(np.array([lw_self] + lw))
    Spe = np.linalg.inv(np.linalg.inv(Sig) + A.T @ Wt @ A)
    mean = w[0]*Spe @ np.linalg.solve(Sig, mu) + sum(
        wj*(Sp @ np.linalg.solve(Sig, mu) + Sp @ SQ.T @ Wt @ Sk @ k) for wj, Sk, k in zip(w[1:], SK, kj))
    print(f'[1 {name}] weights MC {np.round(w_mc,4)} closed {np.round(w,4)}; mean err {np.abs(mean-mean_mc).max():.2e}')
    if name == 'tied':
        # sink factor relative to the formula of (a) with r_ii = 0
        naive_self = np.log(pi[0])  # exp form of (a): pi * exp(-0)
        naive_other = [np.log(p) - 0.5*(SQ@mu-Sk@k) @ np.linalg.solve(Spred, SQ@mu-Sk@k) for p, Sk, k in zip(pi[1:], SK, kj)]
        ratio = (w[0]/w[1]) / np.exp(naive_self - naive_other[0])
        print('   sink factor', ratio, ' det(I+Wt SQ Sig SQ^T)^{1/2} =', np.sqrt(np.linalg.det(np.eye(d) + Wt @ SQ @ Sig @ SQ.T)))
        resid = w[0]*np.eye(K) + (1-w[0])*Sp @ np.linalg.inv(Sig)
        mean2 = resid @ mu + sum(wj*Sp @ SQ.T @ Wt @ Sk @ k for wj, Sk, k in zip(w[1:], SK, kj))
        print('   residual-weight form err', np.abs(mean2-mean).max())

# ---------- (2) dot-product form of w_ij (no self copy) ----------
P = np.linalg.inv(Spred)
lg = [np.log(p) + (SQ@mu) @ P @ (Sk@k) - 0.5*(Sk@k) @ P @ (Sk@k) for p, Sk, k in zip(pi[1:], SK, kj)]
lw = [np.log(p) - 0.5*(SQ@mu-Sk@k) @ P @ (SQ@mu-Sk@k) for p, Sk, k in zip(pi[1:], SK, kj)]
print('[2] dot-product row err', np.abs(softmax(lg) - softmax(lw)).max())

# ---------- (3) H-head update ----------
H = 2; dims = [2, 1]
SQh = [rng.normal(size=(dh, K)) for dh in dims]; Wh = [spd(dh) for dh in dims]; tauh = [0.8, 1.3]
Wth = [w_/t for w_, t in zip(Wh, tauh)]
ns = 3; ksrc = [rng.normal(size=K)*1.5 for _ in range(ns)]
SKh = [[rng.normal(size=(dh, K)) for _ in range(ns)] for dh in dims]
pih = [softmax(rng.normal(size=ns)) for _ in range(H)]
Spl = np.linalg.inv(np.linalg.inv(Sig) + sum(S.T @ Wt_ @ S for S, Wt_ in zip(SQh, Wth)))
L = [np.stack([noise_h.pdf(ks @ S.T - (SKh[h][j] @ ksrc[j])) for j in range(ns)])
     for h, (S, noise_h) in enumerate(zip(SQh, [mvn(mean=np.zeros(dh), cov=np.linalg.inv(w)) for dh, w in zip(dims, Wth)]))]
joint = np.einsum('a,b,an,bn->abn', pih[0], pih[1], L[0], L[1])
tab_mc = joint.sum(2)/joint.sum(); mean_mc = (joint.sum((0, 1))[:, None]*ks).sum(0)/joint.sum()
# exact table via stacked predictive
Sst = np.vstack(SQh); Wst_inv = np.zeros((sum(dims),)*2); Wst_inv[:2, :2] = np.linalg.inv(Wth[0]); Wst_inv[2:, 2:] = np.linalg.inv(Wth[1])
Spr = Wst_inv + Sst @ Sig @ Sst.T
lt = np.array([[np.log(pih[0][a]) + np.log(pih[1][b]) + mvn(mean=np.concatenate([SQh[0]@mu - SKh[0][a]@ksrc[a], SQh[1]@mu - SKh[1][b]@ksrc[b]]), cov=Spr).logpdf(np.zeros(3))
                for b in range(ns)] for a in range(ns)])
tab = np.exp(lt - logsumexp(lt))
marg = [tab.sum(1), tab.sum(0)]
mean_op = Spl @ np.linalg.solve(Sig, mu) + sum(marg[h][j]*Spl @ SQh[h].T @ Wth[h] @ SKh[h][j] @ ksrc[j] for h in range(H) for j in range(ns))
mean_joint = sum(tab[a, b]*Spl @ (np.linalg.solve(Sig, mu) + SQh[0].T @ Wth[0] @ SKh[0][a] @ ksrc[a] + SQh[1].T @ Wth[1] @ SKh[1][b] @ ksrc[b]) for a in range(ns) for b in range(ns))
print('[3] table MC vs exact', np.abs(tab_mc - tab).max(), ' factorization defect', np.abs(tab - np.outer(*marg)).max())
print('    operator form vs joint mean', np.abs(mean_op - mean_joint).max(), ' vs MC', np.abs(mean_op - mean_mc).max())
print('    Sigma+ Sigma^-1 identity', np.abs(Spl @ np.linalg.inv(Sig) - (np.eye(K) - Spl @ sum(S.T @ w @ S for S, w in zip(SQh, Wth)))).max())

# ---------- (4) constant-row gap, self clause ----------
Kb = 2; Wb = spd(Kb); tb = 0.9
rho = {0: np.linalg.qr(rng.normal(size=(Kb, Kb)))[0] @ np.diag([1.2, 0.7]), 1: rng.normal(size=(Kb, Kb)), 'self': np.eye(Kb)}
# recognition: Y_0 (receiver i), Y_1, Y_2 independent Gaussians
means = [rng.normal(size=Kb) for _ in range(3)]; covs = [spd(Kb)*0.3 for _ in range(3)]
def D(yi, yj, R): r = yi - R @ yj; return 0.5*r @ Wb @ r
srcs = [('self', 0), (0, 1), (1, 2)]  # (link key, source agent)
for selfR_name, selfR in [('identity', np.eye(Kb)), ('holonomy', rng.normal(size=(Kb, Kb)))]:
    R = {'self': selfR, 0: rho[0], 1: rho[1]}
    d_exact = []
    for key, j in srcs:
        Rk = R[key]
        if j == 0:
            Am = np.eye(Kb) - Rk; mr = Am @ means[0]; Cr = Am @ covs[0] @ Am.T
        else:
            mr = means[0] - Rk @ means[j]; Cr = covs[0] + Rk @ covs[j] @ Rk.T
        d_exact.append(0.5*mr @ Wb @ mr + 0.5*np.trace(Wb @ Cr))
    pirow = np.array([0.2, 0.5, 0.3])
    bQ = softmax(np.log(pirow) - np.array(d_exact)/tb)
    Dmean = np.array([D(means[0], means[j], R[key]) for key, j in srcs])
    shift = []
    for key, j in srcs:
        if j == 0:
            Am = np.eye(Kb) - R[key]
            shift.append((np.trace(Wb @ covs[0]) - np.trace(Wb @ Am @ covs[0] @ Am.T))/(2*tb))
        else:
            shift.append(-np.trace(Wb @ R[key] @ covs[j] @ R[key].T)/(2*tb))
    bpred = softmax(np.log(pirow) - Dmean/tb + np.array(shift))
    print(f'[4 {selfR_name}] row err {np.abs(bQ-bpred).max():.2e}; self shift {shift[0]:.4f} vs Tr(W Cov Y_i)/(2tau) {np.trace(Wb@covs[0])/(2*tb):.4f}')
