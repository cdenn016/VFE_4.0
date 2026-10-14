"""L11 check s2: link-sector formulas an implementation must reproduce.

(a) prop:link-recognition-flow(e): one BLR step with dt=1 on a quadratic integrand gives
    N(Q^-1 b, Q^-1); run through MultiAgentELBO gaussian_blr_step.
(b) prop:link-anchor-m-coordinate(c), eq:link-anchor-quadratic: prior mean of the tilted
    Gaussian equals the recognition mean at alpha = (I + Sigma sum c C^T C) m.
(c) cor:link-delta-rule, eq:link-delta-current: d/dM E 1/2|k_i - M k_j|_W^2 = -J with a
    correlated (nonproduct) matter law, by central differences.
(d) g11 prop:glink-law-level-energy (i)-(ii): closed form of KL(q_i || exp(w)_# q_j) and
    gradient I - L(-w^T, A_e), against direct Gaussian KL and central differences.
(e) g11 prop:glink-expected-loop (ii)-(iii): expected Frobenius loop defect from the
    product of exponential moments vs Monte Carlo, and the flat-mean value 1/2 sum Tr Lambda^-1.
"""
import sys
import numpy as np
import torch
from scipy.linalg import expm, expm_frechet

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\MultiAgentELBO\src")
from multiagent_elbo.realizations.gaussian.latent_natural import gaussian_blr_step

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(11)

# (a) BLR exactness on a quadratic integrand
D = 4
M = rng.normal(size=(D, D)); Q = M @ M.T + D*np.eye(D); b = rng.normal(size=D)
mu0 = rng.normal(size=D); L0 = np.eye(D)*2.0
# expected gradient and Hessian of i(w)=1/2 w^T Q w - b^T w under N(mu0, L0^-1): Q mu0 - b and Q
mu1, L1 = gaussian_blr_step(torch.tensor(mu0), torch.tensor(L0), torch.tensor(Q @ mu0 - b), torch.tensor(Q), 1.0)
print("(a) BLR dt=1: |mu1 - Q^-1 b| =", np.abs(mu1.numpy() - np.linalg.solve(Q, b)).max(),
      " |L1 - Q| =", np.abs(L1.numpy() - Q).max())

# (b) anchor M-coordinate with a quadratic loop action
n = 5
C = rng.normal(size=(2, n)); cs = np.array([0.7, 1.3])
Sig = np.diag(rng.uniform(0.5, 2.0, size=n))
m = rng.normal(size=n)
CtC = sum(c*np.outer(Ci, Ci) for c, Ci in zip(cs, C))
alpha_new = (np.eye(n) + Sig @ CtC) @ m
prior_prec = np.linalg.inv(Sig) + CtC
prior_mean = np.linalg.solve(prior_prec, np.linalg.solve(Sig, alpha_new))
print("(b) anchor formula: |E_prior[w] - m| =", np.abs(prior_mean - m).max())

# (c) delta-rule current with correlated matter
K = 3
mui, muj = rng.normal(size=K), rng.normal(size=K)
B = rng.normal(size=(2*K, 2*K)); Cov = B @ B.T + np.eye(2*K)
Sij, Sj = Cov[:K, K:], Cov[K:, K:]
W = rng.normal(size=(K, K)); W = W @ W.T + np.eye(K)
Mx = rng.normal(size=(K, K))
def Erec(M_):
    mean = mui - M_ @ muj
    T_ = np.hstack([np.eye(K), -M_])
    return 0.5*(mean @ W @ mean + np.trace(W @ T_ @ Cov @ T_.T))
Jth = W @ ((mui - Mx @ muj)[:, None]*muj[None, :] + Sij - Mx @ Sj)
h = 1e-6; G = np.zeros((K, K))
for r in range(K):
    for s in range(K):
        Ep = Mx.copy(); Ep[r, s] += h; Em = Mx.copy(); Em[r, s] -= h
        G[r, s] = (Erec(Ep) - Erec(Em))/(2*h)
print("(c) delta current: |dE/dM + J| =", np.abs(G + Jth).max())

# (d) law-level transported divergence in generator coordinates
def kl(m1, S1, m2, S2):
    S2i = np.linalg.inv(S2); d = m1 - m2
    return 0.5*(np.trace(S2i @ S1) + d @ S2i @ d - len(m1) + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1])
Si = np.cov(rng.normal(size=(K, 10))) + 0.5*np.eye(K); Sjj = np.cov(rng.normal(size=(K, 10))) + 0.5*np.eye(K)
w = 0.4*rng.normal(size=(K, K))
def closed(w_):
    E_ = expm(-w_); Sji = np.linalg.inv(Sjj); r = E_ @ mui - muj
    return 0.5*np.trace(Sji @ E_ @ Si @ E_.T) + 0.5*r @ Sji @ r - K/2 + 0.5*(np.linalg.slogdet(Sjj)[1] - np.linalg.slogdet(Si)[1]) + np.trace(w_)
direct = kl(mui, Si, expm(w) @ muj, expm(w) @ Sjj @ expm(w).T)
Ae = np.linalg.inv(Sjj) @ (expm(-w) @ (Si + np.outer(mui, mui)) - np.outer(muj, mui))
grad_th = np.eye(K) - expm_frechet(-w.T, Ae, compute_expm=False)
Gd = np.zeros((K, K))
for r in range(K):
    for s in range(K):
        Ep = w.copy(); Ep[r, s] += h; Em = w.copy(); Em[r, s] -= h
        Gd[r, s] = (closed(Ep) - closed(Em))/(2*h)
print("(d) law-level energy: |closed - direct| =", abs(closed(w) - direct), " |grad_th - FD| =", np.abs(grad_th - Gd).max())

# (e) expected loop defect (Frobenius metric), triangle of distinct pairs, K=2
K2, n_mc = 2, 200000
mus = [0.3*rng.normal(size=(K2, K2)) for _ in range(3)]; sds = [0.15, 0.2, 0.1]
def exp_moments(mu_, sd_, n_):
    Z = mu_ + sd_*rng.normal(size=(n_, K2, K2)); Ms = torch.linalg.matrix_exp(torch.tensor(Z)).numpy()
    return Ms.mean(0), np.einsum('sac,sbd->abcd', Ms, Ms).reshape(K2*K2, K2*K2)/n_, Ms
firsts, seconds, samples = zip(*[exp_moments(m_, s_, n_mc) for m_, s_ in zip(mus, sds)])
v = np.eye(K2).ravel()
Pm = firsts[0] @ firsts[1] @ firsts[2]; Pk = seconds[0] @ seconds[1] @ seconds[2]
formula = 0.5*v @ Pk @ v - np.trace(Pm) + K2/2
Hs = samples[0] @ samples[1] @ samples[2]
mc = 0.5*((Hs - np.eye(K2))**2).sum((1, 2)).mean()
print(f"(e) expected defect: product-moment formula {formula:.6f} vs direct MC {mc:.6f} (same draws, diff {formula-mc:.2e})")
small = 1e-3
flat = [exp_moments(np.zeros((K2, K2)), small, 200000)[2] for _ in range(3)]
Hf = flat[0] @ flat[1] @ flat[2]
print(f"    flat means, sd={small}: E Q / (1/2 sum Tr Lambda^-1) = {0.5*((Hf-np.eye(K2))**2).sum((1,2)).mean()/(0.5*3*K2*K2*small**2):.4f}")

# (f) prop:obs-selective-receiver-update(a): posterior weights and mean of the receiver,
#     K=2 receiver, comparison dim 1 (rectangular S^Q), three sources, against 2D quadrature.
Kq, Rc, tau_f = 2, 1, 0.6
SQ = rng.normal(size=(Rc, Kq)); SK = [rng.normal(size=(Rc, Kq)) for _ in range(3)]
Wf = np.array([[1.7]]); kj = [rng.normal(size=Kq) for _ in range(3)]; pif = np.array([0.2, 0.5, 0.3])
mu_r = rng.normal(size=Kq); Bq = rng.normal(size=(Kq, Kq)); Sr = Bq @ Bq.T + 0.4*np.eye(Kq)
Wt = Wf/tau_f
Sp = np.linalg.inv(np.linalg.inv(Sr) + SQ.T @ Wt @ SQ)
Spred = np.linalg.inv(Wt) + SQ @ Sr @ SQ.T
r = [SQ @ mu_r - SK[j] @ kj[j] for j in range(3)]
wl = np.array([np.log(pif[j]) - 0.5*r[j] @ np.linalg.solve(Spred, r[j]) for j in range(3)]); wf = np.exp(wl - wl.max()); wf /= wf.sum()
mean_th = Sp @ np.linalg.solve(Sr, mu_r) + sum(wf[j]*Sp @ SQ.T @ Wt @ SK[j] @ kj[j] for j in range(3))
g = np.linspace(-12, 12, 1601); X, Y = np.meshgrid(g, g, indexing='ij'); pts = np.stack([X, Y], -1)
d = pts - mu_r; logprior = -0.5*np.einsum('...i,ij,...j->...', d, np.linalg.inv(Sr), d)
comp = []
for j in range(3):
    res = np.einsum('ri,...i->...r', SQ, pts) - SK[j] @ kj[j]
    comp.append(np.log(pif[j]) - 0.5*np.einsum('...r,rs,...s->...', res, Wt, res))
comp = np.stack(comp, -1); joint = logprior[..., None] + comp
mx = joint.max(); pj = np.exp(joint - mx)
post_w = pj.sum((0, 1)); post_w /= post_w.sum()
dens = pj.sum(-1); dens /= dens.sum()
mean_q = np.einsum('ab,abi->i', dens, pts)
print("(f) receiver update: |w - w_quad| =", np.abs(wf - post_w).max(), " |mean - mean_quad| =", np.abs(mean_th - mean_q).max())
