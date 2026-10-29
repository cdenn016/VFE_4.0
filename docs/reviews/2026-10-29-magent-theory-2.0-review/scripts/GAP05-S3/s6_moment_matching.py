r"""GAP05-S3 s6: the moment-matching remark at chapters/18_link_dynamics.tex:300.

A. Identity kernel (the compared law is the exponential-family source itself):
   d_eta KL(p || q_eta) = E_q t - E_p t for a categorical source (logits) and a von Mises
   source (natural parameters (k cos a, k sin a), t = (cos x, sin x)).
B. Kernel-mediated source, the structure of it2:law-potential (B_e = q_j K_src):
   d_eta KL(p || q_eta K) = E_q t - E_ptilde t with ptilde = int p(dh) q_eta(. | h), and
   the stationary source is NOT the moment match E_q t = E_p t.
C. Gaussian tangent potential: it2:source-covariance equals the EM form
   1/2 Sigma^{-1} (Sigma - E_ptilde (x - mu)(x - mu)^T) Sigma^{-1}.
D. Non-Gaussian prior: the exact posterior variance depends on the registered record
   (the caveat of it2:certainty (i)).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import laplace, norm

rng = np.random.default_rng(11)
h = 1e-30
def cstep(f, x, v): return np.imag(f(x + 1j * h * v)) / h

print("== A1. categorical source, identity kernel ==")
n = 4
p = rng.dirichlet(np.ones(n) * 2)
def q_of(eta):
    z = np.concatenate([eta, [0.0]]); w = np.exp(z - 0); return w / np.sum(w)
def KLpq(eta): return np.sum(p * (np.log(p) - np.log(q_of(eta))))
eta0 = rng.standard_normal(n - 1)
grad = np.array([cstep(KLpq, eta0.astype(complex), np.eye(n - 1)[k]) for k in range(n - 1)])
print("  max |grad - (E_q t - E_p t)| = %.1e" % np.abs(grad - (q_of(eta0)[:-1] - p[:-1])).max())

print("== A2. von Mises source, identity kernel ==")
N = 512; x = 2 * np.pi * np.arange(N) / N; dx = 2 * np.pi / N
pc = 1 + 0.7 * np.cos(x - 1.0) + 0.2 * np.cos(2 * x); pc /= np.sum(pc) * dx
def vm_eta(eta):
    f = np.exp(eta[0] * np.cos(x) + eta[1] * np.sin(x)); return f / (np.sum(f) * dx)
def KLvm(eta): return np.sum(pc * (np.log(pc) - np.log(vm_eta(eta)))) * dx
eta0 = np.array([0.8, -0.4])
grad = np.array([cstep(KLvm, eta0.astype(complex), np.eye(2)[k]) for k in range(2)])
tq = np.array([np.sum(np.cos(x) * vm_eta(eta0)), np.sum(np.sin(x) * vm_eta(eta0))]) * dx
tp = np.array([np.sum(np.cos(x) * pc), np.sum(np.sin(x) * pc)]) * dx
print("  max |grad - (E_q t - E_p t)| = %.1e" % np.abs(grad - (tq - tp)).max())

print("== B. categorical source seen through a noisy code K (n contents -> n messages) ==")
K = 0.7 * np.eye(n) + 0.3 * rng.dirichlet(np.ones(n), size=n)        # row-stochastic, invertible
qstar = rng.dirichlet(np.ones(n) * 3)
pmsg = qstar @ K                                                       # receiver message law, in the image of K
def KLpqK(eta): return np.sum(pmsg * (np.log(pmsg) - np.log(q_of(eta) @ K)))
eta0 = rng.standard_normal(n - 1)
grad = np.array([cstep(KLpqK, eta0.astype(complex), np.eye(n - 1)[k]) for k in range(n - 1)])
qe = q_of(eta0)
post = (qe[:, None] * K) / (qe @ K)[None, :]                           # q_eta(x | h), columns h
ptilde = post @ pmsg
print("  max |grad - (E_q t - E_ptilde t)| = %.1e   (EM form)" % np.abs(grad - (qe[:-1] - ptilde[:-1])).max())
print("  max |grad - (E_q t - E_p t)|      = %.3e   (displayed formula with p read on the content alphabet)" % np.abs(grad - (qe[:-1] - pmsg[:-1])).max())
res = minimize(KLpqK, np.zeros(n - 1), method='BFGS', options={'gtol': 1e-12})
qs = q_of(res.x)
post = (qs[:, None] * K) / (qs @ K)[None, :]
ptilde = post @ pmsg
print("  stationary source: KL(p || q K) = %.2e, |q - q*| = %.1e" % (res.fun, np.abs(qs - qstar).max()))
print("  at stationarity |E_q t - E_ptilde t| = %.1e but |E_q t - E_p t| = %.3f" % (np.abs(qs - ptilde).max(), np.abs(qs - pmsg).max()))
print("  the naive moment match q = p gives KL(p || p K) = %.4f > 0" % np.sum(pmsg * (np.log(pmsg) - np.log(pmsg @ K))))

print("== C. Gaussian tangent potential: it2:source-covariance is the EM form ==")
Kd, nd = 3, 2
T = rng.standard_normal((nd, Kd))
A = rng.standard_normal((Kd, Kd)); Sig = A @ A.T + 0.5 * np.eye(Kd)
Rm = np.diag([0.3, 0.6])
mu = rng.standard_normal(Kd)
mp = rng.standard_normal(nd); Bm = rng.standard_normal((nd, nd)); Cp = Bm @ Bm.T + 0.2 * np.eye(nd)
def cost(Sigma):
    S = T @ Sigma @ T.T + Rm
    d = mp - T @ mu
    Si = np.linalg.inv(S)
    return 0.5 * (np.trace(Si @ Cp) + d @ Si @ d + np.log(np.linalg.det(S)))     # Sigma-dependent part of KL(p || N(T mu, S))
G = np.zeros((Kd, Kd))
for a in range(Kd):
    for b in range(Kd):
        E = np.zeros((Kd, Kd)); E[a, b] = E[b, a] = 1.0
        G[a, b] = cstep(cost, Sig.astype(complex), E) / (1.0 if a == b else 2.0)
S = T @ Sig @ T.T + Rm; Si = np.linalg.inv(S)
Cm = Cp + np.outer(mp - T @ mu, mp - T @ mu)                 # second moment of p about T mu
form = 0.5 * T.T @ Si @ (S - Cm) @ Si @ T
O = Sig @ T.T @ Si
second_ptilde = Sig - O @ T @ Sig + O @ Cm @ O.T          # E_ptilde (x - mu)(x - mu)^T
em = 0.5 * np.linalg.inv(Sig) @ (Sig - second_ptilde) @ np.linalg.inv(Sig)
print("  max |complex-step gradient - it2:source-covariance| = %.1e" % np.abs(G - form).max())
print("  max |it2:source-covariance - EM form| = %.1e" % np.abs(form - em).max())

print("== D. a non-Gaussian prior: the posterior variance depends on the record ==")
yy = np.linspace(-25, 25, 200001); dyy = yy[1] - yy[0]
prior = laplace.pdf(yy)
for o in (0.0, 1.0, 3.0, 6.0):
    post = prior * norm.pdf(o, loc=yy, scale=0.5); post /= np.sum(post) * dyy
    m = np.sum(yy * post) * dyy
    print("  Laplace prior, Gaussian record noise 0.5, o = %.1f: posterior variance %.5f" % (o, np.sum((yy - m) ** 2 * post) * dyy))

print("== B2. von Mises source through von Mises code noise (compared family not exponential) ==")
Kc = np.exp(1.5 * np.cos(x[None, :] - x[:, None])); Kc = Kc / (np.sum(Kc, axis=1, keepdims=True) * dx)   # Kc[y, h]
pm = 1 + 0.5 * np.cos(x - 2.0) + 0.3 * np.sin(2 * x); pm /= np.sum(pm) * dx                          # receiver message law
def KLvmK(eta):
    qK = vm_eta(eta) @ Kc * dx
    return np.sum(pm * (np.log(pm) - np.log(qK))) * dx
eta0 = np.array([1.1, 0.6])
grad = np.array([cstep(KLvmK, eta0.astype(complex), np.eye(2)[k]) for k in range(2)])
qe = vm_eta(eta0)
qK = qe @ Kc * dx
post = (qe[:, None] * Kc) / qK[None, :]
ptilde = post @ pm * dx
tq = np.array([np.sum(np.cos(x) * qe), np.sum(np.sin(x) * qe)]) * dx
tpt = np.array([np.sum(np.cos(x) * ptilde), np.sum(np.sin(x) * ptilde)]) * dx
tqK = np.array([np.sum(np.cos(x) * qK), np.sum(np.sin(x) * qK)]) * dx
tpm = np.array([np.sum(np.cos(x) * pm), np.sum(np.sin(x) * pm)]) * dx
print("  gradient %s" % np.round(grad, 10))
print("  EM form E_q t - E_ptilde t                 %s  (residual %.1e)" % (np.round(tq - tpt, 10), np.abs(grad - (tq - tpt)).max()))
print("  message-space match E_qK t_H - E_p t_H       %s  (residual %.3e)" % (np.round(tqK - tpm, 10), np.abs(grad - (tqK - tpm)).max()))
print("  content-space reading E_q t - E_p t          %s  (residual %.3e)" % (np.round(tq - tpm, 10), np.abs(grad - (tq - tpm)).max()))
