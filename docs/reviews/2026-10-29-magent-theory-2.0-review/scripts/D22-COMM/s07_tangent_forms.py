"""D22-COMM s07: tangent forms at H = I (it1:loop-comparisons, chapters/17_communication_links.tex:251-257).

Compares the Hessian at X = 0 of X -> KL(q || (e^X)_# q) for Gaussian q with the Hessian of X -> Q(e^X; Sigma).
The KL form is the Gaussian Fisher pullback of the action, degenerate on the stabilizer algebra
{X : X Sigma + Sigma X^T = 0, X mu = 0}; the dressed defect has a positive-definite tangent form on gl(K).
"""
import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(17)
K = 3


def kl_gauss(m0, S0, m1, S1):
    S1i = np.linalg.inv(S1); d = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + d @ S1i @ d - len(m0) + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


def Qdef(H, S):
    X = np.eye(len(S)) - H
    return 0.5 * np.trace(X.T @ np.linalg.inv(S) @ X @ S)


def hessian(fun, n, eps=1e-4):
    Hm = np.zeros((n, n))
    E = np.eye(n)
    for a in range(n):
        for b in range(n):
            Hm[a, b] = (fun(eps * (E[a] + E[b])) - fun(eps * (E[a] - E[b])) - fun(eps * (-E[a] + E[b])) + fun(-eps * (E[a] + E[b]))) / (4 * eps ** 2)
    return 0.5 * (Hm + Hm.T)


X0 = rng.standard_normal((K, K)); Sigma = X0 @ X0.T + 0.3 * np.eye(K)
for mu, lab in [(np.zeros(K), "zero mean"), (rng.standard_normal(K), "nonzero mean")]:
    def f_kl(v):
        H = expm(v.reshape(K, K))
        return kl_gauss(mu, Sigma, H @ mu, H @ Sigma @ H.T)
    def f_q(v):
        return Qdef(expm(v.reshape(K, K)), Sigma)
    Hk = hessian(f_kl, K * K); Hq = hessian(f_q, K * K)
    ek = np.linalg.eigvalsh(Hk); eq = np.linalg.eigvalsh(Hq)
    print(f"{lab}: KL tangent form rank = {int(np.sum(ek > 1e-6 * ek.max()))} of {K*K} (null dim predicted"
          f" {K*(K-1)//2 if lab == 'zero mean' else 1}); dressed-defect tangent min eigenvalue = {eq.min():.4f} (rank {int(np.sum(eq > 1e-6 * eq.max()))})")
