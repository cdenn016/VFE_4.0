r"""X2-FRAME check: in the conditional-formation construction (chapter file
11_conditional_formation.tex, thm form:full-flow, form:selection), the shape
equation alpha_i' = -2(alpha_i - alpha_0) does not depend on the coupling kappa.

Independent route: natural-gradient (Fisher) flow on the one-dimensional analog
W_1(nu, p) = Gamma(shape nu, rate p), with energy
  E = sum_i [ KL(q_i || Gamma(alpha_0, 1)) + KL(q_i || Gamma(alpha_0, Lambda)) ],
  Lambda = 1 + (kappa/N) M,  M = sum_i alpha_i / p_i,
gradient by central finite differences, Fisher matrix of Gamma in (alpha, p).
Prints alpha_dot_i + 2(alpha_i - alpha_0) (should be ~0 for every kappa) and
the precision residual p_dot_i - (-2 p_i + 1 + C), C = Lambda + (kappa/N)(M - N alpha_0/Lambda).
"""
import numpy as np
from scipy.special import gammaln, digamma, polygamma

def kl_gamma(a, b, c, d):
    return (a - c) * digamma(a) - gammaln(a) + gammaln(c) + c * (np.log(b) - np.log(d)) + a * (d - b) / b

def energy(theta, kappa, alpha0, N):
    al, p = theta[:N], theta[N:]
    M = np.sum(al / p)
    Lam = 1.0 + kappa * M / N
    return np.sum(kl_gamma(al, p, alpha0, 1.0) + kl_gamma(al, p, alpha0, Lam))

def nat_grad(theta, kappa, alpha0, N, h=1e-6):
    g = np.zeros_like(theta)
    for k in range(theta.size):
        e = np.zeros_like(theta); e[k] = h
        g[k] = (energy(theta + e, kappa, alpha0, N) - energy(theta - e, kappa, alpha0, N)) / (2 * h)
    al, p = theta[:N], theta[N:]
    out = np.zeros_like(theta)
    for i in range(N):
        G = np.array([[polygamma(1, al[i]), -1.0 / p[i]], [-1.0 / p[i], al[i] / p[i] ** 2]])
        v = -np.linalg.solve(G, np.array([g[i], g[N + i]]))
        out[i], out[N + i] = v
    return out

rng = np.random.default_rng(0)
N, alpha0 = 4, 1.3
for kappa in [0.0, 0.5, 1.0, 3.0]:
    worst_a, worst_p = 0.0, 0.0
    for trial in range(5):
        al = rng.uniform(0.8, 3.0, N)
        p = rng.uniform(0.5, 2.0, N)
        th = np.concatenate([al, p])
        v = nat_grad(th, kappa, alpha0, N)
        M = np.sum(al / p); Lam = 1 + kappa * M / N
        C = Lam + (kappa / N) * (M - N * alpha0 / Lam)
        worst_a = max(worst_a, np.max(np.abs(v[:N] + 2 * (al - alpha0))))
        worst_p = max(worst_p, np.max(np.abs(v[N:] - (-2 * p + 1 + C))))
    print(f"kappa={kappa:4.1f}  max|alpha_dot+2(alpha-alpha0)|={worst_a:.2e}  max|p_dot-(-2p+1+C)|={worst_p:.2e}")

# Membership schedule of form:selection depends only on initial belief shapes:
a = np.array([1.0, 1.2, 1.8, 2.1]); delta = 0.1
d = np.diff(a)
tL, tR = 0.5 * np.log(d[0] / delta), 0.5 * np.log(d[2] / delta)
t2 = 0.5 * np.log((d[1] + (d[0] + d[2]) / 2) / delta)
print("event times (kappa-free):", tL, tR, t2, " expected 0.5log2, 0.5log3, 0.5log(17/2):",
      0.5 * np.log(2), 0.5 * np.log(3), 0.5 * np.log(8.5))
