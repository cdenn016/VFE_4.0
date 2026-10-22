"""L03-DYN script 6: records versus the declared potential, and a normalized law-point model.

(1) A Bayesian record update (app:records, eq:predict-and-assimilate) applied at a minimizer of the
    chapter-3 potential raises the potential: no single functional is decreased by both the flow and
    record assimilation.
(2) Law-point Gibbs model for two Bernoulli agents with identity channels and Gibbs rows:
    P_tau(du) = exp(-E_red(u)/tau) mu_J(du) / Z_tau, mu_J the Jeffreys (Fisher volume) measure.
    Z_tau is finite (E_red is nonnegative and the Jeffreys volume of [0,1] is pi); records o_i ~ Ber(p_i)
    give a normalized joint, an evidence Z_tau(o)/Z_tau, and a posterior over law configurations.
"""
import numpy as np

# (1) single agent, binary K and M, product recognition q x s, E = KL(q x s || r G)
r = np.array([0.5, 0.5])
G = np.array([[0.8, 0.2], [0.3, 0.7]])          # G[m, k]


def E0(q, s):
    R = r[:, None] * G                             # R[m, k]
    Q = s[:, None] * q[None, :]
    return np.sum(Q * np.log(Q / R))


# coordinate minimization to the product minimizer (mean-field fixed point)
q, s = np.array([0.5, 0.5]), np.array([0.5, 0.5])
for _ in range(500):
    q = np.exp(s @ np.log(G))
    q /= q.sum()
    s = r * np.exp(-np.array([np.sum(q * np.log(q / G[m])) for m in range(2)]))
    s /= s.sum()
E_before = E0(q, s)
k_obs = np.array([0.9, 0.1])                       # k(o | m) for the observed record o
s_post = s * k_obs / (s @ k_obs)
print("(1) E at product minimizer %.6f; after Bayesian record update of s: %.6f (increase %.6f)"
      % (E_before, E0(q, s_post), E0(q, s_post) - E_before))

# (2) law-point model, two Bernoulli agents, identity peer channels, self excluded, Gibbs rows trivial
def kl_b(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


rr = np.array([0.3, 0.6])                          # E_0 = KL(Ber p_i || Ber r_i)
w = 1.0


def E(p1, p2):
    return kl_b(p1, rr[0]) + kl_b(p2, rr[1]) + w * (kl_b(p1, p2) + kl_b(p2, p1))


# Gauss-Jacobi quadrature for the Jeffreys weight (p(1-p))^{-1/2}: substitute p = sin^2(theta)
n = 400
th = (np.arange(n) + 0.5) * (np.pi / 2) / n
pgrid = np.sin(th) ** 2
dmu = 2.0 * (np.pi / 2) / n                         # Jeffreys measure dp/sqrt(p(1-p)) = 2 dtheta
P1, P2 = np.meshgrid(pgrid, pgrid, indexing="ij")
for tau in (0.5, 0.1):
    wgt = np.exp(-E(P1, P2) / tau) * dmu * dmu
    Z = wgt.sum()
    counts = np.array([[7, 3], [2, 8]])            # records: (successes, failures) for each agent
    like = (P1 ** counts[0, 0]) * ((1 - P1) ** counts[0, 1]) * (P2 ** counts[1, 0]) * ((1 - P2) ** counts[1, 1])
    Zo = (wgt * like).sum()
    post_mean = ((wgt * like * P1).sum() / Zo, (wgt * like * P2).sum() / Zo)
    print("(2) tau=%.2f: Z_tau = %.6f (Jeffreys volume pi^2 = %.4f), log evidence of the records = %.4f,"
          " posterior means of (p1, p2) = (%.4f, %.4f)" % (tau, Z, np.pi ** 2, np.log(Zo / Z), *post_mean))
