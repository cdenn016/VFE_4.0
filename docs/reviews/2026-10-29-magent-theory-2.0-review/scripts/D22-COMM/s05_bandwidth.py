"""D22-COMM s05: bandwidth and source rows (chapters/17_communication_links.tex:261-275; companion c17:177-198).

Checks:
  (a) it1:bandwidth: normalized posterior log-odds against E_nu[log nu_j - log nu_k] and the KL difference for a
      Gaussian misspecified example (SLLN; agreement is evidence, not proof).
  (b) companion Bernoulli example: log odds (2k-n) log 3 exactly.
  (c) the 2026-10-23 G6 example: nu(x) = 1/(x log(1/x)^2) on (0, e^-1) has mass 1 and E_nu log nu = +infinity,
      so KL(nu||Unif(0,1)) = +infinity while E_nu|log nu_j| = 0 (repaired statement needed).
  (d) remark after it1:bandwidth: the data-processing bound "message-law divergence <= full-belief divergence"
      requires one common code; with agent-specific codes it fails (q = q_j, different codes).
"""
import numpy as np
import mpmath as mp

rng = np.random.default_rng(5)

# (a) Gaussian message laws: truth nu = N(0.3, 1.2), candidates nu_j = N(0, 1), nu_k = N(1, 2)
mu, s2 = 0.3, 1.2
cand = {"j": (0.0, 1.0), "k": (1.0, 2.0)}


def kl1(m0, v0, m1, v1):
    return 0.5 * (v0 / v1 + (m1 - m0) ** 2 / v1 - 1 + np.log(v1 / v0))


def logpdf(x, m, v):
    return -0.5 * np.log(2 * np.pi * v) - 0.5 * (x - m) ** 2 / v


n = 200000
h = mu + np.sqrt(s2) * rng.standard_normal(n)
pij, pik = 0.3, 0.7
lo = (np.log(pij / pik) + np.sum(logpdf(h, *cand["j"]) - logpdf(h, *cand["k"]))) / n
pred = kl1(mu, s2, *cand["k"]) - kl1(mu, s2, *cand["j"])
print("(a) normalized log-odds at n=2e5:", lo, " KL(nu||nu_k) - KL(nu||nu_j) =", pred, " difference =", lo - pred)

# (b) Bernoulli example
for (n_b, k_b) in [(10, 7), (100, 41)]:
    lo_b = k_b * np.log(0.75 / 0.25) + (n_b - k_b) * np.log(0.25 / 0.75)
    print(f"(b) n={n_b}, k={k_b}: log odds = {lo_b}, (2k-n) log 3 = {(2 * k_b - n_b) * np.log(3)}")

# (c) G6 example
mp.mp.dps = 30
f = lambda x: 1 / (x * mp.log(1 / x) ** 2)
mass_t = mp.quad(lambda t: 1 / t ** 2, [1, mp.inf])  # substitution t = log(1/x), dx/x = -dt
mass_x = mp.quad(f, [mp.e ** -60, mp.e ** -20, mp.e ** -5, mp.e ** -1]) + 1 / mp.mpf(60)  # exact tail below e^-60 is 1/60
print("(c) mass of nu: t-substitution =", mass_t, "; x-quadrature with exact tail =", mass_x)
# E log nu = int_1^inf (t - 2 log t) t^-2 dt after t = log(1/x); partial integrals up to L
for L in [1e2, 1e4, 1e8]:
    part = mp.quad(lambda t: (t - 2 * mp.log(t)) / t ** 2, [1, L])
    print(f"(c) partial E_nu log nu up to t={L:.0e}: {mp.nstr(part, 8)} (grows like log L = {mp.nstr(mp.log(L), 8)})")

# (d) different codes, equal beliefs
v_q = 1.0
T_true, T_cand = 1.0, 2.0
kl_msg = kl1(0.0, T_true ** 2 * v_q, 0.0, T_cand ** 2 * v_q)
print("(d) q = q_j = N(0,1), true code 1, candidate code 2: KL(message laws) =", kl_msg, " > KL(q||q_j) = 0")
# common code: data processing holds
qa, qb = (0.5, 1.0), (0.0, 2.0)
Tm = np.array([[1.0, 0.0]])
Sa = np.diag([1.0, 3.0]); Sb = np.diag([2.0, 1.0]); ma = np.array([0.5, 1.0]); mb = np.array([0.0, -1.0])
def klK(m0, S0, m1, S1):
    S1i = np.linalg.inv(S1); d = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + d @ S1i @ d - len(m0) + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])
print("(d) common code: KL(Tq||Tq_j) =", klK(Tm @ ma, Tm @ Sa @ Tm.T, Tm @ mb, Tm @ Sb @ Tm.T), " <= KL(q||q_j) =", klK(ma, Sa, mb, Sb))
