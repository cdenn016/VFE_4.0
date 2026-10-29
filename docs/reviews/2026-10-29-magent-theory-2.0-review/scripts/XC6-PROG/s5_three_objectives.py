r"""XC6-PROG s5: two exact statements bearing on op:dyn-three-pictures (chapters/03_dynamics.tex).

 (a) Exact-potential (Monderer-Shapley) test.  The per-agent dynamics is coordinate descent of
     some global functional Phi (F_i = Phi + c_i(x_{-i})) only if the mixed partials agree:
     d^2 F_1/dq_1 dq_2 = d^2 F_2/dq_2 dq_1.  We evaluate both bilinear forms on tangent vectors
     for per-agent free energies with unit-temperature Gibbs rows (reduced form), identity
     channels, uniform evaluators, and (i) one-way attention 2 -> 1, (ii) mutual attention,
     at random interior Bernoulli laws.
 (b) Per-agent stationary points versus critical points of F_tot = sum_i F_i.  At a per-agent
     stationary point the F_tot gradient on q_j equals the centered source-role term
     sum_i beta_ij * (-K_ij (q_i/nu_ij)); with identity channels it vanishes only if the pooled
     receiver law equals the source law.  We compute a per-agent stationary point of two
     mutually attending Bernoulli agents with distinct references and evaluate that term.
"""
import numpy as np

rng = np.random.default_rng(5)


def ber(p: float) -> np.ndarray:
    return np.array([1 - p, p])


def kl(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * np.log(a / b)))


def F_reduced(qi: np.ndarray, peers: list, pi: np.ndarray, G: np.ndarray) -> float:
    r"""Unit-temperature reduced per-agent free energy without records:
    F_i = -log( pi_ii e^{-KL(q_i||G)} + sum_j pi_ij e^{-KL(q_i||q_j)} )."""
    costs = [kl(qi, G)] + [kl(qi, qj) for qj in peers]
    return float(-np.log(np.sum(pi * np.exp(-np.array(costs)))))


def mixed(Ffun, p1: float, p2: float, h: float = 1e-4) -> float:
    r"""d^2 F / dp_1 dp_2 by central differences (Bernoulli parameters as coordinates)."""
    return (Ffun(p1 + h, p2 + h) - Ffun(p1 + h, p2 - h) - Ffun(p1 - h, p2 + h) + Ffun(p1 - h, p2 - h)) / (4 * h * h)


G = ber(0.5)
print("(a) mixed partials d2F1/dp1dp2 vs d2F2/dp2dp1 (Bernoulli coordinates):")
for trial in range(4):
    p1, p2 = rng.uniform(0.15, 0.85, 2)
    pi1 = np.array([0.4, 0.6])                    # agent 1: self, peer 2
    # (i) one-way attention: agent 2 attends to itself only
    F1 = lambda a, b: F_reduced(ber(a), [ber(b)], pi1, G)
    F2_one = lambda a, b: kl(ber(b), G)           # independent of p1
    m12 = mixed(F1, p1, p2)
    m21_one = mixed(F2_one, p1, p2)
    # (ii) mutual attention with its own prior row
    pi2 = np.array([0.3, 0.7])
    F2_mut = lambda a, b: F_reduced(ber(b), [ber(a)], pi2, G)
    m21_mut = mixed(F2_mut, p1, p2)
    print(f"    p=({p1:.3f},{p2:.3f}): one-way {m12:+.5f} vs {m21_one:+.5f};  mutual {m12:+.5f} vs {m21_mut:+.5f}")

# (b) ---------------------------------------------------------------------------------
rho = np.array([1.2, -0.8])                       # log odds of the two references (self sources)
pi = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]


def per_agent_force(l: np.ndarray) -> np.ndarray:
    r"""Own-block derivative dF_i/dp_i of the reduced per-agent free energy (self source = reference)."""
    p = 1 / (1 + np.exp(-l))
    out = np.zeros(2)
    for i in range(2):
        j = 1 - i
        Gi = ber(1 / (1 + np.exp(-rho[i])))
        Fi = lambda a: F_reduced(ber(a), [ber(p[j])], pi[i], Gi)
        h = 1e-6
        out[i] = (Fi(p[i] + h) - Fi(p[i] - h)) / (2 * h)
    return out


l = np.array([0.0, 0.0])
for _ in range(20000):                            # per-agent Fisher flow in log odds: dl_i/dt = -dF_i/dp_i
    l = l - 0.01 * per_agent_force(l)
p = 1 / (1 + np.exp(-l))
print(f"(b) per-agent stationary point: p = {np.round(p, 6)}, residual own-block force {np.abs(per_agent_force(l)).max():.2e}")
# rows at the stationary point and the source-role derivative of the other agent's objective
for j in range(2):
    i = 1 - j
    Gi = ber(1 / (1 + np.exp(-rho[i])))
    costs = np.array([kl(ber(p[i]), Gi), kl(ber(p[i]), ber(p[j]))])
    w = pi[i] * np.exp(-costs)
    beta_ij = w[1] / w.sum()
    Fi_of_pj = lambda b: F_reduced(ber(p[i]), [ber(b)], pi[i], Gi)
    h = 1e-6
    src = (Fi_of_pj(p[j] + h) - Fi_of_pj(p[j] - h)) / (2 * h)
    pred = beta_ij * (p[j] - p[i]) / (p[j] * (1 - p[j]))       # beta_ij * d/dp_j KL(q_i||q_j)
    print(f"    source-role derivative of F_{i+1} on p_{j+1}: {src:+.6f} (beta*(p_j-p_i)/(p_j(1-p_j)) = {pred:+.6f}); "
          f"zero only if p_i = p_j")
