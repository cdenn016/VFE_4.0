r"""D23-LEARN check 4: Item_1_communication_links.tex:512 identifies the per-agent picture with the
receiver-factorized KL projection of the joint posterior and concludes that ownership leaves the
dictionary means unchanged.  Scalar two-cycle test (K = 1, codes 1, no contents):
edge 1->0 carries w1 with a grounded term g/2 (w1 - what)^2; edge 0->1 carries w2, unread;
anchor 1/(2 s2) w^2 on each edge; loop term lam/2 (w1 + w2)^2.
  exact posterior mean                     : precision [[1/s2+g+lam, lam],[lam, 1/s2+lam]]
  receiver-factorized global (CAVI)        : same means, covariances 1/diagonal
  plug-in per-agent (chapter 18 it2:agent-energy with Omega = lam W): same means
  law-level per-agent (Omega_i = (1/lam + V_peer)^{-1}, chapter 18 eq. it2:widening): fixed point
"""
import numpy as np

s2, lam, g, what = 1.0, 4.0, 10.0, 1.0
P = np.array([[1 / s2 + g + lam, lam], [lam, 1 / s2 + lam]])
eta = np.array([g * what, 0.0])
exact = np.linalg.solve(P, eta)
print("exact posterior mean          :", exact, " exact marginal variances:", np.diag(np.linalg.inv(P)))

def per_agent(law_level: bool):
    a1 = a2 = 0.0
    V1 = V2 = s2
    for _ in range(2000):
        O0 = 1 / (1 / lam + V2) if law_level else lam
        V1 = 1 / (1 / s2 + g + O0)
        a1 = (g * what - O0 * a2) * V1
        O1 = 1 / (1 / lam + V1) if law_level else lam
        V2 = 1 / (1 / s2 + O1)
        a2 = -O1 * a1 * V2
    return np.array([a1, a2]), np.array([V1, V2])

for lab, ll in [("plug-in per-agent fixed point ", False), ("law-level per-agent fixed pt  ", True)]:
    m, V = per_agent(ll)
    print(f"{lab}: means {m}, variances {V}, |means - exact| = {np.abs(m - exact).max():.3e}")
