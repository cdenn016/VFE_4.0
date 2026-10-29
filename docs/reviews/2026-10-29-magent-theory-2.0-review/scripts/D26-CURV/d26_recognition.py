r"""D26-CURV check 6: exact dictionary marginal (it3:induced) versus the factorized global recognition of
chapter 18 (Q = product over agents of Q_i(y_i, omega_in(i))).

Directed scalar ring 0 -> 1 -> 2 -> 0, each agent owns its single incoming dictionary. The exact posterior
precision of omega is the Schur complement Lambda_omega; the factorized recognition has block precision
J_ii, so its omega covariance is block diagonal across owners. Compare the variance of the loop sum.
"""
from d26_common import *

K = 1; nV = 3; ring = [(0, 1), (1, 2), (2, 0)]
codes = [np.eye(1)] * 3; cs = [np.ones(1)] * 3
D, V = coboundary(K, nV, ring, codes, cs)
for p in (1.0, 0.1, 0.01):
    s2 = 1.0; R = np.eye(3)
    L = np.hstack([D, -V]); J = blockdiag([p * np.eye(3), np.eye(3) / s2]) + L.T @ L
    Sig_exact = np.linalg.inv(J)[3:, 3:]
    Lam = np.eye(3) / s2 + V.T @ np.linalg.solve(R + D @ D.T / p, V)
    # factorized recognition: agent i block (y_i, omega_e) with e the edge into i
    owner = {1: 0, 2: 1, 0: 2}                          # receiver -> its incoming edge index
    Sig_fact = np.zeros((3, 3))
    for i, e in owner.items():
        idx = [i, 3 + e]
        Sig_fact[e, e] = np.linalg.inv(J[np.ix_(idx, idx)])[1, 1]
    one = np.ones(3)
    print("p =", p, ": |Sigma_exact - Lambda^-1| =", np.abs(Sig_exact - np.linalg.inv(Lam)).max(),
          " exact Var(loop sum) =", round(float(one @ Sig_exact @ one), 6),
          " factorized Var(loop sum) =", round(float(one @ Sig_fact @ one), 6),
          " exact cross-owner covariance =", round(float(Sig_exact[0, 1]), 6))
