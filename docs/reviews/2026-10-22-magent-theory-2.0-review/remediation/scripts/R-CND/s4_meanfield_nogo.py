r"""R-CND check 4: mean-field identity behind the no-go remark.

Finite alphabet, shared evaluator g, common channel target nu[M] depending on
the laws only through the additive moment M = sum_i E_{q_i} T. Unit Fisher
flow \dot q_i = -q_i (h_i - E_{q_i} h_i) with h_i = dE/dq_i (finite differences).
Checks that d/dt log(q_i/q_j) + 2 log(q_i/q_j) is constant over the alphabet,
and that the identity fails once targets depend on the receiver.
"""
import numpy as np

rng = np.random.default_rng(3)
m, N, d = 6, 4, 2
T = rng.normal(size=(m, d))
g = rng.dirichlet(np.ones(m))
w = rng.dirichlet(np.ones(m))
Q = rng.dirichlet(np.ones(m), size=N)


def target(M: np.ndarray, shift: np.ndarray) -> np.ndarray:
    lam = np.tanh(M) + 0.3 * M**2 + shift
    v = w * np.exp(-T @ lam)
    return v / v.sum()


def kl(a, b):
    return float(np.sum(a * np.log(a / b)))


def energy(Qf: np.ndarray, receiver_shift: bool) -> float:
    # unnormalized extension: normalize inside so the energy is defined off the simplex
    Qn = Qf / Qf.sum(axis=1, keepdims=True)
    M = (Qn @ T).sum(axis=0)
    E = 0.0
    for i in range(N):
        sh = 0.5 * i * np.ones(d) if receiver_shift else np.zeros(d)
        E += kl(Qn[i], g) + kl(Qn[i], target(M, sh))
    return E


for receiver_shift in (False, True):
    h = np.zeros_like(Q)
    eps = 1e-6
    for i in range(N):
        for z in range(m):
            Qp, Qm = Q.copy(), Q.copy()
            Qp[i, z] += eps
            Qm[i, z] -= eps
            h[i, z] = (energy(Qp, receiver_shift) - energy(Qm, receiver_shift)) / (2 * eps)
    Qdot = -Q * (h - (Q * h).sum(axis=1, keepdims=True))
    worst = 0.0
    for i in range(N):
        for j in range(N):
            r = Qdot[i] / Q[i] - Qdot[j] / Q[j] + 2 * (np.log(Q[i]) - np.log(Q[j]))
            worst = max(worst, np.ptp(r))
    print(f"receiver-dependent targets={receiver_shift}: max spread over z of "
          f"d/dt log(q_i/q_j) + 2 log(q_i/q_j) = {worst:.2e}")
