r"""R-CND check 1: kappa-independence of the Wishart shape law.

Natural gradient \dot\theta_i = -G_i^{-1}\nabla_i E on four 2x2 Wishart agents
W_2(alpha_i, P_i) (belief sector), with the full energy of form:energy
(channel derivative included), for several couplings kappa and for the null
flow (peer costs deleted, channel evaluated with peer laws frozen).
Checks alpha_i' = -2(alpha_i - alpha_0) and P_i' = -2P_i + I + C_i.
"""
import numpy as np
from scipy.special import gammaln, digamma

N = 4


def lg2(a: float) -> float:
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a: float) -> float:
    return digamma(a) + digamma(a - 0.5)


def kl(a: float, P: np.ndarray, b: float, Q: np.ndarray) -> float:
    r"""KL(W_2(a,P) || W_2(b,Q)) = (a-b)psi_2(a) - log Gamma_2(a) + log Gamma_2(b)
    + b log det(P Q^{-1}) + a(Tr(Q P^{-1}) - 2)."""
    return ((a - b) * psi2(a) - lg2(a) + lg2(b)
            + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2.0))


def unpack(th: np.ndarray):
    a, p11, p12, p22 = th
    return a, np.array([[p11, p12], [p12, p22]])


def energy_full(TH: np.ndarray, a0: float, kap: float) -> float:
    M = sum(unpack(t)[0] * np.linalg.inv(unpack(t)[1]) for t in TH)
    Lam = np.eye(2) + kap / N * M
    return sum(kl(*unpack(t), a0, np.eye(2)) + kl(*unpack(t), a0, Lam) for t in TH)


def energy_null(TH: np.ndarray, TH0: np.ndarray, a0: float, kap: float) -> float:
    E = 0.0
    for i, t in enumerate(TH):
        a, P = unpack(t)
        M = a * np.linalg.inv(P) + sum(unpack(TH0[j])[0] * np.linalg.inv(unpack(TH0[j])[1])
                                       for j in range(N) if j != i)
        Lam = np.eye(2) + kap / N * M
        E += kl(a, P, a0, np.eye(2)) + kl(a, P, a0, Lam)
    return E


def fisher(th: np.ndarray, h: float = 1e-4) -> np.ndarray:
    def phi(x):
        a, P = unpack(x)
        return lg2(a) - a * np.log(np.linalg.det(P))
    G = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            e_i, e_j = np.eye(4)[i] * h, np.eye(4)[j] * h
            G[i, j] = (phi(th + e_i + e_j) - phi(th + e_i - e_j)
                       - phi(th - e_i + e_j) + phi(th - e_i - e_j)) / (4 * h * h)
    return G


def natgrad(TH: np.ndarray, Efun, h: float = 1e-6) -> np.ndarray:
    V = np.zeros_like(TH)
    for i in range(N):
        g = np.zeros(4)
        for k in range(4):
            Tp, Tm = TH.copy(), TH.copy()
            Tp[i, k] += h
            Tm[i, k] -= h
            g[k] = (Efun(Tp) - Efun(Tm)) / (2 * h)
        V[i] = -np.linalg.solve(fisher(TH[i]), g)
    return V


rng = np.random.default_rng(1)
a0 = 1.0
TH = np.array([[1.0, 1.3, 0.2, 0.9], [1.2, 0.8, -0.1, 1.1],
               [1.8, 1.0, 0.3, 1.4], [2.1, 1.6, 0.0, 0.7]])
print("unequal precisions; alpha' + 2(alpha - alpha0) residual, P' residual")
for kap in [0.0, 1.0, 5.0]:
    V = natgrad(TH, lambda T: energy_full(T, a0, kap))
    M = sum(unpack(t)[0] * np.linalg.inv(unpack(t)[1]) for t in TH)
    Lam = np.eye(2) + kap / N * M
    C = np.eye(2) + 2 * kap / N * M - kap * a0 * np.linalg.inv(Lam)
    ra = np.max(np.abs(V[:, 0] + 2 * (TH[:, 0] - a0)))
    rP = max(np.max(np.abs(unpack(V[i])[1] - (-2 * unpack(TH[i])[1] + np.eye(2) + C)))
             for i in range(N))
    print(f"full  kappa={kap}: shape residual {ra:.2e}, precision residual {rP:.2e}, "
          f"alpha' = {np.round(V[:, 0], 6)}")
    Vn = natgrad(TH, lambda T: energy_null(T, TH, a0, kap))
    ran = np.max(np.abs(Vn[:, 0] + 2 * (TH[:, 0] - a0)))
    print(f"null  kappa={kap}: shape residual {ran:.2e}, P1' null vs full differ by "
          f"{np.max(np.abs(Vn[0,1:]-V[0,1:])):.3f}")
