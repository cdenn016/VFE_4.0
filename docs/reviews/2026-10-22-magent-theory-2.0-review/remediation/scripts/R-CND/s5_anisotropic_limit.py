r"""R-CND check 5: persistent geometry with anisotropic references (L08-CND-12).

Belief reference W_2(alpha_0, Gamma(a,chi)), Gamma = R_chi diag(e^a, e^{-a}) R_chi^T,
channel precision Lambda = Gamma + (kappa/N) M, model reference W_2(eta_0, diag(e^{at}, e^{-at})).
Eigenvalue law (gamma = eigenvalue of the reference):
  \dot p = -2p + 2 gamma + 2 kappa abar/p - kappa alpha_0 p/(gamma p + kappa abar).
(i) natural-gradient check of P' = -2P + Gamma + C at one unequal-precision point;
(ii) time-integrated block geometry versus the fixed-point limit.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln, digamma

N = 4


def lg2(a):
    return 0.5 * np.log(np.pi) + gammaln(a) + gammaln(a - 0.5)


def psi2(a):
    return digamma(a) + digamma(a - 0.5)


def kl(a, P, b, Q):
    return ((a - b) * psi2(a) - lg2(a) + lg2(b) + b * np.log(np.linalg.det(P) / np.linalg.det(Q))
            + a * (np.trace(Q @ np.linalg.inv(P)) - 2.0))


def unpack(t):
    return t[0], np.array([[t[1], t[2]], [t[2], t[3]]])


# (i) natural gradient check
Gam = np.array([[1.4, 0.3], [0.3, 0.8]])
a0, kap = 1.0, 1.0
TH = np.array([[1.0, 1.3, 0.2, 0.9], [1.2, 0.8, -0.1, 1.1], [1.8, 1.0, 0.3, 1.4], [2.1, 1.6, 0.0, 0.7]])


def E(TH):
    M = sum(unpack(t)[0] * np.linalg.inv(unpack(t)[1]) for t in TH)
    Lam = Gam + kap / N * M
    return sum(kl(*unpack(t), a0, Gam) + kl(*unpack(t), a0, Lam) for t in TH)


def fisher(th, h=1e-4):
    phi = lambda x: lg2(x[0]) - x[0] * np.log(np.linalg.det(unpack(x)[1]))
    G = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ei, ej = np.eye(4)[i] * h, np.eye(4)[j] * h
            G[i, j] = (phi(th + ei + ej) - phi(th + ei - ej) - phi(th - ei + ej) + phi(th - ei - ej)) / (4 * h * h)
    return G


M = sum(unpack(t)[0] * np.linalg.inv(unpack(t)[1]) for t in TH)
Lam = Gam + kap / N * M
C = Lam + kap / N * (M - N * a0 * np.linalg.inv(Lam))
res = 0.0
for i in range(N):
    g = np.zeros(4)
    for k in range(4):
        Tp, Tm = TH.copy(), TH.copy()
        Tp[i, k] += 1e-6
        Tm[i, k] -= 1e-6
        g[k] = (E(Tp) - E(Tm)) / 2e-6
    v = -np.linalg.solve(fisher(TH[i]), g)
    res = max(res, abs(v[0] + 2 * (TH[i, 0] - a0)),
              np.max(np.abs(unpack(v)[1] - (-2 * unpack(TH[i])[1] + Gam + C))))
print(f"(i) natural-gradient residual for alpha' and P' = -2P + Gamma + C: {res:.2e}")


# (ii) geometry, block {1,2}, alpha0 = eta0 = kappa = 1, a = at = 0.3
def f(p, ab, gam, kap, a0):
    return -2 * p + 2 * gam + 2 * kap * ab / p - kap * a0 * p / (gam * p + kap * ab)


def path(p0, gam, ab0, a0, kap, ts):
    ab = lambda t: a0 + np.exp(-2 * t) * (ab0 - a0)
    return solve_ivp(lambda t, y: [f(y[0], ab(t), gam, kap, a0)], (0, ts[-1]), [p0], t_eval=ts,
                     rtol=1e-12, atol=1e-14).y[0]


def block(a, at, ts, n=2, a0=1.0, e0=1.0, kap=1.0):
    al, et = np.array([1, 1.2, 1.8, 2.1]), np.array([0.8, 1.1, 1.4, 1.7])
    pp = path(np.exp(a), np.exp(a), al.mean(), a0, kap, ts)
    pm = path(np.exp(-a), np.exp(-a), al.mean(), a0, kap, ts)
    qp = path(np.exp(at), np.exp(at), et.mean(), e0, kap, ts)
    qm = path(np.exp(-at), np.exp(-at), et.mean(), e0, kap, ts)
    A = n * a0 + np.exp(-2 * ts) * (al[:2].sum() - n * a0)
    B = n * e0 + np.exp(-2 * ts) * (et[:2].sum() - n * e0)
    return A, pp, pm, B, qp, qm


ts = np.array([0.0, 1, 2, 4, 8, 16])
h = 1e-5
A, pp, pm, B, qp, qm = block(0.3, 0.3, ts)
Ap, ppp, pmp, *_ = block(0.3 + h, 0.3, ts)
Am, ppm, pmm, *_ = block(0.3 - h, 0.3, ts)
k1 = 2 * A * (pp - pm) ** 2 / (pp * pm)
k2 = 2 * B * (qp - qm) ** 2 / (qp * qm)
k1p = 2 * Ap * (ppp - pmp) ** 2 / (ppp * pmp)
k1m = 2 * Am * (ppm - pmm) ** 2 / (ppm * pmm)
dk1 = (k1p - k1m) / (2 * h)
gaa = A * (((ppp - ppm) / (2 * h) / pp) ** 2 + ((pmp - pmm) / (2 * h) / pm) ** 2)
F = -dk1 * k2 / (k1 + k2) ** 2
for k, t in enumerate(ts):
    print(f"(ii) t={t:4.0f}: g_aa={gaa[k]:.4f} h_chichi={k1[k]*k2[k]/(k1[k]+k2[k]):.4f} "
          f"A_chi={-k1[k]/(k1[k]+k2[k]):.4f} F={F[k]:.4f}")
