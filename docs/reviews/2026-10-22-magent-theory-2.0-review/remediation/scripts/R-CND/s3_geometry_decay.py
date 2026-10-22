r"""R-CND check 3: decay of the inherited contextual geometry of form:geometry.

Eigenvalue flow \dot p = f(p, abar(t)) of form:eigenvalue, closed-form inertias
k_1, k_2, radial coefficient g_aa, connection A = -k_1/(k_1+k_2) and curvature
F = -(d_a k_1) k_2/(k_1+k_2)^2 (finite differences in a). Reports the decay
rate of g_aa and k_1 and compares it with 2|f_p(p*, alpha_0)|.
"""
import numpy as np
from scipy.integrate import solve_ivp


def f(p, ab, kap, a0):
    return -2 * p + 2 + 2 * kap * ab / p - kap * a0 * p / (p + kap * ab)


def fp(p, ab, kap, a0):
    return -2 - 2 * kap * ab / p**2 - kap**2 * a0 * ab / (p + kap * ab) ** 2


def eig_path(p0, ab0, a0, kap, ts):
    abar = lambda t: a0 + np.exp(-2 * t) * (ab0 - a0)
    sol = solve_ivp(lambda t, y: [f(y[0], abar(t), kap, a0), fp(y[0], abar(t), kap, a0) * y[1]],
                    (0, ts[-1]), [p0, 1.0], t_eval=ts, rtol=1e-12, atol=1e-14)
    return sol.y  # p(t), dPhi/dp0


def geometry(a, b, a0, e0, kap, ts, A0, B0, ab0, eb0, n=2):
    pp, dpp = eig_path(np.exp(a), ab0, a0, kap, ts)
    pm, dpm = eig_path(np.exp(-a), ab0, a0, kap, ts)
    qp, _ = eig_path(np.exp(b), eb0, e0, kap, ts)
    qm, _ = eig_path(np.exp(-b), eb0, e0, kap, ts)
    A = n * a0 + np.exp(-2 * ts) * (A0 - n * a0)
    B = n * e0 + np.exp(-2 * ts) * (B0 - n * e0)
    k1 = 2 * A * (pp - pm) ** 2 / (pp * pm)
    k2 = 2 * B * (qp - qm) ** 2 / (qp * qm)
    gaa = A * ((dpp * np.exp(a) / pp) ** 2 + (dpm * np.exp(-a) / pm) ** 2)
    return k1, k2, gaa


ts = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
alphas, etas = np.array([1, 1.2, 1.8, 2.1]), np.array([0.8, 1.1, 1.4, 1.7])
for (a0, e0, kap) in [(1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 0.0)]:
    a, b, h = 0.3, 0.3, 1e-5
    args = (a0, e0, kap, ts, alphas[:2].sum(), etas[:2].sum(), alphas.mean(), etas.mean())
    k1, k2, gaa = geometry(a, b, *args)
    k1p, _, _ = geometry(a + h, b, *args)
    k1m, _, _ = geometry(a - h, b, *args)
    dk1 = (k1p - k1m) / (2 * h)
    F = -dk1 * k2 / (k1 + k2) ** 2
    hcc = k1 * k2 / (k1 + k2)
    # fixed point at abar = alpha0
    from scipy.optimize import brentq
    pstar = brentq(lambda p: f(p, a0, kap, a0), 1e-6, 50)
    lam = abs(fp(pstar, a0, kap, a0))
    rate = -np.log(gaa[4] / gaa[3]) / (ts[4] - ts[3])
    print(f"alpha0={a0} eta0={e0} kappa={kap}: p*={pstar:.6f}, 2|f_p(p*)|={2*lam:.6f}, "
          f"fitted g_aa rate (t=4..6) {rate:.6f}")
    for k in range(len(ts)):
        print(f"   t={ts[k]:.0f}  g_aa={gaa[k]:.3e}  h_chichi={hcc[k]:.3e}  A_chi={-k1[k]/(k1[k]+k2[k]):.4f}  F={F[k]:.4f}")
print("12 - 4 sqrt 2 =", 12 - 4 * np.sqrt(2))
