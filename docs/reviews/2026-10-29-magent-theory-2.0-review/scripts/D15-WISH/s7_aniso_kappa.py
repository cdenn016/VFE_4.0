r"""D15-WISH s7: limiting geometry of form:anisotropic-reference as a function of kappa.

At the fixed points (abar = alpha0, ebar = eta0) of
  dp/dt = -2p + 2 gamma + 2 kappa alpha0/p - kappa alpha0 p/(gamma p + kappa alpha0),
gamma = e^{+-a} for the belief sector (context dependent) and gamma = e^{+-at} for the model
sector (fixed anisotropic reference), implicit differentiation gives d_a p* = -(f_gamma/f_p) d_a gamma.
Block of two agents: A = 2 alpha0, B = 2 eta0.
"""
import numpy as np
from scipy.optimize import brentq


def f(p, g, kap, a0):
    return -2 * p + 2 * g + 2 * kap * a0 / p - kap * a0 * p / (g * p + kap * a0)


def fp(p, g, kap, a0):
    return -2 - 2 * kap * a0 / p**2 - kap**2 * a0 * a0 / (g * p + kap * a0) ** 2


def fg(p, g, kap, a0):
    return 2 + kap * a0 * p**2 / (g * p + kap * a0) ** 2


def limit(a, at, kap, a0=1.0, e0=1.0):
    ps, dps = [], []
    for sg in (1, -1):
        g = np.exp(sg * a)
        p = brentq(lambda x: f(x, g, kap, a0), 1e-8, 100)
        ps.append(p)
        dps.append(-fg(p, g, kap, a0) / fp(p, g, kap, a0) * sg * g)
    qs = [brentq(lambda x: f(x, np.exp(sg * at), kap, e0), 1e-8, 100) for sg in (1, -1)]
    A, B = 2 * a0, 2 * e0
    pp, pm = ps
    dpp, dpm = dps
    I1 = 2 * A * (pp - pm) ** 2 / (pp * pm)
    I2 = 2 * B * (qs[0] - qs[1]) ** 2 / (qs[0] * qs[1])
    dI1 = 2 * A * (2 * (pp - pm) * (dpp - dpm) / (pp * pm) - (pp - pm) ** 2 * (dpp * pm + pp * dpm) / (pp * pm) ** 2)
    gaa = A * ((dpp / pp) ** 2 + (dpm / pm) ** 2)
    return gaa, I1 * I2 / (I1 + I2), -dI1 * I2 / (I1 + I2) ** 2


for kap in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]:
    gaa, hcc, F = limit(0.3, 0.3, kap)
    print(f"kappa={kap:5.1f}: limit g_aa={gaa:.4f} h_chichi={hcc:.4f} F={F:+.4f}")
