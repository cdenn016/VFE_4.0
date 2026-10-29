r"""D06-PRED s9: cmem:response for a non-Gaussian prior and a nonlinear result kernel.

Prior: X ~ Laplace(0, 1); result Y | X = x ~ N(sin(2x) + x^2/4, 0.3); record O = X + eps, eps ~ N(0, C), C = 0.5.
Checks at several records xi, by adaptive quadrature over x (Y integrated analytically):
  d/dxi E[Y|xi] (central finite difference) vs Cov(Y,X|xi)/C
  E[score^2] with score = (X - E[X|xi])/C vs 1/C + d^2/dxi^2 log Z(xi) (finite difference)
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats

C = 0.5
prior = lambda x: 0.5 * np.exp(-abs(x))
g = lambda x: np.sin(2 * x) + x ** 2 / 4          # E[Y | X = x]


def moments(xi: float) -> tuple[float, float, float, float, float]:
    lik = lambda x: stats.norm.pdf(xi, x, np.sqrt(C))
    pts = [xi, 0.0]
    Z = integrate.quad(lambda x: prior(x) * lik(x), -40, 40, points=pts, limit=400, epsabs=1e-14)[0]
    EX = integrate.quad(lambda x: x * prior(x) * lik(x), -40, 40, points=pts, limit=400, epsabs=1e-14)[0] / Z
    EY = integrate.quad(lambda x: g(x) * prior(x) * lik(x), -40, 40, points=pts, limit=400, epsabs=1e-14)[0] / Z
    EXY = integrate.quad(lambda x: x * g(x) * prior(x) * lik(x), -40, 40, points=pts, limit=400, epsabs=1e-14)[0] / Z
    EXX = integrate.quad(lambda x: x * x * prior(x) * lik(x), -40, 40, points=pts, limit=400, epsabs=1e-14)[0] / Z
    return Z, EX, EY, EXY - EX * EY, EXX - EX ** 2


h = 1e-4
for xi in [-1.3, 0.2, 0.9, 2.5]:
    Z0, EX, EY, CYX, CXX = moments(xi)
    Zp, _, EYp, _, _ = moments(xi + h)
    Zm, _, EYm, _, _ = moments(xi - h)
    dEY = (EYp - EYm) / (2 * h)
    d2logZ = (np.log(Zp) - 2 * np.log(Z0) + np.log(Zm)) / h ** 2
    print(f"xi={xi:5.2f}: dE[Y]/dxi = {dEY:.8f}, Cov(Y,X)/C = {CYX / C:.8f};  "
          f"1/C + (log Z)'' = {1 / C + d2logZ:.6f}, Cov(X)/C^2 = {CXX / C ** 2:.6f}")
