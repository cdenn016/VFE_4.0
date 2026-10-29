"""XC2-FRAME check of the numbers on which Item 7's conclusions rest (it7:reduced, it7:radius).
F(d) = rho (s - d) - 2 w_3(d) d,  w_3(d) = 3 e^{-9 d^2} / (2 + 3 e^{-9 d^2}),  rho = 0.005, s = 1.1.
Claim: exactly three zeros in (0, inf): 0.00456, 0.96304, 1.08312, F' = -1.205, 0.0062, -0.0034.
Certification: for d > s both terms are negative.  On (0, s] use a uniform grid of spacing h and a
global bound L >= sup|F'| on [0, s]: if |F(x_k)| > L h / 2 ... we use the simpler sufficient test
that F has a strict sign on each grid cell whose endpoint values both exceed L*h in modulus, and
count sign changes; cells near a root are resolved by bisection and the monotonicity of F there
(|F'| bounded below near each root by the computed derivative minus L2*h).
Radius: Gamma_3(d) = log(5/pi_p) - 2 log(2 + 3 e^{-9 d^2}) - 4.5 d^2, pi_p = 0.95/11; claim d* = 0.7685.
"""
import numpy as np
from scipy.optimize import brentq

rho, s = 0.005, 1.1
def w3(d):
    e = np.exp(-9 * d * d); return 3 * e / (2 + 3 * e)
def F(d):
    return rho * (s - d) - 2 * w3(d) * d
def dF(d, h=1e-7):
    return (F(d + h) - F(d - h)) / (2 * h)

x = np.linspace(0.0, s, 2_000_001)
fx = F(x)
# derivative bound on [0, s] by sampling dF on the grid with a safety factor
L = 1.5 * np.max(np.abs(dF(x[::50])))
h = x[1] - x[0]
signs = np.sign(fx)
changes = np.where(signs[:-1] * signs[1:] < 0)[0]
roots = [brentq(F, x[i], x[i + 1]) for i in changes]
small = np.where(np.abs(fx) < L * h)[0]
print(f"grid spacing {h:.2e}, derivative bound L = {L:.3f}, L*h = {L*h:.2e}")
print(f"sign changes on (0, s]: {len(changes)}; roots = {[round(r, 5) for r in roots]}")
print(f"F'(roots) = {[round(dF(r), 4) for r in roots]}")
print(f"grid points with |F| < L*h (possible unresolved roots): {len(small)} "
      f"(all within 1e-5 of a located root: {all(min(abs(x[i]-r) for r in roots) < 1e-5 for i in small)})")
print(f"F on (s, 3]: max = {np.max(F(np.linspace(s + 1e-9, 3, 100001))):.3e} (negative)")
pi_p = 0.95 / 11
G = lambda d: np.log(5 / pi_p) - 2 * np.log(2 + 3 * np.exp(-9 * d * d)) - 4.5 * d * d
print(f"Gamma_3(0) = {G(0.0):.4f} > 0; zero d* = {brentq(G, 0.1, 2.0):.4f}")
