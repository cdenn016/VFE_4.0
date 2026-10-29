# XC3-SCOPE E4 check: full-law-space Fisher distance between LN(m1,1) and LN(m2,1)
# d = 2 arccos BC, BC = int sqrt(p q). Compare with |dm| (in-family Fisher distance).
import numpy as np
import sympy as sp

# Bhattacharyya coefficient of N(0,1), N(x,1) by quadrature (lognormal: same by pushforward)
z = np.linspace(-30, 30, 600001); dz = z[1] - z[0]
for x in [0.5, 2.0, 5.0]:
    p = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi); q = np.exp(-(z - x)**2 / 2) / np.sqrt(2 * np.pi)
    bc = np.sum(np.sqrt(p * q)) * dz
    print("x", x, "| BC quadrature", "%.12f" % bc, "| exp(-x^2/8)", "%.12f" % np.exp(-x**2 / 8))

X = sp.symbols("x", positive=True)
f = 2 * sp.acos(sp.exp(-X**2 / 8))
print("series of f at 0:", sp.series(f, X, 0, 5))
f2 = sp.lambdify(X, sp.diff(f, X, 2), "numpy")
xs = np.linspace(1e-3, 20, 20000)
v = f2(xs)
print("max f'' on (0,20]:", "%.3e" % np.max(v), "| min f'':", "%.4f" % np.min(v), "at x =", "%.3f" % xs[np.argmin(v)])
print("f(20) =", float(f.subs(X, 20)), "| pi =", float(sp.pi))
