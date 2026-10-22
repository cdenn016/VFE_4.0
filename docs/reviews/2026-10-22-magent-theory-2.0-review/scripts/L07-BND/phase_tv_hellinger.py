r"""Phase-rotation continuity: TV and squared Hellinger along alpha -> alpha + delta
for one child q = m (1 + a sigma), sigma = sign cos(theta - alpha), on a fine
circle grid (exact for delta a multiple of the grid step).  Also the Fisher-Rao
(spherical Hellinger) length of a discretized phase path, which grows like sqrt(n).
"""
import numpy as np

Mgrid = 72000
th = (np.arange(Mgrid) + 0.5) * 2 * np.pi / Mgrid
a = 0.6
def q(alpha):
    s = np.sign(np.cos(th - alpha))
    return 1 + a * s               # density against Haar (base trivial, m = 1)
q0 = q(0.0)
print(" delta      TV        TV_formula   H^2=int(sqrt p - sqrt q)^2   H^2/delta")
for k in (36, 72, 144, 288, 576):
    delta = k * 2 * np.pi / Mgrid
    qd = q(delta)
    tv = 0.5 * np.mean(np.abs(q0 - qd))
    x, y = 1 + a, 1 - a
    tv_formula = delta / (2 * np.pi) * abs(x - y)
    h2 = np.mean((np.sqrt(q0) - np.sqrt(qd)) ** 2)
    print(f"{delta:.5f}  {tv:.6e}  {tv_formula:.6e}  {h2:.6e}  {h2/delta:.6f}  (pred {(np.sqrt(x)-np.sqrt(y))**2/np.pi:.6f})")

print("Fisher-Rao length of the phase path alpha in [0, pi/2] with n pieces:")
for n in (10, 40, 160, 640):
    L = 0.0
    alphas = np.linspace(0, np.pi / 2, n + 1)
    prev = q(alphas[0])
    for al in alphas[1:]:
        cur = q(al)
        bc = np.mean(np.sqrt(prev * cur))
        L += 2 * np.arccos(min(1.0, bc))
        prev = cur
    print(f"n={n:4d}  length={L:.4f}  length/sqrt(n)={L/np.sqrt(n):.4f}")
