r"""Low-cost extension of prop:dyn-no-global-vfe(b) to fixed non-identity channels.

Claim checked: for a fixed Markov matrix K (rows = source states) the mixed third derivative of
b * KL(q_i || q_j K) once in a mass-preserving direction w of q_i and twice in a mass-preserving direction
v of q_j equals  b * sum_y w(y) (vK)(y)^2 / (q_j K)(y)^2,  which is nonzero for suitable w, v whenever
(vK)^2/(q_jK)^2 is not constant in y for some mass-preserving v, in particular whenever K is not constant
in its input.  The free energy of a fixed model with product recognition has this mixed derivative
identically zero (affine in each factor), so the proof of (b) goes through for such channels.
Finite differences at small size; a numerical agreement supports the identity, it does not prove it.
"""
import numpy as np

rng = np.random.default_rng(7)
n = 4
K = rng.random((n, n)) + 0.1
K /= K.sum(axis=1, keepdims=True)
qi = rng.random(n) + 0.2; qi /= qi.sum()
qj = rng.random(n) + 0.2; qj /= qj.sum()
w = rng.standard_normal(n); w -= w.mean()
v = rng.standard_normal(n); v -= v.mean()
b = 0.7


def f(x, y):
    nu = y @ K
    return b * np.sum(x * np.log(x / nu))


h = 1e-3
# third mixed derivative d_x[w] d_y[v] d_y[v] by central differences
def d3(x, y):
    tot = 0.0
    for sx, cx in ((1, 0.5), (-1, -0.5)):
        for sy, cy in ((1, 1.0), (0, -2.0), (-1, 1.0)):
            tot += cx * cy * f(x + sx * h * w, y + sy * h * v)
    return tot / (h ** 3)


fd = d3(qi, qj)
nu = qj @ K
formula = b * np.sum(w * (v @ K) ** 2 / nu ** 2)
out = [f"finite-difference third derivative {fd:.6f}; formula {formula:.6f}; relative error {abs(fd - formula) / abs(formula):.2e}"]
# constant-in-input channel: every row equal, so vK = 0 and the derivative vanishes
Kc = np.tile(K[0], (n, 1))
formula_c = b * np.sum(w * (v @ Kc) ** 2 / (qj @ Kc) ** 2)
out.append(f"constant-in-input channel: formula gives {formula_c:.2e} (vanishes, so (b)'s argument does not apply)")
print("\n".join(out))
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
