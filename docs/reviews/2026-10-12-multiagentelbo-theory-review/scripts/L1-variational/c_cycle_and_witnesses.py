r"""L1-variational check C: symbolic and exact checks of c08 (four-cycle, correlated metric, partial refresh),
c03 (access-restriction Chernoff floor), ca_measure witness (c), c05 random-link VFE, c04 appended energy."""
import numpy as np
import sympy as sp
from fractions import Fraction as Fr
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

T = sp.symbols('T', positive=True)
k = sp.symbols('k0:4'); m = sp.symbols('m0:4'); w = sp.symbols('w0:4')
x = list(k) + list(m) + list(w)
rb = [k[(i + 1) % 4] - k[i] - w[i] for i in range(4)]
rm = [m[(i + 1) % 4] - m[i] - w[i] for i in range(4)]
D = lambda i: (rb[i] ** 2 + rm[i] ** 2) / 2
V = 2 * D(0) + 2 * D(2)
U = V + T / 2 * sum(v ** 2 for v in x) + T / 2 * 2 * sum((k[i] - m[i]) ** 2 for i in range(4))
Hs = sp.hessian(U, x)
s2 = sp.sqrt(2)


def unit(var):
    return [1 if v == var else 0 for v in x]


basis = []
for i in (0, 2):
    a, b = unit(k[i + 1]), unit(k[i]); c_, d_ = unit(m[i + 1]), unit(m[i])
    km = [(p - q) / s2 for p, q in zip(a, b)]
    mmn = [(p - q) / s2 for p, q in zip(c_, d_)]
    kp = [(p + q) / s2 for p, q in zip(a, b)]
    mp = [(p + q) / s2 for p, q in zip(c_, d_)]
    basis += [km, mmn, unit(w[i]), kp, mp]
basis += [unit(w[1]), unit(w[3])]
P = sp.Matrix(basis)
Hn = sp.simplify(P * Hs * P.T)
blk3 = sp.Matrix([[4 + 3 * T, -2 * T, -2 * s2], [-2 * T, 4 + 3 * T, -2 * s2], [-2 * s2, -2 * s2, 4 + T]])
blk2 = T * sp.Matrix([[3, -2], [-2, 3]])
target = sp.diag(blk3, blk2, blk3, blk2, T, T)
print("[1] held-sheet Hessian blocks residual zero:", sp.simplify(Hn - target) == sp.zeros(12, 12),
      "; P orthonormal:", sp.simplify(P * P.T) == sp.eye(12))
print("    leading minors:", [sp.factor(blk3[:j, :j].det()) for j in (1, 2, 3)])
em, ew, z = sp.symbols('e_m e_w z', nonnegative=True)
A0 = blk3.subs(T, 0)
Mflow = sp.diag(1, em, ew) * A0
cp = sp.expand((z * sp.eye(3) - Mflow).det())
claim = sp.expand(z * (z ** 2 - 4 * (1 + em + ew) * z + 8 * (2 * em + ew + em * ew)))
print("    char poly residual:", sp.simplify(cp - claim), "; kernel:", list(A0.nullspace()[0].T))
qd = z ** 2 - 4 * (1 + em + ew) * z + 8 * (2 * em + ew + em * ew)
disc = sp.expand(sp.discriminant(qd, z))
print("    discriminant - 16[(1-em)^2+ew^2]:", sp.simplify(disc - 16 * ((1 - em) ** 2 + ew ** 2)))
A0n = np.array(A0.evalf(), float)


def residuals(e_m, e_w, t):
    Mf = np.diag([1.0, e_m, e_w]) @ A0n
    x0 = np.array([0.0, 0.0, 1.0])
    out = []
    for tt in t:
        xt = expm(-Mf * tt) @ x0
        out.append((np.sqrt(2) * xt[0] - xt[2], np.sqrt(2) * xt[1] - xt[2]))
    return np.array(out)


tg = np.linspace(0, 40, 4001)
viol = 0.0
for e_m in np.linspace(0, 1, 11):
    R = residuals(e_m, 1.0, tg)
    viol = max(viol, (np.abs(R[:, 1]) - np.exp(-(4 - 2 * np.sqrt(2)) * tg)).max())
print("[2] fast link: max(|r^m| - e^{-(4-2sqrt2)t}) over e_m grid:", viol)
viol = 0.0
for e_w in np.linspace(0.1, 1, 10):
    R = residuals(1.0, e_w, tg)
    viol = max(viol, np.abs(R[:, 1] + np.exp(-(4 + 4 * e_w) * tg)).max())
print("    fast models: max|r^m + e^{-(4+4e_w)t}|:", viol)
vb = vlo = vhi = -np.inf
for e in [0.5, 0.3, 0.1, 0.04, 0.01, 0.002]:
    tl = np.linspace(0, 3 / e, 3001)
    R = residuals(e, e, tl)
    vb = max(vb, (np.abs(R[:, 0]) - ((1 + e) * np.exp(-4 * tl) + e)).max())
    vlo = max(vlo, ((1 - 1.5 * e) * np.exp(-6 * e * tl) - 1.5 * e - np.abs(R[:, 1])).max())
    vhi = max(vhi, (np.abs(R[:, 1]) - (np.exp(-(6 * e - 2 * e * e) * tl) + 1.5 * e)).max())
print("    slow pair bounds worst margins (<=0 means the bound holds):", vb, vlo, vhi)
for e in [1 / 25, 1 / 100]:
    tl = np.linspace(0.25 * np.log(4 / e), 1 / (12 * e), 2001)
    R = residuals(e, e, tl)
    print("    e=%.4f: max|r^b|/e = %.3f (claim <=2), min|r^m| = %.4f (claim >=1/2)"
          % (e, np.abs(R[:, 0]).max() / e, np.abs(R[:, 1]).min()))
