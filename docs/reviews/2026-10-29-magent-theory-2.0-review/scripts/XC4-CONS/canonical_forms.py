r"""Check that variant forms of canonical equations across chapters are algebraically the same.

1. Row-gap identity: ch03 eq:dyn-row-gap and ch18 it2:row-gap,
   \sum_j b_j D_j + T KL(b||pi) = -T log Z + T KL(b||b*),  b* = pi e^{-D/T}/Z.
2. Log-odds row law: ch15 ctx:c:eq:logrow and companion comp:ctx:log-odds follow from the Fisher row
   flow \dot b_j = -b_j (h_j - sum_k b_k h_k) with h_j = D_j + T(log(b_j/pi_j) + 1):
   d/dt log(b_j/b_l) = -(D_j - D_l) - T(log(b_j/b_l) - log(pi_j/pi_l)).
3. Envelope derivative: d(-T log Z)/dD_j = b*_j (ch03 eq:dyn-envelope, ch18 dE_T = dF_0 + sum b* dC).
4. Operating-map form of the tangent record model: kappa_i(omega) = omega c_i, so
   T(y_i - y_j - kappa_i omega) (ch18, ch19, Item 3, companion c18) equals T(y_i - y_j - omega c_i)
   (companion c19); adjoint kappa_i^T v = v c_i^T and kappa_i kappa_i^T = |c_i|^2 I (ch19:163).
Single-threaded, a few milliseconds.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(29102026)
out = []

# 1. row-gap identity, numeric over random instances
worst = 0.0
for _ in range(200):
    n = rng.integers(2, 7)
    pi = rng.random(n) + 0.05
    pi /= pi.sum()
    D = rng.random(n) * 5
    T = rng.random() * 3 + 0.05
    b = rng.random(n) + 0.01
    b /= b.sum()
    Z = np.sum(pi * np.exp(-D / T))
    bstar = pi * np.exp(-D / T) / Z
    lhs = np.sum(b * D) + T * np.sum(b * np.log(b / pi))
    rhs = -T * np.log(Z) + T * np.sum(b * np.log(b / bstar))
    worst = max(worst, abs(lhs - rhs))
out.append(f"1. row-gap identity: max |lhs-rhs| over 200 random rows = {worst:.2e}")

# 2. log-odds law from the Fisher row flow, symbolic for three entries
b1, b2, b3, D1, D2, D3, T = sp.symbols("b1 b2 b3 D1 D2 D3 T", positive=True)
p1, p2, p3 = sp.symbols("p1 p2 p3", positive=True)
bs = [b1, b2, b3]
Ds = [D1, D2, D3]
ps = [p1, p2, p3]
h = [Ds[k] + T * (sp.log(bs[k] / ps[k]) + 1) for k in range(3)]
hbar = sum(bs[k] * h[k] for k in range(3))
bdot = [-bs[k] * (h[k] - hbar) for k in range(3)]
# d/dt log(b_j/b_l) = bdot_j/b_j - bdot_l/b_l with j=1, l=3
lhs = sp.simplify(bdot[0] / b1 - bdot[2] / b3)
rhs = -(D1 - D3) - T * (sp.log(b1 / b3) - sp.log(p1 / p3))
out.append(f"2. log-odds law residual (symbolic): {sp.simplify(sp.expand_log(lhs - rhs, force=True))}")

# 3. envelope derivative
Dj = sp.symbols("d1:4", real=True)
Zs = sum(ps[k] * sp.exp(-Dj[k] / T) for k in range(3))
red = -T * sp.log(Zs)
res3 = [sp.simplify(sp.diff(red, Dj[k]) - ps[k] * sp.exp(-Dj[k] / T) / Zs) for k in range(3)]
out.append(f"3. envelope derivative residuals: {res3}")

# 4. operating map
K = 3
c = rng.standard_normal(K)
om = rng.standard_normal((K, K))
kappa = lambda w: w @ c  # kappa_i(omega) = omega c_i
v = rng.standard_normal(K)
# adjoint for the Frobenius inner product: <kappa w, v> = <w, kappa^T v>
w = rng.standard_normal((K, K))
adj = np.outer(v, c)
r_adj = abs(kappa(w) @ v - np.sum(w * adj))
# kappa kappa^T v = (v c^T) c = |c|^2 v
r_kk = np.max(np.abs(adj @ c - (c @ c) * v))
out.append(f"4. operating map: adjoint residual {r_adj:.2e}, kappa kappa^T - |c|^2 I residual {r_kk:.2e}")

txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
