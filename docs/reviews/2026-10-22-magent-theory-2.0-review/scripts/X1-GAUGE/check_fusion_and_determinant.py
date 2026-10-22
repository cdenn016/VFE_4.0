"""X1-GAUGE: (1) ch6 fus:links / fus:distance: the law-dependent scalar link is a coboundary
and the linked cost equals the transport-free cost; (2) app determinant: eta_det(H,H) = -det H,
congruence scaling, signature; (3) circle completion gauge cancellation."""
import numpy as np
import sympy as sp

# (1) fusion links
rng = np.random.default_rng(5)
mu = rng.normal(size=4); chi = 0.37; Ufr = np.exp(rng.normal(size=4))
KL_LN = lambda a, b: 0.5 * (a - b) ** 2          # KL(LN(a,1) || LN(b,1))
res_cost, res_cob = 0.0, 0.0
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        L = chi * (mu[i] - mu[j])
        d_link = np.sqrt(2 * KL_LN(mu[i], mu[j] + L)) / (1 - chi)
        d_none = np.sqrt(2 * KL_LN(mu[i], mu[j]))
        res_cost = max(res_cost, abs(d_link - d_none))
        V = Ufr * np.exp(chi * mu)                 # law-dressed vertex frames
        Theta = Ufr[i] * np.exp(L) / Ufr[j]
        res_cob = max(res_cob, abs(Theta - V[i] / V[j]))
print("fusion: |d_link - d_identity| max =", res_cost, "; coboundary |Theta_ij - V_i V_j^-1| max =", res_cob)

# (2) determinant form
t, x, y, z = sp.symbols('t x y z', real=True)
H = sp.Matrix([[t + z, x - sp.I * y], [x + sp.I * y, t - z]])
eta = lambda A, B: sp.Rational(1, 2) * ((A * B).trace() - A.trace() * B.trace())
print("eta(H,H) + det H =", sp.simplify(eta(H, H) + H.det()))
print("eta(H,H) =", sp.expand(eta(H, H)))
g = sp.Matrix(2, 2, sp.symbols('g0:4'))
gd = g.conjugate().T
Hp = g * H * gd
print("det(g H g^dag) - |det g|^2 det H =", sp.simplify(sp.expand(Hp.det() - (g.det() * gd.det()) * H.det())))
# (3) Im Tr(dg g^-1) = d arg det g for a one-parameter g(s)
s = sp.symbols('s', real=True)
gs = sp.Matrix([[sp.exp(sp.I * s) * (1 + s), s], [0, 2 + sp.I * s]])
lhs = sp.im((gs.diff(s) * gs.inv()).trace())
rhs = sp.diff(sp.arg(gs.det()), s)
print("Im Tr(g' g^-1) - (arg det g)' at s=0.3:", sp.N((lhs - rhs).subs(s, 0.3)))
