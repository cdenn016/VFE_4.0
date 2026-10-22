r"""X3-PIFB check 2: why the 2.0 peer potential is not a sample-level ELBO, and what its source force is.

(a) Mixed third derivative invariant. For ANY positive table P on {-1,1}^2 the mean-field ELBO
    F(x1,x2) = sum q1 q2 log(q1 q2 / P),  q_i(k) = (1+k x_i)/2,
    is (entropy of x1) + (entropy of x2) + multi-affine, so d_{x1} d_{x2}^2 F = 0.
    The 2.0 peer potential E = d(x1,x2) + d(x2,x1) (+ reference terms) has
    d_{x1} d_{x2}^2 E = -2 x2/(1-x2^2)^2, nonzero for x2 != 0.
(b) Unboundedness: for full-support P every coupling rule has KL(Q||P) <= -log min P,
    while E is unbounded on the open square (x1 -> 1, x2 -> -1).
(c) Aggregate (compensation) identity and the incoming force:
    sum_i w_i KL(q_i || nu) = sum_i w_i KL(q_i || qbar) + W KL(qbar || nu),  nu = q_j K,
    so the source-side force on q_j equals the gradient of W KL(qbar || q_j K).
(d) Gaussian comparison with the PIFB2 normalized linear-Gaussian record (A=B=1):
    KL(N(m_i,v_i)||N(m_j,v_j)) = [record cost with tau*Gamma := v_j] - H(q_i) - 1/2,
    and the sign of the source force on v_j differs between the two models.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

x1, x2 = sp.symbols("x1 x2", real=True)
P = sp.symbols("P_mm P_mp P_pm P_pp", positive=True)
Ptab = {(-1, -1): P[0], (-1, 1): P[1], (1, -1): P[2], (1, 1): P[3]}
q1 = {k: (1 + k * x1) / 2 for k in (-1, 1)}
q2 = {k: (1 + k * x2) / 2 for k in (-1, 1)}
F = sum(q1[a] * q2[b] * (sp.log(q1[a]) + sp.log(q2[b]) - sp.log(Ptab[(a, b)])) for a in (-1, 1) for b in (-1, 1))
dF = sp.simplify(sp.diff(F, x1, 1, x2, 2))
print("(a) d_x1 d_x2^2 of mean-field ELBO for arbitrary positive P:", dF)


def kl_bin(x: sp.Expr, z: sp.Expr) -> sp.Expr:
    return sum(((1 + k * x) / 2) * sp.log((1 + k * x) / (1 + k * z)) for k in (-1, 1))


E = kl_bin(x1, x2) + kl_bin(x2, x1)
dE = sp.simplify(sp.diff(E, x1, 1, x2, 2))
print("(a) d_x1 d_x2^2 of 2.0 mutual peer potential:", sp.factor(dE))
print("    value at (x1,x2)=(0.1,0.3):", float(dE.subs({x1: 0.1, x2: 0.3})))

# (b) boundedness
Pnum = np.array([0.1, 0.2, 0.3, 0.4])
print("(b) bound for any coupling with full-support P: -log min P =", -np.log(Pnum.min()))
Ef = sp.lambdify((x1, x2), E, "numpy")
for t in (0.9, 0.99, 0.999, 0.9999):
    print(f"    E at (x1,x2)=({t},{-t}) = {Ef(t, -t):.3f}")

# (c) aggregate identity with a random channel
rng = np.random.default_rng(1)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


A_in, A_out = 4, 3
K = rng.dirichlet(np.ones(A_out), size=A_in)  # K[v, w]
qj = rng.dirichlet(np.ones(A_in))
qs = [rng.dirichlet(np.ones(A_out)) for _ in range(4)]
w = rng.uniform(0.1, 1.0, size=4)
W = w.sum()
qbar = sum(wi * qi for wi, qi in zip(w, qs)) / W
nu = qj @ K
lhs = sum(wi * kl(qi, nu) for wi, qi in zip(w, qs))
rhs = sum(wi * kl(qi, qbar) for wi, qi in zip(w, qs)) + W * kl(qbar, nu)
print("(c) aggregate identity residual:", abs(lhs - rhs))
# gradient in q_j (ambient, before centering): d/dq_j(v) of sum_i w_i KL(q_i||q_j K) = -sum_i w_i sum_u q_i(u) K(v,u)/nu(u)
g1 = -sum(wi * (K @ (qi / nu)) for wi, qi in zip(w, qs))
g2 = -W * (K @ (qbar / nu))
print("    incoming-force identity residual:", float(np.max(np.abs(g1 - g2))))

# (d) Gaussian comparison
mi, mj, vi, vj, s = sp.symbols("m_i m_j v_i v_j s", positive=True)
KLg = sp.Rational(1, 2) * (vi / vj + (mi - mj) ** 2 / vj - 1 + sp.log(vj / vi))
rec = ((mi - mj) ** 2 + vi + vj) / (2 * s) + sp.log(2 * sp.pi * s) / 2  # E_Q D/tau - log c with tau*Gamma = s
H_i = sp.log(2 * sp.pi * sp.E * vi) / 2
diff = sp.simplify(KLg - (rec.subs(s, vj) - H_i - sp.Rational(1, 2)))
print("(d) KL - [record(s=v_j) - H(q_i) - 1/2] =", diff)
print("    source force on v_j, normalized record model (fixed s):", sp.diff(rec, vj))
print("    source force on v_j, 2.0 peer KL:", sp.simplify(sp.diff(KLg, vj)),
      " -> zero at v_j = v_i + (m_i-m_j)^2")
