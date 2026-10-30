"""Exact checks for the linear-Gaussian three-objective theorem on the unit-variance mean slice."""
from __future__ import annotations

import sympy as sp

N = 3
a = sp.symbols("a1:%d" % (N + 1))
mu = sp.symbols("m1:%d" % (N + 1))
b = {(i, j): sp.Symbol(f"b{i+1}{j+1}", positive=True) for i in range(N) for j in range(N) if i != j}
n = sp.Symbol("n", positive=True)
tau = sp.Symbol("tau", positive=True)

# Reciprocal potential on the slice q_i = N(mu_i, 1), identity channels, fixed rows.
E = sum((mu[i] - a[i]) ** 2 / 2 for i in range(N)) + sum(b[i, j] * (mu[i] - mu[j]) ** 2 / 2 for (i, j) in b)
W = sp.Matrix(N, N, lambda i, j: 0 if i == j else b[i, j] + b[j, i])
LW = sp.diag(*[sum(W[i, k] for k in range(N)) for i in range(N)]) - W
grad = sp.Matrix([sp.diff(E, m) for m in mu])
A = sp.eye(N) + LW
assert sp.simplify(grad - (A * sp.Matrix(mu) - sp.Matrix(a))) == sp.zeros(N, 1)
print("(i) reciprocal stationarity is (I+L_W)mu=a")

# Global model: X_i ~ N(a_i, 1/n), record o_ij ~ N(X_j - X_i, 1/(n b_ij)) registered at 0.
# -log p(X, o=0) = n E(X) + const; Gaussian recognition N(mu, tau I).
X = sp.symbols("x1:%d" % (N + 1))
neglogp = n * E.subs(dict(zip(mu, X)))
H = sp.hessian(neglogp, X)
assert sp.simplify(H - n * A) == sp.zeros(N, N)
# E_Q[-log p] for quadratic -log p equals value at mean plus tau/2 trace(Hessian).
FQ = neglogp.subs(dict(zip(X, mu))) + tau / 2 * H.trace() - sp.Rational(N, 2) * sp.log(2 * sp.pi * sp.E * tau)
assert sp.simplify(FQ / n - E - (tau / 2 * A.trace() - sp.Rational(N, 2) * sp.log(2 * sp.pi * sp.E * tau) / n)) == 0
print("(ii) T F(N(mu,tau I)) = E(mu) + const(tau,n)")

# Mean-field Gaussian factors of the exact posterior: block minimizer means solve the same linear system.
P = n * A
mf = [sp.solve(sp.diff(FQ.subs(tau, 1), mu[i]), mu[i])[0] for i in range(N)]
print("(ii) mean-field block-update means:", [sp.simplify(m) for m in mf][:1], "...")

# Per-agent, frozen-source instance: force (mu_i - a_i) + sum_j b_ij (mu_i - mu_j).
Bm = sp.Matrix(N, N, lambda i, j: 0 if i == j else b[i, j])
Lout = sp.diag(*[sum(Bm[i, k] for k in range(N)) for i in range(N)]) - Bm
M = sp.eye(N) + Lout
force = sp.Matrix([(mu[i] - a[i]) + sum(b[i, j] * (mu[i] - mu[j]) for j in range(N) if j != i) for i in range(N)])
assert sp.simplify(force - (M * sp.Matrix(mu) - sp.Matrix(a))) == sp.zeros(N, 1)
print("(iii) per-agent stationarity is (I+L_out)mu=a")

# Reversible rows: b_ij = lam_ij / p_i with lam symmetric. Global model with priors N(a_i, 1/p_i), precisions lam_ij.
p = sp.symbols("p1:%d" % (N + 1), positive=True)
lam = {}
for i in range(N):
    for j in range(i + 1, N):
        lam[i, j] = lam[j, i] = sp.Symbol(f"l{i+1}{j+1}", positive=True)
brev = {(i, j): lam[i, j] / p[i] for (i, j) in b}
Mrev = M.subs({b[k]: brev[k] for k in b})
L0 = sp.diag(*p)
Llam = sp.Matrix(N, N, lambda i, j: -lam[i, j] if i != j else sum(lam[i, k] for k in range(N) if k != i))
assert sp.simplify(L0 * Mrev - (L0 + Llam)) == sp.zeros(N, N)
print("(iv) reversible rows: Lambda0 (I+L_out) = Lambda0 + L_lambda, so posterior means agree")

# Non-reversible rows: 3-cycle with only forward edges; no symmetric lam and positive p solve L0*M = L0 + Llam.
Mcyc = M.subs({b[0, 1]: 1, b[1, 2]: 1, b[2, 0]: 1, b[1, 0]: 0, b[2, 1]: 0, b[0, 2]: 0})
S = L0 * Mcyc
print("(iv) forward 3-cycle: L0*M symmetric?", sp.simplify(S - S.T) == sp.zeros(N, N), "(must be False)")

# Two-agent gaps with symmetric unit rows.
a1, a2 = sp.symbols("alpha beta_")
E2 = (mu[0] - a1) ** 2 / 2 + (mu[1] - a2) ** 2 / 2 + 2 * (mu[0] - mu[1]) ** 2 / 2
s = sp.solve([sp.diff(E2, mu[0]), sp.diff(E2, mu[1])], [mu[0], mu[1]])
print("reciprocal gap:", sp.simplify(s[mu[0]] - s[mu[1]]))
s = sp.solve([(mu[0] - a1) + (mu[0] - mu[1]), (mu[1] - a2) + (mu[1] - mu[0])], [mu[0], mu[1]])
print("per-agent gap:", sp.simplify(s[mu[0]] - s[mu[1]]))

# Off the slice: two agents, free Gaussian variances, symmetric unit rows, evaluators N(a_i,1).
v1, v2 = sp.symbols("v1 v2", positive=True)
def kl(m1, s1, m2, s2):
    return (s1 / s2 + (m1 - m2) ** 2 / s2 - 1 + sp.log(s2 / s1)) / 2
Ev = kl(mu[0], v1, a1, 1) + kl(mu[1], v2, a2, 1) + kl(mu[0], v1, mu[1], v2) + kl(mu[1], v2, mu[0], v1)
num = {a1: 1, a2: 0}
sol = sp.nsolve([sp.diff(Ev, x).subs(num) for x in (mu[0], mu[1], v1, v2)], [mu[0], mu[1], v1, v2], [0.6, 0.4, 0.5, 0.5])
print("free-variance reciprocal stationary point (a-b=1):", [float(x) for x in sol], "gap", float(sol[0] - sol[1]), "vs slice gap 0.2")
