"""Checks for the chapter 3 free-energy amendments (finite alphabets).

(1) Per-agent mixture-source model: P_i(m,k,j,l) = pi^s_l rho_l(m) pi^q_j kappa_j(m,k) l(o|k)
    with rho_self = r, rho_l = s_l K^s, kappa_self(m,.) = G(m,.), kappa_j = q_j K^q.
    Claim: KL(s x q x beta x gamma || P_i(.|o)) - log P_i(o)
         = sum_l gamma_l KL(s||rho_l) + KL(gamma||pi^s)
           + sum_j beta_j E_s KL(q||kappa_j(m,.)) + KL(beta||pi^q) + E_q[-log l(o|k)].
(2) Global mean-field: F(Q) = F_i(Q_i;Q_-i) - sum_{j!=i} H(Q_j), so the Q_i-minimizer of F_i
    (exp E log P) lowers F.
(3) Mixed derivative d_{q_i} d^2_{q_j} of KL(q_i||q_j) is nonzero (mean-field ELBO: zero).
"""
import numpy as np
rng = np.random.default_rng(0)
def simplex(*shape):
    x = rng.random(shape) + 0.1
    return x / x.sum(axis=-1, keepdims=True)
def kl(a, b):
    return float(np.sum(a * (np.log(a) - np.log(b))))

nK, nM, nJ, nL, nO = 3, 2, 3, 2, 2
q, s = simplex(nK), simplex(nM)
beta, gamma = simplex(nJ), simplex(nL)
piq, pis = simplex(nJ), simplex(nL)
G = simplex(nM, nK)                     # own evaluator G(m, k)
r = simplex(nM)                          # own model reference
nu = simplex(nJ - 1, nK)                 # peer channelled beliefs q_j K
rho_peer = simplex(nL - 1, nM)           # peer channelled model laws s_l K^s
lik = simplex(nK, nO); o = 1             # record kernel l(o|k)

kappa = np.zeros((nJ, nM, nK)); kappa[0] = G
for j in range(1, nJ): kappa[j] = nu[j - 1][None, :]
rho = np.vstack([r, rho_peer])
# joint P(m,k,j,l) * l(o|k)
P = (pis[None, None, None, :] * rho.T[:, None, None, :]
     * piq[None, None, :, None] * np.transpose(kappa, (1, 2, 0))[:, :, :, None]
     * lik[None, :, o][:, :, None, None])            # shape m,k,j,l
Po = P.sum()
Q = s[:, None, None, None] * q[None, :, None, None] * beta[None, None, :, None] * gamma[None, None, None, :]
F = kl(Q.ravel(), (P / Po).ravel()) - np.log(Po)
rhs = (sum(gamma[l] * kl(s, rho[l]) for l in range(nL)) + kl(gamma, pis)
       + sum(beta[j] * sum(s[m] * kl(q, kappa[j, m]) for m in range(nM)) for j in range(nJ))
       + kl(beta, piq) + float(-(q * np.log(lik[:, o])).sum()))
print("(1) per-agent mixture-source identity residual:", abs(F - rhs))

# (2) global mean-field coordinate step
nA = 3
Pg = rng.random((nA, nA, nA)) + 0.05; Pg /= Pg.sum()
Qs = [simplex(nA) for _ in range(3)]
def Fglob(Qs):
    Qj = np.einsum('a,b,c->abc', *Qs)
    return float(np.sum(Qj * (np.log(Qj) - np.log(Pg))))
f0 = Fglob(Qs)
Elog = np.einsum('abc,b,c->a', np.log(Pg), Qs[1], Qs[2])
Qs[0] = np.exp(Elog) / np.exp(Elog).sum()
print("(2) global F before/after agent-0 local step:", f0, Fglob(Qs))

# (3) mixed derivative of KL(qi||qj) on Bernoulli tables, d/dx_i d^2/dx_j^2
import sympy as sp
xi, xj = sp.symbols('x_i x_j')
qi = [(1 + xi) / 2, (1 - xi) / 2]; qj = [(1 + xj) / 2, (1 - xj) / 2]
E = sum(qi[a] * sp.log(qi[a] / qj[a]) for a in range(2))
mixed = sp.simplify(sp.diff(E, xi, 1, xj, 2))
print("(3) d_xi d^2_xj KL =", mixed, "value at (0.1,0.3):", float(mixed.subs({xi: 0.1, xj: 0.3})))
