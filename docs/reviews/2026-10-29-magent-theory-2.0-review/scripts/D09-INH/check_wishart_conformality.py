r"""D09-INH extension check for inh:type-closure-open (first step: horizontal conformality of
matrix addition for the Wishart Fisher metrics).

Family (chapters/11_conditional_formation.tex, form:wishart): W_2(nu, P) on 2x2 SPD matrices,
log-density nu log det P - log Gamma_2(nu) + (nu - 3/2) log det X - tr(P X).
Fisher metric in (nu, P):  g_W = psi_2'(nu) dnu^2 - 2 dnu tr(P^{-1} dP) + nu tr((P^{-1} dP)^2),
with psi_2'(nu) = psi'(nu) + psi'(nu - 1/2).
Fine space: the common-precision family (alpha_1, alpha_2, P) with metric g_W(alpha_1,P) + g_W(alpha_2,P)
(shared dP).  Parent (form:parent): R(alpha_1, alpha_2, P) = (A, P), A = alpha_1 + alpha_2, metric g_W(A, P).

1. Verify the Fisher metric formula by Monte Carlo-free exact differentiation: compare with the
   negative expected Hessian computed from E[log det X] = psi_2(nu) - log det P (exact identity)
   and E[X] = nu P^{-1}, at a random point (symbolic in sympy for the nu-nu entry; numeric otherwise).
2. Horizontal lift norm versus parent norm.  Conformality with factor d^2 forces d = 1 (take dA = 0)
   and then psi_2'(A) = 1 / (1/psi_2'(alpha_1) + 1/psi_2'(alpha_2)).  Evaluate the defect.
"""
import numpy as np
from scipy.special import polygamma, digamma

def psi2p(nu):
    return polygamma(1, nu) + polygamma(1, nu - 0.5)

rng = np.random.default_rng(3)

# ---- 1. metric check by finite differences of the exact expected log-likelihood ----
def expected_loglik(theta, theta0):
    """E_{theta0}[log density at theta]; theta = (nu, P)."""
    nu, P = theta; nu0, P0 = theta0
    ElogdetX = digamma(nu0) + digamma(nu0 - 0.5) - np.log(np.linalg.det(P0))   # E log det X
    EX = nu0 * np.linalg.inv(P0)
    from scipy.special import gammaln
    logG2 = 0.5 * np.log(np.pi) + gammaln(nu) + gammaln(nu - 0.5)
    return nu * np.log(np.linalg.det(P)) - logG2 + (nu - 1.5) * ElogdetX - np.trace(P @ EX)

nu0 = 2.7
A_ = rng.normal(size=(2, 2)); P0 = A_ @ A_.T + np.eye(2)
dnu = 0.3; dP = rng.normal(size=(2, 2)); dP = dP + dP.T
h = 1e-4
f = lambda s: expected_loglik((nu0 + s * dnu, P0 + s * dP), (nu0, P0))
fd = -(f(h) - 2 * f(0) + f(-h)) / h**2          # = Fisher quadratic form at the true parameter
Pi = np.linalg.inv(P0)
formula = psi2p(nu0) * dnu**2 - 2 * dnu * np.trace(Pi @ dP) + nu0 * np.trace(Pi @ dP @ Pi @ dP)
print(f"1. Fisher quadratic form: finite difference {fd:.8f}, formula {formula:.8f}, rel. residual {abs(fd - formula) / abs(formula):.2e}")

# ---- 2. conformality defect ----
print("2. alpha_1, alpha_2 | psi_2'(A) | harmonic combination | ratio (1 means conformal)")
for a1, a2 in ((0.8, 0.8), (2.0, 2.0), (2.0, 7.0), (10.0, 10.0), (50.0, 80.0)):
    lhs = psi2p(a1 + a2)
    rhs = 1.0 / (1.0 / psi2p(a1) + 1.0 / psi2p(a2))
    print(f"   {a1:5.1f} {a2:5.1f} | {lhs:.6f} | {rhs:.6f} | {lhs / rhs:.6f}")

# direct check at one point: random horizontal vectors, ratio of norms
a1, a2 = 2.0, 7.0
A_ = rng.normal(size=(2, 2)); P = A_ @ A_.T + np.eye(2); Pi = np.linalg.inv(P)
def g_fine(u):
    u1, u2, U = u
    return (psi2p(a1) * u1**2 + psi2p(a2) * u2**2 - 2 * (u1 + u2) * np.trace(Pi @ U)
            + (a1 + a2) * np.trace(Pi @ U @ Pi @ U))
def g_par(dA, U):
    A = a1 + a2
    return psi2p(A) * dA**2 - 2 * dA * np.trace(Pi @ U) + A * np.trace(Pi @ U @ Pi @ U)
ratios = []
for k in range(5):
    s = rng.normal(); U = rng.normal(size=(2, 2)); U = U + U.T
    if k == 0:
        U = np.zeros((2, 2))
    if k == 1:
        s = 0.0
    u1, u2 = s / psi2p(a1), s / psi2p(a2)        # horizontal: psi2'(a1) u1 = psi2'(a2) u2
    ratios.append(g_par(u1 + u2, U) / g_fine((u1, u2, U)))
print("   ratios g_c(DR u, DR u) / g(u, u) over random horizontal u at (2, 7):", np.round(ratios, 6))
