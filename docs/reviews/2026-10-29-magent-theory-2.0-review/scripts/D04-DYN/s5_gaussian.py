r"""D04-DYN s5: Gaussian restricted flow, non-invariance, and the full velocity of a Gaussian source.

(a) eq:dyn-gaussian-flow (mu' = -Sigma grad_mu E, Sigma' = -2 Sigma grad_Sigma E Sigma) for E = KL(q || N(m, S)) on R^2
    against the exact Fisher-Rao solution (natural parameters interpolate); the factor 2 is checked by comparison.
(b) prop:dyn-gaussian-noninvariance: the first variation of a Gaussian source is not a quadratic polynomial.
(c) the full Fisher velocity of a Gaussian source contains beta q_k/q_i; its Fisher norm is beta^2 chi^2(q_k||q_i),
    infinite iff 2 Sigma_k^{-1} - Sigma_i^{-1} is not positive definite (sympy, 1D).
(d) the restricted velocity equals the score-pairing projection of the full velocity, also when the latter is not in L^2.
(e) prop:dyn-compensation identity (5-point alphabet, common fixed channel) and the Gaussian source variance force.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad

rng = np.random.default_rng(2)

print("== (a) eq:dyn-gaussian-flow vs the exact Fisher-Rao flow of KL(. || N(m, S)) ==")
A = rng.normal(size=(2, 2))
S = A @ A.T + 0.5 * np.eye(2)
m = rng.normal(size=2)
B = rng.normal(size=(2, 2))
Sig0 = B @ B.T + 0.3 * np.eye(2)
mu0 = rng.normal(size=2)
Si = np.linalg.inv(S)


def rhs(t, y, factor):
    mu = y[:2]
    Sig = y[2:].reshape(2, 2)
    gmu = Si @ (mu - m)
    gS = 0.5 * (Si - np.linalg.inv(Sig))        # symmetric gradient: dE = tr(gS dSigma)
    dmu = -Sig @ gmu
    dS = -factor * Sig @ gS @ Sig
    return np.concatenate([dmu, dS.ravel()])


# check the symmetric-gradient convention by finite differences
def Egauss(mu, Sig):
    return 0.5 * (np.trace(Si @ Sig) + (m - mu) @ Si @ (m - mu) - 2 + np.log(np.linalg.det(S) / np.linalg.det(Sig)))


dSym = rng.normal(size=(2, 2))
dSym = dSym + dSym.T
t = 1e-6
fd = (Egauss(mu0, Sig0 + t * dSym) - Egauss(mu0, Sig0 - t * dSym)) / (2 * t)
print(f"  symmetric-gradient check: tr(gS dSigma) - FD = {np.trace(0.5 * (Si - np.linalg.inv(Sig0)) @ dSym) - fd:.2e}")
L0 = np.linalg.inv(Sig0)
eta0 = L0 @ mu0
for factor in (2.0, 1.0):
    sol = solve_ivp(rhs, (0, 3), np.concatenate([mu0, Sig0.ravel()]), args=(factor,), rtol=1e-12, atol=1e-13, t_eval=[0.5, 1.5, 3.0])
    err = 0.0
    for k, tt in enumerate(sol.t):
        Lt = np.exp(-tt) * L0 + (1 - np.exp(-tt)) * Si
        et = np.exp(-tt) * eta0 + (1 - np.exp(-tt)) * Si @ m
        St = np.linalg.inv(Lt)
        err = max(err, np.max(np.abs(sol.y[2:, k].reshape(2, 2) - St)), np.max(np.abs(sol.y[:2, k] - St @ et)))
    print(f"  covariance factor {factor:.0f}: max deviation from the exact Fisher-Rao solution = {err:.2e}")

print("== (b)-(c) a Gaussian source with one receiver ==")
x, mu, v, mk, vk = sp.symbols("x mu v m_k v_k", real=True)
vpos, vkpos = sp.symbols("v_i v_k", positive=True)
qi = sp.exp(-x**2 / (2 * vpos)) / sp.sqrt(2 * sp.pi * vpos)
qk = sp.exp(-x**2 / (2 * vkpos)) / sp.sqrt(2 * sp.pi * vkpos)
integrand = sp.simplify(qk**2 / qi)
expo = sp.simplify(sp.log(integrand).expand(force=True).coeff(x, 2))
print(f"  log(q_k^2/q_i) has x^2 coefficient {sp.simplify(expo)}: integrable iff v_k < 2 v_i")
dpos = sp.symbols("d", positive=True)
chi = sp.simplify(sp.integrate(sp.simplify(integrand.subs(vkpos, 2 * vpos - dpos)), (x, -sp.oo, sp.oo))) - 1
print(f"  chi^2(q_k||q_i) with v_k = 2 v_i - d, d > 0 (zero means): {sp.simplify(chi)}")
for vv in (1.5, 1.9, 1.99):
    val = float(chi.subs({vpos: 1, dpos: 2 - vv}))
    num = quad(lambda z: np.exp(-z**2 / vv + z**2 / 2) / (2 * np.pi * vv) * np.sqrt(2 * np.pi), -np.inf, np.inf)[0] - 1
    print(f"    v_i = 1, v_k = {vv}: chi^2 = {val:.4f} (quadrature {num:.4f})")
# non-polynomiality of the first variation phi = log q_i - log g - beta q_k/q_i on a grid
beta = 0.5
g_m, g_v = 0.3, 2.0
mi, vi, mkk, vkk = 0.0, 1.0, 0.7, 1.5
z = np.linspace(-4, 4, 801)
logq = -0.5 * np.log(2 * np.pi * vi) - (z - mi)**2 / (2 * vi)
logg = -0.5 * np.log(2 * np.pi * g_v) - (z - g_m)**2 / (2 * g_v)
logk = -0.5 * np.log(2 * np.pi * vkk) - (z - mkk)**2 / (2 * vkk)
phi = logq - logg - beta * np.exp(logk - logq)
w = np.exp(logq)
V = np.vstack([np.ones_like(z), z, z**2]).T
coef = np.linalg.lstsq(V * np.sqrt(w)[:, None], phi * np.sqrt(w), rcond=None)[0]
resid = np.sqrt(np.sum(w * (phi - V @ coef)**2) / np.sum(w))
print(f"  weighted L2 distance of phi from quadratics on [-4,4]: {resid:.4f} (zero for a Gaussian-preserving velocity)")

print("== (d) restricted Gaussian velocity of a source = score-pairing projection of the full velocity ==")
g_m, g_v, beta = 0.3, 2.0, 0.5
for (m_k, v_k) in ((0.7, 1.5), (0.7, 3.0)):
    mu_i, v_i = 0.1, 1.0
    dEdmu = (mu_i - g_m) / g_v - beta * (m_k - mu_i) / v_i
    dEdv = 0.5 * (1 / g_v - 1 / v_i) + 0.5 * beta * (-v_k / v_i**2 - (m_k - mu_i)**2 / v_i**2 + 1 / v_i)
    restricted = np.array([-v_i * dEdmu, -2 * v_i**2 * dEdv])
    lqi = lambda z: -(z - mu_i)**2 / (2 * v_i) - 0.5 * np.log(2 * np.pi * v_i)
    lg = lambda z: -(z - g_m)**2 / (2 * g_v) - 0.5 * np.log(2 * np.pi * g_v)
    qi_f = lambda z: np.exp(lqi(z))
    qk_f = lambda z: np.exp(-(z - m_k)**2 / (2 * v_k)) / np.sqrt(2 * np.pi * v_k)
    s_mu = lambda z: (z - mu_i) / v_i
    s_v = lambda z: ((z - mu_i)**2 - v_i) / (2 * v_i**2)
    # E_q[s phi] with phi = log(q_i/g) + 1 - beta q_k/q_i ; the source part is written as an integral against q_k
    pair = []
    for s in (s_mu, s_v):
        loc = quad(lambda z: qi_f(z) * s(z) * (lqi(z) - lg(z) + 1), -40, 40, limit=200)[0]
        src = -beta * quad(lambda z: qk_f(z) * s(z), -60, 60, limit=200)[0]
        pair.append(loc + src)
    Ginv = np.diag([v_i, 2 * v_i**2])
    proj = -Ginv @ np.array(pair)
    chi2 = np.inf if v_k >= 2 * v_i else v_i / np.sqrt(v_k * (2 * v_i - v_k)) * np.exp((m_k - mu_i)**2 / (2 * v_i - v_k)) - 1
    print(f"  receiver N({m_k}, {v_k}), source N({mu_i}, {v_i}): restricted {np.round(restricted, 10)}, projection {np.round(proj, 10)}; chi^2(q_k||q_i) = {chi2}")

print("== (e) prop:dyn-compensation and the Gaussian source variance force ==")
L = 5
K = rng.dirichlet(np.ones(L), size=L)
qj = rng.dirichlet(np.ones(L))
qs = [rng.dirichlet(np.ones(L)) for _ in range(4)]
bw = rng.uniform(0.1, 1.0, size=4)
W = bw.sum()
qbar = sum(bw[a] * qs[a] for a in range(4)) / W
klf = lambda a, b: float(np.sum(a * np.log(a / b)))
lhs = sum(bw[a] * klf(qs[a], qj @ K) for a in range(4))
rhs = sum(bw[a] * klf(qs[a], qbar) for a in range(4)) + W * klf(qbar, qj @ K)
print(f"  compensation identity residual = {abs(lhs - rhs):.2e} (magnitude {lhs:.3f})")
mi_, vi_, mj_, vj_ = sp.symbols("m_i v_i m_j v_j", positive=True)
KLg = sp.Rational(1, 2) * (vi_ / vj_ + (mi_ - mj_)**2 / vj_ - 1 + sp.log(vj_ / vi_))
force = sp.simplify(sp.diff(KLg, vj_) - (vj_ - vi_ - (mi_ - mj_)**2) / (2 * vj_**2))
print(f"  d/dv_j KL(N(m_i,v_i)||N(m_j,v_j)) - (v_j - v_i - (m_i-m_j)^2)/(2 v_j^2) = {force}")
