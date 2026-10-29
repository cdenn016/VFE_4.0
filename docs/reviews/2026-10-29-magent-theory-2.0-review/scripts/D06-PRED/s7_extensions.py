r"""D06-PRED s7: checks behind the improvements and extensions proposed in the D06-PRED report.

  (a) Fisher bound on the signed response: D I^{-1} D^T <= Cov(Y|xi) (Cauchy-Schwarz for the posterior
      score), with D = Cov(Y,X|xi) C^{-1}, I = C^{-1} Cov(X|xi) C^{-1}; numeric on a 2D two-component bank
      and the companion scalar example (2 <= 9/4).
  (b) Free-energy (exclusive) product carrier for the label/result posterior Pi(a,dy) = w_a N(m_a,P_a):
      coordinate minimization of KL(w' x q || Pi) gives a Gaussian q with precision sum_a w'_a P_a^{-1};
      compare with the moment carrier N(mbar, S) in both divergences.
  (c) Joint-law Fisher flow of the local free energy on a finite table: d/dt log zeta = -(phi - E phi),
      phi = log(zeta/zeta^o); closed form zeta_t propto zeta_0^{e^{-t}} (zeta^o)^{1-e^{-t}}; from the prior
      reference r G this is the tempered posterior with likelihood power beta(t) = 1 - e^{-t}; for a Gaussian
      record the power is the posterior with record covariance C/beta.
  (d) Gaussian predictive criterion with trivial side information (2026-10-22 X4-PRIOR-13): for jointly
      Gaussian (X,Y) and U = A X of rank r, I(Y;X|U) >= -1/2 sum_{k>r} log(1-rho_k^2), attained by the
      top-r canonical directions.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, linalg, stats

rng = np.random.default_rng(2029)

# (a) Fisher bound on the response, 2D two-component bank
pis = np.array([1 / 3, 2 / 3])
L = [np.array([[2, .5], [.5, 1]]), np.array([[1, -1 / 3], [-1 / 3, 1.5]])]
Qm = [np.array([[.5, .2], [.2, 1]]), np.array([[1, 0], [0, .25]])]
F = [np.array([[-2, 1], [.5, 3]]), np.array([[1, 2], [.5, 1]])]
C = np.array([[1, .25], [.25, .5]])
nu = [np.array([0, 1.]), np.array([1.5, -1])]
om = [np.array([1, -1.]), np.array([0, 2.])]
Ci = np.linalg.inv(C)
worst = np.inf
for _ in range(200):
    xi = rng.normal(0, 2, 2)
    Z = np.array([stats.multivariate_normal.pdf(xi, nu[a], L[a] + C) for a in range(2)])
    w = pis * Z / (pis * Z).sum()
    mX = [nu[a] + L[a] @ np.linalg.solve(L[a] + C, xi - nu[a]) for a in range(2)]
    PX = [np.linalg.inv(np.linalg.inv(L[a]) + Ci) for a in range(2)]
    mY = [om[a] + F[a] @ (mX[a] - nu[a]) for a in range(2)]
    mbX = sum(w[a] * mX[a] for a in range(2)); mbY = sum(w[a] * mY[a] for a in range(2))
    CXX = sum(w[a] * (PX[a] + np.outer(mX[a] - mbX, mX[a] - mbX)) for a in range(2))
    CYX = sum(w[a] * (F[a] @ PX[a] + np.outer(mY[a] - mbY, mX[a] - mbX)) for a in range(2))
    CYY = sum(w[a] * (Qm[a] + F[a] @ PX[a] @ F[a].T + np.outer(mY[a] - mbY, mY[a] - mbY)) for a in range(2))
    D = CYX @ Ci
    I = Ci @ CXX @ Ci
    gap = CYY - D @ np.linalg.solve(I, D.T)
    worst = min(worst, np.linalg.eigvalsh((gap + gap.T) / 2).min())
print("(a) min eigenvalue of Cov(Y|xi) - D I^-1 D^T over 200 records (>= 0):", worst)
print("(a) companion scalar example: D^2/I =", (-1.0) ** 2 / 0.5, "<= Var(Y|xi) =", 9 / 4, "; gap = Q =", 9 / 4 - 2)

# (b) exclusive product carrier vs moment carrier, 1D label/result posterior
wv = np.array([0.5, 0.3, 0.2]); mv = np.array([-1.0, 0.7, 2.5]); Pv = np.array([0.6, 1.3, 0.4])
mbar = float(wv @ mv); S = float(wv @ (Pv + (mv - mbar) ** 2))


def kl_prod_to_pi(wp, m, s2):
    """KL(w' x N(m,s2) || Pi) with Pi(a,y) = w_a N(y; m_a, P_a), closed form."""
    tot = float(np.sum(wp * np.log(wp / wv)))
    for a in range(3):
        tot += wp[a] * 0.5 * (s2 / Pv[a] + (m - mv[a]) ** 2 / Pv[a] - 1 + np.log(Pv[a] / s2))
    return tot


def kl_pi_to_prod(wp, m, s2):
    """KL(Pi || w' x N(m,s2)), closed form."""
    tot = float(np.sum(wv * np.log(wv / wp)))
    for a in range(3):
        tot += wv[a] * 0.5 * (Pv[a] / s2 + (mv[a] - m) ** 2 / s2 - 1 + np.log(s2 / Pv[a]))
    return tot


wp = wv.copy()
for _ in range(500):
    prec = float(np.sum(wp / Pv)); s2 = 1 / prec; m = s2 * float(np.sum(wp * mv / Pv))
    logits = np.log(wv) - np.array([0.5 * (s2 / Pv[a] + (m - mv[a]) ** 2 / Pv[a] - 1 + np.log(Pv[a] / s2)) for a in range(3)])
    wp = np.exp(logits - logits.max()); wp /= wp.sum()
print(f"(b) exclusive carrier: w' = {wp.round(4)}, mean {m:.4f}, var {s2:.4f};  moment carrier: w = {wv}, mean {mbar:.4f}, var {S:.4f}")
print(f"    KL(w'xq_MF||Pi) = {kl_prod_to_pi(wp, m, s2):.5f} vs KL(w x q_G||Pi) = {kl_prod_to_pi(wv, mbar, S):.5f}")
print(f"    KL(Pi||w x q_G) = {kl_pi_to_prod(wv, mbar, S):.5f} vs KL(Pi||w'x q_MF) = {kl_pi_to_prod(wp, m, s2):.5f}")
print(f"    at w' = w the exclusive variance (sum w_a/P_a)^-1 = {1 / np.sum(wv / Pv):.4f} <= sum w_a P_a = {wv @ Pv:.4f} <= S = {S:.4f}")

# (c) joint-law Fisher flow on a finite table
nM, nX, nY, nO = 2, 3, 2, 3
r = rng.dirichlet(np.ones(nM)); mu = rng.dirichlet(np.ones(nX), nM)
K = rng.dirichlet(np.ones(nY), (nM, nX)); Lrec = rng.dirichlet(np.ones(nO), nX)
o = 1
prior = r[:, None, None] * mu[:, :, None] * K
post = prior * Lrec[None, :, o, None]
post /= post.sum()


def rhs(t, y):
    z = np.exp(y); z /= z.sum()
    phi = np.log(z) - np.log(post.ravel())
    return -(phi - np.sum(z * phi))


sol = integrate.solve_ivp(rhs, (0, 3.0), np.log(prior.ravel()), rtol=1e-11, atol=1e-12, dense_output=True)
errs = []
for t in [0.5, 1.0, 2.0, 3.0]:
    zt = np.exp(sol.sol(t)); zt /= zt.sum()
    closed = prior.ravel() ** np.exp(-t) * post.ravel() ** (1 - np.exp(-t)); closed /= closed.sum()
    beta = 1 - np.exp(-t)
    tempered = (prior * Lrec[None, :, o, None] ** beta).ravel(); tempered /= tempered.sum()
    errs.append((t, np.abs(zt - closed).max(), np.abs(zt - tempered).max()))
print("(c) Fisher flow vs closed form and vs tempered posterior (t, err_closed, err_tempered):",
      [(t, f"{a:.1e}", f"{b:.1e}") for t, a, b in errs])
# Gaussian record: N(xi;u,C)^beta propto N(xi;u,C/beta) (constant in u)
uu = np.linspace(-3, 3, 7); xi0, C0, beta = 0.4, 0.7, 0.35
ratio = stats.norm.pdf(xi0, uu, np.sqrt(C0)) ** beta / stats.norm.pdf(xi0, uu, np.sqrt(C0 / beta))
print("(c) N(xi;u,C)^beta / N(xi;u,C/beta) constant in u: spread =", ratio.max() - ratio.min())

# (d) Gaussian predictive criterion, trivial side information
dx, dy, rk = 4, 3, 2
Araw = rng.normal(size=(dx + dy, dx + dy)); Sig = Araw @ Araw.T + 0.5 * np.eye(dx + dy)
Sxx, Sxy, Syy = Sig[:dx, :dx], Sig[:dx, dx:], Sig[dx:, dx:]
Wx = linalg.inv(linalg.sqrtm(Sxx)).real; Wy = linalg.inv(linalg.sqrtm(Syy)).real
Ucan, rho, Vt = np.linalg.svd(Wx @ Sxy @ Wy)
IXY = -0.5 * np.sum(np.log(1 - rho ** 2))
logdet_y = np.linalg.slogdet(Syy)[1]


def I_YU(A):
    Suu = A @ Sxx @ A.T; Suy = A @ Sxy
    cond = Syy - Suy.T @ np.linalg.solve(Suu, Suy)
    return 0.5 * (logdet_y - np.linalg.slogdet(cond)[1])


A_opt = (Wx @ Ucan[:, :rk]).T
D_opt = IXY - I_YU(A_opt)
D_rand = min(IXY - I_YU(rng.normal(size=(rk, dx))) for _ in range(2000))
print(f"(d) canonical correlations {rho.round(4)}; D(top-{rk}) = {D_opt:.6f}, formula -1/2 sum_(k>r) log(1-rho^2) = "
      f"{-0.5 * np.sum(np.log(1 - rho[rk:] ** 2)):.6f}; best of 2000 random rank-{rk} encoders {D_rand:.6f}")

# (e) composition through depth: the next cue is the previous result, U_{l+1} = V_l, fixed label (T = I).
# Exact filter = one scalar Kalman filter per label plus label weights; check the per-label evidence
# against the stacked joint Gaussian of (O_0..O_n) given the label, and compare the label posterior
# with the fresh-emission filter of cmem:filter.
pi2 = np.array([0.4, 0.6]); nu2 = np.array([0.0, 0.5]); L2 = np.array([1.0, 0.7])
om2 = np.array([1.0, -0.5]); F2 = np.array([0.8, -1.2]); Q2 = np.array([0.3, 0.5]); Cr = 0.6
obs = np.array([0.2, 1.1, -0.4, 0.9])
n = len(obs)


def kalman_loglik(a):
    m, P, ll = nu2[a], L2[a], 0.0
    for l in range(n):
        S_ = P + Cr
        ll += stats.norm.logpdf(obs[l], m, np.sqrt(S_))
        K_ = P / S_
        m, P = m + K_ * (obs[l] - m), (1 - K_) * P          # filtered cue U_l
        m, P = om2[a] + F2[a] * (m - nu2[a]), F2[a] ** 2 * P + Q2[a]   # V_l = next cue U_{l+1}
    return ll


def stacked_loglik(a):
    # U_0 ~ N(nu, L); U_{l+1} = om + F (U_l - nu) + eta_l; O_l = U_l + eps_l
    mean = np.zeros(n); cov = np.zeros((n, n))
    # represent U_l = c_l + sum_k G[l,k] xi_k with xi_0 = U_0 - nu (var L), xi_k = eta_{k-1} (var Q)
    G = np.zeros((n, n)); c = np.zeros(n)
    c[0], G[0, 0] = nu2[a], 1.0
    for l in range(1, n):
        c[l] = om2[a] + F2[a] * (c[l - 1] - nu2[a])
        G[l] = F2[a] * G[l - 1]
        G[l, l] = 1.0
    var = np.array([L2[a]] + [Q2[a]] * (n - 1))
    cov = G @ np.diag(var) @ G.T + Cr * np.eye(n)
    return stats.multivariate_normal.logpdf(obs, c, cov)


kl_ll = np.array([kalman_loglik(a) for a in range(2)])
st_ll = np.array([stacked_loglik(a) for a in range(2)])
w_exact = pi2 * np.exp(kl_ll - kl_ll.max()); w_exact /= w_exact.sum()
fresh = pi2 * np.exp(np.array([np.sum(stats.norm.logpdf(obs, nu2[a], np.sqrt(L2[a] + Cr))) for a in range(2)]))
fresh /= fresh.sum()
print("(e) per-label evidence: Kalman vs stacked joint, max |diff| =", np.abs(kl_ll - st_ll).max())
print("(e) exact label posterior (parallel Kalman filters) =", w_exact.round(6), "; fresh-emission filter =", fresh.round(6))
