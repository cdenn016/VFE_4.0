r"""Numerical checks of MAgent Theory 2.0 identities that coincide with known results.

Each block prints a residual. Numerical agreement supports an identity; it does not prove it.
Run single-threaded:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python lit_identity_checks.py
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0, i1, gammaln, logsumexp

rng = np.random.default_rng(20261022)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL(p || q) = \sum_x p(x) \log(p(x)/q(x)) on a finite alphabet."""
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


# 1. Gibbs variational principle (prop:dyn-gibbs): min_beta sum beta_j D_j + T KL(beta||pi)
#    = -T log sum pi_j exp(-D_j/T), with gap T KL(beta||beta*).
def check_gibbs() -> float:
    n, T = 5, 0.7
    D = rng.uniform(0, 3, n)
    pi = rng.dirichlet(np.ones(n))
    beta = rng.dirichlet(np.ones(n))
    Z = np.sum(pi * np.exp(-D / T))
    bstar = pi * np.exp(-D / T) / Z
    lhs = np.sum(beta * D) + T * kl(beta, pi)
    rhs = -T * np.log(Z) + T * kl(beta, bstar)
    return abs(lhs - rhs)


# 2. Fisher-Rao gradient flow of KL(kappa||A) has the explicit geometric-interpolation solution
#    kappa_t = kappa_0^{a} A^{1-a} / Z(a), a = e^{-t}  (smooth:spectrum-solution; open:angular-flow
#    with rate 2).  Also dH/dt = -Var_{kappa_t}(log kappa_t/A) and H = a psi'(a) - psi(a),
#    psi(a) = log Z(a) = (a-1) D_a(kappa_0||A) (Renyi divergence of order a).
def check_fr_flow() -> tuple[float, float, float]:
    m = 7
    A = rng.dirichlet(np.ones(m))
    k0 = rng.dirichlet(np.ones(m))
    l0 = np.log(k0 / A)

    def kt(t: float) -> np.ndarray:
        a = np.exp(-t)
        w = A * np.exp(a * l0)
        return w / w.sum()

    t, h = 0.6, 1e-5
    # residual of d/dt log(kappa/A) = H - log(kappa/A)
    lt = np.log(kt(t) / A)
    dl = (np.log(kt(t + h) / A) - np.log(kt(t - h) / A)) / (2 * h)
    H = float(np.sum(kt(t) * lt))
    r1 = float(np.max(np.abs(dl - (H - lt))))
    # entropy rate
    Hp = kl(kt(t + h), A)
    Hm = kl(kt(t - h), A)
    var = float(np.sum(kt(t) * lt**2) - H**2)
    r2 = abs((Hp - Hm) / (2 * h) + var)
    # Renyi representation
    a = np.exp(-t)
    psi = lambda b: float(np.log(np.sum(A * np.exp(b * l0))))
    dpsi = (psi(a + h) - psi(a - h)) / (2 * h)
    r3 = abs(H - (a * dpsi - psi(a)))
    return r1, r2, r3


# 3. Jeffrey conditioning (prop:conditional-lift): qhat = sum_h p(.|h) nu(h) has the T-marginal nu,
#    KL(q||qhat) = E_nu KL(q(.|h)||p(.|h)), and the Pythagorean identity
#    KL(q||p) = KL(q||qhat) + KL(qhat||p) with KL(qhat||p) = KL(nu||T#p) (Diaconis-Zabell Thm 5.1).
def check_jeffrey() -> tuple[float, float, float]:
    nh, nv = 3, 4
    q = rng.dirichlet(np.ones(nh * nv)).reshape(nh, nv)
    p = rng.dirichlet(np.ones(nh * nv)).reshape(nh, nv)
    nu = q.sum(1)
    pT = p.sum(1)
    pc = p / pT[:, None]
    qc = q / nu[:, None]
    qhat = pc * nu[:, None]
    r1 = float(np.max(np.abs(qhat.sum(1) - nu)))
    lhs = kl(q.ravel(), qhat.ravel())
    rhs = sum(nu[h] * kl(qc[h], pc[h]) for h in range(nh))
    r2 = abs(lhs - rhs)
    r3 = abs(kl(q.ravel(), p.ravel()) - lhs - kl(qhat.ravel(), p.ravel())) + abs(
        kl(qhat.ravel(), p.ravel()) - kl(nu, pT))
    return r1, r2, r3


# 4. open:limit is the Euclidean gradient flow of the log-sum-exp (Gaussian-kernel attention) energy
#    E(u) = sum u_i^2/2 - sum_i log sum_k exp(-(u_i-u_k)^2/2).
def check_lse_gradient() -> float:
    u = rng.normal(0, 2, 6)

    def E(v: np.ndarray) -> float:
        d = -(v[:, None] - v[None, :]) ** 2 / 2
        return float(np.sum(v**2) / 2 - np.sum(logsumexp(d, axis=1)))

    d = -(u[:, None] - u[None, :]) ** 2 / 2
    w = np.exp(d - logsumexp(d, axis=1, keepdims=True))  # row-normalized, self included
    field = -u + np.array([np.sum((w[i] + w[:, i]) * (u - u[i])) for i in range(len(u))])
    h = 1e-6
    grad = np.array([(E(u + h * e) - E(u - h * e)) / (2 * h) for e in np.eye(len(u))])
    return float(np.max(np.abs(field + grad)))


# 5. Replicator / Shahshahani dissipation (thm:dyn-finite-flow): dE/dt = -Var_u(h) for
#    du = -u (h - <h>), on a random smooth energy over one simplex.
def check_replicator() -> float:
    n = 6
    Q = rng.normal(size=(n, n))
    Q = Q + Q.T
    c = rng.normal(size=n)
    E = lambda u: float(u @ Q @ u / 2 + c @ u + np.sum(u * np.log(u)))
    grad = lambda u: Q @ u + c + np.log(u) + 1
    u = rng.dirichlet(np.ones(n))
    hz = grad(u)
    udot = -u * (hz - u @ hz)
    dt = 1e-6
    dE = (E(u + dt * udot) - E(u - dt * udot)) / (2 * dt)
    var = float(u @ hz**2 - (u @ hz) ** 2)
    return abs(dE + var)


# 6. Sticky collisions (fus:two-stages): event-driven solution of z' in -lambda sum n_j Sign(z_i-z_j)
#    with minimal-norm (mass-weighted) merging, compared with tau_1 = d/(2 lambda), tau_2 = 5d/(4 lambda).
def check_sticky() -> tuple[float, float]:
    lam, d = 0.8, 1.3
    pos = [-3 * d, -2 * d, 2 * d, 3 * d]
    mass = [1, 1, 1, 1]
    tau, events = 0.0, []
    while len(pos) > 1:
        M = sum(mass)
        left = np.cumsum([0] + mass[:-1])
        vel = [lam * ((M - left[i] - mass[i]) - left[i]) for i in range(len(pos))]
        gaps = [(pos[i + 1] - pos[i]) / (vel[i] - vel[i + 1]) if vel[i] > vel[i + 1] else np.inf
                for i in range(len(pos) - 1)]
        dt = min(gaps)
        tau += dt
        pos = [p + v * dt for p, v in zip(pos, vel)]
        merge = [i for i, g in enumerate(gaps) if abs(g - dt) < 1e-12]
        for i in reversed(merge):
            pos[i] = pos[i]
            mass[i] = mass[i] + mass[i + 1]
            del pos[i + 1], mass[i + 1]
        events.append(tau)
    return abs(events[0] - d / (2 * lam)), abs(events[1] - 5 * d / (4 * lam))


# 7. Wishart (form:wishart, form:fisher): Fisher metric on the precision at fixed shape equals the
#    Hessian of the log normalizer -alpha log det P, i.e. alpha tr(P^{-1} U P^{-1} V); and
#    log Gamma_2(nu) = (1/2) log pi + log Gamma(nu) + log Gamma(nu - 1/2) (Muirhead Thm 2.1.12 form).
def check_wishart() -> tuple[float, float]:
    alpha = 1.7
    B = rng.normal(size=(2, 2))
    P = B @ B.T + np.eye(2)
    U = rng.normal(size=(2, 2)); U = U + U.T
    V = rng.normal(size=(2, 2)); V = V + V.T
    f = lambda s, t: -alpha * np.log(np.linalg.det(P + s * U + t * V))
    h = 1e-4
    hess = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    Pi = np.linalg.inv(P)
    target = alpha * np.trace(Pi @ U @ Pi @ V)
    r1 = abs(hess - target)
    # normalizer by quadrature in the coordinates (u, v, z) of the manuscript's proof at P = I
    nu = 1.3
    from scipy import integrate
    gu = integrate.quad(lambda x: x ** (nu - 1.0) * np.exp(-x), 0, np.inf)[0]   # u^{nu-3/2} sqrt(u)
    gv = integrate.quad(lambda x: x ** (nu - 1.5) * np.exp(-x), 0, np.inf)[0]
    gz = np.sqrt(np.pi)
    r2 = abs(np.log(gu * gv * gz) - (0.5 * np.log(np.pi) + gammaln(nu) + gammaln(nu - 0.5)))
    return r1, r2


# 8. von Mises (ch. 12): rotation Fisher information E[kappa^2 sin^2] = kappa I1/I0 and
#    d/dkappa (I1/I0) = Var(cos theta) (Mardia-Jupp facts).
def check_vonmises() -> tuple[float, float]:
    kap = 1.9
    th = np.linspace(-np.pi, np.pi, 200001)
    pdf = np.exp(kap * np.cos(th)) / (2 * np.pi * i0(kap))
    E = lambda g: float(np.trapezoid(g * pdf, th))
    r1 = abs(E(kap**2 * np.sin(th) ** 2) - kap * i1(kap) / i0(kap))
    h = 1e-5
    A = lambda k: i1(k) / i0(k)
    r2 = abs((A(kap + h) - A(kap - h)) / (2 * h) - (E(np.cos(th) ** 2) - E(np.cos(th)) ** 2))
    return r1, r2


# 9. Prony identity (ch. 12 binary inverse): F = sum_j b_j e^{v_j r}; H = m0 m2 - m1^2 and the
#    quadratic z^2 - (m0 m3 - m1 m2)/H z + (m1 m3 - m2^2)/H has roots v_1, v_2.
def check_prony() -> float:
    b = rng.uniform(0.2, 2, 2)
    v = rng.normal(0, 2, 2)
    r0 = 0.3
    m = [float(np.sum(b * v**k * np.exp(v * r0))) for k in range(4)]
    H = m[0] * m[2] - m[1] ** 2
    roots = np.sort(np.roots([1, -(m[0] * m[3] - m[1] * m[2]) / H, (m[1] * m[3] - m[2] ** 2) / H]).real)
    return float(np.max(np.abs(roots - np.sort(v))))


# 10. Quotient (Riemannian-submersion) metric vs canonical-slice metric (open:two-metrics).
def check_two_metrics() -> tuple[float, float]:
    I1, I2, dF = 0.7, 2.3, 1.0
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda x: I1 * x**2 + I2 * (dF - x) ** 2)
    r1 = abs(res.fun - dF**2 / (1 / I1 + 1 / I2))
    r2 = abs((I1 + I2) / 4 * dF**2 - (I1 * (dF / 2) ** 2 + I2 * (dF / 2) ** 2))
    return r1, r2


# 11. Joint Fisher example (geo:joint-fisher): zeta_rho(k,m) = (1 + rho k m)/4 has Fisher 1/(1-rho^2).
def check_joint_fisher() -> float:
    rho = 0.37
    tot = 0.0
    for k in (-1, 1):
        for m in (-1, 1):
            p = (1 + rho * k * m) / 4
            s = k * m / (1 + rho * k * m)
            tot += p * s * s
    return abs(tot - 1 / (1 - rho**2))


if __name__ == "__main__":
    print("1 Gibbs variational identity residual:", check_gibbs())
    print("2 Fisher-Rao KL flow (log-density eq, entropy rate, Renyi form):", check_fr_flow())
    print("3 Jeffrey lift (marginal, KL identity, Pythagoras):", check_jeffrey())
    print("4 open:limit = -grad log-sum-exp energy, max residual:", check_lse_gradient())
    print("5 replicator dissipation dE/dt + Var residual:", check_replicator())
    print("6 sticky collision times residuals:", check_sticky())
    print("7 Wishart Fisher = Hessian of -alpha logdet; Gamma_2 normalizer:", check_wishart())
    print("8 von Mises inertia and derivative:", check_vonmises())
    print("9 Prony roots residual:", check_prony())
    print("10 quotient vs slice metric residuals:", check_two_metrics())
    print("11 joint Fisher 1/(1-rho^2) residual:", check_joint_fisher())


# 12. Pushforward by a statistic is a Riemannian submersion for the Fisher-Rao (Shahshahani)
#     metrics: horizontal vectors are q*s(C) (C-measurable scores), their Fisher norm equals the
#     pushforward norm, they are orthogonal to ker(dC_#), and the Fisher-Rao gradient of E_c o C_#
#     is the horizontal lift of the Fisher-Rao gradient of E_c (links inh:gradient-inheritance and
#     smooth:mean-force-criterion; congruent embeddings in the sense of Cencov).
def check_submersion() -> tuple[float, float, float]:
    ny, nz = 7, 3
    C = np.array([0, 0, 1, 1, 1, 2, 2])
    q = rng.dirichlet(np.ones(ny))
    mu = np.array([q[C == z].sum() for z in range(nz)])
    s = rng.normal(size=nz)
    s = s - mu @ s
    vh = q * s[C]
    push = np.array([vh[C == z].sum() for z in range(nz)])
    r1 = abs(np.sum(vh**2 / q) - np.sum(push**2 / mu))
    w = rng.normal(size=ny)
    for z in range(nz):  # project w into ker(dC_#)
        idx = C == z
        w[idx] -= w[idx].mean()
    r2 = abs(np.sum(vh * w / q))
    # gradient descent: E_c(mu) = sum mu log(mu/b) + mu.a ; FR gradient field -mu(h - <h>)
    b = rng.dirichlet(np.ones(nz)); a = rng.normal(size=nz)
    hc = np.log(mu / b) + 1 + a
    fine_h = hc[C]                       # derivative of E_c(C_# q) w.r.t. q(y)
    qdot = -q * (fine_h - q @ fine_h)
    mudot = -mu * (hc - mu @ hc)
    r3 = float(np.max(np.abs(np.array([qdot[C == z].sum() for z in range(nz)]) - mudot)))
    return r1, r2, r3


if __name__ == "__main__":
    print("12 statistic pushforward is a Fisher-Rao Riemannian submersion (norm, orthogonality, gradient):",
          check_submersion())
