r"""D19-CTXF s7: the finite elimination does not need squared Fisher-Rao distance.

Replace c_e = 1/2 d^2(p_v, U p_u) by the G-invariant cost c_e = KL(p_v || U p_u) on FULL Gaussian laws
N(mu, Sigma) on R^2 (closed form, unlike the Fisher-Rao distance of the full Gaussian family), two agents
per context, with G = GL(2) acting by x -> A x (an isometry of the Fisher metric and an invariance of KL).
Checks:
 (i)  horizontality from invariance alone: at the minimizing U, dc_e(J_{p_v} xi, 0) = 0 for all xi in gl(2)
      (because c(g x, g z) = c(x, z) and stationarity in U), with a non-stationary U as baseline;
 (ii) first-order recovery of the Fisher normal equation ctx:f:eq:normal: for a smooth section p(c) and
      U*(eps) = exp(-eps a(eps)) minimizing KL(p(c + eps X) || U p(c)), a(eps) -> A^F(X), where
      k(A^F, xi) = -G(dp(X), J xi) with the full-Gaussian Fisher metric
      G = dmu^T Sigma^{-1} dmu + 1/2 tr(Sigma^{-1} dSigma Sigma^{-1} dSigma).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

rng = np.random.default_rng(11)
BASIS = [np.eye(2)[:, [i]] @ np.eye(2)[[j], :] for i in range(2) for j in range(2)]


def _x_minus_log1p(x: np.ndarray) -> np.ndarray:
    """Stable x - log(1 + x); series for small |x| avoids cancellation when the laws nearly agree."""
    out = np.empty_like(x)
    small = np.abs(x) < 0.05
    xs = x[small]
    out[small] = sum(((-1) ** k) * xs**k / k for k in range(2, 16))
    out[~small] = x[~small] - np.log1p(x[~small])
    return out


def kl_gauss(m0, S0, m1, S1):
    r"""KL(N(m0,S0)||N(m1,S1)) = 1/2 sum_i (l_i - 1 - log l_i) + 1/2 d^T S1^{-1} d, l = eig(S1^{-1/2} S0 S1^{-1/2})."""
    w, V = np.linalg.eigh(S1)
    S1ih = (V / np.sqrt(w)) @ V.T
    lam = np.linalg.eigvalsh(S1ih @ S0 @ S1ih)
    d = S1ih @ (m1 - m0)
    return 0.5 * float(np.sum(_x_minus_log1p(lam - 1.0))) + 0.5 * float(d @ d)


def act(A, p):
    return [(A @ m, A @ S @ A.T) for m, S in p]


def cost(x, z):
    return sum(kl_gauss(mx, Sx, mz, Sz) for (mx, Sx), (mz, Sz) in zip(x, z))


def U_of(a, U0=np.eye(2)):
    return expm(sum(ai * Ti for ai, Ti in zip(a, BASIS))) @ U0


def best_U(x, y, a0=None):
    f = lambda a: cost(x, act(U_of(a), y))
    res = minimize(f, np.zeros(4) if a0 is None else a0, method="BFGS", options={"gtol": 1e-12, "maxiter": 500})
    return U_of(res.x), res


def Jt(p, xi):
    return [(xi @ m, xi @ S + S @ xi.T) for m, S in p]


def perturb(p, d, s):
    return [(m + s * dm, S + s * dS) for (m, S), (dm, dS) in zip(p, d)]


def fisher(p, a, b):
    tot = 0.0
    for (m, S), (am, aS), (bm, bS) in zip(p, a, b):
        Si = np.linalg.inv(S)
        tot += am @ Si @ bm + 0.5 * np.trace(Si @ aS @ Si @ bS)
    return tot


def rand_spd():
    B = rng.normal(size=(2, 2))
    return B @ B.T + 0.5 * np.eye(2)


def main():
    L = []
    # (i) horizontality of the endpoint force at the minimizer, for all xi in gl(2)
    x = [(rng.normal(size=2), rand_spd()) for _ in range(2)]
    y = [(rng.normal(size=2), rand_spd()) for _ in range(2)]
    U, res = best_U(x, y)
    z = act(U, y)
    h = 1e-6
    vals_min, vals_non = [], []
    Unon = U_of(np.array([0.2, -0.1, 0.15, 0.05]), U)
    znon = act(Unon, y)
    for xi in BASIS:
        d = Jt(x, xi)
        vals_min.append((cost(perturb(x, d, h), z) - cost(perturb(x, d, -h), z)) / (2 * h))
        vals_non.append((cost(perturb(x, d, h), znon) - cost(perturb(x, d, -h), znon)) / (2 * h))
    L.append(f"(i) GL(2), KL edge cost, 2 full-Gaussian agents: BFGS gradient norm {np.linalg.norm(res.jac):.1e}")
    L.append(f"    dc(J_x xi, 0) at the minimizer, xi = E11,E12,E21,E22: {[f'{v:+.2e}' for v in vals_min]}")
    L.append(f"    same at a non-stationary U (baseline): {[f'{v:+.3f}' for v in vals_non]}")

    # (ii) first-order recovery of the Fisher normal equation
    mu0 = [rng.normal(size=2) for _ in range(2)]
    S0 = [rand_spd() for _ in range(2)]
    dmu = [rng.normal(size=2) for _ in range(2)]
    dS = []
    for _ in range(2):
        B = rng.normal(size=(2, 2)); dS.append(0.5 * (B + B.T))
    p0 = list(zip(mu0, S0))
    dp = list(zip(dmu, dS))
    # Fisher normal equation: k a = -b, k_ij = G(J T_i, J T_j), b_i = G(dp, J T_i)
    k = np.array([[fisher(p0, Jt(p0, Ti), Jt(p0, Tj)) for Tj in BASIS] for Ti in BASIS])
    b = np.array([fisher(p0, dp, Jt(p0, Ti)) for Ti in BASIS])
    AF = np.linalg.solve(k, -b)
    L.append(f"(ii) locked inertia eigenvalues {np.round(np.linalg.eigvalsh(k), 4).tolist()} (free action); A^F = {np.round(AF, 6).tolist()}")
    a_prev = np.zeros(4)
    a_by_eps = {}
    for eps in (1e-1, 5e-2, 2.5e-2, 1.25e-2, 1e-3):
        xe = perturb(p0, dp, eps)
        # objective scaled by 1/eps^2 so that its curvature in a is O(1) for every eps
        f = lambda a: cost(xe, act(expm(-eps * sum(ai * Ti for ai, Ti in zip(a, BASIS))), p0)) / eps**2
        r = minimize(f, np.zeros(4), method="BFGS", options={"gtol": 1e-11, "maxiter": 2000})
        a_prev = r.x
        a_by_eps[eps] = r.x
        L.append(f"     eps = {eps:g}: a(eps) = {np.round(r.x, 6).tolist()}, |a(eps) - A^F| = {np.linalg.norm(r.x - AF):.2e}, scaled-gradient norm {np.linalg.norm(r.jac):.1e}")
    for e in (5e-2, 2.5e-2, 1.25e-2):
        rich = 2 * a_by_eps[e] - a_by_eps[2 * e]
        L.append(f"     Richardson 2a(eps) - a(2 eps) at eps = {e:g}: |. - A^F| = {np.linalg.norm(rich - AF):.2e}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
