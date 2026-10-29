r"""XC6-PROG s1: coupled-recognition identification of the reciprocal potential.

Claim tested.  On atomless spaces, for fixed positive rows, the reciprocal potential
    E(x) = sum_i KL(q_i || r_i) + sum_{ij} beta_ij KL(q_i || q_j)   (+ row constants)
equals the variational free energy of the FIXED normalized model P = r_1 (x) ... (x) r_N
(no record, Z = 1) evaluated at a recognition law Q(x) whose one-dimensional marginals are
the agent laws q_i.  Q(x) is a Gaussian-copula coupling of the q_i with a correlation chosen
so that its total correlation equals E(x) - E_0(x).  The identity used is the chain rule
    KL(Q || (x)_i r_i) = sum_i KL(q_i || r_i) + TC(Q),   TC(Q) = KL(Q || (x)_i q_i),
together with invariance of TC under strictly increasing marginal maps (copula invariance).

Parts
 (a) two Gaussian agents, closed form;
 (b) two non-Gaussian agents (Gaussian mixture, logistic), grid quadrature;
 (c) four Gaussian agents, fixed rows with row-entropy constant, equicorrelated copula;
 (d) finite alphabets: the total correlation of any coupling of two binary laws is at most
     log 2, while KL(q1||q2) is unbounded (consistent with prop:dyn-no-global-vfe (a)).

Numerical agreement supports the identity at the instances run; the identity itself is the
chain rule plus copula invariance of mutual information.
"""
import numpy as np
from scipy import special, stats, optimize

np.set_printoptions(precision=6)


def kl_gauss(m0: np.ndarray, S0: np.ndarray, m1: np.ndarray, S1: np.ndarray) -> float:
    r"""KL(N(m0,S0) || N(m1,S1)) = 1/2 [tr(S1^{-1}S0) + d^T S1^{-1} d - k + log det S1 - log det S0]."""
    m0, m1 = np.atleast_1d(m0), np.atleast_1d(m1)
    S0, S1 = np.atleast_2d(S0), np.atleast_2d(S1)
    k = m0.size
    S1i = np.linalg.inv(S1)
    d = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) + d @ S1i @ d - k
                  + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])


def part_a() -> None:
    rng = np.random.default_rng(1)
    worst, used, skipped = 0.0, 0, 0
    for _ in range(400):
        a1, a2, b1, b2 = rng.normal(size=4)
        s1, s2, t1, t2 = np.exp(rng.normal(scale=0.5, size=4))
        beta = rng.uniform(0.1, 2.0)
        E0 = kl_gauss(a1, s1**2, b1, t1**2) + kl_gauss(a2, s2**2, b2, t2**2)
        Eint = beta * kl_gauss(a1, s1**2, a2, s2**2)
        if Eint > 5.0:            # keep the covariance matrix numerically nonsingular
            skipped += 1
            continue
        rho = np.sqrt(-np.expm1(-2.0 * Eint))              # -1/2 log(1-rho^2) = Eint
        Sig = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
        F = kl_gauss(np.array([a1, a2]), Sig, np.array([b1, b2]), np.diag([t1**2, t2**2]))
        worst = max(worst, abs(F - (E0 + Eint)))
        used += 1
    print(f"(a) two Gaussian agents, {used} random instances (skipped {skipped} with beta*KL > 5): "
          f"max |F(Q(x)) - E(x)| = {worst:.3e}")


def part_b() -> None:
    # agent laws (atomless, non-Gaussian) and references
    w, mA, sA, mB, sB = 0.6, -1.0, 0.5, 1.5, 0.8
    loc2, sc2 = 0.3, 0.7
    beta = 0.7
    x = np.linspace(-6.0, 7.0, 2601)
    y = np.linspace(-9.0, 10.0, 2601)
    dx, dy = x[1] - x[0], y[1] - y[0]
    q1 = w * stats.norm.pdf(x, mA, sA) + (1 - w) * stats.norm.pdf(x, mB, sB)
    F1 = w * stats.norm.cdf(x, mA, sA) + (1 - w) * stats.norm.cdf(x, mB, sB)
    q2 = stats.logistic.pdf(y, loc2, sc2)
    F2 = stats.logistic.cdf(y, loc2, sc2)
    r1 = stats.norm.pdf(x, 0.0, 1.0)
    r2 = stats.norm.pdf(y, 0.5, 1.5)
    # KL(q1||q2) on a common grid (x grid), logistic evaluated there
    q2x = stats.logistic.pdf(x, loc2, sc2)
    kl12 = np.sum(q1 * np.log(q1 / q2x)) * dx
    E0 = np.sum(q1 * np.log(q1 / r1)) * dx + np.sum(q2 * np.log(q2 / r2)) * dy
    Eint = beta * kl12
    rho = np.sqrt(1.0 - np.exp(-2.0 * Eint))
    # Gaussian copula joint density on the grid
    eps = 1e-15
    A = special.ndtri(np.clip(F1, eps, 1 - eps))[:, None]
    B = special.ndtri(np.clip(F2, eps, 1 - eps))[None, :]
    logc = (-0.5 * np.log(1 - rho**2)
            - (rho**2 * (A**2 + B**2) - 2 * rho * A * B) / (2 * (1 - rho**2)))
    logf = logc + np.log(q1)[:, None] + np.log(q2)[None, :]
    f = np.exp(logf)
    mass = f.sum() * dx * dy
    m1 = f.sum(axis=1) * dy
    m2 = f.sum(axis=0) * dx
    KLQ = np.sum(f * (logf - np.log(r1)[:, None] - np.log(r2)[None, :])) * dx * dy
    MI = np.sum(f * logc) * dx * dy
    print("(b) two non-Gaussian agents (mixture, logistic), grid quadrature:")
    print(f"    mass of Q = {mass:.8f}; marginal errors: {np.max(np.abs(m1 - q1)):.2e}, {np.max(np.abs(m2 - q2)):.2e}")
    print(f"    E0 = {E0:.8f}, beta*KL(q1||q2) = {Eint:.8f}, rho = {rho:.6f}")
    print(f"    copula MI numeric = {MI:.8f}  vs  -1/2 log(1-rho^2) = {-0.5*np.log(1-rho**2):.8f}")
    print(f"    F(Q(x)) = KL(Q||r1 x r2) = {KLQ:.8f}  vs  E(x) = {E0 + Eint:.8f};  diff = {KLQ - E0 - Eint:.2e}")


def part_c() -> None:
    rng = np.random.default_rng(7)
    N = 4
    T = 0.4
    worst = 0.0
    for _ in range(100):
        mu = rng.normal(size=N)
        sd = np.exp(rng.normal(scale=0.4, size=N))
        rm = rng.normal(size=N)
        rs = np.exp(rng.normal(scale=0.4, size=N))
        beta = rng.uniform(0.1, 1.0, size=(N, N))
        np.fill_diagonal(beta, 0.0)
        beta /= beta.sum(axis=1, keepdims=True)
        pi = np.full((N, N), 1.0 / (N - 1))
        np.fill_diagonal(pi, 0.0)
        E0 = sum(kl_gauss(mu[i], sd[i]**2, rm[i], rs[i]**2) for i in range(N))
        Eint = sum(beta[i, j] * kl_gauss(mu[i], sd[i]**2, mu[j], sd[j]**2)
                   for i in range(N) for j in range(N) if i != j)
        Erow = T * sum(np.sum(beta[i][beta[i] > 0] * np.log(beta[i][beta[i] > 0] / pi[i][beta[i] > 0]))
                       for i in range(N))
        target = Eint + Erow                       # nonnegative for fixed rows

        def tc(r: float) -> float:
            # total correlation of the equicorrelated Gaussian copula
            return -0.5 * ((N - 1) * np.log(1 - r) + np.log(1 + (N - 1) * r))

        r = optimize.brentq(lambda z: tc(z) - target, 0.0, 1.0 - 1e-15)
        C = (1 - r) * np.eye(N) + r * np.ones((N, N))
        D = np.diag(sd)
        F = kl_gauss(mu, D @ C @ D, rm, np.diag(rs**2))
        worst = max(worst, abs(F - (E0 + target)))
    print(f"(c) four Gaussian agents, fixed rows, 100 instances: max |F(Q(x)) - E(x)| = {worst:.3e}")


def part_d() -> None:
    print("(d) binary alphabets: TC of any coupling <= min(H(q1),H(q2)) <= log 2 = 0.693147,")
    for t in [0.9, 0.99, 0.999, 0.9999]:
        p1, p2 = 0.5 * (1 + t), 0.5 * (1 - t)
        kl = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
        H1 = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
        print(f"    t = {t}: KL(q1||q2) = {kl:.4f}, H(q1) = {H1:.4f}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
