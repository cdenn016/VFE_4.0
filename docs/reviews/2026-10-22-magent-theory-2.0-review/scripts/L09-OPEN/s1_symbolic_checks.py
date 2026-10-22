r"""L09-OPEN symbolic checks for chapters/12_open_formation.tex.

Checks (each prints a residual that should be 0 or a boolean):
  1. open:binary-field equals the negative Fisher gradient of the binary
     Gibbs-reduced energy (N=3, generic numeric point, finite differences).
  2. open:full-field restricted to the product family reproduces
     open:binary-field (symbolic, N=2 with self rows).
  3. open:core is the reflection reduction of open:limit (N=4).
  4. Prony/Hankel identities of the identity-channel test.
  5. The 32/3 coefficient of the generated-loss example.
  6. open:two-metrics (quotient vs slice metric).
  7. derivative bound |g_c'(r)| <= (1+4 r^2) e^{-2 r^2} on a grid.
Run: OMP_NUM_THREADS=1 timeout 20 python s1_symbolic_checks.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp


def check_binary_field() -> None:
    r"""Compare open:binary-field with -g^{-1} dE for
    E = \sum_i \psi(x_i) - T \sum_i \log \frac1N \sum_j e^{-d(x_i,x_j)/T}."""
    rng = np.random.default_rng(0)
    N, T = 3, 0.37
    x = rng.uniform(-0.8, 0.8, N)

    def psi(u: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + u) * np.log1p(u) + 0.5 * (1 - u) * np.log1p(-u)

    def d(u: float, v: float) -> float:
        return 0.5 * (1 + u) * np.log((1 + u) / (1 + v)) + 0.5 * (1 - u) * np.log((1 - u) / (1 - v))

    def E(xx: np.ndarray) -> float:
        tot = psi(xx).sum()
        for i in range(N):
            tot -= T * np.log(np.mean([np.exp(-d(xx[i], xx[j]) / T) for j in range(N)]))
        return float(tot)

    eps = 1e-6
    grad = np.array([(E(x + eps * np.eye(N)[i]) - E(x - eps * np.eye(N)[i])) / (2 * eps) for i in range(N)])
    flow_fd = -(1 - x**2) * grad
    D = np.array([[d(x[i], x[j]) for j in range(N)] for i in range(N)])
    W = np.exp(-D / T)
    beta = W / W.sum(1, keepdims=True)
    at = np.arctanh(x)
    flow = np.array([
        (1 - x[i]**2) * (-at[i] - sum(beta[i, j] * (at[i] - at[j]) for j in range(N)))
        + sum(beta[k, i] * (x[k] - x[i]) for k in range(N))
        for i in range(N)])
    print("[1] open:binary-field vs FD Fisher gradient, max|diff| =", np.max(np.abs(flow - flow_fd)))


def check_product_reduction() -> None:
    r"""Product family q_i = mu_i(z) f_i(theta): centered force on the bit marginal.
    x' = -(1-x^2)(H(1)-H(-1))/2 with
    H(z) = 2 log mu_i - sum_j beta_ij log mu_j - sum_k beta_ki mu_k/mu_i."""
    x1, x2, b11, b12, b21, b22 = sp.symbols("x1 x2 b11 b12 b21 b22", real=True)
    xs = [x1, x2]
    B = [[b11, b12], [b21, b22]]
    i = 0
    def mu(j, z):
        return 1 + z * xs[j]
    def H(z):
        return (2 * sp.log(mu(i, z)) - sum(B[i][j] * sp.log(mu(j, z)) for j in range(2))
                - sum(B[k][i] * mu(k, z) / mu(i, z) for k in range(2)))
    xdot = -(1 - x1**2) * (H(1) - H(-1)) / 2
    target = ((1 - x1**2) * (-sp.atanh(x1) - sum(B[i][j] * (sp.atanh(x1) - sp.atanh(xs[j])) for j in range(2)))
              + sum(B[k][i] * (xs[k] - x1) for k in range(2)))
    diff = (xdot - target).subs(b12, 1 - b11)  # row normalization
    diff = diff.rewrite(sp.log)
    val = [float(diff.subs({x1: 0.3, x2: -0.45, b11: 0.6, b21: 0.25, b22: 0.75})),
           float(diff.subs({x1: -0.7, x2: 0.2, b11: 0.1, b21: 0.9, b22: 0.1}))]
    print("[2] product reduction of open:full-field to open:binary-field, residuals =", val)


def check_core() -> None:
    a, b = sp.symbols("a b", positive=True)
    u = [-a - b, -a + b, a - b, a + b]
    G = [[sp.exp(-(u[i] - u[j])**2 / 2) for j in range(4)] for i in range(4)]
    Z = [sum(G[i]) for i in range(4)]
    w = [[G[i][j] / Z[i] for j in range(4)] for i in range(4)]
    ud = [-u[i] + sum((w[i][j] + w[j][i]) * (u[j] - u[i]) for j in range(4)) for i in range(4)]
    adot = -(ud[0] + ud[1]) / 2
    bdot = (ud[1] - ud[0]) / 2
    W = sp.exp(-2 * b**2); v0 = sp.exp(-2 * a**2)
    vp = sp.exp(-2 * (a + b)**2); vm = sp.exp(-2 * (a - b)**2)
    Zp = 1 + W + v0 + vp; Zm = 1 + W + v0 + vm
    cp = W * (1 / Zp + 1 / Zm); c0 = v0 * (1 / Zp + 1 / Zm)
    adot_ms = -a - 2 * a * c0 - 2 * ((a + b) * vp / Zp + (a - b) * vm / Zm)
    bdot_ms = -b - 2 * b * cp - 2 * ((a + b) * vp / Zp - (a - b) * vm / Zm)
    res = []
    for av, bv in [(0.7, 0.3), (1.9, 1.1), (4.2, 0.05)]:
        res.append(float((adot - adot_ms).subs({a: av, b: bv})))
        res.append(float((bdot - bdot_ms).subs({a: av, b: bv})))
    # mirror consistency: agents 3,4 equations equal the reflected 2,1
    mir = [float((ud[2] + ud[1]).subs({a: 1.3, b: 0.4})), float((ud[3] + ud[0]).subs({a: 1.3, b: 0.4}))]
    print("[3] open:core residuals =", res, " reflection invariance residuals =", mir)


def check_prony() -> None:
    c1, c2, v1, v2, z = sp.symbols("c1 c2 v1 v2 z")
    m = [c1 * v1**k + c2 * v2**k for k in range(4)]
    H = sp.expand(m[0] * m[2] - m[1]**2)
    print("[4a] Hankel H - c1 c2 (v1-v2)^2 =", sp.simplify(H - c1 * c2 * (v1 - v2)**2))
    s1 = sp.simplify((m[0] * m[3] - m[1] * m[2]) / H)
    s2 = sp.simplify((m[1] * m[3] - m[2]**2) / H)
    print("[4b] sigma1 - (v1+v2) =", sp.simplify(s1 - (v1 + v2)), "; sigma2 - v1 v2 =", sp.simplify(s2 - v1 * v2))
    # The roots are v_j = atanh(x_j)/T, not atanh(x_j); check F(r) form
    r, x, T, p = sp.symbols("r x T p", positive=True)
    dKL = ((1 + r) / 2 * sp.log((1 + r) / (1 + x)) + (1 - r) / 2 * sp.log((1 - r) / (1 - x)))
    psi = (1 + r) / 2 * sp.log(1 + r) + (1 - r) / 2 * sp.log(1 - r)
    expr = sp.exp(psi / T) * p * sp.exp(-dKL / T)
    target = p * (1 - x**2)**(1 / (2 * T)) * sp.exp(r * sp.atanh(x) / T)
    val = [float((sp.log(expr) - sp.log(target)).rewrite(sp.log).subs({r: rv, x: xv, T: 0.3, p: 0.5}))
           for rv, xv in [(0.2, 0.5), (-0.6, 0.1)]]
    print("[4c] F(r) = sum_j b_j exp(v_j r), v_j = atanh(x_j)/T : log-residuals =", val)


def check_32_over_3() -> None:
    mu1 = [sp.Rational(1, 2)] * 2
    mu2 = [sp.Rational(1, 2)] * 2
    v = [1, -1]
    k1 = [sp.Rational(1, 4), sp.Rational(3, 4)]   # kappa_1(y|z) = Bern(1/4) for both z
    k2 = [sp.Rational(3, 4), sp.Rational(1, 4)]   # kappa_2(y|z) = Bern(3/4) for both z
    chi2 = lambda P, Q: sum((P[y] - Q[y])**2 / Q[y] for y in range(2))
    coef = sum(v[z]**2 * (chi2(k2, k1) / mu1[z] + mu1[z]**2 / mu2[z]**3 * chi2(k1, k2)) for z in range(2))
    print("[5] generated-loss coefficient =", coef)


def check_two_metrics() -> None:
    I1, I2, F = sp.symbols("I1 I2 F", positive=True)
    lam = sp.symbols("lam")
    t1 = lam / (2 * I1); t2 = lam / (2 * I2)
    lam_s = sp.solve(sp.Eq(t1 + t2, F), lam)[0]
    q = sp.simplify((I1 * t1**2 + I2 * t2**2).subs(lam, lam_s))
    print("[6] quotient metric - F^2/(1/I1+1/I2) =", sp.simplify(q - F**2 / (1 / I1 + 1 / I2)),
          "; slice - quotient =", sp.factor(sp.simplify((I1 + I2) / 4 * F**2 - q)))


def check_gprime() -> None:
    r = np.linspace(2.0, 8.0, 20001)
    worst = 0.0
    for c in [1.0, 1.5, 3.0, 10.0]:
        e = np.exp(-2 * r**2)
        gp = e * (c * (1 - 4 * r**2) + e) / (c + e)**2
        worst = max(worst, float(np.max(np.abs(gp) / ((1 + 4 * r**2) * e))))
    print("[7] max |g_c'| / ((1+4r^2)e^{-2r^2}) on r in [2,8], c in {1,1.5,3,10} =", worst,
          "; 34 e^-8 =", 34 * np.exp(-8), "; 68 e^-8 =", 68 * np.exp(-8))


if __name__ == "__main__":
    check_binary_field()
    check_product_reduction()
    check_core()
    check_prony()
    check_32_over_3()
    check_two_metrics()
    check_gprime()
