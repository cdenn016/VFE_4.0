r"""D16-OPEN s7: remaining identities of chapter 12.

 [1] open:two-metrics (Lagrange minimization) symbolically.
 [2] Prony inverse of the two-test section: H = c1 c2 (v1-v2)^2 and the quadratic's roots v_j = a_j/T;
     F(r) = e^{psi(r)/T} Z_I(r) = sum_j b_j e^{v_j r} with b_j = pi_j (1-x_j^2)^{1/(2T)}.
 [3] 32/3 coefficient of open:generated-loss (both conditional chi^2 terms).
 [4] open:open-laws: d/deps R_rho(l + eps s) = rho^2 Cov_{p}(l, s), and
     |c(1,t)-c(-1,t)| <= 2 e^{-4t} M ||l_1 - l_{-1}||_inf on random profiles.
 [5] open:global-anchor: the non-symmetric representative is H_I-canonical (random test).
 [6] open:final-merge limits: block rows and contrasts for the partition sequence.
"""
import numpy as np
import sympy as sp

out = []
# [1]
I1, I2, dF, l = sp.symbols('I1 I2 dF lam', positive=True)
d1, d2 = sp.symbols('d1 d2')
Lg = I1 * d1 ** 2 + I2 * d2 ** 2 - l * (d1 + d2 - dF)
sol = sp.solve([sp.diff(Lg, d1), sp.diff(Lg, d2), d1 + d2 - dF], [d1, d2, l], dict=True)[0]
qmin = sp.simplify((I1 * d1 ** 2 + I2 * d2 ** 2).subs(sol))
out.append("[1] quotient metric min = %s ; equals dF^2/(1/I1+1/I2): %s ; slice (equal phase) = %s ; difference = %s"
           % (qmin, sp.simplify(qmin - dF ** 2 / (1 / I1 + 1 / I2)) == 0,
              sp.simplify(I1 * (dF / 2) ** 2 + I2 * (dF / 2) ** 2),
              sp.factor(sp.simplify(I1 * (dF / 2) ** 2 + I2 * (dF / 2) ** 2 - qmin))))
# [2]
c1, c2, v1, v2 = sp.symbols('c1 c2 v1 v2', positive=True)
m = [c1 * v1 ** k + c2 * v2 ** k for k in range(4)]
H = sp.expand(m[0] * m[2] - m[1] ** 2)
s1 = sp.simplify((m[0] * m[3] - m[1] * m[2]) / H)
s2 = sp.simplify((m[1] * m[3] - m[2] ** 2) / H)
out.append("[2] H - c1c2(v1-v2)^2 = %s ; sigma1-(v1+v2) = %s ; sigma2 - v1v2 = %s"
           % (sp.simplify(H - c1 * c2 * (v1 - v2) ** 2), sp.simplify(s1 - v1 - v2), sp.simplify(s2 - v1 * v2)))
r, x1, x2, T, p1, p2 = 0.23, 0.4, -0.55, 0.3, 0.6, 0.4
psi = lambda r: (1 + r) / 2 * np.log(1 + r) + (1 - r) / 2 * np.log(1 - r)
kl = lambda r, x: 0.5 * ((1 + r) * np.log((1 + r) / (1 + x)) + (1 - r) * np.log((1 - r) / (1 - x)))
ZI = p1 * np.exp(-kl(r, x1) / T) + p2 * np.exp(-kl(r, x2) / T)
Fr = sum(p * (1 - x * x) ** (1 / (2 * T)) * np.exp(np.arctanh(x) / T * r) for p, x in ((p1, x1), (p2, x2)))
out.append("    e^{psi/T} Z_I - sum b_j e^{v_j r} (relative) = %.3e" % (np.exp(psi(r) / T) * ZI / Fr - 1))
# [3]
mu1 = mu2 = np.array([0.5, 0.5]); v = np.array([1.0, -1.0])
k1 = np.array([[0.75, 0.25], [0.75, 0.25]])     # rows z, columns y in (0,1) with y=1 prob 1/4
k2 = np.array([[0.25, 0.75], [0.25, 0.75]])
chi2 = lambda p, q: (p * p / q).sum(-1) - 1
coef = (v ** 2 * (chi2(k2, k1) / mu1 + mu1 ** 2 / mu2 ** 3 * chi2(k1, k2))).sum()
out.append("[3] generated-loss coefficient = %.12f (32/3 = %.12f)" % (coef, 32 / 3))
# [4]
rng = np.random.default_rng(5)
Mg = 1024; th = 2 * np.pi * np.arange(Mg) / Mg
def R_rho(l, rho):
    p = np.exp(rho * l); p /= p.mean()
    return (p * np.log(p)).mean(), p
worst_id, worst_bd = 0.0, 0.0
for _ in range(50):
    l1 = sum(rng.normal(0, 1 / k) * np.cos(k * th + rng.uniform(0, 6.3)) for k in range(1, 4))
    l2 = l1 + sum(rng.normal(0, 0.3 / k) * np.cos(k * th + rng.uniform(0, 6.3)) for k in range(1, 4))
    rho = rng.uniform(0.05, 1)
    sdir = l2 - l1
    e = 1e-6
    num = (R_rho(l1 + e * sdir, rho)[0] - R_rho(l1 - e * sdir, rho)[0]) / (2 * e)
    p = R_rho(l1, rho)[1]
    cov = (p * l1 * sdir).mean() - (p * l1).mean() * (p * sdir).mean()
    worst_id = max(worst_id, abs(num - rho ** 2 * cov))
    Mosc = max(np.ptp(l1), np.ptp(l2))
    lhs = abs(2 * R_rho(l2, rho)[0] - 2 * R_rho(l1, rho)[0])
    rhs = 2 * rho ** 2 * Mosc * np.abs(sdir).max()
    worst_bd = max(worst_bd, lhs / rhs)
out.append("[4] max |dR/deps - rho^2 Cov| = %.3e ; max ratio |c(1)-c(-1)| / (2 rho^2 M ||dl||) = %.4f (<= 1)" % (worst_id, worst_bd))
# [5]
n = 4
phi = rng.uniform(0, 2 * np.pi, n)
def rep(ph):
    a = np.empty(n); a[1:] = -ph[1:]; a[0] = ph[1:].sum()
    return np.mod(ph + a, 2 * np.pi)
b = rng.normal(0, 1, n); b -= b.mean()            # element of H_I (sum zero)
out.append("[5] non-symmetric representative: phases %s vs after H_I rotation %s"
           % (np.round(rep(phi), 12), np.round(rep(np.mod(phi + b, 2 * np.pi)), 12)))
# [6] limits of block rows as rows -> 1/6
Bl = np.full((6, 6), 1 / 6)
br = lambda I, J: Bl[np.ix_(I, J)].sum() / len(I)
out.append("[6] first-stage contrast -> %.4f ; core-parent contrast vs third pair -> %.4f ; two-block rows -> %.4f, %.4f"
           % (br([0], [1]) - br([0], [2, 3, 4, 5]), br([0, 1], [2, 3]) - br([0, 1], [4, 5]),
              br([0, 1, 2, 3], [4, 5]), br([4, 5], [0, 1, 2, 3])))
print("\n".join(out))
