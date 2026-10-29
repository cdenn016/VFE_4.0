r"""GAP04-S4 s4: differentiability in quadratic mean of adaptive history experiments.

Toy version of the history experiment Pi_c of lp:actual-experiment / pa:joint-record-geometry.
A context parameter c sets the public law of a port, P_c(z=1) = sigma(c) (public odds L = c,
mu(z) = e^{Lz/2}/(2 cosh(L/2))).  A policy uses independent randomization to choose a number N
of fresh public queries and issues them at times 1 - 2^{-k}, k = 1..N, all before t = 1 (the
protocol of 14:84 admits independent randomization and adaptive waiting times).
History law: Pi_c(n, z_1..z_n) = P(N=n) prod_k sigma(c)^{[z_k=1]} (1-sigma(c))^{[z_k=-1]}.
Squared Hellinger distance (convention h^2 = 1 - affinity):
    h^2(c, c+d) = 1 - sum_n P(N=n) rho_1^n,   rho_1 = sqrt(p p') + sqrt((1-p)(1-p')).
DQM with Fisher information I gives h^2 = I d^2 / 8 + o(d^2).  For P(N=n) = 6/(pi^2 n^2),
h^2 = 1 - (6/pi^2) Li_2(rho_1), and h^2/d^2 grows like log(1/d): no DQM, E[score^2] = E[N] I_1 = oo.
Controls: N = 5 fixed (h^2/d^2 -> 5 I_1/8) and P(N=n) = n^-3/zeta(3) (E N finite, h^2/d^2 -> E[N] I_1/8).
Zeno control: N = oo surely gives affinity 0 for every d != 0 (Kakutani): mutually singular laws.
"""
import mpmath as mp

mp.mp.dps = 40
c0 = mp.mpf("0.3")


def sig(x):
    return 1 / (1 + mp.e ** (-x))


p0 = sig(c0)
I1 = p0 * (1 - p0)
print(f"per-query Fisher information I_1 = sigma(1-sigma) = {mp.nstr(I1, 12)}")
z3 = mp.zeta(3)
EN3 = mp.zeta(2) / z3
print(f"{'d':>8} {'heavy n^-2: h2/d2':>20} {'predicted ~(6/pi^2)(I/8)(1+log(8/(I d^2)))':>45} "
      f"{'N=5: h2/d2':>14} {'n^-3: h2/d2':>14}")
for k in range(1, 8):
    d = mp.mpf(10) ** (-k)
    p1 = sig(c0 + d)
    rho1 = mp.sqrt(p0 * p1) + mp.sqrt((1 - p0) * (1 - p1))
    h2_heavy = 1 - 6 / mp.pi ** 2 * mp.polylog(2, rho1)
    eps = 1 - rho1
    pred = 6 / mp.pi ** 2 * (I1 / 8) * (1 + mp.log(8 / (I1 * d ** 2)))
    h2_fixed = 1 - rho1 ** 5
    h2_cubic = 1 - mp.polylog(3, rho1) / z3
    print(f"{mp.nstr(d, 2):>8} {mp.nstr(h2_heavy / d ** 2, 10):>20} {mp.nstr(pred, 10):>45} "
          f"{mp.nstr(h2_fixed / d ** 2, 10):>14} {mp.nstr(h2_cubic / d ** 2, 10):>14}")
print(f"limits: N=5 -> 5 I_1/8 = {mp.nstr(5 * I1 / 8, 10)};  n^-3 -> E[N] I_1/8 = {mp.nstr(EN3 * I1 / 8, 10)}")
print(f"heavy tail: E[N] = sum 6/(pi^2 n) = oo, so E[score^2] = E[N] I_1 = oo")
d = mp.mpf("1e-3")
p1 = sig(c0 + d)
rho1 = mp.sqrt(p0 * p1) + mp.sqrt((1 - p0) * (1 - p1))
for n in (10 ** 3, 10 ** 6, 10 ** 9):
    print(f"Zeno control: affinity after n queries at d=1e-3: rho_1^n = {mp.nstr(rho1 ** n, 6)}  (n = {n})")
