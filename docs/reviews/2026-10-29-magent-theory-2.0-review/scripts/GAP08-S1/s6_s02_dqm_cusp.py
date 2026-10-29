r"""GAP08-S1 script 6: supplements/s02_geometry.tex:69 (the sentence that differentiating
KL(nu_i || Theta_# nu_j) "requires" differentiability in quadratic mean of Theta -> Theta_# nu_j).

Independent check of the panel's counterexample (D02-LINKS-07): nu with density
p(x) = exp(-|x - 1|^{1/2}) / 4 on R has infinite scale Fisher information (the scale family is
not DQM at sigma = 1), yet sigma -> KL(nu || sigma_# nu) is differentiable at sigma = 1 with
derivative 0. With sigma = 1 + eps,
    KL = log(sigma) + int p(x) (|x/sigma - 1|^{1/2} - |x - 1|^{1/2}) dx.
The truncated scale Fisher information over |x - 1| > delta is reported for delta = 1e-2 ... 1e-8;
its increment per factor 100 in delta should approach log(100)/8 (logarithmic divergence).
"""
import mpmath as mp

mp.mp.dps = 40
p = lambda x: mp.e ** (-mp.sqrt(abs(x - 1))) / 4
print("normalization:", mp.nstr(mp.quad(p, [-mp.inf, 0, 1, 2, mp.inf]), 20))


def KL(eps):
    s = 1 + eps
    f = lambda x: p(x) * (mp.sqrt(abs(x / s - 1)) - mp.sqrt(abs(x - 1)))
    J = mp.quad(f, [-mp.inf, -10, 0, 1, s, 2, 10, mp.inf])
    return mp.log(s) + J


for k in [2, 3, 4, 5]:
    eps = mp.mpf(10) ** (-k)
    for sgn in (1, -1):
        val = KL(sgn * eps)
        print(f"eps = {'+' if sgn > 0 else '-'}1e-{k}: KL = {mp.nstr(val, 6)}, KL/|eps| = {mp.nstr(val / eps, 6)}, "
              f"KL/(eps^2 log(1/eps)) = {mp.nstr(val / (eps ** 2 * mp.log(1 / eps)), 6)}")


# scale score at sigma = 1: -1 - x p'(x)/p(x), with p'/p = -(1/2) |x-1|^{-1/2} sign(x-1)
def score2(x):
    sgn = 1 if x > 1 else -1
    sc = -1 + x * mp.mpf(1) / 2 * abs(x - 1) ** mp.mpf(-0.5) * sgn
    return sc ** 2 * p(x)


prev = None
for k in [2, 4, 6, 8]:
    dl = mp.mpf(10) ** (-k)
    val = mp.quad(score2, [-mp.inf, -10, 0, 1 - dl]) + mp.quad(score2, [1 + dl, 2, 10, mp.inf])
    inc = '' if prev is None else f", increment {mp.nstr(val - prev, 6)} (log(100)/8 = {mp.nstr(mp.log(100) / 8, 6)})"
    print(f"truncated scale Fisher information over |x-1| > 1e-{k}: {mp.nstr(val, 8)}{inc}")
    prev = val
