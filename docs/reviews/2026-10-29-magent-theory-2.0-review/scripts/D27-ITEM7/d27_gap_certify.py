r"""D27-ITEM7: interval-arithmetic certification of the gap equation it7:gap and of the radius it7:radius.

F(d) = rho (s - d) - 2 d w_3(d),  w_3(d) = 3 e^{-9 d^2} / (2 + 3 e^{-9 d^2}) = 3 / (3 + 2 e^{9 d^2}),
rho = 0.005, s = 1.1 (instance of Item 7). Gamma_3(d) = log(5/pi_p) - 2 log(2 + 3 e^{-9 d^2}) - 4.5 d^2,
pi_p = 0.95/11. mpmath interval arithmetic (outward rounding) with adaptive bisection; every enclosure
printed below is rigorous for the stated formula (it does not certify the reduction r(t) of it7:reduced).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
from mpmath import iv, mpf

iv.dps = 30
RHO = iv.mpf("0.005")
S = iv.mpf("1.1")
PP = iv.mpf("0.95") / 11


def F(d):
    w = 3 / (3 + 2 * iv.exp(9 * d ** 2))
    return RHO * (S - d) - 2 * d * w


def dF(d):
    e = iv.exp(9 * d ** 2)
    den = 3 + 2 * e
    w = 3 / den
    wp = -108 * d * e / den ** 2
    return -RHO - 2 * w - 2 * d * wp


def Gam(d):
    return iv.log(5 / PP) - 2 * iv.log(2 + 3 * iv.exp(-9 * d ** 2)) - iv.mpf("4.5") * d ** 2


def certify(fun, a, b, test, maxdepth=40):
    """Adaptive bisection: True if test(fun(I)) holds on every subinterval I of [a, b]."""
    stack = [(mpf(a), mpf(b), 0)]
    count = 0
    while stack:
        lo, hi, dep = stack.pop()
        count += 1
        val = fun(iv.mpf([lo, hi]))
        if test(val):
            continue
        if dep >= maxdepth:
            return False, count, (lo, hi, val)
        mid = (lo + hi) / 2
        stack.append((lo, mid, dep + 1))
        stack.append((mid, hi, dep + 1))
    return True, count, None


def pt(fun, x):
    return fun(iv.mpf(x))


neg = lambda bound: (lambda v: v.b < bound)
pos = lambda bound: (lambda v: v.a > bound)

print("F(0) =", pt(F, "0"))
r = certify(dF, "0", "0.1291", neg(0)); print("F' < 0 on [0, 0.1291]:", r[:2])
print("F(0.00455), F(0.00457):", pt(F, "0.00455"), pt(F, "0.00457"))
r = certify(F, "0.1291", "0.95804", neg(mpf("-3.2e-5"))); print("F < -3.2e-5 on [0.1291, 0.95804]:", r[:2])
r0 = certify(F, "0.12909", "0.1291", neg(mpf("-3.2e-5"))); print("F < -3.2e-5 on [0.12909, 0.1291]:", r0[:2])
r = certify(dF, "0.95804", "0.96804", pos(0)); print("F' > 0 on [0.95804, 0.96804]:", r[:2])
print("F(0.96303), F(0.96305):", pt(F, "0.96303"), pt(F, "0.96305"))
r = certify(F, "0.96804", "1.07812", pos(mpf("1.6e-5"))); print("F > 1.6e-5 on [0.96804, 1.07812]:", r[:2])
r = certify(dF, "1.07812", "1.08812", neg(0)); print("F' < 0 on [1.07812, 1.08812]:", r[:2])
print("F(1.08311), F(1.08313):", pt(F, "1.08311"), pt(F, "1.08313"))
r = certify(F, "1.08812", "1.6", neg(mpf("-1.7e-5"))); print("F < -1.7e-5 on [1.08812, 1.6]:", r[:2])
# above s both terms of F are negative; at s itself F = -2 s w(s) < 0
print("F(1.1) =", pt(F, "1.1"))
# slopes near the roots (trapping neighborhoods for |r| <= 1e-10)
print("F' on [d_m +- 1e-6]:", dF(iv.mpf(["0.004555", "0.004575"])))
print("F' on [d_u +- 1e-6]:", dF(iv.mpf(["0.963035", "0.963045"])))
print("F' on [d_s +- 1e-6]:", dF(iv.mpf(["1.083115", "1.083125"])))
# radius
print("Gamma_3(0.76849), Gamma_3(0.76851):", pt(Gam, "0.76849"), pt(Gam, "0.76851"))
print("Gamma_3(0) = -log(5 pi_p):", pt(Gam, "0"))
r = certify(Gam, "1.09", "1.11", neg(mpf("-2.5"))); print("Gamma_3 < -2.5 on [1.09, 1.11]:", r[:2])
r = certify(Gam, "0.96804", "1.6", neg(mpf("-1.5"))); print("Gamma_3 < -1.5 on [0.96804, 1.6]:", r[:2])
# slope bounds on the root brackets, used for the trapping statement in the D27 report
r = certify(dF, "1.07812", "1.08812", neg(mpf("-0.002"))); print("F' < -0.002 on [1.07812, 1.08812]:", r[:2])
r = certify(dF, "0.95804", "0.96804", pos(mpf("0.005"))); print("F' > 0.005 on [0.95804, 0.96804]:", r[:2])
r = certify(dF, "0", "0.1291", neg(mpf("-0.5"))); print("F' < -0.5 on [0, 0.1291]:", r[:2])
