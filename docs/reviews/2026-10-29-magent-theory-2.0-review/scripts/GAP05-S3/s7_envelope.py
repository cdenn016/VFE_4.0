r"""GAP05-S3 s7: the envelope hypothesis of it2:law-cost-variation is load-bearing.

Message space (0, inf). Compared family B_eta(h) = exp(-h^eta) / Gamma(1 + 1/eta), smooth and
positive in eta. Receiver message law A(h) = c / ((1 + h)^2 log(e + h)^2). Then
KL(A || B_eta) = -H(A) + E_A h^eta + log Gamma(1 + 1/eta):
E_A h < inf (finite cost at eta = 1), but E_A h^eta = inf for eta > 1 and the one-sided derivative
E_A[h log h] + d/deta log Gamma(1 + 1/eta) is +inf. The truncated integrals show this.
"""
import mpmath as mp
mp.mp.dps = 30
dens = lambda h: 1 / ((1 + h) ** 2 * mp.log(mp.e + h) ** 2)
c = 1 / mp.quad(dens, [0, 1, 10, 100, mp.inf])
print("normalizer c = %s" % mp.nstr(c, 12))
for X in [1e2, 1e4, 1e8, 1e16, 1e32]:
    Eh = c * mp.quad(lambda h: h * dens(h), [0, 1, 10, 100, X])
    Ehlog = c * mp.quad(lambda h: h * mp.log(h) * dens(h), [0, 1, 10, 100, X])
    Eh11 = c * mp.quad(lambda h: h ** 1.1 * dens(h), [0, 1, 10, 100, X])
    print("X = %.0e: int_0^X h A = %s   int_0^X h log h A = %s   int_0^X h^1.1 A = %s"
          % (X, mp.nstr(Eh, 8), mp.nstr(Ehlog, 8), mp.nstr(Eh11, 8)))
print("E_A h converges; E_A h log h grows like log log X; E_A h^1.1 grows like a power of X.")
