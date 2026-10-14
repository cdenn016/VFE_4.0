r"""Scale of the seeded lower bound of prop:phil-seeded-contrast-formation(b).

The first display is (e^{-1/4}/16) * sum_{+-} N(m0, I_12){|x -+ x0| < rho}, rho = 1/100.
By Anderson's inequality each ball mass is at most the centered mass P(chi2_12 <= rho^2),
so the display is at most (e^{-1/4}/16) * 2 * P(chi2_12 <= rho^2) for every m0.
"""
from mpmath import mp, mpf, exp, gammainc, gamma, pi

mp.dps = 30
rho = mpf(1) / 100
k = 12
# P(chi2_k <= r2) = regularized lower incomplete gamma P(k/2, r2/2)
central = gammainc(k / 2, 0, rho**2 / 2, regularized=True)
sup_bound = exp(mpf(-1) / 4) / 16 * 2 * central
best_single = exp(mpf(-1) / 4) / 16 * central
# second display at m0 = x0: e^{-1/4} rho^12/737280 * exp(-rho^2/2)
second = exp(mpf(-1) / 4) * rho**12 / 737280 * exp(-rho**2 / 2)
# check 737280 = 16 * 46080 and 1/46080 = (2 pi)^{-6} vol(B_12(1)) = (2pi)^-6 pi^6/6!
vol = pi**6 / 720
print("central ball mass P(chi2_12<=1e-4) =", central)
print("display at m0=x0 (one ball)        =", best_single)
print("sup over m0 of first display  <=   ", sup_bound)
print("second display at m0=x0            =", second)
print("(2pi)^-6 * pi^6/6! * 46080        =", (2 * pi) ** -6 * vol * 46080)
