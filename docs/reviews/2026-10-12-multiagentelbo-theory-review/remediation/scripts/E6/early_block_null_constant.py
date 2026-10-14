"""L7-05: sup over m0 of c(m0) = e^{-1/4} rho^12/737280 * exp[-(|x-m0|+rho)^2/2], rho = 1/100.
The exponential factor is at most exp(-rho^2/2) < 1 (attained at m0 = x), so
c(m0) <= e^{-1/4} rho^12 / 737280."""
from mpmath import mp, mpf, exp
mp.dps = 30
rho = mpf(1) / 100
ub = exp(mpf(-1) / 4) * rho**12 / 737280
sup = ub * exp(-rho**2 / 2)
print("e^{-1/4} rho^12/737280 =", ub)
print("sup_m0 c(m0)           =", sup)
print("ub < 1.06e-30:", ub < mpf("1.06e-30"))
