r"""G1 s3: analyticity and h^2 rate of the scaled exact binary field (12:127-139, 231).

d(x, y) = KL(Bern((1+x)/2) || Bern((1+y)/2)).  Checks that h^{-2} d(hu, hv) is even in h with
leading term (u - v)^2 / 2, prints the h^2 coefficient, and the h^2 coefficients of the decay
factor (1/h)(1 - h^2 u^2) atanh(hu) and of the receiver force
(1/h)(1 - h^2u^2)(atanh(hu) - atanh(hv)).  These give the C^1 rate of the rows (crossing
observables) and of the field on any compact set with h max|u| < 1.
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 20 python s3_h_expansion.py
"""
import sympy as sp

h, u, v = sp.symbols("h u v", real=True)
x, y = h * u, h * v
d = (1 + x) / 2 * sp.log((1 + x) / (1 + y)) + (1 - x) / 2 * sp.log((1 - x) / (1 - y))
ser = sp.series(d / h**2, h, 0, 5).removeO()
ser = sp.expand(ser)
c0 = sp.factor(ser.coeff(h, 0))
c1 = ser.coeff(h, 1)
c2 = sp.factor(ser.coeff(h, 2))
c3 = ser.coeff(h, 3)
print("h^0 coefficient of h^-2 d(hu,hv):", c0)
print("h^1 coefficient (must vanish):", sp.simplify(c1))
print("h^2 coefficient:", c2)
print("h^3 coefficient (must vanish):", sp.simplify(c3))
print("check even in h: d(-x,-y) - d(x,y) =",
      sp.simplify(d.subs(h, -h) - d))

dec = sp.series((1 - x**2) * sp.atanh(x) / h, h, 0, 5).removeO()
print("(1/h)(1-h^2u^2) atanh(hu) =", sp.expand(dec))
rec = sp.series((1 - x**2) * (sp.atanh(x) - sp.atanh(y)) / h, h, 0, 5).removeO()
print("(1/h)(1-h^2u^2)(atanh(hu)-atanh(hv)) =", sp.factor(sp.expand(rec).coeff(h, 0)),
      "+ h^2 *", sp.factor(sp.expand(rec).coeff(h, 2)))
# within a far cluster at common position U: u = U + p, v = U + q, leading h^2 U^2 relative effect
U, p, q = sp.symbols("U p q", real=True)
c2_far = sp.expand(c2.subs({u: U + p, v: U + q}))
lead = sp.factor(sp.Poly(c2_far, U).coeff_monomial(U**2))
print("h^2 coefficient within a cluster at U, coefficient of U^2:", lead,
      " (relative exponent change h^2 U^2 for small within-cluster gaps)")
