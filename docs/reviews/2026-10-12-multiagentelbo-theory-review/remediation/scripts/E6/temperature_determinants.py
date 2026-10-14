"""Recheck the active determinant bounds of ca_temperature_bounds.tex (L7-03).

Active determinants are the sheet determinants of eq:phil-selection-sheet-determinants
divided by the passive determinant 5T^5. All factors have positive coefficients, so
each normalized polynomial is increasing in T and its sup over 0<T<=1/100 is its value
at T = 1/100 (exact rational arithmetic).
"""
from fractions import Fraction as F
import math
import sympy as sp

T = sp.symbols('T', positive=True)
dM = 25*T**8*(T+8)**2*(5*T+4)**2
dE = 5*T**6*(T**3+16*T**2+77*T+112)*(125*T**3+200*T**2+85*T+8)
dN = 5*T**6*(T+4)*(5*T+2)*(T**2+12*T+28)*(25*T**2+30*T+4)
dC = 5*T**5*(T+2)*(T+4)**2*(T+6)*(5*T+2)**2*(5*T+4)
norm = {'M': sp.expand(dM/(5*T**5)/T**3), 'E': sp.expand(dE/(5*T**5)/T),
        'N': sp.expand(dN/(5*T**5)/T), 'C': sp.expand(dC/(5*T**5))}
stated_upper = {'M': 5262, 'E': 1017, 'N': 1000, 'C': 5120}
stated_lower = {'M': 5120, 'E': 896, 'N': 896, 'C': 3072}
for k, poly in norm.items():
    coeffs = sp.Poly(poly, T).all_coeffs()
    assert all(c >= 0 for c in coeffs), k
    lo = poly.subs(T, 0)
    hi = poly.subs(T, sp.Rational(1, 100))
    print(f"d_{k}: value at T->0 = {lo}, at T=1/100 = {float(hi):.4f} (exact {hi}); "
          f"stated [{stated_lower[k]}, {stated_upper[k]}] ok={lo >= stated_lower[k] and hi <= stated_upper[k]}")
