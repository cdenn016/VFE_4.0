"""Timing probe for mpmath.iv (rigorous interval arithmetic)."""
import time
import mpmath
from mpmath import iv
print("mpmath", mpmath.__version__, "backend", mpmath.libmp.BACKEND)
iv.prec = 53
a = iv.mpf([1.0, 1.001]); b = iv.mpf([0.2, 0.21])
t0 = time.perf_counter()
n = 2000
for _ in range(n):
    w = iv.exp(-2*b*b); v0 = iv.exp(-2*a*a)
    Z = 1 + w + v0
    r = w / Z
t1 = time.perf_counter()
print(f"per (2 exp + 6 arith) block: {(t1-t0)/n*1e6:.1f} us")
