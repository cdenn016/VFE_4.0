"""Fraction of det>0 Gaussian 2x2 matrices outside exp(gl(2,R)) and outside its closure
(two distinct negative eigenvalues: an open set), relevant to thm:link-held-limit scope."""
import numpy as np
rng = np.random.default_rng(0)
A = rng.standard_normal((400000, 2, 2)); A = A[np.linalg.det(A) > 0]
ev = np.linalg.eigvals(A)
real = np.all(np.abs(ev.imag) < 1e-12, axis=1)
neg = real & np.all(ev.real < 0, axis=1) & (np.abs(ev[:, 0] - ev[:, 1]) > 1e-9)
print("det>0 samples:", len(A), " fraction with two distinct negative eigenvalues (no real log, open set):", neg.mean())
