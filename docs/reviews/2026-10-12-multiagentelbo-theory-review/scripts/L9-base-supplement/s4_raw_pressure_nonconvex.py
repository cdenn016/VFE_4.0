r"""L9 check S4: the natural log-partition density A is convex (Hessian = Fisher density), but the raw
pressure p = A - 2 log cosh h - log cosh j of eq:rg-thermo-native-pressure-conversion is not.
At j=0 (product law, signs of mean b): H_jj = 1 + 4 b^6 - 5 b^8 per cell in the bulk, so
d^2 p / dj^2 = 4 b^6 - 5 b^8 < 0 for b^2 > 4/5.  Checked with the exact dual transfer matrix (S2).
"""
import itertools
import numpy as np
def logZ_transfer(L, b, t):
    n = L - 1
    states = np.array(list(itertools.product([0, 1], repeat=n)))
    def row_w(s):
        return t**s.sum() * b**int(np.sum(s[1:] != s[:-1])) * b**int(s[0]) * b**int(s[-1])
    rw = np.array([row_w(s) for s in states])
    T = b**(states[:, None, :] != states[None, :, :]).sum(-1)
    v = (b**states.sum(1)) * rw; ls = 0.0
    for _ in range(n - 1):
        v = (v @ T) * rw; m = v.max(); v /= m; ls += np.log(m)
    return np.log((v * b**states.sum(1)).sum()) + ls
for b in (0.5, 0.9, 0.95):
    h = np.arctanh(b)
    for L in (8, 11):
        N = L * L
        p = lambda j: logZ_transfer(L, b, np.tanh(j)) / N
        e = 1e-3
        d2 = (p(2 * e) - 2 * p(e) + p(0.0)) / e**2      # one-sided at j=0
        print(f"b={b} L={L}: d2p/dj2 ~ {d2:+.4f}   bulk prediction 4b^6-5b^8 = {4*b**6-5*b**8:+.4f}")
