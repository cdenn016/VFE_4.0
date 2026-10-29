"""D05-OBJ s7: the absorption construction behind D05-OBJ-01.

Model P = Unif(K_i) (x) Unif(K_j) (x) pi_theta with pi_theta(n) = 2^-(n+1) on n = 0, 1, 2, ...
Recognition Q(x) = q_i (x) q_j (x) Q''(x), Q''(x) = (1-t) pi_theta + t delta_{n*}, with t chosen so that
KL(Q''||pi_theta) = E(x) - KL(q_i||Unif) - KL(q_j||Unif) + 2 log 2  (nonnegative because E >= 0).
Then F(Q(x)) = E(x) + 2 log 2 on the whole interior.  Binary alphabets, E as in s3.  < 2 s.
"""
import numpy as np
from math import log, ceil
def sig(z): return 1/(1+np.exp(-z))
def klb(p, r): return p*np.log(p/r) + (1-p)*np.log((1-p)/(1-r))
g, h, beta = 0.3, 0.65, 0.6
def E(u, x): return klb(u, g) + klb(x, h) + beta*klb(u, x)
def pi_theta(n): return 2.0**(-(n+1))
def kl_mix_direct(t, nstar, cutoff=1100):
    tot = 0.0
    for n in range(cutoff):
        p = pi_theta(n)
        if p == 0.0: break
        qv = (1-t)*p + (t if n == nstar else 0.0)
        if qv > 0: tot += qv*np.log(qv/p)
    return tot
def solve_t(target):
    nstar = int(ceil(target/log(2))) + 2          # KL at t = 1 is (n*+1) log 2 > target
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = 0.5*(lo+hi)
        if kl_mix_direct(mid, nstar) < target: lo = mid
        else: hi = mid
    return 0.5*(lo+hi), nstar
worst = 0.0; maxtarget = 0.0
for lu in np.linspace(-25, 25, 11):
    for lx in np.linspace(-25, 25, 11):
        u, x = sig(lu), sig(lx)
        target = E(u, x) - klb(u, 0.5) - klb(x, 0.5) + 2*log(2)
        assert target >= 0
        t, nstar = solve_t(target); maxtarget = max(maxtarget, target)
        F = klb(u, 0.5) + klb(x, 0.5) + kl_mix_direct(t, nstar)
        worst = max(worst, abs(F - E(u, x) - 2*log(2)))
print("absorption construction: max |F(Q(x)) - E(x) - 2 log 2| over 121 interior points (logits in [-25, 25]): %.2e" % worst)
print("largest absorbed target KL(Q''||pi_theta): %.3f" % maxtarget)
