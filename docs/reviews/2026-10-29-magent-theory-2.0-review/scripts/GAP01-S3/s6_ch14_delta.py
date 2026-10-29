r"""GAP01-S3 s6: explicit (eps, delta) constants for the chapter 14 grouping import (14:71-80).

Transmitted hidden sector B_i = delta b_i: relative to c_i, the peer log-weights become
  -KL(Bern(L_i)||Bern(L_j)) - KL(vM(B_i)||vM(B_j)),  self: -KL(Bern(L_i)||Bern(0)) - KL(vM(B_i)||Haar) + J t_i - log I_0(J),
with KL(vM(B)||vM(B')) <= |B-B'|^2/2 and KL(vM(B)||Haar) <= |B|^2/2 (Hessian of log I_0(|.|) bounded by 1), and
max_i |B_i(t)| nonincreasing under dB_i/dt = -B_i + sum_{j!=i} beta_ij B_j.  So osc gains at most 2 (delta bmax)^2.
Same Lyapunov/shadowing argument as s1/s2; the null box needs delta bmax <= 1/10 and eps <= 1/100.
"""
import numpy as np

A_PR, P_PR = 0.2, 0.4
ustar = np.array([15 / 7, 15 / 7, 5 / 7]); uin = np.array([4.0, 0.5, -1.0])
e0 = uin - ustar; c1 = np.full(3, e0.mean()); cp = e0 - c1
NS = [np.array(v, float) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [4, -1, 0], [-1, 4, 0], [0, 4, -1], [0, -1, 4])]

def Rbound(rho, eps, A0, dB):
    J = eps ** 3
    dbar = 2 * eps ** 2 * (max(A0, eps) + eps ** 3)
    Rs = []
    for i in range(3):
        Di = max([abs(ustar[i]) + rho + dbar] + [abs(ustar[i] - ustar[j]) + np.sqrt(2) * rho + 2 * dbar for j in range(3) if j != i])
        osc = eps ** 2 * Di ** 2 / 8 + 2 * J + J ** 2 / 4 + 2 * dB ** 2
        so = sum(abs(ustar[j]) for j in range(3) if j != i) + np.sqrt(2) * rho
        Rs.append(P_PR * np.expm1(osc) * so + dbar)
    return float(np.sqrt(np.sum(np.square(Rs)))), dbar

def margin(n, t):
    return n @ (ustar + np.exp(-t / 5) * c1 + np.exp(-7 * t / 5) * cp) / np.linalg.norm(n)

def entrance_time(r0, sh, dbar):
    T = 0.0
    for n in NS:
        extra = np.abs(n).sum() * dbar / np.linalg.norm(n)
        a1, ap = n @ c1, n @ cp
        if a1 <= 0 and ap <= 0:
            f = lambda t: margin(n, t) - (np.exp(-t / 5) * r0 + sh + extra)
            if f(200.0) <= 0:
                return np.inf
            if f(0.0) > 0:
                continue
            lo, hi = 0.0, 200.0
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(mid) <= 0 else (lo, mid)
            T = max(T, hi)
        else:
            if (n @ ustar + min(0, a1) + min(0, ap)) / np.linalg.norm(n) <= r0 + sh + extra:
                return np.inf
    return T

r0 = 0.05; A0 = 0.05; rho0 = np.linalg.norm(e0) + r0
print("chapter 14 grouping, r0=%.2f, |A(0)|<=%.2f, |C(0)|<=eps^3, |B_i(0)| = delta|b_i| <= dB:" % (r0, A0))
for eps in (0.005, 0.01):
    for dB in (0.0, 0.02, 0.05, 0.08, 0.1):
        Rb, db = Rbound(rho0, eps, A0, dB)
        print("   eps=%.3f dB=%.2f: Rbar=%.3e ball invariant=%s explicit T=%.4f" % (eps, dB, Rb, Rb < rho0 / 5, entrance_time(r0, 5 * Rb, db)))
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        Rb, db = Rbound(rho0, eps, A0, mid)
        lo, hi = (mid, hi) if (Rb < rho0 / 5 and np.isfinite(entrance_time(r0, 5 * Rb, db))) else (lo, mid)
    print("   eps=%.3f: sup of delta*bmax with finite explicit T = %.4f (null box additionally needs <= 0.1)" % (eps, lo))
