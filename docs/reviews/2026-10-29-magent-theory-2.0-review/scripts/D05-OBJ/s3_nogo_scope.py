"""D05-OBJ s3: prop:dyn-no-global-vfe, its proof, and the scope of the informal claims.

(a) path along which E -> infinity (finite model free energies are bounded above).
(b) mixed derivative d_{x_i} d^2_{x_j}: zero for every mean-field free energy, nonzero for E.
(c) a COUNTABLE normalized model and a recognition rule Q(x) = q_i (x) q_j (x) Q'(x) (x) Q''(x),
    in which the agent laws q_i, q_j are recognition factors of the agents' content variables,
    with (1+beta) F(Q(x)) = E(x) + U' on the whole interior (0,1)^2.
(d) the same construction truncated to a FINITE model, exact on an open set.
Two agents, binary alphabet, identity channel j -> i with fixed row beta, trivial model sector,
E = KL(q_i||G_i) + KL(q_j||G_j) + beta KL(q_i||q_j) (+ constants from fixed rows).
Single-threaded, < 5 s.
"""
import numpy as np, sympy as sp
from math import log, ceil, pi as PI, tanh, exp

def klb(p, r):  # KL(Ber p || Ber r)
    return p*np.log(p/r) + (1-p)*np.log((1-p)/(1-r))
def sig(z): return 1/(1+np.exp(-z))
def logit(p): return np.log(p/(1-p))

g, h, beta = 0.3, 0.65, 0.6
print("(a) E along q_i -> delta_1, q_j -> delta_2 (u = 1 - t, x = t):")
for t in (1e-1, 1e-2, 1e-4, 1e-8):
    u, x = 1 - t, t
    print("   t = %.0e  E = %.4f" % (t, klb(u, g) + klb(x, h) + beta*klb(u, x)))

print("\n(b) mixed derivatives (q = ((1+x)/2, (1-x)/2))")
xi, xj = sp.symbols('x_i x_j', real=True)
qi = [(1+xi)/2, (1-xi)/2]; qj = [(1+xj)/2, (1-xj)/2]
Ekl = sum(qi[a]*sp.log(qi[a]/qj[a]) for a in range(2))
mixed = sp.simplify(sp.diff(Ekl, xi, 1, xj, 2))
print("   d_xi d^2_xj KL(q_i||q_j) =", mixed, "; at (0.1, 0.3):", float(mixed.subs({xi: 0.1, xj: 0.3})))
P = sp.symbols('P0:4', positive=True)
Ptab = [[P[0], P[1]], [P[2], P[3]]]
Fmf = sum(qi[a]*sp.log(qi[a]) for a in range(2)) + sum(qj[b]*sp.log(qj[b]) for b in range(2)) \
      - sum(qi[a]*qj[b]*sp.log(Ptab[a][b]) for a in range(2) for b in range(2))
print("   d_xi d^2_xj of a mean-field free energy with arbitrary positive table P:", sp.simplify(sp.diff(Fmf, xi, 1, xj, 2)))

print("\n(c) countable model: (1+beta) F(Q(x)) - E(u, x) is constant on (0,1)^2")
ap, bp = 1/(1+beta), beta/(1+beta)
cdelta = 1/(PI/tanh(PI))                      # sum_{m in Z} 1/(1+m^2) = pi coth pi
def pi_delta(m): return cdelta/(1+m*m)
def pi_theta(n): return 2.0**(-(n+1))          # n = 0, 1, 2, ...
def logpy(yi, delta):
    z = ap*logit(g) + bp*delta              # logit of p(y_i = 1 | delta)
    return -np.logaddexp(0.0, -z) if yi == 1 else -np.logaddexp(0.0, z)
def Qprime(x):
    lam = float(logit(x)); M = ceil(lam*lam + abs(lam) + 2); eps = abs(lam)/M
    s = 1 if lam >= 0 else -1
    return [(0, 1-eps), (s*M, eps)]
def F_part(u, x):
    # direct KL( Ber(u) (x) Ber(x) (x) Q'(x) || p(y_i|delta) G_j(y_j) pi_delta(delta) ), summed over the support
    tot = 0.0
    for d, w in Qprime(x):
        if w <= 0: continue
        for yi, qyi in ((1, u), (0, 1-u)):
            lpyi = logpy(yi, d)
            for yj, qyj in ((1, x), (0, 1-x)):
                pyj = h if yj == 1 else 1 - h
                Qv = qyi*qyj*w
                tot += Qv*(np.log(Qv) - lpyi - np.log(pyj) - np.log(pi_delta(d)))
    return tot
def E(u, x): return klb(u, g) + klb(x, h) + beta*klb(u, x)
xs = sig(np.linspace(-30, 30, 241)); us = sig(np.linspace(-20, 20, 9))
Lvals = np.array([[(1+beta)*F_part(u, x) - E(u, x) for x in xs] for u in us])
print("   max over x of the spread of L across nine values of u (u-independence):", float(np.max(Lvals.max(0) - Lvals.min(0))))
L = Lvals[0]
print("   L(x) on logit(x) in [-30, 30]: min %.4f max %.4f" % (L.min(), L.max()))
# analytic upper bound from the report: beta*max(log 1/h, log 1/(1-h)) + (1+beta)*[2 log(1/c) + sup_M log(2M^2)/sqrt(M) + log 2]
Ms = np.arange(2, 10**6); supM = float(np.max(np.log(2*Ms**2)/np.sqrt(Ms)))
Ubound = beta*max(log(1/h), log(1/(1-h))) + (1+beta)*(2*log(1/cdelta) + supM + log(2))
print("   analytic upper bound U on L: %.4f (sup_M log(2M^2)/sqrt(M) = %.4f)" % (Ubound, supM))
Uprime = Ubound + 1.0
def kl_mix(t, nstar):
    p = pi_theta(nstar)
    a = (1-t)*(1-p)*np.log(1-t) if t < 1 else 0.0
    return a + ((1-t)*p + t)*np.log((1-t) + t/p)
def Qsecond(target):
    nstar = int(ceil(target/log(2))) + 2
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if kl_mix(mid, nstar) < target: lo = mid
        else: hi = mid
    return 0.5*(lo+hi), nstar
def kl_Qsecond_direct(t, nstar, cutoff=400):
    tot = 0.0
    for n in range(cutoff):
        qv = (1-t)*pi_theta(n) + (t if n == nstar else 0.0)
        tot += qv*np.log(qv/pi_theta(n))
    tail = (1-t)*2.0**(-cutoff)*np.log(1-t) if t < 1 else 0.0   # sum_{n>=cutoff} (1-t) pi(n) log(1-t)
    return tot + tail
worst = 0.0
for u in us:
    for x in xs:
        Lx = (1+beta)*F_part(u, x) - E(u, x)
        target = (Uprime - Lx)/(1+beta)
        assert target > 0
        t, nstar = Qsecond(target)
        Ffull = F_part(u, x) + kl_Qsecond_direct(t, nstar)
        worst = max(worst, abs((1+beta)*Ffull - E(u, x) - Uprime))
print("   max |(1+beta) F(Q(x)) - E(u,x) - U'| over 9 x 241 interior points, U' = %.4f: %.2e" % (Uprime, worst))

print("\n(d) finite truncation: delta in {-50..50}, theta in {0..60}; exact where M(x) <= 50 and the target is reachable")
Mmax, nmax = 50, 60
cfin = 1/sum(1/(1+m*m) for m in range(-Mmax, Mmax+1)); zfin = sum(2.0**(-(n+1)) for n in range(nmax+1))
def pi_delta_f(m): return cfin/(1+m*m)
def pi_theta_f(n): return 2.0**(-(n+1))/zfin
xs_f = sig(np.linspace(-6, 6, 61)); assert all(ceil(logit(x)**2 + abs(logit(x)) + 2) <= Mmax for x in xs_f)
def F_part_f(u, x):
    tot = 0.0
    for d, w in Qprime(x):
        if w <= 0: continue
        for yi, qyi in ((1, u), (0, 1-u)):
            lpyi = logpy(yi, d)
            for yj, qyj in ((1, x), (0, 1-x)):
                pyj = h if yj == 1 else 1 - h
                Qv = qyi*qyj*w
                tot += Qv*(np.log(Qv) - lpyi - np.log(pyj) - np.log(pi_delta_f(d)))
    return tot
Lf = np.array([(1+beta)*F_part_f(0.4, x) - E(0.4, x) for x in xs_f]); Uf = Lf.max() + 0.5
worst = 0.0
for u in us:
    for x in xs_f:
        target = (Uf - ((1+beta)*F_part_f(u, x) - E(u, x)))/(1+beta)
        nstar = 50; p = pi_theta_f(nstar)
        def klf(t):
            tot = 0.0
            for n in range(nmax+1):
                qv = (1-t)*pi_theta_f(n) + (t if n == nstar else 0.0)
                tot += qv*np.log(qv/pi_theta_f(n))
            return tot
        assert klf(1.0 - 1e-15) > target
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5*(lo+hi)
            if klf(mid) < target: lo = mid
            else: hi = mid
        worst = max(worst, abs((1+beta)*(F_part_f(u, x) + klf(0.5*(lo+hi))) - E(u, x) - Uf))
print("   max |(1+beta) F - E - U'| on |logit x| <= 6, nine u values: %.2e" % worst)
