r"""GAP06-S2 focus question 1(b): fus:sufficiency (chapters/06_fusion.tex:606-620) beyond Gaussian/Gamma.

Family: Bernoulli children on {0,1} (belief bit k, model bit m), n children, statistic
T = (S_k, S_m) = (sum k_j, sum m_j).  Checked:
 (1) the conditional law of the children given T is parameter free (exact, sympy);
 (2) the Fisher information of the parent law Bin(n,p) is n times that of a child;
 (3) the one-child conditional law (message kernel) restores the child law: Bin(n,p) B_n = Ber(p);
 (4) with a coupled evaluator G(k|m) = e^{J k m}/(1+e^{J m}) and model reference Ber(rho),
     the parent joint reference factorizes as Bin(n,rho)(S_m) G_n(S_k|S_m) (evaluator inherited);
 (5) the KL residual  Res(p,p') = n KL(q x s || R) - KL(Bin(n,p) x Bin(n,p') || R^(n))
     depends on (p,p') (state-dependent information loss), whereas in the lognormal family of
     fus:law-family it equals the constant (n-1) C;
 (6) semiconvexity of the square-root divergence sqrt(2 KL(Ber(p1)||Ber(p2))) in Fisher arclength
     coordinates phi = 2 arcsin sqrt(p): bounded below near the diagonal on compact interior sets,
     unbounded below as p2 -> 0 at fixed p1 (not globally semiconvex).
Single-threaded, a few seconds.
"""
import itertools
import numpy as np
import sympy as sp

print("(1)-(3) Bernoulli children, n = 3")
n = 3
p = sp.symbols('p', positive=True)
arr = list(itertools.product([0, 1], repeat=n))


def cond_law(pv):
    P = {x: pv**sum(x) * (1 - pv)**(n - sum(x)) for x in arr}
    out = {}
    for t in range(n + 1):
        Z = sum(P[x] for x in arr if sum(x) == t)
        for x in arr:
            if sum(x) == t:
                out[x] = sp.simplify(P[x] / Z)
    return out


c1 = cond_law(sp.Rational(1, 5)); c2 = cond_law(sp.Rational(2, 3)); cs = cond_law(p)
print("  conditional law parameter free (p = 1/5 vs 2/3 vs symbolic):",
      all(sp.simplify(c1[x] - c2[x]) == 0 and sp.simplify(cs[x] - c1[x]) == 0 for x in arr))
binom = [sp.binomial(n, t) * p**t * (1 - p)**(n - t) for t in range(n + 1)]
I_par = sp.simplify(sum(sp.diff(sp.log(b), p)**2 * b for b in binom))
I_child = sp.simplify(1 / (p * (1 - p)))
print("  Fisher(Bin(n,p)) / Fisher(Ber(p)) =", sp.simplify(I_par / I_child))
restored = sp.simplify(sum(binom[t] * sp.Rational(t, n) for t in range(n + 1)))
print("  restored message P(X_1 = 1) = sum_t Bin(t) * t/n =", restored)

print("\n(4)-(5) coupled evaluator; parent reference and KL residual")
J, rho = sp.symbols('J rho', positive=True)


def G(k, m, Jv):
    return sp.exp(Jv * k * m) / (1 + sp.exp(Jv * m))


def parent_laws(nn, Jv, rhov, pv, pvp):
    # recognition parent: Bin(nn,pv) x Bin(nn,pvp) on (S_k, S_m); reference parent: law of (S_k,S_m)
    arrs = list(itertools.product([0, 1], repeat=nn))
    Q = {}; R = {}
    for ks in arrs:
        for ms in arrs:
            Sk, Sm = sum(ks), sum(ms)
            w_ref = 1
            w_rec = 1
            for kj, mj in zip(ks, ms):
                w_ref *= (rhov if mj == 1 else 1 - rhov) * G(kj, mj, Jv)
                w_rec *= (pv if kj == 1 else 1 - pv) * (pvp if mj == 1 else 1 - pvp)
            R[(Sk, Sm)] = R.get((Sk, Sm), 0) + w_ref
            Q[(Sk, Sm)] = Q.get((Sk, Sm), 0) + w_rec
    return Q, R


def KL(Q, R):
    return sum(Q[k] * sp.log(Q[k] / R[k]) for k in Q if Q[k] != 0)


# evaluator inheritance: R^(n)(Sk|Sm) depends only on Sm and equals the law of
# Bin(Sm, G(1|1)) * Bin(n-Sm, G(1|0)); check for n = 3 exactly
Jv, rv = sp.Rational(7, 10), sp.Rational(2, 5)
_, R3 = parent_laws(3, Jv, rv, sp.Rational(1, 2), sp.Rational(1, 2))
ok = True
for Sm in range(4):
    marg = sum(R3[(Sk, Sm)] for Sk in range(4))
    if sp.simplify(marg - sp.binomial(3, Sm) * rv**Sm * (1 - rv)**(3 - Sm)) != 0:
        ok = False
    g1, g0 = G(1, 1, Jv), G(1, 0, Jv)
    for Sk in range(4):
        conv = sum(sp.binomial(Sm, o) * g1**o * (1 - g1)**(Sm - o) *
                   sp.binomial(3 - Sm, Sk - o) * g0**(Sk - o) * (1 - g0)**(3 - Sm - (Sk - o))
                   for o in range(0, min(Sm, Sk) + 1) if Sk - o <= 3 - Sm)
        if sp.simplify(R3[(Sk, Sm)] / marg - conv) != 0:
            ok = False
print("  parent reference = Bin(n,rho)(S_m) x G_n(S_k|S_m) with G_n the inherited evaluator:", ok)

print("  KL residual n*KL(child) - KL(parent), n = 2, J = 0.7, rho = 0.4:")
for (pv, pvp) in [(0.5, 0.5), (0.2, 0.7), (0.05, 0.05), (0.9, 0.9), (0.5, 0.1)]:
    pvS, pvpS = sp.Float(pv, 30), sp.Float(pvp, 30)
    Q1, R1 = parent_laws(1, Jv, rv, pvS, pvpS)
    Q2, R2 = parent_laws(2, Jv, rv, pvS, pvpS)
    res = 2 * KL(Q1, R1) - KL(Q2, R2)
    print("    p = %.2f, p' = %.2f : residual = %.12f" % (pv, pvp, float(res)))

# lognormal control: residual is the constant (n-1) C in log coordinates (Gaussian KL)
print("  lognormal control (fus:local-formulas): residual n*E_{0,1} - E_{0,n} for a = 2, c = 1:")
a_, c_ = 2.0, 1.0
A = np.array([[a_, -c_], [-c_, a_]])


def gaussKL(mu, S0, mu1, S1):
    k = len(mu); S1i = np.linalg.inv(S1); d = mu1 - mu
    return 0.5 * (np.trace(S1i @ S0) - k + d @ S1i @ d + np.log(np.linalg.det(S1) / np.linalg.det(S0)))


Cc = a_ - 1 - 0.5 * np.log(a_**2 - c_**2)
for mu in ([0.3, -0.2], [1.5, 0.7], [-2.0, 3.0]):
    mu = np.array(mu)
    for nn in (2, 5):
        E1 = gaussKL(mu, np.eye(2), np.zeros(2), np.linalg.inv(A))
        En = gaussKL(mu, np.eye(2) / nn, np.zeros(2), np.linalg.inv(A) / nn)
        print("    mu = %s, n = %d: residual = %.12f, (n-1)C = %.12f" % (mu, nn, nn * E1 - En, (nn - 1) * Cc))

print("\n(6) semiconvexity of sqrt(2 KL(Ber(p1)||Ber(p2))) in arclength coordinates phi = 2 arcsin sqrt p")
f1, f2 = sp.symbols('phi1 phi2', positive=True)
P1 = sp.sin(f1 / 2)**2; P2 = sp.sin(f2 / 2)**2
KLb = P1 * sp.log(P1 / P2) + (1 - P1) * sp.log((1 - P1) / (1 - P2))
dfun = sp.sqrt(2 * KLb)
H = sp.hessian(dfun, (f1, f2))
Hn = sp.lambdify((f1, f2), H, 'mpmath')
import mpmath as mp
mp.mp.dps = 30


def min_eig(a1, a2):
    M = mp.matrix(Hn(mp.mpf(a1), mp.mpf(a2)))
    tr = M[0, 0] + M[1, 1]; det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    return float(tr / 2 - mp.sqrt(tr**2 / 4 - det))


print("  check: 2KL / (phi1-phi2)^2 -> 1 near the diagonal:",
      float((2 * KLb / (f1 - f2)**2).subs({f1: 1.2, f2: 1.2 + 1e-4})))
phis = np.linspace(0.6, np.pi - 0.6, 25)
mins = [min_eig(x, y) for x in phis for y in phis if abs(x - y) > 1e-3]
print("  compact interior box phi in [0.6, pi-0.6] (p in [0.087, 0.913]), off-diagonal: min Hessian eigenvalue = %.4f" % min(mins))
near = [min_eig(x, x + dd) for x in np.linspace(0.6, np.pi - 0.7, 20) for dd in (1e-3, -1e-3, 1e-2, -1e-2)]
print("  within 1e-2 of the diagonal on that box: min eigenvalue = %.4f" % min(near))
print("  p1 = 1/2 (phi1 = pi/2), p2 -> 0:")
for p2 in (1e-2, 1e-4, 1e-6, 1e-9, 1e-12):
    ph2 = 2 * np.arcsin(np.sqrt(p2))
    print("    p2 = %.0e : min eigenvalue = %.4f" % (p2, min_eig(np.pi / 2, ph2)))
