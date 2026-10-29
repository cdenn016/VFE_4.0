"""XC3-SCOPE miscellaneous exact checks.
(a) Gaussian-feature check: mean-field (reverse-KL product) means differ from exact posterior means
    for a non-Gaussian (binary Ising pair) target, as 18:125 and 17:483 state and companion c18:83 omits.
(b) appendices/interacting_memory binary two-way exchange: exact Fraction arithmetic for
    eq:imem:binary-first-posterior ... eq:imem:binary-repeated-samples and the parity collision."""
from fractions import Fraction as Fr
import itertools, math

# (a) ----------------------------------------------------------------------------
def ising_pair(hx=0.3, hy=-0.2, J=0.9):
    states = [(x, y) for x in (-1, 1) for y in (-1, 1)]
    w = {s: math.exp(hx*s[0] + hy*s[1] + J*s[0]*s[1]) for s in states}; Z = sum(w.values())
    ex = sum(s[0]*w[s] for s in states)/Z; ey = sum(s[1]*w[s] for s in states)/Z
    mx, my = 0.0, 0.0
    for _ in range(2000):
        mx = math.tanh(hx + J*my); my = math.tanh(hy + J*mx)
    return (ex, ey), (mx, my)
print("(a) exact means vs mean-field means (non-Gaussian target):", ising_pair())

# (b) ----------------------------------------------------------------------------
q = Fr(3, 4)
def bern(val, cond, p=q):          # P(val | cond) with P(val == cond) = p
    return p if val == cond else 1 - p
def prep(rho=None):
    """joint over (x0,x1,a0,a1,y0,y1,d0,d1); rho=None: P(X1=X0)=3/4; else parity model with P(B=0)=rho."""
    P = {}
    for x0, x1, a0, a1, y0, y1, d0, d1 in itertools.product((0, 1), repeat=8):
        if rho is None:
            p = Fr(1, 2)*bern(x1, x0)
            p *= bern(d0, x0)*bern(d1, x1)
        else:
            b = x0 ^ x1; p = Fr(1, 2)*(rho if b == 0 else 1 - rho)
            p *= (Fr(1, 2) if d0 == 0 else 0)  # D0 uninformative: point mass at 0
            p *= bern(d1, x1)
        p *= bern(a0, x0)*bern(a1, x1)*bern(y0, 1 - x0)*bern(y1, 1 - x1)
        if p: P[(x0, x1, a0, a1, y0, y1, d0, d1)] = p
    return P
def cond(P, pred):
    tot = sum(p for s, p in P.items() if pred(s)); return {s: p/tot for s, p in P.items() if pred(s)}, tot
def prob(P, f): return sum(p for s, p in P.items() if f(s))
X0, X1, A0, A1, Y0, Y1, D0, D1 = range(8)
P = prep()
def k1(d1, P):          # sender 1 samples Y1 from P(Y1 | D1=d1, history encoded in P)
    Pd, _ = cond(P, lambda s: s[D1] == d1); return prob(Pd, lambda s: s[Y1] == 1)
k10 = {d: k1(d, P) for d in (0, 1)}
print("(b) k_{1,0}(1|0), k_{1,0}(1|1):", k10[0], k10[1])
# joint with first message H1=1
P1 = {s: p*k10[s[D1]] for s, p in P.items()}
tot1 = sum(P1.values()); P1 = {s: p/tot1 for s, p in P1.items()}
print("    P(H1=1) =", tot1)
for d0 in (1, 0):
    Pd, _ = cond(P1, lambda s, d0=d0: s[D0] == d0)
    print("    P(X0=1 | D0=%d, H1=1) =" % d0, prob(Pd, lambda s: s[X0] == 1))
k0 = {}
for d0 in (0, 1):
    Pd, _ = cond(P1, lambda s, d0=d0: s[D0] == d0); k0[d0] = prob(Pd, lambda s: s[Y0] == 1)
print("    reply k_{0,1}(1|D0=0,H1=1), k_{0,1}(1|D0=1,H1=1):", k0[0], k0[1])
lam1 = [prob(P1, lambda s, a=a, b=b: (s[D0], s[D1]) == (a, b)) for a in (0, 1) for b in (0, 1)]
P2 = {s: p*k0[s[D0]] for s, p in P1.items()}; tot2 = sum(P2.values()); P2 = {s: p/tot2 for s, p in P2.items()}
lam2 = [prob(P2, lambda s, a=a, b=b: (s[D0], s[D1]) == (a, b)) for a in (0, 1) for b in (0, 1)]
lam0 = [prob(P, lambda s, a=a, b=b: (s[D0], s[D1]) == (a, b)) for a in (0, 1) for b in (0, 1)]
print("    lambda0 =", lam0, " lambda1 =", lam1, " P(H2=1|H1=1) =", tot2, " lambda2 =", lam2)
Pd, _ = cond(P2, lambda s: s[D1] == 0)
print("    P(D0=1|D1=0,H1=H2=1) =", prob(Pd, lambda s: s[D0] == 1), " P(X1=1|D1=0,H1=H2=1) =", prob(Pd, lambda s: s[X1] == 1))
# repeated samples by agent 1 before reply
Pr = {s: p*k10[s[D1]]**2 for s, p in P.items()}
for x1 in (0, 1):
    Px, _ = cond(P, lambda s, x1=x1: s[X1] == x1)
    print("    P(H1=H2=1 | X1=%d) =" % x1, sum(p*k10[s[D1]]**2 for s, p in Px.items()))
Pc, _ = cond(Pr, lambda s: s[D0] == 1)
print("    P(X0=1 | D0=1, H1=H2=1) =", prob(Pc, lambda s: s[X0] == 1))
# squared once-marginalized likelihoods 9/16, 7/16 -> conditional on X1, then X0
lx1 = {0: Fr(9, 16)**2, 1: Fr(7, 16)**2}
num = sum(Fr(1, 2)*bern(x1, 1)*bern(1, 1)*lx1[x1] for x1 in (0, 1))  # placeholder not used
# compute P(X0=1|D0=1) with likelihood L(x0)=sum_x1 P(x1|x0) lx1[x1]
L = {x0: sum(bern(x1, x0)*lx1[x1] for x1 in (0, 1)) for x0 in (0, 1)}
post = (Fr(1, 2)*bern(1, 1)*L[1])/(Fr(1, 2)*bern(1, 1)*L[1] + Fr(1, 2)*bern(1, 0)*L[0])
print("    squared once-marginalized likelihood gives P(X0=1|D0=1) =", post)
# parity collision
for rho in (Fr(1, 3),):
    Pp = prep(rho)
    k = {d: k1(d, Pp) for d in (0, 1)}
    for C in (0, 1):
        Pc, _ = cond(Pp, lambda s, C=C: (s[X0] ^ s[X1]) == C)
        PH = {s: p*k[s[D1]] for s, p in Pc.items()}; t = sum(PH.values()); PH = {s: p/t for s, p in PH.items()}
        print("    parity C=%d: P(H=1|C)=%s P(X0=1|C,H=1)=%s P(A0=1|C,H=1)=%s P(Y0=1|C,H=1)=%s" %
              (C, t, prob(PH, lambda s: s[X0] == 1), prob(PH, lambda s: s[A0] == 1), prob(PH, lambda s: s[Y0] == 1)))
