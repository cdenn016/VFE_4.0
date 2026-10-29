"""GAP09-S1 s2: the separated form of the per-agent free energy can be infinity minus infinity
at the exact optimum, while eq:agent-free-energy gives a finite value.

Per-agent model with trivial presentation sector and the self source only (so the product
recognition family contains the exact posterior).  Content k = u in (0,1), evaluator
G = Uniform(0,1), record O | U=u ~ N(0, sigma(u)^2) with sigma(u) = u log(u)^2 / sqrt(2 pi)
on (0, 1/e] and sigma(1/e) on (1/e, 1) (the D06-PRED-04 kernel).  At o = 0 the record density
is l(0|u) = 1/(u log(u)^2) on (0, 1/e] and e on (1/e, 1).

Checks (sympy, exact):
 (1) Z = P(o=0) = int_0^1 l(0|u) du = e.
 (2) int_0^{1/e} l log l du diverges: with s = -log u it is int_1^S (1/s - 2 log s / s^2) ds,
     which grows like log S.
 (3) Hence KL(Pi_0 || G) = +inf, the record term E_{Pi_0}[-log l] = -inf, the separated
     expression of eq:dyn-per-agent-free-energy is inf - inf at q = Pi_0, while
     F(Pi_0; 0) = KL(Pi_0 || Pi_0) - log Z = -1.
 (4) Donsker-Varadhan control fails only because KL(Q||P^0) is infinite: for the truncated
     posteriors Q_eps = Pi_0 restricted to (eps, 1), KL(Q_eps || G) is finite, the record term
     is finite, the DV bound E[(log l)^+] <= KL(Q_eps||G) + log(1+Z) holds, and the split
     F(Q_eps) = KL(Q_eps||G) + E[-log l] agrees with KL(Q_eps||Pi_0) - log Z.
"""
import sympy as sp
import mpmath as mp

u, s, S = sp.symbols('u s S', positive=True)
l_small = 1 / (u * sp.log(u) ** 2)
Z = sp.integrate(l_small, (u, 0, sp.exp(-1))) + sp.integrate(sp.E, (u, sp.exp(-1), 1))
print('(1) Z = P(o=0) =', sp.simplify(Z))

inner = sp.integrate(1 / s - 2 * sp.log(s) / s ** 2, (s, 1, S))
print('(2) int_1^S (1/s - 2 log s/s^2) ds =', sp.simplify(inner), '; limit S->oo:', sp.limit(inner, S, sp.oo))

mp.mp.dps = 30
Zf = mp.e
def l(uu):
    return 1 / (uu * mp.log(uu) ** 2) if uu <= mp.e ** -1 else mp.e

rows = []
for eps in [mp.mpf('1e-3'), mp.mpf('1e-8'), mp.mpf('1e-20'), mp.mpf('1e-60')]:
    # Q_eps = Pi_0 restricted to (eps, 1) and renormalized
    mass = mp.quad(lambda uu: l(uu) / Zf, [eps, mp.e ** -1, 1])
    q = lambda uu: l(uu) / (Zf * mass)
    klG = mp.quad(lambda uu: q(uu) * mp.log(q(uu)), [eps, mp.e ** -1, 1])          # KL(Q||Unif)
    rec = mp.quad(lambda uu: -q(uu) * mp.log(l(uu)), [eps, mp.e ** -1, 1])         # E_Q[-log l]
    pos = mp.quad(lambda uu: q(uu) * max(mp.log(l(uu)), 0), [eps, mp.e ** -1, 1])  # E_Q[(log l)^+]
    klPi = mp.quad(lambda uu: q(uu) * mp.log(q(uu) / (l(uu) / Zf)), [eps, mp.e ** -1, 1])
    split = klG + rec
    direct = klPi - mp.log(Zf)
    dv = klG + mp.log(1 + Zf)
    rows.append((eps, klG, rec, split, direct, pos, dv))
print('(3)-(4) eps, KL(Q_eps||G), E[-log l], split F, KL(Q_eps||Pi_0) - log Z, E[(log l)^+], DV bound')
for r in rows:
    print('  %8s  %10.4f  %11.4f  %10.6f  %10.6f  %10.4f  %10.4f' % tuple(mp.nstr(x, 8) if i == 0 else float(x) for i, x in enumerate(r)))
print('  F(Pi_0; 0) = -log Z =', float(-mp.log(Zf)), '(finite), while KL(Pi_0||G) and -E[-log l] diverge like log log(1/eps).')
