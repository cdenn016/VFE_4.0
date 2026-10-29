"""D03-GEOM s01: regular tier checks (geo:regular-family, geo:joint-fisher, geo:transported-kl remark).

(a) Binary correlation law zeta_rho(k,m) = (1+rho k m)/4: joint Fisher, marginals,
    conditional decomposition geo:conditional-fisher, and the unweighted sum of
    squared scores (the manuscript's wording "summing its squared score").
(b) A DQM location family with parameter-dependent support whose relative entropy
    KL(P_0 || P_h) is infinite for h>0: f(x) = x^2 e^{-x}/2 on x>0.  Shows that
    DQM alone does not give the second-order KL expansion (remark after
    geo:transported-kl) while the Hellinger expansion h^2 I/8 still holds.
"""
import sympy as sp
import mpmath as mp

print("== (a) binary correlation example")
rho = sp.symbols('rho', real=True)
outcomes = [(k, m) for k in (-1, 1) for m in (-1, 1)]
zeta = {o: (1 + rho*o[0]*o[1])/4 for o in outcomes}
score = {o: sp.diff(sp.log(zeta[o]), rho) for o in outcomes}
fisher = sp.simplify(sum(zeta[o]*score[o]**2 for o in outcomes))
print("joint Fisher:", fisher, " residual vs 1/(1-rho^2):", sp.simplify(fisher - 1/(1-rho**2)))
margk = {k: sp.simplify(sum(zeta[(k, m)] for m in (-1, 1))) for k in (-1, 1)}
margm = {m: sp.simplify(sum(zeta[(k, m)] for k in (-1, 1))) for m in (-1, 1)}
print("k-marginal:", margk, " m-marginal:", margm)
# conditional decomposition: s = m-marginal (uniform, zero score), q(k|m) = zeta/s
cond_fisher = 0
for m in (-1, 1):
    s = margm[m]
    qkm = {k: sp.simplify(zeta[(k, m)]/s) for k in (-1, 1)}
    gq = sum(qkm[k]*sp.diff(sp.log(qkm[k]), rho)**2 for k in (-1, 1))
    cond_fisher += s*gq
print("marginal + integrated conditional Fisher residual:", sp.simplify(cond_fisher - fisher))
unweighted = sp.simplify(sum(score[o]**2 for o in outcomes))
print("unweighted sum of squared scores:", sp.factor(unweighted),
      " (differs from Fisher; equals 4(1+rho^2)/(1-rho^2)^2):",
      sp.simplify(unweighted - 4*(1+rho**2)/(1-rho**2)**2))

print("\n== (b) DQM location family with infinite KL")
mp.mp.dps = 40
f = lambda x: x**2*mp.e**(-x)/2 if x > 0 else mp.mpf(0)
fp = lambda x: (x - x**2/2)*mp.e**(-x) if x > 0 else mp.mpf(0)
I = mp.quad(lambda x: fp(x)**2/f(x), [0, 1, 10, mp.inf])
print("normalization:", mp.quad(f, [0, mp.inf]), " Fisher information I =", I)
for h in [mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.001')]:
    # squared Hellinger H^2 = int (sqrt f(x-h) - sqrt f(x))^2 dx
    H2 = mp.quad(lambda x: (mp.sqrt(f(x - h)) - mp.sqrt(f(x)))**2, [0, h, 1, 10, mp.inf])
    massgap = mp.quad(f, [0, h])  # P_0 mass where P_h has zero density
    print(f"h={mp.nstr(h,3)}: H^2/(h^2 I/4) = {mp.nstr(H2/(h**2*I/4), 8)};"
          f" P_0(0<x<h) = {mp.nstr(massgap, 5)} > 0, so KL(P_0||P_h) = +inf")
print("Interpretation: Hellinger (DQM) expansion holds with ratio -> 1 while KL is infinite;")
print("DQM alone does not justify the second-order KL expansion.")
