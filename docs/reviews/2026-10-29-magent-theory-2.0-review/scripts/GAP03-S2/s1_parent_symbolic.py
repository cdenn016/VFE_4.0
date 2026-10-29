r"""GAP03-S2 s1: symbolic proof checks for form:full-flow (tangency/natural gradient)
and form:parent (parent Fisher flow = pushforward of the fine flow) at EVERY state.

Structure of the proof being checked (see report):
  Lemma A (Bregman): for the Wishart family in coordinates x=(nu,p11,p12,p22),
      grad_x KL(W(x) || W(y)) = Hess F(x) (x - y),  F(nu,P) = log Gamma_2(nu) - nu log det P,
    so the Fisher natural gradient of KL(.||W(y)) is x - y.
  Lemma B (mean parameters): for Phi(m), m = E_x X = nu P^{-1}, with dPhi = Tr(G dm),
      grad_x Phi(m(x)) = Hess F(x) (0, -G),
    so the natural gradient is (0, -G).
  Assembly: fine and parent flows follow; intertwining is linear algebra.
Part (a), (b): fully symbolic in all variables (one agent, arbitrary targets).
Part (c): exact rational evaluation of the assembled fine natural gradient (N=3, unequal
          precisions) and parent natural gradient (several partitions); residual must be 0.
Part (d): horizontal conformality test for the sum retention (inh:type-closure-open).
Single-threaded, short.
"""
import sympy as sp
import random

nu, b = sp.symbols('nu b', positive=True)
p11, p12, p22, q11, q12, q22 = sp.symbols('p11 p12 p22 q11 q12 q22', real=True)


def logGamma2(v):
    return sp.log(sp.sqrt(sp.pi)) + sp.loggamma(v) + sp.loggamma(v - sp.Rational(1, 2))


def F(v, P):
    return logGamma2(v) - v * sp.log(P.det())


def KL(v, P, w, Q):
    """KL(W_2(v,P) || W_2(w,Q)) closed form (derived from E log det X = psi_2(v) - log det P,
    E X = v P^{-1})."""
    psi2 = sp.polygamma(0, v) + sp.polygamma(0, v - sp.Rational(1, 2))
    return ((v - w) * psi2 - logGamma2(v) + logGamma2(w)
            + w * sp.log(P.det()) - w * sp.log(Q.det())
            + v * ((Q * P.inv()).trace() - 2))


P = sp.Matrix([[p11, p12], [p12, p22]])
Q = sp.Matrix([[q11, q12], [q12, q22]])
x = [nu, p11, p12, p22]
y = [b, q11, q12, q22]

out = []

# (a) Lemma A, fully symbolic
K = KL(nu, P, b, Q)
gradK = sp.Matrix([sp.diff(K, xi) for xi in x])
HF = sp.hessian(F(nu, P), x)
resA = [sp.cancel(sp.together(sp.expand_func(e))) for e in (gradK - HF * sp.Matrix([xi - yi for xi, yi in zip(x, y)]))]
out.append("(a) Lemma A residual grad KL - HessF (x-y), symbolic: " + str(resA))

# (b) Lemma B. Symbolic in all variables for Phi linear in m with an arbitrary symmetric G0;
#     by the chain rule grad_x Phi(m(x)) = (Dm)^T grad_m Phi is linear in G = grad_m Phi at the
#     point, so the linear case is the general case pointwise. The nonlinear channel energy
#     Phi = Tr(Lam M) - N a0 log det Lam is also checked exactly at a rational point.
kap, a0, Nb = sp.symbols('kappa alpha0 N', positive=True)
o11, o12, o22 = sp.symbols('o11 o12 o22', real=True)
g11, g12, g22 = sp.symbols('g11 g12 g22', real=True)
Mo = sp.Matrix([[o11, o12], [o12, o22]])
I2 = sp.eye(2)
m = nu * P.inv()
G0 = sp.Matrix([[g11, g12], [g12, g22]])
PhiL = (G0 * m).trace()
gradPhiL = sp.Matrix([sp.diff(PhiL, xi) for xi in x])
resB = [sp.cancel(sp.together(e)) for e in (gradPhiL - HF * sp.Matrix([0, -g11, -g12, -g22]))]
out.append("(b) Lemma B residual grad Tr(G0 m(x)) - HessF (0,-G0), symbolic: " + str(resB))
Mt = m + Mo
Lam = I2 + (kap / Nb) * Mt
Phi = (Lam * Mt).trace() - Nb * a0 * sp.log(Lam.det())
gradPhi = sp.Matrix([sp.diff(Phi, xi) for xi in x])
G = Lam + (kap / Nb) * (Mt - Nb * a0 * Lam.inv())
vecB = sp.Matrix([0, -G[0, 0], -G[0, 1], -G[1, 1]])
valsB = {nu: sp.Rational(7, 5), p11: sp.Rational(3, 2), p12: sp.Rational(1, 5), p22: sp.Rational(4, 5),
         kap: 2, a0: sp.Rational(6, 5), Nb: 4, o11: 2, o12: sp.Rational(-1, 3), o22: sp.Rational(5, 2)}
resB2 = [sp.cancel(sp.expand_func(e.subs(valsB))) for e in (gradPhi - HF * vecB)]
out.append("(b) Lemma B with the channel energy, exact at a rational point: " + str(resB2))

# (c) exact assembled checks at random rational states
random.seed(20261029)


def rnd_spd():
    a = sp.Rational(random.randint(5, 30), 10)
    c = sp.Rational(random.randint(5, 30), 10)
    lim = int(9 * min(a, c))
    bb = sp.Rational(random.randint(-lim, lim), 10) if lim > 0 else sp.Integer(0)
    M = sp.Matrix([[a, bb], [bb, c]])
    assert M.det() > 0
    return M


def _block_residual(shapes, precs, targ_shapes, kapv, Nprim, a0v):
    """For a population of Wishart laws W(shapes[j], precs[j]) with targets of shape
    targ_shapes[j] (reference W(targ,I) and channel W(targ, Lam(M))), Lam = I + kap M / Nprim,
    M = sum_j shapes[j] precs[j]^{-1}, return max over j of |H_j v_j + grad_j E| where v_j is
    the claimed velocity (-2(shape - targ), -2P_j + I + C_q).  The gradient for agent j is
    computed symbolically in agent j's own variables with the others held at their exact values."""
    n = len(shapes)
    Mv = sp.zeros(2)
    for j in range(n):
        Mv += shapes[j] * precs[j].inv()
    Lv = I2 + (sp.Rational(kapv) / Nprim) * Mv
    Cq = Lv + (sp.Rational(kapv) / Nprim) * (Mv - Nprim * a0v * Lv.inv())
    worst = 0
    zs = sp.symbols('zs zu zv zw', real=True)
    Pz = sp.Matrix([[zs[1], zs[2]], [zs[2], zs[3]]])
    for j in range(n):
        Mz = Mv - shapes[j] * precs[j].inv() + zs[0] * Pz.inv()
        Lz = I2 + (sp.Rational(kapv) / Nprim) * Mz
        E = 0
        for k in range(n):
            if k == j:
                E += KL(zs[0], Pz, targ_shapes[k], I2) + KL(zs[0], Pz, targ_shapes[k], Lz)
            else:
                # other agents' own-law terms depend on agent j only through Lz
                E += KL(shapes[k], precs[k], targ_shapes[k], Lz)
        vals = {zs[0]: shapes[j], zs[1]: precs[j][0, 0], zs[2]: precs[j][0, 1], zs[3]: precs[j][1, 1]}
        g = sp.Matrix([sp.diff(E, z) for z in zs]).subs(vals)
        H = sp.hessian(F(zs[0], Pz), list(zs)).subs(vals)
        Pd = -2 * precs[j] + I2 + Cq
        v = sp.Matrix([-2 * (shapes[j] - targ_shapes[j]), Pd[0, 0], Pd[0, 1], Pd[1, 1]])
        r = [sp.cancel(sp.expand(sp.expand_func(e))) for e in (H * v + g)]
        worst = max(worst, max(abs(e) for e in r))
    return worst, Cq


def fine_check(Nagents, kapv, a0v):
    shapes = [sp.Rational(random.randint(6, 30), 10) for _ in range(Nagents)]
    precs = [rnd_spd() for _ in range(Nagents)]       # unequal precisions
    w, _ = _block_residual(shapes, precs, [a0v] * Nagents, kapv, Nagents, a0v)
    return w


for (Nag, kv, av) in [(3, 1, sp.Rational(7, 5)), (3, 5, sp.Rational(3, 2)), (2, sp.Rational(1, 3), 1)]:
    out.append("(c1) fine natural gradient, N=%d, kappa=%s, alpha0=%s, unequal precisions: max |H v + grad E| = %s"
               % (Nag, kv, av, fine_check(Nag, kv, av)))


def parent_check(ns, kapv, a0v):
    """Parent potential with primitive N = sum(ns): parent laws W(A_I, P), targets of shape
    n_I a0, channel precision Lam = I + kap M / N with the PRIMITIVE N.  Returns the residual
    of the parent natural gradient against the claimed parent ODE, and the intertwining
    residual (parent shape velocity minus summed fine shape velocities; parent total moment
    minus fine total moment; parent C_q minus fine C_q)."""
    Nprim = sum(ns)
    Pc = rnd_spd()
    alphas = [[sp.Rational(random.randint(6, 30), 10) for _ in range(n)] for n in ns]
    A = [sum(al) for al in alphas]
    wpar, Cpar = _block_residual(A, [Pc] * len(ns), [n * a0v for n in ns], kapv, Nprim, a0v)
    # fine description at the same state (common precision Pc)
    fl = [a for al in alphas for a in al]
    wfine, Cfine = _block_residual(fl, [Pc] * Nprim, [a0v] * Nprim, kapv, Nprim, a0v)
    maxint = 0
    for I, al in enumerate(alphas):
        fine_sum = sum(-2 * (a - a0v) for a in al)
        maxint = max(maxint, abs(fine_sum - (-2 * (A[I] - ns[I] * a0v))))
    maxint = max(maxint, max(abs(e) for e in (Cpar - Cfine)))
    return max(wpar, wfine), maxint


for ns in [(2, 2), (1, 2, 1), (4,), (1, 3)]:
    r1, r2 = parent_check(ns, sp.Rational(1), sp.Rational(6, 5))
    out.append("(c2) parent natural gradient, blocks %s of N=%d, kappa=1, alpha0=6/5: max |H v + grad E^c| = %s; "
               "intertwining residual (parent shape velocity - summed fine shape velocities; parent C_q - fine C_q) = %s" % (ns, sum(ns), r1, r2))
r1, r2 = parent_check((2, 2), sp.Rational(5, 2), sp.Rational(1))
out.append("(c2) parent natural gradient, blocks (2,2), kappa=5/2, alpha0=1: %s; intertwining %s" % (r1, r2))

# (d) horizontal conformality test of the sum retention on the common-precision family.
#  Fine metric on (alpha_1..alpha_n, P): sum_i [psi2'(a_i) da_i^2 - 2 da_i Tr(P^-1 U) + a_i Tr(P^-1 U P^-1 U)]
#  Horizontal vectors: da_i = s / psi2'(a_i). Parent/fine norm difference = s^2 H (psi2'(A) H - 1),
#  H = sum_i 1/psi2'(a_i). Conformality would force factor d=1 (precision directions) and psi2'(A)H = 1.
import mpmath as mp
mp.mp.dps = 30


def psi2p(a):
    return mp.psi(1, a) + mp.psi(1, a - mp.mpf(1) / 2)


worst = []
for n in [2, 3, 4]:
    for a in [0.6, 0.8, 1.0, 1.5, 2.0, 5.0, 20.0, 100.0]:
        A = n * a
        Hs = n / psi2p(a)
        worst.append((n, a, float(psi2p(A) * Hs)))
out.append("(d) conformal factor psi2'(A) * sum 1/psi2'(alpha_i) for equal shapes (n, alpha, value):")
for w in worst:
    out.append("     n=%d alpha=%6.2f  factor=%.10f" % w)
# unequal shapes
for al in [(0.6, 2.0), (1.0, 1.2), (0.7, 0.9, 3.0)]:
    A = sum(al)
    Hs = sum(1 / psi2p(mp.mpf(a)) for a in al)
    out.append("     shapes %s factor=%.10f" % (al, float(psi2p(A) * Hs)))

# symbolic check of the horizontal-norm identity for n=2
s_, tau, Qu = sp.symbols('s tau Q_U', real=True)
w1, w2, A1, A2 = sp.symbols('w1 w2 a1 a2', positive=True)   # w_i = psi2'(a_i), Wp = psi2'(a1+a2)
Wp = sp.symbols('Wp', positive=True)
da = [s_ / w1, s_ / w2]
fine = w1 * da[0]**2 + w2 * da[1]**2 - 2 * tau * (da[0] + da[1]) + (A1 + A2) * Qu
dA = da[0] + da[1]
par = Wp * dA**2 - 2 * tau * dA + (A1 + A2) * Qu
Hn = 1 / w1 + 1 / w2
out.append("(d) symbolic: parent - fine horizontal norm = %s ; expected s^2 H (Wp H - 1): diff %s"
           % (sp.factor(sp.simplify(par - fine)), sp.simplify(par - fine - s_**2 * Hn * (Wp * Hn - 1))))

for line in out:
    print(line)
