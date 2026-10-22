r"""X4-PRIOR identity checks (single-threaded, short).

(a) 2.0 geo:joint-fisher example: Fisher information of zeta_rho=(1+rho k m)/4 is 1/(1-rho^2).
(b) Earlier prop:pb-collective-gaussian-soldering (b02): the rational jets satisfy
    sum_i M_i^{-1} r_{i mu} = 0 and sum_i r_{i mu} = sigma_mu; collective isotropy is i R I_2.
(c) Source-prior learning as the unit-Fisher flow of the Gibbs-reduced row potential
    E_red(pi) = -T log sum_j pi_j exp(-D_j/T):  pi_dot = T (beta* - pi).
    Self-isolation: with D_self = 0 < D_j, d/dt log(pi_self/pi_j) > 0.
(d) 2.0 prop:dyn-jensen on a finite alphabet:
    E_{q x p}[-log k] = H(q) + KL(q || pK) + int KL(p || p^w) q(dw).
(e) 2.0 geo:retained-connection witness: coarse minimizer a = 1/3.
(f) 2.0 open:two-metrics: quotient metric dF^2/(1/I1+1/I2) vs slice (I1+I2)/4 dF^2.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(0)

# (a)
rho = sp.symbols('rho', real=True)
I = 0
for k in (-1, 1):
    for m in (-1, 1):
        p = (1 + rho * k * m) / 4
        score = sp.diff(sp.log(p), rho)
        I += p * score**2
print("(a) Fisher - 1/(1-rho^2) =", sp.simplify(I - 1 / (1 - rho**2)))

# (b)
i = sp.I
M = [sp.eye(2), sp.Matrix([[2, 0], [0, 1]]), sp.Matrix([[2, 1], [1, 2]]), sp.Matrix([[2, i], [-i, 2]])]
sig = [sp.eye(2), sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -i], [i, 0]]), sp.Matrix([[1, 0], [0, -1]])]
R = sp.Rational(1, 2) * sp.Matrix([
    [-5, 1, -1, -1], [4, -2, 4, -4], [0, 0, 2, 0], [3, -3, 3, -1],
    [4, -4, 4, 4], [-4, 4, -4, 4], [0, 0, 0, 0], [0, 0, 0, 0],
    [3, 3, 3, -3], [0, 0, 0, 0], [0, 0, 0, 0], [-3, 3, -3, 3],
    [0, 0, -6, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
maxres = 0
for mu in range(4):
    rs = [sum((R[4 * a + nu, mu] * sig[nu] for nu in range(4)), sp.zeros(2)) for a in range(4)]
    s1 = sp.simplify(sum((M[a].inv() * rs[a] for a in range(4)), sp.zeros(2)))
    s2 = sp.simplify(sum(rs, sp.zeros(2)) - sig[mu])
    maxres = max(maxres, max(abs(complex(x)) for x in list(s1) + list(s2)))
print("(b) max residual of the two jet identities:", maxres)
# isotropy: a with a M_i + M_i a^dagger = 0 for all i (real-linear in 8 real params)
avars = sp.symbols('a0:8', real=True)
A = sp.Matrix([[avars[0] + i * avars[1], avars[2] + i * avars[3]], [avars[4] + i * avars[5], avars[6] + i * avars[7]]])
eqs = []
for Mi in M:
    E = A * Mi + Mi * A.H
    for x in E:
        eqs += [sp.re(sp.expand(x)), sp.im(sp.expand(x))]
J = sp.Matrix([[sp.diff(e, v) for v in avars] for e in eqs])
print("(b) rank of collective action map (expect 7):", J.rank())
ns = J.nullspace()
print("(b) kernel basis:", [list(v) for v in ns])

# (c)
T = sp.symbols('T', positive=True)
p1, p2, p3, D1, D2, D3 = sp.symbols('p1 p2 p3 D1 D2 D3', positive=True)
pis = [p1, p2, p3]
Ds = [D1, D2, D3]
Z = sum(pj * sp.exp(-Dj / T) for pj, Dj in zip(pis, Ds))
Ered = -T * sp.log(Z)
h = [sp.diff(Ered, pj) for pj in pis]
mean_h = sum(pj * hj for pj, hj in zip(pis, h))
pidot = [-pj * (hj - mean_h) for pj, hj in zip(pis, h)]
beta = [pj * sp.exp(-Dj / T) / Z for pj, Dj in zip(pis, Ds)]
sub = {p3: 1 - p1 - p2}
res = [sp.simplify((pidot[k] - T * (beta[k] - pis[k])).subs(sub)) for k in range(3)]
print("(c) pi_dot - T(beta*-pi) =", res)
# numeric self-isolation: receiver row over (self, a, b) with D_self=0
Tn = 0.05
Dn = np.array([0.0, 0.01, 0.02])
pi = np.array([1 / 3, 1 / 3, 1 / 3])
dt = 0.01
for step in range(int(200 / Tn / dt) // 10):
    b = pi * np.exp(-Dn / Tn); b /= b.sum()
    pi = pi + dt * Tn * (b - pi)
print("(c) learned row after time %.0f (self first):" % (dt * step), np.round(pi, 4))

# (d)
nv, nw = 3, 4
for trial in range(3):
    p = rng.dirichlet(np.ones(nv)); a = rng.dirichlet(np.ones(nw))
    k = rng.dirichlet(np.ones(nw), size=nv)  # k[v,w], rows normalized in w (reference = counting)
    b = p @ k
    lhs = -(a[None, :] * p[:, None] * np.log(k)).sum()
    H = -(a * np.log(a)).sum()
    KLq = (a * np.log(a / b)).sum()
    pw = (k * p[:, None]) / b[None, :]  # pw[v,w]
    KLp = sum(a[w] * (p * np.log(p / pw[:, w])).sum() for w in range(nw))
    print("(d) jensen identity residual:", lhs - (H + KLq + KLp))

# (e)
aa = sp.symbols('a', real=True)
cost = sp.Rational(1, 2) * (1 + aa)**2 + (-1 + aa)**2
print("(e) coarse minimizer:", sp.solve(sp.diff(cost, aa), aa), " fine minimizer:", sp.solve(sp.diff((1 + aa)**2 + (aa - 1)**2, aa), aa))

# (f)
I1, I2, th1, F = sp.symbols('I1 I2 th1 Fdot', positive=True)
c = I1 * th1**2 + I2 * (F - th1)**2
t1 = sp.solve(sp.diff(c, th1), th1)[0]
print("(f) quotient metric - F^2/(1/I1+1/I2):", sp.simplify(c.subs(th1, t1) - F**2 / (1 / I1 + 1 / I2)))
