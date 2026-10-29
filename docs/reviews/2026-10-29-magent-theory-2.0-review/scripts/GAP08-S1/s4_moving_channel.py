r"""GAP08-S1 script 4: chapters/02_geometry.tex:316-332, geo:moving-channel and its sign claim.

(a) Exact rational check of
        I(Ybar) = I(Y) + E I(K_theta(Y, .)) - E Cov(l + b | Ybar)
    for a two-parameter exponential-family input on three points and a moving softmax channel
    with three outputs, at theta = 0 (all quantities rational).  I(Ybar) is computed directly from
    the derivative of the output marginal, not from the formula.
(b) The same identity for a continuous linear-Gaussian moving channel, Y ~ N(theta, 1),
    Ybar | Y ~ N(theta Y, 1), symbolically in theta (sympy), with the conditional covariance of the
    quadratic score computed from the Gaussian conditional law of Y given Ybar.
(c) The logistic example of :332 and a fixed noisy channel (opposite sign).
"""
import sympy as sp
from sympy import Rational as Q
import random

random.seed(5)

print("== (a) finite, two parameters, exact ==")
ny, nb = 3, 3
p0raw = [random.randint(1, 5) for _ in range(ny)]
p0 = [Q(r, sum(p0raw)) for r in p0raw]
k0 = []
for y in range(ny):
    raw = [random.randint(1, 5) for _ in range(nb)]
    k0.append([Q(r, sum(raw)) for r in raw])
a = [[Q(random.randint(-4, 4), 3) for _ in range(2)] for _ in range(ny)]          # input statistics
c = [[[Q(random.randint(-4, 4), 3) for _ in range(2)] for _ in range(nb)] for _ in range(ny)]  # channel statistics
th = sp.symbols('th1 th2')


def P(y, t):
    num = [p0[yy] * sp.exp(t[0] * a[yy][0] + t[1] * a[yy][1]) for yy in range(ny)]
    return num[y] / sum(num)


def Kk(y, yb, t):
    num = [k0[y][bb] * sp.exp(t[0] * c[y][bb][0] + t[1] * c[y][bb][1]) for bb in range(nb)]
    return num[yb] / sum(num)


pbar = [sum(P(y, th) * Kk(y, yb, th) for y in range(ny)) for yb in range(nb)]
at0 = {th[0]: 0, th[1]: 0}
grad = [[sp.nsimplify(sp.diff(pbar[yb], th[k]).subs(at0)) for k in range(2)] for yb in range(nb)]
pb0 = [sp.nsimplify(pbar[yb].subs(at0)) for yb in range(nb)]
I_out = sp.Matrix(2, 2, lambda i, j: sum(grad[yb][i] * grad[yb][j] / pb0[yb] for yb in range(nb)))
# scores at theta = 0
abar = [sum(p0[y] * a[y][k] for y in range(ny)) for k in range(2)]
ell = [[a[y][k] - abar[k] for k in range(2)] for y in range(ny)]
cbar = [[sum(k0[y][bb] * c[y][bb][k] for bb in range(nb)) for k in range(2)] for y in range(ny)]
bsc = [[[c[y][yb][k] - cbar[y][k] for k in range(2)] for yb in range(nb)] for y in range(ny)]
I_in = sp.Matrix(2, 2, lambda i, j: sum(p0[y] * ell[y][i] * ell[y][j] for y in range(ny)))
EIK = sp.Matrix(2, 2, lambda i, j: sum(p0[y] * k0[y][yb] * bsc[y][yb][i] * bsc[y][yb][j]
                                       for y in range(ny) for yb in range(nb)))
ECov = sp.zeros(2, 2)
for yb in range(nb):
    post = [p0[y] * k0[y][yb] / pb0[yb] for y in range(ny)]
    s = [[ell[y][k] + bsc[y][yb][k] for k in range(2)] for y in range(ny)]
    m = [sum(post[y] * s[y][k] for y in range(ny)) for k in range(2)]
    cov = sp.Matrix(2, 2, lambda i, j: sum(post[y] * (s[y][i] - m[i]) * (s[y][j] - m[j]) for y in range(ny)))
    ECov += pb0[yb] * cov
    # conditional mean of the joint score equals the output score
    out_score = [grad[yb][k] / pb0[yb] for k in range(2)]
    assert all(sp.simplify(m[k] - out_score[k]) == 0 for k in range(2))
print("output score = E[l + b | Ybar]: exact at every output")
print("E[b | Y] for each y:", [[sum(k0[y][yb] * bsc[y][yb][k] for yb in range(nb)) for k in range(2)] for y in range(ny)])
print("I(Ybar) direct =", I_out.tolist())
print("I(Y) + E I(K) - E Cov(l + b | Ybar) =", (I_in + EIK - ECov).tolist())
print("residual matrix:", sp.simplify(I_out - (I_in + EIK - ECov)).tolist())
print("eigenvalues of I(Ybar) - I(Y):", [sp.N(e, 6) for e in (I_out - I_in).eigenvals()])

print("\n== (b) linear-Gaussian moving channel, symbolic ==")
t = sp.Symbol('theta', real=True)
z1, z2 = sp.symbols('z1 z2', real=True)
phi = lambda z: sp.exp(-z ** 2 / 2) / sp.sqrt(2 * sp.pi)
# direct output Fisher information of Ybar ~ N(theta^2, theta^2 + 1)
mu, var = t ** 2, t ** 2 + 1
I_direct = sp.simplify(sp.diff(mu, t) ** 2 / var + sp.diff(var, t) ** 2 / (2 * var ** 2))
# conditional law of Y given Ybar = z
zz = sp.Symbol('z', real=True)
s2 = 1 / (1 + t ** 2)
m = t + t * (zz - t ** 2) / (1 + t ** 2)
alpha, beta = -t, 1 + zz  # l + b = alpha Y^2 + beta Y - theta
condvar = alpha ** 2 * (2 * s2 ** 2 + 4 * m ** 2 * s2) + beta ** 2 * s2 + 2 * alpha * beta * (2 * m * s2)
# expectation over z ~ N(theta^2, theta^2 + 1) by direct integration in a standard variable
Ez = sp.simplify(sp.integrate(sp.expand(condvar.subs(zz, t ** 2 + sp.sqrt(1 + t ** 2) * z1)) * phi(z1),
                              (z1, -sp.oo, sp.oo)))
I_in_g = 1
EIK_g = 1 + t ** 2  # channel Fisher information at input y is y^2; E Y^2 = 1 + theta^2
I_formula = sp.simplify(I_in_g + EIK_g - Ez)
print("I(Ybar) direct =", I_direct)
print("I(Y) + E I(K) - E Var(l + b | Ybar) =", I_formula)
print("residual =", sp.simplify(I_direct - I_formula))
# Monte-Carlo-free check of the conditional-law step: E over (Y, Ybar) of (l+b)^2 equals I(Y) + E I(K)
Yv = t + z1
Ybv = t * Yv + z2
score = (Yv - t) + (Ybv - t * Yv) * Yv
inner = sp.integrate(sp.expand(score ** 2) * phi(z2), (z2, -sp.oo, sp.oo))
joint_fisher = sp.simplify(sp.integrate(sp.expand(inner) * phi(z1), (z1, -sp.oo, sp.oo)))
print("joint Fisher information E[(l + b)^2] =", joint_fisher, "; I(Y) + E I(K) =", sp.expand(I_in_g + EIK_g))
print("I(Ybar) - I(Y) at theta = 0, 1, 3:", [sp.nsimplify(sp.simplify((I_direct - 1).subs(t, v))) for v in (0, 1, 3)])

print("\n== (c) logistic example and a fixed channel ==")
sig = 1 / (1 + sp.exp(-t))
I_logistic = sp.simplify(sp.diff(sig, t) ** 2 / (sig * (1 - sig)))
print("constant input, Bernoulli(sigmoid(theta)) output: I =", I_logistic,
      "; minus e^-theta/(1+e^-theta)^2 =", sp.simplify((I_logistic - sp.exp(-t) / (1 + sp.exp(-t)) ** 2).rewrite(sp.exp)))
# fixed channel: Y ~ N(theta, 1), Ybar = Y + N(0, 1): I(Ybar) = 1/2 < 1 = I(Y)
print("fixed unit-noise channel on N(theta, 1): I(Ybar) = 1/2 < I(Y) = 1 (negative change)")
