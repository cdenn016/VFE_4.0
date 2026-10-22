r"""L02-GEOM s01: Fisher examples and channel identities in chapters/02_geometry.tex.

Checks (each prints a residual):
  1. geo:joint-fisher example: zeta_rho(k,m)=(1+rho k m)/4 has Fisher 1/(1-rho^2), and the
     conditional decomposition g_s + E_s g_{q|m} reproduces it.
  2. companion Bernoulli pullback h = p'^2/(p(1-p)).
  3. geo:fisher-loss: I(P)-I(PK) = E Cov(l | Ybar) on a random finite family and fixed kernel.
  4. geo:moving-channel: I(Ybar) = I(Y) + E I(K_theta(Y,.)) - E Cov(l+b | Ybar), on (a) a Gaussian
     channel Ybar|Y ~ N(Y+beta*theta, s^2) symbolically and (b) a random finite moving channel.
  5. geo:moving-channel example: logistic output information.
  6. geo:fisher-loss Bernoulli recovery example: discarded score vanishes at theta=0.
Single-threaded, a few milliseconds.
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(20261022)

# 1. correlation example
rho = sp.symbols('rho', real=True)
tot = 0
for k in (-1, 1):
    for m in (-1, 1):
        z = (1 + rho * k * m) / 4
        tot += z * sp.diff(sp.log(z), rho) ** 2
print('1. joint Fisher - 1/(1-rho^2):', sp.simplify(tot - 1 / (1 - rho ** 2)))
cond = 0
for m in (-1, 1):
    for k in (-1, 1):
        q = (1 + rho * k * m) / 2
        cond += sp.Rational(1, 2) * q * sp.diff(sp.log(q), rho) ** 2
print('1. conditional decomposition residual:', sp.simplify(cond - tot))

# 2. Bernoulli pullback
c = sp.symbols('c', real=True)
p = sp.Function('p')(c)
h = p * sp.diff(sp.log(p), c) ** 2 + (1 - p) * sp.diff(sp.log(1 - p), c) ** 2
print('2. Bernoulli pullback residual:', sp.simplify(h - sp.diff(p, c) ** 2 / (p * (1 - p))))

# 3. fixed-kernel Fisher loss on finite spaces, d=2 parameters
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

nY, nB, d = 6, 4, 2
W0 = rng.normal(size=nY)
W1 = rng.normal(size=(nY, d))
theta0 = rng.normal(size=d)
def P(theta):
    return softmax(W0 + W1 @ theta)
K = rng.random((nY, nB)); K /= K.sum(1, keepdims=True)
eps = 1e-6
def fisher(fun, theta):
    pr = fun(theta)
    J = np.stack([(np.log(fun(theta + eps * e)) - np.log(fun(theta - eps * e))) / (2 * eps)
                  for e in np.eye(d)], axis=1)
    return J.T @ (pr[:, None] * J), J
IP, lY = fisher(P, theta0)
IPK, _ = fisher(lambda t: P(t) @ K, theta0)
p0 = P(theta0)
joint = p0[:, None] * K                      # P(y, ybar)
pb = joint.sum(0)
ECov = np.zeros((d, d))
for b in range(nB):
    w = joint[:, b] / pb[b]
    mu = w @ lY
    ECov += pb[b] * ((lY - mu).T @ (w[:, None] * (lY - mu)))
print('3. fixed-kernel loss residual (max abs):', np.abs((IP - IPK) - ECov).max(),
      ' |I(P)|=', np.abs(IP).max())
print('   eigenvalues of loss:', np.linalg.eigvalsh(IP - IPK))

# 4a. Gaussian moving channel, symbolic
beta, s = sp.symbols('beta s', positive=True)
IY = 1
Ichan = beta ** 2 / s ** 2
# Y|Ybar has variance s^2/(1+s^2); l+b = Y(1-beta/s^2) + f(Ybar)
ECov_g = (1 - beta / s ** 2) ** 2 * s ** 2 / (1 + s ** 2)
IYbar = (1 + beta) ** 2 / (1 + s ** 2)
print('4a. Gaussian moving-channel residual:', sp.simplify(IY + Ichan - ECov_g - IYbar))

# 4b. random finite moving channel, scalar parameter
V0 = rng.normal(size=(nY, nB)); V1 = rng.normal(size=(nY, nB))
def Kth(t):
    L = V0 + t * V1
    E = np.exp(L - L.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)
t0 = 0.3
def Pt(t):
    return softmax(W0 + W1[:, 0] * t)
def out(t):
    return Pt(t) @ Kth(t)
d1 = lambda f, t: (f(t + eps) - f(t - eps)) / (2 * eps)
lYs = d1(lambda t: np.log(Pt(t)), t0)
bsc = d1(lambda t: np.log(Kth(t)), t0)               # b(y, ybar)
pY = Pt(t0); Kt = Kth(t0)
IYs = pY @ lYs ** 2
Ich = sum(pY[y] * Kt[y] @ bsc[y] ** 2 for y in range(nY))
jt = pY[:, None] * Kt
pbar = jt.sum(0)
tot_score = lYs[:, None] + bsc
ECv = 0.0
for b in range(nB):
    w = jt[:, b] / pbar[b]
    mu = w @ tot_score[:, b]
    ECv += pbar[b] * (w @ (tot_score[:, b] - mu) ** 2)
Iout = pbar @ d1(lambda t: np.log(out(t)), t0) ** 2
print('4b. finite moving-channel residual:', abs(IYs + Ich - ECv - Iout), ' I(Ybar)=', Iout,
      ' I(Y)=', IYs)
print('    E[b|Y] max abs:', np.abs((Kt * bsc).sum(1)).max())

# 5. logistic output
th = sp.symbols('theta', real=True)
pp = 1 / (1 + sp.exp(-th))
Iout_s = pp * sp.diff(sp.log(pp), th) ** 2 + (1 - pp) * sp.diff(sp.log(1 - pp), th) ** 2
print('5. logistic output information residual:',
      sp.simplify(Iout_s - sp.exp(-th) / (1 + sp.exp(-th)) ** 2))

# 6. Bernoulli recovery example
p2 = sp.Rational(1, 2) + th ** 2 / 4
score2 = sp.diff(sp.log(p2), th)
print('6. discarded score at 0:', score2.subs(th, 0),
      '; discarded Fisher at theta:', sp.simplify(p2 * sp.diff(sp.log(p2), th) ** 2
                                                  + (1 - p2) * sp.diff(sp.log(1 - p2), th) ** 2))
