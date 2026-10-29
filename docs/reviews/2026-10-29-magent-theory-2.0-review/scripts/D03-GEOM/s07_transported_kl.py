"""D03-GEOM s07: geo:transported-kl / geo:kl-pullback on a non-Gaussian family.

Von Mises laws vM(mu, kappa) on the circle, SO(2) acting by rotation (J xi = xi d_mu),
section over R^2 and a supplied abelian connection A.  Transport back along the
straight line gamma(t) = c0 + tX shifts mu by +int_0^t A(gamma') ds.
Fisher metric: g_mumu = kappa A1(kappa), g_kk = A1'(kappa) = 1 - A1/kappa - A1^2.
Closed-form KL(vM1 || vM2) = log(I0(k2)/I0(k1)) + A1(k1) (k1 - k2 cos(mu1 - mu2)).
"""
import mpmath as mp
mp.mp.dps = 50
def A1(k): return mp.besseli(1, k)/mp.besseli(0, k)
def KL(m1, k1, m2, k2): return mp.log(mp.besseli(0, k2)/mp.besseli(0, k1)) + A1(k1)*(k1 - k2*mp.cos(m1 - m2))
mu = lambda c1, c2: mp.mpf('0.3')*c1 + mp.mpf('0.5')*mp.sin(c2)
ka = lambda c1, c2: 2 + mp.mpf('0.4')*c1 - mp.mpf('0.3')*c2**2
a1 = lambda c1, c2: mp.mpf('0.2') + mp.mpf('0.1')*c2
a2 = lambda c1, c2: -mp.mpf('0.4')*c1
c0 = (mp.mpf('0.2'), mp.mpf('-0.3')); X = (mp.mpf('0.7'), mp.mpf('0.4'))
def gam(t): return (c0[0] + t*X[0], c0[1] + t*X[1])
def transported(t):
    shift = mp.quad(lambda s: a1(*gam(s))*X[0] + a2(*gam(s))*X[1], [0, t]) if t != 0 else 0
    c = gam(t); return mu(*c) + shift, ka(*c)
dmu = mp.diff(lambda t: mu(*gam(t)), 0); dka = mp.diff(lambda t: ka(*gam(t)), 0)
AX = a1(*c0)*X[0] + a2(*c0)*X[1]
k0 = ka(*c0)
gmm = k0*A1(k0); gkk = 1 - A1(k0)/k0 - A1(k0)**2
h = gmm*(dmu + AX)**2 + gkk*dka**2
# Fisher metric check by numerical second derivative of KL
fd_kk = mp.diff(lambda k: KL(0, k0, 0, k), k0, 2)
fd_mm = mp.diff(lambda m: KL(0, k0, m, k0), 0, 2)
print("Fisher check: g_kk - d2KL =", mp.nstr(gkk - fd_kk, 5), "; g_mumu - d2KL =", mp.nstr(gmm - fd_mm, 5))
print("h(X,X) =", mp.nstr(h, 12))
m0 = mu(*c0)
for t in ['1e-1', '1e-2', '1e-3', '1e-4']:
    t = mp.mpf(t); mt, kt = transported(t)
    ratio = KL(m0, k0, mt, kt)/(t**2/2)
    print(f"t={mp.nstr(t,1)}: KL/(t^2/2) = {mp.nstr(ratio, 12)}; relative error = {mp.nstr(abs(ratio/h - 1), 4)}")
