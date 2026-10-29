"""D24-LDYN script 1: symbolic checks of the ring (it2:ring), the echo (it2:echo),
the scalar certainty function (it2:certainty iii), Gaussian convolution of the loop
factor (it2:widening) and the law-level ring covariance fixed point."""
import sympy as sp

print("== it2:ring: block A_i, coupling C', A^{-1}C', det, w")
p, q, s, c = sp.symbols('p qt s c', positive=True)
A = sp.Matrix([[p + q, -q*c], [-q*c, s + q*c**2]])
# energy E_i = p/2 (y_i-m)^2 + s/2 w^2 + q/2 (o - y_i + y_{i+1} + c w)^2
yi, wi, yn, wn, o, m = sp.symbols('y_i w_i y_n w_n o m')
E = p/2*(yi - m)**2 + s/2*wi**2 + q/2*(o - yi + yn + c*wi)**2
Hess = sp.hessian(E, (yi, wi))
Cp = sp.Matrix([[sp.diff(E, a, b) for b in (yn, wn)] for a in (yi, wi)])
print("A_i matches:", sp.simplify(Hess - A) == sp.zeros(2, 2))
print("C' =", Cp)
detA = sp.factor(A.det())
print("det A_i =", sp.expand(detA))
X = sp.simplify(A.inv()*Cp)
print("A^{-1}C' =", X)
w = q*s/(p*s + p*q*c**2 + q*s)
print("first diagonal entry + w == 0:", sp.simplify(X[0, 0] + w) == 0, " second column zero:", sp.simplify(X[:, 1]) == sp.zeros(2, 1))
qeff = 1/(1/q + c**2/s)
print("w == qeff/(p+qeff):", sp.simplify(w - qeff/(p + qeff)) == 0)
print("lim s->oo w =", sp.limit(w, s, sp.oo), "; lim s->0 w (c>0) =", sp.limit(w, s, 0), "; w at c=0:", sp.simplify(w.subs(c, 0)))
print("1 - w > 0:", sp.simplify((1 - w)*(p*s + p*q*c**2 + q*s)), "(numerator, positive)")

# numerical full block-circulant check for several ell, random parameters
import numpy as np
rng = np.random.default_rng(1)
for ell in (3, 4, 5, 7):
    pv, qv, sv, cv = rng.uniform(0.3, 3, size=4)
    An = np.array([[pv + qv, -qv*cv], [-qv*cv, sv + qv*cv**2]])
    Cn = np.array([[-qv, 0.0], [qv*cv, 0.0]])
    Ainv = np.linalg.inv(An)
    M = np.zeros((2*ell, 2*ell))
    for i in range(ell):
        M[2*i:2*i+2, 2*i:2*i+2] = -np.eye(2)
        k = (i + 1) % ell
        M[2*i:2*i+2, 2*k:2*k+2] = -Ainv @ Cn
    ev = np.sort_complex(np.linalg.eigvals(M))
    wv = qv*sv/(pv*sv + pv*qv*cv**2 + qv*sv)
    zeta = np.exp(2j*np.pi*np.arange(ell)/ell)
    pred = np.sort_complex(np.concatenate([-np.ones(ell), -(1 - zeta*wv)]))
    # match by greedy nearest
    used = np.zeros(len(pred), bool); err = 0.0
    for e in ev:
        d = np.abs(pred - e); d[used] = np.inf; j = np.argmin(d); used[j] = True; err = max(err, d[j])
    # global ring: J of the same ring with record precision q, gradient flow D^{-1}J
    nv = 2*ell
    J = np.zeros((nv, nv))
    for i in range(ell):
        J[2*i, 2*i] += pv; J[2*i+1, 2*i+1] += sv
        k = (i + 1) % ell
        L = np.zeros(nv); L[2*i] = 1; L[2*k] = -1; L[2*i+1] = -cv
        J += qv*np.outer(L, L)
    D = np.diag(np.diag(J)) * 0
    for i in range(ell):
        D[2*i:2*i+2, 2*i:2*i+2] = J[2*i:2*i+2, 2*i:2*i+2]
    evg = np.linalg.eigvals(-np.linalg.solve(D, J))
    print(f"ell={ell}: max|eig - predicted| = {err:.2e}, max Re = {ev.real.max():.4f}, max|Im| = {np.abs(ev.imag).max():.4f}; global max|Im| = {np.abs(evg.imag).max():.1e}, global max Re = {evg.real.max():.4f}")

print("\n== law-level ring: symmetric content-variance fixed point Sigma = (A^{-1})_{11} with qt = 1/(R+Sigma)")
R, Sg = sp.symbols('R Sigma', positive=True)
qt = 1/(R + Sg)
phi = sp.simplify(((s + qt*c**2)/(p*s + p*qt*c**2 + qt*s)))
print("phi(Sigma) =", sp.simplify(phi))
num = sp.expand(sp.numer(sp.together(phi - Sg)))
print("fixed-point numerator:", sp.factor(num))
dphi = sp.simplify(sp.diff(phi, Sg))
print("phi'(Sigma) =", sp.factor(dphi))

print("\n== it2:echo")
sig, lam, v, eps = sp.symbols('sigma lambda v epsilon', positive=True)
for ell in (2, 3, 4, 5):
    Jm = sp.eye(ell)/sig**2 + lam*sp.ones(ell, ell)
    exact = sp.simplify(Jm.inv()[0, 0])
    claim = sig**2*(1 + (ell - 1)*lam*sig**2)/(1 + ell*lam*sig**2)
    cav = 1/(1/sig**2 + 1/(1/lam + (ell - 1)*sig**2))
    quad = sp.expand(((ell - 1)/sig**2*v**2 + (eps/sig**2 + 2 - ell)*v - eps))
    fp = sp.expand(sp.numer(sp.together(1/v - 1/sig**2 - 1/(eps + (ell - 1)*v))))
    roots = sp.solve(sp.Eq(quad, 0), v)
    print(f"ell={ell}: exact==claim {sp.simplify(exact-claim)==0}; cavity==exact {sp.simplify(cav-exact)==0}; "
          f"lim exact = {sp.limit(claim, lam, sp.oo)}; fixed-point numerator proportional to quadratic: {sp.simplify(fp + quad*sig**2) == 0 or sp.simplify(fp - quad*sig**2) == 0}")
    pos = [r for r in roots if sp.limit(r.subs(eps, sp.Rational(1, 10**6)).subs(sig, 1), v, 0) is not None]
    rp = sp.simplify(roots[1]) if len(roots) > 1 else roots[0]
    for r in roots:
        val = r.subs({sig: 1, eps: sp.Rational(1, 10**8)})
        print("    root at sigma=1, eps=1e-8:", sp.N(val, 12))
print("ell=2 positive root asymptotics: v^2 + eps v - eps sigma^2 = 0 ->", sp.series((-eps + sp.sqrt(eps**2 + 4*eps*sig**2))/2, eps, 0, 2))

print("\n== it2:certainty (iii): f(v,d2) = p - 1/v + 1/(v+R) - (vk+R+d2)/(v+R)^2")
pj, vk, d2 = sp.symbols('p_j v_k d2', positive=True)
f = pj - 1/v + 1/(v + R) - (vk + R + d2)/(v + R)**2
# check that f is twice the v-derivative of the variance part of the objective
obj = sp.Rational(1, 2)*(pj*v - 1 - sp.log(pj*v)) + sp.Rational(1, 2)*((vk + R + d2)/(v + R) - 1 + sp.log(v + R) - sp.log(vk + R))
print("f == 2 dObj/dv:", sp.simplify(2*sp.diff(obj, v) - f) == 0)
fv = sp.simplify(sp.diff(f, v))
print("df/dv =", sp.factor(sp.together(fv)))
# decompose f = p - R/(v(v+R)) - (vk+R+d2)/(v+R)^2 : both subtracted terms decreasing in v
print("f - [p - R/(v(v+R)) - (vk+R+d2)/(v+R)^2] =", sp.simplify(f - (pj - R/(v*(v + R)) - (vk + R + d2)/(v + R)**2)))
print("df/d(d2) =", sp.diff(f, d2))
print("limits: v->0+:", sp.limit(f, v, 0, '+'), " v->oo:", sp.limit(f, v, sp.oo))

print("\n== loop-factor convolution (it2:widening), scalar check")
u, a_, Vb, W = sp.symbols('u a V W', positive=True)
sv_ = sp.symbols('s', real=True)
integrand = sp.exp(-lam/2*W*(u + sv_)**2) * sp.exp(-(sv_ - a_)**2/(2*Vb))/sp.sqrt(2*sp.pi*Vb)
I = sp.simplify(sp.integrate(integrand, (sv_, -sp.oo, sp.oo)))
Om = 1/(1/(lam*W) + Vb)
ratio = sp.simplify(I/sp.exp(-Om*(u + a_)**2/2))
print("integral / exp(-Omega (u+a)^2/2) =", ratio, " (independent of u:", sp.simplify(sp.diff(ratio, u)) == 0, ")")
