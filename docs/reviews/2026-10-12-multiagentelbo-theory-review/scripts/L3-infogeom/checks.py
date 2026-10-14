"""L3-infogeom recomputation checks for c07, ca_proofs_c07, g01, g05, g06."""
import numpy as np, sympy as sp
from scipy import integrate, linalg
from numpy.polynomial.hermite_e import hermegauss, hermeval
from math import factorial
rng = np.random.default_rng(0)
np.set_printoptions(precision=4)

def gauss_kl(m0, S0, m1, S1):
    """KL(N(m0,S0) || N(m1,S1))."""
    n = len(m0); S1i = np.linalg.inv(S1); d = m1 - m0
    return 0.5*(np.trace(S1i@S0) + d@S1i@d - n + np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1])

print("== 1. Normal-family Moebius counterexample (prop:pb-statistical-tensor-descent)")
mu, y = sp.symbols("mu y", real=True); s = sp.Symbol("s", positive=True)
logp = -sp.log(s) - (y-mu)**2/(2*s**2) - sp.log(sp.sqrt(2*sp.pi))
sc = [sp.diff(logp, v) for v in (mu, s)]
pdf = sp.exp(logp)
def E(expr):
    return sp.simplify(sp.integrate(sp.expand(expr)*pdf, (y, -sp.oo, sp.oo)))
G = sp.Matrix(2, 2, lambda i, j: E(sc[i]*sc[j]))
print("Fisher (mu,s):", G)
T = {}
for i in range(2):
    for j in range(2):
        for k in range(2):
            T[(i, j, k)] = E(sc[i]*sc[j]*sc[k])
print("T components:", {k: v for k, v in T.items() if v != 0})
z = sp.Symbol("z"); f = (1+z)/(1-z)
print("fixed point f(i) =", sp.simplify(f.subs(z, sp.I)), "; derivative at i =", sp.simplify(sp.diff(f, z).subs(z, sp.I)))
X, S = sp.symbols("X S", real=True)
w = f.subs(z, X + sp.I*S); U, V = sp.re(sp.expand_complex(w)), sp.im(sp.expand_complex(w))
J = sp.Matrix([[sp.diff(U, X), sp.diff(U, S)], [sp.diff(V, X), sp.diff(V, S)]])
pt = {X: 0.3, S: 0.7}
Jn = np.array(J.subs(pt).evalf(), dtype=float); Vn = float(V.subs(pt))
g_src = 2*np.eye(2)/0.7**2; g_tgt = 2*np.eye(2)/Vn**2
print("isometry residual at (X,S)=(0.3,0.7):", np.abs(Jn.T@g_tgt@Jn - g_src).max())
Tn = np.zeros((2, 2, 2))
for (i, j, k), v in T.items(): Tn[i, j, k] = float(v.subs(s, 1))
e_mu = np.array([1., 0.]); e_img = np.array([0., 1/np.sqrt(2)])
cube = lambda v: np.einsum("ijk,i,j,k->", Tn, v, v, v)
print("cubic values T(e_mu)^3 =", cube(e_mu), "; T(image)^3 =", cube(e_img), "; 2sqrt2 =", 2*np.sqrt(2))

print("== 2. Gaussian KL Hessian = Fisher (moment chart), K=3; natural gradient")
K = 3
A0 = rng.standard_normal((K, K)); Sig = A0@A0.T + K*np.eye(K); m = rng.standard_normal(K); Lam = np.linalg.inv(Sig)
iu = np.triu_indices(K)
def unpack(v):
    dm = v[:K]; dS = np.zeros((K, K)); dS[iu] = v[K:]; dS = dS + dS.T - np.diag(np.diag(dS)); return dm, dS
dim = K + len(iu[0]); h = 1e-4; H = np.zeros((dim, dim))
def kl_at(v):
    dm, dS = unpack(v); return gauss_kl(m, Sig, m+dm, Sig+dS)
for a in range(dim):
    for b in range(dim):
        ea = np.zeros(dim); eb = np.zeros(dim); ea[a] = h; eb[b] = h
        H[a, b] = (kl_at(ea+eb) - kl_at(ea-eb) - kl_at(-ea+eb) + kl_at(-ea-eb))/(4*h*h)
Gf = np.zeros((dim, dim))
for a in range(dim):
    for b in range(dim):
        dma, dSa = unpack(np.eye(dim)[a]); dmb, dSb = unpack(np.eye(dim)[b])
        Gf[a, b] = dma@Lam@dmb + 0.5*np.trace(Lam@dSa@Lam@dSb)
print("max |Hess KL - Fisher| =", np.abs(H-Gf).max(), "; scale", np.abs(Gf).max())
gm = rng.standard_normal(K); GS = rng.standard_normal((K, K)); GS = (GS+GS.T)/2
cov = np.concatenate([gm, np.array([GS[i, j]*(1 if i == j else 2) for i, j in zip(*iu)])])
nat = np.linalg.solve(Gf, cov); nm, nS = unpack(nat)
print("nat_mu - Sigma grad_mu:", np.abs(nm - Sig@gm).max(), "; nat_S - 2 Sigma G Sigma:", np.abs(nS - 2*Sig@GS@Sig).max())

print("== 3. Orbit Fisher form, Gaussian GL(K) congruence (cor:pb-orbit-fisher-form)")
xi = rng.standard_normal((K, K))
def kbil(a, b):
    da, Sa = a@m, a@Sig+Sig@a.T; db, Sb = b@m, b@Sig+Sig@b.T
    return da@Lam@db + 0.5*np.trace(Lam@Sa@Lam@Sb)
for t in (1e-1, 1e-2, 1e-3):
    g = linalg.expm(t*xi); val = gauss_kl(m, Sig, g@m, g@Sig@g.T)
    print(f" t={t:g}: KL/(t^2/2 k) = {val/(0.5*t*t*kbil(xi, xi)):.8f}")
Lc = np.linalg.cholesky(Sig); Z = np.linalg.solve(Lc, xi@Lc); eta = np.linalg.solve(Lc, m)
print(" k minus (||Z eta||^2 + 2||sym Z||_F^2):", kbil(xi, xi) - (np.sum((Z@eta)**2) + 2*np.sum(((Z+Z.T)/2)**2)))
basis = [np.eye(K*K)[a].reshape(K, K) for a in range(K*K)]
Kmat = np.array([[kbil(A, B) for B in basis] for A in basis])
ev, evec = np.linalg.eigh(Kmat)
print(" orbit-form eigenvalues:", ev, "; #zero:", int(np.sum(np.abs(ev) < 1e-10)), "; expected dim stabilizer (K-1)(K-2)/2 =", (K-1)*(K-2)//2)
nullv = evec[:, 0].reshape(K, K)
for t in (0.5, 2.0):
    g = linalg.expm(t*nullv)
    print(f"  stabilizer direction, t={t}: KL = {gauss_kl(m, Sig, g@m, g@Sig@g.T):.2e}")

print("== 4. Duration examples (c07)")
val, _ = integrate.quad(lambda z: np.sqrt(1+16*z**6), 0, 1)
print(" int_0^1 sqrt(1+16 z^6) dz =", val, "; sqrt2 =", np.sqrt(2))
tt = 0.7; b = 1/(1+np.exp(-tt)); bd = b*(1-b)
print(" Bernoulli record speed:", bd/np.sqrt(b*(1-b)), "; 1/(2cosh(t/2)) =", 1/(2*np.cosh(tt/2)))
z1, z2 = sp.symbols("z1 z2", positive=True)
F = z1*z2; n = sp.sqrt(sp.diff(F, z1)**2 + sp.diff(F, z2)**2)
alpha = [-sp.diff(F, z1)/n, -sp.diff(F, z2)/n]
dal = sp.simplify(sp.diff(alpha[1], z1) - sp.diff(alpha[0], z2))
print(" d alpha coefficient:", dal, "; residual vs claim:", sp.simplify(dal - (z1**2-z2**2)/(z1**2+z2**2)**sp.Rational(3, 2)))
ts = np.linspace(0, 1, 2001); sp1 = np.abs(np.sin(5*ts)); sp2 = np.abs(np.cos(3*ts))+0.1
t1 = np.trapezoid(sp1, ts); t2 = np.trapezoid(sp2, ts); tt_ = np.trapezoid(np.sqrt(sp1**2+sp2**2), ts)
print(" block clocks: max", max(t1, t2), "<= l2", np.hypot(t1, t2), "<= total", tt_, "<= sum", t1+t2)

print("== 5. g01: Bregman form and affine Pythagorean identity, 1-D Gaussian T=(y,y^2)")
def natp(mu_, v): return np.array([mu_/v, -0.5/v])
def Alog(l): return -l[0]**2/(4*l[1]) - 0.5*np.log(-2*l[1]) + 0.5*np.log(2*np.pi)
def gradA(l): v = -0.5/l[1]; mu_ = l[0]*v; return np.array([mu_, v+mu_**2])
def kl1(m0, v0, m1, v1): return 0.5*(v0/v1 + (m1-m0)**2/v1 - 1 + np.log(v1/v0))
lp, l = natp(0.4, 1.3), natp(-0.7, 0.6)
DA = Alog(l) - Alog(lp) - gradA(lp)@(l-lp)
print(" KL(Q_lp || Q_l) - D_A(l, lp) =", kl1(0.4, 1.3, -0.7, 0.6) - DA)
mtarget = 0.9; s2p = 2.1; m0, v0 = -0.3, 0.8
lhs = kl1(mtarget, s2p, m0, v0); rhs = kl1(mtarget, s2p, mtarget, v0) + kl1(mtarget, v0, m0, v0)
print(" Pythagorean residual (projection onto E[y]=m keeps precision):", lhs - rhs)

print("== 6. g05 witness and fixed-covariance KL")
Lw = np.array([[2., 1], [1, 2]]); P = np.array([[1., 0], [1, 1]]); Lp = P.T@Lw@P
gap = lambda M: 0.5*(np.log(M[0, 0]) + np.log(M[1, 1]) - np.log(np.linalg.det(M)))
print(" P^T L P =", Lp.tolist(), "; gaps:", gap(Lw), 0.5*np.log(4/3), gap(Lp), 0.5*np.log(4))
W = A0@A0.T + np.eye(K); mi, mj = rng.standard_normal(K), rng.standard_normal(K)
print(" fixed-cov KL minus 1/2 dmu^T W dmu:", gauss_kl(mi, np.linalg.inv(W), mj, np.linalg.inv(W)) - 0.5*(mi-mj)@W@(mi-mj))

print("== 7. g06 Hermite: Z_2, Mehler eigenvalues")
xq, wq = hermegauss(120); wq = wq/np.sqrt(2*np.pi)
for t in (0.3, 1.0, -0.2):
    print(f" Z_2({t}) quad={np.sum(wq*np.exp(-t*(xq**2-1))):.14f} closed={np.exp(t)/np.sqrt(1+2*t):.14f}")
X1, X2 = np.meshgrid(xq, xq); W2 = np.outer(wq, wq)
for bb in (2, 3):
    rho = 1/np.sqrt(bb); err = 0
    for k in range(1, 7):
        ck = np.zeros(k+1); ck[k] = 1
        bar = rho*X1 + np.sqrt(1-rho**2)*X2
        cov_k = np.sum(W2*hermeval(X1, ck)*hermeval(bar, ck))
        err = max(err, abs(cov_k/factorial(k) - bb**(-k/2)))
    print(f" b={bb}: max abs(E[He_k(xi)He_k(bar)]/k! - b^(-k/2)), k<=6: {err:.2e}")
print(" HS sum b=3:", sum(3.0**(2-k) for k in range(1, 200)), "; b^2/(b-1) =", 9/2)
print(" E He_3^2 =", np.sum(wq*(xq**3-3*xq)**2))

print("== 8. c07 bivariate normal joint-vs-marginal")
c = 0.4; Lb = np.array([[1, c], [c, 1]]); D_ = Lb - (1-c**2)*np.eye(2)
o = np.array([1, 1]); mo = np.array([1, -1])
print(" difference:", D_.tolist(), "; (1,1):", o@D_@o, 2*c*(1+c), "; (1,-1):", mo@D_@mo, 2*c*(c-1))
