"""L2-gauge: time-like defect gradients (prop:link-map-inertia (b)), linearized operator
(prop:link-linearized-equation (a)), and cor:link-cycle-splitting vs the Hodge decomposition."""
import numpy as np
from scipy.linalg import expm, sqrtm, null_space, orth
rng = np.random.default_rng(7)
inv = np.linalg.inv
def spd(K):
    A = rng.standard_normal((K, K)); return A@A.T + 0.5*np.eye(K)
def Lex(B, T):  # exact Frechet derivative via block exponential
    n = B.shape[0]; M = np.zeros((2*n, 2*n)); M[:n, :n] = B; M[n:, n:] = B; M[:n, n:] = T
    return expm(M)[:n, n:]
K = 3; G = spd(K)
ip = lambda A, B: np.trace(inv(G) @ A.T @ G @ B)
star = lambda A: inv(G) @ A.T @ G
# full gl(K): pr_g = identity
w = 0.5*rng.standard_normal((K, K)); wp = 0.5*rng.standard_normal((K, K))
U = lambda wp, w: 0.5*ip(expm(-wp)@expm(w) - np.eye(K), expm(-wp)@expm(w) - np.eye(K))
Th = expm(-wp)@expm(w)
g_early = Lex(star(w), star(expm(-wp)) @ (Th - np.eye(K)))
g_late = -Lex(-star(wp), (Th - np.eye(K)) @ star(expm(w)))
h = 1e-6; eta = rng.standard_normal((K, K))
fd1 = (U(wp, w + h*eta) - U(wp, w - h*eta))/(2*h); fd2 = (U(wp + h*eta, w) - U(wp - h*eta, w))/(2*h)
print("timelike grad earlier arg vs FD:", abs(ip(g_early, eta) - fd1), "  later arg:", abs(ip(g_late, eta) - fd2))
# mass tensor: U(w+dw, w) ~ 1/2 ||L(-w, dw) e^w||_G^2
dw = 1e-3*rng.standard_normal((K, K))
print("mass tensor rel. err (O(|dw|) expected ~1e-3):", abs(U(w + dw, w) - 0.5*ip(Lex(-w, dw)@expm(w), Lex(-w, dw)@expm(w)))/U(w+dw, w))

# linearized equation: one record on e: j->i, defining representation, pulled-back G norm on T=gl(K)
W = spd(K); mui = rng.standard_normal(K); muj = rng.standard_normal(K); Sj = spd(K); Si = spd(K)
def EE(om):  # expected record energy, product law
    M = expm(om); r = mui - M@muj
    return 0.5*(r@W@r + np.trace(W@Si) + np.trace(W@M@Sj@M.T))
Ja = W @ (np.outer(mui - muj, muj) - Sj)
om = rng.standard_normal((K, K))
t = 1e-4
first = (EE(t*om) - EE(-t*om))/(2*t); second = (EE(t*om) - 2*EE(0*om) + EE(-t*om))/t**2
Ejj = Sj + np.outer(muj, muj)
print("lin (a) gradient at 0: dE[om] + Tr(J^T om):", abs(first + np.trace(Ja.T@om)))
print("lin (a) Hessian form: d2E[om,om] - [Tr(W om Ejj om^T) - Tr(J^T om^2)]:", abs(second - (np.trace(W@om@Ejj@om.T) - np.trace(Ja.T@om@om))))

# cycle splitting: graph with 4 vertices, 6 edges (K4), declared loop family = one triangle
V = 4; Ep = [(1, 0), (2, 1), (0, 2), (3, 0), (3, 1), (3, 2)]  # (j, i): e: j->i
nE = len(Ep)
d = np.zeros((nE, V))
for k, (j, i) in enumerate(Ep): d[k, i] += 1; d[k, j] -= 1
# triangle p = (e0, e1, e2) : e0:1->0, e1:2->1, e2:0->2 ; closed walk 0<-1<-2<-0, z_p = +1 on e0,e1,e2
zp = np.zeros(nE); zp[[0, 1, 2]] = 1
print("d^T z_p = 0:", np.abs(d.T @ zp).max())
imd = orth(d); Z = null_space(d.T)
Har = null_space(np.vstack([d.T, zp[None, :]]))  # Z cap z_p^perp
print("dims: im d", imd.shape[1], "(|V|-kappa=3)  Z", Z.shape[1], "(3)  har", Har.shape[1], "(|E|-|V|+1-r=2)")
# per-edge prior variances (scalar per edge, gl(1)-like scalar fiber suffices since splitting tensors with g)
Sig = np.diag(rng.uniform(0.3, 3.0, nE)); Sigi = inv(Sig)
coex = Sig @ zp[:, None]
basis = np.hstack([imd, Har, coex])
print("direct sum rank:", np.linalg.matrix_rank(basis), "of", nE)
print("coexact Sigma^-1-orthogonal to exact:", np.abs(coex.T @ Sigi @ imd).max(), " to harmonic:", np.abs(coex.T @ Sigi @ Har).max())
print("exact vs manuscript-harmonic in Sigma^-1 metric (Hodge needs 0):", np.abs(imd.T @ Sigi @ Har).max())
# true Hodge harmonic in Sigma^-1 metric: ker C+ cap ker(d^T Sigma^-1)
Har_true = null_space(np.vstack([d.T @ Sigi, zp[None, :]]))
# compare subspaces via principal angles
Qa = orth(Har); Qb = orth(Har_true)
s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
print("principal-angle cosines manuscript T_har vs Sigma^-1-Hodge harmonic:", s)
