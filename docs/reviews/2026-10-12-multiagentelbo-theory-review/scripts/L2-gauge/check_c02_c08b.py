"""L2-gauge float64 checks for c02_geometry.tex and c08b_lie_links.tex (GL(K), K<=4,
non-orthogonal gauge transforms). Each line prints a residual."""
import numpy as np
from scipy.linalg import expm, logm, sqrtm
rng = np.random.default_rng(20261012)
def rgl(K, s=0.6):
    while True:
        A = np.eye(K) + s*rng.standard_normal((K, K))
        if np.linalg.det(A) > 0.2: return A
def spd(K):
    A = rng.standard_normal((K, K)); return A@A.T + 0.5*np.eye(K)
def gnorm2(H, G): return np.trace(np.linalg.solve(G, H.T) @ G @ H)
def P(*Ms):
    out = np.eye(Ms[0].shape[0])
    for M in Ms: out = out @ M
    return out
inv = np.linalg.inv
K = 3
print("== c02 ==")
U = {i: rgl(K) for i in (1, 2, 3)}
Th = {(1, 2): rgl(K), (2, 3): rgl(K), (3, 1): rgl(K)}
def twist(Th, U, t, s): return inv(U[t]) @ Th[(t, s)] @ U[s]
g = {i: rgl(K) for i in (1, 2, 3)}
Thp = {(t, s): g[t] @ M @ inv(g[s]) for (t, s), M in Th.items()}
Up = {i: g[i] @ U[i] for i in U}
print("twist passive invariance (non-orth g):", max(np.abs(twist(Th, U, t, s) - twist(Thp, Up, t, s)).max() for (t, s) in Th))
order = [(1, 2), (2, 3), (3, 1)]
Hol = P(*[Th[k] for k in order]); Holp = P(*[Thp[k] for k in order])
tw = [twist(Th, U, *k) for k in order]
print("Hol = U1 (twist product) U1^-1:", np.abs(Hol - U[1] @ P(*tw) @ inv(U[1])).max())
print("Hol' = g1 Hol g1^-1:", np.abs(Holp - g[1] @ Hol @ inv(g[1])).max())
Gf = spd(K); Gfp = inv(g[1]).T @ Gf @ inv(g[1])
pen = lambda Uh, G: np.trace(np.linalg.solve(G, (np.eye(K) - Uh).T) @ G @ (np.eye(K) - Uh))
print("GL loop penalty passive invariance:", abs(pen(Hol, Gf) - pen(Holp, Gfp)))
print("same penalty with metric held (not transformed), deviation:", abs(pen(Hol, Gf) - pen(Holp, Gf)))
E = lambda i, j: np.outer(np.eye(3)[i], np.eye(3)[j])
Rz = lambda a: expm(a*(E(1, 0) - E(0, 1))); Rx = lambda a: expm(a*(E(2, 1) - E(1, 2)))
H = Rz(np.pi/2) @ Rx(np.pi/2)
print("running example: tr Hol =", round(np.trace(H), 14), " cyclic perm resid:", np.abs(H - np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])).max(),
      " angle/pi=", np.arccos((np.trace(H)-1)/2)/np.pi)
S = np.array([[4., 1], [1, 1]]); Sh = np.real(sqrtm(S)); R7 = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
T12 = Sh @ R7 @ inv(Sh)
print("three-agreements witness: |T S T^T - S|:", np.abs(T12 @ S @ T12.T - S).max(), " |eig|:", np.abs(np.linalg.eigvals(T12)), " |TT^T-I|:", np.abs(T12@T12.T-np.eye(2)).max())

print("== c08b ==")
om = {k: 0.4*rng.standard_normal((K, K)) for k in order}
Thc = {(t, s): U[t] @ expm(om[(t, s)]) @ inv(U[s]) for (t, s) in om}
g0 = rgl(K); g0i = inv(g0)
Thc2 = {(t, s): (U[t] @ g0) @ expm(g0i @ om[(t, s)] @ g0) @ inv(U[s] @ g0) for (t, s) in om}
print("gen-gauge (b) exp_{sigma'} o Ad_{g0^-1} = exp_sigma:", max(np.abs(Thc[k] - Thc2[k]).max() for k in om))
Thc3 = {(t, s): (g[t]@U[t]) @ expm(om[(t, s)]) @ inv(g[s]@U[s]) for (t, s) in om}
print("gen-gauge (a) passive moves links by vertex law, same generators:", max(np.abs(Thc3[(t, s)] - g[t]@Thc[(t, s)]@inv(g[s])).max() for (t, s) in om))
Gf = spd(K)
Tp = P(*[expm(om[k]) for k in order])
Q = lambda Tp, G: 0.5*gnorm2(np.eye(K) - Tp, G)
R = g0; Tp2 = g0i @ Tp @ g0
print("defect (b) sigma' with R^T G R:", abs(Q(Tp2, R.T @ Gf @ R) - Q(Tp, Gf)))
print("defect (b) sigma' with G held == sigma with R^-T G R^-1:", abs(Q(Tp2, Gf) - Q(Tp, inv(R).T @ Gf @ inv(R))))
print("defect (b) G held, non-isometric R, deviation (nonzero expected):", abs(Q(Tp2, Gf) - Q(Tp, Gf)))
Gh = np.real(sqrtm(Gf)); O0 = np.linalg.qr(rng.standard_normal((K, K)))[0]; Oiso = inv(Gh) @ O0 @ Gh
Rc = 2.7*Oiso; Tp3 = inv(Rc) @ Tp @ Rc
print("defect (b) R = 2.7*G-isometry, G held, deviation:", abs(Q(Tp3, Gf) - Q(Tp, Gf)))
e0 = expm(om[order[0]])
Trot = expm(om[order[1]]) @ expm(om[order[2]]) @ e0
print("defect (b) rotated walk:", abs(Q(Trot, Gf) - Q(Tp, inv(e0).T @ Gf @ inv(e0))), " raw base-vertex dependence:", abs(Q(Trot, Gf)-Q(Tp, Gf)))
Ui = U[1]; Gi = inv(Ui).T @ Gf @ inv(Ui); Uh = Ui @ Tp @ inv(Ui)
print("defect (c) local form:", abs(Q(Uh, Gi) - Q(Tp, Gf)))
print("defect (d) Q = dim - tr for isometry:", abs(Q(Oiso, Gf) - (K - np.trace(Oiso))))
worst = 0
for _ in range(2000):
    s = rng.uniform(0.01, 1.0)
    Om = [rng.standard_normal((K, K)) for _ in range(3)]
    nrm = [np.sqrt(gnorm2(X, Gf)) for X in Om]; tot = sum(nrm); Om = [X*s/tot for X in Om]
    lhs = abs(Q(P(*[expm(X) for X in Om]), Gf) - 0.5*gnorm2(sum(Om), Gf))
    rhs = s*(np.exp(s)-1-s) + 0.5*(np.exp(s)-1-s)**2
    worst = max(worst, lhs/rhs); assert rhs <= 2.3*s**3
print("defect (e) max lhs/rhs over 2000 draws (<=1 required):", worst)
G1 = spd(K); G2 = 3.1*G1; G3 = spd(K)
b = rng.standard_normal(K); c = rng.standard_normal(K); Hr = np.outer(b, c)
print("lem trace-form scale: multiple:", abs(gnorm2(Hr, G1) - gnorm2(Hr, G2)), " non-multiple:", abs(gnorm2(Hr, G1) - gnorm2(Hr, G3)))
A = expm(np.diag([0, np.log(2)])) @ expm(np.pi*np.array([[0., -1], [1, 0]]))
print("chart (e): product =", np.round(A, 12).tolist(), " imag(logm diag(-1,-2)):", np.abs(np.imag(logm(np.diag([-1., -2])))).max())
B = rgl(4); P_ = np.real(sqrtm(B @ B.T)); O1 = np.linalg.solve(P_, B); T_ = np.real(logm(O1)); S_ = np.real(logm(P_))
print("chart (c) exp(S)exp(T)-B:", np.abs(expm(S_) @ expm(T_) - B).max(), " skew:", np.abs(T_ + T_.T).max(), " sym:", np.abs(S_ - S_.T).max())
def Lfr(Bm, Tm, n=3000):
    ts = (np.arange(n)+0.5)/n
    return sum(expm(t*Bm) @ Tm @ expm((1-t)*Bm) for t in ts)/n
Bm = 0.7*rng.standard_normal((K, K)); Tm = rng.standard_normal((K, K)); Sm = rng.standard_normal((K, K))
h = 1e-6; fd = (expm(Bm + h*Tm) - expm(Bm - h*Tm))/(2*h)
print("L(B,T) vs FD (midpoint quad n=3000):", np.abs(Lfr(Bm, Tm) - fd).max())
print("adjoint <S,L(B,T)> - <L(B^T,S),T>:", abs(np.trace(Sm.T @ Lfr(Bm, Tm)) - np.trace(Lfr(Bm.T, Sm).T @ Tm)))
xi = 0.5*rng.standard_normal((K, K)); W = spd(K); k1 = rng.standard_normal(K); k2 = rng.standard_normal(K)
hfun = lambda Mx: 0.5*(k1 - Mx@k2) @ W @ (k1 - Mx@k2)
gr = -W @ np.outer(k1 - expm(xi)@k2, k2)
X = Lfr(xi.T, gr); Y = np.linalg.solve(Gf, X @ Gf)
eta = rng.standard_normal((K, K)); fdd = (hfun(expm(xi + h*eta)) - hfun(expm(xi - h*eta)))/(2*h)
print("field-eq (c) gradient (drho)^T L(xi^T, J) in <,>_G vs FD:", abs(np.trace(np.linalg.solve(Gf, Y.T) @ Gf @ eta) - fdd))
J2 = np.array([[0., -1], [1, 0]]); w = -np.log(3)*np.eye(2) + 2*np.pi*J2
mu = np.array([1., 0]); l = lambda Mx: 0.5*((mu - Mx@mu)@(mu - Mx@mu) + np.trace(Mx@Mx.T))
Eb = lambda a, b2: np.outer(np.eye(2)[a], np.eye(2)[b2])
gradw = np.array([[(l(expm(w + h*Eb(a, b2))) - l(expm(w - h*Eb(a, b2))))/(2*h) for b2 in range(2)] for a in range(2)])
print("prior-necessary (a)(iii): |exp(w)-I/3|:", np.abs(expm(w) - np.eye(2)/3).max(), " FD gradient:", np.abs(gradw).max())
