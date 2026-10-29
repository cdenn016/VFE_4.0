"""D02-LINKS checks for chapters/02_interactions.tex (MAgent Theory 2.0, 2026-10-29).

Single-threaded, CPU only.  Numerical agreement supports an identity at the
tested instances; it does not prove it.
"""
import numpy as np
import scipy.linalg as sla
import sympy as sp

rng = np.random.default_rng(20261029)
inv = np.linalg.inv
nrm = np.linalg.norm


def randgl(d, lo=0.3):
    while True:
        A = rng.standard_normal((d, d))
        if abs(np.linalg.det(A)) > lo:
            return A


def randspd(d):
    A = rng.standard_normal((d, d))
    return A @ A.T + 0.5 * np.eye(d)


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def sym2(g):
    """Congruence action S -> g S g^T of a 2x2 g on Sym^2(R^2), basis (a,b,c) for [[a,b],[b,c]]."""
    basis = [np.array([[1., 0], [0, 0]]), np.array([[0., 1], [1, 0]]), np.array([[0., 0], [0, 1]])]
    M = np.zeros((3, 3))
    for k, B in enumerate(basis):
        S = g @ B @ g.T
        M[:, k] = [S[0, 0], S[0, 1], S[1, 1]]
    return M


def psqrt(A):
    return sla.sqrtm(A)


print("== 1. prop:int-mismatch: passive covariance in G = GL(3)")
d = 3
U1, U2 = randgl(d), randgl(d)
om, omt = 0.4 * rng.standard_normal((d, d)), 0.4 * rng.standard_normal((d, d))
th = U1 @ sla.expm(om) @ inv(U2)
tht = U1 @ sla.expm(omt) @ inv(U2)
Xi = tht @ inv(th)
g1, g2 = randgl(d), randgl(d)
thp = (g1 @ U1) @ sla.expm(om) @ inv(g2 @ U2)
thtp = (g1 @ U1) @ sla.expm(omt) @ inv(g2 @ U2)
Xip = thtp @ inv(thp)
print("  |Xi' - g_i Xi g_i^-1| =", nrm(Xip - g1 @ Xi @ inv(g1)))
h = randgl(d)
thr = (U1 @ h) @ sla.expm(inv(h) @ om @ h) @ inv(U2 @ h)
thtr = (U1 @ h) @ sla.expm(inv(h) @ omt @ h) @ inv(U2 @ h)
print("  reference change with equivariant generators: |Xi_ref - Xi| =", nrm(thtr @ inv(thr) - Xi))
print("  sender-side mismatch theta^-1 theta~ equals theta^-1 Xi theta:", nrm(inv(th) @ tht - inv(th) @ Xi @ th))

print("== 2. Non-faithful representations: Xi != e but one G-valued link realizes both displayed links")
rq = lambda a: rot(2 * a)
rs = lambda a: rot(3 * a)
theta, thetat = 0.0, np.pi / 3
Xi_angle = (thetat - theta) % (2 * np.pi)
star = np.pi
print("  G = SO(2), rho_q charge 2, rho_s charge 3; Xi angle =", Xi_angle, "(nonidentity)")
print("  |rho_q(theta*) - rho_q(theta)| =", nrm(rq(star) - rq(theta)),
      " |rho_s(theta*) - rho_s(theta~)| =", nrm(rs(star) - rs(thetat)))
grid = np.linspace(0.01, 2 * np.pi - 0.01, 200001)
kerq = grid[(np.abs(np.cos(2 * grid) - 1) < 1e-9) & (np.abs(np.cos(3 * grid) - 1) < 1e-9)]
print("  nonidentity angles in ker(rho_q + rho_s) on grid:", kerq.size, "(joint representation faithful)")
print("  Sym^2 congruence kernel: |sym2(R(pi)) - I| =", nrm(sym2(rot(np.pi)) - np.eye(3)))

print("== 3. prop:int-endpoint; channel-level versus generator-level reference consistency")
mu = [rng.standard_normal(d) for _ in range(2)]
Sg = [randspd(d) for _ in range(2)]
lam = 0.7
Lmom = lambda mi, mj, Sj: lam * np.outer(mi, mj) @ inv(Sj)
L = Lmom(mu[0], mu[1], Sg[1])
Th = U1 @ sla.expm(L) @ inv(U2)
Thp = (g1 @ U1) @ sla.expm(L) @ inv(g2 @ U2)
print("  passive: |Theta' - g_i Theta g_j^-1| =", nrm(Thp - g1 @ Th @ inv(g2)))
mur = [inv(h) @ m for m in mu]
Sr = [inv(h) @ S @ inv(h).T for S in Sg]
Lr = Lmom(mur[0], mur[1], Sr[1])
print("  moment family reference equivariance |L' - h^-1 L h| =", nrm(Lr - inv(h) @ L @ h))
Thr = (U1 @ h) @ sla.expm(Lr) @ inv(U2 @ h)
print("  displayed link after reference change |Theta_ref - Theta| =", nrm(Thr - Th))
J = np.array([[0., -1], [1, 0]])
h2 = np.array([[2.0, 0.3], [0.1, 0.5]])
Lc = 2 * np.pi * J
print("  constant rule 2piJ: |e^{L'} - h^-1 e^L h| =", nrm(sla.expm(Lc) - inv(h2) @ sla.expm(Lc) @ h2),
      " |L' - h^-1 L h| =", nrm(Lc - inv(h2) @ Lc @ h2))

print("== 4. prop:int-rank")
for (dd, r) in [(4, 1), (5, 2), (6, 3)]:
    O = rng.standard_normal((dd, r))
    T = rng.standard_normal((r, dd))
    E = sla.expm(O @ T)
    K = sla.null_space(T)
    print(f"  d={dd} r={r}: max|e^OT v - v| on ker T = {nrm(E @ K - K):.2e};",
          "rank[O, e^OT - I] =", np.linalg.matrix_rank(np.hstack([O, E - np.eye(dd)]), tol=1e-9),
          "rank O =", np.linalg.matrix_rank(O))

print("== 5. Laplace example (exact, sympy)")
I2 = sp.eye(2)
E12 = sp.Matrix([[0, 1], [0, 0]])
E21 = E12.T
Th12, Th21 = I2 + E12, I2 + E21
W = Th12 * Th21
N2 = Th21 * Th12
print("  Theta12 Theta21 =", W.tolist(), " N2 =", N2.tolist())
sqrtN2 = (N2 + sp.sqrt(N2.det()) * I2) / sp.sqrt(N2.trace() + 2 * sp.sqrt(N2.det()))
print("  sqrt(N2)^2 - N2 =", sp.simplify(sqrtN2 ** 2 - N2).tolist())
Om12 = sp.simplify(Th12 * sqrtN2.inv())
print("  Omega12 =", Om12.tolist(), " minus 5^-1/2[[2,1],[-1,2]]:",
      sp.simplify(Om12 - sp.Matrix([[2, 1], [-1, 2]]) / sp.sqrt(5)).tolist())
N1 = Th12 * Th21
sqrtN1 = (N1 + sp.sqrt(N1.det()) * I2) / sp.sqrt(N1.trace() + 2 * sp.sqrt(N1.det()))
Om21 = sp.simplify(Th21 * sqrtN1.inv())
print("  Omega21 Omega12 =", sp.simplify(Om21 * Om12).tolist(), " det Omega12 =", sp.simplify(Om12.det()),
      " Omega12^T Omega12 =", sp.simplify(Om12.T * Om12).tolist())
Qs = lambda Wm, S: sp.Rational(1, 2) * ((S * (I2 - Wm).T * S.inv() * (I2 - Wm)).trace())
print("  Q12 =", Qs(W, I2), "; Q21 =", Qs(N2, I2), "; Q(shear I+E12) =", Qs(I2 + E12, I2))
print("  Theta21 equals Theta12^{-1}?", (Th21 - Th12.inv()) == sp.zeros(2, 2))

print("== 6. prop:int-reciprocal: random pairs, covariance, balanced uniqueness")
worst_inv, worst_cov, used = 0.0, 0.0, 0
while used < 60:
    A = randgl(d)
    B = randgl(d)
    Nj = B @ A
    ev = np.linalg.eigvals(Nj)
    if np.any((np.abs(ev.imag) < 1e-9) & (ev.real <= 0)):
        continue
    used += 1
    Ni = A @ B
    Oij = A @ inv(psqrt(Nj))
    Oji = B @ inv(psqrt(Ni))
    worst_inv = max(worst_inv, nrm(Oji @ Oij - np.eye(d)))
    gi, gj = randgl(d), randgl(d)
    Ap, Bp = gi @ A @ inv(gj), gj @ B @ inv(gi)
    Oijp = Ap @ inv(psqrt(Bp @ Ap))
    worst_cov = max(worst_cov, nrm(Oijp - gi @ Oij @ inv(gj)))
print(f"  worst |Omega_ji Omega_ij - I| = {worst_inv:.2e}; worst covariance residual = {worst_cov:.2e} over {used} pairs")
Sj = psqrt(Nj)
Si = psqrt(Ni)
print("  balance |N_i^{1/2} - Omega_ij N_j^{1/2} Omega_ij^-1| =", nrm(Si - Oij @ Sj @ inv(Oij)))
Ot = randgl(d)
Bj, Bi = inv(Ot) @ A, Ot @ B
print("  arbitrary Omega~: factorization residuals", nrm(Ot @ Bj - A), nrm(inv(Ot) @ Bi - B),
      "; |B_j^2 - N_j| =", round(nrm(Bj @ Bj - Nj), 3), "(unbalanced factorizations exist)")

print("== 7. Domain of prop:int-reciprocal on the moment family (identity frames, Sigma = I, mu1 = e1, mu2 = e2)")
for p in [-3.0, -4.0, -5.0]:
    l12 = 2.0
    l21 = p / l12
    T12 = np.eye(2) + l12 * np.array([[0., 1], [0, 0]])
    T21 = np.eye(2) + l21 * np.array([[0., 0], [1, 0]])
    N2n = T21 @ T12
    print(f"  lambda12*lambda21 = {p}: N2 = {N2n.tolist()}, eigenvalues = {np.round(np.linalg.eigvals(N2n), 6)}")
Ns = sp.Matrix([[1, 2], [-2, -3]])
P, Jf = Ns.jordan_form()
print("  product -4: Jordan form of N2 =", Jf.tolist())
X = sp.Matrix(2, 2, sp.symbols("x0:4", real=True))
sol = sp.solve(list(X * X - sp.Matrix([[1, 2], [sp.Rational(-5, 2), -4]])), list(X), dict=True)
realsol = [s for s in sol if all(sp.im(v) == 0 for v in s.values())]
print("  product -5: real solutions of X^2 = N2 (symbols declared real):", len(sol))
sol4 = sp.solve(list(X * X - Ns), list(X), dict=True)
print("  product -4: real solutions of X^2 = N2:", [s for s in sol4 if all(sp.im(v) == 0 for v in s.values())])

print("== 8. Principal root versus representations: tied links lose the tie, Omega leaves rho(G)")
a, b = 0.3, 0.8 * np.pi - 0.3
rho_q = lambda gm: gm
rho_s = lambda gm: sla.block_diag(gm, sym2(gm))
k = rot(a) @ inv(psqrt(rot(b) @ rot(a)))
for name, rho in [("charge 1 (fundamental)", rho_q), ("fundamental + Sym^2", rho_s)]:
    Tij, Tji = rho(rot(a)), rho(rot(b))
    N = Tji @ Tij
    Om = Tij @ inv(psqrt(N))
    print(f"  {name}: |Omega^sector - rho(Omega^G)| = {nrm(Om - rho(k)):.3f}")
Nsec = rho_s(rot(0.8 * np.pi))
root = psqrt(Nsec)
kk = np.real(root[:2, :2])
print("  fund + Sym^2: fundamental block of the principal root is a rotation by",
      round(np.arctan2(kk[1, 0], kk[0, 0]) / np.pi, 4), "pi; |Sym^2 block - Sym^2(fund block)| =",
      round(nrm(root[2:, 2:] - sym2(kk)), 4))
Om_s = np.real(rho_s(rot(a)) @ inv(root))
fb = Om_s[:2, :2]
print("  Omega^s in rho_s(SO(2))?  |Omega^s - rho_s(its fund block)| =", round(nrm(Om_s - rho_s(fb)), 4))
r2 = lambda ang: rot(2 * ang)
N2c = r2(b) @ r2(a)
Om2 = np.real(r2(a) @ inv(psqrt(N2c)))
ang2 = np.arctan2(Om2[1, 0], Om2[0, 0]) / 2
angq = np.arctan2(k[1, 0], k[0, 0])
print("  charge 2 alone: Omega^s = rho_s(psi), psi =", round(ang2 / np.pi, 4), "pi (mod pi); Omega^q angle =",
      round(angq / np.pi, 4), "pi; mismatch of reciprocal parts (mod pi) =", round(((ang2 - angq) % np.pi) / np.pi, 4), "pi")
X0 = np.array([[0., -0.8 * np.pi], [0.8 * np.pi, 0.]])
print("  safe domain: n = exp(X) with spectrum of d rho_s(X) in |Im| < pi gives rho(n)^{1/2} = rho(exp(X/2)):",
      "max |Im spec dSym2(X)| / pi =", round(1.6, 3), "> 1 here")

print("== 9. prop:int-loop-blindness: stabilizers of centered Gaussians")
def kl_gauss0(S1, S2):
    kk_ = S1.shape[0]
    return 0.5 * (np.trace(inv(S2) @ S1) - kk_ + np.log(np.linalg.det(S2) / np.linalg.det(S1)))
Sa = np.diag([1.0, 3.0])
for ang in [0.7, np.pi / 2, np.pi]:
    R = rot(ang)
    print(f"  Sigma=diag(1,3), rotation {ang:.4f}: KL(nu || R#nu) = {kl_gauss0(Sa, R @ Sa @ R.T):.6f};"
          f" isotropic: {kl_gauss0(np.eye(2), R @ R.T):.2e}")
R3 = sla.block_diag(rot(0.7), np.eye(1))
S3 = np.diag([1.0, 1.0, 3.0])
print("  Sigma=diag(1,1,3), rotation 0.7 in the (1,2)-plane: KL =", kl_gauss0(S3, R3 @ S3 @ R3.T))
for dd in [2, 3, 5]:
    S = randspd(dd)
    w, V = np.linalg.eigh(S)
    R = np.eye(dd) - 2 * (np.outer(V[:, 0], V[:, 0]) + np.outer(V[:, 1], V[:, 1]))
    print(f"  random SPD d={dd}: half-turn in an eigenplane, det {np.linalg.det(R):+.0f}, |R S R^T - S| = {nrm(R @ S @ R.T - S):.1e}")
Y = rng.standard_normal((100000, 2)) @ np.diag([1.0, np.sqrt(3.0)])
print("  shared-sample pair separates H = -I from H = I: mean |HY - Y| =", round(float(np.mean(nrm(-Y - Y, axis=1))), 4), "versus 0")

print("== 10. prop:int-loop-defect: invariance and the reference-consistency hypothesis")
def Qdef(Wm, S):
    kk_ = S.shape[0]
    return 0.5 * np.trace(S @ (np.eye(kk_) - Wm).T @ inv(S) @ (np.eye(kk_) - Wm))
A, B = randgl(d), randgl(d)
Wm = A @ B
S = randspd(d)
g = randgl(d)
sq = np.real(psqrt(S))
print("  trace form minus Frobenius form:", Qdef(Wm, S) - 0.5 * nrm(inv(sq) @ (np.eye(d) - Wm) @ sq) ** 2)
print("  passive change residual:", Qdef(g @ Wm @ inv(g), g @ S @ g.T) - Qdef(Wm, S))
mu1, mu2 = rng.standard_normal(d), rng.standard_normal(d)
S1, S2 = randspd(d), randspd(d)
def loop_Q(m1, m2, Sa_, Sb_, rule):
    L12, L21 = rule(m1, m2, Sb_), rule(m2, m1, Sa_)
    return Qdef(sla.expm(L12) @ sla.expm(L21), Sa_)
eucl = lambda mi, mj, Sj: 0.7 * np.outer(mi, mj)
moment = lambda mi, mj, Sj: 0.7 * np.outer(mi, mj) @ inv(Sj)
hh = randgl(d)
for nm, rule in [("moment (equivariant)", moment), ("Euclidean (not equivariant)", eucl)]:
    q0 = loop_Q(mu1, mu2, S1, S2, rule)
    q1 = loop_Q(inv(hh) @ mu1, inv(hh) @ mu2, inv(hh) @ S1 @ inv(hh).T, inv(hh) @ S2 @ inv(hh).T, rule)
    print(f"  {nm}: Q before = {q0:.6f}, Q after common reference change = {q1:.6f}")
Ytmp = rng.standard_normal((400000, d)) @ np.real(psqrt(S)).T
mc = 0.5 * np.mean(np.einsum("ni,ij,nj->n", Ytmp @ (np.eye(d) - Wm).T, inv(S), Ytmp @ (np.eye(d) - Wm).T))
print("  Q as half the expected Mahalanobis displacement of a centered shared sample: MC", round(mc, 4), "vs", round(Qdef(Wm, S), 4))

print("== 11. Quiver invariants depend on the vertex group")
Th = randgl(d)
Rs = [sla.expm((lambda X_: X_ - X_.T)(rng.standard_normal((d, d)))) for _ in range(2)]
Gs = [randgl(d) for _ in range(2)]
f = lambda T_: np.trace(T_.T @ T_)
print("  open edge tr(Theta^T Theta): SO(3) vertex change residual", f(Rs[0] @ Th @ Rs[1].T) - f(Th),
      "; GL(3) vertex change residual", round(f(Gs[0] @ Th @ inv(Gs[1])) - f(Th), 4))

print("== 12. Culver: real logarithms")
J3 = np.array([[-1., 1, 0], [0, -1, 1], [0, 0, -1]])
for nm, M in [("J2(-1)", np.array([[-1., 1], [0, -1]])),
              ("diag(-1,-2)", np.diag([-1., -2.])),
              ("J1(-1)+J3(-1)", sla.block_diag(np.array([[-1.]]), J3)),
              ("-I2 = two J1(-1)", -np.eye(2))]:
    Lg = sla.logm(M)
    print(f"  {nm}: det = {np.linalg.det(M):+.1f}, max |Im principal log| = {np.max(np.abs(np.imag(Lg))):.4f}")
print("  -I2 has the real logarithm pi*J: |expm(pi J) + I| =", nrm(sla.expm(np.pi * J) + np.eye(2)))

print("== 13. Moment family: a mutual pair with independent means and nonzero coefficients is never reciprocal")
mins = []
for _ in range(2000):
    dd = int(rng.integers(2, 5))
    m1, m2 = rng.standard_normal(dd), rng.standard_normal(dd)
    Sa_, Sb_ = randspd(dd), randspd(dd)
    l12, l21 = rng.standard_normal(2) * 2
    T12 = sla.expm(l12 * np.outer(m1, m2) @ inv(Sb_))
    T21 = sla.expm(l21 * np.outer(m2, m1) @ inv(Sa_))
    mins.append(Qdef(T12 @ T21, Sa_))
print("  min Q12 over 2000 random draws:", min(mins))
m1 = rng.standard_normal(3)
Sc = randspd(3)
c, l12 = 1.7, 0.9
T12 = sla.expm(l12 * np.outer(m1, c * m1) @ inv(Sc))
T21 = sla.expm(-l12 * np.outer(c * m1, m1) @ inv(Sc))
print("  parallel means, equal covariances, lambda21 = -lambda12: |Theta12 Theta21 - I| =", nrm(T12 @ T21 - np.eye(3)))

print("== 14. Law-adjoint pairs: decomposition always exists and Omega is a congruence isometry")
worst, wmin = 0.0, np.inf
for _ in range(200):
    Mi, Mj = randspd(d), randspd(d)
    Tij = randgl(d)
    Tji = inv(Mj) @ Tij.T @ Mi
    Nj = Tji @ Tij
    wmin = min(wmin, np.min(np.linalg.eigvals(Nj).real))
    Om = np.real(Tij @ inv(psqrt(Nj)))
    worst = max(worst, nrm(Om.T @ Mi @ Om - Mj))
print(f"  min real eigenvalue of N_j = {wmin:.3e}; worst |Omega^T M_i Omega - M_j| = {worst:.2e}")
Sc = randspd(d)
m1, m2 = rng.standard_normal(d), rng.standard_normal(d)
lam_s = 0.8
T12 = sla.expm(lam_s * np.outer(m1, m2) @ inv(Sc))
T21 = sla.expm(lam_s * np.outer(m2, m1) @ inv(Sc))
print("  moment family, lambda12 = lambda21, Sigma1 = Sigma2: |Theta21 - Sigma Theta12^T Sigma^-1| =", nrm(T21 - Sc @ T12.T @ inv(Sc)))
Om = np.real(T12 @ inv(psqrt(T21 @ T12)))
print("  and |Omega12 Sigma Omega12^T - Sigma| =", nrm(Om @ Sc @ Om.T - Sc))
S1b = randspd(d)
T21b = sla.expm(lam_s * np.outer(m2, m1) @ inv(S1b))
print("  unequal covariances: |Theta21 - Sigma2 Theta12^T Sigma1^-1| =", round(nrm(T21b - Sc @ T12.T @ inv(S1b)), 4))

print("== 15. Self-loop case of prop:int-reciprocal: Omega_ii = Theta_ii (Theta_ii^2)^{-1/2} = sign(Theta_ii)")
P = randgl(3)
Tii = P @ np.diag([1.5, -0.7, 2.0]) @ inv(P)          # nonidentity self link, one eigenvalue with negative real part
Oii = np.real(Tii @ inv(psqrt(Tii @ Tii)))
sgn = P @ np.diag([1.0, -1.0, 1.0]) @ inv(P)
print("  |Omega_ii - sign(Theta_ii)| =", nrm(Oii - sgn), "; |Omega_ii^2 - I| =", nrm(Oii @ Oii - np.eye(3)),
      "; |Omega_ii - I| =", round(nrm(Oii - np.eye(3)), 4))

print("== 16. Exponential image versus loop holonomy")
cands = {"diag(-2,-1/2) in SL(2)": np.diag([-2.0, -0.5]), "-I + E12 in SL(2)": np.array([[-1., 1], [0, -1]])}
for nm, M in cands.items():
    print(f"  {nm}: trace {np.trace(M):+.2f}, det {np.linalg.det(M):+.2f}, max |Im principal log| = {np.max(np.abs(np.imag(sla.logm(M)))):.4f}")
worst = 0.0
for _ in range(500):
    A = randgl(2)
    if np.linalg.det(A) < 0:
        A[:, 0] *= -1
    Qp, Sp = sla.polar(A)                 # A = Qp Sp, Qp in SO(2), Sp SPD
    Lq = np.real(sla.logm(Qp))
    Ls = np.real(sla.logm(Sp))
    worst = max(worst, nrm(sla.expm(Lq) @ sla.expm(Ls) - A))
print(f"  every sampled A in GL+(2) equals e^X e^Y (polar): worst residual {worst:.2e};"
      " so two-cycle holonomies U_i e^{w_ij} e^{w_ji} U_i^-1 are unrestricted although single reference links are")
