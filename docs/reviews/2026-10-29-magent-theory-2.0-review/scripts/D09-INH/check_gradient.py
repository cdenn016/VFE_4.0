r"""D09-INH check: deterministic gradient-inheritance claims in chapters/05_inheritance.tex:168-372.

  A. Center-of-mass example (lines 219-227): criterion (inh:horizontal-criterion) with a = 1,
     and the side claim "E != chi o E_c o R whenever W' != 0", tested against W(s) = -s^2/4.
  B. Fiber-minimization example (lines 245-255).
  C. Uniform necessity of the Riemannian-submersion condition (lines 227-234), symbolic check
     for a linear retention.
  D. inh:no-underdamped: the stated linear-algebra conclusion on random instances, and two
     counterexamples to the heading/remark ("exact inheritance of a gradient flow creates no
     oscillatory mode"; "oscillatory collective modes require nonreciprocal fields, explicit
     driving, or declared second-order kinetics") at parent equilibria whose fibers contain
     no fine critical point:
       D1. one agent, non-product metric, potential bounded below (no fine critical point);
       D2. Euclidean (agent-local) metric on R^3, coupled potential, helical retention.
  E. inh:role-loss numbers.
"""
import numpy as np
import sympy as sp

# ---------- A ----------
x1, x2, s = sp.symbols('x1 x2 s', real=True)
W = sp.Function('W')
E = sp.Rational(1, 2) * x1**2 + sp.Rational(1, 2) * x2**2 + W(x1 - x2)
z = (x1 + x2) / sp.sqrt(2)
v = sp.Matrix([1, 1]) / sp.sqrt(2)          # horizontal unit vector, DR v = 1
dE = sp.Matrix([sp.diff(E, x1), sp.diff(E, x2)])
crit = sp.simplify((dE.T * v)[0] - z)          # d^2 dE(v) - a dE_c(DR v), d = a = 1, E_c' = z
print("A. criterion residual (any W):", crit)
Wq = -s**2 / 4
Eq = E.subs(W(x1 - x2), Wq.subs(s, x1 - x2)).doit()
Eq = sp.simplify(Eq.replace(W, sp.Lambda(s, Wq)))
print("   with W(s) = -s^2/4:  E =", sp.factor(sp.expand(Eq)), "; E - E_c(R(x)) =", sp.simplify(sp.expand(Eq - z**2 / 2)))
print("   W'(s) =", sp.diff(Wq, s), "(not identically zero), yet E = E_c o R exactly")
# separation dynamics with W = -s^2/4
B = -sp.Matrix([sp.diff(Eq, x1), sp.diff(Eq, x2)])
print("   d/dt (x1 - x2) under W = -s^2/4:", sp.simplify(B[0] - B[1]))

# ---------- B ----------
u, vv = sp.symbols('u v', real=True)
Eb = sp.Rational(1, 2) * (u - vv)**2 + sp.Rational(1, 2) * vv**2
Ec = sp.minimum(Eb, u) if False else sp.simplify(Eb.subs(u, sp.solve(sp.diff(Eb, u), u)[0]))
print("B. E_c(v) = inf_u E =", Ec, "; dE(d_v) - dE_c =", sp.simplify(sp.diff(Eb, vv) - sp.diff(Ec, vv)),
      "; pushed field =", sp.simplify(-sp.diff(Eb, vv)))

# ---------- C ----------
rng = np.random.default_rng(1)
M = rng.normal(size=(2, 4))
G = rng.normal(size=(4, 4)); G = G @ G.T + 4 * np.eye(4)        # fine co-metric g^{-1}
Gc = M @ G @ M.T                                                  # pushed co-metric
c = rng.normal(size=2)                                            # dE_c at R(x)
lhs = M @ (G @ (M.T @ c))                                         # DR grad_g (E_c o R)
print("C. DR grad_g(E_c o R) - (M g^{-1} M^T) dE_c =", np.abs(lhs - Gc @ c).max(),
      "(equals g_c^{-1} dE_c for all dE_c iff g_c^{-1} = M g^{-1} M^T)")

# ---------- D: stated conclusion ----------
worst = 0.0
for trial in range(200):
    n, m = 5, 2
    A = rng.normal(size=(n, n)); g = A @ A.T + np.eye(n)
    # quadratic E with Hessian H preserving ker M for a submersion M:
    Mx = rng.normal(size=(m, n))
    Q, _ = np.linalg.qr(np.hstack([np.linalg.pinv(Mx), rng.normal(size=(n, n - m))]))
    # build L = -g^{-1} H with ker M L-invariant: choose H = g P for P symmetric wrt g? simpler:
    # pick L self-adjoint for g with an invariant subspace ker M: L = V D V^{-1} with g-orthonormal V
    gi_half = np.linalg.cholesky(np.linalg.inv(g))
    # g-orthonormal basis adapted to ker M
    K = np.linalg.svd(Mx)[2][m:].T            # columns span ker M (Euclidean)
    basis = np.hstack([K, rng.normal(size=(n, m))])
    # g-Gram-Schmidt
    Vb = []
    for col in basis.T:
        w = col.copy()
        for b in Vb:
            w -= (b @ g @ w) * b
        Vb.append(w / np.sqrt(w @ g @ w))
    V = np.array(Vb).T
    D = np.diag(rng.normal(size=n))
    L = V @ D @ np.linalg.inv(V)
    # check ker M invariant (first n-m columns of V span ker M)
    Lc = Mx @ L @ np.linalg.pinv(Mx)
    ev = np.linalg.eigvals(Lc)
    evL = np.linalg.eigvals(L)
    dist = max(min(abs(e - evL)) for e in ev)
    worst = max(worst, dist + max(abs(ev.imag)))
print("D. stated conclusion, 200 random g-self-adjoint L with invariant ker M: max(dist(spec L^c, spec L) + |Im|) =", f"{worst:.2e}")

# ---------- D1: non-product metric, bounded-below potential, no fine critical point ----------
z1, z2, y = sp.symbols('z1 z2 y', real=True)
delta, eps = sp.Rational(1, 2), sp.Rational(3, 1)
wz = sp.Matrix([-eps * z2, eps * z1])                     # w(z) = eps J z
Gco = sp.Matrix([[1, 0, wz[0] * sp.exp(y)],
                 [0, 1, wz[1] * sp.exp(y)],
                 [wz[0] * sp.exp(y), wz[1] * sp.exp(y), 1 + 2 * (wz.dot(wz)) * sp.exp(2 * y)]])
E1 = delta / 2 * (z1**2 + z2**2) + sp.exp(-y)
grad_coords = sp.Matrix([sp.diff(E1, z1), sp.diff(E1, z2), sp.diff(E1, y)])
B1 = -(Gco * grad_coords)
B1 = sp.simplify(B1)
schur = sp.simplify(Gco[2, 2] - (Gco[0, 2]**2 + Gco[1, 2]**2))
print("D1. co-metric Schur complement G_yy - |G_zy|^2 =", schur, "(> 0, so G and g = G^{-1} are positive definite on R^3)")
print("    fine field B =", list(B1))
print("    dB_z/dy =", [sp.simplify(sp.diff(B1[i], y)) for i in range(2)], "(zero: R(z,y) = z is projectable)")
Jc = sp.Matrix([[sp.diff(B1[i], v_) for v_ in (z1, z2)] for i in range(2)])
print("    parent linearization at z = 0:", Jc.subs({z1: 0, z2: 0}), " eigenvalues:", list(Jc.subs({z1: 0, z2: 0}).eigenvals()))
print("    dE/dy = -exp(-y) never vanishes: the fiber over z = 0 has no critical point of E; E >= 0")
# numerical integration: fine flow descends E, parent spirals
f1 = sp.lambdify((z1, z2, y), list(B1), 'numpy')
e1 = sp.lambdify((z1, z2, y), E1, 'numpy')
X = np.array([0.05, 0.0, 0.0]); dt = 1e-3  # small |z0| keeps y finite (Riccati bound in report)
Es = [e1(*X)]; th = []
for k in range(8000):
    k1 = np.array(f1(*X)); k2 = np.array(f1(*(X + dt / 2 * k1))); k3 = np.array(f1(*(X + dt / 2 * k2))); k4 = np.array(f1(*(X + dt * k3)))
    X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Es.append(e1(*X)); th.append(np.arctan2(X[1], X[0]))
dEs = np.diff(Es)
print(f"    RK4 to t=8: max increment of E = {dEs.max():.2e} (<= 0 means descent); final z = {X[:2]}, y = {X[2]:.3f};"
      f" winding of z about 0 = {np.unwrap(th)[-1] / (2 * np.pi):.2f} turns")

# ---------- D2: Euclidean metric, three 1-d agents, coupled potential, helical retention ----------
a1, a2, a3 = sp.symbols('a1 a2 a3', real=True)
kap, cc, dl, mu = sp.Rational(1), sp.Rational(2), sp.Rational(1, 10), sp.Rational(1, 2)
Rot = sp.Matrix([[sp.cos(kap * a3), -sp.sin(kap * a3)], [sp.sin(kap * a3), sp.cos(kap * a3)]])
zz = Rot * sp.Matrix([a1, a2])
Psi = mu / 2 * zz[0]**2                                   # anisotropic, so it couples a3 to (a1,a2)
E2 = dl / 2 * (a1**2 + a2**2) + cc * a3 + Psi
Bf = -sp.Matrix([sp.diff(E2, a1), sp.diff(E2, a2), sp.diff(E2, a3)])   # Euclidean gradient flow
DR = zz.jacobian([a1, a2, a3])
zdot = sp.simplify(DR * Bf)
Z1, Z2 = sp.symbols('Z1 Z2', real=True)
# express zdot in z-coordinates: substitute (a1,a2) = Rot^{-1} z
sub = {a1: sp.cos(kap * a3) * Z1 + sp.sin(kap * a3) * Z2, a2: -sp.sin(kap * a3) * Z1 + sp.cos(kap * a3) * Z2}
zdot_z = sp.simplify(sp.expand_trig(zdot.subs(sub)))
print("D2. parent field in z (should not contain a3):", list(zdot_z), " free symbols:", zdot_z.free_symbols)
mixed = sp.simplify(sp.diff(E2, a1, a3))
print("    d^2E/da1 da3 =", mixed, "(not identically zero: the agents (a1,a2) and a3 are energetically coupled)")
Jz = zdot_z.jacobian([Z1, Z2]).subs({Z1: 0, Z2: 0})
print("    parent linearization at 0:", Jz, " eigenvalues:", list(Jz.eigenvals()))

# ---------- E ----------
for p1, p2 in ((0.8, 0.2), (0.2, 0.8)):
    law = ((1 - p1) * (1 - p2), p1 * (1 - p2) + (1 - p1) * p2, p1 * p2)
    print(f"E. (p1,p2)=({p1},{p2}): law of K1+K2 = {tuple(round(t, 10) for t in law)}, d/dt E = {-(p1 + 2 * p2):.2f}")

# ---------- D3: two non-interacting agents, agent-local metrics, energy bounded below ----------
# agent A: z in R^2 with Euclidean metric; agent B: y in R with metric g_B(y) = exp(-y)/k.
# E = (mu/2)|z|^2 + exp(-y) >= 0, no critical point.  Retention w = Rot(kappa*y) z ("A read in B's frame").
zz1, zz2, yy = sp.symbols('zz1 zz2 yy', real=True)
mu3, k3, kap3 = sp.Rational(1, 2), sp.Rational(2), sp.Rational(3, 2)
E3 = mu3 / 2 * (zz1**2 + zz2**2) + sp.exp(-yy)
gB = sp.exp(-yy) / k3
B3 = sp.Matrix([-sp.diff(E3, zz1), -sp.diff(E3, zz2), -sp.diff(E3, yy) / gB])   # -g^{-1} dE, block metric
Rot3 = sp.Matrix([[sp.cos(kap3 * yy), -sp.sin(kap3 * yy)], [sp.sin(kap3 * yy), sp.cos(kap3 * yy)]])
w3 = Rot3 * sp.Matrix([zz1, zz2])
wdot = sp.simplify(w3.jacobian([zz1, zz2, yy]) * B3)
W1, W2 = sp.symbols('W1 W2', real=True)
sub3 = {zz1: sp.cos(kap3 * yy) * W1 + sp.sin(kap3 * yy) * W2, zz2: -sp.sin(kap3 * yy) * W1 + sp.cos(kap3 * yy) * W2}
wdot_w = sp.simplify(sp.expand_trig(wdot.subs(sub3)))
print("D3. fine field (block-diagonal metric diag(1,1,exp(-y)/k)):", list(B3))
print("    parent field in w (no y):", list(wdot_w), " free symbols:", wdot_w.free_symbols)
J3 = wdot_w.jacobian([W1, W2])
print("    parent linearization:", J3, " eigenvalues:", list(J3.eigenvals()))
