"""D03-GEOM s04: geo:fisher-loss, geo:moving-channel, geo:signed-comparison witness,
geo:retained-connection, geo:minimal-contraction, geo:defect-composition.

(a) fixed kernel on a finite exponential family: I(P) - I(PK) = E Cov(l | Ybar);
    equality case with a kernel sufficient for one direction.
(b) Bernoulli pair (1/2 + t/4, 1/2 + t^2/4): discarded score and Fisher.
(c) moving channel identity on a random finite channel; logistic example.
(d) horizontal-defect witness (exact rationals).
(e) two-location retention example and minimal-contraction identity (exact).
(f) SO(3) acting on three unit-covariance normal location laws in R^3, isotropic
    noise channels (fixed, rotation equivariant): random jets, identity
    geo:minimal-contraction-identity, retention criterion geo:connection-retention.
(g) defect composition for two successive noise channels.
"""
import numpy as np
import sympy as sp
rng = np.random.default_rng(29102026)

print("== (a) fixed-kernel Fisher loss")
T = rng.normal(size=(6, 2)); th = np.array([0.3, -0.4])
def law(t):
    e = np.exp(T @ t); return e/e.sum()
Pv = law(th); ell = T - Pv @ T
K = rng.random((6, 3)); K /= K.sum(1, keepdims=True)
PK = Pv @ K
post = (Pv[:, None]*K)/PK[None, :]           # P(y | ybar)
lbar = post.T @ ell                            # E[l | ybar], 3 x 2
I_in = (Pv[:, None, None]*np.einsum('yi,yj->yij', ell, ell)).sum(0)
I_out = (PK[:, None, None]*np.einsum('bi,bj->bij', lbar, lbar)).sum(0)
ECov = sum(PK[b]*((post[:, b][:, None, None]*np.einsum('yi,yj->yij', ell - lbar[b], ell - lbar[b])).sum(0)) for b in range(3))
# output Fisher by finite differences of log PK
eps = 1e-6; Ifd = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        def lp(t): return np.log(law(t) @ K)
        ei = np.eye(2)[i]*eps
        si = (lp(th + ei) - lp(th - ei))/(2*eps)
        ej = np.eye(2)[j]*eps
        sj = (lp(th + ej) - lp(th - ej))/(2*eps)
        Ifd[i, j] = PK @ (si*sj)
print("  |I_in - I_out - E Cov| =", f"{np.max(np.abs(I_in - I_out - ECov)):.2e}", "; |I_out - FD output Fisher| =", f"{np.max(np.abs(I_out - Ifd)):.2e}",
      "; eig(E Cov) =", np.round(np.linalg.eigvalsh(ECov), 4))
# equality case: kernel that is a function of T[:,0] level sets -> make T[:,0] take 2 values
T2 = T.copy(); T2[:, 0] = np.array([1, 1, 1, -1, -1, -1.0])
def law2(t):
    e = np.exp(T2 @ t); return e/e.sum()
P2 = law2(th); ell2 = T2 - P2 @ T2
Kd = np.zeros((6, 2)); Kd[:3, 0] = 1; Kd[3:, 1] = 1     # deterministic statistic sign(T0)
PK2 = P2 @ Kd; post2 = (P2[:, None]*Kd)/PK2[None, :]; lb2 = post2.T @ ell2
EC2 = sum(PK2[b]*((post2[:, b][:, None, None]*np.einsum('yi,yj->yij', ell2 - lb2[b], ell2 - lb2[b])).sum(0)) for b in range(2))
print("  equality case: v=e1 loss E Var(l_1|Ybar) =", f"{EC2[0,0]:.2e}", "; v=e2 loss =", f"{EC2[1,1]:.4f}")

print("== (b) Bernoulli pair")
t = sp.symbols('t', real=True)
p2 = sp.Rational(1, 2) + t**2/4
s2 = sp.diff(sp.log(p2), t); I2 = sp.simplify(sp.diff(p2, t)**2/(p2*(1 - p2)))
print("  second-coordinate score at 0:", s2.subs(t, 0), "; discarded Fisher:", sp.factor(I2), "; at 0:", I2.subs(t, 0))

print("== (c) moving channel")
th0 = 0.37
Ttab = rng.normal(size=5); Wtab = rng.normal(size=(5, 4))
def law1(t): e = np.exp(Ttab*t); return e/e.sum()
def chan(t): e = np.exp(Wtab*t + np.arange(4)[None, :]*0.1); return e/e.sum(1, keepdims=True)
def dt(f, t, e=1e-6): return (f(t + e) - f(t - e))/(2*e)
P = law1(th0); Kt = chan(th0); joint = P[:, None]*Kt; out = joint.sum(0)
l = Ttab - P @ Ttab
b = Wtab - (Kt*Wtab).sum(1, keepdims=True)
print("  max |E[b|Y]| =", f"{np.max(np.abs((Kt*b).sum(1))):.1e}")
s = l[:, None] + b
Iy = P @ l**2; Ichan = P @ (Kt*b**2).sum(1)
cmean = (joint*s).sum(0)/out
ECov = (joint*(s - cmean[None, :])**2).sum()
Iout_formula = Iy + Ichan - ECov
Iout_fd = out @ (dt(lambda u: np.log((law1(u)[:, None]*chan(u)).sum(0)), th0))**2
print("  I(Ybar) formula =", f"{Iout_formula:.10f}", "; finite-difference =", f"{Iout_fd:.10f}", "; residual", f"{abs(Iout_formula - Iout_fd):.1e}")
sg = 1/(1 + sp.exp(-t))
print("  logistic output Fisher residual:", sp.simplify(sp.diff(sg, t)**2/(sg*(1 - sg)) - sp.exp(-t)/(1 + sp.exp(-t))**2))

print("== (d) horizontal-defect witness")
for cc in (sp.Rational(1, 10), sp.Rational(1, 2)):
    diff = 1 - sp.Rational(1, 2)*(1 + cc)**2
    crit = 2*sp.Rational(1, 2)*1*cc + sp.Rational(1, 2)*cc**2 <= sp.Rational(1, 2)
    print(f"  c={cc}: h - f^*hbar = {diff}; criterion 2g(Psi u,a)+|a|^2 <= Delta(u,u): {crit}")

print("== (e) two-location retention example")
a = sp.symbols('a')
cost = sp.Rational(1, 2)*(1 + a)**2 + (-1 + a)**2
amin = sp.solve(sp.diff(cost, a), a)[0]
hbar = cost.subs(a, amin)
print("  coarse minimizer a =", amin, "; hbar_min =", hbar, "; identity 2 - hbar - 1/2 - (3/2) a^2 =", 2 - hbar - sp.Rational(1, 2) - sp.Rational(3, 2)*amin**2)

print("== (f) SO(3), three agents, isotropic noise channels")
def hat(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def Jm(mus): return np.vstack([np.column_stack([hat(e) @ m for e in np.eye(3)]) for m in mus])
def minimize(Gd, Jmat, v):
    k = Jmat.T @ Gd @ Jmat
    a = -np.linalg.solve(k, Jmat.T @ Gd @ v)
    r = v + Jmat @ a
    return a, r, r @ Gd @ r, k
worst = 0.0; worst_ret = 0.0
for trial in range(200):
    mus = rng.normal(size=(3, 3)); w = rng.random(3) + 0.2; tau2 = rng.random(3)*2
    G = np.diag(np.repeat(w, 3)); Gb = np.diag(np.repeat(w/(1 + tau2), 3)); D = G - Gb
    Jx = Jm(mus); v = rng.normal(size=9)
    aF, r, hmin, k = minimize(G, Jx, v)
    abF, rb, hbmin, kb = minimize(Gb, Jx, v)
    lhs = hmin - hbmin
    rhs = r @ D @ r + (aF - abF) @ kb @ (aF - abF)
    worst = max(worst, abs(lhs - rhs)/max(1, abs(lhs)))
    # retention criterion: Delta(r, J xi) = 0 iff aF minimizes coarse
    crit = np.linalg.norm(Jx.T @ D @ r); shift = np.linalg.norm(aF - abF)
    worst_ret = max(worst_ret, abs(crit) < 1e-12 and shift > 1e-8)
print("  max relative residual of minimal-contraction identity over 200 trials:", f"{worst:.1e}")
# equal noise -> Delta proportional to G -> retention holds
mus = rng.normal(size=(3, 3)); w = rng.random(3) + 0.2; tau2 = np.full(3, 0.7)
G = np.diag(np.repeat(w, 3)); Gb = G/(1.7); D = G - Gb; Jx = Jm(mus); v = rng.normal(size=9)
aF, r, hmin, k = minimize(G, Jx, v); abF, *_ = minimize(Gb, Jx, v)
print("  equal noise: |Delta(r,J.)| =", f"{np.linalg.norm(Jx.T @ D @ r):.1e}", "; |A_fine - A_coarse| =", f"{np.linalg.norm(aF - abF):.1e}")
tau2 = np.array([0.1, 0.9, 2.0]); Gb = np.diag(np.repeat(w/(1 + tau2), 3)); D = G - Gb
aF, r, hmin, k = minimize(G, Jx, v); abF, *_ = minimize(Gb, Jx, v)
print("  unequal noise: |Delta(r,J.)| =", f"{np.linalg.norm(Jx.T @ D @ r):.3f}", "; |A_fine - A_coarse| =", f"{np.linalg.norm(aF - abF):.3f}")

print("== (g) defect composition")
t1, t2 = sp.symbols('tau1 tau2', positive=True)
g0, g1, g2 = 1, 1/(1 + t1), 1/(1 + t1 + t2)
print("  residual:", sp.simplify((g0 - g2) - ((g0 - g1) + (g1 - g2))))
