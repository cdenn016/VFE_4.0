"""D05-OBJ s5: miscellaneous checks in scope.

(a) Monderer-Shapley cross derivatives of KL(q1||q2) and KL(q2||q1) on a binary alphabet.
(b) Linear-Gaussian, fixed unit variances, two agents: stationary means of the reciprocal potential,
    the receiver-only agent-wise objective, the per-agent mixture-source free energy (T=1), and a global
    model with one agreement record; directed versus mutual edges.
(c) Record-interrupted reciprocal dynamics on a bistable local term: one assimilation event switches the
    basin, so a record has a persistent effect without any declared record dependence.
(d) Global free energy with streaming records: jumps at records, optimized value nondecreasing.
(e) Per-agent dynamics with unit-mobility rows (def:dyn-per-agent-flow) versus Gibbs-reduced rows
    (the convention of chapters 13 and 14): stability at common equilibria, random search.
Single-threaded, < 20 s.
"""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def sig(z): return 1/(1+np.exp(-z))
def klb(p, r): return p*np.log(p/r) + (1-p)*np.log((1-p)/(1-r))

print("(a) Monderer-Shapley test on Bernoulli laws (probability coordinates x1, x2)")
x1, x2 = sp.symbols('x1 x2', positive=True)
K12 = x1*sp.log(x1/x2) + (1-x1)*sp.log((1-x1)/(1-x2)); K21 = K12.subs({x1: x2, x2: x1}, simultaneous=True)
m1 = sp.simplify(sp.diff(K12, x1, x2)); m2 = sp.simplify(sp.diff(K21, x1, x2))
print("   d2 KL(q1||q2)/dx1dx2 =", m1, " ; d2 KL(q2||q1)/dx1dx2 =", m2)
print("   equality set:", sp.solve(sp.Eq(m1, m2), x1), " (includes x1 = 1 - x2, not only q1 = q2)")

print("\n(b) linear Gaussian, unit variances, references a, b; weights beta")
a, b, be, tau = sp.symbols('a b beta tau', positive=True)
m_1, m_2 = sp.symbols('m1 m2')
def gap(eqs): s = sp.solve(eqs, [m_1, m_2], dict=True)[0]; return sp.simplify(s[m_1]-s[m_2]), s
Erec = (m_1-a)**2/2 + (m_2-b)**2/2 + be*(m_1-m_2)**2            # mutual edges, each beta*(m_i-m_j)^2/2
g_rec, _ = gap([sp.diff(Erec, m_1), sp.diff(Erec, m_2)])
g_agent, _ = gap([(m_1-a) + be*(m_1-m_2), (m_2-b) + be*(m_2-m_1)])
g_mix, _ = gap([(1-be)*(m_1-a) + be*(m_1-m_2), (1-be)*(m_2-b) + be*(m_2-m_1)])
Eglob = (m_1-a)**2/2 + (m_2-b)**2/2 + tau*(m_1-m_2)**2/2           # one agreement record o = y1 - y2 + noise, o = 0, precision tau
g_glob, _ = gap([sp.diff(Eglob, m_1), sp.diff(Eglob, m_2)])
print("   mutual edges: reciprocal gap", g_rec, "| agent-wise (additive local) gap", g_agent, "| per-agent mixture gap", g_mix, "| global gap", g_glob)
print("   at beta = 1: reciprocal", g_rec.subs(be, 1), " agent-wise", g_agent.subs(be, 1))
Edir = (m_1-a)**2/2 + (m_2-b)**2/2 + be*(m_1-m_2)**2/2              # directed edge 2 -> 1 only
_, s_rec = gap([sp.diff(Edir, m_1), sp.diff(Edir, m_2)])
_, s_mix = gap([(1-be)*(m_1-a) + be*(m_1-m_2), m_2 - b])
print("   directed edge 2->1: reciprocal source mean m2 =", sp.simplify(s_rec[m_2]), " ; per-agent source mean m2 =", s_mix[m_2])

print("\n(c) bistable local term KL(q (x) s || R), R diagonal 0.49, off-diagonal 0.01; reciprocal Fisher flow + one record")
R = np.array([[0.49, 0.01], [0.01, 0.49]])
def E0(x, y):
    q = np.array([1-x, x]); s = np.array([1-y, y]); J = np.outer(q, s); return float(np.sum(J*np.log(J/R)))
def flow(t, z):
    lx, ly = z; x, y = sig(lx), sig(ly); h = 1e-7
    dEx = (E0(x+h, y) - E0(x-h, y))/(2*h); dEy = (E0(x, y+h) - E0(x, y-h))/(2*h)
    return [-dEx, -dEy]                       # Bernoulli Fisher flow in log odds: l' = -dE/dp
solA = solve_ivp(flow, (0, 60), [1.0, 1.2], rtol=1e-10, atol=1e-12); zA = solA.y[:, -1]
xA, yA = sig(zA); print("   minimum A: q(1) = %.4f, s(1) = %.4f, E = %.6f" % (xA, yA, E0(xA, yA)))
lik = np.array([0.99, 0.01])                  # record likelihood l(o | value), value = 0, 1; one record per sector
def bayes(p1): return p1*lik[1]/(p1*lik[1] + (1-p1)*lik[0])
for label, (x0, y0) in (("record on the state sector only", (bayes(xA), yA)), ("records on both sectors", (bayes(xA), bayes(yA)))):
    print("   %s: after assimilation q(1) = %.4f, s(1) = %.4f, E = %.6f (change %+.6f)" % (label, x0, y0, E0(x0, y0), E0(x0, y0) - E0(xA, yA)))
    solB = solve_ivp(flow, (0, 80), [np.log(x0/(1-x0)), np.log(y0/(1-y0))], rtol=1e-10, atol=1e-12); xB, yB = sig(solB.y[:, -1])
    which = "minimum A (record effect transient)" if xB > 0.5 else "the other minimum B (record effect persistent)"
    print("      flow then converges to q(1) = %.4f, s(1) = %.4f, E = %.6f: %s" % (xB, yB, E0(xB, yB), which))

print("\n(d) global free energy with streaming records (finite model, Bayes-optimal recognition)")
rng = np.random.default_rng(11)
P0 = rng.random(5) + 0.1; P0 /= P0.sum(); liks = [rng.random(5)*0.9 + 0.05 for _ in range(4)]
logZ = 0.0; post = P0.copy(); out = []
for lk in liks:
    jump = float(np.sum(post*(-np.log(lk))))       # F_{k+1}(Q) - F_k(Q) at the current optimizer Q = posterior_k
    unn = post*lk; z = unn.sum(); logZ += np.log(z); post = unn/z
    out.append((jump, -logZ))
print("   jumps of F at the current recognition when a record arrives:", [round(j, 4) for j, _ in out])
print("   optimized free energy -log P(o_1:k):", [round(float(v), 4) for _, v in out], " (nondecreasing, discrete records)")

print("\n(e) per-agent flows: unit-mobility rows versus Gibbs-reduced rows, identity channels, Bernoulli agents")
def make(N, rho, L0):
    p = [(i+1) % N for i in range(N)]
    def full(z):
        l, Lr = z[:N], z[N:]; bp = sig(Lr); lp = l[p]
        dl = -(1-bp)*(l-rho) - bp*(l-lp)
        dL = -(klb(sig(l), sig(lp)) - klb(sig(l), sig(rho)) + Lr - L0)
        return np.concatenate([dl, dL])
    def Lstar(l): return L0 - (klb(sig(l), sig(l[p])) - klb(sig(l), sig(rho)))
    def red(l):
        bp = sig(Lstar(l)); return -(1-bp)*(l-rho) - bp*(l-l[p])
    return full, red, Lstar
def jac(f, z, h=1e-6):
    return np.array([(f(z+h*e) - f(z-h*e))/(2*h) for e in np.eye(len(z))]).T
found = []; ntried = 0
for trial in range(60):
    N = int(rng.choice([2, 3])); rho = rng.normal(0, 4, N); L0 = rng.normal(1.0, 2.0, N)
    full, red, Lstar = make(N, rho, L0)
    for start in range(3):
        l0 = rng.normal(0, 4, N)
        sol = solve_ivp(lambda t, l: red(l), (0, 80), l0, rtol=1e-9, atol=1e-11)
        l = sol.y[:, -1]
        if np.linalg.norm(red(l)) > 1e-8: continue
        ntried += 1
        z = np.concatenate([l, Lstar(l)])
        er = np.max(np.linalg.eigvals(jac(red, l)).real); ef = np.max(np.linalg.eigvals(jac(full, z)).real)
        if np.sign(er) != np.sign(ef): found.append((N, rho.round(3), L0.round(3), l.round(4), er, ef))
print("   equilibria compared:", ntried, " stability disagreements found:", len(found))
for f in found[:3]: print("   ", f)
