r"""D12-SMOOTH check 5.
(a) smooth:exponential-family: for A = N(0,1), T = (v, v^2), kappa_0 = exp(theta.T - psi(theta)) A,
    H(t) = a theta.grad psi(a theta) - psi(a theta) equals KL(N(m_t, s_t^2) || N(0,1)) (sympy).
(b) Record example of sec:smooth-records (5/8, 3/8; posteriors (9/10,1/10), (1/2,1/2)).
(c) smooth:feedback / smooth:feedback-update against direct Bayes on fine tables, through
    record, explicit private relaxation, second record; within-mark versus cross-mark
    rearrangements.
"""
import numpy as np
import sympy as sp

# (a)
th1, th2, a = sp.symbols("theta1 theta2 a", real=True)
psi = -sp.Rational(1, 2) * sp.log(1 - 2 * th2) + th1 ** 2 / (2 * (1 - 2 * th2))
grad = [sp.diff(psi, th1), sp.diff(psi, th2)]
sub = {th1: a * th1, th2: a * th2}
H_formula = (a * th1 * grad[0].subs(sub, simultaneous=True) + a * th2 * grad[1].subs(sub, simultaneous=True)
             - psi.subs(sub, simultaneous=True))
s2 = 1 / (1 - 2 * a * th2); m = a * th1 * s2
KL = sp.Rational(1, 2) * (s2 + m ** 2 - 1 - sp.log(s2))
print("(a) H_formula - KL simplified:", sp.simplify(H_formula - KL))
num = {th1: 0.7, th2: -0.9, a: 0.37}
print("    numeric check:", float(H_formula.subs(num)), float(KL.subs(num)))
# moment map: grad psi = (E v, E v^2) under P_theta
Ev = sp.simplify(grad[0]); Ev2 = sp.simplify(grad[1])
print("    grad psi = (E v, E v^2):", sp.simplify(Ev - th1 / (1 - 2 * th2)), sp.simplify(Ev2 - (1 / (1 - 2 * th2) + (th1 / (1 - 2 * th2)) ** 2)))

# (b)
k1, k2 = np.array([0.75, 0.25]), np.array([0.25, 0.75])
Lrec = np.array([0.75, 0.25])
print("(b) record probabilities:", k1 @ Lrec, k2 @ Lrec, "; posteriors:", k1 * Lrec / (k1 @ Lrec), k2 * Lrec / (k2 @ Lrec))

# (c)
rng = np.random.default_rng(3)
nV = 4
A = np.array([0.3, 0.3, 0.2, 0.2])           # A(0)=A(1), A(2)=A(3)
Tm = np.array([0, 0, 1, 1])                  # mark
nZ, nO = 2, 3
mu = np.array([0.4, 0.6])
kap = rng.dirichlet(np.ones(nV), size=nZ)
Lk = rng.dirichlet(np.ones(nO), size=(nZ, 2))  # L[z, d, o]


def marked(kap):
    return [[(np.log(kap[z, v] / A[v]), Tm[v], A[v]) for v in range(nV)] for z in range(nZ)]


def eta_canon(eta, nd=10):
    out = []
    for z in range(nZ):
        d = {}
        for r, dd, m in eta[z]:
            key = (round(r, nd), dd)
            d[key] = d.get(key, 0.0) + m
        out.append(sorted((k, round(v, nd)) for k, v in d.items()))
    return out


def marked_record(mu, eta, o):
    Nz = np.array([sum(np.exp(r) * Lk[z, dd, o] * m for r, dd, m in eta[z]) for z in range(nZ)])
    Nn = Nz @ mu
    mu_p = Nz * mu / Nn
    eta_p = [[(r + np.log(Lk[z, dd, o]) - np.log(Nz[z]), dd, m) for r, dd, m in eta[z]] for z in range(nZ)]
    return Nn, mu_p, eta_p


def direct_record(mu, kap, o):
    q = mu[:, None] * kap
    lik = np.array([[Lk[z, Tm[v], o] for v in range(nV)] for z in range(nZ)])
    Po = np.sum(q * lik)
    qp = q * lik / Po
    return Po, qp.sum(1), qp / qp.sum(1, keepdims=True)


def marked_flow(eta, t):
    a = np.exp(-t)
    out = []
    for z in range(nZ):
        Zs = sum(np.exp(a * r) * m for r, dd, m in eta[z])
        out.append([(a * r - np.log(Zs), dd, m) for r, dd, m in eta[z]])
    return out


def direct_flow(kap, t):
    a = np.exp(-t)
    w = kap ** a * A[None, :] ** (1 - a)
    return w / w.sum(1, keepdims=True)


worst = 0.0
for o1, o2 in [(0, 1), (2, 2), (1, 0)]:
    Pm, mu_m, eta_m = marked_record(mu, marked(kap), o1)
    Pd, mu_d, kap_d = direct_record(mu, kap, o1)
    worst = max(worst, abs(Pm - Pd), np.max(np.abs(mu_m - mu_d)))
    eta_m = marked_flow(eta_m, 0.7); kap_d = direct_flow(kap_d, 0.7)
    worst = max(worst, max(abs(a_[0] - b_[0]) for z in range(nZ) for a_, b_ in zip(eta_m[z], marked(kap_d)[z])))
    Pm2, mu_m2, eta_m2 = marked_record(mu_m, eta_m, o2)
    Pd2, mu_d2, kap_d2 = direct_record(mu_d, kap_d, o2)
    worst = max(worst, abs(Pm2 - Pd2), np.max(np.abs(mu_m2 - mu_d2)))
    worst = max(worst, max(abs(a_[0] - b_[0]) for z in range(nZ) for a_, b_ in zip(eta_m2[z], marked(kap_d2)[z])))
print(f"(c) marked feedback vs direct Bayes through record-flow-record: max residual {worst:.2e}")
kw = kap.copy(); kw[:, [0, 1]] = kap[:, [1, 0]]           # within-mark rearrangement (A(0)=A(1))
kx = kap.copy(); kx[:, [1, 2]] = kap[:, [2, 1]]           # cross-mark exchange of values
kx = kx * A[None, :] / A[None, [0, 2, 1, 3]]; kx /= kx.sum(1, keepdims=True)
for name, kk in [("within-mark", kw), ("cross-mark", kx)]:
    same_eta = eta_canon(marked(kk)) == eta_canon(marked(kap))
    P0 = direct_record(mu, kap, 0)[0]; P1 = direct_record(mu, kk, 0)[0]
    post_same = eta_canon(marked_record(mu, marked(kk), 0)[2]) == eta_canon(marked_record(mu, marked(kap), 0)[2])
    print(f"    {name}: same marked spectrum {same_eta}; record law {P0:.6f} vs {P1:.6f}; same posterior marked state {post_same}")
