r"""D17-PA check 3: angular extension (pa:angular-family ... pa:angular-grouping).

  (a) quadrature check of Z, public odds L, Gibbs row weights pa:angular-weights;
  (b) exactness of pa:angular-ode: density-level own-law Fisher velocity projected on the sufficient statistics;
  (c) same for the null pa:angular-null;
  (d) null box pa:null-box / pa:null-bound / pa:null-sign-bound at random box points;
  (e) parameter ODE, three agents: full selects a persistent full block on actual L, null keeps (+,+,-).
"""
import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import solve_ivp

rng = np.random.default_rng(11)
a_pr, p_pr = 0.2, 0.4
NG = 64
ph = 2 * np.pi * np.arange(NG) / NG
Zv = np.array([-1.0, 1.0])
ZZ, PH, PS = np.meshgrid(Zv, ph, ph, indexing="ij")     # [z, phi, psi]

def logI0(r):
    return np.log(i0e(r)) + np.abs(r)

def h_of(Avec):
    return logI0(np.linalg.norm(Avec))

def grad_h(Avec):
    r = np.linalg.norm(Avec)
    if r < 1e-14:
        return np.zeros(2)
    return (i1e(r) / i0e(r)) * Avec / r

def logq_grid(l, A, C, B):
    e = l * ZZ / 2 + (A[0] + ZZ * C[0]) * np.cos(PH) + (A[1] + ZZ * C[1]) * np.sin(PH) + B[0] * np.cos(PS) + B[1] * np.sin(PS)
    return e

def expect(w, f):
    return float(np.mean(w * f) )   # reference: uniform z (mean over 2), Haar x Haar (grid mean)

def public_odds(l, A, C):
    return l + h_of(A + C) - h_of(A - C)

def logZ(l, A, C, B):
    b = np.linalg.norm(B)
    return logI0(b) + np.log((np.exp(l / 2 + h_of(A + C)) + np.exp(-l / 2 + h_of(A - C))) / 2)

def moments(l, A, C, B):
    # E z, E e(phi), E z e(phi), E e(psi), and t = E z cos(phi)
    Pp = np.exp(l / 2 + h_of(A + C)); Pm = np.exp(-l / 2 + h_of(A - C)); s = Pp + Pm
    Pp, Pm = Pp / s, Pm / s
    gp, gm = grad_h(A + C), grad_h(A - C)
    Ez = Pp - Pm
    Ee = Pp * gp + Pm * gm
    Eze = Pp * gp - Pm * gm
    Eps = grad_h(B)
    return Ez, Ee, Eze, Eps

print("(a) quadrature checks of Z, L and the row weights")
worst = 0.0
for _ in range(10):
    l = rng.normal(); A = rng.normal(size=2); C = rng.normal(size=2); B = rng.normal(size=2)
    lq = logq_grid(l, A, C, B)
    Zq = np.mean(np.exp(lq))
    worst = max(worst, abs(np.log(Zq) - logZ(l, A, C, B)))
    q = np.exp(lq) / Zq
    pz = np.array([np.mean(q[0]), np.mean(q[1])]) / 2       # probability masses of z=-1, z=+1
    worst = max(worst, abs(np.log(pz[1] / pz[0]) - public_odds(l, A, C)))
print("  max |log Z - formula|, |log-odds - L| : %.2e" % worst)

def costs_numeric(i, st, eps):
    # D_ii, D_ij by quadrature; st is a list of (l, A, C, B) for 3 agents
    Jc = eps ** 3
    l, A, C, B = st[i]
    lq = logq_grid(l, A, C, B); q = np.exp(lq - logZ(l, A, C, B))
    logG = Jc * ZZ * np.cos(PH) - logI0(Jc)
    D = np.zeros(3)
    for j in range(3):
        if j == i:
            D[j] = expect(q, (lq - logZ(l, A, C, B)) - logG)
        else:
            Lj = public_odds(*st[j][:3])
            lognu = Lj * ZZ / 2 - np.log(np.cosh(Lj / 2))
            D[j] = expect(q, (lq - logZ(l, A, C, B)) - lognu)
    return D, q, lq - logZ(l, A, C, B), logG

def weights_formula(i, st, eps):
    Jc = eps ** 3
    l, A, C, B = st[i]
    Ez, Ee, Eze, Eps = moments(l, A, C, B)
    t_i = Eze[0]
    m_i = np.tanh(public_odds(l, A, C) / 2)
    w = np.zeros(3)
    for j in range(3):
        if j == i:
            w[j] = a_pr * np.exp(Jc * t_i) / np.exp(logI0(Jc))
        else:
            Lj = public_odds(*st[j][:3])
            w[j] = p_pr * np.exp(Lj * m_i / 2) / np.cosh(Lj / 2)
    return w / w.sum(), m_i, t_i

worst = 0.0
for _ in range(10):
    eps = rng.uniform(0.2, 1.2)
    st = [(rng.normal(), rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)) for _ in range(3)]
    for i in range(3):
        D, *_ = costs_numeric(i, st, eps)
        wn = np.array([a_pr if j == i else p_pr for j in range(3)]) * np.exp(-D)
        wn /= wn.sum()
        wf, m_i, t_i = weights_formula(i, st, eps)
        worst = max(worst, np.abs(wn - wf).max())
print("  max |Gibbs row (quadrature) - pa:angular-weights| : %.2e" % worst)

def project(vel, q):
    # least-squares coefficients of vel on centered statistics, weighted by q; returns coef, residual
    T = [ZZ / 2, np.cos(PH), np.sin(PH), ZZ * np.cos(PH), ZZ * np.sin(PH), np.cos(PS), np.sin(PS)]
    Tc = [t - expect(q, t) for t in T]
    G = np.array([[expect(q, x * y) for y in Tc] for x in Tc])
    rhs = np.array([expect(q, x * vel) for x in Tc])
    coef = np.linalg.solve(G, rhs)
    fit = sum(c * t for c, t in zip(coef, Tc))
    res = np.sqrt(expect(q, (vel - fit) ** 2))
    return coef, res

print("(b,c) exactness of the full and null parameter fields")
wf_full = wn_null = wres = 0.0
for _ in range(6):
    eps = rng.uniform(0.3, 1.0); lam = eps; Jc = eps ** 3
    st = [(rng.normal(), rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)) for _ in range(3)]
    o = rng.choice([-1.0, 1.0], size=3); om = rng.uniform(0, 2 * np.pi, size=3)
    for i in range(3):
        D, q, logq, logG = costs_numeric(i, st, eps)
        beta = np.array([a_pr if j == i else p_pr for j in range(3)]) * np.exp(-D); beta /= beta.sum()
        logl = eps * o[i] * ZZ / 2 + lam * np.cos(om[i] - PH)
        hfull = np.zeros_like(logq)
        for j in range(3):
            if j == i:
                hfull += beta[j] * (logq - logG)
            else:
                Lj = public_odds(*st[j][:3])
                hfull += beta[j] * (logq - (Lj * ZZ / 2 - np.log(np.cosh(Lj / 2))))
        hfull -= logl
        vel = -(hfull - expect(q, hfull))
        coef, res = project(vel, q)
        l, A, C, B = st[i]
        pred = np.concatenate([[-l + sum(beta[j] * public_odds(*st[j][:3]) for j in range(3) if j != i) + eps * o[i]],
                               -A + lam * np.array([np.cos(om[i]), np.sin(om[i])]),
                               -C + beta[i] * Jc * np.array([1.0, 0.0]),
                               -B])
        wf_full = max(wf_full, np.abs(coef - pred).max()); wres = max(wres, res)
        # null: delete nonself costs, keep labels and priors
        b0 = a_pr * np.exp(-D[i]) / (a_pr * np.exp(-D[i]) + 2 * p_pr)
        hnull = b0 * (logq - logG) - logl
        veln = -(hnull - expect(q, hnull))
        coefn, resn = project(veln, q)
        predn = np.concatenate([[-b0 * l + eps * o[i]], -b0 * A + lam * np.array([np.cos(om[i]), np.sin(om[i])]),
                                b0 * (Jc * np.array([1.0, 0.0]) - C), -b0 * B])
        wn_null = max(wn_null, np.abs(coefn - predn).max()); wres = max(wres, resn)
print("  max coefficient error full: %.2e, null: %.2e; max L2(q) residual outside the span: %.2e" % (wf_full, wn_null, wres))
