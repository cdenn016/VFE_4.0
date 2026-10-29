r"""D17-PA check 1: four-state conditional channel of chapters/13_per_agent_inheritance.tex.

Checks (single-threaded, small):
  (a) channel identities pa:four-channel / pa:four-substitution: row sums, det K[p], pK[p]=pW, D(pK[p])[v]=vW;
  (b) own-block potentials pa:four-potentials and null potentials pa:four-null against finite differences
      of the Gibbs-reduced per-agent free energy (prop:dyn-per-agent with exact unit-temperature rows);
  (c) linearization pa:four-linear at the record-free uniform table: full 12x12 spectrum;
  (d) conditional cone ODE pa:four-cone against the null replicator flow;
  (e) pa:four-grouping: full flow selects the pair on a window, null never does; random open perturbations;
  (f) extension probe: long-time behavior of the full flow.
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(20261029)
V = np.array([-1.0, 1.0])            # index 0 is -1, index 1 is +1
J = 1.5
kappa, alpha = 0.6, 0.3
rho_q = np.cosh(J) / (1.0 + np.cosh(J))
rho_s = 0.5
A_SIGN = V[:, None] * np.ones((2, 2))  # a value at [a_idx, x_idx]

def Wker(kap=kappa):
    W = np.zeros((2, 2, 2))  # [b, x, y]
    for bi, b in enumerate(V):
        for xi, x in enumerate(V):
            for yi, y in enumerate(V):
                W[bi, xi, yi] = (1 + kap * b * x * y) / 2
    return W
W = Wker()

def pW(p):
    return np.einsum("bx,bxy->by", p, W)

def Kmat(p):
    mu = p.sum(axis=0)                 # mu_p(x)
    L = p / mu[None, :]                # L_p(b|x) at [b, x]
    K = np.zeros((2, 2, 2, 2))         # [a, x, b, y]
    for ai in range(2):
        for xi in range(2):
            for bi in range(2):
                for yi in range(2):
                    K[ai, xi, bi, yi] = (alpha * (ai == bi) + (1 - alpha) * L[bi, xi]) * W[bi, xi, yi]
    return K

def randtab():
    t = rng.uniform(0.2, 1.0, size=(2, 2))
    return t / t.sum()

print("(a) channel identities")
worst = 0.0
for _ in range(20):
    p = randtab()
    K = Kmat(p)
    rows = K.reshape(4, 4).sum(axis=1)
    out = np.einsum("ax,axby->by", p, K)
    det = np.linalg.det(K.reshape(4, 4))
    worst = max(worst, np.abs(rows - 1).max(), np.abs(out - pW(p)).max())
    v = rng.normal(size=(2, 2)); v -= v.mean()
    eps = 1e-6
    d = (np.einsum("ax,axby->by", p + eps * v, Kmat(p + eps * v)) - np.einsum("ax,axby->by", p - eps * v, Kmat(p - eps * v))) / (2 * eps)
    worst = max(worst, np.abs(d - pW(v)).max())
print("  max residual of row sums, pK[p]=pW, D(pK[p])[v]=vW :", worst)
print("  det K[p] =", det, " expected -alpha^2 kappa^2 =", -alpha**2 * kappa**2)

def kl(p, r):
    return float(np.sum(p * np.log(p / r)))

def mean_a(p):
    return float(np.sum(A_SIGN * p))

def costs(qi, si, qj, sj):
    mi, ni = mean_a(qi), mean_a(si)
    Dq_ii = float(np.sum(qi * np.log(qi))) - J * mi * ni + np.log(4 * np.cosh(J))
    Ds_ii = float(np.sum(si * np.log(si))) + np.log(4.0)
    Dq_ij = kl(qi, pW(qj))
    Ds_ij = kl(si, pW(sj))
    return Dq_ii, Dq_ij, Ds_ii, Ds_ij

def F_red(qi, si, qj, sj, Hq, Hs, null=False):
    Dq_ii, Dq_ij, Ds_ii, Ds_ij = costs(qi, si, qj, sj)
    if null:
        Dq_ij = 0.0; Ds_ij = 0.0
    Zq = rho_q * np.exp(-Dq_ii) + (1 - rho_q) * np.exp(-Dq_ij)
    Zs = rho_s * np.exp(-Ds_ii) + (1 - rho_s) * np.exp(-Ds_ij)
    return -np.log(Zq) - np.log(Zs) - Hq * mean_a(qi) - Hs * mean_a(si)

def potentials(qi, si, qj, sj, Hq, Hs, null=False):
    Dq_ii, Dq_ij, Ds_ii, Ds_ij = costs(qi, si, qj, sj)
    mi, ni = mean_a(qi), mean_a(si)
    if null:
        b0 = rho_q * np.exp(-Dq_ii) / (rho_q * np.exp(-Dq_ii) + 1 - rho_q)
        g0 = rho_s * np.exp(-Ds_ii) / (rho_s * np.exp(-Ds_ii) + 1 - rho_s)
        hq = b0 * (np.log(qi) - J * ni * A_SIGN) - Hq * A_SIGN
        hs = g0 * np.log(si) - b0 * J * mi * A_SIGN - Hs * A_SIGN
        return hq, hs, b0, g0
    bii = rho_q * np.exp(-Dq_ii) / (rho_q * np.exp(-Dq_ii) + (1 - rho_q) * np.exp(-Dq_ij))
    gii = rho_s * np.exp(-Ds_ii) / (rho_s * np.exp(-Ds_ii) + (1 - rho_s) * np.exp(-Ds_ij))
    hq = np.log(qi) - (1 - bii) * np.log(pW(qj)) - bii * J * ni * A_SIGN - Hq * A_SIGN
    hs = np.log(si) - (1 - gii) * np.log(pW(sj)) - bii * J * mi * A_SIGN - Hs * A_SIGN
    return hq, hs, bii, gii

print("(b) own-block and null potentials vs finite differences of the reduced per-agent free energy")
worst = {False: 0.0, True: 0.0}
for _ in range(20):
    qi, si, qj, sj = randtab(), randtab(), randtab(), randtab()
    Hq, Hs = rng.normal(), rng.normal()
    for null in (False, True):
        hq, hs, _, _ = potentials(qi, si, qj, sj, Hq, Hs, null)
        for _k in range(3):
            v = rng.normal(size=(2, 2)); v -= v.mean()
            e = 1e-6
            dq = (F_red(qi + e * v, si, qj, sj, Hq, Hs, null) - F_red(qi - e * v, si, qj, sj, Hq, Hs, null)) / (2 * e)
            ds = (F_red(qi, si + e * v, qj, sj, Hq, Hs, null) - F_red(qi, si - e * v, qj, sj, Hq, Hs, null)) / (2 * e)
            worst[null] = max(worst[null], abs(dq - np.sum(hq * v)), abs(ds - np.sum(hs * v)))
print("  max directional-derivative residual  full: %.2e   null: %.2e" % (worst[False], worst[True]))

# flow on the four tables q1, s1, q2, s2
def unpack(y):
    t = y.reshape(4, 2, 2)
    return t[0], t[1], t[2], t[3]

def field(y, H, null=False):
    q1, s1, q2, s2 = unpack(y)
    out = []
    for (qi, si, qj, sj, HH) in ((q1, s1, q2, s2, H[0]), (q2, s2, q1, s1, H[1])):
        hq, hs, _, _ = potentials(qi, si, qj, sj, HH[0], HH[1], null)
        out.append(-qi * (hq - np.sum(qi * hq)))
        out.append(-si * (hs - np.sum(si * hs)))
    return np.concatenate([o.ravel() for o in out])

def to_full(z):
    tabs = []
    for k in range(4):
        w = z[3 * k:3 * k + 3]
        tabs.append(np.array([w[0], w[1], w[2], 1 - w.sum()]).reshape(2, 2))
    return np.concatenate([t.ravel() for t in tabs])

def free_field(z, H, null=False):
    f = field(to_full(z), H, null).reshape(4, 4)
    return f[:, :3].ravel()

def jac(z, H, null=False):
    Jm = np.zeros((12, 12))
    for k in range(12):
        e = np.zeros(12); e[k] = 1e-6
        Jm[:, k] = (free_field(z + e, H, null) - free_field(z - e, H, null)) / 2e-6
    return Jm

z0 = np.full(12, 0.25)
H0 = ((0.0, 0.0), (0.0, 0.0))
ev = np.sort(np.linalg.eigvals(jac(z0, H0)).real)
expected = np.sort([0.25, -0.75, -1.25, -2.25] + [-1 + kappa / 2] * 4 + [-1 - kappa / 2] * 4)
print("(c) full linearization spectrum at uniform:", np.round(ev, 6))
print("    expected                               :", np.round(expected, 6), " max residual %.1e" % np.abs(ev - expected).max())
print("    null linearization spectrum           :", np.round(np.sort(np.linalg.eigvals(jac(z0, H0, True)).real), 6))

def cond_means(p):
    return (p[1, :] - p[0, :]) / p.sum(axis=0)    # E[a | x] for x = -1, +1

print("(d) conditional cone ODE pa:four-cone vs null replicator")
worst = 0.0
for _ in range(10):
    y = np.concatenate([randtab().ravel() for _ in range(4)])
    H = ((rng.normal(), rng.normal()), (rng.normal(), rng.normal()))
    f = field(y, H, True)
    q1, s1, q2, s2 = unpack(y)
    dq1, ds1, _, _ = unpack(f)
    e = 1e-7
    cq_dot_num = (cond_means(q1 + e * dq1) - cond_means(q1 - e * dq1)) / (2 * e)
    cs_dot_num = (cond_means(s1 + e * ds1) - cond_means(s1 - e * ds1)) / (2 * e)
    hq, hs, b0, g0 = potentials(q1, s1, q2, s2, H[0][0], H[0][1], True)
    m1, n1 = mean_a(q1), mean_a(s1)
    cq, cs = cond_means(q1), cond_means(s1)
    cq_dot = (1 - cq**2) * (b0 * (J * n1 - np.arctanh(cq)) + H[0][0])
    cs_dot = (1 - cs**2) * (b0 * J * m1 - g0 * np.arctanh(cs) + H[0][1])
    worst = max(worst, np.abs(cq_dot - cq_dot_num).max(), np.abs(cs_dot - cs_dot_num).max())
print("  max residual %.2e" % worst)

# (e) selection: full versus null
def initial(eta, pert=0.0):
    m = eta * np.array([3.0, -1.0])
    tabs = []
    for i in range(2):
        for _sec in range(2):
            t = (1 + m[i] * A_SIGN) / 4.0
            if pert > 0:
                t = t * np.exp(pert * rng.normal(size=(2, 2)))
                t /= t.sum()
            tabs.append(t)
    return np.concatenate([t.ravel() for t in tabs])

def all_cond(y):
    q1, s1, q2, s2 = unpack(y)
    return np.concatenate([cond_means(q1), cond_means(s1), cond_means(q2), cond_means(s2)])

def selected(y, thr):
    c = all_cond(y)
    return bool(np.all(c > thr) or np.all(c < -thr))

def run(y0, H, T, null, n=4001):
    ts = np.linspace(0, T, n)
    sol = solve_ivp(lambda t, y: field(y, H, null), (0, T), y0, t_eval=ts, rtol=1e-10, atol=1e-13, method="LSODA")
    return sol.t, sol.y.T

eta, h = 0.02, 2e-4
thr = eta / (8 * np.sqrt(2))
H = ((h, h), (-h, -h))
y0 = initial(eta)
t, Y = run(y0, H, 8.0, False)
sel = np.array([selected(y, thr) for y in Y])
k = np.argmin(np.abs(t - np.log(4)))
print("(e) eta=%g, h=%g, threshold=%.5f" % (eta, h, thr))
print("  conditional means at t*=log4 in units of eta:", np.round(all_cond(Y[k]) / eta, 5))
print("  linear prediction: agent1 %.4f, agent2 %.4f" % (3 / np.sqrt(2), 1 / np.sqrt(2)))
if sel.any():
    print("  full flow: first selected at t=%.4f; selected at t=8: %s" % (t[sel.argmax()], sel[-1]))
t0, Y0 = run(y0, H, 200.0, True, n=20001)
sel0 = np.array([selected(y, thr) for y in Y0])
c0 = np.array([all_cond(y) for y in Y0])
print("  null flow on [0,200]: ever selected=%s; min agent-1 cond=%.3e; max agent-2 cond=%.3e" % (sel0.any(), c0[:, :4].min(), c0[:, 4:].max()))
print("  null final conditional means:", np.round(c0[-1], 4))

nfull = nnull = 0
NS = 30
for _ in range(NS):
    y0p = initial(eta, pert=0.05 * eta)
    t1, Y1 = run(y0p, H, 6.0, False, n=1201)
    s1 = np.array([selected(y, thr) for y in Y1])
    nfull += int(s1.any() and not s1[0])
    t2, Y2 = run(y0p, H, 60.0, True, n=3001)
    nnull += int(any(selected(y, thr) for y in Y2))
print("  random log-table perturbations (size 5 percent of eta): full selected %d/%d, null selected %d/%d" % (nfull, NS, nnull, NS))

# (f) long-time behavior of the full flow
t, Y = run(initial(eta), H, 120.0, False, n=12001)
s = np.array([selected(y, thr) for y in Y])
c = np.array([all_cond(y) for y in Y])
i_on = s.argmax()
print("(f) full flow on [0,120]: selected from t=%.3f; selected at every later sample: %s" % (t[i_on], bool(s[i_on:].all())))
print("    final conditional means:", np.round(c[-1], 5))
q1, s1, q2, s2 = unpack(Y[-1])
print("    final m1,n1,m2,n2 =", np.round([mean_a(q1), mean_a(s1), mean_a(q2), mean_a(s2)], 6))
print("    max |field| at t=120: %.2e" % np.abs(field(Y[-1], H, False)).max())
zf = np.concatenate([Y[-1].reshape(4, 4)[kk, :3] for kk in range(4)])
print("    max real eigenvalue of the Jacobian at the final state: %.4f" % np.linalg.eigvals(jac(zf, H)).real.max())
