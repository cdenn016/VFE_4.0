r"""D17-PA check 5: geometry at eps=0 and the context-varying phase test.

  (a) identities: x^2 E sin^2 = I(x) = x I1/I0, I'(x) = I1/I0 + x V(x), h1'' = V;
  (b) pa:nonflat-limit: connection, curvature and minimized tensor of the three-agent eps=0 section by quadrature
      of the actual scores (finite differences in the contextual coordinates), against the closed forms;
  (c) a section with context-varying erased phase theta_2(c): fine and parent connections and tensors.
"""
import numpy as np
from scipy.special import i0, i1

NG = 256
ph = 2 * np.pi * np.arange(NG) / NG

def vm_mean(f, kappa, mu=0.0):
    w = np.exp(kappa * np.cos(ph - mu)); w /= w.mean()
    return np.mean(w * f)

def I_(x):
    return x * i1(x) / i0(x)

def V_(x):
    m1 = i1(x) / i0(x)
    return vm_mean(np.cos(ph) ** 2, x) - m1 ** 2

print("(a) Bessel identities")
worst = 0.0
for x in (0.01, 0.1, 0.5, 1.0, 3.0):
    lhs = x ** 2 * vm_mean(np.sin(ph) ** 2, x)
    d = 1e-5
    Ip = (I_(x + d) - I_(x - d)) / (2 * d)
    h1pp = (np.log(i0(x + d)) - 2 * np.log(i0(x)) + np.log(i0(x - d))) / d ** 2
    worst = max(worst, abs(lhs - I_(x)), abs(Ip - (i1(x) / i0(x) + x * V_(x))), abs(h1pp - V_(x)) / max(V_(x), 1e-12) * 1e-3)
print("  max residual %.2e" % worst)

# (b) eps=0 section: agent 1 von Mises vector s e(chi); agents 2,3 fixed vectors; product law
def scores_agent(Avec_fun, c, dc=1e-5):
    # returns contextual scores (da, dchi) and orbit score of a von Mises law exp(A(c).e(phi)) on the grid
    A = Avec_fun(c)
    logq = A[0] * np.cos(ph) + A[1] * np.sin(ph)
    logq -= np.log(np.mean(np.exp(logq)))
    q = np.exp(logq)
    sc = []
    for k in range(2):
        e = np.zeros(2); e[k] = dc
        Ap, Am = Avec_fun(c + e), Avec_fun(c - e)
        lp = Ap[0] * np.cos(ph) + Ap[1] * np.sin(ph); lp -= np.log(np.mean(np.exp(lp)))
        lm = Am[0] * np.cos(ph) + Am[1] * np.sin(ph); lm -= np.log(np.mean(np.exp(lm)))
        sc.append((lp - lm) / (2 * dc))
    # orbit score of the pushforward phi -> phi + g: d/dg log q(phi - g)
    orb = -(A[0] * (-np.sin(ph)) + A[1] * np.cos(ph))
    return q, sc, orb

def cost_data(c, t, d=0.03, d3=0.025, al3=1.0):
    agents = [lambda cc: np.exp(-t) * cc[0] * np.array([np.cos(cc[1]), np.sin(cc[1])]),
              lambda cc: np.exp(-t) * d * np.array([1.0, 0.0]),
              lambda cc: np.exp(-t) * d3 * np.array([np.cos(al3), np.sin(al3)])]
    K = 0.0; j = np.zeros(2); Bm = np.zeros((2, 2))
    for f in agents:
        q, sc, orb = scores_agent(f, c)
        K += np.mean(q * orb * orb)
        for x in range(2):
            j[x] += np.mean(q * sc[x] * orb)
            for y in range(2):
                Bm[x, y] += np.mean(q * sc[x] * sc[y])
    return K, j, Bm

def connection(c, t):
    K, j, Bm = cost_data(c, t)
    return -j / K, Bm - np.outer(j, j) / K

print("(b) eps=0 nonflat limit: numeric vs closed form")
d, d3 = 0.03, 0.025
for (a, chi, t) in ((0.04, 0.3, 0.0), (0.04, 0.3, 1.0), (0.02, -0.5, 3.0)):
    c = np.array([a, chi])
    Acon, h = connection(c, t)
    dd = 1e-4
    Achi_p = connection(c + np.array([dd, 0]), t)[0][1]; Achi_m = connection(c - np.array([dd, 0]), t)[0][1]
    Aa_p = connection(c + np.array([0, dd]), t)[0][0]; Aa_m = connection(c - np.array([0, dd]), t)[0][0]
    F = (Achi_p - Achi_m) / (2 * dd) - (Aa_p - Aa_m) / (2 * dd)
    s = np.exp(-t) * a
    Bt = I_(np.exp(-t) * d) + I_(np.exp(-t) * d3)
    Ach = -I_(s) / (I_(s) + Bt)
    Ipr = i1(s) / i0(s) + s * V_(s)
    Fcl = -Bt * np.exp(-t) * Ipr / (I_(s) + Bt) ** 2
    print("  a=%.3f chi=%.2f t=%.1f  A=(%.3e, %.6f) closed (0, %.6f) | F=%.6f closed %.6f | h_aa=%.4e closed %.4e | h_cc=%.4e closed %.4e | h_ac=%.1e"
          % (a, chi, t, Acon[0], Acon[1], Ach, F, Fcl, h[0, 0], np.exp(-2 * t) * V_(s), h[1, 1], I_(s) * Bt / (I_(s) + Bt), h[0, 1]))
    if t == 3.0:
        # infinite-time behaviour at eps=0: connection and curvature limits
        D2 = d ** 2 + d3 ** 2
        print("    eps=0, t to infinity: A_chi -> %.6f (=-a^2/(a^2+D)), F -> %.4f (=-2aD/(a^2+D)^2), tensor -> 0" % (-a**2 / (a**2 + D2), -2 * a * D2 / (a**2 + D2) ** 2))

# (c) context-varying hidden phase on agent 2: fine (with theta_2(c)) vs parent (theta_2 = 0)
print("(c) context-varying erased phase theta_2(c) = 0.7 a + 0.4 chi, hidden concentration b=0.05")
PH, PS = np.meshgrid(ph[::4], ph[::4], indexing="ij")
bconc = 0.05
def agent2_fine_cost(c, t, xi, X, keep_theta=True, dc=1e-5):
    # quadratic cost E (s_X + xi s_orb)^2 of agent 2's (phi, psi) law; X is a contextual tangent (2-vector)
    def logq(cc):
        th = (0.7 * cc[0] + 0.4 * cc[1]) if keep_theta else 0.0
        A = np.exp(-t) * d * np.array([1.0, 0.0])
        lq = A[0] * np.cos(PH) + A[1] * np.sin(PH) + bconc * np.cos(PS - th)
        return lq - np.log(np.mean(np.exp(lq)))
    lq = logq(c); q = np.exp(lq)
    sX = (logq(c + dc * X) - logq(c - dc * X)) / (2 * dc)
    A = np.exp(-t) * d * np.array([1.0, 0.0])
    orb = -(A[0] * (-np.sin(PH)) + A[1] * np.cos(PH))
    return np.mean(q * (sX + xi * orb) ** 2)

def total_cost(c, t, X, xi, keep_theta):
    # agents 1 and 3 from the 1D quadrature, agent 2 from the 2D grid
    tot = 0.0
    for f in (lambda cc: np.exp(-t) * cc[0] * np.array([np.cos(cc[1]), np.sin(cc[1])]),
              lambda cc: np.exp(-t) * d3 * np.array([np.cos(1.0), np.sin(1.0)])):
        q, sc, orb = scores_agent(f, c)
        sX = sc[0] * X[0] + sc[1] * X[1]
        tot += np.mean(q * (sX + xi * orb) ** 2)
    return tot + agent2_fine_cost(c, t, xi, X, keep_theta)

c = np.array([0.04, 0.3]); t = 1.0
for X in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
    out = []
    for keep in (True, False):
        # cost is quadratic in xi: fit c0 + c1 xi + c2 xi^2
        vals = [total_cost(c, t, X, xi, keep) for xi in (-1.0, 0.0, 1.0)]
        c2 = (vals[0] + vals[2] - 2 * vals[1]) / 2; c1 = (vals[2] - vals[0]) / 2; c0 = vals[1]
        xi_star = -c1 / (2 * c2); hmin = c0 - c1 ** 2 / (4 * c2)
        out.append((xi_star, hmin))
    dth = 0.7 * X[0] + 0.4 * X[1]
    print("  X=%s: minimizing xi fine %.8f parent %.8f | h_min fine %.6e parent %.6e | difference %.6e vs I(b) dtheta^2 = %.6e"
          % (X, out[0][0], out[1][0], out[0][1], out[1][1], out[0][1] - out[1][1], I_(bconc) * dth ** 2))
