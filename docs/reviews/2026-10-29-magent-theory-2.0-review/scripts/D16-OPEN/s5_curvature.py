r"""D16-OPEN s5: recompute the declared-family collective geometry (focus question 5).

Contexts c = (kappa, chi).  At time t (rho = e^{-2t}) each of n agents in a block has belief
angular factor p_{rho kappa}(theta - chi) and model factor p_{rho lambda}(psi); binary
coordinates are context independent (they contribute no contextual score).  The common
collective group S^1 rotates every belief and model angle (tied representation); J is its
infinitesimal action, D^A p = dp + J A.

Direct computation (no closed form used): Fisher inner products of scores on an angular grid,
contextual derivatives by central differences in (kappa, chi), minimizing connection
A(X) = -g(dpX, J)/g(J,J), tensor h(X,Y) = g(dpX + J A(X), dpY + J A(Y)),
curvature F = d_kappa A_chi - d_chi A_kappa by central differences.
Compared with open:curvature, open:rank, open:geometry-limits.
Also: the untied group S^1_q x S^1_s (flat, rank one), and the parent-law cost equality
open:cost-equality for a 2-agent block with distinct factors and a constant H_I rotation.
"""
import numpy as np
from scipy.special import ive

M = 2048
th = 2 * np.pi * np.arange(M) / M
dth = th[1] - th[0]
A1 = lambda v: ive(1, v) / ive(0, v)
Icl = lambda v: v * A1(v)
Ipr = lambda v: v * (1 - A1(v) ** 2)          # I'(v) = v(1 - A^2)


def logp(k, x):
    return k * np.cos(th - x) - np.log(np.mean(np.exp(k * np.cos(th - x))))


def scores(kappa, chi, lam, rho, n, eps=1e-5):
    """per-agent scores (belief, model) for d_kappa, d_chi and J; all n agents identical."""
    lq = lambda k, x: logp(rho * k, x)
    q = np.exp(lq(kappa, chi))
    s = np.exp(logp(rho * lam, 0.0))
    sk = (lq(kappa + eps, chi) - lq(kappa - eps, chi)) / (2 * eps)
    sc = (lq(kappa, chi + eps) - lq(kappa, chi - eps)) / (2 * eps)
    # J: common rotation theta -> theta - xi of both factors: score = -d_theta log
    Jq = -(lq(kappa, chi + eps) - lq(kappa, chi - eps)) / (2 * eps) * (-1)  # d/dxi log q(theta - xi) = -d_theta log q
    Jq = -np.gradient(lq(kappa, chi), dth)
    Js = -np.gradient(logp(rho * lam, 0.0), dth)
    return q, s, sk, sc, Jq, Js


def ip(p, a, b):
    return np.mean(p * a * b)


def geometry(kappa, chi, lam, t, n, tied=True):
    rho = np.exp(-2 * t)
    q, s, sk, sc, Jq, Js = scores(kappa, chi, lam, rho, n)
    # block cost: n copies, belief + model
    gJJ = n * (ip(q, Jq, Jq) + ip(s, Js, Js))
    gXJ = {"k": n * ip(q, sk, Jq), "c": n * ip(q, sc, Jq)}
    gXY = {("k", "k"): n * ip(q, sk, sk), ("c", "c"): n * ip(q, sc, sc), ("k", "c"): n * ip(q, sk, sc)}
    if tied:
        A = {X: -gXJ[X] / gJJ for X in ("k", "c")}
        h = {}
        for X, Y in (("k", "k"), ("c", "c"), ("k", "c")):
            h[(X, Y)] = gXY[(X, Y)] + A[X] * gXJ[Y] + A[Y] * gXJ[X] + A[X] * A[Y] * gJJ
        return A, h
    else:
        # untied: separate generators for belief and model rotations
        gqq = n * ip(q, Jq, Jq)
        Aq = {X: -gXJ[X] / gqq for X in ("k", "c")}
        h = {}
        for X, Y in (("k", "k"), ("c", "c"), ("k", "c")):
            h[(X, Y)] = gXY[(X, Y)] + Aq[X] * gXJ[Y] + Aq[Y] * gXJ[X] + Aq[X] * Aq[Y] * gqq
        return Aq, h


def curvature(kappa, chi, lam, t, n, tied=True, e=1e-4):
    Ac_p = geometry(kappa + e, chi, lam, t, n, tied)[0]["c"]
    Ac_m = geometry(kappa - e, chi, lam, t, n, tied)[0]["c"]
    Ak_p = geometry(kappa, chi + e, lam, t, n, tied)[0]["k"]
    Ak_m = geometry(kappa, chi - e, lam, t, n, tied)[0]["k"]
    return (Ac_p - Ac_m) / (2 * e) - (Ak_p - Ak_m) / (2 * e)


out = []
kap, chi, lam = 1.5, 0.4, 1.0
for n in (1, 2, 4):
    for t in (0.0, 1.0, 2.0, 3.0):
        rho = np.exp(-2 * t)
        a, b = Icl(rho * kap), Icl(rho * lam)
        A, h = geometry(kap, chi, lam, t, n)
        Fnum = curvature(kap, chi, lam, t, n)
        Fcl = -b * rho * Ipr(rho * kap) / (a + b) ** 2
        f = np.exp(logp(rho * kap, 0.0)); varcos = np.mean(f * np.cos(th) ** 2) - np.mean(f * np.cos(th)) ** 2
        out.append("n=%d t=%.0f: A_chi num %.8f cl %.8f | F num %.8f cl %.8f | h_cc num %.6e cl %.6e | "
                   "h_kk num %.6e cl %.6e | h_kc %.1e | A_kappa %.1e"
                   % (n, t, A["c"], -a / (a + b), Fnum, Fcl, h[("c", "c")], n * a * b / (a + b),
                      h[("k", "k")], n * rho ** 2 * varcos, h[("k", "c")], A["k"]))
# limits
Finf = -2 * lam ** 2 * kap / (kap ** 2 + lam ** 2) ** 2
for t in (4.0, 6.0, 8.0):
    rho = np.exp(-2 * t)
    a, b = Icl(rho * kap), Icl(rho * lam)
    Fcl = -b * rho * Ipr(rho * kap) / (a + b) ** 2
    out.append("t=%.0f: F (closed form) %.10f vs F_inf %.10f; e^{4t} h_cc/n %.8f vs %.8f; e^{4t} h_kk/n (num) %.6f vs 0.5"
               % (t, Fcl, Finf, np.exp(4 * t) * a * b / (a + b), kap ** 2 * lam ** 2 / (2 * (kap ** 2 + lam ** 2)),
                  np.exp(4 * t) * geometry(kap, chi, lam, t, 1)[1][("k", "k")]))
# uniform lower bound of |F| on [0, inf) on a compact kappa patch
ks = np.linspace(0.2, 5, 97); ts = np.linspace(0, 12, 601)
KK, TT = np.meshgrid(ks, ts, indexing="ij")
R = np.exp(-2 * TT)
FF = -Icl(R * lam) * R * Ipr(R * KK) / (Icl(R * KK) + Icl(R * lam)) ** 2
out.append("min |F| over kappa in [0.2,5], t in [0,12] (lambda=1): %.4e; at t=12 vs limit: max rel dev %.2e"
           % (np.abs(FF).min(), np.max(np.abs(FF[:, -1] / (-2 * lam ** 2 * ks / (ks ** 2 + lam ** 2) ** 2) - 1))))
# untied group: flat connection, rank one
A, h = geometry(kap, chi, lam, 1.0, 2, tied=False)
out.append("untied (S^1_q x S^1_s): A_q(chi) = %.8f (expected -1), F = %.2e, h_cc = %.2e, h_kk = %.4e (rank one)"
           % (A["c"], curvature(kap, chi, lam, 1.0, 2, tied=False), h[("c", "c")], h[("k", "k")]))

# open:cost-equality for a two-agent product parent with distinct factors and constant H_I rotation
rng = np.random.default_rng(3)
Mq = 256
tq = 2 * np.pi * np.arange(Mq) / Mq
def prof(c):
    l = c[0] * np.cos(tq) + c[1] * np.sin(tq) + c[2] * np.cos(2 * tq)
    f = np.exp(l); return f / f.mean()
c1 = rng.normal(0, 1, 3); c2 = rng.normal(0, 1, 3); d1 = rng.normal(0, 1, 3); d2 = rng.normal(0, 1, 3)
def sec(eps):   # a one-parameter contextual curve of child laws
    return prof(c1 + eps * d1), prof(c2 + eps * d2)
e = 1e-5
f1, f2 = sec(0.0)
s1 = (np.log(sec(e)[0]) - np.log(sec(-e)[0])) / (2 * e)
s2 = (np.log(sec(e)[1]) - np.log(sec(-e)[1])) / (2 * e)
kk = np.fft.fftfreq(Mq, 1.0 / Mq)
Dsp = lambda f, ax=-1: np.real(np.fft.ifft(np.fft.fft(f, axis=ax) * (1j * kk).reshape([-1 if i == (ax % f.ndim) else 1 for i in range(f.ndim)]), axis=ax))
J1 = -Dsp(np.log(f1)); J2 = -Dsp(np.log(f2))
xi = 0.37
fine = np.mean(f1 * (s1 + xi * J1) ** 2) + np.mean(f2 * (s2 + xi * J2) ** 2)
# parent: product law on the torus grid after constant rotation (a, -a) applied to every context
a = 0.81
k = np.fft.fftfreq(Mq, 1.0 / Mq)
R_ = lambda f, a: np.real(np.fft.ifft(np.fft.fft(f) * np.exp(-1j * k * a)))
def parent_log(eps):
    g1, g2 = sec(eps)
    return np.log(R_(g1, a))[:, None] + np.log(R_(g2, -a))[None, :]
P = np.exp(parent_log(0.0))
sP = (parent_log(e) - parent_log(-e)) / (2 * e)
JP = -(Dsp(parent_log(0.0), 0) + Dsp(parent_log(0.0), 1))
par = np.mean(P * (sP + xi * JP) ** 2)
out.append("open:cost-equality: fine sum %.10f, parent-law Fisher cost %.10f, rel diff %.2e" % (fine, par, abs(par / fine - 1)))
print("\n".join(out))
