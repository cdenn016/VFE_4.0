r"""GAP06-S2 focus question 2: finite-time arrival at the final partition for the
heterogeneous-reference fusion flow (2026-10-29 D11-FUS-08 variant of chapters/06_fusion.tex):
    F(mu) = sum_i n_i U(mu_i - x_i) + lam sum_{i<j} n_i n_j (|dq| + |ds|),  W mu' in -dF(mu).
Theorem (report): if 0 lies in the relative interior of dF(mu*) (strict complementarity, margin
theta > 0), the coincidence partition of mu* is reached at a finite time
    T_arr <= T_1 + 2 r/(theta c_0),  r = min(sep*/2, theta c_0/(2 (a+c))),
    T_1 = max(0, log(||mu(0)-mu*||_W / r)/kappa),
and kept afterward; c_0 = min of Lambda_coinc on the W-unit sphere of the normal space.
Without strict complementarity it can fail: explicit two-agent datum with a X^q - c X^s = 2 lam.
Checks: (A) nondegenerate three-agent datum, observed arrival vs the bound; (B) the degenerate datum,
Delta^q(t) = Delta^q(t_s) exp(-a (t - t_s)) > 0 after the s-sector sticks; (C) the opposite side
arrives at a finite time; (D) near-degenerate data, arrival time ~ log(1/delta)/a;
(E) a synchronized pair that splits (membership is not monotone under heterogeneous references).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_events import FusionFlow

a, c = 1.0, 0.5
kappa = a - c

print("(A) nondegenerate three-agent datum")
lam = 0.5
xref = np.array([[0.30, 0.20], [-0.30, -0.10], [3.00, 2.50]])
fl = FusionFlow([1, 1, 1], a, c, lam, xref=xref)
mu0 = np.array([[2.0, -1.0], [-1.5, 1.0], [0.5, 0.0]])
hist, mus, sh = fl.run(mu0, T=70.0, dt=2e-3, record_sector=True)
print("  sync history:", [(round(t, 10), p) for t, p in hist])
print("  sector events:", [(round(e[0], 10),) + e[1:] for e in sh])
# strict-complementarity margin theta at mu*: for every cluster C of every sector and every proper
# subset S, sum_{j in S} y_j <= (1 - theta) lam n_S n_{C minus S}, with y_j = F0_j + ext_C (v_C = 0)
from itertools import combinations
F0 = np.stack([-a * (mus[:, 0] - xref[:, 0]) + c * (mus[:, 1] - xref[:, 1]),
               -a * (mus[:, 1] - xref[:, 1]) + c * (mus[:, 0] - xref[:, 0])], 1)
cl = fl.initial_clusters(np.round(mus, 9))
ratios = []; seps = []
for al in (0, 1):
    vals = [mus[C[0], al] for C in cl[al]]
    seps += list(np.diff(vals))
    for p_, C in enumerate(cl[al]):
        below = sum(len(D) for D in cl[al][:p_]); above = sum(len(D) for D in cl[al][p_ + 1:])
        ext = -lam * (below - above)
        y = {j: F0[j, al] + ext for j in C}
        print("  sector %d cluster %s: stationarity residual %.2e" % (al, C, abs(sum(y.values()))))
        for k in range(1, len(C)):
            for S in combinations(C, k):
                ratios.append(sum(y[j] for j in S) / (lam * len(S) * (len(C) - len(S))))
theta = 1 - max(ratios)
c0 = np.sqrt(2) * lam          # rigorous lower bound for unit multiplicities (see report)
sep = min(seps) if seps else np.inf
r = min(sep / 2, theta * c0 / (2 * (a + c)))
print("  sector partitions at mu*:", cl)
dist0 = np.linalg.norm(mu0 - mus)
T1 = max(0.0, np.log(dist0 / r) / kappa)
Tbound = T1 + 2 * r / (theta * c0)
t_arr = hist[-1][0]
print("  mu* =", np.round(mus, 12).tolist())
print("  theta = %.6f, sep* = %.4f, r = %.6f" % (theta, sep, r))
print("  observed arrival at final partition t = %.6f ; bound T_1 + 2r/(theta c0) = %.4f + %.4f = %.4f"
      % (t_arr, T1, 2 * r / (theta * c0), Tbound))

print("\n(B) degenerate datum: a X^q - c X^s = 2 lam exactly")
lam = 1.0
X = np.array([2.5, 1.0])
print("  eps*_q = (aX^q - cX^s)/(2lam) = %.3f, eps*_s = (aX^s - cX^q)/(2lam) = %.3f"
      % ((a * X[0] - c * X[1]) / (2 * lam), (a * X[1] - c * X[0]) / (2 * lam)))
fl = FusionFlow([1, 1], a, c, lam, xref=np.array([X / 2, -X / 2]))
mu0 = np.array([[1.0, 0.6], [-0.2, 0.1]])
hist, mu_end, sh = fl.run(mu0, T=25.0, dt=2e-3, record_sector=True)
print("  sector events:", [(round(e[0], 12),) + e[1:] for e in sh])
print("  sync history:", hist)
ts = sh[0][0]
# re-run to t_s and to several later times to compare Delta^q with the exponential law
fl2 = FusionFlow([1, 1], a, c, lam, xref=np.array([X / 2, -X / 2]))
_, mu_s, _ = fl2.run(mu0, T=ts + 1e-9, dt=2e-3)
Dq_s = mu_s[0, 0] - mu_s[1, 0]
for T in (ts + 2.0, ts + 8.0, 25.0):
    _, mT, _ = fl2.run(mu0, T=T, dt=2e-3)
    Dq = mT[0, 0] - mT[1, 0]; Ds = mT[0, 1] - mT[1, 1]
    print("  t = %7.4f: Delta^q = %.6e, predicted %.6e, Delta^s = %.1e"
          % (T, Dq, Dq_s * np.exp(-a * (T - ts)), Ds))

print("\n(C) same degenerate references, opposite side Delta^q(0) < 0")
mu0c = np.array([[-0.5, 0.6], [0.4, 0.1]])
hist, _, sh = fl.run(mu0c, T=25.0, dt=2e-3, record_sector=True)
print("  sync history:", [(round(t, 10), p) for t, p in hist])

print("\n(D) near-degenerate references X^q = 2.5 - delta: arrival time")
for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
    Xd = np.array([2.5 - delta, 1.0])
    fld = FusionFlow([1, 1], a, c, lam, xref=np.array([Xd / 2, -Xd / 2]))
    hist, _, sh = fld.run(mu0, T=40.0, dt=2e-3, record_sector=True)
    tarr = hist[-1][0] if len(hist) > 1 else np.inf
    print("  delta = %.0e: arrival t = %.6f, log(1/delta)/a = %.4f" % (delta, tarr, np.log(1 / delta) / a))

print("\n(E) a synchronized pair splits when a|X^q| - ... exceeds the binding")
Xe = np.array([6.0, 0.0])
fle = FusionFlow([1, 1], a, c, lam, xref=np.array([Xe / 2, -Xe / 2]))
hist, mE, sh = fle.run(np.zeros((2, 2)), T=20.0, dt=2e-3, record_sector=True)
print("  sync history:", [(round(t, 10), p) for t, p in hist])
print("  final state:", np.round(mE, 8).tolist())
