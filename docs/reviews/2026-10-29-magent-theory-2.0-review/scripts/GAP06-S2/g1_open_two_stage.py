r"""GAP06-S2 focus question 1(a): openness of fus:open-two-stage near D = 0, with quantifiers.

Analytic claim checked here (proof in the report): for a common-offset datum xbar with gaps
g1 <= g3 (reflect otherwise), H = g3, D = g2 + (g1+g3)/2 - 2H > 0, tau_f = (2 g2 + g1 + g3)/(8 lam),
t_f = log(1 + kappa tau_f)/kappa, every initial point within Euclidean distance
    eps_low(xbar) = D exp(-kappa t_f) / 16
has the two-stage history, while the common-offset point obtained by moving the third sorted
agent down by 2D/5 in both sectors (distance 2 sqrt(2) D / 5) has D' = 0 and a four-agent merger.
Hence the largest admissible radius eps*(xbar) satisfies eps_low <= eps* <= (2 sqrt 2/5) D.

Integrator: exact event-driven solver fusion_events.FusionFlow (independent of D11-FUS's prox).
Single-threaded, a few seconds.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_events import FusionFlow

a, c, lam = 1.0, 0.5, 1.0
kappa = a - c


def common_offset(u0, v0):
    u0 = np.asarray(u0, float)
    return np.stack([u0 + v0, u0 - v0], 1)


def two_stage_ok(hist):
    seq = [h[1] for h in hist]
    A, B = (0, 1), (2, 3)
    s0 = ((0,), (1,), (2,), (3,))
    good1 = [s0, ((0, 1), (2,), (3,)), ((0, 1), (2, 3)), ((0, 1, 2, 3),)]
    good2 = [s0, ((0,), (1,), (2, 3)), ((0, 1), (2, 3)), ((0, 1, 2, 3),)]
    good3 = [s0, ((0, 1), (2, 3)), ((0, 1, 2, 3),)]   # simultaneous outer events (symmetric data)
    return seq == good1 or seq == good2 or seq == good3


def gaps_D(u0):
    us = np.sort(u0)
    g1, g2, g3 = np.diff(us)
    H = max(g1, g3)
    D = g2 + (g1 + g3) / 2 - 2 * H
    tauf = (2 * g2 + g1 + g3) / (8 * lam)
    return g1, g2, g3, D, tauf


flow = FusionFlow([1, 1, 1, 1], a, c, lam)

print("Part A: fus:two-stages, u0 = (-3d,-2d,2d,3d), d = 0.7, v0 = 0.3")
d = 0.7
hist, mu, _ = flow.run(common_offset([-3 * d, -2 * d, 2 * d, 3 * d], 0.3), T=6.0)
t1 = np.log(1 + kappa * d / (2 * lam)) / kappa
t2 = np.log(1 + 5 * kappa * d / (4 * lam)) / kappa
for h in hist:
    print("  t = %.15f  %s" % h)
print("  analytic t1 = %.15f, t2 = %.15f" % (t1, t2))
print("  |t1 - event| = %.2e, |t2 - event| = %.2e" % (abs(hist[1][0] - t1), abs(hist[2][0] - t2)))

print("\nPart B: random common-offset data; history vs the D criterion and tau_final")
rng = np.random.default_rng(1)
nok = nbad = 0; maxerr = 0.0
for trial in range(60):
    g = rng.uniform(0.2, 2.0, 3)
    u0 = np.concatenate([[0.0], np.cumsum(g)]) - 1.5
    v0 = rng.uniform(-0.5, 0.5)
    g1, g2, g3, D, tauf = gaps_D(u0)
    hist, mu, _ = flow.run(common_offset(u0, v0), T=8.0)
    ok = two_stage_ok(hist)
    if (D > 0) == ok:
        nok += 1
    else:
        nbad += 1
    if ok:
        tf = np.log(1 + kappa * tauf) / kappa
        maxerr = max(maxerr, abs(hist[-1][0] - tf))
print("  agreement of history with D>0: %d of %d (disagreements %d)" % (nok, nok + nbad, nbad))
print("  max |final time - log(1+kappa tau_final)/kappa| = %.2e" % maxerr)

print("\nPart C: perturbations of size eps of a datum with small D (Euclidean norm in R^8)")
for D_target in (1e-2, 1e-3):
    g1, g3 = 0.8, 1.0
    g2 = D_target - (g1 + g3) / 2 + 2 * g3
    u0 = np.array([0.0, g1, g1 + g2, g1 + g2 + g3]) - (g1 + g2 + g3) / 2
    v0 = 0.25
    _, _, _, D, tauf = gaps_D(u0)
    tf = np.log(1 + kappa * tauf) / kappa
    eps_low = D * np.exp(-kappa * tf) / 16
    eps_up = 2 * np.sqrt(2) * D / 5
    xbar = common_offset(u0, v0)
    hist0, _, _ = flow.run(xbar, T=6.0, dt=5e-4)
    print("  D = %.3e, t_f = %.6f, eps_low = %.3e, eps_up = 2sqrt2 D/5 = %.3e, base history ok: %s"
          % (D, tf, eps_low, eps_up, two_stage_ok(hist0)))
    print("    unperturbed events:", [(round(h[0], 9), h[1]) for h in hist0[1:]])
    # explicit failing witness inside the family: move sorted agent 3 down by 2D/5
    u_w = u0.copy(); u_w[2] -= 2 * D / 5
    xw = common_offset(u_w, v0)
    hw, _, _ = flow.run(xw, T=6.0, dt=5e-4)
    print("    witness at distance %.3e: D' = %.2e, history ok: %s, events %s"
          % (np.linalg.norm(xw - xbar), gaps_D(u_w)[3], two_stage_ok(hw),
             [(round(h[0], 9), h[1]) for h in hw[1:]]))
    for mult in (0.5, 1.0, 4.0, 16.0, 64.0, 'D/4', 'D/2', 'D'):
        if isinstance(mult, str):
            eps = {'D/4': D / 4, 'D/2': D / 2, 'D': D}[mult]
        else:
            eps = mult * eps_low
        nok = 0; ntr = 25
        for trial in range(ntr):
            dirn = rng.normal(size=(4, 2)); dirn /= np.linalg.norm(dirn)
            x0 = xbar + eps * dirn
            h, _, _ = flow.run(x0, T=6.0, dt=1e-3)
            nok += two_stage_ok(h)
        print("    eps = %-6s (%.3e): two-stage history in %d of %d random directions"
              % (str(mult) if isinstance(mult, str) else "%g*low" % mult, eps, nok, ntr))
