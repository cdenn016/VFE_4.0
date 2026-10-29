r"""GAP07-S3, focus question 4, part 1: the D27-ITEM7-04 counterexample inside the Items 2/3 Gaussian tangent joint.

D27's energy  sum_B sum_{i in B} [1/2 (y_i - o_i)^2 + 1/2 (y_i - m_B)^2] + sum_B 4 (m_B - 0)^2  is the supplied hierarchy
it3:hierarchy over it3:energy with no edges (no bridges, no internal records), child self precision r = 1 at m_i = o_i,
cross-scale precision tau = 1 with c_A = 0 (cross-scale dictionaries inert), parent self law N(0, 1/8). D27 integrated it
in the metric diag(20, 20, 20, 20, 6, 6), which it chose. The joint's own global flow (it2:global-flow, ch18:114-118) uses
the Fisher metric at the optimal factor covariances, D = block diagonal of J = diag(2, 2, 2, 2, 10, 10). This script
(a) reproduces D27's result in D27's metric, (b) runs the same handoff states in the joint's own metric, (c) activates
the cross-scale dictionaries (c_A != 0) of it3:hierarchy, child- or parent-owned, and (d) scans this no-edge family in the
joint's own metric, with handoff states reachable by its own pre-handoff flow, for any union dwell >= tau_p.
Flows are solved exactly (eigendecomposition); the selector is evaluated on a grid of step 0.001 (part a-c) or 0.01 (d).
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_joint import two_cluster_joint, LinearFlow, dwell_union, longest_dwell, first_selection

T0 = time.time()
rec = np.array([-0.775, -0.725, 0.725, 0.775])
hand = {"actual (block gap 0.5)": np.array([-0.275, -0.225, 0.225, 0.275]),
        "counterfactual (block gap 0.7)": np.array([-0.375, -0.325, 0.325, 0.375])}
ts = np.arange(0, 400.0, 0.001)


def run_post(par, metric, y_hand, c_child=(0, 0, 0, 0)):
    Jt, y, yA = two_cluster_joint(rec, 1.0, 0.0, 0.0, 1.0, c_child, parents=par)
    J, h = Jt.build()
    D = Jt.metric(J) if metric == "joint" else np.diag(metric)
    x0 = np.zeros(len(Jt.names)); x0[y] = y_hand; x0[yA[0]] = y_hand[:2].mean(); x0[yA[1]] = y_hand[2:].mean()
    F = LinearFlow(J, h, D); X = F.traj(x0, ts)
    g = X[yA[1]] - X[yA[0]]
    return dwell_union(g, ts), np.min(np.abs(g)), longest_dwell(g, ts), F.xs[yA[1]] - F.xs[yA[0]], np.diag(D)


par = dict(p=8.0, a=0.0, tau=1.0, sigA2=1.0, cA=0.0)
for metric in ([20, 20, 20, 20, 6, 6], "joint"):
    for name, yh in hand.items():
        u, gmin, dw, gs, dg = run_post(par, metric, yh)
        print(f"(a/b) metric {'D27 diag(20,20,20,20,6,6)' if metric != 'joint' else 'joint D = blockdiag(J) = diag' + str(tuple(np.round(dg, 3)))}:"
              f" {name}: union selected at {u}; min parent gap {gmin:.4f}; longest coherent dwell {dw:.3f}; attractor gap {gs:.4f}"
              f" (coherence needs < 0.1291)")
# analytic 2x2 reduction in the joint metric: u' = -u + kA v, v' = -v + (r obar + tau u)/(r + tau); undershoot iff alpha < 0
r, tau, p, n, ob = 1.0, 1.0, 8.0, 2, 0.75
kA, km = n * tau / (n * tau + p), tau / (r + tau)
vs = r * ob / ((r + tau) * (1 - kA * km)); us = kA * vs; s = np.sqrt(kA / km)
for v0 in (0.25, 0.35):
    alpha = ((v0 - us) + (v0 - vs) * s) / 2
    print(f"     2x2 reduction, joint metric, v0 = {v0}: u* = {us:.4f}, v* = {vs:.4f}, slow-mode coefficient alpha = {alpha:.4f}"
          f" ({'monotone approach from above, no undershoot' if alpha > 0 else 'undershoot'})")

print("(c) cross-scale dictionaries active (c_A != 0), joint metric:")
for owner in ("child", "parent"):
    for cA in (0.25, 0.75):
        for sigA2 in (0.1, 1.0, 10.0):
            pr = dict(p=8.0, a=0.0, tau=1.0, sigA2=sigA2, cA=cA, owner=owner)
            res = [run_post(pr, "joint", yh) for yh in hand.values()]
            print(f"    owner {owner:6s} c_A {cA:4.2f} sigma_A^2 {sigA2:5.1f}: union (actual, counterfactual) = "
                  f"({res[0][0]}, {res[1][0]}); min gaps ({res[0][1]:.4f}, {res[1][1]:.4f}); attractor gap {res[0][3]:.4f}")

# (d) scan of the no-edge family in the joint metric with reachable handoffs
print("(d) no-edge family, joint metric, handoff states produced by the joint's own pre-handoff flow:")
ts2 = np.arange(0, 60.0, 0.01)
best = (0.0, None); n_tried = n_sel = 0
for ob in (0.3, 0.5, 0.75, 1.0):
    recd = np.array([-ob - 0.025, -ob + 0.025, ob - 0.025, ob + 0.025])
    for f0 in (0.0, 0.3, 0.6, 1.5):
        y00 = f0 * recd + np.array([-0.002, 0.001, -0.001, 0.002])
        # pre-handoff: children only, no edges -> each child relaxes to its record at unit rate in the joint metric
        Jt0, y0i, _ = two_cluster_joint(recd, 1.0, 0.0, 0.0, 1.0, (0, 0, 0, 0))
        J0, h0 = Jt0.build(); F0 = LinearFlow(J0, h0, Jt0.metric(J0)); X0 = F0.traj(y00, ts2)
        th, sel = first_selection(X0[y0i], ts2)
        if th is None or sorted(tuple(sorted(c)) for c in sel) != [(0, 1), (2, 3)]: continue
        yh = X0[y0i, int(round(th / 0.01))]
        for r_ in (0.2, 1.0, 5.0):
            for tau_ in (0.1, 0.5, 1.0, 3.0, 10.0, 30.0):
                for p_ in (0.5, 2.0, 8.0, 30.0, 100.0):
                    n_tried += 1
                    Jt, y, yA = two_cluster_joint(recd, r_, 0.0, 0.0, 1.0, (0, 0, 0, 0), parents=dict(p=p_, a=0.0, tau=tau_, sigA2=1, cA=0.0))
                    J, h = Jt.build(); F = LinearFlow(J, h, Jt.metric(J))
                    x0 = np.zeros(len(Jt.names)); x0[y] = yh; x0[yA[0]] = yh[:2].mean(); x0[yA[1]] = yh[2:].mean()
                    g = (F.traj(x0, ts2)[yA[1]] - F.traj(x0, ts2)[yA[0]])
                    gs = F.xs[yA[1]] - F.xs[yA[0]]
                    if 3 * gs ** 2 < 0.05: continue                      # attractor inside S: union from every handoff
                    n_sel += 1
                    dw = longest_dwell(g, ts2)
                    if dw > best[0]: best = (dw, (ob, f0, r_, tau_, p_, round(th, 2), round(float(gs), 4)))
print(f"    {n_tried} parameter/handoff combinations with a reachable first stage; {n_sel} with the attractor outside S;"
      f" longest transient coherent dwell of the parents {best[0]:.3f} (tau_p = 2) at (obar, f0, r, tau, p, t_handoff, g*) = {best[1]}")
print("elapsed", round(time.time() - T0, 1), "s")
