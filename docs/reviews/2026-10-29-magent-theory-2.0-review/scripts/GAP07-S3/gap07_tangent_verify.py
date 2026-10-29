r"""GAP07-S3, focus question 4, part 3: verification of the best (F6b) instances found by gap07_tangent_search.py in the
Items 2/3 Gaussian tangent joint with its own global flow.

For the most robust instance of the unrestricted search and of the same-side search: positive definiteness of J (strict
convexity), the first stage from the admitted initial state by an independent RK4 integration (step 0.002) with the
general selector on a 0.002 grid, the union from the actual handoff by RK4, agreement of RK4 with the exact solution, and
a sweep of counterfactual handoffs (I shifted by -Delta, J by +Delta, walled at member level) with the exact solution on
a 0.002 grid. Each instance prints its parameters so that it can be rebuilt from gap07_joint.two_cluster_joint.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gap07_joint import two_cluster_joint, LinearFlow, first_selection, dwell_union, longest_dwell

LS, DC, DW, TP = 6.0, 0.05, 0.5, 2.0
T0 = time.time()


def rk4(J, h, D, x0, T, dt):
    Dinv = np.linalg.inv(D); A = -Dinv @ J; b = Dinv @ h
    f = lambda x: A @ x + b
    xs = [x0.copy()]; x = x0.copy()
    for _ in range(int(round(T / dt))):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); xs.append(x.copy())
    return np.array(xs).T


for fname in ("gap07_tangent_search_found.json", "gap07_tangent_search_found_same_side.json"):
    d = json.load(open(os.path.join(HERE, fname), encoding="utf-8"))[0]
    ob, par = d["ob"], d["par"]
    m = np.array([-ob - 0.025, -ob + 0.025, ob - 0.025, ob + 0.025])
    print(f"== {fname}: obar {ob:.5f}, r {d['r']:.5f}, q_in {d['q_in']:.5f}, q_br {d['q_br']:.5f}, sigma^2 {d['sig2']:.5f},"
          f" y(0) {np.round(d['y00'], 6).tolist()}; parents {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in par.items()})}")
    Jt0, yi, _ = two_cluster_joint(m, d["r"], d["q_in"], d["q_br"], d["sig2"], m)
    J0, h0 = Jt0.build(); D0 = Jt0.metric(J0)
    x00 = np.zeros(len(Jt0.names)); x00[yi] = d["y00"]
    dt = 0.002; Tpre = 10.0
    X0 = rk4(J0, h0, D0, x00, Tpre, dt); ts0 = np.arange(X0.shape[1]) * dt
    F0 = LinearFlow(J0, h0, D0); err0 = np.abs(F0.traj(x00, ts0) - X0).max()
    th, sel = first_selection(X0[yi], ts0)
    kh = int(round(th / dt)); xh = X0[:, kh]
    print(f"   pre-handoff: J eigenvalues in [{np.linalg.eigvalsh(J0).min():.4f}, {np.linalg.eigvalsh(J0).max():.2f}];"
          f" first selection (RK4, grid 0.002) at t = {th:.3f}: {sorted(tuple(sorted(c)) for c in sel)};"
          f" |RK4 - exact| = {err0:.1e}; handoff contents {np.round(xh[yi], 5).tolist()}")
    yhh = xh[yi]
    KL = lambda a_, b_: LS / 2 * (a_ - b_) ** 2
    print(f"   at the handoff: internal KL {KL(yhh[0], yhh[1]):.4f}, {KL(yhh[2], yhh[3]):.4f} (< 0.05);"
          f" smallest cross KL {min(KL(yhh[i], yhh[j]) for i in (0, 1) for j in (2, 3)):.4f} (> 0.5)")
    Jt, y, yA = two_cluster_joint(m, d["r"], d["q_in"], d["q_br"], d["sig2"], m, parents=par)
    J, h = Jt.build(); D = Jt.metric(J); F = LinearFlow(J, h, D)
    idx = {nm: k for k, nm in enumerate(Jt.names)}
    def post_state(shift):
        x0 = np.zeros(len(Jt.names))
        for k, nm in enumerate(Jt0.names): x0[idx[nm]] = xh[k]
        x0[y[0]] -= shift; x0[y[1]] -= shift; x0[y[2]] += shift; x0[y[3]] += shift
        x0[yA[0]] = x0[y[:2]].mean(); x0[yA[1]] = x0[y[2:]].mean()
        return x0
    gs = F.xs[yA[1]] - F.xs[yA[0]]
    Tpost = 60.0 if d["dwell_actual"] < 50 else 250.0
    Xp = rk4(J, h, D, post_state(0.0), Tpost, dt); tsp = np.arange(Xp.shape[1]) * dt
    errp = np.abs(F.traj(post_state(0.0), tsp) - Xp).max()
    gp = Xp[yA[1]] - Xp[yA[0]]
    print(f"   post-handoff: J eigenvalues in [{np.linalg.eigvalsh(J).min():.4f}, {np.linalg.eigvalsh(J).max():.2f}]"
          f" (strictly convex); slowest rate of D^-1 J {F.lam.min():.4f}; attractor parent gap {gs:.5f}"
          f" (KL {LS / 2 * gs ** 2:.4f} > 0.05); |RK4 - exact| = {errp:.1e}")
    print(f"   actual handoff (RK4): union selected at {dwell_union(gp, tsp)}; min parent gap {np.abs(gp).min():.5f}"
          f" (radius 0.12910); longest coherent dwell {longest_dwell(gp, tsp):.3f}")
    ts = np.arange(0, 400.0, 0.002)
    rows = []
    for sh in np.round(np.arange(-0.15, 0.51, 0.05), 3):
        x0 = post_state(sh); yy = x0[y]
        walled = all(KL(yy[i], yy[j]) > DW for i in (0, 1) for j in (2, 3))
        g = F.traj(x0, ts)[yA[1]] - F.traj(x0, ts)[yA[0]]
        rows.append((float(sh), walled, dwell_union(g, ts), round(float(np.abs(g).min()), 5)))
    print("   counterfactual sweep (Delta, walled, union time, min parent gap):", rows)
print("elapsed", round(time.time() - T0, 1), "s")
