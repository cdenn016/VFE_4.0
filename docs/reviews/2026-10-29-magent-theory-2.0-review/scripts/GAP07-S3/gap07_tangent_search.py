r"""GAP07-S3, focus question 4, part 2: random search in the Items 2/3 Gaussian tangent joint, in its own global flow,
for an instance meeting (F6b) under the finite-dwell declared selector, with the first-stage handoff produced by the
joint's own pre-handoff flow from an admitted initial state.

Family (mirror symmetric, K = 1): children a1, a2 | b1, b2 with self laws N(m_i, 1/r) at m = (-ob-0.025, -ob+0.025,
ob-0.025, ob+0.025); internal edges a1 <-> a2, b1 <-> b2 and bridges a2 <-> b1 with registered agreement o_e = 0, receiver-
owned dictionaries (anchor 1/sigma^2, operating contents c_i = m_i); after the handoff, supplied parents (it3:hierarchy)
with self laws N(-+a, 1/p), cross-scale factors tau with dictionaries (anchor 1/sigma_A^2, operating vector -+c_A), owned
by the child or by the parent. Pre-handoff flow and post-handoff flow: x' = -D^{-1}(J x - h), D = blockdiag(J) (exact).
Declared selector it7:selector: presented precision 6, delta_c = 0.05, delta_w = 0.5, tau_p = 2, grid step 0.02.
Handoff rule: parents start at the pools of their members, new cross-scale dictionaries at 0; children keep their state.
Success: the first selection is exactly {I, J}; the union is selected from the actual handoff; some counterfactual handoff
(I shifted by -Delta, J by +Delta, still walled at member level) never selects it; the attractor gap has KL > delta_c.
"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import json
import numpy as np
from gap07_joint import two_cluster_joint, LinearFlow

LS, DC, DW, TP, DT = 6.0, 0.05, 0.5, 2.0, 0.02
GC = np.sqrt(2 * DC / LS)                                   # coherence radius 0.1291 in mean gap
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
SUBSETS = [c for k in (2, 3, 4) for c in __import__("itertools").combinations(range(4), k)]


def connected(C, coh):
    r"""Boolean array over the grid: the coherence graph restricted to C is connected (|C| <= 4)."""
    C = list(C)
    if len(C) == 2: return coh[(C[0], C[1])]
    # connectivity by brute force over spanning structures: union-find on edge patterns, vectorized via reachability
    reach = {i: np.zeros_like(coh[(0, 1)]) for i in C}
    reach[C[0]] = np.ones_like(coh[(0, 1)])
    for _ in range(len(C)):
        for i in C:
            for j in C:
                if i < j:
                    e = coh[(i, j)]
                    ri, rj = reach[i], reach[j]
                    reach[i] = ri | (rj & e); reach[j] = rj | (ri & e)
    out = np.ones_like(coh[(0, 1)])
    for i in C: out &= reach[i]
    return out


def first_run(b, ts):
    idx = np.arange(len(b)); lf = np.maximum.accumulate(np.where(~b, idx, -1)); st = lf + 1
    ok = b & (ts - ts[np.minimum(st, len(ts) - 1)] >= TP - 1e-9)
    k = np.argmax(ok)
    return (ts[k], k) if ok[k] else (None, None)


def longest(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best * DT


def first_stage(Y, ts):
    KL = {pq: LS / 2 * (Y[pq[0]] - Y[pq[1]]) ** 2 for pq in PAIRS}
    coh = {pq: KL[pq] < DC for pq in PAIRS}
    times = {}
    for C in SUBSETS:
        out = [j for j in range(4) if j not in C]
        b = connected(C, coh)
        for i in C:
            for j in out:
                b = b & (KL[(min(i, j), max(i, j))] > DW)
        t, k = first_run(b, ts)
        if t is not None: times[C] = (t, k)
    if not times: return None
    tmin = min(t for t, k in times.values())
    first = sorted(C for C, (t, k) in times.items() if t <= tmin + 1e-9)
    return tmin, times[first[0]][1], first


MODE = sys.argv[1] if len(sys.argv) > 1 else "free"
SAME_SIDE = MODE == "same_side"; OWN = MODE == "own_anchor"
rng = np.random.default_rng({"free": 29102026, "same_side": 29102027, "own_anchor": 29102028}[MODE])
T0 = time.time(); trials = stage1 = unions = inside = 0; found = []
ts_pre = np.arange(0, 40.0, DT)
while time.time() - T0 < 24.0:
    trials += 1
    ob = rng.uniform(0.3, 1.2); m = np.array([-ob - 0.025, -ob + 0.025, ob - 0.025, ob + 0.025])
    r = 10 ** rng.uniform(-1, 1)
    q_in = 0.0 if rng.random() < 0.2 else 10 ** rng.uniform(-1, 1.5)
    q_br = (10 ** rng.uniform(-2, 1.5)) if OWN else (0.0 if rng.random() < 0.2 else 10 ** rng.uniform(-2, 1))
    sig2 = 10 ** rng.uniform(-2, 2)
    par = dict(p=10 ** rng.uniform(-1, 2), a=ob * (1.0 if OWN else (rng.uniform(0.0, 1.2) if SAME_SIDE else rng.uniform(-0.2, 1.2))), tau=10 ** rng.uniform(-1.3, 1.7),
               sigA2=10 ** rng.uniform(-2, 2), cA=ob * rng.uniform(0.0, 1.5), owner="parent" if rng.random() < 0.5 else "child")
    f0 = rng.uniform(0.0, 1.5)
    y00 = f0 * m + rng.uniform(-0.003, 0.003, 4) * np.array([1, 1, -1, -1])
    y00 = np.array([y00[0], y00[1], -y00[1], -y00[0]])                                 # mirror symmetric
    Jt0, yi, _ = two_cluster_joint(m, r, q_in, q_br, sig2, m)
    J0, h0 = Jt0.build(); F0 = LinearFlow(J0, h0, Jt0.metric(J0))
    x00 = np.zeros(len(Jt0.names)); x00[yi] = y00
    X0 = F0.traj(x00, ts_pre)
    fs = first_stage(X0[yi], ts_pre)
    if fs is None or fs[2] != [(0, 1), (2, 3)]: continue
    stage1 += 1
    th, kh, _ = fs; xh = X0[:, kh]
    Jt, y, yA = two_cluster_joint(m, r, q_in, q_br, sig2, m, parents=par)
    J, h = Jt.build(); F = LinearFlow(J, h, Jt.metric(J))
    gs = F.xs[yA[1]] - F.xs[yA[0]]
    if LS / 2 * gs ** 2 <= DC:
        inside += 1; continue                                   # it7:attractor: union from every handoff
    Tpost = min(300.0, max(40.0, 15.0 / F.lam.min())); ts = np.arange(0, Tpost, DT)
    idx = {nm: k for k, nm in enumerate(Jt.names)}
    def post_state(shift):
        x0 = np.zeros(len(Jt.names))
        for k, nm in enumerate(Jt0.names): x0[idx[nm]] = xh[k]
        x0[y[0]] -= shift; x0[y[1]] -= shift; x0[y[2]] += shift; x0[y[3]] += shift
        x0[yA[0]] = x0[y[:2]].mean(); x0[yA[1]] = x0[y[2:]].mean()
        return x0
    def union(shift):
        X = F.traj(post_state(shift), ts); g = X[yA[1]] - X[yA[0]]
        b = LS / 2 * g ** 2 < DC
        t, _ = first_run(b, ts)
        return t, longest(b), np.abs(g).min()
    tu, dwa, gmin_a = union(0.0)
    if tu is None: continue
    unions += 1
    for sh in (-0.1, -0.05, 0.05, 0.1, 0.2, 0.3):
        x0 = post_state(sh); yy = x0[y]
        walled = all(LS / 2 * (yy[i] - yy[j]) ** 2 > DW for i in (0, 1) for j in (2, 3))
        if not walled: continue
        tc, dwc, gmin_c = union(sh)
        if tc is None:
            rob = min(dwa - TP, TP - dwc, (gmin_c - GC) if dwc == 0 else 0.0, gs - GC)
            found.append(dict(ob=ob, r=r, q_in=q_in, q_br=q_br, sig2=sig2, f0=f0, y00=y00.tolist(), par=par,
                              t_first=float(th), t_union=float(tu), dwell_actual=float(dwa), shift=sh, dwell_cf=float(dwc),
                              gmin_cf=float(gmin_c), g_star=float(gs), robustness=float(rob)))
            break
print(f"mode {MODE}: anchors a >= 0 only: {SAME_SIDE}; anchors at the own cluster record means with bridges always present: {OWN}")
print(f"trials {trials}; reachable first stage {{I, J}} {stage1}; post-handoff attractor inside S (union from every handoff): {inside}; union from the actual handoff with attractor outside S: {unions};"
      f" (F6b) instances found: {len(found)}; elapsed {time.time() - T0:.1f} s")
found.sort(key=lambda d: -d["robustness"])
for d in found[:5]:
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items() if k not in ("y00", "par")}),
          json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in d["par"].items()}))
with open(os.path.join(HERE, "gap07_tangent_search_found" + ("" if MODE == "free" else "_" + MODE) + ".json"), "w", encoding="utf-8") as fh:
    json.dump(found[:20], fh, indent=1)
own = [d["par"]["owner"] for d in found]
print("owners among instances:", {o_: own.count(o_) for o_ in set(own)})
