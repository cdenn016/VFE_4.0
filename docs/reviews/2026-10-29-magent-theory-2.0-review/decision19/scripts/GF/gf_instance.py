r"""The declared GF instance: histories, controls, certificates, parent, counterfactuals.

Constitutive data: N = 4, s_i^2 = 1, rho_i^2 = 1/4, tau^2 = 1/10, sigma^2 = 4,
catalog = all 60 nested pairs (Pi1, Pi2) on four agents, uniform prior.
Center of the admitted class: records o* = (2.9, 3.1, 4.4, 4.6),
initial means m0* = (-9.2, -3.0, 4.2, 10.4), any initial variances.
Floating point (double); every strict inequality is reported with its margin.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from gf_model import Model, FINEST4, A_SEP, A_JOIN, name, compress  # noqa: E402
from gf_controls import run_full, run_lp, run_rn, run_frozen  # noqa: E402

M = Model(np.ones(4), np.full(4, 0.25), 0.1, 4.0)
o = np.array([2.9, 3.1, 4.4, 4.6])
m0 = np.array([-9.2, -3.0, 4.2, 10.4])
a, b = M.a[0], M.b[0]
v0 = 1 + 0.1 + 4.0
kappa = v0 / (v0 + 0.25)
kfin = 4.1 / 5.1
print(f"a={a:.6f} b={b:.6f} kappa={kappa:.6f} k_fin={kfin:.6f} eta={1/(1+4):.6f}")


def events(sel):
    out = [(0, sel[0])]
    for n in range(1, len(sel)):
        if sel[n] != sel[n - 1]:
            out.append((n, sel[n]))
    return out


N = 120
res = {}
for lab, f in (("full", run_full), ("lp", run_lp), ("rn", run_rn), ("frozen", run_frozen)):
    sel, ms, gaps = f(M, m0, o, N)
    res[lab] = (sel, ms, gaps)
    print(f"{lab:7s} events:", [(n, name(S)) for n, S in events(sel)], f" min gap over {N} sweeps = {min(gaps):.4g}")

# tail certificates: ball around the limit on which the final structure is the strict argmax
def ball_radius(Sf, mstar):
    r_best = np.inf
    for S in M.cat:
        if S == Sf:
            continue
        d0 = M.score(Sf, mstar) - M.score(S, mstar)
        H = M.Ci[Sf] - M.Ci[S]
        g = np.linalg.norm(H @ mstar)
        h = 0.5 * np.linalg.norm(H, 2)
        # need d0 - g r - h r^2 > 0
        r = (-g + np.sqrt(g * g + 4 * h * d0)) / (2 * h) if h > 0 else d0 / g
        r_best = min(r_best, r)
    return r_best


for lab, Sf, mstar, rate in (("full", A_JOIN, M.fixed_point(A_JOIN, o), b * np.linalg.norm(M.K(A_JOIN), 2)),
                             ("lp", A_JOIN, kappa * o, b * kfin)):
    r = ball_radius(Sf, mstar)
    ms = res[lab][1]
    n_in = next(n for n in range(len(ms)) if np.linalg.norm(ms[n] - mstar) < r and all(res[lab][0][k] == Sf for k in range(n, len(res[lab][0]))))
    print(f"{lab}: limit {np.round(mstar, 5)}, certificate radius {r:.4g}, contraction {rate:.4f}, inside from sweep {n_in}")
print("rn limit kappa o selects", name(M.select(kappa * o)[0]), f"gap {M.select(kappa*o)[1]:.4g}")

# openness: random perturbations of (m0, o)
rng = np.random.default_rng(0)
ref = {lab: compress(res[lab][0]) for lab in res}
for eps in (1e-3, 5e-3, 1e-2):
    bad = 0
    for _ in range(200):
        dm, do = rng.uniform(-eps, eps, 4), rng.uniform(-eps, eps, 4)
        for lab, f in (("full", run_full), ("lp", run_lp), ("rn", run_rn)):
            if compress(f(M, m0 + dm, o + do, 80)[0]) != ref[lab]:
                bad += 1
                break
    print(f"perturbation eps={eps}: {bad}/200 changed a sequence")

# selector-free observables
ms_full, ms_lp = res["full"][1], res["lp"][1]
print("within-block contrasts after the first-stage sweep: full", np.round([ms_full[-1][1] - ms_full[-1][0], ms_full[-1][3] - ms_full[-1][2]], 6),
      " a*do =", np.round([a * (o[1] - o[0]), a * (o[3] - o[2])], 6), " nulls kappa*do =", np.round([kappa * (o[1] - o[0]), kappa * (o[3] - o[2])], 6))
print("contraction factor a/kappa =", a / kappa)
mstar = M.fixed_point(A_JOIN, o)
print("full limit m* =", np.round(mstar, 5), " null limit kappa o =", np.round(kappa * o, 5))
t1 = events(res["full"][0])[1][0]
print(f"first event sweep {t1}: full state == lp state:", np.allclose(ms_full[t1], ms_lp[t1]))

# exact posterior over structures given o, and soft stationary factor at m*
ex = M.exact_log_post(o)
lps = np.array([v for v, _ in ex]); p = np.exp(lps - lps.max()); p /= p.sum()
print("exact posterior top 4:", [(name(S), round(float(pp), 4)) for (v, S), pp in zip(ex[:4], p[:4])])
sc = M.scores(mstar)
ls = np.array([v for v, _ in sc]); q = np.exp(ls - ls.max()); q /= q.sum()
print("soft structure factor at m* top 4:", [(name(S), round(float(qq), 4)) for (v, S), qq in zip(sc[:4], q[:4])])

# parent: block retention and inherited dynamics
blocks = [(0, 1), (2, 3)]
PAR = [A_SEP, A_JOIN]
par = Model(np.full(2, 0.5), np.full(2, 0.125), 0.1, 4.0,
            catalog=[(((0,), (1,)), ((0,), (1,))), (((0,), (1,)), ((0, 1),))])
mapS = {A_SEP: par.cat[0], A_JOIN: par.cat[1]}
ohat = np.array([o[list(B)].mean() for B in blocks])
sel = res["full"][0]
err = 0.0
mh = np.array([ms_full[t1][list(B)].mean() for B in blocks])
for n in range(t1, N):
    S, _ = par.select(mh)
    assert S == mapS[sel[n]], (n, S, sel[n])
    mh = par.a * ohat + par.b * (par.K(S) @ mh)
    err = max(err, np.max(np.abs(mh - np.array([ms_full[n + 1][list(B)].mean() for B in blocks]))))
print("inherited parent reproduces the fine block means and selections from the handoff, max error", err)
for S in par.cat:
    fp = par.fixed_point(S, ohat)
    s_, g_ = par.select(fp)
    print(f"parent structure {S[1]} self-consistent: {s_ == S}  (selected {s_[1]}, gap {g_:.4g}) at fixed point {np.round(fp, 5)}")

# counterfactual handoffs with records fixed: grid of parent states at which sep is selected
cnt, unions, maxsteps = 0, 0, 0
for u in np.linspace(-30, 30, 121):
    for v in np.linspace(-30, 30, 121):
        mh = np.array([u, v])
        if par.select(mh)[0] != par.cat[0]:
            continue
        cnt += 1
        for k in range(400):
            S, _ = par.select(mh)
            if S == par.cat[1]:
                unions += 1
                maxsteps = max(maxsteps, k)
                break
            mh = par.a * ohat + par.b * (par.K(S) @ mh)
print(f"counterfactual parent states in the selector domain: {cnt}, union occurs from {unions}, max sweeps {maxsteps}")

# amplifying grade at the first event: lp state equals full state, so the ratio is 1
# a non-nested interaction-dependent path in the parent-bistable regime (observation)
M2 = Model(np.ones(4), np.full(4, 0.25), 0.05, 4.0)
o2 = np.array([2.9, 3.1, 5.65, 5.85])
m02 = np.array([-4.0, 2.0, -9.0, -1.0])
for lab, f in (("full", run_full), ("lp", run_lp), ("rn", run_rn)):
    print("bistable-regime example", lab, [name(S) for S in compress(f(M2, m02, o2, 150)[0])])
