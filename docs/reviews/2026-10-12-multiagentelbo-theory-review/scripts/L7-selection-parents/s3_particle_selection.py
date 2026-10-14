r"""L7 check 3: particle (McKean--Vlasov) simulation of pairing selection.

Model: four-cycle kinetics of c13 def:rg-formation-cycle-experiment at fixed T,
rows learned by  dp_i/dt = eps (q_t(sigma_i=+1) - p_i)  (eq:phil-source-learning),
with q_t replaced by the empirical law of N particles. Exact OU transition on
the held sheet over each step dt; at most one clock ring per step with
probability 1 - e^{-dt}, the redraw using the configuration at the end of the
step. This is a finite-N, time-discretized surrogate of the law-level system;
it is evidence, not proof.

Outputs:
  (A) imprint j_T(m0) = int_0^inf E f_v(sigma_t) dt from frozen runs;
  (B) learned runs: matching occupancy and rows on [t_L, 2 t_L],
      t_L = 2 eps^{-1} log(1/eps), and whether the selected matching equals
      sign(j_T(m0)) (S^(0) for j > 0);
  (C) frozen run: q(S^(0)) - q(S^(1)) at late times.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import expm, cholesky

def build(T: float, dt: float):
    AB = np.zeros((4, 12)); AM = np.zeros((4, 12))
    for i in range(4):
        j = (i + 1) % 4
        AB[i, j] += 1; AB[i, i] -= 1; AB[i, 8 + i] -= 1
        AM[i, 4 + j] += 1; AM[i, 4 + i] -= 1; AM[i, 8 + i] -= 1
    Dfr = np.zeros((4, 12)); Dfr[:, :4] = np.eye(4); Dfr[:, 4:8] = -np.eye(4)
    Om = np.eye(12) + 2 * Dfr.T @ Dfr
    sigmas = list(itertools.product([1, -1], repeat=4))
    E = np.zeros((16, 12, 12)); L = np.zeros((16, 12, 12))
    for k, s in enumerate(sigmas):
        m = [(s[e] == 1) + (s[(e + 1) % 4] == -1) for e in range(4)]
        H = sum(m[e] * (np.outer(AB[e], AB[e]) + np.outer(AM[e], AM[e])) for e in range(4))
        A = H + T * Om
        E[k] = expm(-A * dt)
        C = T * np.linalg.solve(A, np.eye(12) - E[k] @ E[k])
        C = (C + C.T) / 2
        L[k] = cholesky(C, lower=True)
    return AB, AM, E, L, np.array(sigmas)

def sig_index(sig: np.ndarray) -> np.ndarray:
    # sigmas ordered by itertools.product([1,-1]) -> bit=1 for -1, MSB first
    bits = (sig == -1).astype(int)
    return bits[:, 0] * 8 + bits[:, 1] * 4 + bits[:, 2] * 2 + bits[:, 3]

def run(cfg: dict) -> dict:
    T, dt, eps, N, tmax, seed = cfg["T"], cfg["dt"], cfg["eps"], cfg["N"], cfg["tmax"], cfg["seed"]
    m0 = np.array(cfg["m0"])
    AB, AM, E, L, sigmas = build(T, dt)
    rng = np.random.default_rng(seed)
    x = m0 + rng.standard_normal((N, 12))
    sig = rng.choice([1, -1], size=(N, 4))
    p = np.full(4, 0.5)
    v = np.array([1, -1, 1, -1])
    pring = 1 - np.exp(-dt)
    nsteps = int(round(tmax / dt))
    rec_every = max(1, int(round(cfg.get("rec", 5.0) / dt)))
    rec = []
    jint = 0.0
    S0 = np.array([1, -1, 1, -1]); S1 = -S0
    for n in range(nsteps):
        idx = sig_index(sig)
        x = np.matmul(E[idx], x[:, :, None])[:, :, 0] + np.matmul(L[idx], rng.standard_normal((N, 12, 1)))[:, :, 0]
        ring = rng.random(N) < pring
        if ring.any():
            xr = x[ring]
            D = 0.5 * ((xr @ AB.T) ** 2 + (xr @ AM.T) ** 2)  # edge costs (n,4)
            # receiver i: +1 -> edge i, -1 -> edge i-1
            logit = np.log(p) - np.log1p(-p) - (D - np.roll(D, 1, axis=1)) / T
            prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
            sig[ring] = np.where(rng.random(prob.shape) < prob, 1, -1)
        a = (sig == 1).mean(axis=0)
        fv = 0.5 * (sig @ v).mean()
        jint += fv * dt
        if eps > 0:
            p = p + dt * eps * (a - p)
        if n % rec_every == 0:
            q0 = np.all(sig == S0, axis=1).mean(); q1 = np.all(sig == S1, axis=1).mean()
            rec.append([n * dt, q0, q1, fv, *p.tolist()])
    return {"cfg": cfg, "rec": rec, "jint": jint}

if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    rng = np.random.default_rng(12345)
    ucirc = np.r_[np.zeros(8), [0, 1, 0, -1]]
    m_gen = rng.normal(0, 1.5, 12)                    # a generic mean
    rot = lambda m: np.r_[np.roll(m[:4], 1), np.roll(m[4:8], 1), np.roll(m[8:], 1)]
    means = {"generic": m_gen, "generic_rot": rot(m_gen), "s_ucirc": 4.0 * ucirc}
    dt = 0.05
    cfgs = []
    # (A) frozen imprint runs
    for name, m in means.items():
        cfgs.append({"tag": f"frozen_{name}", "T": T, "dt": dt, "eps": 0.0, "N": 20000, "tmax": 400.0, "seed": 1, "m0": m.tolist(), "rec": 5.0})
    # (B) learned runs at two mobilities, two seeds for the generic mean
    for eps in [0.02, 0.01]:
        tL = 2 / eps * np.log(1 / eps)
        for name in ["generic", "generic_rot"]:
            for seed in ([2, 3] if name == "generic" else [2]):
                cfgs.append({"tag": f"learn_{name}_eps{eps}_seed{seed}", "T": T, "dt": dt, "eps": eps, "N": 4000, "tmax": 2 * tL, "seed": seed, "m0": means[name].tolist(), "rec": 10.0})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(run, cfgs))
    out = {"T": T, "means": {k: v.tolist() for k, v in means.items()}, "runs": res, "wall": time.time() - t0}
    with open(f"s3_results_T{T}.json", "w") as f:
        json.dump(out, f)
    print("wall", out["wall"])
    for r in res:
        c = r["cfg"]; rec = np.array(r["rec"])
        if c["eps"] == 0:
            late = rec[rec[:, 0] > 300]
            print(f"{c['tag']}: j_T estimate {r['jint']:+.3f}; late q(S0)-q(S1) mean {np.mean(late[:,1]-late[:,2]):+.4f}, late q(S0)={late[:,1].mean():.4f}, q(S1)={late[:,2].mean():.4f}")
        else:
            eps = c["eps"]; tL = 2 / eps * np.log(1 / eps)
            w = rec[(rec[:, 0] >= tL) & (rec[:, 0] <= 2 * tL)]
            sel = "S0" if w[:, 1].mean() > w[:, 2].mean() else "S1"
            qnotM = 1 - np.maximum(w[:, 1], w[:, 2])
            print(f"{c['tag']}: selected {sel}; sup_window q(S!=M) {qnotM.max():.4f}; mean {qnotM.mean():.4f}; rows at 2tL {np.round(rec[-1,4:],4)}")
