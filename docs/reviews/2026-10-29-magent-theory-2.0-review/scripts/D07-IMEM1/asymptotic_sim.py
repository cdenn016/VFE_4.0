r"""Numerical illustration for lens D07-IMEM1 (not a proof).

Binary preparation of imem:binary-preparation, alternating speakers starting with agent 1,
identity encoders.  Two filters run side by side on the same simulated transcripts:

  * the four-weight filter imem:reduced-update on lambda_t(d0, d1);
  * the product-tilt filter, which stores only the two cumulative one-agent likelihood
    ratios r_i = F_i(1)/F_i(0) and reconstructs lambda_t(d) propto F_0(d0) F_1(d1) lambda_0(d).

Reported: the maximum discrepancy between the two filters along all runs, and the average
public posterior mass P(D = true pair | H_{1:t}) at several horizons, which illustrates the
asymptotic-revelation extension (both private records become public in the limit).
Single thread, seeded, float arithmetic, a few seconds.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

T0 = time.perf_counter()
order = [(0, 0), (0, 1), (1, 0), (1, 1)]
lam0 = {(0, 0): 9 / 32, (0, 1): 7 / 32, (1, 0): 7 / 32, (1, 1): 9 / 32}
V = [{(0, 0): 2 / 3, (0, 1): 4 / 7, (1, 0): 3 / 7, (1, 1): 1 / 3},   # P(Y0=1 | D)
     {(0, 0): 2 / 3, (0, 1): 3 / 7, (1, 0): 4 / 7, (1, 1): 1 / 3}]   # P(Y1=1 | D)


def kernel(lam: dict, j: int) -> dict:
    """P(H=1 | D_j = dj, H_{1:t}) = sum_{d_other} V_j(d) lam(d | d_j)."""
    out = {}
    for dj in (0, 1):
        m = sum(lam[d] for d in order if d[j] == dj)
        out[dj] = sum(V[j][d] * lam[d] for d in order if d[j] == dj) / m
    return out


def tilt(r0: float, r1: float) -> dict:
    un = {d: lam0[d] * (r0 if d[0] == 1 else 1.0) * (r1 if d[1] == 1 else 1.0) for d in order}
    z = sum(un.values())
    return {d: v / z for d, v in un.items()}


def sample_prep(rng: random.Random) -> tuple:
    x0 = rng.random() < 0.5
    x1 = x0 if rng.random() < 0.75 else (not x0)
    d0 = x0 if rng.random() < 0.75 else (not x0)
    d1 = x1 if rng.random() < 0.75 else (not x1)
    return int(d0), int(d1)


rng = random.Random(12345)
horizons = [10, 50, 200, 1000]
runs = 300
mass = {T: 0.0 for T in horizons}
maxdiff = 0.0
for _ in range(runs):
    D = sample_prep(rng)
    lam = dict(lam0)
    logr = [0.0, 0.0]
    for t in range(1, max(horizons) + 1):
        j = 1 if t % 2 == 1 else 0
        k = kernel(lam, j)
        h = 1 if rng.random() < k[D[j]] else 0
        lik = {dj: (k[dj] if h == 1 else 1 - k[dj]) for dj in (0, 1)}
        un = {d: lam[d] * lik[d[j]] for d in order}
        z = sum(un.values())
        lam = {d: v / z for d, v in un.items()}
        # product-tilt filter: kernel computed from the tilt state itself
        lt = tilt(2.718281828459045 ** logr[0], 2.718281828459045 ** logr[1])
        kt = kernel(lt, j)
        likt = {dj: (kt[dj] if h == 1 else 1 - kt[dj]) for dj in (0, 1)}
        import math
        logr[j] += math.log(likt[1]) - math.log(likt[0])
        lt2 = tilt(math.exp(logr[0]), math.exp(logr[1]))
        maxdiff = max(maxdiff, max(abs(lt2[d] - lam[d]) for d in order))
        if t in mass:
            mass[t] += lam[D] / runs
out = {
    "runs": runs,
    "max_abs_difference_four_weight_vs_two_ratio_filter": maxdiff,
    "mean_public_mass_on_true_private_pair": {str(T): round(m, 4) for T, m in mass.items()},
    "wall_seconds": round(time.perf_counter() - T0, 3),
    "scope": "Monte Carlo illustration on one finite model; not evidence for a general theorem.",
}
Path(__file__).with_name("asymptotic_sim_output.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
