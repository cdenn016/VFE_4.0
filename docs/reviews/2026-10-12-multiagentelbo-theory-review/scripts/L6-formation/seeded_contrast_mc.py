r"""L6-formation: Monte Carlo of the contrast-formation event of c14 prop:phil-seeded-contrast-formation
(frozen rows 1/2), regime eps = 1/100, T = eps^8, horizon 2/eps, initial law N(m0, I_12) x uniform S.

Between proposals the configuration is the exact sheet Ornstein--Uhlenbeck flow
x(t) = e^{-K_S t} x + noise, K_S = H_S + T (I + R_ev); the noise has scale sqrt(2T) ~ 1.4e-8 and is
included exactly through its covariance.  Proposals: rate-one clock, all four receivers redrawn from
posterior rows beta_i(j) propto pi_ij exp(-D_{e(i,j)}/T).  The event is checked on a time grid (step 0.05),
so the estimate is approximate in the interval clause.  Reports the empirical probability next to the
proved lower bound e^{-1/4}/16 * sum_pm N(m0,I){|x -+ x0| < 1/100}.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, ".")
from four_cycle_checks import ASSIGN, H_of, R_ev, cov_b, cov_m, edge_of  # noqa: E402

EPS = 0.01
T = EPS ** 8
HOR = 2 / EPS
DT = 0.1


def main(nsamp: int = 400, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    K = {S: H_of(S) + T * (np.eye(12) + R_ev) for S in ASSIGN}
    eig = {S: np.linalg.eigh(K[S]) for S in ASSIGN}
    Ab = np.stack([cov_b(i) for i in range(4)])
    Am = np.stack([cov_m(i) for i in range(4)])
    blocks = []
    for i in range(4):
        blocks.append(([i], [(i - 1) % 4, (i + 1) % 4]))
        blocks.append(([i, (i + 1) % 4], [(i - 1) % 4, (i + 2) % 4]))
    xo = np.concatenate([np.zeros(8), np.ones(4)])
    for name, m0 in (("m0 = 0", np.zeros(12)), ("m0 = x_circ", xo), ("m0 = random", rng.normal(size=12))):
        hits = 0
        for _ in range(nsamp):
            x = m0 + rng.normal(size=12)
            S = ASSIGN[rng.integers(16)]
            rb0 = Ab @ x
            entr = [any(rb0[e] ** 2 / 2 > 0.25 for e in internal) for internal, _ in blocks]
            if not any(entr):
                continue
            t = 0.0
            good_since = [None] * len(blocks)
            found = False
            while t < HOR and not found:
                tau = rng.exponential(1.0)
                tend = min(t + tau, HOR)
                lam, V = eig[S]
                c0 = V.T @ x
                ss = np.arange(0.0, tend - t + 1e-12, DT)
                XS = (V * 1.0) @ (np.exp(-np.outer(lam, ss)) * c0[:, None])  # 12 x len(ss)
                RB, RM = Ab @ XS, Am @ XS
                for b, (internal, boundary) in enumerate(blocks):
                    if not entr[b] or found:
                        continue
                    okv = np.all(RB[internal] ** 2 / 2 <= EPS ** 4, axis=0) & np.all(RM[internal] ** 2 <= EPS ** 4, axis=0)                         & np.all(RB[boundary] ** 2 > 1.0, axis=0)
                    for k, ok in enumerate(okv):
                        if ok:
                            if good_since[b] is None:
                                good_since[b] = t + ss[k]
                            if t + ss[k] - good_since[b] >= 1.0:
                                found = True
                                break
                        else:
                            good_since[b] = None
                # exact noise over the segment (negligible at T = 1e-16, included for completeness)
                d = tend - t
                var = T * (1 - np.exp(-2 * lam * d)) / np.where(lam > 0, lam, 1.0)
                var = np.where(lam > 1e-30, var, 2 * T * d)
                x = V @ (np.exp(-lam * d) * c0 + np.sqrt(var) * rng.normal(size=12))
                t = tend
                if t >= HOR:
                    break
                D = 0.5 * ((Ab @ x) ** 2 + (Am @ x) ** 2)
                newS = []
                for i in range(4):
                    js = [(i - 1) % 4, (i + 1) % 4]
                    costs = np.array([D[edge_of(i, j)] for j in js]) / T
                    w = np.exp(-(costs - costs.min()))
                    w /= w.sum()
                    newS.append(js[rng.choice(2, p=w)])
                S = tuple(newS)
            hits += found
        # proved lower bound (frozen rows): e^{-1/4}/16 * sum_pm N(m0,I){|x -+ x0|<0.01}
        def ball(c):
            nc = float(np.sum((m0 - c) ** 2))
            from scipy.stats import ncx2
            return ncx2.cdf(1e-4, 12, nc) if nc > 0 else chi2.cdf(1e-4, 12)
        lb = math.exp(-0.25) / 16 * (ball(xo) + ball(-xo))
        p = hits / nsamp
        print(f"{name}: empirical P(contrast formation) = {p:.3f} +- {math.sqrt(p*(1-p)/nsamp):.3f} "
              f"(n={nsamp}); proved lower bound = {lb:.3e}")


if __name__ == "__main__":
    main()
