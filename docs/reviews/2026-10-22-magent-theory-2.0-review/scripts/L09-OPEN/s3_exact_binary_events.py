r"""L09-OPEN: exact binary field open:binary-field at T = h^2, both sectors independent.

Integrated in scaled coordinates u = x/h (no time change):
  u_i' = (1 - h^2 u_i^2)[ -atanh(h u_i)/h - \sum_j beta_ij (atanh(h u_i) - atanh(h u_j))/h ]
         + \sum_k beta_ki (u_k - u_i),
  beta_ij = exp(-d(h u_i, h u_j)/h^2) / \sum_k exp(-d(h u_i, h u_k)/h^2),
with d computed cancellation-free:
  d(x,y) = delta^2/(1-y^2) + (1+x)/2 L(delta/(1+y)) + (1-x)/2 L(-delta/(1-y)),  L(s)=log1p(s)-s.
The fixed selector (eps=1/10, delta=1/20) is applied in both sectors and both directions.
Initial points: reference (both sectors equal) plus independent perturbations of all 12
coordinates. Single-threaded; a numerical test of qualitative claims only.
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.integrate import solve_ivp

EPS, DELTA = 0.1, 0.05


def Lfun(s: np.ndarray) -> np.ndarray:
    small = np.abs(s) < 1e-3
    out = np.empty_like(s)
    ss = s[small]
    out[small] = -ss**2 / 2 + ss**3 / 3 - ss**4 / 4 + ss**5 / 5
    out[~small] = np.log1p(s[~small]) - s[~small]
    return out


def dscaled(u: np.ndarray, h: float) -> np.ndarray:
    r"""Matrix D_ij = d(h u_i, h u_j) / h^2 (receiver i, source j)."""
    x = h * u
    X = x[:, None]; Y = x[None, :]
    dl = (u[:, None] - u[None, :])  # delta / h
    sp = h * dl / (1 + Y); sm = -h * dl / (1 - Y)
    with np.errstate(all="ignore"):
        D = dl**2 / (1 - Y**2) + (1 + X) / 2 * Lfun(sp) / h**2 + (1 - X) / 2 * Lfun(sm) / h**2
    return D


def rows(u: np.ndarray, h: float) -> np.ndarray:
    D = dscaled(u, h)
    D = D - D.min(axis=1, keepdims=True)
    W = np.exp(-D)
    return W / W.sum(axis=1, keepdims=True)


def sector_field(u: np.ndarray, h: float) -> np.ndarray:
    x = h * u
    beta = rows(u, h)
    at = np.arctanh(x) / h
    adiff = at[:, None] - at[None, :]
    rec = (1 - x**2) * (-at - (beta * adiff).sum(axis=1))
    inc = (beta * (u[:, None] - u[None, :])).sum(axis=0)  # sum_k beta_ki (u_k - u_i)
    return rec + inc


def make_field(h: float, N: int):
    def f(t: float, z: np.ndarray) -> np.ndarray:
        return np.concatenate([sector_field(z[:N], h), sector_field(z[N:], h)])
    return f


def block_rows(beta: np.ndarray, P: list) -> np.ndarray:
    n = len(P)
    B = np.zeros((n, n))
    for I, bi in enumerate(P):
        for J, bj in enumerate(P):
            B[I, J] = beta[np.ix_(list(bi), list(bj))].sum() / len(bi)
    return B


def qualifies(Bs: list, I: int, J: int) -> tuple[bool, float]:
    ok = True; m = np.inf
    n = Bs[0].shape[0]
    for B in Bs:
        for (r, s) in [(I, J), (J, I)]:
            ext = sum(B[r, K] for K in range(n) if K not in (I, J))
            ok = ok and (B[r, s] > EPS) and (B[r, s] - ext > DELTA)
            m = min(m, B[r, s] - EPS, B[r, s] - ext - DELTA)
    return ok, m


def selector(Z: np.ndarray, ts: np.ndarray, h: float, N: int):
    P = [frozenset([i]) for i in range(N)]
    events = []
    for k, t in enumerate(ts):
        bq = rows(Z[:N, k], h); bs = rows(Z[N:, k], h)
        Bs = [block_rows(bq, P), block_rows(bs, P)]
        n = len(P); used = set(); newP = []
        for I in range(n):
            for J in range(I + 1, n):
                ok, m = qualifies(Bs, I, J)
                if ok:
                    if I in used or J in used:
                        raise RuntimeError("two partners")
                    used |= {I, J}; newP.append(P[I] | P[J])
                    events.append((float(t), sorted(P[I] | P[J])))
        if used:
            P = newP + [P[K] for K in range(n) if K not in used]
    return events, P


def run(A: float, R: float, h: float, sigma: float, T_end: float, seed: int, nts: int = 6001) -> None:
    N = 6
    ref = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    rng = np.random.default_rng(seed)
    z0 = np.concatenate([ref, ref]) + sigma * rng.standard_normal(2 * N)
    if np.max(np.abs(h * z0)) >= 1:
        print("  inadmissible: h*|u| >= 1"); return
    f = make_field(h, N)
    sol = solve_ivp(f, (0, T_end), z0, rtol=1e-9, atol=1e-9, dense_output=True, method="LSODA")
    ts = np.linspace(0, T_end, nts)
    Z = sol.sol(ts)
    ev, P = selector(Z, ts, h, N)
    init_offdiag = max(np.max(rows(z0[:N], h) - np.diag(np.diag(rows(z0[:N], h)))),
                       np.max(rows(z0[N:], h) - np.diag(np.diag(rows(z0[N:], h)))))
    print(f"  A={A}, R={R:.3e}, h={h:.1e} (T={h*h:.1e}), sigma={sigma}, max|x0|={np.max(np.abs(h*z0)):.3f}, "
          f"max initial off-diag weight={init_offdiag:.2e}")
    for e in ev:
        print(f"    merge t={e[0]:.3f} block {e[1]}")
    mind = min(np.min(np.abs(np.diff(np.sort(Z[:N, k])))) for k in range(0, nts, 50))
    print(f"    final partition {[sorted(b) for b in P]}; min within-sector gap of u over grid = {mind:.2e}")


if __name__ == "__main__":
    print("[1] proof reference (A=4096, R=1.7546e7), h=1e-9, perturbation sigma=1e-2 in u units")
    run(4096.0, 1.7546e7, 1e-9, 1e-2, 18.0, 1, nts=9001)
    print("[2] moderate parameters, several h")
    for h in [1e-3, 5e-3, 1e-2, 1.5e-2]:
        run(6.0, 60.0, h, 5e-2, 6.0, 2)
    print("[3] moderate A=10, R=300")
    for h in [1e-3, 2.5e-3]:
        run(10.0, 300.0, h, 5e-2, 8.0, 3)
