r"""L09-OPEN: limiting field open:limit at the proof's reference point.

u_i' = -u_i + \sum_j (w_ij + w_ji)(u_j - u_i),  w_ij = e^{-(u_i-u_j)^2/2} / \sum_k e^{-(u_i-u_k)^2/2}.

Part A: four-agent reflection core open:core from a(0)=A=4096, b(0)=2: checks
        open:early-bounds, tau <= 2 log 4, t_A >= log(A/4)/2, b(t_A) <= 2 (A/4)^{-1/4}.
Part B: b=0 comparison from a=4: crossing a_c = sqrt(log 9 / 2), S bound.
Part C: six agents (core + pair at R +- 2), R from the proof's prescription, run the
        fixed selector (eps=1/10, delta=1/20) on [0, L] and beyond, to record all merges.
This is a numerical test of qualitative claims; it proves nothing.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

EPS, DELTA = 0.1, 0.05


def weights(u: np.ndarray) -> np.ndarray:
    D = (u[:, None] - u[None, :]) ** 2 / 2.0
    D = D - D.min(axis=1, keepdims=True)  # self entry is 0, so min is 0 anyway
    W = np.exp(-D)
    return W / W.sum(axis=1, keepdims=True)


def field(t: float, u: np.ndarray) -> np.ndarray:
    w = weights(u)
    S = w + w.T
    return -u + (S * (u[None, :] - u[:, None])).sum(axis=1)


def core_field(t: float, y: np.ndarray) -> np.ndarray:
    a, b = y
    W = np.exp(-2 * b * b); v0 = np.exp(-2 * a * a)
    vp = np.exp(-2 * (a + b) ** 2); vm = np.exp(-2 * (a - b) ** 2)
    Zp = 1 + W + v0 + vp; Zm = 1 + W + v0 + vm
    cp = W * (1 / Zp + 1 / Zm); c0 = v0 * (1 / Zp + 1 / Zm)
    ad = -a - 2 * a * c0 - 2 * ((a + b) * vp / Zp + (a - b) * vm / Zm)
    bd = -b - 2 * b * cp - 2 * ((a + b) * vp / Zp - (a - b) * vm / Zm)
    return np.array([ad, bd])


def block_rows(beta: np.ndarray, P: list[frozenset]) -> np.ndarray:
    n = len(P)
    B = np.zeros((n, n))
    for I, bi in enumerate(P):
        for J, bj in enumerate(P):
            B[I, J] = sum(beta[i, j] for i in bi for j in bj) / len(bi)
    return B


def qualifying(B: np.ndarray) -> list[tuple[int, int, float]]:
    out = []
    n = B.shape[0]
    for I in range(n):
        for J in range(I + 1, n):
            ok = True; marg = []
            for (r, s) in [(I, J), (J, I)]:
                ext = sum(B[r, K] for K in range(n) if K not in (I, J))
                cond = B[r, s] > EPS and B[r, s] - ext > DELTA
                marg.append(min(B[r, s] - EPS, B[r, s] - ext - DELTA))
                ok = ok and cond
            if ok:
                out.append((I, J, min(marg)))
    return out


def run_selector(sol, ts: np.ndarray, N: int, label: str) -> list:
    P = [frozenset([i]) for i in range(N)]
    events = []
    U = sol.sol(ts)
    for k, t in enumerate(ts):
        u = U[:, k]
        beta = weights(u)
        B = block_rows(beta, P)
        q = qualifying(B)
        if q:
            used = set(); newP = []
            for (I, J, m) in q:
                if I in used or J in used:
                    raise RuntimeError("block with two partners")
                used |= {I, J}
                newP.append(P[I] | P[J])
                events.append((float(t), sorted(P[I] | P[J]), float(m)))
            newP += [P[K] for K in range(len(P)) if K not in used]
            P = newP
    return events, P


def main() -> None:
    A = 4096.0
    # Part A
    solA = solve_ivp(core_field, (0, 20), [A, 2.0], rtol=1e-11, atol=1e-13, dense_output=True,
                     events=[lambda t, y: y[0] - 4.0, lambda t, y: y[1] - 0.5])
    t_A = solA.t_events[0][0]; tau = solA.t_events[1][0]
    tt = np.linspace(0, t_A, 4001)
    Y = solA.sol(tt); D = np.array([core_field(0, Y[:, k]) for k in range(len(tt))]).T
    ra = D[0] / Y[0]; rb = D[1] / Y[1]
    print(f"[A] tau={tau:.4f} (bound 2log4={2*np.log(4):.4f}); t_A={t_A:.4f} (lower bound {0.5*np.log(A/4):.4f})")
    print(f"[A] b(t_A)={solA.sol(t_A)[1]:.3e} (bound {2*(A/4)**-0.25:.3e}); a'/a in [{ra.min():.4f},{ra.max():.4f}] (claim [-2,-1]);"
          f" b'/b in [{rb.min():.4f},{rb.max():.4f}] (claim [-4,-1/2])")
    # pair weight at tau
    a_t, b_t = solA.sol(tau)
    W = np.exp(-2 * b_t**2)
    print(f"[A] p_+(tau) = {W/(1+W+np.exp(-2*a_t**2)+np.exp(-2*(a_t+b_t)**2)):.4f} (claim > 1/4), a(tau)={a_t:.1f} (claim >= A/256={A/256:.1f})")
    # Part B
    solB = solve_ivp(core_field, (0, 10), [4.0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True,
                     events=[lambda t, y: np.exp(-2*y[0]**2)/(1+np.exp(-2*y[0]**2)) - 0.1,
                             lambda t, y: np.exp(-2*y[0]**2)/(1+np.exp(-2*y[0]**2)) - 0.2])
    tc = solB.t_events[0][0]; S = solB.t_events[1][0]
    print(f"[B] parent crossing at a={solB.sol(tc)[0]:.6f} (claim {np.sqrt(np.log(9)/2):.6f}), t={tc:.4f}; S={S:.4f} (bound {np.log(4/np.sqrt(np.log(4)/2)):.4f})")
    L = t_A + S
    # Part C: R from the proof: D_R = R e^{-L} - A - 6 >= 12
    R = (A + 6 + 12) * np.exp(L)
    u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    print(f"[C] L={L:.4f}, R={R:.4e}")
    T_end = L + 14.0
    sol = solve_ivp(field, (0, T_end), u0, rtol=1e-10, atol=1e-10, dense_output=True, method="LSODA")
    ts = np.linspace(0, T_end, 40001)
    events, P = run_selector(sol, ts, 6, "limit")
    for e in events:
        print(f"[C] merge at t={e[0]:.4f}: block {e[1]}, qualifying margin at detection {e[2]:.2e}")
    print("[C] final partition:", [sorted(b) for b in P])
    # exterior masses through L
    mx = 0.0
    for t in np.linspace(0, L, 4001):
        beta = weights(sol.sol(t))
        mx = max(mx, beta[:4, 4:].sum(axis=1).max(), beta[4:, :4].sum(axis=1).max())
    print(f"[C] max primitive core<->third-pair mass on [0,L] = {mx:.2e} (claim < 1e-3)")


if __name__ == "__main__":
    main()
