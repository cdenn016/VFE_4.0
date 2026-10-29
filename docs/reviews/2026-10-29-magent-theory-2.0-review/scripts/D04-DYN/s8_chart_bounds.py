r"""D04-DYN s8: support for the proposed global chart-flow extension (fixed channels, general spaces).

Three agents with densities against the uniform probability lambda on the circle [0,1), discretized on L points.
Fixed kernels k(v,w) = 1 + 0.8 cos(2 pi (w - v - shift)) (densities in [0.2, 1.8] against lambda) and identity
channels; evaluators with bounded log density; learnable rows at T = 0.3.  The flow is eq:dyn-fisher-flow for
the rows and the chart flow d/dt log q = -(phi - E_q phi) for the laws.  If the essential-inf/sup argument
sketched in the report is right, min and max of log(dq/dlambda) along the trajectory are bounded uniformly in L.
"""
import numpy as np
from scipy.integrate import solve_ivp

T = 0.3
src = {0: [1, 2], 1: [0, 1], 2: [0, 1]}            # (receiver: sources); 1 -> 1 is an identity self channel
kind = {(0, 1): ("kernel", 0.1), (0, 2): ("identity", 0.0), (1, 0): ("kernel", 0.35), (1, 1): ("identity", 0.0),
        (2, 0): ("identity", 0.0), (2, 1): ("kernel", -0.2)}
pi = {i: np.array([0.5, 0.5]) for i in range(3)}


def run(L, tmax=30.0):
    z = (np.arange(L) + 0.5) / L
    Kmat = {}
    for e, (kd, sh) in kind.items():
        if kd == "kernel":
            Kmat[e] = 1 + 0.8 * np.cos(2 * np.pi * (z[None, :] - z[:, None] - sh))   # k(v, w), rows v
    logg = [a * np.cos(2 * np.pi * (z - c)) for a, c in ((1.5, 0.2), (2.0, 0.7), (1.0, 0.45))]
    logg = [lg - np.log(np.mean(np.exp(lg))) for lg in logg]

    def unpack(y):
        lq = y[:3 * L].reshape(3, L)
        q = np.exp(lq)
        q /= q.mean(axis=1, keepdims=True)
        b = np.exp(y[3 * L:].reshape(3, 2))
        b /= b.sum(axis=1, keepdims=True)
        return q, b

    def rhs(t, y):
        q, b = unpack(y)
        lq = np.log(q)
        h = lq - np.array(logg)
        hb = np.zeros((3, 2))
        for i in range(3):
            for n, j in enumerate(src[i]):
                kd, _ = kind[(i, j)]
                nu = q[j] if kd == "identity" else (q[j] @ Kmat[(i, j)]) / L
                ratio = q[i] / nu
                h[i] += b[i, n] * np.log(ratio)
                if kd == "identity":
                    h[j] -= b[i, n] * ratio
                else:
                    h[j] -= b[i, n] * (Kmat[(i, j)] @ ratio) / L
                hb[i, n] = np.mean(q[i] * np.log(ratio)) + T * (np.log(b[i, n] / pi[i][n]) + 1)
        dl = -(h - np.mean(q * h, axis=1, keepdims=True))
        db = -(hb - np.sum(b * hb, axis=1, keepdims=True))
        return np.concatenate([dl.ravel(), db.ravel()])

    lq0 = np.array([2.5 * np.sin(2 * np.pi * (k + 1) * z + k) for k in range(3)])
    lq0 -= np.log(np.mean(np.exp(lq0), axis=1, keepdims=True))
    y0 = np.concatenate([lq0.ravel(), np.log(np.full(6, 0.5))])
    sol = solve_ivp(rhs, (0, tmax), y0, method="RK45", rtol=1e-8, atol=1e-10, t_eval=np.linspace(0, tmax, 301))
    lo, hi = [], []
    for k in range(sol.y.shape[1]):
        q, b = unpack(sol.y[:, k])
        lo.append(np.log(q).min())
        hi.append(np.log(q).max())
    v = rhs(0, sol.y[:, -1])
    return min(lo), max(hi), lo[0], hi[0], np.abs(v).max(), sol.status


for L in (50, 200, 800):
    lo, hi, lo0, hi0, vend, st = run(L)
    print(f"L = {L:4d}: log density range at t=0 [{lo0:.4f}, {hi0:.4f}], over [0,30] [{lo:.4f}, {hi:.4f}]; max abs velocity at t=30 {vend:.1e}; status {st}")
