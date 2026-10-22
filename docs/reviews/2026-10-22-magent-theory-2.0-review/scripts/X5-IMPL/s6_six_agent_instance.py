"""Ch.12 prop:open-routing: search for an explicit moderate-size instance of the two-stage
preferential routing under the limiting field open:limit with the selector open:selector
(epsilon=1/10, delta=1/20), both sectors identical. The proof uses A>=4096 and an
unspecified large R and small h; this script asks whether moderate values already realize
the event structure, as a candidate implementation test (numerical evidence only)."""
import numpy as np
from scipy.integrate import solve_ivp

EPS, DEL = 0.1, 0.05
def weights(u):
    D = (u[:, None] - u[None, :])**2 / 2
    w = np.exp(-(D - D.min(1, keepdims=True))); return w / w.sum(1, keepdims=True)
def field(t, u):
    w = weights(u); S = w + w.T
    return -u + (S * (u[None, :] - u[:, None])).sum(1)
def block_rows(w, P):
    return np.array([[w[np.ix_(I, J)].sum() / len(I) for J in P] for I in P])
def qualifies(B, a, b):
    ext_a = sum(B[a, k] for k in range(len(B)) if k not in (a, b))
    ext_b = sum(B[b, k] for k in range(len(B)) if k not in (a, b))
    return B[a, b] > EPS and B[b, a] > EPS and B[a, b] - ext_a > DEL and B[b, a] - ext_b > DEL
def run(A, R, b0=2.0, T=12.0, n=24001):
    u0 = np.array([-A - b0, -A + b0, A - b0, A + b0, R - b0, R + b0])
    ts = np.linspace(0, T, n)
    sol = solve_ivp(field, (0, T), u0, t_eval=ts, method='DOP853', rtol=1e-11, atol=1e-13)
    P = [[i] for i in range(6)]; events = []
    w0 = weights(u0); off = w0[~np.eye(6, dtype=bool)].max()
    for k, t in enumerate(sol.t):
        w = weights(sol.y[:, k]); B = block_rows(w, P)
        merged = False
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                if qualifies(B, a, b):
                    events.append((round(t, 4), sorted(P[a] + P[b]), round(B[a, b], 4)))
                    P = [P[c] for c in range(len(P)) if c not in (a, b)] + [P[a] + P[b]]
                    merged = True; break
            if merged: break
    return off, events, P
for A, R in ((4096, None), (8, 40), (6, 30), (5, 25), (4, 20)):
    if R is None:
        print('A=4096 omitted: the proof scale; see report'); continue
    off, ev, P = run(A, R)
    print(f'A={A}, R={R}: max initial off-diagonal weight={off:.2e}; events={ev}; final partition={P}')
