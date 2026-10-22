r"""X2-FRAME severity check for the open-law construction (12_open_formation.tex,
prop open:routing, def open:selector).

Compare the membership events produced by
  (full) the limiting routing field  u_i' = -u_i + sum_j (w_ij + w_ji)(u_j - u_i),
  (null) the same population with every peer force deleted, u_i' = -u_i,
when the SAME native rows w_ij = exp(-(u_i-u_j)^2/2)/sum_k exp(-(u_i-u_k)^2/2)
(self included) are read by the SAME preferential pair rule (eps=1/10, delta=1/20).
Initial data are the manuscript's: (-A-2,-A+2,A-2,A+2,R-2,R+2). Both sectors are
taken equal, so one sector suffices. RK4, dt=1e-3, single thread.
This is a numerical illustration of the limiting field, not of the exact
scaled binary field at a fixed small h.
"""
import numpy as np

EPS, DELTA = 0.1, 0.05

def rows(u):
    D = (u[:, None] - u[None, :]) ** 2 / 2.0
    W = np.exp(-(D - D.min(axis=1, keepdims=True)))
    return W / W.sum(axis=1, keepdims=True)

def field_full(u):
    w = rows(u)
    S = w + w.T
    return -u + (S * (u[None, :] - u[:, None])).sum(axis=1)

def field_null(u):
    return -u

def block_rows(w, blocks):
    n = len(blocks)
    B = np.zeros((n, n))
    for a, I in enumerate(blocks):
        for b, J in enumerate(blocks):
            B[a, b] = w[np.ix_(I, J)].sum() / len(I)
    return B

def qualifies(B, a, b):
    ext_a = sum(B[a, k] for k in range(len(B)) if k not in (a, b))
    ext_b = sum(B[b, k] for k in range(len(B)) if k not in (a, b))
    return (B[a, b] > EPS and B[b, a] > EPS and B[a, b] - ext_a > DELTA and B[b, a] - ext_b > DELTA)

def run(field, u0, T, dt=1e-3):
    u = u0.copy(); blocks = [[i] for i in range(len(u))]; events = []
    t = 0.0
    w0 = rows(u0); off = w0 - np.diag(np.diag(w0))
    while t < T:
        k1 = field(u); k2 = field(u + dt / 2 * k1); k3 = field(u + dt / 2 * k2); k4 = field(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        B = block_rows(rows(u), blocks)
        merged = True
        while merged:
            merged = False
            for a in range(len(blocks)):
                for b in range(a + 1, len(blocks)):
                    if qualifies(B, a, b):
                        new = blocks[a] + blocks[b]
                        events.append((round(t, 3), sorted(new)))
                        blocks = [blk for k, blk in enumerate(blocks) if k not in (a, b)] + [new]
                        B = block_rows(rows(u), blocks); merged = True
                        break
                if merged:
                    break
    return events, blocks, off.max()

for A, R, T in [(8.0, 200.0, 6.0), (64.0, 2000.0, 7.5), (4096.0, 4.0e5, 10.5)]:
    u0 = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2])
    for name, f in [("full", field_full), ("null", field_null)]:
        ev, bl, w0max = run(f, u0, T)
        print(f"A={A:g} R={R:g} {name}: max initial off-diagonal weight={w0max:.2e}")
        print("   events:", ev)
        print("   final blocks:", bl)
