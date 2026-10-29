"""XC2-FRAME independent re-run of the integrations quoted in open:null-control
(12_open_formation.tex:248-249): limit field open:limit versus null limit du/dt = -u,
read by the same native rows and the preferential pair rule open:selector
(eps = 1/10, delta = 1/20), from u(0) = (-A-2, -A+2, A-2, A+2, R-2, R+2).

limit field: du_i/dt = -u_i + sum_j (w_ij + w_ji)(u_j - u_i),
             w_ij = exp(-(u_i-u_j)^2/2) / sum_k exp(-(u_i-u_k)^2/2)   (self included)
block rows:  B_IJ = |I|^{-1} sum_{i in I} sum_{j in J} w_ij        (open:block-row)
rule:        I,J qualify when B_IJ > eps and B_IJ - sum_{K != I,J} B_IK > delta in both
             directions (the two sectors are identical here); merge at first contact.
RK4, fixed step dt; we report event times at two step sizes to show convergence.
"""
import numpy as np

EPS, DEL = 0.1, 0.05

def rows(u):
    d2 = (u[:, None] - u[None, :]) ** 2
    W = np.exp(-0.5 * d2)
    return W / W.sum(axis=1, keepdims=True)

def limit_field(u):
    W = rows(u); S = W + W.T
    return -u + (S * (u[None, :] - u[:, None])).sum(axis=1)

def null_field(u):
    return -u

def block_row(W, I, J):
    return W[np.ix_(I, J)].sum() / len(I)

def qualifies(W, blocks, a, b):
    I, J = blocks[a], blocks[b]
    for (X, Y, idx) in ((I, J, a), (J, I, b)):
        bxy = block_row(W, X, Y)
        ext = sum(block_row(W, X, blocks[k]) for k in range(len(blocks)) if k not in (a, b))
        if not (bxy > EPS and bxy - ext > DEL):
            return False
    return True

def run(field, A, R, T, dt):
    u = np.array([-A - 2, -A + 2, A - 2, A + 2, R - 2, R + 2], dtype=float)
    blocks = [[i] for i in range(6)]
    events = []
    t = 0.0
    n = int(round(T / dt))
    for _ in range(n):
        k1 = field(u); k2 = field(u + 0.5 * dt * k1); k3 = field(u + 0.5 * dt * k2); k4 = field(u + dt * k3)
        u = u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        W = rows(u)
        merged = True
        while merged:
            merged = False
            for a in range(len(blocks)):
                for b in range(a + 1, len(blocks)):
                    if qualifies(W, blocks, a, b):
                        events.append((round(t, 3), tuple(sorted(blocks[a] + blocks[b]))))
                        new = sorted(blocks[a] + blocks[b])
                        blocks = [blocks[k] for k in range(len(blocks)) if k not in (a, b)] + [new]
                        merged = True
                        break
                if merged:
                    break
    return events, blocks

if __name__ == "__main__":
    for (A, R, T) in ((8, 200, 3.0), (64, 2000, 5.0)):
        for dt in (2e-3, 1e-3):
            ef, bf = run(limit_field, A, R, T, dt)
            en, bn = run(null_field, A, R, T, dt)
            print(f"A={A}, R={R}, dt={dt}: full {ef}")
            print(f"{'':>24s}null {en}")
    A, R, T, dt = 4096, 4e5, 10.5, 2e-3
    ef, bf = run(limit_field, A, R, T, dt)
    en, bn = run(null_field, A, R, T, dt)
    print(f"A={A}, R={R:g}, dt={dt}, horizon {T}: full {ef}; final blocks {bf}")
    print(f"{'':>34s}null {en}; final blocks {bn}")
