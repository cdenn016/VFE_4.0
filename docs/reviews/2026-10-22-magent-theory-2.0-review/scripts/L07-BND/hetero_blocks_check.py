r"""Extension check: a two-block heterogeneous family at depth 2.

Leaves LL, LR use belief profile f^L and model profile g^L; leaves RL, RR use
f^R, g^R (f^L != f^R, different base marginals).  The circle/sign structure is
the same as in depth2_flow_check.py.  Rows are class-constant within each block
but differ between blocks (b^L, b^R).  Claimed closed profile equations, for a
leaf in block X with other block Y:
  h^X = 2 log f^X + V^X - b^X_0 P_0 log f^Y - b^X_1 P_1 log f^X
        - b^Y_0 (P_0 f^Y)/f^X - b^X_1 (P_1 f^X)/f^X,
  D^X_0 = int f^X (log f^X - P_0 log f^Y),  D^X_1 = int f^X (log f^X - P_1 log f^X).
The fine system code (generic energy, generic gradient, Fisher flow) is reused
from depth2_flow_check.py by executing its definitions section only.
"""
import os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "depth2_flow_check.py")).read().split("# ---------------- set up ----------------")[0]
exec(src)

rng = np.random.default_rng(23)
def norm(f, base): return f / (base[:, None, None] * U * f).sum()
fL = norm(np.exp(0.6 * rng.normal(size=(nr, 2, 2))), A); fR = norm(np.exp(0.6 * rng.normal(size=(nr, 2, 2))), A)
gL = norm(np.exp(0.6 * rng.normal(size=(nw, 2, 2))), B); gR = norm(np.exp(0.6 * rng.normal(size=(nw, 2, 2))), B)
bL = np.array([0.6, 0.4]); bR = np.array([0.35, 0.65]); cL = np.array([0.5, 0.5]); cR = np.array([0.7, 0.3])
phase_q = [np.array([0, 2, 1]), np.array([1, 0, 3]), np.array([3, 3, 0])]
phase_s = [np.array([2, 1]), np.array([0, 3]), np.array([1, 1])]
bits_q = path_bits(phase_q, nr); bits_s = path_bits(phase_s, nw)
block = [0, 0, 1, 1]

def lift_blocks(fs, bits):
    out = []
    for i in range(N):
        out.append(lift(fs[block[i]], [bits[i]])[0])
    return out

Q = lift_blocks([fL, fR], bits_q); S_ = lift_blocks([gL, gR], bits_s)
P = [q * refq for q in Q]; S = [s * refs for s in S_]
beta = np.zeros((N, N)); gamma = np.zeros((N, N))
for i in range(N):
    bq = [bL, bR][block[i]]; bs = [cL, cR][block[i]]
    for j in range(N):
        if i != j:
            k = cls(i, j); beta[i, j] = bq[k] / n_k[k]; gamma[i, j] = bs[k] / n_k[k]

def prof2_rhs(s):
    fs, gs, bqs, bss = s[0:2], s[2:4], s[4:6], s[6:8]
    refA = A[:, None, None] * U; refB = B[:, None, None] * U
    out_f, out_g, out_bq, out_bs = [], [], [], []
    for X in (0, 1):
        Y = 1 - X
        fX, fY, gX, gY = fs[X], fs[Y], gs[X], gs[Y]
        Vq = -(B[None, :] * gX.mean(axis=(1, 2))[None, :] * logrho).sum(axis=1)
        Vs = -(A[:, None] * fX.mean(axis=(1, 2))[:, None] * logrho).sum(axis=0)
        def force(x, y, V, bx, by):
            lx, ly = np.log(x), np.log(y)
            return (2 * lx + V[:, None, None] - bx[0] * P0(ly) - bx[1] * P1(lx)
                    - by[0] * P0(y) / x - bx[1] * P1(x) / x)
        hq = force(fX, fY, Vq, bqs[X], bqs[Y]); hs = force(gX, gY, Vs, bss[X], bss[Y])
        out_f.append(-fX * (hq - (refA * fX * hq).sum())); out_g.append(-gX * (hs - (refB * gX * hs).sum()))
        def costs(x, y, ref):
            lx, ly = np.log(x), np.log(y)
            return np.array([(ref * x * (lx - P0(ly))).sum(), (ref * x * (lx - P1(lx))).sum()])
        Dq = costs(fX, fY, refA); Ds = costs(gX, gY, refB)
        aq = Dq + Tq * np.log(bqs[X] / pi_k); as_ = Ds + Ts * np.log(bss[X] / pi_k)
        out_bq.append(-bqs[X] * (aq - (bqs[X] * aq).sum())); out_bs.append(-bss[X] * (as_ - (bss[X] * as_).sum()))
    return out_f + out_g + out_bq + out_bs

def p2rk4(s, dt, steps):
    for _ in range(steps):
        k1 = prof2_rhs(s); k2 = prof2_rhs([a + dt / 2 * b for a, b in zip(s, k1)])
        k3 = prof2_rhs([a + dt / 2 * b for a, b in zip(s, k2)]); k4 = prof2_rhs([a + dt * b for a, b in zip(s, k3)])
        s = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
    return s

dt, steps = 0.002, 500
fin = rk4(fine_rhs, [P, S, beta, gamma], dt, steps)
pro = p2rk4([fL, fR, gL, gR, bL, bR, cL, cR], dt, steps)
QT = lift_blocks(pro[0:2], bits_q); ST = lift_blocks(pro[2:4], bits_s)
print("two-block family: max|q_i - f^X(u(i))| =", max(np.abs(fin[0][i] / refq - QT[i]).max() for i in range(N)),
      " model:", max(np.abs(fin[1][i] / refs - ST[i]).max() for i in range(N)))
rb = 0.0
for i in range(N):
    for j in range(N):
        if i != j:
            k = cls(i, j)
            rb = max(rb, abs(fin[2][i, j] - pro[4 + block[i]][k] / n_k[k]), abs(fin[3][i, j] - pro[6 + block[i]][k] / n_k[k]))
print("rows residual:", rb, "  block base marginals differ:", np.abs(fL.mean(axis=(1, 2)) - fR.mean(axis=(1, 2))).max())

# root fresh-uniform response in the heterogeneous family: does it depend on the root phase?
Qn = [fin[0][i] / refq for i in range(N)]
root = sum(Qn) / 4
phase_q2 = [phase_q[0] + 1, phase_q[1], phase_q[2]]          # rotate the root phase by pi/2
bits_q2 = path_bits(phase_q2, nr)
Q2 = lift_blocks(pro[0:2], bits_q2)
root2 = sum(Q2) / 4
print("heterogeneous blocks: TV(root response, root-phase-rotated root response) =",
      0.5 * np.abs((root - root2) * refq).sum())
# same quantity for the homogeneous family (f^L = f^R) is zero:
Qh = lift_blocks([pro[0], pro[0]], bits_q); Qh2 = lift_blocks([pro[0], pro[0]], bits_q2)
print("homogeneous check (f^L = f^R): TV =", 0.5 * np.abs((sum(Qh) - sum(Qh2)) / 4 * refq).sum())
