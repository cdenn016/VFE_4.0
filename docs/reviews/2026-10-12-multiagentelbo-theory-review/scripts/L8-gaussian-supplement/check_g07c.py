"""L8 checks for g07 part C: fixed-anchor pair tests, congruence-diagonal closure, boundary classification."""
import numpy as np
from numpy.linalg import inv, eigvalsh
from common import rng, out, rep, note, spd, gl_plus

def schur(L, r, hdn):
    return L[np.ix_(r, r)] - L[np.ix_(r, hdn)] @ inv(L[np.ix_(hdn, hdn)]) @ L[np.ix_(hdn, r)]

K = 3; Q0 = spd(K); W = spd(K); Wp = spd(K); t1, t2 = 0.7, 1.9
M = Q0 + t1 * W + t2 * Wp; Wout = t1 * t2 * W @ inv(M) @ Wp
pred = t1 * t2 * Q0 @ inv(M) @ (W @ inv(Q0) @ Wp - Wp @ inv(Q0) @ W) @ inv(M) @ Q0
rep("g07 pair-test antisymmetry identity", np.abs(Wout - Wout.T - pred).max(), np.abs(Wout).max())
H = gl_plus(K); d1, d2 = rng.random(K), rng.random(K)
Q0 = H @ H.T; W = H @ np.diag(d1) @ H.T; Wp = H @ np.diag(d2) @ H.T
M = Q0 + t1 * W + t2 * Wp; Wout = t1 * t2 * W @ inv(M) @ Wp
v = t1 * t2 * d1 * d2 / (1 + t1 * d1 + t2 * d2)
rep("g07 congruence-diagonal pair output", np.abs(Wout - H @ np.diag(v) @ H.T).max(), np.abs(Wout).max())
W = np.diag([1., 0]); Wp = np.diag([0, 1.]); Q0 = np.array([[2, 1], [1, 2.]])
note(f"g07 anchor-outside-cone coefficient {(W @ inv(Q0 + W + Wp) @ Wp).round(12).tolist()} (claimed [[0,-1/8],[0,0]])")
Hh = np.array([[1, 1], [0, 1.]]); Wa = Hh @ np.diag([1, 2.]) @ Hh.T; Wb = Hh @ np.diag([3, 1.]) @ Hh.T
note(f"g07 W_H commutator {(Wa @ Wb - Wb @ Wa).tolist()} (claimed [[0,-5],[5,0]])")

K = 3; N = 5; H = gl_plus(K, 10.)
def cone():
    return H @ np.diag(rng.random(K) * (rng.random(K) > 0.2)) @ H.T
Lc = np.zeros((N*K, N*K))
for i in range(N):
    Lc[i*K:(i+1)*K, i*K:(i+1)*K] += cone()
for i in range(N):
    for j in range(i+1, N):
        W = cone(); Lc[i*K:(i+1)*K, i*K:(i+1)*K] += W; Lc[j*K:(j+1)*K, j*K:(j+1)*K] += W
        Lc[i*K:(i+1)*K, j*K:(j+1)*K] -= W; Lc[j*K:(j+1)*K, i*K:(i+1)*K] -= W
Se = schur(Lc, list(range(6)), list(range(6, 15)))
Hi = inv(H); worst = 0
for a in range(2):
    for b in range(2):
        Dg = Hi @ Se[a*K:(a+1)*K, b*K:(b+1)*K] @ Hi.T
        worst = max(worst, np.abs(Dg - np.diag(np.diag(Dg))).max())
note(f"g07 congruence-diagonal Kron closure: max off-diagonal of H^-1 block H^-T = {worst:.3e} (block scale {np.abs(Se).max():.2e})")

K = 2
def lap(nv, edges):
    L = np.zeros((nv*K, nv*K))
    for (i, j, W) in edges:
        L[i*K:(i+1)*K, i*K:(i+1)*K] += W; L[j*K:(j+1)*K, j*K:(j+1)*K] += W
        L[i*K:(i+1)*K, j*K:(j+1)*K] -= W; L[j*K:(j+1)*K, i*K:(i+1)*K] -= W
    return L
E2 = [(0, 2, spd(K)), (2, 3, spd(K)), (3, 4, spd(K)), (2, 4, spd(K)), (4, 1, spd(K)), (3, 1, spd(K)), (0, 3, spd(K))]
S2 = schur(lap(5, E2), list(range(4)), list(range(4, 10)))
note(f"g07 two-terminal loopy component: off-block asym {np.abs(S2[:2,2:]-S2[:2,2:].T).max():.3e}, row-sum {np.abs(S2[:2,:2]+S2[:2,2:]).max():.3e}, W_Z eigs {eigvalsh(-S2[:2,2:]).round(4).tolist()}")
E3 = [(0, 3, spd(K)), (1, 3, spd(K)), (2, 4, spd(K)), (3, 4, spd(K))]
S3 = schur(lap(5, E3), list(range(6)), list(range(6, 10)))
note(f"g07 three-terminal component: off-block 0-1 asym {np.abs(S3[:2,2:4]-S3[:2,2:4].T).max():.3e} (nonzero expected)")
print("\n".join(out))
