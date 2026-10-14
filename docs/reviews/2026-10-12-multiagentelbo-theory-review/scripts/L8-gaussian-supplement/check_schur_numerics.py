"""L8 numerical-hazard probe: Kron/Schur elimination and closure tests under ill-conditioning (float64 vs mpmath)."""
import numpy as np
import mpmath as mp
from numpy.linalg import inv, cholesky
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from common import rng, spd, gl_plus

mp.mp.dps = 60
K = 2
print("Exactly closed congruence-diagonal star (3 retained, 1 hidden); closure test = symmetry of off-diagonal Schur blocks")
print("cond(H)   cond(M)     |antisym| inv-form   |antisym| chol-form   block scale   mp60 Schur rel.err (inv / chol)")
for ch in [1e0, 1e2, 1e4, 1e6]:
    H = gl_plus(K, ch)
    diag = lambda: H @ np.diag(rng.random(K) + 0.05) @ H.T
    Ws = [diag() for _ in range(3)]; A = [diag() for _ in range(4)]
    n = 4; L = np.zeros((n * K, n * K))
    for i in range(n): L[i*K:(i+1)*K, i*K:(i+1)*K] += A[i]
    for i, W in enumerate(Ws):
        j = 3
        L[i*K:(i+1)*K, i*K:(i+1)*K] += W; L[j*K:(j+1)*K, j*K:(j+1)*K] += W
        L[i*K:(i+1)*K, j*K:(j+1)*K] -= W; L[j*K:(j+1)*K, i*K:(i+1)*K] -= W
    r = slice(0, 6); h = slice(6, 8); M = L[h, h]
    S_inv = L[r, r] - L[r, h] @ inv(M) @ L[h, r]
    c = cho_factor(M); S_ch = L[r, r] - L[r, h] @ cho_solve(c, L[h, r])
    asym = lambda S: max(np.abs(S[0:2, 2:4] - S[0:2, 2:4].T).max(), np.abs(S[0:2, 4:6] - S[0:2, 4:6].T).max())
    Lm = mp.matrix(L.tolist()); Smp = Lm[0:6, 0:6] - Lm[0:6, 6:8] * (Lm[6:8, 6:8] ** -1) * Lm[6:8, 0:6]
    Smp = np.array(Smp.tolist(), dtype=float)
    e1 = np.abs(S_inv - Smp).max() / np.abs(Smp).max(); e2 = np.abs(S_ch - Smp).max() / np.abs(Smp).max()
    print(f"{ch:7.0e}  {np.linalg.cond(M):9.2e}   {asym(S_inv):12.3e}       {asym(S_ch):12.3e}       {np.abs(S_inv).max():9.2e}     {e1:.2e} / {e2:.2e}")
