"""L8 checks for g07 part B: Kron reduction, pair tests, obstruction, nonflat series."""
import numpy as np
import sympy as sp
from numpy.linalg import inv, slogdet, eigvalsh, det
from common import rng, out, rep, note, spd, gl_plus

def schur(L, r, hdn):
    return L[np.ix_(r, r)] - L[np.ix_(r, hdn)] @ inv(L[np.ix_(hdn, hdn)]) @ L[np.ix_(hdn, r)]

def add(L, i, j, W):
    L[2*i:2*i+2, 2*i:2*i+2] += W; L[2*j:2*j+2, 2*j:2*j+2] += W
    L[2*i:2*i+2, 2*j:2*j+2] -= W; L[2*j:2*j+2, 2*i:2*i+2] -= W

# prop:cg-kron-leaves-family (exact)
I2 = sp.eye(2); W13 = sp.Matrix([[1, 0], [0, 2]]); W23 = sp.Matrix([[2, 1], [1, 2]])
L = sp.zeros(6, 6)
for i in range(3): L[2*i:2*i+2, 2*i:2*i+2] += I2
add(L, 0, 2, W13); add(L, 1, 2, W23)
minors = [L[:k, :k].det() for k in range(1, 7)]
Wk = W13 * L[4:6, 4:6].inv() * W23
note(f"g07 Kron witness: leading minors {minors} (claimed 2,6,18,48,102,233); 19*W12^Kron = {Wk*19}")

# three-terminal star defect
W14 = sp.Matrix([[1, 0], [0, 2]]); W24 = sp.eye(2); W34 = sp.Matrix([[2, 1], [1, 2]]); Mst = W14 + W24 + W34
Wk12 = W14 * Mst.inv() * W24
note(f"g07 three-terminal 19*W14 M^-1 W24 = {Wk12*19} (claimed [[5,-1],[-2,8]]); 19*antisym = {(Wk12-Wk12.T)*19}")

# rational obstruction
Lst = sp.zeros(8, 8)
for i, W in enumerate([W14, W24, W34]): add(Lst, i, 3, W)
Lr = Lst[:6, :6] - Lst[:6, 6:] * Lst[6:, 6:].inv() * Lst[6:, :6]
note(f"g07 obstruction: diag traces {[Lr[2*i:2*i+2, 2*i:2*i+2].trace() for i in range(3)]} (claimed 36/19,29/19,39/19); Tr L_st = {Lr.trace()}")
t = sp.Rational(1, 104); Jm = sp.Matrix([[0, 1], [-1, 0]])
Y = sp.zeros(6, 6)
for i in range(3):
    for j in range(3):
        Y[2*i:2*i+2, 2*j:2*j+2] = (1 + t**2) * I2 if i == j else I2
Y[0:2, 2:4] = I2 + t * Jm; Y[2:4, 0:2] = I2 - t * Jm
TrYL = (Y * Lr).trace()
note(f"g07 obstruction Tr(Y L_st) = {TrYL} (claimed -1/1976)")
mins = []
for a in range(3):
    for b in range(a + 1, 3):
        idx = [2*a, 2*a+1, 2*b, 2*b+1]
        mins.append(min(eigvalsh(np.array(Y.extract(idx, idx), dtype=float))))
note(f"g07 obstruction: min eig of 2-node principal submatrices of Y {min(mins):.3e}; min eig of Y {min(eigvalsh(np.array(Y, dtype=float))):.3e}")
delta = sp.Rational(1, 20000)
note(f"g07 obstruction Tr(Y (L_st + delta I)) = {TrYL + delta * Y.trace()} (negative claimed)")
Lfull = Lst + sp.diag(delta, delta, delta, delta, delta, delta, 0, 0)
note(f"g07 obstruction full precision min eig {min(eigvalsh(np.array(Lfull, dtype=float))):.3e}")
print("\n".join(out))
