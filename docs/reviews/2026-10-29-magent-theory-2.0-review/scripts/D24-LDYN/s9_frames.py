"""D24-LDYN script 9: passive frame changes and the records (ch18 line 33).
Exact record T theta^{-1} y_i - T y_j is invariant under y_i -> g_i y_i, y_j -> g_j y_j,
T -> T g_j^{-1}, theta -> g_i theta g_j^{-1}. The tangent record T(y_i - y_j - omega c_i) with
c_i -> g_i c_i is invariant for all y_i only when T(g_j^{-1} g_i - I) = 0 (e.g. g_i = g_j)."""
import numpy as np
from scipy.linalg import expm
rng = np.random.default_rng(5)
K = 3
T = rng.normal(size=(2, K))
w = 0.1 * rng.normal(size=(K, K))
yi, yj, c = rng.normal(size=K), rng.normal(size=K), rng.normal(size=K)
gi, gj = rng.normal(size=(K, K)) + 2 * np.eye(K), rng.normal(size=(K, K)) + 2 * np.eye(K)
th = expm(w)
exact = T @ np.linalg.solve(th, yi) - T @ yj
Tn, thn = T @ np.linalg.inv(gj), gi @ th @ np.linalg.inv(gj)
exact_new = Tn @ np.linalg.solve(thn, gi @ yi) - Tn @ (gj @ yj)
print("exact record, independent frames: |change| =", np.abs(exact - exact_new).max())
tang = T @ (yi - yj - w @ c)
g = gi
tang_common = (T @ np.linalg.inv(g)) @ (g @ yi - g @ yj - (g @ w @ np.linalg.inv(g)) @ (g @ c))
print("tangent record, common frame g, omega -> g omega g^{-1}: |change| =", np.abs(tang - tang_common).max())
# independent frames: best omega' (least squares over all y_i) cannot restore invariance
# record difference as a function of y_i: T(g_j^{-1} g_i - I) y_i must vanish for every y_i
print("independent frames: |T (g_j^{-1} g_i - I)| =", np.abs(T @ (np.linalg.solve(gj, gi) - np.eye(K))).max(), "(nonzero => no omega' works for all y_i)")
# anchor: ||g w g^{-1}||_F vs ||w||_F
print("Frobenius anchor under conjugation (non-orthogonal g): ||w||, ||g w g^-1|| =", np.linalg.norm(w), np.linalg.norm(g @ w @ np.linalg.inv(g)))
