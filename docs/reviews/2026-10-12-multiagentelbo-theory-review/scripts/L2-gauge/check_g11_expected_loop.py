"""MC check of prop:glink-expected-loop (iii) at flat means with a non-identity fiber metric, K=2."""
import numpy as np
from scipy.linalg import expm, sqrtm
rng = np.random.default_rng(3)
K = 2; A = rng.standard_normal((K, K)); G = A@A.T + 0.5*np.eye(K); Gh = np.real(sqrtm(G)); Ghi = np.linalg.inv(Gh)
# G-orthonormal basis of gl(K): H -> Gh H Ghi is an isometry onto Frobenius, so E_k = Ghi e_k Gh
basis = [Ghi @ np.outer(np.eye(K)[a], np.eye(K)[b]) @ Gh for a in range(K) for b in range(K)]
ip = lambda X, Y: np.trace(np.linalg.solve(G, X.T) @ G @ Y)
print("basis orthonormality err:", max(abs(ip(X, Y) - (1.0 if i == j else 0.0)) for i, X in enumerate(basis) for j, Y in enumerate(basis)))
s = 0.03; n = 100000; acc = 0.0
for _ in range(n):
    zs = [s*sum(c*E for c, E in zip(rng.standard_normal(4), basis)) for _ in range(3)]
    Th = expm(zs[0]) @ expm(zs[1]) @ expm(zs[2]) - np.eye(K)
    acc += 0.5*ip(Th, Th)
pred = 0.5*3*4*s**2
print("MC E[Q] =", acc/n, " predicted 1/2 sum Tr Lambda^-1 =", pred, " rel diff =", (acc/n - pred)/pred, " (O(s^2) bias + MC error expected)")
