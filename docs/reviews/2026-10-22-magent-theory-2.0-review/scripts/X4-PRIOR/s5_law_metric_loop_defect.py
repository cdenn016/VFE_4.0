r"""X4-PRIOR: a loop defect with its fiber metric supplied by the base-vertex law.

Q(Theta_p; M_i) = tr((Theta_p - I)^T M_i^{-1} (Theta_p - I) M_i) = || M_i^{-1/2} (Theta_p - I) M_i^{1/2} ||_F^2,
where M_i is the reference covariance of the law at the base vertex of the closed walk p.
Checks: invariance under passive vertex changes (Theta_p -> g Theta_p g^{-1}, M_i -> g M_i g^T) and under
common reference changes (same form), nonnegativity, vanishing exactly at Theta_p = I, and nonzero value on a
unipotent holonomy (which every continuous class function on GL(K) fails to separate from I).
Uses the 2.0 moment-family loop of eq. int-nongaussian-loop as one holonomy.
"""
import numpy as np

rng = np.random.default_rng(5)


def Q(Th, M):
    A = Th - np.eye(len(M))
    return float(np.trace(A.T @ np.linalg.inv(M) @ A @ M))


K = 3
worst = 0.0
for trial in range(200):
    Th = rng.normal(size=(K, K)) + 2 * np.eye(K)
    B = rng.normal(size=(K, K)); M = B @ B.T + 0.1 * np.eye(K)
    g = rng.normal(size=(K, K)) + 3 * np.eye(K)
    q0 = Q(Th, M)
    q1 = Q(g @ Th @ np.linalg.inv(g), g @ M @ g.T)
    worst = max(worst, abs(q1 - q0) / max(1.0, abs(q0)))
    assert q0 >= -1e-12
print("max relative passive/reference invariance error:", worst)
print("Q(I) =", Q(np.eye(K), np.eye(K)))
U = np.array([[1.0, 1.0], [0.0, 1.0]])
print("unipotent holonomy, M=I: Q =", Q(U, np.eye(2)), "; M=diag(4,1): Q =", Q(U, np.diag([4.0, 1.0])))
L12 = np.array([[1.0, 1.0], [0.0, 1.0]]); L21 = np.array([[1.0, 0.0], [1.0, 1.0]])
loop = L12 @ L21
print("2.0 moment-family two-cycle holonomy", loop.tolist(), " defect at base vertex with M=I:", Q(loop, np.eye(2)))
