"""D24-LDYN script 10: sufficient condition for law-level per-agent stability (lambda = 0, c = 0):
max_j sum_{e in in(j)} s_e/(1+s_e) <= 1, with s_e = lambda_max(R_e^{-1/2} T_e Sigma_i T_e^T R_e^{-1/2}).
Check on random complete exchanges (in-degree 2): every instance meeting the condition is stable;
report the condition value for unstable instances."""
import time
import numpy as np
rng = np.random.default_rng(11)
N, K = 3, 2
edges = [(j, i) for i in range(N) for j in range(N) if j - i]
def rand_P():
    th = rng.uniform(0, np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Rm @ np.diag(10 ** rng.uniform(-2, 2, size=2)) @ Rm.T
t0 = time.time()
n_cond = n_cond_unstable = n_unst = 0
cond_of_unstable = []
for n in range(1500):
    P = [rand_P() for _ in range(N)]
    T = []
    for k in range(len(edges)):
        th = rng.uniform(0, np.pi)
        T.append(np.array([[np.cos(th), np.sin(th)]]))
    R = [np.array([[10 ** rng.uniform(-3, 1)]]) for _ in edges]
    Sig = [np.zeros((K, K)) for _ in range(N)]
    for it in range(400):
        A = [P[i].copy() for i in range(N)]
        for k, (j, i) in enumerate(edges):
            A[i] += T[k].T @ np.linalg.inv(R[k] + T[k] @ Sig[j] @ T[k].T) @ T[k]
        S2 = [np.linalg.inv(a) for a in A]
        err = max(np.abs(S2[i] - Sig[i]).max() for i in range(N))
        Sig = S2
        if err < 1e-11:
            break
    Big = np.zeros((N * K, N * K))
    for k, (j, i) in enumerate(edges):
        Big[i*K:(i+1)*K, j*K:(j+1)*K] += np.linalg.solve(A[i], T[k].T @ np.linalg.inv(R[k] + T[k] @ Sig[j] @ T[k].T) @ T[k])
    mr = np.linalg.eigvals(-np.eye(N * K) + Big).real.max()
    theta = np.zeros(N)
    for k, (j, i) in enumerate(edges):
        s = (T[k] @ Sig[j] @ T[k].T)[0, 0] / R[k][0, 0]
        theta[i] += s / (1 + s)
    cond = theta.max()
    if cond <= 1:
        n_cond += 1
        n_cond_unstable += mr > 1e-12
    if mr > 1e-12:
        n_unst += 1
        cond_of_unstable.append(cond)
print(f"{1500} instances; condition met in {n_cond}, of which unstable: {n_cond_unstable}")
print(f"unstable instances: {n_unst}; their condition values (should all exceed 1): min {min(cond_of_unstable):.4f}, max {max(cond_of_unstable):.4f}")
print("elapsed", round(time.time() - t0, 1), "s")
