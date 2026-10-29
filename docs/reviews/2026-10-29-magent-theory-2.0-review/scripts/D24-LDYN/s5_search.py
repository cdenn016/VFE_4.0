"""D24-LDYN script 5: bounded random search for unstable per-agent mean flows.
(a) content-only (c=0) per-agent flows, K=2, three agents in complete exchange (in-degree 2),
rank-one codes, anisotropic self precisions; plug-in and law-level (at the covariance fixed point).
(b) K=1 full tangent model with dictionaries, loops and grounding (plug-in and law-level)."""
import time
import numpy as np
from tangent_lib import Model, spd

t0 = time.time()
rng = np.random.default_rng(11)
N, K = 3, 2
edges = [(j, i) for i in range(N) for j in range(N) if j - i]
def rand_P():
    th = rng.uniform(0, np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Rm @ np.diag(10 ** rng.uniform(-2, 2, size=2)) @ Rm.T
def flow_maxre(P, T, R, law):
    Sig = [np.zeros((K, K)) for _ in range(N)]
    for it in range(400):
        A = [P[i].copy() for i in range(N)]
        for k, (j, i) in enumerate(edges):
            Rt = R[k] + (T[k] @ Sig[j] @ T[k].T if law else 0)
            A[i] += T[k].T @ np.linalg.inv(Rt) @ T[k]
        S2 = [np.linalg.inv(a) for a in A]
        err = max(np.abs(S2[i] - Sig[i]).max() for i in range(N))
        Sig = S2
        if not law or err < 1e-11:
            break
    Big = np.zeros((N * K, N * K))
    for k, (j, i) in enumerate(edges):
        Rt = R[k] + (T[k] @ Sig[j] @ T[k].T if law else 0)
        Big[i*K:(i+1)*K, j*K:(j+1)*K] += np.linalg.solve(A[i], T[k].T @ np.linalg.inv(Rt) @ T[k])
    ev = np.linalg.eigvals(-np.eye(N * K) + Big)
    return ev.real.max()
res = {True: [], False: []}
worst = {True: (-np.inf, None), False: (-np.inf, None)}
ntr = 0
while time.time() - t0 < 14 and ntr < 1500:
    ntr += 1
    P = [rand_P() for _ in range(N)]
    T = []
    for k in range(len(edges)):
        th = rng.uniform(0, np.pi)
        T.append(np.array([[np.cos(th), np.sin(th)]]))
    R = [np.array([[10 ** rng.uniform(-3, 1)]]) for _ in edges]
    for law in (False, True):
        mr = flow_maxre(P, T, R, law)
        res[law].append(mr)
        if mr > worst[law][0]:
            worst[law] = (mr, (P, T, R))
for law in (False, True):
    arr = np.array(res[law])
    print(f"(a) {'law-level' if law else 'plug-in'}: {len(arr)} instances, fraction unstable {np.mean(arr > 1e-9):.3f}, max Re {arr.max():+.4f}")
print("   elapsed", round(time.time() - t0, 1), "s")

# (b) K=1, full model with dictionaries, loops and grounding
t1 = time.time()
worst_b = {'plug': -np.inf, 'law': -np.inf}
cnt = 0
edges1 = [(1, 0), (0, 1), (2, 1), (0, 2), (2, 0), (1, 2)]
while time.time() - t1 < 12 and cnt < 300:
    cnt += 1
    M = Model(3, 1, edges1, rng, sigma2=10 ** rng.uniform(-1, 1), lam=10 ** rng.uniform(-1, 1.5),
              gscale=10 ** rng.uniform(-2, 1))
    for k in range(M.E):
        M.R[k] = np.array([[10 ** rng.uniform(-2, 1)]])
    H, b, A = M.per_agent_H_b('plug')
    worst_b['plug'] = max(worst_b['plug'], np.linalg.eigvals(-np.linalg.solve(A, H)).real.max())
    Sg, Vg, it = M.law_level_fixed_point(0.0, tol=1e-11, maxit=2000)
    Hl, bl, Al = M.per_agent_H_b('law', Sg, Vg)
    worst_b['law'] = max(worst_b['law'], np.linalg.eigvals(-np.linalg.solve(Al, Hl)).real.max())
print(f"(b) K=1 with dictionaries, loops, grounding: {cnt} instances; max Re plug-in {worst_b['plug']:+.4f}, law-level {worst_b['law']:+.4f}")
print("   elapsed", round(time.time() - t1, 1), "s")
