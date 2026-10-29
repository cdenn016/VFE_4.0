"""D24-LDYN script 6: reproduce the worst law-level instance of s5 part (a) (same seed and draw
sequence), then verify it in the full tangent model (tangent_lib.Model, operating contents zero so
the dictionaries decouple from the records): covariance fixed point from two starts, spectrum of the
law-level per-agent mean flow, RK4 integration, and the global flow for contrast."""
import numpy as np
from tangent_lib import Model
rng = np.random.default_rng(11)
N, K = 3, 2
edges = [(j, i) for i in range(N) for j in range(N) if j - i]
def rand_P():
    th = rng.uniform(0, np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Rm @ np.diag(10 ** rng.uniform(-2, 2, size=2)) @ Rm.T
def law_maxre(P, T, R):
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
    return np.linalg.eigvals(-np.eye(N * K) + Big).real.max()
best = (-np.inf, None)
for n in range(1500):
    P = [rand_P() for _ in range(N)]
    T = []
    for k in range(len(edges)):
        th = rng.uniform(0, np.pi)
        T.append(np.array([[np.cos(th), np.sin(th)]]))
    R = [np.array([[10 ** rng.uniform(-3, 1)]]) for _ in edges]
    mr = law_maxre(P, T, R)
    if mr > best[0]:
        best = (mr, (P, T, R))
mr, (P, T, R) = best
print("worst law-level instance (content-only search): max Re =", mr)
np.set_printoptions(precision=5, suppress=False)
for i in range(N):
    w, U = np.linalg.eigh(P[i])
    print(f"  P_{i} eigenvalues {w}, first eigenvector angle {np.degrees(np.arctan2(U[1,0], U[0,0])):.2f} deg")
for k, (j, i) in enumerate(edges):
    print(f"  edge {j}->{i}: code angle {np.degrees(np.arctan2(T[k][0,1], T[k][0,0])):.2f} deg, R = {R[k][0,0]:.5g}")
for lam, sig2 in [(0.0, 1.0), (1.0, 1.0), (5.0, 0.3)]:
    M = Model(N, K, edges, np.random.default_rng(5), sigma2=sig2, lam=lam, P=P, Tlist=T, Rlist=R,
              cvec=[np.zeros(K) for _ in range(N)], grounded=True)
    S_lo, V_lo, it_lo = M.law_level_fixed_point(0.0, tol=1e-13)
    S_hi, V_hi, it_hi = M.law_level_fixed_point(30.0, tol=1e-13)
    dfp = max(max(np.abs(a - b).max() for a, b in zip(S_lo, S_hi)), max(np.abs(a - b).max() for a, b in zip(V_lo, V_hi)))
    Hl, bl, Al = M.per_agent_H_b('law', S_lo, V_lo)
    ev = np.linalg.eigvals(-np.linalg.solve(Al, Hl))
    J, eta = M.global_J_eta()
    D = np.zeros_like(J)
    for blk in M.blocks:
        D[np.ix_(blk, blk)] = J[np.ix_(blk, blk)]
    evg = np.linalg.eigvals(-np.linalg.solve(D, J))
    mu_star = np.linalg.solve(Hl, bl)
    Fl = lambda mu: -np.linalg.solve(Al, Hl @ mu - bl)
    mu = mu_star + 1e-6 * np.random.default_rng(1).normal(size=M.n)
    dt = 0.02
    for step in range(5000):
        k1 = Fl(mu); k2 = Fl(mu + dt / 2 * k1); k3 = Fl(mu + dt / 2 * k2); k4 = Fl(mu + dt * k3)
        mu = mu + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    print(f"lam={lam} sigma2={sig2}: loops {len(M.loops)}; covariance fixed point from two starts differs by {dfp:.1e} "
          f"(iterations {it_lo}, {it_hi}); law-level per-agent max Re {ev.real.max():+.5f}; global max Re {evg.real.max():+.4f}; "
          f"RK4 growth |mu(100)-mu*|/1e-6 = {np.abs(mu - mu_star).max() / 1e-6:.3g}")
