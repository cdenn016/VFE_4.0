"""R-BND: contrast decay rate 3+2/(N-1) and the Jensen gap condition for the profile flow."""
import itertools
import numpy as np

def P(k, d):
    U = list(itertools.product([-1, 1], repeat=d))
    idx = {u: n for n, u in enumerate(U)}
    M = np.zeros((2**d, 2**d))
    for u in U:
        tails = list(itertools.product([-1, 1], repeat=d - k - 1))
        for t in tails:
            v = u[:k] + (-u[k],) + t
            M[idx[u], idx[v]] += 1.0 / len(tails)
    return M

rng = np.random.default_rng(0)
for d in (2, 3, 4):
    N = 2**d
    pi = np.array([2**(d - k - 1) / (N - 1) for k in range(d)])
    L = 3 * np.eye(N) - 2 * sum(pi[k] * P(k, d) for k in range(d))
    ones = np.ones(N) / np.sqrt(N)
    Q = np.eye(N) - np.outer(ones, ones)
    ev = np.linalg.eigvalsh(Q @ L @ Q)
    ev = np.sort(ev)[1:]  # drop the constant mode
    print(f"d={d} N={N} contrast eigenvalues min={ev.min():.12f} max={ev.max():.12f} predicted={3+2/(N-1):.12f}")
    # random rows: every Walsh rate >= 1
    b = rng.dirichlet(np.ones(d))
    Lb = 3 * np.eye(N) - 2 * sum(b[k] * P(k, d) for k in range(d))
    evb = np.sort(np.linalg.eigvals(Q @ Lb @ Q).real)[1:]
    print(f"   random rows min rate {evb.min():.6f} (>=1 expected)")
    # Jensen gap: int f [log P_k f - P_k log f]
    f = np.exp(rng.normal(size=N)); f /= f.mean()
    for k in range(d):
        Pk = P(k, d)
        gap = np.mean(f * (np.log(Pk @ f) - Pk @ np.log(f)))
        print(f"   Jensen gap k={k}: {gap:.3e}")
    # a profile independent of bits k+2..d has zero gap for k
    k = 0
    U = list(itertools.product([-1, 1], repeat=d))
    g = np.array([1 + 0.4 * u[0] for u in U]); g /= g.mean()
    gap = np.mean(g * (np.log(P(k, d) @ g) - P(k, d) @ np.log(g)))
    print(f"   profile depending on bit 1 only, k=0 gap: {gap:.3e} (zero iff no dependence on bits k+2..d)")
