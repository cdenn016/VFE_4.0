"""D24-LDYN script 7: in-degree-one graphs (two-cycle, directed 3-ring), K=2, anisotropic self
precisions, random codes of rank 1 or 2, nonzero operating contents, grounding, lambda = 0.
Claim tested: the law-level per-agent mean flow is stable (contraction in the norms
||dy_i||_{Sigma_i^{-1}}), while the plug-in flow need not be. Also check the contraction factor."""
import time
import numpy as np
from tangent_lib import Model
rng = np.random.default_rng(23)
def rand_P(K=2):
    th = rng.uniform(0, np.pi)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Rm @ np.diag(10 ** rng.uniform(-2, 2, size=K)) @ Rm.T
t0 = time.time()
stats = {}
for name, N, edges in [("two-cycle", 2, [(1, 0), (0, 1)]), ("3-ring", 3, [(1, 0), (2, 1), (0, 2)])]:
    worst_law, worst_plug, n_unst_plug, cnt, worst_rho = -np.inf, -np.inf, 0, 0, 0.0
    while time.time() - t0 < (12 if name == "two-cycle" else 25) and cnt < 400:
        cnt += 1
        P = [rand_P() for _ in range(N)]
        M = Model(N, 2, edges, rng, sigma2=10 ** rng.uniform(-1, 1), lam=0.0, P=P,
                  gscale=10 ** rng.uniform(-2, 1))
        for k in range(M.E):
            n = M.T[k].shape[0]
            M.R[k] = np.diag(10 ** rng.uniform(-3, 1, size=n))
        H, b, A = M.per_agent_H_b('plug')
        mp = np.linalg.eigvals(-np.linalg.solve(A, H)).real.max()
        worst_plug = max(worst_plug, mp)
        n_unst_plug += mp > 1e-9
        Sg, Vg, it = M.law_level_fixed_point(0.0, tol=1e-12, maxit=3000)
        Hl, bl, Al = M.per_agent_H_b('law', Sg, Vg)
        Xit = np.eye(M.n) - np.linalg.solve(Al, Hl)        # best-response iteration matrix A^{-1}C
        worst_rho = max(worst_rho, np.abs(np.linalg.eigvals(Xit)).max())
        worst_law = max(worst_law, np.linalg.eigvals(-np.linalg.solve(Al, Hl)).real.max())
    print(f"{name}: {cnt} instances; plug-in unstable fraction {n_unst_plug/cnt:.3f} (max Re {worst_plug:+.4f}); "
          f"law-level max Re {worst_law:+.5f}, max spectral radius of A^-1 C {worst_rho:.5f}")
print("elapsed", round(time.time() - t0, 1), "s")
