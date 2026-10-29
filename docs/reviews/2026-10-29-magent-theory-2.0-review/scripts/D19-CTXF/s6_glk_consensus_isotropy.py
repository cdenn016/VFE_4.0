r"""D19-CTXF s6: infinitesimal isotropy of GL(K) acting on collectives of Gaussian laws.

GL(K) acts on N(mu, Sigma) by sample pushforward x -> A x: (mu, Sigma) -> (A mu, A Sigma A^T), an isometry
of the Fisher metric.  Fundamental tangent: J_p xi = (xi mu, xi Sigma + Sigma xi^T) per agent.  The edge-cost
group Hessian at coincident endpoints is the locked inertia k = J^T G J (ctx:f:eq:normal), so
ker k = ker J = Lie algebra of the collective stabilizer.  We count dim ker J for
  (a) n agents in consensus (all equal to one generic law), K = 2, 3, 4;
  (b) n generic distinct agents, K = 2, 3, 4.
A positive kernel means ctx:f:ass:branch fails at every endpoint pair on that stratum.
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(7)


def Jmatrix(agents, K):
    cols = []
    for a in range(K):
        for b in range(K):
            xi = np.zeros((K, K)); xi[a, b] = 1.0
            col = []
            for mu, S in agents:
                col.append(xi @ mu)
                T = xi @ S + S @ xi.T
                col.append(T[np.triu_indices(K)])
            cols.append(np.concatenate(col))
    return np.array(cols).T


def generic_law(K):
    A = rng.normal(size=(K, K))
    return rng.normal(size=K), A @ A.T + 0.5 * np.eye(K)


def main():
    lines = []
    for K in (2, 3, 4):
        for n in (1, 2, 5):
            law = generic_law(K)
            Jc = Jmatrix([law] * n, K)
            Jg = Jmatrix([generic_law(K) for _ in range(n)], K)
            sv_c = np.linalg.svd(Jc, compute_uv=False)
            sv_g = np.linalg.svd(Jg, compute_uv=False)
            kc = K * K - int(np.sum(sv_c > 1e-10 * sv_c[0]))
            kg = K * K - int(np.sum(sv_g > 1e-10 * sv_g[0]))
            lines.append(f"K={K} n={n}: dim gl(K) = {K*K:2d}; dim ker J at consensus = {kc} "
                         f"(predicted (K-1)(K-2)/2 = {(K-1)*(K-2)//2}); generic distinct agents: {kg}; "
                         f"smallest nonzero singular value at consensus = {sv_c[sv_c > 1e-10*sv_c[0]].min():.3e}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
