r"""L02-GEOM s08: the collective minimizing connection near consensus.

GL(K) acting linearly on n=2 Gaussian laws N(mu_i, Sigma_i) on R^K (K=3), Fisher metric
g(dmu,dSigma) = dmu^T Sigma^{-1} dmu + 1/2 tr(Sigma^{-1} dSigma Sigma^{-1} dSigma), weights 1.
At exact agreement (mu_1,Sigma_1) = (mu_2,Sigma_2) the infinitesimal stabilizer grows (orbit
dimension drops), so the constant-rank hypothesis of geo:collective fails there. Measure the
smallest eigenvalue of the locked inertia k and the norm of A^F for a fixed random first jet as the
two laws approach each other at distance eps.
"""
import numpy as np
rng = np.random.default_rng(4)
K = 3
iu = np.triu_indices(K)
def sym_from(v):
    S = np.zeros((K, K)); S[iu] = v; return S + S.T - np.diag(np.diag(S))
def fisher_block(mu, Sig, v1, v2):
    Si = np.linalg.inv(Sig)
    dm1, dS1 = v1[:K], sym_from(v1[K:]); dm2, dS2 = v2[:K], sym_from(v2[K:])
    return dm1 @ Si @ dm2 + 0.5 * np.trace(Si @ dS1 @ Si @ dS2)
dimL = K + K * (K + 1) // 2
def metric(laws):
    n = len(laws); G = np.zeros((n * dimL, n * dimL))
    E = np.eye(dimL)
    for i, (mu, Sig) in enumerate(laws):
        for a in range(dimL):
            for b in range(dimL):
                G[i * dimL + a, i * dimL + b] = fisher_block(mu, Sig, E[a], E[b])
    return G
def Jmat(laws):
    cols = []
    for i in range(K):
        for j in range(K):
            Eij = np.zeros((K, K)); Eij[i, j] = 1
            col = []
            for mu, Sig in laws:
                col.append(Eij @ mu)
                col.append((Eij @ Sig + Sig @ Eij.T)[iu])
            cols.append(np.concatenate(col))
    return np.array(cols).T
mu0 = rng.normal(size=K); L = rng.normal(size=(K, K)); Sig0 = L @ L.T + K * np.eye(K)
dmu = rng.normal(size=K); M = rng.normal(size=(K, K)); dSig = (M + M.T) / 2
jet = rng.normal(size=2 * dimL)
print('orbit dimension at exact agreement:', np.linalg.matrix_rank(Jmat([(mu0, Sig0), (mu0, Sig0)]), tol=1e-9),
      '; at distinct laws:', np.linalg.matrix_rank(Jmat([(mu0, Sig0), (mu0 + dmu, Sig0 + dSig)]), tol=1e-9))
for eps in (1e-1, 1e-2, 1e-3, 1e-4):
    laws = [(mu0, Sig0), (mu0 + eps * dmu, Sig0 + eps * dSig)]
    G = metric(laws); J = Jmat(laws)
    k = J.T @ G @ J
    ev = np.linalg.eigvalsh(k)
    AF = -np.linalg.lstsq(k, J.T @ G @ jet, rcond=None)[0]
    r = jet + J @ AF
    print('eps=%.0e: smallest nonzero-branch eig(k)=%.3e, |A^F|=%.3e, h^min=%.4f, |J^T G r|=%.1e'
          % (eps, ev[0], np.linalg.norm(AF), r @ G @ r, np.abs(J.T @ G @ r).max()))
