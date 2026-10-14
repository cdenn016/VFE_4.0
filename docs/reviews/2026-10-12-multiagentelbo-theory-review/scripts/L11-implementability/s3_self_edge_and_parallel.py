"""L11 check s3: two implementation hazards the theory leaves to the implementer.

(a) Loop copies (self edges). Coarse depths carry loop copies e: I -> I whose link is a
    based holonomy (def:full-graph-coarse-links). The product-matter expected record cost
    of MultiAgentELBO costs_from_maps assumes receiver != source (residual covariance
    Sigma_i + M Sigma_j M^T). For a loop copy the residual is (I - M) z_i, with covariance
    (I-M) Sigma (I-M)^T. CoupledModel rejects self sources, so the code is safe as is;
    the check measures the error a naive extension would make.
(b) Parallel block updates. corollary prop:elbo-exact-e-coordinate(iv) and the remark after
    it cover sequential exact or damped block updates; simultaneous replacement of
    several blocks is not covered. For a Gaussian target and product recognition, the
    simultaneous (Jacobi) exact update can increase the collective free energy.
"""
import sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\MultiAgentELBO\src")
from multiagent_elbo.realizations.gaussian.coupled_learning import costs_from_maps

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(5)
K = 2
mu = torch.tensor(rng.normal(size=(1, K)))
B = rng.normal(size=(K, K)); Sigma = torch.tensor(B @ B.T + 0.5*np.eye(K))[None]
M = torch.tensor(np.eye(K) + 0.3*rng.normal(size=(K, K)))
H = torch.eye(K)[None]
fake = SimpleNamespace(H=H, sources=torch.tensor([[0]]))
naive, _ = costs_from_maps(mu, Sigma, M[None, None], torch.zeros(1, 1, K), fake)
R = torch.eye(K) - M
exact = 0.5*(R @ mu[0]) @ (R @ mu[0]) + 0.5*torch.trace(R @ Sigma[0] @ R.T)
z = mu[0] + torch.tensor(rng.normal(size=(400000, K))) @ torch.linalg.cholesky(Sigma[0]).T
mc = 0.5*((z @ R.T)**2).sum(1).mean()
print(f"(a) loop copy: naive product cost {float(naive):.5f}, exact {float(exact):.5f}, MC {float(mc):.5f}")

# (b) Jacobi vs Gauss-Seidel mean updates for product recognition of N(m*, Lam^-1)
c = 0.6
Lam = np.full((3, 3), c) + (1 - c)*np.eye(3)
mstar = np.array([1.0, -0.5, 0.25])
F = lambda m: 0.5*(m - mstar) @ Lam @ (m - mstar)       # mean-dependent part of the collective VFE
def block(m, i):
    return mstar[i] - sum(Lam[i, j]*(m[j] - mstar[j]) for j in range(3) if j != i)/Lam[i, i]
mj = np.zeros(3); mg = np.zeros(3); hist_j, hist_g = [F(mj)], [F(mg)]
for _ in range(6):
    mj = np.array([block(mj, i) for i in range(3)]); hist_j.append(F(mj))
    for i in range(3):
        mg[i] = block(mg, i)
    hist_g.append(F(mg))
print("(b) eigenvalues of Lam:", np.round(np.linalg.eigvalsh(Lam), 3), " Jacobi spectral radius:",
      np.round(max(abs(np.linalg.eigvals(-(Lam - np.diag(np.diag(Lam)))/np.diag(Lam)[:, None]))), 3))
print("    Jacobi (parallel) F:", np.round(hist_j, 4))
print("    Gauss-Seidel (sweep) F:", np.round(hist_g, 6))
