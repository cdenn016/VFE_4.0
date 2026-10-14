"""L11 check s1: MultiAgentELBO latent_links objective against the theory free energy.

Independent evaluation of F = E_Q[log q - log p(omega, z, J, o)] for the latent-link
model of def:link-latent-model with the per-receiver comparison record of
prop:obs-selective-receiver-update (S^Q = I, S^K = exp(omega)), K = 1, N = 3 agents
on a triangle, each row attending to both other agents, one triangle loop in the
generator prior. Expectations are tensor Gauss-Hermite quadrature on the raw log
densities; nothing is taken from the code's analytic Gaussian formulas.
The code side is latent_links.relative_free_energy with its sample means replaced by
the same quadrature weights; it omits log Z_S by design, which is added back here.
"""
import sys, itertools
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\chris and christine\Desktop\MultiAgentELBO\src")
from multiagent_elbo.realizations.gaussian.coupled_learning import CoupledModel
from multiagent_elbo.realizations.gaussian.latent_links import (
    LatentLinkModel, LatentLinkState, generator_kl, loop_action, matter_free_energy, record_costs)

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)
N, K, E, A = 3, 1, 3, 2
tau = 0.7
a = rng.normal(size=(N, K)); J0 = rng.uniform(0.8, 1.6, size=(N, 1, 1)); Hm = rng.uniform(0.6, 1.4, size=(N, 1, 1))
sources = np.array([[1, 2], [0, 2], [0, 1]])
endpoints = np.array([[0, 1], [1, 2], [2, 0]])          # (receiver, source) of stored copy
link_index = np.array([[0, 2], [0, 1], [2, 1]])
reverse = np.array([[False, True], [True, False], [False, True]])
pi = np.array([[0.3, 0.7], [0.55, 0.45], [0.2, 0.8]])
o = rng.normal(size=(N, K))
alpha = rng.normal(scale=0.2, size=(E, 1)); P = rng.uniform(2.0, 4.0, size=(E, 1, 1))
c_loop = 0.9
mu = rng.normal(size=(N, K)); Sig = rng.uniform(0.3, 0.8, size=(N, 1, 1))
beta = np.array([[0.4, 0.6], [0.5, 0.5], [0.35, 0.65]])
om_mu = rng.normal(scale=0.2, size=(E, 1)); Lam = rng.uniform(3.0, 6.0, size=(E, 1, 1))

T = lambda x, dt=None: torch.tensor(x) if dt is None else torch.tensor(x, dtype=dt)
matter = CoupledModel(T(a), T(J0), T(Hm), T(sources, torch.long), T(endpoints, torch.long),
                      T(link_index, torch.long), T(reverse, torch.bool), tau)
model = LatentLinkModel(matter, T(pi), T(alpha), T(P), T([[0, 1, 2]], torch.long), T([[1, 1, 1]], torch.long),
                        T([0], torch.long), T([c_loop]), T(np.repeat(o[:, None, :], A, 1)))
state = LatentLinkState(T(mu), T(Sig), T(beta), T(om_mu), T(Lam))

# ---------- quadrature grid over the three generators under R ----------
x, w = np.polynomial.hermite_e.hermegauss(40); w = w/w.sum()
sd = 1/np.sqrt(Lam[:, 0, 0])
grid = np.array(list(itertools.product(range(len(x)), repeat=3)))
om_nodes = om_mu[:, 0][None, :] + sd[None, :]*x[grid]           # (S,3)
wR = np.prod(w[grid], axis=1)
loop_sign = np.array([1, 1, 1])
S_of = lambda om: 0.5*(np.exp(om @ loop_sign) - 1.0)**2*c_loop   # K=1 loop defect, Frobenius metric
# ---------- log Z_S under the base law nu (independent grid) ----------
sdn = 1/np.sqrt(P[:, 0, 0])
nu_nodes = alpha[:, 0][None, :] + sdn[None, :]*x[grid]
logZ = np.log(np.sum(wR*np.exp(-S_of(nu_nodes))))
# ---------- theory: E_Q[log q] - E_Q[log p] ----------
log_nu = lambda om: np.sum(-0.5*np.log(2*np.pi/P[:, 0, 0]) - 0.5*P[:, 0, 0]*(om-alpha[:, 0])**2, axis=-1)
neg_ent_R = np.sum(-0.5*np.log(2*np.pi*np.e/Lam[:, 0, 0]))
neg_ent_q = np.sum(-0.5*np.log(2*np.pi*np.e*Sig[:, 0, 0]))
neg_ent_b = np.sum(beta*np.log(beta))
E_log_prior_om = np.sum(wR*(-logZ - S_of(om_nodes) + log_nu(om_nodes)))
xz, wz = np.polynomial.hermite_e.hermegauss(30); wz = wz/wz.sum()
E_log_pz = 0.0
for i in range(N):
    z = mu[i, 0] + np.sqrt(Sig[i, 0, 0])*xz
    E_log_pz += np.sum(wz*(-0.5*np.log(2*np.pi/J0[i, 0, 0]) - 0.5*J0[i, 0, 0]*(z-a[i, 0])**2))
E_log_rec = 0.0
x1, w1 = np.polynomial.hermite_e.hermegauss(30); w1 = w1/w1.sum()
for i in range(N):
    for s in range(A):
        j, e = sources[i, s], link_index[i, s]; sg = -1.0 if reverse[i, s] else 1.0
        om = om_mu[e, 0] + sd[e]*x1[:, None, None]
        zi = mu[i, 0] + np.sqrt(Sig[i, 0, 0])*xz[None, :, None]
        zj = mu[j, 0] + np.sqrt(Sig[j, 0, 0])*xz[None, None, :]
        var = tau/Hm[i, 0, 0]
        logN = -0.5*np.log(2*np.pi*var) - 0.5*(o[i, 0] - (zi - np.exp(sg*om)*zj))**2/var
        W = w1[:, None, None]*wz[None, :, None]*wz[None, None, :]
        E_log_rec += beta[i, s]*(np.log(pi[i, s]) + np.sum(W*logN))
F_theory = neg_ent_R + neg_ent_q + neg_ent_b - E_log_prior_om - E_log_pz - E_log_rec
# ---------- code: relative_free_energy pieces with quadrature weights ----------
omt = T(om_nodes[:, :, None]); wt = T(wR)
rc = record_costs(omt, state, model)                                   # (S,N,A)
mfe_uniform = matter_free_energy(omt, state, model)
rec_uniform = (state.beta*rc.mean(0)).sum()/tau
rec_weighted = (state.beta*(wt[:, None, None]*rc).sum(0)).sum()/tau
code_rel = float(generator_kl(state, model) + (wt*loop_action(omt, model)).sum() + mfe_uniform - rec_uniform + rec_weighted)
print(f"log Z_S (quadrature)              = {logZ:.12f}")
print(f"F theory (direct E_Q log q/p)     = {F_theory:.12f}")
print(f"F code (relative_free_energy+logZ)= {code_rel + logZ:.12f}")
print(f"residual                          = {code_rel + logZ - F_theory:.3e}")
