r"""XC7-IMPL s9: VFE_4.0's scalar is the 2.0 global free energy (eq:dyn-fixed-vfe,
prop:dyn-global-local) of its declared normalized joint, at an arbitrary recognition state.

Read-only use of the vfe4 package (no run_simulation, no files written; run with
PYTHONDONTWRITEBYTECODE=1).  Tiny point configuration: N=4 agents, K=2, held links,
window radius 1 with self copies (three candidates per row), both channels.

We rebuild F(Q;o) = E_Q[log Q - log P(m,k,J,o)] independently in numpy:
  m_s ~ N(0, H^-1 I), k_s | m_s ~ N(m_s, Q I), private o ~ N(k_s, s_p^2 I),
  per channel and receiving row: J ~ pi_row, o_row | J=c ~ N(A y_r - B_c y_src(c), tau I),
  with B_c = exp(omega_c) for nonself copies (scipy expm) and B = I for self copies,
and compare with vfe4.objective.evaluate_vfe at a random non-optimal state.  A Monte Carlo
estimate of the same expectation is printed as a second, implementation-free check.
Then we test the coordinate-descent property: after one chromatic sweep (beliefs only),
the last colour's belief blocks are stationary for F, the first colour's are not.
"""
import os, sys
sys.dont_write_bytecode = True
ROOT = r"C:\Users\chris and christine\Desktop\VFE_4.0"
sys.path.insert(0, ROOT)
import numpy as np
import torch
from scipy.linalg import expm
from copy import deepcopy
from vfe4.presets import PRESETS, strict_merge
from vfe4.config import load_config
from vfe4.model import build_model, initialize_sections
from vfe4.objective import evaluate_vfe, optimal_rows
from vfe4.inference import make_schedule, _channel_sweep
torch.set_num_threads(1)

raw = strict_merge(PRESETS['tiny_point'], {'inference': {'model_every': 1000}})
cfg = load_config(raw); model = build_model(cfg); state = initialize_sections(model, cfg)
K = cfg['bundle']['K']; S = model.incidence.agent_id.numel()
rng = np.random.default_rng(0)
for laws in (state.belief, state.model):
    for b in laws.buckets:
        b.mu[:] = torch.tensor(rng.normal(size=tuple(b.mu.shape)))
        Lr = np.tril(rng.normal(size=tuple(b.chol.shape)) * 0.3)
        for i in range(Lr.shape[0]):
            np.fill_diagonal(Lr[i], np.abs(np.diag(Lr[i])) + 0.4)
        b.chol[:] = torch.tensor(Lr)
lens = model.rows.row_offset.diff().numpy()
beta = np.concatenate([rng.dirichlet(np.ones(n)) for n in lens]); state.beta = torch.tensor(beta)
vfe4_value = float(evaluate_vfe(state, model).total)

def gauss(laws):
    mus, covs = {}, {}
    for b in laws.buckets:
        for a_idx, inc in enumerate(b.incidence_ids[:, 0].tolist()):
            mus[inc] = b.mu[a_idx].numpy().copy(); Lc = b.chol[a_idx].numpy(); covs[inc] = Lc @ Lc.T
    return mus, covs
mk, Sk = gauss(state.belief); mm, Sm = gauss(state.model)
Hp = cfg['generative']['model_prior_std'] ** -2; Qb = cfg['generative']['bridge_std'] ** 2
tau = cfg['communication']['tau']; logpi = state.log_pi.numpy()
def ent(Sg): return 0.5 * np.linalg.slogdet(2 * np.pi * np.e * Sg)[1]
def E_neglog_normal(o_mean_resid, cov_resid, prec):
    d = len(o_mean_resid)
    return 0.5 * (o_mean_resid @ prec @ o_mean_resid + np.trace(prec @ cov_resid) - np.linalg.slogdet(prec / (2 * np.pi))[1])
F = 0.0
for s in range(S):
    F -= ent(Sk[s]) + ent(Sm[s])
    F += E_neglog_normal(mm[s], Sm[s], Hp * np.eye(K))                       # -log N(m; 0, H^-1)
    F += E_neglog_normal(mk[s] - mm[s], Sk[s] + Sm[s], np.eye(K) / Qb)      # -log N(k; m, Q), independent factors
F += np.sum(beta * (np.log(beta) - logpi))
omega = state.links.omega_mean.numpy(); copies = model.copies
for bank in model.records:
    o = bank.observed.numpy(); prec = bank.precision.numpy(); A = bank.query.numpy(); Bk = bank.key.numpy()
    mus, covs = (mk, Sk) if bank.channel == 0 else (mm, Sm)
    for f in range(len(o)):
        r = int(bank.receiver[f]); sidx = int(bank.sender[f])
        if bank.kind == 'communication':
            cid = int(bank.candidate_id[f]); prim = int(copies.primitive_id[int(model.rows.copy_id[cid])])
            Bmap = Bk[f] @ (expm(omega[prim].reshape(K, K)) if prim >= 0 else np.eye(K))
            w = beta[cid]
            if sidx == r:                                        # self copy: one variable
                Mmap = A[f] - Bmap; res = o[f] - Mmap @ mus[r]; cv = Mmap @ covs[r] @ Mmap.T
            else:
                res = o[f] - A[f] @ mus[r] + Bmap @ mus[sidx]; cv = A[f] @ covs[r] @ A[f].T + Bmap @ covs[sidx] @ Bmap.T
            F += w * E_neglog_normal(res, cv, prec[f])
        else:
            if sidx < 0:
                F += E_neglog_normal(o[f] - A[f] @ mus[r], A[f] @ covs[r] @ A[f].T, prec[f])
            else:
                res = o[f] - A[f] @ mus[r] + Bk[f] @ mus[sidx]
                F += E_neglog_normal(res, A[f] @ covs[r] @ A[f].T + Bk[f] @ covs[sidx] @ Bk[f].T, prec[f])
print(f"VFE_4.0 evaluate_vfe total = {vfe4_value:.10f}")
print(f"independent global free energy KL(Q||P0) - E_Q log l = {F:.10f}   difference {vfe4_value - F:.2e}")

# Monte Carlo estimate of E_Q[log Q - log P] (labels sampled as well)
ns = 200000; tot = np.zeros(ns)
def logN(x, mean, cov):
    d = x.shape[-1]; diff = x - mean; ci = np.linalg.inv(cov)
    return -0.5 * (np.einsum('ni,ij,nj->n', diff, ci, diff) + np.linalg.slogdet(2 * np.pi * cov)[1])
ks = {s: mk[s] + rng.normal(size=(ns, K)) @ np.linalg.cholesky(Sk[s]).T for s in range(S)}
ms = {s: mm[s] + rng.normal(size=(ns, K)) @ np.linalg.cholesky(Sm[s]).T for s in range(S)}
for s in range(S):
    tot += logN(ks[s], mk[s], Sk[s]) + logN(ms[s], mm[s], Sm[s])
    tot -= logN(ms[s], np.zeros(K), np.eye(K) / Hp) + logN(ks[s], ms[s], Qb * np.eye(K))
offs = model.rows.row_offset.numpy()
labels = np.zeros((len(lens), ns), int)
for rrow in range(len(lens)):
    pr = beta[offs[rrow]:offs[rrow + 1]]; labels[rrow] = offs[rrow] + rng.choice(len(pr), size=ns, p=pr)
    tot += np.log(beta[labels[rrow]]) - logpi[labels[rrow]]
for bank in model.records:
    o = bank.observed.numpy(); prec = bank.precision.numpy(); A = bank.query.numpy(); Bk = bank.key.numpy()
    ys = ks if bank.channel == 0 else ms
    for f in range(len(o)):
        r = int(bank.receiver[f]); sidx = int(bank.sender[f]); cov = np.linalg.inv(prec[f])
        if bank.kind == 'communication':
            cid = int(bank.candidate_id[f]); prim = int(copies.primitive_id[int(model.rows.copy_id[cid])])
            Bmap = Bk[f] @ (expm(omega[prim].reshape(K, K)) if prim >= 0 else np.eye(K))
            rrow = int(model.rows.row_id[cid]); active = labels[rrow] == cid
            ysrc = ys[sidx] if sidx != r else ys[r]
            pred = ys[r] @ A[f].T - ysrc @ Bmap.T
            tot -= active * logN(np.broadcast_to(o[f], pred.shape), pred, cov)
        else:
            pred = ys[r] @ A[f].T - (ys[sidx] @ Bk[f].T if sidx >= 0 else 0)
            tot -= logN(np.broadcast_to(o[f], pred.shape), pred, cov)
print(f"Monte Carlo E_Q[log Q - log P] = {tot.mean():.4f} +- {tot.std() / np.sqrt(ns):.4f}")

# coordinate descent: one chromatic belief sweep at fixed rows
schedule = make_schedule(model, cfg)
state.beta = optimal_rows(model.record_costs(state), model.rows, state.log_pi)
_channel_sweep(state, model, schedule, cfg, 0, 1.0)
def grad_mean(agent, h=1e-6):
    g = []
    for comp in range(K):
        out = []
        for sgn in (1, -1):
            st = deepcopy(state)
            for b in st.belief.buckets:
                pos = (b.agent_ids == agent).nonzero().flatten()
                if pos.numel(): b.mu[pos[0], comp] += sgn * h
            out.append(float(evaluate_vfe(st, model).total))
        g.append((out[0] - out[1]) / (2 * h))
    return np.array(g)
cols = [c.tolist() for c in schedule.colors]
print("colours:", cols)
print("|dF/dmu| for last-colour agents:", [np.abs(grad_mean(a)).max() for a in cols[-1]],
      " first-colour agents:", [np.abs(grad_mean(a)).max() for a in cols[0]])
