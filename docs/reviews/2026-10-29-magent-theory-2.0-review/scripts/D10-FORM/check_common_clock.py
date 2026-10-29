r"""D10-FORM check 3: inh:common-clock and its corollary (ch05 lines 1329-1373).

Model (belief sector only; the model sector decouples under a model-independent evaluator):
  E(q, beta) = sum_i KL(q_i || G_i) + sum_i sum_{j in N_i} beta_ij KL(q_i || nu_ij) + T sum_i KL(beta_i || pi_i),
with nu_ij = C[M] (constant-output channel of the additive moment M = sum_l E_{q_l} T_stat),
C[M] = softmax(W M + b) on K = 3 symbols, N = 3 agents.

  (a) hypotheses of inh:common-clock (shared G_0, self excluded, nu_ij = C[M]):
      r_i(k) := h_i(k) - 2 log q_i(k) must agree across agents up to a k-independent constant.
  (b) Gibbs-reduced potential: same statement.
  (c) null flow (peer costs deleted): h_i - log q_i common -> coefficient -1.
  (d) boundary cases where one hypothesis is dropped:
      (d1) self in the source set with identity self channel, unequal self rows;
      (d2) heterogeneous evaluators G_i;
      (d3) input-dependent fixed channel nu_ij = q_j K;
      (d4) model-dependent evaluator with unequal model laws s_i.
  (e) integration of the full and null flows of a binary three-agent model with
      heterogeneous evaluators and a receiver-independent target C[M]:
      closed linear law for log-odds differences, and a threshold selector whose
      partition sequence differs between full and null flows on an open set.
  (f) finite-horizon truncation: with the hypotheses of the proposition, a horizon tau
      produces P != P^0 although the full flow is the null flow with time halved.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

rng = np.random.default_rng(11)
out: list[str] = []
N, K = 3, 3

qs = [[sp.Symbol(f"q{i}_{k}", positive=True) for k in range(K)] for i in range(N)]
Tstat = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])  # T: K -> R^2
Wm = sp.Matrix([[0.4, -0.3], [0.2, 0.5], [-0.6, 0.1]])
bvec = sp.Matrix([0.1, -0.2, 0.05])
M = [sum(qs[i][k] * Tstat[k, d] for i in range(N) for k in range(K)) for d in range(2)]
logits = Wm * sp.Matrix(M) + bvec
den = sum(sp.exp(logits[u]) for u in range(K))
Cm = [sp.exp(logits[u]) / den for u in range(K)]


def kl(p, qv):
    return sum(p[u] * (sp.log(p[u]) - sp.log(qv[u])) for u in range(K))


def build(G, sources, nu_fun, betas):
    E = 0
    for i in range(N):
        E += kl(qs[i], G[i])
        for j in sources[i]:
            E += betas[i][j] * kl(qs[i], nu_fun(i, j))
    return E


def residual_common(E, coeff, point):
    """max over pairs (i,j) of the k-spread of r_i(k) - r_j(k), r_i = h_i - coeff*log q_i."""
    r = []
    for i in range(N):
        ri = []
        for k in range(K):
            hik = sp.diff(E, qs[i][k])
            ri.append(float((hik - coeff * sp.log(qs[i][k])).evalf(25, subs=point)))
        r.append(np.array(ri))
    spread = 0.0
    for i in range(N):
        for j in range(N):
            d = r[i] - r[j]
            spread = max(spread, float(np.max(d) - np.min(d)))
    return spread


point = {}
for i in range(N):
    p = rng.dirichlet(np.ones(K))
    for k in range(K):
        point[qs[i][k]] = float(p[k])
G0 = [0.2, 0.5, 0.3]
Gshared = [G0] * N
others = {i: [j for j in range(N) if j != i] for i in range(N)}
rowsd = {i: dict(zip(others[i], rng.dirichlet(np.ones(N - 1)))) for i in range(N)}

E_full = build(Gshared, others, lambda i, j: Cm, rowsd)
out.append(f"(a) dynamic rows, hypotheses met: k-spread of r_i-r_j with coeff 2: {residual_common(E_full, 2, point):.2e}")
E_red = sum(kl(qs[i], Gshared[i]) for i in range(N)) + sum(kl(qs[i], Cm) for i in range(N))
out.append(f"(b) Gibbs-reduced (all peer costs equal, Z_i = exp(-D_i/T)): k-spread with coeff 2: {residual_common(E_red, 2, point):.2e}")
E_null = sum(kl(qs[i], Gshared[i]) for i in range(N))
out.append(f"(c) null flow: k-spread with coeff 1: {residual_common(E_null, 1, point):.2e}; with coeff 2: {residual_common(E_null, 2, point):.2e}")

# (d1) self included with identity self channel, unequal self rows
srcs_self = {i: list(range(N)) for i in range(N)}
rows_self = {i: dict(zip(range(N), rng.dirichlet(np.ones(N)))) for i in range(N)}
E_d1 = build(Gshared, srcs_self, lambda i, j: (qs[i] if i == j else Cm), rows_self)
out.append(f"(d1) self in source set (identity self channel, unequal beta_ii): spread coeff 2: {residual_common(E_d1, 2, point):.3e}")
# (d2) heterogeneous evaluators
Ghet = [[0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.25, 0.25, 0.5]]
E_d2 = build(Ghet, others, lambda i, j: Cm, rowsd)
out.append(f"(d2) heterogeneous evaluators: spread coeff 2: {residual_common(E_d2, 2, point):.3e}")
# (d3) input-dependent fixed channel
Kfix = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
E_d3 = build(Gshared, others, lambda i, j: [sum(qs[j][v] * Kfix[v, u] for v in range(K)) for u in range(K)], rowsd)
out.append(f"(d3) input-dependent fixed channel q_j K: spread coeff 2: {residual_common(E_d3, 2, point):.3e}")
# (d4) model-dependent evaluator G(m,k) with unequal fixed s_i: the effective evaluator is exp(sum_m s_i(m) log G(m,.))
Gm = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
s_laws = [np.array([0.9, 0.1]), np.array([0.3, 0.7]), np.array([0.5, 0.5])]
E_d4 = 0
for i in range(N):
    E_d4 += sum(qs[i][k] * (sp.log(qs[i][k]) - float(s_laws[i] @ np.log(Gm[:, k]))) for k in range(K))
    for j in others[i]:
        E_d4 += rowsd[i][j] * kl(qs[i], Cm)
out.append(f"(d4) model-dependent evaluator, unequal s_i: spread coeff 2: {residual_common(E_d4, 2, point):.3e}")

# (e) binary three-agent model with heterogeneous evaluators and receiver-independent target C[M]
gam = np.array([0.0, 0.75, 3.0])  # evaluator log-odds; delta = 0.5 below
wM, b0 = 0.8, -0.4  # C[M] = Ber(sigmoid(wM * M + b0)), M = sum_i q_i(1)
sig = lambda u: 1.0 / (1.0 + np.exp(-u))


def rhs_binary(t, lam, peers: bool):
    p = sig(lam)
    Mv = p.sum()
    cpar = sig(wM * Mv + b0)
    clog = np.log(cpar / (1 - cpar))
    # belief force difference h(1)-h(0) for agent i:
    #  local: (lam_i - gam_i); receiver: (lam_i - clog); source through M:
    #  d/dM sum_l KL(q_l || C) * dM/dq_i(1)-dM/dq_i(0) = sum_l [ -(p_l/c - (1-p_l)/(1-c)) c(1-c) wM ] * (1 - 0)
    dh = lam - gam
    if peers:
        src = np.sum(-(p / cpar - (1 - p) / (1 - cpar)) * cpar * (1 - cpar) * wM)
        dh = dh + (lam - clog) + src
    # replicator in log-odds: dlam/dt = -(h(1) - h(0))
    return -dh


def run(lam0, peers, tmax=12.0):
    sol = solve_ivp(rhs_binary, (0, tmax), lam0, args=(peers,), rtol=1e-11, atol=1e-12, dense_output=True, max_step=0.02)
    return sol


lam0 = np.array([0.2, 2.0, 4.6])
sol_f = run(lam0, True)
sol_n = run(lam0, False)
ts = np.linspace(0, 12, 601)
Lf = sol_f.sol(ts)
Ln = sol_n.sol(ts)
pairs = [(0, 1), (0, 2), (1, 2)]
res_lin = 0.0
for (i, j) in pairs:
    d0 = lam0[i] - lam0[j]
    dg = gam[i] - gam[j]
    pred_f = np.exp(-2 * ts) * d0 + (1 - np.exp(-2 * ts)) * dg / 2
    pred_n = np.exp(-ts) * d0 + (1 - np.exp(-ts)) * dg
    res_lin = max(res_lin, np.max(np.abs((Lf[i] - Lf[j]) - pred_f)), np.max(np.abs((Ln[i] - Ln[j]) - pred_n)))
out.append(f"(e) closed linear law for log-odds differences (full: rate 2 to dgamma/2; null: rate 1 to dgamma): max residual {res_lin:.2e}")


def partition_sequence(Lt, delta=0.5):
    """Join a pair when |lam_i - lam_j| reaches delta from above and the relation |.|<delta then
    persists to the end of the window (an invariant block relation on the window)."""
    seq = []
    for (i, j) in pairs:
        d = np.abs(Lt[i] - Lt[j])
        if d[0] <= delta or d[-1] >= delta:
            continue
        last_above = np.where(d >= delta)[0][-1]
        seq.append(((i, j), round(float(ts[last_above + 1]), 2)))
    return sorted(seq, key=lambda e: e[1])


out.append(f"(e) full-flow joins (pair, time): {partition_sequence(Lf)}")
out.append(f"(e) null-flow joins (pair, time): {partition_sequence(Ln)}")
# openness: perturb initial data
diffs = 0
trials = 20
for _ in range(trials):
    l0 = lam0 + rng.uniform(-0.1, 0.1, size=3)
    a = [e[0] for e in partition_sequence(run(l0, True).sol(ts))]
    b = [e[0] for e in partition_sequence(run(l0, False).sol(ts))]
    diffs += int(a != b)
out.append(f"(e) partition sequences differ (full vs null) in {diffs}/{trials} random perturbations of size 0.1")

# (f) finite horizon: shared evaluator, differences scale e^{-2t} (full) vs e^{-t} (null)
delta, tau = 0.5, 1.0
D0 = delta * np.exp(1.5 * tau)
full_joins = D0 * np.exp(-2 * tau) < delta
null_joins = D0 * np.exp(-1 * tau) < delta
out.append(f"(f) horizon tau={tau}: pair with |Delta(0)|=delta*e^(1.5 tau) joins in full flow: {full_joins}; in null flow: {null_joins}; set of such Delta(0) is the open interval (delta e^tau, delta e^(2 tau))")

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
