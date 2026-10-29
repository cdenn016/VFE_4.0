"""XC2-FRAME check: does the (F5) verdict of pa:persistent-grouping depend on the
declared selector constant?

Model (chapters/13_per_agent_inheritance.tex, eq. pa:bit-flow, pa:bit-null, pa:selector):
three agents, Bernoulli belief laws with log odds l_i on z in {-1,1}, uniform self
reference and evaluator, identity peer channels, priors pi_ii = a = 1/5, pi_ij = p = 2/5,
unit-temperature Gibbs rows, records o = (1,1,-1) of strength eps.

  full:  dl_i/dt = -l_i + sum_{j != i} beta_ij l_j + eps*o_i
  null:  dl_i/dt = -beta0_ii(l_i) l_i + eps*o_i,
         beta0_ii = a e^{-D_ii} / (a e^{-D_ii} + 2p),  D_ii = KL(q_i || uniform)

Selector (pa:selector with a variable constant c; the manuscript uses c = 1/4):
connect i,j when l_i, l_j have the same nonzero sign and min/max(|l|) > c; blocks are
connected components.  Partition sequence P = event times dropped.
"""
import numpy as np

A_SELF, P_PEER = 0.2, 0.4
O = np.array([1.0, 1.0, -1.0])
PRI = np.full((3, 3), P_PEER)
np.fill_diagonal(PRI, A_SELF)

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def kl_bern(l, m):
    p, r = sig(l), sig(m)
    return p * (np.log(p) - np.log(r)) + (1 - p) * (np.log(1 - p) - np.log(1 - r))

def full_field(l, eps):
    C = kl_bern(l[:, None], l[None, :])          # C[i,j] = KL(q_i || q_j)
    np.fill_diagonal(C, kl_bern(l, 0.0))          # self cost KL(q_i || uniform)
    W = PRI * np.exp(-C)
    B = W / W.sum(axis=1, keepdims=True)
    Bo = B.copy(); np.fill_diagonal(Bo, 0.0)
    return -l + Bo @ l + eps * O

def null_field(l, eps):
    dii = kl_bern(l, 0.0)
    b0 = A_SELF * np.exp(-dii) / (A_SELF * np.exp(-dii) + 2 * P_PEER)
    return -b0 * l + eps * O

def integrate(field, l0, eps, T=100.0, dt=0.02):
    n = int(T / dt)
    traj = np.empty((n + 1, 3)); traj[0] = l0
    l = l0.copy()
    for k in range(n):
        k1 = field(l, eps); k2 = field(l + 0.5 * dt * k1, eps)
        k3 = field(l + 0.5 * dt * k2, eps); k4 = field(l + dt * k3, eps)
        l = l + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        traj[k + 1] = l
    return traj

def partition(l, c):
    parent = [0, 1, 2]
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(3):
        for j in range(i + 1, 3):
            if l[i] * l[j] > 0:
                r = min(abs(l[i]), abs(l[j])) / max(abs(l[i]), abs(l[j]))
                if r > c:
                    parent[find(i)] = find(j)
    blocks = {}
    for i in range(3):
        blocks.setdefault(find(i), []).append(i + 1)
    return tuple(sorted(tuple(b) for b in blocks.values()))

def seq_of(traj, c):
    seq = [partition(traj[0], c)]
    for row in traj[1:]:
        p = partition(row, c)
        if p != seq[-1]:
            seq.append(p)
    return seq

def fmt(seq):
    return " -> ".join("|".join("".join(str(i) for i in b) for b in p) for p in seq)

if __name__ == "__main__":
    M = -1.4 * np.eye(3) + 0.4 * np.ones((3, 3))
    us = -np.linalg.solve(M, O)
    print("Limit equilibrium u* = -M^{-1}o:", np.round(us, 6), " u3/u1 =", round(us[2] / us[0], 6))
    u_in = np.array([4.0, 0.5, -1.0])
    for eps in (0.01, 0.05):
        l0 = eps * u_in
        tf = integrate(full_field, l0, eps)
        t0 = integrate(null_field, l0, eps)
        print(f"\neps = {eps}; final full l/eps = {np.round(tf[-1]/eps, 4)}; final null l/eps = {np.round(t0[-1]/eps, 4)}")
        # transient maximum of agent-3 ratio while agent 3 is positive, full flow
        pos = tf[:, 2] > 0
        r3 = np.where(pos, tf[:, 2] / np.maximum(tf[:, 0], tf[:, 1]), np.nan)
        r3b = np.where(pos, np.minimum(tf[:, 2], tf[:, 1]) / np.maximum(tf[:, 2], tf[:, 1]), np.nan)
        print(f"  full flow: sup over t of ratio(agent3, agent2) while agent 3 > 0 = {np.nanmax(r3b):.4f}")
        for c in (0.13, 0.2, 0.25, 0.3, 0.33, 0.34, 0.4, 0.5, 0.6, 0.75, 0.95):
            sf, s0 = seq_of(tf, c), seq_of(t0, c)
            v = "P != P0: (F5) holds" if sf != s0 else "P == P0: (F5) fails"
            print(f"  c={c:4.2f}  full: {fmt(sf):34s} null: {fmt(s0):16s} {v}")
