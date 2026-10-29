r"""GAP07-S3 module: the Gaussian tangent global joint of Items 2 and 3 (it2:global-energy with lambda = 0, it3:energy,
it3:hierarchy) for K = 1, its global flow (it2:global-flow), and the declared selector it7:selector.

Energy terms, each 1/2 prec (g.x - target)^2:
  child self        g = e_{y_i},                              prec r_i,     target m_i
  edge e: j -> i    g = e_{y_i} - e_{y_j} - c_i e_{omega_e},  prec q_e,     target o_e (= 0, registered agreement)
  dictionary anchor g = e_{omega},                            prec 1/sigma^2, target 0
  parent self       g = e_{y_A},                              prec p,       target a_A
  cross-scale       g = e_{y_i} - e_{y_A} - c_A e_{omega_iA}, prec tau,     target 0
Agent blocks: child i holds y_i, the dictionaries of its received edges and (default) its cross-scale dictionary;
parent A holds y_A. Global flow: mu' = -D^{-1}(J mu - h), D = block diagonal of J over agent blocks (the Fisher metric at
the optimal factor covariances S_a = J_aa^{-1}). Linear with constant coefficients, so it is solved exactly:
with D = L L^T and L^{-1} J L^{-T} = U diag(lam) U^T, mu(t) - x* = L^{-T} U diag(e^{-lam t}) U^T L^T (mu(0) - x*).
"""
import itertools
import numpy as np


class Joint:
    def __init__(self):
        self.names, self.agent_of, self.terms = [], [], []

    def var(self, name, agent):
        self.names.append(name); self.agent_of.append(agent); return len(self.names) - 1

    def term(self, coefs, prec, target=0.0):
        self.terms.append((dict(coefs), float(prec), float(target)))

    def build(self):
        n = len(self.names); J = np.zeros((n, n)); h = np.zeros(n)
        for coefs, prec, target in self.terms:
            g = np.zeros(n)
            for k, v in coefs.items(): g[k] += v
            J += prec * np.outer(g, g); h += prec * target * g
        return J, h

    def metric(self, J):
        D = np.zeros_like(J)
        for a in set(self.agent_of):
            idx = [k for k, ag in enumerate(self.agent_of) if ag == a]
            D[np.ix_(idx, idx)] = J[np.ix_(idx, idx)]
        return D


class LinearFlow:
    r"""Exact solution of x' = -D^{-1}(J x - h)."""

    def __init__(self, J, h, D):
        self.xs = np.linalg.solve(J, h)
        L = np.linalg.cholesky(D); Li = np.linalg.inv(L)
        S = Li @ J @ Li.T; S = (S + S.T) / 2
        self.lam, U = np.linalg.eigh(S)
        self.P = Li.T @ U; self.Q = U.T @ L.T

    def traj(self, x0, ts):
        c = self.Q @ (x0 - self.xs)
        return self.xs[:, None] + self.P @ (c[:, None] * np.exp(-np.outer(self.lam, ts)))


def components(adj):
    m = adj.shape[0]; lab = -np.ones(m, int); c = 0
    for s in range(m):
        if lab[s] >= 0: continue
        st = [s]; lab[s] = c
        while st:
            u = st.pop()
            for v in np.nonzero(adj[u])[0]:
                if lab[v] < 0: lab[v] = c; st.append(v)
        c += 1
    return [frozenset(np.nonzero(lab == j)[0].tolist()) for j in range(c)]


def first_selection(M, ts, lam_sel=6.0, dc=0.05, dw=0.5, tp=2.0):
    r"""Declared selector it7:selector on units with presented means M[u, t] (presented precision lam_sel).
    Returns (time, list of selected unit sets) of the first handoff on the grid, or (None, None)."""
    K = M.shape[0]; onset = {}
    for k, t in enumerate(ts):
        m = M[:, k]; Dm = lam_sel / 2 * (m[:, None] - m[None, :]) ** 2
        found = []
        for C in components(Dm < dc):
            if len(C) < 2: continue
            out = [j for j in range(K) if j not in C]
            if all(Dm[i, j] > dw for i in C for j in out): found.append(C)
        onset = {C: onset.get(C, t) for C in found}
        due = [C for C, t0 in onset.items() if t - t0 >= tp - 1e-9]
        if due:
            return t, due
    return None, None


def dwell_union(g, ts, lam_sel=6.0, dc=0.05, tp=2.0):
    r"""Two units with presented-mean gap g(t): first time at which KL < dc has held for tp (None if never)."""
    coh = lam_sel / 2 * g ** 2 < dc; t0 = None
    for k, t in enumerate(ts):
        if coh[k]:
            if t0 is None: t0 = t
            if t - t0 >= tp - 1e-9: return t
        else:
            t0 = None
    return None


def longest_dwell(g, ts, lam_sel=6.0, dc=0.05):
    coh = lam_sel / 2 * g ** 2 < dc; best = cur = 0.0
    dt = ts[1] - ts[0]
    for c in coh:
        cur = cur + dt if c else 0.0; best = max(best, cur)
    return best


def two_cluster_joint(rec, r, q_in, q_br, sig2, c_child, parents=None):
    r"""Four children a1, a2 | b1, b2 (indices 0-3) with self laws N(rec_i, 1/r), internal edges a1 <-> a2, b1 <-> b2
    (precision q_in), bridges a2 <-> b1 (precision q_br), receiver-owned dictionaries with anchor 1/sig2 and operating
    contents c_child[i]; edge records o_e = 0. parents = dict(p, a, tau, sigA2, cA, owner) adds the supplied hierarchy."""
    Jt = Joint()
    y = [Jt.var(f"y{i}", f"c{i}") for i in range(4)]
    for i in range(4): Jt.term({y[i]: 1.0}, r, rec[i])
    edges = []
    if q_in > 0: edges += [(1, 0, q_in), (0, 1, q_in), (3, 2, q_in), (2, 3, q_in)]
    if q_br > 0: edges += [(2, 1, q_br), (1, 2, q_br)]
    for (j, i, q) in edges:                                    # edge j -> i, receiver i owns omega_e
        w = Jt.var(f"w{j}{i}", f"c{i}")
        Jt.term({y[i]: 1.0, y[j]: -1.0, w: -c_child[i]}, q, 0.0)
        Jt.term({w: 1.0}, 1.0 / sig2, 0.0)
    yA = None
    if parents is not None:
        P = parents
        yA = [Jt.var("yA_I", "A_I"), Jt.var("yA_J", "A_J")]
        Jt.term({yA[0]: 1.0}, P["p"], -P["a"]); Jt.term({yA[1]: 1.0}, P["p"], P["a"])
        for i in range(4):
            B = 0 if i < 2 else 1
            owner = f"c{i}" if P.get("owner", "child") == "child" else ("A_I" if B == 0 else "A_J")
            if P["cA"] != 0.0:
                w = Jt.var(f"wA{i}", owner)
                cA = -P["cA"] if B == 0 else P["cA"]
                Jt.term({y[i]: 1.0, yA[B]: -1.0, w: -cA}, P["tau"], 0.0)
                Jt.term({w: 1.0}, 1.0 / P["sigA2"], 0.0)
            else:
                Jt.term({y[i]: 1.0, yA[B]: -1.0}, P["tau"], 0.0)
    return Jt, y, yA
