r"""L6-formation: checks of c13 prop:rg-gated-depth-one on small graphs.

(c) rank a_F = |F| + N - cc(F); k_S >= 2(N - n) with equality iff att(S) is a minimum edge cover;
    every minimum edge cover is attended; sheet weight bounds (eq:rg-gated-sheet-weight);
    merge cost q{cc <= n-1} <= K_1 T (eq:rg-gated-merge-cost), computed exactly from determinants.
(d) block-law bound (eq:rg-gated-block-law) by exact sampling from q_Omega (Gaussian mixture).
Configuration x = (kbar, mbar, omega) in R^{2N+|E|}; edge e: j -> i,
r^b_e = kbar_i - kbar_j - omega_e, r^m_e = mbar_i - mbar_j - omega_e.
"""
from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
from scipy.linalg import null_space


def residual_map(N: int, edges: list[tuple[int, int]], e: int) -> np.ndarray:
    j, i = edges[e]
    a = np.zeros((2, 2 * N + len(edges)))
    a[0, i] += 1; a[0, j] -= 1; a[0, 2 * N + e] -= 1
    a[1, N + i] += 1; a[1, N + j] -= 1; a[1, 2 * N + e] -= 1
    return a


def edge_index(edges: list[tuple[int, int]], u: int, v: int) -> int:
    for k, (a, b) in enumerate(edges):
        if {a, b} == {u, v}:
            return k
    raise KeyError


def analyze(name: str, G: nx.Graph, T_list=(1e-2, 1e-3), a_res: float | None = None, nsamp: int = 200000,
            seed: int = 0) -> None:
    N = G.number_of_nodes()
    edges = [tuple(e) for e in G.edges()]
    E = len(edges)
    n = len(nx.max_weight_matching(G, maxcardinality=True))
    dim = 2 * N + E
    A = {e: residual_map(N, edges, e) for e in range(E)}
    # rank formula on all edge subsets (small graphs)
    worst = 0
    for r in range(E + 1):
        for F in itertools.combinations(range(E), r):
            if F:
                aF = np.vstack([A[e] for e in F])
                rk = np.linalg.matrix_rank(aF)
            else:
                rk = 0
            H = nx.Graph(); H.add_nodes_from(range(N)); H.add_edges_from([edges[e] for e in F])
            worst = max(worst, abs(rk - (len(F) + N - nx.number_connected_components(H))))
    # assignments
    nbrs = [sorted(G.neighbors(i)) for i in range(N)]
    assigns = list(itertools.product(*nbrs))
    D = np.zeros((N, dim))
    for i in range(N):
        D[i, i] = 1; D[i, N + i] = -1
    R_ev = 2 * D.T @ D
    att = {S: frozenset(edge_index(edges, i, j) for i, j in enumerate(S)) for S in assigns}
    HS = {S: sum(A[e].T @ A[e] for e in (edge_index(edges, i, j) for i, j in enumerate(S))) for S in assigns}
    kS = {S: np.linalg.matrix_rank(HS[S]) for S in assigns}
    min_cover_size = N - n
    covers_attended = {att[S] for S in assigns if len(att[S]) == min_cover_size and nx.is_forest(G.edge_subgraph([edges[e] for e in att[S]]))}
    # enumerate all minimum edge covers
    all_min_covers = set()
    for F in itertools.combinations(range(E), min_cover_size):
        if set(itertools.chain(*[edges[e] for e in F])) == set(range(N)):
            all_min_covers.add(frozenset(F))
    eq_ok = all((kS[S] == 2 * (N - n)) == (att[S] in all_min_covers) for S in assigns)
    ge_ok = all(kS[S] >= 2 * (N - n) for S in assigns)
    print(f"[{name}] N={N} |E|={E} n={n}: rank-formula max residual {worst}; k_S>=2(N-n): {ge_ok}; "
          f"equality iff min edge cover: {eq_ok}; #min covers {len(all_min_covers)}, all attended: "
          f"{all_min_covers <= set(att.values())}")
    for Omega_name, Omega in (("I+R_ev", np.eye(dim) + R_ev), ("I", np.eye(dim))):
        normO = np.linalg.norm(Omega, 2)
        for T in T_list:
            logZ, zlist, okw = {}, {}, True
            hS = {}
            for S in assigns:
                M = HS[S] / T + Omega
                sign, ld = np.linalg.slogdet(M)
                logZ[S] = 0.5 * dim * math.log(2 * math.pi) - 0.5 * ld
                ev = np.linalg.eigvalsh(HS[S])
                pos = ev[ev > 1e-9]
                hS[S] = pos.min()
                W = null_space(HS[S])
                zS = (np.prod(pos) * np.linalg.det(W.T @ Omega @ W)) ** -0.5
                ratio = math.exp(logZ[S]) / ((2 * math.pi) ** (dim / 2) * zS * T ** (kS[S] / 2))
                lo = (1 + T * normO / hS[S]) ** (-kS[S] / 2)
                okw &= (lo - 1e-12 <= ratio <= 1 + 1e-12)
                zlist[S] = zS
            prior = {S: 1.0 / len(assigns) for S in assigns}  # uniform product prior
            mx = max(logZ.values())
            w = {S: prior[S] * math.exp(logZ[S] - mx) for S in assigns}
            tot = sum(w.values())
            ccS = {S: nx.number_connected_components(nx.Graph([edges[e] for e in att[S]]) if att[S] else nx.Graph()) for S in assigns}
            # include isolated vertices count: att(S) covers V so fine
            mass_low = sum(w[S] for S in assigns if ccS[S] <= n - 1) / tot
            h_star = min(hS.values())
            Mset = [S for S in assigns if kS[S] == 2 * (N - n)]
            K1 = (1 + normO / h_star) ** (N - n) * sum(prior[S] * zlist[S] for S in assigns if ccS[S] <= n - 1) / sum(prior[S] * zlist[S] for S in Mset)
            print(f"  Omega={Omega_name} T={T:g}: sheet-weight bounds hold: {okw}; q(cc<=n-1)={mass_low:.3e} "
                  f"<= K_1 T={K1 * T:.3e}: {mass_low <= K1 * T}")
            # (d) exact sampling of block partition
            if a_res is not None:
                rng = np.random.default_rng(seed)
                Slist = list(assigns)
                probs = np.array([w[S] for S in Slist]) / tot
                idx = rng.choice(len(Slist), size=nsamp, p=probs)
                # min-edge-cover partitions
                mec_parts = set()
                for F in all_min_covers:
                    g = nx.Graph(); g.add_nodes_from(range(N)); g.add_edges_from([edges[e] for e in F])
                    mec_parts.add(frozenset(frozenset(c) for c in nx.connected_components(g)))
                bad = 0
                # s_*
                s_star = math.inf
                for S in assigns:
                    Wp = null_space(HS[S]); P = Wp @ Wp.T
                    comps = list(nx.connected_components(nx.Graph([edges[e] for e in att[S]])))
                    lab = {v: k for k, c in enumerate(comps) for v in c}
                    for e in range(E):
                        if e not in att[S] and lab[edges[e][0]] != lab[edges[e][1]]:
                            sv = np.linalg.svd(A[e] @ P, compute_uv=False)
                            s_star = min(s_star, sv[sv > 1e-9].min())
                for k in range(len(Slist)):
                    m = int((idx == k).sum())
                    if m == 0:
                        continue
                    S = Slist[k]
                    Cov = T * np.linalg.inv(HS[S] + T * Omega)
                    X = rng.multivariate_normal(np.zeros(dim), Cov, size=m)
                    for x in X:
                        coh = [e for e in range(E) if np.linalg.norm(A[e] @ x) <= a_res]
                        g = nx.Graph(); g.add_nodes_from(range(N)); g.add_edges_from([edges[e] for e in coh])
                        part = frozenset(frozenset(c) for c in nx.connected_components(g))
                        bad += part not in mec_parts
                bound = K1 * T + E * normO * a_res ** 2 / (2 * s_star ** 2) + 4 * E * math.exp(-a_res ** 2 / (4 * T))
                pb = bad / nsamp
                se = math.sqrt(max(pb * (1 - pb), 1 / nsamp) / nsamp)
                print(f"    (d) a={a_res}: P(partition not MEC) = {pb:.3e} +- {se:.1e}; bound = {bound:.3e} "
                      f"(K1T={K1*T:.2e}, area term={E*normO*a_res**2/(2*s_star**2):.2e}, tail={4*E*math.exp(-a_res**2/(4*T)):.2e}); s_*={s_star:.3f}")


def main() -> None:
    analyze("C4", nx.cycle_graph(4), T_list=(1e-2, 1e-3), a_res=0.3, nsamp=20000)
    analyze("C5", nx.cycle_graph(5), T_list=(1e-2, 1e-3), a_res=0.3, nsamp=20000)
    analyze("triangle+pendant", nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)]), T_list=(1e-2, 1e-3), a_res=0.3, nsamp=20000)
    analyze("K4", nx.complete_graph(4), T_list=(1e-2,), a_res=None)
    analyze("star K1,3", nx.star_graph(3), T_list=(1e-2,), a_res=None)


if __name__ == "__main__":
    main()
