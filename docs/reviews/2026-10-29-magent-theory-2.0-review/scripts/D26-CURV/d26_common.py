r"""Shared helpers for the D26-CURV checks (float64, tiny sizes, single thread).

Conventions follow chapters/19_curvature_hierarchy.tex: an edge e = (j, i) means e: j -> i,
receiver i, sender j; vec(omega_e) is row-major, so kappa_i omega_e = omega_e c_i is the matrix
kron(I_K, c_i^T) acting on vec(omega_e).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np


def spd(rng, n, shift=1.0):
    """Random symmetric positive-definite n x n matrix."""
    a = rng.standard_normal((n, n))
    return a @ a.T / n + shift * np.eye(n)


def kap(c):
    """Matrix of omega -> omega c on row-major vec(omega)."""
    return np.kron(np.eye(len(c)), np.asarray(c)[None, :])


def blockdiag(blocks):
    rows = sum(b.shape[0] for b in blocks); cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols)); r = c = 0
    for b in blocks:
        out[r:r + b.shape[0], c:c + b.shape[1]] = b; r += b.shape[0]; c += b.shape[1]
    return out


def coboundary(K, nV, edges, codes, c):
    r"""delta: Y -> H with (delta y)_e = T_e (y_i - y_j); V: X -> H with (V omega)_e = T_e omega_e c_i."""
    nH = sum(T.shape[0] for T in codes)
    D = np.zeros((nH, nV * K)); V = np.zeros((nH, len(edges) * K * K)); r = 0
    for e, (j, i) in enumerate(edges):
        T = codes[e]; n = T.shape[0]
        D[r:r + n, i * K:(i + 1) * K] += T
        D[r:r + n, j * K:(j + 1) * K] -= T
        V[r:r + n, e * K * K:(e + 1) * K * K] = T @ kap(c[i])
        r += n
    return D, V


def harmonic_form(D, R):
    r"""Pi_H = R^{-1} - R^{-1} delta (delta^T R^{-1} delta)^+ delta^T R^{-1}."""
    Ri = np.linalg.inv(R)
    return Ri - Ri @ D @ np.linalg.pinv(D.T @ Ri @ D) @ D.T @ Ri


def underlying_components(nV, edges):
    parent = list(range(nV))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for j, i in edges:
        a, b = find(j), find(i)
        if a != b:
            parent[a] = b
    return len({find(v) for v in range(nV)})


def simple_directed_cycles(nV, edges):
    """All simple directed cycles as lists of edge indices (parallel edges distinct). Brute force DFS."""
    out = []
    adj = {v: [] for v in range(nV)}
    for e, (j, i) in enumerate(edges):
        adj[j].append((i, e))
    for s in range(nV):
        stack = [(s, [s], [])]
        while stack:
            v, path, epath = stack.pop()
            for (w, e) in adj[v]:
                if w == s:
                    out.append(epath + [e])
                elif w > s and w not in path:
                    stack.append((w, path + [w], epath + [e]))
    return out


def sccs(nV, edges):
    """Strongly connected components by reachability closure (tiny graphs)."""
    reach = np.eye(nV, dtype=bool)
    for j, i in edges:
        reach[j, i] = True
    for k in range(nV):
        reach = reach | (reach[:, [k]] & reach[[k], :])
    comp = [-1] * nV; n = 0
    for v in range(nV):
        if comp[v] < 0:
            for w in range(nV):
                if reach[v, w] and reach[w, v]:
                    comp[w] = n
            n += 1
    return comp, n
