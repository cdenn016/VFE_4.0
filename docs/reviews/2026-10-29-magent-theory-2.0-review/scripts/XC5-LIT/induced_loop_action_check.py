r"""Numerical checks supporting XC5-LIT findings on induced loop actions and quotient distances.

Block A (Forman 1993): for a U(1) connection on a graph with unit edge weights, the
magnetic Laplacian Delta = D - A_theta satisfies
    det(Delta) = sum over cycle-rooted spanning forests F of prod_{cycles C in F} (2 - 2 cos hol(C)).
So integrating out a Gaussian content field whose precision depends on the links,
    -log int exp(-1/2 y^* Delta y) dy = (1/2) log det Delta + const,
is a holonomy-dependent loop action.  Checked on the theta graph (5 edges, 4 vertices),
where the five CRSFs are the spanning unicyclic subgraphs obtained by deleting one edge.

Block B (quotient distance): for SO(2) acting diagonally on columns of a 2 x n matrix,
    min_R 1/2 ||X - R Y||_F^2 = 1/2 (||X||^2 + ||Y||^2) - sqrt(C^2 + S^2),
the orbit distance used in ctx:f:eq:procrustes, and it equals 1/2 d_Q([X],[Y])^2 for the
quotient metric only where the minimizing rotation is unique (rho > 0).  Here we check the
Procrustes formula against brute-force minimization over a fine angle grid.

Single-threaded; runs in well under a second.
"""
from __future__ import annotations

import itertools

import numpy as np

rng = np.random.default_rng(20261029)

# ---------------- Block A: Forman's determinant formula on the theta graph
# vertices a=0, b=1, c=2, d=3; edges (oriented u->v) with phases
edges = [(0, 1), (0, 2), (2, 1), (0, 3), (3, 1)]
worst = 0.0
for trial in range(200):
    phase = rng.uniform(-np.pi, np.pi, size=len(edges))
    n = 4
    L = np.zeros((n, n), dtype=complex)
    for (u, v), t in zip(edges, phase):
        L[u, u] += 1.0
        L[v, v] += 1.0
        L[u, v] -= np.exp(1j * t)       # parallel transport v -> u convention
        L[v, u] -= np.exp(-1j * t)
    det = np.linalg.det(L).real

    def hol(cycle_edges: list[tuple[int, int]]) -> float:
        """Signed sum of edge phases along an oriented vertex cycle."""
        s = 0.0
        for (u, v) in cycle_edges:
            if (u, v) in edges:
                s += phase[edges.index((u, v))]
            else:
                s -= phase[edges.index((v, u))]
        return s

    # enumerate spanning connected unicyclic subgraphs (delete one edge, all are such here)
    total = 0.0
    for drop in range(len(edges)):
        sub = [e for k, e in enumerate(edges) if k != drop]
        # find the unique cycle by brute force over vertex cycles
        cyc_found = None
        for perm_len in (3, 4):
            for cyc in itertools.permutations(range(4), perm_len):
                if cyc[0] != min(cyc):
                    continue
                pairs = [(cyc[k], cyc[(k + 1) % perm_len]) for k in range(perm_len)]
                if all(((u, v) in sub) or ((v, u) in sub) for (u, v) in pairs):
                    cyc_found = pairs
                    break
            if cyc_found:
                break
        total += 2.0 - 2.0 * np.cos(hol(cyc_found))
    worst = max(worst, abs(det - total))
print(f"Block A: max |det(magnetic Laplacian) - Forman CRSF sum| over 200 random theta-graph connections = {worst:.2e}")

# ---------------- Block B: SO(2) Procrustes closed form vs grid
J = np.array([[0.0, -1.0], [1.0, 0.0]])
worstB = 0.0
for trial in range(200):
    X = rng.normal(size=(2, 4))
    Y = rng.normal(size=(2, 4))
    C = np.sum(X * Y)                       # <X, Y>_F  (manuscript: C = <Y,X>, target X, source Y)
    S = np.sum(X * (J @ Y))                 # <X, J Y>_F
    rho = np.hypot(C, S)
    closed = 0.5 * (np.sum(X**2) + np.sum(Y**2)) - rho
    th = np.linspace(-np.pi, np.pi, 200001)
    vals = 0.5 * (np.sum(X**2) + np.sum(Y**2)) - (C * np.cos(th) + S * np.sin(th))
    worstB = max(worstB, abs(vals.min() - closed))
print(f"Block B: max |grid minimum - Procrustes closed form| over 200 random pairs = {worstB:.2e}")

# ---------------- Block C: one loop with a mass term (the ch19 open-fluctuation setting)
# For the n-cycle with unit weights, U(1) holonomy theta and on-site precision p > 0,
#   det(p I + Delta_theta) = 2 cosh(n a) - 2 cos(theta),   2 cosh(a) = 2 + p.
# Real SO(2) links acting on R^2 contents square this (realification).  Hence the induced
# action (1/2) log det is minimized at flat holonomy and its loop dependence decays like
# exp(-n a) with loop length: an induced Wilson-type loop term with a definite sign.
worstC = 0.0
worstR = 0.0
for trial in range(200):
    n = int(rng.integers(3, 9))
    p = float(rng.uniform(0.05, 2.0))
    ph = rng.uniform(-np.pi, np.pi, size=n)
    theta = ph.sum()
    L = np.zeros((n, n), dtype=complex)
    LR = np.zeros((2 * n, 2 * n))
    for k in range(n):
        u, v = k, (k + 1) % n
        L[u, u] += 1.0; L[v, v] += 1.0
        L[u, v] -= np.exp(1j * ph[k]); L[v, u] -= np.exp(-1j * ph[k])
        Rk = np.array([[np.cos(ph[k]), -np.sin(ph[k])], [np.sin(ph[k]), np.cos(ph[k])]])
        LR[2*u:2*u+2, 2*u:2*u+2] += np.eye(2); LR[2*v:2*v+2, 2*v:2*v+2] += np.eye(2)
        LR[2*u:2*u+2, 2*v:2*v+2] -= Rk; LR[2*v:2*v+2, 2*u:2*u+2] -= Rk.T
    a = np.arccosh(1.0 + p / 2.0)
    closed = 2.0 * np.cosh(n * a) - 2.0 * np.cos(theta)
    detC = np.linalg.det(p * np.eye(n) + L).real
    detR = np.linalg.det(p * np.eye(2 * n) + LR)
    worstC = max(worstC, abs(detC - closed) / closed)
    worstR = max(worstR, abs(detR - closed**2) / closed**2)
print(f"Block C: max relative |det(pI+Delta_theta) - (2cosh(na)-2cos theta)| (U(1)) = {worstC:.2e}; "
      f"SO(2) realification vs square = {worstR:.2e}")
