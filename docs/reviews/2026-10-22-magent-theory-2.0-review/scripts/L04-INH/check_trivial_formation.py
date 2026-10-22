r"""Trivial instance of the letter of inh:recursive-formation (report L04-INH-01).

Four unit-variance Gaussian location agents q_i = N(x_i, 1) with uniform peer KL potential
E = sum_{i<j} KL(q_i||q_j) + KL(q_j||q_i) = sum_{i<j} (x_i - x_j)^2 and Euclidean (= Fisher) metric.
Flow: xdot = -grad E = -2 L x, L the complete-graph Laplacian; every pairwise difference decays as e^{-8t}.
Selector: blocks = connected components of {|x_i - x_j| < delta} among current parents
(parent distance = distance of block means). Retention: identity on the children tuple (exact).
Shows two pairs selected first and their union later, with no compression anywhere.
"""
import numpy as np
from scipy.linalg import expm

x0 = np.array([0.0, 0.3, 2.0, 2.5])
n = len(x0)
L = n * np.eye(n) - np.ones((n, n))
delta = 0.05
blocks = [[i] for i in range(n)]
events = []
for t in np.linspace(0, 1.2, 12001):
    x = expm(-2 * L * t) @ x0
    means = [x[B].mean() for B in blocks]
    merged = False
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            if abs(means[a] - means[b]) < delta:
                events.append((round(t, 4), blocks[a] + blocks[b]))
                blocks = [B for k, B in enumerate(blocks) if k not in (a, b)] + [blocks[a] + blocks[b]]
                merged = True
                break
        if merged:
            break
print("selection events (time, new block):", events)
# exactness of the identity retention is trivial; the parent of {0,1} keeps (x0, x1) entirely.
# pairwise differences: d/dt (x_i - x_j) = -8 (x_i - x_j), so a block relation once satisfied persists.
print("decay rate check:", np.allclose(expm(-2 * L * 0.1) @ np.array([1., -1, 0, 0]), np.exp(-0.8) * np.array([1., -1, 0, 0])))
