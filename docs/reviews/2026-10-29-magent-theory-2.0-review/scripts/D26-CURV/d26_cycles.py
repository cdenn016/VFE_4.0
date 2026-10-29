r"""D26-CURV check 2: directed walks, cycle spanning, and record-favored classes (it3:induced, it3:dag-example).

(a) The DAG 1->2, 2->3, 1->3 with v = (0,0,1): squared residual 1/3.
(b) Directed cycles exist but do not span: {1->2, 2->1, 2->3, 1->3}, v = (0,0,0,1).
(c) Strongly connected, but the declared family contains only two-cycles: triangle circulation unseen.
(d) Random digraphs: directed cycles span the cycle space iff the condensation multigraph
    (strongly connected components contracted, inter-component edges kept with multiplicity) is a forest,
    i.e. #inter-SCC edges = #SCC - #weak components. Strong connectivity => spanning.
(e) Nonzero records: the weak-anchor minimizer carries a nonzero loop sum and a nonzero H^1 class.
"""
from d26_common import *
import itertools

def residual_sq(nV, edges, v, R=None):
    D, _ = coboundary(1, nV, edges, [np.eye(1)] * len(edges), [np.ones(1)] * nV)
    R = np.eye(len(edges)) if R is None else R
    return float(v @ harmonic_form(D, R) @ v)

def walk_chain(nE, cyc):
    z = np.zeros(nE)
    for e in cyc:
        z[e] += 1
    return z

# (a)
print("(a) DAG residual =", residual_sq(3, [(0, 1), (1, 2), (0, 2)], np.array([0., 0., 1.])),
      " directed cycles:", simple_directed_cycles(3, [(0, 1), (1, 2), (0, 2)]))

# (b)
eb = [(0, 1), (1, 0), (1, 2), (0, 2)]
vb = np.array([0., 0., 0., 1.])
cyc_b = simple_directed_cycles(3, eb)
print("(b) directed cycles:", cyc_b, " their sums of v:", [float(sum(vb[e] for e in cy)) for cy in cyc_b],
      " residual =", residual_sq(3, eb, vb), " cycle-space dim =", len(eb) - 3 + 1,
      " span of directed cycles =", np.linalg.matrix_rank(np.array([walk_chain(4, cy) for cy in cyc_b])))

# (c) triangle with both directions; declared family = the three two-cycles only
ec = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]
vc = np.array([1., 1., 1., -1., -1., -1.])
two = [[0, 3], [1, 4], [2, 5]]
print("(c) two-cycle sums:", [float(vc[a] + vc[b]) for a, b in two], " triangle sum:", float(vc[0] + vc[1] + vc[2]),
      " residual =", residual_sq(3, ec, vc), " span of two-cycles =",
      np.linalg.matrix_rank(np.array([walk_chain(6, cy) for cy in two])), " cycle-space dim =", 6 - 3 + 1)

# (d) random digraphs
rng = np.random.default_rng(7)
agree = total = sc_total = sc_span = 0
for _ in range(3000):
    n = int(rng.integers(3, 6)); mE = int(rng.integers(2, 9))
    E = []
    for _ in range(mE):
        j, i = rng.choice(n, 2, replace=False); E.append((int(j), int(i)))
    cyc = simple_directed_cycles(n, E)
    zdim = len(E) - n + underlying_components(n, E)
    span = np.linalg.matrix_rank(np.array([walk_chain(len(E), cy) for cy in cyc])) if cyc else 0
    comp, ncomp = sccs(n, E)
    inter = sum(1 for (j, i) in E if comp[j] != comp[i])
    forest = (inter == ncomp - underlying_components(n, E))
    agree += ((span == zdim) == forest); total += 1
    if ncomp == 1:
        sc_total += 1; sc_span += (span == zdim)
print("(d) criterion 'condensation multigraph is a forest' agrees with 'directed cycles span' in",
      agree, "of", total, "random digraphs;", "strongly connected cases spanning:", sc_span, "of", sc_total)

# (e) directed triangle 0->1->2->0, unit codes and noise, c = 1, weak anchors, o = (1, 0, 0)
et = [(0, 1), (1, 2), (2, 0)]
D, V = coboundary(1, 3, et, [np.eye(1)] * 3, [np.ones(1)] * 3)
PiH = harmonic_form(D, np.eye(3))
s2 = 1.0; o = np.array([1., 0., 0.])
w_star = -np.linalg.solve(np.eye(3) / s2 + V.T @ PiH @ V, V.T @ PiH @ o)
print("(e) weak-anchor minimizer omega* =", np.round(w_star, 6), " loop sum =", round(float(w_star.sum()), 6),
      " class norm^2 of V omega* =", round(float((V @ w_star) @ PiH @ (V @ w_star)), 6),
      " at o = 0 the minimizer is 0")
# finite anchors with conflicting self means and zero records: the delta m shift alone moves omega
P = np.eye(3); mm = np.array([0., 1., 3.])
CP = np.eye(3) + D @ np.linalg.inv(P) @ D.T
w_fin = -np.linalg.solve(np.eye(3) / s2 + V.T @ np.linalg.solve(CP, V), V.T @ np.linalg.solve(CP, -D @ mm))
print("    finite anchors, o = 0, self means", mm, ": omega* =", np.round(w_fin, 6),
      " loop sum =", round(float(w_fin.sum()), 6))
