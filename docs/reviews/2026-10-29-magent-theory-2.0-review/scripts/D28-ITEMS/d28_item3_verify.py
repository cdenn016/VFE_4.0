r"""D28-ITEMS independent checks of Item 3 (curvature and hierarchy) claims.

Single-threaded, tiny sizes; sympy for exact algebra.

Blocks
  A  it3:dag-example: residual 1/3, and the rank of the limiting quadratic part on that DAG.
  B  A connected digraph WITH a directed cycle on which directed cycles do not span the cycle space:
     zero directed-loop sums but positive limiting quadratic part (G1 beyond the acyclic case);
     the rank K(|E|-|V|+1) holds without the cycle-spanning hypothesis.
  C  A strongly connected non-complete digraph: zero iff declared directed-loop sums of omega c vanish.
  D  it3:rigid normalizer counterexample: two copies of (1/3) z^(-2/3) on (0,1).
  E  it3:cap convolution identity (scalar, symbolic).
  F  it3:rank-one under the tangent reciprocal potential: stationary omega_e c = F_e Delta_e with
     R_e replaced by (R_e + T_e Sigma_j T_e^T)/beta_e.
  G  it3:flat-wall counterexample without the internal path hypothesis.
"""
import numpy as np
import sympy as sp
from scipy.linalg import null_space

rng = np.random.default_rng(29103)


def coboundary(edges, nV, K, codes):
    r"""(delta y)_e = T_e (y_i - y_j) for e = (j -> i), edges given as (sender j, receiver i)."""
    rows = []
    for (j, i), T in zip(edges, codes):
        Rw = np.zeros((T.shape[0], nV * K))
        Rw[:, i * K:(i + 1) * K] += T
        Rw[:, j * K:(j + 1) * K] -= T
        rows.append(Rw)
    return np.vstack(rows)


def reinterp(edges, K, codes, c):
    r"""(V omega)_e = T_e omega_e c on row-major vec(omega)."""
    kap = np.kron(np.eye(K), c[None, :])
    nE = len(edges); mm = K * K
    rows = []
    for e, T in enumerate(codes):
        Rw = np.zeros((T.shape[0], nE * mm))
        Rw[:, e * mm:(e + 1) * mm] = T @ kap
        rows.append(Rw)
    return np.vstack(rows)


def pi_H(delta, R):
    Ri = np.linalg.inv(R)
    return Ri - Ri @ delta @ np.linalg.pinv(delta.T @ Ri @ delta) @ delta.T @ Ri


# ---------------- A ----------------
edges = [(0, 1), (1, 2), (0, 2)]          # 1->2, 2->3, 1->3 in the document's 1-based labels
d = coboundary(edges, 3, 1, [np.eye(1)] * 3)
v = np.array([0.0, 0.0, 1.0])
print("A  DAG residual v^T Pi_H v =", v @ pi_H(d, np.eye(3)) @ v, " (document: 1/3)")
Vm = reinterp(edges, 1, [np.eye(1)] * 3, np.array([1.0]))
print("   rank of limiting quadratic on the DAG =", np.linalg.matrix_rank(Vm.T @ pi_H(d, np.eye(3)) @ Vm, tol=1e-9),
      " K(|E|-|V|+1) =", 1 * (3 - 3 + 1))

# ---------------- B ----------------
K = 2
edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3)]   # directed triangle plus a sink vertex 3
nV = 4
codes = [rng.standard_normal((2, K)) + 2 * np.eye(2) for _ in edges]   # injective
c = rng.standard_normal(K)
d = coboundary(edges, nV, K, codes); Vm = reinterp(edges, K, codes, c)
R = np.eye(2 * len(edges))
Lim = Vm.T @ pi_H(d, R) @ Vm
mm = K * K
tri = [0, 1, 2]
Bc = np.zeros((K, len(edges) * mm))
kap = np.kron(np.eye(K), c[None, :])
for e in tri:
    Bc[:, e * mm:(e + 1) * mm] += kap
Zdir = null_space(Bc)                    # omega with zero directed-loop sum applied to c
vals = np.linalg.eigvalsh(Zdir.T @ Lim @ Zdir)
print("B  digraph with one directed cycle, 5 edges, 4 vertices:")
print("   rank of limiting quadratic =", np.linalg.matrix_rank(Lim, tol=1e-9), " K(|E|-|V|+1) =", K * (5 - 4 + 1))
print("   max of the quadratic on {directed-loop sums of omega c = 0} (normalized) =", vals.max(),
      " -> positive, so zero directed-loop sums do not imply zero penalty")

# ---------------- C ----------------
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 0)]   # strongly connected: cycles 0-1-2-3-0 and 0-1-2-0
codes = [rng.standard_normal((2, K)) + 2 * np.eye(2) for _ in edges]
d = coboundary(edges, 4, K, codes); Vm = reinterp(edges, K, codes, c)
Lim = Vm.T @ pi_H(d, np.eye(2 * len(edges))) @ Vm
loops = [[0, 1, 2, 3], [0, 1, 4]]
Bc = np.zeros((K * len(loops), len(edges) * mm))
for g, gam in enumerate(loops):
    for e in gam:
        Bc[g * K:(g + 1) * K, e * mm:(e + 1) * mm] += kap
Zdir = null_space(Bc)
print("C  strongly connected digraph: rank =", np.linalg.matrix_rank(Lim, tol=1e-9), " K(|E|-|V|+1) =", K * 2,
      " max quadratic on zero loop sums =", np.abs(Zdir.T @ Lim @ Zdir).max(),
      " dim of that set =", Zdir.shape[1], " expected |E|K^2 - rank =", len(edges) * mm - K * 2)

# ---------------- D ----------------
z, e_ = sp.symbols("z epsilon", positive=True)
p = sp.Rational(1, 3) * z ** sp.Rational(-2, 3)
print("D  int_0^1 p =", sp.integrate(p, (z, 0, 1)), "; int_eps^1 p^2 =", sp.simplify(sp.integrate(p ** 2, (z, e_, 1))),
      " -> diverges as eps -> 0")

# ---------------- E ----------------
x, w, tau, s2, cA = sp.symbols("x omega tau sigma2 c_A", positive=True)
integrand = sp.exp(-tau / 2 * (x - w * cA) ** 2) * sp.exp(-w ** 2 / (2 * s2)) / sp.sqrt(2 * sp.pi * s2)
res = sp.simplify(sp.integrate(integrand, (w, -sp.oo, sp.oo)))
logres = sp.simplify(sp.expand_log(sp.log(res), force=True))
prec = sp.simplify(-2 * sp.diff(logres, x, 2) / 2)
print("E  effective precision of the integrated cross-scale factor:", sp.simplify(prec),
      " document:", sp.simplify(1 / (1 / tau + s2 * cA ** 2)), " difference:", sp.simplify(prec - 1 / (1 / tau + s2 * cA ** 2)))

# ---------------- F ----------------
K = 2; mm = K * K
edges = [(1, 0), (0, 1), (2, 1), (0, 2)]
nA = 3
codes = [rng.standard_normal((1 + e % 2, K)) for e in range(len(edges))]
Rn = [np.eye(T.shape[0]) * (0.5 + e) for e, T in enumerate(codes)]
Sig = [np.eye(K) * (0.3 + 0.2 * i) for i in range(nA)]
beta = [0.4, 0.9, 0.6, 1.3]
cvec = rng.standard_normal(K)
s2n = 0.8
m = [rng.standard_normal(K) * 2 for _ in range(nA)]
P = [np.eye(K) * 1.5 for _ in range(nA)]
kapm = np.kron(np.eye(K), cvec[None, :])
n = nA * K + len(edges) * mm
H = np.zeros((n, n)); b = np.zeros(n)
for i in range(nA):
    H[i * K:(i + 1) * K, i * K:(i + 1) * K] += P[i]; b[i * K:(i + 1) * K] += P[i] @ m[i]
for e, ((j, i), T) in enumerate(zip(edges, codes)):
    off = nA * K + e * mm
    H[off:off + mm, off:off + mm] += np.eye(mm) / s2n
    L = np.zeros((T.shape[0], n)); L[:, i * K:(i + 1) * K] += T; L[:, j * K:(j + 1) * K] -= T
    L[:, off:off + mm] -= T @ kapm
    Seff = (Rn[e] + T @ Sig[j] @ T.T) / beta[e]          # reciprocal widened noise, registered value zero
    H += L.T @ np.linalg.solve(Seff, L)
x_st = np.linalg.solve(H, b)
err = 0.0
for e, ((j, i), T) in enumerate(zip(edges, codes)):
    off = nA * K + e * mm
    om = x_st[off:off + mm].reshape(K, K)
    Dl = x_st[i * K:(i + 1) * K] - x_st[j * K:(j + 1) * K]
    Reff = (Rn[e] + T @ Sig[j] @ T.T) / beta[e]
    a = s2n * (cvec @ cvec)
    Fe = a * T.T @ np.linalg.solve(Reff + a * T @ T.T, T)
    err = max(err, np.abs(om @ cvec - Fe @ Dl).max(), np.linalg.svd(om, compute_uv=False)[1])
print("F  reciprocal mean-stationary point: max(|omega c - F Delta|, second singular value) =", err)

# ---------------- G ----------------
th = {("b1", "a1"): 1.0, ("a1", "b1"): 1.0, ("b2", "a2"): 2.0, ("a2", "b2"): 0.5}
print("G  two-cycle holonomies:", th[("a1", "b1")] * th[("b1", "a1")], th[("a2", "b2")] * th[("b2", "a2")],
      " J->I bridge dictionaries:", th[("b1", "a1")], th[("b2", "a2")])

# ---------------- H: general rank formula (no injectivity, no common c, no cycle-spanning) ----------------
# rank of V^T Pi_H V = sum_e rank T_e - K|V| + dim H^0, with H^0 = {y : T_e(y_i - y_j) = 0 on every edge}.
K = 3; mm = K * K
for trial in range(4):
    nV = 4
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (3, 2)][: 4 + trial % 3]
    codes = [rng.standard_normal((1 + (e + trial) % 3, K)) for e in range(len(edges))]
    cs = [rng.standard_normal(K) for _ in range(nV)]
    d = coboundary(edges, nV, K, codes)
    rows = []
    for e, ((j, i), T) in enumerate(zip(edges, codes)):
        Rw = np.zeros((T.shape[0], len(edges) * mm))
        Rw[:, e * mm:(e + 1) * mm] = T @ np.kron(np.eye(K), cs[i][None, :])
        rows.append(Rw)
    Vm = np.vstack(rows)
    Lim = Vm.T @ pi_H(d, np.eye(d.shape[0])) @ Vm
    h0 = nV * K - np.linalg.matrix_rank(d, tol=1e-9)
    pred = sum(np.linalg.matrix_rank(T) for T in codes) - K * nV + h0
    print("H  trial", trial, ": rank =", np.linalg.matrix_rank(Lim, tol=1e-9), " sum rank T_e - K|V| + dim H0 =", pred)
