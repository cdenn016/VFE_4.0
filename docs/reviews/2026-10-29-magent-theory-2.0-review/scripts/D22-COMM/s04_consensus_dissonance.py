"""D22-COMM s04: curvature as dissonance (chapters/17_communication_links.tex:222-259).

Checks:
  (a) it1:consensus: consensus space = common fixed space of based loop holonomies on strongly connected graphs
      (3-cycle with SO(3) links: nontrivial holonomy, one-dimensional consensus space).
  (b) it1:frustration: least eigenvalue of B^T B; directed 3-cycle with planar rotations, holonomy angle 0.7.
  (c) without strong connectivity (acyclic 1->2, 2->3, 1->3) there are no closed walks, yet ker B = 0 for generic
      dictionaries: the "exactly when" of it1:frustration needs the theorem's strong-connectivity hypothesis.
  (d) it1:loop-comparisons: Q(H;Sigma) = 1/2 ||Sigma^{-1/2}(I-H)Sigma^{1/2}||_F^2 equals the trace form, is invariant
      under joint re-expression, quarter turn gives Q = 2 with KL(q||H#q) = 0; unipotent H = I + E12 has Q = 1/2 while
      its conjugates approach I; Q depends on Sigma, so it is not a class function of H alone.
  (e) lossy codes: grounded records do not identify the two-cycle holonomy (companion c17 line 175 qualifier).
"""
import numpy as np
from scipy.linalg import expm, null_space, sqrtm


def rot3(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    X = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return expm(ang * X)


def incidence(edges, n, K, w=None):
    """Rows: for edge (j -> i, theta), block sqrt(w)(I at i, -theta at j)."""
    B = np.zeros((len(edges) * K, n * K))
    for e, (j, i, th) in enumerate(edges):
        ww = 1.0 if w is None else w[e]
        B[e * K:(e + 1) * K, i * K:(i + 1) * K] = np.sqrt(ww) * np.eye(K)
        B[e * K:(e + 1) * K, j * K:(j + 1) * K] = -np.sqrt(ww) * th
    return B


# (a) directed 3-cycle 0->1->2->0 with SO(3) links; holonomy at 0 is th20 th12 th01... (receiver-first indices)
rng = np.random.default_rng(3)
A01 = rot3(rng.standard_normal(3), 0.9)
A12 = rot3(rng.standard_normal(3), 0.6)
A20 = rot3(rng.standard_normal(3), 1.1)
edges = [(0, 1, A01), (1, 2, A12), (2, 0, A20)]  # (sender j, receiver i, theta_ij)
H = A20 @ A12 @ A01
B = incidence(edges, 3, 3)
print("(a) holonomy rotation angle =", np.arccos((np.trace(H) - 1) / 2), " dim ker B =", null_space(B).shape[1],
      " dim Fix(H) =", null_space(H - np.eye(3)).shape[1])

# (b) planar directed 3-cycle with total holonomy angle 0.7
def rot2(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

edges2 = [(0, 1, rot2(0.2)), (1, 2, rot2(0.3)), (2, 0, rot2(0.2))]
B2 = incidence(edges2, 3, 2)
lam = np.linalg.eigvalsh(B2.T @ B2).min()
print("(b) least eigenvalue =", lam, " predicted 2-2cos(0.7/3) =", 2 - 2 * np.cos(0.7 / 3))
edges2f = [(0, 1, rot2(0.2)), (1, 2, rot2(0.3)), (2, 0, rot2(-0.5))]
print("(b) flat triangle: dim ker B =", null_space(incidence(edges2f, 3, 2)).shape[1])

# (c) acyclic graph: no closed walks, but generic dictionaries give ker B = 0
K = 2
th21 = expm(0.5 * rng.standard_normal((K, K))); th32 = expm(0.5 * rng.standard_normal((K, K))); th31 = expm(0.5 * rng.standard_normal((K, K)))
edges3 = [(0, 1, th21), (1, 2, th32), (0, 2, th31)]
B3 = incidence(edges3, 3, K)
print("(c) acyclic graph: closed walks = none (common fixed space of an empty family = R^2), dim ker B =",
      null_space(B3).shape[1], ", least eigenvalue of B^T B =", np.linalg.eigvalsh(B3.T @ B3).min())

# (d) dressed defect
def Q(Hm, S):
    Sh = np.real(sqrtm(S)); Shi = np.linalg.inv(Sh)
    return 0.5 * np.linalg.norm(Shi @ (np.eye(len(S)) - Hm) @ Sh, "fro") ** 2


def Qtr(Hm, S):
    X = np.eye(len(S)) - Hm
    return 0.5 * np.trace(X.T @ np.linalg.inv(S) @ X @ S)


Hq = np.array([[0.0, -1.0], [1.0, 0.0]])
print("(d) quarter turn: Q(H;I) =", Q(Hq, np.eye(2)), "; H H^T = I so KL(N(0,I)||H#N(0,I)) = 0:", np.allclose(Hq @ Hq.T, np.eye(2)))
Hg = expm(0.7 * rng.standard_normal((3, 3))); Sg = np.cov(rng.standard_normal((3, 10))) + 0.1 * np.eye(3)
g = expm(0.9 * rng.standard_normal((3, 3)))
print("(d) Q formula vs trace form:", Q(Hg, Sg) - Qtr(Hg, Sg),
      "; joint re-expression residual:", Q(np.linalg.inv(g) @ Hg @ g, np.linalg.inv(g) @ Sg @ np.linalg.inv(g).T) - Q(Hg, Sg))
Hu = np.array([[1.0, 1.0], [0.0, 1.0]])
for t in [1.0, 1e-2, 1e-4]:
    D = np.diag([t, 1.0])
    Hc = D @ Hu @ np.linalg.inv(D)
    print(f"(d) unipotent conjugate by diag({t},1): entries {Hc.ravel().round(6)}, tr = {np.trace(Hc)}, det = {np.linalg.det(Hc)},"
          f" Q(Hc;I) = {Q(Hc, np.eye(2)):.3e}")
print("(d) Q(I+E12; I) =", Q(Hu, np.eye(2)), "; Q(I+E12; diag(1,4)) =", Q(Hu, np.diag([1.0, 4.0])),
      " (same H, different Sigma: not a class function of H alone)")

# (e) lossy codes: two-cycle holonomy not identified by grounded records
T1 = np.array([[1.0, 0.0]]); T0 = np.array([[0.0, 1.0]])
s1 = np.array([[1.0, 0.0], [0.7, 1.5]])   # T1 s1 = T1, in Stab(T1)
s0 = np.array([[0.6, -0.4], [0.0, 1.0]])  # T0 s0 = T0, in Stab(T0)
print("(e) T1 s1 - T1 =", (T1 @ s1 - T1).ravel(), " T0 s0 - T0 =", (T0 @ s0 - T0).ravel())
# true dictionaries identity (flat two-cycle); alternative theta01 = s1, theta10 = s0 reproduces both edge laws
H_true = np.eye(2); H_alt = s1 @ s0
print("(e) effective codes unchanged:", np.allclose(T1 @ np.linalg.inv(s1), T1), np.allclose(T0 @ np.linalg.inv(s0), T0),
      "; holonomy true = I, alternative =", H_alt.round(3).tolist(), " Q(H_alt;I) =", Q(H_alt, np.eye(2)))
