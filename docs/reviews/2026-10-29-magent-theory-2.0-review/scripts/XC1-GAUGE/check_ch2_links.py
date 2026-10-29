"""XC1-GAUGE check 1: chapter 2 link algebra, reciprocal split, loop defect,
quiver-invariant scope, and the content of reference consistency.

Single-threaded, float64 / exact rationals. A numerical agreement supports an
identity at the tested instance; it does not prove it.
"""
import numpy as np
import scipy.linalg as sla
import sympy as sp

rng = np.random.default_rng(20261029)
out = []

def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)

# ---------------------------------------------------------------- (a) prop:int-reciprocal
def psqrt(M):
    return sla.sqrtm(M).real

T12 = np.array([[1.0, 1.0], [0.0, 1.0]])
T21 = np.array([[1.0, 0.0], [1.0, 1.0]])
N2 = T21 @ T12
O12 = T12 @ np.linalg.inv(psqrt(N2))
N1 = T12 @ T21
O21 = T21 @ np.linalg.inv(psqrt(N1))
claimed = np.array([[2.0, 1.0], [-1.0, 2.0]]) / np.sqrt(5)
p("(a) N2 =", N2.tolist())
p("    |Omega12 - claimed| =", np.abs(O12 - claimed).max(),
  " Omega21 Omega12 - I =", np.abs(O21 @ O12 - np.eye(2)).max(),
  " Omega12 orthogonal residual =", np.abs(O12.T @ O12 - np.eye(2)).max())
# random directed pair, no inverse imposed
worst = 0.0
for _ in range(200):
    A = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
    B = np.eye(3) + 0.4 * rng.standard_normal((3, 3))
    Nj = B @ A
    ev = np.linalg.eigvals(Nj)
    if np.any((np.abs(ev.imag) < 1e-9) & (ev.real <= 0)):
        continue
    Oij = A @ np.linalg.inv(psqrt(Nj))
    Oji = B @ np.linalg.inv(psqrt(A @ B))
    worst = max(worst, np.abs(Oji @ Oij - np.eye(3)).max())
p("    random directed pairs: max |Omega_ji Omega_ij - I| =", worst)

# ---------------------------------------------------------------- (b) prop:int-loop-defect
def Qdef(W, S):
    Sh = psqrt(S)
    M = np.linalg.inv(Sh) @ (np.eye(len(W)) - W) @ Sh
    return 0.5 * np.sum(M * M)

W = T12 @ T21
p("(b) Q12 for eq:int-nongaussian-loop =", Qdef(W, np.eye(2)), "(claimed 3/2)")
p("    shear I+E12 with Sigma=I:", Qdef(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2)), "(claimed 1/2)")
S = np.array([[2.0, 0.3], [0.3, 0.7]])
g = np.array([[1.3, -0.8], [0.4, 2.1]])
p("    passive/reference GL change residual:",
  abs(Qdef(g @ W @ np.linalg.inv(g), g @ S @ g.T) - Qdef(W, S)))

# ---------------------------------------------------------------- (c) quiver invariants: scope of Le Bruyn-Procesi
# Single edge j -> i, no oriented cycle. Vertex group SO(2) x SO(2) acting Theta -> g_i Theta g_j^{-1}.
def R(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

Th = np.array([[1.7, 0.2], [-0.5, 0.9]])
vals = []
for _ in range(5):
    gi, gj = R(rng.uniform(0, 6.3)), R(rng.uniform(0, 6.3))
    Tp = gi @ Th @ gj.T
    vals.append((np.trace(Tp.T @ Tp), np.linalg.det(Tp)))
vals = np.array(vals)
p("(c) single-edge quiver, SO(2) vertex groups: Tr(Th^T Th), det Th over random gauges:",
  np.round(vals, 12).tolist())
p("    spread of invariants:", np.ptp(vals[:, 0]), np.ptp(vals[:, 1]))
Th2 = np.array([[0.3, 0.0], [0.0, 2.0]])
p("    a second link has different values:", np.trace(Th2.T @ Th2), np.linalg.det(Th2),
  " => nonconstant invariants exist although the quiver has no oriented cycle")
gi = np.array([[2.0, 0.0], [0.0, 1.0]])
Tp = gi @ Th
p("    under GL x GL the same function is not invariant:", np.trace(Th.T @ Th), "->", np.trace(Tp.T @ Tp))

# ---------------------------------------------------------------- (d) reference consistency with re-expressed supplied data
# Rule without any metric: L = mu_i mu_j^T  (not equivariant under nonorthogonal h)
mu_i = np.array([1.0, 0.5]); mu_j = np.array([-0.3, 1.2])
h = np.array([[1.0, 0.7], [0.0, 1.5]])
hi = np.linalg.inv(h)
L = np.outer(mu_i, mu_j)
Lp = np.outer(hi @ mu_i, hi @ mu_j)
p("(d) bare rule L=mu_i mu_j^T: |L' - h^-1 L h| =", np.abs(Lp - hi @ L @ h).max(), "(fails reference consistency)")
# same rule with a declared 'supplied' metric M=I that is re-expressed as M' = h^T M h
M = np.eye(2)
LM = np.outer(mu_i, mu_j) @ M
LMp = np.outer(hi @ mu_i, hi @ mu_j) @ (h.T @ M @ h)
p("    same rule with supplied metric M re-expressed: |L' - h^-1 L h| =", np.abs(LMp - hi @ LM @ h).max(),
  "(passes: re-expressed background data dress the reference change)")
# the manuscript's moment family uses Sigma_j^{-1} instead: law-dressed, passes with no supplied data
Sj = np.array([[1.0, 0.2], [0.2, 0.5]])
Lm = np.outer(mu_i, mu_j) @ np.linalg.inv(Sj)
Lmp = np.outer(hi @ mu_i, hi @ mu_j) @ np.linalg.inv(hi @ Sj @ hi.T)
p("    moment family (Sigma_j^-1 from the law): |L' - h^-1 L h| =", np.abs(Lmp - hi @ Lm @ h).max())

# ---------------------------------------------------------------- (e) tied links and per-sector principal roots
# G = S^1, rho_q charge 1, rho_s charge 2 (rotation by 2t). Tied directed pair theta_ij = e^{ia}, theta_ji = e^{ib}.
def recip(Tij, Tji):
    Nj = Tji @ Tij
    return Tij @ np.linalg.inv(psqrt(Nj))

for (a, b) in [(0.3, 0.5), (1.2, 0.9), (1.4, 1.1)]:
    Oq = recip(R(a), R(b))
    Os = recip(R(2 * a), R(2 * b))
    tq = np.arctan2(Oq[1, 0], Oq[0, 0])
    ts = np.arctan2(Os[1, 0], Os[0, 0])
    mismatch = Os @ R(2 * tq).T  # sector-s reciprocal part vs rho_s of the sector-q reciprocal element
    p("(e) a=%.2f b=%.2f |a+b|=%.2f: Omega^q angle %.4f, Omega^s angle %.4f, rho_s(Omega^q) angle %.4f, mismatch angle %.4f"
      % (a, b, a + b, tq, ts, (2 * tq + np.pi) % (2 * np.pi) - np.pi,
         np.arctan2(mismatch[1, 0], mismatch[0, 0])))

# ---------------------------------------------------------------- (f) open:global-anchor obstruction
n = 3
v = 2 * np.pi / n
x = np.array([0.0, v, 2 * v])
p("(f) x=(e,v,v^2) rotated by v is a cyclic relabeling:",
  np.allclose(np.sort((x + v) % (2 * np.pi)), np.sort(x % (2 * np.pi))))
# common rotation by 2pi/n lies in H_I
p("    sum of common rotation by 2pi/n over n agents mod 2pi =", (n * v) % (2 * np.pi))

with open(__file__.replace(".py", "_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
