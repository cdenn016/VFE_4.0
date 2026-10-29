r"""D26-CURV check 1: the Gaussian induced action (it3:induced, it3:open-fluctuation).

(a) Independent quadrature of int exp(-E(y, omega)) dy for a two-agent scalar model, compared with
    the closed form S_marg including the affine term and the delta m shift.
(b) Schur complement of the joint precision versus Lambda_omega = s^-2 I + V^T C_P^-1 V; posterior mean.
(c) Weak-anchor limit C_P^{-1} -> Pi_H at rate O(eps).
(d) Quotient identity v^T Pi_H v = min_a ||v - delta a||^2_{R^-1}; Pi_H delta = 0.
(e) Rank of the limiting penalty: injective codes -> K(|E|-|V|+1); full-row-rank (surjective) codes ->
    dim H^1 = sum n_e - rank(delta).
(f) Anchor bound (delta a)^T C_P^-1 delta a <= a^T P a.
(g) Nonlinear fluctuation formula k - b^T J^-1 b / 2 + log det J / 2 by quadrature at a fixed omega.
"""
from d26_common import *
from scipy import integrate

rng = np.random.default_rng(291026)

# (a) scalar two-agent model, one edge 0 -> 1 (receiver 1), K = 1, code t, noise r, operating content c.
t, r, c, s2 = 0.7, 0.4, 1.3, 0.9
P = np.array([1.5, 0.8]); m = np.array([0.3, -1.1]); o = 0.25
def E(y0, y1, w):
    return 0.5 * (P[0] * (y0 - m[0])**2 + P[1] * (y1 - m[1])**2) + 0.5 * w**2 / s2 \
        + 0.5 * (o - t * (y1 - y0 - w * c))**2 / r
def S_quad(w):
    val, err = integrate.dblquad(lambda y1, y0: np.exp(-E(y0, y1, w)), -12, 12, -12, 12,
                                 epsabs=1e-13, epsrel=1e-12)
    return -np.log(val)
D = np.array([[-t, t]]); Vw = t * c
CP = r + (D @ np.diag(1 / P) @ D.T)[0, 0]
def S_formula(w):
    u = o + Vw * w - (D @ m)[0]
    return 0.5 * w**2 / s2 + 0.5 * u**2 / CP
const_exact = 0.5 * np.log(P[0] * P[1]) - 0.5 * np.log(r) + 0.5 * np.log(CP) - np.log(2 * np.pi)
ws = [-1.0, 0.0, 0.6, 1.7]
diffs = [S_quad(w) - S_formula(w) for w in ws]
print("(a) quadrature S(w) - closed form, at w =", ws, ":", np.round(diffs, 12))
print("    predicted constant (1/2)log det P - (1/2)log r + (1/2)log C_P - log 2pi =", round(const_exact, 12),
      " max deviation from it:", max(abs(d - const_exact) for d in diffs))
# dropping the affine record term or the delta m shift changes the omega dependence:
alt = [S_quad(w) - (0.5 * w**2 / s2 + 0.5 * (Vw * w)**2 / CP) for w in ws]
print("    same with the affine term dropped: spread over w =", max(alt) - min(alt))

# (b)-(f) random directed multigraph with two-cycles, K = 2.
K = 2; nV = 4
edges = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 1), (0, 2), (2, 0)]
nE = len(edges)
codes = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in edges]          # injective (square, invertible)
c_common = rng.standard_normal(K); cs = [c_common] * nV
Ps = [spd(rng, K) for _ in range(nV)]; Pm = blockdiag(Ps)
Rs = [spd(rng, K, 0.5) for _ in edges]; Rm = blockdiag(Rs)
m = rng.standard_normal(nV * K); o = rng.standard_normal(nE * K)
D, V = coboundary(K, nV, edges, codes, cs)
nX = V.shape[1]
# joint precision of (y, omega)
L = np.hstack([D, -V])                       # residual o - L x with x = (y, omega)
Ri = np.linalg.inv(Rm)
J = blockdiag([Pm, np.eye(nX) / s2]) + L.T @ Ri @ L
eta = np.concatenate([Pm @ m, np.zeros(nX)]) + L.T @ Ri @ o
ny = nV * K
schur = J[ny:, ny:] - J[ny:, :ny] @ np.linalg.solve(J[:ny, :ny], J[:ny, ny:])
CPm = Rm + D @ np.linalg.inv(Pm) @ D.T
Lam = np.eye(nX) / s2 + V.T @ np.linalg.solve(CPm, V)
print("(b) |Schur complement - Lambda_omega| =", np.abs(schur - Lam).max())
xstar = np.linalg.solve(J, eta)
w_from_marg = -np.linalg.solve(Lam, V.T @ np.linalg.solve(CPm, o - D @ m))
print("    |posterior dictionary mean - argmin S_marg| =", np.abs(xstar[ny:] - w_from_marg).max())

# (c) weak-anchor limit
Pbar = blockdiag([spd(rng, K) for _ in range(nV)])
PiH = harmonic_form(D, Rm)
for eps in (1e-2, 1e-4, 1e-6):
    Ce = Rm + D @ np.linalg.inv(eps * Pbar) @ D.T
    print("(c) eps =", eps, ": |C_P^-1 - Pi_H| =", np.abs(np.linalg.inv(Ce) - PiH).max())

# (d) quotient identity and Pi_H delta = 0
v = V @ rng.standard_normal(nX)
Rh = np.linalg.cholesky(Ri)                  # Ri = Rh Rh^T
a_ls = np.linalg.lstsq(Rh.T @ D, Rh.T @ v, rcond=None)[0]
print("(d) |v^T Pi_H v - min_a ||v - delta a||^2_{R^-1}| =",
      abs(v @ PiH @ v - (v - D @ a_ls) @ Ri @ (v - D @ a_ls)), "  |Pi_H delta| =", np.abs(PiH @ D).max())

# (e) rank
lim = V.T @ PiH @ V
print("(e) injective codes: rank =", np.linalg.matrix_rank(lim, tol=1e-9), " K(|E|-|V|+1) =", K * (nE - nV + 1))
for trial in range(3):
    K3 = 3
    codes3 = [rng.standard_normal((int(rng.integers(1, K3 + 1)), K3)) for _ in edges]   # full row rank n_e <= K
    D3, V3 = coboundary(K3, nV, edges, codes3, [rng.standard_normal(K3)] * nV)
    R3 = blockdiag([spd(rng, T.shape[0], 0.5) for T in codes3])
    lim3 = V3.T @ harmonic_form(D3, R3) @ V3
    nH = sum(T.shape[0] for T in codes3); rk = np.linalg.matrix_rank(D3, tol=1e-9)
    print("    surjective codes, widths", [T.shape[0] for T in codes3], ": rank =",
          np.linalg.matrix_rank(lim3, tol=1e-9), " dim H^1 = sum n_e - rank delta =", nH - rk,
          " [K(|E|-|V|+1) =", K3 * (nE - nV + 1), "]")

# (f) anchor bound
worst = -np.inf
for _ in range(2000):
    a = rng.standard_normal(ny)
    worst = max(worst, (D @ a) @ np.linalg.solve(CPm, D @ a) - a @ Pm @ a)
print("(f) max over 2000 draws of (delta a)^T C_P^-1 delta a - a^T P a =", worst, "(must be <= 0)")

# (g) nonlinear fluctuation formula by quadrature (n = 2), omega-dependent J(w), b(w), k(w)
w = 0.37
Jw = np.array([[2.0 + w**2, 0.3 - w], [0.3 - w, 1.0 + 0.5 * np.cos(w)]])
bw = np.array([np.sin(w), 1.0 - w]); kw = 0.2 * w**2
val, _ = integrate.dblquad(lambda y1, y0: np.exp(-(kw - bw @ np.array([y0, y1]) +
                           0.5 * np.array([y0, y1]) @ Jw @ np.array([y0, y1]))), -15, 15, -15, 15,
                           epsabs=1e-13, epsrel=1e-12)
lhs = -np.log(val)
rhs = kw - 0.5 * bw @ np.linalg.solve(Jw, bw) + 0.5 * np.log(np.linalg.det(Jw)) - np.log(2 * np.pi)
print("(g) quadrature -log int e^-E - [k - b^T J^-1 b/2 + log det J/2 - (n/2) log 2pi] =", lhs - rhs)
