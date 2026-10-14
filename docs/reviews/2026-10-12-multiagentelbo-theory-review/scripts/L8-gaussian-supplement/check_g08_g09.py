"""L8 checks for g08 (nonflat relaxation, barycenters, quotient volume) and g09 (operator coarse-graining)."""
import numpy as np
import sympy as sp
from numpy.linalg import inv, slogdet, eigvalsh, det
from scipy.linalg import expm, sqrtm, null_space
from scipy.optimize import minimize
from common import rng, out, rep, note, spd, gl_plus, kl

# example loop weights of g07 series example
Ta = sp.Matrix([[1, 1], [0, 1]]); Wa = sp.Matrix([[1, 0], [0, 2]]); Wb = sp.Matrix([[2, 1], [1, 2]]); Tb = sp.Matrix([[1, 0], [1, 1]])
W2h = (Wa.inv() + Ta * Wb.inv() * Ta.T).inv(); W3h = (sp.eye(2) + Tb * Tb.T).inv()
note(f"g07 example loop weights 11*W2 = {(11*W2h).tolist()}, 5*W3 = {(5*W3h).tolist()}; Q_loop2 = {((sp.eye(2)-Ta).T*W2h*(sp.eye(2)-Ta)).tolist()}, Q_loop3 = {((sp.eye(2)-Tb).T*W3h*(sp.eye(2)-Tb)).tolist()}")

# prop:cg-effective-support (v) triangle
W12 = np.diag([1., 0]); T12 = np.diag([1., 2]); W23 = np.diag([0, 1.]); T23 = np.diag([2., 1]); W31 = np.eye(2); T31 = np.eye(2)
res = [np.abs(sqrtm(W) @ (np.eye(2) - T)).max() for W, T in [(W12, T12), (W23, T23), (W31, T31)]]
L = np.zeros((6, 6))
for (i, j, W, T) in [(0, 1, W12, T12), (1, 2, W23, T23), (2, 0, W31, T31)]:
    R = np.zeros((2, 6)); R[:, 2*i:2*i+2] += np.eye(2); R[:, 2*j:2*j+2] += -T; L += R.T @ W @ R
ns = null_space(L)
note(f"g08 triangle: weighted residuals {res}; loop product {T12 @ T23 @ T31}; kernel dim {ns.shape[1]}; constant-section check {np.abs(L @ np.kron(np.ones(3), [1., 0])).max()}, {np.abs(L @ np.kron(np.ones(3), [0, 1.])).max()}")

# prop:cg-gaussian-fixed-relaxation
d = 4; G = spd(d, 1e2); Bk = rng.standard_normal((d, 2)); L = rng.standard_normal((d, d)); L = L @ L.T
L = L - L @ Bk @ inv(Bk.T @ L @ Bk) @ Bk.T @ L  # make ker L = span(Bk)
Pi0 = Bk @ inv(Bk.T @ G @ Bk) @ Bk.T @ G
z0 = rng.standard_normal(d); Gi = inv(G)
zt = lambda t: expm(-t * Gi @ L) @ z0
rep("g08 relaxation limit = Pi_0 z0", np.abs(zt(200.) - Pi0 @ z0).max(), np.abs(z0).max())
Gh = sqrtm(G).real; ev = eigvalsh(inv(Gh) @ L @ inv(Gh)); rplus = min(e for e in ev if e > 1e-9)
viol = 0
for t in [0.1, 1., 5.]:
    lhs = np.sqrt((zt(t) - Pi0 @ z0) @ G @ (zt(t) - Pi0 @ z0)); rhs = np.exp(-rplus * t) * np.sqrt((z0 - Pi0 @ z0) @ G @ (z0 - Pi0 @ z0))
    viol = max(viol, lhs - rhs)
note(f"g08 relaxation decay bound max violation {viol:.3e} (<=0 expected)")
g = gl_plus(d, 10.); Gp = inv(g).T @ G @ inv(g); Lp = inv(g).T @ L @ inv(g)
rep("g08 flow congruence", np.abs(expm(-inv(Gp) @ Lp) @ (g @ z0) - g @ zt(1.)).max(), np.abs(g @ z0).max())

# prop:cg-gaussian-forward-kl-barycenter, compact holonomy (nonorthogonal conjugate of a finite rotation group)
K = 2; mus = [rng.standard_normal(K) for _ in range(3)]; Ss = [spd(K) for _ in range(3)]; w = np.array([0.2, 0.5, 0.3])
mb = sum(wi * m for wi, m in zip(w, mus)); Sb = sum(wi * (S + np.outer(m - mb, m - mb)) for wi, m, S in zip(w, mus, Ss))
obj = lambda mu, S: sum(wi * kl(m, Si, mu, S) for wi, m, Si in zip(w, mus, Ss))
score = 0.5 * (slogdet(Sb)[1] - sum(wi * slogdet(S)[1] for wi, S in zip(w, Ss)))
rep("g08 barycenter score", abs(obj(mb, Sb) - score), score)
P = gl_plus(K, 5.); Rq = np.array([[0, -1], [1, 0.]]); grp = [P @ np.linalg.matrix_power(Rq, k) @ inv(P) for k in range(4)]
Xi = sum(wi * (S + np.outer(m, m)) for wi, m, S in zip(w, mus, Ss))
muH = sum(gk @ mb for gk in grp) / 4; XiH = sum(gk @ Xi @ gk.T for gk in grp) / 4; SH = XiH - np.outer(muH, muH)
inv_ok = max(np.abs(gk @ SH @ gk.T - SH).max() for gk in grp)
scoreH = 0.5 * (slogdet(SH)[1] - sum(wi * slogdet(S)[1] for wi, S in zip(w, Ss)))
rep("g08 Haar-projected barycenter score (nonorthogonal conjugated C4)", abs(obj(muH, SH) - scoreH), scoreH)
note(f"   invariance of Sigma_Hol under group {inv_ok:.3e}; |det g| = {[round(abs(det(gk)), 12) for gk in grp]}")
# invariant-constrained optimum check: perturb within invariant family; invariant SPD for this group = P (a I + b J) P^T
best = np.inf
def inv_cov(x):
    a, b = np.exp(x[0]), x[1]
    return P @ (a * np.eye(2)) @ P.T  # symmetric invariant forms of rotation group: P (aI) P^T only
for _ in range(200):
    a = np.exp(rng.standard_normal()); best = min(best, obj(np.zeros(2), P @ (a * np.eye(2)) @ P.T))
note(f"   min over 200 invariant candidates {best:.6f} >= Haar score {scoreH:.6f}")
# fixed-covariance pairwise identity
S0 = spd(K); lhs = sum(wi * kl(m, S0, mb, S0) for wi, m in zip(w, mus))
skl = lambda a, b: 0.5 * (kl(a, S0, b, S0) + kl(b, S0, a, S0))
rhs = sum(w[i] * w[j] * skl(mus[i], mus[j]) for i in range(3) for j in range(i + 1, 3))
rep("g08 fixed-covariance pairwise identity", abs(lhs - rhs), lhs)

# prop:cg-quotient-metric
for sizes in [[1, 1], [2, 3, 1], [4, 1, 2, 2]]:
    n = np.array(sizes, float); N = n.sum(); G = np.diag(n) - np.outer(n, n) / N
    ev = eigvalsh(G); pd = np.prod(ev[ev > 1e-12])
    Sh = np.zeros((int(N), len(n))); c = 0
    for I, s in enumerate(sizes):
        Sh[c:c+s, I] = 1; c += s
    x = rng.standard_normal(len(n)); one = np.ones(int(N))
    tau = -one @ (Sh @ x) / N
    note(f"g08 quotient metric sizes {sizes}: pdet {pd:.12f} vs formula {len(n)*np.prod(n)/N:.12f}; min_tau identity residual {abs(np.sum((Sh @ x + tau)**2) - x @ G @ x):.2e}")

# g09 prop:grg-interior-law-step and spectral bounds
n = 6; Lf = spd(n); S = np.kron(np.array([[1, 0], [1, 0], [0, 1.]]), np.eye(2)); Lc = S.T @ Lf @ S; Pl = inv(Lc) @ S.T @ Lf
for z in [0.3, 1.0]:
    cov = z**2 * Pl @ inv(Lf) @ Pl.T + z * (1 - z) * inv(Lc)
    rep(f"g09 interior step kernel covariance zeta={z}", np.abs(cov - z * inv(Lc)).max())
Se = np.kron(np.array([[1, 0], [1, 0], [0, 1], [0, 1.]]), np.eye(2)); Lf8 = spd(8); Lce = Se.T @ Lf8 @ Se
note(f"g09 spectral bounds (equal clusters b=2): b*lmin {2*eigvalsh(Lf8).min():.4f} <= {eigvalsh(Lce).min():.4f}; {eigvalsh(Lce).max():.4f} <= b*lmax {2*eigvalsh(Lf8).max():.4f}")
print("\n".join(out))
