r"""GAP05-S3 s3: it2:gauge-general under linear and nonlinear dictionary actions.

(a) G = GL^+(K), K = 2, 3, records T(theta^{-1} y_i - y_j) + noise with Laplace and Student-t
    noise: build theta2_e = s R_e (scaled rotation, inside the exponential image) with
    theta2_e v_e = y2_i, and compare coded discrepancies and record densities.
(b) G = SO(2) acting on contents in S^1 by rotation, von Mises records on the discrepancy
    y_i - w_e - y_j: w2_e = w_e + dy_i - dy_j works for every displacement.
(c) G = SO(2) acting linearly on R^2 (does not contain GL^+): injective code T = I_2 and the
    rank-one code T = (1, 0). Minimal record mismatch over all rotations and the fraction of
    random displacements that the dictionaries can absorb; groups R_{>0} x SO(2) and SL(2).
(d) Tangent version: dimension of the belief projection of the gauge space for gl(2) and so(2).
"""
import numpy as np
from scipy.linalg import logm, expm
from scipy.stats import t as student_t, laplace, vonmises

rng = np.random.default_rng(7)

def rot_to(u, w):
    """A rotation in span(u, w) taking u/|u| to w/|w| (identity on the complement)."""
    K = len(u)
    a = u / np.linalg.norm(u); b = w / np.linalg.norm(w)
    c = np.clip(a @ b, -1, 1)
    bp = b - c * a
    if np.linalg.norm(bp) < 1e-14:
        if c > 0:
            return np.eye(K)
        e = np.zeros(K); e[np.argmin(abs(a))] = 1.0
        bp = e - (e @ a) * a
    bp /= np.linalg.norm(bp)
    ang = np.arccos(c)
    R = np.eye(K) + np.sin(ang) * (np.outer(bp, a) - np.outer(a, bp)) + (np.cos(ang) - 1) * (np.outer(a, a) + np.outer(bp, bp))
    return R

def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

print("== (a) GL+(K): explicit absorbing dictionaries, non-Gaussian noise ==")
for K in (2, 3):
    worst_disc = worst_dens = worst_map = 0.0
    min_det = np.inf
    max_imag_log = 0.0
    for trial in range(200):
        n_e = rng.integers(1, K + 1)
        T = rng.standard_normal((n_e, K))
        yi, yj = rng.standard_normal(K), rng.standard_normal(K)
        th = expm(0.5 * rng.standard_normal((K, K)))
        yi2, yj2 = yi + rng.standard_normal(K), yj + rng.standard_normal(K)
        v = np.linalg.solve(th, yi) + yj2 - yj
        s = np.linalg.norm(yi2) / np.linalg.norm(v)
        th2 = s * rot_to(v, yi2)
        worst_map = max(worst_map, np.linalg.norm(th2 @ v - yi2))
        min_det = min(min_det, np.linalg.det(th2))
        L = logm(th2)
        max_imag_log = max(max_imag_log, np.abs(np.imag(L)).max() if np.iscomplexobj(L) else 0.0)
        d1 = T @ (np.linalg.solve(th, yi) - yj)
        d2 = T @ (np.linalg.solve(th2, yi2) - yj2)
        worst_disc = max(worst_disc, np.abs(d1 - d2).max())
        o = rng.standard_normal(n_e) * 2
        for dist in (laplace, student_t):
            args = () if dist is laplace else (3,)
            l1 = np.prod(dist.pdf(o - d1, *args)); l2 = np.prod(dist.pdf(o - d2, *args))
            worst_dens = max(worst_dens, abs(l1 - l2) / l1)
    print("K=%d: |theta2 v - y2_i| <= %.1e, min det theta2 = %.3f, max |Im logm theta2| = %.1e," % (K, worst_map, min_det, max_imag_log))
    print("      coded discrepancy mismatch <= %.1e, relative record-density mismatch (Laplace, t_3) <= %.1e" % (worst_disc, worst_dens))

print("== (b) SO(2) on S^1, von Mises records on y_i - w_e - y_j ==")
worst = 0.0
for trial in range(1000):
    yi, yj, w = rng.uniform(-np.pi, np.pi, 3)
    dyi, dyj = rng.uniform(-np.pi, np.pi, 2)
    w2 = w + dyi - dyj
    o = rng.uniform(-np.pi, np.pi)
    l1 = vonmises.pdf(o, 3.0, loc=(yi - w - yj))
    l2 = vonmises.pdf(o, 3.0, loc=((yi + dyi) - w2 - (yj + dyj)))
    worst = max(worst, abs(l1 - l2) / l1)
print("max relative record-density mismatch over 1000 random displacements: %.1e (no exceptional set)" % worst)
print("law level: s2 gives dC_e/dmu_0 = -dC_e/dmu_1 = -dC_e/dalpha_e = 0.162284769479, so C_e depends on")
print("(mu_0, mu_1, alpha_e) only through mu_0 - alpha_e - mu_1 for von Mises contents and dictionary laws.")

print("== (c) SO(2) acting linearly on R^2 ==")
def best_mismatch(T, th, yi, yj, yi2, yj2, ngrid=4000):
    target = T @ (np.linalg.solve(th, yi) - yj)
    best = np.inf
    for a in np.linspace(-np.pi, np.pi, ngrid, endpoint=False):
        best = min(best, np.linalg.norm(T @ (rot(a).T @ yi2 - yj2) - target))
    return best

ntr = 400
absorb_I = 0
absorb_r1 = 0
mis_I = []
for trial in range(ntr):
    yi, yj = rng.standard_normal(2), rng.standard_normal(2)
    th = rot(rng.uniform(-np.pi, np.pi))
    yi2, yj2 = yi + 0.7 * rng.standard_normal(2), yj + 0.7 * rng.standard_normal(2)
    v = th.T @ yi + yj2 - yj
    mis = abs(np.linalg.norm(yi2) - np.linalg.norm(v))
    mis_I.append(mis)
    absorb_I += mis < 1e-12
    absorb_r1 += abs(v[0]) <= np.linalg.norm(yi2)
print("closed form vs brute force over all rotations (injective code; rank-one code):")
for trial in range(5):
    yi, yj = rng.standard_normal(2), rng.standard_normal(2)
    th = rot(rng.uniform(-np.pi, np.pi))
    yi2, yj2 = yi + 0.7 * rng.standard_normal(2), yj + 0.7 * rng.standard_normal(2)
    v = th.T @ yi + yj2 - yj
    bI = best_mismatch(np.eye(2), th, yi, yj, yi2, yj2)
    b1 = best_mismatch(np.array([[1.0, 0.0]]), th, yi, yj, yi2, yj2)
    print("   injective %.4f vs ||y2_i|-|v|| %.4f;   rank-one %.4f vs max(0,|v_1|-|y2_i|) %.4f"
          % (bI, abs(np.linalg.norm(yi2) - np.linalg.norm(v)), b1, max(0.0, abs(v[0]) - np.linalg.norm(yi2))))
print("injective code: absorbable displacements %d/%d; median minimal mismatch %.3f" % (absorb_I, ntr, np.median(mis_I)))
print("rank-one code: absorbable displacements %d/%d (an inequality, not a subspace)" % (absorb_r1, ntr))
worst_co = 0.0
worst_sl = 0.0
for trial in range(200):
    v, w = rng.standard_normal(2), rng.standard_normal(2)
    g_co = (np.linalg.norm(w) / np.linalg.norm(v)) * rot_to(v, w)
    worst_co = max(worst_co, np.linalg.norm(g_co @ v - w))
    u = np.array([-v[1], v[0]]) / (v @ v)
    z = np.array([-w[1], w[0]]) / (w @ w)
    g_sl = np.column_stack([w, z]) @ np.linalg.inv(np.column_stack([v, u]))
    worst_sl = max(worst_sl, np.linalg.norm(g_sl @ v - w) + abs(np.linalg.det(g_sl) - 1))
print("R_{>0} x SO(2): max |g v - w| = %.1e;  SL(2): max |g v - w| + |det g - 1| = %.1e" % (worst_co, worst_sl))

print("== (d) tangent gauge space: belief projection dimension, one edge, K = 2 ==")
J = np.array([[0.0, -1.0], [1.0, 0.0]])
def belief_proj_dim(T, c, basis):
    A = np.hstack([T, -T, np.column_stack([-T @ (B @ c) for B in basis])])
    U, S, Vt = np.linalg.svd(A)
    rank = int(np.sum(S > 1e-12))
    N = Vt[rank:].T
    return np.linalg.matrix_rank(N[:4], tol=1e-10)
gl = [np.outer(np.eye(2)[a], np.eye(2)[b]) for a in range(2) for b in range(2)]
so = [J]
for T, Tn in [(np.eye(2), 'T = I_2'), (np.array([[1.0, 0.0]]), 'T = (1,0)')]:
    for c, cn in [(np.array([1.0, 0.0]), 'c = e_1'), (np.array([1.0, 1.0]) / np.sqrt(2), 'c = (1,1)/sqrt2')]:
        print("  %-9s %-14s gl(2): %d of 4   so(2): %d of 4" % (Tn, cn, belief_proj_dim(T, c, gl), belief_proj_dim(T, c, so)))
