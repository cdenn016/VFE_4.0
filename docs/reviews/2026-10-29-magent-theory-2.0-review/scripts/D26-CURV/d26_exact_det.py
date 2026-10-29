r"""D26-CURV check 7: the exact Gaussian-content record versus the tangent model at weak anchors.

Exact record: o_e = T_e(exp(-omega_e) y_i - y_j) + noise, contents y ~ N(m, (eps Pbar)^-1). Integrating out y
gives -log Z(omega) = (1/2) log det J(omega) + record term + anchor, with J(omega) = eps Pbar + L(omega)^T R^-1 L(omega).
The tangent model replaces L(omega) by the constant coboundary delta, so its determinant is constant.

Prediction (Schur complement on the K-dimensional zero mode N = (1 x I_K)/sqrt(n) of delta):
(1/2) log det J(omega) = const + (1/2) log det(eps N^T Pbar N + G(omega)) + O(|omega|), with
G(omega)[u, u'] = (V_u omega)^T Pi_H (V_u' omega) / n and (V_u omega)_e = T_e omega_e u.
So the curvature of the determinant at omega = 0 is about tr((N^T Pbar N)^-1 G'')/eps, of order 1/eps, and
it acts on every content direction u, including omega with omega c = 0 that the tangent cost does not see.
"""
from d26_common import *
from scipy.linalg import expm

rng = np.random.default_rng(31)
K = 2; nV = 3
edges = [(0, 1), (1, 2), (2, 0), (1, 0)]
nE = len(edges)
codes = [rng.standard_normal((K, K)) + 2 * np.eye(K) for _ in edges]
R = blockdiag([spd(rng, K, 0.5) for _ in edges]); Ri = np.linalg.inv(R)
Pbar = blockdiag([spd(rng, K) for _ in range(nV)])
c = np.array([1.0, 0.0])
D, Vc = coboundary(K, nV, edges, codes, [c] * nV)
PiH = harmonic_form(D, R)
N = np.kron(np.ones((nV, 1)), np.eye(K)) / np.sqrt(nV)
P0 = N.T @ Pbar @ N


def Lmat(W):
    nH = K * nE; L = np.zeros((nH, nV * K))
    for e, (j, i) in enumerate(edges):
        L[e * K:(e + 1) * K, i * K:(i + 1) * K] = codes[e] @ expm(-W[e])
        L[e * K:(e + 1) * K, j * K:(j + 1) * K] -= codes[e]
    return L


def half_logdet_J(W, eps):
    L = Lmat(W)
    return 0.5 * np.linalg.slogdet(eps * Pbar + L.T @ Ri @ L)[1]


def G(W):
    cols = []
    for u in np.eye(K):
        _, Vu = coboundary(K, nV, edges, codes, [u] * nV)
        cols.append(Vu @ np.concatenate([w.ravel() for w in W]))
    M = np.array(cols).T
    return M.T @ PiH @ M / nV


dirs = {"omega_e = E22 on every edge (omega c = 0)": [np.array([[0., 0.], [0., 1.]])] * nE,
        "random direction": [rng.standard_normal((K, K)) for _ in range(nE)]}
for name, Wh in dirs.items():
    tan = (Vc @ np.concatenate([w.ravel() for w in Wh]))
    tangent_curv = tan @ PiH @ tan
    print(name, ": tangent weak-anchor curvature (V w)^T Pi_H (V w) =", round(float(tangent_curv), 6))
    for eps in (1e-2, 1e-3, 1e-4):
        t = 1e-2 * np.sqrt(eps)
        f = lambda s: half_logdet_J([s * w for w in Wh], eps)
        curv = (f(t) - 2 * f(0.0) + f(-t)) / t ** 2
        pred = np.trace(np.linalg.solve(P0, G(Wh))) / eps
        big = 10 * np.sqrt(eps)
        exact_big = f(big) - f(0.0)
        pred_big = 0.5 * (np.linalg.slogdet(eps * P0 + big ** 2 * G(Wh))[1] - np.linalg.slogdet(eps * P0)[1])
        print("   eps =", eps, ": determinant curvature at 0 =", round(float(curv), 3), " predicted tr(P0^-1 G)/eps =",
              round(float(pred), 3), " | at t = 10 sqrt(eps): exact increase", round(float(exact_big), 4),
              " predicted", round(float(pred_big), 4))

# Small-noise regime: R = rho Rbar with fixed anchors P = kappa Pbar. The tangent record precision times rho tends to
# V^T Pi_H(Rbar) V (the same cohomological form), while the ratio of the exact determinant curvature to the tangent
# record curvature scales like (common-mode prior variance)/|c|^2, small when the tangent approximation is valid.
print("small-noise regime, random direction, |c|^2 =", c @ c)
Rbar = R.copy()
Wh = dirs["random direction"]; wv = np.concatenate([w.ravel() for w in Wh])
PiHbar = harmonic_form(D, Rbar)
for kappa in (1.0, 10.0, 100.0):
    for rho in (1e-2, 1e-4):
        Rr = rho * Rbar; Rri = np.linalg.inv(Rr); P = kappa * Pbar
        CP = Rr + D @ np.linalg.inv(P) @ D.T
        tang = (Vc @ wv) @ np.linalg.solve(CP, Vc @ wv)
        def fdet(s):
            L = Lmat([s * w for w in Wh]); return 0.5 * np.linalg.slogdet(P + L.T @ Rri @ L)[1]
        t = 1e-4 * np.sqrt(rho)
        dcurv = (fdet(t) - 2 * fdet(0.0) + fdet(-t)) / t ** 2
        print("   kappa =", kappa, " rho =", rho, ": rho*tangent =", round(float(rho * tang), 5),
              " (V w)^T Pi_H(Rbar) (V w) =", round(float((Vc @ wv) @ PiHbar @ (Vc @ wv)), 5),
              " determinant/tangent curvature ratio =", round(float(dcurv / tang), 5))
