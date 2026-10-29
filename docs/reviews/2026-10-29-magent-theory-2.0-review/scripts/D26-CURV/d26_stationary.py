r"""D26-CURV check 3: stationary dictionaries (it3:rank-one, it3:dissonance).

(a) Global joint with nonzero records and per-receiver operating contents: the closed form
    omega_e = s2 T^T (R + s2|c_i|^2 T T^T)^-1 (T Delta - o) c_i^T and rank <= 1.
(b) Plug-in and law-level per-agent fixed points (best-response linear systems).
(c) Reciprocal potential computed from the Gaussian KL between message laws, minimized numerically
    in the means at fixed covariances; compared with the replacement R -> (R + T Sigma_j T^T)/beta.
(d) Exact ordered product of exp(omega_e) at a stationary state with common c: (H - I) annihilates
    the orthogonal complement of c exactly, H = I + h c^T; equal gains cancel sum(omega) but not the
    exact product.
"""
from d26_common import *
from scipy.linalg import expm
from scipy.optimize import minimize

rng = np.random.default_rng(1029)
K = 2; nV = 3
edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)]
nE = len(edges); mm = K * K
codes = [rng.standard_normal((1 + (e % 2), K)) for e in range(nE)]
Rs = [spd(rng, T.shape[0], 0.5) for T in codes]
Ps = [spd(rng, K) for _ in range(nV)]
m = [2.0 * rng.standard_normal(K) for _ in range(nV)]
s2 = 0.7


def gain_omega(T, R, c, Delta, o):
    a = s2 * (c @ c)
    return s2 * np.outer(T.T @ np.linalg.solve(R + a * T @ T.T, T @ Delta - o), c)


def solve_global(cs, os_, Rs_):
    D, V = coboundary(K, nV, edges, codes, cs)
    Rm = blockdiag(Rs_); Ri = np.linalg.inv(Rm); L = np.hstack([D, -V]); o = np.concatenate(os_)
    J = blockdiag([blockdiag(Ps), np.eye(nE * mm) / s2]) + L.T @ Ri @ L
    eta = np.concatenate([blockdiag(Ps) @ np.concatenate(m), np.zeros(nE * mm)]) + L.T @ Ri @ o
    x = np.linalg.solve(J, eta)
    y = x[:nV * K].reshape(nV, K)
    W = [x[nV * K + e * mm: nV * K + (e + 1) * mm].reshape(K, K) for e in range(nE)]
    return y, W


def check(y, W, cs, os_, Rs_, label):
    err = sv = 0.0
    for e, (j, i) in enumerate(edges):
        pred = gain_omega(codes[e], Rs_[e], cs[i], y[i] - y[j], os_[e])
        err = max(err, np.abs(W[e] - pred).max())
        sv = max(sv, np.linalg.svd(W[e], compute_uv=False)[1])
    print("   ", label, ": max |omega_e - closed form| =", err, " max second singular value =", sv)


cs = [rng.standard_normal(K) for _ in range(nV)]
os_ = [rng.standard_normal(T.shape[0]) for T in codes]
y, W = solve_global(cs, os_, Rs)
print("(a) global joint")
check(y, W, cs, os_, Rs, "global, o != 0, per-receiver c_i")


def per_agent(cs, os_, law):
    """Best-response fixed point of the per-agent models (plug-in or law-level convolution)."""
    Sig = [np.zeros((K, K)) for _ in range(nV)]
    for _ in range(500):
        new = []
        for i in range(nV):
            ins = [e for e, (j, h) in enumerate(edges) if h == i]
            A = np.zeros((K + len(ins) * mm, K + len(ins) * mm)); A[:K, :K] += Ps[i]
            for k, e in enumerate(ins):
                j = edges[e][0]; T = codes[e]
                Rt = Rs[e] + (T @ Sig[j] @ T.T if law else 0)
                Le = np.zeros((T.shape[0], A.shape[0])); Le[:, :K] = T
                Le[:, K + k * mm: K + (k + 1) * mm] = -T @ kap(cs[i])
                A += Le.T @ np.linalg.solve(Rt, Le)
                A[K + k * mm: K + (k + 1) * mm, K + k * mm: K + (k + 1) * mm] += np.eye(mm) / s2
            new.append(np.linalg.inv(A)[:K, :K])
        done = max(np.abs(a - b).max() for a, b in zip(new, Sig)) < 1e-15
        Sig = new
        if done:
            break
    n = nV * K + nE * mm; M = np.zeros((n, n)); rhs = np.zeros(n)
    yi = lambda i: slice(i * K, (i + 1) * K)
    wi = lambda e: slice(nV * K + e * mm, nV * K + (e + 1) * mm)
    for i in range(nV):
        M[yi(i), yi(i)] += Ps[i]; rhs[yi(i)] += Ps[i] @ m[i]
    Rts = []
    for e, (j, i) in enumerate(edges):
        T = codes[e]; Rt = Rs[e] + (T @ Sig[j] @ T.T if law else 0); Rts.append(Rt); Rti = np.linalg.inv(Rt)
        Lfull = np.zeros((T.shape[0], n)); Lfull[:, yi(i)] = T; Lfull[:, yi(j)] = -T
        Lfull[:, wi(e)] = -T @ kap(cs[i])
        for blk in (yi(i), wi(e)):
            M[blk, :] += Lfull[:, blk].T @ Rti @ Lfull; rhs[blk] += Lfull[:, blk].T @ Rti @ os_[e]
        M[wi(e), wi(e)] += np.eye(mm) / s2
    x = np.linalg.solve(M, rhs)
    return x[:nV * K].reshape(nV, K), [x[wi(e)].reshape(K, K) for e in range(nE)], Rts


print("(b) per-agent")
for law in (False, True):
    for oz in (True, False):
        oo = [np.zeros(T.shape[0]) for T in codes] if oz else os_
        y, W, Rts = per_agent(cs, oo, law)
        check(y, W, cs, oo, Rts, ("law-level" if law else "plug-in") + (", o = 0" if oz else ", o != 0"))

Sig = [spd(rng, K, 0.3) for _ in range(nV)]
Vd = [spd(rng, mm, 0.2) * 0.1 for _ in range(nE)]
beta = rng.uniform(0.3, 1.5, nE)


def kl(a, A, b, B):
    """KL(N(a, A) to N(b, B)) for Gaussians."""
    Bi = np.linalg.inv(B)
    return 0.5 * (np.trace(Bi @ A) - len(a) + (a - b) @ Bi @ (a - b) + np.log(np.linalg.det(B) / np.linalg.det(A)))


def potential(z):
    """Reciprocal potential at fixed covariances: self KL, row-weighted averaged message KL, dictionary KL mean part."""
    mu = z[:nV * K].reshape(nV, K)
    A = [z[nV * K + e * mm: nV * K + (e + 1) * mm] for e in range(nE)]
    val = 0.0
    for i in range(nV):
        val += kl(mu[i], Sig[i], m[i], np.linalg.inv(Ps[i]))
    for e, (j, i) in enumerate(edges):
        T = codes[e]; Si = T @ Sig[i] @ T.T + Rs[e]; Sj = T @ Sig[j] @ T.T + Rs[e]
        kk = T @ kap(cs[i])
        val += beta[e] * (kl(T @ mu[i] - kk @ A[e], Si, T @ mu[j], Sj)
                          + 0.5 * np.trace(np.linalg.solve(Sj, kk @ Vd[e] @ kk.T)))
        val += 0.5 * A[e] @ A[e] / s2
    return val


z0 = rng.standard_normal(nV * K + nE * mm)
res = minimize(potential, z0, method="BFGS", options={"gtol": 1e-11, "maxiter": 20000})
mu = res.x[:nV * K].reshape(nV, K)
A = [res.x[nV * K + e * mm: nV * K + (e + 1) * mm].reshape(K, K) for e in range(nE)]
Rrec = [(codes[e] @ Sig[edges[e][0]] @ codes[e].T + Rs[e]) / beta[e] for e in range(nE)]
print("(c) reciprocal potential, BFGS success:", res.success, " gradient norm:", np.linalg.norm(res.jac))
check(mu, A, cs, [np.zeros(T.shape[0]) for T in codes], Rrec, "reciprocal, R to (R + T Sigma_j T^T)/beta, o = 0")

c = rng.standard_normal(K); cs_c = [c] * nV
y, W = solve_global(cs_c, os_, Rs)
cperp = np.array([-c[1], c[0]])
for cyc in ([0, 1], [0, 2, 4], [5, 3, 1]):
    H = np.eye(K)
    for e in cyc:
        H = expm(W[e]) @ H
    S = sum(W[e] for e in cyc)
    print("(d) loop", cyc, ": |(H - I) c_perp| =", np.abs((H - np.eye(K)) @ cperp).max(),
          " |H - I| =", round(float(np.abs(H - np.eye(K)).max()), 6),
          " |sum omega c| =", round(float(np.abs(S @ c).max()), 6))
codes_orig = codes
n_z = nV * K + nE * mm
z_ref = np.zeros(n_z)
f0 = potential(z_ref)
Hq = np.zeros((n_z, n_z)); g1 = np.zeros(n_z)
I_ = np.eye(n_z)
fp = np.array([potential(I_[a]) for a in range(n_z)]); fm = np.array([potential(-I_[a]) for a in range(n_z)])
for a in range(n_z):
    Hq[a, a] = fp[a] + fm[a] - 2 * f0
    g1[a] = 0.5 * (fp[a] - fm[a])
    for b in range(a + 1, n_z):
        Hq[a, b] = Hq[b, a] = 0.5 * (potential(I_[a] + I_[b]) - fp[a] - fp[b] + f0 + potential(-I_[a] - I_[b]) - fm[a] - fm[b] + f0)
z_star = np.linalg.solve(Hq, -g1)
mu2 = z_star[:nV * K].reshape(nV, K)
A2 = [z_star[nV * K + e * mm: nV * K + (e + 1) * mm].reshape(K, K) for e in range(nE)]
print("(c') exact quadratic solve of the reciprocal potential")
check(mu2, A2, cs, [np.zeros(T.shape[0]) for T in codes], Rrec, "reciprocal (exact solve), R to (R + T Sigma_j T^T)/beta, o = 0")
T1 = rng.standard_normal((2, K)); R1 = spd(rng, 2, 0.5)
codes = [T1] * nE; Rs_eq = [R1] * nE
m_dir = [3.0 * rng.standard_normal(K) for _ in range(nV)]
for scale in (1.0, 0.5, 0.25, 0.125):
    m = [scale * v for v in m_dir]
    y, W = solve_global(cs_c, [np.zeros(2)] * nE, Rs_eq)
    cyc = [0, 2, 4]
    H = np.eye(K)
    for e in cyc:
        H = expm(W[e]) @ H
    S = sum(W[e] for e in cyc)
    wn = max(np.abs(W[e]).max() for e in cyc)
    print("    equal gains, self-mean scale", scale, ": |sum omega| =", np.abs(S).max(),
          " |H - I| =", round(float(np.abs(H - np.eye(K)).max()), 8), " max|omega_e|^2 =", round(float(wn ** 2), 8),
          " |(H - I) c_perp| =", np.abs((H - np.eye(K)) @ cperp).max())
