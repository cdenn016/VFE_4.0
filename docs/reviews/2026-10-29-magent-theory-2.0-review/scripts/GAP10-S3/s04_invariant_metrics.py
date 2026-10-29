r"""GAP10-S3 s04: invariant metrics on SL(K) link-tuple strata (ibt:group / ibt:alignment).

A. Free but non-proper: SL(2) conjugation on pairs. z=(A,E12), z'=(A,E21), A=diag(1,2) have
   trivial infinitesimal stabilizer; z_n=(A,E12+t^2 E21) -> z and g_n z_n g_n^{-1} -> z' with
   g_n = diag(t,1/t) unbounded.  (No invariant Riemannian metric on any invariant open set
   containing z and z'; proof in the report.)
B. Canonical Kempf-Ness (KN) dressed metric on the absolutely irreducible locus:
   S_z = argmin_{P>0, det P=1} f_z(P),  f_z(P) = sum_e Tr(P^{-1} A_e P A_e^T),
   G_z(dz,dz) = sum_e Tr(S_z^{-1} dA_e S_z dA_e^T).  Checks: equivariance S_{h^{-1}zh} ~ h^{-1}S_z h^{-T},
   invariance of G, blow-up of cond(S) toward the reducible locus.
C. Alignment connection (ibt:eq:augmented-normal) of SL(2) link fields with the KN metric
   preserves the KN fiber metric: dS + aS + Sa^T = 0, holonomy orthogonal for S^{-1}.
   Contrast: an invariant metric dressed by the KN point of a *different* norm is not metric.
D. ibt:group for g = sl(K) with M = H_X^{-1}: the minimizer is the traceless part of -dX Q_X.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

t0 = time.time()
rng = np.random.default_rng(20261029)
np.set_printoptions(precision=6, linewidth=140)


def sl_basis(K):
    B = []
    for i in range(K):
        for j in range(K):
            if i != j:
                E = np.zeros((K, K)); E[i, j] = 1; B.append(E)
    for i in range(K - 1):
        E = np.zeros((K, K)); E[i, i] = 1; E[i + 1, i + 1] = -1; B.append(E)
    return B


def Jmat(z, basis):
    r"""Columns: vec of ([xi, A_e])_e for xi in basis (infinitesimal conjugation)."""
    return np.stack([np.concatenate([(xi @ A - A @ xi).ravel() for A in z]) for xi in basis], axis=1)


print("=== A. free but non-proper SL(2) pairs ===")
A = np.diag([1.0, 2.0])
E12 = np.array([[0, 1.0], [0, 0]]); E21 = E12.T
bas2 = sl_basis(2)
for name, z in [("z=(A,E12)", [A, E12]), ("z'=(A,E21)", [A, E21])]:
    s = np.linalg.svd(Jmat(z, bas2), compute_uv=False)
    print(f"  {name}: singular values of J_z = {s}  (injective: {s.min() > 1e-12})")
for t in [1e-1, 1e-2, 1e-3, 1e-4]:
    zn = [A, E12 + t ** 2 * E21]
    g = np.diag([t, 1 / t])
    gz = [g @ X @ np.linalg.inv(g) for X in zn]
    gz_on_z = [g @ X @ np.linalg.inv(g) for X in [A, E12]]
    sv = np.linalg.svd(Jmat(zn, bas2), compute_uv=False).min()
    print(f"  t={t:.0e}: |z_n - z| = {t**2:.1e}, |g_n z_n - z'| = {np.linalg.norm(gz[1]-E21):.1e}, "
          f"|g_n| = {np.linalg.norm(g,2):.1e}, min sv J(z_n) = {sv:.3f}, g_n.z -> (A, {gz_on_z[1][0,1]:.1e} E12)")


print("  group-valued version in SL(2)^2: z = (D, I+E12), z' = (D, I+E21), D = diag(2, 1/2)")
D_ = np.diag([2.0, 0.5])
for name, z in [("z ", [D_, np.eye(2) + E12]), ("z'", [D_, np.eye(2) + E21])]:
    s = np.linalg.svd(Jmat(z, bas2), compute_uv=False)
    print(f"    {name}: dets {[round(np.linalg.det(X), 12) for X in z]}, singular values of J_z = {np.round(s, 6)}")
for t in [1e-1, 1e-2, 1e-3]:
    Kn = np.array([[1 + t ** 2, 1.0], [t ** 2, 1.0]])
    g = np.diag([t, 1 / t])
    gK = g @ Kn @ np.linalg.inv(g)
    gz1 = g @ (np.eye(2) + E12) @ np.linalg.inv(g)
    print(f"    t={t:.0e}: det K_n = {np.linalg.det(Kn):.12f}, |K_n - (I+E12)| = {np.linalg.norm(Kn - np.eye(2) - E12):.1e}, "
          f"|g K_n g^-1 - (I+E21)| = {np.linalg.norm(gK - np.eye(2) - E21):.1e}, g D g^-1 = D, "
          f"g (I+E12) g^-1 = I + {gz1[0,1]:.1e} E12 -> I (not I+E21)")


def sym_basis(K):
    B = []
    for i in range(K):
        for j in range(i + 1, K):
            E = np.zeros((K, K)); E[i, j] = E[j, i] = 1; B.append(E)
    for i in range(K - 1):
        E = np.zeros((K, K)); E[i, i] = 1; E[i + 1, i + 1] = -1; B.append(E)
    return B


def kn(z, K, w=None, P0=None, tol=1e-15, maxit=60):
    r"""KN point by Newton's method on the symmetric space, with exact gradient and Hessian.

    In balanced coordinates B_e = L^{-1} A_e L (P = L L^T), f(L e^Y L^T) = sum_e w_e Tr(e^{-Y} B_e e^{Y} B_e^T),
    gradient g_k = sum_e w_e Tr(sigma_k (B_e^T B_e - B_e B_e^T)), Hessian K_kl = sum_e w_e Tr([sigma_k,B_e][sigma_l,B_e]^T).
    """
    if w is None:
        w = np.ones(len(z))
    sb = sym_basis(K)
    P = np.eye(K) if P0 is None else P0.copy()
    it = 0
    for it in range(maxit):
        L = np.linalg.cholesky(P)
        Li = np.linalg.inv(L)
        Bz = [Li @ A_ @ L for A_ in z]
        g = np.array([sum(we * np.trace(sg @ (B_.T @ B_ - B_ @ B_.T)) for we, B_ in zip(w, Bz)) for sg in sb])
        C = [[sg @ B_ - B_ @ sg for B_ in Bz] for sg in sb]
        Hs = np.array([[sum(we * np.trace(C[k][e] @ C[l][e].T) for e, we in enumerate(w)) for l in range(len(sb))]
                       for k in range(len(sb))])
        y = -np.linalg.solve(Hs, g)
        Y = sum(yk * sg for yk, sg in zip(y, sb))
        P = L @ expm(Y) @ L.T
        P = (P + P.T) / 2
        if np.linalg.norm(g) < tol * max(1.0, sum(np.linalg.norm(A_) ** 2 for A_ in z)):
            break
    S = P / np.linalg.det(P) ** (1 / K)

    class R:
        pass
    res = R(); res.jac = g; res.nit = it
    return S, res


def Gmetric(z, dz, S):
    Si = np.linalg.inv(S)
    return sum(np.trace(Si @ dA @ S @ dA.T) for dA in dz)


print("\n=== B. KN-dressed invariant metric on the absolutely irreducible locus ===")
for K in (2, 3):
    z = [rng.normal(size=(K, K)) for _ in range(2)]
    S, res = kn(z, K)
    h = rng.normal(size=(K, K)) + 2 * np.eye(K)
    hi = np.linalg.inv(h)
    zh = [hi @ X @ h for X in z]
    Sh, _ = kn(zh, K)
    pred = hi @ S @ hi.T
    pred /= np.linalg.det(pred) ** (1 / K)
    dz = [rng.normal(size=(K, K)) for _ in range(2)]
    dzh = [hi @ X @ h for X in dz]
    print(f"  K={K}: |S_(h^-1 z h) - h^-1 S_z h^-T| (det-normalized) = {np.linalg.norm(Sh - pred):.2e}; "
          f"G invariance |G_zh(h^-1 dz h) - G_z(dz)| = {abs(Gmetric(zh, dzh, Sh) - Gmetric(z, dz, S)):.2e} "
          f"(G = {Gmetric(z, dz, S):.6f}); grad norm {np.linalg.norm(res.jac):.1e}")
    # balanced condition at S: in the gauge g with g g^T = S, sum [B,B^T] = 0
    L = np.linalg.cholesky(S)
    zb = [np.linalg.inv(L) @ X @ L for X in z]
    mu = sum(B @ B.T - B.T @ B for B in zb)
    print(f"        moment map at the balanced representative |sum [B,B^T]| = {np.linalg.norm(mu):.2e}")
print("  blow-up toward the reducible pair (A, E12): z_eps = (A, E12 + eps E21)")
for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
    S, _ = kn([A, E12 + eps * E21], 2)
    print(f"    eps={eps:.0e}: cond(S) = {np.linalg.cond(S):.3e}  (predicted 1/eps = {1/eps:.1e})")

print("\n=== C. alignment connection with the KN metric is metric for S^{-1} ===")
K = 2
L1c = [np.array([[0.2, 0.7], [-0.5, -0.2]]), np.array([[0.3, -0.2], [0.1, -0.3]]), np.array([[-0.1, 0.2], [0.3, 0.1]])]
L2c = [np.array([[0.6, 0.1], [0.3, -0.6]]), np.array([[0.0, 0.4], [-0.2, 0.0]]), np.array([[0.2, 0.1], [0.0, -0.2]])]


def links(c):
    L1 = L1c[0] + c[0] * L1c[1] + c[1] * L1c[2] + 0.5 * c[0] * c[1] * L1c[1] @ L1c[2]
    L2 = L2c[0] + c[0] * L2c[1] + c[1] * L2c[2] + 0.3 * c[0] ** 2 * L2c[2]
    L1 = L1 - np.trace(L1) / 2 * np.eye(2); L2 = L2 - np.trace(L2) / 2 * np.eye(2)
    return [expm(L1), expm(L2)]


_warm = {}


def connection(c, wts=(1.0, 1.0), kn_w=(1.0, 1.0), hd=1e-6):
    r"""Alignment connection of ibt:eq:augmented-normal for G = SL(2) acting by conjugation on two
    link fields, with invariant target metric sum_e wts_e Tr(S^{-1} dK_e S dK_e^T), S = KN point of
    the norm with weights kn_w (self-balanced iff kn_w = wts)."""
    z = links(c)
    key = kn_w
    S, _ = kn(z, 2, w=np.array(kn_w), P0=_warm.get(key))
    _warm[key] = S
    Si = np.linalg.inv(S)
    Jz = [[xi @ Kk - Kk @ xi for Kk in z] for xi in bas2]

    def ip(u, v):
        return sum(w_ * np.trace(Si @ a_ @ S @ b_.T) for w_, a_, b_ in zip(wts, u, v))
    Gm = np.array([[ip(Jz[i], Jz[j]) for j in range(3)] for i in range(3)])
    a = []
    for k in range(2):
        dc = np.zeros(2); dc[k] = hd
        zp, zm = links(c + dc), links(c - dc)
        dZ = [(p_ - m_) / (2 * hd) for p_, m_ in zip(zp, zm)]
        rhs = -np.array([ip(dZ, Jz[i]) for i in range(3)])
        coef = np.linalg.solve(Gm, rhs)
        a.append(sum(cf * xi for cf, xi in zip(coef, bas2)))
    return a, S


c0 = np.array([0.05, -0.05])
rad, nst = 0.2, 160
dets = []; conds = []
for tt in np.linspace(0, 2 * np.pi, 33):
    zz = links(c0 + rad * np.array([np.cos(tt), np.sin(tt)]))
    dets.append(abs(np.linalg.det(zz[0] @ zz[1] - zz[1] @ zz[0])))
    conds.append(np.linalg.cond(kn(zz, 2)[0]))
print(f"  loop: radius {rad} about {c0}; min |det[K1,K2]| = {min(dets):.3f} (absolutely irreducible iff != 0), "
      f"max cond(S_KN) = {max(conds):.3f}")
cases = [("KN metric, weights (1,1), self-balanced", dict(wts=(1.0, 1.0), kn_w=(1.0, 1.0))),
         ("metric weights (1,10) dressed by KN point of weights (1,1): NOT self-balanced", dict(wts=(1.0, 10.0), kn_w=(1.0, 1.0))),
         ("metric weights (1,10) dressed by KN point of weights (1,10): self-balanced", dict(wts=(1.0, 10.0), kn_w=(1.0, 10.0)))]
for label, kw in cases:
    _warm.clear()
    a, S = connection(c0, **kw)
    res = []
    for k in range(2):
        dc = np.zeros(2); dc[k] = 1e-5
        Sp = connection(c0 + dc, **kw)[1]; Sm = connection(c0 - dc, **kw)[1]
        dS = (Sp - Sm) / 2e-5
        res.append(np.linalg.norm(dS + a[k] @ S + S @ a[k].T))
    _warm.clear()
    Tm = np.eye(2)

    def avel(tt):
        p_ = c0 + rad * np.array([np.cos(tt), np.sin(tt)])
        v_ = rad * np.array([-np.sin(tt), np.cos(tt)])
        aa, _ = connection(p_, **kw)
        return -(aa[0] * v_[0] + aa[1] * v_[1])
    hstep = 2 * np.pi / nst
    for kk in range(nst):
        tt = kk * hstep
        k1 = avel(tt) @ Tm
        k2 = avel(tt + hstep / 2) @ (Tm + hstep / 2 * k1)
        k3 = avel(tt + hstep / 2) @ (Tm + hstep / 2 * k2)
        k4 = avel(tt + hstep) @ (Tm + hstep * k3)
        Tm = Tm + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    _warm.clear()
    S0 = connection(c0 + rad * np.array([1.0, 0.0]), **kw)[1]
    M0 = np.linalg.inv(S0)
    ev = np.linalg.eigvals(Tm)
    print(f"  {label}:")
    print(f"     |dS + aS + Sa^T| at c0 = {res[0]:.2e}, {res[1]:.2e}  (|a(e1)|={np.linalg.norm(a[0]):.3f}, |a(e2)|={np.linalg.norm(a[1]):.3f})")
    print(f"     holonomy: eigenvalues {np.round(ev, 9)}, moduli {np.round(np.abs(ev), 9)}, det {np.linalg.det(Tm):.10f}, "
          f"|Hol - I| = {np.linalg.norm(Tm - np.eye(2)):.3e}, isometry defect |Hol^T S0^-1 Hol - S0^-1| = {np.linalg.norm(Tm.T @ M0 @ Tm - M0):.2e}")
print(f"  ({time.time()-t0:.1f} s)")

print("\n=== D. ibt:group with g = sl(K): traceless part of -dX Q_X when M = H_X^{-1} ===")
K, r = 2, 4
co = rng.normal(size=(3, K, r))
def Xf(c):
    return co[0] + c[0] * co[1] + c[1] * co[2] + 0.4 * c[0] * c[1] * co[1] @ np.diag([1, -1, 0.5, 2])
cc = np.array([0.2, 0.3])
X = Xf(cc); H = X @ X.T; Q = X.T @ np.linalg.inv(H)
for mname, Mm in [("H_X^{-1}", np.linalg.inv(H)), ("I", np.eye(K))]:
    for k in range(2):
        dc = np.zeros(2); dc[k] = 1e-6
        dX = (Xf(cc + dc) - Xf(cc - dc)) / 2e-6
        agl = -dX @ Q
        # constrained normal equation over sl(2)
        Gm = np.array([[np.trace(Mm @ (bi @ X) @ (bj @ X).T) for bj in bas2] for bi in bas2])
        rhs = -np.array([np.trace(Mm @ dX @ (bj @ X).T) for bj in bas2])
        coef = np.linalg.solve(Gm, rhs)
        asl = sum(cf * b for cf, b in zip(coef, bas2))
        tl = agl - np.trace(agl) / K * np.eye(K)
        if k == 0:
            print(f"  M = {mname}: |a_sl - traceless(a_gl)| = {np.linalg.norm(asl - tl):.2e} (direction 1)", end='')
        else:
            print(f", {np.linalg.norm(asl - tl):.2e} (direction 2)")
print(f"\nruntime {time.time()-t0:.1f} s")

print("\n=== E. SL(2) pairs: balanced branches and curvature of the KN-dressed alignment connection ===")
print("  branch I (det[K1,K2] > 0): balanced links both symmetric; branch II (det < 0): skew parts nonzero,")
print("  traceless-symmetric parts parallel.  Prediction: connection flat on branch II, curved on branch I.")
rng3 = np.random.default_rng(11)
kw = dict(wts=(1.0, 1.0), kn_w=(1.0, 1.0))


def curvature(c, h=1e-4):
    def Aof(cc):
        return connection(cc, **kw)[0]
    a = Aof(c)
    d2a1 = (Aof(c + np.array([h, 0]))[1] - Aof(c - np.array([h, 0]))[1]) / (2 * h)
    d1a2 = (Aof(c + np.array([0, h]))[0] - Aof(c - np.array([0, h]))[0]) / (2 * h)
    return d2a1 - d1a2 + a[0] @ a[1] - a[1] @ a[0]


Jsk = np.array([[0.0, -1.0], [1.0, 0.0]])
count = 0
def symdom(scale):
    M_ = rng3.normal(scale=scale, size=(2, 2))
    return (M_ + M_.T) / 2 + 0.05 * (M_ - M_.T) / 2


for trial in range(80):
    if trial % 2 == 0:
        L1c[:] = [rng3.normal(scale=0.6, size=(2, 2)) for _ in range(3)]
        L2c[:] = [rng3.normal(scale=0.6, size=(2, 2)) for _ in range(3)]
    else:
        L1c[:] = [symdom(0.6) for _ in range(3)]
        L2c[:] = [symdom(0.6) for _ in range(3)]
    c = np.zeros(2)
    z = links(c)
    dC = np.linalg.det(z[0] @ z[1] - z[1] @ z[0])
    if abs(dC) < 0.1:
        continue
    _warm.clear()
    F = curvature(c)
    a, S = connection(c, **kw)
    w_, V_ = np.linalg.eigh(S)
    Lh = V_ @ np.diag(np.sqrt(w_)) @ V_.T
    Bz = [np.linalg.inv(Lh) @ K_ @ Lh for K_ in z]
    skew = [np.linalg.norm(B_ - B_.T) / 2 for B_ in Bz]
    Ts = [((B_ + B_.T) / 2 - np.trace(B_) / 2 * np.eye(2)) for B_ in Bz]
    par = abs(Ts[0][0, 0] * Ts[1][0, 1] - Ts[0][0, 1] * Ts[1][0, 0])
    print(f"  det[K1,K2] = {dC: .3f}: |F_12| = {np.linalg.norm(F):.2e} (|a| = {np.linalg.norm(a[0]) + np.linalg.norm(a[1]):.2f}); "
          f"balanced reps: skew parts {skew[0]:.2e}, {skew[1]:.2e}; T-part parallelism defect {par:.2e}")
    count += 1
    if count >= 14:
        break
print(f"\nruntime {time.time()-t0:.1f} s")

print("\n=== F. nontrivial holonomy on branch I (det[K1,K2] > 0), self-balanced vs not ===")
rng4 = np.random.default_rng(5)
found = False
for trial in range(200):
    L1c[:] = [symdom(0.6) for _ in range(3)]
    L2c[:] = [symdom(0.6) for _ in range(3)]
    ok = True
    c0f = np.zeros(2); radf = 0.15
    for tt in np.linspace(0, 2 * np.pi, 17):
        zz = links(c0f + radf * np.array([np.cos(tt), np.sin(tt)]))
        if np.linalg.det(zz[0] @ zz[1] - zz[1] @ zz[0]) < 0.1:
            ok = False; break
    if ok:
        found = True
        break
print(f"  field found on branch I along the whole loop: {found} (trial {trial})")
for label, kwf in [("self-balanced KN (1,1)", dict(wts=(1.0, 1.0), kn_w=(1.0, 1.0))),
                   ("weights (1,10) with KN point of (1,1)", dict(wts=(1.0, 10.0), kn_w=(1.0, 1.0)))]:
    _warm.clear()
    Tm = np.eye(2); nst = 160; hstep = 2 * np.pi / nst

    def avel(tt, kwf=kwf):
        p_ = c0f + radf * np.array([np.cos(tt), np.sin(tt)])
        v_ = radf * np.array([-np.sin(tt), np.cos(tt)])
        aa, _ = connection(p_, **kwf)
        return -(aa[0] * v_[0] + aa[1] * v_[1])
    for kk in range(nst):
        tt = kk * hstep
        k1 = avel(tt) @ Tm
        k2 = avel(tt + hstep / 2) @ (Tm + hstep / 2 * k1)
        k3 = avel(tt + hstep / 2) @ (Tm + hstep / 2 * k2)
        k4 = avel(tt + hstep) @ (Tm + hstep * k3)
        Tm = Tm + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    _warm.clear()
    S0 = connection(c0f + radf * np.array([1.0, 0.0]), **kwf)[1]
    M0 = np.linalg.inv(S0)
    ev = np.linalg.eigvals(Tm)
    print(f"  {label}: eigenvalues {np.round(ev, 8)}, moduli {np.round(np.abs(ev), 10)}, |Hol - I| = {np.linalg.norm(Tm - np.eye(2)):.3e}, "
          f"isometry defect {np.linalg.norm(Tm.T @ M0 @ Tm - M0):.2e}")
print(f"\nruntime {time.time()-t0:.1f} s")
