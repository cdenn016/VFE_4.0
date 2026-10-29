"""D02-LINKS checks for supplements/s02_geometry.tex (MAgent Theory 2.0, 2026-10-29).

Single-threaded, CPU only.  Numerical agreement supports an identity at the tested
instances; it does not prove it.
"""
import numpy as np
import scipy.linalg as sla
import scipy.integrate as si
import sympy as sp

rng = np.random.default_rng(29102026)
inv = np.linalg.inv
nrm = np.linalg.norm


def phi1(Z):
    """phi_1(Z) = sum_{n>=1} Z^{n-1}/n!, from the top-right block of expm([[Z, I], [0, 0]])."""
    r = Z.shape[0]
    big = np.zeros((2 * r, 2 * r))
    big[:r, :r] = Z
    big[:r, r:] = np.eye(r)
    return sla.expm(big)[:r, r:]


def phi1_series(Z, n=60):
    r = Z.shape[0]
    S = np.zeros_like(Z)
    P = np.eye(r)
    fact = 1.0
    for k in range(1, n):
        fact *= k
        S = S + P / fact
        P = P @ Z
    return S


print("== lem:sgeom-lowrank for random rectangular O (d x r), T (r x d), including r > d")
for (d, r) in [(3, 1), (4, 2), (5, 3), (6, 6), (3, 5), (7, 2)]:
    O = rng.standard_normal((d, r))
    T = rng.standard_normal((r, d))
    lhs = sla.expm(O @ T)
    rhs = np.eye(d) + O @ phi1(T @ O) @ T
    de = np.linalg.det(lhs)
    print(f"  (d,r)=({d},{r}): |e^OT - I - O phi1(TO) T| = {nrm(lhs - rhs):.2e};"
          f" |det e^OT - e^tr(OT)| / e^tr = {abs(de - np.exp(np.trace(O @ T))) / np.exp(np.trace(O @ T)):.2e};"
          f" tr(OT) - tr(TO) = {np.trace(O @ T) - np.trace(T @ O):.1e}")
Z = 0.7 * rng.standard_normal((4, 4))
print("  phi1 block exponential vs truncated series:", nrm(phi1(Z) - phi1_series(Z)))
print("  invertible Z: |phi1(Z) - (e^Z - I) Z^-1| =", nrm(phi1(Z) - (sla.expm(Z) - np.eye(4)) @ inv(Z)))
# singular TO and nilpotent TO
d, r = 5, 3
O = rng.standard_normal((d, r))
T = rng.standard_normal((r, d))
T[2, :] = 0.0                        # TO singular
print("  singular TO (rank", np.linalg.matrix_rank(T @ O), "): residual", nrm(sla.expm(O @ T) - np.eye(d) - O @ phi1(T @ O) @ T))
O = np.zeros((4, 2)); O[0, 0] = 1.0; O[1, 1] = 1.0
T = np.zeros((2, 4)); T[0, 1] = 1.0; T[1, 2] = 1.0     # TO = [[0,1],[0,0]] nilpotent
print("  nilpotent TO =", (T @ O).tolist(), ": residual", nrm(sla.expm(O @ T) - np.eye(4) - O @ phi1(T @ O) @ T))

print("== Stacked interfaces: generic cross products versus block-diagonal replacement")
d = 6
H = [2, 1, 2]
Os = [rng.standard_normal((d, k)) for k in H]
Ts = [rng.standard_normal((k, d)) for k in H]
O = np.hstack(Os)
T = np.vstack(Ts)
full = sla.expm(O @ T)
print("  full stacked formula residual:", nrm(full - np.eye(d) - O @ phi1(T @ O) @ T))
diag = np.eye(d) + sum(Oh @ phi1(Th @ Oh) @ Th for Oh, Th in zip(Os, Ts))
print("  replacing TO by its diagonal blocks, generic: error", round(nrm(full - diag), 4))
P = rng.standard_normal((d, d))
Pi = inv(P)
idx = [list(range(0, 2)), list(range(2, 3)), list(range(3, 5))]
Os2 = [P[:, ix] @ rng.standard_normal((len(ix), k)) for ix, k in zip(idx, H)]
Ts2 = [rng.standard_normal((k, len(ix))) @ Pi[ix, :] for ix, k in zip(idx, H)]
O2, T2 = np.hstack(Os2), np.vstack(Ts2)
cross = max(nrm(Ts2[a] @ Os2[b]) for a in range(3) for b in range(3) if a != b)
full2 = sla.expm(O2 @ T2)
diag2 = np.eye(d) + sum(Oh @ phi1(Th @ Oh) @ Th for Oh, Th in zip(Os2, Ts2))
print(f"  with T_h O_k = 0 (h != k; max cross norm {cross:.1e}): error {nrm(full2 - diag2):.2e}")

print("== prop:sgeom-rankone")
for t_target in [-0.874, 0.0, 1.188, 3.5]:
    u = rng.standard_normal(4)
    v = rng.standard_normal(4)
    if t_target == 0.0:
        v = v - (v @ u) / (u @ u) * u
    else:
        v = v * (t_target / (v @ u))
    L = np.outer(u, v)
    t = v @ u
    E = sla.expm(L)
    f1 = 1.0 if abs(t) < 1e-14 else (np.exp(t) - 1) / t
    print(f"  t = {t:+.3f}: |e^L - I - phi1(t) L| = {nrm(E - np.eye(4) - f1 * L):.2e}; |det e^L - e^t| = {abs(np.linalg.det(E) - np.exp(t)):.2e}"
          + (f"; |L^2| = {nrm(L @ L):.1e}" if t_target == 0.0 else ""))

print("== lem:sgeom-frechet")
for d in [3, 5]:
    L = 0.8 * rng.standard_normal((d, d))
    Hm = rng.standard_normal((d, d))
    x, w = np.polynomial.legendre.leggauss(40)
    ts = 0.5 * (x + 1)
    ws = 0.5 * w
    quad = sum(wk * sla.expm((1 - tk) * L) @ Hm @ sla.expm(tk * L) for tk, wk in zip(ts, ws))
    _, fr = sla.expm_frechet(L, Hm)
    eps = 1e-5
    cd = (sla.expm(L + eps * Hm) - sla.expm(L - eps * Hm)) / (2 * eps)
    print(f"  d={d}: |quadrature - expm_frechet| = {nrm(quad - fr):.2e}; |central difference - expm_frechet| = {nrm(cd - fr):.2e}")
    C = 0.3 * L + 0.2 * L @ L
    _, frc = sla.expm_frechet(L, C)
    print(f"        commuting direction: |D exp_L[H] - e^L H| = {nrm(frc - sla.expm(L) @ C):.2e};"
          f" noncommuting: |D exp_L[H] - e^L H| = {nrm(fr - sla.expm(L) @ Hm):.3f}")

print("== eq:sgeom-moment-derivative (sympy, d = 2, symbolic curves)")
t, lam = sp.symbols("t lambda", real=True)
a = sp.symbols("a1:3", real=True); da = sp.symbols("da1:3", real=True)
b = sp.symbols("b1:3", real=True); db = sp.symbols("db1:3", real=True)
s11, s12, s22, ds11, ds12, ds22 = sp.symbols("s11 s12 s22 ds11 ds12 ds22", real=True)
mui = sp.Matrix(a) + t * sp.Matrix(da)
muj = sp.Matrix(b) + t * sp.Matrix(db)
Sj = sp.Matrix([[s11, s12], [s12, s22]]) + t * sp.Matrix([[ds11, ds12], [ds12, ds22]])
Lt = lam * mui * muj.T * Sj.inv()
dL_direct = sp.simplify(Lt.diff(t).subs(t, 0))
S0 = sp.Matrix([[s11, s12], [s12, s22]]); dS = sp.Matrix([[ds11, ds12], [ds12, ds22]])
mi0, mj0 = sp.Matrix(a), sp.Matrix(b)
dL_formula = lam * (sp.Matrix(da) * mj0.T * S0.inv() + mi0 * sp.Matrix(db).T * S0.inv()
                    - mi0 * mj0.T * S0.inv() * dS * S0.inv())
print("  direct derivative minus formula:", sp.simplify(dL_direct - dL_formula).tolist())

print("== Regularity sentence: DQM of Theta -> Theta#nu_j versus differentiability of KL(nu_i || Theta#nu_j)")
# (a) the manuscript counterexample: nu_j uniform on [1,2]x[0,1], nu_i = nu_j (Theta0 = I)
for e in [1e-2, 1e-4]:
    up = np.array([1 + e / 2, 0.5]) / (1 + e)
    dn = np.array([2 - e / 2, 0.5]) / (1 - e)
    print(f"  eps={e}: witness of supp nu_i outside (1+eps) supp: preimage x1 = {up[0]:.6f} < 1;"
          f" outside (1-eps) supp: preimage x1 = {dn[0]:.6f} > 2")
# (b) DQM fails but the cost is differentiable at the diagonal:
#     nu with density p(x) = exp(-|x-1|^(1/2))/4 on R, scale family p_s(x) = p(x/s)/s.
import mpmath as mp
mp.mp.dps = 30
p = lambda x: mp.e ** (-mp.sqrt(abs(x - 1))) / 4
score = lambda x: (x * mp.sign(x - 1) / (2 * mp.sqrt(abs(x - 1))) - 1) if x != 1 else mp.inf
for delta in [mp.mpf("1e-2"), mp.mpf("1e-4"), mp.mpf("1e-6")]:
    I_tr = mp.quad(lambda x: score(x) ** 2 * p(x), [-mp.inf, 0, 1 - delta]) + mp.quad(lambda x: score(x) ** 2 * p(x), [1 + delta, 2, mp.inf])
    print(f"  truncated scale Fisher information, |x-1| > {mp.nstr(delta, 2)}: {mp.nstr(I_tr, 8)}")
def KL(sig):
    f = lambda x: p(x) * (mp.sqrt(abs(x / sig - 1)) - mp.sqrt(abs(x - 1)))
    return mp.quad(f, [-mp.inf, 0, 1, sig, 3, mp.inf]) + mp.log(sig)
for e in ["1e-2", "1e-3", "1e-4", "1e-5"]:
    ee = mp.mpf(e)
    kp, km = KL(1 + ee), KL(1 - ee)
    print(f"  eps={e}: KL(nu||(1+eps)#nu) = {mp.nstr(kp, 6)}, KL/eps = {mp.nstr(kp / ee, 4)}, KL/eps^2 = {mp.nstr(kp / ee ** 2, 5)};"
          f" (1-eps): KL/eps = {mp.nstr(km / ee, 4)}")
# (c) DQM holds (Gaussian scale family) but the cost is identically +infinity: nu_i atomic with finite moments
print("  nu_j Gaussian (DQM scale family), nu_i = (delta_{-1} + delta_{+1})/2 (finite moments, variance 1):",
      "nu_i is not absolutely continuous w.r.t. any Theta#nu_j, so KL = +inf for every Theta")
