r"""D06-PRED s2: conditional-information identities of chapters/04_prediction.tex on finite spaces.

  eq:evaluator-compatibility    s_i = Theta_# s_j and K_i(Theta m) = K_j(m) give equal predictions
  thm:predictive-chain          KL(s||t) = KL(sK||tK) + int KL(s^o||t^o) sK(do); both sides infinite
                                together without finiteness; Petz/Bayes-inverse recovery iff the
                                conditional term vanishes (finite case); infinite case: equality
                                without recovery
  prop:conditional-lift         T_# qhat = mu and KL(q||qhat) = int KL(q(.|h)||p(.|h)) mu(dh)
  cor:pythagorean               KL(q||p) = KL(q||qhat) + KL(qhat||p), KL(qhat||p) = KL(mu||T_# p),
                                qhat beats random r with T_# r = mu
  prop:common-family-recovery   (C_# P_a) R_0 = Lbar_a(C(x)) P_0(dx); recovery iff L_a is sigma(C)-measurable
  cor:local-family-recovery     CI case and dependent case; two-point witness
  thm:predictive-distortion     decoder decomposition and I(Y;X|U,Z) = I(Y;X|Z) - I(Y;U|Z)
  inh:causal-memory (ch. 5)     path-divergence identity (named by focus question 6 as the
                                "cost of discarding record history")
  companion examples            four-point Jeffrey example (log 2), two-bit compression example
"""
from __future__ import annotations

import itertools

import numpy as np

rng = np.random.default_rng(1029)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, float).ravel()
    q = np.asarray(q, float).ravel()
    m = p > 0
    if np.any(q[m] <= 0):
        return float("inf")
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def simplex(*shape: int) -> np.ndarray:
    x = rng.gamma(1.0, size=shape)
    return x / x.sum(axis=-1, keepdims=True)


def H(p: np.ndarray) -> float:
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# eq:evaluator-compatibility
nM, nO = 4, 3
Kj = simplex(nM, nO)
perm = rng.permutation(nM)                 # Theta: m -> perm[m]
Ki = np.empty_like(Kj)
Ki[perm] = Kj                              # K_i(Theta m) = K_j(m)
sj = simplex(nM)
si = np.zeros(nM)
si[perm] = sj                              # s_i = Theta_# s_j
print("evaluator compatibility: |s_i K_i - s_j K_j| =", np.abs(si @ Ki - sj @ Kj).max())


def chain_terms(s: np.ndarray, t: np.ndarray, K: np.ndarray) -> tuple[float, float, float]:
    sK, tK = s @ K, t @ K
    cond = 0.0
    for o in range(K.shape[1]):
        if sK[o] == 0:
            continue
        so = s * K[:, o] / sK[o]
        if tK[o] == 0:
            cond = float("inf")
            continue
        to = t * K[:, o] / tK[o]
        cond += sK[o] * kl(so, to)
    return kl(s, t), kl(sK, tK), cond


res = 0.0
for _ in range(200):
    s, t, K = simplex(5), simplex(5), simplex(5, 3)
    a, b, c = chain_terms(s, t, K)
    res = max(res, abs(a - b - c))
print("thm:predictive-chain max residual (200 random, finite) =", res)
s = np.array([0.5, 0.5, 0.0])
t = np.array([0.0, 0.5, 0.5])
K = np.array([[1, 0], [0, 1], [0, 1.0]])
print("infinite case (KL(s||t), KL(sK||tK), cond) =", chain_terms(s, t, K))


def bayes_inverse_recovers(s: np.ndarray, t: np.ndarray, K: np.ndarray) -> float:
    sK, tK = s @ K, t @ K
    rec = np.zeros_like(s)
    for o in range(K.shape[1]):
        if tK[o] > 0:
            rec += sK[o] * t * K[:, o] / tK[o]
    return float(np.abs(rec - s).max())


# sufficient channel for (s, t): s/t constant on the fibres of a deterministic map
t = simplex(4)
lr = np.array([2.0, 2.0, 0.5, 0.5])
s = t * lr
s /= s.sum()
Kdet = np.array([[1, 0], [1, 0], [0, 1], [0, 1.0]])
a, b, c = chain_terms(s, t, Kdet)
print("sufficient case: conditional term =", c, " recovery error =", bayes_inverse_recovers(s, t, Kdet))
s2 = simplex(4)
a, b, c = chain_terms(s2, t, Kdet)
print("non-sufficient case: conditional term =", c, " recovery error =", bayes_inverse_recovers(s2, t, Kdet))
print("infinite case recovery error (equality of infinities without recovery) =",
      bayes_inverse_recovers(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5]), K))

# prop:conditional-lift and cor:pythagorean
nE, nH = 6, 3
Tmap = np.array([0, 0, 1, 1, 2, 2])
q, p = simplex(nE), simplex(nE)
mu = np.array([q[Tmap == h].sum() for h in range(nH)])
Tp = np.array([p[Tmap == h].sum() for h in range(nH)])
qhat = np.zeros(nE)
for h in range(nH):
    qhat[Tmap == h] = p[Tmap == h] / Tp[h] * mu[h]
pushed = np.array([qhat[Tmap == h].sum() for h in range(nH)])
loss = sum(mu[h] * kl(q[Tmap == h] / mu[h], p[Tmap == h] / Tp[h]) for h in range(nH))
print("conditional-lift: |T# qhat - mu| =", np.abs(pushed - mu).max(), " loss residual =", abs(kl(q, qhat) - loss))
print("pythagorean residual =", abs(kl(q, p) - kl(q, qhat) - kl(qhat, p)),
      " |KL(qhat||p) - KL(mu||T#p)| =", abs(kl(qhat, p) - kl(mu, Tp)))
best = np.inf
for _ in range(3000):
    rr = np.zeros(nE)
    for h in range(nH):
        rr[Tmap == h] = simplex(2) * mu[h]
    best = min(best, kl(rr, p))
print("I-projection: KL(qhat||p) =", kl(qhat, p), " best random competitor =", best)
# q << p with zeros in p (and q) on the same atom
p0 = p.copy(); p0[1] = 0.0; p0 /= p0.sum()
q0 = q.copy(); q0[1] = 0.0; q0 /= q0.sum()
mu0 = np.array([q0[Tmap == h].sum() for h in range(nH)])
Tp0 = np.array([p0[Tmap == h].sum() for h in range(nH)])
qhat0 = np.zeros(nE)
for h in range(nH):
    qhat0[Tmap == h] = p0[Tmap == h] / Tp0[h] * mu0[h]
print("zero-atom case pythagorean residual =", abs(kl(q0, p0) - kl(q0, qhat0) - kl(qhat0, p0)))

# prop:common-family-recovery
nX, nZ = 6, 3
C = np.array([0, 1, 1, 2, 2, 2])
P0 = simplex(nX)
CP0 = np.array([P0[C == z].sum() for z in range(nZ)])
R0 = np.zeros((nZ, nX))
for z in range(nZ):
    R0[z, C == z] = P0[C == z] / CP0[z]
for label, Lr in [("generic L", rng.gamma(1.0, size=nX)), ("sigma(C)-measurable L", np.array([0.3, 1.7, 1.7, 0.9, 0.9, 0.9]))]:
    Pa = Lr * P0
    Pa /= Pa.sum()
    L = Pa / P0
    Lbar = np.array([np.sum(L[C == z] * P0[C == z]) / CP0[z] for z in range(nZ)])
    CPa = np.array([Pa[C == z].sum() for z in range(nZ)])
    lift = CPa @ R0
    print(f"common recovery [{label}]: density residual =", np.abs(lift - Lbar[C] * P0).max(),
          " recovery error =", np.abs(lift - Pa).max())

# cor:local-family-recovery
nXl, nEl = 4, 2
Cl = np.array([0, 0, 1, 1])
pz = simplex(2)
px_z = np.zeros((2, nXl))
for z in range(2):
    px_z[z, Cl == z] = simplex(2)
pe_z = simplex(2, nEl)
P_ci = np.einsum("z,zx,ze->xe", pz, px_z, pe_z)            # X indep E given Z
P_dep = simplex(nXl * nEl).reshape(nXl, nEl)


def local_lift(Pxe: np.ndarray, Pref: np.ndarray) -> np.ndarray:
    """Retain (Z,E) of Pxe, redraw X from Pref(dx|Z)."""
    Pz_ref = np.array([Pref[Cl == z].sum() for z in range(2)])
    Rz = np.zeros((2, nXl))
    for z in range(2):
        Rz[z, Cl == z] = Pref[Cl == z].sum(axis=1) / Pz_ref[z]
    PZE = np.array([[Pxe[Cl == z, e].sum() for e in range(nEl)] for z in range(2)])
    return np.einsum("ze,zx->xe", PZE, Rz)


print("local recovery CI case: error =", np.abs(local_lift(P_ci, P_ci) - P_ci).max())
print("local recovery dependent case: error =", np.abs(local_lift(P_dep, P_dep) - P_dep).max())
Ldesc = np.array([[1.4, 0.6], [1.4, 0.6], [0.8, 1.2], [0.8, 1.2]])   # depends on (Z,E) only
Pa = Ldesc * P_ci
Pa /= Pa.sum()
print("local recovery CI case, descending family: error =", np.abs(local_lift(Pa, P_ci) - Pa).max())
Lnd = rng.gamma(1.0, size=(nXl, nEl))
Pb = Lnd * P_ci
Pb /= Pb.sum()
print("local recovery CI case, non-descending family: error =", np.abs(local_lift(Pb, P_ci) - Pb).max())
W = np.array([[0.5, 0.0], [0.0, 0.5]])                     # two-point witness, X in {0,1}, C constant
lift_w = np.outer(W.sum(axis=0), W.sum(axis=1)).T           # X uniform, independent of E
print("two-point witness: lift =", lift_w.ravel(), " vs P0 =", W.ravel())

# thm:predictive-distortion
nXd, nYd, nZd, nUd = 3, 2, 2, 2
Pxyz = simplex(nXd * nYd * nZd).reshape(nXd, nYd, nZd)
Tenc = simplex(nXd, nUd)
Delta = simplex(nUd, nZd, nYd)                              # Delta(y|u,z)
Pxyzu = Pxyz[:, :, :, None] * Tenc[:, None, None, :]      # U - X - (Y,Z)


def cond_mi_YX_given(PJ: np.ndarray, given: tuple[int, ...]) -> float:
    """I(Y;X|given) for a joint array with axes (x, y, z, u)."""
    axes = {"x": 0, "y": 1, "z": 2, "u": 3}
    keep = sorted(set([0, 1] + list(given)))
    sumax = tuple(a for a in range(4) if a not in keep)
    Pk = PJ.sum(axis=sumax) if sumax else PJ
    names = [a for a in range(4) if a in keep]
    gi = [names.index(g) for g in given]
    xi, yi = names.index(0), names.index(1)
    Pg = Pk.sum(axis=(xi, yi), keepdims=True)
    Pxg = Pk.sum(axis=yi, keepdims=True)
    Pyg = Pk.sum(axis=xi, keepdims=True)
    m = Pk > 0
    val = Pk * (np.log(np.where(m, Pk, 1)) + np.log(np.where(m, Pg, 1)) - np.log(np.where(m, Pxg, 1)) - np.log(np.where(m, Pyg, 1)))
    return float(val[m].sum())


lhs = 0.0
P_y_xz = Pxyz / Pxyz.sum(axis=1, keepdims=True)              # P(y|x,z)
Puz_y = Pxyzu.sum(axis=0)                                    # (y,z,u)
P_y_uz = Puz_y / Puz_y.sum(axis=0, keepdims=True)            # P(y|u,z)
second = 0.0
for x, z, u in itertools.product(range(nXd), range(nZd), range(nUd)):
    pr = Pxyzu[x, :, z, u].sum()
    lhs += pr * kl(P_y_xz[x, :, z], Delta[u, z])
for z, u in itertools.product(range(nZd), range(nUd)):
    pr = Puz_y[:, z, u].sum()
    second += pr * kl(P_y_uz[:, z, u], Delta[u, z])
I_YX_UZ = cond_mi_YX_given(Pxyzu, (2, 3))
I_YX_Z = cond_mi_YX_given(Pxyzu, (2,))
# I(Y;U|Z): swap roles of x and u
I_YU_Z = cond_mi_YX_given(np.transpose(Pxyzu, (3, 1, 2, 0)), (2,))
print("decoder decomposition residual =", abs(lhs - I_YX_UZ - second))
print("distortion identity residual |I(Y;X|U,Z) - (I(Y;X|Z) - I(Y;U|Z))| =", abs(I_YX_UZ - (I_YX_Z - I_YU_Z)))

# inh:causal-memory (chapter 5, outside the file scope): binary path of length 3
n = 3
Ppath = simplex(2 ** n).reshape((2,) * n)
stats = [lambda h: 0, lambda h: h[-1], lambda h: sum(h) % 2]  # f_0 trivial, f_1 last, f_2 parity
Tt = [simplex(1, 2), simplex(2, 2), simplex(2, 2)]
Ptil = np.ones((2,) * n)
for path in itertools.product(range(2), repeat=n):
    pr = 1.0
    for tt in range(n):
        pr *= Tt[tt][stats[tt](path[:tt]), path[tt]]
    Ptil[path] = pr
lhs = kl(Ppath, Ptil)
rhs = 0.0
for tt in range(n):
    # joint of (H_t, Z_t)
    PH = Ppath.sum(axis=tuple(range(tt + 1, n))) if tt + 1 < n else Ppath
    # conditional MI I(Z_t; H_t | M_t) and the KL term
    PM = {}
    for hz in itertools.product(range(2), repeat=tt + 1):
        mval = stats[tt](hz[:tt])
        PM.setdefault(mval, np.zeros(2))
        PM[mval][hz[tt]] += PH[hz]
    cmi = 0.0
    for hz in itertools.product(range(2), repeat=tt + 1):
        h, zt = hz[:tt], hz[tt]
        mval = stats[tt](h)
        p_hz = PH[hz]
        if p_hz == 0:
            continue
        p_h = PH[h].sum() if tt > 0 else 1.0
        p_z_h = p_hz / p_h
        p_z_m = PM[mval][zt] / PM[mval].sum()
        cmi += p_hz * np.log(p_z_h / p_z_m)
    klterm = sum(PM[mv].sum() * kl(PM[mv] / PM[mv].sum(), Tt[tt][mv]) for mv in PM)
    rhs += cmi + klterm
print("inh:causal-memory identity residual =", abs(lhs - rhs))

# companion four-point example: q uniform on {00,11}, T = H, independent fair reference
qv = np.array([0.5, 0, 0, 0.5])                              # order 00,01,10,11
print("companion four-point: KL(q||uniform reconstruction) - log 2 =", kl(qv, np.full(4, 0.25)) - np.log(2))
# two-bit compression: X=(A,B) fair independent, U=A
PAB = np.full((2, 2), 0.25)
print("compression: D for Y=A is 0; D for Y=B equals H(B|A) - log2 =", H(PAB.ravel()) - H(PAB.sum(axis=1)) - np.log(2))
print("values: I(Y;X|U,Z) =", I_YX_UZ, " I(Y;X|Z) =", I_YX_Z, " I(Y;U|Z) =", I_YU_Z)
# independent brute-force I(Y;U|Z) from the (y,z,u) marginal
Pyzu = Pxyzu.sum(axis=0)
Pz_ = Pyzu.sum(axis=(0, 2))
brute = 0.0
for y, z, u in itertools.product(range(nYd), range(nZd), range(nUd)):
    pj = Pyzu[y, z, u]
    brute += pj * np.log(pj * Pz_[z] / (Pyzu[:, z, u].sum() * Pyzu[y, z, :].sum()))
print("brute-force I(Y;U|Z) residual =", abs(brute - I_YU_Z))
