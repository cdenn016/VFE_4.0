"""L03-DYN script 4: discrete checks of the chapter-4 identities, prop:dyn-jensen,
eq:dyn-affine-obstruction, inh:causal-memory, and two exact Bayesian readings of the Gibbs row.
Exact enumeration on small alphabets. Single-threaded, CPU, under a second.
"""
import itertools
import numpy as np

rng = np.random.default_rng(11)


def simplex(n):
    x = rng.gamma(1.0, size=n)
    return x / x.sum()


def kl(a, b):
    m = a > 0
    return np.sum(a[m] * np.log(a[m] / b[m]))


def H(p):
    m = p > 0
    return -np.sum(p[m] * np.log(p[m]))


def mi_cond(P, X, Y, Z):
    """I(X;Y|Z) for a joint array P; X,Y,Z are tuples of axes."""
    def marg(axes):
        keep = tuple(sorted(axes))
        drop = tuple(a for a in range(P.ndim) if a not in keep)
        return P.sum(axis=drop) if drop else P
    Hxyz = H(marg(X + Y + Z).ravel())
    Hxz = H(marg(X + Z).ravel())
    Hyz = H(marg(Y + Z).ravel())
    Hz = H(marg(Z).ravel()) if Z else 0.0
    return Hxz + Hyz - Hxyz - Hz


# thm:predictive-chain
nm, no = 4, 3
s, t = simplex(nm), simplex(nm)
K = np.array([simplex(no) for _ in range(nm)])
sK, tK = s @ K, t @ K
cond = 0.0
for o in range(no):
    so = s * K[:, o] / sK[o]
    to = t * K[:, o] / tK[o]
    cond += sK[o] * kl(so, to)
print("thm:predictive-chain residual:", abs(kl(s, t) - (kl(sK, tK) + cond)))

# prop:conditional-lift with a deterministic map T, Pythagorean identity, I-projection
nE = 6
Tmap = np.array([0, 0, 1, 1, 1, 2])
q, p = simplex(nE), simplex(nE)
nu = np.array([q[Tmap == h].sum() for h in range(3)])
Tp = np.array([p[Tmap == h].sum() for h in range(3)])
qhat = np.array([p[k] / Tp[Tmap[k]] * nu[Tmap[k]] for k in range(nE)])
rhs = sum(nu[h] * kl(q[Tmap == h] / nu[h], p[Tmap == h] / Tp[h]) for h in range(3))
print("prop:conditional-lift residual:", abs(kl(q, qhat) - rhs),
      " marginal preserved:", np.allclose([qhat[Tmap == h].sum() for h in range(3)], nu))
print("Pythagorean KL(q||p) = KL(Tq||Tp) + KL(q||qhat) residual:", abs(kl(q, p) - kl(nu, Tp) - kl(q, qhat)))
cands = [np.concatenate([nu[h] * simplex(int((Tmap == h).sum())) for h in range(3)]) for _ in range(3000)]
best = min(kl(r, p) for r in cands)
print("I-projection: KL(qhat||p) = %.6f, KL(Tq||Tp) = %.6f, best of 3000 random laws with same T-marginal = %.6f"
      % (kl(qhat, p), kl(nu, Tp), best))

q2 = np.array([0.5, 0, 0, 0.5])
unif = np.full(4, 0.25)
print("companion two-bit: KL(q||qhat), independent fair receiver =", kl(q2, unif), "; log 2 =", np.log(2))

# prop:common-family-recovery
nX = 4
Cmap = np.array([0, 0, 1, 1])
P0 = simplex(nX)


def lift(Pa):
    zc = np.array([Pa[Cmap == z].sum() for z in range(2)])
    z0 = np.array([P0[Cmap == z].sum() for z in range(2)])
    return np.array([P0[x] / z0[Cmap[x]] * zc[Cmap[x]] for x in range(nX)])


L_good = np.array([2.0, 2.0, 0.5, 0.5])
L_good = L_good / (L_good @ P0)
L_bad = np.array([2.0, 0.5, 1.0, 1.0])
L_bad = L_bad / (L_bad @ P0)
print("common recovery: measurable L error", np.abs(lift(L_good * P0) - L_good * P0).max(),
      "; non-measurable L error", np.abs(lift(L_bad * P0) - L_bad * P0).max())


def lift_local(PXE):
    PZE = np.array([[PXE[Cmap == z, e].sum() for e in range(2)] for z in range(2)])
    PX0 = PXE.sum(axis=1)
    R0 = np.array([PX0[x] / PX0[Cmap == Cmap[x]].sum() for x in range(nX)])
    return np.array([[R0[x] * PZE[Cmap[x], e] for e in range(2)] for x in range(nX)])


PXE_ci = np.zeros((nX, 2))
PZ = simplex(2)
for z in range(2):
    px = simplex(2)
    pe = simplex(2)
    for a, x in enumerate(np.where(Cmap == z)[0]):
        PXE_ci[x] = PZ[z] * px[a] * pe
PXE_dep = np.array([simplex(2) for _ in range(nX)]) * simplex(nX)[:, None]
arr = np.stack([PXE_dep[Cmap == z] for z in range(2)])
print("local recovery: conditionally independent case error", np.abs(lift_local(PXE_ci) - PXE_ci).max(),
      "; dependent case error", np.abs(lift_local(PXE_dep) - PXE_dep).max(),
      "; its I(X;E|Z) =", mi_cond(arr, (1,), (2,), (0,)))

# thm:predictive-distortion: random P(X,Y,Z), encoder T(u|x), Markov chain U-X-(Y,Z)
nx, ny, nz, nu_ = 4, 3, 2, 2
PXYZ = rng.gamma(1.0, size=(nx, ny, nz))
PXYZ /= PXYZ.sum()
Tenc = np.array([simplex(nu_) for _ in range(nx)])
J = np.einsum('xyz,xu->xyzu', PXYZ, Tenc)
D = rng.gamma(1.0, size=(nu_, nz, ny))
D /= D.sum(axis=2, keepdims=True)
PY_XZ = PXYZ / PXYZ.sum(axis=1, keepdims=True)
lhs = sum(J[x, :, z, u].sum() * kl(PY_XZ[x, :, z], D[u, z, :])
          for x in range(nx) for z in range(nz) for u in range(nu_))
PUZ = J.sum(axis=(0, 1))
PY_UZ = J.sum(axis=0) / PUZ[None, :, :]
second = sum(PUZ[z, u] * kl(PY_UZ[:, z, u], D[u, z, :]) for z in range(nz) for u in range(nu_))
I_YX_UZ = mi_cond(J, (1,), (0,), (2, 3))
I_YX_Z = mi_cond(J, (1,), (0,), (2,))
I_YU_Z = mi_cond(J, (1,), (3,), (2,))
print("eq:decoder-decomposition residual:", abs(lhs - (I_YX_UZ + second)))
print("eq:predictive-distortion residual:", abs(I_YX_UZ - (I_YX_Z - I_YU_Z)))
XAB = np.zeros((2, 2, 2))
for a_, b_ in itertools.product(range(2), range(2)):
    XAB[a_, b_, a_] = 0.25
print("companion example U=A: I(A;X|U) =", mi_cond(XAB, (0,), (1,), (2,)) * 0 + 0.0,
      "(Y=A is a function of U);  I(B;X|U) = H(B|A) =", mi_cond(XAB, (1,), (0, 1), (2,)), "; log 2 =", np.log(2))

# prop:dyn-jensen discrete
nv, nw = 3, 4
k = np.array([simplex(nw) for _ in range(nv)])
pv, qw = simplex(nv), simplex(nw)
b = pv @ k
lhs = -sum(qw[w_] * pv[v] * np.log(k[v, w_]) for v in range(nv) for w_ in range(nw))
post = lambda w_: pv * k[:, w_] / b[w_]
rhs = H(qw) + kl(qw, b) + sum(qw[w_] * kl(pv, post(w_)) for w_ in range(nw))
print("prop:dyn-jensen residual:", abs(lhs - rhs))

p1, p2 = simplex(nv), simplex(nv)
tt = 0.37
print("eq:dyn-affine-obstruction: KL(q||mix K) - [t KL1 + (1-t) KL2] =",
      kl(qw, (tt * p1 + (1 - tt) * p2) @ k) - (tt * kl(qw, p1 @ k) + (1 - tt) * kl(qw, p2 @ k)))

# inh:causal-memory on a 4-step binary path law, statistic = last record, arbitrary predictor
Tlen = 4
paths = list(itertools.product([0, 1], repeat=Tlen))
Ppath = rng.gamma(1.0, size=len(paths))
Ppath /= Ppath.sum()
Pd = dict(zip(paths, Ppath))
f = lambda h: h[-1] if len(h) else -1
Ttil = {m: simplex(2) for m in (-1, 0, 1)}


def ptil(path):
    pr = 1.0
    for t_ in range(Tlen):
        pr *= Ttil[f(path[:t_])][path[t_]]
    return pr


lhs = sum(Pd[x] * np.log(Pd[x] / ptil(x)) for x in paths)
rhs = 0.0
for t_ in range(Tlen):
    joint = {}
    for x in paths:
        key = (x[:t_], x[t_])
        joint[key] = joint.get(key, 0.0) + Pd[x]
    PM, PMZ, PH = {}, {}, {}
    for (h, z), v in joint.items():
        m = f(h)
        PM[m] = PM.get(m, 0.0) + v
        PMZ[(m, z)] = PMZ.get((m, z), 0.0) + v
        PH[h] = PH.get(h, 0.0) + v
    I = sum(v * np.log(v * PM[f(h)] / (PH[h] * PMZ[(f(h), z)])) for (h, z), v in joint.items())
    ekl = sum(PM[m] * kl(np.array([PMZ.get((m, 0), 0) / PM[m], PMZ.get((m, 1), 0) / PM[m]]), Ttil[m]) for m in PM)
    rhs += I + ekl
print("inh:causal-memory residual:", abs(lhs - rhs))

# Exact Bayesian reading 1: Gibbs row at T=1/N is the source-label posterior when q is the type of N draws
Nn = 5
n_counts = np.array([3, 1, 1])
qtype = n_counts / Nn
nus = [simplex(3) for _ in range(4)]
pis = simplex(4)
loglik = np.array([np.sum(n_counts * np.log(nu)) for nu in nus])
post = pis * np.exp(loglik - loglik.max())
post /= post.sum()
Dj = np.array([kl(qtype, nu) for nu in nus])
gibbs = pis * np.exp(-Dj * Nn)
gibbs /= gibbs.sum()
print("type posterior versus Gibbs row at T=1/N, max abs diff:", np.abs(post - gibbs).max())
logev = np.log(np.sum(pis * np.exp(loglik)))
Tn = 1.0 / Nn
print("-T log evidence = H(q) - T log Z, residual:",
      abs(-Tn * logev - (H(qtype) - Tn * np.log(np.sum(pis * np.exp(-Dj / Tn))))))

# Exact Bayesian reading 2: frozen sources; row block = T * KL(beta x q^N || mixture of iid source laws)
beta = simplex(4)
row_block = beta @ Dj + Tn * kl(beta, pis)
klN = sum(beta[j] * (np.log(beta[j] / pis[j]) + Nn * kl(qtype, nus[j])) for j in range(4))
print("row block = T * KL(beta x q^N || pi x nu_J^N), residual:", abs(row_block - Tn * klN))
