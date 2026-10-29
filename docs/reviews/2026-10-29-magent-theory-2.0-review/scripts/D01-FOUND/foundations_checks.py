r"""Finite and closed-form checks for chapters/01_foundations.tex and companion/c01_objects.tex.

Lens D01-FOUND, 2026-10-29.  Every check is a finite enumeration or a closed form.
A numerical agreement supports an identity at the tested instance; it is not a proof.

Sections
  A  prop:kl-coordinate (injective maps, infinite case)
  C  disintegration commutes with the diagonal action (eq:agent-disintegration)
  D  prop:bundle-morphism (equivariant kernel, data processing, linear intertwiners)
  E  prop:record-covariance (covariant transition, equivariant readout)
  F  prop:loop-blindness (i), (ii)
  G  single-edge control (passively removable) and self-loop variant of the stabilizer question
  H  eq:agent-free-energy: posterior form versus split ELBO form
  I  NEW: constant evaluator with m-dependent record kernel is not predictively inert
  J  NEW: species "only when" claim -- predictive equivalence without a shared model
  L  correlation section and companion worked examples
"""
from __future__ import annotations

import itertools
import math

import numpy as np

rng = np.random.default_rng(20261029)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    r"""KL(p || q) = \sum p log(p/q) on a finite set, +inf if p is not << q."""
    p = np.asarray(p, float).ravel()
    q = np.asarray(q, float).ravel()
    if np.any((q == 0) & (p > 0)):
        return math.inf
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def rand_simplex(n: int) -> np.ndarray:
    x = rng.random(n) + 0.05
    return x / x.sum()


def push(p: np.ndarray, f: list[int], n_out: int) -> np.ndarray:
    out = np.zeros(n_out)
    for y, fy in enumerate(f):
        out[fy] += p[y]
    return out


print("== A  prop:kl-coordinate")
mu, nu = rand_simplex(5), rand_simplex(5)
f = [6, 0, 3, 7, 2]  # injective map {0..4} -> {0..7}
print("  KL(mu||nu) =", kl(mu, nu), " KL(f#mu||f#nu) =", kl(push(mu, f, 8), push(nu, f, 8)))
nu0 = nu.copy(); nu0[2] = 0.0; nu0 /= nu0.sum()
print("  infinite case:", kl(mu, nu0), kl(push(mu, f, 8), push(nu0, f, 8)))

print("== C  disintegration commutes with the diagonal Z3 action")
zeta = rng.random((3, 3)); zeta /= zeta.sum()  # zeta[k, m]
g = 1
zeta_g = np.zeros_like(zeta)
for k in range(3):
    for m in range(3):
        zeta_g[(k + g) % 3, (m + g) % 3] += zeta[k, m]
s, sg = zeta.sum(0), zeta_g.sum(0)
res = 0.0
for mp in range(3):
    cond_g = zeta_g[:, mp] / sg[mp]                       # conditional of g#zeta given m'
    m = (mp - g) % 3
    cond = zeta[:, m] / s[m]                               # q(.|g^{-1} m')
    pushed = np.array([cond[(k - g) % 3] for k in range(3)])  # g# q(.|g^{-1}m')
    res = max(res, float(np.max(np.abs(cond_g - pushed))))
print("  max |cond(g#zeta)(.|m') - g# q(.|g^-1 m')| =", res)

print("== D  prop:bundle-morphism")
# Y = Z3 x Z2 with g.(a,b) = (a+g, b); Ybar = Z3 with g.a = a+g; phi = id.
Y = [(a, b) for a in range(3) for b in range(2)]
idx = {y: i for i, y in enumerate(Y)}


def act_Y(g: int, y: tuple[int, int]) -> tuple[int, int]:
    return ((y[0] + g) % 3, y[1])


kappa0 = rng.random((6, 3)); kappa0 /= kappa0.sum(1, keepdims=True)
# symmetrize: kappa(y, B) = (1/3) sum_h kappa0(h y, h B)
kappa = np.zeros((6, 3))
for y in Y:
    for h in range(3):
        hy = idx[act_Y(h, y)]
        for b in range(3):
            kappa[idx[y], b] += kappa0[hy, (b + h) % 3] / 3.0
eq_res = 0.0
for g in range(3):
    for y in Y:
        lhs = kappa[idx[act_Y(g, y)]]
        rhs = np.array([kappa[idx[y], (b - g) % 3] for b in range(3)])  # phi(g)# kappa(y,.)
        eq_res = max(eq_res, float(np.max(np.abs(lhs - rhs))))
print("  equivariance residual of symmetrized kappa:", eq_res)
z = rand_simplex(6); z2 = rand_simplex(6)
wd = 0.0
for g in range(3):
    gz = np.zeros(6)
    for y in Y:
        gz[idx[act_Y(g, y)]] += z[idx[y]]
    lhs = gz @ kappa
    zk = z @ kappa
    rhs = np.array([zk[(b - g) % 3] for b in range(3)])
    wd = max(wd, float(np.max(np.abs(lhs - rhs))))
print("  (g#zeta)kappa - phi(g)#(zeta kappa) residual:", wd)
print("  data processing: KL(zeta kappa||zeta' kappa) =", kl(z @ kappa, z2 @ kappa), "<= KL(zeta||zeta') =", kl(z, z2))
# non-equivariant kernel: well-definedness fails
gz = np.zeros(6)
for y in Y:
    gz[idx[act_Y(1, y)]] += z[idx[y]]
zk0 = z @ kappa0
print("  non-equivariant kappa0 mismatch:", float(np.max(np.abs(gz @ kappa0 - np.array([zk0[(b - 1) % 3] for b in range(3)])))))


def rot(t: float, charge: int = 1) -> np.ndarray:
    c, s_ = math.cos(charge * t), math.sin(charge * t)
    return np.array([[c, -s_], [s_, c]])


Jm = np.array([[0.0, -1.0], [1.0, 0.0]])
A = 0.7 * np.eye(2) + 0.3 * Jm        # commutes with charge-1 rotations
At = -0.2 * np.eye(2) + 1.1 * Jm      # commutes with charge-2 rotations
k = rng.normal(size=2); mvec = rng.normal(size=2)
wd_lin = 0.0
for _ in range(20):
    u, gg = rng.uniform(0, 2 * math.pi, 2)
    # coordinates in the reference frame e of [u,k] -> [f(u), A k] with f = id, phi = id
    img1 = rot(u) @ A @ k
    img2 = rot(u + gg) @ A @ (np.linalg.inv(rot(gg)) @ k)   # representative (ug, rho(g)^-1 k)
    img1s = rot(u, 2) @ At @ mvec
    img2s = rot(u + gg, 2) @ At @ (np.linalg.inv(rot(gg, 2)) @ mvec)
    wd_lin = max(wd_lin, float(np.max(np.abs(img1 - img2))), float(np.max(np.abs(img1s - img2s))))
print("  content morphisms Phi, Phi~ well defined (intertwiners), residual:", wd_lin)
B = np.diag([1.0, 2.0])
u, gg = 0.4, 1.1
print("  non-intertwining A=diag(1,2) mismatch:", float(np.max(np.abs(rot(u) @ B @ k - rot(u + gg) @ B @ np.linalg.inv(rot(gg)) @ k))))

print("== E  prop:record-covariance")
nS = 4
G0 = rng.random((nS, nS)); G0 /= G0.sum(1, keepdims=True)
Gam = np.zeros((nS, nS))
for x in range(nS):
    for h in range(nS):
        for b in range(nS):
            Gam[x, b] += G0[(x + h) % nS, (b + h) % nS] / nS
rho = [x % 2 for x in range(nS)]                 # readout to O = Z2
def tau(gm: int, o: int) -> int:
    return (o + gm) % 2                          # rho(gamma x) = tau(gamma) rho(x)
def Rk(x: int, Aset: set[int]) -> float:
    return float(sum(Gam[x, xp] for xp in range(nS) if rho[xp] in Aset))
res = 0.0
for gm in range(nS):
    for x in range(nS):
        for Aset in [{0}, {1}, {0, 1}]:
            lhs = Rk((x + gm) % nS, Aset)
            rhs = Rk(x, {o for o in range(2) if tau(gm, o) in Aset})
            res = max(res, abs(lhs - rhs))
print("  max |R(gamma x, A) - R(x, tau^-1 A)| =", res)

print("== E2  NEW: dressed readout passes passive covariance but depends on the common reference")
# displayed mean mu = U mubar; readout = first coordinate of the reference mean U^{-1} mu
mu_disp = np.array([1.0, 0.0]); U0 = np.eye(2)
read = lambda Uf, m_: float((np.linalg.inv(Uf) @ m_)[0])
gp = rot(math.pi / 2)
print("  readout before / after passive change (mu->g mu, U->g U):", read(U0, mu_disp), read(gp @ U0, gp @ mu_disp))
hc = rot(math.pi / 2)
print("  readout before / after common reference change (U->U h), same bundle point:", read(U0, mu_disp), round(read(U0 @ hc, mu_disp), 12))

print("== F  prop:loop-blindness")
nuU = np.full(4, 0.25)
H = [1, 2, 3, 0]; Hp = [2, 3, 0, 1]
print("  KL(nu||H#nu) =", kl(nuU, push(nuU, H, 4)), " KL(nu||H'#nu) =", kl(nuU, push(nuU, Hp, 4)))
graph = lambda Hm: np.array([[nuU[y] if Hm[y] == z_ else 0.0 for z_ in range(4)] for y in range(4)])
print("  TV distance of graph laws (id,H)#nu vs (id,H')#nu:", 0.5 * float(np.abs(graph(H) - graph(Hp)).sum()))
th = 0.7
Rm = rot(th)
S1 = np.eye(2); S3 = np.diag([1.0, 3.0])
def kl_gauss0(S_a: np.ndarray, S_b: np.ndarray) -> float:
    d = S_a.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(S_b, S_a)) - d + math.log(np.linalg.det(S_b) / np.linalg.det(S_a)))
print("  rotation 0.7 rad: isotropic KL =", kl_gauss0(S1, Rm @ S1 @ Rm.T), " diag(1,3) KL =", kl_gauss0(S3, Rm @ S3 @ Rm.T))

print("== G  single edge with a stabilizing transport (passively removable control; see loop_dynamics_check.py) and self-loop variant")
# Finite tables on Z4.  Potential E = KL(nu_i||u) + KL(nu_j||u) + beta KL(nu_i || H# nu_j)
# with the unit Fisher flow d/dt log p = -(phi - E_p phi), phi the first variation.
beta = 1.0
u4 = np.full(4, 0.25)


def first_variations(ni: np.ndarray, nj: np.ndarray, Hm: list[int]) -> tuple[np.ndarray, np.ndarray]:
    Hnj = push(nj, Hm, 4)
    ratio = ni / Hnj
    phi_i = np.log(ni / u4) + beta * np.log(ratio)
    phi_j = np.log(nj / u4) - beta * np.array([ratio[Hm[y]] for y in range(4)])
    return phi_i, phi_j


def velocity(ni: np.ndarray, nj: np.ndarray, Hm: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pi_, pj_ = first_variations(ni, nj, Hm)
    return -(pi_ - ni @ pi_), -(pj_ - nj @ pj_)


def flow(ni: np.ndarray, nj: np.ndarray, Hm: list[int], T: float = 1.0, steps: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    li, lj = np.log(ni), np.log(nj)
    dt = T / steps
    def f_(li_: np.ndarray, lj_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.exp(li_ - li_.max()); a /= a.sum()
        b = np.exp(lj_ - lj_.max()); b /= b.sum()
        return velocity(a, b, Hm)
    for _ in range(steps):
        k1 = f_(li, lj); k2 = f_(li + dt / 2 * k1[0], lj + dt / 2 * k1[1])
        k3 = f_(li + dt / 2 * k2[0], lj + dt / 2 * k2[1]); k4 = f_(li + dt * k3[0], lj + dt * k3[1])
        li = li + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        lj = lj + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    a = np.exp(li - li.max()); a /= a.sum(); b = np.exp(lj - lj.max()); b /= b.sum()
    return a, b


ni0 = np.array([0.4, 0.3, 0.2, 0.1]); nj0 = u4.copy()  # source uniform: stabilized by every shift
Hid = [0, 1, 2, 3]; Hsh = [1, 2, 3, 0]
print("  cost values at t=0: id", kl(ni0, push(nj0, Hid, 4)), " shift", kl(ni0, push(nj0, Hsh, 4)))
vi_a, vj_a = velocity(ni0, nj0, Hid); vi_b, vj_b = velocity(ni0, nj0, Hsh)
print("  receiver log-velocity difference:", float(np.max(np.abs(vi_a - vi_b))))
print("  source log-velocity, id   :", np.round(vj_a, 6))
print("  source log-velocity, shift:", np.round(vj_b, 6))
ai, aj = flow(ni0, nj0, Hid); bi, bj = flow(ni0, nj0, Hsh)
print("  t=1 source laws differ by (max abs):", float(np.max(np.abs(aj - bj))), " receiver:", float(np.max(np.abs(ai - bi))))
print("  t=1 cost values: id", kl(ai, push(aj, Hid, 4)), " shift", kl(bi, push(bj, Hsh, 4)))
# Gaussian family with unit covariances: E = |mu_i|^2/2+|mu_j|^2/2+beta|mu_i-R mu_j|^2/2.
mu_i = np.array([1.0, 0.0]); mu_j = np.zeros(2)
for ang in [0.0, 0.7]:
    Rr = rot(ang)
    grad_j = mu_j - beta * Rr.T @ (mu_i - Rr @ mu_j)
    print(f"  Gaussian source mean velocity, rotation {ang}:", np.round(-grad_j, 6))
# Self-loop variant: E = KL(nu||r) + beta KL(nu || H# nu), nu(0) uniform (in Stab(H))
r = np.array([0.5, 0.25, 0.15, 0.10])


def self_loop_vel(p: np.ndarray, Hm: list[int]) -> np.ndarray:
    Hp_ = push(p, Hm, 4); ratio = p / Hp_
    phi = np.log(p / r) + beta * (np.log(ratio) + 1.0 - np.array([ratio[Hm[y]] for y in range(4)]))
    return -(phi - p @ phi)


def self_flow(Hm: list[int], T: float = 1.0, steps: int = 2000) -> np.ndarray:
    lp = np.log(u4.copy()); dt = T / steps
    def f_(l_: np.ndarray) -> np.ndarray:
        p = np.exp(l_ - l_.max()); p /= p.sum(); return self_loop_vel(p, Hm)
    for _ in range(steps):
        k1 = f_(lp); k2 = f_(lp + dt / 2 * k1); k3 = f_(lp + dt / 2 * k2); k4 = f_(lp + dt * k3)
        lp = lp + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = np.exp(lp - lp.max()); return p / p.sum()


print("  self-loop: loop-term velocity at uniform, id vs shift:",
      float(np.max(np.abs(self_loop_vel(u4, Hid) - self_loop_vel(u4, Hsh)))))
print("  self-loop: t=1 laws differ by:", float(np.max(np.abs(self_flow(Hid) - self_flow(Hsh)))))

print("== H  eq:agent-free-energy: posterior form = split form")
nM, nK, nO = 2, 3, 2
r_ = rand_simplex(nM)
Gk = rng.random((nM, nK)); Gk /= Gk.sum(1, keepdims=True)
L = rng.random((nK, nM, nO)); L /= L.sum(2, keepdims=True)
o = 1
Rjoint = (r_[:, None] * Gk)                          # R[m,k]
lik = L[:, :, o].T                                    # lik[m,k] = l(o|k,m)
Po = float(np.sum(Rjoint * lik))
post = Rjoint * lik / Po
zt = rng.random((nM, nK)); zt /= zt.sum()
F_post = kl(zt, post) - math.log(Po)
F_split = kl(zt, Rjoint) - float(np.sum(zt * np.log(lik)))
print("  F_post =", F_post, " F_split =", F_split, " residual =", abs(F_post - F_split))
L2 = L.copy(); L2[0, 0, o] = 0.0; L2[0, 0, 1 - o] = 1.0
lik2 = L2[:, :, o].T; Po2 = float(np.sum(Rjoint * lik2)); post2 = Rjoint * lik2 / Po2
with np.errstate(divide="ignore"):
    split2 = kl(zt, Rjoint) - float(np.sum(zt * np.log(lik2)))
print("  zeta charging a zero-likelihood cell: posterior form", kl(zt, post2) - math.log(Po2), " split form", split2)

print("== I  NEW: constant evaluator, m-dependent record kernel")
# K = {0}, M = {0,1}, O = {0,1}; G(m,.) = delta_0; l(k,m; o=1) = 0.2 + 0.6 m
for s_m in [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 1.0])]:
    p1 = float(s_m @ np.array([0.2, 0.8]))
    print("  model belief", s_m, "-> predicted P(o=1) =", round(p1, 6))
s0 = np.array([0.5, 0.5]); post_m = s0 * np.array([0.2, 0.8]); post_m /= post_m.sum()
print("  posterior over m after o=1 from uniform s:", np.round(post_m, 6))

print("== J  NEW: predictive equivalence without a shared generative model")
p_m = np.array([0.2, 0.7])
# agent i: G_i(m,.) = Ber(p_m) on K={0,1}, l_i(k,m;.) = delta_k
# agent j: G_j(m,.) = delta_0,      l_j(k,m;.) = Ber(p_m) (reads m directly)
Gi = np.array([[1 - p, p] for p in p_m]); Li = np.zeros((2, 2, 2))
for kk in range(2):
    Li[kk, :, kk] = 1.0
Gj = np.array([[1.0, 0.0], [1.0, 0.0]]); Lj = np.zeros((2, 2, 2))
for kk in range(2):
    for mm in range(2):
        Lj[kk, mm] = [1 - p_m[mm], p_m[mm]]
Krec_i = np.einsum("mk,kmo->mo", Gi, Li); Krec_j = np.einsum("mk,kmo->mo", Gj, Lj)
print("  composite kernels equal:", float(np.max(np.abs(Krec_i - Krec_j))))
# predictions and model posteriors after every record history of length 3 agree for every s
worst = 0.0
for s_ in [np.array([0.3, 0.7]), np.array([0.9, 0.1])]:
    for hist in itertools.product([0, 1], repeat=3):
        wi = s_.copy(); wj = s_.copy()
        for ob in hist:
            wi = wi * Krec_i[:, ob]; wj = wj * Krec_j[:, ob]
        worst = max(worst, float(np.max(np.abs(wi / wi.sum() - wj / wj.sum()))))
print("  max model-posterior difference over histories:", worst)
found = False
for phi_ in itertools.permutations(range(2)):
    for psi_ in itertools.permutations(range(2)):
        ok_r = True  # take r_i = r_j uniform, invariant under phi
        ok_G = all(np.allclose(Gj[phi_[m]], [Gi[m][psi_.index(kk)] for kk in range(2)]) for m in range(2))
        ok_l = all(np.allclose(Lj[psi_[kk], phi_[m]], Li[kk, m]) for kk in range(2) for m in range(2))
        found = found or (ok_r and ok_G and ok_l)
print("  any (phi, psi) satisfying eq:shared-model:", found)
print("  content posteriors after o=1 from m=1: agent i", Gi[1] * Li[:, 1, 1] / (Gi[1] @ Li[:, 1, 1]),
      " agent j", Gj[1] * Lj[:, 1, 1] / (Gj[1] @ Lj[:, 1, 1]))
# shared model (identical), same s, different belief conditionals -> different recognition predictions
s_ = np.array([0.5, 0.5])
q_a = np.array([[0.9, 0.1], [0.2, 0.8]]); q_b = np.array([[0.5, 0.5], [0.5, 0.5]])
pred_a = np.einsum("m,mk,kmo->o", s_, q_a, Li); pred_b = np.einsum("m,mk,kmo->o", s_, q_b, Li)
print("  identical models, equal s, different q(.|m): recognition predictions", np.round(pred_a, 6), np.round(pred_b, 6))

print("== K  prop:content-law, eq:content-projected (finite coefficient space)")
# Z = {0..11}; F_D: Z -> W = {0..3} non-injective; prior P0, record likelihood l_o(z) reading only F_D(z)
nZ, nW = 12, 4
FD = [z % nW for z in range(nZ)]
P0 = rand_simplex(nZ)
lw = rng.random(nW) + 0.1
lo = np.array([lw[FD[z]] for z in range(nZ)])
Zo = float(P0 @ lo); Pi = P0 * lo / Zo
Q = rand_simplex(nZ)
F_direct = kl(Q, P0) - float(Q @ np.log(lo))
QD = np.array([Q[[z for z in range(nZ) if FD[z] == w]].sum() for w in range(nW)])
PiD = np.array([Pi[[z for z in range(nZ) if FD[z] == w]].sum() for w in range(nW)])
cond = 0.0
for w in range(nW):
    zs = [z for z in range(nZ) if FD[z] == w]
    cond += QD[w] * kl(Q[zs] / QD[w], Pi[zs] / PiD[w])
F_proj = kl(QD, PiD) - math.log(Zo) + cond
print("  F(Q;o) direct =", F_direct, " projected form =", F_proj, " residual =", abs(F_direct - F_proj))
print("  records read only F_D, yet the conditional term is", cond, "(posterior conditional = prior conditional:",
      float(max(np.max(np.abs(Pi[[z for z in range(nZ) if FD[z] == w]] / PiD[w]
                               - P0[[z for z in range(nZ) if FD[z] == w]] / P0[[z for z in range(nZ) if FD[z] == w]].sum()))
                for w in range(nW))), ")")

print("== L  correlation section and companion examples")
prior = np.full((2, 2), 0.25); agree = np.array([[1.0, 0.0], [0.0, 1.0]])
postA = prior * agree; postA /= postA.sum()
print("  marginals", postA.sum(0), postA.sum(1), " E[XY] prior", prior[1, 1], " posterior", postA[1, 1],
      " MI", kl(postA, np.outer(postA.sum(1), postA.sum(0))))
best = math.inf; arg = None
grid = np.linspace(0, 1, 201)
for a in grid:
    for b in grid:
        Q = np.outer([1 - a, a], [1 - b, b])
        v = kl(Q, postA)
        if v < best:
            best, arg = v, (a, b)
print("  hard record: min over product laws of KL(Q||Pi) =", best, "at", arg, " log2 =", math.log(2))


def Hb(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log(p) + (1 - p) * math.log(1 - p))


ms = np.linspace(-0.999999, 0.999999, 200001)
for Jc in [0.5, 0.9, 1.0, 1.1, 2.0]:
    fvals = np.array([-2 * Hb((1 + m) / 2) - Jc * m * m for m in ms])
    m_opt = abs(float(ms[np.argmin(fvals)]))
    m_fix = 0.0
    if Jc > 1:
        lo, hi = 1e-9, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if math.tanh(Jc * mid) - mid > 0:
                lo = mid
            else:
                hi = mid
        m_fix = 0.5 * (lo + hi)
    print(f"  soft record J={Jc}: |m_opt| (grid) = {m_opt:.6f}, positive root of m=tanh(Jm) = {m_fix:.6f}")
mm_ = np.linspace(-0.99, 0.99, 397)
f2 = np.array([[-Hb((1 + a) / 2) - Hb((1 + b) / 2) - 2.0 * a * b for b in mm_] for a in mm_])
ia, ib = np.unravel_index(np.argmin(f2), f2.shape)
print("  J=2 unrestricted product optimum (m1, m2) =", (round(float(mm_[ia]), 4), round(float(mm_[ib]), 4)))
print("  companion mixture s_iG_i(k=1) =", 0.6 * 0.9 + 0.4 * 0.2)
sP = np.array([0.4, 0.6]); gP = np.array([[0.8, 0.2], [0.1, 0.9]])  # g[m, k]
geo = np.exp(sP @ np.log(gP)); geo /= geo.sum()
best = math.inf; qbest = None
for a in np.linspace(1e-4, 1 - 1e-4, 9999):
    q = np.array([1 - a, a]); val = float(sum(sP[m] * kl(q, gP[m]) for m in range(2)))
    if val < best:
        best, qbest = val, q
print("  product-restricted belief minimizer: geometric average", np.round(geo, 4), " brute force", np.round(qbest, 4),
      " arithmetic mixture", np.round(sP @ gP, 4))
