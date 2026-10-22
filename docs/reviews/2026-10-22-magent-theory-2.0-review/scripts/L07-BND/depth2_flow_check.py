r"""Depth-two check of thm:boundary-tree-flow and thm:boundary-tree-inheritance.

Fine system: N = 4 leaf agents of a depth-2 binary tree.  Belief sample space is
r in {0,1,2} (reference A) times three circles (root, L, R).  Model sample space
is w in {0,1} (reference B) times three separate circles.  Each circle is the
uniform law on M = 4 points; with half-integer phase offsets every sign
sigma_v = sign cos(theta_v - alpha_v(r)) is exactly fair.  A finite uniform
circle replaces the Haar circle: the theorem only uses fair independent signs,
and on the continuous circle a fine state that is piecewise constant on these
cells evolves (identity channels, circle-free evaluator) exactly as this finite
table, by uniqueness.

The fine energy is the generic coupled potential of eq:dyn-potential with
identity channels, all nonself sources, uniform priors:
  E = sum_i KL(q_i x s_i || R) + sum_i [sum_j beta_ij KL(q_i||q_j) + T_q KL(beta_i||pi)]
      + (model sector),
and the fine flow is the unit-mobility Fisher flow eq:dyn-fisher-flow on all
probability tables.  The fine gradient is coded from the generic KL derivative
(receiver + incoming terms) and checked against finite differences of E.

Profile system: eq:boundary-tree-flow and eq:boundary-tree-rows.
"""
import numpy as np

rng = np.random.default_rng(7)
M = 4
nr, nw = 3, 2
A = np.array([0.2, 0.5, 0.3])
B = np.array([0.4, 0.6])
N = 4
Tq, Ts = 0.7, 1.3

# coupled evaluator density rho(r,w) against A x B, normalized
rho = np.exp(rng.normal(size=(nr, nw)) * 0.8)
rho /= (A[:, None] * B[None, :] * rho).sum()
logrho = np.log(rho)

def signs(offsets):
    """offsets[r] integer; sigma[r, m] = sign cos(2 pi (m - a - 1/2)/M)."""
    m = np.arange(M)
    s = np.sign(np.cos(2 * np.pi * (m[None, :] - offsets[:, None] - 0.5) / M))
    assert np.all(s != 0) and np.all(s.sum(axis=1) == 0)
    return s

# leaves: (eps1, eps2, internal node used at depth 1)
LEAVES = [(+1, +1, 1), (+1, -1, 1), (-1, +1, 2), (-1, -1, 2)]
NAMES = ["LL", "LR", "RL", "RR"]
# class k(i,j): number of shared initial edges
def cls(i, j):
    return 1 if LEAVES[i][2] == LEAVES[j][2] else 0
n_k = np.array([2, 1])
pi_k = n_k / (N - 1)

def bit_index(u):
    return np.where(u > 0, 0, 1)

def path_bits(phase, nb):
    """phase: list of 3 offset arrays (len nb). Returns list over leaves of (u1,u2)
    arrays with shape (nb, M, M, M) indexed [base, th0, th1, th2]."""
    s0, s1, s2 = (signs(phase[v]) for v in range(3))
    out = []
    for (e1, e2, v1) in LEAVES:
        u1 = e1 * s0[:, :, None, None] * np.ones((1, 1, M, M))
        sv = s1 if v1 == 1 else s2
        if v1 == 1:
            u2 = e2 * sv[:, None, :, None] * np.ones((1, M, 1, M))
        else:
            u2 = e2 * sv[:, None, None, :] * np.ones((1, M, M, 1))
        out.append((u1, u2))
    return out

def lift(f, bits):
    """f shape (nb,2,2) -> list of fine densities shape (nb,M,M,M)."""
    nb = f.shape[0]
    res = []
    rr = np.arange(nb)[:, None, None, None] * np.ones((1, M, M, M), dtype=int)
    for (u1, u2) in bits:
        res.append(f[rr, bit_index(u1), bit_index(u2)])
    return res

def pullback(qs, bits, nb):
    """average fine densities over leaves and cells with the same bits."""
    f = np.zeros((nb, 2, 2)); cnt = np.zeros((nb, 2, 2))
    rr = np.arange(nb)[:, None, None, None] * np.ones((1, M, M, M), dtype=int)
    for q, (u1, u2) in zip(qs, bits):
        np.add.at(f, (rr, bit_index(u1), bit_index(u2)), q)
        np.add.at(cnt, (rr, bit_index(u1), bit_index(u2)), 1)
    return f / cnt

# ---------------- fine system ----------------
refq = A[:, None, None, None] * np.ones((1, M, M, M)) / M**3
refs = B[:, None, None, None] * np.ones((1, M, M, M)) / M**3

def fine_energy(P, S, beta, gamma):
    E = 0.0
    for i in range(N):
        pb = P[i].sum(axis=(1, 2, 3)); sb = S[i].sum(axis=(1, 2, 3))
        E += (P[i] * np.log(P[i] / refq)).sum() + (S[i] * np.log(S[i] / refs)).sum()
        E -= (pb[:, None] * sb[None, :] * logrho).sum()
        for j in range(N):
            if j == i: continue
            E += beta[i, j] * (P[i] * np.log(P[i] / P[j])).sum()
            E += gamma[i, j] * (S[i] * np.log(S[i] / S[j])).sum()
        off = [j for j in range(N) if j != i]
        E += Tq * sum(beta[i, j] * np.log(beta[i, j] * (N - 1)) for j in off)
        E += Ts * sum(gamma[i, j] * np.log(gamma[i, j] * (N - 1)) for j in off)
    return E

def fine_grad(P, S, beta, gamma, incoming=True):
    hP, hS = [], []
    hb = np.zeros((N, N)); hg = np.zeros((N, N))
    for i in range(N):
        pb = P[i].sum(axis=(1, 2, 3)); sb = S[i].sum(axis=(1, 2, 3))
        Vq = -(sb[None, :] * logrho).sum(axis=1)       # shape (nr,)
        Vs = -(pb[:, None] * logrho).sum(axis=0)       # shape (nw,)
        h = np.log(P[i] / refq) + 1 + Vq[:, None, None, None]
        k = np.log(S[i] / refs) + 1 + Vs[:, None, None, None]
        for j in range(N):
            if j == i: continue
            h = h + beta[i, j] * (np.log(P[i] / P[j]) + 1)
            k = k + gamma[i, j] * (np.log(S[i] / S[j]) + 1)
            if incoming:
                h = h - beta[j, i] * P[j] / P[i]
                k = k - gamma[j, i] * S[j] / S[i]
            hb[i, j] = (P[i] * np.log(P[i] / P[j])).sum() + Tq * (np.log(beta[i, j] * (N - 1)) + 1)
            hg[i, j] = (S[i] * np.log(S[i] / S[j])).sum() + Ts * (np.log(gamma[i, j] * (N - 1)) + 1)
        hP.append(h); hS.append(k)
    return hP, hS, hb, hg

def fisher(u, h):
    return -u * (h - (u * h).sum())

def fine_rhs(state, incoming=True, gibbs=False):
    P, S, beta, gamma = state
    if gibbs:
        beta, gamma = gibbs_rows(P, S)
    hP, hS, hb, hg = fine_grad(P, S, beta, gamma, incoming)
    dP = [fisher(P[i], hP[i]) for i in range(N)]
    dS = [fisher(S[i], hS[i]) for i in range(N)]
    db = np.zeros((N, N)); dg = np.zeros((N, N))
    if not gibbs:
        for i in range(N):
            off = [j for j in range(N) if j != i]
            db[i, off] = fisher(beta[i, off], hb[i, off])
            dg[i, off] = fisher(gamma[i, off], hg[i, off])
    return [dP, dS, db, dg]

def gibbs_rows(P, S):
    beta = np.zeros((N, N)); gamma = np.zeros((N, N))
    for i in range(N):
        off = [j for j in range(N) if j != i]
        dq = np.array([(P[i] * np.log(P[i] / P[j])).sum() for j in off])
        ds = np.array([(S[i] * np.log(S[i] / S[j])).sum() for j in off])
        wq = np.exp(-dq / Tq); ws = np.exp(-ds / Ts)
        beta[i, off] = wq / wq.sum(); gamma[i, off] = ws / ws.sum()
    return beta, gamma

def add(state, d, c):
    P, S, b, g = state
    return [[P[i] + c * d[0][i] for i in range(N)], [S[i] + c * d[1][i] for i in range(N)],
            b + c * d[2], g + c * d[3]]

def rk4(rhs, state, dt, steps, **kw):
    for _ in range(steps):
        k1 = rhs(state, **kw)
        k2 = rhs(add(state, k1, dt / 2), **kw)
        k3 = rhs(add(state, k2, dt / 2), **kw)
        k4 = rhs(add(state, k3, dt), **kw)
        st = state
        for kk, c in ((k1, dt / 6), (k2, dt / 3), (k3, dt / 3), (k4, dt / 6)):
            st = add(st, kk, c)
        state = st
    return state

# ---------------- profile system ----------------
U = 0.25
def P0(phi):  # keep 0 bits, flip bit1, average bit2
    out = np.empty_like(phi)
    m = phi.mean(axis=2)  # (nb, 2) indexed by u1
    out[:, 0, :] = m[:, 1][:, None]
    out[:, 1, :] = m[:, 0][:, None]
    return out
def P1(phi):  # keep bit1, flip bit2
    return phi[:, :, ::-1]
PK = [P0, P1]

def D_k(f, ref):
    lf = np.log(f)
    return np.array([(ref * f * (lf - Pk(lf))).sum() for Pk in PK])

def prof_rhs(pstate, incoming=True, gibbs=False):
    f, g, bq, bs = pstate
    refA = A[:, None, None] * U; refB = B[:, None, None] * U
    f0 = f.mean(axis=(1, 2)); g0 = g.mean(axis=(1, 2))
    Vq = -(B[None, :] * g0[None, :] * logrho).sum(axis=1)
    Vs = -(A[:, None] * f0[:, None] * logrho).sum(axis=0)
    Dq, Ds = D_k(f, refA), D_k(g, refB)
    if gibbs:
        wq = pi_k * np.exp(-Dq / Tq); bq = wq / wq.sum()
        ws = pi_k * np.exp(-Ds / Ts); bs = ws / ws.sum()
    def force(x, V, b):
        lx = np.log(x)
        h = 2 * lx + V[:, None, None]
        for k, Pk in enumerate(PK):
            h = h - b[k] * (Pk(lx) + (Pk(x) / x if incoming else 0))
        return h
    hq = force(f, Vq, bq); hs = force(g, Vs, bs)
    df = -f * (hq - (refA * f * hq).sum())
    dg = -g * (hs - (refB * g * hs).sum())
    if gibbs:
        dbq = np.zeros(2); dbs = np.zeros(2)
    else:
        aq = Dq + Tq * np.log(bq / pi_k); as_ = Ds + Ts * np.log(bs / pi_k)
        dbq = -bq * (aq - (bq * aq).sum()); dbs = -bs * (as_ - (bs * as_).sum())
    return [df, dg, dbq, dbs]

def padd(s, d, c):
    return [s[i] + c * d[i] for i in range(4)]

def prk4(rhs, s, dt, steps, **kw):
    for _ in range(steps):
        k1 = rhs(s, **kw); k2 = rhs(padd(s, k1, dt / 2), **kw)
        k3 = rhs(padd(s, k2, dt / 2), **kw); k4 = rhs(padd(s, k3, dt), **kw)
        s = [s[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)]
    return s

def normalize_profile(f, base):
    return f / (base[:, None, None] * U * f).sum()

# ---------------- set up ----------------
f = normalize_profile(np.exp(rng.normal(size=(nr, 2, 2)) * 0.6), A)
g = normalize_profile(np.exp(rng.normal(size=(nw, 2, 2)) * 0.6), B)
bq = np.array([0.55, 0.45]); bs = np.array([0.3, 0.7])
phase_q = [np.array([0, 2, 1]), np.array([1, 0, 3]), np.array([3, 3, 0])]
phase_s = [np.array([2, 1]), np.array([0, 3]), np.array([1, 1])]
bits_q = path_bits(phase_q, nr); bits_s = path_bits(phase_s, nw)

def fine_from_profile(f, g, bq, bs, bits_q, bits_s):
    Q = lift(f, bits_q); Sd = lift(g, bits_s)
    P = [q * refq for q in Q]; S = [s * refs for s in Sd]
    beta = np.zeros((N, N)); gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                beta[i, j] = bq[cls(i, j)] / n_k[cls(i, j)]
                gamma[i, j] = bs[cls(i, j)] / n_k[cls(i, j)]
    return [P, S, beta, gamma]

fine0 = fine_from_profile(f, g, bq, bs, bits_q, bits_s)
P, S, beta, gamma = fine0
for i in range(N):
    assert abs(P[i].sum() - 1) < 1e-12 and abs(S[i].sum() - 1) < 1e-12

print("== 1. class costs: KL(q_i||q_j) vs D_k(f) ==")
Dq = D_k(f, A[:, None, None] * U)
res = 0.0
for i in range(N):
    for j in range(N):
        if i != j:
            kl = (P[i] * np.log(P[i] / P[j])).sum()
            res = max(res, abs(kl - Dq[cls(i, j)]))
print("D_k(f) =", Dq, " max |KL - D_k| =", res)

print("== 2. energy identity E = N F ==")
def prof_energy(f, g, bq, bs):
    refA = A[:, None, None] * U; refB = B[:, None, None] * U
    f0 = f.mean(axis=(1, 2)); g0 = g.mean(axis=(1, 2))
    F = (refA * f * np.log(f)).sum() + (refB * g * np.log(g)).sum()
    F -= (A[:, None] * f0[:, None] * B[None, :] * g0[None, :] * logrho).sum()
    F += (bq * D_k(f, refA)).sum() + Tq * (bq * np.log(bq / pi_k)).sum()
    F += (bs * D_k(g, refB)).sum() + Ts * (bs * np.log(bs / pi_k)).sum()
    return F
E = fine_energy(*fine0); F = prof_energy(f, g, bq, bs)
print("E =", E, " N F =", N * F, " residual =", abs(E - N * F))

print("== 3. fine gradient vs finite differences of E ==")
hP, hS, hb, hg = fine_grad(*fine0)
eps = 1e-6
worst = 0.0
for trial in range(3):
    dPv = [rng.normal(size=P[0].shape) * refq for _ in range(N)]
    dSv = [rng.normal(size=S[0].shape) * refs for _ in range(N)]
    dbv = rng.normal(size=(N, N)) * 0.01; np.fill_diagonal(dbv, 0)
    dgv = rng.normal(size=(N, N)) * 0.01; np.fill_diagonal(dgv, 0)
    sp = [[P[i] + eps * dPv[i] for i in range(N)], [S[i] + eps * dSv[i] for i in range(N)], beta + eps * dbv, gamma + eps * dgv]
    sm = [[P[i] - eps * dPv[i] for i in range(N)], [S[i] - eps * dSv[i] for i in range(N)], beta - eps * dbv, gamma - eps * dgv]
    fd = (fine_energy(*sp) - fine_energy(*sm)) / (2 * eps)
    an = sum((hP[i] * dPv[i]).sum() + (hS[i] * dSv[i]).sum() for i in range(N)) + (hb * dbv).sum() + (hg * dgv).sum()
    worst = max(worst, abs(fd - an) / max(1, abs(an)))
print("max relative FD residual =", worst)

print("== 4. fine Fisher flow vs profile flow, t in [0,1] ==")
dt, steps = 0.002, 500
fineT = rk4(fine_rhs, fine0, dt, steps)
profT = prk4(prof_rhs, [f, g, bq, bs], dt, steps)
fT, gT, bqT, bsT = profT
QT = lift(fT, bits_q); ST = lift(gT, bits_s)
rq = max(np.abs(fineT[0][i] / refq - QT[i]).max() for i in range(N))
rs = max(np.abs(fineT[1][i] / refs - ST[i]).max() for i in range(N))
rb = 0.0
for i in range(N):
    for j in range(N):
        if i != j:
            rb = max(rb, abs(fineT[2][i, j] - bqT[cls(i, j)] / n_k[cls(i, j)]),
                     abs(fineT[3][i, j] - bsT[cls(i, j)] / n_k[cls(i, j)]))
print("max|q_i(t) - f(t)(u(i))| =", rq, " model:", rs, " rows:", rb)
print("profile moved by |f(1)-f(0)|_max =", np.abs(fT - f).max(), " |b(1)-b(0)| =", np.abs(bqT - bq).max())
print("energy F(0) =", prof_energy(f, g, bq, bs), " F(1) =", prof_energy(*profT))

print("== 5. drop the incoming ratio P_k f/f in the profile flow ==")
profNI = prk4(prof_rhs, [f, g, bq, bs], dt, steps, incoming=False)
print("max|f_noincoming(1) - f(1)| =", np.abs(profNI[0] - fT).max())

print("== 5b. replace D_k by KL(f||P_k f) (averaged source) ==")
def prof_rhs_avg(pstate):
    f, g, bq, bs = pstate
    refA = A[:, None, None] * U; refB = B[:, None, None] * U
    f0 = f.mean(axis=(1, 2)); g0 = g.mean(axis=(1, 2))
    Vq = -(B[None, :] * g0[None, :] * logrho).sum(axis=1)
    Vs = -(A[:, None] * f0[:, None] * logrho).sum(axis=0)
    def kavg(x, ref):
        return np.array([(ref * x * (np.log(x) - np.log(Pk(x)))).sum() for Pk in PK])
    Dq, Ds = kavg(f, refA), kavg(g, refB)
    def force(x, V, b):
        lx = np.log(x)
        h = lx + V[:, None, None]
        for k, Pk in enumerate(PK):
            # d/dx KL(x||P_k x) = log x + 1 - log P_k x - P_k(x / P_k x)
            h = h + b[k] * (lx - np.log(Pk(x)) - Pk(x / Pk(x)))
        return h
    hq = force(f, Vq, bq); hs = force(g, Vs, bs)
    df = -f * (hq - (refA * f * hq).sum()); dg = -g * (hs - (refB * g * hs).sum())
    aq = Dq + Tq * np.log(bq / pi_k); as_ = Ds + Ts * np.log(bs / pi_k)
    return [df, dg, -bq * (aq - (bq * aq).sum()), -bs * (as_ - (bs * as_).sum())]
profAVG = prk4(lambda s: prof_rhs_avg(s), [f, g, bq, bs], dt, steps)
print("max|f_avgsource(1) - f(1)| =", np.abs(profAVG[0] - fT).max())

print("== 6. non-family fine initial condition leaves the family ==")
Ppert = [p.copy() for p in P]
noise = np.exp(0.05 * rng.normal(size=P[0].shape)); Ppert[0] = P[0] * noise; Ppert[0] /= Ppert[0].sum()
def family_defect(st):
    Q = [st[0][i] / refq for i in range(N)]
    fh = pullback(Q, bits_q, nr)
    L = lift(fh, bits_q)
    return max(np.abs(Q[i] - L[i]).max() for i in range(N))
pert0 = [Ppert, S, beta, gamma]
pertT = rk4(fine_rhs, pert0, dt, steps)
print("defect at t=0:", family_defect(pert0), " at t=1:", family_defect(pertT),
      " (family run defect at t=1:", family_defect(fineT), ")")

print("== 7. Gibbs-reduced variant ==")
fineG = rk4(fine_rhs, fine0, dt, steps, gibbs=True)
profG = prk4(prof_rhs, [f, g, bq, bs], dt, steps, gibbs=True)
QG = lift(profG[0], bits_q)
print("reduced flow max|q_i - f(u(i))| =", max(np.abs(fineG[0][i] / refq - QG[i]).max() for i in range(N)))

print("== 8. responses at t=1 (fine family run) ==")
Q = [fineT[0][i] / refq for i in range(N)]
root = sum(Q) / 4
f0T = fT.mean(axis=(1, 2))
print("root response residual:", np.abs(root - f0T[:, None, None, None]).max())
f1T = fT.mean(axis=2)   # (nr, 2) over u1
u1L = bits_q[0][0]      # leaf LL's first bit = +sigma_0 = node L path bit
rr = np.arange(nr)[:, None, None, None] * np.ones((1, M, M, M), dtype=int)
respL = (Q[0] + Q[1]) / 2
print("node-L response residual:", np.abs(respL - f1T[rr, bit_index(u1L)]).max())

print("== 9. phase rotation at node L: same profile flow, different child laws ==")
phase_q2 = [phase_q[0], phase_q[1] + 1, phase_q[2]]   # delta = pi/2 at node L
bits_q2 = path_bits(phase_q2, nr)
fine0b = fine_from_profile(f, g, bq, bs, bits_q2, bits_s)
fineTb = rk4(fine_rhs, fine0b, dt, steps)
Qb = [fineTb[0][i] / refq for i in range(N)]
fb_pull = pullback(Qb, bits_q2, nr)
print("pulled-back profile difference between phase variants:", np.abs(fb_pull - fT).max())
print("root response difference:", np.abs(sum(Qb) / 4 - root).max(),
      " node-L response difference:", np.abs((Qb[0] + Qb[1]) / 2 - respL).max())
tv_child = 0.5 * np.abs(Q[0] * refq - Qb[0] * refq).sum()
x = fT[:, :, 0]; y = fT[:, :, 1]   # branch responses at node L: f(r,u1,+1), f(r,u1,-1)
tv_formula = (np.pi / 2) / (2 * np.pi) * (A * (np.abs(x - y).mean(axis=1))).sum()
print("TV(child LL, rotated child LL) =", tv_child, " formula eq:boundary-tree-tv =", tv_formula)

print("== 10. base-dependent automorphism equivariance of the profile flow ==")
def tau(f):
    h = f.copy()
    h[0, 0, :] = f[0, 0, ::-1]         # r=0: swap children of the left node
    h[1] = f[1, ::-1, :]               # r=1: swap the root's branches
    return h                           # r=2: identity
profTau = prk4(prof_rhs, [tau(f), g, bq, bs], dt, steps)
print("max|(tau f)(t) - tau(f(t))| =", np.abs(profTau[0] - tau(fT)).max(),
      " rows:", np.abs(profTau[2] - bqT).max())

print("== 11. common base-record update commutes, family survives ==")
Lo = np.array([0.9, 0.2, 0.5])          # L(o|r) for the observed o
mid = rk4(fine_rhs, fine0, dt, 250)
pmid = prk4(prof_rhs, [f, g, bq, bs], dt, 250)
Z = (A * pmid[0].mean(axis=(1, 2)) * Lo).sum()
Zi = [(mid[0][i] * Lo[:, None, None, None]).sum() for i in range(N)]
print("component normalizers Z_i:", np.round(Zi, 12), " profile Z_o:", Z)
mid[0] = [mid[0][i] * Lo[:, None, None, None] / Zi[i] for i in range(N)]
pmid[0] = pmid[0] * Lo[:, None, None] / Z
fin = rk4(fine_rhs, mid, dt, 250); pfin = prk4(prof_rhs, pmid, dt, 250)
QF = lift(pfin[0], bits_q)
print("after record + flow, max|q_i - f(u(i))| =", max(np.abs(fin[0][i] / refq - QF[i]).max() for i in range(N)))

print("== 12. reuse of one hidden child (two draws) vs fresh draws ==")
def two_draw_reuse(Qa, Qb_):
    pa = Qa * refq; pb = Qb_ * refq
    return 0.5 * (np.multiply.outer(pa, pa) + np.multiply.outer(pb, pb))
R1 = two_draw_reuse(Q[0], Q[1]); R2 = two_draw_reuse(Qb[0], Qb[1])
print("TV(reuse law, rotated reuse law) =", 0.5 * np.abs(R1 - R2).sum())
F1 = np.multiply.outer(respL * refq, respL * refq); F2 = np.multiply.outer((Qb[0] + Qb[1]) / 2 * refq, (Qb[0] + Qb[1]) / 2 * refq)
print("TV(fresh two-draw law, rotated) =", 0.5 * np.abs(F1 - F2).sum())

print("== 13. shared belief/model label ==")
SM = [fineT[1][i] for i in range(N)]
SMb = [fineTb[1][i] for i in range(N)]
J1 = sum(np.multiply.outer(fineT[0][i], SM[i]) for i in range(N)) / 4
J2 = sum(np.multiply.outer(fineTb[0][i], SMb[i]) for i in range(N)) / 4
print("TV(shared-label joint law, belief-rotated) =", 0.5 * np.abs(J1 - J2).sum())
