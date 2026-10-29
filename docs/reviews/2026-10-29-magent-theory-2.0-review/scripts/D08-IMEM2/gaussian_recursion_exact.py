r"""Exact rational check of imem:gaussian-closure and a Gaussian local-retention test.

Lens D08-IMEM2.  Scope: appendices/interacting_memory.tex lines 560-580
(imem:gaussian-closure, imem:gaussian-reduced-observation,
imem:gaussian-reduced-recursion) and lines 617-661 (local retention).

Part A (closure).  Ground truth: the joint Gaussian of (Z, D, H_1..H_T) built
step by step from the protocol definition: the speaker's law
P(Y_j | D_j, H_{1:t}) is obtained by generic Gaussian conditioning of the
current joint law, a fresh sample is drawn and encoded H = B_j Ytilde + eta.
lambda_t = P(D | H_{1:t} = h) is then generic conditioning.  This is compared
EXACTLY (sympy rationals) with the manuscript's information-form recursion
driven only by (m_t, S_t) and the fixed reconstruction W, and with the
reconstruction Gamma_t = W lambda_t.

Part B (extension, not a manuscript claim).  With singleton presentations
the covariance S_t is history independent, so the pair of local recognition
laws at the actual private values is an affine image M_t m_t + const of the
mean m_t.  If d_0 != d_1, rank M_t <= 2 min(d_0, d_1) < d_0 + d_1, and two
histories can share both local laws while giving different receiver updates
after the same next message.  For d_0 = d_1 = 1 the map is injective.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import sympy as sp

Rat = sp.Rational


def blocks(sizes: Sequence[int]) -> List[List[int]]:
    out, k = [], 0
    for s in sizes:
        out.append(list(range(k, k + s)))
        k += s
    return out


class Model:
    """One-presentation affine Gaussian two-agent preparation."""

    def __init__(self, dx: Tuple[int, int], dy: Tuple[int, int], Gx, nu, F, om, Qc, Nc, B, Rc):
        self.dx, self.dy = dx, dy
        self.B, self.Rc = B, Rc
        SigX = Gx * Gx.T
        # variable layout of the preparation: Z = (X0, Y0, X1, Y1), D = (D0, D1)
        self.sizes = [dx[0], dy[0], dx[1], dy[1], dx[0], dx[1]]
        bl = blocks(self.sizes)
        self.iX = [bl[0], bl[2]]
        self.iY = [bl[1], bl[3]]
        self.iD = [bl[4], bl[5]]
        n = sum(self.sizes)
        # linear representation over sources (X jointly, Y-noise_i, D-noise_i)
        ns = dx[0] + dx[1] + dy[0] + dy[1] + dx[0] + dx[1]
        src = sp.zeros(ns, ns)
        sb = blocks([dx[0] + dx[1], dy[0], dy[1], dx[0], dx[1]])
        src[sb[0][0]:sb[0][-1] + 1, sb[0][0]:sb[0][-1] + 1] = SigX
        src[sb[1][0]:sb[1][-1] + 1, sb[1][0]:sb[1][-1] + 1] = Qc[0]
        src[sb[2][0]:sb[2][-1] + 1, sb[2][0]:sb[2][-1] + 1] = Qc[1]
        src[sb[3][0]:sb[3][-1] + 1, sb[3][0]:sb[3][-1] + 1] = Nc[0]
        src[sb[4][0]:sb[4][-1] + 1, sb[4][0]:sb[4][-1] + 1] = Nc[1]
        Cm = sp.zeros(n, ns)
        mean = sp.zeros(n, 1)
        xs = blocks(dx)
        for i in (0, 1):
            for r, v in enumerate(self.iX[i]):
                Cm[v, sb[0][xs[i][r]]] = 1
                mean[v] = nu[i][r]
            for r, v in enumerate(self.iD[i]):
                Cm[v, sb[0][xs[i][r]]] = 1
                Cm[v, sb[3 + i][r]] = 1
                mean[v] = nu[i][r]
            for r, v in enumerate(self.iY[i]):
                for c in range(dx[i]):
                    Cm[v, sb[0][xs[i][c]]] = F[i][r, c]
                Cm[v, sb[1 + i][r]] = 1
                mean[v] = om[i][r]
        self.mean0 = mean
        self.cov0 = Cm * src * Cm.T
        self.n0 = n


def sub(M, I, J):
    return M.extract(I, J)


def ground_truth(model: Model, schedule: Sequence[int], hs: Sequence[sp.Matrix]):
    """Joint (mean, cov) of (Z, D, H_1..H_T); returns lambda_t for t = 0..T."""
    mean, cov = model.mean0, model.cov0
    hidx: List[List[int]] = []
    for t, j in enumerate(schedule):
        U = model.iD[j] + [k for blk in hidx for k in blk]
        Yj = model.iY[j]
        CUU = sub(cov, U, U)
        K = sub(cov, Yj, U) * CUU.inv()
        mu0 = sub(mean, Yj, [0]) - K * sub(mean, U, [0])
        Sig_t = sub(cov, Yj, Yj) - K * sub(cov, U, Yj)
        Bj, Rj = model.B[j], model.Rc[j]
        n = cov.shape[0]
        m = Bj.shape[0]
        new_mean = Bj * (mu0 + K * sub(mean, U, [0]))
        cross = Bj * K * sub(cov, U, list(range(n)))           # Cov(H, all existing)
        varH = Bj * K * CUU * K.T * Bj.T + Bj * Sig_t * Bj.T + Rj
        cov2 = sp.zeros(n + m, n + m)
        cov2[:n, :n] = cov
        cov2[n:, :n] = cross
        cov2[:n, n:] = cross.T
        cov2[n:, n:] = varH
        mean2 = sp.zeros(n + m, 1)
        mean2[:n, 0] = mean
        mean2[n:, 0] = new_mean
        mean, cov = mean2, cov2
        hidx.append(list(range(n, n + m)))
    Didx = model.iD[0] + model.iD[1]
    Zidx = model.iX[0] + model.iY[0] + model.iX[1] + model.iY[1]
    lam = []
    post_joint = []
    for t in range(len(schedule) + 1):
        Hi = [k for blk in hidx[:t] for k in blk]
        ZD = Zidx + Didx
        if not Hi:
            lam.append((sub(mean, Didx, [0]), sub(cov, Didx, Didx)))
            post_joint.append((sub(mean, ZD, [0]), sub(cov, ZD, ZD)))
            continue
        hv = sp.Matrix.vstack(*hs[:t])
        G = sub(cov, Hi, Hi).inv()
        for I, store in ((Didx, lam), (ZD, post_joint)):
            CIH = sub(cov, I, Hi)
            mI = sub(mean, I, [0]) + CIH * G * (hv - sub(mean, Hi, [0]))
            SI = sub(cov, I, I) - CIH * G * CIH.T
            store.append((sp.simplify(mI), sp.simplify(SI)))
    return lam, post_joint, (mean, cov, hidx)


def reconstruction(model: Model):
    Didx = model.iD[0] + model.iD[1]
    Zidx = model.iX[0] + model.iY[0] + model.iX[1] + model.iY[1]
    SDD = sub(model.cov0, Didx, Didx)
    Cz = sub(model.cov0, Zidx, Didx) * SDD.inv()             # E[Z|D] = muZ + Cz (D - muD)
    Om = sub(model.cov0, Zidx, Zidx) - Cz * sub(model.cov0, Didx, Zidx)
    muZ = sub(model.mean0, Zidx, [0])
    muD = sub(model.mean0, Didx, [0])
    return Zidx, Didx, Cz, Om, muZ, muD


def recursion(model: Model, schedule: Sequence[int], hs: Sequence[sp.Matrix]):
    """The manuscript's reduced filter driven by (m_t, S_t) and W only."""
    Zidx, Didx, Cz, Om, muZ, muD = reconstruction(model)
    pos = {v: k for k, v in enumerate(Zidx)}
    dpos = blocks([model.dx[0], model.dx[1]])
    m, S = muD, sub(model.cov0, Didx, Didx)
    out = [(m, S)]
    lin = []
    for t, j in enumerate(schedule):
        Ij, Io = dpos[j], dpos[1 - j]
        rows = [pos[v] for v in model.iY[j]]
        Wy = Cz.extract(rows, list(range(len(Didx))))
        w0 = muZ.extract(rows, [0]) - Wy * muD
        OmY = Om.extract(rows, rows)
        Sjj, Soj = S.extract(Ij, Ij), S.extract(Io, Ij)
        reg = Soj * Sjj.inv()
        Vo = S.extract(Io, Io) - reg * S.extract(Ij, Io)
        coef_j = Wy.extract(list(range(Wy.rows)), Ij) + Wy.extract(list(range(Wy.rows)), Io) * reg
        const = w0 + Wy.extract(list(range(Wy.rows)), Io) * (m.extract(Io, [0]) - reg * m.extract(Ij, [0]))
        CovY = OmY + Wy.extract(list(range(Wy.rows)), Io) * Vo * Wy.extract(list(range(Wy.rows)), Io).T
        Bj, Rj = model.B[j], model.Rc[j]
        A = sp.zeros(Bj.rows, len(Didx))
        A[:, Ij[0]:Ij[-1] + 1] = Bj * coef_j
        bt = Bj * const
        Rt = Bj * CovY * Bj.T + Rj
        Sn = (S.inv() + A.T * Rt.inv() * A).inv()
        mn = Sn * (S.inv() * m + A.T * Rt.inv() * (hs[t] - bt))
        # linear part of m_{t+1} in m_t at fixed h (b_t is affine in m_t)
        dB = sp.zeros(Bj.rows, len(Didx))
        dB[:, Io[0]:Io[-1] + 1] = Bj * Wy.extract(list(range(Wy.rows)), Io)
        dB[:, Ij[0]:Ij[-1] + 1] = -Bj * Wy.extract(list(range(Wy.rows)), Io) * reg
        Phi = Sn * (S.inv() - A.T * Rt.inv() * dB)
        lin.append(Phi)
        m, S = sp.simplify(mn), sp.simplify(Sn)
        out.append((m, S))
    return out, lin


def local_mean_map(model: Model, S, dvals):
    """Linear part (in m) of the stacked local means E[Z_i | d_i, H], i = 0, 1."""
    Zidx, Didx, Cz, Om, muZ, muD = reconstruction(model)
    pos = {v: k for k, v in enumerate(Zidx)}
    dpos = blocks([model.dx[0], model.dx[1]])
    rows_all = []
    for i in (0, 1):
        Ii, Io = dpos[i], dpos[1 - i]
        rows = [pos[v] for v in model.iX[i] + model.iY[i]]
        reg = S.extract(Io, Ii) * S.extract(Ii, Ii).inv()
        Co = Cz.extract(rows, Io)
        M = sp.zeros(len(rows), len(Didx))
        M[:, Io[0]:Io[-1] + 1] = Co
        M[:, Ii[0]:Ii[-1] + 1] = -Co * reg
        rows_all.append(M)
    return sp.Matrix.vstack(*rows_all), rows_all


def run(label: str, model: Model, schedule, hs) -> Dict[str, object]:
    lam_gt, post_gt, _ = ground_truth(model, schedule, hs)
    rec, lin = recursion(model, schedule, hs)
    Zidx, Didx, Cz, Om, muZ, muD = reconstruction(model)
    ok_rec = all(sp.simplify(a[0] - b[0]) == sp.zeros(*a[0].shape) and
                 sp.simplify(a[1] - b[1]) == sp.zeros(*a[1].shape) for a, b in zip(lam_gt, rec))
    ok_W = True
    for (mt, St), (mj, Sj) in zip(rec, post_gt):
        mz = muZ + Cz * (mt - muD)
        Sz = Om + Cz * St * Cz.T
        mj_rec = sp.Matrix.vstack(mz, mt)
        Sj_rec = sp.Matrix(sp.BlockMatrix([[Sz, Cz * St], [St * Cz.T, St]]))
        ok_W &= (sp.simplify(mj - mj_rec) == sp.zeros(*mj.shape) and sp.simplify(Sj - Sj_rec) == sp.zeros(*Sj.shape))
    # Sylvester criterion: all leading principal minors positive (exact rationals)
    pd_ok = all(all(St[:k, :k].det() > 0 for k in range(1, St.rows + 1)) for _, St in rec)
    print(f"[{label}] recursion == generic conditioning for t=0..{len(schedule)}: {ok_rec}")
    print(f"[{label}] Gamma_t == W lambda_t (joint mean and covariance)       : {ok_W}")
    print(f"[{label}] S_t positive definite at every step                    : {pd_ok}")
    return {"rec": rec, "lin": lin, "lam": lam_gt}


# ---------------------------------------------------------------- Part A
Gx = sp.Matrix([[1, 0, 0], [Rat(1, 2), 1, 0], [Rat(1, 3), Rat(-1, 2), 1]])
model_v = Model(
    dx=(1, 2), dy=(1, 2), Gx=Gx,
    nu=[sp.Matrix([0]), sp.Matrix([1, -1])],
    F=[sp.Matrix([[2]]), sp.Matrix([[1, -1], [Rat(1, 2), 1]])],
    om=[sp.Matrix([Rat(1, 2)]), sp.Matrix([0, 1])],
    Qc=[sp.Matrix([[1]]), sp.Matrix([[1, Rat(1, 4)], [Rat(1, 4), 1]])],
    Nc=[sp.Matrix([[Rat(1, 2)]]), sp.Matrix([[1, 0], [0, 2]])],
    B=[sp.Matrix([[1]]), sp.Matrix([[1, 0], [1, 1]])],
    Rc=[sp.Matrix([[Rat(1, 3)]]), sp.Matrix([[1, 0], [0, Rat(1, 2)]])],
)
schedule = [1, 0, 1]
hs = [sp.Matrix([Rat(1, 2), -1]), sp.Matrix([2]), sp.Matrix([0, 1])]
resv = run("vector d=(1,2)", model_v, schedule, hs)

# S_t is history independent: rerun with different messages
hs2 = [sp.Matrix([-3, Rat(2, 5)]), sp.Matrix([Rat(-1, 7)]), sp.Matrix([5, 0])]
rec2, _ = recursion(model_v, schedule, hs2)
print("[vector] S_t identical for two different histories           :",
      all(sp.simplify(a[1] - b[1]) == sp.zeros(3, 3) for a, b in zip(resv["rec"], rec2)))

# ---------------------------------------------------------------- Part B
print("\nPart B: local-only retention in the affine Gaussian family")
rec = resv["rec"]
t = 2                                            # after H1 (agent 1) and H2 (agent 0)
St, St1 = rec[t][1], rec[t + 1][1]
M_t, _ = local_mean_map(model_v, St, None)
print(f"[vector] rank of local-mean map M_{t} (m_t in R^3)              : {M_t.rank()}")
ker = M_t.nullspace()
dm = ker[0]
print(f"[vector] kernel direction dm                                   : {list(dm)}")
# reachability: m_t = E[D | H_{1:t}] has covariance Cov(D) - S_t
Didx = model_v.iD[0] + model_v.iD[1]
CovMean = sub(model_v.cov0, Didx, Didx) - St
print(f"[vector] rank Cov(m_{t}) = rank(Cov(D) - S_{t})                     : {CovMean.rank()}  (3 = m_t has a density on R^3)")
Phi = resv["lin"][t]                              # linear part of the update at step t+1 (agent 1 speaks)
dm_next = sp.simplify(Phi * dm)
M_t1, (M0_t1, M1_t1) = local_mean_map(model_v, St1, None)
d_recv = sp.simplify(M0_t1 * dm_next)             # agent 0 is the receiver at step t+1
d_send = sp.simplify(M1_t1 * dm_next)
print(f"[vector] equal local laws at t={t}: M_t dm                         : {list(sp.simplify(M_t * dm))}")
print(f"[vector] receiver (agent 0) local-mean change after same H3     : {list(d_recv)}")
print(f"[vector] sender  (agent 1) local-mean change after same H3      : {list(d_send)}")

# explicit pair of histories: h' = h + dh with Psi dh = dm, Psi = d m_t / d h_{1:t}
mean, cov, hidx = ground_truth(model_v, schedule, hs)[2]
Hi = [k for blk in hidx[:t] for k in blk]
Psi = sub(cov, Didx, Hi) * sub(cov, Hi, Hi).inv()
sol = Psi.gauss_jordan_solve(dm)[0]
free = sol.free_symbols
dh = sp.simplify(sol.subs({s_: 0 for s_ in free}))
hs_alt = [hs[0] + dh[0:2, 0], hs[1] + dh[2:3, 0], hs[2]]
rec_alt, _ = recursion(model_v, schedule, hs_alt)


def local_laws(model, rec_t, dvals):
    """Exact local means/covariances of Z_i at the actual private values d_i."""
    m, S = rec_t
    Zidx, Didx_, Cz, Om, muZ, muD = reconstruction(model)
    pos = {v: k for k, v in enumerate(Zidx)}
    dpos = blocks([model.dx[0], model.dx[1]])
    res = []
    for i in (0, 1):
        Ii, Io = dpos[i], dpos[1 - i]
        reg = S.extract(Io, Ii) * S.extract(Ii, Ii).inv()
        mo = m.extract(Io, [0]) + reg * (dvals[i] - m.extract(Ii, [0]))
        Vo = S.extract(Io, Io) - reg * S.extract(Ii, Io)
        rows = [pos[v] for v in model.iX[i] + model.iY[i]]
        Ci, Co = Cz.extract(rows, Ii), Cz.extract(rows, Io)
        mean_i = muZ.extract(rows, [0]) + Ci * (dvals[i] - muD.extract(Ii, [0])) + Co * (mo - muD.extract(Io, [0]))
        cov_i = Om.extract(rows, rows) + Co * Vo * Co.T
        res.append((sp.simplify(mean_i), sp.simplify(cov_i)))
    return res


dvals = [sp.Matrix([Rat(1, 3)]), sp.Matrix([-1, 2])]
LA = local_laws(model_v, rec[t], dvals)
LB = local_laws(model_v, rec_alt[t], dvals)
print(f"[vector] histories h, h' differ (dh = {list(dh)})")
print("[vector] lambda_2 means differ                               :", sp.simplify(rec[t][0] - rec_alt[t][0]) != sp.zeros(3, 1))
print("[vector] both local laws equal at t=2 (means and covariances) :",
      all(sp.simplify(a[0] - b[0]) == sp.zeros(*a[0].shape) and sp.simplify(a[1] - b[1]) == sp.zeros(*a[1].shape)
          for a, b in zip(LA, LB)))
LA3 = local_laws(model_v, rec[t + 1], dvals)
LB3 = local_laws(model_v, rec_alt[t + 1], dvals)
print("[vector] receiver local law after the same H3 differs         :", sp.simplify(LA3[0][0] - LB3[0][0]) != sp.zeros(2, 1),
      "  difference:", list(sp.simplify(LA3[0][0] - LB3[0][0])))
print("[vector] sender local law after H3 unchanged and equal        :", sp.simplify(LA3[1][0] - LB3[1][0]) == sp.zeros(4, 1))

# scalar contrast: d0 = d1 = 1
Gs = sp.Matrix([[1, 0], [Rat(2, 3), 1]])
model_s = Model(
    dx=(1, 1), dy=(1, 1), Gx=Gs,
    nu=[sp.Matrix([0]), sp.Matrix([1])],
    F=[sp.Matrix([[-1]]), sp.Matrix([[2]])],
    om=[sp.Matrix([0]), sp.Matrix([1])],
    Qc=[sp.Matrix([[1]]), sp.Matrix([[Rat(1, 2)]])],
    Nc=[sp.Matrix([[1]]), sp.Matrix([[Rat(1, 3)]])],
    B=[sp.Matrix([[1]]), sp.Matrix([[2]])],
    Rc=[sp.Matrix([[1]]), sp.Matrix([[Rat(1, 2)]])],
)
sched_s = [1, 0, 1, 0]
hs_s = [sp.Matrix([1]), sp.Matrix([-1]), sp.Matrix([2]), sp.Matrix([0])]
ress = run("scalar d=(1,1)", model_s, sched_s, hs_s)
ranks = []
for tt, (m_, S_) in enumerate(ress["rec"]):
    Mt, _ = local_mean_map(model_s, S_, None)
    ranks.append(Mt.rank())
print("[scalar] rank of local-mean map at t=0..4 (2 = injective)     :", ranks)


# ---------------------------------------------------------------- Part B, independent confirmation
# Local laws computed directly from the generic joint of (Z, D, H_1..H_3), conditioning
# Z_i on (D_i = d_i, H_{1:t} = h); no use of the reduced recursion.
def local_from_joint(model, schedule, hs_, dvals_, t_):
    mean, cov, hidx = ground_truth(model, schedule, hs_)[2]
    out = []
    for i in (0, 1):
        Zi = model.iX[i] + model.iY[i]
        Ci = model.iD[i] + [k for blk in hidx[:t_] for k in blk]
        vals = sp.Matrix.vstack(dvals_[i], *hs_[:t_]) if t_ else dvals_[i]
        G = sub(cov, Ci, Ci).inv()
        CZ = sub(cov, Zi, Ci)
        mz = sub(mean, Zi, [0]) + CZ * G * (vals - sub(mean, Ci, [0]))
        Sz = sub(cov, Zi, Zi) - CZ * G * CZ.T
        out.append((sp.simplify(mz), sp.simplify(Sz)))
    return out


gtA2 = local_from_joint(model_v, schedule, hs, dvals, 2)
gtB2 = local_from_joint(model_v, schedule, hs_alt, dvals, 2)
gtA3 = local_from_joint(model_v, schedule, hs, dvals, 3)
gtB3 = local_from_joint(model_v, schedule, hs_alt, dvals, 3)
print("\n[independent] generic-joint local laws equal at t=2 (both agents):",
      all(sp.simplify(a[0] - b[0]) == sp.zeros(*a[0].shape) and sp.simplify(a[1] - b[1]) == sp.zeros(*a[1].shape)
          for a, b in zip(gtA2, gtB2)))
diff3 = sp.simplify(gtA3[0][0] - gtB3[0][0])
print("[independent] receiver mean after same H3 differs:", diff3 != sp.zeros(2, 1),
      " float:", [float(v) for v in diff3])
print("[independent] sender local law after H3 equal   :",
      sp.simplify(gtA3[1][0] - gtB3[1][0]) == sp.zeros(4, 1) and sp.simplify(gtA3[1][1] - gtB3[1][1]) == sp.zeros(4, 4))
lam_alt_gt = ground_truth(model_v, schedule, hs_alt)[0]
print("[independent] recursion == generic conditioning on h' too   :",
      all(sp.simplify(a[0] - b[0]) == sp.zeros(3, 1) and sp.simplify(a[1] - b[1]) == sp.zeros(3, 3)
          for a, b in zip(lam_alt_gt, rec_alt)))
print("[independent] local laws from joint == local laws via W lambda (t=2,3):",
      all(sp.simplify(a[0] - b[0]) == sp.zeros(*a[0].shape) for a, b in zip(gtA2 + gtA3, LA + LA3)))


# ---------------------------------------------------------------- Part B, almost-sure version
# Under the protocol law of H_{1:3}, is the receiver's local mean after H_3 a
# function of (both local means at t=2, H_3)?  Compute Var(T3 | T2, H3) exactly.
mean, cov, hidx = ground_truth(model_v, schedule, hs)[2]
H12 = [k for blk in hidx[:2] for k in blk]
H123 = [k for blk in hidx[:3] for k in blk]
nH = len(H123)


def local_mean_coeff(i, Hset):
    """Coefficient row-block of E[Z_i | D_i = d_i, H_Hset] on H_{1:3} (zeros beyond Hset)."""
    Zi = model_v.iX[i] + model_v.iY[i]
    Ci = model_v.iD[i] + Hset
    Gm = sub(cov, Ci, Ci).inv()
    full = sub(cov, Zi, Ci) * Gm                     # coefficients on (D_i, H_Hset)
    co = sp.zeros(len(Zi), nH)
    nd = len(model_v.iD[i])
    for c, hk in enumerate(Hset):
        co[:, H123.index(hk)] = full[:, nd + c]
    return co


T2 = sp.Matrix.vstack(local_mean_coeff(0, H12), local_mean_coeff(1, H12))
H3sel = sp.zeros(len(hidx[2]), nH)
for r, hk in enumerate(hidx[2]):
    H3sel[r, H123.index(hk)] = 1
Cond = sp.Matrix.vstack(T2, H3sel)                   # conditioning variables, linear in H_{1:3}
Tgt = local_mean_coeff(0, H123)                      # receiver local mean after H_3
SH = sub(cov, H123, H123)                            # protocol covariance of H_{1:3}
CC = Cond * SH * Cond.T
# CC may be singular (T2 has repeated information); use a basis of its row space
rows = []
basis = sp.zeros(0, nH)
for r in range(Cond.rows):
    trial = sp.Matrix.vstack(basis, Cond[r, :])
    if trial.rank() > basis.rows:
        basis = trial
CCb = basis * SH * basis.T
resid = Tgt * SH * Tgt.T - Tgt * SH * basis.T * CCb.inv() * basis * SH * Tgt.T
resid = sp.simplify(resid)
print("\n[a.s.] rank of conditioning information (T2, H3)         :", basis.rows, "of", Cond.rows)
print("[a.s.] Var(receiver local mean after H3 | T2, H3), trace :", float(resid.trace()),
      " rank:", resid.rank())
