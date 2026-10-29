r"""D14-BND s6: thm:boundary-tree-inheritance response/record clauses in both directions, and the
phase-loss geometry of sec:boundary-tree-loss (circle realization on a uniform grid).

Two fine configurations with the SAME profile (same descriptor at the relevant level) but a phase
rotated by an integer number of grid steps are compared under each protocol by total variation.
(a) fresh uniform-descendant responses (root, node L) and fresh independent two-draw laws;
    formula eq:boundary-tree-response
(b) reused hidden child two-draw law; formula eq:boundary-tree-reuse
(c) shared belief/model child label
(d) base-only broadcast record: equal component normalizers, posterior = lift of L f / Z_o,
    equal record laws, and continued fine flow = lift of the profile flow
(e) record reading the root circle coordinate; base-only record delivered to one root branch
(f) eq:boundary-tree-tv, eq:boundary-tree-hellinger, Fisher-Rao bounds and chord-sum growth
"""
import numpy as np

from bnd_common import (tree, make_space, lift, class_rows, fine_velocity, rk4, profile_velocity,
                        normalize_profile, setup, FineData)

rng = np.random.default_rng(99)
d = 2; Mc = 8
N, M, paths = tree(d)
A = np.array([0.45, 0.55]); B = np.array([0.6, 0.4])
alpha_q = np.array([[0.0, 2.0, 5.0], [3.0, 1.0, 6.0]])
alpha_s = np.array([[1.0, 4.0, 0.0], [2.0, 7.0, 3.0]])
f = normalize_profile(np.exp(0.7 * rng.normal(size=(2, 4))), A)
g = normalize_profile(np.exp(0.7 * rng.normal(size=(2, 4))), B)


def conf(aq, as_):
    spq = make_space(d, A, "circle", Mc=Mc, alpha=aq)
    sps = make_space(d, B, "circle", Mc=Mc, alpha=as_)
    return spq, sps, lift(f, spq), lift(g, sps)


def rot(alpha, node, j):
    a = alpha.copy(); a[:, node] = (a[:, node] + j) % Mc; return a


def tv(p, p2, w):
    return 0.5 * np.sum(w * np.abs(p - p2))


spq, sps, q, s = conf(alpha_q, alpha_s)
lam = A[:, None] * spq.Uc[None, :]
mu = B[:, None] * sps.Uc[None, :]
leftL = [0, 1]   # leaves below internal node 1 (L)
print("(a) fresh responses under a rotation of the phase at node L by 2 grid steps (pi/2)")
spq2, _, q2, _ = conf(rot(alpha_q, 1, 2), alpha_s)
root1, root2 = q.mean(axis=0), q2.mean(axis=0)
L1, L2 = q[leftL].mean(axis=0), q2[leftL].mean(axis=0)
print(f"   TV root response {tv(root1, root2, lam):.2e}; TV node-L response {tv(L1, L2, lam):.2e}")
# formula eq:boundary-tree-response: root density f_0(r); node L density f_1(r, u_1) with u_1 = sigma_0
f0 = f.mean(axis=1); f1 = np.stack([f[:, :2].mean(axis=1), f[:, 2:].mean(axis=1)], axis=1)  # (R, 2): u1=+1, -1
sig0 = spq.sig[:, :, 0]
L_formula = np.where(sig0 > 0, f1[:, [0]], f1[:, [1]])
print(f"   sup|root - f_0| = {np.max(np.abs(root1 - f0[:, None])):.2e}; sup|node L - f_1(r, sigma_0)| = {np.max(np.abs(L1 - L_formula)):.2e}")
# ancestor phase: rotating the ROOT phase changes the node-L response (ancestor data retained at level 1)
_, _, q3, _ = conf(rot(alpha_q, 0, 2), alpha_s)
print(f"   control: rotating the root phase changes the node-L response by TV {tv(L1, q3[leftL].mean(axis=0), lam):.4f} (retained at level 1)")
two1 = np.einsum('rc,sd->rcsd', L1, L1); two2 = np.einsum('rc,sd->rcsd', L2, L2)
w2 = np.einsum('rc,sd->rcsd', lam, lam)
print(f"   fresh independent two-draw law at node L: TV {tv(two1, two2, w2):.2e}")

print("(b) reused hidden child at node L, two draws")
re1 = 0.5 * sum(np.einsum('rc,sd->rcsd', q[i], q[i]) for i in leftL)
re2 = 0.5 * sum(np.einsum('rc,sd->rcsd', q2[i], q2[i]) for i in leftL)
print(f"   TV under the node-L rotation: {tv(re1, re2, w2):.4f}")
# formula: p(xi)p(xi')[1 + a(xi) a(xi') sigma(xi) sigma(xi')] with p = f_1(r,sigma_0), a from the lowest pair
pL = L_formula
fl = np.where(sig0 > 0, f[:, [0]], f[:, [2]]); fr = np.where(sig0 > 0, f[:, [1]], f[:, [3]])
aL = (fl - fr) / (fl + fr)
sL = spq.sig[:, :, 1]
formula = np.einsum('rc,sd->rcsd', pL, pL) * (1 + np.einsum('rc,sd->rcsd', aL * sL, aL * sL))
print(f"   sup|reuse law - eq:boundary-tree-reuse| = {np.max(np.abs(re1 - formula)):.2e}")

print("(c) shared belief/model child label at node L under a model-sector rotation at node L")
_, sps4, _, s4 = conf(alpha_q, rot(alpha_s, 1, 2))
sh1 = 0.5 * sum(np.einsum('rc,sd->rcsd', q[i], s[i]) for i in leftL)
sh2 = 0.5 * sum(np.einsum('rc,sd->rcsd', q[i], s4[i]) for i in leftL)
wqs = np.einsum('rc,sd->rcsd', lam, mu)
ind1 = np.einsum('rc,sd->rcsd', q[leftL].mean(axis=0), s[leftL].mean(axis=0))
ind2 = np.einsum('rc,sd->rcsd', q[leftL].mean(axis=0), s4[leftL].mean(axis=0))
print(f"   shared-label TV {tv(sh1, sh2, wqs):.4f}; separate fresh labels TV {tv(ind1, ind2, wqs):.2e}")

print("(d) base-only broadcast record L(o|r)")
Lrec = np.array([[0.7, 0.3], [0.2, 0.8]])  # L(o | r), rows r, columns o
o = 1
Z = np.array([np.sum(lam * Lrec[:, [o]] * q[i]) for i in range(N)])
qp = Lrec[None, :, [o]] * q / Z[:, None, None]
fp = Lrec[:, [o]] * f / np.sum(A * Lrec[:, o] * f0)
print(f"   component normalizers Z_i: {np.round(Z, 12)}; Z_o = {np.sum(A * f0 * Lrec[:, o]):.12f}")
print(f"   sup|posterior - lift(L f / Z_o)| = {np.max(np.abs(qp - lift(fp, spq))):.2e}")
# record law for a phase-rotated configuration
Pq1 = [np.sum(lam * Lrec[:, [oo]] * q.mean(axis=0)) for oo in range(2)]
Pq2 = [np.sum(lam * Lrec[:, [oo]] * q2.mean(axis=0)) for oo in range(2)]
print(f"   record law (root response readout): {np.round(Pq1, 12)} vs rotated {np.round(Pq2, 12)}")
# continue the coupled fine flow after the update and compare with the profile flow
N_, M_, paths_, FD, PD = setup(d, 2, 2, rng, realization="circle", Mc=Mc)
FD = FineData(spq, sps, PD.logrho, FD.Tq, FD.Ts, FD.pi_row, FD.edges)
PD.A, PD.B = A, B
logrho = PD.logrho
bq = np.array([0.55, 0.45]); bs = np.array([0.35, 0.65])
xf = rk4(fine_velocity, (qp, s, class_rows(bq, d, paths), class_rows(bs, d, paths)), 0.01, 50, FD)
xp = rk4(profile_velocity, (fp, g, bq, bs), 0.01, 50, PD)
print(f"   after update + flow to t=0.5: sup|q - lift f| = {np.max(np.abs(xf[0] - lift(xp[0], spq))):.2e}")

print("(e) records that read a public coordinate, or reach one root branch only")
# record kernel reading theta_0 through a fixed partition (beta = 0 grid units): ell(o | r, s)
beta0 = 0.0
theta0 = 2 * np.pi * (spq.vals[:, 0] + 0.5) / Mc
srec = np.sign(np.cos(theta0 - 2 * np.pi * beta0 / Mc))     # record's own sign of theta_0, (C,)
ell = {1: np.array([0.8, 0.3]), -1: np.array([0.2, 0.7])}  # ell(o=1 | r, s) for s = +-1, per r
def sign_record(qq):
    like = np.where(srec[None, :] > 0, ell[1][:, None], ell[-1][:, None])  # (R, C), o = 1
    Zi = np.array([np.sum(lam * like * qq[i]) for i in range(N)])
    return like[None] * qq / Zi[:, None, None]
_, _, q5, _ = conf(rot(alpha_q, 0, 2), alpha_s)   # root phase rotated: same root descriptor
print(f"   before: TV root responses (root phase rotated) {tv(q.mean(0), q5.mean(0), lam):.2e}")
print(f"   after the theta_0-reading record: TV root responses {tv(sign_record(q).mean(0), sign_record(q5).mean(0), lam):.4f}")
leftroot = [i for i in range(N) if paths[i][0][1] == 1]
def one_branch(qq):
    out = qq.copy()
    for i in leftroot:
        Zi = np.sum(lam * Lrec[:, [o]] * qq[i]); out[i] = Lrec[:, [o]] * qq[i] / Zi
    return out
print(f"   base-only record delivered to the left root branch only: TV root responses {tv(one_branch(q).mean(0), one_branch(q5).mean(0), lam):.4f}")
print(f"   base-only record broadcast to every agent:               TV root responses "
      f"{tv((Lrec[None,:,[o]]*q/np.array([np.sum(lam*Lrec[:,[o]]*q[i]) for i in range(N)])[:,None,None]).mean(0), (Lrec[None,:,[o]]*q5/np.array([np.sum(lam*Lrec[:,[o]]*q5[i]) for i in range(N)])[:,None,None]).mean(0), lam):.2e}")

print("(f) phase-loss geometry for one child-parent response (d = 1 response at the root, two-point base)")
Mcf = 25600
theta = 2 * np.pi * (np.arange(Mcf) + 0.5) / Mcf
pp = np.array([1.5, 0.8]); pm = np.array([0.5, 1.2]); Ab = np.array([0.4, 0.6])  # p_+, p_- per base point
def resp(shift):
    sg = np.sign(np.cos(theta - 2 * np.pi * shift / Mcf))
    return np.where(sg[None, :] > 0, pp[:, None], pm[:, None])  # density against Ab x uniform circle
w = Ab[:, None] / Mcf
base = resp(0)
EpmTV = np.sum(Ab * np.abs(pp - pm)); EpmH = np.sum(Ab * (np.sqrt(pp) - np.sqrt(pm)) ** 2)
for j in (256, 3200, 6400, 12800, 19200):
    delta = 2 * np.pi * j / Mcf
    r2 = resp(j)
    TV = 0.5 * np.sum(w * np.abs(base - r2))
    H2 = 0.5 * np.sum(w * (np.sqrt(base) - np.sqrt(r2)) ** 2)
    dl = min(delta, 2 * np.pi - delta)
    BC = np.sum(w * np.sqrt(base * r2)); dFR = 2 * np.arccos(min(1.0, BC))
    print(f"   delta={delta:.4f}: TV={TV:.8f} formula={dl/(2*np.pi)*EpmTV:.8f}; H^2={H2:.8f} formula={dl/(2*np.pi)*EpmH:.8f}; "
          f"d_FR={dFR:.6f} in [{2*np.sqrt(2*H2):.6f}, {np.pi*np.sqrt(2*H2):.6f}]")
for n in (10, 40, 160, 640):
    step = (Mcf // 4) // n
    chord = 0.0
    for k in range(n):
        a, b = resp(k * step), resp((k + 1) * step)
        chord += 2 * np.arccos(min(1.0, np.sum(w * np.sqrt(a * b))))
    print(f"   chord sum of the phase path over [0, pi/2], n={n}: {chord:.5f}; chord/sqrt(n) = {chord/np.sqrt(n):.5f}")
