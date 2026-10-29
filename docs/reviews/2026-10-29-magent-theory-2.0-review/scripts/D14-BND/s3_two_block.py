r"""D14-BND s3: rem:boundary-heterogeneous.

(a) Invariance of the two-block family (left leaves f^L(r,u(i,xi)), right leaves f^R(r,u(i,xi)),
    block-class-constant rows) under the fine flow, tested directly on the fine solution
    (no two-block profile equations are used): at t = 1 every left leaf must still be read
    from one common profile, and likewise on the right.
(b) The conjugacy condition f^L_1(r,s) + f^R_1(r,-s) independent of s: start on it with
    f^L_1 = f^R_1 but different deeper structure, and measure the root-sign dependence of the
    fresh root response at later times (coin realization, d = 2 and 3).
(c) Control: a two-block state in the S-orbit of the symmetric family keeps a sign-free root response.
"""
import numpy as np

from bnd_common import (setup, lift, class_rows, fine_velocity, rk4, normalize_profile,
                        lca_class, tree)

rng = np.random.default_rng(314159)


def two_block_state(fL, fR, gL, gR, FD, d, paths, rng, same_rows=False):
    N = 2 ** d
    qL, qR = lift(fL, FD.spq), lift(fR, FD.spq)
    sL, sR = lift(gL, FD.sps), lift(gR, FD.sps)
    left = np.array([paths[i][0][1] == 1 for i in range(N)])
    q = np.where(left[:, None, None], qL, qR)
    s = np.where(left[:, None, None], sL, sR)
    bL, bR = rng.dirichlet(np.ones(d) * 2), rng.dirichlet(np.ones(d) * 2)
    cL, cR = rng.dirichlet(np.ones(d) * 2), rng.dirichlet(np.ones(d) * 2)
    if same_rows:
        bR, cR = bL, cL
    beta = np.where(left[:, None], class_rows(bL, d, paths), class_rows(bR, d, paths))
    gamma = np.where(left[:, None], class_rows(cL, d, paths), class_rows(cR, d, paths))
    return (q, s, beta, gamma), left


def block_spread(q, sp, block_mask):
    """max over (r, u) of the spread of q_i(r,c) over block leaves i and cells c with u(i,r,c) = u."""
    N, R, C = q.shape
    worst = 0.0
    for r in range(R):
        vals = {}
        for i in np.where(block_mask)[0]:
            for c in range(C):
                vals.setdefault(sp.uidx[i, r, c], []).append(q[i, r, c])
        for v in vals.values():
            worst = max(worst, max(v) - min(v))
    return worst


def row_spread(beta, paths, block_mask, d):
    worst = 0.0
    for blk in (block_mask, ~block_mask):
        for k in range(d):
            vals = [beta[i, j] for i in np.where(blk)[0] for j in range(len(blk))
                    if i != j and lca_class(paths, i, j) == k]
            worst = max(worst, max(vals) - min(vals))
    return worst


def root_sign_dependence(q, sp):
    """TV between the fresh root response density and its root-sign-reversed version."""
    N, R, C = q.shape
    resp = q.mean(axis=0)                      # (R, C) density against A x Uc
    sig0 = sp.sig[:, :, 0]                     # root sign
    tv = 0.0
    for r in range(R):
        plus = resp[r, sig0[r] > 0].mean()
        minus = resp[r, sig0[r] < 0].mean()
        # within a sign cell the response is constant on the two-block family; TV of the flip:
        tv += sp.A[r] * 0.5 * abs(plus - minus)
    return tv


print("(a) two-block family under the fine flow (generic f^L != f^R, different base marginals)")
for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    fL = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    fR = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    gL = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    gR = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    x0, left = two_block_state(fL, fR, gL, gR, FD, d, paths, rng)
    x1 = rk4(fine_velocity, x0, 0.01, 100, FD)
    q1, s1, b1, g1 = x1
    print(f"   d={d}: belief block spreads L,R = {block_spread(q1, FD.spq, left):.2e}, {block_spread(q1, FD.spq, ~left):.2e};"
          f" model = {block_spread(s1, FD.sps, left):.2e}, {block_spread(s1, FD.sps, ~left):.2e};"
          f" row spreads = {row_spread(b1, paths, left, d):.2e}, {row_spread(g1, paths, left, d):.2e};"
          f" base-marginal gap |f^L_0 - f^R_0| at t=1 = {np.max(np.abs(q1[left].mean(axis=(0,2)) - q1[~left].mean(axis=(0,2)))):.3f}")
    # a perturbed state (break the block structure slightly) leaves the family: control
    q0p = x0[0].copy(); q0p[1] *= np.exp(0.05 * rng.normal(size=q0p[1].shape))
    lam = FD.spq.A[:, None] * FD.spq.Uc[None, :]
    q0p[1] /= np.sum(lam * q0p[1])
    x1p = rk4(fine_velocity, (q0p, x0[1], x0[2], x0[3]), 0.01, 100, FD)
    print(f"        control (non-family perturbation of one leaf): left block spread at t=1 = {block_spread(x1p[0], FD.spq, left):.2e}")

print("(b) conjugacy condition at t=0 (f^L_1 = f^R_1, deeper structure differs); root-sign TV of the fresh root response")
for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    base = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    # build fR with the same first-bit means as fL = base but different deeper structure
    fL = base.copy()
    half = 2 ** (d - 1)
    fR = base.copy()
    for r in range(2):
        for blk in (slice(0, half), slice(half, 2 ** d)):
            v = fR[r, blk]
            w = np.exp(0.8 * rng.normal(size=v.shape)); w *= v.mean() / w.mean()
            fR[r, blk] = w
    g = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    x0, left = two_block_state(fL, fR, g, g, FD, d, paths, rng, same_rows=True)
    out = []
    x = x0
    for t in range(4):
        out.append(root_sign_dependence(x[0], FD.spq))
        x = rk4(fine_velocity, x, 0.01, 50, FD)
    print(f"   d={d}: TV(root response, root-sign reversed) at t = 0, 0.5, 1.0, 1.5: " + ", ".join(f"{v:.2e}" for v in out))

print("(c) control: two-block state in the S-orbit of the symmetric family (f^R = f^L o tau, tau below level 1)")
for d in (2, 3):
    N, M, paths, FD, PD = setup(d, 2, 2, rng)
    A, B = FD.spq.A, FD.sps.A
    fL = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), A)
    # tau: reverse bit 2 under prefix u1 = +1 only (an automorphism acting below level 1)
    from bnd_common import all_bits, bits_to_idx
    U = all_bits(d)
    perm = [bits_to_idx((u[0], -u[1]) + u[2:]) if u[0] == 1 else bits_to_idx(u) for u in U]
    fR = fL[:, perm]
    g = normalize_profile(np.exp(0.6 * rng.normal(size=(2, 2 ** d))), B)
    x0, left = two_block_state(fL, fR, g, g, FD, d, paths, rng, same_rows=True)
    out = []
    x = x0
    for t in range(4):
        out.append(root_sign_dependence(x[0], FD.spq))
        x = rk4(fine_velocity, x, 0.01, 50, FD)
    print(f"   d={d}: TV at t = 0, 0.5, 1.0, 1.5: " + ", ".join(f"{v:.2e}" for v in out))
