r"""D11-FUS exact event-driven solution of the common-offset family in ORIGINAL time t.

On the common-offset class, u_i obeys \dot u_I = -kappa u_I + b_I between events, with
b_I = lambda (n_right - n_left).  An adjacent gap Delta between blocks I<J obeys
\dot\Delta = -kappa Delta - lambda (n_I + n_J), so it closes after
    dt = kappa^{-1} log(1 + kappa Delta / (lambda (n_I+n_J))).
No tau change of variables is used here, so this is an independent check of
fus:collision-times, fus:open-gap-condition, tau_final, fus:dyadic, and of the claim that the
membership sequence is the normalized-centroid-linkage (NCL) dendrogram of the initial data
(independent of lambda>0), which is the evidence for finding D11-FUS-01.
Single-threaded, a few seconds.
"""
import itertools
import numpy as np
import sympy as sp

# ---------------------------------------------------------------- symbolic four-agent times
kap, lam, d = sp.symbols('kappa lambda d', positive=True)
t1 = sp.log(1 + kap*d/(2*lam))/kap                       # outer gap d, n_I+n_J = 2
gap_mid_t1 = (4*d + 2*lam/kap)*sp.exp(-kap*t1) - 2*lam/kap   # middle gap 4d, rate 2 lambda
dt2 = sp.log(1 + kap*gap_mid_t1/(4*lam))/kap              # parents, n_I+n_J = 4
t2 = sp.simplify(t1 + dt2)
print("A1 parent gap at t1:", sp.simplify(gap_mid_t1), "(z-frame value 3d times e^{-kappa t1})")
print("A2 t2 - kappa^{-1} log(1+5 kappa d/(4 lambda)):",
      sp.simplify(sp.expand_log(t2 - sp.log(1 + 5*kap*d/(4*lam))/kap, force=True)))


def evolve(u0, n0, lam, kap, tol=1e-12):
    """Exact sticky evolution in original time; returns list of (t, partition) events."""
    order = np.argsort(u0)
    blocks = [[float(u0[i]), float(n0[i]), [int(i)]] for i in order]
    t = 0.0
    hist = []
    while len(blocks) > 1:
        ntot = sum(b[1] for b in blocks)
        left = np.cumsum([0.0] + [b[1] for b in blocks[:-1]])
        bvec = [lam*((ntot - left[k] - blocks[k][1]) - left[k]) for k in range(len(blocks))]
        dts = []
        for k in range(len(blocks) - 1):
            gap = blocks[k + 1][0] - blocks[k][0]
            dts.append(np.log1p(kap*gap/(lam*(blocks[k][1] + blocks[k + 1][1])))/kap)
        dtmin = min(dts)
        e = np.exp(-kap*dtmin)
        for k, b in enumerate(blocks):
            b[0] = bvec[k]/kap + (b[0] - bvec[k]/kap)*e
        t += dtmin
        merged = []
        k = 0
        while k < len(blocks):
            cur = blocks[k]
            while k + 1 < len(blocks) and dts[k] <= dtmin*(1 + 1e-12) + tol:
                nxt = blocks[k + 1]
                cur = [(cur[0]*cur[1] + nxt[0]*nxt[1])/(cur[1] + nxt[1]), cur[1] + nxt[1], cur[2] + nxt[2]]
                k += 1
            merged.append(cur)
            k += 1
        blocks = merged
        hist.append((t, tuple(sorted(tuple(sorted(b[2])) for b in blocks))))
    return hist


def ncl(u0, n0):
    """Normalized centroid linkage: repeatedly merge adjacent current blocks minimizing
    r = (c_J - c_I)/(n_I + n_J), computed from the INITIAL positions.  Returns (r, partition) list."""
    order = np.argsort(u0)
    blocks = [[float(u0[i]), float(n0[i]), [int(i)]] for i in order]
    out = []
    while len(blocks) > 1:
        rs = [(blocks[k + 1][0] - blocks[k][0])/(blocks[k][1] + blocks[k + 1][1]) for k in range(len(blocks) - 1)]
        rmin = min(rs)
        merged = []
        k = 0
        while k < len(blocks):
            cur = blocks[k]
            while k + 1 < len(blocks) and rs[k] <= rmin*(1 + 1e-12) + 1e-14:
                nxt = blocks[k + 1]
                cur = [(cur[0]*cur[1] + nxt[0]*nxt[1])/(cur[1] + nxt[1]), cur[1] + nxt[1], cur[2] + nxt[2]]
                k += 1
            merged.append(cur)
            k += 1
        blocks = merged
        out.append((rmin, tuple(sorted(tuple(sorted(b[2])) for b in blocks))))
    return out


def single_linkage(u0):
    """Single-linkage partition sequence of 1D points (merge adjacent gaps in increasing size)."""
    order = np.argsort(u0)
    xs = np.asarray(u0)[order]
    gaps = np.diff(xs)
    parent = list(range(len(xs)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    out = []
    for g in sorted(set(np.round(gaps, 12))):
        for k in np.where(np.abs(gaps - g) < 1e-12)[0]:
            parent[find(k + 1)] = find(k)
        groups = {}
        for k in range(len(xs)):
            groups.setdefault(find(k), []).append(int(order[k]))
        out.append(tuple(sorted(tuple(sorted(v)) for v in groups.values())))
    return out

# ---------------------------------------------------------------- B: four-agent example numerically
for (aa, cc, ll, dd) in [(1.0, 0.5, 0.3, 1.0), (2.0, 1.5, 1.0, 0.5), (3.0, 0.2, 0.05, 2.0)]:
    kk = aa - cc
    hist = evolve(np.array([-3*dd, -2*dd, 2*dd, 3*dd]), np.ones(4), ll, kk)
    t1c = np.log1p(kk*dd/(2*ll))/kk; t2c = np.log1p(5*kk*dd/(4*ll))/kk
    print(f"B  (a,c,lam,d)=({aa},{cc},{ll},{dd}): events {[round(h[0],10) for h in hist]} "
          f"closed forms ({t1c:.10f}, {t2c:.10f}); residuals {abs(hist[0][0]-t1c):.1e}, {abs(hist[-1][0]-t2c):.1e}")

# ---------------------------------------------------------------- C: open-gap condition iff and tau_final
rng = np.random.default_rng(7)
bad = 0; tested = 0; maxres = 0.0
for _ in range(4000):
    g = rng.uniform(0.05, 3.0, size=3)
    u0 = np.concatenate([[0.0], np.cumsum(g)])
    perm = rng.permutation(4); u0 = u0[perm]            # labels arbitrary
    ll = rng.uniform(0.05, 2.0); kk = rng.uniform(0.05, 2.0)
    hist = evolve(u0, np.ones(4), ll, kk)
    H = max(g[0], g[2]); D = g[1] + (g[0] + g[2])/2 - 2*H
    inv = np.argsort(perm)        # position index -> label
    A_ = tuple(sorted((int(inv[0]), int(inv[1])))); B_ = tuple(sorted((int(inv[2]), int(inv[3]))))
    two_stage = any(set(p) == {A_, B_} for _, p in hist) and len(hist) in (2, 3)
    # two-stage means: partition {A,B} occurs, preceded only by A or B forming
    parts = [p for _, p in hist]
    ok_pattern = (tuple(sorted((A_, B_))) in parts) and all(
        all(len(blk) <= 2 for blk in p) or len(p) == 1 for p in parts)
    if abs(D) > 1e-9:
        tested += 1
        if (D > 0) != ok_pattern:
            bad += 1
        if D > 0:
            tau_f = (np.exp(kk*hist[-1][0]) - 1)/kk
            maxres = max(maxres, abs(tau_f - (2*g[1] + g[0] + g[2])/(8*ll)))
print(f"C  open-gap iff: {tested} random cases, mismatches {bad}; max |tau_final - (2g2+g1+g3)/(8 lam)| = {maxres:.1e}")

# ---------------------------------------------------------------- D: NCL theorem and lambda-independence
mism = 0; lam_dep = 0; cases = 0; tres = 0.0
for _ in range(3000):
    r = rng.integers(2, 9)
    u0 = rng.normal(0, 2, size=r)
    n0 = rng.integers(1, 4, size=r).astype(float)
    kk = rng.uniform(0.05, 2.0)
    seqs = []
    for ll in (0.01, 0.3, 5.0):
        hist = evolve(u0, n0, ll, kk)
        seqs.append([p for _, p in hist])
        nc = ncl(u0, n0)
        # event times predicted by NCL heights: t = kappa^{-1} log(1 + kappa r/lam)
        tpred = [np.log1p(kk*rv/ll)/kk for rv, _ in nc]
        if [p for _, p in nc] == seqs[-1] and len(tpred) == len(hist):
            tres = max(tres, max(abs(tp - h[0]) for tp, h in zip(tpred, hist)))
    cases += 1
    if seqs[1] != [p for _, p in ncl(u0, n0)]:
        mism += 1
    if not (seqs[0] == seqs[1] == seqs[2]):
        lam_dep += 1
print(f"D  NCL: {cases} random common-offset configurations (r=2..8, multiplicities 1..3): "
      f"sequence != NCL in {mism}; lambda-dependence of the sequence in {lam_dep}; max event-time residual {tres:.1e}")

# ---------------------------------------------------------------- E: single linkage vs Phi on four agents
agree_pos = 0; n_pos = 0; disagree_neg = 0; n_negpos = 0
for _ in range(3000):
    g = rng.uniform(0.05, 3.0, size=3)
    u0 = np.concatenate([[0.0], np.cumsum(g)])
    H = max(g[0], g[2]); D = g[1] + (g[0] + g[2])/2 - 2*H
    seq = [p for _, p in evolve(u0, np.ones(4), 0.5, 0.5)]
    sl = single_linkage(u0)
    if D > 1e-9:
        n_pos += 1; agree_pos += (seq == sl)
    if D < -1e-9 and g[1] > H + 1e-9:
        n_negpos += 1; disagree_neg += (seq != sl)
print(f"E  admitted class D>0: Phi sequence == single-linkage sequence in {agree_pos}/{n_pos};"
      f" region D<0<g2-H: Phi != single linkage in {disagree_neg}/{n_negpos}")
u0 = np.array([0.0, 1.0, 3.2, 5.2])     # gaps (1, 2.2, 2): D=-0.3, g2-H=0.2
print("E2 gaps (1,2.2,2): Phi:", [p for _, p in evolve(u0, np.ones(4), 0.5, 0.5)], " SL:", single_linkage(u0))

# ---------------------------------------------------------------- F: dyadic depth 3 and within-level reordering
def dyadic_ok(u0, N):
    seq = [p for _, p in evolve(u0, np.ones(N), 1.0, 0.7)]
    k = int(np.log2(N))
    # every event must only join blocks inside the smallest dyadic block not yet complete, level by level
    level_done = 0
    for p in seq:
        sizes = sorted(len(b) for b in p)
        # each block must be a union of consecutive (in sorted label = position order) dyadic sub-blocks
        for b in p:
            lo, hi = min(b), max(b)
            sz = len(b)
            if hi - lo + 1 != sz or (sz & (sz - 1)) != 0 or lo % sz != 0:
                return False, seq
        # condition: no block of size > 2^l exists while some block of size < 2^l (dyadic incomplete) exists
        for l in range(1, k + 1):
            if any(len(b) > 2**l for b in p) and any(len(b) < 2**l for b in p):
                return False, seq
    return True, seq

ok_count = 0; total = 0
for _ in range(300):
    N = 8
    G1 = rng.uniform(0.5, 1.0, size=4)
    G2 = rng.uniform(4*1.01*G1.max(), 4*1.01*G1.max()*1.5, size=2)
    G3 = rng.uniform(4*1.01*G2.max(), 4*1.01*G2.max()*1.5)
    gaps = [G1[0], G2[0], G1[1], G3, G1[2], G2[1], G1[3]]
    u0 = np.concatenate([[0.0], np.cumsum(gaps)])
    ok, _ = dyadic_ok(u0, N); ok_count += ok; total += 1
print(f"F  dyadic condition (N/2)max_l < min_(l+1): dyadic history in {ok_count}/{total}")

# sharper condition 2^l max_l < min_(>l) (weaker than the manuscript's) still gives the dyadic history
ok_count = 0; total = 0; fails_ms = 0
for _ in range(300):
    G1 = rng.uniform(0.5, 1.0, size=4)
    G2 = rng.uniform(2*1.01*G1.max(), 2*1.01*G1.max()*1.3, size=2)
    G3 = rng.uniform(4*1.01*G2.max(), 4*1.01*G2.max()*1.3)
    gaps = [G1[0], G2[0], G1[1], G3, G1[2], G2[1], G1[3]]
    ms_ok = 4*G1.max() < G2.min() and 4*G2.max() < G3
    fails_ms += (not ms_ok)
    u0 = np.concatenate([[0.0], np.cumsum(gaps)])
    ok, _ = dyadic_ok(u0, 8); ok_count += ok; total += 1
print(f"F2 sharper condition 2^l max_l < min_(>l): dyadic history in {ok_count}/{total}; "
      f"of these, {fails_ms} violate the manuscript's condition")

# within-level reordering: level-2 gaps G_L=10 < G_R=10.5, but right quad's pairs merge earlier
gaps = [1.0, 10.0, 1.0, 50.0, 0.1, 10.5, 0.1]
u0 = np.concatenate([[0.0], np.cumsum(gaps)])
seq = [p for _, p in evolve(u0, np.ones(8), 1.0, 0.7)]
print("F3 Phi:", seq)
print("F3 SL :", single_linkage(u0))
print("F3 NCL:", [p for _, p in ncl(u0, np.ones(8))])

# ---------------------------------------------------------------- G: sync-time bounds on the common-offset class
worst_ratio_sharp = 0.0; worst_ms = 0.0
for _ in range(2000):
    r = rng.integers(2, 9)
    u0 = rng.normal(0, 3, size=r) + rng.normal(0, 5)
    n0 = rng.integers(1, 4, size=r).astype(float); N = n0.sum()
    aa = rng.uniform(0.2, 3.0); cc = rng.uniform(0.0, 0.95)*aa; kk = aa - cc
    ll = rng.uniform(0.05, 3.0); v0 = rng.normal(0, 2)
    Tsync = evolve(u0, n0, ll, kk)[-1][0]
    mq0 = u0 + v0; ms0 = u0 - v0
    R0 = np.sqrt((n0*(mq0**2 + ms0**2)).sum())
    t0 = np.log(max(1.0, 4*cc*R0/(ll*N)))/kk
    Bms = t0 + 4*R0*np.exp(-kk*t0)/(ll*N)
    M0 = max(np.ptp(mq0), np.ptp(ms0))
    Bsh = np.log1p(kk*M0/(ll*N))/kk
    worst_ratio_sharp = max(worst_ratio_sharp, Tsync/Bsh)
    worst_ms = max(worst_ms, Tsync/Bms)
print(f"G  common-offset sync: max T_sync/B_manuscript = {worst_ms:.4f}; max T_sync/B_sharp = {worst_ratio_sharp:.4f} (both must be <= 1)")
