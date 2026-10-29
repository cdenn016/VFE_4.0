r"""D10-FORM check 4: inh:scalar-additive and the anisotropic remark (ch05 lines 1287-1327).

  (a) Jensen step: a non-affine G violates pairwise sum closure (numerical witness);
      an affine G satisfies it.
  (b) prefix property: for G = a(g) theta + c(g) with nonincreasing m(t), the selected partition
      sequence (centroid distance first reaches delta from above) equals an initial segment of
      count-weighted centroid agglomeration of the initial data; random trials.
  (c) the proposition does not constrain the null flow: two explicit cases where
      P != P^0 although the full flow is scalar-additive:
        (c1) finite horizon tau, full a = -3, null a^0 = -1;
        (c2) infinite horizon, full a = -lambda (interaction-only contraction), null a^0 = 0.
  (d) anisotropic example: diag(-1,-3), differences (1,0), (0,1.5) at t = 1.
  (e) reviewer extension (D10-FORM-16): scalar reciprocal pairwise dynamics
      x_i' = -(x_i - c_i) - sum_j phi_ij(x_i - x_j) with odd phi; the centroid retention of {0,1}
      passes the fiber test for linear member-independent cross forces and an arbitrary odd
      within-block force, and fails for a bounded-confidence cross force phi(u) = u exp(-u^2).
"""
from __future__ import annotations

import itertools as it

import numpy as np

rng = np.random.default_rng(5)
out: list[str] = []

# (a)
Gnon = lambda x: -x**3
x, y = 0.3, -0.7
xp, yp = -0.1, -0.3  # same sum
out.append(f"(a) non-affine G=-x^3: G(x)+G(y)={Gnon(x)+Gnon(y):.6f} vs G(x')+G(y')={Gnon(xp)+Gnon(yp):.6f} (same sum {x+y:.2f})")
Gaff = lambda x: -2 * x + 0.4
out.append(f"(a) affine G: {Gaff(x)+Gaff(y):.6f} vs {Gaff(xp)+Gaff(yp):.6f}")


def centroid_agglomeration(theta0):
    """count-weighted centroid agglomeration of scalar data: returns list of (merged blocks, height)."""
    blocks = [((i,), theta0[i], 1) for i in range(len(theta0))]
    seq = []
    while len(blocks) > 1:
        best = None
        for (p, q) in it.combinations(range(len(blocks)), 2):
            d = abs(blocks[p][1] - blocks[q][1])
            if best is None or d < best[0]:
                best = (d, p, q)
        d, p, q = best
        (mp, cp, npp), (mq, cq, nq) = blocks[p], blocks[q]
        new = (tuple(sorted(mp + mq)), (npp * cp + nq * cq) / (npp + nq), npp + nq)
        seq.append((new[0], d))
        blocks = [b for k, b in enumerate(blocks) if k not in (p, q)] + [new]
    return seq


def selector_run(theta0, mfun, tgrid, delta):
    """threshold selector with exact scalar-additive evolution: theta_i(t) = m(t) theta_i(0) + b(t)."""
    blocks = [((i,), 1) for i in range(len(theta0))]
    events = []
    cen0 = lambda blk: np.mean([theta0[i] for i in blk])
    for t in tgrid:
        mt = mfun(t)
        changed = True
        while changed:
            changed = False
            cand = []
            for (p, q) in it.combinations(range(len(blocks)), 2):
                D = abs(cen0(blocks[p][0]) - cen0(blocks[q][0]))
                if D > delta and D * mt <= delta:  # reached from above (initial distance above threshold)
                    cand.append((D, p, q))
            if cand:
                D, p, q = min(cand)
                new = (tuple(sorted(blocks[p][0] + blocks[q][0])), 1)
                blocks = [b for k, b in enumerate(blocks) if k not in (p, q)] + [new]
                events.append((new[0], round(D, 6)))
                changed = True
    return events


delta = 0.05
tgrid = np.linspace(0, 6, 6001)
ok = 0
trials = 30
for _ in range(trials):
    theta0 = np.sort(rng.uniform(-1, 1, size=5))
    if np.min(np.diff(theta0)) <= delta:
        theta0 = theta0 + np.arange(5) * delta
    mfun = lambda t: np.exp(-1.3 * t) * (1 + 0.3 * np.sin(t) ** 2) ** 0 # nonincreasing common factor
    ev = selector_run(theta0, mfun, tgrid, delta)
    agg = centroid_agglomeration(list(theta0))
    ok += int([e[0] for e in ev] == [a[0] for a in agg[: len(ev)]])
out.append(f"(b) selector sequence equals initial segment of centroid agglomeration in {ok}/{trials} random trials")

# (c1) finite horizon: full m = e^{-3t}, null m^0 = e^{-t}, horizon tau
theta0 = np.array([0.0, 0.3, 1.1, 1.5])
tau = 1.0
tg = np.linspace(0, tau, 2001)
evf = selector_run(theta0, lambda t: np.exp(-3 * t), tg, delta)
evn = selector_run(theta0, lambda t: np.exp(-1 * t), tg, delta)
out.append(f"(c1) horizon {tau}: full sequence {[e[0] for e in evf]}; null sequence {[e[0] for e in evn]}; differ: {[e[0] for e in evf] != [e[0] for e in evn]}")
# (c2) infinite horizon with interaction-only contraction
tg2 = np.linspace(0, 40, 4001)
evf2 = selector_run(theta0, lambda t: np.exp(-0.5 * t), tg2, delta)
evn2 = selector_run(theta0, lambda t: 1.0 + 0 * t, tg2, delta)
out.append(f"(c2) full a=-0.5, null a^0=0: full {[e[0] for e in evf2]}; null {[e[0] for e in evn2]}")

# (d) anisotropic propagator
U = np.diag(np.exp([-1.0, -3.0]))
d1, d2 = U @ np.array([1.0, 0.0]), U @ np.array([0.0, 1.5])
out.append(f"(d) diag(-1,-3) at t=1: (1,0)->{np.round(d1, 4)}, (0,1.5)->{np.round(d2, 4)}; norms {np.linalg.norm(d1):.4f} vs {np.linalg.norm(d2):.4f}")

# (e) centroid fiber test for scalar reciprocal pairwise dynamics
cc = np.array([0.0, 0.5, 1.4, 2.1])


def pair_field(xv, phi_in, phi_cross):
    n = len(xv)
    v = -(xv - cc)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = (i in (0, 1)) and (j in (0, 1))
            v[i] -= (phi_in if same else phi_cross)(xv[i] - xv[j])
    return v


lin = lambda u: 0.7 * u
odd_in = lambda u: 0.7 * u + 0.4 * u**3  # arbitrary odd within-block force
bc = lambda u: u * np.exp(-u**2)  # bounded-confidence cross force


def fiber_residual(phi_in, phi_cross, trials=200):
    res = 0.0
    for _ in range(trials):
        xv = rng.normal(size=4) * 1.5
        e = rng.normal()
        xp = xv.copy()
        xp[0] += e
        xp[1] -= e
        v, vp = pair_field(xv, phi_in, phi_cross), pair_field(xp, phi_in, phi_cross)
        r = np.array([(v[0] + v[1]) / 2 - (vp[0] + vp[1]) / 2, v[2] - vp[2], v[3] - vp[3]])
        res = max(res, float(np.max(np.abs(r))))
    return res


out.append(f"(e) centroid fiber residual, linear cross force + cubic odd within-block force: {fiber_residual(odd_in, lin):.2e}")
out.append(f"(e) centroid fiber residual, bounded-confidence cross force u*exp(-u^2): {fiber_residual(lin, bc):.3e}")

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
