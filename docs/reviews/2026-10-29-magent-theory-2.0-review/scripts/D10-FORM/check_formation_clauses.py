r"""D10-FORM check 5: clause-by-clause test of inh:recursive-formation on an anchored linear
consensus model (reviewer construction, not a manuscript example).

Agents i = 0..3 have belief laws N(x_i, 1) on R (restricted Gaussian location family, Fisher metric 1),
evaluators N(c_i, 1), identity channels, fixed rows w = 1/4 on every nonself source. The reciprocal
potential is E = sum_i (x_i - c_i)^2 / 2 + w sum_i sum_{j != i} (x_i - x_j)^2 / 2, so
  dx_i/dt = -(x_i - c_i) - 2 w sum_{j != i} (x_i - x_j).
Null flow (peer costs deleted): dx_i/dt = -(x_i - c_i).
Selector: join two current blocks when their centroid distance reaches delta from above and stays
below delta for the rest of the run (invariant block relation). Retention of a block I: its
centroid, with all other coordinates kept. Parent protocol class: report the average of one fresh
sample from each child of I. Child-level probe: one fresh sample from a named child.

  (F3,D) centroid retention passes the fiber test (random fiber pairs, residual);
  (F3,B) averaging protocol law N(xbar_I, 1/|I|) factors through the retention;
  (F4)   a child-level probe separates two points of one fiber;
  (F5)   full flow: {0,1}, {2,3}, then union; null flow: no event; checked on random perturbations;
  (F6a)  the union occurs under the inherited centroid dynamics;
  (F6b)  counterfactual handoffs: every handoff with parent gap > delta unites (fails under a
         restricted class); handoffs with parent gap < delta never trigger the from-above rule
         (holds under the unrestricted class).
"""
from __future__ import annotations

import itertools as it

import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(3)
out: list[str] = []
Nag, w = 4, 0.25
c = np.array([0.0, 0.6, 1.3, 1.95])
delta = 0.5


def field(t, x, peers=True):
    v = -(x - c)
    if peers:
        v = v - 2 * w * np.array([np.sum(x[i] - x) for i in range(Nag)])
    return v


# (F3,D) fiber test for the centroid retention of block {0,1}
res = 0.0
for _ in range(200):
    xx = rng.normal(size=Nag) * 2
    e = rng.normal()
    xp = xx.copy()
    xp[0] += e
    xp[1] -= e
    def proj(v):
        return np.array([(v[0] + v[1]) / 2, v[2], v[3]])
    res = max(res, np.max(np.abs(proj(field(0, xx)) - proj(field(0, xp)))))
out.append(f"(F3,D) centroid retention of {{0,1}}: max |DR B(x) - DR B(x')| over 200 fiber pairs: {res:.2e}")
out.append("(F3,B) averaging protocol: law N((x0+x1)/2, 1/2), a function of the retained centroid (direct)")
x_a = np.array([0.2, 0.8, 1.3, 1.95])
x_b = np.array([0.4, 0.6, 1.3, 1.95])
out.append(f"(F4) fiber points {x_a[:2]} and {x_b[:2]} share centroid {np.mean(x_a[:2]):.2f}; child-0 probe laws N({x_a[0]},1) vs N({x_b[0]},1) differ")


def run(x0, peers, T=12.0):
    sol = solve_ivp(field, (0, T), x0, args=(peers,), rtol=1e-11, atol=1e-12, dense_output=True, max_step=0.01)
    return sol


ts = np.linspace(0, 12, 2401)


def membership(sol):
    X = sol.sol(ts)
    blocks = [(i,) for i in range(Nag)]
    events = []
    # pairwise candidate joins evaluated on current blocks, with persistence to the end of the run
    while True:
        cand = []
        for p, q in it.combinations(range(len(blocks)), 2):
            cp = X[list(blocks[p])].mean(axis=0)
            cq = X[list(blocks[q])].mean(axis=0)
            d = np.abs(cp - cq)
            start = 0 if not events else events[-1][1]
            idx0 = np.searchsorted(ts, start)
            dd = d[idx0:]
            if dd[0] <= delta or dd[-1] >= delta:
                continue
            last_above = np.where(dd >= delta)[0][-1]
            cand.append((ts[idx0 + last_above + 1], p, q))
        if not cand:
            break
        tj, p, q = min(cand)
        new = tuple(sorted(blocks[p] + blocks[q]))
        blocks = [b for k, b in enumerate(blocks) if k not in (p, q)] + [new]
        events.append((new, float(tj)))
    return [(b, round(t, 3)) for b, t in events]


x0 = c + np.array([0.03, -0.03, 0.03, -0.03])
out.append(f"(F5) full flow events: {membership(run(x0, True))}")
out.append(f"(F5) null flow events: {membership(run(x0, False))}")
diff = 0
trials = 15
for _ in range(trials):
    xr = x0 + rng.uniform(-0.01, 0.01, size=Nag)
    diff += int([e[0] for e in membership(run(xr, True))] != [e[0] for e in membership(run(xr, False))])
out.append(f"(F5) partition sequences differ in {diff}/{trials} random perturbations of size 0.01")

# (F6a/F6b) inherited dynamics of the two parent centroids after the first stage
# d/dt (xbar_23 - xbar_01) = -(1 + 2 w N) gap + (cbar_23 - cbar_01)
k = 1 + 2 * w * Nag
target = (c[2:].mean() - c[:2].mean()) / k
out.append(f"(F6a) inherited parent gap relaxes at rate {k} to {target:.4f} < delta={delta}: union occurs from the actual handoff")
gaps = np.linspace(-3, 3, 601)
unite_restricted = all(abs(np.exp(-k * 50) * g + (1 - np.exp(-k * 50)) * target) < delta for g in gaps if abs(g) > delta)
no_event_degenerate = [g for g in gaps if abs(g) < delta and g * target >= 0]
out.append(f"(F6b) every counterfactual handoff with |gap| > delta unites: {unite_restricted}  -> (F6b) fails for the class {{|gap| > delta}}")
out.append(f"(F6b) {len(no_event_degenerate)} grid handoffs with |gap| < delta (same sign as target) never reach delta from above -> (F6b) holds if such handoffs are in the class")

text = "\n".join(out)
print(text)
with open(__file__.replace(".py", ".out"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
