"""XC2-FRAME check of the sentence
  'Interaction-sensitive selection under the reciprocal potential therefore requires
   receiver-dependent targets' (05_inheritance:1372-1373, 11_conditional_formation after
   form:mean-field, 06_outlook:16, companion c05).
inh:common-clock / form:mean-field prove a common clock only when every channel equals one
law C[M] of the ADDITIVE moment M = sum_i E_{q_i} T.  Here every receiver uses the same
target, but it depends on the population non-additively: the normalized geometric mean of
all agent laws.  For Bernoulli laws with log odds l_i this target has log odds mean(l).

Reciprocal potential (belief sector; model sector omitted, it decouples), source sets
exclude self, so all costs in a row are equal and the Gibbs rows equal the priors:
  E = sum_i [ KL(q_i || G0) + KL(q_i || C) ],  C = Bern(log odds = mean(l)),  G0 uniform.
Unit Fisher flow in log odds:  dl_i/dt = -(1/sigma'(l_i)) dE/dl_i
  dE/dl_i = sigma'(l_i) l_i + sigma'(l_i)(l_i - lbar) + (1/N) sum_j (sigma(lbar) - sigma(l_j)).
Null flow (inh:null-flow, peer costs deleted): dl_i/dt = -l_i.
Selector (the class named in inh:common-clock: reads only pairwise log-ratio differences
l_i - l_j): join two current blocks when their count-weighted centroid distance in l first
reaches delta from above; recompute after each join.
We (1) verify by finite differences that the analytic field is the natural gradient,
(2) show d/dt (l_i - l_j) is not -2 (l_i - l_j), and (3) search for an initial datum whose
partition sequence differs between the full and null flows.
"""
import numpy as np

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsig(x):
    s = sig(x); return s * (1 - s)

def kl_bern(l, m):
    p, r = sig(l), sig(m)
    return p * (np.log(p) - np.log(r)) + (1 - p) * (np.log(1 - p) - np.log(1 - r))

def energy(l):
    lbar = l.mean()
    return np.sum(kl_bern(l, 0.0) + kl_bern(l, lbar))

def full_field(l):
    N = len(l); lbar = l.mean()
    S = np.sum(sig(lbar) - sig(l))
    grad = dsig(l) * l + dsig(l) * (l - lbar) + S / N
    return -grad / dsig(l)

def null_field(l):
    return -l

def rk4(field, l0, T, dt):
    n = int(round(T / dt)); traj = np.empty((n + 1, len(l0))); traj[0] = l0; l = l0.copy()
    for k in range(n):
        k1 = field(l); k2 = field(l + 0.5 * dt * k1); k3 = field(l + 0.5 * dt * k2); k4 = field(l + dt * k3)
        l = l + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6; traj[k + 1] = l
    return traj

def selector_history(traj, delta, dt):
    """Blocks join when centroid distance first reaches delta from above."""
    N = traj.shape[1]
    blocks = [[i] for i in range(N)]
    events = []
    def cdist(state, I, J):
        return abs(state[I].mean() - state[J].mean())
    prev = traj[0]
    for k in range(1, traj.shape[0]):
        cur = traj[k]
        changed = True
        while changed:
            changed = False
            for a in range(len(blocks)):
                for b in range(a + 1, len(blocks)):
                    I, J = blocks[a], blocks[b]
                    if cdist(prev, I, J) > delta and cdist(cur, I, J) <= delta:
                        events.append((round(k * dt, 3), tuple(sorted(I + J))))
                        blocks = [x for x in blocks if x is not I and x is not J] + [sorted(I + J)]
                        changed = True
                        break
                if changed:
                    break
        prev = cur
    return events

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # (1) natural-gradient check
    l = np.array([3.6, 5.0, 6.5]); h = 1e-6
    g_fd = np.array([(energy(l + h * e) - energy(l - h * e)) / (2 * h) for e in np.eye(3)])
    print("(1) max |analytic - finite-difference| of -sigma' * field vs dE/dl:",
          f"{np.max(np.abs(-dsig(l) * full_field(l) - g_fd)):.2e}")
    # (2) common clock test
    v = full_field(l)
    for (i, j) in [(0, 1), (1, 2), (0, 2)]:
        print(f"(2) d/dt(l_{i+1}-l_{j+1}) = {v[i]-v[j]:+.4f}  vs  -2(l_{i+1}-l_{j+1}) = {-2*(l[i]-l[j]):+.4f}")
    # (3) search
    T, dt = 6.0, 0.002
    found = 0
    for trial in range(400):
        N = 3
        base = rng.uniform(2.0, 7.0)
        gaps = rng.uniform(0.6, 2.0, size=N - 1)
        l0 = base + np.concatenate([[0.0], np.cumsum(gaps)])
        d0 = min(gaps)
        for delta in (0.15, 0.3, 0.5):
            if delta >= 0.9 * d0:
                continue
            ef = selector_history(rk4(full_field, l0, T, dt), delta, dt)
            en = selector_history(rk4(null_field, l0, T, dt), delta, dt)
            Pf = [e[1] for e in ef]; Pn = [e[1] for e in en]
            if Pf and Pn and Pf[0] != Pn[0]:
                found += 1
                if found <= 3:
                    print(f"(3) l0 = {np.round(l0, 4)}, delta = {delta}:")
                    print(f"     full events {ef}")
                    print(f"     null events {en}")
        if found >= 3:
            break
    print(f"(3) initial data with a different first join found: {found} (of the trials scanned)")
