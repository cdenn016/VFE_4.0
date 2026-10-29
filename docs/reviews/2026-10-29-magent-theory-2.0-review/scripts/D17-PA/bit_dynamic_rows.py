r"""D17-PA check 8: pa:persistent-grouping under def:dyn-per-agent-flow (rows evolve with unit Fisher mobility)
instead of the Gibbs-reduced convention used in chapter 13. Rows start at the priors and are integrated in
log-row coordinates (exact positivity) with DOP853. Null: nonself costs deleted; labels, priors and row entropy
kept; rows still dynamic. (A first version integrating the rows directly with default RK45 tolerances produced
spurious late block splits; those were integration artifacts and are not reported.)"""
import numpy as np
from scipy.integrate import solve_ivp

a, p = 0.2, 0.4
o = np.array([1.0, 1.0, -1.0])
PI = np.full((3, 3), p); np.fill_diagonal(PI, a)

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def klb(l1, l2):
    p1, p2 = sig(l1), sig(l2)
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))

def field(y, eps, null):
    l = y[:3]; LB = y[3:].reshape(3, 3)
    B = np.exp(LB - LB.max(axis=1, keepdims=True)); B /= B.sum(axis=1, keepdims=True)
    dl = np.zeros(3); dLB = np.zeros((3, 3))
    for i in range(3):
        D = np.array([klb(l[i], 0.0) if j == i else (0.0 if null else klb(l[i], l[j])) for j in range(3)])
        if null:
            dl[i] = -B[i, i] * l[i] + eps * o[i]
        else:
            dl[i] = -l[i] + sum(B[i, j] * l[j] for j in range(3) if j != i) + eps * o[i]
        hrow = D + np.log(B[i] / PI[i]) + 1.0
        dLB[i] = -(hrow - np.sum(B[i] * hrow))
    return np.concatenate([dl, dLB.ravel()])

def partition(l):
    comp = [0, 1, 2]
    for i in range(3):
        for j in range(i + 1, 3):
            if l[i] * l[j] > 0 and min(abs(l[i]), abs(l[j])) / max(abs(l[i]), abs(l[j])) > 0.25:
                ci, cj = comp[i], comp[j]
                comp = [ci if c == cj else c for c in comp]
    blocks = {}
    for k, c in enumerate(comp):
        blocks.setdefault(c, []).append(k + 1)
    return tuple(sorted(tuple(b) for b in blocks.values()))

rng = np.random.default_rng(3)
u_in = np.array([4.0, 0.5, -1.0])
for eps in (0.05, 0.3):
    nf = nn = nstart = 0
    for trial in range(12):
        l0 = eps * (u_in + 0.1 * rng.normal(size=3))
        y0 = np.concatenate([l0, np.log(PI).ravel()])
        out = {}
        for null, T in ((False, 120.0), (True, 300.0)):
            sol = solve_ivp(lambda t, y: field(y, eps, null), (0, T), y0, t_eval=np.linspace(0, T, 2401), rtol=1e-10, atol=1e-13, method="DOP853")
            seq = []
            for k in range(sol.y.shape[1]):
                P = partition(sol.y[:3, k])
                if not seq or seq[-1][1] != P:
                    seq.append((round(float(sol.t[k]), 3), P))
            out[null] = seq
        nstart += int(out[False][0][1] == ((1,), (2,), (3,)))
        nf += int(out[False][0][1] == ((1,), (2,), (3,)) and out[False][-1][1] == ((1, 2, 3),) and len(out[False]) <= 3)
        nn += int(any(P == ((1, 2, 3),) for _, P in out[True]))
        if trial == 0:
            print("eps=%.2f sample full %s | null %s" % (eps, out[False], out[True]))
    print("eps=%.2f dynamic rows: start singletons %d/12; singletons -> pair -> full block, persisting to t=120: %d/12; null ever full block %d/12" % (eps, nstart, nf, nn))
