r"""X4-PRIOR checks on the 2.0 selectors and contextual geometry (single-threaded, short).

(a) Chapter 11 threshold selector (form:selector): every block score obeys b_I(t)=e^{-2t}b_I(0)
    and unions take count-weighted means. Claim checked: the event sequence equals count-weighted
    centroid-linkage (UPGMC) agglomeration of the initial scalar scores, with event times
    t_k = (1/2) log(h_k/delta), h_k the k-th merge height; in 1D the merge heights never invert.
(b) Chapter 12 collective geometry: a_t = I(rho k), b_t = I(rho l), I(x)=x I1(x)/I0(x),
    rho = e^{-2t}. A^F = -a/(a+b) dchi and F = -b rho I'(rho k)/(a+b)^2 dk^dchi (open:curvature).
    Limits as t -> infinity: A^F coefficient -> -k^2/(k^2+l^2), F -> -2 k l^2/(k^2+l^2)^2 (nonzero),
    while h_chichi and h_kk decay like e^{-4t}.
"""
import numpy as np
from scipy.special import i0e, i1e

rng = np.random.default_rng(1)

# (a)
def simulate_threshold(b0, delta):
    blocks = [([i], b0[i]) for i in range(len(b0))]
    events = []
    t = 0.0
    while len(blocks) > 1:
        best = None
        for x in range(len(blocks)):
            for y in range(x + 1, len(blocks)):
                d = abs(blocks[x][1] - blocks[y][1])
                if best is None or d < best[0]:
                    best = (d, x, y)
        d, x, y = best
        tj = 0.5 * np.log(d / delta)
        if tj < t - 1e-12:
            return events, "INVERSION"
        t = tj
        I, J = blocks[x], blocks[y]
        nI, nJ = len(I[0]), len(J[0])
        new = (sorted(I[0] + J[0]), (nI * I[1] + nJ * J[1]) / (nI + nJ))
        blocks = [blk for k, blk in enumerate(blocks) if k not in (x, y)] + [new]
        events.append((round(t, 10), tuple(new[0]), d))
    return events, "ok"


def upgmc(b0):
    clusters = [([i], b0[i]) for i in range(len(b0))]
    heights = []
    while len(clusters) > 1:
        pairs = [(abs(clusters[x][1] - clusters[y][1]), x, y) for x in range(len(clusters)) for y in range(x + 1, len(clusters))]
        h, x, y = min(pairs)
        I, J = clusters[x], clusters[y]
        n1, n2 = len(I[0]), len(J[0])
        new = (sorted(I[0] + J[0]), (n1 * I[1] + n2 * J[1]) / (n1 + n2))
        clusters = [c for k, c in enumerate(clusters) if k not in (x, y)] + [new]
        heights.append((tuple(new[0]), h))
    return heights

mism, inv = 0, 0
for trial in range(2000):
    N = rng.integers(3, 8)
    b0 = rng.uniform(0, 10, size=N)
    delta = 1e-3
    if np.min(np.abs(b0[:, None] - b0[None, :]) + np.eye(N) * 1e9) <= delta:
        continue  # the theorem's hypothesis requires every initial gap above the threshold
    ev, status = simulate_threshold(b0, delta)
    if status != "ok":
        inv += 1
        continue
    up = upgmc(b0)
    for (t, blk, d), (blk2, h) in zip(ev, up):
        if blk != blk2 or abs(t - 0.5 * np.log(h / delta)) > 1e-9:
            mism += 1
            break
print("(a) 1D trials: mismatches with UPGMC =", mism, " inversions =", inv)
# the example of the manuscript: shapes (1, 6/5, 9/5, 21/10), alpha0=1 -> b0 = shapes-1, delta=1/10
b0 = np.array([1, 1.2, 1.8, 2.1]) - 1.0
ev, st = simulate_threshold(b0, 0.1)
print("(a) manuscript example events:", ev, " expected times", 0.5*np.log(2), 0.5*np.log(3), 0.5*np.log(17/2))
# 2D vector scores: an inversion example (the scalar rule would never fire again)
P = np.array([[0, 0], [1, 0], [0.5, 0.9]])
c01 = P[:2].mean(axis=0)
print("(a) 2D inversion: first merge height 1.0, new centroid distance", np.linalg.norm(c01 - P[2]))

# (b)
def Ifun(x):
    return x * i1e(x) / i0e(x)

def Iprime(x, eps=1e-6):
    return (Ifun(x + eps) - Ifun(x - eps)) / (2 * eps)

k, l = 1.3, 0.7
for t in (0.0, 1.0, 3.0, 6.0, 10.0):
    r = np.exp(-2 * t)
    a, b = Ifun(r * k), Ifun(r * l)
    Acoef = -a / (a + b)
    Fcoef = -b * r * Iprime(r * k) / (a + b)**2
    # finite-difference check of dA/dk
    dk = 1e-5
    a2 = Ifun(r * (k + dk)); a1 = Ifun(r * (k - dk))
    fd = (-(a2) / (a2 + b) + a1 / (a1 + b)) / (2 * dk)
    hchichi = a * b / (a + b)
    print("(b) t=%4.1f A=%.6f F=%.6f dA/dk(fd)=%.6f h_chichi=%.3e" % (t, Acoef, Fcoef, fd, hchichi))
print("(b) predicted limits: A ->", -k**2 / (k**2 + l**2), " F ->", -2 * k * l**2 / (k**2 + l**2)**2)
