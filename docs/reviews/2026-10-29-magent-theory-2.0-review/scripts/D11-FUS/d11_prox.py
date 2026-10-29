r"""D11-FUS proximal (implicit Euler) integration of fus:inclusion for OFF-family data.

Step (fus:prox-step): mu^{k+1} = argmin_x F(x) + (1/2h)||x - mu^k||_W^2, with
F = sum_i n_i U(mu_i - x_i) + lam sum_{i<j} n_i n_j (|dq| + |ds|)   (x_i = 0 in the chapter).
Solved exactly by block coordinate descent over the sectors; each sector subproblem
    min_q sum_i (n_i beta/2)(q_i - y_i)^2 + lam sum_{i<j} n_i n_j |q_i - q_j|,  beta = a + 1/h,
preserves the order of y (weights proportional to n_i), so on the monotone cone the penalty is
linear and the problem is weighted isotonic regression (PAVA) with shifted targets.
PAVA pools produce exact floating ties, so coincidence events are exact in the discrete scheme.
Single-threaded; each part a few seconds.
"""
import sys
import numpy as np


def pava(y, w):
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi*wi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2]/wts[-2] >= vals[-1]/wts[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            vals[-1] += v2; wts[-1] += w2; cnt[-1] += c2
    out = []
    for v_, w_, c_ in zip(vals, wts, cnt):
        out += [v_/w_]*c_
    return np.array(out)


def sector_prox(y, n, beta, lam):
    order = np.argsort(y, kind='mergesort')
    ys, ns = y[order], n[order]
    L = np.concatenate([[0.0], np.cumsum(ns)[:-1]])
    R = ns.sum() - L - ns
    yp = ys - lam*(L - R)/beta
    qs = pava(yp, ns*beta)
    out = np.empty_like(y)
    out[order] = qs
    return out


def prox_step(mu, n, a, c, lam, h, xref=None):
    if xref is None:
        xref = np.zeros_like(mu)
    beta = a + 1.0/h
    q, s = mu[:, 0].copy(), mu[:, 1].copy()
    for _ in range(60):
        q_old, s_old = q.copy(), s.copy()
        yq = (a*xref[:, 0] + c*(s - xref[:, 1]) + mu[:, 0]/h)/beta
        q = sector_prox(yq, n, beta, lam)
        ys = (a*xref[:, 1] + c*(q - xref[:, 0]) + mu[:, 1]/h)/beta
        s = sector_prox(ys, n, beta, lam)
        if max(np.abs(q - q_old).max(), np.abs(s - s_old).max()) < 1e-15:
            break
    return np.stack([q, s], 1)


def partition(mu, tol=1e-12):
    r = len(mu); lab = list(range(r))
    for i in range(r):
        for j in range(i):
            if np.abs(mu[i] - mu[j]).max() <= tol*(1 + np.abs(mu[i]).max()):
                lab[i] = lab[j]; break
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(i)
    return tuple(sorted(tuple(g) for g in groups.values()))


def run(mu0, n, a, c, lam, h, T, xref=None, record=False):
    mu = mu0.copy(); t = 0.0; hist = []; traj = [mu.copy()] if record else None
    last = partition(mu)
    while t < T - 1e-15:
        mu = prox_step(mu, n, a, c, lam, h, xref); t += h
        p = partition(mu)
        if p != last:
            hist.append((t, p)); last = p
        if record:
            traj.append(mu.copy())
    return hist, mu, traj


part = sys.argv[1] if len(sys.argv) > 1 else "all"
rng = np.random.default_rng(11)

if part in ("sync", "all"):
    # ------------------------------------------------ fus:sync bound and the sharper bound off the family
    a, c = 1.0, 0.6; kap = a - c
    print("SYNC  off-family random data, (a,c)=(1,0.6), h=1e-3")
    worst = []
    for trial in range(8):
        r = int(rng.integers(3, 6))
        n = rng.integers(1, 4, size=r).astype(float); N = n.sum()
        lam = float(rng.uniform(0.2, 1.0))
        mu0 = rng.normal(0, 1.5, size=(r, 2)) + rng.normal(0, 3, size=2)   # offset center
        R0 = np.sqrt((n[:, None]*mu0**2).sum())
        t0 = np.log(max(1.0, 4*c*R0/(lam*N)))/kap
        Bms = t0 + 4*R0*np.exp(-kap*t0)/(lam*N)
        M0 = max(np.ptp(mu0[:, 0]), np.ptp(mu0[:, 1]))
        Bsh = np.log1p(kap*M0/(lam*N))/kap
        h = 1e-3
        hist, mu, _ = run(mu0, n, a, c, lam, h, Bsh + 0.05)
        Ts = hist[-1][0] if hist and len(hist[-1][1]) == 1 else np.inf
        print(f"  r={r} N={N:.0f} lam={lam:.3f} R0={R0:.2f}: T_sync={Ts:.4f}  B_sharp={Bsh:.4f}  B_manuscript={Bms:.4f}"
              f"  events={len(hist)}  sequence={[p for _, p in hist]}")
        worst.append(Ts/Bsh)
    print(f"  max T_sync/B_sharp over trials = {max(worst):.4f} (discretization error O(h))")

if part in ("open", "all"):
    # ------------------------------------------------ fus:open-two-stage: perturbations of an admitted datum
    a, c, lam, h = 1.0, 0.5, 0.3, 1e-3
    u = np.array([-3.0, -2.0, 2.0, 3.0]); v0 = 0.4
    xbar = np.stack([u + v0, u - v0], 1)
    print("OPEN  perturbations of the four-agent datum (d=1), eps up to 0.05")
    for trial in range(6):
        pert = rng.normal(size=(4, 2)); pert *= 0.05/np.linalg.norm(pert)
        hist, mu, _ = run(xbar + pert, np.ones(4), a, c, lam, h, 2.6)
        print(f"  trial {trial}: {[(round(t,3), p) for t, p in hist]}")
    # ------------------------------------------------ block averages of separated blocks close (claim at 06:542-546)
    print("AVG  block averages of the UNSYNCHRONIZED blocks A={0,1}, B={2,3} vs the coarse two-parent flow")
    x0 = xbar + 0.05*rng.normal(size=(4, 2))
    nfine = np.ones(4)
    coarse0 = np.stack([x0[:2].mean(0), x0[2:].mean(0)])
    mu_f = x0.copy(); mu_c = coarse0.copy(); t = 0.0; maxdev = 0.0; sep_ok = True; tsep = None
    while t < 1.0:
        mu_f = prox_step(mu_f, nfine, a, c, lam, h); mu_c = prox_step(mu_c, np.array([2.0, 2.0]), a, c, lam, h); t += h
        sep = all(mu_f[:2, al].max() < mu_f[2:, al].min() for al in (0, 1))
        if not sep:
            sep_ok = False; break
        maxdev = max(maxdev, np.abs(np.stack([mu_f[:2].mean(0), mu_f[2:].mean(0)]) - mu_c).max())
    print(f"  separated on [0,{t:.3f}]: {sep_ok}; max |block average - coarse| = {maxdev:.2e}")
    print(f"  fine partition at t={t:.3f}: {partition(mu_f)} (A and B are still distinct blocks of distinct agents)")

if part in ("lamdep", "all"):
    # ------------------------------------------------ lambda-dependence of the sequence OFF the common-offset class
    a, c = 1.0, 0.5
    q0 = np.array([0.0, 2.0, 12.0]); s0 = np.array([0.0, 10.0, 11.0])
    mu0 = np.stack([q0, s0], 1)
    print("LAMDEP three agents, q=(0,2,12), s=(0,10,11); u=(q+s)/2=(0,6,11.5), u-gaps (6,5.5)")
    for lam, h, T in [(50.0, 2e-5, 0.12), (10.0, 1e-4, 0.6), (0.02, 1e-2, 40.0)]:
        hist, mu, _ = run(mu0, np.ones(3), a, c, lam, h, T)
        print(f"  lam={lam}: {[(round(t,4), p) for t, p in hist]}")
    # null flow (lam=0) with a small threshold single-linkage selector on Euclidean distances
    A = np.array([[a, -c], [-c, a]])
    w, V = np.linalg.eigh(A)
    def null_mu(t):
        return (V @ np.diag(np.exp(-w*t)) @ V.T @ mu0.T).T
    for delta in (1e-1, 1e-3):
        joined = []
        ts = np.linspace(0, 60, 60001)
        for t in ts:
            m = null_mu(t)
            for (i, j) in [(0, 1), (1, 2), (0, 2)]:
                if (i, j) not in [p for p, _ in joined] and np.linalg.norm(m[i] - m[j]) <= delta:
                    joined.append(((i, j), round(t, 3)))
        print(f"  null flow, threshold delta={delta}: first pair reaching delta: {joined[:2]}")

if part in ("hetero", "all"):
    # ------------------------------------------------ extension: heterogeneous single-well references x_i
    a, c, h = 1.0, 0.5, 2e-3
    xref = np.array([[0.0, 0.2], [0.3, 0.1], [0.5, 0.4], [4.0, 3.8], [4.4, 4.1], [9.0, 9.5]])
    mu0 = xref + rng.normal(0, 1.0, size=xref.shape)
    print("HETERO convex single-well references; stationary partition vs lambda (persistent blocks?)")
    for lam in (0.05, 0.2, 0.6, 2.0):
        hist, mu, _ = run(mu0, np.ones(6), a, c, lam, h, 12.0, xref=xref)
        mu2 = prox_step(mu, np.ones(6), a, c, lam, 1e3, xref)      # large-h step = argmin F (stationary point)
        print(f"  lam={lam}: final partition {partition(mu)} | argmin-F partition {partition(mu2, 1e-9)} | events {len(hist)}"
              f" | last event t={hist[-1][0] if hist else None}")
