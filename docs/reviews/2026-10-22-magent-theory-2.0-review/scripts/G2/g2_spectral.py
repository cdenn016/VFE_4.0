r"""G2 checks for the conditional-spectrum flow of chapter 7 (smooth:spectrum-flow).

Part A: forward invariance of the essential range of \ell_z under
        \ell_t = a \ell_0 - \log Z(a),  a = e^{-t},  Z(a) = \int e^{a r} \xi_0(dr),
        i.e. supp \xi_t \subset [min supp \xi_0, max supp \xi_0] and width_t = a width_0.
Part B: uniqueness reduction of the nonlinear weak equation to a scalar ODE
        c' = -c + e^c (a Z'(a) + c Z(a)),  c(0) = 0, whose solution must be -log Z(a);
        weak form checked for two test functions by finite differences.
Part C: |dH/dt| = Var_{e^r \xi_t}(r) <= (a width_0)^2 / 4.
Part D: per-z bounded but non-uniform private log densities: H(z,t) unbounded in z for
        t < log 2, and the public log density of a symmetric two-agent identity-channel
        system becomes unbounded below for t > 0.
Single-threaded, numpy/scipy only.
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(20261022)


def normalize_atoms(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    r"""Shift atoms so that \sum_k w_k e^{r_k} = 1 (normalization of a spectrum)."""
    return r - np.log(np.sum(w * np.exp(r)))


def evolve_atoms(r0: np.ndarray, w: np.ndarray, t: float) -> np.ndarray:
    a = np.exp(-t)
    logZ = np.log(np.sum(w * np.exp(a * r0)))
    return a * r0 - logZ


def H_of(r: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * r * np.exp(r)))


print("Part A: range invariance")
worst_lo, worst_hi, worst_width = 0.0, 0.0, 0.0
for trial in range(3000):
    K = rng.integers(2, 8)
    w = rng.dirichlet(np.ones(K))
    r0 = normalize_atoms(rng.normal(0, rng.uniform(0.1, 6.0), K), w)
    m0, M0 = r0.min(), r0.max()
    for t in np.concatenate([np.linspace(0.001, 0.2, 20), np.linspace(0.2, 12, 40)]):
        rt = evolve_atoms(r0, w, t)
        worst_lo = max(worst_lo, m0 - rt.min())          # should be <= 0
        worst_hi = max(worst_hi, rt.max() - M0)          # should be <= 0
        worst_width = max(worst_width, abs((rt.max() - rt.min()) - np.exp(-t) * (M0 - m0)))
print(f"  max(m0 - min supp xi_t) = {worst_lo:.3e} (<= 0 means lower end never below m0)")
print(f"  max(max supp xi_t - M0) = {worst_hi:.3e} (<= 0 means upper end never above M0)")
print(f"  max |width_t - a width_0| = {worst_width:.3e}")

# continuous-reference example: V = [0,1] grid, ell_0 = c cos(2 pi v) normalized
v = (np.arange(4000) + 0.5) / 4000
for c in [0.5, 3.0, 8.0]:
    l0 = c * np.cos(2 * np.pi * v) + 2.0 * (v > 0.9)
    l0 = l0 - np.log(np.mean(np.exp(l0)))
    w = np.full(v.size, 1.0 / v.size)
    m0, M0 = l0.min(), l0.max()
    ok = True
    for t in np.linspace(0.0, 6.0, 61):
        lt = evolve_atoms(l0, w, t)
        ok &= (lt.min() >= m0 - 1e-12) and (lt.max() <= M0 + 1e-12)
    print(f"  grid example c={c}: range [{m0:.3f},{M0:.3f}] invariant on t in [0,6]: {ok}")

print("Part B: scalar ODE reduction of the weak equation")
for trial in range(5):
    K = 5
    w = rng.dirichlet(np.ones(K))
    r0 = normalize_atoms(rng.normal(0, 2.5, K), w)

    def rhs(t, y):
        a = np.exp(-t)
        Z = np.sum(w * np.exp(a * r0))
        Zp = np.sum(w * r0 * np.exp(a * r0))
        c = y[0]
        return [-c + np.exp(c) * (a * Zp + c * Z)]

    T = 5.0
    sol = solve_ivp(rhs, (0, T), [0.0], rtol=1e-12, atol=1e-14, dense_output=True)
    ts = np.linspace(0, T, 51)
    err = max(abs(sol.sol(t)[0] + np.log(np.sum(w * np.exp(np.exp(-t) * r0)))) for t in ts)
    print(f"  trial {trial}: max |c(t) + log Z(e^-t)| on [0,5] = {err:.2e}")

# weak form d/dt int f dxi_t = int f'(r)(H - r) dxi_t
K = 6
w = rng.dirichlet(np.ones(K))
r0 = normalize_atoms(rng.normal(0, 2.0, K), w)
for name, f, fp in [("sin(3r)", lambda r: np.sin(3 * r), lambda r: 3 * np.cos(3 * r)),
                    ("r^3", lambda r: r ** 3, lambda r: 3 * r ** 2)]:
    worst = 0.0
    for t in np.linspace(0.05, 4, 40):
        h = 1e-5
        lhs = (np.sum(w * f(evolve_atoms(r0, w, t + h))) - np.sum(w * f(evolve_atoms(r0, w, t - h)))) / (2 * h)
        rt = evolve_atoms(r0, w, t)
        Ht = H_of(rt, w)
        rhs_val = np.sum(w * fp(rt) * (Ht - rt))
        worst = max(worst, abs(lhs - rhs_val))
    print(f"  weak form residual for f={name}: {worst:.2e}")

print("Part C: |dH/dt| = Var_{e^r xi}(r) <= (a width0)^2/4")
worst_ratio = 0.0
for trial in range(2000):
    K = rng.integers(2, 8)
    w = rng.dirichlet(np.ones(K))
    r0 = normalize_atoms(rng.normal(0, rng.uniform(0.1, 6.0), K), w)
    width0 = r0.max() - r0.min()
    for t in np.linspace(0.0, 8, 33):
        rt = evolve_atoms(r0, w, t)
        p = w * np.exp(rt)
        var = np.sum(p * rt ** 2) - np.sum(p * rt) ** 2
        worst_ratio = max(worst_ratio, var / ((np.exp(-t) * width0) ** 2 / 4))
print(f"  max Var / ((a width0)^2/4) = {worst_ratio:.6f} (<= 1 expected)")

print("Part D: per-z bounded, non-uniform private log density (p(z) = z/2, eps = p)")


def spectrum_D(z: float):
    p = z / 2.0
    alpha = np.log((1 - p) / p)
    r0 = np.array([alpha, -alpha])
    w = np.array([p, 1 - p])
    return r0, w


def H_zt(z: float, t: float) -> float:
    r0, w = spectrum_D(z)
    rt = evolve_atoms(r0, w, t)
    return H_of(rt, w)


zs = 10.0 ** -np.arange(1, 16)
print("  z        sup|ell_z^0|   H(z,0)     H(z,0.2)   H(z,0.5)   log(2/z) [asymptote for t < log 2]")
for z in zs[::2]:
    r0, w = spectrum_D(z)
    print(f"  {z:8.1e} {np.max(np.abs(r0)):10.3f} {H_zt(z,0):10.3f} {H_zt(z,0.2):10.3f} {H_zt(z,0.5):10.3f} {np.log(2/z):10.3f}")

# finite energy: int_0^1 H(z,0) dz
zg = (np.arange(200000) + 0.5) / 200000
pg = zg / 2
Hg = (1 - 2 * pg) * np.log((1 - pg) / pg)
print(f"  int_0^1 H(z,0) dz = {np.mean(Hg):.4f} (finite)")

# public log density in the symmetric identity-channel system: m(z,t) = -int_0^t e^{-(t-s)} H(z,s) ds + c(t)
def m_public(z: float, t: float, n: int = 4000) -> float:
    s = np.linspace(0, t, n + 1)
    vals = np.array([np.exp(-(t - si)) * H_zt(z, si) for si in s])
    return -np.trapezoid(vals, s)


for t in [0.1, 0.3, 0.6]:
    ms = np.array([m_public(z, t, 800) for z in zs])
    slope = (ms[-1] - ms[-5]) / (np.log(2 / zs[-1]) - np.log(2 / zs[-5]))
    K_pred = 1 - np.exp(-t)
    print(f"  t={t}: m(z,t)-m(0.1,t) for z=1e-5,1e-10,1e-15: "
          f"{ms[4]-ms[0]:.3f}, {ms[9]-ms[0]:.3f}, {ms[14]-ms[0]:.3f};"
          f" slope dm/dlog(2/z) = {slope:.4f}, predicted -(1-e^-t) as z->0 = {-K_pred:.4f}")

print("Part E: sup_z H(z,0) = infinity but uniformly bounded time integral (mass 1/2 on a set of A-measure p)")


def H_half(p: float, t: float) -> float:
    r0 = np.array([np.log(0.5 / p), np.log(0.5 / (1 - p))])
    w = np.array([p, 1 - p])
    return H_of(evolve_atoms(r0, w, t), w)


for t in [0.001, 0.01, 0.1]:
    row = []
    for k in [5, 20, 80, 320, 640]:
        p = np.exp(-k)
        s = np.concatenate([np.linspace(0, min(t, 20.0 / k), 4001), np.linspace(min(t, 20.0 / k), t, 4001)[1:]])
        vals = np.array([H_half(p, si) for si in s])
        row.append(np.trapezoid(vals, s))
    print(f"  t={t}: int_0^t H ds for log(1/p) = 5,20,80,320,640: " + ", ".join(f"{x:.4f}" for x in row)
          + f"; H(.,0) = " + ", ".join(f"{H_half(np.exp(-k),0):.1f}" for k in [5, 20, 80, 320, 640]))
print("  (as log(1/p) -> infinity the time integral tends to log 2 for every fixed t > 0: bounded, but not continuous at t = 0 in L^infinity)")
