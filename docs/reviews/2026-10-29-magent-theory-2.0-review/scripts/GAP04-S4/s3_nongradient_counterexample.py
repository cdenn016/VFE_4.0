r"""GAP04-S4 s3: mechanism of the non-gradient counterexample (report, finding GAP04-S4-02).

On L^infty_0([0,1], Leb) put e_s = 1_[0,s] - s, g0 = 1_[0,1/2] - 1_(1/2,1],
c(f) = max(0, 1/8 - dist_inf(f, {e_s : s in E})), V(f) = c(f) g0.
V is globally 1-Lipschitz and bounded in the sup norm, so it has a global unique flow.
The report proves Phi_t(e_s) = e_s + a(t) 1_E(s) g0 with a(t) = (1 - e^{-t})/8, hence
<Phi_t(e_s), g0> = <e_s, g0> + a(t) 1_E(s): for a non-Borel E the time-t map is not Borel for
the evaluation sigma-algebra.  A non-Borel set cannot be computed; here E is a finite set of
grid values and the script checks the separation ||e_s - e_s'||_inf >= 1/2, the orbit formula,
the stationarity off E, the sampled sup-norm Lipschitz constant of V, and the resulting
L^1-discontinuity of s -> Phi_t(e_s) at the points of E (s -> e_s itself is L^1-continuous).
"""
import numpy as np

G = 2000
x = (np.arange(G) + 0.5) / G                      # cell midpoints; Lebesgue = uniform weights 1/G
w = 1.0 / G
g0 = np.where(x <= 0.5, 1.0, -1.0)
S_grid = np.arange(1, G) / G                      # admissible s values (cell boundaries)


def e(s):
    return (x <= s).astype(float) - s


E = [0.25, 0.40, 0.731]
SE = np.stack([e(s) for s in E])


def c(f):
    d = np.abs(SE - f[None, :]).max(axis=1).min()
    return max(0.0, 0.125 - d)


def V(f):
    return c(f) * g0


def flow(f, T=2.0, dt=0.01):
    for _ in range(int(round(T / dt))):
        k1 = V(f); k2 = V(f + 0.5 * dt * k1); k3 = V(f + 0.5 * dt * k2); k4 = V(f + dt * k3)
        f = f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return f


rng = np.random.default_rng(3)
sep = min(np.abs(e(s1) - e(s2)).max() for s1, s2 in rng.choice(S_grid, size=(2000, 2)) if s1 != s2)
print(f"min sampled sup-distance between distinct e_s: {sep:.4f} (claim >= 1/2)")
lip = 0.0
for _ in range(2000):
    p1 = rng.uniform(-1, 1, size=G)
    f1 = e(rng.choice(E)) + rng.uniform(0, 0.15) * p1 / np.abs(p1).max()
    p2 = rng.uniform(-1, 1, size=G)
    f2 = f1 + rng.uniform(1e-4, 0.05) * p2 / np.abs(p2).max()
    lip = max(lip, np.abs(V(f1) - V(f2)).max() / np.abs(f1 - f2).max())
print(f"max sampled sup-norm Lipschitz quotient of V: {lip:.4f} (claim <= 1)")
T = 2.0
aT = (1 - np.exp(-T)) / 8
for s in E + [0.3, 0.5, 0.9]:
    fT = flow(e(s), T)
    jump = w * ((fT - e(s)) * g0).sum()
    print(f"s={s:5.3f} in E: {s in E!s:5}  <Phi_T(e_s)-e_s, g0> = {jump:.12f}   predicted a(T)1_E(s) = {aT * (s in E):.12f}")
# L^1 discontinuity in s of the composite at a point of E
s0 = E[1]
for h in (1e-2, 2e-3, 5e-4):
    s1 = round((s0 + h) * G) / G
    d_in = w * np.abs(e(s0) - e(s1)).sum()
    d_out = w * np.abs(flow(e(s0), T) - flow(e(s1), T)).sum()
    print(f"|s-s'|={h:.0e}: ||e_s - e_s'||_1 = {d_in:.5f},  ||Phi_T(e_s) - Phi_T(e_s')||_1 = {d_out:.5f}")
