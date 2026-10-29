r"""D27-ITEM7: a strictly convex Gaussian hierarchy with a unique global attractor that meets (F6b)
under the finite-dwell selector of it7:selector (counterexample to the paragraph after it7:attractor).

Two first-stage blocks I = {a1, a2}, J = {b1, b2} of members with records, declared parents A_I, A_J added at
the handoff, initialized at the pools of their members, with a common parent anchor a (a declared parent prior)
and cross-scale top-down terms. Global quadratic energy (tangent Gaussian type):
  E = sum_B sum_{i in B} [ r/2 (y_i - o_i)^2 + tau/2 (y_i - m_B)^2 ] + sum_B p/2 (m_B - a)^2 .
Constant Fisher metric: members N(y_i, 1/P_y), parents N(m_B, 1/P_m). Gradient flow x' = -G^{-1} grad E.
Selector of it7:selector on presented laws N(., 1/6): coherence KL < 0.05, walls KL > 0.5, window tau_p = 2.
Before the handoff the units are the members; at the handoff each block is internally coherent and the two
blocks are walled, so a first stage selecting I and J separately is consistent with it7:selector.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

r, tau, p = 1.0, 1.0, 8.0
P_y, P_m = 20.0, 6.0
lam, dc, dw, tp = 6.0, 0.05, 0.5, 2.0
o = np.array([-0.775, -0.725, 0.725, 0.775])          # records: block means -0.75 and 0.75 (gap 1.5)
a = 0.0                                                # common parent anchor
blocks = [(0, 1), (2, 3)]

# variables x = (y_a1, y_a2, y_b1, y_b2, m_I, m_J)
H = np.zeros((6, 6)); b = np.zeros(6)
for k, B in enumerate(blocks):
    for i in B:
        H[i, i] += r + tau; H[4 + k, 4 + k] += tau; H[i, 4 + k] -= tau; H[4 + k, i] -= tau; b[i] += r * o[i]
    H[4 + k, 4 + k] += p; b[4 + k] += p * a
xstar = np.linalg.solve(H, b)
G = np.diag([P_y] * 4 + [P_m] * 2)
A = -np.linalg.solve(G, H); c = np.linalg.solve(G, b)
ds = xstar[5] - xstar[4]
print("Hessian eigenvalues:", np.round(np.linalg.eigvalsh(H), 4))
print("minimizer:", np.round(xstar, 4), "; parent gap d* =", round(ds, 4), "; KL(d*) =", round(lam / 2 * ds ** 2, 4),
      "(coherence needs < 0.05)")
print("generator eigenvalues:", np.round(np.sort(np.linalg.eigvals(A).real), 4))


def run(y0, T=400.0, dt=0.005):
    x = np.concatenate([y0, [np.mean(y0[:2]), np.mean(y0[2:])]])     # parents start at the pools
    t = 0.0; onset = None; sel = None; dmin = np.inf; dwell = 0.0; cur = 0.0
    while t < T - 1e-12:
        f = lambda z: A @ z + c
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        d = x[5] - x[4]; dmin = min(dmin, abs(d))
        if lam / 2 * d * d < dc:                     # two units only: walls vacuous
            cur += dt; dwell = max(dwell, cur)
            if onset is None: onset = t
            if sel is None and t - onset >= tp - 1e-12: sel = round(t, 3)
        else:
            onset = None; cur = 0.0
    return sel, dmin, dwell, x


for name, y0 in (("actual handoff (block gap 0.5)", np.array([-0.275, -0.225, 0.225, 0.275])),
                 ("counterfactual handoff (block gap 0.7)", np.array([-0.375, -0.325, 0.325, 0.375]))):
    gap0 = y0[2:].mean() - y0[:2].mean(); inner = max(abs(y0[1] - y0[0]), abs(y0[3] - y0[2]))
    sel, dmin, dwell, xf = run(y0)
    print(f"{name}: block gap {gap0:.2f} (KL {lam / 2 * gap0 ** 2:.3f}, walled {lam / 2 * gap0 ** 2 > dw}); "
          f"internal member gap {inner:.2f} (KL {lam / 2 * inner ** 2:.4f}, coherent {lam / 2 * inner ** 2 < dc}); "
          f"min |parent gap| {dmin:.4f}; longest coherent dwell {dwell:.2f}; union selected at {sel}; "
          f"|x(400) - x*| = {np.abs(xf - xstar).max():.1e}")
