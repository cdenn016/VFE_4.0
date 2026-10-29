r"""GAP01-S3 s5: sanity check of the chart-flow Lipschitz bound (report, Q4) on a discretized {-1,1} x S^1 x S^1.

Chart: f_i in L^infty_0(lambda), q_i = e^{f_i} lambda / E_lambda e^{f_i}; Gibbs-reduced per-agent first variation
   phi_i = log q_i - beta_ii log G_i - sum_{j!=i} beta_ij log nu_j - log l_i,  nu_j = (z-marginal of q_j) x Haar x Haar,
   V_i = -(phi_i - E_lambda phi_i).
Claimed bound on {max_i ||f_i||_inf <= R}: ||V(f)-V(g)||_inf <= (8 + 32 R + 32 R^2) max_i ||f_i - g_i||_inf  (R >= Gamma/2).
Also checks the a priori growth bound ||P_i||_inf <= 2 max(J, R) + 3 eps for V_i = -f_i + P_i.
The discretization (24 x 24 grid on the circles) is a finite-table instance of the same estimate; floating point.
"""
import numpy as np

rng = np.random.default_rng(11)
NZ, NP, NS = 2, 24, 24
z = np.array([-1.0, 1.0])[:, None, None] * np.ones((NZ, NP, NS))
ph = (2 * np.pi * np.arange(NP) / NP)[None, :, None] * np.ones((NZ, NP, NS))
eps = 0.05; J = eps ** 3
omega = rng.uniform(0, 2 * np.pi, 3); orec = np.array([1.0, 1.0, -1.0])
logG = J * z * np.cos(ph) - np.log(np.mean(np.exp(J * np.cos(ph[0]))))
pi = np.array([[0.2 if i == j else 0.4 for j in range(3)] for i in range(3)])

def V(F):
    logq = []; q = []
    for f in F:
        Lam = np.log(np.mean(np.exp(f)))
        logq.append(f - Lam); q.append(np.exp(f - Lam))
    lognu = []
    for j in range(3):
        mz = np.mean(q[j], axis=(1, 2))           # E_lambda[e^{f}|Z=z]/E_lambda e^f  (density of z-marginal x Haar^2)
        lognu.append(np.log(mz)[:, None, None] * np.ones((NZ, NP, NS)))
    out = []; Ps = []
    for i in range(3):
        logl = eps * orec[i] * z / 2 + eps * np.cos(omega[i] - ph)
        D = np.empty(3)
        for k in range(3):
            T = logG if k == i else lognu[k]
            D[k] = np.mean(q[i] * (logq[i] - T))
        w = pi[i] * np.exp(-(D - D.min())); b = w / w.sum()
        stuff = b[i] * logG + sum(b[k] * lognu[k] for k in range(3) if k != i) + logl
        P = stuff - stuff.mean()
        phi = logq[i] - stuff
        out.append(-(phi - phi.mean())); Ps.append(P)
    return out, Ps

def rand_f(R):
    f = rng.normal(size=(NZ, NP, NS)); f -= f.mean(); f *= R * rng.uniform(0.3, 1.0) / np.abs(f).max()
    return f

for R in (0.1, 0.5, 1.0, 2.0):
    worst = 0.0; worst_P = 0.0
    for trial in range(60):
        F = [rand_f(R) for _ in range(3)]
        G = [f + 1e-3 * R * rng.normal(size=f.shape) for f in F]
        G = [g - g.mean() for g in G]
        G = [g * min(1.0, R / np.abs(g).max()) for g in G]
        VF, PF = V(F); VG, _ = V(G)
        num = max(np.abs(a - b).max() for a, b in zip(VF, VG))
        den = max(np.abs(a - b).max() for a, b in zip(F, G))
        worst = max(worst, num / den)
        Rnow = max(np.abs(f).max() for f in F)
        worst_P = max(worst_P, max(np.abs(P).max() for P in PF) / (2 * max(J, Rnow) + 3 * eps))
    print("R=%.1f: sampled max Lipschitz ratio %.3f  vs bound 8+32R+32R^2 = %.1f ; max ||P||/(2max(J,R)+3eps) = %.3f"
          % (R, worst, 8 + 32 * R + 32 * R ** 2, worst_P))
