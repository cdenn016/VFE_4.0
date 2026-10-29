r"""D28-ITEMS independent checks of Item 2 (link dynamics) claims.

Single-threaded, tiny sizes; sympy for exact algebra. Agreement supports the statements at the tested
instances only.

Blocks
  A  it2:split: the per-agent covariance S' equals the global S identically (symbolic, scalar K=1),
     so the receiver's single-record shift is the same in both pictures.
  B  Item 2 remark after it2:potential ("With Gibbs rows in place of fixed rows, the Gibbs envelope
     identity gives the same derivatives ..."): compare d/dx of (i) the entropy-free weighted sum with
     Gibbs rows substituted, (ii) the entropy-regularized reduced potential -T log Z, and (iii) the
     fixed-row derivative evaluated at the Gibbs rows.
  C  it2:echo: exact algebra of (i), (iii), (iv) and uniqueness of the law-level fixed point among
     ALL (not only symmetric) fixed points, by the monotone-map argument checked numerically.
  D  it2:ring: spectrum formula at c = 0 and c != 0, weak-anchor limit.
  E  it2:certainty(iii): implicit derivative dv/d(d^2) > 0 at a nondegenerate minimum (symbolic).
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(29102)

# ---------------- A ----------------
Si, Sj, C, R, Tt = sp.symbols("Sigma_i Sigma_j C_rho R T", positive=True)
S_glob = Tt * (Si + Sj + C) * Tt + R
S_pa = Tt * (Si + C) * Tt + R + Tt * Sj * Tt
print("A  S' - S =", sp.simplify(S_pa - S_glob))

# ---------------- B ----------------
T = 0.7
pi = np.array([0.5, 0.3, 0.2])
A = rng.standard_normal((3, 2))


def costs(x):
    return 0.5 * np.sum((A * 0 + x[None, :] - A) ** 2, axis=1) + 0.3 * np.arange(3)


def gibbs(c):
    w = pi * np.exp(-c / T)
    return w / w.sum()


def E_free(x):
    c = costs(x)
    return float(gibbs(c) @ c)


def E_red(x):
    c = costs(x)
    return float(-T * np.log(np.sum(pi * np.exp(-c / T))))


x0 = np.array([0.3, -0.4]); h = 1e-6
g_free = np.array([(E_free(x0 + h * e) - E_free(x0 - h * e)) / (2 * h) for e in np.eye(2)])
g_red = np.array([(E_red(x0 + h * e) - E_red(x0 - h * e)) / (2 * h) for e in np.eye(2)])
b = gibbs(costs(x0))
dC = np.array([[(costs(x0 + h * e)[k] - costs(x0 - h * e)[k]) / (2 * h) for e in np.eye(2)] for k in range(3)])
g_fixed = b @ dC
print("B  fixed-row derivative at Gibbs rows      =", g_fixed)
print("   derivative of -T log Z (with entropy)    =", g_red, " diff", np.abs(g_red - g_fixed).max())
print("   derivative of sum beta*(x) C(x) (no entropy) =", g_free, " diff", np.abs(g_free - g_fixed).max())

# ---------------- C ----------------
s2, lam, l = sp.symbols("sigma2 lambda ell", positive=True)
v = sp.symbols("v", positive=True)
exact = sp.simplify((sp.Matrix.eye(3) / s2 + lam * sp.ones(3, 3)).inv()[0, 0])
print("C  exact marginal l=3 from matrix inverse:", sp.simplify(exact - s2 * (1 + 2 * lam * s2) / (1 + 3 * lam * s2)))
eps = sp.symbols("epsilon", positive=True)
eq = sp.together(1 / v - (1 / s2 + 1 / (eps + (l - 1) * v)))
num = sp.expand(sp.numer(eq) * (-1))
target = sp.expand((l - 1) / s2 * v ** 2 + (eps / s2 + 2 - l) * v - eps)
print("   law-level quadratic check (ratio should be constant):", sp.simplify(num / target))
cav = 1 / (1 / s2 + 1 / (1 / lam + (l - 1) * s2))
print("   cavity minus exact (general l):",
      sp.simplify(cav - s2 * (1 + (l - 1) * lam * s2) / (1 + l * lam * s2)))
# uniqueness among all fixed points: v_e = phi(S - v_e), phi increasing => every fixed point is symmetric
sig2n, lamn = 1.0, 50.0
for ell in (2, 3, 5):
    sols = []
    for trial in range(200):
        vv = rng.uniform(0.01, 3.0, ell)
        for _ in range(20000):
            new = np.array([1.0 / (1.0 / sig2n + 1.0 / (1.0 / lamn + vv.sum() - vv[e])) for e in range(ell)])
            if np.abs(new - vv).max() < 1e-14:
                break
            vv = 0.5 * vv + 0.5 * new
        sols.append(vv)
    sols = np.array(sols)
    print("   l =", ell, ": spread of fixed points over 200 random starts =", np.ptp(sols, axis=0).max(),
          " max asymmetry =", np.max(np.ptp(sols, axis=1)), " value", sols[0, 0])

# ---------------- D ----------------
def ring_spectrum(p, q, s, c, ell=3):
    Ai = np.array([[p + q, -q * c], [-q * c, s + q * c * c]])
    Cp = np.array([[-q, 0.0], [q * c, 0.0]])
    n = 2 * ell
    H = np.zeros((n, n)); Ab = np.zeros((n, n))
    for i in range(ell):
        H[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Ai; Ab[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Ai
        j = (i + 1) % ell
        H[2 * i:2 * i + 2, 2 * j:2 * j + 2] = Cp
    return np.linalg.eigvals(-np.linalg.solve(Ab, H))


for c in (0.0, 0.9):
    for s in (1e-6, 1.0, 1e6):
        ev = ring_spectrum(1.3, 2.1, s, c)
        w = 2.1 * s / (1.3 * s + 1.3 * 2.1 * c * c + 2.1 * s)
        pred = np.array([-1.0] * 3 + [-(1 - z * w) for z in np.exp(2j * np.pi * np.arange(3) / 3)])
        err = max(min(abs(e - pr) for pr in pred) for e in ev)
        print("D  c =", c, " s =", s, ": w =", round(w, 6), " max |Im| =", round(np.abs(ev.imag).max(), 6),
              " |eig - formula| =", err)

# ---------------- E ----------------
pj, vk, Rr, d2 = sp.symbols("p_j v_k R d2", positive=True)
f = pj - 1 / v + 1 / (v + Rr) - (vk + Rr + d2) / (v + Rr) ** 2
print("E  df/d(d2) =", sp.simplify(sp.diff(f, d2)), " => dv/d(d2) = -f_d2/f_v > 0 when f_v > 0")
