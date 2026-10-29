"""XC1-GAUGE check 4: dependence of minimized contextual tensors and curvature
on the declared collective group.

(a) ch14-type family: three agents, each with a marked von Mises factor
    exp(A_i . e_phi) and a private factor exp(B_i . e_psi).  Section along one
    contextual direction c: A_1 = a e(chi(c)), A_2, A_3 fixed; B_i = b_i e(beta_i + lam c)
    (a common psi-phase gradient lam).  Fisher metric of the product law,
    orbit scores of (i) the marked group only (as declared in ch14) and
    (ii) the marked group together with a common psi rotation.
(b) ch11/ch12 tied versus untied belief/model group (sympy).
(c) ch12 open:curvature large-time limit (mpmath Bessel functions).
"""
import numpy as np
import sympy as sp
import mpmath as mp

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# rotation Fisher inertia of a von Mises factor with concentration r: I(r) = r I1(r)/I0(r)
def inertia(r):
    return float(r * mp.besseli(1, r) / mp.besseli(0, r))
def radial_var(r):
    # Var(cos phi) under von Mises(r): d/dr (I1/I0) = 1 - I1/(r I0) - (I1/I0)^2
    m = mp.besseli(1, r) / mp.besseli(0, r)
    return float(1 - m / r - m * m)

def min_tensor(a, dchi, A_other, b_list, lam, include_psi):
    """Minimized Fisher length of the contextual tangent; scores of independent
    factors are orthogonal, so work in the basis of per-factor rotation/radial scores."""
    # coordinates: [radial_1, rot_phi_1, rot_phi_2, rot_phi_3, rot_psi_1, rot_psi_2, rot_psi_3]
    I_phi = [inertia(a)] + [inertia(r) for r in A_other]
    I_psi = [inertia(b) for b in b_list]
    G = np.diag([radial_var(a)] + I_phi + I_psi)
    v = np.array([0.0, dchi, 0, 0, lam, lam, lam])          # contextual tangent (da = 0 here)
    orbits = [np.array([0, 1, 1, 1, 0, 0, 0.0])]              # common marked rotation
    if include_psi:
        orbits.append(np.array([0, 0, 0, 0, 1, 1, 1.0]))       # common psi rotation
    Jm = np.column_stack(orbits)
    k = Jm.T @ G @ Jm
    xi = -np.linalg.solve(k, Jm.T @ G @ v)
    r = v + Jm @ xi
    return r @ G @ r, xi

a, A_other, b_list = 0.6, [0.4, 0.5], [0.3, 0.45, 0.2]
for lam in (0.0, 0.5, 1.0):
    hm, xim = min_tensor(a, 1.0, A_other, b_list, lam, include_psi=False)
    hp, xip = min_tensor(a, 1.0, A_other, b_list, lam, include_psi=True)
    p("(a) lam=%.1f  h_min(marked group)=%.6f   h_min(marked x common psi)=%.6f   psi-connection component=%s"
      % (lam, hm, hp, np.round(xip[1:], 6)))
I_phys = sum(inertia(b) for b in b_list)
p("    predicted lam^2 * I_phys increment at lam=1:", I_phys)

# (b) tied versus untied: cost I1 (dchi + xi_q)^2 + I2 xi_s^2 (+ g_aa da^2), connection forms in chi
I1, I2 = sp.symbols('I1 I2', positive=True); xq, xs, xi = sp.symbols('xi_q xi_s xi', real=True)
tied = sp.solve(sp.diff(I1 * (1 + xi)**2 + I2 * xi**2, xi), xi)[0]
p("(b) tied minimizer xi =", sp.simplify(tied), " residual =", sp.simplify((I1 * (1 + xi)**2 + I2 * xi**2).subs(xi, tied)))
sol = sp.solve([sp.diff(I1 * (1 + xq)**2 + I2 * xs**2, xq), sp.diff(I1 * (1 + xq)**2 + I2 * xs**2, xs)], [xq, xs])
p("    untied minimizer =", sol, " residual =", sp.simplify((I1 * (1 + xq)**2 + I2 * xs**2).subs(sol)),
  " (connection -dchi on beliefs, 0 on models: flat for any I1(a), I2)")

# (c) ch12 curvature limit
mp.mp.dps = 30
def Ifun(v): return v * mp.besseli(1, v) / mp.besseli(0, v)
def Iprime(v): return mp.diff(Ifun, v)
kap, lamv = mp.mpf('0.8'), mp.mpf('1.3')
for t in (0, 2, 6, 12):
    rho = mp.e**(-2 * t)
    at, bt = Ifun(rho * kap), Ifun(rho * lamv)
    F = -bt * rho * Iprime(rho * kap) / (at + bt)**2
    p("(c) t=%2d  F_t coefficient = %s" % (t, mp.nstr(F, 12)))
p("    claimed limit -2 lam^2 kappa/(kappa^2+lam^2)^2 =", mp.nstr(-2 * lamv**2 * kap / (kap**2 + lamv**2)**2, 12))

with open(__file__.replace(".py", "_output.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
