r"""L05-FUS: checks of the spectrum and marked-feedback formulas in chapters/07_smooth_closure.tex.

 R1  swapped binary example: record probabilities 5/8, 3/8; posteriors (9/10,1/10), (1/2,1/2)
 R2  entropy rate dH/dt = -Var_{e^r xi}(r) along the explicit solution (smooth:entropy-rate)
 R3  inverse map (smooth:spectrum-inverse)
 R4  marked feedback (smooth:record-normalizers, smooth:feedback-update): marked-spectrum formulas
     equal direct Bayes; within-mark rearrangements give identical record laws and next states
     after record -> flow -> record; a cross-mark rearrangement with equal unmarked spectrum differs
 R5  injectivity of the unmarked spectrum for a generic non-uniform private reference
"""
import numpy as np

# R1
L1 = np.array([0.75, 0.25])
for k in (np.array([0.75, 0.25]), np.array([0.25, 0.75])):
    p1 = k @ L1
    print("R1: conditional", k, "P(record 1) =", p1, "posterior", k * L1 / p1)

# helpers: spectrum as a list of (A-mass, r, mark) atoms
def flow(kappa, A, t):
    a = np.exp(-t)
    k = kappa**a * A**(1 - a)
    return k / k.sum()

def H_of(kappa, A):
    return np.sum(kappa * np.log(kappa / A))

# R2
A = np.array([0.1, 0.2, 0.3, 0.4])
k0 = np.array([0.4, 0.3, 0.2, 0.1])
t, h = 0.7, 1e-5
dH = (H_of(flow(k0, A, t + h), A) - H_of(flow(k0, A, t - h), A)) / (2 * h)
kt = flow(k0, A, t)
r = np.log(kt / A)
var = np.sum(kt * r**2) - np.sum(kt * r)**2
print("R2: dH/dt numeric", dH, " -Var formula", -var, " residual", dH + var)

# R3 inverse map on the atoms of xi^t: y -> y/a - log int e^{r/a} xi^t(dr)
a = np.exp(-t)
r0 = np.log(k0 / A)
rt = np.log(kt / A)
rec = rt / a - np.log(np.sum(A * np.exp(rt / a)))
print("R3: inverse map residual", np.abs(rec - r0).max())

# R4 marked feedback, uniform A on 4 points, marks T = (0,0,1,1)
A4 = np.full(4, 0.25)
Tm = np.array([0, 0, 1, 1])
Lik = {0: np.array([0.2, 0.7]), 1: np.array([0.8, 0.3])}   # L(o | d), o in {0,1}, d in {0,1}

def marked_spectrum(kappa, A, T):
    r = np.log(kappa / A)
    return sorted(zip(np.round(r, 12), T, A))

def marked_update(atoms, o):
    # atoms: list of (r, d, A-mass); N = int e^r L(o|d) eta(dr,dd)
    N = sum(np.exp(r) * Lik[o][d] * m for r, d, m in atoms)
    new = [(r + np.log(Lik[o][d]) - np.log(N), d, m) for r, d, m in atoms]
    return N, new

def marked_flow(atoms, t):
    a = np.exp(-t)
    Z = sum(m * np.exp(a * r) for r, d, m in atoms)
    return [(a * r - np.log(Z), d, m) for r, d, m in atoms]

def direct(kappa, o):
    w = kappa * Lik[o][Tm]
    return w.sum(), w / w.sum()

kap = np.array([0.4, 0.2, 0.1, 0.3])
kap_within = np.array([0.2, 0.4, 0.3, 0.1])     # swaps inside each mark class
kap_cross = np.array([0.1, 0.3, 0.4, 0.2])      # same unmarked spectrum, different marked spectrum
for name, k in (("kappa", kap), ("within-mark", kap_within), ("cross-mark", kap_cross)):
    atoms = [(np.log(x / 0.25), d, 0.25) for x, d in zip(k, Tm)]
    N1, at1 = marked_update(atoms, 1)
    Nd, post = direct(k, 1)
    at1f = marked_flow(at1, 0.5)
    N2, at2 = marked_update(at1f, 0)
    Nd2, post2 = direct(flow(post, A4, 0.5), 0)
    chk = max(abs(N1 - Nd), abs(N2 - Nd2),
              np.abs(np.sort([r for r, d, m in at2]) - np.sort(np.log(post2 / 0.25))).max())
    print(f"R4 {name:12s}: unmarked spectrum {np.sort(np.round(np.log(k/0.25),6))}, P(o=1)={N1:.6f},",
          f"then flow 0.5 and P(o=0)={N2:.6f}; marked-vs-direct residual {chk:.2e};",
          f"H after = {sum(np.exp(r)*r*m for r,d,m in at2):.6f}")

# R5 generic non-uniform reference: spectrum separates swapped conditionals
A2 = np.array([0.6, 0.4])
for k in (np.array([0.75, 0.25]), np.array([0.25, 0.75])):
    print("R5: A=(0.6,0.4), conditional", k, "spectrum atoms (mass, value):",
          sorted(zip(A2, np.round(np.log(k / A2), 6))))
