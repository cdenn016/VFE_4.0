r"""XC7-IMPL s8: exact finite conformance checks for the two memory appendices.

(a) imem binary two-way exchange: brute-force enumeration with exact fractions of every
    rational in imem:binary-first-likelihood ... imem:binary-repeated-samples.
(b) imem parity collision imem:parity-message ... imem:parity-local-output.
(c) imem:gaussian-closure: the reduced recursion for lambda_t after two messages vs
    brute-force conditioning of the jointly Gaussian (D, H_1, H_2); imem:scalar-calibration.
(d) cmem: component posteriors, response and Fisher identities, projection identities.
"""
from fractions import Fraction as Fr
from itertools import product
import numpy as np

q34, q14 = Fr(3, 4), Fr(1, 4)
def kern(out, cue, same):
    hit = (out == cue) if same else (out != cue)
    return q34 if hit else q14
def joint_xd(x0, x1, d0, d1):
    return Fr(1, 2) * kern(x1, x0, True) * kern(d0, x0, True) * kern(d1, x1, True)
def P(event):
    return sum(joint_xd(*z) for z in product((0, 1), repeat=4) if event(*z))
def py_given_x(y, x): return kern(y, x, False)
def k10(h, d1):
    num = sum(joint_xd(x0, x1, d0, d1) * py_given_x(h, x1) for x0, x1, d0 in product((0, 1), repeat=3))
    return num / P(lambda x0, x1, d0, dd1: dd1 == d1)
print("(a) k10(1|0), k10(1|1) =", k10(1, 0), k10(1, 1), " claimed 5/8, 3/8")
def lik_H1_given(idx, val):
    num = sum(joint_xd(*z) * k10(1, z[3]) for z in product((0, 1), repeat=4) if z[idx] == val)
    return num / P(lambda *z: z[idx] == val)
print("    P(H1=1|X1=0,1) =", lik_H1_given(1, 0), lik_H1_given(1, 1), " P(H1=1|X0=0,1) =", lik_H1_given(0, 0), lik_H1_given(0, 1))
def post_x0(d0, h1=1):
    num = sum(joint_xd(1, x1, d0, d1) * k10(h1, d1) for x1, d1 in product((0, 1), repeat=2))
    den = sum(joint_xd(x0, x1, d0, d1) * k10(h1, d1) for x0, x1, d1 in product((0, 1), repeat=3))
    return num / den
print("    P(X0=1|D0=1,H1=1) =", post_x0(1), " (45/62);  at D0=0:", post_x0(0), " (5/22)")
def k01(h, d0, h1=1):
    p1 = post_x0(d0, h1)
    return p1 * py_given_x(h, 1) + (1 - p1) * py_given_x(h, 0)
print("    k01(1|D0=0,H1=1), k01(1|D0=1,H1=1) =", k01(1, 0), k01(1, 1), " (7/11, 12/31)")
pairs = list(product((0, 1), repeat=2))
lam0 = [P(lambda x0, x1, d0, d1, a=a, b=b: d0 == a and d1 == b) for a, b in pairs]
lam1 = [l * k10(1, b) for l, (a, b) in zip(lam0, pairs)]; Z1 = sum(lam1); lam1 = [l / Z1 for l in lam1]
lam2 = [l * k01(1, a) for l, (a, b) in zip(lam1, pairs)]; Z2 = sum(lam2); lam2 = [l / Z2 for l in lam2]
print("    lambda0 =", lam0, " lambda1 =", lam1, " lambda2 =", lam2, " normalizers", Z1, Z2)
px0_D = [P(lambda x0, x1, d0, d1, a=a, b=b: x0 == 1 and d0 == a and d1 == b) / l for l, (a, b) in zip(lam0, pairs)]
px1_D = [P(lambda x0, x1, d0, d1, a=a, b=b: x1 == 1 and d0 == a and d1 == b) / l for l, (a, b) in zip(lam0, pairs)]
V0 = [p * q14 + (1 - p) * q34 for p in px0_D]; V1 = [p * q14 + (1 - p) * q34 for p in px1_D]
print("    P(X0=1|D) =", px0_D, " V0 =", V0, " V1 =", V1)
pD0 = lam2[2] / (lam2[0] + lam2[2])
print("    P(D0=1|D1=0,H1=H2=1) =", pD0, " (308/959);  P(X1=1|D1=0,H1=H2=1) =",
      (1 - pD0) * px1_D[0] + pD0 * px1_D[2], " (437/1918)")
num = sum(joint_xd(1, x1, 1, d1) * py_given_x(1, x1) for x1, d1 in product((0, 1), repeat=2))
den = sum(joint_xd(x0, x1, 1, d1) * py_given_x(1, x1) for x0, x1, d1 in product((0, 1), repeat=3))
print("    direct Y1=1: P(X0=1|D0=1,Y1=1) =", num / den, " (9/14)")
rep = [sum(kern(d1, x1, True) * k10(1, d1) ** 2 for d1 in (0, 1)) for x1 in (0, 1)]
print("    repeated samples P(H1=H2=1|X1=0,1) =", rep, " (21/64, 13/64)")
num = sum(joint_xd(1, x1, 1, d1) * k10(1, d1) ** 2 for x1, d1 in product((0, 1), repeat=2))
den = sum(joint_xd(x0, x1, 1, d1) * k10(1, d1) ** 2 for x0, x1, d1 in product((0, 1), repeat=3))
lx = [Fr(9, 16) ** 2, Fr(7, 16) ** 2]
num2 = sum(Fr(1, 2) * kern(1, 1, True) * kern(x1, 1, True) * lx[x1] for x1 in (0, 1))
den2 = sum(Fr(1, 2) * kern(1, x0, True) * kern(x1, x0, True) * lx[x1] for x0, x1 in product((0, 1), repeat=2))
print("    P(X0=1|D0=1,H1=H2=1) =", num / den, " (45/64); with squared marginal likelihoods:", num2 / den2, " (171/244)")

# ---------------------------------------------------------------- (b) parity collision
for rho in (Fr(1, 3), Fr(1, 2)):
    def jp(x0, b, d1, rho=rho): return Fr(1, 2) * (rho if b == 0 else 1 - rho) * kern(d1, x0 ^ b, True)
    def kH(d1):
        tot = sum(jp(x0, b, d1) for x0, b in product((0, 1), repeat=2))
        px1 = sum(jp(x0, b, d1) for x0, b in product((0, 1), repeat=2) if x0 ^ b == 1) / tot
        return px1 * q14 + (1 - px1) * q34
    res = []
    for C in (0, 1):
        pH = sum(jp(x0, C, d1) * kH(d1) for x0, d1 in product((0, 1), repeat=2)) / sum(jp(x0, C, d1) for x0, d1 in product((0, 1), repeat=2))
        px0 = sum(jp(1, C, d1) * kH(d1) for d1 in (0, 1)) / sum(jp(x0, C, d1) * kH(d1) for x0, d1 in product((0, 1), repeat=2))
        res.append((pH, px0, q14 + px0 / 2, q34 - px0 / 2))
    print(f"(b) rho={rho}: P(H=1|C)=({res[0][0]},{res[1][0]}), P(X0=1|C,H=1)=({res[0][1]},{res[1][1]}), "
          f"P(A0=1|C,H=1)=({res[0][2]},{res[1][2]}), P(Y0=1|C,H=1)=({res[0][3]},{res[1][3]}), k(1|D1=0,1)=({kH(0)},{kH(1)})")

# ---------------------------------------------------------------- (c) scalar Gaussian closure, two messages
nu0, nu1, L0, L1, S01 = 0.3, -0.5, 1.2, 0.9, 0.6
N0, N1 = 0.5, 0.4
om0, om1, f0, f1, Q0, Q1 = 0.2, -0.1, -1.3, 0.8, 0.3, 0.25
b0, b1, R0, R1 = 0.9, 1.1, 0.2, 0.15
def npdf(z, m, v): return np.exp(-(z - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
SD = np.array([[L0 + N0, S01], [S01, L1 + N1]]); mD = np.array([nu0, nu1])
g1, P1 = L1 / (L1 + N1), L1 * N1 / (L1 + N1)
def k1(h, d1): return npdf(h, b1 * (om1 + f1 * g1 * (d1 - nu1)), b1 ** 2 * (Q1 + f1 ** 2 * P1) + R1)
grid = np.linspace(-9, 9, 3601); dg = grid[1] - grid[0]
def agent0_posterior_x0(d0, h1):
    # (X0, D1) | D0 = d0 is Gaussian; integrate D1 against the actual message density k1(h1 | D1)
    Sfull = np.array([[L0, S01, L0], [S01, L1 + N1, S01], [L0, S01, L0 + N0]])     # (X0, D1, D0)
    mfull = np.array([nu0, nu1, nu0])
    G = Sfull[:2, 2] / Sfull[2, 2]; mc = mfull[:2] + G * (d0 - mfull[2]); Sc = Sfull[:2, :2] - np.outer(G, Sfull[2, :2])
    gx = Sc[0, 1] / Sc[1, 1]; vx = Sc[0, 0] - gx * Sc[0, 1]
    wts = npdf(grid, mc[1], Sc[1, 1]) * k1(h1, grid); wts /= wts.sum()
    Ex_d1 = mc[0] + gx * (grid - mc[1])
    Ex = np.sum(wts * Ex_d1); Vx = vx + np.sum(wts * (Ex_d1 - Ex) ** 2)
    return Ex, Vx
def k0(h2, d0, h1):
    Ex, Vx = agent0_posterior_x0(d0, h1)
    return npdf(h2, b0 * (om0 + f0 * (Ex - nu0)), b0 ** 2 * (Q0 + f0 ** 2 * Vx) + R0)
h1, h2 = 0.7, -0.4
# brute-force posterior of D = (D0, D1) on a grid
gd = np.linspace(-6, 6, 241); D0g, D1g = np.meshgrid(gd, gd, indexing='ij')
Sinv = np.linalg.inv(SD); dd = np.stack([D0g - mD[0], D1g - mD[1]], -1)
prior = np.exp(-0.5 * np.einsum('...i,ij,...j->...', dd, Sinv, dd))
k0vals = np.array([k0(h2, d0, h1) for d0 in gd])[:, None]
post = prior * k1(h1, D1g) * k0vals; post /= post.sum()
mb = np.array([np.sum(post * D0g), np.sum(post * D1g)])
Sb = np.array([[np.sum(post * (D0g - mb[0]) * (D0g - mb[0])), np.sum(post * (D0g - mb[0]) * (D1g - mb[1]))],
               [0, np.sum(post * (D1g - mb[1]) * (D1g - mb[1]))]]); Sb[1, 0] = Sb[0, 1]
# reduced recursion (imem:gaussian-reduced-recursion) with message coefficients from the current Gaussian lambda_t
def coeffs(j, m, Sg):
    # E[X_j | D] from the fixed reconstruction W, then condition the lambda_t Gaussian on D_j
    SXD = np.array([[L0, S01], [S01, L1]]); WG = SXD @ np.linalg.inv(SD); WC = SXD - WG @ SD @ WG.T
    mX = np.array([nu0, nu1]) + WG @ (m - mD)
    cXjD = (WG @ Sg)[j]; VXj = (WG @ Sg @ WG.T + WC)[j, j]
    G = cXjD[j] / Sg[j, j]; Px = VXj - G * cXjD[j]
    f, om_, nu_, b, R = (f0, om0, nu0, b0, R0) if j == 0 else (f1, om1, nu1, b1, R1)
    A = np.zeros(2); A[j] = b * f * G
    bb = b * (om_ + f * (mX[j] - G * m[j] - nu_))
    return bb, A, b ** 2 * (Q0 if j == 0 else Q1) + b ** 2 * f ** 2 * Px + R
m, Sg = mD.copy(), SD.copy()
for j, h in ((1, h1), (0, h2)):
    bb, A, R = coeffs(j, m, Sg)
    Sn = np.linalg.inv(np.linalg.inv(Sg) + np.outer(A, A) / R)
    m = Sn @ (np.linalg.solve(Sg, m) + A * (h - bb) / R); Sg = Sn
print(f"(c) lambda_2 mean: recursion {m}, grid {mb};  cov: recursion {Sg.ravel()}, grid {Sb.ravel()}")
import sympy as sp
L1s, N1s, Q1s, f1s, bs, Rs_ = sp.symbols('L1 N1 Q1 f1 b R', positive=True)
gs = L1s / (L1s + N1s); Ps = L1s * N1s / (L1s + N1s); a_ = bs * f1s * gs
Neff = a_ ** 2 * N1s + bs ** 2 * (Q1s + f1s ** 2 * Ps) + Rs_
print("    imem:scalar-calibration Var(H) - Var(H_dir) =", sp.simplify(a_ ** 2 * L1s + Neff - (bs ** 2 * (f1s ** 2 * L1s + Q1s) + Rs_)))

# ---------------------------------------------------------------- (d) conditional memory, 1-D, M = 2
pis = np.array([0.35, 0.65]); nus = np.array([-0.8, 1.1]); Ls = np.array([0.6, 1.4]); Cn = 0.5
oms = np.array([0.4, -0.7]); Fs = np.array([-1.5, 0.9]); Qs = np.array([0.3, 0.2])
xg = np.linspace(-8, 8, 801); yg = np.linspace(-10, 10, 1001); XX, YY = np.meshgrid(xg, yg, indexing='ij')
def joint_post(xi):
    dens = []
    for a in range(2):
        dens.append(pis[a] * npdf(XX, nus[a], Ls[a]) * npdf(YY, oms[a] + Fs[a] * (XX - nus[a]), Qs[a]) * npdf(xi, XX, Cn))
    dens = np.array(dens); return dens / dens.sum()
def formula(xi):
    W = Ls + Cn; w = pis * npdf(xi, nus, W); w /= w.sum()
    mX = nus + Ls / W * (xi - nus); PX = 1 / (1 / Ls + 1 / Cn); mY = oms + Fs * (mX - nus); PY = Qs + Fs ** 2 * PX
    return w, mX, PX, mY, PY
xi = 0.4
Pj = joint_post(xi); w, mX, PX, mY, PY = formula(xi)
wq = Pj.sum(axis=(1, 2)); mXq = np.array([np.sum(Pj[a] * XX) / wq[a] for a in range(2)])
mYq = np.array([np.sum(Pj[a] * YY) / wq[a] for a in range(2)]); PYq = np.array([np.sum(Pj[a] * (YY - mYq[a]) ** 2) / wq[a] for a in range(2)])
print("(d) cmem:gaussian-posterior-formula: |w|", np.abs(w - wq).max(), " |m_X|", np.abs(mX - mXq).max(),
      " |m_Y|", np.abs(mY - mYq).max(), " |P_Y|", np.abs(PY - PYq).max())
def EY(xi): Pj = joint_post(xi); return np.sum(Pj * YY[None])
h = 1e-4; resp_fd = (EY(xi + h) - EY(xi - h)) / (2 * h)
Pj = joint_post(xi); EX = np.sum(Pj * XX[None]); EYv = np.sum(Pj * YY[None])
covYX = np.sum(Pj * (YY[None] - EYv) * (XX[None] - EX)); varX = np.sum(Pj * (XX[None] - EX) ** 2)
lp_p, lp_m = np.log(joint_post(xi + h) + 1e-300), np.log(joint_post(xi - h) + 1e-300)
score = (lp_p - lp_m) / (2 * h); fisher = np.sum(Pj * score ** 2)
print("    cmem:response-fisher: dE[Y|xi]/dxi", resp_fd, "vs Cov(Y,X|xi)/C", covYX / Cn, ";  Fisher", fisher, "vs Var(X|xi)/C^2", varX / Cn ** 2)
for t, c in ((0.5, 0.3), (0.8, 1.0)):
    v1 = 1 - t ** 2 + c
    lhs = npdf(0.0, t, v1)
    print(f"    cmem:readout-loss t={t},c={c}: (p_t K_c)(0)={lhs:.10f} formula {np.exp(-t**2/(2*v1))/np.sqrt(2*np.pi*v1):.10f}, (q_G K_c)(0)={1/np.sqrt(2*np.pi*(1+c)):.10f}")
tt = 0.6; hh = 1e-5
def logfam(tv, a, yv): return np.log(0.5) + np.log(npdf(yv, a * tv, 1 - tv ** 2))
ygr = np.linspace(-9, 9, 20001); fis = 0
for a in (1, -1):
    sc = (logfam(tt + hh, a, ygr) - logfam(tt - hh, a, ygr)) / (2 * hh)
    fis += np.sum(0.5 * npdf(ygr, a * tt, 1 - tt ** 2) * sc ** 2) * (ygr[1] - ygr[0])
print(f"    cmem:joint-fisher coefficient along t at t={tt}: {fis:.8f} vs (1+t^2)/(1-t^2)^2 = {(1+tt**2)/(1-tt**2)**2:.8f}")
for a in (1.0, 3.0, 6.0):
    yv = np.linspace(-30, 30, 200001); dy = yv[1] - yv[0]
    q = npdf(yv, 0, 1 + a * a); p = 0.5 * npdf(yv, -a, 1) + 0.5 * npdf(yv, a, 1)
    klqp = np.sum(q * (np.log(q) - np.log(p))) * dy
    bound = a * a - a * np.sqrt(2 * (1 + a * a) / np.pi) - 0.5 * np.log(1 + a * a)
    print(f"    cmem:reverse-counterexample a={a}: KL(q_a||p_a) = {klqp:.5f} >= bound {bound:.5f}: {klqp >= bound}")
