r"""L06-ERA script 1: symbolic and scalar checks for chapters/08_smooth_erasure.tex.

Checks (residuals printed):
  (a) erase:local from KL(q (x) s || R) on the three-bit product family;
  (b) d-derivatives, erase:gate-derivative and the constant B;
  (c) constants m, M, V, M_chi and a feasible (delta, rho) pair;
  (d) erase:well-margins numerically;
  (e) erase:public-derivatives and their O(eps^2) order;
  (f) erase:crossed-pair-gap identity;
  (g) erase:mixed-term coefficient from the cumulant expansion of -T log Z;
  (h) erase:mixed-witness closed form;
  (i) erase:separated-energy after closure.
"""
import sympy as sp
import numpy as np

x, z, c, v, eps, J, k, a, b, r, zeta, rho, al, T = sp.symbols(
    'x z c v epsilon J k a b r zeta rho alpha T', real=True)


def bern(mean, kk):
    return (1 + kk * mean) / 2


def d(xx, zz):
    """KL(b_x || b_z) on {-1,1}."""
    return sum(bern(xx, kk) * sp.log(bern(xx, kk) / bern(zz, kk)) for kk in (-1, 1))


def psi(xx):
    return d(xx, 0)


# (a) local term: KL(q (x) s || r G) with q,s products of three bits
xs = sp.symbols('x1 z1 h1 y1 w1 hh1', real=True)
X1, Z1, H1, Y1, W1, HH1 = xs
hstar = sp.Rational(-1, 2)
Jv = sp.log(3)
E = 0
for k1 in (-1, 1):
    for k2 in (-1, 1):
        for k3 in (-1, 1):
            for m1 in (-1, 1):
                for m2 in (-1, 1):
                    for m3 in (-1, 1):
                        qk = bern(X1, k1) * bern(Z1, k2) * bern(H1, k3)
                        sm = bern(Y1, m1) * bern(W1, m2) * bern(HH1, m3)
                        rm = bern(0, m1) * bern(0, m2) * bern(hstar, m3)
                        Gk = sp.exp(Jv * k1 * m1) / (2 * sp.cosh(Jv)) * bern(0, k2) * bern(hstar, k3)
                        E += qk * sm * sp.log(qk * sm / (rm * Gk))
claimed = (psi(X1) + psi(Y1) - Jv * X1 * Y1 + sp.log(sp.cosh(Jv))
           + psi(Z1) + psi(W1) + d(H1, hstar) + d(HH1, hstar))
rng = np.random.default_rng(0)
res = 0.0
for _ in range(5):
    vals = {s: float(rng.uniform(-0.9, 0.9)) for s in xs}
    res = max(res, abs(float((E - claimed).subs(vals).evalf())))
print('(a) erase:local max residual over 5 random points:', res)

# (b) derivatives of d
d1 = sp.simplify(sp.diff(d(x, z), x) - (sp.atanh(x) - sp.atanh(z)))
d2 = sp.simplify(sp.diff(d(x, z), z) - (z - x) / (1 - z**2))
print('(b) d_1 d - (atanh x - atanh z):', sp.simplify(sp.expand_log(sp.diff(d(x, z), x).rewrite(sp.log) - (sp.atanh(x) - sp.atanh(z)).rewrite(sp.log), force=True)))
print('(b) d_2 d - (z-x)/(1-z^2):', d2)
za, zj = sp.symbols('z_a z_j', real=True)
gd = sp.diff(d(za, al * rho * zj), al)
claim_gd = rho * zj * (al * rho * zj - za) / (1 - al**2 * rho**2 * zj**2)
print('(b) erase:gate-derivative residual:', sp.simplify(gd - claim_gd))
p = 0.75
B = 3 * p**2 / (2 * (1 - p**2 / 4))
# brute-force max of |claim_gd|/rho over |z|<=p, rho<=1/2, alpha in [0,1]
f_gd = sp.lambdify((za, zj, al, rho), claim_gd / rho, 'numpy')
grid = np.linspace(-p, p, 61)
mx = 0.0
for rr in np.linspace(1e-3, 0.5, 11):
    for aa in np.linspace(0, 1, 11):
        Za, Zj = np.meshgrid(grid, grid)
        mx = max(mx, np.abs(f_gd(Za, Zj, aa, rr)).max())
print(f'(b) B claimed = {B:.6f}, grid max of |d_alpha d|/rho = {mx:.6f}')

# (c) constants
hs = np.linspace(1 / 8, 1 / 2, 20001)
gate = 2 * (1 - hs**2) * (np.arctanh(hs) - np.arctanh(-0.5))
m_c, M_c = gate.min(), gate.max()
V_c = M_c + m_c / 2


def chi(rr):
    rr = np.asarray(rr, float)
    out = np.zeros_like(rr)
    inside = (rr > 0) & (rr < 1)
    e1 = np.exp(-1 / rr[inside])
    e2 = np.exp(-1 / (1 - rr[inside]))
    out[inside] = e1 / (e1 + e2)
    out[rr >= 1] = 1.0
    return out


rr = np.linspace(1e-6, 1 - 1e-6, 200001)
Mchi = np.max(np.gradient(chi(rr), rr))
delta_max = (1 / 8) * 2 * m_c / (3 * V_c)
print(f'(c) m={m_c:.6f} M={M_c:.6f} V={V_c:.6f} M_chi={Mchi:.6f}; delta < {delta_max:.6f} (and <= 1/3)')
dl = 0.04
rho_max = m_c * dl / (2 * B * Mchi)
print(f'(c) with delta=0.04: rho <= {rho_max:.6e} (gate condition)')

# (d) well margins
l, u = 0.49, 0.51
Jn = np.log(3.0)
print(f'(d) J l - atanh l = {Jn*l-np.arctanh(l):.6e}, atanh u - J u = {np.arctanh(u)-Jn*u:.6e}, '
      f'J*1/2-atanh(1/2) = {Jn/2-np.arctanh(0.5):.2e}')

# (e) public derivatives
cc = (1 - eps) * x + eps * v
De = d(x, cc)
p1 = sp.atanh(x) - sp.atanh(cc) + (1 - eps) * (cc - x) / (1 - cc**2)
p2 = eps * (cc - x) / (1 - cc**2)
fp1 = sp.lambdify((x, v, eps), sp.diff(De, x) - p1)
fp2 = sp.lambdify((x, v, eps), sp.diff(De, v) - p2)
mx1 = max(abs(fp1(xx, vv, 0.03)) for xx in np.linspace(-.51, .51, 7) for vv in np.linspace(-.51, .51, 7))
mx2 = max(abs(fp2(xx, vv, 0.03)) for xx in np.linspace(-.51, .51, 7) for vv in np.linspace(-.51, .51, 7))
print(f'(e) erase:public-derivatives residuals: {mx1:.2e}, {mx2:.2e}')
ser1 = sp.series(p1, eps, 0, 3).removeO()
ser2 = sp.series(p2, eps, 0, 3).removeO()
print('(e) d1 D_eps series to eps^2:', sp.simplify(ser1))
print('(e) d2 D_eps series to eps^2:', sp.simplify(ser2))

# (f) crossed-pair gap
gap = sp.simplify(sp.expand_log(d(zeta, -c) - d(zeta, c), force=True))
print('(f) d(zeta,-c)-d(zeta,c) - 2 zeta atanh c =',
      sp.simplify((gap - 2 * zeta * sp.atanh(c)).rewrite(sp.log)))

# (g) cumulant expansion of -T log E exp(-X/T): mixed coefficient
s = sp.symbols('s', positive=True)   # bookkeeping
A_ = sp.symbols('A1:4', real=True)
Zs = sp.symbols('Z1:4', real=True)
zi = sp.symbols('z_i', real=True)
Xs = [eps**2 * A_[j] - al * rho * zi * Zs[j] for j in range(3)]
F = -T * sp.log(sum(sp.exp(-Xj / T) for Xj in Xs) / 3)
# mixed coefficient: d^2/(d eps^2 d rho) at 0, divided appropriately
mixed = sp.diff(F, eps, 2, rho, 1).subs({eps: 0, rho: 0}) / 2
cov = sum(A_[j] * Zs[j] for j in range(3)) / 3 - (sum(A_) / 3) * (sum(Zs) / 3)
print('(g) mixed coefficient - alpha z_i Cov(A,z)/T =',
      sp.simplify(mixed - al * zi * cov / T))

# (h) witness
vv = [-b, -a, a, b]
zz = [zeta, -zeta, zeta, -zeta]
xr = [r * t for t in vv]
tot = 0
for i in range(4):
    js = [j for j in range(4) if j != i]
    Aij = [(xr[j] - xr[i])**2 / (2 * (1 - xr[i]**2)) for j in js]
    zj_ = [zz[j] for j in js]
    covi = sum(Aij[t] * zj_[t] for t in range(3)) / 3 - (sum(Aij) / 3) * (sum(zj_) / 3)
    tot += zz[i] * covi
claim_w = 2 * zeta**2 * r**2 / 9 * ((a**2 - 3 * b**2 + 6 * a * b) / (1 - r**2 * b**2)
                                    + (b**2 - 3 * a**2 + 6 * a * b) / (1 - r**2 * a**2))
print('(h) erase:mixed-witness residual:', sp.simplify(sp.together(tot - claim_w)))
num1 = lambda A, Bb: A**2 - 3 * Bb**2 + 6 * A * Bb
print('(h) numerators at (a,b)=(0.49,0.51):', num1(0.49, 0.51), num1(0.51, 0.49))

# (i) separated energy: -T log Z_i after closure
Ds = sp.symbols('D1:4', real=True)
Pz, Hh = sp.symbols('Pz Hh', real=True)
lhs = -T * sp.log(sum(sp.exp(-(Ds[j] + Pz + Hh) / T) for j in range(3)) / 3)
rhs = Pz + Hh - T * sp.log(sum(sp.exp(-Ds[j] / T) for j in range(3)) / 3)
vals = {Ds[0]: 0.3, Ds[1]: 0.7, Ds[2]: 1.1, Pz: 0.2, Hh: 0.45, T: 0.6}
print('(i) row separation residual (numeric):', float((lhs - rhs).subs(vals).evalf()))
