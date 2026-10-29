r"""GAP05-S3 s2: a continuous non-Gaussian instance of it2:law-potential on the circle.

Contents, messages and dictionaries live on S^1; SO(2) acts by rotation (a nonlinear
dictionary action on laws). Two agents, edges e: 1->0 and e': 0->1 (a two-cycle), one
candidate per row (beta = 1). Code kernel k(h|x) = vM(h; x, kc); receiver kernel
K_rec(w)(y, .) = vM(.; y - w, kc); source kernel K_src(y, .) = vM(.; y, kc).
q_i = vM(mu_i, kap_i); r_e = vM(alpha_e, k_e); references p_i are non-von-Mises
(cardioid-type); joint dictionary prior pi(w, w') ~ exp(s cos w + s cos w' + lam cos(w + w')).
Quadrature: periodic trapezoid rule on N points (spectrally accurate); convolutions by
dense matrices so that complex-step derivatives are exact up to rounding.
Checks: (0) quadrature convergence N=256 vs 512; (a) it2:law-cost-variation for parameter
directions of q_0, q_1, r_e; (b) own-block first variation in the vM chart (alpha_e, k_e);
(c) Fisher tensor of vM in the (alpha, k) chart and in natural coordinates.
"""
import numpy as np
from scipy.special import i0, i1

def setup(N):
    x = 2 * np.pi * np.arange(N) / N
    dx = 2 * np.pi / N
    return x, dx

def vm(x, mu, kap, dx):
    f = np.exp(kap * np.cos(x - mu))
    return f / (np.sum(f) * dx)

def build(N, params):
    x, dx = setup(N)
    mu0, kap0, mu1, kap1, al_e, k_e, al_f, k_f = params
    kc, s, lam = 2.0, 0.7, 1.1
    diff = x[:, None] - x[None, :]
    Kmat = np.exp(kc * np.cos(diff)); Kmat = Kmat / (np.sum(Kmat[:, 0]) * dx)   # Kmat[h, y] = k(h | y)
    q0 = vm(x, mu0, kap0, dx); q1 = vm(x, mu1, kap1, dx)
    re = vm(x, al_e, k_e, dx); rf = vm(x, al_f, k_f, dx)
    p0 = (1 + 0.6 * np.cos(x - 0.3)) ; p0 = p0 / (np.sum(p0) * dx)
    p1 = (1 + 0.4 * np.cos(2 * x + 1.0)); p1 = p1 / (np.sum(p1) * dx)
    W, Wp = np.meshgrid(x, x, indexing='ij')
    pi = np.exp(s * np.cos(W) + s * np.cos(Wp) + lam * np.cos(W + Wp)); pi = pi / (np.sum(pi) * dx * dx)
    return dict(x=x, dx=dx, N=N, Kmat=Kmat, q=[q0, q1], r=[re, rf], p=[p0, p1], pi=pi)

def KLd(a, b, dx):
    return np.sum(a * (np.log(a) - np.log(b))) * dx

def Delta_fun(M, i, j):
    """Delta(w_k) = KL(A(w_k) || B), A(w)(h) = (q_i * k)(h + w), B = q_j * k."""
    dx, N = M['dx'], M['N']
    a = M['Kmat'] @ M['q'][i] * dx
    b = M['Kmat'] @ M['q'][j] * dx
    D = np.empty(N, dtype=a.dtype)
    for k in range(N):
        A = np.roll(a, -k)          # A(h) = a(h + w_k)
        D[k] = np.sum(A * (np.log(A) - np.log(b))) * dx
    return D, a, b

def energy(M):
    dx = M['dx']
    De, _, _ = Delta_fun(M, 0, 1)      # edge e: sender 1 -> receiver 0
    Df, _, _ = Delta_fun(M, 1, 0)      # edge e': sender 0 -> receiver 1
    Ce = np.sum(M['r'][0] * De) * dx
    Cf = np.sum(M['r'][1] * Df) * dx
    joint = np.outer(M['r'][0], M['r'][1])
    F0 = KLd(M['q'][0], M['p'][0], dx) + KLd(M['q'][1], M['p'][1], dx) \
         + np.sum(joint * (np.log(joint) - np.log(M['pi']))) * dx * dx
    return F0 + Ce + Cf, Ce, Cf

P0 = np.array([0.4, 1.5, -1.0, 2.5, 0.8, 1.2, -0.5, 0.9])

def E_of(params, N=256, which=0):
    M = build(N, params)
    return energy(M)[which]

print("== (0) quadrature convergence ==")
for which, name in [(0, 'E'), (1, 'C_e')]:
    v256 = E_of(P0, 256, which).real; v512 = E_of(P0, 512, which).real
    print("%s: N=256 %.15f  N=512 %.15f  diff %.1e" % (name, v256, v512, abs(v256 - v512)))

h = 1e-30
def cdir(fun, k):
    P = P0.astype(complex); P[k] += 1j * h
    return np.imag(fun(P)) / h

print("== (a) it2:law-cost-variation along parameter directions ==")
M0 = build(256, P0)
x, dx = M0['x'], M0['dx']
De, a, b = Delta_fun(M0, 0, 1)
names = ['mu_0 (receiver)', 'kap_0 (receiver)', 'mu_1 (sender)', 'kap_1 (sender)', 'alpha_e (dictionary)', 'k_e (dictionary)']
worst = 0.0
for k in range(6):
    num = cdir(lambda P: energy(build(256, P))[1], k)
    # parameter derivatives of the laws by complex step (exact up to rounding)
    Pc = P0.astype(complex); Pc[k] += 1j * h
    Mc = build(256, Pc)
    dq0 = np.imag(Mc['q'][0]) / h; dq1 = np.imag(Mc['q'][1]) / h; dre = np.imag(Mc['r'][0]) / h
    da = M0['Kmat'] @ dq0 * dx      # (dq_i) K
    db = M0['Kmat'] @ dq1 * dx      # (dq_j) K_src
    form = np.sum(dre * De) * dx
    inner = np.empty(256)
    for kk in range(256):
        A = np.roll(a, -kk); dA = np.roll(da, -kk)
        inner[kk] = np.sum(np.log(A / b) * dA) * dx - np.sum((A / b) * db) * dx
    form += np.sum(M0['r'][0] * inner) * dx
    worst = max(worst, abs(num - form))
    print("  %-22s complex step %+.12f  formula %+.12f" % (names[k], num, form))
print("max residual %.2e" % worst)

print("== (b) own-block first variation of E in the vM chart of r_e ==")
Df, _, _ = Delta_fun(M0, 1, 0)
logpi = np.log(M0['pi'])
Elog = logpi @ M0['r'][1] * dx                       # E_{r_e'} log pi(w, w')
g = np.log(M0['r'][0]) - Elog + 1.0 * De            # beta_e = 1
marg = np.sum(M0['pi'], axis=1) * dx
g_ind = np.log(M0['r'][0]) - np.log(marg) + De
for k in (4, 5):
    num = cdir(lambda P: energy(build(256, P))[0], k)
    Pc = P0.astype(complex); Pc[k] += 1j * h
    dre = np.imag(build(256, Pc)['r'][0]) / h
    print("  %-22s dE complex step %+.12f  int dr g %+.12f  diff %.1e  (independent-prior version %+.6f)"
          % (names[k], num, np.sum(dre * g) * dx, abs(num - np.sum(dre * g) * dx), np.sum(dre * g_ind) * dx))

print("== (c) Fisher tensor of vM(alpha, k) ==")
for kap in [2.0, 0.5, 0.05, 0.005]:
    A1 = i1(kap) / i0(kap)
    gaa = kap * A1
    gkk = 1 - A1 / kap - A1 ** 2
    xx, ddx = setup(4096)
    f = vm(xx, 0.0, kap, ddx)
    s_a = kap * np.sin(xx); s_k = np.cos(xx) - A1
    num = np.array([[np.sum(f * s_a * s_a), np.sum(f * s_a * s_k)], [np.sum(f * s_k * s_a), np.sum(f * s_k * s_k)]]) * ddx
    print("  k=%-6g g_aa=%.6e (quad %.6e)  g_kk=%.6e (quad %.6e)  g_ak quad %.1e" % (kap, gaa, num[0, 0], gkk, num[1, 1], num[0, 1]))
# natural coordinates eta = (k cos a, k sin a), t = (cos x, sin x): Fisher = Cov_f(t)
for kap in [0.005, 0.0]:
    xx, ddx = setup(4096)
    f = vm(xx, 0.0, kap, ddx)
    t = np.vstack([np.cos(xx), np.sin(xx)])
    mean = t @ f * ddx
    cov = (t * f) @ t.T * ddx - np.outer(mean, mean)
    print("  natural chart at k=%g: Fisher = Cov(t) = [[%.4f, %.4f], [%.4f, %.4f]]" % (kap, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]))
print("g_aa = k A1(k) -> 0 as k -> 0: the polar chart (alpha, k) is singular at the uniform law,")
print("while the natural chart has Fisher matrix I/2 there; a parametric Fisher flow must use a chart")
print("with positive-definite Fisher tensor, as it2:law-cost-variation requires.")
