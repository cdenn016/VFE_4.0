r"""XC7-IMPL s3: conformance checks for chapter 15 (contextual coupling).

(a) finite two-context witness ctx:f:eq:witnessvelocity and the cusp ctx:f:eq:cusp
    (orthogonal Procrustes in SO(2)), exact arithmetic.
(b) continuum witness: a Fourier pseudo-spectral RK4 solver of the full nonlinear
    matrix flow P_t = -P H + lam D_x D_x P with alpha eliminated by ctx:c:eq:alpha,
    started inside the exact family P = R_phi(x) a(0,x) V_0 with a nonconstant phase,
    against ctx:c:eq:exact and the energy ctx:c:eq:exactenergy.
(c) the same solver from a generic initial field: discrete check of the energy
    identity dF_*/dt = -int ||P_t||^2 (ctx:c:statement:5 in this symmetric example).
(d) ctx:c:eq:nonbasic at lam = 0 reproduces the Cartesian Fisher flow p_t = -(p - a e_1).
"""
import numpy as np
from fractions import Fraction as Fr

Jm = np.array([[0., -1.], [1., 0.]])
def R(th): return np.cos(th) * np.eye(2) + np.sin(th) * Jm

# ---------------------------------------------------------------- (a)
X = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float)
Y = np.array([[2, 3, 0, 0], [0, 0, 1, 2]], float)
C, S = np.sum(Y * X), np.sum(Y * (Jm @ X)); rho = np.hypot(C, S)
Rs = (C * np.eye(2) + S * Jm) / rho
cstar = 0.5 * (np.sum(X ** 2) + np.sum(Y ** 2)) - rho
mu1, nu1, mu2, nu2 = X.T
gE = np.stack([(mu1 - nu1) + 2 * (mu1 - mu2), -(mu1 - nu1) + nu1 + 2 * (nu1 - nu2),
               (mu2 - nu2) + 2 * (mu2 - mu1), -(mu2 - nu2) + nu2 + 2 * (nu2 - nu1)], 1)
gc = X - Rs.T @ Y
lam = 0.7
Xdot = -gE - lam * gc
claimed = np.array([[-2 + lam, -3 + 2 * lam, 2, 2], [2, 2, -2, -3 + lam]])
print(f"(a) (C,S,rho)=({C},{S},{rho}), c*={cstar}, |Xdot - claimed| = {np.abs(Xdot - claimed).max():.1e}")
for eps in (-0.1, 0.0, 0.1):
    x_e = np.array([[1, 0], [0, 1 + eps]]); y = np.array([[1, 0], [0, -1]])
    th = np.linspace(0, 2 * np.pi, 200001)
    vals = [0.5 * np.sum((x_e - R(t) @ y) ** 2) for t in th[::50]]
    print(f"    cusp eps={eps:+.1f}: numerical min {min(vals):.8f}, formula {2 + eps + eps**2/2 - abs(eps):.8f}")

# ---------------------------------------------------------------- (b)
H = np.array([[3, -1, -2, 0], [-1, 4, 0, -2], [-2, 0, 3, -1], [0, -2, -1, 4]], float)
tau = (np.sqrt(5) - 1) / 2; kap = 5 - tau
v = np.array([1, tau, -1, -tau]); V0 = np.outer([1., 0.], v); w = v @ v
N = 64; xg = 2 * np.pi * np.arange(N) / N; kk = np.fft.fftfreq(N, 1.0 / N)
DEALIAS = {'on': True}
def dx(f, order=1):        # spectral derivative along the last axis
    return np.real(np.fft.ifft((1j * kk) ** order * np.fft.fft(f, axis=-1), axis=-1))
def filt(f):               # 2/3-rule dealiasing of the right-hand side
    if not DEALIAS['on']:
        return f
    return np.real(np.fft.ifft((np.abs(kk) <= N / 3) * np.fft.fft(f, axis=-1), axis=-1))
def rhs(P, lam):           # P shape (2,4,N)
    Px, Pxx = dx(P), dx(P, 2)
    JP = np.einsum('ab,bcn->acn', Jm, P); JPx = np.einsum('ab,bcn->acn', Jm, Px)
    nrm = np.sum(P * P, axis=(0, 1))
    alpha = -np.sum(Px * JP, axis=(0, 1)) / nrm
    alx = dx(alpha)
    DD = Pxx + alx * JP + 2 * alpha * JPx - alpha ** 2 * P
    return filt(-np.einsum('acn,cd->adn', P, H) + lam * DD)
def rk4(P, lam, dt, steps):
    for _ in range(steps):
        k1 = rhs(P, lam); k2 = rhs(P + dt / 2 * k1, lam); k3 = rhs(P + dt / 2 * k2, lam); k4 = rhs(P + dt * k3, lam)
        P = P + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return P
def energy(P, lam):
    Px = dx(P); JP = np.einsum('ab,bcn->acn', Jm, P)
    nrm = np.sum(P * P, axis=(0, 1))
    dens = 2 + 0.5 * np.sum(P * np.einsum('acn,cd->adn', P, H), axis=(0, 1)) \
        + lam / 2 * (np.sum(Px * Px, axis=(0, 1)) - np.sum(Px * JP, axis=(0, 1)) ** 2 / nrm)
    return dens.mean()
lam, n, a0, a1 = 0.5, 2, 1.0, 0.6
phase = 0.8 * np.sin(xg) + 0.3 * np.cos(2 * xg)
def exact(t):
    a = a0 * np.exp(-kap * t) + a1 * np.exp(-(kap + lam * n ** 2) * t) * np.cos(n * xg)
    return np.stack([R(phase[i]) @ (a[i] * V0) for i in range(N)], -1)
T, dt = 1.0, 1e-3
DEALIAS['on'] = False
P_raw = rk4(exact(0.0), lam, dt, int(round(0.3 / dt)))
print(f"(b) without dealiasing (N={N}): rel error at t=0.3 = {np.abs(P_raw - exact(0.3)).max() / np.abs(exact(0.3)).max():.2e}"
      "  [aliasing instability in the undamped gauge direction]")
# off-family growth without dealiasing (N = 64, dt = 1e-3), and resolution dependence (dt = 2.5e-4, t = 0.3)
def a_of(P):
    return np.array([np.sum(P[:, :, i] * (R(phase[i]) @ V0)) for i in range(N)]) / w
def off_family(P):
    an = a_of(P); return np.abs(P - np.stack([R(phase[i]) @ (an[i] * V0) for i in range(N)], -1)).max()
Pt_ = exact(0.0); trace = []
for k in range(1, 8):
    Pt_ = rk4(Pt_, lam, 1e-3, 25); trace.append(off_family(Pt_))
print("    off-family component, t = 0.025..0.175 (no dealiasing):", ["%.1e" % v for v in trace])
def run_res(Nr, dealias, T=0.3, dtr=2.5e-4):
    xr = 2 * np.pi * np.arange(Nr) / Nr; kr = np.fft.fftfreq(Nr, 1.0 / Nr); mask = (np.abs(kr) <= Nr / 3) if dealias else np.ones(Nr, bool)
    def d(f, o=1): return np.real(np.fft.ifft((1j * kr) ** o * np.fft.fft(f, axis=-1), axis=-1))
    def fl(f): return np.real(np.fft.ifft(mask * np.fft.fft(f, axis=-1), axis=-1))
    def rr(P):
        Px, Pxx = d(P), d(P, 2); JP = np.einsum('ab,bcn->acn', Jm, P); JPx = np.einsum('ab,bcn->acn', Jm, Px)
        al = -np.sum(Px * JP, axis=(0, 1)) / np.sum(P * P, axis=(0, 1))
        return fl(-np.einsum('acn,cd->adn', P, H) + lam * (Pxx + d(al) * JP + 2 * al * JPx - al ** 2 * P))
    ph = 0.8 * np.sin(xr) + 0.3 * np.cos(2 * xr)
    def ex(t):
        a = a0 * np.exp(-kap * t) + a1 * np.exp(-(kap + lam * n ** 2) * t) * np.cos(n * xr)
        return np.stack([R(ph[i]) @ (a[i] * V0) for i in range(Nr)], -1)
    P = ex(0.0)
    for _ in range(int(round(T / dtr))):
        k1 = rr(P); k2 = rr(P + dtr / 2 * k1); k3 = rr(P + dtr / 2 * k2); k4 = rr(P + dtr * k3); P = P + dtr / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    E = ex(T); return np.abs(P - E).max() / np.abs(E).max()
print("    resolution dependence at t=0.3, dt=2.5e-4 (no dealias / 2/3 dealias):",
      [(Nr, "%.1e" % run_res(Nr, False), "%.1e" % run_res(Nr, True)) for Nr in (32, 64, 128)])
DEALIAS['on'] = True
P = rk4(exact(0.0), lam, dt, int(round(T / dt)))
rel = np.abs(P - exact(T)).max() / np.abs(exact(T)).max()
Fex = 2 + w * kap * a0 ** 2 / 2 * np.exp(-2 * kap * T) + w * (kap + lam * n ** 2) * a1 ** 2 / 4 * np.exp(-2 * (kap + lam * n ** 2) * T)
print(f"    with 2/3 dealiasing, N={N}, dt={dt}: max rel error vs ctx:c:eq:exact at T={T}: {rel:.2e}; "
      f"F_*(T) numerical {energy(P, lam):.12f} vs exact {Fex:.12f}")
P2 = rk4(exact(0.0), lam, 2 * dt, int(round(T / (2 * dt))))
print(f"    halving dt changes P by {np.abs(P2 - P).max():.2e} (RK4 time error)")

# ---------------------------------------------------------------- (c)
rng = np.random.default_rng(3)
P0 = np.zeros((2, 4, N))
for a in range(2):
    for c in range(4):
        P0[a, c] = rng.normal() + 0.3 * rng.normal() * np.cos(xg + rng.uniform(0, 6)) + 0.2 * rng.normal() * np.sin(2 * xg)
P0 += 1.5 * V0[:, :, None]                          # keep ||P|| away from the singular orbit P = 0
dt = 5e-4
Ea = energy(P0, lam); Pt = rhs(P0, lam)
P1 = rk4(P0, lam, dt, 1); Eb = energy(P1, lam)
dEdt_fd = (Eb - Ea) / dt; diss = -np.mean(np.sum(Pt * Pt, axis=(0, 1)))
P_half = rk4(P0, lam, dt / 2, 1); Pt_half = rhs(P_half, lam)
diss_mid = -np.mean(np.sum(Pt_half * Pt_half, axis=(0, 1)))
print(f"(c) generic field: (F(dt)-F(0))/dt = {dEdt_fd:.6f}; -int||P_t||^2 at t=0: {diss:.6f}, at dt/2: {diss_mid:.6f}; "
      f"min ||P||_F over x: {np.sqrt(np.sum(P0*P0,axis=(0,1))).min():.3f}")
# horizontality: <P_t, JP> = 0 pointwise for symmetric H
JP0 = np.einsum('ab,bcn->acn', Jm, P0)
print(f"    max |<P_t, JP>_F| = {np.abs(np.sum(Pt * JP0, axis=(0, 1))).max():.2e} (horizontal vector field)")

# ---------------------------------------------------------------- (d)
a_, eta = 1.3, 0.4
for (u, r) in [(0.7, 0.2), (2.0, -1.1)]:
    ph = r + eta * u
    ut = -u + a_ * np.cos(ph)
    rt = -eta * ut - a_ / u * np.sin(ph)
    # p = u (cos ph, sin ph); dp/dt = u_t e(ph) + u (r_t + eta u_t) e(ph)^perp
    pt = ut * np.array([np.cos(ph), np.sin(ph)]) + u * (rt + eta * ut) * np.array([-np.sin(ph), np.cos(ph)])
    p = u * np.array([np.cos(ph), np.sin(ph)])
    print(f"(d) ctx:c:eq:nonbasic at (u,r)=({u},{r}): |p_t + (p - a e1)| = {np.abs(pt + (p - np.array([a_, 0.]))).max():.1e}")
