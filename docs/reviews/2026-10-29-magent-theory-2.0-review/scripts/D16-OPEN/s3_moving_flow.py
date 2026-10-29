r"""D16-OPEN s3: direct check of the inherited moving-representative field (focus question 4).

Three agents on a discretized bit-circle (M angles), correlated non-von-Mises laws (so the anchor
phases move), selective channels, all-source Gibbs rows, uniform references/evaluators.
Block I = {0,1}, singleton {2}. Anchor = the single context.

The fine trajectory is integrated directly from open:full-field (RK4).  The representative is
q~_i(t) = tau_{a_i(t)} q_i(t), a_i = phi_I - vartheta_i, vartheta_i = arg E_{q_i} e^{i theta},
phi_I = mean over I; rotation by a non-grid angle is done spectrally (exact for the band-limited
part; the laws are exp of trigonometric polynomials, so the aliasing error is negligible).

Checks at a time t0:
 [1] d/dt q~_i (centered difference of the rotated fine trajectory) vs
     V_i(x~) - (omegabar_I - omega_i) d_theta q~_i                       (open:moving-flow)
     and the size of the error if the moving term is omitted.
 [2] same for the model sector with the belief-anchored angles.
 [3] equivariance V_i(x~) = tau_{a_i} V_i(x); omega_i invariance.
 [4] Fisher orthogonality of V to each single-agent rotation direction (open:quotient-gradient).
 [5] block response kernel (Z_0, Z_1, Theta_0+Theta_1) equal for x and x~ (open:response descent);
     singleton first moment changes.
"""
import numpy as np

rng = np.random.default_rng(12)
N, M, T = 3, 128, 0.5
th = 2 * np.pi * np.arange(M) / M
w = 1.0 / (2 * M)
Zv = np.array([1.0, -1.0])
modes = np.fft.fftfreq(M, d=1.0 / M)          # integer wavenumbers


def law(scale=0.8, harm=3):
    lq = np.zeros((2, M))
    for zi in range(2):
        for m in range(1, harm + 1):
            a, b = rng.normal(0, scale / m, 2)
            lq[zi] += a * np.cos(m * th) + b * np.sin(m * th)
    lq += rng.normal(0, 0.5) * Zv[:, None]
    q = np.exp(lq)
    return q / (q.sum() * w)


def rot(f, a):           # (tau_a f)(z, theta) = f(z, theta - a), spectral
    F = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(F * np.exp(-1j * modes * a), axis=-1))


def dth(f):
    F = np.fft.fft(f, axis=-1)
    return np.real(np.fft.ifft(F * (1j * modes), axis=-1))


mu = lambda q: q.mean(-1)


def D(qi, qj):
    return (w * qi * (np.log(qi) - np.log(mu(qj))[:, None])).sum()


def field(sec):
    Dm = np.array([[D(sec[i], sec[j]) for j in range(N)] for i in range(N)])
    B = np.exp(-(Dm - Dm.min(1, keepdims=True)) / T); B /= B.sum(1, keepdims=True)
    m = np.array([mu(s) for s in sec])
    V = np.zeros_like(sec)
    for i in range(N):
        h = 2 * np.log(sec[i]) - (B[i][:, None] * np.log(m)).sum(0)[:, None] \
            - ((B[:, i][:, None] * m).sum(0) / m[i])[:, None]
        V[i] = -sec[i] * (h - (w * sec[i] * h).sum())
    return V


def F(state):
    Q, S = state
    return np.array([field(Q), field(S)])


def rk4(state, dt):
    k1 = F(state); k2 = F(state + dt / 2 * k1); k3 = F(state + dt / 2 * k2); k4 = F(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


u1 = lambda q: (w * q * np.exp(1j * th)).sum()

Q0 = np.array([law() for _ in range(N)]); S0 = np.array([law() for _ in range(N)])
state = np.array([Q0, S0])
t0, dt, de = 0.3, 2e-3, 1e-4
for _ in range(int(round((t0 - de) / dt))):
    state = rk4(state, dt)
# remaining partial step to reach t0 - de exactly
rem = (t0 - de) - dt * int(round((t0 - de) / dt))
if abs(rem) > 1e-15:
    state = rk4(state, rem)
sm = state.copy(); s0 = rk4(sm, de); sp = rk4(s0, de)


def angles(st):
    th_ = np.array([np.angle(u1(st[0, i])) for i in range(N)])
    return th_


def rep(st, thref):
    # anchored rotation for partition {{0,1},{2}}; unwrap relative to reference angles
    v = angles(st)
    v = thref + np.angle(np.exp(1j * (v - thref)))
    phiI = (v[0] + v[1]) / 2
    a = np.array([phiI - v[0], phiI - v[1], 0.0])
    Qt = np.array([rot(st[0, i], a[i]) for i in range(N)])
    St = np.array([rot(st[1, i], a[i]) for i in range(N)])
    return np.array([Qt, St]), a


thref = angles(s0)
rm, am = rep(sm, thref); r0, a0 = rep(s0, thref); rp, ap = rep(sp, thref)
lhs = (rp - rm) / (2 * de)
V0 = F(s0)
omega = np.array([np.imag(u1(V0[0, i]) / u1(s0[0, i])) for i in range(N)])
obar = np.array([(omega[0] + omega[1]) / 2] * 2 + [omega[2]])
coef = obar - omega                           # adot_i
Vt = F(r0)
rhs = Vt.copy()
for sec in range(2):
    for i in range(N):
        rhs[sec, i] = Vt[sec, i] - coef[i] * dth(r0[sec, i])
out = []
out.append("anchor phases at t0: %s ; omega_i = %s ; adot = %s" %
           (np.round(angles(s0), 4), np.array2string(omega, precision=6), np.array2string(coef, precision=6)))
for sec, name in ((0, "belief"), (1, "model")):
    err_full = np.abs(lhs[sec] - rhs[sec]).max()
    err_omit = np.abs(lhs[sec] - Vt[sec]).max()
    out.append("[%d] %s: max|d/dt q~ - (V(x~) - adot d_theta q~)| = %.3e ; omitting the moving term: %.3e"
               % (sec + 1, name, err_full, err_omit))
# [3] equivariance and omega invariance
eqv = max(np.abs(Vt[s, i] - rot(V0[s, i], a0[i])).max() for s in range(2) for i in range(N))
om_t = np.array([np.imag(u1(Vt[0, i]) / u1(r0[0, i])) for i in range(N)])
out.append("[3] max|V(x~) - tau_a V(x)| = %.3e ; max|omega(x~) - omega(x)| = %.3e" % (eqv, np.abs(om_t - omega).max()))
# slice condition and adot from finite differences
pha = np.array([np.angle(u1(r0[0, i])) for i in range(N)])
out.append("    slice: representative anchor phases %s (agents 0,1 equal); finite-difference adot = %s"
           % (np.round(pha, 10), np.array2string((ap - am) / (2 * de), precision=6)))
# [4] orthogonality: g(V, J_i) = int V_i d_theta log q_i dnu for each single law
orth = max(abs((w * V0[s, i] * dth(np.log(s0[s, i]))).sum()) for s in range(2) for i in range(N))
scale = max(np.sqrt((w * V0[s, i] ** 2 / s0[s, i]).sum()) for s in range(2) for i in range(N))
out.append("[4] max over single-law rotation directions |g(V, J)| = %.3e  (|V|_g scale %.3e)" % (orth, scale))
# [5] block response kernel
def block_kernel(st):
    # joint law of (Z_0, Z_1, Theta_0 + Theta_1 mod 2pi) as an array [z0, z1, k]
    out = np.zeros((2, 2, M))
    for z0 in range(2):
        for z1 in range(2):
            f0 = st[0, 0, z0] / M; f1 = st[0, 1, z1] / M          # mass per grid point (times 1/2 each bit)
            conv = np.real(np.fft.ifft(np.fft.fft(f0) * np.fft.fft(f1)))
            out[z0, z1] = 0.25 * conv
    return out
Kx, Kt = block_kernel(s0), block_kernel(r0)
m_single = abs(u1(s0[0, 0]) - u1(r0[0, 0]))
out.append("[5] block kernel max|K(x) - K(x~)| = %.3e (total mass %.6f); singleton first-moment change |u_0(x)-u_0(x~)| = %.3e"
           % (np.abs(Kx - Kt).max(), Kx.sum(), m_single))
print("\n".join(out))
