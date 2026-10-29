r"""D18-LP check 3: lp:cone, lp:closed-reference, lp:jet-flow, lp:causal-flow,
lp:frozen-local-initial-frame and lp:causal-phase-jets by direct integration.

1-dimensional context c. The native field gets an explicit c-dependence through a record
embedding Lam_i(c) = R(rho c) Lam_i (to exercise the explicit partials W_r, W_rs).
One evidence append at t1 (H, Lam jump; x continuous). Generic J=0.3, moderate amplitudes:
the identities checked are exact for every J, not only J = eps^3.
"""
import numpy as np
from scipy.integrate import solve_ivp
import lp_model as M

J = 0.3
RHO = 0.4
T1, TEND = 0.7, 2.0
H0 = np.array([0.4, 0.4, -0.4]); H1 = H0 + np.array([0.3, -0.2, 0.1])
L0 = np.array([[0.2, 0.1], [-0.1, 0.25], [0.15, -0.2]])
L1 = L0 + np.array([[0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])


def records(t):
    return (H0, L0) if t < T1 else (H1, L1)


def Vc(x, c, t):
    H, L = records(t)
    Lam = L @ M.rot(RHO * c).T
    return M.V(x, H, Lam, J)


def section(c):
    x = np.zeros((3, 8))
    x[:, 0] = np.array([0.5, 0.2, -0.3]) + 0.3 * np.sin(c + np.arange(3))
    a, chi = 0.6 + 0.2 * c, 0.5 * c
    x[0, 1:3] = a * np.array([np.cos(chi), np.sin(chi)])
    x[1, 1:3] = [0.4 + 0.05 * c, 0.1 * c ** 2]
    x[2, 1:3] = [0.3, -0.2 + 0.1 * np.cos(c)]
    x[:, 3:5] = 0.1 * np.stack([np.cos(c + np.arange(3)), np.sin(2 * c + np.arange(3))], axis=1)
    th = np.array([0.1, 0.6, 1.0]) + np.array([0.3, -0.2, 0.5]) * c
    amp = np.array([0.8, 0.6, 0.9]) + 0.1 * c * np.array([1, -1, 0.5])
    x[:, 5:7] = amp[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
    x[:, 7] = np.array([0.2, -0.3, 0.1]) + 0.1 * c
    return x


def integrate(rhs, y0, ts, rtol=1e-12, atol=1e-13):
    """Integrate across the record append at T1 (field discontinuous, state continuous)."""
    out = []
    seg1 = solve_ivp(rhs, (0, T1), y0, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    seg2 = solve_ivp(rhs, (T1, TEND), seg1.y[:, -1], method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    for t in ts:
        out.append(seg1.sol(t) if t <= T1 else seg2.sol(t))
    return np.array(out)


ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
native = {}
for c in [0.0, 0.7 - 1e-3, 0.7, 0.7 + 1e-3]:
    native[c] = integrate(lambda t, y, c=c: Vc(y, c, t).ravel(), section(c).ravel(), ts)

c0, c = 0.0, 0.7
U0 = M.Umat(M.S_of(section(c0)))
# (3) jet flow with reference held fixed under c-differentiation
h = 1e-4


def jet_rhs(t, Y):
    y, yc, ycc, xr = Y[:24], Y[24:48], Y[48:72], Y[72:]
    H, L = records(t)
    Om = M.omega_of(xr.reshape(3, 8), H, L @ M.rot(RHO * c0).T, J)
    Wf = lambda yy, cc: (Vc(yy.reshape(3, 8), cc, t) - Om * M.calJ(yy.reshape(3, 8))).ravel()
    W0 = Wf(y, c)
    dy = W0
    Wp, Wm = Wf(y + h * yc, c + h), Wf(y - h * yc, c - h)
    dyc = (Wp - Wm) / (2 * h)              # D_y W[y_c] + W_c
    phi2 = (Wp - 2 * W0 + Wm) / h ** 2     # D^2W[y_c,y_c] + 2 D W_c[y_c] + W_cc
    dlin = (Wf(y + h * ycc, c) - Wf(y - h * ycc, c)) / (2 * h)
    dycc = dlin + phi2
    dxr = (Vc(xr.reshape(3, 8), c0, t) - Om * M.calJ(xr.reshape(3, 8))).ravel()
    return np.concatenate([dy, dyc, dycc, dxr])


d = 1e-3
x0c = section(c); xp, xm = section(c + d), section(c - d)
Y0 = np.concatenate([M.rotB(x0c, U0).ravel(),
                     M.rotB((xp - xm) / (2 * d), U0).ravel(),
                     M.rotB((xp - 2 * x0c + xm) / d ** 2, U0).ravel(),
                     M.rotB(section(c0), U0).ravel()])
jets = integrate(jet_rhs, Y0, ts, rtol=1e-9, atol=1e-11)
e1 = e2 = 0.0
for k, t in enumerate(ts):
    U = M.Umat(M.S_of(native[c0][k]))
    yp = M.rotB(native[c + d][k], U).ravel(); ym = M.rotB(native[c - d][k], U).ravel(); y0 = M.rotB(native[c][k], U).ravel()
    fd1 = (yp - ym) / (2 * d); fd2 = (yp - 2 * y0 + ym) / d ** 2
    e1 = max(e1, np.abs(jets[k, 24:48] - fd1).max()); e2 = max(e2, np.abs(jets[k, 48:72] - fd2).max())
print("(3) lp:jet-flow: max |y_c(jet) - FD| =", e1, "; max |y_cc(jet) - FD| =", e2, "(FD step 1e-3)")

