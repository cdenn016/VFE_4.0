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
for c in [0.0, 0.7 - 2e-3, 0.7 - 1e-3, 0.7, 0.7 + 1e-3, 0.7 + 2e-3]:
    native[c] = integrate(lambda t, y, c=c: Vc(y, c, t).ravel(), section(c).ravel(), ts)

# (1) cone estimate with v the bisector of the initial angular range [0.1-0.2, 1.0+0.35]
v = np.array([np.cos(0.6), np.sin(0.6)])
worst = np.inf
for c, sol in native.items():
    B0 = section(c)[:, 5:7]
    for k, t in enumerate(ts[1:], start=1):
        Bt = sol[k].reshape(3, 8)[:, 5:7]
        worst = min(worst, np.min(Bt @ v - np.exp(-t) * (B0 @ v)))
print("(1) lp:cone: min over c, t>0, i of v.B_i(t) - e^{-t} v.B_i(0) =", worst, "(must be >= 0)")

# (2) closed reference system
c0, c = 0.0, 0.7


def W_closed(t, Y):
    y = Y[:24].reshape(3, 8); xr = Y[24:].reshape(3, 8)
    Om = M.omega_of(xr, *records(t)[:1], records(t)[1] @ M.rot(RHO * c0).T, J)
    dy = Vc(y, c, t) - Om * M.calJ(y)
    dxr = Vc(xr, c0, t) - Om * M.calJ(xr)
    return np.concatenate([dy.ravel(), dxr.ravel()])


U0 = M.Umat(M.S_of(section(c0)))
Y0 = np.concatenate([M.rotB(section(c), U0).ravel(), M.rotB(section(c0), U0).ravel()])
closed = integrate(W_closed, Y0, ts)
err = 0.0
for k, t in enumerate(ts):
    U = M.Umat(M.S_of(native[c0][k]))
    err = max(err, np.abs(closed[k, :24] - M.rotB(native[c][k], U).ravel()).max(),
              np.abs(closed[k, 24:] - M.rotB(native[c0][k], U).ravel()).max())
anch = max(abs(M.S_of(closed[k, 24:])[1]) for k in range(len(ts)))
print("(2) lp:closed-reference: max |closed - R_U(native)| =", err, "; max |S_y(anchor)| =", anch)

