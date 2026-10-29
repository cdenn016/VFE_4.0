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
for c in [0.0, 0.7]:
    native[c] = integrate(lambda t, y, c=c: Vc(y, c, t).ravel(), section(c).ravel(), ts)

c0, c = 0.0, 0.7
h = 1e-4
d = 1e-3
# (4) causal accumulator from t_* = 0, local slice at c only


def causal_rhs(t, Y):
    xb = Y[:24].reshape(3, 8)
    H, L = records(t)
    om = M.omega_of(xb, H, L @ M.rot(RHO * c).T, J)
    dxb = Vc(xb, c, t) - om * M.calJ(xb)
    return np.concatenate([dxb.ravel(), [om]])


xb0 = M.rotB(section(c), M.Umat(M.S_of(section(c))))
cs = integrate(causal_rhs, np.concatenate([xb0.ravel(), [0.0]]), ts)
Uloc0 = M.Umat(M.S_of(section(c)))
err4 = 0.0; slice_err = 0.0; refdiff = 0.0
for k, t in enumerate(ts):
    xt = M.rotB(cs[k, :24], M.rot(cs[k, 24]))
    err4 = max(err4, np.abs(xt.ravel() - M.rotB(native[c][k], Uloc0).ravel()).max())
    slice_err = max(slice_err, abs(M.S_of(cs[k, :24])[1]))
    U = M.Umat(M.S_of(native[c0][k]))
    refdiff = max(refdiff, np.abs(xt.ravel() - M.rotB(native[c][k], U).ravel()).max())
print("(4) lp:frozen-local-initial-frame: max |R_theta xbar - R_{U(S(x(c,0)))} x(c,t)| =", err4,
      "; max |S_y(slice)| =", slice_err)
print("    (non-coherent preparation) max |causal parent - reference parent| =", refdiff, "(differ, as expected)")

# (5) causal phase jet: theta_c(t) from FD of theta(c,t) vs. the jet equation of lp:causal-phase-jets


def causal_jet_rhs(t, Y, cc=c):
    xb, xbc, th, thc = Y[:24], Y[24:48], Y[48], Y[49]
    H, L = records(t)

    def Wl(z, ccc):
        z3 = z.reshape(3, 8)
        om = M.omega_of(z3, H, L @ M.rot(RHO * ccc).T, J)
        return (Vc(z3, ccc, t) - om * M.calJ(z3)).ravel(), om
    W0, om0 = Wl(xb, cc)
    Wp, omp = Wl(xb + h * xbc, cc + h)
    Wm, omm = Wl(xb - h * xbc, cc - h)
    return np.concatenate([W0, (Wp - Wm) / (2 * h), [om0, (omp - omm) / (2 * h)]])


def slice_section(cc):
    xx = section(cc)
    return M.rotB(xx, M.Umat(M.S_of(xx)))


xbc0 = (slice_section(c + d) - slice_section(c - d)) / (2 * d)
cj = integrate(causal_jet_rhs, np.concatenate([xb0.ravel(), xbc0.ravel(), [0.0, 0.0]]), ts, rtol=1e-9, atol=1e-11)


def causal_rhs_c(t, Y, cc):
    xb = Y[:24].reshape(3, 8)
    H, L = records(t)
    om = M.omega_of(xb, H, L @ M.rot(RHO * cc).T, J)
    return np.concatenate([(Vc(xb, cc, t) - om * M.calJ(xb)).ravel(), [om]])


solp = integrate(lambda t, Y: causal_rhs_c(t, Y, c + d), np.concatenate([slice_section(c + d).ravel(), [0.0]]), ts)
solm = integrate(lambda t, Y: causal_rhs_c(t, Y, c - d), np.concatenate([slice_section(c - d).ravel(), [0.0]]), ts)
e5 = max(abs(cj[k, 49] - (solp[k, 24] - solm[k, 24]) / (2 * d)) for k in range(len(ts)))
print("(5) lp:causal-phase-jets (first order): max |theta_c(jet) - FD| =", e5,
      "; theta_c(t=2) =", cj[-1, 49])
# alpha(c,t) - alpha_0(c) = theta (lp:initial-phase-datum)
ang = lambda S: np.arctan2(S[1], S[0])
e6 = max(abs(np.angle(np.exp(1j * (ang(M.S_of(native[c][k])) - ang(M.S_of(section(c))) - cs[k, 24])))) for k in range(len(ts)))
print("(6) lp:initial-phase-datum: max |alpha(c,t)-alpha_0(c)-theta(c,t)| =", e6)
