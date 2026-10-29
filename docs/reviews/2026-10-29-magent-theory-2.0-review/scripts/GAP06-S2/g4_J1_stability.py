r"""GAP06-S2 focus question 3: stability of the common reference of chapters/09_public_compression.tex
at |J| = 1 (2026-10-29 D13-ERAPUB-08; companion c09:11).  Finite-alphabet instance of the chart flow
eq:public-phase-chart-flow: A = B = uniform on 4 points, phi = psi = (1,1,-1,-1), N = 3 agents,
identity channels, uniform nonself priors, Gibbs-reduced reciprocal potential, temperatures
T_q = 0.7, T_s = 1.3, unit Fisher mobility (Fisher-Rao flow in log coordinates).
Reported along one trajectory from a heterogeneous initial state:
  ||delta|| (agent differences, chart sup norm)  -> exponential, rate ~ kappa_N = 2N/(N-1) = 3 at |J| = 1,
  ||u_perp|| (mean component orthogonal to phi)   -> exponential, rate ~ 1,
  a^2 + b^2 (mean feature amplitudes)              -> algebraic, (a^2+b^2) t -> 3 (center direction
  s' = -s + tanh s = -s^3/3 + ..., s^2 ~ 3/(2t), a = b = s).
Also J = 0.9 (exponential, rate 0.1) and J = 1.1 (leaves the reference).  Numerical illustration of the
analytic cascade proof in the report, not a proof.  Single-threaded, a few seconds.
"""
import numpy as np
from scipy.integrate import solve_ivp

K = 4
A = np.full(K, 1 / K); B = np.full(K, 1 / K)
phi = np.array([1.0, 1.0, -1.0, -1.0]); psi = phi.copy()
N = 3; Tq, Ts = 0.7, 1.3


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def field(t, y, J):
    L = y.reshape(N, 2, K)
    q = np.exp(L[:, 0]); q /= q.sum(1, keepdims=True)
    s = np.exp(L[:, 1]); s /= s.sum(1, keepdims=True)
    out = np.zeros_like(L)
    for sec, (P, ref, T) in enumerate(((q, A, Tq), (s, B, Ts))):
        other = s if sec == 0 else q
        feat_self = phi if sec == 0 else psi
        feat_other = psi if sec == 0 else phi
        D = np.array([[kl(P[i], P[j]) if i != j else np.inf for j in range(N)] for i in range(N)])
        W = np.exp(-(D - D[np.isfinite(D)].min()) / T); np.fill_diagonal(W, 0.0)
        beta = W / W.sum(1, keepdims=True)
        for i in range(N):
            h = np.log(P[i] / ref) - J * feat_self * float(other[i] @ feat_other)
            h = h + sum(beta[i, j] * (np.log(P[i]) - np.log(P[j])) for j in range(N) if j != i)
            h = h - sum(beta[l, i] * P[l] / P[i] for l in range(N) if l != i)
            out[i, sec] = -(h - P[i] @ h)
    return out.ravel()


def chart(y):
    L = y.reshape(N, 2, K)
    q = np.exp(L[:, 0]); q /= q.sum(1, keepdims=True)
    s = np.exp(L[:, 1]); s /= s.sum(1, keepdims=True)
    u = np.log(q / A); u -= (u * A).sum(1, keepdims=True)
    v = np.log(s / B); v -= (v * B).sum(1, keepdims=True)
    return u, v


rng = np.random.default_rng(3)
base_u = 0.08 * phi + 0.05 * np.array([1.0, -1.0, 0.5, -0.5])
base_v = 0.06 * psi + 0.04 * np.array([-0.5, 0.5, 1.0, -1.0])
y0 = []
for i in range(N):
    du = 0.03 * rng.normal(size=K); dv = 0.03 * rng.normal(size=K)
    lu = np.log(A) + base_u + du; lv = np.log(B) + base_v + dv
    y0.append(np.stack([lu, lv]))
y0 = np.array(y0).ravel()

for J in (1.0, 0.9, 1.1):
    tt = [0, 1, 2, 4, 8, 50, 100, 200, 400]
    sol = solve_ivp(field, (0, tt[-1]), y0, args=(J,), method='DOP853', t_eval=tt,
                    rtol=1e-13, atol=1e-15)
    print("J = %.2f  (status %d, %d field evaluations)" % (J, sol.status, sol.nfev))
    print("   t     ||delta||      ||u_perp||+||v_perp||   a          b          (a^2+b^2) t   center-ODE prediction")
    s8 = None
    for k, t in enumerate(sol.t):
        u, v = chart(sol.y[:, k])
        ub, vb = u.mean(0), v.mean(0)
        dl = max(np.abs(u - ub).max(), np.abs(v - vb).max())
        aa = float(ub @ (phi * A)); bb = float(vb @ (psi * B))
        perp = np.abs(ub - aa * phi).max() + np.abs(vb - bb * psi).max()
        if t == 8:
            s8 = 0.5 * (aa + bb)
        pred = ''
        if J == 1.0 and s8 is not None and t > 8:
            # s' = -s + tanh s ~ -s^3/3 from t = 8: s^2 = 1/(1/s8^2 + 2(t-8)/3); V = 2 s^2
            pred = "%.5f" % (2 * t / (1 / s8**2 + 2 * (t - 8) / 3))
        print("  %5g   %.3e      %.3e             %+.5f   %+.5f   %.5f      %s" % (t, dl, perp, aa, bb, (aa**2 + bb**2) * t, pred))
    print()

# reduced feature system at J = 1: V = a^2+b^2 strictly decreasing, V' <= -V^2/5 on the unit ball
print("reduced feature system a' = -a + J tanh b, b' = -b + J tanh a at J = 1:")
g = np.linspace(-1, 1, 801)
Aa, Bb = np.meshgrid(g, g)
Vd = 2 * (-Aa**2 - Bb**2 + Aa * np.tanh(Bb) + Bb * np.tanh(Aa))
V = Aa**2 + Bb**2
mask = V > 0
print("  max of V'/V^2 over the grid on [-1,1]^2 minus origin: %.5f (claimed bound -1/5 = -0.2)"
      % float((Vd[mask] / V[mask]**2).max()))
Vdm = 2 * (-Aa**2 - Bb**2 - Aa * np.tanh(Bb) - Bb * np.tanh(Aa))
print("  J = -1: max of V'/V^2: %.5f" % float((Vdm[mask] / V[mask]**2).max()))
