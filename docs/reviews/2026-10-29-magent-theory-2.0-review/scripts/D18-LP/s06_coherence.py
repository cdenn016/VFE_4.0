r"""D18-LP check 6: is initial collective-phase coherence necessary, and does a coherent
preparation stay coherent?

For the causal parent  xtilde(c,t) = R_{-alpha0(c)} x(c,t)  (lp:frozen-local-initial-frame),
check 5 showed  hbar - h = -2 (X alpha0) b_t(X) + (X alpha0)^2 I_t  with
b_t(X) = sum_i (X arg B_i(c,t)) I(|B_i|), I_t = sum_i I(|B_i|), while the connection is unchanged.

(a) A NON-coherent preparation (X alpha0 = 2 mu != 0) built with amplitude gradients so that
    b_0 = mu I_0: the causal parent preserves connection AND tensor at t = 0, at every c of the
    patch; at t > 0 the defect is evaluated along the exact native flow (fixed record episode).
(b) A coherent noncollinear preparation (context-constant B, context-dependent marked A_1):
    d_c arg S(c,T) at T>0 (companion c14:183 says it "need not remain context-constant").
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0e, i1e
import lp_model as M

Ifun = lambda r: r * i1e(r) / i0e(r)
J = 0.3
H = np.array([0.4, 0.4, -0.4]); Lam = np.array([[0.2, 0.1], [-0.1, 0.25], [0.15, -0.2]])
th0 = np.array([0.1, 0.6, 1.0]); amp0 = np.array([0.8, 0.6, 0.9])
E = np.stack([np.cos(th0), np.sin(th0)])  # 2x3
S0 = E @ amp0
Epinv = np.linalg.pinv(E)
mu = 0.4


def section_noncoh(c):
    x = np.zeros((3, 8))
    x[:, 0] = [0.5 + 0.2 * c, 0.2, -0.3]
    x[0, 1:3] = (0.6 + 0.2 * c) * np.array([np.cos(0.5 * c), np.sin(0.5 * c)])
    x[1, 1:3] = [0.4, 0.0]; x[2, 1:3] = [0.3, -0.2]
    amp = amp0 + Epinv @ (M.rot(mu * c) @ S0 - S0)   # sum_i amp_i e_{th0_i} = R(mu c) S0
    th = th0 + mu * c                                  # common phase gradient mu
    x[:, 5:7] = amp[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
    x[:, 7] = [0.2, -0.3, 0.1]
    return x


def evolve(x0, ts):
    sol = solve_ivp(lambda t, y: M.V(y, H, Lam, J).ravel(), (0, ts[-1]), x0.ravel(), method="DOP853",
                    rtol=1e-12, atol=1e-13, dense_output=True)
    return [sol.sol(t).reshape(3, 8) for t in ts]


ang = lambda B: np.arctan2(B[:, 1], B[:, 0])
ts = [0.0, 0.25, 0.5, 1.0, 2.0]
hfd = 1e-4
for c in [0.0, 0.3]:
    xp, x0, xm = evolve(section_noncoh(c + hfd), ts), evolve(section_noncoh(c), ts), evolve(section_noncoh(c - hfd), ts)
    alpha0 = lambda cc: np.arctan2(*M.S_of(section_noncoh(cc))[::-1])
    Xa0 = (alpha0(c + hfd) - alpha0(c - hfd)) / (2 * hfd)
    print(f"(a) c={c}: X alpha0 = {Xa0:.6f} (2 mu = {2*mu})")
    for k, t in enumerate(ts):
        B = x0[k][:, 5:7]
        dth = (np.unwrap(np.stack([ang(xm[k][:, 5:7]), ang(xp[k][:, 5:7])]), axis=0)[1]
               - np.unwrap(np.stack([ang(xm[k][:, 5:7]), ang(xp[k][:, 5:7])]), axis=0)[0]) / (2 * hfd)
        I = np.array([Ifun(np.hypot(*Bi)) for Bi in B])
        b = float(dth @ I); It = float(I.sum())
        defect = -2 * Xa0 * b + Xa0 ** 2 * It
        print(f"    t={t:4.2f}: b_t/I_t = {b/It:.6f}, tensor defect hbar-h = {defect:+.3e}")

# (b) coherent noncollinear preparation; does arg S(c,T) stay context-constant?
def section_coh(c, eps=None):
    x = np.zeros((3, 8))
    x[:, 0] = [0.5, 0.2, -0.3]
    x[0, 1:3] = 0.6 * np.array([np.cos(c), np.sin(c)])   # A_1 = a e_chi, chi = c
    x[1, 1:3] = [0.4, 0.0]; x[2, 1:3] = [0.3, -0.2]
    x[:, 5:7] = amp0[:, None] * np.stack([np.cos(th0), np.sin(th0)], axis=1)  # context-constant, noncollinear
    x[:, 7] = [0.2, -0.3, 0.1]
    return x


for T in [0.5, 1.0, 2.0]:
    vals = []
    for c in [1.0 - hfd, 1.0 + hfd]:
        xT = evolve(section_coh(c), [0.0, T])[-1]
        vals.append(np.arctan2(*M.S_of(xT)[::-1]))
    print(f"(b) coherent start, J=0.3, at chi=1: d/dchi arg S(chi,T={T}) = {(vals[1]-vals[0])/(2*hfd):+.3e}")
# same with the chapter's small-parameter scaling: eps=0.05, J=eps^3, records eps*o, B = delta b
eps, delta = 0.05, 0.3
Jsmall = eps ** 3
Hs = eps * np.array([1.0, 1.0, -1.0]); Ls = eps * np.stack([np.cos([0.3, 1.2, 2.0]), np.sin([0.3, 1.2, 2.0])], axis=1)
def section_small(c):
    x = np.zeros((3, 8))
    x[:, 0] = eps * np.array([4.0, 0.5, -1.0])
    x[0, 1:3] = 0.08 * np.array([np.cos(c), np.sin(c)]); x[1, 1:3] = [0.06, 0.0]; x[2, 1:3] = [0.04, 0.03]
    x[:, 5:7] = delta * amp0[:, None] * np.stack([np.cos(th0), np.sin(th0)], axis=1) / 10
    return x
for T in [1.0, 4.0]:
    vals = []
    for c in [1.0 - 1e-3, 1.0 + 1e-3]:
        sol = solve_ivp(lambda t, y: M.V(y, Hs, Ls, Jsmall).ravel(), (0, T), section_small(c).ravel(), method="DOP853", rtol=1e-13, atol=1e-15)
        vals.append(np.arctan2(*M.S_of(sol.y[:, -1])[::-1]))
    print(f"(b) chapter scaling eps={eps}, at chi=1: d/dchi arg S(chi,T={T}) = {(vals[1]-vals[0])/2e-3:+.3e}")
