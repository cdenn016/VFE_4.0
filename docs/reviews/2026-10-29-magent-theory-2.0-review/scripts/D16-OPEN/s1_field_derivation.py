r"""D16-OPEN s1: independent derivation of the chapter 12 field (focus question 1).

Discretized bit-circle K = {-1,1} x {theta_k = 2 pi k / M}, reference measure nu with
weight 1/(2M) per point (the exact finite analogue of (1/2) sum_z delta_z (x) dtheta/2pi).
Six agents, belief and model densities q_i, s_i relative to nu, uniform references and
evaluators r_i = G_i = 1, all-source priors pi_ij = 1/N including self, selective channel
K((z,theta), .) = delta_z (x) Haar on every edge including self, exact Gibbs rows, unit Fisher
mobility.

Checks:
 [1] reduced energy  E_red = sum_i KL(q_i||1)+KL(s_i||1) - T sum_i log( (1/N) sum_j e^{-D_ij/T} )
     functional derivative by central finite differences vs the closed form of open:full-field,
     h_i = 2 log q_i - sum_j beta_ij log mu_j - sum_k beta_ki mu_k/mu_i  (+2 exactly).
 [2] rows from the full costs D_ij vs rows from KL(mu_i||mu_j) (open:row).
 [3] binary family (uniform angle): xdot from the full field vs open:binary-field.
 [4] correlated laws: d/dt log f(theta|z) vs -2(log f - E[log f|z]) (open:conditional-flow),
     and xdot vs binary field + forcing -(1/2)(1-x^2)(c(1)-c(-1)) (open:bit-forcing).
 [5] null flow (inh:null-flow), both readings of "deleting peer costs":
     (R) remove peer sources: E0 = sum KL(q||1)+KL(s||1)+D_ii (+const);
     (Z) set peer costs to zero with rows still optimized over all sources:
         E0 = sum KL - T log((1/N)(e^{-D_ii/T} + N-1)).
     On the binary family both give xdot = -(1-x^2) atanh x.
 [6] open:interior extremum comparison sanity: max_i atanh x_i decays at least like e^{-t}.
"""
import numpy as np

rng = np.random.default_rng(20261029)
N, M, T = 6, 16, 0.37
th = 2 * np.pi * np.arange(M) / M
w = 1.0 / (2 * M)               # nu weight per point
Z = np.array([1.0, -1.0])        # z values, index 0 -> z=+1, 1 -> z=-1


def normalize(q):
    return q / (q.sum() * w)


def rand_law(corr=True):
    # log q(z,theta) = trig polynomial with z-dependent coefficients (correlated) or not
    lq = np.zeros((2, M))
    for zi in range(2):
        a1, b1, a2, b2 = rng.normal(0, 0.7, 4)
        if not corr and zi == 1:
            lq[1] = lq[0] - lq[0].mean()
            break
        lq[zi] = a1 * np.cos(th) + b1 * np.sin(th) + a2 * np.cos(2 * th) + b2 * np.sin(2 * th)
    lq += rng.normal(0, 0.6) * Z[:, None]      # bit bias
    return normalize(np.exp(lq))


def mu(q):           # bit-marginal density relative to the uniform bit measure
    return q.mean(axis=1)


def D(qi, qj):       # KL(q_i || mu_j (x) Haar)
    mj = mu(qj)
    return (w * qi * (np.log(qi) - np.log(mj)[:, None])).sum()


def E_red(Q, S, T=T):
    E = 0.0
    for sec in (Q, S):
        for i in range(N):
            E += (w * sec[i] * np.log(sec[i])).sum()
            Dij = np.array([D(sec[i], sec[j]) for j in range(N)])
            E += -T * np.log(np.mean(np.exp(-Dij / T)))
    return E


def rows(sec, T=T):
    B = np.zeros((N, N))
    for i in range(N):
        Dij = np.array([D(sec[i], sec[j]) for j in range(N)])
        e = np.exp(-(Dij - Dij.min()) / T)
        B[i] = e / e.sum()
    return B


def h_closed(sec, T=T):
    B = rows(sec, T)
    m = np.array([mu(s) for s in sec])        # (N,2)
    H = []
    for i in range(N):
        hi = 2 * np.log(sec[i]) - (B[i][:, None] * np.log(m)).sum(0)[:, None] \
             - ((B[:, i][:, None] * m).sum(0) / m[i])[:, None]
        H.append(hi)
    return np.array(H)


def fisher_V(sec, H):
    V = []
    for i in range(N):
        Eh = (w * sec[i] * H[i]).sum()
        V.append(-sec[i] * (H[i] - Eh))
    return np.array(V)


out = []
# ---------- [1] functional derivative ----------
Q = np.array([rand_law() for _ in range(N)])
S = np.array([rand_law() for _ in range(N)])
eps = 1e-6
fd = np.zeros_like(Q)
for i in range(N):
    for zi in range(2):
        for k in range(M):
            Qp = Q.copy(); Qp[i, zi, k] += eps
            Qm = Q.copy(); Qm[i, zi, k] -= eps
            fd[i, zi, k] = (E_red(Qp, S) - E_red(Qm, S)) / (2 * eps) / w
hc = h_closed(Q) + 2.0
out.append("[1] max |FD functional derivative - (h_closed+2)| over 6 agents x 2 x %d = %.3e"
           % (M, np.abs(fd - hc).max()))
out.append("    (scale of h: max|h| = %.3f)" % np.abs(hc).max())

# ---------- [2] rows depend only on bits ----------
B_full = rows(Q)
m = np.array([mu(q) for q in Q])
x = (m[:, 0] - m[:, 1]) / 2
def dbin(u, v):
    return 0.5 * ((1 + u) * np.log((1 + u) / (1 + v)) + (1 - u) * np.log((1 - u) / (1 - v)))
Db = np.array([[dbin(x[i], x[j]) for j in range(N)] for i in range(N)])
Bb = np.exp(-Db / T); Bb /= Bb.sum(1, keepdims=True)
out.append("[2] max |rows from full D_ij - rows from KL(mu_i||mu_j)| = %.3e" % np.abs(B_full - Bb).max())


def binary_field(x, T=T):
    Db = np.array([[dbin(x[i], x[j]) for j in range(N)] for i in range(N)])
    B = np.exp(-(Db - Db.min(1, keepdims=True)) / T); B /= B.sum(1, keepdims=True)
    at = np.arctanh(x)
    xd = np.zeros(N)
    for i in range(N):
        xd[i] = (1 - x[i] ** 2) * (-at[i] - (B[i] * (at[i] - at)).sum()) + (B[:, i] * (x - x[i])).sum()
    return xd


# ---------- [3] binary family ----------
xb = rng.uniform(-0.6, 0.6, N)
Qb = np.array([np.outer(1 + Z * xb[i], np.ones(M)) for i in range(N)])
Vb = fisher_V(Qb, h_closed(Qb))
xdot_full = np.array([(mu(Vb[i])[0] - mu(Vb[i])[1]) / 2 for i in range(N)])
out.append("[3] binary family: max |xdot(full field) - open:binary-field| = %.3e  (|xdot| max %.3f)"
           % (np.abs(xdot_full - binary_field(xb)).max(), np.abs(xdot_full).max()))

# ---------- [4] correlated laws: conditional flow and bit forcing ----------
V = fisher_V(Q, h_closed(Q))
res_f, res_x = 0.0, 0.0
xd_pred = binary_field(x)
for i in range(N):
    mi = mu(Q[i]); f = Q[i] / mi[:, None]
    dmu = mu(V[i])
    dlogf = V[i] / Q[i] - (dmu / mi)[:, None]
    lf = np.log(f)
    pred = -2 * (lf - (f * lf).mean(1, keepdims=True))
    res_f = max(res_f, np.abs(dlogf - pred).max())
    c = 2 * (f * lf).mean(1)                       # c(z) = 2 KL(f(.|z) || Haar)
    xd_pred[i] += -0.5 * (1 - x[i] ** 2) * (c[0] - c[1])
    res_x = max(res_x, abs((dmu[0] - dmu[1]) / 2 - xd_pred[i]))
out.append("[4] correlated: max |dlog f/dt - (-2)(log f - E[log f|z])| = %.3e" % res_f)
out.append("    correlated: max |xdot - (binary field + forcing)|        = %.3e" % res_x)

# ---------- [5] null flow ----------
def E_null(Q, S, reading):
    E = 0.0
    for sec in (Q, S):
        for i in range(N):
            E += (w * sec[i] * np.log(sec[i])).sum()
            Dii = D(sec[i], sec[i])
            if reading == "R":
                E += Dii
            else:
                E += -T * np.log((np.exp(-Dii / T) + N - 1) / N)
    return E

for reading in ("R", "Z"):
    # gradient of E_null on the binary family by finite differences, then Fisher flow
    Qn = Qb.copy()
    g = np.zeros_like(Qn)
    for i in range(N):
        for zi in range(2):
            for k in range(M):
                Qp = Qn.copy(); Qp[i, zi, k] += eps
                Qm = Qn.copy(); Qm[i, zi, k] -= eps
                g[i, zi, k] = (E_null(Qp, S, reading) - E_null(Qm, S, reading)) / (2 * eps) / w
    Vn = fisher_V(Qn, g)
    xdn = np.array([(mu(Vn[i])[0] - mu(Vn[i])[1]) / 2 for i in range(N)])
    out.append("[5] null reading %s, binary family: max |xdot + (1-x^2) atanh x| = %.3e"
               % (reading, np.abs(xdn + (1 - xb ** 2) * np.arctanh(xb)).max()))
    # correlated laws: angular relaxation rate under the null
    g = np.zeros_like(Q)
    for i in range(1):
        for zi in range(2):
            for k in range(M):
                Qp = Q.copy(); Qp[i, zi, k] += eps
                Qm = Q.copy(); Qm[i, zi, k] -= eps
                g[i, zi, k] = (E_null(Qp, S, reading) - E_null(Qm, S, reading)) / (2 * eps) / w
    Eh = (w * Q[0] * g[0]).sum()
    V0 = -Q[0] * (g[0] - Eh)
    mi = mu(Q[0]); f = Q[0] / mi[:, None]; lf = np.log(f)
    dlogf = V0 / Q[0] - (mu(V0) / mi)[:, None]
    cen = lf - (f * lf).mean(1, keepdims=True)
    rate = -(dlogf * cen).sum() / (cen * cen).sum()
    Dii = D(Q[0], Q[0]); bii = np.exp(-Dii / T) / (np.exp(-Dii / T) + N - 1)
    out.append("    null reading %s, correlated agent 0: fitted angular relaxation rate = %.6f"
               "  (2 for R; 1+beta_ii = %.6f for Z)" % (reading, rate, 1 + bii))

# ---------- [6] extremum comparison ----------
from scipy.integrate import solve_ivp
x0 = rng.uniform(-0.9, 0.9, N)
sol = solve_ivp(lambda t, x: binary_field(x), (0, 6), x0, rtol=1e-10, atol=1e-12, dense_output=True)
ts = np.linspace(0, 6, 61)
th_ = np.arctanh(sol.sol(ts))
viol_hi = (th_.max(0) - np.exp(-ts) * np.arctanh(x0).max()).max()
viol_lo = (np.exp(-ts) * np.arctanh(x0).min() - th_.min(0)).max()
out.append("[6] open:logit-bound: max(max logit - e^-t max0) = %.3e, max(e^-t min0 - min logit) = %.3e (both <= 0 expected)"
           % (viol_hi, viol_lo))

print("\n".join(out))
